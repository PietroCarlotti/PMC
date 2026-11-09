# Import packages
import math
import torch

from abc import ABC, abstractmethod
from cmdstanpy import CmdStanModel
from numbers import Integral, Real
from pathlib import Path
from sklearn.cluster import KMeans

class Regressor(ABC):
    """Base class for regressors that learn E[h(C) | D = d]."""

    @abstractmethod
    def _fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        """Fit the model to the pairs (D^(s), C^(s)) for s = 1, ..., S.

        Args:
            d: S draws of the discrepancy, a tensor of shape (S, m).
            c: S draws of the target h(C), a tensor of shape (S, r).
        """

    @abstractmethod
    def _predict(self) -> torch.Tensor:
        """Predict the target at d = 0.

        Returns:
            Tensor of shape (r,).
        """

    def fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        """Fit the regressor to the pairs of discrepancy and target draws.

        Args:
            d: S draws of the discrepancy, a tensor of shape (S, m).
            c: S draws of the target h(C), a tensor of shape (S, r).
        """
        # Check that d is a tensor
        if not isinstance(d, torch.Tensor):
            raise TypeError("d must be a torch.Tensor")

        # Check that c is a tensor
        if not isinstance(c, torch.Tensor):
            raise TypeError("c must be a torch.Tensor")

        # Check that d and c have shapes (S, m) and (S, r)
        if d.ndim != 2 or c.ndim != 2:
            raise ValueError(
                f"d and c must have shapes (S, m) and (S, r), "
                f"got {tuple(d.shape)} and {tuple(c.shape)}"
            )

        # Unpack the number of draws S, discrepancy dimension m, and target dimension r
        S, m = d.shape
        r = c.shape[1]

        # Check that d and c have the same number of draws S
        if c.shape[0] != S:
            raise ValueError(f"d and c must have the same S, got {S} and {c.shape[0]}")

        # Check that there is at least one draw, discrepancy coordinate, and target coordinate
        if S < 1 or m < 1 or r < 1:
            raise ValueError(f"S, m, and r must be at least 1, got S={S}, m={m}, r={r}")

        # Check that d and c are floating-point tensors
        if not d.is_floating_point() or not c.is_floating_point():
            raise ValueError(f"d and c must be floating point, got {d.dtype} and {c.dtype}")

        # Check that d and c have the same device and dtype
        if d.device != c.device or d.dtype != c.dtype:
            raise ValueError("d and c must have the same device and dtype")

        # Store the dimensions m, r, the device, and dtype
        self._m = d.shape[1]
        self._r = c.shape[1]
        self._device = d.device
        self._dtype = d.dtype

        # Fit the subclass's model
        self._fit(d, c)

    def predict(self) -> torch.Tensor:
        """Predict the target at a zero discrepancy.

        Returns:
            Tensor of shape (r,).
        """
        # Check that the model has been fitted
        if not hasattr(self, "_m"):
            raise RuntimeError("call fit before predict")

        # Apply the subclass's model at d = 0
        output = self._predict()

        # Check that the output is a tensor
        if not isinstance(output, torch.Tensor):
            raise TypeError("regressor must return a torch.Tensor")

        # Check that the output has shape (r,)
        if output.shape != (self._r,):
            raise ValueError(
                f"regressor must return shape (r,) = {(self._r,)}, got {tuple(output.shape)}"
            )

        # Check that the output has the same device and dtype as the data passed to fit
        if output.device != self._device or output.dtype != self._dtype:
            raise ValueError(
                f"regressor must return device {self._device} and dtype {self._dtype}, "
                f"got device {output.device} and dtype {output.dtype}"
            )

        # Check that the output is finite
        if not torch.isfinite(output).all():
            raise ValueError("regressor must return finite values")
        return output


class OLS(Regressor):
    """Ordinary least squares for h(C) = a + d B."""

    def _fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        """Fit the intercept and the coefficients by least squares."""
        # When there are at least twice as many draws as parameters, solve the centered normal equations
        if d.shape[0] >= 2 * (d.shape[1] + 1):
            # Compute the discrepancy and target means
            d_mean = d.mean(dim=0)
            c_mean = c.mean(dim=0)

            # Form the centered Gram matrix and cross-product
            gram = d.T @ d - d.shape[0] * torch.outer(d_mean, d_mean)
            cross = d.T @ c - d.shape[0] * torch.outer(d_mean, c_mean)

            # Solve for the coefficients, recover the intercept, and stack the solution
            coef = torch.linalg.solve(gram, cross)
            intercept = c_mean - d_mean @ coef
            beta = torch.cat([intercept.unsqueeze(0), coef], dim=0)
        # When there are enough draws, use the ordinary least-squares solution
        elif d.shape[0] >= d.shape[1] + 1:
            # Add a column of ones for the intercept: (S, m) -> (S, m + 1)
            X = torch.cat([torch.ones_like(d[:, :1]), d], dim=1)
            beta = torch.linalg.lstsq(X, c, rcond=1e-5).solution
        else:
            # Compute the discrepancy and target means
            d_mean = d.mean(dim=0)
            c_mean = c.mean(dim=0)

            # Center the discrepancies and targets
            centered_d = d - d_mean
            centered_c = c - c_mean

            # Solve for the minimum-norm slopes and recover the intercept
            coef = torch.linalg.pinv(centered_d) @ centered_c
            intercept = c_mean - d_mean @ coef
            beta = torch.cat([intercept.unsqueeze(0), coef], dim=0)

        # Split the solution into the intercept (r,) and the coefficients (m, r)
        self.intercept = beta[0]
        self.coef = beta[1:]

    def _predict(self) -> torch.Tensor:
        """Return the intercept, which is the prediction at a zero discrepancy."""
        # At d = 0 the prediction is the intercept
        return self.intercept


class GaussianProcess(Regressor):
    """Base class for Gaussian-process regressors."""

    def __init__(
        self,
        chains: int,
        warmup: int,
        samples: int,
        length_scale_prior_shape: float,
        length_scale_prior_scale: float,
        signal_sd_prior_scale: float,
        noise_sd_prior_scale: float,
        adapt_delta: float,
        seed: int | None = None,
    ):
        # Check that the chain count is an integer
        if isinstance(chains, bool) or not isinstance(chains, Integral):
            raise TypeError("chains must be an integer")

        # Check that the chain count is positive
        if chains < 1:
            raise ValueError("chains must be at least 1")

        # Check that the warmup count is an integer
        if isinstance(warmup, bool) or not isinstance(warmup, Integral):
            raise TypeError("warmup must be an integer")

        # Check that the warmup count is positive
        if warmup < 1:
            raise ValueError("warmup must be at least 1")

        # Check that the sample count is an integer
        if isinstance(samples, bool) or not isinstance(samples, Integral):
            raise TypeError("samples must be an integer")

        # Check that the sample count is positive
        if samples < 1:
            raise ValueError("samples must be at least 1")

        # Check that the seed is an integer
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise TypeError("seed must be an integer or None")

        # Check that the seed is nonnegative
        if seed is not None and seed < 0:
            raise ValueError("seed must be nonnegative or None")

        # Check that the prior settings are finite and positive
        for name, value in (
            ("length_scale_prior_shape", length_scale_prior_shape),
            ("length_scale_prior_scale", length_scale_prior_scale),
            ("signal_sd_prior_scale", signal_sd_prior_scale),
            ("noise_sd_prior_scale", noise_sd_prior_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        # Check that the adaptation target is a finite probability
        if isinstance(adapt_delta, bool) or not isinstance(adapt_delta, Real):
            raise TypeError("adapt_delta must be a real number")
        if not math.isfinite(adapt_delta) or not 0 < adapt_delta < 1:
            raise ValueError("adapt_delta must be finite and strictly between 0 and 1")

        # Store the Stan sampling configuration and prior settings
        self.chains = chains
        self.warmup = warmup
        self.samples = samples
        self.length_scale_prior_shape = length_scale_prior_shape
        self.length_scale_prior_scale = length_scale_prior_scale
        self.signal_sd_prior_scale = signal_sd_prior_scale
        self.noise_sd_prior_scale = noise_sd_prior_scale
        self.adapt_delta = adapt_delta
        self.seed = seed

    def _standardize(self, d: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Standardize the discrepancies
        self.d_mean = d.mean(dim=0)
        self.d_sd = d.std(dim=0, unbiased=False)
        d = (d - self.d_mean) / self.d_sd

        # Standardize the targets
        self.c_mean = c.mean(dim=0)
        self.c_sd = c.std(dim=0, unbiased=False)
        c = (c - self.c_mean) / self.c_sd

        return d, c

    # Return the posterior mean at zero discrepancy
    def _predict(self) -> torch.Tensor:
        return torch.as_tensor(
            self.f_zero.mean(axis=0),
            device=self._device,
            dtype=self._dtype,
        )


class ExactGaussianProcess(GaussianProcess):
    """Full Bayesian Gaussian-process regression."""

    # Locate the packaged Stan model
    _stan_file = Path(__file__).parent / "stan" / "regressor_exact_gaussian_process.stan"

    # Fit the exact Gaussian process regression through Stan
    def _fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        d, c = self._standardize(d, c)

        # Assemble the Stan data
        data = {
            "S": d.shape[0],
            "m": self._m,
            "r": self._r,
            "D": d.detach().cpu().numpy(),
            "C": c.detach().cpu().numpy(),
            "D_zero": (-self.d_mean / self.d_sd).detach().cpu().numpy(),
            "length_scale_prior_shape": self.length_scale_prior_shape,
            "length_scale_prior_scale": self.length_scale_prior_scale,
            "signal_sd_prior_scale": self.signal_sd_prior_scale,
            "noise_sd_prior_scale": self.noise_sd_prior_scale,
        }

        # Compile the Stan model
        self.model = CmdStanModel(stan_file=self._stan_file)

        # Sample from the Stan model
        self.stan_fit = self.model.sample(
            data=data,
            chains=self.chains,
            iter_warmup=self.warmup,
            iter_sampling=self.samples,
            adapt_delta=self.adapt_delta,
            seed=self.seed,
        )

        # Store the posterior samples of the latent function at zero discrepancy
        self.f_zero = self.stan_fit.stan_variable("f_zero")

        # De-standardize the posterior samples of the latent function
        self.f_zero = (self.f_zero * self.c_sd.detach().cpu().numpy() + self.c_mean.detach().cpu().numpy())


class SparseGaussianProcess(GaussianProcess):
    """Sparse Bayesian Gaussian-process regression."""

    # Locate the packaged Stan model
    _stan_file = Path(__file__).parent / "stan" / "regressor_sparse_gaussian_process.stan"

    def __init__(
        self,
        chains: int,
        warmup: int,
        samples: int,
        length_scale_prior_shape: float,
        length_scale_prior_scale: float,
        signal_sd_prior_scale: float,
        noise_sd_prior_scale: float,
        num_inducing: int,
        adapt_delta: float,
        seed: int | None = None,
    ):
        super().__init__(
            chains,
            warmup,
            samples,
            length_scale_prior_shape,
            length_scale_prior_scale,
            signal_sd_prior_scale,
            noise_sd_prior_scale,
            adapt_delta,
            seed,
        )

        # Check that the inducing-point count is a positive integer
        if isinstance(num_inducing, bool) or not isinstance(num_inducing, Integral):
            raise TypeError("num_inducing must be an integer")
        if num_inducing < 1:
            raise ValueError("num_inducing must be at least 1")

        self.num_inducing = num_inducing

    def _select_inducing_points(self, d: torch.Tensor) -> torch.Tensor:
        # Check that there are enough discrepancy draws
        if self.num_inducing > d.shape[0]:
            raise ValueError(
                f"num_inducing must not exceed the number of draws, got "
                f"num_inducing={self.num_inducing}, S={d.shape[0]}"
            )

        # Fit k-means to the standardized discrepancies
        kmeans = KMeans(
            n_clusters=self.num_inducing,
            n_init="auto",
            random_state=self.seed,
        )
        centers = kmeans.fit(d.detach().cpu().numpy()).cluster_centers_

        # Return the centers on the original tensor device and dtype
        return torch.as_tensor(centers, device=d.device, dtype=d.dtype)

    # Fit the sparse Gaussian process regression through Stan
    def _fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        d, c = self._standardize(d, c)

        # Select and store the fixed inducing locations
        self.inducing_points = self._select_inducing_points(d)

        # Assemble the Stan data
        data = {
            "S": d.shape[0],
            "m": self._m,
            "r": self._r,
            "num_inducing": self.num_inducing,
            "D": d.detach().cpu().numpy(),
            "C": c.detach().cpu().numpy(),
            "Z": self.inducing_points.detach().cpu().numpy(),
            "D_zero": (-self.d_mean / self.d_sd).detach().cpu().numpy(),
            "length_scale_prior_shape": self.length_scale_prior_shape,
            "length_scale_prior_scale": self.length_scale_prior_scale,
            "signal_sd_prior_scale": self.signal_sd_prior_scale,
            "noise_sd_prior_scale": self.noise_sd_prior_scale,
        }

        # Compile the Stan model
        self.model = CmdStanModel(stan_file=self._stan_file)

        # Sample from the Stan model
        self.stan_fit = self.model.sample(
            data=data,
            chains=self.chains,
            iter_warmup=self.warmup,
            iter_sampling=self.samples,
            adapt_delta=self.adapt_delta,
            seed=self.seed,
        )

        # Store the posterior samples of the latent function at zero discrepancy
        self.f_zero = self.stan_fit.stan_variable("f_zero")

        # De-standardize the posterior samples of the latent function
        self.f_zero = self.f_zero * self.c_sd.detach().cpu().numpy() + self.c_mean.detach().cpu().numpy()


class NeuralRegression(Regressor):
    """Base class for neural regressors."""

    def __init__(self, optimizer, epochs: int, batch_size: int, lr: float, schedule: str, clip: float | None, fused: bool, seed: int | None = None):
        """Set the training parameters.

        Args:
            optimizer: optimizer class.
            epochs: number of passes over the data.
            batch_size: number of draws in each minibatch.
            lr: learning rate at the start of the training.
            schedule: how the learning rate changes during the training, either "constant" or "cosine".
            clip: largest norm of the gradient
            fused: whether to use the fused implementation of the optimizer.
        """
        # Check that fused is Boolean
        if not isinstance(fused, bool):
            raise TypeError("fused must be a bool")

        # Check that optimizer is callable
        if not callable(optimizer):
            raise TypeError("optimizer must be callable")

        # Check that epoch and batch counts are integers
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, Integral)
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
        ):
            raise TypeError("epochs and batch_size must be integers")

        # Check that the training parameters are valid
        if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
            raise ValueError(
                f"epochs and batch_size must be at least 1 and lr finite and positive, "
                f"got epochs={epochs}, batch_size={batch_size}, lr={lr}"
            )
        if schedule not in ("constant", "cosine"):
            raise ValueError(f'schedule must be "constant" or "cosine", got {schedule!r}')
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError(f"clip must be finite and positive or None, got clip={clip}")

        # Store the training parameters
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.schedule = schedule
        self.clip = clip
        self.fused = fused
        self.seed = seed

    @abstractmethod
    def _build(self) -> torch.nn.Module:
        """Build a new network that maps the discrepancy to the target.

        Returns:
            The network, which maps a tensor of shape (S, m) to a tensor of shape (S, r).
        """

    def _fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        """Build a new network and train it."""
        # Build the network with the configured seed, device, and dtype
        if self.seed is None:
            self.net = self._build().to(
                device=self._device,
                dtype=self._dtype,
            )
        else:
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                self.net = self._build().to(
                    device=self._device,
                    dtype=self._dtype,
                )

        # Set up the optimizer
        if self.fused and self._device.type == "cuda":
            try:
                optimizer = self.optimizer(
                    self.net.parameters(),
                    lr=self.lr,
                    fused=True,
                )
            except TypeError:
                optimizer = self.optimizer(self.net.parameters(), lr=self.lr)
        else:
            optimizer = self.optimizer(self.net.parameters(), lr=self.lr)

        # Set up the learning rate scheduler
        if self.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        else:
            scheduler = None

        # Set the network to training mode
        self.net.train()

        # Record the mean loss from each epoch
        self.loss_history = []

        # Set the random state for minibatch shuffling from the configured seed
        rng = None
        if self.seed is not None:
            rng = torch.Generator(device=d.device)
            rng.manual_seed(self.seed)

        # Train the network with minibatches
        for _ in range(self.epochs):
            # Initialize the epoch loss and batch count
            total_loss = torch.zeros((), device=d.device, dtype=d.dtype)
            num_batches = 0

            # Shuffle the draws at the start of each epoch
            perm = torch.randperm(
                d.shape[0],
                device=d.device,
                generator=rng,
            )

            for start in range(0, d.shape[0], self.batch_size):
                # Select the draws in the minibatch
                idx = perm[start:start + self.batch_size]

                # Reset the gradients
                optimizer.zero_grad()

                # Compute the squared error on the minibatch
                loss = torch.nn.functional.mse_loss(self.net(d[idx]), c[idx])

                # Accumulate the detached minibatch loss
                total_loss += loss.detach()
                num_batches += 1

                # Backpropagate the loss
                loss.backward()

                # Scale down the gradient if its norm is above the limit
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)

                # Update the parameters
                optimizer.step()

            # Update the learning rate at the end of each epoch
            if scheduler is not None:
                scheduler.step()

            self.loss_history.append((total_loss / num_batches).item())

    def _predict(self) -> torch.Tensor:
        """Evaluate the network at a zero discrepancy."""
        # Build the zero vector d = 0: shape (m,)
        d = torch.zeros(self._m, device=self._device, dtype=self._dtype)

        # Set the network to evaluation mode
        self.net.eval()

        # Evaluate the network without tracking gradients: (m,) -> (r,)
        with torch.no_grad():
            return self.net(d)


class MLP(NeuralRegression):
    """Multilayer perceptron trained on the squared error, so that it estimates E[h(C) | D = d]."""

    def __init__(self, hidden_sizes, activation, optimizer, epochs: int, batch_size: int, lr: float, schedule: str, clip: float | None, fused: bool, seed: int | None = None):
        """Set the architecture and the training parameters.

        Args:
            hidden_sizes: sizes of the hidden layers.
            activation: activation class.
            optimizer: optimizer class.
            epochs: number of passes over the data.
            batch_size: number of draws in each minibatch.
            lr: learning rate at the start of the training.
            schedule: how the learning rate changes during the training, either "constant" or "cosine".
            clip: largest norm of the gradient.
            fused: whether to use the fused implementation of the optimizer.
        """
        super().__init__(optimizer, epochs, batch_size, lr, schedule, clip, fused, seed)

        # Store the architecture
        self.hidden_sizes = hidden_sizes
        self.activation = activation

    def _build(self) -> torch.nn.Module:
        """Build the layers of the MLP."""
        # Set up the layer sizes m -> hidden_sizes
        sizes = [self._m, *self.hidden_sizes]

        # Add a linear layer and an activation for each hidden size
        layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers += [torch.nn.Linear(n_in, n_out), self.activation()]
        layers.append(torch.nn.Linear(sizes[-1], self._r))
        return torch.nn.Sequential(*layers)


class CNN(NeuralRegression):
    """Convolutional network trained on the squared error, so that it estimates E[h(C) | D = d]."""

    def __init__(self, shape: tuple[int, int], channels, kernel_size: int, activation, optimizer, epochs: int, batch_size: int, lr: float, schedule: str, clip: float | None, fused: bool):
        """Set the architecture and the training parameters.

        Args:
            shape: height and width of the target images.
            channels: number of channels of each hidden convolution.
            kernel_size: odd size of the convolution kernels.
            activation: activation class, for example torch.nn.ReLU.
            optimizer: optimizer class, for example torch.optim.Adam.
            epochs: number of passes over the data.
            batch_size: number of draws in each minibatch.
            lr: learning rate at the start of the training.
            schedule: how the learning rate changes during the training, either "constant" or "cosine".
            clip: largest norm of the gradient.
            fused: whether to use the fused implementation of the optimizer.
        """
        super().__init__(optimizer, epochs, batch_size, lr, schedule, clip, fused)

        # Check that the kernels have an odd size
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        # Store the architecture
        self.shape = shape
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation

    def _build(self) -> torch.nn.Module:
        """Build the convolutions of the CNN."""
        # Check that the target is an image
        height, width = self.shape
        if self._r != height * width:
            raise ValueError(f"the target must have {height * width} coordinates, got r={self._r}")

        # Turn each flattened discrepancy into an image with one channel: (S, m) -> (S, 1, height, width)
        layers = [torch.nn.Unflatten(-1, (1, height, width))]

        # If the discrepancy is not an image, learn the image from it first
        if self._m != height * width:
            layers.insert(0, torch.nn.Linear(self._m, height * width))

        # Add a convolution and an activation for each hidden number of channels
        sizes = [1, *self.channels]
        padding = self.kernel_size // 2
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers += [torch.nn.Conv2d(n_in, n_out, self.kernel_size, padding=padding), self.activation()]

        # Map to one channel with a convolution of size 1 and flatten the images: (S, channels, height, width) -> (S, r)
        layers += [torch.nn.Conv2d(sizes[-1], 1, 1), torch.nn.Flatten(start_dim=-3)]
        net = torch.nn.Sequential(*layers)

        # Add a learned bias for each pixel to the input, so that a zero input can give a different output at each pixel
        net.bias = torch.nn.Parameter(torch.zeros(self._m))
        net.register_forward_pre_hook(lambda module, inputs: (inputs[0] + module.bias,))
        return net
