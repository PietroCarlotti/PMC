# Import packages
import math
import gpytorch
import pyro
import torch
from linear_operator.utils.errors import NanError, NotPSDError

from abc import ABC, abstractmethod
from numbers import Integral, Real
from pyro.infer import MCMC, NUTS
from pyro.ops.integrator import register_exception_handler

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


class _MPSGammaPrior(gpytorch.priors.GammaPrior):
    """Sample Gamma priors on CPU when MPS lacks the Gamma sampler."""

    def expand(self, batch_shape: torch.Size) -> "_MPSGammaPrior":
        return _MPSGammaPrior(
            self.concentration.expand(batch_shape), self.rate.expand(batch_shape)
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if self.concentration.device.type == "mps":
            cpu_prior = torch.distributions.Gamma(
                self.concentration.cpu(), self.rate.cpu()
            )
            return cpu_prior.rsample(sample_shape).to(self.concentration.device)
        return super().rsample(sample_shape)


class GaussianProcess(Regressor):
    """Base class for Gaussian-process regressors of h(C) on D."""

    @staticmethod
    def _build_mean(mean: str) -> gpytorch.means.Mean:
        if mean == "zero":
            return gpytorch.means.ZeroMean()
        raise ValueError(f"unsupported mean {mean!r}")

    @staticmethod
    def _build_kernel(
        kernel: str,
        lengthscale_prior_shape: float,
        lengthscale_prior_scale: float,
        signal_sd_prior_scale: float,
    ) -> gpytorch.kernels.Kernel:
        # Build the kernel
        if kernel == "rbf":
            # Build the base kernel
            base_kernel = gpytorch.kernels.RBFKernel()

            # Register the inverse lengthscale prior on the base kernel
            base_kernel.register_prior(
                "inverse_lengthscale_prior",
                _MPSGammaPrior(lengthscale_prior_shape, lengthscale_prior_scale),
                lambda module: module.lengthscale.reciprocal(),
                lambda module, value: module.initialize(lengthscale=value.reciprocal()),
            )
        else:
            raise ValueError(f"unsupported kernel {kernel!r}")

        # Build the scale kernel
        return gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=_MPSGammaPrior(
                0.5, 1 / (2 * signal_sd_prior_scale**2)
            ),
        )

    @staticmethod
    def _build_likelihood(
        likelihood: str,
        noise_sd_prior_scale: float,
    ) -> gpytorch.likelihoods.Likelihood:
        if likelihood == "gaussian":
            return gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=_MPSGammaPrior(
                    0.5, 1 / (2 * noise_sd_prior_scale**2)
                ),
                noise_constraint=gpytorch.constraints.Positive(),
            )
        raise ValueError(f"unsupported likelihood {likelihood!r}")

    def __init__(
        self,
        mean: str,
        kernel: str,
        likelihood: str,
        lengthscale_prior_shape: float,
        lengthscale_prior_scale: float,
        signal_sd_prior_scale: float,
        noise_sd_prior_scale: float,
    ):
        """Build the chosen mean, kernel, and likelihood."""
        self.mean_module = self._build_mean(mean)
        self.covar_module = self._build_kernel(
            kernel, lengthscale_prior_shape, lengthscale_prior_scale, signal_sd_prior_scale
        )
        self.likelihood = self._build_likelihood(likelihood, noise_sd_prior_scale)

    def _standardize(self, d: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardize discrepancies and targets, retaining scales for prediction."""
        # Standardize the discrepancies
        self.d_mean = d.mean(dim=0)
        self.d_sd = d.std(dim=0, unbiased=False)
        self.d_sd = torch.where(self.d_sd > 0, self.d_sd, torch.ones_like(self.d_sd))

        # Standardize the targets
        self.c_mean = c.mean(dim=0)
        self.c_sd = c.std(dim=0, unbiased=False)
        self.c_sd = torch.where(self.c_sd > 0, self.c_sd, torch.ones_like(self.c_sd))

        return (d - self.d_mean) / self.d_sd, (c - self.c_mean) / self.c_sd

    def _unstandardize(self, c: torch.Tensor) -> torch.Tensor:
        """Return standardized targets to their original scale."""
        return c * self.c_sd + self.c_mean


class _ExactGP(gpytorch.models.ExactGP):
    """GPyTorch exact model for one target coordinate."""

    def __init__(
        self,
        d: torch.Tensor,
        c: torch.Tensor,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood,
    ):
        # Check that the likelihood is Gaussian
        if not isinstance(likelihood, (gpytorch.likelihoods.GaussianLikelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)):
            raise TypeError("exact GP inference requires a Gaussian likelihood")

        super().__init__(d, c, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, d: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(d), self.covar_module(d))


class ExactGaussianProcess(GaussianProcess):
    """Exact GP regressor configured for Bayesian hyperparameter sampling."""

    def __init__(
        self,
        mean: str,
        kernel: str,
        likelihood: str,
        lengthscale_prior_shape: float,
        lengthscale_prior_scale: float,
        signal_sd_prior_scale: float,
        noise_sd_prior_scale: float,
        chains: int,
        warmup: int,
        samples: int,
        adapt_delta: float,
        seed: int | None,
    ):
        super().__init__(
            mean, kernel, likelihood,
            lengthscale_prior_shape, lengthscale_prior_scale,
            signal_sd_prior_scale, noise_sd_prior_scale,
        )

        # Check that chains, warmup, and samples are positive integers
        for name, value in (("chains", chains), ("warmup", warmup), ("samples", samples)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be at least 1")

        # Check that adapt_delta is a real number
        if isinstance(adapt_delta, bool) or not isinstance(adapt_delta, Real):
            raise TypeError("adapt_delta must be a real number")

        # Check that adapt_delta is finite and strictly between 0 and 1
        if not math.isfinite(adapt_delta) or not 0 < adapt_delta < 1:
            raise ValueError("adapt_delta must be finite and strictly between 0 and 1")

        # Check that seed is an integer or None
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise TypeError("seed must be an integer or None")

        self.chains = chains
        self.warmup = warmup
        self.samples = samples
        self.adapt_delta = adapt_delta
        self.seed = seed

    def _fit(self, d: torch.Tensor, c: torch.Tensor) -> None:
        # Check that there is only one target coordinate
        if c.shape[1] != 1:
            raise NotImplementedError("exact GP fitting currently supports one target coordinate")

        # Reject NUTS proposals whose GP covariance is numerically invalid
        register_exception_handler(
            "gpytorch_invalid_covariance",
            lambda error: isinstance(error, (NanError, NotPSDError)),
            warn_on_overwrite=False,
        )

        # Standardize the discrepancies and targets
        d, c = self._standardize(d, c)

        # Build the exact GP model
        self._model = _ExactGP(
            d, c[:, 0], self.mean_module, self.covar_module, self.likelihood
        ).to(device=d.device, dtype=d.dtype)

        # Define the joint model for MCMC sampling
        def joint_model() -> None:
            sampled_model = self._model.pyro_sample_from_prior()
            pyro.sample("obs", sampled_model.likelihood(sampled_model(d)), obs=c[:, 0])

        # Run MCMC sampling
        self._mcmc = []

        # Run MCMC sampling for each chain
        for chain in range(self.chains):
            # Set the random seed for reproducibility if provided
            if self.seed is not None:
                pyro.set_rng_seed(self.seed + chain)

            # Set up the NUTS kernel
            nuts_kernel = NUTS(joint_model, target_accept_prob=self.adapt_delta)

            # Set up the MCMC sampler
            mcmc = MCMC(
                nuts_kernel,
                num_samples=self.samples,
                warmup_steps=self.warmup,
                num_chains=1,
            )

            # Run the MCMC sampler
            with gpytorch.settings.fast_computations(False, False, False):
                mcmc.run()
            self._mcmc.append(mcmc)

        # Concatenate the samples from all chains
        chain_samples = [run.get_samples() for run in self._mcmc]
        self._posterior_samples = {
            name: torch.cat([draws[name] for draws in chain_samples], dim=0)
            for name in chain_samples[0]
        }

        # Load the posterior samples into the model
        self._model.pyro_load_from_samples(self._posterior_samples)

        # Set the model to evaluation mode
        self._model.eval()
        self.f_zero_samples = None

    def _predict(self) -> torch.Tensor:
        if self.f_zero_samples is None:
            # Draw f(0) for each retained hyperparameter draw
            zero = (-self.d_mean / self.d_sd).reshape(1, 1, self._m)
            zero = zero.expand(self.chains * self.samples, -1, -1)

            # Evaluate the model at D = 0
            with torch.no_grad(), gpytorch.settings.fast_computations(False, False, False):
                self.f_zero_samples = self._unstandardize(self._model(zero).sample())

        return self.f_zero_samples.mean(dim=0)
