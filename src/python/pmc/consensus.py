# Import packages
import torch

from abc import ABC, abstractmethod

class Consensus(ABC):
    """Base class for consensus functions c: X^K -> X.

    A consensus function must satisfy c(x, ..., x) = x for all x.
    """

    @abstractmethod
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Map K samples to a single point.

        Args:
            samples: S draws of K points, a tensor of shape (S, K, p).

        Returns:
            Tensor of shape (S, p).
        """

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply the consensus function to the samples.

        Args:
            samples: S draws of K points, a tensor of shape (S, K, p).

        Returns:
            Tensor of shape (S, p).
        """
        # Check that the input is a tensor
        if not isinstance(samples, torch.Tensor):
            raise TypeError("samples must be a torch.Tensor")

        # Check that the input has shape (S, K, p)
        if samples.ndim != 3:
            raise ValueError(
                f"samples must have shape (S, K, p), got {tuple(samples.shape)}"
            )

        # Unpack the number of draws S, the number of components K, and the dimension p
        S, K, p = samples.shape

        # Check that the input dimensions satisfy the consensus contract
        if S < 1 or K < 2 or p < 1:
            raise ValueError(
                f"S and p must be at least 1 and K must be at least 2, "
                f"got S={S}, K={K}, p={p}"
            )

        # Check that the input is a floating-point tensor
        if not samples.is_floating_point():
            raise ValueError(f"samples must be floating point, got dtype {samples.dtype}")

        # Apply the subclass's consensus function
        output = self.forward(samples)

        # Check that the output is a tensor
        if not isinstance(output, torch.Tensor):
            raise TypeError("consensus must return a torch.Tensor")

        # Check that the output has shape (S, p)
        if output.shape != (S, p):
            raise ValueError(
                f"consensus must return shape (S, p) = {(S, p)}, got {tuple(output.shape)}"
            )

        # Check that the output has the same device and dtype as the input
        if output.device != samples.device or output.dtype != samples.dtype:
            raise ValueError(
                f"consensus must return device {samples.device} and dtype {samples.dtype}, "
                f"got device {output.device} and dtype {output.dtype}"
            )
        return output


class WeightedAverage(Consensus):
    """Weighted average c(x_1, ..., x_K) = sum_k W_k x_k, with sum_k W_k = I."""

    def __init__(self, weights: torch.Tensor):
        """Set the matrix weights of the weighted average.

        Args:
            weights: one matrix per component, a tensor of shape (K, p, p). The matrices must sum to the identity.
        """
        # Convert weights to a tensor
        weights = torch.as_tensor(weights)

        # Check that the weights satisfy the consensus dimension contract
        if (
            weights.ndim != 3
            or weights.shape[0] < 2
            or weights.shape[1] < 1
            or weights.shape[1] != weights.shape[2]
        ):
            raise ValueError(f"weights must have shape (K, p, p), got {tuple(weights.shape)}")

        # Check that the weights have a floating-point dtype
        if not weights.is_floating_point():
            raise ValueError(f"weights must be floating point, got dtype {weights.dtype}")

        # Check that the weights sum to the identity, which implies c(x, ..., x) = x
        eye = torch.eye(weights.shape[1], dtype=weights.dtype, device=weights.device)
        if not torch.allclose(weights.sum(dim=0), eye):
            raise ValueError("weights must sum to the identity matrix over K")

        self.weights = weights

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Average the points with the matrix weights."""
        # Check that there is one weight matrix per component and that dimensions match
        _, K, p = samples.shape
        if self.weights.shape[0] != K or self.weights.shape[1] != p:
            raise ValueError(
                f"weights have shape {tuple(self.weights.shape)}, "
                f"but samples have K={K}, p={p}"
            )

        # Match the weights to the device and dtype of the samples
        W = self.weights.to(samples)

        # Apply W_k to x_k and sum over the components: (S, K, p) -> (S, p)
        return torch.einsum("kab,skb->sa", W, samples)


class Mean(Consensus):
    """Arithmetic mean c(x_1, ..., x_K) = (1/K) sum_k x_k.

    Equivalent to `WeightedAverage` with W_k = I_p / K.
    """

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Average the points."""
        # Average over the components: (S, K, p) -> (S, p)
        return samples.mean(dim=1)
