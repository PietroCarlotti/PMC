# Import packages
import math
import torch

from abc import ABC, abstractmethod

class Discrepancy(ABC):
    """Base class for discrepancy functions d: X^K -> R^m.

    A discrepancy function must satisfy d(x_1, ..., x_K) = 0 iff x_1 = ... = x_K.
    """

    @abstractmethod
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Map K samples to a discrepancy vector.

        Args:
            samples: S draws of K points, a tensor of shape (S, K, p).

        Returns:
            Tensor of shape (S, m).
        """

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply the discrepancy function to the samples.

        Args:
            samples: S draws of K points, a tensor of shape (S, K, p).

        Returns:
            Tensor of shape (S, m).
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

        # Check that the input dimensions satisfy the PMC contract
        if S < 1 or K < 2 or p < 1:
            raise ValueError(
                f"S and p must be at least 1 and K must be at least 2, "
                f"got S={S}, K={K}, p={p}"
            )

        # Check that the input is a floating-point tensor
        if not samples.is_floating_point():
            raise ValueError(f"samples must be floating point, got dtype {samples.dtype}")

        # Apply the subclass's discrepancy function
        output = self.forward(samples)

        # Check that the output is a tensor
        if not isinstance(output, torch.Tensor):
            raise TypeError("discrepancy must return a torch.Tensor")

        # Check that the output has shape (S, m) for m at least 1
        if output.ndim != 2 or output.shape[0] != S or output.shape[1] < 1:
            raise ValueError(
                f"discrepancy must return shape (S, m) with S = {S} and m at least 1, "
                f"got {tuple(output.shape)}"
            )

        # Check that the output has the same device and dtype as the input
        if output.device != samples.device or output.dtype != samples.dtype:
            raise ValueError(
                f"discrepancy must return device {samples.device} and dtype {samples.dtype}, "
                f"got device {output.device} and dtype {output.dtype}"
            )
        return output


class AnchoredDifferences(Discrepancy):
    """Differences from the first component d(x_1, ..., x_K) = (x_2 - x_1, ..., x_K - x_1).

    The output has dimension m = (K - 1) p.
    """

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Subtract the first coordinate from the others and flatten the differences."""
        # Check that there are at least two components
        if samples.shape[1] < 2:
            raise ValueError(f"K must be at least 2, got K={samples.shape[1]}")

        # Subtract the first component from the others: (S, K, p) -> (S, K - 1, p)
        diffs = samples[:, 1:] - samples[:, :1]

        # Flatten the differences into one vector: (S, K - 1, p) -> (S, (K - 1) p)
        return diffs.flatten(start_dim=1)


class Norm(Discrepancy):
    """Norm of `AnchoredDifferences` d(x_1, ..., x_K) = ||(x_2 - x_1, ..., x_K - x_1)||_ord.

    The output has dimension m = 1.
    """

    def __init__(self, ord: float = 2):
        """Set the order of the norm.

        Args:
            ord: order of the norm. Use infinity for the sup norm.
        """
        # Check that ord defines a norm
        if math.isnan(ord) or ord < 1:
            raise ValueError(f"ord must be at least 1, got {ord}")

        # Store the anchored differences and the norm order
        self.anchored = AnchoredDifferences()
        self.ord = ord

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Take the norm of the anchored differences."""
        # Compute the anchored differences: (S, K, p) -> (S, (K - 1) p)
        d = self.anchored(samples)

        # Take the norm over the differences vector: (S, (K - 1) p) -> (S, 1)
        return torch.linalg.vector_norm(d, ord=self.ord, dim=1, keepdim=True)
