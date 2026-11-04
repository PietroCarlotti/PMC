# Import packages
import math
import torch

from abc import ABC, abstractmethod

class Transform(ABC):
    """Base class for transforms h: X -> R^r, applied to the consensus."""

    @abstractmethod
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Map each consensus point to R^r.

        Args:
            c: S draws of the consensus, a tensor of shape (S, p).

        Returns:
            Tensor of shape (S, r).
        """

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the consensus.

        Args:
            c: S draws of the consensus, a tensor of shape (S, p).

        Returns:
            Tensor of shape (S, r).
        """
        # Check that the input is a tensor
        if not isinstance(c, torch.Tensor):
            raise TypeError("c must be a torch.Tensor")

        # Check that the input has shape (S, p)
        if c.ndim != 2:
            raise ValueError(f"c must have shape (S, p), got {tuple(c.shape)}")

        # Unpack the number of draws S and the dimension p
        S, p = c.shape

        # Check that there is at least one draw and one coordinate
        if S < 1 or p < 1:
            raise ValueError(f"S and p must be at least 1, got S={S}, p={p}")

        # Check that the input is a floating-point tensor
        if not c.is_floating_point():
            raise ValueError(f"c must be floating point, got dtype {c.dtype}")

        # Apply the subclass's transform
        output = self.forward(c)

        # Check that the output is a tensor
        if not isinstance(output, torch.Tensor):
            raise TypeError("transform must return a torch.Tensor")

        # Check that the output has shape (S, r) for r at least 1
        if output.ndim != 2 or output.shape[0] != S or output.shape[1] < 1:
            raise ValueError(
                f"transform must return shape (S, r) with S = {S} and r at least 1, "
                f"got {tuple(output.shape)}"
            )

        # Check that the output has the same device and dtype as the input
        if output.device != c.device or output.dtype != c.dtype:
            raise ValueError(
                f"transform must return device {c.device} and dtype {c.dtype}, "
                f"got device {output.device} and dtype {output.dtype}"
            )
        return output


class Identity(Transform):
    """Identity transform h(x) = x, with r = p."""

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Return the consensus unchanged."""
        # Return the consensus unchanged: (S, p) -> (S, p)
        return c


class HadamardProduct(Transform):
    """Elementwise square h(x)_j = x_j^2 for j = 1, ..., p, with r = p."""

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Square each coordinate."""
        # Square each coordinate: (S, p) -> (S, p)
        return c ** 2


class InnerProduct(Transform):
    """Inner product of x with itself h(x) = x^T x, with r = 1."""

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Sum the squared coordinates."""
        # Sum the squared coordinates: (S, p) -> (S, 1)
        return (c ** 2).sum(dim=1, keepdim=True)


class OuterProduct(Transform):
    """Outer product h(x) = x x^T, with r = p^2."""

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Take the outer product of each point with itself."""
        # Take the outer product of each draw with itself: (S, p) -> (S, p, p)
        outer = c.unsqueeze(2) * c.unsqueeze(1)

        # Flatten the matrix into one vector: (S, p, p) -> (S, p^2)
        return outer.flatten(start_dim=1)


class Indicator(Transform):
    """Base class for indicators h(x) = 1{x in A} of a set A."""

    @abstractmethod
    def contains(self, c: torch.Tensor) -> torch.Tensor:
        """Check which draws lie in the set.

        Args:
            c: S draws of the consensus, a tensor of shape (S, p).

        Returns:
            Boolean tensor of shape (S, r).
        """

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Return 1.0 for points in the set and 0.0 for the others."""
        # Check which draws lie in the set
        contains = self.contains(c)

        # Check that membership indicators are tensors
        if not isinstance(contains, torch.Tensor):
            raise TypeError("indicator membership must be a torch.Tensor")

        # Check that membership indicators are Boolean
        if contains.dtype != torch.bool:
            raise ValueError(
                f"indicator membership must have dtype torch.bool, got {contains.dtype}"
            )

        # Convert the booleans to floats: (S, p) -> (S, r)
        return contains.to(c.dtype)


class IndicatorThreshold(Indicator):
    """Coordinatewise threshold h(x)_j = 1{x_j > t_j} for j = 1, ..., p, with r = p."""

    def __init__(self, thresholds: torch.Tensor):
        """Set the thresholds.

        Args:
            thresholds: one threshold for each coordinate, a tensor of shape (p,).
        """
        thresholds = torch.as_tensor(thresholds)

        # Check that the thresholds form a nonempty vector
        if thresholds.ndim != 1 or thresholds.shape[0] < 1:
            raise ValueError(
                f"thresholds must have shape (p,) for p at least 1, "
                f"got {tuple(thresholds.shape)}"
            )

        # Check that the thresholds have a floating-point dtype
        if not thresholds.is_floating_point():
            raise ValueError(
                f"thresholds must be floating point, got dtype {thresholds.dtype}"
            )

        # Check that the thresholds are finite
        if not torch.isfinite(thresholds).all():
            raise ValueError("thresholds must be finite")

        self.thresholds = thresholds

    def contains(self, c: torch.Tensor) -> torch.Tensor:
        """Check which coordinates exceed their thresholds."""
        # Check that there is one threshold per coordinate
        p = c.shape[1]
        if self.thresholds.shape[0] != p:
            raise ValueError(f"expected {self.thresholds.shape[0]} coordinates, got p={p}")

        # Compare each coordinate with its threshold: (S, p) -> (S, p)
        return c > self.thresholds.to(c)


class IndicatorBall(Indicator):
    """Euclidean ball h(x) = 1{||x - a||_2 <= rho}, with r = 1."""

    def __init__(self, center: torch.Tensor, radius: float):
        """Set the center and the radius.

        Args:
            center: the center of the ball, a tensor of shape (p,).
            radius: radius of the ball.
        """
        center = torch.as_tensor(center)

        # Check that the center is a nonempty vector
        if center.ndim != 1 or center.shape[0] < 1:
            raise ValueError(
                f"center must have shape (p,) for p at least 1, got {tuple(center.shape)}"
            )

        # Check that the center is finite
        if not torch.isfinite(center).all():
            raise ValueError("center must be finite")

        # Check that the radius is finite and positive
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"radius must be finite and positive, got {radius}")

        self.center = center
        self.radius = radius

    def contains(self, c: torch.Tensor) -> torch.Tensor:
        """Check which points lie within the radius of the center."""
        # Check that the center has one entry per coordinate
        p = c.shape[1]
        if self.center.shape[0] != p:
            raise ValueError(f"expected {self.center.shape[0]} coordinates, got p={p}")

        # Compute the distance of each draw to the center: (S, p) -> (S, 1)
        distance = torch.linalg.vector_norm(c - self.center.to(c), dim=1, keepdim=True)

        # Compare the distance with the radius: (S, 1) -> (S, 1)
        return distance <= self.radius


class IndicatorBox(Indicator):
    """Coordinatewise interval h(x)_j = 1{l_j <= x_j <= u_j} for j = 1, ..., p, with r = p."""

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        """Set the lower and upper bounds.

        Args:
            lower: the lower bound of each coordinate, a tensor of shape (p,).
            upper: the upper bound of each coordinate, a tensor of shape (p,).
        """
        lower = torch.as_tensor(lower)
        upper = torch.as_tensor(upper)

        # Check that the bounds are nonempty vectors of the same shape
        if lower.ndim != 1 or lower.shape != upper.shape or lower.shape[0] < 1:
            raise ValueError(
                f"lower and upper must have the same shape (p,) for p at least 1, "
                f"got {tuple(lower.shape)} and {tuple(upper.shape)}"
            )

        # Check that the bounds are finite
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("lower and upper must be finite")

        # Check that the lower bounds do not exceed the upper bounds
        if (lower > upper).any():
            raise ValueError("lower must not exceed upper in any coordinate")

        self.lower = lower
        self.upper = upper

    def contains(self, c: torch.Tensor) -> torch.Tensor:
        """Check which coordinates lie within their bounds."""
        # Check that there is one pair of bounds per coordinate
        p = c.shape[1]
        if self.lower.shape[0] != p:
            raise ValueError(f"expected {self.lower.shape[0]} coordinates, got p={p}")

        # Compare each coordinate with its bounds: (S, p) -> (S, p)
        return (c >= self.lower.to(c)) & (c <= self.upper.to(c))
