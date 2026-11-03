# Import packages
import torch

from collections.abc import Callable
from .consensus import Consensus
from .discrepancy import Discrepancy
from .regressor import Regressor

class PMC:
    """Product Monte Carlo estimator from a consensus, a discrepancy, a transform, and a regressor."""

    def __init__(self, consensus: Consensus, discrepancy: Discrepancy, h: Callable[[torch.Tensor], torch.Tensor], regressor: Regressor):
        """Set the components of the estimator.

        Args:
            consensus: the consensus function.
            discrepancy: the discrepancy function.
            h: callable applied to the consensus.
            regressor: the regressor.
        """
        # Check that consensus is a consensus
        if not isinstance(consensus, Consensus):
            raise TypeError("consensus must be a Consensus")

        # Check that discrepancy is a discrepancy
        if not isinstance(discrepancy, Discrepancy):
            raise TypeError("discrepancy must be a Discrepancy")

        # Check that h is callable
        if not callable(h):
            raise TypeError("h must be callable")

        # Check that regressor is a regressor
        if not isinstance(regressor, Regressor):
            raise TypeError("regressor must be a Regressor")

        self.consensus = consensus
        self.discrepancy = discrepancy
        self.h = h
        self.regressor = regressor

    def estimate(self, samples: torch.Tensor) -> torch.Tensor:
        """Estimate E[h(c(X_1, ..., X_K)) | d(X_1, ..., X_K) = 0] from the samples.

        Args:
            samples: S draws of K components, a tensor of shape (S, K, p).

        Returns:
            Tensor of shape (r,).
        """
        # Compute the consensus and the discrepancy: (S, K, p) -> (S, p) and (S, m)
        c = self.consensus(samples)
        d = self.discrepancy(samples)

        # Apply the transform to the consensus: (S, p) -> (S, r)
        h_c = self.h(c)

        # Fit the regressor to predict the transformed consensus from the discrepancy
        self.regressor.fit(d, h_c)

        # Evaluate the fitted regressor at d = 0
        return self.regressor.predict()
