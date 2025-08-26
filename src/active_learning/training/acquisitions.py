from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import torch


class Acquisition(ABC):
    requires_mc: bool = False

    def fit(self, **kwargs) -> "Acquisition":
        return self

    @abstractmethod
    def score(
        self, *, mean_probs: torch.Tensor, mc_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Score samples for acquisition priority.

        Args:
            mean_probs: [N,K] mean predictive probabilities
            mc_probs: [T,N,K] per-sample probs over T stochastic passes
        Returns:
            scores: [N] higher scores = acquire first
        """
        raise NotImplementedError


def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Compute entropy along last dimension."""
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1)


class BALD(Acquisition):
    """Bayesian Active Learning by Disagreement: H[mean] - mean_t H[p_t]"""

    requires_mc = True

    def score(
        self, *, mean_probs: torch.Tensor, mc_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mc_probs is None:
            raise ValueError("BALD requires mc_probs with shape [T,N,K]")
        return _entropy(mean_probs) - _entropy(mc_probs).mean(dim=0)


class EntropyUncertainty(Acquisition):
    """Predictive entropy of the mean distribution."""

    def score(
        self, *, mean_probs: torch.Tensor, mc_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return _entropy(mean_probs)


class LeastConfidence(Acquisition):
    """Uncertainty as 1 - max class probability."""

    def score(
        self, *, mean_probs: torch.Tensor, mc_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return 1.0 - mean_probs.max(dim=-1).values


class Margin(Acquisition):
    """Uncertainty as negative margin between top-2 class probabilities."""

    def score(
        self, *, mean_probs: torch.Tensor, mc_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        top2 = torch.topk(mean_probs, k=2, dim=-1).values
        return (top2[:, 0] - top2[:, 1]).neg()


class TieCloseness(Acquisition):
    """For binary classification: uncertainty when probabilities are close to 0.5."""

    def score(
        self, *, mean_probs: torch.Tensor, mc_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mean_probs.size(-1) != 2:
            raise ValueError("TieCloseness expects binary classification (K=2)")
        return (mean_probs[:, 0] - 0.5).abs().neg()


class CoresetKCenter(Acquisition):
    """K-center coreset method: select samples farthest from current labeled set."""

    def __init__(self, embed_dim: Optional[int] = None):
        self.embed_dim = embed_dim
        self.bank: Optional[torch.Tensor] = None

    def fit(self, *, labeled_embeds: torch.Tensor) -> "CoresetKCenter":  # type: ignore
        """Fit on embeddings of current labeled set."""
        self.bank = labeled_embeds
        if self.embed_dim is None:
            self.embed_dim = labeled_embeds.size(-1)
        return self

    def score(
        self,
        *,
        mean_probs: torch.Tensor,
        mc_probs: Optional[torch.Tensor] = None,
        candidate_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.bank is None or candidate_embeds is None:
            raise ValueError("Must call fit() first and provide candidate_embeds")

        dists = torch.cdist(candidate_embeds, self.bank)
        return dists.min(dim=1).values


_REGISTRY: Dict[str, Type[Acquisition]] = {
    "bald": BALD,
    "entropy": EntropyUncertainty,
    "least_confidence": LeastConfidence,
    "margin": Margin,
    "tie": TieCloseness,
    "coreset_kcenter": CoresetKCenter,
}


def make_acquisition(name: str, **kwargs) -> Acquisition:
    """Factory function to create acquisition strategies by name."""
    key = name.lower()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(f"Unknown acquisition '{name}'. Available: {available}")
    return _REGISTRY[key](**kwargs)
