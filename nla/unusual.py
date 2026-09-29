"""
nla/unusual.py

"Is the model in an unusual internal state here?"

Why it exists: the report's reliability check asks whether the reconstructor can
rebuild the state from the explanation. For states far from anything the NLA was
trained on it cannot, however right the words are: under activation steering every
read named the planted concept yet all were marked unreliable (/loop iteration 9).
So an unusual state must be flagged explicitly, not silently reported as "no signal".

Method: Mahalanobis distance to the training activations with a Ledoit-Wolf shrunk
covariance - a simple, strong detector of out-of-distribution inputs from hidden
states (Lee et al., NeurIPS 2018; for language models, Ren et al., ICLR 2023).
Scores are reported as percentiles of the held-out validation distances, and a
state is flagged above the validation `flag_quantile` (default 99th percentile),
so about 1 in 100 ordinary states is flagged by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


class UnusualnessModel:
    def __init__(self, mean: torch.Tensor, precision: torch.Tensor, val_dist: torch.Tensor,
                 flag_quantile: float = 0.99):
        self.mean, self.precision = mean.float(), precision.float()
        self.val_dist = val_dist.float().sort().values
        self.threshold = torch.quantile(self.val_dist, flag_quantile).item()
        self.flag_quantile = flag_quantile

    @classmethod
    def fit(cls, train: torch.Tensor, val: torch.Tensor, max_fit: int = 10000,
            flag_quantile: float = 0.99, seed: int = 0) -> "UnusualnessModel":
        from sklearn.covariance import LedoitWolf

        g = torch.Generator().manual_seed(seed)
        x = train[torch.randperm(len(train), generator=g)[:max_fit]].double()
        lw = LedoitWolf().fit(x.numpy())
        mean = torch.from_numpy(lw.location_)
        precision = torch.from_numpy(lw.precision_)
        tmp = cls(mean, precision, torch.zeros(1))
        return cls(mean, precision, tmp.distance(val), flag_quantile)

    def distance(self, h: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance of each row of h [..., d]."""
        z = h.float().cpu() - self.mean
        return torch.einsum("...i,ij,...j->...", z, self.precision, z).clamp(min=0).sqrt()

    def percentile(self, dist: torch.Tensor) -> torch.Tensor:
        """Share of ordinary (validation) states that are LESS unusual than this one."""
        return torch.searchsorted(self.val_dist, dist.float().contiguous()) / len(self.val_dist)

    def assess(self, h: torch.Tensor) -> list:
        d = self.distance(h.reshape(-1, h.shape[-1]))
        p = self.percentile(d)
        return [{"distance": round(a, 2), "percentile": round(b, 4), "flag": a > self.threshold}
                for a, b in zip(d.tolist(), p.tolist())]

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        torch.save({"mean": self.mean, "precision": self.precision, "val_dist": self.val_dist,
                    "flag_quantile": self.flag_quantile}, path)

    @classmethod
    def load(cls, path) -> Optional["UnusualnessModel"]:
        if not Path(path).exists():
            return None
        s = torch.load(path, weights_only=True)
        return cls(s["mean"], s["precision"], s["val_dist"], s["flag_quantile"])

    @classmethod
    def for_data(cls, data, cache_path) -> "UnusualnessModel":
        """Load the cached model for this dataset, or fit and cache it (a few seconds)."""
        m = cls.load(cache_path)
        if m is None:
            m = cls.fit(data.acts[data.split["train"]], data.acts[data.split["val"]])
            m.save(cache_path)
        return m


UNUSUAL_TEXT = ("The model was in an unusual internal state for {n} of {total} words read "
                "(further from typical than {q:.0%} of ordinary states). Our reliability check cannot "
                "verify explanations of unusual states, so those reads are shown as UNVERIFIED rather "
                "than dropped: they may be the most informative part, or they may be wrong.")


def unusual_summary(flags: list, flag_quantile: float, min_flagged: int = 2) -> Optional[Dict]:
    """
    Banner only when at least `min_flagged` words are flagged. Each ordinary word is
    flagged ~1.4% of the time, so with one flag enough, ~1 in 5 ordinary 16-word
    answers showed the banner (15% measured); requiring 2 gives ~2% (binomial).
    """
    n = sum(flags)
    if n < min_flagged:
        return None
    return {"n_flagged": n, "n_tokens": len(flags), "share": round(n / len(flags), 3),
            "text": UNUSUAL_TEXT.format(n=n, total=len(flags), q=flag_quantile)}
