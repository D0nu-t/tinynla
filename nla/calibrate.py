"""
nla/calibrate.py

Plain-language "signal strength" for a self-check FVE (stakeholder report, idea 3).

Raw FVE is meaningless to a non-specialist (and negative values look like
failure even when informative - see wiki nla_touchstone). Instead each
explanation's per-sample FVE is placed against two reference distributions
measured once on validation activations:

    shuffled   FVE of explanations paired with the WRONG activation
    own        FVE of explanations paired with their own activation

    none      at or below the 95th percentile of `shuffled`
              - indistinguishable from a description of some other text
    weak      above that, below the median of `own`
    moderate  between the `own` median and 90th percentile
    strong    at or above the `own` 90th percentile

Validity is checked by test-retest: the bucket of one sample must predict
the FVE of an INDEPENDENT second sample for the same activation
(`retest_by_bucket`). Buckets that don't predict a fresh sample would be
decoration, not information.

Over-trust matters here: explanations raise acceptance even when wrong
(Bansal et al., CHI 2021) and practitioners over-trust interpretability
tools (Kaur et al., CHI 2020), so "none" is shown prominently, not hidden.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BUCKETS = ["none", "weak", "moderate", "strong"]
CALIBRATION_FILE = "calibration.json"

BUCKET_TEXT = {
    "none": "no reliable signal: no better than a description of some other text",
    "weak": "weak signal: fewer reliable reads than this tool usually gets",
    "moderate": "moderate signal: about as many reliable reads as this tool usually gets",
    "strong": "strong signal: more reliable reads than this tool usually gets",
    "uncalibrated": "not calibrated yet: run python -m training.calibrate_nla",
}


@dataclass
class Calibration:
    shuffled_p95: float
    own_p50: float
    own_p90: float
    n: int
    retest: Optional[Dict[str, Dict[str, float]]] = None

    @classmethod
    def from_samples(cls, own: Sequence[float], shuffled: Sequence[float]) -> "Calibration":
        own, shuffled = np.asarray(own, float), np.asarray(shuffled, float)
        return cls(
            shuffled_p95=float(np.quantile(shuffled, 0.95)),
            own_p50=float(np.quantile(own, 0.50)),
            own_p90=float(np.quantile(own, 0.90)),
            n=int(len(own)),
        )

    def bucket(self, fve: float) -> str:
        if fve <= self.shuffled_p95:
            return "none"
        if fve < self.own_p50:
            return "weak"
        if fve < self.own_p90:
            return "moderate"
        return "strong"

    @property
    def reliable_rate(self) -> Optional[float]:
        """Share of single reads above 'none' in calibration (the tool's typical rate)."""
        if not self.retest:
            return None
        total = sum(r["n"] for r in self.retest.values())
        return 1 - self.retest["none"]["n"] / total if total else None

    def save(self, directory: str) -> Path:
        path = Path(directory) / CALIBRATION_FILE
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str) -> Optional["Calibration"]:
        path = Path(directory) / CALIBRATION_FILE
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def retest_by_bucket(cal: Calibration, first: Sequence[float], second: Sequence[float]) -> Dict:
    """Mean FVE of an independent second sample, grouped by the first sample's bucket."""
    groups: Dict[str, List[float]] = {b: [] for b in BUCKETS}
    for a, b in zip(first, second):
        groups[cal.bucket(a)].append(b)
    return {
        b: {"n": len(v), "second_sample_fve_mean": float(np.mean(v)) if v else None}
        for b, v in groups.items()
    }


def is_monotonic(retest: Dict, min_n: int = 5) -> bool:
    means = [retest[b]["second_sample_fve_mean"] for b in BUCKETS
             if retest[b]["n"] >= min_n]
    return all(x < y for x, y in zip(means, means[1:]))


def answer_verdict(reliable_tokens: int, total_tokens: int, typical_rate: Optional[float]) -> str:
    """
    Answer-level verdict from the share of tokens with at least one reliable read,
    relative to this tool's typical rate. Counting (not a median of noisy labels)
    keeps the verdict stable across re-runs.
    """
    if total_tokens == 0 or reliable_tokens == 0:
        return "none"
    return _label_for_share(reliable_tokens / total_tokens, typical_rate)


def _label_for_share(share: float, typical_rate: Optional[float]) -> str:
    base = typical_rate or 0.35
    if share < 0.5 * base:
        return "weak"
    if share < 1.5 * base:
        return "moderate"
    return "strong"


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (well behaved for small n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def verdict_range(reliable_tokens: int, total_tokens: int,
                  typical_rate: Optional[float]) -> Dict:
    """
    Verdict plus the labels at both ends of the 95% Wilson interval on the
    reliable-token share. If they differ, the honest statement is a range
    ("moderate to strong"): with ~20 words, a single label is not supported.
    """
    point = answer_verdict(reliable_tokens, total_tokens, typical_rate)
    if point == "none":
        return {"point": "none", "low": "none", "high": "none", "interval": (0.0, 0.0), "is_range": False}
    lo, hi = wilson_interval(reliable_tokens, total_tokens)
    low, high = _label_for_share(lo, typical_rate), _label_for_share(hi, typical_rate)
    return {"point": point, "low": low, "high": high,
            "interval": (round(lo, 3), round(hi, 3)), "is_range": low != high}


def overall_bucket(buckets: Sequence[str]) -> str:
    """Median bucket of an answer's tokens (robust to a few lucky reads)."""
    if not buckets:
        return "none"
    ranks = sorted(BUCKETS.index(b) for b in buckets if b in BUCKETS)
    return BUCKETS[ranks[len(ranks) // 2]] if ranks else "uncalibrated"
