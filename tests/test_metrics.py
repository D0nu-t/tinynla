import math

import torch

from nla.metrics import fve, rescale, resolve_scale

D = 64


def _data(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    # anisotropic: a shared offset direction, like transformer residuals
    offset = torch.randn(D, generator=g) * 3
    return torch.randn(n, D, generator=g) + offset


def test_resolve_scale():
    assert resolve_scale("sqrt_d_model", 768) == math.sqrt(768)
    assert resolve_scale(2.0, 768) == 2.0
    assert resolve_scale(None, 768) is None


def test_fve_bounds():
    gold = _data()
    s = resolve_scale("sqrt_d_model", D)
    mu = rescale(gold, s).mean(dim=0)

    assert abs(fve(gold, gold, mu, s) - 1.0) < 1e-6
    # the mean itself (un-normalised) scores exactly 0
    mean_pred = mu.expand_as(gold)
    assert abs(fve(mean_pred, gold, mu, s, normalize_pred=False)) < 1e-6


def test_shuffled_predictions_score_below_zero():
    gold = _data()
    s = resolve_scale("sqrt_d_model", D)
    mu = rescale(gold, s).mean(dim=0)
    shuffled = gold[torch.randperm(len(gold), generator=torch.Generator().manual_seed(1))]
    # a real-looking but wrong activation is worse than the mean
    assert fve(shuffled, gold, mu, s) < 0


def test_mean_direction_is_below_zero_for_normalised_predictions():
    """The fair constant baseline under direction-normalised scoring is < 0
    whenever the mean is shorter than the scale (it always is, by Jensen)."""
    gold = _data()
    s = resolve_scale("sqrt_d_model", D)
    mu = rescale(gold, s).mean(dim=0)
    assert mu.norm() < s
    assert fve(mu.expand_as(gold), gold, mu, s) < 0     # normalize_pred=True


def test_cosine_high_but_fve_zero_under_anisotropy():
    """The trap from the Sep-2026 diagnosis: cosine looks good, FVE says no."""
    gold = _data()
    s = resolve_scale("sqrt_d_model", D)
    mu = rescale(gold, s).mean(dim=0)
    const = mu.expand_as(gold)
    cos = torch.nn.functional.cosine_similarity(const, gold, dim=-1).mean()
    assert cos > 0.8
    assert fve(const, gold, mu, s, normalize_pred=False) < 1e-6
