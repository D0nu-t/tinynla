import torch

from nla.unusual import UnusualnessModel, unusual_summary


def test_flags_outliers_not_ordinary_points(tmp_path):
    g = torch.Generator().manual_seed(0)
    train = torch.randn(3000, 16, generator=g) * torch.linspace(0.5, 3, 16)
    val = torch.randn(1000, 16, generator=g) * torch.linspace(0.5, 3, 16)
    m = UnusualnessModel.fit(train, val)
    test = torch.randn(1000, 16, generator=g) * torch.linspace(0.5, 3, 16)
    rate = sum(a["flag"] for a in m.assess(test)) / 1000
    assert rate < 0.03                                  # ~1% by construction
    # an outlier along a LOW-variance direction is unusual even at a small norm
    odd = torch.zeros(1, 16)
    odd[0, 0] = 4.0
    assert m.assess(odd)[0]["flag"]
    # the same step along the highest-variance direction is ordinary
    ok = torch.zeros(1, 16)
    ok[0, -1] = 4.0
    assert not m.assess(ok)[0]["flag"]

    p = tmp_path / "u.pt"
    m.save(p)
    m2 = UnusualnessModel.load(p)
    assert abs(m2.threshold - m.threshold) < 1e-6


def test_summary_only_when_flagged():
    assert unusual_summary([False, False], 0.99) is None
    assert unusual_summary([True, False, False, False], 0.99) is None     # one flag: chance level
    s = unusual_summary([True, True, False, False], 0.99)
    assert s["n_flagged"] == 2 and "UNVERIFIED" in s["text"]
