import math

import pytest
import torch

from nla.ar import ActivationReconstructor
from nla.av import INJECT_TOKEN, ActivationVerbalizer
from nla.rl import GRPOConfig, frozen_reference, group_advantages, grpo_step, rewards_for

PROMPT = "Explain: <concept>" + INJECT_TOKEN + "</concept>\n"
SCALE = math.sqrt(768)


def test_group_advantages_are_zero_mean_unit_std():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, 5.0])
    a = group_advantages(r, group=4, eps=0.0).view(2, 4)
    assert torch.allclose(a.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(a.std(dim=1), torch.ones(2), atol=1e-5)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    av = ActivationVerbalizer("gpt2", PROMPT, "<explanation>", "</explanation>", 120.0)
    ar = ActivationReconstructor("gpt2", 2, "<text>{explanation}</text> <summary>")
    return av, ar


def test_failed_parse_gets_penalty(models):
    _, ar = models
    cfg = GRPOConfig()
    r, _ = rewards_for(ar, ["a", "b"], [True, False], torch.randn(2, 768), SCALE, cfg)
    assert r[1].item() == cfg.failed_reward
    assert -4.0 <= r[0].item() <= 0.0


def test_grpo_step_runs_and_kl_starts_at_zero(models):
    av, ar = models
    ref = frozen_reference(av)
    cfg = GRPOConfig(group_size=3, max_new_tokens=12)
    av_opt = torch.optim.AdamW(av.parameters(), lr=1e-5)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=1e-5)
    acts = torch.randn(2, 768) * 120

    first = grpo_step(av, ref, ar, av_opt, ar_opt, acts, SCALE, cfg)
    assert first["kl"] < 1e-8          # policy == reference before any update
    assert 0.0 <= first["parse_ok"] <= 1.0
    assert math.isfinite(first["reward"])

    second = grpo_step(av, ref, ar, av_opt, ar_opt, acts, SCALE, cfg)
    assert second["kl"] > 0            # the policy has moved
