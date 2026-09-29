"""
The NLA machinery on one tiny model per family (CPU, seconds each): the
adapter layer, AV/AR construction with and without LoRA, SFT loss, generation,
one GRPO step with the LoRA SFT reference adapter, and save/load.

Tiny random checkpoints are downloaded once (a few MB); tests skip when offline.
"""

import pytest
import torch

from nla import model_adapter as ma
from nla.ar import ActivationReconstructor
from nla.av import INJECT_TOKEN, ActivationVerbalizer
from nla.rl import REF_ADAPTER, GRPOConfig, frozen_reference, grpo_step

FAMILIES = [
    ("sshleifer/tiny-gpt2", None),
    ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}),
]
PROMPT = f"Explain: <concept>{INJECT_TOKEN}</concept>\n"


def _build(name, lora, tmp_path):
    try:
        av = ActivationVerbalizer(name, PROMPT, "<explanation>", "</explanation>", injection_scale=5.0,
                                  max_explanation_tokens=8, lora=lora)
        ar = ActivationReconstructor(name, layer=0, template="<text>{explanation}</text> <summary>",
                                     max_explanation_tokens=8)
    except OSError as e:          # offline and not cached
        pytest.skip(f"{name} unavailable: {e}")
    return av, ar


@pytest.mark.parametrize("name,lora", FAMILIES, ids=[f[0].split("/")[-1] for f in FAMILIES])
def test_family_end_to_end(name, lora, tmp_path):
    torch.manual_seed(0)
    av, ar = _build(name, lora, tmp_path)
    d = ma.hidden_size(av.lm)
    assert ma.blocks(av.lm) is not None and ma.final_norm(av.lm) is not None

    acts = torch.randn(2, d)
    loss = av.sft_loss(acts, ["a short test", "another one"])
    loss.backward()
    assert torch.isfinite(loss)

    outs = av.generate(acts, n_samples=2, max_new_tokens=6)
    assert len(outs) == 4 and all(av.inject_id not in o["token_ids"] for o in outs)
    assert ar(["hello there"]).shape == (1, d)

    # save -> load keeps the embedding table (the LoRA branch used to shrink Qwen's)
    n_emb = av.lm.get_input_embeddings().num_embeddings
    av.save(str(tmp_path / "sft"))
    pol = ActivationVerbalizer.load(str(tmp_path / "sft"))
    assert pol.lm.get_input_embeddings().num_embeddings == n_emb

    ref = frozen_reference(pol, sft_dir=str(tmp_path / "sft"))
    if lora:
        assert ref == REF_ADAPTER                     # no model copy
        ref_w = {k: v.clone() for k, v in pol.lm.named_parameters() if REF_ADAPTER in k}
        assert ref_w and not any(v.requires_grad for k, v in pol.lm.named_parameters() if REF_ADAPTER in k)
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=1e-2)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=1e-3)
    cfg = GRPOConfig(group_size=2, max_new_tokens=6)
    stats = grpo_step(pol, ref, ar, opt, ar_opt, acts, None, cfg)
    assert stats["kl"] < 1e-6                         # policy == SFT reference at step 0
    if lora:
        # the reference adapter must not move; the policy adapter is still the active one
        after = dict(pol.lm.named_parameters())
        assert all(torch.equal(after[k], v) for k, v in ref_w.items())
        assert pol.lm.active_adapter == "default"
        pol.save(str(tmp_path / "rl"))
        saved = {p.name for p in (tmp_path / "rl" / "model").rglob("*")}
        assert REF_ADAPTER not in saved               # only the policy adapter is written


def test_micro_batched_update_matches_full_batch():
    """Chunked policy updates must give the same gradient as one big batch."""
    import copy
    from unittest import mock

    torch.manual_seed(0)
    try:
        av = ActivationVerbalizer("sshleifer/tiny-gpt2", PROMPT, "<explanation>", "</explanation>",
                                  injection_scale=5.0, max_explanation_tokens=8)
        ar = ActivationReconstructor("sshleifer/tiny-gpt2", layer=0, template="<text>{explanation}</text> <summary>")
    except OSError as e:
        pytest.skip(str(e))
    acts = torch.randn(2, ma.hidden_size(av.lm))
    samples = av.generate(acts, n_samples=4, max_new_tokens=6)
    grads = []
    for mb in (None, 4, 3):
        pol = copy.deepcopy(av)
        opt = torch.optim.SGD(pol.parameters(), lr=0.0)          # lr 0: inspect grads only
        with mock.patch.object(pol, "generate", return_value=samples):
            grpo_step(pol, frozen_reference(pol), copy.deepcopy(ar), opt, torch.optim.SGD(ar.parameters(), lr=0.0),
                      acts, None, GRPOConfig(group_size=4, max_new_tokens=6, micro_batch=mb, train_ar=False))
        grads.append(torch.cat([p.grad.flatten() for p in pol.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6) and torch.allclose(grads[0], grads[2], atol=1e-6)


@pytest.mark.parametrize("name", [f[0] for f in FAMILIES], ids=[f[0].split("/")[-1] for f in FAMILIES])
def test_block_output_matches_full_forward(name):
    """Early-stopped extraction gives exactly the state a full forward pass records."""
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(name).eval()
    except OSError as e:
        pytest.skip(str(e))
    ids = torch.randint(0, 100, (2, 7))
    layer = max(0, ma.num_layers(m) - 2)
    with torch.no_grad(), ma.capture_block_output(m, layer) as cap:
        m(input_ids=ids)
    fast = ma.block_output(m, layer, input_ids=ids)
    assert torch.equal(fast, cap["hidden"])
    assert not m._forward_hooks and not ma.blocks(m)[layer]._forward_hooks   # hook removed
