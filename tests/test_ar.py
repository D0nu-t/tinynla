import torch
from transformers import AutoModelForCausalLM

from nla import model_adapter as ma
from nla.ar import ActivationReconstructor

TEMPLATE = "<text>{explanation}</text> <summary>"


def test_identity_head_reads_block_output_at_last_token():
    ar = ActivationReconstructor("gpt2", layer=8, template=TEMPLATE).eval()
    lm = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    expl = ["A recipe for bread.", "A much longer explanation about a football match."]
    with torch.no_grad():
        pred = ar(expl)

    # same thing computed by the full target model, one prompt at a time
    for i, e in enumerate(expl):
        ids, _ = ar._encode([e])
        with torch.no_grad(), ma.capture_block_output(lm, 8) as cap:
            lm(input_ids=ids)
        assert torch.allclose(pred[i], cap["hidden"][0, -1], atol=1e-3)


def test_explanation_truncation_keeps_suffix():
    ar = ActivationReconstructor("gpt2", 8, TEMPLATE, max_explanation_tokens=5)
    ids, mask = ar._encode(["word " * 50])
    row = ids[0, : mask[0].sum()].tolist()
    assert row[-len(ar._suffix_ids):] == ar._suffix_ids
    assert len(row) == len(ar._prefix_ids) + 5 + len(ar._suffix_ids)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    ar = ActivationReconstructor("gpt2", 2, TEMPLATE)
    targets = torch.randn(4, 768) * 100
    expl = ["alpha", "beta gamma", "delta epsilon zeta", "eta"]
    opt = torch.optim.AdamW(ar.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        loss = ar.loss(expl, targets, scale=768 ** 0.5)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2 * first
