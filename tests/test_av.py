import math

import pytest
import torch

from nla.av import INJECT_TOKEN, ActivationVerbalizer

PROMPT = "Explain: <concept>" + INJECT_TOKEN + "</concept>\n"


@pytest.fixture(scope="module")
def av():
    torch.manual_seed(0)
    return ActivationVerbalizer(
        "gpt2", PROMPT, "<explanation>", "</explanation>",
        injection_scale=math.sqrt(768),  # tests don't depend on the chosen scale
    ).eval()


def test_injection_writes_scaled_vector_at_marker(av):
    v = torch.randn(3, 768) * 120
    emb = av.embed_prompts(v)
    injected = emb[:, av.inject_pos]
    assert torch.allclose(injected.norm(dim=-1), torch.full((3,), math.sqrt(768)), atol=1e-3)
    assert torch.allclose(
        torch.nn.functional.normalize(injected, dim=-1),
        torch.nn.functional.normalize(v, dim=-1), atol=1e-5,
    )
    # every other position is the plain token embedding
    ids = torch.tensor(av.prompt_ids)
    plain = av.lm.get_input_embeddings()(ids)
    others = [i for i in range(len(ids)) if i != av.inject_pos]
    assert torch.allclose(emb[0, others], plain[others])


def test_sft_loss_learns_vector_dependent_text(av):
    """Two different vectors -> two different explanations must be learnable."""
    torch.manual_seed(0)
    av.train()
    v = torch.randn(2, 768) * 120
    expl = ["A cooking recipe for bread.", "A football match report."]
    opt = torch.optim.AdamW(av.parameters(), lr=5e-4)
    for _ in range(40):
        loss = av.sft_loss(v, expl)
        opt.zero_grad(); loss.backward(); opt.step()
    av.eval()
    outs = av.generate(v, n_samples=1, temperature=0)
    assert [o["text"] for o in outs] == expl
    assert all(o["ok"] for o in outs)
