import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla import model_adapter as ma
from nla.lens import logit_lens, top_words


def test_lens_on_last_block_equals_model_output():
    tok = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    last = ma.num_layers(lm) - 1
    with torch.no_grad(), ma.capture_block_output(lm, last) as cap:
        out = lm(ids)
    lens = logit_lens(lm, cap["hidden"][0, -1])
    assert torch.allclose(lens, out.logits[0, -1].log_softmax(-1), atol=1e-4)

    words = top_words(lens, tok, k=5)
    assert len(words) == 5 and all(w["word"].isalpha() for w in words)
    assert words[0]["prob"] >= words[-1]["prob"]


def test_tuned_lens_is_closer_to_the_final_output_than_logit_lens():
    import pytest
    import torch.nn.functional as F
    from nla.lens import load_tuned_lens, tuned_lens
    params = load_tuned_lens("gpt2")
    if params is None:
        pytest.skip("tuned lens not downloadable (offline)")
    tok = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids = tok("After three days of negotiations, the two parties finally agreed to a ceasefire",
              return_tensors="pt").input_ids
    with torch.no_grad(), ma.capture_block_output(lm, 8) as cap:
        final = lm(ids).logits[0].log_softmax(-1)
    h = cap["hidden"][0]
    kl = lambda lp: F.kl_div(lp, final, log_target=True, reduction="batchmean").item()
    assert kl(tuned_lens(lm, h, 8, params)) < 0.5 * kl(logit_lens(lm, h))
