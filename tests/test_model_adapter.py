import copy

import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from nla import model_adapter as ma

MODEL = "gpt2"
LAYER = 8


@pytest.fixture(scope="module")
def gpt2():
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    return tok, lm


def test_accessors(gpt2):
    _, lm = gpt2
    assert ma.num_layers(lm) == 12
    assert ma.hidden_size(lm) == 768
    assert ma.embed_scale(lm) == 1.0
    assert ma.default_layer(lm) == 8
    assert ma.embed_tokens(lm) is lm.transformer.wte


def test_capture_matches_hidden_states(gpt2):
    tok, lm = gpt2
    ids = tok("Once upon a time there was a cat.", return_tensors="pt").input_ids
    with torch.no_grad(), ma.capture_block_output(lm, LAYER) as cap:
        out = lm(ids, output_hidden_states=True)
    # hidden_states[k + 1] is block k's output (except the last, which has ln_f)
    assert torch.allclose(cap["hidden"], out.hidden_states[LAYER + 1], atol=1e-5)


def test_truncated_base_model_equals_block_output(gpt2):
    tok, lm = gpt2
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids

    with torch.no_grad(), ma.capture_block_output(lm, LAYER) as cap:
        lm(ids)

    base = AutoModel.from_pretrained(MODEL).eval()
    ma.truncate(base, LAYER)
    assert ma.num_layers(base) == LAYER + 1
    with torch.no_grad():
        trunc = base(ids).last_hidden_state

    assert torch.allclose(trunc, cap["hidden"], atol=1e-4)
