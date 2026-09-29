import torch
from torch import nn

from training.hidden_state_nla import identify, steer


class Block(nn.Module):
    def forward(self, x):
        return (x * 1.0,)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([Block(), Block()])


def test_identify_rank():
    sims = torch.tensor([0.1, 0.9, 0.5])
    assert identify(sims, 1) == {"hit": True, "rank": 1}
    assert identify(sims, 0) == {"hit": False, "rank": 3}


def test_steer_adds_vector_and_cleans_up():
    m = Tiny()
    v = torch.ones(4)
    x = torch.zeros(1, 2, 4)
    with steer(m, 0, v):
        out = m.transformer.h[0](x)[0]
    assert torch.allclose(out, torch.ones(1, 2, 4))
    assert torch.allclose(m.transformer.h[0](x)[0], x)     # hook removed
