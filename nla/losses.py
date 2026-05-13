import torch
import torch.nn.functional as F


def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    sim = F.cosine_similarity(pred, target)

    return 1 - sim.mean()