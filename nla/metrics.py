import torch
import torch.nn.functional as F


def cosine_similarity_metric(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    return F.cosine_similarity(
        pred,
        target
    ).mean().item()