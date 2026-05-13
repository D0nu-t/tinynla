import torch
import torch.nn.functional as F


def kl_divergence(logits_a, logits_b):

    p = F.log_softmax(logits_a, dim=-1)

    q = F.softmax(logits_b, dim=-1)

    return F.kl_div(
        p,
        q,
        reduction="batchmean"
    ).item()


def topk_overlap(logits_a, logits_b, k=10):

    top_a = torch.topk(
        logits_a,
        k=k,
        dim=-1
    ).indices

    top_b = torch.topk(
        logits_b,
        k=k,
        dim=-1
    ).indices

    overlap = (
        (top_a == top_b)
        .float()
        .mean()
        .item()
    )

    return overlap


def logit_cosine_similarity(
    logits_a,
    logits_b
):

    return F.cosine_similarity(
        logits_a,
        logits_b,
        dim=-1
    ).mean().item()