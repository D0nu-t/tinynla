import torch.nn.functional as F


def sequence_cosine_loss(
    pred,
    target
):

    pred = F.normalize(
        pred,
        dim=-1
    )

    target = F.normalize(
        target,
        dim=-1
    )

    cosine = (
        pred * target
    ).sum(dim=-1)

    return 1 - cosine.mean()