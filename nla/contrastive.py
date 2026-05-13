import torch
import torch.nn.functional as F


def info_nce(
    z_text,
    z_act,
    temperature=0.07
):

    z_text = F.normalize(
        z_text,
        dim=-1
    )

    z_act = F.normalize(
        z_act,
        dim=-1
    )

    logits = (
        z_text @ z_act.T
    ) / temperature

    labels = torch.arange(
        len(z_text)
    ).to(z_text.device)

    return F.cross_entropy(
        logits,
        labels
    )