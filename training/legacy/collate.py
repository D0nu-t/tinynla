# training/collate.py

from torch.nn.utils.rnn import pad_sequence
import torch


def sequence_collate(batch):
    descriptions = [x["description"] for x in batch]

    activations = [
        x["activation"]
        for x in batch
    ]

    lengths = torch.tensor(
        [a.shape[0] for a in activations]
    )

    padded = pad_sequence(
        activations,
        batch_first=True,
    )

    mask = torch.arange(
        padded.size(1)
    )[None, :] < lengths[:, None]

    return {
        "description": descriptions,
        "activation": padded,
        "mask": mask,
        "lengths": lengths,
    }