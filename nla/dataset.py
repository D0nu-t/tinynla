"""
nla/dataset.py

Dataset classes and collators for both pooled and sequence activation buffers.

ActivationDataset         — legacy pooled dataset; each sample has activation [hidden_dim]
SequenceActivationDataset — v3 sequence dataset; each sample has activation [seq_len, hidden_dim]

sequence_collate          — pads variable-length sequences to batch max_seq_len;
                            returns a padding mask for use in masked losses
"""

import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset


# ===========================================================================
# Legacy pooled dataset
# ===========================================================================

class ActivationDataset(Dataset):
    """
    Pooled activation dataset. Each sample:
        description: str
        activation:  Tensor[hidden_dim]

    Used with training.reconstructor_type = "pooled_mlp".
    """

    def __init__(self, path: str):
        self.samples = torch.load(path, weights_only=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "description": item["description"],
            "activation": item["activation"],
        }


# ===========================================================================
# Sequence dataset
# ===========================================================================

class SequenceActivationDataset(Dataset):
    """
    Sequence activation dataset. Each sample:
        description:         str
        activation_sequence: Tensor[seq_len, hidden_dim]
        seq_len:             int

    Used with training.reconstructor_type = "token_decoder".

    The seq_len varies across samples — use sequence_collate to batch.
    """

    def __init__(self, path: str):
        self.samples = torch.load(path, weights_only=False)
        # Validate that samples contain sequence activations
        if self.samples and "activation_sequence" not in self.samples[0]:
            raise ValueError(
                "Buffer does not contain 'activation_sequence'. "
                "Rebuild with activation.pooling='sequence'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        seq = item["activation_sequence"]   # [seq_len, hidden_dim]
        return {
            "description": item["description"],
            "activation_sequence": seq,
            "seq_len": seq.shape[0],
        }


# ===========================================================================
# Collators
# ===========================================================================

def pooled_collate(batch: List[Dict]) -> Dict:
    """Standard collator for pooled activations."""
    return {
        "texts": [x["description"] for x in batch],
        "activations": torch.stack([x["activation"] for x in batch]),
    }


def sequence_collate(batch: List[Dict]) -> Dict:
    """
    Collator for variable-length sequence activations.

    Pads all sequences in the batch to the batch max seq_len.
    Returns a boolean mask: True at valid positions, False at padding.

    Returns:
        texts:               List[str]  — descriptions
        activation_sequences: Tensor[batch, max_seq_len, hidden_dim]
        seq_lens:            Tensor[batch]  — original lengths before padding
        mask:                Tensor[batch, max_seq_len]  — True at valid positions
    """
    texts = [x["description"] for x in batch]
    seqs = [x["activation_sequence"] for x in batch]   # list of [seq_i, hidden]
    seq_lens = [s.shape[0] for s in seqs]
    max_len = max(seq_lens)
    hidden_dim = seqs[0].shape[-1]

    padded = torch.zeros(len(seqs), max_len, hidden_dim)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)

    for i, (s, l) in enumerate(zip(seqs, seq_lens)):
        padded[i, :l] = s
        mask[i, :l] = True

    return {
        "texts": texts,
        "activation_sequences": padded,           # [batch, max_len, hidden]
        "seq_lens": torch.tensor(seq_lens),
        "mask": mask,                             # [batch, max_len]
    }


# ===========================================================================
# Serialization
# ===========================================================================

def save_dataset(samples: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)