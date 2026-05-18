"""
nla/dataset.py

Dataset classes, validation utilities, and collators for pooled and
sequence activation buffers.

Primary v3 path:
    SequenceActivationDataset
    sequence_collate

Legacy pooled path:
    ActivationDataset
    pooled_collate

Key upgrades:
  - Strict validation of activation tensors
  - Automatic dtype normalization
  - Sequence truncation support
  - Dataset statistics helpers
  - Safer padding/collation
  - Optional memory-efficient loading preparation
"""

import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ===========================================================================
# Validation helpers
# ===========================================================================

def _validate_tensor(
    tensor: torch.Tensor,
    expected_dim: int,
    name: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if tensor.dim() != expected_dim:
        raise ValueError(
            f"{name} must have dim={expected_dim}, "
            f"got shape={tuple(tensor.shape)}"
        )

    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty")

    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _ensure_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.contiguous()


# ===========================================================================
# Legacy pooled dataset
# ===========================================================================

class ActivationDataset(Dataset):
    """
    Legacy pooled activation dataset.

    Each sample:
        description: str
        activation:  Tensor[hidden_dim]
    """

    def __init__(self, path: str):
        self.samples = torch.load(path, weights_only=False)

        if len(self.samples) == 0:
            raise ValueError("Dataset is empty")

        first = self.samples[0]

        if "activation" not in first:
            raise ValueError(
                "Buffer missing 'activation'. "
                "Expected pooled activation dataset."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]

        activation = _ensure_float32(item["activation"])
        _validate_tensor(
            activation,
            expected_dim=1,
            name="activation",
        )

        return {
            "description": item["description"],
            "activation": activation,
        }

    @property
    def hidden_dim(self) -> int:
        return self[0]["activation"].shape[-1]


# ===========================================================================
# Sequence dataset (v3 primary)
# ===========================================================================

class SequenceActivationDataset(Dataset):
    """
    Token-level activation trajectory dataset.

    Each sample:
        description:         str
        activation_sequence: Tensor[seq_len, hidden_dim]
        seq_len:             int

    Supports:
      - variable-length trajectories
      - optional truncation
      - strict validation
      - sequence statistics
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
    ):
        self.samples = torch.load(path, weights_only=False)

        if len(self.samples) == 0:
            raise ValueError("Dataset is empty")

        first = self.samples[0]

        if "activation_sequence" not in first:
            raise ValueError(
                "Buffer missing 'activation_sequence'. "
                "Rebuild with sequence extraction enabled."
            )

        self.max_seq_len = max_seq_len

        self._hidden_dim = first["activation_sequence"].shape[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]

        seq = item["activation_sequence"]

        _validate_tensor(
            seq,
            expected_dim=2,
            name="activation_sequence",
        )

        seq = _ensure_float32(seq)

        if self.max_seq_len is not None:
            seq = seq[: self.max_seq_len]

        seq_len = seq.shape[0]

        if seq_len < 1:
            raise ValueError("Sequence length must be >= 1")

        return {
            "description": item["description"],
            "activation_sequence": seq,
            "seq_len": seq_len,
        }

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def sequence_lengths(self) -> List[int]:
        return [
            min(
                s["activation_sequence"].shape[0],
                self.max_seq_len or 10**9,
            )
            for s in self.samples
        ]

    def stats(self) -> Dict:
        lengths = self.sequence_lengths

        return {
            "num_samples": len(self.samples),
            "hidden_dim": self.hidden_dim,
            "min_seq_len": min(lengths),
            "max_seq_len": max(lengths),
            "mean_seq_len": sum(lengths) / len(lengths),
        }


# ===========================================================================
# Collators
# ===========================================================================

def pooled_collate(batch: List[Dict]) -> Dict:
    """
    Standard collator for pooled activations.
    """

    activations = torch.stack(
        [x["activation"] for x in batch]
    )

    return {
        "texts": [x["description"] for x in batch],
        "activations": activations,
    }


def sequence_collate(batch: List[Dict]) -> Dict:
    """
    Collator for variable-length activation trajectories.

    Pads sequences to batch max length.

    Returns:
        texts:
            List[str]

        activation_sequences:
            Tensor[batch, max_seq_len, hidden_dim]

        seq_lens:
            Tensor[batch]

        mask:
            BoolTensor[batch, max_seq_len]
            True at valid positions
    """

    if len(batch) == 0:
        raise ValueError("Empty batch")

    texts = [x["description"] for x in batch]

    seqs = [
        _ensure_float32(x["activation_sequence"])
        for x in batch
    ]

    seq_lens = [s.shape[0] for s in seqs]

    max_len = max(seq_lens)
    hidden_dim = seqs[0].shape[-1]
    batch_size = len(seqs)

    padded = torch.zeros(
        batch_size,
        max_len,
        hidden_dim,
        dtype=torch.float32,
    )

    mask = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    for i, seq in enumerate(seqs):
        length = seq.shape[0]

        padded[i, :length] = seq
        mask[i, :length] = True

    return {
        "texts": texts,
        "activation_sequences": padded,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "mask": mask,
    }


# ===========================================================================
# Serialization
# ===========================================================================

def save_dataset(
    samples: list,
    output_path: str,
) -> None:
    """
    Save dataset safely.
    """

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(samples, output_path)


def load_dataset_file(path: str):
    """
    Convenience wrapper around torch.load().
    """

    return torch.load(path, weights_only=False)