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

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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

    def position_mean(
        self,
        indices: Sequence[int],
        max_len: int,
    ) -> torch.Tensor:
        """
        Per-position mean trajectory over the given samples.

        This is the "mean ablation" baseline: the best a reconstructor can
        do if it ignores the description entirely.

        Returns:
            [max_len, hidden_dim]
        """
        total = torch.zeros(max_len, self.hidden_dim)
        count = torch.zeros(max_len)

        for i in indices:
            seq = _ensure_float32(
                self.samples[i]["activation_sequence"]
            )[:max_len]
            length = seq.shape[0]

            total[:length] += seq
            count[:length] += 1

        return total / count.clamp(min=1).unsqueeze(-1)

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
# Train / val / test split
# ===========================================================================

SPLIT_FILENAME = "split.json"


def make_split(
    n: int,
    seed: int = 42,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> Dict[str, List[int]]:
    """
    Deterministic index split into train / val / test.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    return {
        "test": sorted(indices[:n_test]),
        "val": sorted(indices[n_test:n_test + n_val]),
        "train": sorted(indices[n_test + n_val:]),
    }


def save_split(
    split: Dict[str, List[int]],
    dataset_dir: str,
    seed: int,
) -> Path:
    path = Path(dataset_dir) / SPLIT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)

    n = sum(len(v) for v in split.values())

    with open(path, "w") as f:
        json.dump({"num_samples": n, "seed": seed, **split}, f)

    return path


def load_or_create_split(
    dataset_dir: str,
    n: int,
    seed: int = 42,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> Dict[str, List[int]]:
    """
    Load <dataset_dir>/split.json, creating it if missing.

    Every stage (training and all evaluations) must use this so that
    evaluation runs only on held-out samples.
    """
    path = Path(dataset_dir) / SPLIT_FILENAME

    if path.exists():
        with open(path) as f:
            data = json.load(f)

        if data["num_samples"] != n:
            raise ValueError(
                f"{path} was made for {data['num_samples']} samples "
                f"but the buffer has {n}. Rebuild the buffer or "
                f"delete the stale split file."
            )

        return {k: data[k] for k in ("train", "val", "test")}

    split = make_split(n, seed, val_frac, test_frac)
    save_split(split, dataset_dir, seed)

    print(f"[INFO] Created new split: {path}")

    return split


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