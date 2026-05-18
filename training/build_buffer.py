"""
training/build_buffer.py

Builds a sequence-level activation dataset.

Pipeline:
  text
    -> frozen LM hidden-state extraction
    -> [seq_len, hidden_dim] activation trajectory
    -> semantic description generation
    -> serialized buffer.pt

Phase 1/2/3 upgrades:
  - Sequence-native extraction (no pooled activations)
  - Robust dataset filtering + validation
  - Exact seq_len preservation
  - Safe failure handling
  - Periodic checkpointing for long runs
  - Metadata sanity tracking
  - CUDA memory cleanup between failures
  - Duplicate text filtering
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Set

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from nla.activations import ActivationExtractor
from nla.dataset import save_dataset
from nla.labeler import SemanticLabeler
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()
print("[OK] Environment variables loaded.")


def _is_valid_text(
    text: str,
    min_chars: int = 20,
) -> bool:
    """Basic text quality filter."""
    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < min_chars:
        return False

    # Remove pathological low-information strings
    unique_chars = len(set(text))
    if unique_chars < 5:
        return False

    return True


def _save_checkpoint(
    samples: List[Dict],
    output_path: Path,
    metadata_path: Path,
    cfg: dict,
    skipped: int,
) -> None:
    """Safely save intermediate progress."""
    save_dataset(samples, str(output_path))

    metadata = {
        "model": cfg["model"]["target_name"],
        "layer_idx": cfg["activation"]["layer_idx"],
        "normalize": cfg["activation"]["normalize"],
        "num_samples": len(samples),
        "skipped_samples": skipped,
        "hidden_dim": (
            samples[0]["activation_sequence"].shape[-1]
            if samples else None
        ),
        "sequence_mode": True,
        "max_length": cfg["activation"]["max_length"],
        "dataset_source": cfg["dataset"]["source"],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def build_buffer(cfg: dict) -> None:
    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    print(f"[INFO] Device: {device}")

    output_dir = Path(cfg["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "buffer.pt"
    metadata_path = output_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Activation extractor
    # ------------------------------------------------------------------
    extractor = ActivationExtractor(
        model_name=cfg["model"]["target_name"],
        layer_idx=cfg["activation"]["layer_idx"],
        device=device,
        max_length=cfg["activation"]["max_length"],
        normalize=cfg["activation"]["normalize"],
    )

    # ------------------------------------------------------------------
    # Semantic labeler
    # ------------------------------------------------------------------
    labeler = SemanticLabeler(use_local_model=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(
        f"[INFO] Loading dataset: "
        f"{cfg['dataset']['source']}"
    )

    dataset = load_dataset(
        cfg["dataset"]["source"],
        split="train",
    )

    n = cfg["dataset"]["num_samples"]
    dataset = dataset.select(range(min(n, len(dataset))))

    print(
        f"[INFO] Processing "
        f"{len(dataset)} samples"
    )

    samples: List[Dict] = []
    skipped = 0
    duplicate_count = 0
    seen_texts: Set[str] = set()

    checkpoint_every = cfg["dataset"].get(
        "checkpoint_every",
        250,
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for idx, item in enumerate(
        tqdm(dataset, desc="build_buffer")
    ):
        try:
            # ----------------------------------------------------------
            # Extract text
            # ----------------------------------------------------------
            text = item.get("text", None)

            if not _is_valid_text(text):
                skipped += 1
                continue

            text = text.strip()

            # Hard truncate before extraction
            text = text[
                : cfg["dataset"]["max_text_chars"]
            ]

            # ----------------------------------------------------------
            # Duplicate filtering
            # ----------------------------------------------------------
            text_hash = hash(text)

            if text_hash in seen_texts:
                duplicate_count += 1
                skipped += 1
                continue

            seen_texts.add(text_hash)

            # ----------------------------------------------------------
            # Extract activation trajectory
            # Shape: [seq_len, hidden_dim]
            # ----------------------------------------------------------
            activation_sequence = (
                extractor.extract_sequence(text)
            )

            if activation_sequence is None:
                skipped += 1
                continue

            if activation_sequence.ndim != 2:
                skipped += 1
                print(
                    f"[WARN] Invalid activation "
                    f"shape for sample {idx}: "
                    f"{activation_sequence.shape}"
                )
                continue

            seq_len = activation_sequence.shape[0]
            hidden_dim = activation_sequence.shape[1]

            if seq_len < 2:
                skipped += 1
                continue

            # ----------------------------------------------------------
            # Generate semantic description
            # ----------------------------------------------------------
            description = labeler.describe(text)

            if not isinstance(description, str):
                skipped += 1
                continue

            description = description.strip()

            if len(description) == 0:
                skipped += 1
                continue

            # ----------------------------------------------------------
            # Store sample
            # ----------------------------------------------------------
            samples.append(
                {
                    "id": idx,
                    "text": text,
                    "description": description,
                    "activation_sequence": (
                        activation_sequence.cpu()
                    ),
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                }
            )

            # ----------------------------------------------------------
            # Periodic checkpoint
            # ----------------------------------------------------------
            if (
                len(samples) > 0
                and len(samples)
                % checkpoint_every
                == 0
            ):
                _save_checkpoint(
                    samples=samples,
                    output_path=output_path,
                    metadata_path=metadata_path,
                    cfg=cfg,
                    skipped=skipped,
                )

                print(
                    f"\n[INFO] Checkpoint saved "
                    f"({len(samples)} samples)"
                )

        except torch.cuda.OutOfMemoryError:
            skipped += 1

            print(
                f"[WARN] CUDA OOM "
                f"at sample {idx}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        except Exception as e:
            skipped += 1

            print(
                f"[WARN] skipped sample "
                f"{idx}: {e}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    _save_checkpoint(
        samples=samples,
        output_path=output_path,
        metadata_path=metadata_path,
        cfg=cfg,
        skipped=skipped,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUFFER BUILD COMPLETE")
    print("=" * 60)

    print(
        f"Saved samples:     "
        f"{len(samples)}"
    )
    print(
        f"Skipped samples:   "
        f"{skipped}"
    )
    print(
        f"Duplicates removed:"
        f" {duplicate_count}"
    )

    if samples:
        seq_lens = [
            s["seq_len"]
            for s in samples
        ]

        print(
            f"Mean seq len:      "
            f"{sum(seq_lens)/len(seq_lens):.2f}"
        )
        print(
            f"Min seq len:       "
            f"{min(seq_lens)}"
        )
        print(
            f"Max seq len:       "
            f"{max(seq_lens)}"
        )

        print(
            f"Hidden dim:        "
            f"{samples[0]['hidden_dim']}"
        )

    print(
        f"\n[OK] Saved buffer:"
        f" {output_path}"
    )
    print(
        f"[OK] Saved metadata:"
        f" {metadata_path}"
    )


def main():
    cfg = load_config()
    build_buffer(cfg)


if __name__ == "__main__":
    main()