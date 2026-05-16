"""
training/build_buffer.py

Stage 0: Build the activation buffer.

Reads config from TINYNLA_CONFIG env var (set by layer_sweep.py)
or falls back to configs/base.yaml.

Output per run:
    <output_dir>/buffer.pt       — list of {id, text, description, activation}
    <output_dir>/metadata.json   — config snapshot + dataset statistics
"""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from nla.activations import ActivationExtractor
from nla.dataset import save_dataset
from nla.labeler import SemanticLabeler
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()


def build_buffer(cfg: dict) -> None:
    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(cfg["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "buffer.pt"
    metadata_path = output_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Extractor + Labeler
    # ------------------------------------------------------------------
    extractor = ActivationExtractor(
        model_name=cfg["model"]["target_name"],
        layer_idx=cfg["activation"]["layer_idx"],
        device=device,
        max_length=cfg["activation"]["max_length"],
        normalize=cfg["activation"]["normalize"],
    )
    labeler = SemanticLabeler()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = load_dataset(cfg["dataset"]["source"], split="train")
    n = cfg["dataset"]["num_samples"]
    dataset = dataset.select(range(min(n, len(dataset))))
    print(f"\n[INFO] Building buffer: {len(dataset)} samples from {cfg['dataset']['source']}")

    samples = []
    skipped = 0
    pooling = cfg["activation"].get("pooling", "mean")

    for idx, item in enumerate(tqdm(dataset)):
        try:
            text = item["text"]
            if not isinstance(text, str) or len(text.strip()) < 20:
                skipped += 1
                continue

            text = text.strip()[:cfg["dataset"]["max_text_chars"]]

            activation = extractor.extract(text, mode="sequence", pooling=pooling)
            description = labeler.describe(text)

            samples.append({
                "id": idx,
                "text": text,
                "description": description,
                "activation": activation.cpu(),
            })

        except Exception as e:
            skipped += 1
            print(f"\n[WARN] Skipped sample {idx}: {e}")

    save_dataset(samples, str(output_path))

    metadata = {
        "model": cfg["model"]["target_name"],
        "layer_idx": cfg["activation"]["layer_idx"],
        "pooling": pooling,
        "normalize": cfg["activation"]["normalize"],
        "num_samples": len(samples),
        "skipped_samples": skipped,
        "hidden_dim": int(samples[0]["activation"].shape[-1]) if samples else None,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n========== BUFFER SUMMARY ==========")
    print(f"Saved:   {len(samples)}")
    print(f"Skipped: {skipped}")
    if samples:
        print(f"Dim:     {samples[0]['activation'].shape[-1]}")
    print(f"Output:  {output_path}")
    print(f"[OK] Buffer complete.")


def main():
    cfg = load_config()
    build_buffer(cfg)


if __name__ == "__main__":
    main()