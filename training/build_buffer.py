"""
training/build_buffer.py
"""

import json
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
print("[OK] Environment variables loaded.")

def build_buffer(cfg: dict) -> None:
    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    output_dir = Path(cfg["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "buffer.pt"
    metadata_path = output_dir / "metadata.json"

    extractor = ActivationExtractor(
        model_name=cfg["model"]["target_name"],
        layer_idx=cfg["activation"]["layer_idx"],
        device=device,
        max_length=cfg["activation"]["max_length"],
        normalize=cfg["activation"]["normalize"],
    )

    labeler = SemanticLabeler(use_local_model=True)

    dataset = load_dataset(
        cfg["dataset"]["source"],
        split="train"
    )

    n = cfg["dataset"]["num_samples"]
    dataset = dataset.select(range(min(n, len(dataset))))

    samples = []
    skipped = 0

    for idx, item in enumerate(tqdm(dataset)):
        try:
            text = item["text"]

            if not isinstance(text, str):
                skipped += 1
                continue

            text = text.strip()

            if len(text) < 20:
                skipped += 1
                continue

            text = text[:cfg["dataset"]["max_text_chars"]]

            activation_sequence = extractor.extract_sequence(text)

            description = labeler.describe(text)
            #description = text

            samples.append(
                {
                    "id": idx,
                    "text": text,
                    "description": description,
                    "activation_sequence": activation_sequence.cpu(),
                    "seq_len": activation_sequence.shape[0],
                }
            )

        except Exception as e:
            skipped += 1
            print(f"[WARN] skipped sample {idx}: {e}")

    save_dataset(samples, str(output_path))

    metadata = {
        "model": cfg["model"]["target_name"],
        "layer_idx": cfg["activation"]["layer_idx"],
        "normalize": cfg["activation"]["normalize"],
        "num_samples": len(samples),
        "skipped_samples": skipped,
        "hidden_dim": samples[0]["activation_sequence"].shape[-1]
        if samples
        else None,
        "sequence_mode": True,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] saved {len(samples)} samples")
    print(f"[OK] skipped {skipped}")


def main():
    cfg = load_config()
    build_buffer(cfg)


if __name__ == "__main__":
    main()