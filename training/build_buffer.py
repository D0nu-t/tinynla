import json
import yaml
import torch

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

from nla.activations import ActivationExtractor
from nla.dataset import save_dataset
from nla.labeler import SemanticLabeler
load_dotenv()


def main():

    cfg = yaml.safe_load(
        open("configs/base.yaml")
    )

    # ---------------------------------------
    # Device Resolution
    # ---------------------------------------

    if cfg["device"] == "auto":

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = cfg["device"]

    print(f"\n[INFO] Using device: {device}")

    if device == "cuda":

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------
    # Paths
    # ---------------------------------------

    output_dir = Path(
        cfg["dataset"]["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "buffer.pt"

    metadata_path = output_dir / "metadata.json"

    # ---------------------------------------
    # Activation Extractor
    # ---------------------------------------

    extractor = ActivationExtractor(
        model_name=cfg["model"]["target_name"],
        layer_idx=cfg["activation"]["layer_idx"],
        device=device,
        max_length=cfg["activation"]["max_length"],
        normalize=cfg["activation"]["normalize"]
    )

    # ---------------------------------------
    # Semantic Labeler
    # ---------------------------------------

    labeler = SemanticLabeler()

    # ---------------------------------------
    # Dataset
    # ---------------------------------------

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train"
    )

    dataset = dataset.select(
        range(cfg["dataset"]["num_samples"])
    )

    print(
        f"\n[INFO] Building activation buffer "
        f"with {len(dataset)} samples"
    )

    samples = []

    skipped = 0

    # ---------------------------------------
    # Main Extraction Loop
    # ---------------------------------------

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

            # ---------------------------------------
            # Truncate Text
            # ---------------------------------------

            text = text[:512]

            # ---------------------------------------
            # Extract Activation
            # ---------------------------------------

            activation = extractor.extract_pooled(text)
            # ---------------------------------------
            # Generate Semantic Description
            # ---------------------------------------

            description = labeler.describe(text)

            samples.append({
                "id": idx,
                "text": text,
                "description": description,
                "activation": activation.cpu()
            })

        except Exception as e:

            skipped += 1

            print(
                f"\n[WARNING] Failed sample "
                f"{idx}: {str(e)}"
            )

    # ---------------------------------------
    # Save Dataset
    # ---------------------------------------

    save_dataset(
        samples,
        str(output_path)
    )

    # ---------------------------------------
    # Save Metadata
    # ---------------------------------------

    metadata = {
        "model": cfg["model"]["target_name"],
        "layer_idx": cfg["activation"]["layer_idx"],
        "normalize": cfg["activation"]["normalize"],
        "max_length": cfg["activation"]["max_length"],
        "num_samples": len(samples),
        "skipped_samples": skipped,
        "hidden_dim": (
            samples[0]["activation"].shape[-1]
            if len(samples) > 0
            else None
        )
    }

    with open(metadata_path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # ---------------------------------------
    # Final Summary
    # ---------------------------------------

    print("\n========== BUFFER SUMMARY ==========")

    print(f"Saved samples: {len(samples)}")
    print(f"Skipped samples: {skipped}")

    if len(samples) > 0:

        print(
            f"Activation dim: "
            f"{samples[0]['activation'].shape[-1]}"
        )

    print(f"\nDataset saved to:")
    print(output_path)

    print(f"\nMetadata saved to:")
    print(metadata_path)

    print("\n[OK] Activation buffer completed.")


if __name__ == "__main__":
    main()