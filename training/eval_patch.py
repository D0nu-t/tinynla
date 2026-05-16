"""
training/eval_patch.py

Stage 2a: Geometric reconstruction evaluation.

Loads the trained AR and measures mean cosine similarity between
reconstructed and target activation vectors across the full dataset.

This is a necessary but insufficient success criterion:
    high cosine -> geometry is recovered
    but functional patching can still fail if the vector is out-of-distribution
    for the downstream transformer blocks.

Run eval_functional.py for the causal test.
"""

from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from nla.dataset import ActivationDataset
from nla.reconstructor import TokenLevelReconstructor
from nla.dataset import SequenceActivationDataset
from nla.metrics import cosine_similarity_metric
from nla.reconstructor import ActivationReconstructor
from nla.utils import load_config, resolve_device

load_dotenv()


def collate(batch):
    return {
        "texts": [x["description"] for x in batch],
        "activations": torch.stack([x["activation"] for x in batch]),
    }


def main():
    cfg = load_config()
    device = resolve_device(cfg)

    buffer_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"
    dataset = SequenceActivationDataset(str(buffer_path))
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate)

    sample_dim = dataset[0]["activation"].shape[-1]
    model = TokenLevelReconstructor(
        encoder_name=cfg["training"].get("encoder_name", "distilbert-base-uncased"),
        output_dim=sample_dim,
        hidden_dim=cfg["training"]["hidden_dim"],
    ).to(device)

    checkpoint = Path(cfg["training"]["save_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    sims = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval_patch"):
            target = batch["activations"].to(device)
            pred = model(batch["texts"], device)
            sims.append(cosine_similarity_metric(pred, target))

    mean_sim = sum(sims) / len(sims)
    print(f"\nmean cosine similarity: {mean_sim:.4f}")
    print("[OK] Geometric evaluation complete.")


if __name__ == "__main__":
    main()