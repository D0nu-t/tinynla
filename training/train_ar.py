"""
training/train_ar.py

Stage 1: Train the Activation Reconstructor.

Reads config via load_config() which checks TINYNLA_CONFIG env var first,
allowing layer_sweep.py to inject per-layer configs without modifying this file.

Saves:
    <save_dir>/best_model.pt   — lowest training-loss checkpoint
    <save_dir>/latest_model.pt — end-of-last-epoch checkpoint
    <save_dir>/config.json     — config snapshot for reproducibility
    <save_dir>/metrics.json    — per-epoch training metrics
"""

import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from nla.dataset import ActivationDataset
from nla.losses import cosine_loss
from nla.metrics import cosine_similarity_metric
from nla.reconstructor import ActivationReconstructor
from nla.tracking import WandbTracker
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()


def collate(batch):
    return {
        "texts": [x["description"] for x in batch],
        "activations": torch.stack([x["activation"] for x in batch]),
    }


def train_ar(cfg: dict) -> None:
    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    tracker = None
    if cfg["tracking"]["use_wandb"]:
        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=cfg["tracking"]["run_name"],
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    buffer_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"
    dataset = ActivationDataset(str(buffer_path))
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    print(f"[INFO] Dataset: {len(dataset)} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    sample_dim = dataset[0]["activation"].shape[-1]
    model = ActivationReconstructor(
        encoder_name=cfg["training"].get("encoder_name", "distilbert-base-uncased"),
        output_dim=sample_dim,
        hidden_dim=cfg["training"]["hidden_dim"],
    ).to(device)

    # ------------------------------------------------------------------
    # Optimizer + AMP
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    use_amp = device == "cuda"
    device_type = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    grad_clip = float(cfg["training"]["grad_clip"])
    patience = cfg["training"].get("early_stopping_patience", 5)
    epochs = cfg["training"]["epochs"]

    best_loss = float("inf")
    epochs_without_improvement = 0
    all_metrics = []

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        losses, cosines = [], []

        bar = tqdm(enumerate(loader), total=len(loader), desc=f"epoch={epoch}")
        for step, batch in bar:
            target = batch["activations"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(batch["texts"], device)
                loss = cosine_loss(pred, target)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            cos = cosine_similarity_metric(pred.detach(), target)
            losses.append(loss.item())
            cosines.append(cos)

            bar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{cos:.4f}")

            if tracker and step % 50 == 0:
                tracker.log({"train/step_loss": loss.item(), "train/step_cosine": cos})

        avg_loss = sum(losses) / len(losses)
        avg_cos = sum(cosines) / len(cosines)

        print(f"epoch={epoch}  loss={avg_loss:.4f}  cosine={avg_cos:.4f}")

        epoch_metrics = {"epoch": epoch, "loss": avg_loss, "cosine": avg_cos}
        all_metrics.append(epoch_metrics)

        if tracker:
            tracker.log({"train/loss": avg_loss, "train/cosine": avg_cos, "train/epoch": epoch}, step=epoch)

        # Latest checkpoint (always overwritten)
        torch.save(model.state_dict(), save_dir / "latest_model.pt")

        # Best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  -> Best model saved (loss={best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    with open(save_dir / "metrics.json", "w") as f:
        json.dump({"training_history": all_metrics, "best_loss": best_loss}, f, indent=2)

    if tracker:
        tracker.save_model(str(save_dir / "best_model.pt"))
        tracker.finish()

    print(f"\n[OK] Training complete. Best loss: {best_loss:.4f}")


def main():
    cfg = load_config()
    train_ar(cfg)


if __name__ == "__main__":
    main()