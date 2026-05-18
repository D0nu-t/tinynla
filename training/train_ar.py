"""
training/train_ar.py

Stage 1: Sequence-level activation reconstruction training.

Trains TokenLevelReconstructor to map:
    natural-language description
        ->
    activation trajectory [seq_len, hidden_dim]

Key upgrades:
  - full trajectory supervision
  - masked sequence cosine loss
  - optional combined cosine + MSE loss
  - AMP training
  - gradient clipping
  - early stopping
  - WandB logging
  - checkpoint metadata
  - validation split
  - reconstruction diagnostics

Outputs:
    best_model.pt
    latest_model.pt
    metrics.json
    config.json
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from nla.dataset import (
    SequenceActivationDataset,
    sequence_collate,
)
from nla.losses import (
    masked_sequence_cosine_loss,
    masked_combined_sequence_loss,
)
from nla.reconstructor import TokenLevelReconstructor
from nla.tracking import WandbTracker
from nla.utils import (
    load_config,
    resolve_device,
    set_seed,
)

load_dotenv()
print("[OK] Environment variables loaded.")


# ============================================================================
# Helpers
# ============================================================================

def evaluate_epoch(
    model,
    loader,
    device,
    loss_type="cosine",
    mse_alpha=0.5,
):
    model.eval()

    losses: List[float] = []
    cosine_scores: List[float] = []

    with torch.no_grad():

        for batch in loader:

            target = batch["activation_sequences"].to(device)
            mask = batch["mask"].to(device)

            seq_len = target.shape[1]

            pred = model(
                batch["texts"],
                seq_len=seq_len,
                device=device,
            )

            if loss_type == "combined":
                loss = masked_combined_sequence_loss(
                    pred,
                    target,
                    mask,
                    alpha=mse_alpha,
                )
            else:
                loss = masked_sequence_cosine_loss(
                    pred,
                    target,
                    mask,
                )

            pred_norm = torch.nn.functional.normalize(pred, dim=-1)
            target_norm = torch.nn.functional.normalize(target, dim=-1)

            cosine = (pred_norm * target_norm).sum(dim=-1)
            cosine = cosine * mask.float()

            denom = mask.float().sum().clamp(min=1.0)
            cosine_mean = (cosine.sum() / denom).item()

            losses.append(loss.item())
            cosine_scores.append(cosine_mean)

    return {
        "loss": float(np.mean(losses)),
        "cosine": float(np.mean(cosine_scores)),
    }


# ============================================================================
# Main training
# ============================================================================

def train_ar(cfg: Dict):

    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    print(f"\n[INFO] Device: {device}")

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Tracking
    # ----------------------------------------------------------------------

    tracker = None

    if cfg["tracking"]["use_wandb"]:
        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=cfg["tracking"]["run_name"],
            config=cfg,
            mode=cfg["tracking"].get("mode", "offline"),
        )

    # ----------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------

    buffer_path = (
        Path(cfg["dataset"]["output_dir"])
        / "buffer.pt"
    )

    print(f"[INFO] Loading dataset: {buffer_path}")

    dataset = SequenceActivationDataset(str(buffer_path))

    val_split = cfg["training"].get("val_split", 0.05)

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=sequence_collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=sequence_collate,
        num_workers=0,
    )

    hidden_dim = dataset[0]["activation_sequence"].shape[-1]

    print(f"[INFO] Hidden dim: {hidden_dim}")
    print(f"[INFO] Train samples: {train_size}")
    print(f"[INFO] Val samples: {val_size}")

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------

    model = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=cfg["activation"]["max_length"],
        encoder_name=cfg["training"].get(
            "encoder_name",
            "distilgpt2",
        ),
    ).to(device)

    # ----------------------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        foreach=False,
    )

    scaler = torch.amp.GradScaler(
        enabled=device == "cuda"
    )

    # ----------------------------------------------------------------------
    # Loss configuration
    # ----------------------------------------------------------------------

    loss_type = cfg["training"].get(
        "loss_type",
        "cosine",
    )

    mse_alpha = cfg["training"].get(
        "combined_loss_alpha",
        0.5,
    )

    # ----------------------------------------------------------------------
    # Early stopping
    # ----------------------------------------------------------------------

    best_val_loss = float("inf")
    patience = cfg["training"].get("patience", 5)
    patience_counter = 0

    global_step = 0

    train_history = []
    val_history = []

    # ----------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------

    for epoch in range(cfg["training"]["epochs"]):

        model.train()

        epoch_losses = []
        epoch_cosines = []

        pbar = tqdm(
            train_loader,
            desc=f"epoch={epoch}",
        )

        for batch in pbar:

            target = batch["activation_sequences"].to(device)
            mask = batch["mask"].to(device)

            seq_len = target.shape[1]

            with torch.amp.autocast(
                device_type=device,
                enabled=device == "cuda",
            ):

                pred = model(
                    batch["texts"],
                    seq_len=seq_len,
                    device=device,
                )

                if loss_type == "combined":

                    loss = masked_combined_sequence_loss(
                        pred,
                        target,
                        mask,
                        alpha=mse_alpha,
                    )

                else:

                    loss = masked_sequence_cosine_loss(
                        pred,
                        target,
                        mask,
                    )

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["grad_clip"],
            )

            scaler.step(optimizer)
            scaler.update()

            # --------------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------------

            with torch.no_grad():

                pred_norm = torch.nn.functional.normalize(
                    pred,
                    dim=-1,
                )

                target_norm = torch.nn.functional.normalize(
                    target,
                    dim=-1,
                )

                cosine = (
                    pred_norm * target_norm
                ).sum(dim=-1)

                cosine = cosine * mask.float()

                denom = mask.float().sum().clamp(min=1.0)

                cosine_mean = (
                    cosine.sum() / denom
                ).item()

            epoch_losses.append(loss.item())
            epoch_cosines.append(cosine_mean)

            pbar.set_description(
                f"epoch={epoch} "
                f"loss={loss.item():.4f} "
                f"cos={cosine_mean:.4f}"
            )

            if tracker:

                tracker.log(
                    {
                        "train/loss": loss.item(),
                        "train/cosine": cosine_mean,
                    },
                    step=global_step,
                )

            global_step += 1

        # ------------------------------------------------------------------
        # Epoch aggregation
        # ------------------------------------------------------------------

        train_loss = float(np.mean(epoch_losses))
        train_cosine = float(np.mean(epoch_cosines))

        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            loss_type=loss_type,
            mse_alpha=mse_alpha,
        )

        val_loss = val_metrics["loss"]
        val_cosine = val_metrics["cosine"]

        train_history.append(train_loss)
        val_history.append(val_loss)

        print()
        print("=" * 60)
        print(f"EPOCH {epoch}")
        print("=" * 60)
        print(f"train loss:   {train_loss:.4f}")
        print(f"train cosine: {train_cosine:.4f}")
        print(f"val loss:     {val_loss:.4f}")
        print(f"val cosine:   {val_cosine:.4f}")

        # ------------------------------------------------------------------
        # Save latest checkpoint
        # ------------------------------------------------------------------

        torch.save(
            model.state_dict(),
            save_dir / "latest_model.pt",
        )

        # ------------------------------------------------------------------
        # Save best checkpoint
        # ------------------------------------------------------------------

        improved = val_loss < best_val_loss

        if improved:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                save_dir / "best_model.pt",
            )

            print("[INFO] Saved best_model.pt")

        else:

            patience_counter += 1

            print(
                f"[INFO] Early-stop counter: "
                f"{patience_counter}/{patience}"
            )

        # ------------------------------------------------------------------
        # WandB
        # ------------------------------------------------------------------

        if tracker:

            tracker.log(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/train_cosine": train_cosine,
                    "epoch/val_loss": val_loss,
                    "epoch/val_cosine": val_cosine,
                },
                step=global_step,
            )

        # ------------------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------------------

        if patience_counter >= patience:

            print("\n[INFO] Early stopping triggered.")
            break

    # ----------------------------------------------------------------------
    # Save metadata
    # ----------------------------------------------------------------------

    metrics = {
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "val_history": val_history,
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ----------------------------------------------------------------------
    # Finish tracking
    # ----------------------------------------------------------------------

    if tracker:
        tracker.finish()

    print("\n[OK] Training complete.")


# ============================================================================
# Entry
# ============================================================================

def main():
    cfg = load_config()
    train_ar(cfg)


if __name__ == "__main__":
    main()