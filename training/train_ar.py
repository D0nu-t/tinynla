"""
training/train_ar.py
"""

import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from nla.dataset import (
    SequenceActivationDataset,
    sequence_collate,
)
from nla.losses import masked_sequence_cosine_loss
from nla.reconstructor import TokenLevelReconstructor
from nla.tracking import WandbTracker
from nla.utils import (
    load_config,
    resolve_device,
    set_seed,
)

load_dotenv()


def train_ar(cfg: dict):

    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    tracker = None
    if cfg["tracking"]["use_wandb"]:
        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=cfg["tracking"]["run_name"],
            config=cfg,
            mode="offline",
        )

    buffer_path = (
        Path(cfg["dataset"]["output_dir"])
        / "buffer.pt"
    )

    dataset = SequenceActivationDataset(
        str(buffer_path)
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=sequence_collate,
        num_workers=0,
    )

    hidden_dim = dataset[0][
        "activation_sequence"
    ].shape[-1]

    model = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=cfg["activation"]["max_length"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(
            cfg["training"]["weight_decay"]
        ),
        foreach=False,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=device == "cuda"
    )

    best_loss = float("inf")
    global_step = 0

    for epoch in range(
        cfg["training"]["epochs"]
    ):

        model.train()
        losses = []

        pbar = tqdm(loader)

        for batch in pbar:

            target = batch[
                "activation_sequences"
            ].to(device)

            mask = batch["mask"].to(device)

            seq_len = target.shape[1]

            with torch.cuda.amp.autocast(
                enabled=device == "cuda"
            ):

                pred = model(
                    batch["texts"],
                    seq_len=seq_len,
                    device=device,
                )

                loss = (
                    masked_sequence_cosine_loss(
                        pred,
                        target,
                        mask,
                    )
                )

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"][
                    "grad_clip"
                ],
            )

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            pbar.set_description(
                f"epoch={epoch} loss={loss.item():.4f}"
            )

            if tracker:
                tracker.log(
                    {
                        "train/loss": loss.item(),
                    },
                    step=global_step,
                )

            global_step += 1

        avg_loss = sum(losses) / len(losses)

        torch.save(
            model.state_dict(),
            save_dir / "latest_model.pt",
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(
                model.state_dict(),
                save_dir / "best_model.pt",
            )

    with open(
        save_dir / "config.json",
        "w",
    ) as f:
        json.dump(cfg, f, indent=2)

    if tracker:
        tracker.finish()


def main():
    cfg = load_config()
    train_ar(cfg)


if __name__ == "__main__":
    main()