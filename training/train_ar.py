import json
import yaml
import torch

from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from nla.dataset import ActivationDataset
from nla.reconstructor import ActivationReconstructor
from nla.losses import cosine_loss
from nla.metrics import cosine_similarity_metric
from nla.tracking import WandbTracker

load_dotenv()


def collate(batch):
    return {
        "texts": [x["description"] for x in batch],
        "activations": torch.stack(
            [x["activation"] for x in batch]
        )
    }


def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():

    cfg = yaml.safe_load(open("configs/base.yaml"))

    device = cfg["device"]

    tracker = None

    if cfg["tracking"]["use_wandb"]:

        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=cfg["tracking"]["run_name"],
            config=cfg,
        )

    dataset = ActivationDataset(
        "datasets/activation_buffer/buffer.pt"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate
    )

    sample_dim = dataset[0]["activation"].shape[-1]

    model = ActivationReconstructor(
        output_dim=sample_dim,
        hidden_dim=cfg["training"]["hidden_dim"]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"])
    )

    best_loss = float("inf")

    checkpoint_dir = "checkpoints/ar"

    save_config(
        cfg,
        f"{checkpoint_dir}/config.json"
    )

    for epoch in range(cfg["training"]["epochs"]):

        model.train()

        losses = []
        cosines = []

        progress_bar = tqdm(
            enumerate(loader),
            total=len(loader)
        )

        for batch_idx, batch in progress_bar:

            target = batch["activations"].to(device)

            pred = model(
                batch["texts"],
                device
            )

            loss = cosine_loss(
                pred,
                target
            )

            cosine = cosine_similarity_metric(
                pred,
                target
            )

            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            opt.step()

            losses.append(loss.item())
            cosines.append(cosine.item())

            progress_bar.set_description(
                f"epoch={epoch} "
                f"loss={loss.item():.4f} "
                f"cos={cosine.item():.4f}"
            )

            # Optional lightweight step logging
            if (
                tracker is not None
                and batch_idx % 50 == 0
            ):

                tracker.log({
                    "train/step_loss": loss.item(),
                    "train/step_cosine": cosine.item(),
                })

        avg_loss = sum(losses) / len(losses)
        avg_cosine = sum(cosines) / len(cosines)

        current_lr = opt.param_groups[0]["lr"]

        print(
            f"epoch={epoch} "
            f"loss={avg_loss:.4f} "
            f"cosine={avg_cosine:.4f}"
        )

        if tracker is not None:

            tracker.log({
                "train/loss": avg_loss,
                "train/cosine": avg_cosine,
                "train/lr": current_lr,
                "train/epoch": epoch
            }, step=epoch)

        # Save latest checkpoint
        latest_checkpoint = (
            f"{checkpoint_dir}/latest_model.pt"
        )

        torch.save(
            model.state_dict(),
            latest_checkpoint
        )

        # Save best checkpoint
        if avg_loss < best_loss:

            best_loss = avg_loss

            best_checkpoint = (
                f"{checkpoint_dir}/best_model.pt"
            )

            torch.save(
                model.state_dict(),
                best_checkpoint
            )

            print(
                f"[INFO] New best model saved "
                f"(loss={best_loss:.4f})"
            )

    print("\n[OK] Training completed.")

    if tracker is not None:

        tracker.save_model(
            f"{checkpoint_dir}/best_model.pt"
        )

        tracker.finish()


if __name__ == "__main__":
    main()