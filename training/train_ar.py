import yaml
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from nla.dataset import ActivationDataset
from nla.reconstructor import ActivationReconstructor
from nla.losses import cosine_loss
from nla.metrics import cosine_similarity_metric
from dotenv import load_dotenv

load_dotenv()

def collate(batch):
    return {
        "texts": [x["description"] for x in batch],
        "activations": torch.stack(
            [x["activation"] for x in batch]
        )
    }


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))

    device = cfg["device"]

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
        lr=cfg["training"]["lr"]
    )

    for epoch in range(cfg["training"]["epochs"]):
        model.train()

        losses = []

        for batch in tqdm(loader):
            target = batch["activations"].to(device)

            pred = model(
                batch["texts"],
                device
            )

            loss = cosine_loss(pred, target)

            opt.zero_grad()

            loss.backward()

            opt.step()

            losses.append(loss.item())

        print(
            f"epoch={epoch} "
            f"loss={sum(losses)/len(losses):.4f}"
        )

    torch.save(
        model.state_dict(),
        "checkpoints/ar/model.pt"
    )


if __name__ == "__main__":
    main()