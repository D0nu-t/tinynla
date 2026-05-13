import yaml
import torch

from torch.utils.data import DataLoader

from nla.dataset import ActivationDataset
from nla.reconstructor import ActivationReconstructor
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
        batch_size=64,
        collate_fn=collate
    )

    sample_dim = dataset[0]["activation"].shape[-1]

    model = ActivationReconstructor(
        output_dim=sample_dim
    ).to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/ar/model.pt"
        )
    )

    model.eval()

    sims = []

    with torch.no_grad():
        for batch in loader:
            target = batch["activations"].to(device)

            pred = model(
                batch["texts"],
                device
            )

            sim = cosine_similarity_metric(
                pred,
                target
            )

            sims.append(sim)

    print(
        f"mean cosine similarity: "
        f"{sum(sims)/len(sims):.4f}"
    )


if __name__ == "__main__":
    main()