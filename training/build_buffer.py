import yaml
import random

from datasets import load_dataset
from tqdm import tqdm

from nla.activations import ActivationExtractor
from nla.dataset import save_dataset

from dotenv import load_dotenv

load_dotenv()
TEMPLATES = [
    "Activation related to: {}",
    "The text discusses {}",
    "Semantic feature involving {}"
]


def synthetic_description(text):

    text = text.lower()

    if any(x in text for x in ["king", "queen", "prince"]):
        return "Activation related to royalty."

    if any(x in text for x in ["dog", "cat", "animal"]):
        return "Activation related to animals."

    if any(x in text for x in ["happy", "sad", "angry"]):
        return "Activation related to emotions."

    return "General narrative activation."


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))

    extractor = ActivationExtractor(
        model_name=cfg["model"]["target_name"],
        layer_idx=cfg["activation"]["layer_idx"],
        device=cfg["device"]
    )

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train"
    )

    samples = []

    for item in tqdm(dataset.select(range(cfg["dataset"]["num_samples"]))):
        text = item["text"][:256]

        activation = extractor.extract(text)

        desc = synthetic_description(text)

        samples.append({
            "text": text,
            "description": desc,
            "activation": activation
        })

    save_dataset(
        samples,
        f"{cfg['dataset']['output_dir']}/buffer.pt"
    )


if __name__ == "__main__":
    main()