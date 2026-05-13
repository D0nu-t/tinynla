import yaml
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from dotenv import load_dotenv

load_dotenv()

from nla.activations import ActivationExtractor
from nla.reconstructor import ActivationReconstructor
from nla.patching import ActivationPatcher

from nla.evaluation import (
    kl_divergence,
    topk_overlap,
    logit_cosine_similarity
)

from nla.dataset import ActivationDataset


def main():

    cfg = yaml.safe_load(
        open("configs/base.yaml")
    )

    device = cfg["device"]

    #
    # Load target LM
    #

    model_name = cfg["model"]["target_name"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    ).to(device)

    model.eval()

    #
    # Load AR
    #

    sample_dim = model.config.hidden_size

    ar = ActivationReconstructor(
        output_dim=sample_dim
    ).to(device)

    ar.load_state_dict(
        torch.load(
            "checkpoints/ar/model.pt"
        )
    )

    ar.eval()

    #
    # Dataset
    #

    dataset = ActivationDataset(
        "datasets/activation_buffer/buffer.pt"
    )

    #
    # Metrics
    #

    kls = []
    overlaps = []
    cosine_sims = []

    #
    # Evaluate subset
    #

    for item in tqdm(dataset.samples[:200]):

        text = item["text"]

        description = item["description"]

        #
        # Original logits
        #

        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        ).to(device)

        with torch.no_grad():

            original_outputs = model(**toks)

            original_logits = (
                original_outputs.logits[:, -1, :]
            )

        #
        # Reconstruct activation
        #

        with torch.no_grad():

            reconstructed = ar(
                [description],
                device
            )

        #
        # Patch activation
        #

        patcher = ActivationPatcher(
            reconstructed
        )

        target_layer = (
            model.transformer.h[
                cfg["activation"]["layer_idx"]
            ]
        )

        handle = target_layer.register_forward_hook(
            patcher.hook_fn
        )

        #
        # Patched forward pass
        #

        with torch.no_grad():

            patched_outputs = model(**toks)

            patched_logits = (
                patched_outputs.logits[:, -1, :]
            )

        handle.remove()

        #
        # Metrics
        #

        kls.append(
            kl_divergence(
                original_logits,
                patched_logits
            )
        )

        overlaps.append(
            topk_overlap(
                original_logits,
                patched_logits
            )
        )

        cosine_sims.append(
            logit_cosine_similarity(
                original_logits,
                patched_logits
            )
        )

    #
    # Results
    #

    print()
    print("========== FUNCTIONAL EVALUATION ==========")
    print()

    print(
        f"KL divergence: "
        f"{sum(kls)/len(kls):.4f}"
    )

    print(
        f"Top-k overlap: "
        f"{sum(overlaps)/len(overlaps):.4f}"
    )

    print(
        f"Logit cosine similarity: "
        f"{sum(cosine_sims)/len(cosine_sims):.4f}"
    )


if __name__ == "__main__":
    main()