import json
import yaml
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from nla.reconstructor import (
    ActivationReconstructor
)

from nla.dataset import (
    ActivationDataset
)

from nla.patching import (
    ActivationPatcher
)

from nla.evaluation import (
    kl_divergence,
    topk_overlap,
    logit_cosine_similarity
)

from nla.tracking import (
    WandbTracker
)

load_dotenv()


# =========================================================
# Utilities
# =========================================================

def resolve_device(cfg):

    if cfg["device"] == "auto":

        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return cfg["device"]


def mean(values):

    if len(values) == 0:
        return 0.0

    return float(sum(values) / len(values))


def std(values):

    if len(values) == 0:
        return 0.0

    return float(np.std(values))


# =========================================================
# Forward Helpers
# =========================================================

@torch.no_grad()
def get_logits(
    model,
    tokenizer,
    text,
    device,
    max_length=64
):

    toks = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    outputs = model(**toks)

    logits = outputs.logits[:, -1, :]

    return logits, toks


@torch.no_grad()
def run_patched_forward(
    model,
    toks,
    layer_idx,
    patch_tensor
):

    target_layer = (
        model.transformer.h[layer_idx]
    )

    patcher = ActivationPatcher(
        patch_tensor
    )

    handle = target_layer.register_forward_hook(
        patcher.hook_fn
    )

    outputs = model(**toks)

    logits = outputs.logits[:, -1, :]

    handle.remove()

    return logits


# =========================================================
# Main Evaluation
# =========================================================

def main():

    cfg = yaml.safe_load(
        open("configs/base.yaml")
    )

    device = resolve_device(cfg)

    print(f"\n[INFO] Device: {device}")

    if device == "cuda":

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =====================================================
    # Tracking
    # =====================================================

    tracker = None

    if cfg["tracking"]["use_wandb"]:

        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=(
                cfg["tracking"]["run_name"]
                + "_functional_eval"
            ),
            config=cfg
        )

    # =====================================================
    # Load LM
    # =====================================================

    model_name = cfg["model"]["target_name"]

    print(f"\n[INFO] Loading LM: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(model_name)
        .to(device)
    )

    model.eval()

    # =====================================================
    # Load Dataset
    # =====================================================

    dataset_path = (
        Path(
            cfg["dataset"]["output_dir"]
        ) / "buffer.pt"
    )

    dataset = ActivationDataset(
        str(dataset_path)
    )

    print(
        f"[INFO] Loaded dataset "
        f"with {len(dataset)} samples"
    )

    # =====================================================
    # Determine Activation Dim
    # =====================================================

    sample_dim = (
        dataset[0]["activation"]
        .shape[-1]
    )

    # =====================================================
    # Load AR
    # =====================================================

    model_path = (
        Path(
            cfg["training"]["save_dir"]
        ) / "model.pt"
    )

    print(
        f"[INFO] Loading AR from:"
    )

    print(model_path)

    ar = ActivationReconstructor(
        output_dim=sample_dim,
        hidden_dim=cfg["training"]["hidden_dim"]
    ).to(device)

    ar.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    ar.eval()

    # =====================================================
    # Metrics
    # =====================================================

    kls = []
    overlaps = []
    cosine_sims = []

    random_kls = []
    zero_kls = []

    # =====================================================
    # Evaluation Loop
    # =====================================================

    num_eval = min(
        200,
        len(dataset)
    )

    print(
        f"\n[INFO] Evaluating "
        f"{num_eval} samples"
    )

    for item in tqdm(
        dataset.samples[:num_eval]
    ):

        text = item["text"]

        description = item["description"]

        # -------------------------------------------------
        # Original
        # -------------------------------------------------

        original_logits, toks = get_logits(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=cfg["activation"]["max_length"]
        )

        # -------------------------------------------------
        # Reconstructed Activation
        # -------------------------------------------------

        with torch.no_grad():

            reconstructed = ar(
                [description],
                device
            )

        # -------------------------------------------------
        # Patched Logits
        # -------------------------------------------------

        patched_logits = run_patched_forward(
            model=model,
            toks=toks,
            layer_idx=cfg["activation"]["layer_idx"],
            patch_tensor=reconstructed
        )

        # -------------------------------------------------
        # Random Baseline
        # -------------------------------------------------

        random_activation = torch.randn_like(
            reconstructed
        )

        random_logits = run_patched_forward(
            model=model,
            toks=toks,
            layer_idx=cfg["activation"]["layer_idx"],
            patch_tensor=random_activation
        )

        # -------------------------------------------------
        # Zero Baseline
        # -------------------------------------------------

        zero_activation = torch.zeros_like(
            reconstructed
        )

        zero_logits = run_patched_forward(
            model=model,
            toks=toks,
            layer_idx=cfg["activation"]["layer_idx"],
            patch_tensor=zero_activation
        )

        # -------------------------------------------------
        # Main Metrics
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Baselines
        # -------------------------------------------------

        random_kls.append(
            kl_divergence(
                original_logits,
                random_logits
            )
        )

        zero_kls.append(
            kl_divergence(
                original_logits,
                zero_logits
            )
        )

    # =====================================================
    # Aggregate Metrics
    # =====================================================

    results = {

        "kl_divergence_mean":
            mean(kls),

        "kl_divergence_std":
            std(kls),

        "topk_overlap_mean":
            mean(overlaps),

        "topk_overlap_std":
            std(overlaps),

        "logit_cosine_mean":
            mean(cosine_sims),

        "logit_cosine_std":
            std(cosine_sims),

        "random_baseline_kl":
            mean(random_kls),

        "zero_baseline_kl":
            mean(zero_kls)
    }

    # =====================================================
    # Save Metrics
    # =====================================================

    metrics_path = (
        Path(
            cfg["training"]["save_dir"]
        ) / "metrics.json"
    )

    with open(metrics_path, "w") as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # =====================================================
    # Console Output
    # =====================================================

    print()
    print("=" * 50)
    print("FUNCTIONAL EVALUATION")
    print("=" * 50)

    print()

    print(
        f"KL divergence: "
        f"{results['kl_divergence_mean']:.4f}"
    )

    print(
        f"Top-k overlap: "
        f"{results['topk_overlap_mean']:.4f}"
    )

    print(
        f"Logit cosine similarity: "
        f"{results['logit_cosine_mean']:.4f}"
    )

    print()

    print("Baselines")

    print(
        f"Random KL: "
        f"{results['random_baseline_kl']:.4f}"
    )

    print(
        f"Zero KL: "
        f"{results['zero_baseline_kl']:.4f}"
    )

    # =====================================================
    # WandB Logging
    # =====================================================

    if tracker is not None:

        tracker.log({

            "eval/kl_divergence":
                results["kl_divergence_mean"],

            "eval/topk_overlap":
                results["topk_overlap_mean"],

            "eval/logit_cosine":
                results["logit_cosine_mean"],

            "eval/random_baseline_kl":
                results["random_baseline_kl"],

            "eval/zero_baseline_kl":
                results["zero_baseline_kl"]
        })

        tracker.finish()

    print("\n[OK] Functional evaluation completed.")