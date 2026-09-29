"""
training/train_ar_sft.py

Stage 1: supervised AR training on warm-start explanations
(explanation -> activation), direction-only MSE.

    python -m training.train_ar_sft --config configs/gpt2_small.yaml

Outputs (ar_sft.save_dir):
    ar.pt, ar_config.json     best checkpoint by validation FVE
    metrics.jsonl             step / epoch log (for the future dashboard)
    ar_sft_eval.json          held-out test FVE with controls
"""

import json
import random
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from nla.ar import ActivationReconstructor
from nla.datagen import BUFFER_FILE, train_mean
from nla.dataset import load_or_create_split
from nla.evaluate import ar_scores
from nla.metrics import resolve_scale
from nla.runs import model_opts
from nla.utils import cli_config, resolve_device, retry_on_cuda_oom, set_seed

load_dotenv()


def main():
    cfg, _ = cli_config(__doc__)

    seed = cfg["experiment"]["seed"]
    set_seed(seed, deterministic=False)
    device = resolve_device(cfg)
    tcfg = cfg["ar_sft"]

    # ------------------------------------------------------------------ data
    data_dir = cfg["data"]["output_dir"]
    record = torch.load(Path(data_dir) / BUFFER_FILE, weights_only=False)
    acts = record["activations"]
    expl = record["explanations"]
    split = load_or_create_split(data_dir, n=len(acts), seed=seed)

    scale = resolve_scale(cfg["nla"]["mse_scale"], acts.shape[1])
    mean = train_mean(acts, split["train"], scale)

    def subset(name):
        idx = split[name]
        return [expl[i] for i in idx], acts[idx]

    val_expl, val_acts = subset("val")
    test_expl, test_acts = subset("test")

    # ----------------------------------------------------------------- model
    ar = ActivationReconstructor(
        cfg["model"]["target_name"],
        cfg["model"]["layer"],
        cfg["templates"]["ar_prompt"],
        tcfg.get("max_explanation_tokens", 80),
        **model_opts(cfg),
    ).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": ar.body.parameters(), "lr": float(tcfg["lr"])},
            {"params": ar.head.parameters(), "lr": float(tcfg["head_lr"])},
        ],
        weight_decay=0.0,
        foreach=False,
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    save_dir = Path(tcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log = open(save_dir / "metrics.jsonl", "w")

    def log_row(row):
        log.write(json.dumps(row) + "\n")
        log.flush()

    start = ar_scores(ar, val_expl, val_acts, mean, scale)
    print(f"[ar_sft] step 0 val: {start}")
    log_row({"step": 0, "split": "val", **start})

    # ------------------------------------------------------------ training
    rng = random.Random(seed)
    train_idx = list(split["train"])
    bs = tcfg["batch_size"]
    best_fve, step = float("-inf"), 0

    for epoch in range(tcfg["epochs"]):
        rng.shuffle(train_idx)
        ar.train()
        pbar = tqdm(range(0, len(train_idx), bs), desc=f"epoch {epoch}")
        for i in pbar:
            idx = train_idx[i:i + bs]

            def train_step():
                opt.zero_grad()      # first, so a retried step starts from clean gradients
                with torch.autocast(device, enabled=use_amp):
                    loss = ar.loss([expl[j] for j in idx], acts[idx], scale)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                return loss

            loss = retry_on_cuda_oom(train_step, label=f"ar_sft step {step}")
            step += 1

            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                log_row({"step": step, "split": "train", "loss": loss.item()})

        val = ar_scores(ar, val_expl, val_acts, mean, scale)
        log_row({"step": step, "epoch": epoch, "split": "val", **val})
        print(f"[ar_sft] epoch {epoch} val: {val}")

        if val["fve"] > best_fve:
            best_fve = val["fve"]
            ar.save(str(save_dir))
            print(f"[ar_sft] saved best (val FVE {best_fve:.4f})")

    # ---------------------------------------------------------------- test
    best = ActivationReconstructor.load(str(save_dir)).to(device)
    test = ar_scores(best, test_expl, test_acts, mean, scale)
    report = {"val_fve_best": best_fve, "test": test, "n_test": len(test_expl)}
    with open(save_dir / "ar_sft_eval.json", "w") as f:
        json.dump(report, f, indent=2)
    log.close()

    print("\n" + "=" * 60)
    print("AR SFT — held-out test (warm-start explanations)")
    print("=" * 60)
    for k, v in test.items():
        print(f"{k:<16} {v:>10.4f}")
    print("Pass: fve clearly > 0 and fve_shuffled <= ~0 (touchstone 1-2)")


if __name__ == "__main__":
    main()
