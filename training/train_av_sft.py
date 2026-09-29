"""
training/train_av_sft.py

Stage 2: supervised AV training (activation -> warm-start explanation),
with the activation injected at <|inject|>.

    python -m training.train_av_sft --config configs/gpt2_small.yaml

Each epoch, `eval_samples` validation activations are decoded (T=1) and the
explanations are scored by the stage-1 AR. Two numbers matter:
    av_fve        FVE of AV explanations            (should approach summary_fve)
    summary_fve   FVE of the warm-start summaries   (the bar RL must beat later)
plus av_fve_shuffled (<= 0) and the closing-tag parse rate.
"""

import json
import random
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from nla.ar import ActivationReconstructor
from nla.av import ActivationVerbalizer, choose_injection_scale, token_embedding_norms
from nla.datagen import BUFFER_FILE, META_FILE, train_mean
from nla.dataset import load_or_create_split
from nla.evaluate import ar_scores
from nla.metrics import resolve_scale
from nla.runs import model_opts
from nla.utils import cli_config, is_cuda_oom, resolve_device, set_seed, wait_for_gpu

load_dotenv()


@torch.no_grad()
def decode_and_score(av, ar, acts, summaries, mean, scale, batch=16):
    av.eval()
    outs = []
    for i in range(0, len(acts), batch):
        outs += av.generate(acts[i:i + batch], n_samples=1, temperature=1.0)
    texts = [o["text"] for o in outs]
    av_scores = ar_scores(ar, texts, acts, mean, scale)
    sum_scores = ar_scores(ar, summaries, acts, mean, scale, controls=False)
    av.train()
    return {
        "av_fve": av_scores["fve"],
        "av_fve_shuffled": av_scores["fve_shuffled"],
        "summary_fve": sum_scores["fve"],
        "parse_ok": sum(o["ok"] for o in outs) / len(outs),
        "unique_frac": len(set(texts)) / len(texts),
    }, texts


def main():
    cfg, _ = cli_config(__doc__)

    seed = cfg["experiment"]["seed"]
    set_seed(seed, deterministic=False)
    device = resolve_device(cfg)
    tcfg = cfg["av_sft"]
    tmpl = cfg["templates"]

    data_dir = Path(cfg["data"]["output_dir"])
    record = torch.load(data_dir / BUFFER_FILE, weights_only=False)
    acts, expl = record["activations"], record["explanations"]
    split = load_or_create_split(str(data_dir), n=len(acts), seed=seed)
    d = acts.shape[1]
    scale = resolve_scale(cfg["nla"]["mse_scale"], d)
    mean = train_mean(acts, split["train"], scale)

    # -------------------------------------------------- injection scale
    inj = cfg["nla"].get("injection_scale")
    if inj is None:
        inj = choose_injection_scale(
            acts[split["train"]],
            token_embedding_norms(cfg["model"]["target_name"]),
            d,
        )
    inj = float(inj)
    print(f"[av_sft] injection_scale = {inj:.3f}")

    meta_path = data_dir / META_FILE
    meta = yaml.safe_load(open(meta_path))
    meta["injection_scale"] = inj
    yaml.safe_dump(meta, open(meta_path, "w"), sort_keys=False)

    # ------------------------------------------------------------ models
    av = ActivationVerbalizer(
        cfg["model"]["target_name"],
        tmpl["av_prompt"], tmpl["av_response_open"], tmpl["av_response_close"],
        injection_scale=inj,
        max_explanation_tokens=tcfg["max_explanation_tokens"],
        **model_opts(cfg),
    ).to(device)
    ar = ActivationReconstructor.load(cfg["ar_sft"]["save_dir"]).to(device).eval()
    for q in ar.parameters():
        q.requires_grad_(False)

    rng = random.Random(seed)
    val_idx = rng.sample(split["val"], min(tcfg["eval_samples"], len(split["val"])))
    val_acts = acts[val_idx]
    val_summaries = [expl[i] for i in val_idx]

    opt = torch.optim.AdamW(av.parameters(), lr=float(tcfg["lr"]), foreach=False)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    save_dir = Path(tcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log = open(save_dir / "metrics.jsonl", "w")

    train_idx = list(split["train"])
    bs, accum = tcfg["batch_size"], tcfg["grad_accum"]
    best, step = float("-inf"), 0

    for epoch in range(tcfg["epochs"]):
        rng.shuffle(train_idx)
        av.train()
        pbar = tqdm(range(0, len(train_idx), bs), desc=f"epoch {epoch}")
        for n, i in enumerate(pbar):
            idx = train_idx[i:i + bs]
            try:
                with torch.autocast(device, enabled=use_amp):
                    loss = av.sft_loss(acts[idx], [expl[j] for j in idx]) / accum
                scaler.scale(loss).backward()
            except Exception as e:  # noqa: BLE001
                if not is_cuda_oom(e):
                    raise
                # a failed backward may have added PART of this micro-batch's gradient;
                # drop the window so far (this optimizer step uses fewer micro-batches)
                opt.zero_grad()
                wait_for_gpu(1, label=f"av_sft micro-step {n}")
                continue

            if (n + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    pbar.set_postfix(loss=f"{loss.item() * accum:.3f}")
                    log.write(json.dumps({"step": step, "loss": loss.item() * accum}) + "\n")
                    log.flush()

        scores, samples = decode_and_score(av, ar, val_acts, val_summaries, mean, scale)
        log.write(json.dumps({"step": step, "epoch": epoch, "split": "val", **scores}) + "\n")
        log.flush()
        print(f"[av_sft] epoch {epoch}: {scores}")
        for s, ref in list(zip(samples, val_summaries))[:3]:
            print(f"   AV : {s[:150]}\n   REF: {ref[:150]}")

        if scores["av_fve"] > best:
            best = scores["av_fve"]
            av.save(str(save_dir))
            print(f"[av_sft] saved best (val AV FVE {best:.4f})")

    log.close()
    print("\nPass (touchstone): av_fve near summary_fve, av_fve_shuffled <= ~0, parse_ok high")


if __name__ == "__main__":
    main()
