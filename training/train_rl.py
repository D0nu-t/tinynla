"""
training/train_rl.py

Stage 3: GRPO on the AV (reward = AR reconstruction), AR trained alongside.
Starts from the SFT checkpoints; the SFT AV is also the frozen KL reference.

    python -m training.train_rl --config configs/gpt2_small.yaml

The number that decides whether the NLA "reads thoughts" is logged every
`eval_every` steps on validation activations:
    av_fve  vs  summary_fve   (touchstone 1b: AV must end ABOVE the summary)
Both are scored by the SAME current AR, so the comparison is fair.

Outputs (rl.save_dir): av/, ar/, metrics.jsonl
"""

import json
import random
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from nla.ar import ActivationReconstructor
from nla.av import ActivationVerbalizer
from nla.rl import GRPOConfig, frozen_reference, grpo_step
from nla.runs import load_data
from nla.utils import cli_config, resolve_device, retry_on_cuda_oom, set_seed
from training.train_av_sft import decode_and_score

load_dotenv()


def main():
    cfg, _ = cli_config(__doc__)
    seed = cfg["experiment"]["seed"]
    set_seed(seed, deterministic=False)
    device = resolve_device(cfg)
    rcfg = cfg["rl"]

    data = load_data(cfg)
    acts, expl = data.acts, data.explanations

    # The KL anchor is ALWAYS the SFT policy (the paper penalises divergence from
    # the initial policy), even when resuming from an earlier RL run.
    sft_dir = cfg["av_sft"]["save_dir"]
    init_from = rcfg.get("init_from")            # e.g. checkpoints/<run>/rl to continue training
    if cfg.get("lora"):
        # the SFT reference is a frozen second adapter on the policy's own base
        av = ActivationVerbalizer.load(f"{init_from}/av" if init_from else sft_dir).to(device)
        ref = frozen_reference(av, sft_dir=sft_dir)
    else:
        sft_av = ActivationVerbalizer.load(sft_dir).to(device)
        ref = frozen_reference(sft_av)
        if init_from:
            del sft_av
            av = ActivationVerbalizer.load(f"{init_from}/av").to(device)
        else:
            av = sft_av
    if init_from:
        print(f"[rl] resuming AV and AR from {init_from}")
        ar = ActivationReconstructor.load(f"{init_from}/ar").to(device)
    else:
        ar = ActivationReconstructor.load(cfg["ar_sft"]["save_dir"]).to(device)

    grpo = GRPOConfig(
        group_size=rcfg["group_size"],
        max_new_tokens=rcfg["max_new_tokens"],
        temperature=rcfg["temperature"],
        kl_beta=rcfg["kl_beta"],
        log_reward=rcfg.get("log_reward", False),
        micro_batch=rcfg.get("micro_batch"),
    )
    av_opt = torch.optim.AdamW([p for p in av.parameters() if p.requires_grad],
                               lr=float(rcfg["lr"]), foreach=False)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=float(rcfg["ar_lr"]), foreach=False)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ar_scaler = torch.amp.GradScaler(enabled=use_amp)

    rng = random.Random(seed)
    val_idx = rng.sample(data.split["val"], min(rcfg["eval_samples"], len(data.split["val"])))
    val_acts, val_sum = acts[val_idx], [expl[i] for i in val_idx]

    save_dir = Path(rcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log = open(save_dir / "metrics.jsonl", "w")

    def evaluate(step):
        scores, samples = decode_and_score(av, ar, val_acts, val_sum, data.mean, data.scale)
        log.write(json.dumps({"step": step, "split": "val", **scores}) + "\n")
        log.flush()
        gap = scores["av_fve"] - scores["summary_fve"]
        print(f"\n[rl] step {step}: {scores}  (AV - summary = {gap:+.4f})")
        print(f"   e.g. {samples[0][:160]}")
        if torch.cuda.is_available():
            print(f"   peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return scores

    best = evaluate(0)["av_fve"]
    train_idx = list(data.split["train"])
    bs = rcfg["batch_size"]

    pbar = tqdm(range(1, rcfg["steps"] + 1), desc="grpo")
    for step in pbar:
        idx = rng.sample(train_idx, bs)
        # grpo_step clears gradients before each update, so a whole step can be retried
        stats = retry_on_cuda_oom(lambda: grpo_step(
            av, ref, ar, av_opt, ar_opt, acts[idx], data.scale, grpo,
            scaler=scaler, ar_scaler=ar_scaler,
        ), label=f"rl step {step}")
        log.write(json.dumps({"step": step, "split": "train", **stats}) + "\n")
        log.flush()   # live dashboards read this file
        pbar.set_postfix(r=f"{stats['reward']:.3f}", kl=f"{stats['kl']:.3f}",
                         ok=f"{stats['parse_ok']:.2f}")

        if step % rcfg["eval_every"] == 0 or step == rcfg["steps"]:
            scores = retry_on_cuda_oom(lambda: evaluate(step), label=f"rl eval {step}")
            if scores["av_fve"] > best:
                best = scores["av_fve"]
                av.save(str(save_dir / "av"))
                ar.save(str(save_dir / "ar"))
                print(f"[rl] saved best (val AV FVE {best:.4f})")

    if not (save_dir / "av" / "av_config.json").exists():
        # never improved on SFT: still save so downstream tools find a run
        av.save(str(save_dir / "av"))
        ar.save(str(save_dir / "ar"))
    log.close()


if __name__ == "__main__":
    main()
