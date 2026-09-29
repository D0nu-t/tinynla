"""
nla/rl.py

GRPO for the Activation Verbalizer, following Fraser-Taliente et al. 2026
and the kitft Miles config, reduced to a single-GPU on-policy loop:

  1. For each activation h, sample a group of G explanations z ~ AV(.|h).
  2. Reward r = -MSE_nrm(AR(z), h)   (direction-only MSE; failed parse -> -2.0,
     or -log MSE with log_reward=True).
  3. Advantage = (r - mean_group) / (std_group + eps).
  4. AV loss = -mean(A * mean_t log pi(z_t)) + beta * k2-KL(pi || pi_sft),
     k2 = 0.5 * (log pi - log pi_ref)^2 per token.
  5. The AR keeps training supervised on the sampled explanations, so it
     learns to read the AV's evolving language (kitft: "AR supervised").

On-policy (one gradient step per rollout), so the PPO ratio is exactly 1
and clipping is unnecessary.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from nla.metrics import normalized_mse


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    kl_beta: float = 0.01
    failed_reward: float = -2.0
    log_reward: bool = False
    adv_eps: float = 1e-4
    grad_clip: float = 1.0
    train_ar: bool = True
    # policy update in chunks of this many sequences (gradients accumulated; same
    # maths). With a large vocabulary (Qwen: 152k) the per-token log-softmax over
    # the whole batch is the memory peak; None = whole batch at once.
    micro_batch: Optional[int] = None


REF_ADAPTER = "sft_reference"


def frozen_reference(av, sft_dir: Optional[str] = None) -> Union[torch.nn.Module, str]:
    """
    The KL reference policy: the SFT verbalizer (the paper anchors RL to the
    initial policy, not to the pretrained base).

    With LoRA the SFT adapter is loaded a second time, frozen, as a named adapter
    on the SAME base (a few MB instead of a full copy: ~1 GB saved at Qwen-0.5B);
    returns its name and grpo_step switches to it for the reference pass. Note
    `disable_adapter()` would give the PRETRAINED base, the wrong anchor.
    Without LoRA (GPT-2), a frozen copy of the policy LM in half precision on GPU.
    """
    if getattr(av, "lora", None):
        if sft_dir is None:
            raise ValueError("LoRA reference needs the SFT AV directory (sft_dir)")
        av.lm.load_adapter(str(Path(sft_dir) / "model"), adapter_name=REF_ADAPTER, is_trainable=False)
        av.lm.set_adapter("default")
        return REF_ADAPTER
    ref = copy.deepcopy(av.lm).eval()
    if av.device.type == "cuda":
        ref = ref.half()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def rewards_for(ar, texts, ok, targets, scale, cfg: GRPOConfig) -> torch.Tensor:
    with torch.no_grad():
        was = ar.training
        ar.eval()
        with torch.autocast(ar.device.type, enabled=ar.device.type == "cuda"):
            pred = ar(texts).float()
        ar.train(was)
    mse = normalized_mse(pred, targets.to(pred.device), scale).cpu()
    r = -torch.log(mse.clamp(min=1e-6)) if cfg.log_reward else -mse
    failed = cfg.failed_reward if not cfg.log_reward else -math.log(2.0)
    ok_t = torch.tensor(ok)
    return torch.where(ok_t, r, torch.full_like(r, failed)), mse


def _reference_logprobs(av, ref_lm, targets, responses):
    if isinstance(ref_lm, str):                  # LoRA: switch to the frozen SFT adapter
        av.lm.set_adapter(ref_lm)
        try:
            return av.sequence_logprobs(targets, responses)[0]
        finally:
            av.lm.set_adapter("default")         # also restores requires_grad on "default"
    policy_lm = av.lm
    av.lm = ref_lm                               # same prompt/injection path
    try:
        return av.sequence_logprobs(targets, responses)[0]
    finally:
        av.lm = policy_lm


def group_advantages(rewards: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    r = rewards.view(-1, group)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def grpo_step(
    av,
    ref_lm,
    ar,
    av_opt,
    ar_opt,
    acts: torch.Tensor,
    scale,
    cfg: GRPOConfig,
    scaler=None,
    ar_scaler=None,
) -> Dict[str, float]:
    """One rollout + update on a batch of activations [B, d]."""
    device_type = av.device.type
    use_amp = device_type == "cuda"
    g = cfg.group_size

    # ---------------------------------------------------------- rollout
    av.eval()
    samples = av.generate(
        acts, n_samples=g, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
    )
    texts = [s["text"] for s in samples]
    ok = [s["ok"] for s in samples]
    responses = [s["token_ids"] for s in samples]
    targets = acts.repeat_interleave(g, dim=0)

    rewards, mse = rewards_for(ar, texts, ok, targets, scale, cfg)
    adv = group_advantages(rewards, g, cfg.adv_eps).to(av.device)

    # ----------------------------------------------------- policy update
    # Stay in eval mode: dropout would make log pi differ from the policy
    # that produced the samples (and from the reference, inflating KL).
    # Gradients do not need train mode.
    av.eval()
    n_seq = len(responses)
    mb = cfg.micro_batch or n_seq
    av_opt.zero_grad()
    pg_total, kl_total, all_tokens = 0.0, 0.0, []
    for lo in range(0, n_seq, mb):
        part = slice(lo, lo + mb)
        with torch.autocast(device_type, enabled=use_amp):
            logp, mask = av.sequence_logprobs(targets[part], responses[part])
            with torch.no_grad():
                ref_logp = _reference_logprobs(av, ref_lm, targets[part], responses[part])
        tokens = mask.sum(dim=1).clamp(min=1)
        seq_logp = (logp * mask).sum(dim=1) / tokens
        pg = -(adv[part] * seq_logp).sum() / n_seq          # mean over ALL sequences
        kl_tok = 0.5 * ((logp - ref_logp.float()) ** 2)
        kl = ((kl_tok * mask).sum(dim=1) / tokens).sum() / n_seq
        chunk_loss = pg + cfg.kl_beta * kl
        (scaler.scale(chunk_loss) if scaler is not None else chunk_loss).backward()
        pg_total += pg.item()
        kl_total += kl.item()
        all_tokens.append(tokens)
    tokens = torch.cat(all_tokens)

    if scaler is not None:
        scaler.unscale_(av_opt)
        torch.nn.utils.clip_grad_norm_(av.parameters(), cfg.grad_clip)
        scaler.step(av_opt)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(av.parameters(), cfg.grad_clip)
        av_opt.step()

    # ------------------------------------------------ AR supervised update
    ar_loss = float("nan")
    ok_idx = [i for i, o in enumerate(ok) if o]
    if cfg.train_ar and ok_idx:
        ar.eval()   # dropout off here too: the reward model should be deterministic
        with torch.autocast(device_type, enabled=use_amp):
            loss_ar = ar.loss([texts[i] for i in ok_idx], targets[ok_idx], scale)
        ar_opt.zero_grad()
        if ar_scaler is not None:
            ar_scaler.scale(loss_ar).backward()
            ar_scaler.unscale_(ar_opt)
            torch.nn.utils.clip_grad_norm_(ar.parameters(), cfg.grad_clip)
            ar_scaler.step(ar_opt)
            ar_scaler.update()
        else:
            loss_ar.backward()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), cfg.grad_clip)
            ar_opt.step()
        ar_loss = loss_ar.item()

    return {
        "reward": rewards.mean().item(),
        "mse_nrm": mse.mean().item(),
        "kl": kl_total,
        "pg_loss": pg_total,
        "ar_loss": ar_loss,
        "parse_ok": sum(ok) / len(ok),
        "mean_tokens": tokens.float().mean().item(),
    }
