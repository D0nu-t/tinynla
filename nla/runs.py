"""
nla/runs.py

One place that knows how a run is laid out on disk, so trainers, the
evaluator, the reader and the GUI all agree.

    <data.output_dir>/        buffer.pt, split.json, nla_meta.yaml
    <ar_sft.save_dir>/        ar.pt, ar_config.json
    <av_sft.save_dir>/        model/, av_config.json
    <rl.save_dir>/av, /ar     after GRPO
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from nla.datagen import BUFFER_FILE, train_mean
from nla.dataset import load_or_create_split
from nla.metrics import rescale, resolve_scale


def model_opts(cfg: Dict) -> Dict:
    """LoRA / dtype options shared by AV and AR constructors."""
    return {
        "lora": cfg.get("lora"),
        "base_dtype": cfg.get("base_dtype", "float32"),
    }


def load_target(cfg: Dict, device: str):
    """
    The target model for reading activations. On CUDA it uses the config's
    base_dtype (fp16 for LoRA configs): a fp32 Qwen-0.5B target (2 GB) next to the
    AV and AR would not fit a 4 GB GPU. Callers cast captured states to float.
    """
    from transformers import AutoModelForCausalLM

    dtype = torch.float32
    if device == "cuda" and cfg.get("base_dtype", "float32") != "float32":
        dtype = getattr(torch, cfg["base_dtype"])
    return AutoModelForCausalLM.from_pretrained(cfg["model"]["target_name"], dtype=dtype).to(device).eval()


@dataclass
class DataBundle:
    record: Dict
    split: Dict[str, List[int]]
    scale: Optional[float]
    mean: torch.Tensor        # mean of rescaled TRAIN activations
    variance: float           # E||g - mean||^2 over TRAIN (per-sample FVE denominator)

    @property
    def acts(self) -> torch.Tensor:
        return self.record["activations"]

    @property
    def explanations(self) -> List[str]:
        return self.record["explanations"]


def load_data(cfg: Dict) -> DataBundle:
    data_dir = cfg["data"]["output_dir"]
    record = torch.load(Path(data_dir) / BUFFER_FILE, weights_only=False)
    acts = record["activations"]
    split = load_or_create_split(data_dir, n=len(acts), seed=cfg["experiment"]["seed"])
    scale = resolve_scale(cfg["nla"]["mse_scale"], acts.shape[1])
    mean = train_mean(acts, split["train"], scale)
    g = rescale(acts[split["train"]], scale)
    variance = float(((g - mean) ** 2).sum(dim=-1).mean())
    return DataBundle(record, split, scale, mean, variance)


def model_dirs(cfg: Dict, stage: str = "auto") -> Tuple[str, str]:
    """
    (av_dir, ar_dir) for a stage: "sft", "rl", or "auto" (rl if trained).
    """
    rl_dir = Path(cfg.get("rl", {}).get("save_dir", "__none__"))
    has_rl = (rl_dir / "av" / "av_config.json").exists()
    if stage == "rl" or (stage == "auto" and has_rl):
        return str(rl_dir / "av"), str(rl_dir / "ar")
    return cfg["av_sft"]["save_dir"], cfg["ar_sft"]["save_dir"]
