"""
nla/datagen.py

Build the NLA dataset: (context, raw activation, warm-start explanation).

For each document we take its first `max_context_tokens` tokens, pick a
random position t >= min_position, and record the target model's block-K
output at t. The model is causal, so the activation at t depends only on
tokens 0..t, and one right-padded batched forward serves many snippets.

Activations are stored RAW (no normalisation): the loss/metric scale and
the AV injection scale are applied later, from nla_meta.yaml.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from nla import model_adapter as ma
from nla.dataset import make_split, save_split
from nla.explainers import CachedExplainer, build_explainer
from nla.metrics import rescale, resolve_scale
from nla.utils import retry_on_cuda_oom

ACTIVATIONS_FILE = "activations.pt"
BUFFER_FILE = "buffer.pt"
META_FILE = "nla_meta.yaml"


def _documents(cfg: Dict) -> Iterator[str]:
    data = cfg["data"]
    ds = load_dataset(
        data["source"],
        name=data.get("source_config"),
        split="train",
        streaming=data.get("streaming", False),
    )
    for row in ds:
        yield row["text"]


def extract_activations(cfg: Dict, device: str) -> Dict:
    """Stage 1: sample snippets and record block-K activations."""
    data = cfg["data"]
    layer = cfg["model"]["layer"]
    rng = random.Random(cfg["experiment"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["target_name"])
    from nla.runs import load_target   # local import: nla.runs imports this module
    model = load_target(cfg, device)
    pad_id = tokenizer.eos_token_id

    n_target = data["num_samples"]
    max_tokens = data["max_context_tokens"]
    min_pos = data["min_position"]

    # --- collect snippets -------------------------------------------------
    snippets: List[List[int]] = []
    positions: List[int] = []
    for text in _documents(cfg):
        ids = tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"]
        if len(ids) < min_pos + 2:
            continue
        snippets.append(ids)
        positions.append(rng.randint(min_pos, len(ids) - 2))  # leave a next token
        if len(snippets) >= n_target:
            break

    # --- batched forward ----------------------------------------------------
    acts = torch.empty(len(snippets), ma.hidden_size(model))
    bs = data.get("extract_batch_size", 16)

    for start in tqdm(range(0, len(snippets), bs), desc="extract"):
        batch = snippets[start:start + bs]
        width = max(len(s) for s in batch)
        ids = torch.full((len(batch), width), pad_id)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, s in enumerate(batch):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1

        hidden = retry_on_cuda_oom(
            lambda: ma.block_output(model, layer, input_ids=ids.to(device), attention_mask=mask.to(device)),
            label=f"extract batch {start // bs}")

        pos = torch.tensor(positions[start:start + bs])
        rows = torch.arange(len(batch))
        acts[start:start + len(batch)] = hidden[rows, pos].float().cpu()

    contexts = [
        tokenizer.decode(s[: p + 1]) for s, p in zip(snippets, positions)
    ]
    next_tokens = [s[p + 1] for s, p in zip(snippets, positions)]

    return {
        "fingerprint": extraction_fingerprint(cfg),
        "activations": acts,
        "positions": positions,
        "contexts": contexts,
        "next_tokens": next_tokens,
        "layer": layer,
        "target_name": cfg["model"]["target_name"],
    }


def extraction_fingerprint(cfg: Dict) -> Dict:
    """Everything that determines the extracted activations (for safe reuse)."""
    data = cfg["data"]
    return {"target": cfg["model"]["target_name"], "layer": cfg["model"]["layer"],
            "source": data["source"], "source_config": data.get("source_config"),
            "num_samples": data["num_samples"], "max_context_tokens": data["max_context_tokens"],
            "min_position": data["min_position"], "seed": cfg["experiment"]["seed"]}


def _reusable_activations(cfg: Dict, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rec = torch.load(path, weights_only=False, map_location="cpu")
    except Exception:            # noqa: BLE001 - a half-written file is simply re-extracted
        return False
    return rec.get("fingerprint") == extraction_fingerprint(cfg)


def add_explanations(cfg: Dict, record: Dict) -> Dict:
    """Stage 2: warm-start explanations for every context."""
    ecfg = cfg["explainer"]
    cached = CachedExplainer(build_explainer(ecfg), ecfg["cache_path"])
    record["explanations"] = cached.explain_all(
        record["contexts"], batch_size=ecfg.get("batch_size", 16)
    )
    record["explainer"] = cached.explainer.cache_namespace
    return record


def write_meta(cfg: Dict, record: Dict, out_dir: Path) -> Dict:
    """nla_meta.yaml: everything downstream needs to interpret the buffer."""
    acts = record["activations"]
    d = acts.shape[1]
    norms = acts.norm(dim=-1)
    scale = resolve_scale(cfg["nla"]["mse_scale"], d)

    meta = {
        "kind": "nla_dataset",
        "target_name": record["target_name"],
        "layer": record["layer"],
        "d_model": d,
        "num_samples": len(acts),
        "mse_scale": cfg["nla"]["mse_scale"],
        "injection_scale": cfg["nla"].get("injection_scale"),
        "templates": cfg["templates"],
        "explainer": record.get("explainer"),
        "source": cfg["data"]["source"],
        "activation_norm": {
            "mean": float(norms.mean()),
            "median": float(norms.median()),
            "p99": float(norms.quantile(0.99)),
            "max": float(norms.max()),
        },
    }
    with open(out_dir / META_FILE, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    # the FVE reference mean is computed from TRAIN rows only, by the trainers
    _ = scale
    return meta


def build(cfg: Dict, device: str, stage: str = "all") -> None:
    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["experiment"]["seed"]

    reuse = stage == "all" and _reusable_activations(cfg, out_dir / ACTIVATIONS_FILE)
    if reuse:
        # a restart after a crash in the (long) explanation stage: extraction is done
        print(f"[datagen] reusing {out_dir / ACTIVATIONS_FILE} (same model, layer, data and seed)")
    if stage in ("extract", "all") and not reuse:
        record = extract_activations(cfg, device)
        torch.save(record, out_dir / ACTIVATIONS_FILE)
        save_split(make_split(len(record["activations"]), seed), str(out_dir), seed)
        print(f"[datagen] {len(record['activations'])} activations -> {out_dir}")
        torch.cuda.empty_cache()

    if stage in ("explain", "all"):
        record = torch.load(out_dir / ACTIVATIONS_FILE, weights_only=False)
        record = add_explanations(cfg, record)
        torch.save(record, out_dir / BUFFER_FILE)
        meta = write_meta(cfg, record, out_dir)
        n_unique = len(set(record["explanations"]))
        print(
            f"[datagen] buffer: {meta['num_samples']} rows, "
            f"{n_unique} unique explanations, "
            f"median activation norm {meta['activation_norm']['median']:.1f}"
        )


def train_mean(activations: torch.Tensor, train_idx: List[int], scale) -> torch.Tensor:
    """FVE reference: mean of rescaled TRAIN activations (not re-normalised)."""
    return rescale(activations[train_idx], scale).mean(dim=0)
