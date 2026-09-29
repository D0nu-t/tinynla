"""
End-to-end smoke test: every stage as a real subprocess, tiny sizes, CPU.

    python -m pytest tests/test_e2e_smoke.py -m slow -s

Checks that the pipeline wiring works (files produced, reports readable,
GUI serves a real run) - not that the tiny model learns anything.
Needs network access for the streamed FineWeb-Edu documents.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = "data/e2e_smoke"
CK = "checkpoints/e2e_smoke"

SETS = [
    "device=cpu",
    f"data.output_dir={SMOKE}",
    "data.num_samples=40",
    "data.max_context_tokens=48",
    "explainer.kind=rule",
    f"explainer.cache_path={SMOKE}/cache.jsonl",
    "ar_sft.epochs=1",
    "ar_sft.batch_size=8",
    f"ar_sft.save_dir={CK}/ar_sft",
    "av_sft.epochs=1",
    "av_sft.batch_size=4",
    "av_sft.grad_accum=1",
    "av_sft.eval_samples=2",
    f"av_sft.save_dir={CK}/av_sft",
    "rl.steps=2",
    "rl.eval_every=1",
    "rl.eval_samples=2",
    "rl.batch_size=2",
    "rl.group_size=2",
    "rl.max_new_tokens=12",
    f"rl.save_dir={CK}/rl",
    "eval.max_test=2",
    "eval.n_samples=2",
]


def run(module, *extra):
    args = [sys.executable, "-m", module, "--config", "configs/gpt2_small.yaml"]
    for s in SETS:
        args += ["--set", s]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    res = subprocess.run(
        args + list(extra), cwd=ROOT, env=env, capture_output=True,
        encoding="utf-8", errors="replace",   # children print UTF-8 (nla.utils.utf8_stdio)
    )
    assert res.returncode == 0, f"{module} failed:\n{res.stdout[-2000:]}\n{res.stderr[-4000:]}"
    return res.stdout


@pytest.mark.slow
def test_pipeline_end_to_end():
    run("training.datagen")
    for f in ("buffer.pt", "split.json", "nla_meta.yaml"):
        assert (ROOT / SMOKE / f).exists(), f

    out = run("training.train_ar_sft")
    assert "AR SFT" in out
    report = json.loads((ROOT / CK / "ar_sft" / "ar_sft_eval.json").read_text())
    assert {"fve", "fve_shuffled", "fve_mean"} <= set(report["test"])
    assert abs(report["test"]["fve_mean"]) < 1e-5

    run("training.train_av_sft")
    assert (ROOT / CK / "av_sft" / "av_config.json").exists()
    meta = (ROOT / SMOKE / "nla_meta.yaml").read_text()
    assert "injection_scale: null" not in meta          # chosen and recorded

    run("training.train_rl")
    assert (ROOT / CK / "rl" / "av" / "av_config.json").exists()
    rows = [json.loads(l) for l in (ROOT / CK / "rl" / "metrics.jsonl").read_text().splitlines()]
    assert any(r.get("split") == "train" and "kl" in r for r in rows)

    out = run("training.eval_nla")
    ev = json.loads((ROOT / CK / "rl" / "av" / "nla_eval.json").read_text())
    assert set(ev["checks"]) >= {"1_beats_mean", "1b_beats_input_summary", "2_shuffle_control"}


@pytest.mark.slow
def test_reader_and_gui_on_real_run():
    """Requires test_pipeline_end_to_end to have produced the smoke run."""
    from fastapi.testclient import TestClient

    from nla.gui.server import create_app
    from nla.read import NLAReader
    from nla.utils import load_config, parse_overrides

    os.chdir(ROOT)
    cfg = load_config("configs/gpt2_small.yaml", overrides=parse_overrides(SETS))
    reader = NLAReader(cfg, device="cpu")
    assert reader.stage == "rl"

    text = "The quick brown fox jumps over the lazy dog because it wanted to reach the"
    toks = reader.tokenize(text)
    assert toks[-1]["eligible"] and not toks[0]["eligible"]

    items = list(reader.read(text, -1, 2))
    assert len(items) == 2 and all(-50 < i["fve"] <= 1 for i in items)
    c = reader.controls(text, -1)
    assert abs(c["mean_fve"]) < 5

    client = TestClient(create_app(reader))
    assert client.get("/api/run").json()["stage"] == "rl"
    res = client.post("/api/read", json={"text": text, "position": -1, "n_samples": 1})
    assert "event: done" in res.text
    cmp = client.post("/api/compare", json={"text": text, "position": -1, "explanation": items[0]["text"] or "x"})
    assert len(cmp.json()["dims"]) == 16
