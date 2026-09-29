"""
nla/gui/dashboard.py

Training dashboard: live view of the pipeline from its log and metrics files.

It never loads a model or initialises CUDA, so it is safe to run next to a
training job that is using the whole GPU.

    python -m nla.gui.dashboard --config configs/gpt2_small.yaml   # http://127.0.0.1:8001

Also mounted at /dashboard by the Thought Reader server (nla.gui).

Sources:
    logs/pipeline_<run>.log              stage headers + tqdm progress lines
    <stage save_dir>/metrics.jsonl       per-step / per-eval metrics
    <data.output_dir>/nla_meta.yaml      datagen completion + injection scale
    nvidia-smi                           GPU load, memory, temperature (optional)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"
STAGES = ["datagen", "ar_sft", "av_sft", "rl", "eval"]

_HEADER = re.compile(r"^\[(datagen|ar_sft|av_sft|rl|eval)\] python -m ")
_TQDM = re.compile(
    r"^(?P<desc>[^:|]*?):\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<rate>[\d.?]+)(?P<unit>s/it|it/s)"
    r"(?:,\s*(?P<post>[^\]]*))?\]"
)


# ============================================================================
# Parsing
# ============================================================================

def _hms(s: str) -> Optional[float]:
    if "?" in s:
        return None
    secs = 0.0
    for part in s.split(":"):
        secs = secs * 60 + float(part)
    return secs


def parse_tqdm(line: str) -> Optional[Dict]:
    m = _TQDM.search(line.strip())
    if not m:
        return None
    rate = None if "?" in m["rate"] else float(m["rate"])
    sec_per_it = None if rate is None else (rate if m["unit"] == "s/it" else 1 / max(rate, 1e-9))
    post = {}
    for kv in (m["post"] or "").split(","):
        if "=" in kv:
            k, v = kv.strip().split("=", 1)
            post[k] = v
    return {
        "desc": m["desc"].strip(),
        "n": int(m["n"]),
        "total": int(m["total"]),
        "pct": int(m["pct"]),
        "elapsed_s": _hms(m["elapsed"]),
        "eta_s": _hms(m["eta"]),
        "sec_per_it": sec_per_it,
        "postfix": post,
    }


# housekeeping bars from HF/tqdm that must not replace a stage's training bar
_IGNORED_BARS = ("Loading", "Writing", "Fetching", "Downloading", "Resolving", "Generating")


def _numeric(post: Dict[str, str]) -> Dict[str, float]:
    out = {}
    for k, v in post.items():
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def parse_pipeline_log(text: str) -> Dict:
    """
    Stage order, the live progress bar of each stage, per-update series
    (tqdm postfix values, e.g. RL reward/KL - available even when a
    stage's metrics.jsonl is still buffered), recent messages, and whether
    the pipeline ended (ok / failed).
    """
    lines = [l for chunk in text.split("\n") for l in chunk.split("\r")]
    started: List[str] = []
    progress: Dict[str, Dict] = {}
    series: Dict[str, Dict[tuple, Dict]] = {}
    messages: List[str] = []
    finished, failed = False, None

    current = None
    for raw in lines:
        line = raw.strip()
        if not line or set(line) == {"="}:
            continue
        h = _HEADER.match(line)
        if h:
            current = h.group(1)
            started.append(current)
            continue
        if "[OK] Pipeline complete" in line:
            finished = True
        m = re.search(r"\[ERROR\] stage (\w+) failed", line)
        if m:
            failed = m.group(1)
        t = parse_tqdm(line)
        if t:
            if current and not t["desc"].startswith(_IGNORED_BARS):
                progress[current] = t
                vals = _numeric(t["postfix"])
                if vals:
                    # key by (bar, n): tqdm redraws the same step several times
                    series.setdefault(current, {})[(t["desc"], t["n"])] = {
                        "bar": t["desc"], "n": t["n"], "total": t["total"], **vals,
                    }
        elif current:
            messages.append(line if line.startswith("[") else f"[{current}] {line}")

    return {
        "started": started,
        "current": started[-1] if started and not finished and not failed else None,
        "progress": progress,
        "series": {s: list(v.values()) for s, v in series.items()},
        "finished": finished,
        "failed": failed,
        "messages": messages[-40:],
    }


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass   # a line being written right now
    return rows


def gpu_stats() -> Optional[Dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, util, used, total, temp = [x.strip() for x in out.split(",")]
        return {"name": name, "util": float(util), "mem_used": float(used),
                "mem_total": float(total), "temp": float(temp)}
    except Exception:
        return None


# ============================================================================
# State assembly
# ============================================================================

class DashboardState:
    def __init__(self, cfg: Dict, log_path: Optional[str] = None):
        self.cfg = cfg
        name = cfg["experiment"]["name"]
        guess = Path("logs") / f"pipeline_{name.rsplit('_L', 1)[0]}.log"
        self.log_path = Path(log_path) if log_path else guess
        self.data_dir = Path(cfg["data"]["output_dir"])
        self.dirs = {
            "ar_sft": Path(cfg["ar_sft"]["save_dir"]),
            "av_sft": Path(cfg["av_sft"]["save_dir"]),
            "rl": Path(cfg["rl"]["save_dir"]),
        }

    def _stage_status(self, log: Dict) -> Dict[str, str]:
        status = {s: "pending" for s in STAGES}
        if (self.data_dir / "nla_meta.yaml").exists():
            status["datagen"] = "done"
        # a log may cover only a later run (e.g. continued RL): finished checkpoints still count
        if (self.dirs["ar_sft"] / "ar.pt").exists():
            status["ar_sft"] = "done"
        if (self.dirs["av_sft"] / "av_config.json").exists():
            status["av_sft"] = "done"
        for s in log["started"]:
            status[s] = "done"
        if log["current"]:
            status[log["current"]] = "running"
        if log["failed"]:
            status[log["failed"]] = "failed"
        if (self.dirs["rl"] / "av" / "nla_eval.json").exists() and log["finished"]:
            status["eval"] = "done"
        return status

    def summary(self) -> Dict:
        text = ""
        if self.log_path.exists():
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        log = parse_pipeline_log(text)

        meta = {}
        if (self.data_dir / "nla_meta.yaml").exists():
            meta = yaml.safe_load((self.data_dir / "nla_meta.yaml").read_text(encoding="utf-8"))

        ar_eval = None
        p = self.dirs["ar_sft"] / "ar_sft_eval.json"
        if p.exists():
            ar_eval = json.loads(p.read_text())

        final_eval = None
        p = self.dirs["rl"] / "av" / "nla_eval.json"
        if p.exists():
            final_eval = json.loads(p.read_text())

        return {
            "time": time.time(),
            "run": {
                "name": self.cfg["experiment"]["name"],
                "target": self.cfg["model"]["target_name"],
                "layer": self.cfg["model"]["layer"],
                "num_samples": meta.get("num_samples"),
                "injection_scale": meta.get("injection_scale"),
                "log": str(self.log_path),
                "log_age_s": time.time() - self.log_path.stat().st_mtime if self.log_path.exists() else None,
            },
            "stages": self._stage_status(log),
            "current": log["current"],
            "progress": log["progress"],
            "finished": log["finished"],
            "failed": log["failed"],
            "messages": log["messages"],
            "gpu": gpu_stats(),
            "ar_sft_eval": ar_eval,
            "final_eval": final_eval,
        }

    def metrics(self) -> Dict:
        text = ""
        if self.log_path.exists():
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return {
            "jsonl": {s: read_jsonl(d / "metrics.jsonl") for s, d in self.dirs.items()},
            "live": parse_pipeline_log(text)["series"],
        }


# ============================================================================
# Routes
# ============================================================================

def dashboard_router(state: DashboardState) -> APIRouter:
    r = APIRouter()

    @r.get("/dashboard")
    def page():
        return FileResponse(STATIC / "dashboard.html")

    @r.get("/api/dashboard/summary")
    def summary():
        return state.summary()

    @r.get("/api/dashboard/metrics")
    def metrics():
        return state.metrics()

    return r


def create_dashboard_app(state: DashboardState) -> FastAPI:
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="TinyNLA training dashboard")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.include_router(dashboard_router(state))

    @app.middleware("http")
    async def no_stale_assets(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def root():
        return RedirectResponse("/dashboard")

    return app


def main(argv=None) -> None:
    import argparse
    import webbrowser

    import uvicorn

    from nla.config import load_config, parse_overrides, utf8_stdio   # torch-free

    utf8_stdio()
    p = argparse.ArgumentParser(description="TinyNLA training dashboard (no GPU use)")
    p.add_argument("--config", default="configs/gpt2_small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--log", help="pipeline log (default logs/pipeline_<run>.log)")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    state = DashboardState(cfg, args.log)
    url = f"http://127.0.0.1:{args.port}/dashboard"
    print(f"[dashboard] {state.log_path} -> {url}")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(create_dashboard_app(state), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
