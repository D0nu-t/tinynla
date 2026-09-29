"""
nla/gui/server.py

"Thought reader" GUI backend: a thin FastAPI layer over nla.read.NLAReader.

    python -m nla.gui --config configs/gpt2_small.yaml [--stage auto|sft|rl] [--port 8000]

Endpoints (all JSON):
    GET  /api/run                       run metadata + held-out eval summary
    POST /api/tokenize  {text}          tokens, per-token norm, eligibility
    POST /api/read      {text, position, n_samples}
                                        Server-Sent Events, one explanation per event
    POST /api/controls  {text, position}
    POST /api/compare   {text, position, explanation}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC = Path(__file__).parent / "static"


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class PositionIn(TextIn):
    position: int = -1


class ReadIn(PositionIn):
    n_samples: int = Field(3, ge=1, le=8)


class ReportIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    answer: Optional[str] = Field(None, max_length=4000)   # None: the model answers
    n_samples: int = Field(3, ge=1, le=5)


class ContrastIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    baseline: str = Field(min_length=1, max_length=4000)
    answer: Optional[str] = Field(None, max_length=4000)   # None: the model answers `prompt`
    n_samples: int = Field(4, ge=1, le=6)


class CompareIn(PositionIn):
    explanation: str = Field(min_length=1, max_length=2000)


def create_app(reader) -> FastAPI:
    """`reader` is an NLAReader (or anything with the same methods, for tests)."""
    app = FastAPI(title="TinyNLA Thought Reader")
    gpu = threading.Lock()   # one model, one GPU: serialise model calls

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def no_stale_assets(request, call_next):
        # local tool: always revalidate so a code update is never hidden by the cache
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    if hasattr(reader, "cfg"):   # real reader: also serve the training dashboard
        from nla.gui.dashboard import DashboardState, dashboard_router
        app.include_router(dashboard_router(DashboardState(reader.cfg)))

    @app.get("/report")
    def report_page():
        return FileResponse(STATIC / "report.html")

    @app.post("/api/report")
    def report(body: ReportIn):
        try:
            with gpu:
                return reader.report(body.prompt, body.answer, n_samples=body.n_samples)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/compare")
    def compare_page():
        return FileResponse(STATIC / "compare.html")

    @app.post("/api/contrast")
    def contrast(body: ContrastIn):
        try:
            with gpu:
                return reader.contrast(body.prompt, body.baseline, body.answer, n_samples=body.n_samples)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/run")
    def run():
        return reader.run_info()

    @app.post("/api/tokenize")
    def tokenize(body: TextIn):
        with gpu:
            return {"tokens": reader.tokenize(body.text)}

    def _check_position(text: str, position: int) -> None:
        with gpu:
            n = len(reader.tokenize(text))
        if not -n <= position < n:
            raise HTTPException(422, f"position {position} out of range for {n} tokens")

    @app.post("/api/read")
    def read(body: ReadIn):
        _check_position(body.text, body.position)

        def events():
            try:
                for i in range(body.n_samples):
                    with gpu:
                        item = next(iter(reader.read(body.text, body.position, 1)))
                    item["index"] = i
                    yield f"data: {json.dumps(item)}\n\n"
                yield "event: done\ndata: {}\n\n"
            except Exception as e:  # surface model errors to the page
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/controls")
    def controls(body: PositionIn):
        _check_position(body.text, body.position)
        with gpu:
            return reader.controls(body.text, body.position)

    @app.post("/api/compare")
    def compare(body: CompareIn):
        _check_position(body.text, body.position)
        with gpu:
            return reader.compare_vectors(body.text, body.position, body.explanation)

    return app


def main(argv: Optional[list] = None) -> None:
    import argparse
    import webbrowser

    import uvicorn

    from nla.read import NLAReader
    from nla.utils import load_config, parse_overrides, utf8_stdio

    utf8_stdio()

    p = argparse.ArgumentParser(description="TinyNLA thought reader GUI")
    p.add_argument("--config", default="configs/gpt2_small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--stage", default="auto", choices=["auto", "sft", "rl"])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    print("[gui] loading target, AV and AR ...")
    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    reader = NLAReader(cfg, stage=args.stage)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[gui] {reader.stage} run ready -> {url}")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(reader), host="127.0.0.1", port=args.port, log_level="warning")
