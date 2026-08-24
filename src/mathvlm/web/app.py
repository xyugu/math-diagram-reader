"""The viewer's HTTP layer: JSON endpoints over ``web/runs.py``, and the static page.

GET  /api/meta      schema and render constants the page draws with
GET  /api/runs      runs with saved predictions
GET  /api/samples   ?run=<key>: one summary per sample, in dataset order
GET  /api/sample    ?run=<key>&id=<id>: raw output, decoded prediction, ground truth, eval
GET  /api/image     ?run=<key>&id=<id>: the input image
POST /api/check     {"text": <output JSON>}: schema validation + geometric warnings
POST /api/render    {"text": <output JSON>}: PNG from render.py, the training renderer
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mathvlm.config import Config, get_config
from mathvlm.render import render_page
from mathvlm.schema import (
    GRID_MAX,
    GRID_MIN,
    SCHEMA_VERSION,
    UNNAMED_PREFIX,
    validate_output_json,
)
from mathvlm.web.runs import CheckResult, RunInfo, RunStore, Sample, SampleSummary, check_output

STATIC_DIR = Path(__file__).parent / "static"

# render.py settings the page needs to draw a diagram the way the training images look
_RENDER_KEYS = ("diagram_pad", "figure_width_in", "ink", "line_width", "point_marker_size",
                "label_fontsize", "label_offset_pt", "mark_size")  # fmt: skip


class OutputText(BaseModel):
    text: str  # a complete output, as JSON text


@contextmanager
def _not_found() -> Iterator[None]:
    try:
        yield
    except KeyError as e:
        raise HTTPException(404, str(e.args[0])) from e


def create_app(cfg: Config | None = None, outputs_dir: Path | None = None) -> FastAPI:
    cfg = cfg or get_config()
    store = RunStore(outputs_dir or cfg.paths.resolved("outputs_dir"), cfg)
    render_lock = threading.Lock()  # render.py scopes matplotlib's global rc per call
    app = FastAPI(title="mathvlm viewer")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "grid": [GRID_MIN, GRID_MAX],
            "unnamed_prefix": UNNAMED_PREFIX,
            "collinear_tol": cfg.schema_.consistency_collinear_tol,
            "render": {k: getattr(cfg.render, k) for k in _RENDER_KEYS},
        }

    @app.get("/api/runs")
    def runs() -> list[RunInfo]:
        return store.runs()

    @app.get("/api/samples")
    def samples(run: str) -> list[SampleSummary]:
        with _not_found():
            return store.samples(run)

    @app.get("/api/sample")
    def sample(run: str, id: str) -> Sample:
        with _not_found():
            return store.sample(run, id)

    @app.get("/api/image")
    def image(run: str, id: str) -> FileResponse:
        with _not_found():
            return FileResponse(store.image_path(run, id))

    @app.post("/api/check")
    def check(body: OutputText) -> CheckResult:
        return check_output(body.text)

    @app.post("/api/render", response_class=Response)
    def render(body: OutputText) -> Response:
        report = validate_output_json(body.text)
        if not report.ok:
            raise HTTPException(422, "; ".join(report.errors))
        with render_lock:
            image = render_page(report.value).image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
