"""Saved predictions, read for the viewer: which runs exist, their samples, one sample in
full, and checking an edited output.

A run is a directory under ``<outputs_dir>/infer/`` holding ``predictions.jsonl`` (written
by ``mathvlm.infer``). Its key is that directory's path relative to ``infer/``,
``<split>/<name>``, where the split may itself contain a slash (``pgdp5k/val/sft20k``).
``run.json`` records the split's data directory (without it, the split's default location
is used), and the run's eval, if one has been run, is ``per_example.jsonl`` under the same
relative path in ``<outputs_dir>/eval/``.

Read-only, and independent of FastAPI: ``web/app.py`` only routes requests here, so this
module is tested directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from mathvlm.config import Config, get_config
from mathvlm.eval.metrics import CATEGORIES, Counts
from mathvlm.eval.parse import parse_output
from mathvlm.schema import (
    ValidationReport,
    consistency_issues,
    output_to_obj,
    validate_output_json,
)
from mathvlm.splits import split_dir

Stage = Literal["ok", "invalid", "parse_error"]


class CheckResult(BaseModel):
    """One output's validation: the stage ``schema_valid_rate`` counts, plus the soft
    geometric warnings (``schema.consistency_issues``) of a valid diagram."""

    stage: Stage
    errors: list[str]
    warnings: list[str]
    canonical: Any = None  # schema.output_to_obj of a valid output: what a download holds


class RunInfo(BaseModel):
    key: str  # path under <outputs_dir>/infer/: "<split>/<name>"
    split: str
    name: str
    n: int  # predictions written so far
    adapter: str | None
    has_eval: bool


class SampleSummary(BaseModel):
    id: str
    index: int
    family: str | None
    stage: Stage
    pred: Literal["diagram", "null", "failed"]  # ParsedOutput.diagram_kind
    # From the run's eval. None when there is no eval, or the score does not apply.
    text_exact: bool | None = None
    structure_f1: float | None = None  # strict, categories pooled as in the eval summary
    point_error: float | None = None  # mean, grid units, after alignment


class Sample(BaseModel):
    id: str
    index: int
    family: str | None
    kind: str | None
    has_image: bool
    raw: str | None
    truncated: bool
    n_tokens: int | None
    how: Literal["direct", "fenced"] | None
    failure: str | None  # eval/parse.py FAILURES key when the output did not parse
    check: CheckResult
    pred: Any  # the decoded output, valid or not; None when it did not parse
    gt: Any  # the ground-truth file's content; None without one
    eval: dict[str, Any] | None  # the run's per_example.jsonl row, minus the raw output


def check_report(report: ValidationReport) -> CheckResult:
    t = report.value
    has_diagram = t is not None and t.diagram is not None
    return CheckResult(
        stage=report.stage,
        errors=list(report.errors),
        warnings=consistency_issues(t.diagram) if has_diagram else [],
        canonical=None if t is None else output_to_obj(t),
    )


def check_output(text: str) -> CheckResult:
    """Validate an edited output. Strict (no fence stripping): the page wrote this JSON."""
    return check_report(validate_output_json(text))


@lru_cache(maxsize=16)
def _read_jsonl_cached(path: str, mtime_ns: int, size: int) -> tuple[dict, ...]:
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # blank, or cut short while mathvlm.infer appends to the file
            if isinstance(row, dict):
                rows.append(row)
    return tuple(rows)


def read_jsonl(path: Path) -> tuple[dict, ...]:
    """Rows of a JSONL file, skipping lines that do not decode. Cached until the file
    changes; callers must not mutate the rows."""
    st = path.stat()
    return _read_jsonl_cached(str(path), st.st_mtime_ns, st.st_size)


def _eval_scores(row: dict | None) -> dict[str, Any]:
    if row is None:
        return {}
    strict = (row.get("structure") or {}).get("strict") or {}
    total = sum((Counts(*strict[c]) for c in CATEGORIES if c in strict), Counts())
    return {
        "text_exact": (row.get("text") or {}).get("exact"),
        "structure_f1": total.f1,
        "point_error": (row.get("geometry") or {}).get("error_mean"),
    }


@dataclass(frozen=True)
class _Run:
    data: Path
    preds: dict[str, dict]  # id -> predictions.jsonl row, in file order
    manifest: dict[str, dict]  # id -> manifest row; empty if the split has no manifest
    evals: dict[str, dict]  # id -> per_example.jsonl row; empty without an eval

    def index(self, sid: str) -> int:
        row = self.manifest.get(sid)
        return row["index"] if row is not None else list(self.preds).index(sid)


class RunStore:
    """The runs under one outputs directory."""

    def __init__(self, outputs_dir: Path, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.infer_root = outputs_dir / "infer"
        self.eval_root = outputs_dir / "eval"

    def keys(self) -> list[str]:
        if not self.infer_root.is_dir():
            return []
        found = self.infer_root.rglob("predictions.jsonl")
        return sorted(p.parent.relative_to(self.infer_root).as_posix() for p in found)

    def _dir(self, key: str) -> Path:
        """A run's directory. Only keys from ``keys()`` resolve, so a request can never
        name a path outside the infer directory."""
        if key not in self.keys():
            raise KeyError(f"unknown run {key!r}")
        return self.infer_root / key

    @staticmethod
    def _meta(run_dir: Path) -> dict:
        p = run_dir / "run.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _eval_path(self, key: str) -> Path:
        return self.eval_root / key / "per_example.jsonl"

    def runs(self) -> list[RunInfo]:
        out = []
        for key in self.keys():
            run_dir = self.infer_root / key
            meta = self._meta(run_dir)
            split, _, name = key.rpartition("/")
            out.append(
                RunInfo(
                    key=key,
                    split=meta.get("split", split),
                    name=meta.get("name", name),
                    n=len(read_jsonl(run_dir / "predictions.jsonl")),
                    adapter=meta.get("adapter"),
                    has_eval=self._eval_path(key).exists(),
                )
            )
        return out

    def _load(self, key: str) -> _Run:
        run_dir = self._dir(key)
        meta = self._meta(run_dir)
        if meta.get("data"):
            data = Path(meta["data"])
        else:
            data = split_dir(meta.get("split") or key.rpartition("/")[0], self.cfg)
        manifest_path = data / "manifest.jsonl"
        eval_path = self._eval_path(key)
        return _Run(
            data=data,
            preds={r["id"]: r for r in read_jsonl(run_dir / "predictions.jsonl") if "id" in r},
            manifest={r["id"]: r for r in read_jsonl(manifest_path)}
            if manifest_path.exists()
            else {},
            evals={r["id"]: r for r in read_jsonl(eval_path)} if eval_path.exists() else {},
        )

    def samples(self, key: str) -> list[SampleSummary]:
        """Every sample the run has a prediction for, in dataset order."""
        run = self._load(key)
        out = []
        for sid, row in run.preds.items():
            parsed = parse_output(row.get("output"), bool(row.get("truncated")))
            out.append(
                SampleSummary(
                    id=sid,
                    index=run.index(sid),
                    family=run.manifest.get(sid, {}).get("family"),
                    stage=parsed.report.stage,
                    pred=parsed.diagram_kind,
                    **_eval_scores(run.evals.get(sid)),
                )
            )
        return sorted(out, key=lambda s: s.index)

    def sample(self, key: str, sid: str) -> Sample:
        run = self._load(key)
        if sid not in run.preds:
            raise KeyError(f"run {key!r} has no prediction for {sid!r}")
        row = run.preds[sid]
        m = run.manifest.get(sid, {})
        parsed = parse_output(row.get("output"), bool(row.get("truncated")))
        gt_path = run.data / m["gt"] if "gt" in m else None
        ev = run.evals.get(sid)
        return Sample(
            id=sid,
            index=run.index(sid),
            family=m.get("family"),
            kind=m.get("kind"),
            has_image="image" in m,
            raw=row.get("output"),
            truncated=bool(row.get("truncated")),
            n_tokens=row.get("n_tokens"),
            how=parsed.how,
            failure=parsed.failure,
            check=check_report(parsed.report),
            pred=parsed.obj,
            gt=json.loads(gt_path.read_text()) if gt_path and gt_path.exists() else None,
            eval=None if ev is None else {k: v for k, v in ev.items() if k != "output"},
        )

    def image_path(self, key: str, sid: str) -> Path:
        """The input image of a sample in the run; it must lie inside the split's data
        directory."""
        run = self._load(key)
        m = run.manifest.get(sid)
        if sid not in run.preds or m is None or "image" not in m:
            raise KeyError(f"run {key!r} has no image for {sid!r}")
        root = run.data.resolve()
        p = (root / m["image"]).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise KeyError(f"no image file for {sid!r}")
        return p
