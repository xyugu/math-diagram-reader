"""Per-example metrics and their aggregation. README.md, design invariant 7: every change
here needs a test in tests/test_metrics.py.

Per example (``evaluate_example``)
----------------------------------
- **Parse** (eval/parse.py): direct / fenced / failed, schema validity, failure category.
- **Diagram detection**: GT kind (``diagram``/``null``) against predicted kind
  (``diagram``/``null``/``failed``), read from the decoded output even when the diagram
  is schema-invalid.
- **Text**: normalized edit distance (eval/normalize.py) and raw edit distance against the
  GT text, plus normalized exact match. Scored whenever the output decodes to an object
  with a string ``text``, even if its diagram is invalid; otherwise NED is 1.0.
- **Structure** (when GT or a valid prediction has a diagram): true/false positives and
  false negatives per category -- points, segments, circles, marks, labels -- in two
  variants that differ only in which predicted point stands for which GT point
  (eval/matching.py):
    - ``strict``: labeled points by name, ``_P`` points by position.
    - ``geometric``: every point by position, names ignored.
  The predicted diagram is renamed into GT names through the correspondence, then each
  category is compared as a multiset of canonical forms (schema.canonicalize), so list
  order and endpoint order never matter. Circles match by center (radius is geometry);
  labels match on target *and* normalized text; marks match whole (an equal_length
  group with one segment missing is one false positive plus one false negative). A
  prediction that is missing, invalid or null counts as an empty diagram, so failing on
  hard examples never helps.
- **Geometry** (only when GT and prediction both have a valid diagram): distance of each
  geometrically matched point after alignment, in GT grid units; the same distances
  without alignment; coverage (matched GT points / GT points); the fitted scale; and
  radius error for circles whose centers match.

Aggregates (``summarize``) are micro-averaged: counts and distances are pooled across
examples before dividing. A ratio with a zero denominator is ``None``, never 0 or 1.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from mathvlm.config import EvalConfig, get_config
from mathvlm.eval.matching import (
    Correspondence,
    Similarity,
    align,
    match_geometric,
    match_strict,
    rename_into,
)
from mathvlm.eval.normalize import ned, normalize_label, normalize_text
from mathvlm.eval.parse import ParsedOutput, parse_output
from mathvlm.schema import Diagram, Transcription, canonicalize

CATEGORIES = ("points", "segments", "circles", "marks", "labels")
MARK_TYPES = ("right_angle", "equal_length", "parallel", "equal_angle")
# The categories plus a per-type breakdown of marks ("marks.right_angle", ...).
KEYS = (*CATEGORIES, *(f"marks.{t}" for t in MARK_TYPES))
VARIANTS = ("strict", "geometric")


def _mean(xs) -> float | None:
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def _median(xs) -> float | None:
    xs = list(xs)
    return statistics.median(xs) if xs else None


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        """2TP / (2TP + FP + FN): the harmonic mean of P and R where both exist, 0 when
        nothing matched, None when there was nothing to find and nothing predicted."""
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else None

    def to_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}  # fmt: skip


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def _multisets(d: Diagram | None) -> dict[str, Counter]:
    """Each category as a multiset of canonical, order-free keys."""
    out = {c: Counter() for c in KEYS}
    if d is None:
        return out
    cd = canonicalize(d)
    out["points"] = Counter(cd.points.keys())
    out["segments"] = Counter(cd.segments)
    out["circles"] = Counter(c.center for c in d.circles)
    out["marks"] = Counter(cd.marks)
    out["labels"] = Counter((*lb[:-1], normalize_label(lb[-1])) for lb in cd.labels)
    for m in cd.marks:
        out[f"marks.{m[0]}"][m] += 1
    return out


def structure_counts(gt: Diagram | None, pred: Diagram | None) -> dict[str, Counts]:
    """Counts per category. ``pred`` must already use GT names for matched points
    (``matching.rename_into``); ``None`` on either side is an empty diagram."""
    G, P = _multisets(gt), _multisets(pred)
    return {
        c: Counts(
            sum((G[c] & P[c]).values()),
            sum((P[c] - G[c]).values()),
            sum((G[c] - P[c]).values()),
        )
        for c in G
    }


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    n_gt: int
    n_pred: int
    errors: list[float]  # per geometrically matched GT point, after alignment (grid units)
    raw_errors: list[float]  # the same pairs, before alignment
    strict_errors: list[float]  # per strictly matched (by name) GT point, after alignment
    scale: float
    radius_errors: list[float]

    @property
    def coverage(self) -> float:
        return len(self.errors) / self.n_gt

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "n_matched": len(self.errors),
            "coverage": self.coverage,
            "error_mean": _mean(self.errors),
            "error_median": _median(self.errors),
            "raw_error_mean": _mean(self.raw_errors),
            "strict_error_mean": _mean(self.strict_errors),
            "strict_error_median": _median(self.strict_errors),
            "scale": self.scale,
            "radius_error_mean": _mean(self.radius_errors),
        }


def _distances(gt: Diagram, pred: Diagram, T: Similarity, corr: Correspondence, aligned: bool):
    out = []
    for p, g in sorted(corr.pairs.items(), key=lambda kv: kv[1]):
        px = np.asarray(pred.points[p], dtype=float)
        q = T.apply(px)[0] if aligned else px
        out.append(float(np.linalg.norm(q - np.asarray(gt.points[g], dtype=float))))
    return out


def geometry(
    gt: Diagram, pred: Diagram, T: Similarity, geo: Correspondence, strict: Correspondence
) -> Geometry:
    """Point and radius errors in GT grid units. ``errors`` measure the shape (points
    matched by position); ``strict_errors`` also catch labels put on the wrong point,
    which position alone cannot see when the figure is symmetric."""
    radius_errors: list[float] = []
    gt_r: dict[str, list[float]] = defaultdict(list)
    pred_r: dict[str, list[float]] = defaultdict(list)
    for c in gt.circles:
        gt_r[c.center].append(float(c.radius))
    for c in pred.circles:
        if c.center in geo.pairs:
            pred_r[geo.pairs[c.center]].append(T.scale * float(c.radius))
    for center in sorted(gt_r):
        if pred_r.get(center):  # concentric circles: best one-to-one pairing by radius
            cost = np.abs(np.subtract.outer(gt_r[center], pred_r[center]))
            rows, cols = linear_sum_assignment(cost)
            radius_errors.extend(float(cost[i, j]) for i, j in zip(rows, cols, strict=True))
    return Geometry(
        n_gt=len(gt.points),
        n_pred=len(pred.points),
        errors=_distances(gt, pred, T, geo, aligned=True),
        raw_errors=_distances(gt, pred, T, geo, aligned=False),
        strict_errors=_distances(gt, pred, T, strict, aligned=True),
        scale=T.scale,
        radius_errors=radius_errors,
    )


# ---------------------------------------------------------------------------
# One example
# ---------------------------------------------------------------------------


def text_render_score(pred: str, gt: str) -> float | None:
    """Stub for a rendering-based text metric: render both strings and compare the
    images (in the spirit of CDM), which forgives every spelling that renders the same,
    not just the ones normalize.py knows. Not implemented in Phase 1; always None."""
    return None


@dataclass
class ExampleResult:
    id: str
    meta: dict[str, Any]  # family, kind, ... from the dataset manifest
    parsed: ParsedOutput
    gt_kind: str  # "diagram" | "null"
    pred_kind: str  # "diagram" | "null" | "failed"
    text: dict[str, Any]
    structure: dict[str, dict[str, Counts]] | None  # variant -> category -> counts
    geometry: Geometry | None
    pairs: dict[str, dict[str, str]] = field(default_factory=dict)  # variant -> pred->gt

    def to_row(self) -> dict[str, Any]:
        p = self.parsed
        return {
            "id": self.id,
            **self.meta,
            "gt": self.gt_kind,
            "pred": self.pred_kind,
            "parse": {
                "how": p.how,
                "stage": p.report.stage,
                "failure": p.failure,
                "errors": list(p.report.errors[:5]),
            },  # fmt: skip
            "text": self.text,
            "structure": None
            if self.structure is None
            else {
                v: {c: [n.tp, n.fp, n.fn] for c, n in cats.items()}
                for v, cats in self.structure.items()
            },  # fmt: skip
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "pairs": self.pairs,
            "output": p.raw,
        }


def text_scores(pred: str | None, gt: str) -> dict[str, Any]:
    if pred is None:
        return {"has_text": False, "ned": 1.0, "ned_raw": 1.0, "exact": False}
    a, b = normalize_text(pred), normalize_text(gt)
    return {"has_text": True, "ned": ned(a, b), "ned_raw": ned(pred, gt), "exact": a == b}


def evaluate_example(
    gt: Transcription,
    raw: str | None,
    *,
    id: str = "",
    meta: dict[str, Any] | None = None,
    truncated: bool = False,
    cfg: EvalConfig | None = None,
) -> ExampleResult:
    """Score one raw model output (``None``: no prediction) against its ground truth."""
    cfg = cfg or get_config().eval
    parsed = parse_output(raw, truncated)
    pred = parsed.transcription
    gd = gt.diagram
    pd = pred.diagram if pred is not None else None

    text = text_scores(parsed.text, gt.text)
    text["escape_suspect"] = parsed.escape_suspect

    structure = geo = None
    pairs: dict[str, dict[str, str]] = {}
    if gd is not None and pd is not None:
        T = align(gd.points, pd.points, cfg)
        corr = {
            "strict": match_strict(gd.points, pd.points, T, cfg),
            "geometric": match_geometric(gd.points, pd.points, T, cfg),
        }
        structure = {
            v: structure_counts(gd, rename_into(pd, corr[v].pairs, gd.points)) for v in VARIANTS
        }
        geo = geometry(gd, pd, T, corr["geometric"], corr["strict"])
        pairs = {v: dict(sorted(corr[v].pairs.items())) for v in VARIANTS}
    elif gd is not None or pd is not None:
        structure = {v: structure_counts(gd, pd) for v in VARIANTS}
        pairs = {v: {} for v in VARIANTS}

    return ExampleResult(
        id=id,
        meta=dict(meta or {}),
        parsed=parsed,
        gt_kind="null" if gd is None else "diagram",
        pred_kind=parsed.diagram_kind,
        text=text,
        structure=structure,
        geometry=geo,
        pairs=pairs,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _rate(k: float, n: int) -> float | None:
    return k / n if n else None


_ERROR_REASONS = (
    ("Field required", "missing key"),
    ("Extra inputs", "extra key"),
    ("references unknown point", "unknown point"),
    ("invalid point name", "bad point name"),
    ("valid integer", "not an integer"),
    ("less than or equal", "out of range"),
    ("greater than or equal", "out of range"),
    ("greater than", "out of range"),
    ("duplicate", "duplicate"),
    ("degenerate", "degenerate"),
    ("valid string", "not a string"),
    ("valid dictionary", "not an object"),
    ("valid list", "not a list"),
    ("valid tuple", "not a list"),
)


def schema_error_kind(err: str) -> str:
    """Coarse bucket for one validation error: ``"<field>: <reason>"``, e.g.
    ``"diagram.points: not an integer"``. Point names and list indices are dropped."""
    loc, _, msg = err.partition(": ")
    head = ".".join(loc.split(".")[:2])
    reason = next((r for key, r in _ERROR_REASONS if key in msg), "other")
    return f"{head}: {reason}"


def _structure_summary(results: list[ExampleResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(results)}
    for v in VARIANTS:
        totals: dict[str, Counts] = defaultdict(Counts)
        for r in results:
            for c, n in r.structure[v].items():
                totals[c] = totals[c] + n
        all_ = sum((totals[c] for c in CATEGORIES), Counts())
        out[v] = {c: totals[c].to_dict() for c in KEYS}
        out[v]["all"] = all_.to_dict()
    return out


def summarize(results: list[ExampleResult]) -> dict[str, Any]:
    n = len(results)
    how = Counter(r.parsed.how or "failed" for r in results)
    stage = Counter(r.parsed.report.stage for r in results)
    failures = Counter(r.parsed.failure for r in results if r.parsed.failure)
    schema_errors = Counter(
        schema_error_kind(r.parsed.report.errors[0])
        for r in results
        if r.parsed.report.stage == "invalid" and r.parsed.report.errors
    )

    confusion = {g: {p: 0 for p in ("diagram", "null", "failed")} for g in ("diagram", "null")}
    for r in results:
        confusion[r.gt_kind][r.pred_kind] += 1
    n_gt_diagram = sum(confusion["diagram"].values())

    with_text = [r for r in results if r.text["has_text"]]
    scoped = [r for r in results if r.structure is not None]
    both = [r for r in results if r.geometry is not None]
    errors = [e for r in both for e in r.geometry.errors]
    strict_errors = [e for r in both for e in r.geometry.strict_errors]

    return {
        "n": n,
        "n_gt_diagram": n_gt_diagram,
        "n_gt_null": n - n_gt_diagram,
        "parse": {
            "json_parse_rate": _rate(how["direct"] + how["fenced"], n),
            "json_parse_rate_strict": _rate(how["direct"], n),
            "schema_valid_rate": _rate(stage["ok"], n),
            "tiers": {k: how[k] for k in ("direct", "fenced", "failed")},
            "failures": dict(failures.most_common()),
            "schema_errors": dict(schema_errors.most_common()),
            "escape_suspect": sum(bool(r.text["escape_suspect"]) for r in results),
        },
        "detection": {
            "accuracy": _rate(confusion["diagram"]["diagram"] + confusion["null"]["null"], n),
            "confusion": confusion,
        },
        "text": {
            "ned": _mean(r.text["ned"] for r in results),
            "ned_raw": _mean(r.text["ned_raw"] for r in results),
            "exact": _mean(float(r.text["exact"]) for r in results),
            "n_with_text": len(with_text),
            "ned_with_text": _mean(r.text["ned"] for r in with_text),
            "render": None,  # text_render_score is a stub
        },
        "structure": _structure_summary(scoped),
        "structure_both_valid": _structure_summary(both),
        "geometry": {
            "n": len(both),
            "error_mean": _mean(errors),
            "error_median": _median(errors),
            "raw_error_mean": _mean(e for r in both for e in r.geometry.raw_errors),
            "strict_error_mean": _mean(strict_errors),
            "strict_error_median": _median(strict_errors),
            "coverage": _rate(len(errors), sum(r.geometry.n_gt for r in both)),
            "scale_median": _median(r.geometry.scale for r in both),
            "radius_error_mean": _mean(e for r in both for e in r.geometry.radius_errors),
        },
    }


def summarize_by(results: list[ExampleResult], key: str) -> dict[str, dict[str, Any]]:
    """A compact summary per value of ``meta[key]`` (e.g. per scene family)."""
    groups: dict[str, list[ExampleResult]] = defaultdict(list)
    for r in results:
        groups[str(r.meta.get(key))].append(r)
    out = {}
    for k in sorted(groups):
        s = summarize(groups[k])
        out[k] = {
            "n": s["n"],
            "schema_valid_rate": s["parse"]["schema_valid_rate"],
            "detection_accuracy": s["detection"]["accuracy"],
            "text_ned": s["text"]["ned"],
            "f1_strict": s["structure"]["strict"]["all"]["f1"] if s["structure"]["n"] else None,
            "f1_geometric": (
                s["structure"]["geometric"]["all"]["f1"] if s["structure"]["n"] else None
            ),
            "point_error_mean": s["geometry"]["error_mean"],
        }
    return out
