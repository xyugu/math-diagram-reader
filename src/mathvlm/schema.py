"""The diagram schema. FROZEN: see README.md, design invariant 1.

This is the only module that defines what a valid model output looks like. Changing
the *shape* of anything here is a schema change: bump ``SCHEMA_VERSION``, add a
changelog entry below, and tell the user.

CHANGELOG
---------
1.1 (2026-10-24, approved by the user before implementation)
    - ``right_angle`` marks gain a required ``rays`` pair: ``{"type": "right_angle",
      "at": "C", "rays": ["A", "B"]}`` means angle A-C-B is 90 degrees. In v1, ``at``
      alone was ambiguous wherever 3+ segments meet (e.g. the foot of an altitude),
      and the renderer could not tell which corner to draw the square in.
    - New mark ``{"type": "equal_angle", "angles": [["A","B","D"], ["D","B","C"]]}``.
      v1 had no way to mark an angle bisector.
1.0
    The original brief.

What this module provides
-------------------------
- Pydantic models: ``Transcription`` (the whole output) and ``Diagram``.
- Hard validation (``validate_output`` / ``validate_output_json``): shape, types,
  value ranges, point-name syntax, referential integrity, duplicates. Anything that
  fails here is "invalid" in the eval's ``schema_valid_rate``.
- ``canonicalize()``: one hashable comparison form per structure, so endpoint order
  and list order can never affect a metric.
- ``to_target_json()``: the deterministic, compact JSON string used as the training
  target. Ordering is canonical, so the target does not depend on generator internals.
- ``consistency_issues()``: *soft* geometric checks (a right angle really is ~90
  degrees, a marked segment is actually drawn, ...). Valid for figures drawn to
  scale, i.e. our synthetic data. Not used to reject model predictions.

Conventions fixed by the schema
-------------------------------
- Coordinates are integers on a ``[grid_min, grid_max]`` grid (0-100), origin top-left,
  y pointing down. ``radius`` is a positive number (int or float) in grid units.
- Visible point labels are one uppercase letter, optionally followed by one or two
  primes or a ``_``-subscript of 1-2 digits: ``A``, ``A'``, ``A''``, ``O_1``. This is
  the same spelling the question text uses in LaTeX, so labels match ``$O_1$``.
- Points that matter but are not labeled in the figure are named ``_P<n>`` (``_P1``,
  ``_P2``, ...; no leading zeros). No other underscore-prefixed names are allowed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    Tag,
    ValidationError,
    model_validator,
)

from mathvlm.config import get_config

SCHEMA_VERSION = "1.1"

_SC = get_config().schema_
GRID_MIN: int = _SC.grid_min
GRID_MAX: int = _SC.grid_max
UNNAMED_PREFIX: str = _SC.unnamed_prefix

_NAMED_RE = re.compile(r"[A-Z](?:'{1,2}|_[0-9]{1,2})?")
_UNNAMED_RE = re.compile(re.escape(UNNAMED_PREFIX) + r"(?:0|[1-9][0-9]*)")


# ---------------------------------------------------------------------------
# Atomic types
# ---------------------------------------------------------------------------


def _check_point_name(name: str) -> str:
    if _NAMED_RE.fullmatch(name) or _UNNAMED_RE.fullmatch(name):
        return name
    raise ValueError(
        f"invalid point name {name!r}: expected an uppercase letter with optional "
        f"primes or _subscript (A, A', O_1), or {UNNAMED_PREFIX}<n> for unlabeled points"
    )


def is_unnamed(name: str) -> bool:
    """True for the ``_P<n>`` names given to points not labeled in the figure."""
    return _UNNAMED_RE.fullmatch(name) is not None


PointName = Annotated[StrictStr, AfterValidator(_check_point_name)]
GridInt = Annotated[StrictInt, Field(ge=GRID_MIN, le=GRID_MAX)]
Coord = tuple[GridInt, GridInt]
Radius = Annotated[StrictInt | StrictFloat, Field(gt=0, le=GRID_MAX - GRID_MIN)]
NonEmptyText = Annotated[StrictStr, Field(min_length=1)]


def _check_segment(seg: tuple[str, str]) -> tuple[str, str]:
    if seg[0] == seg[1]:
        raise ValueError(f"degenerate segment {list(seg)}: endpoints must differ")
    return seg


def _check_angle(ang: tuple[str, str, str]) -> tuple[str, str, str]:
    a, v, b = ang
    if len({a, v, b}) != 3:
        raise ValueError(f"degenerate angle {list(ang)}: the three points must differ")
    return ang


Segment = Annotated[tuple[PointName, PointName], AfterValidator(_check_segment)]
# [end, vertex, end] -- the vertex is always the middle element.
Angle = Annotated[tuple[PointName, PointName, PointName], AfterValidator(_check_angle)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Circles
# ---------------------------------------------------------------------------


class Circle(_Strict):
    center: PointName
    radius: Radius


# ---------------------------------------------------------------------------
# Marks (discriminated on "type")
# ---------------------------------------------------------------------------


def _reject_duplicates(items: list, key, what: str) -> None:
    seen = set()
    for it in items:
        k = key(it)
        if k in seen:
            raise ValueError(f"duplicate {what}: {it!r}")
        seen.add(k)


class RightAngleMark(_Strict):
    type: Literal["right_angle"]
    at: PointName
    rays: tuple[PointName, PointName]

    @model_validator(mode="after")
    def _check(self) -> RightAngleMark:
        _check_angle((self.rays[0], self.at, self.rays[1]))
        return self


class EqualLengthMark(_Strict):
    type: Literal["equal_length"]
    segments: Annotated[list[Segment], Field(min_length=2)]

    @model_validator(mode="after")
    def _check(self) -> EqualLengthMark:
        _reject_duplicates(self.segments, _seg_key, "segment in equal_length mark")
        return self


class ParallelMark(_Strict):
    type: Literal["parallel"]
    segments: Annotated[list[Segment], Field(min_length=2)]

    @model_validator(mode="after")
    def _check(self) -> ParallelMark:
        _reject_duplicates(self.segments, _seg_key, "segment in parallel mark")
        return self


class EqualAngleMark(_Strict):
    type: Literal["equal_angle"]
    angles: Annotated[list[Angle], Field(min_length=2)]

    @model_validator(mode="after")
    def _check(self) -> EqualAngleMark:
        _reject_duplicates(self.angles, _angle_key, "angle in equal_angle mark")
        return self


Mark = Annotated[
    RightAngleMark | EqualLengthMark | ParallelMark | EqualAngleMark,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Labels (two shapes, told apart by which key is present)
# ---------------------------------------------------------------------------


class SegmentLabel(_Strict):
    """A length label on a segment: ``{"on": ["A", "C"], "text": "3"}``."""

    on: Segment
    text: NonEmptyText


class AngleLabel(_Strict):
    """An angle label: ``{"angle": ["A", "B", "C"], "text": "40^\\\\circ"}`` (vertex B)."""

    angle: Angle
    text: NonEmptyText


def _label_kind(v: Any) -> str | None:
    if isinstance(v, SegmentLabel):
        return "segment"
    if isinstance(v, AngleLabel):
        return "angle"
    if isinstance(v, dict):
        has_on, has_angle = "on" in v, "angle" in v
        if has_on and not has_angle:
            return "segment"
        if has_angle and not has_on:
            return "angle"
    return None


Label = Annotated[
    Annotated[SegmentLabel, Tag("segment")] | Annotated[AngleLabel, Tag("angle")],
    Discriminator(
        _label_kind,
        custom_error_type="label_kind",
        custom_error_message="a label needs exactly one of 'on' (segment) or 'angle'",
    ),
]


# ---------------------------------------------------------------------------
# Diagram and the full output
# ---------------------------------------------------------------------------


class Diagram(_Strict):
    # A non-null diagram must contain at least one point; a question with no figure
    # uses "diagram": null instead of an empty scene.
    points: Annotated[dict[PointName, Coord], Field(min_length=1)]
    # Missing list keys default to empty, so a prediction for a figure with no circles
    # need not spell out "circles": []. Targets always include all five keys.
    segments: list[Segment] = Field(default_factory=list)
    circles: list[Circle] = Field(default_factory=list)
    marks: list[Mark] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)

    @model_validator(mode="after")
    def _referential_integrity(self) -> Diagram:
        errors: list[str] = []
        known = self.points.keys()

        def need(name: str, where: str) -> None:
            if name not in known:
                errors.append(f"{where} references unknown point {name!r}")

        for s in self.segments:
            for n in s:
                need(n, f"segment {list(s)}")
        for c in self.circles:
            need(c.center, "circle center")
        for m in self.marks:
            for n in _mark_point_names(m):
                need(n, f"{m.type} mark")
        for lb in self.labels:
            names = lb.on if isinstance(lb, SegmentLabel) else lb.angle
            for n in names:
                need(n, f"label {lb.text!r}")

        for items, key, what in (
            (self.segments, _seg_key, "segment"),
            (self.circles, _circle_key, "circle"),
            (self.marks, canonical_mark, "mark"),
            (self.labels, _label_target_key, "label target"),
        ):
            try:
                _reject_duplicates(items, key, what)
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise ValueError("; ".join(errors))
        return self


class Transcription(_Strict):
    """The complete model output: exactly the two keys ``text`` and ``diagram``."""

    text: StrictStr
    diagram: Diagram | None  # required key; null when there is no figure


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating one output.

    ``stage`` is ``"ok"``, ``"parse_error"`` (not JSON at all) or ``"invalid"``
    (JSON, but not a valid ``Transcription``). The eval's parse rate counts every stage
    except ``parse_error``; its validation rate counts only ``ok``.
    """

    stage: Literal["ok", "parse_error", "invalid"]
    value: Transcription | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.stage == "ok"


class _DuplicateKeyError(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKeyError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def loads_strict(text: str) -> Any:
    """``json.loads`` that rejects NaN/Infinity and duplicate object keys.

    Duplicate keys matter here: ``{"points": {"A": [1, 2], "A": [3, 4]}}`` silently
    keeps the last value under plain ``json.loads``.
    """
    return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)


def _format_errors(e: ValidationError) -> tuple[str, ...]:
    out = []
    for err in e.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return tuple(out)


def validate_output(obj: Any) -> ValidationReport:
    """Validate an already-parsed JSON value (dict) against the schema."""
    try:
        value = Transcription.model_validate(obj)
    except ValidationError as e:
        return ValidationReport("invalid", None, _format_errors(e))
    return ValidationReport("ok", value, ())


def validate_output_json(text: str) -> ValidationReport:
    """Validate a raw JSON string. Does *not* strip Markdown fences or extract JSON from
    surrounding prose; that leniency belongs to the inference/eval layer, which reports
    it separately."""
    try:
        obj = loads_strict(text)
    except _DuplicateKeyError as e:
        return ValidationReport("invalid", None, (str(e),))
    except (json.JSONDecodeError, ValueError) as e:
        return ValidationReport("parse_error", None, (f"not valid JSON: {e}",))
    return validate_output(obj)


def output_json_schema() -> dict[str, Any]:
    """JSON Schema for ``Transcription``; useful later for constrained decoding."""
    return Transcription.model_json_schema()


# ---------------------------------------------------------------------------
# Canonical comparison form
# ---------------------------------------------------------------------------

CanonSegment = frozenset[str]
CanonAngle = tuple[str, frozenset[str]]  # (vertex, {end, end})


def _seg_key(seg: tuple[str, str]) -> CanonSegment:
    return frozenset(seg)


def _angle_key(ang: tuple[str, str, str]) -> CanonAngle:
    return (ang[1], frozenset((ang[0], ang[2])))


def _circle_key(c: Circle) -> tuple[str, float]:
    return (c.center, float(c.radius))


def _label_target_key(lb: SegmentLabel | AngleLabel) -> tuple:
    if isinstance(lb, SegmentLabel):
        return ("on", _seg_key(lb.on))
    return ("angle", _angle_key(lb.angle))


def _mark_point_names(m: RightAngleMark | EqualLengthMark | ParallelMark | EqualAngleMark):
    if isinstance(m, RightAngleMark):
        return (m.at, *m.rays)
    if isinstance(m, EqualLengthMark | ParallelMark):
        return tuple(n for s in m.segments for n in s)
    return tuple(n for a in m.angles for n in a)


def canonical_segment(seg: tuple[str, str]) -> CanonSegment:
    return _seg_key(seg)


def canonical_angle(ang: tuple[str, str, str]) -> CanonAngle:
    return _angle_key(ang)


def canonical_mark(m: RightAngleMark | EqualLengthMark | ParallelMark | EqualAngleMark) -> tuple:
    """Hashable, order-free form of a mark.

    right_angle  -> ("right_angle", (vertex, {ray, ray}))
    equal_length -> ("equal_length", {{A,B}, {C,D}, ...})
    parallel     -> ("parallel", {{A,B}, {C,D}, ...})
    equal_angle  -> ("equal_angle", {(vertex, {end, end}), ...})
    """
    if isinstance(m, RightAngleMark):
        return ("right_angle", (m.at, frozenset(m.rays)))
    if isinstance(m, EqualLengthMark | ParallelMark):
        return (m.type, frozenset(_seg_key(s) for s in m.segments))
    return ("equal_angle", frozenset(_angle_key(a) for a in m.angles))


def canonical_label(lb: SegmentLabel | AngleLabel) -> tuple:
    """``("on", {A,B}, text)`` or ``("angle", (vertex, {end, end}), text)``.

    Text is only stripped. LaTeX normalization is an eval concern (eval/normalize.py).
    """
    return (*_label_target_key(lb), lb.text.strip())


@dataclass(frozen=True, eq=True)
class CanonicalDiagram:
    """Order-free view of a ``Diagram``. Compare fields with ``==`` or set operations."""

    points: dict[str, tuple[int, int]]
    segments: frozenset[CanonSegment]
    circles: frozenset[tuple[str, float]]
    marks: frozenset[tuple]
    labels: frozenset[tuple]

    __hash__ = None  # holds a dict; compare, don't hash


def canonicalize(d: Diagram) -> CanonicalDiagram:
    return CanonicalDiagram(
        points={k: (int(v[0]), int(v[1])) for k, v in d.points.items()},
        segments=frozenset(_seg_key(s) for s in d.segments),
        circles=frozenset(_circle_key(c) for c in d.circles),
        marks=frozenset(canonical_mark(m) for m in d.marks),
        labels=frozenset(canonical_label(lb) for lb in d.labels),
    )


# ---------------------------------------------------------------------------
# Deterministic serialization (training targets)
# ---------------------------------------------------------------------------

_MARK_ORDER = {"right_angle": 0, "equal_length": 1, "parallel": 2, "equal_angle": 3}


def name_sort_key(name: str) -> tuple:
    """Natural order: labeled points alphabetically (A, A', A'', A_1, B, ...), then
    unlabeled points numerically (_P2 before _P10)."""
    if is_unnamed(name):
        return (1, int(name[len(UNNAMED_PREFIX) :]), "")
    return (0, 0, name)


def _sorted_pair(a: str, b: str) -> list[str]:
    return sorted((a, b), key=name_sort_key)


def _seg_out(s: tuple[str, str]) -> list[str]:
    return _sorted_pair(*s)


def _seg_order(s: list[str]) -> tuple:
    return tuple(name_sort_key(n) for n in s)


def _angle_out(a: tuple[str, str, str]) -> list[str]:
    lo, hi = _sorted_pair(a[0], a[2])
    return [lo, a[1], hi]


def _angle_order(a: list[str]) -> tuple:
    # vertex first, then ends
    return (name_sort_key(a[1]), name_sort_key(a[0]), name_sort_key(a[2]))


def _mark_out(m) -> dict[str, Any]:
    if isinstance(m, RightAngleMark):
        return {"type": m.type, "at": m.at, "rays": _sorted_pair(*m.rays)}
    if isinstance(m, EqualLengthMark | ParallelMark):
        segs = sorted((_seg_out(s) for s in m.segments), key=_seg_order)
        return {"type": m.type, "segments": segs}
    angs = sorted((_angle_out(a) for a in m.angles), key=_angle_order)
    return {"type": m.type, "angles": angs}


def _mark_order(m: dict[str, Any]) -> tuple:
    t = m["type"]
    if t == "right_angle":
        body: tuple = (name_sort_key(m["at"]), *(name_sort_key(n) for n in m["rays"]))
    elif t == "equal_angle":
        body = tuple(_angle_order(a) for a in m["angles"])
    else:
        body = tuple(_seg_order(s) for s in m["segments"])
    return (_MARK_ORDER[t], body)


def _label_out(lb) -> dict[str, Any]:
    if isinstance(lb, SegmentLabel):
        return {"on": _seg_out(lb.on), "text": lb.text}
    return {"angle": _angle_out(lb.angle), "text": lb.text}


def _label_order(lb: dict[str, Any]) -> tuple:
    if "on" in lb:
        return (0, _seg_order(lb["on"]), lb["text"])
    return (1, _angle_order(lb["angle"]), lb["text"])


def diagram_to_obj(d: Diagram) -> dict[str, Any]:
    """Plain-JSON form with canonical ordering everywhere and all five keys present."""
    points = {k: [d.points[k][0], d.points[k][1]] for k in sorted(d.points, key=name_sort_key)}
    segments = sorted((_seg_out(s) for s in d.segments), key=_seg_order)
    circles = [
        {"center": c.center, "radius": c.radius}
        for c in sorted(d.circles, key=lambda c: (name_sort_key(c.center), float(c.radius)))
    ]
    marks = sorted((_mark_out(m) for m in d.marks), key=_mark_order)
    labels = sorted((_label_out(lb) for lb in d.labels), key=_label_order)
    return {
        "points": points,
        "segments": segments,
        "circles": circles,
        "marks": marks,
        "labels": labels,
    }


def output_to_obj(t: Transcription) -> dict[str, Any]:
    return {"text": t.text, "diagram": None if t.diagram is None else diagram_to_obj(t.diagram)}


def to_target_json(t: Transcription) -> str:
    """The exact training-target string. Compact separators keep generation short; key
    and list order are canonical, so equal inputs always give byte-identical strings."""
    return json.dumps(output_to_obj(t), ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Soft geometric consistency (for figures drawn to scale, i.e. synthetic data)
# ---------------------------------------------------------------------------


def _dist_point_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _angle_deg(p, v, q) -> float:
    a = math.atan2(p[1] - v[1], p[0] - v[0])
    b = math.atan2(q[1] - v[1], q[0] - v[0])
    d = abs(math.degrees(a - b)) % 360.0
    return 360.0 - d if d > 180.0 else d


def _direction_diff_deg(a, b, c, d) -> float:
    """Smallest angle between lines AB and CD, in [0, 90]."""
    t1 = math.atan2(b[1] - a[1], b[0] - a[0])
    t2 = math.atan2(d[1] - c[1], d[0] - c[0])
    diff = abs(math.degrees(t1 - t2)) % 180.0
    return min(diff, 180.0 - diff)


def consistency_issues(d: Diagram) -> list[str]:
    """Geometric sanity problems in a *valid* diagram. Empty list means consistent.

    Checks that every marked or labeled segment/ray is actually drawn (directly, or as
    a collinear piece of a drawn segment, e.g. AM where M lies on AB), that marks hold
    geometrically (right angles ~90 deg, equal lengths/angles ~equal, parallels ~parallel),
    and that circles fit in the grid. Assumes the figure is drawn to scale, so it applies
    to synthetic ground truth, not to arbitrary textbook figures.
    """
    tol = _SC.consistency_collinear_tol
    ang_tol = _SC.consistency_angle_tol_deg
    len_tol = _SC.consistency_length_tol
    P = {k: (float(v[0]), float(v[1])) for k, v in d.points.items()}
    drawn = [(a, b) for a, b in d.segments]
    drawn_set = {frozenset(s) for s in drawn}
    issues: list[str] = []

    def covered(p: str, q: str) -> bool:
        if frozenset((p, q)) in drawn_set:
            return True
        return any(
            _dist_point_segment(P[p], P[a], P[b]) <= tol
            and _dist_point_segment(P[q], P[a], P[b]) <= tol
            for a, b in drawn
        )

    def length(s) -> float:
        return math.dist(P[s[0]], P[s[1]])

    def angle(a) -> float:
        return _angle_deg(P[a[0]], P[a[1]], P[a[2]])

    for m in d.marks:
        if isinstance(m, RightAngleMark):
            for r in m.rays:
                if not covered(m.at, r):
                    issues.append(f"right_angle at {m.at}: ray {m.at}{r} is not drawn")
            got = _angle_deg(P[m.rays[0]], P[m.at], P[m.rays[1]])
            if abs(got - 90.0) > ang_tol:
                issues.append(f"right_angle at {m.at}: angle is {got:.1f} deg, not ~90")
        elif isinstance(m, EqualLengthMark | ParallelMark):
            for s in m.segments:
                if not covered(*s):
                    issues.append(f"{m.type}: segment {''.join(s)} is not drawn")
            if isinstance(m, EqualLengthMark):
                lens = [length(s) for s in m.segments]
                if max(lens) - min(lens) > len_tol:
                    issues.append(f"equal_length: lengths {[round(x, 1) for x in lens]} differ")
            else:
                ref = m.segments[0]
                for s in m.segments[1:]:
                    dd = _direction_diff_deg(P[ref[0]], P[ref[1]], P[s[0]], P[s[1]])
                    if dd > ang_tol:
                        issues.append(
                            f"parallel: {''.join(ref)} and {''.join(s)} differ by {dd:.1f} deg"
                        )
        else:  # EqualAngleMark
            for a in m.angles:
                for end in (a[0], a[2]):
                    if not covered(a[1], end):
                        issues.append(f"equal_angle: ray {a[1]}{end} is not drawn")
            vals = [angle(a) for a in m.angles]
            if max(vals) - min(vals) > ang_tol:
                issues.append(f"equal_angle: angles {[round(x, 1) for x in vals]} differ")

    for lb in d.labels:
        if isinstance(lb, SegmentLabel):
            if not covered(*lb.on):
                issues.append(f"label {lb.text!r}: segment {''.join(lb.on)} is not drawn")
        else:
            a = lb.angle
            for end in (a[0], a[2]):
                if not covered(a[1], end):
                    issues.append(f"label {lb.text!r}: ray {a[1]}{end} is not drawn")

    for c in d.circles:
        cx, cy = P[c.center]
        r = float(c.radius)
        if (
            cx - r < GRID_MIN - tol
            or cx + r > GRID_MAX + tol
            or cy - r < GRID_MIN - tol
            or (cy + r > GRID_MAX + tol)
        ):
            issues.append(f"circle at {c.center} (r={r}) extends outside the grid")

    return issues
