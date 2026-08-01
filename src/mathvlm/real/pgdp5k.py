r"""PGDP5K (real textbook geometry diagrams) -> our schema, for evaluation.

    uv run python -m mathvlm.real.pgdp5k --raw ~/vlm_data/raw/pgdp5k/extracted/PGDP5K

Writes ``<data_dir>/real/pgdp5k/{train,val,test}/`` in the same layout as a synthetic
split (``<id>.png``, ``<id>.json``, ``manifest.jsonl``), so ``infer`` and ``run_eval`` take
``--split pgdp5k/val`` unchanged, plus a QA sheet (original | re-rendered ground truth).

Source: https://huggingface.co/datasets/PeijieWang/PGDP5K (MIT on the card; the images
are screenshots of commercial textbooks: keep them out of the repo). Each diagram has
pixel geometry and symbols (``annotations/<split>.json``) and a logic-form description
(``our_diagram_logic_forms_annot.json``). Mapping:

- points: ``point_positions``. A point whose name is not drawn as a label in the image
  (text symbol of class "point") becomes ``_P1, _P2, ...`` in reading order.
- segments: the drawn lines (``geos.lines``), endpoint to endpoint, i.e. maximal drawn
  segments, our generator's convention.
- circles: ``geos.circles``; the center is the point at that position, or a new
  unnamed point. Partially drawn circles become full circles (recorded as a loss).
- ``Perpendicular(L1, L2)`` -> right_angle at their common point, with the two rays
  chosen so the square sits in the corner where the image draws it.
- ``Parallel`` / ``Equals(LengthOf, LengthOf)`` / ``Equals(MeasureOf(Angle), MeasureOf(Angle))``
  -> parallel / equal_length / equal_angle marks, grouped transitively.
- ``Equals(LengthOf(Line), v)`` -> length label ``v``; ``Equals(MeasureOf(Angle), v)`` ->
  angle label ``v^\circ`` (parenthesized if ``v`` is an expression, ``(4y-10)^\circ``,
  as the images show); ``MeasureOf(angle 4)`` -> angle label ``4``;
  ``Equals(MeasureOf(Arc), v)`` -> arc label ``v^\circ`` (``LengthOf(Arc)``: no degree).
- text symbols of class "line" (``m``, ``\ell``) -> a label on the line's segment;
  "area" and "others" -> free text at the symbol's position.
- ``PointLiesOnLine`` / ``PointLiesOnCircle`` are implied by the coordinates.

Arrows: PGDP5K's ``arrow``/``head``/``head_len`` symbols are callout arrows and dimension
lines, which only say where a label belongs (the logic forms already attach it). Ray
arrowheads are not annotated at all, so the ground truth has no arrow marks, and every
row says ``"unannotated": ["arrow"]``: the eval ignores arrow marks on both sides
instead of penalizing a model that draws arrowheads the dataset never recorded.

Not representable (recorded per diagram in the manifest's ``dropped``, and
``representable`` is false): equal-arc marks, circle perimeters, partial circles, and
anything the mapping above could not place. The ground-truth text is ``""``: these
images contain no question text.

Coordinates are fitted to the grid exactly as the generator does (bounding box of points
and circles, centered, larger side spanning ``[margin, 100 - margin]``, y down).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mathvlm.config import get_config
from mathvlm.schema import GRID_MAX, GRID_MIN, consistency_issues, output_to_obj, validate_output

SPLITS = ("train", "val", "test")
_NAME_RE = re.compile(r"[A-Z](?:'{1,2}|_[0-9]{1,2})?")
_LINE = r"Line\(([^,()]+), ([^,()]+)\)"
_ANGLE = r"Angle\(([^,()]+), ([^,()]+), ([^,()]+)\)"
_ARC = r"Arc\(([^,()]+), ([^,()]+)\)"

Vec = tuple[float, float]


@dataclass
class Converted:
    name: str
    status: str  # "ok" | "skipped" | "invalid"
    obj: dict | None = None  # the schema-valid {"text", "diagram"} object
    dropped: list[str] = field(default_factory=list)
    reason: str = ""
    n_unnamed: int = 0
    n_issues: int = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _split_args(s: str) -> list[str]:
    """Split ``a, f(b, c), d`` at top-level commas."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def _term(s: str) -> tuple[str, Any]:
    """One side of an ``Equals``: a measured object, or a value."""
    for pat, kind in (
        (rf"^LengthOf\({_LINE}\)$", "length"),
        (rf"^MeasureOf\({_ANGLE}\)$", "angle"),
        (rf"^MeasureOf\({_ARC}\)$", "arc_measure"),
        (rf"^LengthOf\({_ARC}\)$", "arc_length"),
    ):
        m = re.match(pat, s)
        if m:
            return kind, tuple(g.strip() for g in m.groups())
    m = re.match(r"^MeasureOf\(angle ([^()]+)\)$", s)
    if m:
        return "angle_name", m.group(1).strip()
    if re.match(r"^[A-Za-z]+Of\(", s):
        return "other", s
    return "value", s.strip()


def _degrees(v: str) -> str:
    """``48`` -> ``48^\\circ``; ``4y-10`` -> ``(4y-10)^\\circ``, as textbooks print it."""
    compound = re.search(r"[+\-]", v.lstrip("-")) is not None
    return f"({v})^\\circ" if compound else f"{v}^\\circ"


class _Groups:
    """Union-find over hashable items, for transitive equal-length/angle/parallel groups."""

    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list]:
        out: dict = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return [sorted(g) for g in out.values() if len(g) >= 2]


def _dist_to_line(p: Vec, a: Vec, b: Vec) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return math.dist(p, a)
    return abs(dy * (p[0] - a[0]) - dx * (p[1] - a[1])) / n


def _unit(v: Vec) -> Vec | None:
    n = math.hypot(*v)
    return None if n < 1e-9 else (v[0] / n, v[1] / n)


# ---------------------------------------------------------------------------
# One diagram
# ---------------------------------------------------------------------------


def convert_one(name: str, lf: dict, ann: dict, margin: float = 8.0) -> Converted:
    """Convert one diagram. Pure: the logic-form entry and the annotation entry in, a
    schema-valid object (or the reason there is none) out."""
    res = Converted(name, "ok")
    drop = res.dropped
    pos: dict[str, Vec] = {k: (float(v[0]), float(v[1])) for k, v in lf["point_positions"].items()}
    symbols = ann.get("symbols", [])
    shown = {s["text_content"] for s in symbols
             if s["sym_class"] == "text" and s["text_class"] == "point"}  # fmt: skip
    diag = math.hypot(ann.get("width", 1), ann.get("height", 1))
    tol = max(3.0, 0.02 * diag)  # pixels: "the same place"

    def nearest(p: Vec, limit: float = tol) -> str | None:
        best = min(pos, key=lambda k: math.dist(pos[k], p), default=None)
        return best if best is not None and math.dist(pos[best], p) <= limit else None

    # --- circles (may add unnamed centers) ---
    circles: list[tuple[str, float]] = []
    for c in ann["geos"].get("circles", []):
        center, r, quadrants = (float(c["loc"][0][0]), float(c["loc"][0][1])), float(c["loc"][1]), str(c["loc"][2])  # fmt: skip
        cname = nearest(center)
        if cname is None:
            cname = f"__center{len(circles)}"
            pos[cname] = center
        circles.append((cname, r))
        if quadrants != "1111":
            drop.append(f"partial circle ({quadrants}) drawn as a full circle")

    # --- names: visible labels keep theirs (B_{3} spelled B_3), the rest become _P1.. in
    # reading order ---
    def clean(n: str) -> str:
        return re.sub(r"_\{([0-9]{1,2})\}", r"_\1", n)

    bad = [n for n in pos if n in shown and not _NAME_RE.fullmatch(clean(n))]
    if bad:
        return Converted(name, "skipped", reason=f"point name(s) not in the schema: {bad}")
    hidden = sorted((n for n in pos if n not in shown), key=lambda n: (pos[n][1], pos[n][0], n))
    rename = {n: clean(n) for n in pos if n in shown}
    rename.update({n: f"_P{i + 1}" for i, n in enumerate(hidden)})
    res.n_unnamed = len(hidden)

    def N(n: str) -> str:
        return rename[n]

    # --- segments: drawn lines, endpoint to endpoint ---
    segments: list[tuple[str, str]] = []
    line_ends: dict[str, tuple[str, str]] = {}
    for ln in ann["geos"].get("lines", []):
        a = nearest((float(ln["loc"][0][0]), float(ln["loc"][0][1])))
        b = nearest((float(ln["loc"][1][0]), float(ln["loc"][1][1])))
        if a is None or b is None or a == b:
            drop.append(f"line {ln['id']}: endpoint not at a point")
            continue
        line_ends[ln["id"]] = (a, b)
        if {a, b} not in [set(s) for s in segments]:
            segments.append((a, b))

    def on_line(p: str, a: str, b: str) -> bool:
        return _dist_to_line(pos[p], pos[a], pos[b]) <= tol

    # --- logic forms ---
    eq_len, eq_ang, par = _Groups(), _Groups(), _Groups()
    labels: list[dict] = []
    marks: list[dict] = []
    perp_syms = [(s["bbox"][0] + s["bbox"][2] / 2, s["bbox"][1] + s["bbox"][3] / 2)
                 for s in symbols if s["sym_class"] == "perpendicular"]  # fmt: skip

    def known(*ns: str) -> bool:
        return all(n in pos for n in ns)

    for f in lf["diagram_logic_forms"]:
        if f.startswith(("PointLiesOnLine(", "PointLiesOnCircle(")):
            continue
        m = re.match(rf"^(Perpendicular|Parallel)\({_LINE}, {_LINE}\)$", f)
        if m:
            kind, a, b, c, d = m.group(1), *(g.strip() for g in m.groups()[1:])
            if not known(a, b, c, d):
                drop.append(f"unknown point in {f}")
            elif kind == "Parallel":
                par.union((a, b), (c, d))
            else:
                rm = _right_angle(a, b, c, d, pos, perp_syms, on_line)
                if rm is None:
                    drop.append(f"no common point: {f}")
                else:
                    marks.append(rm)
            continue
        m = re.match(r"^Equals\((.*)\)$", f)
        args = _split_args(m.group(1)) if m else []
        if len(args) != 2:
            drop.append(f"unparsed: {f}")
            continue
        (k1, v1), (k2, v2) = _term(args[0]), _term(args[1])
        if k2 != "value" and k1 == "value":
            (k1, v1), (k2, v2) = (k2, v2), (k1, v1)
        pts = [x for v in (v1, v2) if isinstance(v, tuple) for x in v]
        if not known(*pts):
            drop.append(f"unknown point in {f}")
        elif (k1, k2) == ("length", "length"):
            eq_len.union(v1, v2)
        elif (k1, k2) == ("angle", "angle"):
            eq_ang.union(v1, v2)
        elif k1 == "length" and k2 == "value":
            labels.append({"on": list(v1), "text": v2})
        elif k1 == "angle" and k2 == "value":
            labels.append({"angle": list(v1), "text": _degrees(v2)})
        elif {k1, k2} == {"angle", "angle_name"}:
            ang, nm = (v1, v2) if k1 == "angle" else (v2, v1)
            labels.append({"angle": list(ang), "text": nm})
        elif k1 == "arc_measure" and k2 == "value":
            labels.append({"arc": list(v1), "text": _degrees(v2)})
        elif k1 == "arc_length" and k2 == "value":
            labels.append({"arc": list(v1), "text": v2})
        else:
            drop.append(f"not representable: {f}")

    for g in eq_len.groups():
        marks.append({"type": "equal_length", "segments": [list(s) for s in g]})
    for g in eq_ang.groups():
        marks.append({"type": "equal_angle", "angles": [list(a) for a in g]})
    for g in par.groups():
        marks.append({"type": "parallel", "segments": [list(s) for s in g]})

    # --- text symbols the logic forms do not carry: line names, area and other text ---
    rel = {sid: targets for sid, targets in ann.get("relations", {}).get("sym2geo", [])}
    free: list[tuple[Vec, str]] = []
    for s in symbols:
        if s["sym_class"] != "text" or s["text_class"] not in ("line", "area", "others"):
            continue
        text = str(s["text_content"] or "").strip()
        if not text:
            continue
        center = (s["bbox"][0] + s["bbox"][2] / 2, s["bbox"][1] + s["bbox"][3] / 2)
        if s["text_class"] == "line":
            lid = next((t for t in rel.get(s["id"], []) if t in line_ends), None)
            if lid is None:
                drop.append(f"line name {text!r}: no line")
            else:
                labels.append({"on": list(line_ends[lid]), "text": text})
        else:
            free.append((center, text))

    # --- one label per target ---
    seen, kept = set(), []
    for lb in labels:
        key = next(k for k in ("on", "angle", "arc") if k in lb)
        target = (key, lb[key][1], frozenset((lb[key][0], lb[key][2]))) if key == "angle" \
            else (key, frozenset(lb[key]))  # fmt: skip
        if target in seen:
            drop.append(f"second label on one target: {lb['text']!r}")
            continue
        seen.add(target)
        kept.append(lb)

    # --- fit to the grid, as the generator does ---
    xs = [p[0] for p in pos.values()] + [pos[c][0] + s * r for c, r in circles for s in (-1, 1)]
    ys = [p[1] for p in pos.values()] + [pos[c][1] + s * r for c, r in circles for s in (-1, 1)]
    xs += [c[0] for c, _ in free]
    ys += [c[1] for c, _ in free]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = (GRID_MAX - GRID_MIN - 2 * margin) / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    mid = (GRID_MIN + GRID_MAX) / 2

    def grid(p: Vec) -> list[int]:
        return [round((p[0] - cx) * scale + mid), round((p[1] - cy) * scale + mid)]

    def ren(x):
        return [N(n) for n in x] if isinstance(x, list) else N(x)

    out_labels = []
    for lb in kept:
        key = next(k for k in ("on", "angle", "arc") if k in lb)
        out_labels.append({key: ren(lb[key]), "text": lb["text"]})
    taken = set()
    for c, text in free:
        at = grid(c)
        if tuple(at) not in taken:
            taken.add(tuple(at))
            out_labels.append({"at": at, "text": text})
    out_marks = []
    for mk in marks:
        mk = dict(mk)
        if mk["type"] == "right_angle":
            mk.update(at=N(mk["at"]), rays=ren(mk["rays"]))
        elif mk["type"] == "arrow":
            mk.update({"at": N(mk["at"]), "from": N(mk["from"])})
        elif mk["type"] == "equal_angle":
            mk["angles"] = [ren(a) for a in mk["angles"]]
        else:
            mk["segments"] = [ren(s) for s in mk["segments"]]
        out_marks.append(mk)

    obj = {
        "text": "",
        "diagram": {
            "points": {N(k): grid(p) for k, p in pos.items()},
            "segments": [[N(a), N(b)] for a, b in segments],
            "circles": [{"center": N(c), "radius": round(r * scale, 1)} for c, r in circles],
            "marks": out_marks,
            "labels": out_labels,
        },
    }
    report = validate_output(obj)
    if not report.ok:
        return Converted(name, "invalid", dropped=drop, reason="; ".join(report.errors[:3]),
                         n_unnamed=res.n_unnamed)  # fmt: skip
    res.obj = output_to_obj(report.value)
    res.n_issues = len(consistency_issues(report.value.diagram))
    return res


def _right_angle(a, b, c, d, pos, perp_syms, on_line) -> dict | None:
    """right_angle mark for Perpendicular(Line(a, b), Line(c, d)): the vertex is the common
    point; each ray goes to a point on its line, on the side where the image draws the
    square (nearest perpendicular symbol)."""
    common = {a, b} & {c, d}
    if common:
        v = common.pop()
    else:
        v = next((p for p in pos if on_line(p, a, b) and on_line(p, c, d)), None)
        if v is None:
            return None
    r1s = [x for x in (a, b) if x != v]
    r2s = [x for x in (c, d) if x != v]
    if not r1s or not r2s:
        return None
    sym = min(perp_syms, key=lambda s: math.dist(s, pos[v]), default=None)
    best, best_score = (r1s[0], r2s[0]), -math.inf
    for r1 in r1s:
        for r2 in r2s:
            u1, u2 = _unit((pos[r1][0] - pos[v][0], pos[r1][1] - pos[v][1])), _unit(
                (pos[r2][0] - pos[v][0], pos[r2][1] - pos[v][1]))  # fmt: skip
            if u1 is None or u2 is None or r1 == r2:
                continue
            score = 0.0
            if sym is not None:
                to_sym = _unit((sym[0] - pos[v][0], sym[1] - pos[v][1]))
                bis = _unit((u1[0] + u2[0], u1[1] + u2[1]))
                if to_sym and bis:
                    score = bis[0] * to_sym[0] + bis[1] * to_sym[1]
            if score > best_score:
                best, best_score = (r1, r2), score
    return {"type": "right_angle", "at": v, "rays": list(best)}


# ---------------------------------------------------------------------------
# Whole dataset
# ---------------------------------------------------------------------------


def convert_split(raw: Path, split: str, out: Path, lfs: dict, margin: float) -> list[Converted]:
    anns = json.loads((raw / "annotations" / f"{split}.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for index, key in enumerate(sorted(anns, key=lambda k: (len(k), k))):
        ann = anns[key]
        stem = Path(ann["file_name"]).stem
        r = convert_one(stem, lfs[key], ann, margin)
        results.append(r)
        if r.status != "ok":
            continue
        sid = f"pgdp5k_{split}_{stem}"
        shutil.copyfile(raw / split / ann["file_name"], out / f"{sid}.png")
        (out / f"{sid}.json").write_text(json.dumps(r.obj, indent=2, ensure_ascii=False) + "\n")
        rows.append({
            "id": sid, "index": index, "split": f"pgdp5k/{split}", "family": "pgdp5k",
            "kind": "circle" if r.obj["diagram"]["circles"] else "lines",
            "image": f"{sid}.png", "gt": f"{sid}.json", "source": f"{split}/{ann['file_name']}",
            "width": ann["width"], "height": ann["height"], "dropped": r.dropped,
            "representable": not r.dropped, "n_unnamed": r.n_unnamed, "unannotated": ["arrow"],
            "consistency_issues": r.n_issues,
        })  # fmt: skip
    with open(out / "manifest.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return results


def qa_sheet(split_out: Path, raw: Path, n: int, path: Path) -> Path:
    """Original image | ground truth re-rendered with our renderer, first ``n`` diagrams."""
    from PIL import Image

    from mathvlm.render import contact_sheet, render_diagram
    from mathvlm.schema import validate_output_json

    rows = [json.loads(line) for line in (split_out / "manifest.jsonl").read_text().splitlines()]
    tiles = []
    for row in rows[:n]:
        gt = validate_output_json((split_out / row["gt"]).read_text()).value
        ren = render_diagram(gt.diagram).resize((320, 320), Image.Resampling.LANCZOS)
        orig = Image.open(split_out / row["image"]).convert("RGB")
        orig.thumbnail((320, 320), Image.Resampling.LANCZOS)
        pair = Image.new("RGB", (650, 320), (255, 255, 255))
        pair.paste(orig, (0, 0))
        pair.paste(ren, (330, 0))
        tiles.append(pair)
    path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(tiles, cols=2, pad=10).save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.real.pgdp5k", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--raw", type=Path, required=True, help="the extracted PGDP5K/ directory")
    ap.add_argument("--out", type=Path, default=None, help="default: <data_dir>/real/pgdp5k")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--sheet", type=int, default=12, help="QA pairs per split (0: none)")
    args = ap.parse_args(argv)
    raw = args.raw.expanduser()
    out = args.out or cfg.paths.resolved("data_dir") / "real" / "pgdp5k"
    lfs = json.loads((raw / "our_diagram_logic_forms_annot.json").read_text())
    for split in args.splits:
        results = convert_split(raw, split, out / split, lfs, cfg.generate.margin)
        status = Counter(r.status for r in results)
        ok = [r for r in results if r.status == "ok"]
        drops = Counter(re.sub(r"\(.*", "", d).split(":")[0].strip() for r in ok for d in r.dropped)
        print(f"{split}: {len(results)} diagrams -> {dict(status)}; "
              f"representable {sum(not r.dropped for r in ok)}/{len(ok)}; "
              f"with consistency issues {sum(r.n_issues > 0 for r in ok)}; "
              f"unnamed points/diagram {sum(r.n_unnamed for r in ok) / max(1, len(ok)):.1f}")  # fmt: skip
        print(f"  dropped: {dict(drops.most_common(8))}")
        for r in [r for r in results if r.status != "ok"][:5]:
            print(f"  {r.status} {r.name}: {r.reason[:150]}")
        if args.sheet:
            sheet = cfg.paths.resolved("outputs_dir") / "real" / f"pgdp5k_{split}_qa.png"
            print(f"  QA sheet: {qa_sheet(out / split, raw, args.sheet, sheet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
