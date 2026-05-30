"""Writing an eval run: per-example JSONL, summary JSON and text, side-by-side images.

A side-by-side image puts the input page (what the model saw) next to the prediction
re-rendered with the same renderer that made the training data. A schema-invalid
prediction cannot be rendered, so its panel shows the validation errors and the raw
output instead, with control characters made visible (an unescaped ``\\frac`` shows up
as ``\\x0c`` followed by ``rac``).
"""

from __future__ import annotations

import json
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mathvlm.eval.metrics import CATEGORIES, MARK_TYPES, ExampleResult

PANEL_MAX_HEIGHT = 1600  # px; a rambling prediction would otherwise make a huge image
_PAD = 10
_FONT_SIZE = 13


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_jsonl(results: list[ExampleResult], path: Path) -> None:
    with open(path, "w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_row(), ensure_ascii=False) + "\n")


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Printable summary
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _counts(d: dict[str, int], limit: int = 6) -> str:
    if not d:
        return "none"
    items = list(d.items())
    s = ", ".join(f"{k} {v}" for k, v in items[:limit])
    return s + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def format_summary(s: dict[str, Any], by_family: dict[str, dict] | None = None, title="") -> str:
    p, det, txt, st, geo = s["parse"], s["detection"], s["text"], s["structure"], s["geometry"]
    c = det["confusion"]
    L = []
    if title:
        L.append(title)
    L.append(f"{s['n']} samples: {s['n_gt_diagram']} with a diagram, {s['n_gt_null']} without")
    L += [
        "",
        "parse",
        f"  json parse rate          {_pct(p['json_parse_rate']):>7}   "
        f"(direct {p['tiers']['direct']}, fenced {p['tiers']['fenced']}, "
        f"failed {p['tiers']['failed']})",
        f"  json parse rate, strict  {_pct(p['json_parse_rate_strict']):>7}   (direct only)",
        f"  schema valid rate        {_pct(p['schema_valid_rate']):>7}",
        f"  parse failures           {_counts(p['failures'])}",
        f"  schema errors (first)    {_counts(p['schema_errors'])}",
        f"  unescaped-LaTeX suspects {p['escape_suspect']}",
        "",
        f"diagram detection          accuracy {_pct(det['accuracy'])}",
        "                           pred diagram   pred null   failed",
        f"  gt diagram               {c['diagram']['diagram']:>12} {c['diagram']['null']:>11}"
        f" {c['diagram']['failed']:>8}",
        f"  gt null                  {c['null']['diagram']:>12} {c['null']['null']:>11}"
        f" {c['null']['failed']:>8}",
        "",
        "text                       normalized      raw",
        f"  edit distance (NED)      {_num(txt['ned']):>10} {_num(txt['ned_raw']):>8}"
        "   (no usable text counts 1.0)",
        f"  exact match              {_pct(txt['exact']):>10}",
        f"  NED where text exists    {_num(txt['ned_with_text']):>10}"
        f"            (n={txt['n_with_text']})",
        "",
        f"structure F1 (micro, {st['n']} samples)   strict  geometric      P / R (geometric)",
    ]
    for cat in (*CATEGORIES, "all"):
        a, b = st["strict"][cat], st["geometric"][cat]
        L.append(
            f"  {cat:<24} {_num(a['f1']):>10} {_num(b['f1']):>10}"
            f"   {_num(b['precision'], 2)} / {_num(b['recall'], 2)}"
        )
    types = ", ".join(f"{t} {_num(st['geometric'][f'marks.{t}']['f1'], 2)}" for t in MARK_TYPES)
    L.append(f"  marks by type (geometric) {types}")
    sb = s["structure_both_valid"]
    L.append(
        f"  where both diagrams are valid (n={sb['n']}): all {_num(sb['strict']['all']['f1'])}"
        f" strict, {_num(sb['geometric']['all']['f1'])} geometric"
    )
    L += [
        "",
        f"geometry ({geo['n']} samples with both diagrams valid; grid units)",
        "                           mean  median",
        f"  point error, by position {_num(geo['error_mean'], 2):>6} {_num(geo['error_median'], 2):>7}"
        f"   (aligned; unaligned mean {_num(geo['raw_error_mean'], 2)})",
        f"  point error, by name     {_num(geo['strict_error_mean'], 2):>6}"
        f" {_num(geo['strict_error_median'], 2):>7}",
        f"  coverage                 {_pct(geo['coverage'])} of GT points matched",
        f"  fitted scale (median)    {_num(geo['scale_median'], 3)}",
        f"  radius error (mean)      {_num(geo['radius_error_mean'], 2)}",
        "  text render score        not implemented (stub)",
    ]
    if by_family:
        L += ["", "by family                      n   valid    NED   F1 strict   F1 geo   pt err"]
        for fam, f in by_family.items():
            L.append(
                f"  {fam:<26} {f['n']:>4} {_pct(f['schema_valid_rate']):>7} "
                f"{_num(f['text_ned']):>6} {_num(f['f1_strict']):>11} "
                f"{_num(f['f1_geometric']):>8} {_num(f['point_error_mean'], 2):>8}"
            )
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Side-by-side images
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _font() -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono"))
    return ImageFont.truetype(path, _FONT_SIZE)


def _visible(s: str) -> str:
    """Control characters (other than newline) as \\xNN, so escape bugs show."""
    return "".join(ch if ch == "\n" or ord(ch) >= 32 else f"\\x{ord(ch):02x}" for ch in s)


def _text_card(lines: list[str], width: int, max_height: int = PANEL_MAX_HEIGHT) -> Image.Image:
    font = _font()
    cols = max(20, int((width - 2 * _PAD) // font.getlength("M")))
    wrapped: list[str] = []
    for line in lines:
        wrapped += textwrap.wrap(line, cols, replace_whitespace=False, drop_whitespace=False) or [
            ""
        ]
    line_h = _FONT_SIZE + 4
    max_lines = (max_height - 2 * _PAD) // line_h
    if len(wrapped) > max_lines:
        wrapped = [*wrapped[: max_lines - 1], "... (cut)"]
    im = Image.new("RGB", (width, 2 * _PAD + line_h * len(wrapped)), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for k, line in enumerate(wrapped):
        draw.text((_PAD, _PAD + k * line_h), line, fill=(0, 0, 0), font=font)
    return im


def _titled(im: Image.Image, title: str) -> Image.Image:
    strip = _text_card([title], im.width)
    out = Image.new("RGB", (im.width, strip.height + im.height), (255, 255, 255))
    out.paste(strip, (0, 0))
    out.paste(im, (0, strip.height))
    return out


def _prediction_panel(r: ExampleResult, width: int) -> Image.Image:
    from mathvlm.render import render_page

    t = r.parsed.transcription
    if t is not None:
        try:
            im = render_page(t).image
            if im.height > PANEL_MAX_HEIGHT:
                im = im.crop((0, 0, im.width, PANEL_MAX_HEIGHT))
                return _titled(im, "prediction, re-rendered (cut)")
            return _titled(im, "prediction, re-rendered")
        except Exception as e:  # the renderer should not raise; show it if it does
            return _text_card([f"render failed: {type(e).__name__}: {e}"], width)
    p = r.parsed
    head = f"prediction: {p.failure}" if p.failure else f"prediction: schema {p.report.stage}"
    lines = [head, ""]
    lines += [f"  {e}" for e in p.report.errors[:6]]
    lines += ["", "raw output:", *_visible(p.raw or "").split("\n")]
    return _text_card(lines, width)


def _header(r: ExampleResult) -> list[str]:
    fam = "/".join(str(r.meta[k]) for k in ("family", "kind") if k in r.meta)
    p = r.parsed
    parse = p.how or f"failed ({p.failure})"
    parts = [f"parse {parse}", f"schema {p.report.stage}", f"NED {r.text['ned']:.3f}"]
    if r.structure is not None:
        parts.append(
            "F1 strict/geo "
            f"{_num(r.structure['strict']['segments'].f1, 2)}/"
            f"{_num(r.structure['geometric']['segments'].f1, 2)} (segments)"
        )
    if r.geometry is not None:
        g = r.geometry.to_dict()
        parts.append(
            f"pt err {_num(g['error_mean'], 1)}, by name {_num(g['strict_error_mean'], 1)}"
        )
    return [f"{r.id}  {fam}  gt {r.gt_kind} -> pred {r.pred_kind}", "  ".join(parts)]


def side_by_side(r: ExampleResult, input_image: Image.Image) -> Image.Image:
    from mathvlm.render import contact_sheet

    left = _titled(input_image.convert("RGB"), "input")
    right = _prediction_panel(r, left.width)
    body = contact_sheet([left, right], cols=2, pad=_PAD, bg=(220, 220, 220))
    head = _text_card(_header(r), body.width)
    out = Image.new("RGB", (body.width, head.height + body.height), (255, 255, 255))
    out.paste(head, (0, 0))
    out.paste(body, (0, head.height))
    return out


def write_side_by_sides(
    results: list[ExampleResult], images: dict[str, Path], out_dir: Path, limit: int
) -> list[Path]:
    """The first ``limit`` samples in dataset order, so runs compare image for image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in results[:limit]:
        src = images.get(r.id)
        if src is None or not src.exists():
            continue
        path = out_dir / f"{r.id}.png"
        side_by_side(r, Image.open(src)).save(path)
        paths.append(path)
    return paths
