"""Renderer: diagram JSON -> image, plus the question-above-diagram page layout.

Used both to create training images and to visualize predictions, so it must draw
*any* schema-valid diagram without raising, including implausible predictions
(coincident points, a "right angle" that is not 90 degrees, malformed LaTeX in a label).

Coordinate contract
-------------------
The diagram panel is a square axes whose data limits are the 0-100 grid plus
``render.diagram_pad`` on every side, with y pointing down. So a grid point maps to a
fixed pixel position inside the panel (``grid_to_pixel``), independent of the figure's
content. Training images keep this mapping exact; augmentation perturbs it afterwards.

Drawing conventions
-------------------
- right_angle: a small square in the corner between the two rays.
- equal_length: perpendicular ticks at the segment's midpoint. The k-th equal_length
  mark in the list gets k ticks (capped at 4), so different groups are distinguishable.
- parallel: chevron arrows at the midpoint, all pointing the same way within a group;
  the k-th parallel mark gets k chevrons.
- equal_angle: an arc crossed by k short ticks for the k-th equal_angle mark (capped
  at 4). Ticks rather than concentric arcs, because the two halves of a bisected
  angle share a ray and their arcs would otherwise merge into one.
- angle labels: text inside the angle on its bisector, over a single arc when
  ``style.angle_arcs`` is set (and the angle has no equal_angle arcs already, and its
  vertex carries fewer than three angle labels).
- point labels: placed in the widest free direction around the point, biased away
  from the figure's centroid. Unlabeled ``_P`` points get no label.
- arrow: a filled arrowhead with its tip at the point, pointing away from ``from``;
  the point gets no dot.
- arc labels: outside the circle, on the radius through the minor arc's midpoint.
- free text: centered on its grid position.

Fonts and LaTeX
---------------
Mathtext only (no TeX install), matplotlib-bundled fonts only. Every string that fails
to parse as mathtext is drawn verbatim instead of raising, so a prediction with broken
LaTeX still renders and the breakage stays visible.

Determinism: all randomness lives in ``sample_style(rng)``. Rendering itself is a pure
function of (input, style).
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import cbook
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import Circle as CirclePatch
from PIL import Image

from mathvlm.config import get_config, sample_rng
from mathvlm.schema import (
    GRID_MAX,
    GRID_MIN,
    AngleLabel,
    ArcLabel,
    ArrowMark,
    Diagram,
    EqualAngleMark,
    EqualLengthMark,
    FreeTextLabel,
    ParallelMark,
    RightAngleMark,
    SegmentLabel,
    Transcription,
    canonical_angle,
    canonical_segment,
    is_unnamed,
)

_RC = get_config().render
if _RC.usetex:
    raise NotImplementedError("render.usetex=true is not supported in Phase 1 (mathtext only)")

PAD: float = _RC.diagram_pad
SPAN: float = (GRID_MAX - GRID_MIN) + 2 * PAD
LO: float = GRID_MIN - PAD
HI: float = GRID_MAX + PAD


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderStyle:
    fontset: str  # mathtext fontset
    text_family: str  # plain-text font paired with the fontset
    ink: str
    line_width: float  # points
    point_dots: bool  # dot every point (circle centers always get one)
    dot_size: float  # marker size, points
    label_fontsize: float
    label_italic: bool  # $A$ (italic) vs \mathrm{A} (upright)
    question_fontsize: float
    mark_size: float  # grid units
    angle_arcs: bool  # arc under angle labels
    background: str
    line_ink: str  # segments, circles, arrowheads, point dots
    mark_ink: str  # right-angle squares, ticks, chevrons, angle arcs
    label_ink: str  # value labels (lengths, angles, arcs, free text)


def default_style() -> RenderStyle:
    """The un-randomized style used for fixtures and for visualizing predictions."""
    return RenderStyle(
        fontset=_RC.mathtext_fontset,
        text_family=_RC.text_family[_RC.mathtext_fontset],
        ink=_RC.ink,
        line_width=_RC.line_width,
        point_dots=True,
        dot_size=_RC.point_marker_size,
        label_fontsize=_RC.label_fontsize,
        label_italic=True,
        question_fontsize=_RC.question_fontsize,
        mark_size=_RC.mark_size,
        angle_arcs=True,
        background=_RC.background,
        line_ink=_RC.ink,
        mark_ink=_RC.ink,
        label_ink=_RC.ink,
    )


def sample_style(rng: np.random.Generator) -> RenderStyle:
    """Draw a random style. The draw order is fixed, so equal RNG states give equal
    styles; this is the only source of randomness in the renderer."""
    fontset = _RC.fontsets[int(rng.integers(len(_RC.fontsets)))]
    ink = _RC.inks[int(rng.integers(len(_RC.inks)))]
    style = dict(
        fontset=fontset,
        text_family=_RC.text_family[fontset],
        ink=ink,
        line_width=float(rng.uniform(*_RC.line_width_jitter)),
        point_dots=bool(rng.random() < _RC.point_dot_prob),
        dot_size=float(rng.uniform(*_RC.point_marker_size_jitter)),
        label_fontsize=float(rng.uniform(*_RC.label_fontsize_jitter)),
        label_italic=bool(rng.random() < _RC.label_italic_prob),
        question_fontsize=float(rng.uniform(*_RC.question_fontsize_jitter)),
        mark_size=float(rng.uniform(*_RC.mark_size_jitter)),
        angle_arcs=bool(rng.random() < _RC.angle_arc_prob),
        background=_RC.background,
    )
    # Drawn after every older field, so adding colors left those draws unchanged.
    line_ink = mark_ink = label_ink = ink
    if rng.random() < _RC.colored_prob:  # textbook color: blue figure, red/orange labels
        line_ink = _RC.line_inks[int(rng.integers(len(_RC.line_inks)))]
        accent = _RC.accent_inks[int(rng.integers(len(_RC.accent_inks)))]
        mark_ink = accent if rng.random() < 0.6 else line_ink
        label_ink = accent if rng.random() < 0.5 else ink
    return RenderStyle(**style, line_ink=line_ink, mark_ink=mark_ink, label_ink=label_ink)


def _rc(style: RenderStyle):
    """Matplotlib's mathtext fontset is global state; scope it to one render."""
    return matplotlib.rc_context(
        {
            "mathtext.fontset": style.fontset,
            "font.family": [style.text_family, "DejaVu Sans"],  # fallback for odd glyphs
            "text.usetex": False,
            "axes.formatter.use_mathtext": True,  # silences the cmr10 formatter warning
        }
    )


# ---------------------------------------------------------------------------
# Safe text
# ---------------------------------------------------------------------------

_PARSER = MathTextParser("path")


@lru_cache(maxsize=8192)
def _mathtext_ok(s: str, fontset: str) -> bool:
    with matplotlib.rc_context({"mathtext.fontset": fontset}):
        try:
            _PARSER.parse(s)
        except Exception:
            return False
    return True


def _safe(s: str, style: RenderStyle) -> tuple[str, bool]:
    """Return ``(string, parse_math)``. Strings with unparseable math are drawn raw."""
    if not cbook.is_math_text(s):
        return s, True  # plain text; matplotlib will un-escape \$ itself
    if _mathtext_ok(s, style.fontset):
        return s, True
    return s, False


def _point_label_text(name: str, style: RenderStyle) -> str:
    return f"${name}$" if style.label_italic else f"$\\mathrm{{{name}}}$"


# ---------------------------------------------------------------------------
# Small geometry helpers (data coordinates, y down)
# ---------------------------------------------------------------------------

Vec = tuple[float, float]


def _unit(dx: float, dy: float) -> Vec | None:
    n = math.hypot(dx, dy)
    return None if n < 1e-9 else (dx / n, dy / n)


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def _ang(v: Vec) -> float:
    return math.atan2(v[1], v[0])


def _angdist(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _dist_point_segment(p: Vec, a: Vec, b: Vec) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def grid_to_pixel(x: float, y: float, panel_box: tuple[float, float, float]) -> Vec:
    """Map a grid coordinate to image pixels, given the diagram panel's
    ``(left_px, top_px, size_px)``. For ``render_diagram`` output the panel is the
    whole image: ``(0, 0, width)``."""
    left, top, size = panel_box
    return (left + (x - LO) / SPAN * size, top + (y - LO) / SPAN * size)


# ---------------------------------------------------------------------------
# Diagram drawing
# ---------------------------------------------------------------------------


@dataclass
class _Deco:
    kind: str  # "ticks" | "arrows"
    count: int
    direction: Vec  # unit vector along the segment (arrows point this way)


class _Painter:
    def __init__(self, ax, d: Diagram, style: RenderStyle, panel_in: float):
        self.ax = ax
        self.d = d
        self.s = style
        self.P: dict[str, Vec] = {k: (float(v[0]), float(v[1])) for k, v in d.points.items()}
        self.ppg = panel_in * 72.0 / SPAN  # points per grid unit
        self.renderer = ax.figure.canvas.get_renderer()
        self.tol = get_config().schema_.consistency_collinear_tol
        pts = list(self.P.values())
        self.centroid: Vec = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )
        # angle key -> outermost arc radius drawn there (grid units)
        self.arc_radius: dict[tuple, float] = {}
        # point name -> directions (radians) that a point label should avoid
        self.obstacles: dict[str, list[float]] = {k: [] for k in self.P}
        self.arrow_tips = {m.at for m in d.marks if isinstance(m, ArrowMark)}

    # -- primitives -------------------------------------------------------

    def line(self, xs, ys, lw: float | None = None, z: int = 2, color: str | None = None) -> None:
        self.ax.plot(
            xs,
            ys,
            color=color or self.s.line_ink,
            lw=self.s.line_width if lw is None else lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=z,
        )

    def mark_lw(self) -> float:
        return max(0.6, 0.8 * self.s.line_width)

    def arc(self, v: Vec, a1: float, a2: float, r: float) -> None:
        delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi  # shorter way round
        ts = [a1 + delta * i / 32 for i in range(33)]
        self.line(
            [v[0] + r * math.cos(t) for t in ts],
            [v[1] + r * math.sin(t) for t in ts],
            lw=self.mark_lw(),
            color=self.s.mark_ink,
            z=3,
        )

    def text(self, x: float, y: float, s: str, fontsize: float) -> None:
        """Angle and length labels. A tight background-colored box keeps them legible
        where they land on a line (e.g. an angle label on a median)."""
        s, pm = _safe(s, self.s)
        self.ax.text(
            x, y, s, ha="center", va="center", fontsize=fontsize, color=self.s.label_ink,
            parse_math=pm, zorder=6, clip_on=False,
            bbox={"boxstyle": "square,pad=0.08", "facecolor": self.s.background,
                  "edgecolor": "none"},
        )  # fmt: skip

    # -- passes -----------------------------------------------------------

    def draw(self) -> None:
        self.draw_circles()
        self.draw_segments()
        self.draw_arrows()
        self.draw_right_angles()
        self.draw_segment_decorations()
        self.draw_equal_angles()
        self.draw_angle_labels()
        self.draw_segment_labels()
        self.draw_arc_labels()
        self.draw_free_text()
        self.draw_points()

    def draw_circles(self) -> None:
        for c in self.d.circles:
            self.ax.add_patch(
                CirclePatch(
                    self.P[c.center],
                    float(c.radius),
                    fill=False,
                    edgecolor=self.s.line_ink,
                    lw=self.s.line_width,
                    zorder=2,
                )
            )

    def draw_segments(self) -> None:
        for a, b in self.d.segments:
            pa, pb = self.P[a], self.P[b]
            self.line([pa[0], pb[0]], [pa[1], pb[1]])

    def arrowhead(self, m: ArrowMark) -> list[Vec] | None:
        """Triangle (tip, two base corners) of an arrowhead, in grid units."""
        tip, tail = self.P[m.at], self.P[m.from_]
        u = _unit(*_sub(tip, tail))
        if u is None:
            return None
        length, half = 1.0 * self.s.mark_size, 0.42 * self.s.mark_size
        base = (tip[0] - u[0] * length, tip[1] - u[1] * length)
        n = (-u[1], u[0])
        return [tip, (base[0] + n[0] * half, base[1] + n[1] * half),
                (base[0] - n[0] * half, base[1] - n[1] * half)]  # fmt: skip

    def draw_arrows(self) -> None:
        for m in self.d.marks:
            if isinstance(m, ArrowMark) and (tri := self.arrowhead(m)) is not None:
                self.ax.fill(
                    [p[0] for p in tri], [p[1] for p in tri], color=self.s.line_ink, lw=0, zorder=3
                )
                self.obstacles[m.at].append(_ang(_unit(*_sub(self.P[m.from_], self.P[m.at]))))

    def _bisector(self, v: Vec, e1: Vec, e2: Vec) -> Vec | None:
        u1, u2 = _unit(*_sub(e1, v)), _unit(*_sub(e2, v))
        if u1 is None or u2 is None:
            return None
        b = _unit(u1[0] + u2[0], u1[1] + u2[1])
        if b is None:  # straight angle: pick the normal pointing away from the centroid
            n = (-u1[1], u1[0])
            away = _sub(v, self.centroid)
            b = n if n[0] * away[0] + n[1] * away[1] >= 0 else (-n[0], -n[1])
        return b

    def draw_right_angles(self) -> None:
        for m in self.d.marks:
            if not isinstance(m, RightAngleMark):
                continue
            v, e1, e2 = self.P[m.at], self.P[m.rays[0]], self.P[m.rays[1]]
            u1, u2 = _unit(*_sub(e1, v)), _unit(*_sub(e2, v))
            if u1 is None or u2 is None:
                continue
            side = min(self.s.mark_size, 0.35 * math.dist(v, e1), 0.35 * math.dist(v, e2))
            p1 = (v[0] + side * u1[0], v[1] + side * u1[1])
            p2 = (p1[0] + side * u2[0], p1[1] + side * u2[1])
            p3 = (v[0] + side * u2[0], v[1] + side * u2[1])
            self.line(
                [p1[0], p2[0], p3[0]],
                [p1[1], p2[1], p3[1]],
                lw=self.mark_lw(),
                color=self.s.mark_ink,
                z=3,
            )
            b = self._bisector(v, e1, e2)
            if b is not None:
                self.obstacles[m.at].append(_ang(b))

    def _segment_decorations(self) -> dict[frozenset, tuple[tuple[str, str], list[_Deco]]]:
        decos: dict[frozenset, tuple[tuple[str, str], list[_Deco]]] = {}
        n_eq = n_par = 0
        for m in self.d.marks:
            if isinstance(m, EqualLengthMark):
                n_eq += 1
                for s in m.segments:
                    u = _unit(*_sub(self.P[s[1]], self.P[s[0]]))
                    if u is not None:
                        key = canonical_segment(s)
                        decos.setdefault(key, (s, []))[1].append(_Deco("ticks", min(n_eq, 4), u))
            elif isinstance(m, ParallelMark):
                n_par += 1
                ref = None
                for s in m.segments:
                    u = _unit(*_sub(self.P[s[1]], self.P[s[0]]))
                    if u is None:
                        continue
                    if ref is None:
                        ref = u
                    elif u[0] * ref[0] + u[1] * ref[1] < 0:
                        u = (-u[0], -u[1])  # same arrow direction across the group
                    key = canonical_segment(s)
                    decos.setdefault(key, (s, []))[1].append(_Deco("arrows", min(n_par, 4), u))
        return decos

    def draw_segment_decorations(self) -> None:
        ms = self.s.mark_size
        spacing = 0.45 * ms
        half = 0.55 * ms
        gap = 0.7 * ms
        for (a, b), items in self._segment_decorations().values():
            pa, pb = self.P[a], self.P[b]
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            widths = [
                (it.count - 1) * spacing + (half if it.kind == "arrows" else 0) for it in items
            ]
            total = sum(widths) + gap * (len(items) - 1)
            base = _unit(*_sub(pb, pa))
            if base is None:
                continue
            cursor = -total / 2
            for it, w in zip(items, widths, strict=True):
                d = it.direction
                n = (-d[1], d[0])
                for i in range(it.count):
                    t = cursor + i * spacing
                    c = (mid[0] + base[0] * t, mid[1] + base[1] * t)
                    if it.kind == "ticks":
                        self.line(
                            [c[0] - n[0] * half, c[0] + n[0] * half],
                            [c[1] - n[1] * half, c[1] + n[1] * half],
                            lw=self.mark_lw(),
                            color=self.s.mark_ink,
                            z=3,
                        )
                    else:
                        tip = (c[0] + d[0] * half * 0.5, c[1] + d[1] * half * 0.5)
                        back = (tip[0] - d[0] * half, tip[1] - d[1] * half)
                        w1 = (back[0] + n[0] * half * 0.7, back[1] + n[1] * half * 0.7)
                        w2 = (back[0] - n[0] * half * 0.7, back[1] - n[1] * half * 0.7)
                        self.line(
                            [w1[0], tip[0], w2[0]],
                            [w1[1], tip[1], w2[1]],
                            lw=self.mark_lw(),
                            color=self.s.mark_ink,
                            z=3,
                        )
                cursor += w + gap

    def _half_extent(self, s: str, fontsize: float) -> tuple[float, float]:
        """Half width and half height of a label (including its background box), in grid
        units, measured with the real font rather than estimated from the font size."""
        s, pm = _safe(s, self.s)
        prop = FontProperties(size=fontsize)  # family comes from the active rc context
        ismath = pm and cbook.is_math_text(s)
        w, h, _ = self.renderer.get_text_width_height_descent(s, prop, ismath=ismath)
        px_per_grid = self.ppg * self.ax.figure.dpi / 72.0
        pad = 0.08 * fontsize * self.ax.figure.dpi / 72.0  # matches the bbox pad in text()
        return ((w / 2 + pad) / px_per_grid, (h / 2 + pad) / px_per_grid)

    def _arc_radius(self, v: Vec, e1: Vec, e2: Vec, min_arc_len: float) -> float:
        """Arc radius that grows for narrow angles, so the arc stays long enough to read
        (and to carry ticks), but never reaches past 45% of the shorter ray."""
        u1, u2 = _unit(*_sub(e1, v)), _unit(*_sub(e2, v))
        angle = _angdist(_ang(u1), _ang(u2)) if u1 and u2 else math.pi
        r = max(1.3 * self.s.mark_size, min_arc_len / max(angle, 1e-3))
        cap = 0.45 * min(math.dist(v, e1), math.dist(v, e2))
        return max(min(r, cap), 0.5 * self.s.mark_size)

    def draw_equal_angles(self) -> None:
        ms = self.s.mark_size
        k = 0
        for m in self.d.marks:
            if not isinstance(m, EqualAngleMark):
                continue
            k += 1
            for a in m.angles:
                v, e1, e2 = self.P[a[1]], self.P[a[0]], self.P[a[2]]
                u1, u2 = _unit(*_sub(e1, v)), _unit(*_sub(e2, v))
                if u1 is None or u2 is None:
                    continue
                r = self._arc_radius(v, e1, e2, min_arc_len=1.8 * ms)
                t1, t2 = _ang(u1), _ang(u2)
                self.arc(v, t1, t2, r)
                # k short radial ticks across the arc's middle. Adjacent equal angles
                # (a bisected angle) share a ray, so their arcs touch and would read as one
                # arc without the ticks.
                mid = t1 + ((t2 - t1 + math.pi) % (2 * math.pi) - math.pi) / 2
                h = 0.35 * ms
                step = 0.45 * ms / r
                n_ticks = min(k, 4)
                for j in range(n_ticks):
                    t = mid + (j - (n_ticks - 1) / 2) * step
                    c, sn = math.cos(t), math.sin(t)
                    self.line(
                        [v[0] + (r - h) * c, v[0] + (r + h) * c],
                        [v[1] + (r - h) * sn, v[1] + (r + h) * sn],
                        lw=self.mark_lw(),
                        color=self.s.mark_ink,
                        z=3,
                    )
                self.arc_radius[canonical_angle(a)] = r + h
                b = self._bisector(v, e1, e2)
                if b is not None:
                    self.obstacles[a[1]].append(_ang(b))

    def draw_angle_labels(self) -> None:
        fs = self.s.label_fontsize
        margin = 0.35 * self.s.mark_size
        # three or more labeled angles at one vertex (numbered angles at a crossing): no
        # arcs there, as textbooks draw it; the arcs would merge into a circle
        per_vertex = Counter(lb.angle[1] for lb in self.d.labels if isinstance(lb, AngleLabel))
        crowded = {v for v, n in per_vertex.items() if n >= 3}
        for lb in self.d.labels:
            if not isinstance(lb, AngleLabel):
                continue
            a = lb.angle
            text = f"${lb.text}$"
            v, e1, e2 = self.P[a[1]], self.P[a[0]], self.P[a[2]]
            b = self._bisector(v, e1, e2)
            if b is None:  # coincident points: put the text just above the vertex
                self.text(v[0], v[1] - 2 * self.s.mark_size, text, fs)
                continue
            u1, u2 = _unit(*_sub(e1, v)), _unit(*_sub(e2, v))
            key = canonical_angle(a)
            r_arc = self.arc_radius.get(key)
            if r_arc is None and self.s.angle_arcs and a[1] not in crowded:
                r_arc = self._arc_radius(v, e1, e2, min_arc_len=1.2 * self.s.mark_size)
                self.arc(v, _ang(u1), _ang(u2), r_arc)
            r_arc = r_arc or 0.0

            # Distance along the bisector at which the label's box clears the arc and
            # both sides of the angle.
            hw, hh = self._half_extent(text, fs)
            r = r_arc + margin + hw * abs(b[0]) + hh * abs(b[1])
            for u in (u1, u2):
                n = (-u[1], u[0])
                nb = n[0] * b[0] + n[1] * b[1]
                if nb < 0:
                    n, nb = (-n[0], -n[1]), -nb
                if nb > 1e-3:
                    r = max(r, (margin + hw * abs(n[0]) + hh * abs(n[1])) / nb)
            r = min(r, max(0.9 * min(math.dist(v, e1), math.dist(v, e2)), r_arc + margin))
            self.text(v[0] + b[0] * r, v[1] + b[1] * r, text, fs)
            self.obstacles[a[1]].append(_ang(b))

    def draw_segment_labels(self) -> None:
        fs = self.s.label_fontsize
        margin = 0.3 * self.s.mark_size
        decos = self._segment_decorations()
        decorated = set(decos)
        # things a length label must not sit on: points (and their labels) and the tick
        # or arrow marks drawn at decorated segments' midpoints
        deco_centers = [
            ((self.P[a][0] + self.P[b][0]) / 2, (self.P[a][1] + self.P[b][1]) / 2)
            for (a, b), _ in decos.values()
        ]
        clear = 2.5 * self.s.mark_size
        for lb in self.d.labels:
            if not isinstance(lb, SegmentLabel):
                continue
            text = f"${lb.text}$"
            pa, pb = self.P[lb.on[0]], self.P[lb.on[1]]
            others = [p for n, p in self.P.items() if n not in lb.on]
            own = canonical_segment(lb.on) in decorated
            obstacles = others + [c for c in deco_centers if not own or c != (
                (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)]  # fmt: skip
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            for t in (0.5, 0.35, 0.65, 0.25, 0.75):  # slide along the segment if crowded
                cand = (pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t)
                if all(math.dist(cand, o) >= clear for o in obstacles):
                    mid = cand
                    break
            u = _unit(*_sub(pb, pa))
            n = (0.0, -1.0) if u is None else (-u[1], u[0])
            away = _sub(mid, self.centroid)
            dot = n[0] * away[0] + n[1] * away[1]
            if dot < -1e-6 or (abs(dot) <= 1e-6 and n[1] > 0):
                n = (-n[0], -n[1])
            hw, hh = self._half_extent(text, fs)
            off = margin + hw * abs(n[0]) + hh * abs(n[1])
            if canonical_segment(lb.on) in decorated:
                off += 0.55 * self.s.mark_size  # clear the ticks/arrows at the midpoint
            self.text(mid[0] + n[0] * off, mid[1] + n[1] * off, text, fs)

    def arc_label_anchor(self, lb: ArcLabel, text: str, fontsize: float) -> Vec:
        """Center of an arc label: outside the circle through both endpoints, on the
        radius through the minor arc's midpoint. Without such a circle (an implausible
        prediction), beside the chord's midpoint, away from the figure."""
        a, b = self.P[lb.arc[0]], self.P[lb.arc[1]]
        hw, hh = self._half_extent(text, fontsize)
        margin = 0.3 * self.s.mark_size
        best = None
        for c in self.d.circles:
            o, r = self.P[c.center], float(c.radius)
            err = abs(math.dist(a, o) - r) + abs(math.dist(b, o) - r)
            if best is None or err < best[0]:
                best = (err, o, r)
        if best is not None and best[0] <= 2 * (self.tol + 1.0):
            _, o, r = best
            t1, t2 = _ang(_sub(a, o)), _ang(_sub(b, o))
            tm = t1 + ((t2 - t1 + math.pi) % (2 * math.pi) - math.pi) / 2
            u = (math.cos(tm), math.sin(tm))
        else:
            o = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            r = 0.0
            u = _unit(*_sub(o, self.centroid)) or (0.0, -1.0)
        off = r + margin + hw * abs(u[0]) + hh * abs(u[1])
        return (o[0] + u[0] * off, o[1] + u[1] * off)

    def draw_arc_labels(self) -> None:
        fs = self.s.label_fontsize
        for lb in self.d.labels:
            if isinstance(lb, ArcLabel):
                text = f"${lb.text}$"
                self.text(*self.arc_label_anchor(lb, text, fs), text, fs)

    def draw_free_text(self) -> None:
        for lb in self.d.labels:
            if isinstance(lb, FreeTextLabel):
                self.text(float(lb.at[0]), float(lb.at[1]), f"${lb.text}$", self.s.label_fontsize)

    def _ray_directions(self, name: str) -> list[float]:
        p = self.P[name]
        dirs: list[float] = []
        for a, b in self.d.segments:
            if name in (a, b):
                other = self.P[b if a == name else a]
                u = _unit(*_sub(other, p))
                if u is not None:
                    dirs.append(_ang(u))
            elif _dist_point_segment(p, self.P[a], self.P[b]) <= self.tol:
                for end in (a, b):  # point in the interior of a segment: two rays
                    u = _unit(*_sub(self.P[end], p))
                    if u is not None:
                        dirs.append(_ang(u))
        for c in self.d.circles:
            radial = _unit(*_sub(p, self.P[c.center]))
            if radial is not None and abs(math.dist(p, self.P[c.center]) - c.radius) <= self.tol:
                t = _ang(radial)
                dirs.extend([t + math.pi / 2, t - math.pi / 2])  # tangent directions
        return dirs

    def label_direction(self, name: str) -> float:
        """Direction (radians, data coords) in which to place a point's label: the
        candidate with the most angular clearance from drawn rays and marks, with a mild
        preference for pointing away from the figure's centroid."""
        p = self.P[name]
        avoid = self._ray_directions(name) + self.obstacles[name]
        away = _unit(*_sub(p, self.centroid))
        away_t = _ang(away) if away is not None else -math.pi / 2  # default: up
        best, best_score = away_t, -math.inf
        for i in range(24):
            t = -math.pi + i * (2 * math.pi / 24)
            clearance = min((_angdist(t, r) for r in avoid), default=math.pi)
            score = clearance + 0.5 * math.cos(t - away_t)
            if score > best_score + 1e-9:
                best, best_score = t, score
        return best

    def draw_points(self) -> None:
        centers = {c.center for c in self.d.circles}
        fs = self.s.label_fontsize
        dist_pt = _RC.label_offset_pt + 0.35 * fs
        for name, p in self.P.items():
            if (self.s.point_dots or name in centers) and name not in self.arrow_tips:
                self.ax.plot(
                    [p[0]], [p[1]], "o", ms=self.s.dot_size, color=self.s.line_ink, zorder=5,
                    markeredgewidth=0,
                )  # fmt: skip
            if is_unnamed(name):
                continue
            t = self.label_direction(name)
            s, pm = _safe(_point_label_text(name, self.s), self.s)
            self.ax.annotate(
                s,
                xy=p,
                # data y points down, display y points up
                xytext=(dist_pt * math.cos(t), -dist_pt * math.sin(t)),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=fs,
                color=self.s.ink,
                parse_math=pm,
                annotation_clip=False,
                zorder=6,
            )


def _setup_axes(ax) -> None:
    ax.set_xlim(LO, HI)
    ax.set_ylim(HI, LO)  # y points down
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _to_image(canvas: FigureCanvasAgg) -> Image.Image:
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    return Image.fromarray(arr, "RGB")


def render_diagram(d: Diagram, style: RenderStyle | None = None) -> Image.Image:
    """Diagram only, as a square image whose pixels map to the grid via
    ``grid_to_pixel(x, y, (0, 0, image.width))``."""
    style = style or default_style()
    size = _RC.figure_width_in
    with _rc(style):
        fig = Figure(figsize=(size, size), dpi=_RC.dpi, facecolor=style.background)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        _setup_axes(ax)
        _Painter(ax, d, style, size).draw()
        return _to_image(canvas)


# ---------------------------------------------------------------------------
# Question text layout
# ---------------------------------------------------------------------------

_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
_PARA_RE = re.compile(r"\n[ \t]*\n")
_FRAC_RE = re.compile(r"\\frac(?![A-Za-z])")


@dataclass
class _Line:
    s: str
    parse_math: bool
    display: bool
    ascent_px: float
    descent_px: float
    gap_before_px: float


def _words(s: str) -> list[str]:
    """Split on whitespace outside $...$ so inline math is never broken across lines."""
    words: list[str] = []
    cur: list[str] = []
    in_math = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "$":
            cur.append(s[i : i + 2])
            i += 2
            continue
        if ch == "$":
            in_math = not in_math
        if ch.isspace() and not in_math:
            if cur:
                words.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        words.append("".join(cur))
    return words


def _logical_lines(text: str) -> list[tuple[str, bool, bool]]:
    """Split question Markdown into ``(content, is_display, starts_paragraph)``.

    Supported subset: paragraphs (blank lines), hard line breaks (single newlines),
    inline ``$...$`` and display ``$$...$$`` math. Other Markdown syntax is drawn as-is.
    """
    out: list[tuple[str, bool, bool]] = []
    for para in _PARA_RE.split(text.strip()):
        first = True
        pos = 0
        pieces: list[tuple[str, bool]] = []
        for m in _DISPLAY_RE.finditer(para):
            pieces.append((para[pos : m.start()], False))
            pieces.append((m.group(1).strip(), True))
            pos = m.end()
        pieces.append((para[pos:], False))
        for content, display in pieces:
            if display:
                # $$...$$ is display style in LaTeX; mathtext needs \dfrac for that.
                # Rendering only: the ground-truth text keeps \frac.
                content = _FRAC_RE.sub(r"\\dfrac", content)
                out.append((f"${content}$", True, first))
                first = False
                continue
            for raw in content.split("\n"):
                if raw.strip():
                    out.append((raw.strip(), False, first))
                    first = False
    return out


class _Measurer:
    def __init__(self, renderer, style: RenderStyle):
        self.r = renderer
        self.style = style
        self.prop = FontProperties(
            family=[style.text_family, "DejaVu Sans"], size=style.question_fontsize
        )
        _, h, d = self.r.get_text_width_height_descent("lp", self.prop, ismath=False)
        self.ref_ascent, self.ref_descent = h - d, d

    def measure(self, s: str) -> tuple[float, float, float, str, bool]:
        s, pm = _safe(s, self.style)
        ismath = pm and cbook.is_math_text(s)
        w, h, d = self.r.get_text_width_height_descent(s, self.prop, ismath=ismath)
        return w, h - d, d, s, pm


def _layout_text(text: str, measurer: _Measurer, max_width_px: float) -> list[_Line]:
    lines: list[_Line] = []
    para_gap = 0.5 * (measurer.ref_ascent + measurer.ref_descent)

    def emit(s: str, display: bool, gap: float) -> None:
        _, asc, desc, s2, pm = measurer.measure(s)
        lines.append(
            _Line(
                s2, pm, display, max(asc, measurer.ref_ascent), max(desc, measurer.ref_descent),
                gap,
            )
        )  # fmt: skip

    for content, display, new_para in _logical_lines(text):
        gap = para_gap if (new_para and lines) else 0.0
        if display:
            emit(content, True, gap)
            continue
        cur: list[str] = []
        for w in _words(content):
            cand = " ".join([*cur, w])
            if cur and measurer.measure(cand)[0] > max_width_px:
                emit(" ".join(cur), False, gap)
                gap = 0.0
                cur = [w]
            else:
                cur.append(w)
        if cur:
            emit(" ".join(cur), False, gap)
    return lines


@dataclass(frozen=True)
class RenderedPage:
    image: Image.Image
    # (left_px, top_px, size_px) of the diagram panel, or None if there is no diagram.
    # Feed it to grid_to_pixel to locate grid points in the image.
    panel_box: tuple[float, float, float] | None


def render_page(t: Transcription, style: RenderStyle | None = None) -> RenderedPage:
    """The question text, wrapped to the page width, above the diagram panel.
    Questions with ``diagram: null`` render as text only."""
    style = style or default_style()
    dpi = _RC.dpi
    W = _RC.question_width_in
    m = _RC.page_margin_in
    diag_w = min(_RC.figure_width_in, W - 2 * m)
    has_diag = t.diagram is not None

    with _rc(style):
        fig = Figure(figsize=(W, 2.0), dpi=dpi, facecolor=style.background)
        canvas = FigureCanvasAgg(fig)
        measurer = _Measurer(canvas.get_renderer(), style)
        lines = _layout_text(t.text, measurer, (W - 2 * m) * dpi)

        spacing = _RC.line_spacing
        text_h = (
            sum(ln.gap_before_px + (ln.ascent_px + ln.descent_px) * spacing for ln in lines) / dpi
        )
        diag_block = (_RC.text_diagram_gap_in if lines else 0.0) + diag_w if has_diag else 0.0
        H = m + text_h + diag_block + m
        fig.set_size_inches(W, H)

        y = H - m  # inches from the bottom edge
        for ln in lines:
            y -= ln.gap_before_px / dpi
            baseline = y - ln.ascent_px / dpi
            x, ha = (W / 2, "center") if ln.display else (m, "left")
            fig.text(
                x, baseline, ln.s, transform=fig.dpi_scale_trans, ha=ha, va="baseline",
                fontsize=style.question_fontsize, color=style.ink, parse_math=ln.parse_math,
            )  # fmt: skip
            y -= (ln.ascent_px + ln.descent_px) * spacing / dpi

        panel_box = None
        if has_diag:
            left = (W - diag_w) / 2
            bottom = m
            ax = fig.add_axes((left / W, bottom / H, diag_w / W, diag_w / H))
            _setup_axes(ax)
            _Painter(ax, t.diagram, style, diag_w).draw()
            panel_box = (left * dpi, (H - bottom - diag_w) * dpi, diag_w * dpi)

        return RenderedPage(_to_image(canvas), panel_box)


# ---------------------------------------------------------------------------
# Contact sheets and CLI
# ---------------------------------------------------------------------------


def contact_sheet(
    images: Iterable[Image.Image], cols: int = 3, pad: int = 12, bg=(205, 205, 205)
) -> Image.Image:
    """Tile images into a grid (top-aligned cells), for eyeballing many renders."""
    images = list(images)
    if not images:
        raise ValueError("no images")
    cw = max(im.width for im in images)
    rows = math.ceil(len(images) / cols)
    row_h = [max(im.height for im in images[r * cols : (r + 1) * cols]) for r in range(rows)]
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * pad, sum(row_h) + (rows + 1) * pad), bg)
    y = pad
    for r in range(rows):
        for c, im in enumerate(images[r * cols : (r + 1) * cols]):
            sheet.paste(im, (pad + c * (cw + pad), y))
        y += row_h[r] + pad
    return sheet


def _render_fixtures(out: Path, n_styles: int, seed: int) -> list[Path]:
    from mathvlm.fixtures import DIAGRAMS, PAGES

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    diagram_imgs = []
    for name, obj in DIAGRAMS.items():
        im = render_diagram(Diagram.model_validate(obj))
        p = out / f"diagram_{name}.png"
        im.save(p)
        written.append(p)
        diagram_imgs.append(im)
    for name, obj in PAGES.items():
        p = out / f"{name}.png"
        render_page(Transcription.model_validate(obj)).image.save(p)
        written.append(p)

    p = out / "contact_diagrams.png"
    contact_sheet(diagram_imgs, cols=4).save(p)
    written.append(p)

    page = Transcription.model_validate(PAGES["page_bisector_display_math"])
    variants = [
        render_page(page, sample_style(sample_rng(seed, "render_fixtures", i))).image
        for i in range(n_styles)
    ]
    p = out / "style_variants.png"
    contact_sheet(variants, cols=3).save(p)
    written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.render", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--fixtures", action="store_true", help="render the reference fixtures")
    ap.add_argument("--out", type=Path, default=None, help="default: <outputs_dir>/fixtures")
    ap.add_argument("--styles", type=int, default=6, help="sampled-style variants to render")
    ap.add_argument("--seed", type=int, default=None, help=f"default: config seed ({cfg.seed})")
    args = ap.parse_args(argv)
    if not args.fixtures:
        ap.error("nothing to do: pass --fixtures")
    out = args.out or cfg.paths.resolved("outputs_dir") / "fixtures"
    seed = cfg.seed if args.seed is None else args.seed
    written = _render_fixtures(out, args.styles, seed)
    print(f"wrote {len(written)} images to {out}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
