"""Random but geometrically valid scene families.

Each builder constructs a figure in a float frame from exact geometry (feet of
altitudes, incenters, tangent points, ...) and records, alongside the drawing, what is
*true* about it: statements the question may make, which lengths and angles are equal,
which angles are exactly 90 degrees, and which points lie on which segments or circles
by construction. ``to_grid`` then rotates/reflects the figure and fits it to the
canonical box on the integer grid (design invariant 5), and ``validate_scene`` rejects
figures that would be ambiguous or ugly once rounded.

All randomness comes from the ``rng`` passed in, so a scene is a pure function of the
generator state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from mathvlm.config import GenerateConfig
from mathvlm.generate import primitives as G
from mathvlm.schema import GRID_MAX, GRID_MIN, UNNAMED_PREFIX, canonical_angle, is_unnamed

Vec = G.Vec
Seg = tuple[str, str]
Ang = tuple[str, str, str]  # (end, vertex, end)

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


# ---------------------------------------------------------------------------
# Random helpers (all draws go through these, in a fixed order)
# ---------------------------------------------------------------------------


def pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def wpick(rng: np.random.Generator, items, weights):
    w = np.asarray(weights, dtype=float)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def chance(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


# ---------------------------------------------------------------------------
# Point naming
# ---------------------------------------------------------------------------


class Namer:
    """Hands out distinct point names according to a per-sample labeling scheme."""

    def __init__(self, rng: np.random.Generator, cfg: GenerateConfig):
        self.rng = rng
        self.cfg = cfg
        schemes = list(cfg.label_schemes)
        self.scheme: str = wpick(rng, schemes, [cfg.label_schemes[s] for s in schemes])
        self.used: set[str] = set()
        self.n_unnamed = 0
        self.cursor = int(rng.integers(0, 26 - 12)) if self.scheme == "consecutive" else 0

    def _take(self, name: str) -> str:
        self.used.add(name)
        return name

    def _random_free(self) -> str:
        return self._take(pick(self.rng, [c for c in LETTERS if c not in self.used]))

    def _next_free(self) -> str:
        while LETTERS[self.cursor % 26] in self.used:
            self.cursor += 1
        name = LETTERS[self.cursor % 26]
        self.cursor += 1
        return self._take(name)

    def _first_free(self) -> str:
        return self._take(next(c for c in LETTERS if c not in self.used))

    def vertices(self, n: int) -> list[str]:
        if self.scheme == "conventional":
            return [self._take(c) for c in LETTERS[:n]]
        if self.scheme == "consecutive":
            return [self._next_free() for _ in range(n)]
        return [self._random_free() for _ in range(n)]

    def point(self, preferred: tuple[str, ...] = (), base: str | None = None) -> str:
        """A labeled constructed point, e.g. a foot, midpoint or center."""
        if base is not None and chance(self.rng, self.cfg.prime_name_prob):
            for cand in (f"{base}'", f"{base}_1", f"{base}''", f"{base}_2"):
                if cand not in self.used:
                    return self._take(cand)
        if self.scheme == "conventional":
            for cand in preferred:
                if cand not in self.used:
                    return self._take(cand)
            return self._first_free()
        if self.scheme == "consecutive":
            return self._next_free()
        return self._random_free()

    def unnamed(self) -> str:
        self.n_unnamed += 1
        return f"{UNNAMED_PREFIX}{self.n_unnamed}"

    def aux(self, preferred: tuple[str, ...] = (), base: str | None = None) -> str:
        """A constructed point that is left unlabeled with probability unnamed_aux_prob."""
        if chance(self.rng, self.cfg.unnamed_aux_prob):
            return self.unnamed()
        return self.point(preferred, base)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    family: str
    kind: str
    points: dict[str, Vec] = field(default_factory=dict)
    segments: list[Seg] = field(default_factory=list)
    circles: list[tuple[str, float]] = field(default_factory=list)  # (center, radius)
    marks: list[dict] = field(default_factory=list)
    # --- what is true about the figure (drives the question text) ---
    statements: list[tuple] = field(default_factory=list)
    lengths: list[Seg] = field(default_factory=list)  # may be labeled / asked
    angles: list[Ang] = field(default_factory=list)  # may be labeled / asked
    equal_lengths: list[list[Seg]] = field(default_factory=list)
    equal_angles: list[list[Ang]] = field(default_factory=list)
    exact_angles: dict[tuple, float] = field(default_factory=dict)  # canonical angle -> deg
    asks: list[tuple] = field(default_factory=list)  # family-specific questions
    max_angle_givens: int = 2
    # --- incidences that hold by construction ---
    polygon: tuple[str, ...] = ()
    on_segment: set[tuple[str, frozenset]] = field(default_factory=set)
    on_circle: set[tuple[str, str]] = field(default_factory=set)

    def add_segment(self, a: str, b: str) -> None:
        if frozenset((a, b)) not in {frozenset(s) for s in self.segments}:
            self.segments.append((a, b))
            self.lengths.append((a, b))

    def add_mark(self, rng: np.random.Generator, cfg: GenerateConfig, mark: dict) -> None:
        if chance(rng, cfg.mark_prob):
            self.marks.append(mark)

    def right_angle(self, rng, cfg, at: str, r1: str, r2: str) -> None:
        self.exact_angles[canonical_angle((r1, at, r2))] = 90.0
        self.add_mark(rng, cfg, {"type": "right_angle", "at": at, "rays": [r1, r2]})


# ---------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------

TRI_KINDS = ("scalene", "right", "isosceles")
TRI_KIND_WEIGHTS = (0.5, 0.25, 0.25)


def _triangle(rng, cfg: GenerateConfig, kind: str) -> tuple[Vec, Vec, Vec]:
    m = cfg.min_angle_deg
    if kind == "right":  # right angle at C
        alpha = rng.uniform(m, 90 - m)
        return G.triangle_from_angles(alpha, 90 - alpha)
    if kind == "isosceles":  # apex C, apex angle 180 - 2*alpha >= m
        alpha = rng.uniform(m, 90 - m / 2)
        return G.triangle_from_angles(alpha, alpha)
    for _ in range(100):  # scalene: clearly unequal angles, clearly not right
        alpha = rng.uniform(m, 180 - 2 * m)
        beta = rng.uniform(m, 180 - alpha - m)
        angs = sorted((alpha, beta, 180 - alpha - beta))
        if min(angs[1] - angs[0], angs[2] - angs[1]) > 5 and abs(angs[2] - 90) > 5:
            break
    return G.triangle_from_angles(alpha, beta)


def _base_triangle(sc: Scene, rng, namer: Namer, cfg: GenerateConfig, kind: str) -> Seg:
    a, b, c = _triangle(rng, cfg, kind)
    A, B, C = namer.vertices(3)
    sc.points.update({A: a, B: b, C: c})
    sc.polygon = (A, B, C)
    for s in ((A, B), (B, C), (C, A)):
        sc.add_segment(*s)
    sc.angles += [(B, A, C), (A, B, C), (A, C, B)]
    sc.statements.append(("triangle", (A, B, C), kind))
    if kind == "right":
        sc.right_angle(rng, cfg, C, A, B)
        sc.statements.append(("right_angle", (A, C, B)))
    elif kind == "isosceles":
        sc.equal_lengths.append([(C, A), (C, B)])
        sc.equal_angles.append([(B, A, C), (A, B, C)])
        sc.add_mark(rng, cfg, {"type": "equal_length", "segments": [[C, A], [C, B]]})
        sc.statements.append(("isosceles", C, (A, B)))
    sc.asks += [("area", "triangle", (A, B, C)), ("perimeter", "triangle", (A, B, C))]
    return A, B, C


def _cevian(
    sc: Scene, rng, namer: Namer, cfg: GenerateConfig, kind: str, V: str, P: str, Q: str
) -> bool:
    v, p, q = sc.points[V], sc.points[P], sc.points[Q]
    if kind == "altitude":
        t = G.project_param(v, p, q)
        if not 0.15 < t < 0.85:  # foot must land well inside the opposite side
            return False
        F = namer.aux(("H", "D", "E", "F"), base=V)
        f = G.lerp(p, q, t)
        sc.points[F] = f
        sc.right_angle(rng, cfg, F, V, Q)
        sc.exact_angles[canonical_angle((V, F, P))] = 90.0
    elif kind == "median":
        F = namer.aux(("M", "N", "D", "E"), base=V)
        sc.points[F] = G.midpoint(p, q)
        sc.equal_lengths.append([(P, F), (F, Q)])
        sc.add_mark(rng, cfg, {"type": "equal_length", "segments": [[P, F], [F, Q]]})
    else:  # bisector
        F = namer.aux(("D", "E", "F"), base=V)
        sc.points[F] = G.bisector_foot(v, p, q)
        sc.equal_angles.append([(P, V, F), (F, V, Q)])
        sc.add_mark(rng, cfg, {"type": "equal_angle", "angles": [[P, V, F], [F, V, Q]]})
        if not is_unnamed(F):
            sc.asks.append(
                ("prove", f"\\frac{{{P}{F}}}{{{F}{Q}}} = \\frac{{{V}{P}}}{{{V}{Q}}}", True)
            )
    sc.add_segment(V, F)
    sc.on_segment.add((F, frozenset((P, Q))))
    sc.lengths += [(P, F), (F, Q)]
    sc.angles += [(P, V, F), (F, V, Q)]
    sc.statements.append((kind, V, F, P, Q))
    return True


def triangle_cevian(rng, namer: Namer, cfg: GenerateConfig) -> Scene:
    kind = wpick(rng, TRI_KINDS, TRI_KIND_WEIGHTS)
    sc = Scene("triangle_cevian", kind)
    A, B, C = _base_triangle(sc, rng, namer, cfg, kind)
    n = 2 if chance(rng, 0.3) else 1
    order = [(A, B, C), (B, C, A), (C, A, B)]
    for i in rng.permutation(3)[:n]:
        P, Q, V = order[int(i)]  # cevian from V to side PQ
        cev = wpick(rng, ("altitude", "median", "bisector"), (1, 1, 1))
        if not _cevian(sc, rng, namer, cfg, cev, V, P, Q):
            _cevian(sc, rng, namer, cfg, "median", V, P, Q)
    # isosceles + cevian from the apex: the two halves are congruent
    if kind == "isosceles":
        for st in sc.statements:
            if st[0] in ("altitude", "median", "bisector") and st[1] == C and not is_unnamed(st[2]):
                F = st[2]
                sc.asks.append(
                    ("prove", f"\\triangle {A}{C}{F} \\cong \\triangle {B}{C}{F}", False)
                )
    return sc


def circle_incircle(rng, namer: Namer, cfg: GenerateConfig) -> Scene:
    kind = wpick(rng, TRI_KINDS, TRI_KIND_WEIGHTS)
    sc = Scene("circle_incircle", kind)
    A, B, C = _base_triangle(sc, rng, namer, cfg, kind)
    a, b, c = (sc.points[n] for n in (A, B, C))
    I = namer.point(("I", "O"))
    i = G.incenter(a, b, c)
    sc.points[I] = i
    sc.circles.append((I, G.inradius(a, b, c)))
    sc.statements.append(("incircle", I, (A, B, C)))
    sc.asks.append(("radius", I))

    sides = [(A, B), (B, C), (C, A)]
    n_touch = wpick(rng, (0, 1, 2, 3), (0.25, 0.4, 0.2, 0.15))
    touch: dict[Seg, str] = {}
    radii: list[Seg] = []
    for k in rng.permutation(3)[:n_touch]:
        P, Q = sides[int(k)]
        T = namer.aux(("D", "E", "F", "T"))
        sc.points[T] = G.foot(i, sc.points[P], sc.points[Q])
        sc.add_segment(I, T)
        radii.append((I, T))
        sc.right_angle(rng, cfg, T, I, Q)
        sc.on_segment.add((T, frozenset((P, Q))))
        sc.on_circle.add((T, I))
        sc.lengths += [(P, T), (T, Q)]
        touch[(P, Q)] = T
        if not is_unnamed(T):
            sc.statements.append(("touches", I, T, P, Q))
    # tangent lengths from a common vertex are equal
    for V in (A, B, C):
        pieces = [(V, T) for (P, Q), T in touch.items() if V in (P, Q)]
        if len(pieces) == 2:
            sc.equal_lengths.append(pieces)
    if len(radii) >= 2:
        sc.equal_lengths.append(radii)
    if chance(rng, 0.25):  # bisectors from the vertices to the incenter
        for k in rng.permutation(3)[: int(rng.integers(1, 3))]:
            V = (A, B, C)[int(k)]
            P, Q = [n for n in (A, B, C) if n != V]
            sc.add_segment(V, I)
            sc.equal_angles.append([(P, V, I), (I, V, Q)])
            sc.add_mark(rng, cfg, {"type": "equal_angle", "angles": [[P, V, I], [I, V, Q]]})
            sc.angles += [(P, V, I), (I, V, Q)]
    return sc


def circle_circumcircle(rng, namer: Namer, cfg: GenerateConfig) -> Scene:
    m = cfg.min_angle_deg
    right = chance(rng, 0.25)
    sc = Scene("circle_circumcircle", "right" if right else "general")
    if right:
        a = rng.uniform(2 * m, 180 - 2 * m)
        arcs = [180.0, a, 180.0 - a]  # the side spanning the 180-degree arc is a diameter
    else:
        for _ in range(100):
            a1 = rng.uniform(2 * m, 360 - 4 * m)
            a2 = rng.uniform(2 * m, 360 - a1 - 2 * m)
            arcs = [a1, a2, 360 - a1 - a2]
            if min(arcs) >= 2 * m and all(abs(x - 180) > 12 for x in arcs):
                break
    phi = rng.uniform(0, 360)
    ts = [phi, phi + arcs[0], phi + arcs[0] + arcs[1]]
    pts = [(math.cos(math.radians(t)), math.sin(math.radians(t))) for t in ts]
    A, B, C = namer.vertices(3)
    O = namer.point(("O",))
    names = (A, B, C)
    sc.points.update(dict(zip(names, pts, strict=True)))
    sc.points[O] = (0.0, 0.0)
    sc.circles.append((O, 1.0))
    sc.polygon = names
    for s in ((A, B), (B, C), (C, A)):
        sc.add_segment(*s)
    sc.angles += [(B, A, C), (A, B, C), (A, C, B)]
    for n in names:
        sc.on_circle.add((n, O))
    sc.statements.append(("circumcircle", O, names))
    sc.asks.append(("radius", O))
    sc.max_angle_givens = 1  # inscribed vs central angles would clash after rounding

    candidates = list(names)
    radii: list[Seg] = []
    if right:  # arc 0 runs from A to B, so AB is a diameter and the right angle is at C
        sc.on_segment.add((O, frozenset((A, B))))
        sc.right_angle(rng, cfg, C, A, B)
        sc.statements.append(("diameter", A, B, O))
        sc.lengths += [(A, O), (O, B)]
        radii += [(O, A), (O, B)]
        candidates = [C]  # radii to A and B would overlap the diameter
    k = wpick(rng, (0, 1, 2, 3), (0.3, 0.3, 0.3, 0.1))
    drawn = [candidates[int(j)] for j in rng.permutation(len(candidates))[:k]]
    for V in drawn:
        sc.add_segment(O, V)
        radii.append((O, V))
    if len(radii) >= 2:
        sc.equal_lengths.append(radii)
        drawn_radii = [r for r in radii if r[1] in drawn]
        if len(drawn_radii) >= 2 and chance(rng, 0.5):
            sc.marks.append({"type": "equal_length", "segments": [list(r) for r in drawn_radii]})
    for i in range(len(drawn)):
        for j in range(i + 1, len(drawn)):
            sc.angles.append((drawn[i], O, drawn[j]))
    return sc


# ---------------------------------------------------------------------------
# Quadrilaterals
# ---------------------------------------------------------------------------

QUAD_KINDS = ("parallelogram", "rectangle", "rhombus", "square", "trapezoid", "kite", "convex")
QUAD_KIND_WEIGHTS = (0.22, 0.14, 0.12, 0.08, 0.18, 0.10, 0.16)


def _quad(rng, cfg: GenerateConfig, kind: str) -> list[Vec]:
    m = cfg.min_angle_deg
    if kind in ("parallelogram", "rhombus", "rectangle", "square"):
        theta = 90.0 if kind in ("rectangle", "square") else rng.uniform(m + 5, 78)
        b = 1.0 if kind in ("rhombus", "square") else rng.uniform(0.45, 0.85)
        t = math.radians(theta)
        a0, b0, d0 = (0.0, 0.0), (1.0, 0.0), (b * math.cos(t), -b * math.sin(t))
        return [a0, b0, G.add(b0, d0), d0]
    if kind == "trapezoid":
        h = rng.uniform(0.35, 0.8)
        c = rng.uniform(0.35, 0.8)
        d1 = (1 - c) / 2 if chance(rng, 0.35) else rng.uniform(-0.1, 1 - c + 0.1)
        return [(0.0, 0.0), (1.0, 0.0), (d1 + c, -h), (d1, -h)]
    if kind == "kite":
        p = rng.uniform(0.3, 0.7)
        q = rng.uniform(p + 0.3, 1.6)
        w = rng.uniform(0.4, 0.8)
        return [(0.0, -q), (w, 0.0), (0.0, p), (-w, 0.0)]  # apex A at the long end
    for _ in range(100):  # general convex quadrilateral
        angs = [math.radians(90 * i + rng.uniform(-35, 35)) for i in range(4)]
        rad = [rng.uniform(0.7, 1.3) for _ in range(4)]
        pts = [(r * math.cos(t), -r * math.sin(t)) for r, t in zip(rad, angs, strict=True)]
        if G.is_convex(pts) and min(G.polygon_angles(pts)) >= m:
            break
    return pts


def quadrilateral_diagonals(rng, namer: Namer, cfg: GenerateConfig) -> Scene:
    kind = wpick(rng, QUAD_KINDS, QUAD_KIND_WEIGHTS)
    sc = Scene("quadrilateral_diagonals", kind)
    pts = _quad(rng, cfg, kind)
    A, B, C, D = namer.vertices(4)
    sc.points.update(dict(zip((A, B, C, D), pts, strict=True)))
    sc.polygon = (A, B, C, D)
    for s in ((A, B), (B, C), (C, D), (D, A), (A, C), (B, D)):
        sc.add_segment(*s)
    e = G.line_intersection(*pts[0::2], *pts[1::2])
    E = namer.aux(("O", "E", "P", "M"))
    sc.points[E] = e
    sc.on_segment |= {(E, frozenset((A, C))), (E, frozenset((B, D)))}
    sc.lengths += [(A, E), (E, C), (B, E), (E, D)]
    sc.angles += [(D, A, B), (A, B, C), (B, C, D), (C, D, A), (A, E, B), (B, A, C), (A, C, D)]
    sc.statements.append(("quad", kind, (A, B, C, D)))
    sc.statements.append(("diagonals", (A, C), (B, D), E))
    sc.asks += [("area", "quad", (A, B, C, D)), ("perimeter", "quad", (A, B, C, D))]
    sc.max_angle_givens = 1  # opposite/adjacent angle relations would clash after rounding

    def mark(p: float, mk: dict) -> None:
        if chance(rng, p):
            sc.marks.append(mk)

    is_par = kind in ("parallelogram", "rectangle", "rhombus", "square")
    if is_par:
        sc.equal_lengths += [[(A, B), (D, C)], [(A, D), (B, C)], [(A, E), (E, C)], [(B, E), (E, D)]]
        sc.equal_angles += [[(D, A, B), (B, C, D)], [(A, B, C), (C, D, A)]]
        p_par = cfg.mark_prob if kind == "parallelogram" else 0.3
        mark(p_par, {"type": "parallel", "segments": [[A, B], [D, C]]})
        mark(p_par, {"type": "parallel", "segments": [[A, D], [B, C]]})
    if kind == "parallelogram":
        mark(0.4, {"type": "equal_length", "segments": [[A, E], [E, C]]})
        mark(0.4, {"type": "equal_length", "segments": [[B, E], [E, D]]})
    if kind in ("rectangle", "square"):
        corners = [(A, D, B), (B, A, C), (C, B, D), (D, C, A)]
        for at, r1, r2 in corners:
            sc.exact_angles[canonical_angle((r1, at, r2))] = 90.0
        sc.add_mark(rng, cfg, {"type": "right_angle", "at": A, "rays": [D, B]})
        for at, r1, r2 in corners[1:]:
            mark(0.5, {"type": "right_angle", "at": at, "rays": [r1, r2]})
        sc.equal_lengths.append([(A, E), (E, C), (B, E), (E, D)])
    if kind in ("rhombus", "square"):
        sc.equal_lengths.append([(A, B), (B, C), (C, D), (D, A)])
        sc.add_mark(
            rng, cfg, {"type": "equal_length", "segments": [[A, B], [B, C], [C, D], [D, A]]}
        )
        sc.exact_angles[canonical_angle((A, E, B))] = 90.0
        mark(0.6, {"type": "right_angle", "at": E, "rays": [A, B]})
    if kind == "trapezoid":
        sc.add_mark(rng, cfg, {"type": "parallel", "segments": [[A, B], [D, C]]})
        if abs(G.dist(pts[0], pts[3]) - G.dist(pts[1], pts[2])) < 1e-6:  # isosceles
            sc.equal_lengths.append([(A, D), (B, C)])
            sc.add_mark(rng, cfg, {"type": "equal_length", "segments": [[A, D], [B, C]]})
    if kind == "kite":
        sc.equal_lengths += [[(A, B), (A, D)], [(C, B), (C, D)], [(B, E), (E, D)]]
        sc.add_mark(rng, cfg, {"type": "equal_length", "segments": [[A, B], [A, D]]})
        sc.add_mark(rng, cfg, {"type": "equal_length", "segments": [[C, B], [C, D]]})
        sc.right_angle(rng, cfg, E, A, B)
    return sc


# ---------------------------------------------------------------------------
# Tangents
# ---------------------------------------------------------------------------


def tangent_line(rng, namer: Namer, cfg: GenerateConfig) -> Scene:
    variant = wpick(rng, ("external", "line"), (0.7, 0.3))
    sc = Scene("tangent_line", variant)
    O = namer.point(("O",))
    o = (0.0, 0.0)
    sc.points[O] = o
    sc.circles.append((O, 1.0))
    sc.asks.append(("radius", O))
    sc.max_angle_givens = 1

    if variant == "external":
        d = rng.uniform(1.6, 3.0)
        t0 = rng.uniform(0, 2 * math.pi)
        p = (d * math.cos(t0), d * math.sin(t0))
        P = namer.point(("P", "A"))
        sc.points[P] = p
        t1, t2 = G.tangent_points(o, 1.0, p)
        two = chance(rng, 0.6)
        touch = [(namer.aux(("T", "A", "B")), t1)]
        if two:
            touch.append((namer.aux(("S", "B", "Q")), t2))
        for T, t in touch:
            sc.points[T] = t
            sc.add_segment(P, T)
            sc.on_circle.add((T, O))
            sc.exact_angles[canonical_angle((O, T, P))] = 90.0
            if chance(rng, 0.85):
                sc.add_segment(O, T)
                sc.right_angle(rng, cfg, T, O, P)
        Ts = [T for T, _ in touch]
        if two:
            sc.equal_lengths.append([(P, Ts[0]), (P, Ts[1])])
            if chance(rng, 0.6):
                sc.marks.append({"type": "equal_length", "segments": [[P, Ts[0]], [P, Ts[1]]]})
            sc.angles.append((Ts[0], P, Ts[1]))
        if chance(rng, 0.55):
            sc.add_segment(O, P)
            sc.angles.append((Ts[0], P, O))
            if (O, Ts[0]) in sc.segments:
                sc.angles.append((Ts[0], O, P))
        radii = [s for s in sc.segments if s[0] == O and s[1] in Ts]
        if len(radii) == 2:
            sc.equal_lengths.append(radii)
        sc.statements.append(("tangents", P, tuple(Ts), O))
        return sc

    # a line tangent at T, optionally with a chord TA
    phi = rng.uniform(0, 2 * math.pi)
    t = (math.cos(phi), math.sin(phi))
    direction = (-math.sin(phi), math.cos(phi))
    T = namer.point(("T", "A"))
    U = namer.aux(("U", "M", "A"))
    V = namer.aux(("V", "N", "B"))
    sc.points[T] = t
    sc.points[U] = G.sub(t, G.mul(direction, rng.uniform(1.0, 2.0)))
    sc.points[V] = G.add(t, G.mul(direction, rng.uniform(1.0, 2.0)))
    sc.add_segment(U, V)
    sc.on_segment.add((T, frozenset((U, V))))
    sc.on_circle.add((T, O))
    sc.lengths += [(U, T), (T, V)]
    sc.exact_angles[canonical_angle((O, T, V))] = 90.0
    sc.exact_angles[canonical_angle((O, T, U))] = 90.0
    if chance(rng, 0.85):
        sc.add_segment(O, T)
        sc.right_angle(rng, cfg, T, O, V)
    if chance(rng, 0.6):  # chord TA: tangent-chord angle
        delta = math.radians(rng.uniform(60, 150)) * (1 if chance(rng, 0.5) else -1)
        A = namer.point(("A", "C"))
        sc.points[A] = (math.cos(phi + delta), math.sin(phi + delta))
        sc.on_circle.add((A, O))
        sc.add_segment(T, A)
        sc.angles += [(V, T, A), (U, T, A)]
        if chance(rng, 0.5):
            sc.add_segment(O, A)
            sc.angles.append((T, O, A))
            if (O, T) in sc.segments:
                sc.equal_lengths.append([(O, T), (O, A)])
    sc.statements.append(("tangent_line", U, V, T, O))
    return sc


BUILDERS = {
    "triangle_cevian": triangle_cevian,
    "circle_incircle": circle_incircle,
    "circle_circumcircle": circle_circumcircle,
    "quadrilateral_diagonals": quadrilateral_diagonals,
    "tangent_line": tangent_line,
}


# ---------------------------------------------------------------------------
# Canonical frame and validation
# ---------------------------------------------------------------------------


def to_grid(sc: Scene, rng, cfg: GenerateConfig) -> Scene:
    """Randomly rotate/reflect the construction, then fit its bounding box (points and
    circles) to [margin, 100 - margin] in its larger dimension, centered, and round
    points to integers and radii to one decimal."""
    theta = 0.0 if chance(rng, cfg.horizontal_base_prob) else rng.uniform(0, 2 * math.pi)
    flip = chance(rng, 0.5)
    pts = {}
    for k, v in sc.points.items():
        x, y = G.rotate(v, theta)
        pts[k] = (-x, y) if flip else (x, y)
    xs = [p[0] for p in pts.values()] + [pts[c][0] + s * r for c, r in sc.circles for s in (-1, 1)]
    ys = [p[1] for p in pts.values()] + [pts[c][1] + s * r for c, r in sc.circles for s in (-1, 1)]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    scale = (GRID_MAX - GRID_MIN - 2 * cfg.margin) / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    mid = (GRID_MIN + GRID_MAX) / 2
    grid = {
        k: (round((x - cx) * scale + mid), round((y - cy) * scale + mid))
        for k, (x, y) in pts.items()
    }
    circles = [(c, round(r * scale, 1)) for c, r in sc.circles]
    return replace(sc, points=grid, circles=circles)


def validate_scene(sc: Scene, cfg: GenerateConfig) -> list[str]:
    """Problems that make a gridded scene unusable. Empty list means accept."""
    P = sc.points
    problems: list[str] = []
    names = list(P)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if G.dist(P[a], P[b]) < cfg.min_point_separation:
                problems.append(f"points {a} and {b} too close")
    if sc.polygon:
        poly = [P[n] for n in sc.polygon]
        n = len(poly)
        for i in range(n):
            if G.dist(poly[i], poly[(i + 1) % n]) < cfg.min_side_len:
                problems.append("polygon side too short")
        if not G.is_convex(poly):
            problems.append("polygon not convex")
        elif min(G.polygon_angles(poly)) < cfg.min_angle_deg - 2:  # 2 deg of rounding slack
            problems.append("polygon angle too small")
    for a, b in sc.segments:
        if G.dist(P[a], P[b]) < cfg.min_side_len / 2:
            problems.append(f"segment {a}{b} too short")
    for _, r in sc.circles:
        if r < cfg.min_side_len / 2:
            problems.append("circle too small")
    # accidental incidences: a point (nearly) on a segment or circle it is not on by
    # construction would read as a different figure
    for name, p in P.items():
        for a, b in sc.segments:
            if name in (a, b) or (name, frozenset((a, b))) in sc.on_segment:
                continue
            if G.dist_point_segment(p, P[a], P[b]) < cfg.incidence_tol:
                problems.append(f"{name} accidentally near segment {a}{b}")
        for c, r in sc.circles:
            if name == c or (name, c) in sc.on_circle:
                continue
            if abs(G.dist(p, P[c]) - r) < cfg.incidence_tol:
                problems.append(f"{name} accidentally near circle {c}")
    # an unlabeled point must still be visible: a segment endpoint, a circle center, or
    # the meeting point of two drawn things
    centers = {c for c, _ in sc.circles}
    for name in names:
        if not is_unnamed(name):
            continue
        endpoint = any(name in s for s in sc.segments)
        hits = sum(1 for n, _ in sc.on_segment if n == name) + sum(
            1 for n, _ in sc.on_circle if n == name
        )
        if not (endpoint or name in centers or hits >= 2):
            problems.append(f"unlabeled point {name} is invisible")
    return problems
