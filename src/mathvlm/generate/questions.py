"""Templated question text that is consistent with a gridded scene.

Consistency rules:
- The text only names points that are labeled in the figure (never ``_P`` points).
- Every statement in the text is true of the figure (it comes from ``Scene.statements``).
- A value that appears both in the text and on the figure is the same string. Values are
  measured from the final integer coordinates, so the figure is drawn to scale; equal
  quantities (by construction) share one value, and right angles are exactly 90.
- Givens are chosen so rounding cannot visibly contradict itself: no two overlapping
  collinear lengths, no two angles at one vertex, and caps on how many are given.

Only a mathtext-safe LaTeX subset is used, so every question renders (render.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mathvlm.config import GenerateConfig
from mathvlm.generate import primitives as G
from mathvlm.generate.scenes import Ang, Scene, Seg, chance, pick, wpick
from mathvlm.schema import canonical_angle, canonical_segment, is_unnamed

QUAD_WORDS = {
    "parallelogram": "parallelogram",
    "rectangle": "rectangle",
    "rhombus": "rhombus",
    "square": "square",
    "trapezoid": "trapezoid",
    "kite": "kite",
    "convex": "quadrilateral",
}


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _named(*names: str) -> bool:
    return not any(is_unnamed(n) for n in names)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s and s[0].isalpha() else s


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


# ---------------------------------------------------------------------------
# Quantities and values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    kind: str  # "length" | "angle"
    pts: tuple[str, ...]

    @property
    def key(self) -> tuple:
        if self.kind == "length":
            return ("length", canonical_segment(self.pts))
        return ("angle", canonical_angle(self.pts))

    @property
    def named(self) -> bool:
        return _named(*self.pts)

    @property
    def latex(self) -> str:
        return "".join(self.pts) if self.kind == "length" else f"\\angle {''.join(self.pts)}"


class Valuer:
    """Turns measured lengths and angles into the value strings shown to the reader."""

    def __init__(self, sc: Scene, rng: np.random.Generator, qcfg):
        self.sc = sc
        longest = max(G.dist(sc.points[a], sc.points[b]) for a, b in sc.lengths)
        self.unit = rng.uniform(*qcfg.length_scale) / longest
        self.decimals = chance(rng, qcfg.decimal_length_prob)
        self.groups: dict[tuple, int] = {}
        for i, grp in enumerate(sc.equal_lengths):
            for s in grp:
                self.groups[("length", canonical_segment(s))] = i
        for i, grp in enumerate(sc.equal_angles):
            for a in grp:
                self.groups[("angle", canonical_angle(a))] = 1000 + i

    def group(self, q: Quantity) -> int | None:
        return self.groups.get(q.key)

    def _measure(self, q: Quantity) -> float:
        P = self.sc.points
        if q.kind == "length":
            return G.dist(P[q.pts[0]], P[q.pts[1]])
        return G.angle_deg(P[q.pts[0]], P[q.pts[1]], P[q.pts[2]])

    def raw(self, q: Quantity) -> float:
        if q.kind == "angle" and canonical_angle(q.pts) in self.sc.exact_angles:
            return self.sc.exact_angles[canonical_angle(q.pts)]
        g = self.group(q)
        if g is None:
            return self._measure(q)
        members = self.sc.equal_lengths[g] if g < 1000 else self.sc.equal_angles[g - 1000]
        kind = "length" if g < 1000 else "angle"
        return sum(self._measure(Quantity(kind, tuple(m))) for m in members) / len(members)

    def number(self, q: Quantity) -> str:
        if q.kind == "angle":
            return str(round(self.raw(q)))
        v = self.raw(q) * self.unit
        if self.decimals:
            s = f"{round(v, 1):.1f}"
            return s[:-2] if s.endswith(".0") else s
        return str(max(1, round(v)))

    def text(self, q: Quantity) -> str:
        return self.number(q) + ("^\\circ" if q.kind == "angle" else "")


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


def _phrases(sc: Scene, rng) -> tuple[str | None, list[str]]:
    """Return (context, facts): an optional leading "In ..." phrase and fact clauses."""
    context: str | None = None
    facts: list[str] = []
    # "circle I is inscribed in triangle ABC" already introduces the triangle, so it
    # replaces the "In triangle ABC" context and comes first.
    has_incircle = any(st[0] == "incircle" for st in sc.statements)
    order = {"incircle": 0, "touches": 1}  # "it touches ..." must follow the incircle
    statements = sorted(sc.statements, key=lambda st: order.get(st[0], 2))
    touches = [st for st in statements if st[0] == "touches"]
    for st in statements:
        k = st[0]
        if k == "triangle" and has_incircle:
            continue
        if k == "triangle":
            tri, kind = "".join(st[1]), st[2]
            if kind == "right" and chance(rng, 0.5):
                context = f"In right triangle ${tri}$"
            else:
                context = pick(rng, [f"In $\\triangle {tri}$", f"In triangle ${tri}$"])
        elif k == "right_angle":
            a, v, b = st[1]
            facts.append(pick(rng, [f"$\\angle {a}{v}{b} = 90^\\circ$", f"${v}{a} \\perp {v}{b}$"]))
        elif k == "isosceles":
            apex, (a, b) = st[1], st[2]
            facts.append(f"${apex}{a} = {apex}{b}$")
        elif k == "altitude":
            V, F, P, Q = st[1:]
            if _named(F):
                facts.append(pick(rng, [
                    f"${V}{F} \\perp {P}{Q}$ at ${F}$",
                    f"${V}{F}$ is the altitude to ${P}{Q}$",
                    f"${V}{F} \\perp {P}{Q}$",
                ]))  # fmt: skip
            else:
                facts.append(f"an altitude is drawn from ${V}$ to ${P}{Q}$")
        elif k == "median":
            V, F, P, Q = st[1:]
            if _named(F):
                facts.append(
                    pick(rng, [f"${F}$ is the midpoint of ${P}{Q}$", f"${V}{F}$ is a median"])
                )
            else:
                facts.append(f"the median from ${V}$ to ${P}{Q}$ is drawn")
        elif k == "bisector":
            V, F, P, Q = st[1:]
            if _named(F):
                facts.append(pick(rng, [
                    f"${V}{F}$ bisects $\\angle {P}{V}{Q}$",
                    f"${V}{F}$ is the bisector of $\\angle {P}{V}{Q}$",
                ]))  # fmt: skip
            else:
                facts.append(f"the bisector of $\\angle {P}{V}{Q}$ is drawn")
        elif k == "incircle":
            I, tri = st[1], "".join(st[2])
            facts.append(pick(rng, [
                f"circle ${I}$ is inscribed in $\\triangle {tri}$",
                f"circle ${I}$ is the incircle of $\\triangle {tri}$",
            ]))  # fmt: skip
        elif k == "touches":
            if st is touches[0]:  # one clause for all tangency points
                sides = _join([f"${P}{Q}$" for _, _, _, P, Q in touches])
                pts = _join([f"${T}$" for _, _, T, _, _ in touches])
                facts.append(
                    f"it touches {sides} at {pts}" + (", respectively" if len(touches) > 1 else "")
                )
        elif k == "circumcircle":
            O, (a, b, c) = st[1], st[2]
            facts.append(pick(rng, [
                f"$\\triangle {a}{b}{c}$ is inscribed in circle ${O}$",
                f"points ${a}$, ${b}$, and ${c}$ lie on circle ${O}$",
            ]))  # fmt: skip
        elif k == "diameter":
            P, Q, O = st[1:]
            facts.append(f"${P}{Q}$ is a diameter of circle ${O}$")
        elif k == "quad":
            kind, quad = st[1], "".join(st[2])
            word = QUAD_WORDS[kind]
            a, b, c, d = st[2]
            if chance(rng, 0.5):
                context = f"In {word} ${quad}$"
            else:
                facts.append(f"${quad}$ is {_article(word)} {word}")
            if kind == "trapezoid":
                facts.append(f"${a}{b} \\parallel {d}{c}$")
            elif kind == "kite":
                facts.append(f"${a}{b} = {a}{d}$ and ${c}{b} = {c}{d}$")
        elif k == "diagonals":
            (a, c), (b, d), E = st[1:]
            if _named(E):
                facts.append(pick(rng, [
                    f"diagonals ${a}{c}$ and ${b}{d}$ intersect at ${E}$",
                    f"the diagonals meet at ${E}$",
                ]))  # fmt: skip
            else:
                facts.append(f"diagonals ${a}{c}$ and ${b}{d}$ are drawn")
        elif k == "tangents":
            P, Ts, O = st[1:]
            if len(Ts) == 1 and _named(Ts[0]):
                facts.append(f"${P}{Ts[0]}$ is tangent to circle ${O}$ at ${Ts[0]}$")
            elif len(Ts) == 2 and _named(*Ts):
                facts.append(
                    f"${P}{Ts[0]}$ and ${P}{Ts[1]}$ are tangent to circle ${O}$ at ${Ts[0]}$ and ${Ts[1]}$"
                )
            elif len(Ts) == 1:
                facts.append(f"a tangent from ${P}$ touches circle ${O}$")
            else:
                facts.append(f"two tangents are drawn from ${P}$ to circle ${O}$")
        elif k == "tangent_line":
            U, V, T, O = st[1:]
            if _named(U, V):
                facts.append(f"line ${U}{V}$ is tangent to circle ${O}$ at ${T}$")
            else:
                facts.append(f"a line is tangent to circle ${O}$ at ${T}$")
        else:
            raise ValueError(f"unknown statement {k!r}")
    return context, facts


# ---------------------------------------------------------------------------
# Asks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ask:
    text: str  # lowercase-first, without the final period
    display: bool = False  # ends in display math, which carries its own period


def _family_ask(sc: Scene, ask: tuple, rng) -> Ask | None:
    kind = ask[0]
    if kind in ("area", "perimeter"):
        shape, pts = ask[1], ask[2]
        if not _named(*pts):
            return None
        name = "".join(pts)
        if shape == "triangle":
            return Ask(f"find the {kind} of $\\triangle {name}$")
        word = QUAD_WORDS[sc.kind]
        return Ask(
            pick(rng, [f"find the {kind} of ${name}$", f"find the {kind} of {word} ${name}$"])
        )
    if kind == "radius":
        return Ask(
            pick(rng, [f"find the radius of circle ${ask[1]}$", "find the radius of the circle"])
        )
    if kind == "prove":
        latex, display = ask[1], ask[2]
        if display:
            return Ask(f"show that\n\n$${latex}.$$", display=True)
        return Ask(pick(rng, [f"prove that ${latex}$", f"show that ${latex}$"]))
    raise ValueError(f"unknown ask {kind!r}")


def _named_ask(q: Quantity, rng) -> Ask:
    if q.kind == "length":
        return Ask(pick(rng, [f"find the length of ${q.latex}$", f"find ${q.latex}$"]))
    return Ask(
        pick(rng, [f"find ${q.latex}$", f"find the measure of ${q.latex}$", f"find $m{q.latex}$"])
    )


def _expr(a: int, b: int, var: str) -> str:
    head = f"{a}{var}"
    return head if b == 0 else f"{head}+{b}" if b > 0 else f"{head}-{-b}"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class Composed:
    text: str
    labels: list[dict]
    meta: dict = field(default_factory=dict)


def _overlap(s1: Seg, s2: Seg, P) -> bool:
    """Collinear segments whose extents overlap (e.g. AM inside AB)."""
    a, b, c, d = P[s1[0]], P[s1[1]], P[s2[0]], P[s2[1]]
    if G.dist_point_line(c, a, b) > 1.5 or G.dist_point_line(d, a, b) > 1.5:
        return False
    t = sorted((G.project_param(c, a, b), G.project_param(d, a, b)))
    return min(1.0, t[1]) - max(0.0, t[0]) > 0.02


def _sectors_overlap(a1: Ang, a2: Ang, P) -> bool:
    """Do two angles at the same vertex overlap, rather than merely share a ray?
    (The two halves of a bisected angle do not; angle ABC and its part ABD do.)"""
    v = P[a1[1]]

    def sector(a: Ang) -> tuple[float, float]:
        t1 = math.atan2(P[a[0]][1] - v[1], P[a[0]][0] - v[0])
        t2 = math.atan2(P[a[2]][1] - v[1], P[a[2]][0] - v[0])
        return t1, (t2 - t1 + math.pi) % (2 * math.pi) - math.pi  # start, signed sweep

    def inside(t: float, sec: tuple[float, float]) -> bool:
        start, sweep = sec
        if abs(sweep) < 1e-9:
            return False
        rel = ((t - start + math.pi) % (2 * math.pi) - math.pi) / sweep
        return 0.0 < rel < 1.0

    s1, s2 = sector(a1), sector(a2)
    probes = (0.15, 0.5, 0.85)
    return any(inside(s2[0] + s2[1] * f, s1) for f in probes) or any(
        inside(s1[0] + s1[1] * f, s2) for f in probes
    )


def compose(sc: Scene, rng: np.random.Generator, cfg: GenerateConfig) -> Composed:
    qc = cfg.questions
    val = Valuer(sc, rng, qc)

    pool: list[Quantity] = []
    seen: set[tuple] = set()
    for s in sc.lengths:
        q = Quantity("length", tuple(s))
        if q.key not in seen:
            seen.add(q.key)
            pool.append(q)
    for a in sc.angles:
        q = Quantity("angle", tuple(a))
        if q.key not in seen and canonical_angle(a) not in sc.exact_angles:
            seen.add(q.key)
            pool.append(q)
    pool = [pool[int(i)] for i in rng.permutation(len(pool))]

    # --- givens --------------------------------------------------------------
    n_givens = int(rng.integers(1, qc.max_givens + 1))
    givens: list[Quantity] = []

    def conflicts(q: Quantity, chosen: list[Quantity]) -> bool:
        if val.group(q) is not None and any(val.group(c) == val.group(q) for c in chosen):
            return True
        same = [c for c in chosen if c.kind == q.kind]
        if q.kind == "length":
            if len(same) >= qc.max_length_givens:
                return True
            return any(_overlap(q.pts, c.pts, sc.points) for c in same)
        if len(same) >= sc.max_angle_givens:
            return True
        return any(c.pts[1] == q.pts[1] for c in same)

    for q in pool:
        if len(givens) >= n_givens:
            break
        if not conflicts(q, givens):
            givens.append(q)

    labels: list[dict] = []
    angle_labels: list[Ang] = []

    def can_label(q: Quantity) -> bool:
        if q.kind == "length":
            # A value written on a segment with a point in its middle would read as
            # labeling one of the pieces, so only piece-free segments get figure labels.
            a, b = (sc.points[n] for n in q.pts)
            return not any(
                G.dist_point_segment(p, a, b) < 1.5 and 0.02 < G.project_param(p, a, b) < 0.98
                for n, p in sc.points.items()
                if n not in q.pts
            )
        return not any(
            a[1] == q.pts[1] and _sectors_overlap(a, q.pts, sc.points) for a in angle_labels
        )

    def label(q: Quantity, text: str) -> bool:
        if not can_label(q):
            return False
        if q.kind == "angle":
            angle_labels.append(q.pts)
            labels.append({"angle": list(q.pts), "text": text})
        else:
            labels.append({"on": list(q.pts), "text": text})
        return True

    # --- unknown -------------------------------------------------------------
    asks: list[Ask] = []
    unknown_meta: dict = {}
    must_show: set[tuple] = set()  # givens the question cannot do without
    # Asking for the radius is trivial once a radius is given.
    centers = {c for c, _ in sc.circles}
    given_radius = {
        c
        for q in givens
        if q.kind == "length"
        for c in centers
        if c in q.pts and (next(n for n in q.pts if n != c), c) in sc.on_circle
    }
    usable = [x for x in sc.asks if not (x[0] == "radius" and x[1] in given_radius)]
    family_asks = [a for a in (_family_ask(sc, x, rng) for x in usable) if a is not None]
    rest = [q for q in pool if q.key not in {g.key for g in givens}]
    if family_asks and (not rest or chance(rng, qc.family_ask_prob)):
        asks.append(pick(rng, family_asks))
        unknown_meta = {"type": "family", "ask": asks[0].text}
    elif any(q.named or can_label(q) for q in rest):
        q = next(q for q in rest if q.named or can_label(q))
        variable = not q.named or chance(rng, qc.variable_unknown_prob)
        var = wpick(rng, ("x", "y"), (0.8, 0.2))
        if variable and label(q, var if q.kind == "length" else f"{var}^\\circ"):
            unknown_meta = {
                "type": "variable",
                "quantity": list(q.pts),
                "label": labels[-1]["text"],
            }
            # Algebraic variant: the unknown's label becomes an expression equal to a
            # partner that is equal by construction, and the partner is shown as a given.
            partner = None
            g = val.group(q)
            if (
                g is not None
                and chance(rng, qc.algebraic_prob)
                and not (q.kind == "length" and val.decimals)
            ):
                members = sc.equal_lengths[g] if g < 1000 else sc.equal_angles[g - 1000]
                kind = "length" if g < 1000 else "angle"
                partner = next(
                    (
                        Quantity(kind, tuple(m))
                        for m in members
                        if Quantity(kind, tuple(m)).key != q.key
                    ),
                    None,
                )
            if (
                partner is not None
                and int(val.number(partner)) >= 4
                and (partner.named or can_label(partner))
            ):
                v = int(val.number(partner))
                a = pick(rng, [2, 3])
                x0 = int(rng.integers(1, max(2, v // a)))
                expr = _expr(a, v - a * x0, var)
                labels[-1]["text"] = expr if q.kind == "length" else f"({expr})^\\circ"
                givens = [gv for gv in givens if gv.key != partner.key] + [partner]
                must_show.add(partner.key)
                unknown_meta = {"type": "algebraic", "quantity": list(q.pts),
                                "label": labels[-1]["text"], "x": x0}  # fmt: skip
            asks.append(Ask(pick(rng, [f"find ${var}$", f"find the value of ${var}$"])))
        if not asks and q.named:
            asks.append(_named_ask(q, rng))
            unknown_meta = {"type": "named", "quantity": list(q.pts)}
    if not asks:  # nothing askable left; fall back to any family ask
        asks.append(family_asks[0] if family_asks else Ask("describe the figure"))
        unknown_meta = {"type": "fallback"}

    # --- where each given appears -------------------------------------------
    given_text: list[str] = []
    givens_meta = []
    for q in givens:
        in_fig = chance(rng, qc.figure_label_prob)
        in_txt = q.named and chance(rng, qc.text_given_prob)
        if not (in_fig or in_txt):
            in_txt = q.named
            in_fig = not q.named
        if in_fig and not label(q, val.text(q)):
            in_fig = False
            in_txt = q.named
        if q.key in must_show and not (in_fig or in_txt):
            raise AssertionError("a required given could not be shown")  # guarded above
        if in_txt:
            given_text.append(f"${q.latex} = {val.text(q)}$")
        if in_fig or in_txt:
            givens_meta.append(
                {
                    "quantity": list(q.pts),
                    "value": val.text(q),
                    "in_text": in_txt,
                    "in_figure": in_fig,
                }
            )

    # --- second part ----------------------------------------------------------
    two_part = False
    if chance(rng, qc.two_part_prob):
        extra = [a for a in family_asks if a.text != asks[0].text and not a.display]
        if extra and not asks[0].display:
            asks.append(pick(rng, extra))
            two_part = True

    # --- assemble -------------------------------------------------------------
    context, facts = _phrases(sc, rng)
    prefix = wpick(rng, ("", "In the figure, ", "As shown in the figure, "), (0.5, 0.25, 0.25))
    form = (
        "inline"
        if (two_part or not given_text or (context and not facts))
        else pick(rng, ["if", "inline", "given"])
    )
    clauses = facts + (given_text if form == "inline" else [])
    if context:
        lead = context + ("," if clauses else "")
        lead = (prefix + lead[0].lower() + lead[1:]) if prefix else lead
        first = f"{lead} {_join(clauses)}".rstrip() + "."
    else:
        first = _cap(prefix + _join(clauses)) + "."

    def ask_sentence(a: Ask) -> str:
        return _cap(a.text) + ("" if a.display else ".")

    if two_part:
        text = first + f"\n\n(a) {ask_sentence(asks[0])}\n\n(b) {ask_sentence(asks[1])}"
    elif form == "if":
        text = f"{first} If {_join(given_text)}, {asks[0].text}" + ("" if asks[0].display else ".")
    elif form == "given":
        text = f"{first} Given that {_join(given_text)}, {asks[0].text}" + (
            "" if asks[0].display else "."
        )
    else:
        text = f"{first} {ask_sentence(asks[0])}"

    meta = {"givens": givens_meta, "unknown": unknown_meta, "form": form, "two_part": two_part}
    return Composed(text=text, labels=labels, meta=meta)


# ---------------------------------------------------------------------------
# Questions without a figure
# ---------------------------------------------------------------------------


def _poly(terms: list[tuple[int, str]]) -> str:
    """Format sum of coef*monomial, e.g. [(1,'x^2'), (-5,'x'), (6,'')] -> 'x^2 - 5x + 6'."""
    out = ""
    for c, mono in terms:
        if c == 0:
            continue
        mag = abs(c)
        body = mono if (mag == 1 and mono) else f"{mag}{mono}"
        if not out:
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out or "0"


def text_only_question(rng: np.random.Generator) -> tuple[str, str]:
    """A question with no figure. Returns (text, kind)."""

    def nz(lo: int, hi: int) -> int:
        while True:
            v = int(rng.integers(lo, hi + 1))
            if v != 0:
                return v

    kind = pick(rng, ["linear", "quadratic", "system", "fraction", "evaluate", "word"])
    display = chance(rng, 0.4)
    if kind == "linear":
        x0, a, b = int(rng.integers(-9, 10)), nz(2, 9), nz(-20, 20)
        eq = f"{_poly([(a, 'x'), (b, '')])} = {a * x0 + b}"
        lead = pick(rng, ["Solve for $x$", "Find $x$ if", "Solve"])
    elif kind == "quadratic":
        r1, r2 = int(rng.integers(-9, 10)), int(rng.integers(-9, 10))
        eq = f"{_poly([(1, 'x^2'), (-(r1 + r2), 'x'), (r1 * r2, '')])} = 0"
        lead = pick(rng, ["Solve", "Find the roots of", "Solve for $x$"])
    elif kind == "system":
        x0, y0 = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
        a, b, c, d = nz(1, 5), nz(-5, 5), nz(1, 5), nz(-5, 5)
        while a * d - b * c == 0:
            d = nz(-5, 5)
        e1 = f"{_poly([(a, 'x'), (b, 'y')])} = {a * x0 + b * y0}"
        e2 = f"{_poly([(c, 'x'), (d, 'y')])} = {c * x0 + d * y0}"
        if display:
            return f"Solve the system of equations:\n\n$${e1}$$\n\n$${e2}$$", kind
        return f"Solve the system ${e1}$ and ${e2}$.", kind
    elif kind == "fraction":
        a, b, c, d = (
            int(rng.integers(1, 10)),
            int(rng.integers(2, 13)),
            int(rng.integers(1, 10)),
            int(rng.integers(2, 13)),
        )
        op = pick(rng, ["+", "-", "\\cdot"])
        eq = f"\\frac{{{a}}}{{{b}}} {op} \\frac{{{c}}}{{{d}}}"
        lead = pick(rng, ["Simplify", "Compute", "Evaluate"])
    elif kind == "evaluate":
        a, b = int(rng.integers(2, 13)), int(rng.integers(2, 6))
        eq = pick(
            rng,
            [f"\\sqrt{{{a * a}}} + {b}^{{2}}", f"{a}^{{2}} - {b}^{{3}}", f"\\sqrt{{{a * a * b}}}"],
        )
        lead = pick(rng, ["Evaluate", "Simplify", "Compute"])
    else:
        L, W = int(rng.integers(3, 25)), int(rng.integers(2, 20))
        unit = pick(rng, ["cm", "m", "in"])
        what = pick(rng, ["perimeter", "area"])
        return f"A rectangle has length ${L}$ {unit} and width ${W}$ {unit}. Find its {what}.", kind
    if display:
        return f"{lead}:\n\n$${eq}$$", kind
    return f"{lead} ${eq}$.", kind
