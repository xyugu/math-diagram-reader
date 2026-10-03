"""Conventions of real textbook figures that the base scene families lack.

Measured against PGDP5K (README.md, Results): over half the points in real figures are
unlabeled line ends and intersections, rays end in arrowheads, many images show the
figure alone (no question text), and angles are often numbered ``1, 2, 3`` around an
intersection. These functions add those to a generated sample:

- ``add_rays`` (float frame, before ``to_grid``): extends drawn segments past an endpoint
  to a new unlabeled tip with an arrowhead, as long as the ray crosses nothing. Done before
  the grid fit, so the canonical frame (README.md, design invariant 5) includes the rays.
- ``number_angles`` (grid diagram): labels the sectors around a vertex where three or
  more rays meet ``1, 2, ...``, at vertices that carry no other angle label or arc.
- ``hide_names`` (grid diagram): turns labeled points into unlabeled ``_P`` points.
- ``densify_labels`` (grid diagram): value labels as dense as real figures put them:
  pieces of a segment split by an interior point, several sectors around one vertex,
  arcs of a circle, sometimes with a unit (``6\\text{ cm}``). Values are measured from
  the figure, so they agree with it.

The last three apply to diagram-only pages only, where no question text mentions a point
or an angle. All draws come from the sample's own realism RNG stream (build.py), so the
scene, text, style and augmentation draws are unaffected.
"""

from __future__ import annotations

import math

import numpy as np

from mathvlm.config import RealismConfig
from mathvlm.generate import primitives as G
from mathvlm.generate.scenes import Namer, Scene, chance
from mathvlm.schema import UNNAMED_PREFIX, is_unnamed, rename_points

Vec = G.Vec


# ---------------------------------------------------------------------------
# Rays
# ---------------------------------------------------------------------------


def _segments_cross(a: Vec, b: Vec, c: Vec, d: Vec) -> bool:
    """Proper intersection of segments ab and cd (touching at an endpoint is not)."""

    def orient(p, q, r):
        return G.cross(G.sub(q, p), G.sub(r, p))

    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _ray_is_clear(sc: Scene, base: str, tail: str, tip: Vec, span: float) -> bool:
    P = sc.points
    b = P[base]
    # leaves no other drawn thing: no crossing, no running into a circle
    for s, t in sc.segments:
        if base in (s, t) and tail in (s, t):
            continue
        if _segments_cross(b, tip, P[s], P[t]):
            return False
        if base in (s, t):  # another segment from the same point: keep a clear angle
            other = P[t if s == base else s]
            if G.angle_deg(other, b, tip) < 15:
                return False
    for c, r in sc.circles:
        for k in range(1, 21):
            q = G.lerp(b, tip, 0.05 + 0.95 * k / 20)
            if G.dist(q, P[c]) < r * 0.98:
                return False
        if abs(G.dist(tip, P[c]) - r) < 0.06 * span:
            return False
    return all(G.dist(tip, p) >= 0.12 * span for p in P.values())


def add_rays(sc: Scene, rng: np.random.Generator, namer: Namer, cfg: RealismConfig) -> int:
    """Extend up to ``cfg.max_rays`` drawn segments past one endpoint into rays. Returns
    how many were added. The drawn segment becomes tail-to-tip (one maximal segment), the
    old endpoint lies on it, and the tip gets an arrowhead."""
    if not sc.segments or not chance(rng, cfg.ray_prob):
        return 0
    pts = list(sc.points.values()) + [
        (sc.points[c][0] + s * r, sc.points[c][1]) for c, r in sc.circles for s in (-1, 1)
    ]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    want = int(rng.integers(1, cfg.max_rays + 1))
    candidates = [(a, b) for a, b in sc.segments] + [(b, a) for a, b in sc.segments]
    order = rng.permutation(len(candidates))
    added = 0
    for i in order:
        if added >= want:
            break
        tail, base = candidates[int(i)]
        if (tail, base) not in sc.segments and (base, tail) not in sc.segments:
            continue  # this segment was already extended
        if is_unnamed(base) and any(m.get("at") == base for m in sc.marks):
            continue
        u = G.sub(sc.points[base], sc.points[tail])
        n = G.norm(u)
        if n < 1e-9:
            continue
        length = float(rng.uniform(*cfg.ray_length)) * span
        tip = G.add(sc.points[base], G.mul(u, length / n))
        if not _ray_is_clear(sc, base, tail, tip, span):
            continue
        name = namer.unnamed()
        sc.points[name] = tip
        old = frozenset((tail, base))
        sc.segments = [(tail, name) if frozenset(s) == old else s for s in sc.segments]
        new = frozenset((tail, name))
        sc.on_segment = {(p, new if seg == old else seg) for p, seg in sc.on_segment}
        sc.on_segment.add((base, new))
        sc.marks.append({"type": "arrow", "at": name, "from": base})
        added += 1
    return added


# ---------------------------------------------------------------------------
# Numbered angles and hidden names (diagram-only pages)
# ---------------------------------------------------------------------------


def _rays_at(d: dict, v: str, tol: float = 1.0) -> list[tuple[float, str]]:
    """(direction, far point) for every drawn ray leaving v."""
    P = {k: (float(x), float(y)) for k, (x, y) in d["points"].items()}
    out = []
    for a, b in d["segments"]:
        if v in (a, b):
            far = b if a == v else a
            out.append((math.atan2(P[far][1] - P[v][1], P[far][0] - P[v][0]), far))
        elif G.dist_point_segment(P[v], P[a], P[b]) <= tol and v not in (a, b):
            for far in (a, b):
                if G.dist(P[far], P[v]) > tol:
                    out.append((math.atan2(P[far][1] - P[v][1], P[far][0] - P[v][0]), far))
    return sorted(out)


def number_angles(d: dict, rng: np.random.Generator, cfg: RealismConfig) -> int:
    """Number the sectors around one vertex with three or more rays ``1, 2, ...``.
    Returns how many angle labels were added."""
    if not chance(rng, cfg.numbered_angles_prob):
        return 0
    busy = {lb["angle"][1] for lb in d["labels"] if "angle" in lb}
    busy |= {a[1] for m in d["marks"] if m["type"] == "equal_angle" for a in m["angles"]}
    candidates = []
    for v in sorted(d["points"]):
        if v in busy:
            continue
        rays = _rays_at(d, v)
        if len(rays) < 3:
            continue
        sectors = []
        for k in range(len(rays)):
            (t1, p1), (t2, p2) = rays[k], rays[(k + 1) % len(rays)]
            width = math.degrees((t2 - t1) % (2 * math.pi))
            if cfg.numbered_min_deg <= width <= 179 and p1 != p2:
                sectors.append((p1, v, p2))
        if len(sectors) >= 2:
            candidates.append(sectors)
    if not candidates:
        return 0
    sectors = candidates[int(rng.integers(len(candidates)))]
    start = int(rng.integers(len(sectors)))
    sectors = sectors[start:] + sectors[:start]
    for k, (p1, v, p2) in enumerate(sectors):
        d["labels"].append({"angle": [p1, v, p2], "text": str(k + 1)})
    return len(sectors)


def hide_names(d: dict, rng: np.random.Generator, cfg: RealismConfig) -> str:
    """Hide all, some or none of the point labels (by ``cfg.hide_names`` weights).
    Hidden points get temporary ``_P`` names; build.py renumbers them in reading order."""
    modes = list(cfg.hide_names)
    w = np.asarray([cfg.hide_names[m] for m in modes], dtype=float)
    mode = modes[int(rng.choice(len(modes), p=w / w.sum()))]
    named = sorted(n for n in d["points"] if not is_unnamed(n))
    if mode == "all":
        hidden = named
    elif mode == "some":
        hidden = [n for n in named if rng.random() < cfg.hide_some_prob]
    else:
        hidden = []
    mapping = {n: f"{UNNAMED_PREFIX}{1000 + i}" for i, n in enumerate(hidden)}
    d.update(rename_points(d, lambda n: mapping.get(n, n)))
    return mode


# ---------------------------------------------------------------------------
# Dense value labels (diagram-only pages)
# ---------------------------------------------------------------------------


def _label_targets(d: dict) -> set:
    out = set()
    for lb in d["labels"]:
        if "on" in lb:
            out.add(("on", frozenset(lb["on"])))
        elif "angle" in lb:
            a = lb["angle"]
            out.add(("angle", a[1], frozenset((a[0], a[2]))))
        elif "arc" in lb:
            out.add(("arc", frozenset(lb["arc"])))
    return out


def _pieces(d: dict, tol: float = 1.0) -> list[tuple[str, str]]:
    """Every drawn segment split at the points on it: consecutive pairs along it."""
    P = {k: (float(x), float(y)) for k, (x, y) in d["points"].items()}
    out = []
    for a, b in d["segments"]:
        on = [n for n in P if n not in (a, b) and G.dist_point_segment(P[n], P[a], P[b]) <= tol]
        chain = [a, *sorted(on, key=lambda n: G.dist(P[a], P[n])), b]
        out += list(zip(chain, chain[1:], strict=False))
    return out


def densify_labels(d: dict, rng: np.random.Generator, cfg: RealismConfig) -> dict[str, int]:
    """Add measured value labels to a diagram-only page. Returns counts by kind."""
    added = {"lengths": 0, "angles": 0, "arcs": 0}
    if not chance(rng, cfg.dense_labels_prob):
        return added
    P = {k: (float(x), float(y)) for k, (x, y) in d["points"].items()}
    taken = _label_targets(d)
    tips = {m["at"] for m in d["marks"] if m["type"] == "arrow"}

    # lengths, in one consistent unit scale (the longest drawn piece maps into 6..24)
    pieces = [p for p in _pieces(d) if not (set(p) & tips) and G.dist(P[p[0]], P[p[1]]) >= 8]
    if pieces:
        scale = float(rng.uniform(6, 24)) / max(G.dist(P[a], P[b]) for a, b in pieces)
        decimals = 1 if chance(rng, 0.2) else 0
        unit = cfg.units[int(rng.integers(len(cfg.units)))] if chance(rng, cfg.units_prob) else ""
        want = int(rng.integers(1, cfg.max_extra_lengths + 1))
        for i in rng.permutation(len(pieces)):
            if added["lengths"] >= want:
                break
            a, b = pieces[int(i)]
            if ("on", frozenset((a, b))) in taken:
                continue
            value = round(G.dist(P[a], P[b]) * scale, decimals)
            text = f"{value:g}" + (f"\\text{{ {unit}}}" if unit else "")
            d["labels"].append({"on": [a, b], "text": text})
            taken.add(("on", frozenset((a, b))))
            added["lengths"] += 1

    # every sector around one free vertex, in degrees (they add up as the figure does)
    busy = {lb["angle"][1] for lb in d["labels"] if "angle" in lb}
    busy |= {a[1] for m in d["marks"] if m["type"] == "equal_angle" for a in m["angles"]}
    busy |= {m["at"] for m in d["marks"] if m["type"] == "right_angle"}
    vertices = [v for v in sorted(d["points"]) if v not in busy and len(_rays_at(d, v)) >= 3]
    if vertices and chance(rng, 0.7):
        v = vertices[int(rng.integers(len(vertices)))]
        rays = _rays_at(d, v)
        for k in range(len(rays)):
            (t1, p1), (t2, p2) = rays[k], rays[(k + 1) % len(rays)]
            width = math.degrees((t2 - t1) % (2 * math.pi))
            if not (cfg.numbered_min_deg <= width <= 179) or p1 == p2:
                continue
            text = "x^\\circ" if chance(rng, 0.15) else f"{round(width)}^\\circ"
            d["labels"].append({"angle": [p1, v, p2], "text": text})
            added["angles"] += 1

    # one arc per circle, between neighbouring points on it, with its measure
    for c in d["circles"]:
        if not chance(rng, cfg.arc_label_prob):
            continue
        o, r = P[c["center"]], float(c["radius"])
        on = sorted((n for n in P if n != c["center"] and abs(G.dist(P[n], o) - r) <= 1.5),
                    key=lambda n: math.atan2(P[n][1] - o[1], P[n][0] - o[0]))  # fmt: skip
        arcs = []
        for k in range(len(on)):
            a, b = on[k], on[(k + 1) % len(on)]
            if a == b or ("arc", frozenset((a, b))) in taken:
                continue
            deg = G.angle_deg(P[a], o, P[b])
            if 30 <= deg <= 170:
                arcs.append((a, b, deg))
        if arcs:
            a, b, deg = arcs[int(rng.integers(len(arcs)))]
            d["labels"].append({"arc": [a, b], "text": f"{round(deg)}^\\circ"})
            taken.add(("arc", frozenset((a, b))))
            added["arcs"] += 1
    return added
