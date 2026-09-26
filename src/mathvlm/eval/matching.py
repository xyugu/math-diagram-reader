"""Point correspondence between a predicted and a ground-truth diagram.

The schema fixes coordinates only up to a similarity transform, so predicted points are
aligned to the GT frame before they are compared.

Alignment (``align``) fits ``q = s R p + t`` taking prediction to GT: translation and
uniform scale by default, rotation and reflection only if the config allows them (off,
because the image's orientation pins them; README.md, design invariant 6). The fit
needs a correspondence and the correspondence needs a fit, so they are found together.
Starting transforms come from two sources: three global guesses (identity; bounding box
onto bounding box; centroid and spread), and two-point hypotheses, where every pair of
GT points matched to every ordered pair of predicted points fixes a transform (as in
RANSAC, but exhaustive, so deterministic). The global guesses fail when the prediction
has extra or missing points; the hypotheses do not. Each hypothesis gets a cheap score,
and the best few plus the global guesses are refined ICP-style: alternate Hungarian
matching and a least-squares fit on the inliers until the matching stops changing.

The refined transform with the lowest *robust* cost wins (``_cost``): each point
contributes its squared distance, capped at ``eval.align_inlier_dist`` squared, and an
unmatched point pays the cap. So three points placed exactly and one far off beat four
points all a little off: the alignment follows the consistent majority instead of
spreading one point's error over the whole figure. Names play no part, so a prediction
with every letter wrong still aligns.

Angles are compared by their rays (``ray_keys``): an angle named through any point on
each of its rays is the same angle.

Two correspondences, both under that one alignment:

- ``match_geometric``: all points by Hungarian assignment on aligned coordinates,
  ignoring names.
- ``match_strict``: labeled points by name; ``_P`` points, which have no name to go by,
  by Hungarian assignment among themselves.

A pair farther apart than ``eval.match_max_dist`` (grid units, after alignment) is not
a match.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from mathvlm.config import EvalConfig, get_config
from mathvlm.schema import (
    UNNAMED_PREFIX,
    Diagram,
    diagram_to_obj,
    is_unnamed,
    name_sort_key,
    rename_points,
)

Points = Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class Similarity:
    """``q = scale * rotation @ p + translation``. ``rotation`` is the identity unless
    rotation is allowed; with reflection allowed it may be improper."""

    scale: float = 1.0
    rotation: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    translation: tuple[float, float] = (0.0, 0.0)

    def apply(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return self.scale * xy @ np.asarray(self.rotation).T + np.asarray(self.translation)


IDENTITY = Similarity()


def fit_similarity(src, dst, rotation: bool = False, reflection: bool = False) -> Similarity:
    """Least-squares similarity taking ``src`` onto ``dst`` (Umeyama, 1991).

    Without rotation, R = I and the scale is clamped at 0 (a negative scale would be a
    180-degree rotation). One point, or all ``src`` points coincident: translation only.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if len(src) == 0:
        return IDENTITY
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    var = float((a * a).sum())
    if var < 1e-12:
        return Similarity(translation=tuple(float(v) for v in mu_d - mu_s))
    if rotation:
        U, S, Vt = np.linalg.svd(b.T @ a)
        D = np.eye(2)
        if not reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
            D[1, 1] = -1.0
        R = U @ D @ Vt
        s = float(np.trace(np.diag(S) @ D)) / var
    else:
        R = np.eye(2)
        s = max(float((a * b).sum()) / var, 0.0)
    t = mu_d - s * R @ mu_s
    return Similarity(
        s,
        ((float(R[0, 0]), float(R[0, 1])), (float(R[1, 0]), float(R[1, 1]))),
        (float(t[0]), float(t[1])),
    )


def assign(gt_xy, pred_xy, max_dist: float) -> list[tuple[int, int, float]]:
    """Hungarian assignment of GT to predicted points, as ``(gt_i, pred_j, dist)``.

    Maximizes the number of pairs within ``max_dist`` first, then minimizes their total
    distance; pairs farther than ``max_dist`` are dropped. ``max_dist=inf`` keeps every
    pair of the full assignment.
    """
    g = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    p = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    if len(g) == 0 or len(p) == 0:
        return []
    d = np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2)
    if math.isfinite(max_dist):
        # Any pair over the threshold costs the same flat penalty, larger than any total of
        # within-threshold pairs: it will be dropped, so how far it is must not steer
        # the assignment of the others.
        penalty = (len(g) + len(p) + 1) * max_dist
        cost = np.where(d <= max_dist, d, penalty)
    else:
        cost = d
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(i), int(j), float(d[i, j]))
        for i, j in zip(rows, cols, strict=True)
        if d[i, j] <= max_dist
    ]


def _arrays(points: Points) -> tuple[list[str], np.ndarray]:
    names = sorted(points, key=name_sort_key)
    return names, np.asarray([points[n] for n in names], dtype=float).reshape(-1, 2)


def _cost(g: np.ndarray, q: np.ndarray, cfg: EvalConfig) -> float:
    """Robust alignment cost: squared distance per matched pair, capped at the inlier
    distance squared, which is also what each unmatched point (either side) pays."""
    cap = cfg.align_inlier_dist**2
    pairs = assign(g, q, cfg.match_max_dist)
    unmatched = len(g) + len(q) - 2 * len(pairs)
    return sum(min(d * d, cap) for _, _, d in pairs) + cap * unmatched


def _global_starts(g: np.ndarray, p: np.ndarray) -> list[Similarity]:
    def box(x):
        lo, hi = x.min(axis=0), x.max(axis=0)
        return (lo + hi) / 2, float(np.linalg.norm(hi - lo))

    def spread(x):
        mu = x.mean(axis=0)
        return mu, float(np.sqrt(((x - mu) ** 2).sum(axis=1).mean()))

    out = [IDENTITY]
    for summary in (box, spread):
        (cg, sg), (cp, sp) = summary(g), summary(p)
        s = sg / sp if sp > 1e-9 and sg > 1e-9 else 1.0
        out.append(Similarity(s, translation=tuple(float(v) for v in cg - s * cp)))
    return out


def _pair_hypotheses(g: np.ndarray, p: np.ndarray, cfg: EvalConfig) -> list[Similarity]:
    """The ``cfg.align_hypotheses_kept`` best transforms fixed by matching a GT point
    pair to a predicted point pair, ranked by a cheap stand-in for ``_cost``."""
    if len(g) < 2 or len(p) < 2:
        return []
    gi, gj = np.triu_indices(len(g), k=1)
    pi, pj = np.nonzero(~np.eye(len(p), dtype=bool))  # ordered pairs
    G = np.repeat(np.arange(len(gi)), len(pi))
    P = np.tile(np.arange(len(pi)), len(gi))
    if len(G) > cfg.align_max_hypotheses:  # deterministic subsample
        keep = np.random.default_rng(0).choice(len(G), cfg.align_max_hypotheses, replace=False)
        G, P = G[np.sort(keep)], P[np.sort(keep)]
    zg = g[:, 0] + 1j * g[:, 1]
    zp = p[:, 0] + 1j * p[:, 1]
    g1, g2 = zg[gi[G]], zg[gj[G]]
    p1, p2 = zp[pi[P]], zp[pj[P]]

    # (z, flip): q = z * p + t, or z * conj(p) + t when flip
    candidates = []
    dp = p1 - p2
    ok = np.abs(dp) > 1e-9
    if cfg.allow_rotation:
        z = np.where(ok, (g1 - g2) / np.where(ok, dp, 1), 1)
        candidates.append((z, g1 - z * p1, False))
        if cfg.allow_reflection:
            zc = np.where(ok, (g1 - g2) / np.where(ok, np.conj(dp), 1), 1)
            candidates.append((zc, g1 - zc * np.conj(p1), True))
    else:  # least-squares scale for the two pairs, no rotation
        s = np.where(ok, np.real((g1 - g2) * np.conj(dp)) / np.where(ok, np.abs(dp) ** 2, 1), 1)
        z = np.maximum(s, 0.0).astype(complex)
        candidates.append((z, (g1 + g2) / 2 - z * (p1 + p2) / 2, False))

    cap = cfg.align_inlier_dist**2
    scored: list[tuple[float, Similarity]] = []
    for z, t, flip in candidates:
        src = np.conj(zp) if flip else zp
        q = z[:, None] * src[None, :] + t[:, None]  # (H, m)
        d2 = np.abs(q[:, None, :] - zg[None, :, None]) ** 2  # (H, n, m)
        # _cost with nearest neighbours instead of a one-to-one assignment
        score = np.minimum(d2.min(axis=2), cap).sum(axis=1)
        score += np.minimum(d2.min(axis=1), cap).sum(axis=1)
        for h in np.argsort(score, kind="stable")[: cfg.align_hypotheses_kept]:
            c, sn = math.cos(np.angle(z[h])), math.sin(np.angle(z[h]))
            R = ((c, -sn), (sn, c)) if not flip else ((c, sn), (sn, -c))
            scored.append(
                (
                    float(score[h]),
                    Similarity(float(abs(z[h])), R, (float(t[h].real), float(t[h].imag))),
                )
            )
    scored.sort(key=lambda x: x[0])
    return [T for _, T in scored[: cfg.align_hypotheses_kept]]


def _refine(g: np.ndarray, p: np.ndarray, T: Similarity, cfg: EvalConfig) -> Similarity:
    """ICP: alternate Hungarian matching and a least-squares fit until stable. The fit
    uses the inlier pairs when there are at least two; otherwise every pair within
    match_max_dist; otherwise (the start is far off) every pair of the full assignment."""
    seen = None
    for _ in range(cfg.align_max_iters):
        q = T.apply(p)
        pairs = assign(g, q, cfg.match_max_dist)
        inliers = [x for x in pairs if x[2] <= cfg.align_inlier_dist]
        if len(inliers) >= 2:
            pairs = inliers
        elif len(pairs) < 2:
            pairs = assign(g, q, math.inf)
        key = [(i, j) for i, j, _ in pairs]
        if key == seen:
            break
        seen = key
        T = fit_similarity(
            p[[j for _, j in key]], g[[i for i, _ in key]],
            cfg.allow_rotation, cfg.allow_reflection,
        )  # fmt: skip
    return T


def align(gt: Points, pred: Points, cfg: EvalConfig | None = None) -> Similarity:
    """The similarity that best maps ``pred`` onto ``gt``, ignoring point names."""
    cfg = cfg or get_config().eval
    _, g = _arrays(gt)
    _, p = _arrays(pred)
    if len(g) == 0 or len(p) == 0:
        return IDENTITY
    best: tuple[float, Similarity] | None = None
    for start in [*_global_starts(g, p), *_pair_hypotheses(g, p, cfg)]:
        T = _refine(g, p, start, cfg)
        c = _cost(g, T.apply(p), cfg)
        if best is None or c < best[0] - 1e-9:
            best = (c, T)
    return best[1]


@dataclass(frozen=True)
class Correspondence:
    pairs: dict[str, str]  # predicted name -> GT name
    dist: dict[str, float]  # GT name -> distance to its match, after alignment


NO_MATCH = Correspondence({}, {})


def _hungarian(gt: Points, gnames, pred: Points, pnames, T: Similarity, max_dist: float):
    if not gnames or not pnames:
        return {}, {}
    g = np.asarray([gt[n] for n in gnames], dtype=float).reshape(-1, 2)
    q = T.apply([pred[n] for n in pnames])
    pairs = assign(g, q, max_dist)
    return (
        {pnames[j]: gnames[i] for i, j, _ in pairs},
        {gnames[i]: d for i, _, d in pairs},
    )


def match_geometric(
    gt: Points, pred: Points, T: Similarity, cfg: EvalConfig | None = None
) -> Correspondence:
    cfg = cfg or get_config().eval
    gnames = sorted(gt, key=name_sort_key)
    pnames = sorted(pred, key=name_sort_key)
    return Correspondence(*_hungarian(gt, gnames, pred, pnames, T, cfg.match_max_dist))


def match_strict(
    gt: Points, pred: Points, T: Similarity, cfg: EvalConfig | None = None
) -> Correspondence:
    cfg = cfg or get_config().eval
    pairs: dict[str, str] = {}
    dist: dict[str, float] = {}
    for n in sorted(pred, key=name_sort_key):
        if not is_unnamed(n) and n in gt:
            pairs[n] = n
            dist[n] = float(np.linalg.norm(T.apply(pred[n])[0] - np.asarray(gt[n], float)))
    gu = sorted((n for n in gt if is_unnamed(n)), key=name_sort_key)
    pu = sorted((n for n in pred if is_unnamed(n)), key=name_sort_key)
    up, ud = _hungarian(gt, gu, pred, pu, T, cfg.match_max_dist)
    return Correspondence({**pairs, **up}, {**dist, **ud})


def ray_keys(gt: Diagram, pred: Diagram, T: Similarity, tol_deg: float):
    """``ray(v, e)``: the GT point nearest ``v`` in the direction of ``e`` (within
    ``tol_deg``), else ``e`` itself. Angles are then compared by their rays, not by which
    points happen to name them: with D on ray BC, angle ABD is angle ABC.

    ``pred`` must already use GT names for matched points (``rename_into``). Positions:
    GT coordinates for GT names, aligned predicted coordinates for the rest."""
    gpos = {k: (float(x), float(y)) for k, (x, y) in gt.points.items()}
    ppos = {k: tuple(T.apply(v)[0]) for k, v in pred.points.items()}
    tol = math.radians(tol_deg)
    cache: dict[tuple[str, str], str] = {}

    def pos(n: str):
        return gpos[n] if n in gpos else ppos[n]

    def ray(v: str, e: str) -> str:
        if v not in gpos or (v, e) in cache:
            return cache.get((v, e), e)
        (vx, vy), (ex, ey) = pos(v), pos(e)
        t = math.atan2(ey - vy, ex - vx)
        best, best_d = e, math.inf
        for g, (gx, gy) in gpos.items():
            d = math.hypot(gx - vx, gy - vy)
            if g == v or d < 1e-9:
                continue
            diff = abs((math.atan2(gy - vy, gx - vx) - t + math.pi) % (2 * math.pi) - math.pi)
            if diff <= tol and d < best_d:
                best, best_d = g, d
        cache[(v, e)] = best
        return best

    return ray


def rename_into(pred: Diagram, pairs: Mapping[str, str], gt_names) -> Diagram:
    """``pred`` with matched points renamed to their GT names and every unmatched point
    to a fresh ``_P`` name that no GT point has, so only matches can coincide.

    The renaming is one-to-one, so the result is a valid diagram with the same structure.
    """
    taken = set(gt_names) | set(pairs.values())
    fresh: dict[str, str] = {}
    k = 1000
    for n in sorted(pred.points, key=name_sort_key):
        if n in pairs:
            continue
        while f"{UNNAMED_PREFIX}{k}" in taken:
            k += 1
        fresh[n] = f"{UNNAMED_PREFIX}{k}"
        taken.add(fresh[n])
        k += 1
    mapping = {**fresh, **pairs}

    return Diagram.model_validate(rename_points(diagram_to_obj(pred), mapping.__getitem__))
