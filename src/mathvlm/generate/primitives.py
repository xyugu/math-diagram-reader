"""Plane-geometry helpers on float points ``(x, y)``.

Pure functions, no randomness. The scene builders work in an arbitrary float frame
(y pointing down, like the schema) and only round to the integer grid at the end.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec = tuple[float, float]


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec, b: Vec) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a: Vec, b: Vec, t: float) -> Vec:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def midpoint(a: Vec, b: Vec) -> Vec:
    return lerp(a, b, 0.5)


def rotate(p: Vec, theta: float, origin: Vec = (0.0, 0.0)) -> Vec:
    """Rotate ``p`` by ``theta`` radians about ``origin``."""
    c, s = math.cos(theta), math.sin(theta)
    x, y = p[0] - origin[0], p[1] - origin[1]
    return (origin[0] + c * x - s * y, origin[1] + s * x + c * y)


def angle_deg(a: Vec, v: Vec, b: Vec) -> float:
    """Interior angle a-v-b in degrees, in [0, 180]. 0 if a ray has zero length."""
    u, w = sub(a, v), sub(b, v)
    nu, nw = norm(u), norm(w)
    if nu == 0 or nw == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(u, w) / (nu * nw)))))


def project_param(p: Vec, a: Vec, b: Vec) -> float:
    """Parameter t of p's projection onto line ab (0 at a, 1 at b)."""
    d = sub(b, a)
    return dot(sub(p, a), d) / dot(d, d)


def foot(p: Vec, a: Vec, b: Vec) -> Vec:
    """Foot of the perpendicular from p to line ab."""
    return lerp(a, b, project_param(p, a, b))


def dist_point_line(p: Vec, a: Vec, b: Vec) -> float:
    return dist(p, foot(p, a, b))


def dist_point_segment(p: Vec, a: Vec, b: Vec) -> float:
    t = max(0.0, min(1.0, project_param(p, a, b)))
    return dist(p, lerp(a, b, t))


def line_intersection(a: Vec, b: Vec, c: Vec, d: Vec) -> Vec | None:
    """Intersection of lines ab and cd, or None if (nearly) parallel."""
    r, s = sub(b, a), sub(d, c)
    den = cross(r, s)
    if abs(den) < 1e-12:
        return None
    return lerp(a, b, cross(sub(c, a), s) / den)


def bisector_foot(v: Vec, a: Vec, b: Vec) -> Vec:
    """Where the bisector of angle a-v-b meets segment ab (angle bisector theorem:
    the foot divides ab in the ratio |va| : |vb|)."""
    da, db = dist(v, a), dist(v, b)
    return lerp(a, b, da / (da + db))


def triangle_area(a: Vec, b: Vec, c: Vec) -> float:
    return abs(cross(sub(b, a), sub(c, a))) / 2


def incenter(a: Vec, b: Vec, c: Vec) -> Vec:
    la, lb, lc = dist(b, c), dist(c, a), dist(a, b)
    s = la + lb + lc
    return ((la * a[0] + lb * b[0] + lc * c[0]) / s, (la * a[1] + lb * b[1] + lc * c[1]) / s)


def inradius(a: Vec, b: Vec, c: Vec) -> float:
    return triangle_area(a, b, c) / ((dist(b, c) + dist(c, a) + dist(a, b)) / 2)


def circumcenter(a: Vec, b: Vec, c: Vec) -> Vec:
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        raise ValueError("collinear points have no circumcenter")
    sa, sb, sc = dot(a, a), dot(b, b), dot(c, c)
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    return (ux, uy)


def tangent_points(center: Vec, r: float, p: Vec) -> tuple[Vec, Vec]:
    """The two points where tangents from external point p touch the circle."""
    d = dist(center, p)
    if d <= r:
        raise ValueError("point is not outside the circle")
    base = math.atan2(p[1] - center[1], p[0] - center[0])
    phi = math.acos(r / d)
    return (
        (center[0] + r * math.cos(base + phi), center[1] + r * math.sin(base + phi)),
        (center[0] + r * math.cos(base - phi), center[1] + r * math.sin(base - phi)),
    )


def triangle_from_angles(alpha: float, beta: float) -> tuple[Vec, Vec, Vec]:
    """Triangle with A=(0,0), B=(1,0) and apex C above AB (negative y, since y points
    down), with angle ``alpha`` (degrees) at A and ``beta`` at B."""
    gamma = 180.0 - alpha - beta
    if min(alpha, beta, gamma) <= 0:
        raise ValueError("angles must be positive and sum to less than 180")
    ac = math.sin(math.radians(beta)) / math.sin(math.radians(gamma))  # law of sines, |AB| = 1
    ar = math.radians(alpha)
    return (0.0, 0.0), (1.0, 0.0), (ac * math.cos(ar), -ac * math.sin(ar))


def polygon_angles(pts: Sequence[Vec]) -> list[float]:
    n = len(pts)
    return [angle_deg(pts[i - 1], pts[i], pts[(i + 1) % n]) for i in range(n)]


def is_convex(pts: Sequence[Vec]) -> bool:
    n = len(pts)
    signs = set()
    for i in range(n):
        c = cross(sub(pts[(i + 1) % n], pts[i]), sub(pts[(i + 2) % n], pts[(i + 1) % n]))
        if abs(c) < 1e-9:
            return False
        signs.add(c > 0)
    return len(signs) == 1
