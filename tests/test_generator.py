"""Tests for the synthetic generator (src/mathvlm/generate/)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pytest
from matplotlib.mathtext import MathTextParser

from mathvlm import schema as S
from mathvlm.config import get_config
from mathvlm.generate import build as B
from mathvlm.generate import primitives as G
from mathvlm.generate.scenes import BUILDERS, Namer, to_grid, validate_scene

GCFG = get_config().generate
N_SAMPLES = 400
PARSER = MathTextParser("path")


@lru_cache(maxsize=1)
def samples() -> tuple[B.Sample, ...]:
    return tuple(B.make_sample(1234, "test", i) for i in range(N_SAMPLES))


def diagram_samples():
    return [s for s in samples() if s.transcription.diagram is not None]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_foot_is_perpendicular():
    p, a, b = (3.0, 4.0), (0.0, 0.0), (10.0, 2.0)
    f = G.foot(p, a, b)
    assert close(G.dot(G.sub(p, f), G.sub(b, a)), 0.0)
    assert G.dist_point_line(f, a, b) < 1e-9


def test_bisector_foot_splits_the_angle_equally():
    v, a, b = (0.0, 0.0), (10.0, 1.0), (2.0, 9.0)
    d = G.bisector_foot(v, a, b)
    assert close(G.angle_deg(a, v, d), G.angle_deg(d, v, b), 1e-9)
    assert G.dist_point_segment(d, a, b) < 1e-9


def test_incenter_is_equidistant_from_the_sides():
    a, b, c = (0.0, 0.0), (7.0, 0.0), (2.0, 5.0)
    i, r = G.incenter(a, b, c), G.inradius(a, b, c)
    for p, q in ((a, b), (b, c), (c, a)):
        assert close(G.dist_point_line(i, p, q), r, 1e-9)


def test_circumcenter_is_equidistant_from_the_vertices():
    a, b, c = (0.0, 0.0), (7.0, 0.0), (2.0, 5.0)
    o = G.circumcenter(a, b, c)
    assert close(G.dist(o, a), G.dist(o, b), 1e-9) and close(G.dist(o, b), G.dist(o, c), 1e-9)
    with pytest.raises(ValueError):
        G.circumcenter((0, 0), (1, 1), (2, 2))


def test_tangent_points_touch_at_right_angles():
    o, r, p = (1.0, 2.0), 3.0, (9.0, 5.0)
    for t in G.tangent_points(o, r, p):
        assert close(G.dist(o, t), r, 1e-9)
        assert close(G.angle_deg(o, t, p), 90.0, 1e-7)
    with pytest.raises(ValueError):
        G.tangent_points(o, r, (1.0, 3.0))


def test_line_intersection():
    assert G.line_intersection((0, 0), (2, 2), (0, 2), (2, 0)) == pytest.approx((1.0, 1.0))
    assert G.line_intersection((0, 0), (1, 0), (0, 1), (1, 1)) is None


def test_triangle_from_angles():
    a, b, c = G.triangle_from_angles(40.0, 75.0)
    assert G.angle_deg(b, a, c) == pytest.approx(40.0)
    assert G.angle_deg(a, b, c) == pytest.approx(75.0)
    assert c[1] < 0  # apex above the base (y points down)


def test_is_convex():
    assert G.is_convex([(0, 0), (4, 0), (4, 3), (0, 3)])
    assert not G.is_convex([(0, 0), (4, 0), (1, 1), (0, 3)])


# ---------------------------------------------------------------------------
# Scenes: geometry survives rotation, fitting and rounding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(BUILDERS))
def test_scene_constructions_hold_on_the_grid(family):
    accepted = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        sc = to_grid(BUILDERS[family](rng, Namer(rng, GCFG), GCFG), rng, GCFG)
        if validate_scene(sc, GCFG):
            continue
        accepted += 1
        P = sc.points
        for name, seg in sc.on_segment:
            a, b = (P[n] for n in seg)
            assert G.dist_point_segment(P[name], a, b) <= 1.5, (family, seed, name, seg)
        for name, c in sc.on_circle:
            r = dict(sc.circles)[c]
            assert abs(G.dist(P[name], P[c]) - r) <= 1.5, (family, seed, name, c)
        for key, deg in sc.exact_angles.items():
            v, (e1, e2) = key[0], tuple(key[1])
            assert abs(G.angle_deg(P[e1], P[v], P[e2]) - deg) <= GCFG.min_angle_deg / 4
        for grp in sc.equal_lengths:
            lens = [G.dist(P[a], P[b]) for a, b in grp]
            assert max(lens) - min(lens) <= 2.0, (family, seed, grp, lens)
        for grp in sc.equal_angles:
            angs = [G.angle_deg(P[a], P[v], P[b]) for a, v, b in grp]
            assert max(angs) - min(angs) <= 6.0, (family, seed, grp, angs)
    assert accepted >= 40, f"{family}: only {accepted}/60 scenes accepted"


def test_scene_fills_the_canonical_box():
    target = 100 - 2 * GCFG.margin
    for s in diagram_samples():
        d = s.transcription.diagram
        xs = [p[0] for p in d.points.values()] + [
            d.points[c.center][0] + k * c.radius for c in d.circles for k in (-1, 1)
        ]
        ys = [p[1] for p in d.points.values()] + [
            d.points[c.center][1] + k * c.radius for c in d.circles for k in (-1, 1)
        ]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        assert target - 2 <= span <= target + 2, (s.id, span)
        assert min(xs) >= GCFG.margin - 2 and max(xs) <= 100 - GCFG.margin + 2


def test_validity_thresholds():
    for s in diagram_samples():
        pts = list(s.transcription.diagram.points.values())
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert G.dist(pts[i], pts[j]) >= GCFG.min_point_separation, s.id


# ---------------------------------------------------------------------------
# Samples: schema, text, labels
# ---------------------------------------------------------------------------


def test_every_sample_is_valid_and_consistent():
    for s in diagram_samples():
        r = S.validate_output(S.output_to_obj(s.transcription))
        assert r.ok, (s.id, r.errors)
        assert S.consistency_issues(s.transcription.diagram) == [], s.id


def test_family_coverage_and_no_diagram_share():
    fam = Counter(s.family for s in samples())
    assert set(fam) == set(BUILDERS) | {"no_diagram"}
    share = fam["no_diagram"] / N_SAMPLES
    assert abs(share - GCFG.no_diagram_fraction) < 0.06, share
    assert all(c > N_SAMPLES * 0.08 for f, c in fam.items() if f != "no_diagram"), fam


def _parse_lines(text: str) -> None:
    for line in text.split("\n"):
        line = re.sub(r"\$\$(.+?)\$\$", r"$\1$", line)
        if line.strip():
            PARSER.parse(line)


def test_text_renders_and_names_only_labeled_points():
    for s in samples():
        t = s.transcription.text
        _parse_lines(t)  # raises on LaTeX that mathtext (and so the renderer) can't handle
        assert "_P" not in t, s.id
        d = s.transcription.diagram
        if d is None:
            continue
        math_parts = " ".join(re.findall(r"\$+([^$]+)\$+", t))
        for nm in re.findall(
            r"[A-Z](?:'{1,2}|_[0-9]{1,2})?", re.sub(r"\\[a-zA-Z]+", " ", math_parts)
        ):
            assert nm in d.points, (s.id, nm, t)


LINE_NAMES = set(get_config().generate.realism.line_names)


def test_labels_parse_and_never_sit_on_a_segment_with_an_interior_point():
    for s in diagram_samples():
        d = s.transcription.diagram
        for lb in d.labels:
            PARSER.parse(f"${lb.text}$")
            # a length on a split segment would be ambiguous; a line's name is not: it
            # names the whole line, interior points and all
            if isinstance(lb, S.SegmentLabel) and lb.text not in LINE_NAMES:
                a, b = (d.points[n] for n in lb.on)
                for n, p in d.points.items():
                    if n in lb.on:
                        continue
                    inside = 0.02 < G.project_param(p, a, b) < 0.98
                    assert not (inside and G.dist_point_segment(p, a, b) < 1.5), (s.id, lb, n)


def test_givens_appear_where_the_metadata_says():
    for s in diagram_samples():
        labels = [lb.text for lb in s.transcription.diagram.labels]
        for g in s.meta["givens"]:
            assert g["in_text"] or g["in_figure"], s.id
            if g["in_text"]:
                assert g["value"] in s.transcription.text, (s.id, g)
            if g["in_figure"]:
                assert g["value"] in labels, (s.id, g)


def test_algebraic_labels_match_their_partner():
    found = 0
    for s in diagram_samples():
        u = s.meta.get("unknown", {})
        if u.get("type") != "algebraic" or s.meta.get("diagram_only"):
            continue  # a figure-only page drops the question, and with it the partner
        found += 1
        m = re.fullmatch(r"\(?(\d+)([xy])([+-]\d+)?\)?(\^\\circ)?", u["label"])
        assert m, u["label"]
        value = int(m.group(1)) * u["x"] + int(m.group(3) or 0)
        assert str(value) in {g["value"].replace("^\\circ", "") for g in s.meta["givens"]}, (
            s.id,
            u,
        )
    assert found >= 3, f"only {found} algebraic samples; the variant is not being exercised"


def test_unknown_variable_is_on_the_figure():
    for s in diagram_samples():
        u = s.meta.get("unknown", {})
        if u.get("type") in ("variable", "algebraic"):
            assert u["label"] in [lb.text for lb in s.transcription.diagram.labels], s.id


# ---------------------------------------------------------------------------
# Determinism (design invariant 4)
# ---------------------------------------------------------------------------


def test_same_index_gives_the_same_sample():
    a, b = B.make_sample(7, "val", 3), B.make_sample(7, "val", 3)
    assert S.to_target_json(a.transcription) == S.to_target_json(b.transcription)
    assert a.style == b.style and a.meta == b.meta


def test_different_indices_and_splits_differ():
    targets = {S.to_target_json(B.make_sample(7, "val", i).transcription) for i in range(20)}
    assert len(targets) == 20
    assert S.to_target_json(B.make_sample(7, "val", 0).transcription) != S.to_target_json(
        B.make_sample(7, "train", 0).transcription
    )


def test_sample_is_identical_alone_or_in_a_batch(tmp_path):
    batch, alone = tmp_path / "batch", tmp_path / "alone"
    B.build("val", 8, 7, batch)
    B.build("val", 1, 7, alone, start=5)
    for ext in ("json", "png"):
        assert (batch / f"val_000005.{ext}").read_bytes() == (
            alone / f"val_000005.{ext}"
        ).read_bytes()


def test_worker_count_does_not_change_output(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    B.build("val", 6, 7, one, images=False)
    B.build("val", 6, 7, two, workers=2, images=False)
    assert (one / "manifest.jsonl").read_text() == (two / "manifest.jsonl").read_text()
    for i in range(6):
        assert (one / f"val_{i:06d}.json").read_bytes() == (two / f"val_{i:06d}.json").read_bytes()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_samples_and_manifest(tmp_path):
    assert (
        B.main(
            ["--split", "val", "--n", "3", "--seed", "1", "--out", str(tmp_path), "--sheet", "0"]
        )
        == 0
    )
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["val_000000", "val_000001", "val_000002"]
    for r in rows:
        assert (tmp_path / r["image"]).exists()
        gt = json.loads((tmp_path / r["gt"]).read_text())
        assert S.validate_output(gt).ok
        assert set(r) >= {"family", "kind", "seed", "style", "attempts"}


def test_cli_no_images(tmp_path):
    B.main(["--split", "val", "--n", "2", "--out", str(tmp_path), "--no-images"])
    assert not list(tmp_path.glob("*.png"))
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert math.isfinite(len((tmp_path / "manifest.jsonl").read_text()))


def test_unnamed_points_are_numbered_in_reading_order():
    seen = 0
    for i in range(300):
        d = B.make_sample(0, "reading", i).transcription.diagram
        if d is None:
            continue
        unnamed = [n for n in d.points if S.is_unnamed(n)]
        if len(unnamed) < 2:
            continue
        seen += 1
        by_position = sorted(unnamed, key=lambda n: (d.points[n][1], d.points[n][0]))
        assert by_position == [f"_P{k + 1}" for k in range(len(unnamed))]
    assert seen >= 3
