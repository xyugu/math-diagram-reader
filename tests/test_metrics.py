"""Tests for the eval metrics (src/mathvlm/eval/): parsing, alignment and matching,
structure F1, geometry error, aggregation. README.md, design invariant 7: every change to
the metrics needs a test here."""

from __future__ import annotations

import copy
import json
import math
import random

import numpy as np
import pytest

from mathvlm.config import get_config
from mathvlm.eval import matching as Mt
from mathvlm.eval import metrics as M
from mathvlm.eval.parse import parse_output
from mathvlm.generate.build import make_sample
from mathvlm.schema import Transcription, to_target_json

EC = get_config().eval
ROT = EC.model_copy(update={"allow_rotation": True})
ROT_REF = EC.model_copy(update={"allow_rotation": True, "allow_reflection": True})
TEXT = "Find $x$."

# Uses every structure kind. Coordinates stay within [10, 45], so 2x + 10 is still on
# the grid. Validity is all the metrics need; the figure need not be consistent.
BASE = {
    "points": {
        "A": [10, 42], "B": [44, 40], "C": [16, 10], "M": [30, 25],
        "O": [28, 32], "_P1": [27, 41], "_P2": [13, 26],
    },
    "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["A", "M"], ["C", "_P1"], ["B", "_P2"]],
    "circles": [{"center": "O", "radius": 5}],
    "marks": [
        {"type": "right_angle", "at": "A", "rays": ["B", "C"]},
        {"type": "equal_length", "segments": [["B", "M"], ["M", "C"]]},
        {"type": "parallel", "segments": [["A", "B"], ["C", "_P2"]]},
        {"type": "equal_angle", "angles": [["B", "A", "M"], ["M", "A", "C"]]},
    ],
    "labels": [{"on": ["A", "B"], "text": "6"}, {"angle": ["A", "B", "C"], "text": "45^\\circ"}],
}  # fmt: skip
N_BASE = {"points": 7, "segments": 6, "circles": 1, "marks": 4, "labels": 2}


def tr(diagram, text: str = TEXT) -> Transcription:
    return Transcription.model_validate({"text": text, "diagram": diagram})


def js(diagram, text: str = TEXT) -> str:
    return json.dumps({"text": text, "diagram": diagram})


def ev(gt, pred, cfg=None, text: str = TEXT) -> M.ExampleResult:
    """Evaluate a prediction given as a diagram dict (or None), or as a raw string."""
    raw = pred if isinstance(pred, str) else js(pred, text)
    return M.evaluate_example(tr(gt, text), raw, cfg=cfg)


def cnt(r: M.ExampleResult, variant: str, cat: str) -> tuple[int, int, int]:
    c = r.structure[variant][cat]
    return (c.tp, c.fp, c.fn)


def all_f1(r: M.ExampleResult, variant: str) -> dict[str, float | None]:
    return {c: r.structure[variant][c].f1 for c in M.CATEGORIES}


def rename(d: dict, m: dict[str, str]) -> dict:
    def f(n):
        return m.get(n, n)

    def mark(mk):
        if mk["type"] == "right_angle":
            return {**mk, "at": f(mk["at"]), "rays": [f(x) for x in mk["rays"]]}
        if mk["type"] == "equal_angle":
            return {**mk, "angles": [[f(x) for x in a] for a in mk["angles"]]}
        return {**mk, "segments": [[f(x) for x in s] for s in mk["segments"]]}

    def label(lb):
        if "on" in lb:
            return {"on": [f(x) for x in lb["on"]], "text": lb["text"]}
        return {"angle": [f(x) for x in lb["angle"]], "text": lb["text"]}

    return {
        "points": {f(k): v for k, v in d["points"].items()},
        "segments": [[f(a), f(b)] for a, b in d.get("segments", [])],
        "circles": [{**c, "center": f(c["center"])} for c in d.get("circles", [])],
        "marks": [mark(x) for x in d.get("marks", [])],
        "labels": [label(x) for x in d.get("labels", [])],
    }


def move(d: dict, fn, radius_scale: float = 1.0) -> dict:
    out = copy.deepcopy(d)
    out["points"] = {k: [round(c) for c in fn(*v)] for k, v in d["points"].items()}
    out["circles"] = [{**c, "radius": c["radius"] * radius_scale} for c in d.get("circles", [])]
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_direct_and_fenced():
    raw = js(BASE)
    assert parse_output(raw).how == "direct"
    assert parse_output(f"  \n{raw}\n ").how == "direct"
    for fenced in (f"```json\n{raw}\n```", f"```\n{raw}\n```", f"```JSON\n{raw}```"):
        p = parse_output(fenced)
        assert p.how == "fenced" and p.report.ok


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        (None, "missing"),
        ("", "empty"),
        ("  \n", "empty"),
        ('{"text": "a", "diagram": {"points": {"A": [1,', "truncated"),
        ('{"text": "a', "truncated"),
        ('{"text": "$\\sqrt{2}$", "diagram": null}', "invalid_escape"),  # JSON has no \s
        ('{"text": "one\ntwo", "diagram": null}', "control_char"),  # raw newline in a string
        ('Here it is: {"text": "a", "diagram": null}', "prose_wrapped"),
        ('{"text": "a", "diagram": null} Hope this helps!', "prose_wrapped"),
        ('```json\n{"text": "a", "diagram": null}\n```\nDone.', "prose_wrapped"),
        ("I cannot read this image.", "no_json"),
        ("{'text': 'a', 'diagram': None}", "syntax"),
    ],
)
def test_parse_failures_are_classified(raw, failure):
    p = parse_output(raw)
    assert p.how is None and p.report.stage == "parse_error"
    assert p.failure == failure
    assert p.diagram_kind == "failed" and p.text is None


def test_generator_truncation_flag_wins():
    assert parse_output('{"text": "a", "diagr', truncated=True).failure == "truncated"
    assert parse_output("prose {", truncated=True).failure == "truncated"
    assert parse_output(js(None), truncated=True).report.ok  # finished exactly at the limit


def test_duplicate_keys_parse_but_are_invalid():
    p = parse_output('{"text": "a", "text": "b", "diagram": null}')
    assert p.how == "direct" and p.report.stage == "invalid"


def test_nan_is_not_json():
    assert parse_output('{"text": "a", "diagram": {"points": {"A": [NaN, 1]}}}').how is None


def test_unescaped_latex_that_happens_to_parse_is_flagged():
    p = parse_output('{"text": "$\\frac{1}{2}$", "diagram": null}')  # \f is a form feed
    assert p.report.ok and p.escape_suspect
    assert p.text == "$\frac{1}{2}$"
    assert not parse_output(js(None, "$\\frac{1}{2}$")).escape_suspect


def test_text_and_diagram_kind_survive_an_invalid_diagram():
    p = parse_output(js({"points": {"A": [1.5, 2]}}, "Find x."))
    assert p.report.stage == "invalid"
    assert p.text == "Find x." and p.diagram_kind == "diagram"


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        (js(None), "null"),
        (js(BASE), "diagram"),
        ('{"text": "a"}', "failed"),
        ('{"text": "a", "diagram": []}', "failed"),
        ("[1, 2]", "failed"),
        ("garbage", "failed"),
    ],
)
def test_diagram_kind(raw, kind):
    assert parse_output(raw).diagram_kind == kind


# ---------------------------------------------------------------------------
# Alignment and matching
# ---------------------------------------------------------------------------


def test_fit_similarity_recovers_known_transforms():
    src = np.random.default_rng(0).uniform(0, 50, (6, 2))
    dst = 1.7 * src + [12, -3]
    T = Mt.fit_similarity(src, dst)
    assert T.scale == pytest.approx(1.7)
    assert np.allclose(T.apply(src), dst)

    th = 0.6
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    rotated = 0.8 * src @ R.T + [5, 5]
    assert np.allclose(Mt.fit_similarity(src, rotated, rotation=True).apply(src), rotated)
    assert not np.allclose(Mt.fit_similarity(src, rotated).apply(src), rotated, atol=1)

    mirrored = src * [-1, 1] + [100, 0]
    assert np.allclose(Mt.fit_similarity(src, mirrored, True, True).apply(src), mirrored)
    assert not np.allclose(Mt.fit_similarity(src, mirrored, True).apply(src), mirrored, atol=1)


def test_fit_similarity_degenerate_inputs():
    assert Mt.fit_similarity([], []) == Mt.IDENTITY
    one = Mt.fit_similarity([[1, 2]], [[4, 6]])
    assert one.scale == 1.0 and one.translation == (3.0, 4.0)
    coincident = Mt.fit_similarity([[1, 1], [1, 1]], [[0, 0], [2, 2]])
    assert coincident.scale == 1.0
    assert np.allclose(coincident.apply([[1, 1]]), [[1, 1]])
    # anti-correlated without rotation: the scale clamps at 0 rather than flipping
    assert Mt.fit_similarity([[0, 0], [10, 0]], [[10, 0], [0, 0]]).scale == 0.0


def test_assign_maximizes_pairs_within_threshold_first():
    g, p = [[0, 0], [12, 0]], [[11, 0], [24, 0]]
    # the minimum-total assignment (g0-p0 11, g1-p1 12) has no pair within 10
    assert Mt.assign(g, p, 10.0) == [(1, 0, 1.0)]
    assert len(Mt.assign(g, p, math.inf)) == 2
    assert Mt.assign([], p, 10.0) == []


def test_align_ignores_names_and_survives_outliers():
    gt = {"A": [10, 10], "B": [90, 10], "C": [50, 80], "D": [50, 40]}
    pred = {"P": [30, 30], "Q": [70, 30], "R": [50, 65], "Z": [95, 95]}  # half size, +extra
    T = Mt.align(gt, pred)
    assert T.scale == pytest.approx(2.0)
    assert np.allclose(T.apply([[30, 30], [70, 30], [50, 65]]), [[10, 10], [90, 10], [50, 80]])


def test_align_follows_the_consistent_majority():
    """Three exact points and one 20 units off: the error stays on the one point instead
    of being spread over all four by a compromise shrink."""
    gt = {"A": [10, 10], "B": [90, 10], "C": [50, 90], "D": [30, 40]}
    pred = {**gt, "D": [30, 60]}
    T = Mt.align(gt, pred)
    assert T.scale == pytest.approx(1.0) and np.allclose(T.translation, 0.0)
    r = ev({"points": gt}, {"points": pred})
    assert sorted(r.geometry.errors) == pytest.approx([0.0, 0.0, 0.0, 20.0])


def test_align_recovers_shrunk_shifted_predictions_with_point_errors():
    """On generated figures: scale 0.6, shift, drop one point, hallucinate another."""
    checked = 0
    for i in range(80):
        d = make_sample(9, "align", i).transcription.diagram
        if d is None or len(d.points) < 4:
            continue
        names = sorted(d.points)
        dropped = names[len(names) // 2]
        pred = {k: [round(0.6 * x + 5), round(0.6 * y + 5)] for k, (x, y) in d.points.items()}
        del pred[dropped]
        extra = next(c for c in "ZYXWVU" if c not in d.points)
        pred[extra] = [97, 97]
        r = ev({"points": {k: list(v) for k, v in d.points.items()}}, {"points": pred})
        assert r.geometry.scale == pytest.approx(1 / 0.6, rel=0.03)
        assert r.geometry.coverage == pytest.approx((len(names) - 1) / len(names))
        assert max(r.geometry.errors) < 1.5  # integer rounding at 0.6x, scaled back up
        assert extra not in r.pairs["geometric"]
        checked += 1
    assert checked > 40


def test_align_with_nothing_to_align():
    assert Mt.align({}, {"A": [1, 1]}) == Mt.IDENTITY
    assert Mt.align({"A": [1, 1]}, {}) == Mt.IDENTITY


def test_rename_into_is_one_to_one():
    pred = tr(rename(BASE, {"A": "_P1000", "B": "X"})).diagram
    out = Mt.rename_into(pred, {"C": "C"}, BASE["points"])
    assert "C" in out.points and len(out.points) == len(pred.points)
    assert not (set(out.points) - {"C"}) & set(BASE["points"])


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_identity_scores_perfectly():
    r = ev(BASE, BASE)
    assert r.parsed.how == "direct" and r.parsed.report.ok
    assert (r.gt_kind, r.pred_kind) == ("diagram", "diagram")
    for v in M.VARIANTS:
        assert all_f1(r, v) == dict.fromkeys(M.CATEGORIES, 1.0)
        for c, n in N_BASE.items():
            assert cnt(r, v, c) == (n, 0, 0)
    g = r.geometry
    assert g.coverage == 1.0 and g.scale == pytest.approx(1.0)
    assert max(g.errors + g.strict_errors + g.raw_errors + g.radius_errors) < 1e-9
    assert r.text == {"has_text": True, "ned": 0.0, "ned_raw": 0.0, "exact": True,
                      "escape_suspect": False}  # fmt: skip


def test_list_order_and_endpoint_order_do_not_matter():
    rnd = random.Random(3)
    d = copy.deepcopy(BASE)
    d["points"] = dict(rnd.sample(list(d["points"].items()), len(d["points"])))
    for key in ("segments", "marks", "labels"):
        rnd.shuffle(d[key])
    d["segments"] = [s[::-1] for s in d["segments"]]
    for m in d["marks"]:
        if m["type"] == "right_angle":
            m["rays"] = m["rays"][::-1]
        elif m["type"] == "equal_angle":
            m["angles"] = [a[::-1] for a in m["angles"]][::-1]
        else:
            m["segments"] = [s[::-1] for s in m["segments"]][::-1]
    for lb in d["labels"]:
        lb["on" if "on" in lb else "angle"] = lb.get("on", lb.get("angle"))[::-1]
    r = ev(BASE, d)
    for v in M.VARIANTS:
        assert all_f1(r, v) == dict.fromkeys(M.CATEGORIES, 1.0)


def test_consistent_relabeling_drops_strict_but_not_geometric():
    r = ev(BASE, rename(BASE, {"A": "X", "B": "Y", "C": "Z", "M": "N", "O": "Q"}))
    assert all_f1(r, "geometric") == dict.fromkeys(M.CATEGORIES, 1.0)
    assert r.geometry.coverage == 1.0 and max(r.geometry.errors) < 1e-9
    # strict: only the two _P points (matched by position) survive
    assert cnt(r, "strict", "points") == (2, 5, 5)
    for c in ("segments", "circles", "marks", "labels"):
        assert r.structure["strict"][c].f1 == 0.0


def test_hand_computed_precision_and_recall():
    gt = {"points": {"A": [10, 10], "B": [90, 10], "C": [50, 80], "D": [50, 10]},
          "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["C", "D"]]}  # fmt: skip
    pred = {"points": {"A": [10, 10], "B": [90, 10], "C": [50, 80], "E": [15, 60]},
            "segments": [["A", "B"], ["B", "C"], ["C", "E"]]}  # fmt: skip
    r = ev(gt, pred)
    for v in M.VARIANTS:  # E is far from D, so the variants agree
        pts, seg = r.structure[v]["points"], r.structure[v]["segments"]
        assert (pts.tp, pts.fp, pts.fn) == (3, 1, 1)
        assert pts.precision == pts.recall == pts.f1 == pytest.approx(0.75)
        assert (seg.tp, seg.fp, seg.fn) == (2, 1, 2)
        assert seg.precision == pytest.approx(2 / 3)
        assert seg.recall == pytest.approx(1 / 2)
        assert seg.f1 == pytest.approx(4 / 7)
        assert r.structure[v]["circles"].f1 is None  # nothing to find, nothing predicted
    assert r.geometry.coverage == pytest.approx(3 / 4)


def test_hungarian_recovers_shuffled_unnamed_points():
    gt = copy.deepcopy(BASE)
    gt["points"]["_P3"] = [40, 12]
    gt["segments"].append(["_P1", "_P3"])
    pred = rename(gt, {"_P1": "_P3", "_P2": "_P1", "_P3": "_P2"})
    pred = move(pred, lambda x, y: (x + 1, y - 1))  # and jitter a little
    r = ev(gt, pred)
    assert all_f1(r, "strict") == dict.fromkeys(M.CATEGORIES, 1.0)
    assert {p: g for p, g in r.pairs["strict"].items() if p.startswith("_P")} == {
        "_P3": "_P1",
        "_P1": "_P2",
        "_P2": "_P3",
    }


def test_unnamed_points_beyond_max_dist_do_not_match():
    gt = {"points": {"A": [10, 10], "B": [90, 10], "C": [50, 90], "_P1": [30, 40]},
          "segments": [["A", "_P1"]]}  # fmt: skip
    pred = copy.deepcopy(gt)
    pred["points"]["_P1"] = [30, round(40 + EC.match_max_dist + 10)]
    r = ev(gt, pred)
    assert cnt(r, "strict", "points") == (3, 1, 1)
    assert cnt(r, "strict", "segments") == (0, 1, 1)


def test_circles_match_by_center_and_radius_is_geometry():
    gt = {"points": {"O": [50, 50], "A": [80, 50]}, "circles": [{"center": "O", "radius": 20}]}
    pred = copy.deepcopy(gt)
    pred["circles"][0]["radius"] = 25
    r = ev(gt, pred)
    assert r.structure["strict"]["circles"].f1 == 1.0
    assert r.geometry.radius_errors == [pytest.approx(5.0)]


def test_concentric_circles_pair_optimally():
    gt = {"points": {"O": [50, 50], "A": [80, 50]},
          "circles": [{"center": "O", "radius": 10}, {"center": "O", "radius": 20}]}  # fmt: skip
    pred = {"points": gt["points"], "circles": [{"center": "O", "radius": 19}]}
    r = ev(gt, pred)
    assert cnt(r, "strict", "circles") == (1, 0, 1)
    assert r.geometry.radius_errors == [pytest.approx(1.0)]


@pytest.mark.parametrize(
    ("label", "match"),
    [
        ({"angle": ["A", "B", "C"], "text": "45^\\circ"}, True),
        ({"angle": ["C", "B", "A"], "text": "45°"}, True),
        ({"angle": ["A", "B", "C"], "text": "$45^{\\circ}$"}, True),
        ({"angle": ["A", "B", "C"], "text": "46^\\circ"}, False),
        ({"angle": ["A", "C", "B"], "text": "45^\\circ"}, False),
        ({"on": ["A", "B"], "text": "45^\\circ"}, False),
    ],
)
def test_labels_match_on_target_and_normalized_text(label, match):
    pts = {"A": [10, 80], "B": [50, 20], "C": [90, 80]}
    gt = {"points": pts, "labels": [{"angle": ["A", "B", "C"], "text": "45^\\circ"}]}
    r = ev(gt, {"points": pts, "labels": [label]})
    assert cnt(r, "strict", "labels") == ((1, 0, 0) if match else (0, 1, 1))


def test_marks_match_whole():
    pts = {"A": [10, 80], "B": [50, 20], "C": [90, 80], "M": [50, 80]}
    eq3 = {"type": "equal_length", "segments": [["A", "B"], ["B", "C"], ["C", "A"]]}
    eq2 = {"type": "equal_length", "segments": [["A", "B"], ["B", "C"]]}
    r = ev({"points": pts, "marks": [eq3]}, {"points": pts, "marks": [eq2]})
    assert cnt(r, "strict", "marks") == (0, 1, 1)
    assert cnt(r, "strict", "marks.equal_length") == (0, 1, 1)
    assert cnt(r, "strict", "marks.right_angle") == (0, 0, 0)

    ra = {"type": "right_angle", "at": "M", "rays": ["A", "B"]}
    swapped = {"type": "right_angle", "at": "M", "rays": ["B", "A"]}
    other = {"type": "right_angle", "at": "M", "rays": ["C", "B"]}
    assert cnt(ev({"points": pts, "marks": [ra]}, {"points": pts, "marks": [swapped]}),
               "strict", "marks.right_angle") == (1, 0, 0)  # fmt: skip
    assert cnt(ev({"points": pts, "marks": [ra]}, {"points": pts, "marks": [other]}),
               "strict", "marks") == (0, 1, 1)  # fmt: skip


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_similarity_invariance():
    r = ev(BASE, move(BASE, lambda x, y: (2 * x + 10, 2 * y + 10), radius_scale=2.0))
    g = r.geometry
    assert max(g.errors) < 1e-9 and max(g.strict_errors) < 1e-9
    assert g.scale == pytest.approx(0.5)
    assert min(g.raw_errors) > 10
    assert g.radius_errors == [pytest.approx(0.0, abs=1e-9)]
    for v in M.VARIANTS:
        assert all_f1(r, v) == dict.fromkeys(M.CATEGORIES, 1.0)


ASYM = {
    "points": {"A": [10, 90], "B": [90, 80], "C": [30, 10], "D": [70, 30], "E": [50, 60]},
    "segments": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]],
}


def test_rotation_is_not_forgiven_by_default():
    rotated = move(ASYM, lambda x, y: (100 - y, x))  # 90 degrees about the grid center
    assert M.summarize([ev(ASYM, rotated)])["geometry"]["error_mean"] > 5
    assert max(ev(ASYM, rotated, cfg=ROT).geometry.errors) < 1e-6


def test_reflection_is_not_forgiven_without_the_flag():
    mirrored = move(ASYM, lambda x, y: (100 - x, y))
    assert M.summarize([ev(ASYM, mirrored)])["geometry"]["error_mean"] > 5
    assert M.summarize([ev(ASYM, mirrored, cfg=ROT)])["geometry"]["error_mean"] > 1
    assert max(ev(ASYM, mirrored, cfg=ROT_REF).geometry.errors) < 1e-6


def test_reflection_requires_rotation_in_config():
    from mathvlm.config import EvalConfig

    with pytest.raises(ValueError, match="requires"):
        EvalConfig.model_validate({**EC.model_dump(), "allow_reflection": True})


def test_labels_on_the_wrong_points_of_a_symmetric_figure_are_caught_by_name():
    rect = {"points": {"A": [20, 70], "B": [80, 70], "C": [80, 30], "D": [20, 30]},
            "segments": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]]}  # fmt: skip
    mirrored = move(rect, lambda x, y: (100 - x, y))  # A and B swap places, as do C and D
    r = ev(rect, mirrored)
    assert max(r.geometry.errors) < 1e-9  # the same point set
    assert r.geometry.strict_errors == [pytest.approx(60.0)] * 4


def test_single_point_aligns_by_translation():
    r = ev({"points": {"A": [50, 50]}}, {"points": {"A": [60, 40]}})
    assert r.geometry.errors == [pytest.approx(0.0)]
    assert r.geometry.raw_errors == [pytest.approx(math.hypot(10, 10))]
    assert r.geometry.scale == 1.0


def test_missing_and_hallucinated_points():
    pred = copy.deepcopy(BASE)
    del pred["points"]["_P1"]
    pred["segments"] = [s for s in pred["segments"] if "_P1" not in s]
    pred["points"]["Z"] = [95, 95]
    r = ev(BASE, pred)
    assert r.geometry.coverage == pytest.approx(6 / 7)
    assert max(r.geometry.errors) < 1e-9
    assert cnt(r, "geometric", "points") == (6, 1, 1)


# ---------------------------------------------------------------------------
# Null diagrams and failed predictions
# ---------------------------------------------------------------------------


def test_gt_diagram_predicted_null():
    r = ev(BASE, None)
    assert (r.gt_kind, r.pred_kind) == ("diagram", "null")
    for v in M.VARIANTS:
        for c, n in N_BASE.items():
            assert cnt(r, v, c) == (0, 0, n)
    assert r.geometry is None
    assert r.text["ned"] == 0.0


def test_gt_null_predicted_diagram():
    r = ev(None, BASE)
    assert (r.gt_kind, r.pred_kind) == ("null", "diagram")
    for v in M.VARIANTS:
        for c, n in N_BASE.items():
            assert cnt(r, v, c) == (0, n, 0)
    assert r.geometry is None


def test_both_null_has_no_structure():
    r = ev(None, None)
    assert r.structure is None and r.geometry is None
    assert r.text["exact"]


@pytest.mark.parametrize(
    "raw",
    [
        js({"points": {}}),  # schema-invalid: a diagram needs a point
        js({"points": {"A": [10.5, 3]}}),  # schema-invalid: float coordinate
        '{"text": "Find $x$."}',  # missing key
        "not json",
        None,  # no prediction at all
    ],
)
def test_failed_predictions_count_as_empty_diagrams(raw):
    r = M.evaluate_example(tr(BASE), raw)
    assert cnt(r, "strict", "points") == (0, 0, N_BASE["points"])
    assert r.geometry is None


def test_text_without_a_usable_output_scores_one():
    r = M.evaluate_example(tr(BASE), "garbage")
    assert r.text["ned"] == r.text["ned_raw"] == 1.0 and not r.text["has_text"]
    r = M.evaluate_example(tr(BASE), '{"text": 5, "diagram": null}')
    assert r.text["ned"] == 1.0


def test_text_is_scored_even_when_the_diagram_is_invalid():
    r = M.evaluate_example(tr(BASE, "Find $x + 1$."), js({"points": {}}, "Find $x+1$."))
    assert r.parsed.report.stage == "invalid"
    assert r.text["ned"] == 0.0 and r.text["ned_raw"] > 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_summary_pools_counts_across_examples():
    gt = {"points": {"A": [10, 10], "B": [90, 10], "C": [50, 80], "D": [50, 10]},
          "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["C", "D"]]}  # fmt: skip
    pred = {"points": {"A": [10, 10], "B": [90, 10], "C": [50, 80], "E": [15, 60]},
            "segments": [["A", "B"], ["B", "C"], ["C", "E"]]}  # fmt: skip
    S = M.summarize([ev(gt, pred), ev(gt, gt)])
    pts, seg = S["structure"]["strict"]["points"], S["structure"]["strict"]["segments"]
    assert (pts["tp"], pts["fp"], pts["fn"]) == (7, 1, 1)
    assert pts["f1"] == pytest.approx(14 / 16)
    assert (seg["tp"], seg["fp"], seg["fn"]) == (6, 1, 2)
    assert seg["f1"] == pytest.approx(12 / 15)
    assert S["structure"]["strict"]["all"]["tp"] == 13
    assert S["geometry"]["coverage"] == pytest.approx(7 / 8)


def test_summary_rates_and_detection_confusion():
    results = [ev(BASE, BASE), ev(BASE, None), ev(None, None), ev(None, "garbage")]
    S = M.summarize(results)
    assert S["n"] == 4 and S["n_gt_diagram"] == 2 and S["n_gt_null"] == 2
    assert S["parse"]["json_parse_rate"] == 0.75
    assert S["parse"]["schema_valid_rate"] == 0.75
    assert S["parse"]["failures"] == {"no_json": 1}
    assert S["detection"]["confusion"] == {
        "diagram": {"diagram": 1, "null": 1, "failed": 0},
        "null": {"diagram": 0, "null": 1, "failed": 1},
    }
    assert S["detection"]["accuracy"] == 0.5
    assert S["text"]["ned"] == pytest.approx(0.25)  # the failure counts as 1.0
    assert S["text"]["ned_with_text"] == 0.0
    assert S["structure"]["n"] == 2 and S["structure_both_valid"]["n"] == 1


def test_fenced_counts_for_parse_rate_but_not_strict_rate():
    S = M.summarize([ev(BASE, f"```json\n{js(BASE)}\n```"), ev(BASE, BASE)])
    assert S["parse"]["json_parse_rate"] == 1.0
    assert S["parse"]["json_parse_rate_strict"] == 0.5


def test_ratios_without_a_denominator_are_none():
    S = M.summarize([ev(None, None)])
    assert S["structure"]["n"] == 0
    assert S["structure"]["strict"]["points"]["f1"] is None
    assert S["geometry"]["error_mean"] is None and S["geometry"]["coverage"] is None
    assert M.summarize([])["parse"]["json_parse_rate"] is None


def test_counts_conventions():
    assert M.Counts(0, 0, 0).f1 is None
    c = M.Counts(0, 3, 0)
    assert c.f1 == 0.0 and c.precision == 0.0 and c.recall is None


@pytest.mark.parametrize(
    ("err", "kind"),
    [
        (
            "diagram.points.A.0: Input should be a valid integer, got a number with a "
            "fractional part",
            "diagram.points: not an integer",
        ),  # fmt: skip
        (
            "diagram: Value error, segment ['A', 'X'] references unknown point 'X'",
            "diagram: unknown point",
        ),  # fmt: skip
        ("diagram: Field required", "diagram: missing key"),
        ("caption: Extra inputs are not permitted", "caption: extra key"),
        ("<root>: Input should be a valid dictionary", "<root>: not an object"),
    ],
)
def test_schema_error_kind(err, kind):
    assert M.schema_error_kind(err) == kind


def test_schema_errors_are_bucketed_in_the_summary():
    S = M.summarize([ev(BASE, {"points": {"A": [1.5, 2]}}), ev(BASE, {"points": {"A": [1, 2]},
                     "segments": [["A", "X"]]})])  # fmt: skip
    assert S["parse"]["schema_errors"] == {
        "diagram.points: not an integer": 1,
        "diagram: unknown point": 1,
    }


def test_summary_by_family():
    results = [
        M.evaluate_example(tr(BASE), js(BASE), meta={"family": "a"}),
        M.evaluate_example(tr(None), "garbage", meta={"family": "b"}),
    ]
    by = M.summarize_by(results, "family")
    assert by["a"]["f1_strict"] == 1.0 and by["a"]["text_ned"] == 0.0
    assert by["b"]["f1_strict"] is None and by["b"]["text_ned"] == 1.0


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_ground_truth_as_prediction_is_perfect_on_generated_samples():
    samples = [make_sample(5, "oracle", i) for i in range(40)]
    S = M.summarize(
        [M.evaluate_example(s.transcription, to_target_json(s.transcription)) for s in samples]
    )
    assert S["parse"]["json_parse_rate_strict"] == S["parse"]["schema_valid_rate"] == 1.0
    assert S["detection"]["accuracy"] == 1.0
    assert S["text"]["ned"] == 0.0 and S["text"]["exact"] == 1.0
    for v in M.VARIANTS:
        for c in (*M.CATEGORIES, "all"):
            assert S["structure"][v][c]["f1"] in (1.0, None), (v, c)
        assert S["structure"][v]["all"]["f1"] == 1.0
    g = S["geometry"]
    assert g["error_mean"] == g["strict_error_mean"] == g["raw_error_mean"] == 0.0
    assert g["coverage"] == 1.0 and g["scale_median"] == pytest.approx(1.0)


def test_point_error_grows_with_coordinate_noise():
    rng = np.random.default_rng(0)
    gt = tr(ASYM)
    means = []
    for sigma in (0.0, 2.0, 6.0):
        errs: list[float] = []
        for _ in range(20):
            noise = rng.normal(0.0, sigma, (len(ASYM["points"]), 2))
            pred = copy.deepcopy(ASYM)
            pred["points"] = {
                k: [int(np.clip(round(v[0] + dx), 0, 100)), int(np.clip(round(v[1] + dy), 0, 100))]
                for (k, v), (dx, dy) in zip(ASYM["points"].items(), noise, strict=True)
            }
            errs += M.evaluate_example(gt, js(pred)).geometry.errors
        means.append(float(np.mean(errs)))
    assert means[0] == 0.0 < means[1] < means[2]


def test_run_eval_cli(tmp_path):
    from mathvlm.eval import run_eval
    from mathvlm.generate.build import build

    data = tmp_path / "data"
    build("cli", 6, 21, data)
    out = tmp_path / "oracle"
    argv = ["--split", "cli", "--data", str(data)]
    assert run_eval.main([*argv, "--oracle", "--out", str(out), "--sbs", "2"]) == 0
    S = json.loads((out / "summary.json").read_text())
    assert S["name"] == "oracle" and S["parse"]["schema_valid_rate"] == 1.0
    assert S["text"]["ned"] == 0.0 and S["structure"]["geometric"]["all"]["f1"] == 1.0
    rows = [json.loads(line) for line in (out / "per_example.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and (out / "summary.txt").exists()
    assert len(list((out / "side_by_side").glob("*.png"))) == 2

    # one perfect prediction, one garbage, four missing
    pred = tmp_path / "zeroshot" / "predictions.jsonl"
    pred.parent.mkdir()
    pred.write_text(
        json.dumps({"id": rows[0]["id"], "output": rows[0]["output"]}) + "\n"
        + json.dumps({"id": rows[1]["id"], "output": "no idea", "truncated": False}) + "\n"
    )  # fmt: skip
    out2 = tmp_path / "pred_eval"
    assert run_eval.main([*argv, "--pred", str(pred), "--out", str(out2), "--sbs", "0"]) == 0
    S = json.loads((out2 / "summary.json").read_text())
    assert S["name"] == "zeroshot"
    assert S["parse"]["failures"] == {"missing": 4, "no_json": 1}
    assert S["parse"]["json_parse_rate"] == pytest.approx(1 / 6)


def test_load_predictions_rejects_bad_files(tmp_path):
    from mathvlm.eval.run_eval import load_predictions

    p = tmp_path / "p.jsonl"
    p.write_text('{"id": "a", "output": "x"}\n{"id": "a", "output": "y"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(p)
    p.write_text('{"id": "a", "output": 5}\n')
    with pytest.raises(ValueError, match="string"):
        load_predictions(p)
    p.write_text('{"id": "a", "output": null}\n\n')
    assert load_predictions(p) == {"a": {"id": "a", "output": None}}


def test_report_handles_failures_and_empty_runs():
    from PIL import Image

    from mathvlm.eval import report

    page = Image.new("RGB", (300, 400), "white")
    for raw in ('{"text": "$\\frac12$", "diagram": {"points": {}}}', "no idea", js(BASE)):
        im = report.side_by_side(M.evaluate_example(tr(BASE), raw, id="x"), page)
        assert im.width > 2 * page.width and im.height > page.height
    assert report._visible("a\x0cb\nc") == "a\\x0cb\nc"
    text = report.format_summary(M.summarize([]))
    assert "0 samples" in text
