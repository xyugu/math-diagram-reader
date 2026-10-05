"""Tests for real-textbook conventions in generated samples (src/mathvlm/generate/realism.py)."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from mathvlm import schema as S
from mathvlm.config import get_config
from mathvlm.generate import realism as R
from mathvlm.generate.build import _streams, make_sample
from mathvlm.generate.scenes import Namer, Scene
from mathvlm.render import render_page

RC = get_config().generate.realism
ALWAYS = RC.model_copy(update={"ray_prob": 1.0, "max_rays": 1, "numbered_angles_prob": 1.0})


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def namer() -> Namer:
    return Namer(rng(1), get_config().generate)


def triangle_scene() -> Scene:
    sc = Scene("t", "t")
    sc.points = {"A": (0.0, 0.0), "B": (10.0, 0.0), "C": (4.0, 8.0)}
    sc.segments = [("A", "B"), ("B", "C"), ("C", "A")]
    return sc


# ---------------------------------------------------------------------------
# Rays
# ---------------------------------------------------------------------------


def test_add_ray_extends_one_segment_to_an_arrow_tip():
    sc = triangle_scene()
    assert R.add_rays(sc, rng(3), namer(), ALWAYS) == 1
    (arrow,) = [m for m in sc.marks if m["type"] == "arrow"]
    tip, base = arrow["at"], arrow["from"]
    assert S.is_unnamed(tip) and base in "ABC"
    (seg,) = [s for s in sc.segments if tip in s]
    tail = seg[0] if seg[1] == tip else seg[1]
    assert (base, frozenset((tail, tip))) in sc.on_segment  # the old end now lies on it
    assert len(sc.segments) == 3  # the segment grew, it was not duplicated
    # collinear and beyond the old endpoint
    b, t, a = (np.array(sc.points[n]) for n in (base, tip, tail))
    u, w = b - a, t - a
    assert abs(u[0] * w[1] - u[1] * w[0]) < 1e-9
    assert np.linalg.norm(w) > np.linalg.norm(u)


def test_rays_never_cross_other_segments_or_circles():
    sc = triangle_scene()
    sc.points["O"] = (14.0, 0.0)
    sc.circles = [("O", 2.5)]  # sits just past B on the line AB
    # extending AB past B would run into the circle: not clear
    assert not R._ray_is_clear(sc, "B", "A", (18.0, 0.0), 10.0)
    # extending CA past A goes away from everything: clear
    assert R._ray_is_clear(sc, "A", "C", (-2.0, -4.0), 10.0)
    # a ray from C through B would cross nothing but end too close to O: not clear
    assert not R._ray_is_clear(sc, "B", "C", (14.2, -0.3), 10.0)


def test_disabled_rays_add_nothing():
    sc = triangle_scene()
    off = RC.model_copy(update={"ray_prob": 0.0})
    assert R.add_rays(sc, rng(0), namer(), off) == 0 and sc.marks == []


# ---------------------------------------------------------------------------
# Numbered angles and hidden names
# ---------------------------------------------------------------------------

X = {
    "points": {"A": [10, 10], "B": [90, 90], "C": [10, 90], "D": [90, 10], "E": [50, 50]},
    "segments": [["A", "B"], ["C", "D"]],
    "circles": [],
    "marks": [],
    "labels": [],
}  # two lines crossing at E


def x_diagram() -> dict:
    import copy

    return copy.deepcopy(X)


def test_number_angles_around_an_intersection():
    d = x_diagram()
    assert R.number_angles(d, rng(0), ALWAYS) == 4
    assert sorted(lb["text"] for lb in d["labels"]) == ["1", "2", "3", "4"]
    assert all(lb["angle"][1] == "E" for lb in d["labels"])
    t = S.validate_output({"text": "", "diagram": d}).value
    assert S.consistency_issues(t.diagram) == []
    sectors = {S.canonical_angle(tuple(lb["angle"])) for lb in d["labels"]}
    assert len(sectors) == 4  # four different sectors


def test_number_angles_skips_a_vertex_that_already_has_an_angle_label():
    d = x_diagram()
    d["labels"] = [{"angle": ["A", "E", "C"], "text": "40^\\circ"}]
    assert R.number_angles(d, rng(0), ALWAYS) == 0


@pytest.mark.parametrize("mode", ["all", "some", "none"])
def test_hide_names(mode):
    d = x_diagram()
    d["labels"] = [{"on": ["A", "B"], "text": "5"}]
    cfg = RC.model_copy(update={"hide_names": {mode: 1.0}, "hide_some_prob": 0.5})
    assert R.hide_names(d, rng(2), cfg) == mode
    named = [n for n in d["points"] if not S.is_unnamed(n)]
    if mode == "all":
        assert named == []
    elif mode == "none":
        assert sorted(named) == list("ABCDE")
    t = S.validate_output({"text": "", "diagram": S.renumber_unnamed(d)})
    assert t.ok, t.errors  # labels follow the renamed points


# ---------------------------------------------------------------------------
# In generated samples
# ---------------------------------------------------------------------------


def test_adding_the_realism_stream_left_the_others_unchanged():
    old = [np.random.default_rng(s) for s in np.random.SeedSequence(7).spawn(4)]
    new = _streams(7)
    assert len(new) == 5
    for a, b in zip(old, new[:4], strict=True):
        assert np.array_equal(a.random(6), b.random(6))


@pytest.fixture(scope="module")
def samples():
    return [make_sample(0, "realism_test", i) for i in range(250)]


def test_generated_mix_follows_the_config(samples):
    diag = [s for s in samples if s.family != "no_diagram"]
    # families with a question of their own become figure-only pages at the configured rate
    questions = [s for s in diag if s.family != "line_crossings"]
    assert abs(sum(s.meta["diagram_only"] for s in questions) / len(questions)
               - RC.diagram_only_prob) < 0.1  # fmt: skip
    only = [s for s in diag if s.meta["diagram_only"]]
    assert all(s.transcription.text == "" for s in only)
    assert all(s.transcription.text for s in diag if not s.meta["diagram_only"])
    rays = [s for s in diag if s.meta["rays"]]
    assert 0.15 < len(rays) / len(diag) < RC.ray_prob + 0.1
    for s in rays:
        arrows = [m for m in s.transcription.diagram.marks if isinstance(m, S.ArrowMark)]
        own = s.family == "line_crossings"  # lines end in arrowheads of their own
        assert (len(arrows) >= s.meta["rays"]) if own else (len(arrows) == s.meta["rays"])
        assert all(S.is_unnamed(m.at) for m in arrows)
    hidden = collections.Counter(s.meta["hidden_names"] for s in only)
    assert set(hidden) <= set(RC.hide_names) and hidden["all"] > 0


def test_generated_samples_stay_valid_and_consistent(samples):
    for s in samples:
        if s.transcription.diagram is not None:
            assert S.consistency_issues(s.transcription.diagram) == [], s.id
        # questions never name a hidden point
        if s.meta.get("diagram_only") is False:
            assert "_P" not in s.transcription.text


def test_figure_only_page_renders_without_a_text_block(samples):
    s = next(s for s in samples if s.meta.get("diagram_only"))
    page = render_page(s.transcription, s.style)
    assert page.panel_box is not None
    assert page.panel_box[1] < 0.6 * page.image.height  # the panel starts near the top


# ---------------------------------------------------------------------------
# Dense value labels
# ---------------------------------------------------------------------------

DENSE = RC.model_copy(update={"dense_labels_prob": 1.0, "max_extra_lengths": 3,
                              "units_prob": 0.0, "arc_label_prob": 1.0})  # fmt: skip

# AB with M on it (AM = 30, MB = 50), a ray from M up to C; circle O through P, Q, R.
SPLIT = {
    "points": {"A": [10, 50], "M": [40, 50], "B": [90, 50], "C": [40, 10],
               "O": [70, 80], "P": [85, 80], "Q": [70, 65], "R": [55, 80]},
    "segments": [["A", "B"], ["M", "C"]],
    "circles": [{"center": "O", "radius": 15}],
    "marks": [], "labels": [],
}  # fmt: skip


def split_diagram() -> dict:
    import copy

    return copy.deepcopy(SPLIT)


def test_dense_lengths_go_on_pieces_in_one_scale():
    d = split_diagram()
    added = R.densify_labels(d, rng(4), DENSE)
    lengths = {frozenset(lb["on"]): float(lb["text"]) for lb in d["labels"] if "on" in lb}
    assert added["lengths"] == len(lengths) >= 1
    assert frozenset("AB") not in lengths  # a split segment is labeled piece by piece
    if {frozenset("AM"), frozenset("MB")} <= set(lengths):
        assert lengths[frozenset("MB")] / lengths[frozenset("AM")] == pytest.approx(
            50 / 30, rel=0.06
        )
    t = S.validate_output({"text": "", "diagram": d})
    assert t.ok and S.consistency_issues(t.value.diagram) == []


def test_dense_angles_cover_the_sectors_at_one_vertex():
    d = split_diagram()
    R.densify_labels(d, np.random.default_rng(0), DENSE.model_copy(update={"max_extra_lengths": 1}))
    angles = [lb for lb in d["labels"] if "angle" in lb]
    if angles:  # chance(0.7) to label a vertex
        assert {lb["angle"][1] for lb in angles} == {"M"}  # the only vertex with 3 rays
        values = sorted(int(lb["text"].split("^")[0]) for lb in angles if lb["text"][0] != "x")
        assert all(v == 90 for v in values)  # MC is perpendicular to AB


def test_dense_arc_label_is_the_central_angle():
    d = split_diagram()
    added = R.densify_labels(d, rng(1), DENSE)
    arcs = [lb for lb in d["labels"] if "arc" in lb]
    assert added["arcs"] == len(arcs) == 1
    assert arcs[0]["text"] == "90^\\circ"  # neighbours on the circle are a quarter apart


def test_dense_units_are_text_that_the_metric_ignores():
    from mathvlm.eval.normalize import normalize_label

    d = split_diagram()
    R.densify_labels(d, rng(4), DENSE.model_copy(update={"units_prob": 1.0, "units": ["cm"]}))
    texts = [lb["text"] for lb in d["labels"] if "on" in lb]
    assert texts and all(t.endswith("\\text{ cm}") for t in texts)
    assert all(normalize_label(t) == t.split("\\text")[0] for t in texts)


def test_dense_labels_can_be_off():
    d = split_diagram()
    off = DENSE.model_copy(update={"dense_labels_prob": 0.0})
    assert R.densify_labels(d, rng(0), off) == {"lengths": 0, "angles": 0, "arcs": 0}
    assert d["labels"] == []


# ---------------------------------------------------------------------------
# Line-only figures
# ---------------------------------------------------------------------------


def test_line_crossings_are_figure_only_pages_of_lines():
    from mathvlm.generate.scenes import line_crossings

    gcfg = get_config().generate
    kinds = collections.Counter()
    for i in range(40):
        sc = line_crossings(
            np.random.default_rng(i), Namer(np.random.default_rng(100 + i), gcfg), gcfg
        )
        kinds[sc.kind] += 1
        assert sc.figure_only and not sc.circles and not sc.polygon
        # every intersection lies on at least two lines, and every arrow points outward
        meets = collections.Counter(p for p, _ in sc.on_segment)
        assert meets and all(n >= 2 for n in meets.values()) or sc.kind == "crossing"
        for m in (m for m in sc.marks if m["type"] == "arrow"):
            tip, base = np.array(sc.points[m["at"]]), np.array(sc.points[m["from"]])
            assert np.linalg.norm(tip - base) > 0.3
        if sc.kind == "parallel":
            (a, b), (c, d) = sc.segments[-3:-1] if len(sc.segments) == 3 else sc.segments[-4:-2]
            u, w = np.subtract(sc.points[b], sc.points[a]), np.subtract(sc.points[d], sc.points[c])
            assert abs(u[0] * w[1] - u[1] * w[0]) < 1e-9  # the two lines are parallel
    assert set(kinds) == {"crossing", "parallel"}


def test_generated_line_figures_have_no_text(samples):
    lines = [s for s in samples if s.family == "line_crossings"]
    assert lines
    for s in lines:
        assert s.transcription.text == "" and s.meta["diagram_only"]
        assert any(isinstance(m, S.ArrowMark) for m in s.transcription.diagram.marks) or any(
            not S.is_unnamed(n) for n in s.transcription.diagram.points
        )
