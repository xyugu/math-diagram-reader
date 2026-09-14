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
    only = [s for s in diag if s.meta["diagram_only"]]
    assert abs(len(only) / len(diag) - RC.diagram_only_prob) < 0.1
    assert all(s.transcription.text == "" for s in only)
    assert all(s.transcription.text for s in diag if not s.meta["diagram_only"])
    rays = [s for s in diag if s.meta["rays"]]
    assert 0.15 < len(rays) / len(diag) < RC.ray_prob + 0.1
    for s in rays:
        arrows = [m for m in s.transcription.diagram.marks if isinstance(m, S.ArrowMark)]
        assert len(arrows) == s.meta["rays"] and all(S.is_unnamed(m.at) for m in arrows)
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
