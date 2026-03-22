"""Tests for the frozen diagram schema (src/mathvlm/schema.py)."""

from __future__ import annotations

import copy
import json
import random

import pytest

from mathvlm import schema as S

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def full_diagram() -> dict:
    """A valid, geometrically consistent diagram using every v1.1 feature.

    Isosceles right triangle ABC (right angle at C, CA == CB). M is the midpoint of AB,
    so CM is both the altitude (right angle at M) and the bisector of angle ACB. CD is
    parallel to AB. The circumcircle is centered at M with radius MA = 40 and fits in
    the grid. _P1 is the unlabeled midpoint of AC.
    """
    return {
        "points": {
            "A": [10, 50],
            "B": [90, 50],
            "C": [50, 10],
            "M": [50, 50],
            "D": [90, 10],
            "_P1": [30, 30],
        },
        "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["C", "M"], ["C", "D"]],
        "circles": [{"center": "M", "radius": 40}],
        "marks": [
            {"type": "right_angle", "at": "C", "rays": ["A", "B"]},
            {"type": "right_angle", "at": "M", "rays": ["C", "B"]},
            {"type": "equal_length", "segments": [["A", "M"], ["M", "B"]]},
            {"type": "parallel", "segments": [["A", "B"], ["C", "D"]]},
            {"type": "equal_angle", "angles": [["A", "C", "M"], ["M", "C", "B"]]},
        ],
        "labels": [
            {"on": ["A", "C"], "text": "3"},
            {"angle": ["A", "B", "C"], "text": "45^\\circ"},
        ],
    }


def output(diagram: dict | None = None, text: str = "In $\\triangle ABC$, find $x$.") -> dict:
    return {"text": text, "diagram": diagram}


def ok(obj) -> S.Transcription:
    r = S.validate_output(obj)
    assert r.stage == "ok", r.errors
    return r.value


def invalid(obj) -> tuple[str, ...]:
    r = S.validate_output(obj)
    assert r.stage == "invalid", f"expected invalid, got {r.stage}"
    assert r.value is None
    assert r.errors
    return r.errors


def with_diagram(**overrides) -> dict:
    d = full_diagram()
    d.update(overrides)
    return output(d)


# ---------------------------------------------------------------------------
# Version and the happy path
# ---------------------------------------------------------------------------


def test_schema_version_is_1_1():
    assert S.SCHEMA_VERSION == "1.1"


def test_full_diagram_is_valid():
    t = ok(output(full_diagram()))
    assert t.diagram is not None
    assert len(t.diagram.marks) == 5


def test_null_diagram_is_valid():
    t = ok(output(None))
    assert t.diagram is None


def test_empty_text_is_valid_shape():
    # Empty text is a scoring problem, not a schema violation.
    ok(output(None, text=""))


def test_missing_list_keys_default_to_empty():
    t = ok(output({"points": {"A": [1, 2]}}))
    d = t.diagram
    assert d.segments == [] and d.circles == [] and d.marks == [] and d.labels == []


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


def test_missing_diagram_key_rejected():
    invalid({"text": "x"})


def test_missing_text_key_rejected():
    invalid({"diagram": None})


def test_extra_top_level_key_rejected():
    invalid({"text": "x", "diagram": None, "answer": "5"})


@pytest.mark.parametrize("bad_text", [None, 5, ["x"], {"md": "x"}])
def test_text_must_be_a_string(bad_text):
    invalid({"text": bad_text, "diagram": None})


def test_extra_key_in_diagram_rejected():
    invalid(with_diagram(angles=[]))


def test_extra_key_in_circle_rejected():
    invalid(with_diagram(circles=[{"center": "M", "radius": 40, "color": "red"}]))


def test_extra_key_in_mark_rejected():
    invalid(with_diagram(marks=[{"type": "right_angle", "at": "C", "rays": ["A", "B"], "x": 1}]))


def test_empty_points_rejected():
    errs = invalid(output({"points": {}}))
    assert any("points" in e for e in errs)


# ---------------------------------------------------------------------------
# Point names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["A", "Z", "A'", "A''", "O_1", "M_12", "_P0", "_P1", "_P10"])
def test_valid_point_names(name):
    ok(output({"points": {name: [5, 5]}}))


@pytest.mark.parametrize(
    "name",
    ["a", "AB", "A1", "A'''", "A_123", "A_", "_P", "_P01", "_Q1", "_p1", "P_", "", " A", "∠A"],
)
def test_invalid_point_names(name):
    invalid(output({"points": {name: [5, 5]}}))


def test_is_unnamed():
    assert S.is_unnamed("_P3")
    assert not S.is_unnamed("P")
    assert not S.is_unnamed("_Q3")


# ---------------------------------------------------------------------------
# Coordinates and radius
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xy", [[0, 0], [100, 100], [0, 100], [37, 64]])
def test_coords_in_range_accepted(xy):
    ok(output({"points": {"A": xy}}))


@pytest.mark.parametrize(
    "xy",
    [
        [-1, 5],
        [5, 101],
        [10.0, 5],  # integral float: still not an int on the grid
        [10.5, 5],
        [True, 5],  # bool is an int subclass in Python; must not slip through
        ["10", 5],
        [10],
        [10, 20, 30],
        {"x": 10, "y": 20},
        None,
    ],
)
def test_bad_coords_rejected(xy):
    invalid(output({"points": {"A": xy}}))


@pytest.mark.parametrize("r", [5, 5.5, 0.1, 100])
def test_valid_radius(r):
    ok(output({"points": {"O": [50, 50]}, "circles": [{"center": "O", "radius": r}]}))


@pytest.mark.parametrize("r", [0, -3, 100.5, True, "5", None])
def test_invalid_radius(r):
    invalid(output({"points": {"O": [50, 50]}, "circles": [{"center": "O", "radius": r}]}))


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def _assert_mentions(errs, needle):
    assert any(needle in e for e in errs), errs


def test_segment_to_unknown_point():
    _assert_mentions(invalid(with_diagram(segments=[["A", "Q"]])), "'Q'")


def test_circle_center_unknown():
    _assert_mentions(invalid(with_diagram(circles=[{"center": "O", "radius": 5}])), "'O'")


@pytest.mark.parametrize(
    "mark",
    [
        {"type": "right_angle", "at": "Q", "rays": ["A", "B"]},
        {"type": "right_angle", "at": "C", "rays": ["A", "Q"]},
        {"type": "equal_length", "segments": [["A", "B"], ["C", "Q"]]},
        {"type": "parallel", "segments": [["A", "Q"], ["C", "D"]]},
        {"type": "equal_angle", "angles": [["A", "C", "M"], ["M", "C", "Q"]]},
    ],
)
def test_mark_references_unknown_point(mark):
    _assert_mentions(invalid(with_diagram(marks=[mark])), "'Q'")


@pytest.mark.parametrize(
    "label",
    [{"on": ["A", "Q"], "text": "3"}, {"angle": ["A", "Q", "C"], "text": "40^\\circ"}],
)
def test_label_references_unknown_point(label):
    _assert_mentions(invalid(with_diagram(labels=[label])), "'Q'")


def test_all_reference_errors_reported_together():
    errs = invalid(with_diagram(segments=[["A", "Q"]], circles=[{"center": "R", "radius": 3}]))
    joined = " ".join(errs)
    assert "'Q'" in joined and "'R'" in joined


# ---------------------------------------------------------------------------
# Degenerate and duplicate structures
# ---------------------------------------------------------------------------


def test_self_segment_rejected():
    invalid(with_diagram(segments=[["A", "A"]]))


def test_duplicate_segment_reversed_rejected():
    invalid(with_diagram(segments=[["A", "B"], ["B", "A"]]))


def test_duplicate_circle_rejected():
    c = {"center": "M", "radius": 40}
    invalid(with_diagram(circles=[c, dict(c)]))


def test_concentric_circles_allowed():
    ok(with_diagram(circles=[{"center": "M", "radius": 40}, {"center": "M", "radius": 20}]))


def test_duplicate_mark_up_to_ray_order_rejected():
    invalid(
        with_diagram(
            marks=[
                {"type": "right_angle", "at": "C", "rays": ["A", "B"]},
                {"type": "right_angle", "at": "C", "rays": ["B", "A"]},
            ]
        )
    )


def test_two_labels_on_same_segment_rejected():
    invalid(with_diagram(labels=[{"on": ["A", "C"], "text": "3"}, {"on": ["C", "A"], "text": "4"}]))


def test_right_angle_without_rays_rejected():
    """The v1 form. Documents the approved v1 -> v1.1 change."""
    invalid(with_diagram(marks=[{"type": "right_angle", "at": "C"}]))


@pytest.mark.parametrize(
    "mark",
    [
        {"type": "right_angle", "at": "C", "rays": ["C", "B"]},  # vertex in rays
        {"type": "right_angle", "at": "C", "rays": ["A", "A"]},  # same ray twice
        {"type": "equal_length", "segments": [["A", "B"]]},  # needs >= 2
        {"type": "equal_length", "segments": [["A", "B"], ["B", "A"]]},  # duplicate
        {"type": "equal_length", "segments": [["A", "A"], ["B", "C"]]},  # self-segment
        {"type": "parallel", "segments": [["A", "B"]]},
        {"type": "equal_angle", "angles": [["A", "C", "M"]]},  # needs >= 2
        {"type": "equal_angle", "angles": [["A", "C", "M"], ["M", "C", "A"]]},  # same angle
        {"type": "equal_angle", "angles": [["A", "C", "C"], ["M", "C", "B"]]},  # degenerate
        {"type": "perpendicular", "segments": [["A", "B"], ["C", "M"]]},  # unknown type
        {"at": "C", "rays": ["A", "B"]},  # missing type
    ],
)
def test_bad_marks_rejected(mark):
    invalid(with_diagram(marks=[mark]))


@pytest.mark.parametrize(
    "label",
    [
        {"on": ["A", "C"], "angle": ["A", "B", "C"], "text": "3"},  # both shapes
        {"text": "3"},  # neither
        {"on": ["A", "C"], "text": ""},  # empty text
        {"on": ["A", "C"]},  # no text
        {"angle": ["A", "B"], "text": "40^\\circ"},  # two-point angle
        {"angle": ["A", "B", "B"], "text": "40^\\circ"},  # degenerate angle
    ],
)
def test_bad_labels_rejected(label):
    invalid(with_diagram(labels=[label]))


# ---------------------------------------------------------------------------
# JSON string entry point
# ---------------------------------------------------------------------------


def test_json_ok():
    r = S.validate_output_json(json.dumps(output(full_diagram())))
    assert r.ok and r.stage == "ok"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        '{"text": "x", "diagram": null',  # truncated
        '{"text": "x", "diagram": null,}',  # trailing comma
        '```json\n{"text": "x", "diagram": null}\n```',  # fences: stripped by eval, not here
    ],
)
def test_json_parse_errors(raw):
    r = S.validate_output_json(raw)
    assert r.stage == "parse_error"
    assert not r.ok


def test_json_nan_rejected():
    r = S.validate_output_json('{"text": "x", "diagram": {"points": {"A": [NaN, 1]}}}')
    assert r.stage == "parse_error"


def test_json_duplicate_point_key_is_invalid():
    raw = '{"text": "x", "diagram": {"points": {"A": [1, 2], "A": [3, 4]}}}'
    r = S.validate_output_json(raw)
    assert r.stage == "invalid"
    assert "duplicate" in r.errors[0]


def test_json_valid_but_wrong_shape_is_invalid_not_parse_error():
    r = S.validate_output_json('{"text": "x"}')
    assert r.stage == "invalid"


def test_error_messages_locate_the_field():
    r = S.validate_output(output({"points": {"A": [5, 500]}}))
    assert any(e.startswith("diagram.points.A") for e in r.errors), r.errors


def test_output_json_schema_exposes_both_keys():
    js = S.output_json_schema()
    assert set(js["properties"]) == {"text", "diagram"}
    assert set(js["required"]) == {"text", "diagram"}


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


def scrambled(d: dict, seed: int) -> dict:
    """Same diagram, with every order the schema treats as meaningless shuffled."""
    rng = random.Random(seed)
    d = copy.deepcopy(d)
    items = list(d["points"].items())
    rng.shuffle(items)
    d["points"] = dict(items)
    d["segments"] = [s[::-1] if rng.random() < 0.5 else s for s in d["segments"]]
    rng.shuffle(d["segments"])
    for m in d["marks"]:
        if m["type"] == "right_angle":
            m["rays"] = m["rays"][::-1]
        elif m["type"] in ("equal_length", "parallel"):
            m["segments"] = [s[::-1] for s in m["segments"]]
            rng.shuffle(m["segments"])
        else:
            m["angles"] = [a[::-1] for a in m["angles"]]  # reversal keeps the vertex
            rng.shuffle(m["angles"])
    rng.shuffle(d["marks"])
    for lb in d["labels"]:
        key = "on" if "on" in lb else "angle"
        lb[key] = lb[key][::-1]
    rng.shuffle(d["labels"])
    return d


@pytest.mark.parametrize("seed", range(5))
def test_canonical_form_ignores_meaningless_order(seed):
    a = ok(output(full_diagram())).diagram
    b = ok(output(scrambled(full_diagram(), seed))).diagram
    assert S.canonicalize(a) == S.canonicalize(b)


def test_canonical_angle_keeps_vertex_distinct():
    # angle A-C-B and angle C-A-B share points but are different angles
    assert S.canonical_angle(("A", "C", "B")) != S.canonical_angle(("C", "A", "B"))
    assert S.canonical_angle(("A", "C", "B")) == S.canonical_angle(("B", "C", "A"))


def test_canonical_form_detects_real_differences():
    base = S.canonicalize(ok(output(full_diagram())).diagram)

    moved = full_diagram()
    moved["marks"][0] = {"type": "right_angle", "at": "A", "rays": ["C", "B"]}
    assert S.canonicalize(ok(output(moved)).diagram).marks != base.marks

    relabeled = full_diagram()
    relabeled["labels"][0] = {"on": ["A", "C"], "text": "4"}
    assert S.canonicalize(ok(output(relabeled)).diagram).labels != base.labels


def test_canonical_mark_shapes():
    m = S.RightAngleMark(type="right_angle", at="C", rays=("A", "B"))
    assert S.canonical_mark(m) == ("right_angle", ("C", frozenset({"A", "B"})))
    e = S.EqualLengthMark(type="equal_length", segments=[("A", "B"), ("C", "D")])
    assert S.canonical_mark(e) == (
        "equal_length",
        frozenset({frozenset({"A", "B"}), frozenset({"C", "D"})}),
    )


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_target_json_is_byte_identical_under_reordering(seed):
    a = S.to_target_json(ok(output(full_diagram())))
    b = S.to_target_json(ok(output(scrambled(full_diagram(), seed))))
    assert a == b


def test_target_json_round_trips():
    t = ok(output(full_diagram()))
    s1 = S.to_target_json(t)
    r = S.validate_output_json(s1)
    assert r.ok, r.errors
    assert S.to_target_json(r.value) == s1
    assert S.canonicalize(r.value.diagram) == S.canonicalize(t.diagram)


def test_target_json_always_has_all_five_keys_in_order():
    t = ok(output({"points": {"A": [1, 2]}}))
    obj = json.loads(S.to_target_json(t))
    assert list(obj) == ["text", "diagram"]
    assert list(obj["diagram"]) == ["points", "segments", "circles", "marks", "labels"]


def test_target_json_null_diagram():
    assert S.to_target_json(ok(output(None, text="Solve $x+1=2$."))) == (
        '{"text":"Solve $x+1=2$.","diagram":null}'
    )


def test_target_json_is_compact_and_keeps_latex():
    s = S.to_target_json(ok(output(None, text="$\\angle ABC = 40^\\circ$")))
    assert ", " not in s and ": " not in s
    assert json.loads(s)["text"] == "$\\angle ABC = 40^\\circ$"


def test_point_order_is_natural():
    pts = {"_P10": [1, 1], "B": [2, 2], "_P2": [3, 3], "A'": [4, 4], "A": [5, 5], "A_1": [6, 6]}
    obj = json.loads(S.to_target_json(ok(output({"points": pts}))))
    assert list(obj["diagram"]["points"]) == ["A", "A'", "A_1", "B", "_P2", "_P10"]


def test_serialized_angles_keep_vertex_in_middle():
    obj = json.loads(S.to_target_json(ok(output(full_diagram()))))
    ang = obj["diagram"]["labels"][-1]["angle"]
    assert ang[1] == "B"
    for a in next(m for m in obj["diagram"]["marks"] if m["type"] == "equal_angle")["angles"]:
        assert a[1] == "C"


# ---------------------------------------------------------------------------
# Soft geometric consistency
# ---------------------------------------------------------------------------


def diag(obj: dict) -> S.Diagram:
    return ok(output(obj)).diagram


def test_full_diagram_is_consistent():
    assert S.consistency_issues(diag(full_diagram())) == []


def test_collinear_subsegment_counts_as_drawn():
    # AM and MB are pieces of the drawn segment AB.
    d = diag(
        {
            "points": {"A": [10, 50], "M": [50, 50], "B": [90, 50]},
            "segments": [["A", "B"]],
            "marks": [{"type": "equal_length", "segments": [["A", "M"], ["M", "B"]]}],
        }
    )
    assert S.consistency_issues(d) == []


def test_right_angle_that_is_not_90_is_flagged():
    d = full_diagram()
    d["points"]["C"] = [30, 30]  # angle ACB no longer 90
    d["marks"] = [{"type": "right_angle", "at": "C", "rays": ["A", "B"]}]
    d["labels"] = []
    issues = S.consistency_issues(diag(d))
    assert any("not ~90" in i for i in issues), issues


def test_mark_on_undrawn_segment_is_flagged():
    d = full_diagram()
    d["segments"] = [s for s in d["segments"] if s != ["C", "D"]]
    issues = S.consistency_issues(diag(d))
    assert any("CD is not drawn" in i for i in issues), issues


def test_unequal_equal_length_is_flagged():
    d = diag(
        {
            "points": {"A": [10, 50], "M": [40, 50], "B": [90, 50]},
            "segments": [["A", "B"]],
            "marks": [{"type": "equal_length", "segments": [["A", "M"], ["M", "B"]]}],
        }
    )
    assert any("differ" in i for i in S.consistency_issues(d))


def test_non_parallel_is_flagged():
    d = diag(
        {
            "points": {"A": [10, 10], "B": [90, 10], "C": [10, 50], "D": [90, 80]},
            "segments": [["A", "B"], ["C", "D"]],
            "marks": [{"type": "parallel", "segments": [["A", "B"], ["C", "D"]]}],
        }
    )
    assert any("parallel" in i for i in S.consistency_issues(d))


def test_unequal_equal_angle_is_flagged():
    d = full_diagram()
    d["points"]["M"] = [30, 50]  # still on AB, but CM no longer bisects ACB
    d["marks"] = [m for m in d["marks"] if m["type"] == "equal_angle"]
    d["labels"] = []
    issues = S.consistency_issues(diag(d))
    assert any("equal_angle" in i and "differ" in i for i in issues), issues


def test_label_on_undrawn_segment_is_flagged():
    d = full_diagram()
    d["labels"] = [{"on": ["A", "D"], "text": "7"}]
    assert any("AD is not drawn" in i for i in S.consistency_issues(diag(d)))


def test_circle_outside_grid_is_flagged():
    d = diag({"points": {"O": [10, 50]}, "circles": [{"center": "O", "radius": 30}]})
    assert any("outside the grid" in i for i in S.consistency_issues(d))
