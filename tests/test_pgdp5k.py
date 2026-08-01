"""Tests for the PGDP5K converter (src/mathvlm/real/pgdp5k.py) on hand-made annotations
whose correct conversion is worked out by hand, plus a smoke test on the real files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathvlm import schema as S
from mathvlm.real.pgdp5k import convert_one

# Triangle ABC (pixels, y down) with the altitude BD onto AC, and AC extended past C to
# an unlabeled line end K. D is the midpoint of AC.
POS = {"A": [20, 180], "C": [220, 180], "B": [120, 30], "D": [120, 180], "K": [260, 180]}


def text_sym(sid, cls, content, x, y):
    return {"id": sid, "sym_class": "text", "text_class": cls, "text_content": content,
            "bbox": [x - 5, y - 5, 10, 10]}  # fmt: skip


def triangle(forms: list[str] | None = None, symbols: list | None = None):
    lf = {
        "point_instances": list(POS),
        "line_instances": [],
        "circle_instances": [],
        "diagram_logic_forms": forms
        if forms is not None
        else [
            "PointLiesOnLine(D, Line(A, C))",
            "Perpendicular(Line(B, D), Line(A, C))",
            "Equals(LengthOf(Line(A, D)), LengthOf(Line(D, C)))",
            "Equals(LengthOf(Line(A, B)), 5\\sqrt{2})",
            "Equals(MeasureOf(Angle(A, B, D)), 4y-10)",
            "Equals(MeasureOf(Angle(D, B, C)), 45)",
            "Equals(MeasureOf(Angle(B, A, C)), MeasureOf(angle 1))",
            "Equals(PerimeterOf(Circle(O)), 15)",
        ],
        "point_positions": POS,
    }
    lines = [("l0", "A", "K"), ("l1", "A", "B"), ("l2", "B", "C"), ("l3", "B", "D")]
    ann = {
        "width": 280,
        "height": 200,
        "geos": {
            "points": [{"id": f"p{i}", "loc": [v]} for i, v in enumerate(POS.values())],
            "lines": [{"id": i, "loc": [POS[a], POS[b]]} for i, a, b in lines],
            "circles": [],
        },
        "symbols": symbols
        if symbols is not None
        else [
            *(text_sym(f"t{n}", "point", n, *POS[n]) for n in "ABCD"),
            {
                "id": "s_perp",
                "sym_class": "perpendicular",
                "text_class": "others",
                "text_content": None,
                "bbox": [124, 166, 10, 10],
            },  # right of D, above AC
            text_sym("s_m", "line", "m", 240, 170),
            text_sym("s_area", "area", "40", 100, 120),
            {
                "id": "s_arrow",
                "sym_class": "arrow",
                "text_class": "others",
                "text_content": None,
                "bbox": [250, 170, 20, 20],
            },
        ],  # fmt: skip
        "relations": {"geo2geo": [], "sym2sym": [], "sym2geo": [["s_m", ["l0"]]]},
    }
    return lf, ann


def converted(lf, ann):
    r = convert_one("t", lf, ann)
    assert r.status == "ok", r.reason
    d = S.validate_output(r.obj).value.diagram
    return r, d


def labels_by_text(d) -> dict[str, S.AnyLabel]:
    return {lb.text: lb for lb in d.labels}


def test_points_hidden_names_and_segments():
    r, d = converted(*triangle())
    assert set(d.points) == {"A", "B", "C", "D", "_P1"}  # K is not drawn as a label
    assert r.n_unnamed == 1
    assert {frozenset(s) for s in d.segments} == {
        frozenset(s) for s in (("A", "_P1"), ("A", "B"), ("B", "C"), ("B", "D"))
    }  # AK is one maximal drawn segment, though D and C lie on it


def test_grid_fit_matches_the_generator_convention():
    _, d = converted(*triangle())
    xs = [p[0] for p in d.points.values()]
    assert (min(xs), max(xs)) == (8, 92)  # the longer side spans [margin, 100 - margin]
    assert d.points["A"] == (8, 76) and d.points["_P1"] == (92, 76)
    assert d.points["B"][1] < d.points["A"][1]  # y still points down


def test_marks_and_labels():
    r, d = converted(*triangle())
    (ra,) = [m for m in d.marks if isinstance(m, S.RightAngleMark)]
    assert ra.at == "D" and set(ra.rays) == {"B", "C"}  # the corner where the square is
    (eq,) = [m for m in d.marks if isinstance(m, S.EqualLengthMark)]
    assert {frozenset(s) for s in eq.segments} == {frozenset("AD"), frozenset("DC")}
    assert not any(isinstance(m, S.ArrowMark) for m in d.marks)  # callouts are not rays
    lb = labels_by_text(d)
    assert set(lb["5\\sqrt{2}"].on) == {"A", "B"}
    assert lb["(4y-10)^\\circ"].angle[1] == "B"  # an expression gets parentheses
    assert lb["45^\\circ"].angle[1] == "B"
    assert lb["1"].angle == ("B", "A", "C")  # a numbered angle
    assert set(lb["m"].on) == {"A", "_P1"}  # a line name
    assert isinstance(lb["40"], S.FreeTextLabel)
    assert r.dropped == ["not representable: Equals(PerimeterOf(Circle(O)), 15)"]
    assert r.obj["text"] == ""


def test_right_angle_corner_follows_the_symbol():
    lf, ann = triangle()
    ann["symbols"][4]["bbox"] = [104, 166, 10, 10]  # left of D instead
    _, d = converted(lf, ann)
    (ra,) = [m for m in d.marks if isinstance(m, S.RightAngleMark)]
    assert set(ra.rays) == {"B", "A"}


def test_circles_partial_arcs_and_a_center_with_no_point():
    pos = {"E": [150, 100], "F": [100, 50], "G": [50, 100]}
    lf = {"point_instances": list(pos), "line_instances": [], "circle_instances": ["O"],
          "diagram_logic_forms": ["PointLiesOnCircle(E, Circle(O, radius_0_0))",
                                  "Equals(MeasureOf(Arc(E, F)), 90)",
                                  "Equals(LengthOf(Arc(E, F)), LengthOf(Arc(F, G)))"],
          "point_positions": pos}  # fmt: skip
    ann = {"width": 200, "height": 200,
           "geos": {"points": [], "lines": [], "circles": [{"loc": [[100, 100], 50, "1110"]}]},
           "symbols": [text_sym(f"t{n}", "point", n, *pos[n]) for n in "EFG"],
           "relations": {"sym2geo": []}}  # fmt: skip
    r, d = converted(lf, ann)
    (c,) = d.circles
    assert S.is_unnamed(c.center) and c.radius == pytest.approx(42.0)  # 84 / (2 * 50) * 50
    (arc,) = d.labels
    assert isinstance(arc, S.ArcLabel) and set(arc.arc) == {"E", "F"} and arc.text == "90^\\circ"
    assert any("partial circle" in x for x in r.dropped)
    assert any("LengthOf(Arc" in x for x in r.dropped)
    assert r.n_issues == 0


def test_a_second_label_on_one_target_is_dropped():
    forms = ["Equals(LengthOf(Line(A, B)), 7)", "Equals(LengthOf(Line(B, A)), x)"]
    r, d = converted(*triangle(forms=forms))
    assert [lb.text for lb in d.labels if isinstance(lb, S.SegmentLabel)] == ["7", "m"]
    assert any("second label" in x for x in r.dropped)


def test_point_names_outside_the_schema_skip_the_diagram():
    lf, ann = triangle()
    ann["symbols"][0]["text_content"] = "DesMoines"
    lf["point_positions"] = {("DesMoines" if k == "A" else k): v for k, v in POS.items()}
    assert convert_one("t", lf, ann).status == "skipped"


RAW = Path("~/vlm_data/raw/pgdp5k/extracted/PGDP5K").expanduser()


@pytest.mark.skipif(not RAW.exists(), reason="PGDP5K is not downloaded")
def test_real_val_diagrams_convert():
    lfs = json.loads((RAW / "our_diagram_logic_forms_annot.json").read_text())
    anns = json.loads((RAW / "annotations" / "val.json").read_text())
    keys = sorted(anns)[:100]
    results = [convert_one(k, lfs[k], anns[k]) for k in keys]
    assert all(r.status == "ok" for r in results), [r.reason for r in results if r.status != "ok"]
    assert sum(not r.dropped for r in results) >= 90


def test_braced_subscripts_are_respelled():
    lf, ann = triangle(forms=["Equals(LengthOf(Line(A_{1}, B)), 3)"])
    lf["point_positions"] = {("A_{1}" if k == "A" else k): v for k, v in POS.items()}
    ann["symbols"][0]["text_content"] = "A_{1}"
    _, d = converted(lf, ann)
    assert "A_1" in d.points and set(d.labels[0].on) == {"A_1", "B"}
