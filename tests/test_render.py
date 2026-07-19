"""Tests for the renderer (src/mathvlm/render.py) and the reference fixtures."""

from __future__ import annotations

import math

import matplotlib
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mathvlm import render as R
from mathvlm import schema as S
from mathvlm.config import get_config, sample_rng
from mathvlm.fixtures import DIAGRAMS, PAGES

RC = get_config().render


def diagram(obj: dict) -> S.Diagram:
    r = S.validate_output({"text": "", "diagram": obj})
    assert r.ok, r.errors
    return r.value.diagram


def transcription(obj: dict) -> S.Transcription:
    r = S.validate_output(obj)
    assert r.ok, r.errors
    return r.value


def no_dots(style: R.RenderStyle | None = None) -> R.RenderStyle:
    from dataclasses import replace

    return replace(style or R.default_style(), point_dots=False)


def dark(img, x: float, y: float, radius: int = 2, thresh: int = 128) -> bool:
    """Is there a dark pixel within `radius` px of image position (x, y)?"""
    a = np.asarray(img.convert("L"))
    xi, yi = int(round(x)), int(round(y))
    win = a[max(yi - radius, 0) : yi + radius + 1, max(xi - radius, 0) : xi + radius + 1]
    return bool((win < thresh).any())


# A single horizontal segment between two unlabeled points: no labels, no dots, so the
# only ink is the segment itself.
HLINE = {"points": {"_P1": [10, 20], "_P2": [90, 20]}, "segments": [["_P1", "_P2"]]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DIAGRAMS))
def test_fixture_diagrams_are_valid_and_consistent(name):
    assert S.consistency_issues(diagram(DIAGRAMS[name])) == []


@pytest.mark.parametrize("name", list(PAGES))
def test_fixture_pages_are_valid(name):
    transcription(PAGES[name])


def test_fixtures_cover_every_feature():
    ds = [diagram(d) for d in DIAGRAMS.values()]
    mark_types = {m.type for d in ds for m in d.marks}
    assert mark_types == {"right_angle", "equal_length", "parallel", "equal_angle", "arrow"}
    label_types = {type(lb) for d in ds for lb in d.labels}
    assert label_types == {S.SegmentLabel, S.AngleLabel, S.ArcLabel, S.FreeTextLabel}
    assert any(d.circles for d in ds)
    names = {n for d in ds for n in d.points}
    assert any(S.is_unnamed(n) for n in names)
    assert any("'" in n for n in names) and any("_" in n and not S.is_unnamed(n) for n in names)
    assert any(p["diagram"] is None for p in PAGES.values())


# ---------------------------------------------------------------------------
# Coordinate contract
# ---------------------------------------------------------------------------


def test_render_diagram_is_square_rgb():
    img = R.render_diagram(diagram(DIAGRAMS["incircle"]))
    side = int(RC.figure_width_in * RC.dpi)
    assert img.mode == "RGB"
    assert img.size == (side, side)


def test_grid_maps_to_expected_pixels_with_y_down():
    img = R.render_diagram(diagram(HLINE), no_dots())
    box = (0, 0, img.width)
    assert dark(img, *R.grid_to_pixel(50, 20, box))  # on the segment
    assert dark(img, *R.grid_to_pixel(12, 20, box))
    assert not dark(img, *R.grid_to_pixel(50, 80, box))  # y=20 is near the top, not bottom
    assert not dark(img, *R.grid_to_pixel(5, 20, box))  # before the segment starts
    assert not dark(img, *R.grid_to_pixel(95, 20, box))  # after it ends
    # nothing else on the canvas: the ink sits in the upper part of the image
    rows = np.where((np.asarray(img.convert("L")) < 128).any(axis=1))[0]
    assert rows.max() < img.height * 0.35


def test_page_panel_box_locates_the_diagram():
    page = R.render_page(
        S.Transcription(text="A long enough question to wrap onto two lines, maybe.",
                        diagram=diagram(HLINE)),
        no_dots(),
    )  # fmt: skip
    left, top, size = page.panel_box
    assert size == pytest.approx(
        min(RC.figure_width_in, RC.question_width_in - 2 * RC.page_margin_in) * RC.dpi
    )
    assert top > 0  # text sits above the panel
    assert top + size <= page.image.height + 1
    assert dark(page.image, *R.grid_to_pixel(50, 20, page.panel_box))
    assert not dark(page.image, *R.grid_to_pixel(50, 80, page.panel_box))


def test_text_only_page_has_no_panel():
    page = R.render_page(transcription(PAGES["page_no_diagram"]))
    assert page.panel_box is None
    assert page.image.height < page.image.width
    assert (np.asarray(page.image.convert("L")) < 128).any()


def test_longer_question_gives_taller_page():
    d = diagram(HLINE)
    short = R.render_page(S.Transcription(text="Find $x$.", diagram=d)).image
    long = R.render_page(
        S.Transcription(text=" ".join(["Find the value of $x$."] * 12), diagram=d)
    ).image
    assert long.height > short.height
    assert long.width == short.width


# ---------------------------------------------------------------------------
# Determinism and styles
# ---------------------------------------------------------------------------


def test_rendering_is_deterministic():
    t = transcription(PAGES["page_bisector_display_math"])
    style = R.sample_style(sample_rng(0, "test", 3))
    a = np.asarray(R.render_page(t, style).image)
    b = np.asarray(R.render_page(t, style).image)
    assert np.array_equal(a, b)


def test_sample_style_is_deterministic_and_varied():
    assert R.sample_style(sample_rng(0, "test", 1)) == R.sample_style(sample_rng(0, "test", 1))
    styles = [R.sample_style(sample_rng(0, "test", i)) for i in range(40)]
    assert len({s.fontset for s in styles}) > 1
    assert len({s.point_dots for s in styles}) == 2
    assert len({s.label_italic for s in styles}) == 2
    lo, hi = RC.line_width_jitter
    assert all(lo <= s.line_width <= hi for s in styles)
    assert all(s.fontset in RC.fontsets for s in styles)


def test_different_styles_give_different_images():
    t = transcription(PAGES["page_right_triangle"])
    a = R.render_page(t, R.sample_style(sample_rng(0, "test", 0))).image
    b = R.render_page(t, R.sample_style(sample_rng(0, "test", 1))).image
    assert a.size != b.size or not np.array_equal(np.asarray(a), np.asarray(b))


def test_global_matplotlib_state_is_restored():
    before = matplotlib.rcParams["mathtext.fontset"], list(matplotlib.rcParams["font.family"])
    for fs in RC.fontsets:
        style = R.RenderStyle(**{**R.default_style().__dict__, "fontset": fs,
                                 "text_family": RC.text_family[fs]})  # fmt: skip
        R.render_page(transcription(PAGES["page_tangents_two_parts"]), style)
    after = matplotlib.rcParams["mathtext.fontset"], list(matplotlib.rcParams["font.family"])
    assert before == after


# ---------------------------------------------------------------------------
# Robustness: predictions can be malformed in ways the schema allows
# ---------------------------------------------------------------------------


def test_bad_latex_in_labels_renders_instead_of_raising():
    d = diagram(
        {
            "points": {"A": [10, 80], "B": [90, 80], "C": [50, 20]},
            "segments": [["A", "B"], ["B", "C"], ["C", "A"]],
            "labels": [
                {"on": ["A", "B"], "text": "\\frac{1}{"},
                {"angle": ["A", "C", "B"], "text": "\\notacommand"},
            ],
        }
    )
    R.render_diagram(d)


def test_bad_latex_in_question_renders_instead_of_raising():
    R.render_page(S.Transcription(text="Find $\\frac{1}{$ and $$\\sqrt{$$ now.", diagram=None))


def test_degenerate_prediction_renders():
    d = diagram(
        {
            "points": {"A": [50, 50], "B": [50, 50], "C": [50, 50], "D": [0, 100]},
            "segments": [["A", "B"], ["C", "D"]],
            "circles": [{"center": "A", "radius": 100}],
            "marks": [
                {"type": "right_angle", "at": "A", "rays": ["B", "C"]},
                {"type": "equal_length", "segments": [["A", "B"], ["C", "D"]]},
                {"type": "parallel", "segments": [["A", "B"], ["C", "D"]]},
                {"type": "equal_angle", "angles": [["B", "A", "C"], ["C", "A", "D"]]},
            ],
            "labels": [{"on": ["A", "B"], "text": "0"}, {"angle": ["B", "A", "C"], "text": "?"}],
        }
    )
    R.render_diagram(d)
    R.render_page(S.Transcription(text="x", diagram=d), R.sample_style(sample_rng(0, "t", 0)))


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------


def test_words_keep_inline_math_intact():
    assert R._words("In $\\triangle ABC$, find $x + y$.") == [
        "In",
        "$\\triangle ABC$,",
        "find",
        "$x + y$.",
    ]


def test_logical_lines_split_paragraphs_breaks_and_display_math():
    lines = R._logical_lines("Show that\n\n$$\\frac{a}{b} = 1$$\n\n(a) one\n(b) two")
    assert lines == [
        ("Show that", False, True),
        ("$\\dfrac{a}{b} = 1$", True, True),  # display style for rendering only
        ("(a) one", False, True),
        ("(b) two", False, False),
    ]


def test_wrapped_lines_fit_the_page():
    style = R.default_style()
    text = " ".join(["In $\\triangle ABC$, the bisector of $\\angle B$ meets $AC$ at $D$."] * 6)
    max_px = (RC.question_width_in - 2 * RC.page_margin_in) * RC.dpi
    with R._rc(style):
        fig = Figure(figsize=(RC.question_width_in, 2), dpi=RC.dpi)
        m = R._Measurer(FigureCanvasAgg(fig).get_renderer(), style)
        lines = R._layout_text(text, m, max_px)
        assert len(lines) > 3
        assert all(m.measure(ln.s)[0] <= max_px for ln in lines)


# ---------------------------------------------------------------------------
# Drawing decisions
# ---------------------------------------------------------------------------


def painter(obj: dict) -> R._Painter:
    style = R.default_style()
    with R._rc(style):
        fig = Figure(figsize=(RC.figure_width_in,) * 2, dpi=RC.dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        R._setup_axes(ax)
        p = R._Painter(ax, diagram(obj), style, RC.figure_width_in)
        p.draw()
    return p


def test_point_labels_go_outside_the_figure():
    p = painter(DIAGRAMS["right_triangle_altitude"])
    # data coords: y points down, so sin(t) > 0 means "below"
    assert math.sin(p.label_direction("D")) > 0.5  # foot of the altitude: below AB
    assert math.sin(p.label_direction("C")) < 0  # apex: above
    assert math.cos(p.label_direction("A")) < 0  # left vertex: to the left
    assert math.cos(p.label_direction("B")) > 0  # right vertex: to the right


def test_tick_counts_distinguish_equal_length_groups():
    decos = painter(DIAGRAMS["isosceles_median"])._segment_decorations()
    counts = {frozenset(k): [d.count for d in v[1]] for k, v in decos.items()}
    assert counts[frozenset({"A", "B"})] == [1] and counts[frozenset({"A", "C"})] == [1]
    assert counts[frozenset({"B", "M"})] == [2] and counts[frozenset({"M", "C"})] == [2]


def test_parallel_arrows_point_the_same_way_even_if_listed_reversed():
    obj = {
        "points": {"A": [10, 80], "B": [60, 80], "C": [90, 20], "D": [40, 20]},
        "segments": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
        "marks": [
            {"type": "parallel", "segments": [["A", "B"], ["C", "D"]]}
        ],  # CD runs right-to-left
    }
    decos = painter(obj)._segment_decorations()
    d_ab = decos[frozenset({"A", "B"})][1][0].direction
    d_cd = decos[frozenset({"C", "D"})][1][0].direction
    assert d_ab[0] * d_cd[0] + d_ab[1] * d_cd[1] > 0.99


# ---------------------------------------------------------------------------
# Contact sheet and CLI
# ---------------------------------------------------------------------------


def test_contact_sheet_size():
    from PIL import Image

    ims = [Image.new("RGB", (100, 50 + 10 * i), "white") for i in range(5)]
    sheet = R.contact_sheet(ims, cols=3, pad=10)
    assert sheet.size == (3 * 100 + 4 * 10, (50 + 20) + (50 + 40) + 3 * 10)


def test_cli_writes_fixture_images(tmp_path):
    assert R.main(["--fixtures", "--out", str(tmp_path), "--styles", "2"]) == 0
    written = {p.name for p in tmp_path.glob("*.png")}
    assert {f"diagram_{n}.png" for n in DIAGRAMS} <= written
    assert {f"{n}.png" for n in PAGES} <= written
    assert {"contact_diagrams.png", "style_variants.png"} <= written


def test_length_label_slides_away_from_a_point_at_the_midpoint():
    obj = {
        "points": {"A": [10, 50], "M": [50, 50], "B": [90, 50], "C": [50, 15]},
        "segments": [["A", "B"], ["C", "M"]],
        "labels": [{"on": ["A", "B"], "text": "7"}],
    }
    p = painter(obj)
    (label,) = [t for t in p.ax.texts if t.get_text() == "$7$"]
    x, _ = label.get_position()
    assert abs(x - 50) > 5  # not on top of M (and M's label)
    assert 10 < x < 90  # still alongside AB


def test_length_label_stays_centered_when_uncrowded():
    obj = {
        "points": {"A": [10, 50], "B": [90, 50]},
        "segments": [["A", "B"]],
        "labels": [{"on": ["A", "B"], "text": "7"}],
    }
    (label,) = [t for t in painter(obj).ax.texts if t.get_text() == "$7$"]
    assert label.get_position()[0] == pytest.approx(50)


# ---------------------------------------------------------------------------
# v1.2: arrowheads, arc labels, free text
# ---------------------------------------------------------------------------


def test_arrowhead_points_away_from_its_tail():
    p = painter(DIAGRAMS["rays_arcs_free_text"])
    m = next(m for m in p.d.marks if isinstance(m, S.ArrowMark) and m.at == "_P1")
    tip, c1, c2 = p.arrowhead(m)
    assert tip == p.P["_P1"]
    base = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    assert base[0] < tip[0] and base[1] == pytest.approx(tip[1])  # behind the tip, toward O
    assert abs(c1[1] - c2[1]) > 0  # it has width across the ray


def test_arrowhead_is_drawn_across_the_line():
    arrow = {**HLINE, "marks": [{"type": "arrow", "at": "_P2", "from": "_P1"}]}
    plain, with_arrow = (R.render_diagram(diagram(d), no_dots()) for d in (HLINE, arrow))
    box = (0, 0, plain.width)
    # behind the tip (head length 4.5, half-width 1.9 at its base), one unit off the line:
    # ink only where the arrowhead is
    x, y = R.grid_to_pixel(86.5, 21.0, box)
    assert dark(with_arrow, x, y, radius=1) and not dark(plain, x, y, radius=1)
    assert not dark(with_arrow, *R.grid_to_pixel(50, 23, box), radius=1)  # mid-ray: none


def test_arc_label_sits_outside_the_minor_arc_midpoint():
    p = painter(DIAGRAMS["rays_arcs_free_text"])
    lb = next(lb for lb in p.d.labels if isinstance(lb, S.ArcLabel))
    x, y = p.arc_label_anchor(lb, f"${lb.text}$", p.s.label_fontsize)
    o = p.P["O"]
    assert math.dist((x, y), o) > 28  # outside the circle
    # B is east of O and D south (y down): the minor arc's midpoint is south-east
    assert math.atan2(y - o[1], x - o[0]) == pytest.approx(math.pi / 4, abs=1e-6)


def test_arc_label_without_a_circle_still_renders():
    d = {"points": {"A": [20, 50], "B": [80, 50]}, "segments": [["A", "B"]],
         "labels": [{"arc": ["A", "B"], "text": "x"}]}  # fmt: skip
    p = painter(d)
    x, y = p.arc_label_anchor(p.d.labels[0], "$x$", p.s.label_fontsize)
    assert 40 < x < 60  # beside the chord's midpoint
    R.render_diagram(diagram(d))


def test_free_text_is_drawn_at_its_position():
    text = {**HLINE, "labels": [{"at": [50, 70], "text": "A=40"}]}
    plain, with_text = (R.render_diagram(diagram(d), no_dots()) for d in (HLINE, text))
    x, y = R.grid_to_pixel(50, 70, (0, 0, plain.width))
    assert dark(with_text, x, y, radius=12) and not dark(plain, x, y, radius=12)


def test_degenerate_v12_prediction_renders():
    d = diagram({
        "points": {"A": [50, 50], "B": [50, 50]},
        "marks": [{"type": "arrow", "at": "A", "from": "B"}],  # zero-length: no head
        "labels": [{"arc": ["A", "B"], "text": "\\frac{"}, {"at": [0, 100], "text": "x"}],
    })  # fmt: skip
    R.render_diagram(d)
