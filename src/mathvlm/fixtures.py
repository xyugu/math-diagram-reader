"""Hand-written reference figures.

Used by ``python -m mathvlm.render --fixtures`` to eyeball the renderer and by the
tests. Every diagram here is valid under the schema and passes
``schema.consistency_issues`` (it is drawn to scale), and together they cover every
mark type, both label kinds, circles, and unlabeled ``_P`` points.

Coordinates were worked out by hand on the 0-100 grid (y down) and rounded to integers.
"""

from __future__ import annotations

# name -> diagram JSON object
DIAGRAMS: dict[str, dict] = {
    # 3-4-5 right triangle with the altitude to the hypotenuse. The foot D is where three
    # segments meet, i.e. exactly the case where a v1 right_angle mark was ambiguous.
    "right_triangle_altitude": {
        "points": {"A": [10, 80], "B": [90, 80], "C": [39, 42], "D": [39, 80]},
        "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["C", "D"]],
        "marks": [
            {"type": "right_angle", "at": "C", "rays": ["A", "B"]},
            {"type": "right_angle", "at": "D", "rays": ["C", "B"]},
        ],
        "labels": [
            {"on": ["A", "C"], "text": "3"},
            {"on": ["B", "C"], "text": "4"},
            {"on": ["A", "B"], "text": "5"},
        ],
    },
    # Isosceles triangle with the median to the base: two equal_length groups (1 and 2
    # ticks) plus a right angle at the midpoint.
    "isosceles_median": {
        "points": {"A": [50, 15], "B": [20, 85], "C": [80, 85], "M": [50, 85]},
        "segments": [["A", "B"], ["A", "C"], ["B", "C"], ["A", "M"]],
        "marks": [
            {"type": "equal_length", "segments": [["A", "B"], ["A", "C"]]},
            {"type": "equal_length", "segments": [["B", "M"], ["M", "C"]]},
            {"type": "right_angle", "at": "M", "rays": ["A", "C"]},
        ],
        "labels": [{"angle": ["B", "A", "C"], "text": "46^\\circ"}],
    },
    # Parallelogram with diagonals meeting at an unlabeled point: two parallel groups
    # (1 and 2 arrows) and two equal_length groups on collinear sub-segments.
    "parallelogram_diagonals": {
        "points": {
            "A": [15, 75],
            "B": [65, 75],
            "C": [85, 25],
            "D": [35, 25],
            "_P1": [50, 50],
        },
        "segments": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"], ["A", "C"], ["B", "D"]],
        "marks": [
            {"type": "parallel", "segments": [["A", "B"], ["D", "C"]]},
            {"type": "parallel", "segments": [["A", "D"], ["B", "C"]]},
            {"type": "equal_length", "segments": [["A", "_P1"], ["_P1", "C"]]},
            {"type": "equal_length", "segments": [["B", "_P1"], ["_P1", "D"]]},
        ],
        "labels": [{"on": ["A", "B"], "text": "x+2"}],
    },
    # Angle bisector BD of angle ABC, with D on AC.
    "angle_bisector": {
        "points": {"A": [55, 15], "B": [15, 80], "C": [90, 80], "D": [73, 48]},
        "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["B", "D"]],
        "marks": [{"type": "equal_angle", "angles": [["A", "B", "D"], ["D", "B", "C"]]}],
        "labels": [
            {"angle": ["B", "A", "C"], "text": "70^\\circ"},
            {"on": ["B", "C"], "text": "12"},
        ],
    },
    # Incircle with center I and one unlabeled tangency point on AB.
    "incircle": {
        "points": {
            "A": [10, 85],
            "B": [90, 85],
            "C": [40, 15],
            "I": [45, 62],
            "_P1": [45, 85],
        },
        "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["I", "_P1"]],
        "circles": [{"center": "I", "radius": 23.1}],
        "marks": [{"type": "right_angle", "at": "_P1", "rays": ["I", "B"]}],
        "labels": [{"on": ["I", "_P1"], "text": "r"}],
    },
    # Triangle inscribed in its circumcircle, with one radius drawn.
    "circumcircle": {
        "points": {"A": [20, 68], "B": [80, 68], "C": [44, 16], "O": [50, 50]},
        "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["O", "A"]],
        "circles": [{"center": "O", "radius": 35}],
        "labels": [{"on": ["O", "A"], "text": "R"}, {"angle": ["A", "C", "B"], "text": "\\theta"}],
    },
    # Two tangents from an external point P. Radii meet the tangents at right angles and
    # the tangent lengths are equal.
    "tangent_lines": {
        "points": {"O": [35, 50], "P": [90, 50], "T": [46, 28], "S": [46, 72]},
        "segments": [["O", "T"], ["O", "S"], ["O", "P"], ["P", "T"], ["P", "S"]],
        "circles": [{"center": "O", "radius": 24.6}],
        "marks": [
            {"type": "right_angle", "at": "T", "rays": ["O", "P"]},
            {"type": "right_angle", "at": "S", "rays": ["O", "P"]},
            {"type": "equal_length", "segments": [["P", "T"], ["P", "S"]]},
        ],
        "labels": [{"on": ["O", "P"], "text": "13"}, {"on": ["O", "T"], "text": "5"}],
    },
    # Primes and subscripts in point names.
    "primed_and_subscript_names": {
        "points": {"A": [15, 80], "A'": [60, 80], "O_1": [37, 40], "B_2": [85, 30]},
        "segments": [["A", "A'"], ["A", "O_1"], ["A'", "O_1"], ["O_1", "B_2"]],
        "marks": [{"type": "equal_length", "segments": [["A", "O_1"], ["A'", "O_1"]]}],
    },
}

# name -> full Transcription JSON object (question text + diagram)
PAGES: dict[str, dict] = {
    "page_right_triangle": {
        "text": (
            "In right triangle $ABC$, $\\angle ACB = 90^\\circ$ and $CD \\perp AB$ at $D$. "
            "If $AC = 3$ and $BC = 4$, find the length of $CD$."
        ),
        "diagram": DIAGRAMS["right_triangle_altitude"],
    },
    "page_bisector_display_math": {
        "text": (
            "In $\\triangle ABC$, $BD$ bisects $\\angle ABC$ and meets $AC$ at $D$. "
            "Show that\n\n$$\\frac{AD}{DC} = \\frac{AB}{BC}.$$"
        ),
        "diagram": DIAGRAMS["angle_bisector"],
    },
    "page_tangents_two_parts": {
        "text": (
            "From an external point $P$, two tangents $PT$ and $PS$ are drawn to a circle "
            "with center $O$ and radius $5$. Given $OP = 13$:\n\n"
            "(a) Find $PT$.\n\n"
            "(b) Find the area of quadrilateral $OTPS$."
        ),
        "diagram": DIAGRAMS["tangent_lines"],
    },
    "page_no_diagram": {
        "text": "Solve for $x$:\n\n$$2x^2 - 3x + 1 = 0$$",
        "diagram": None,
    },
}
