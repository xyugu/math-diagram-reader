"""Tests for text normalization (src/mathvlm/eval/normalize.py)."""

from __future__ import annotations

import pytest

from mathvlm.eval.normalize import ned, normalize_label, normalize_math, normalize_text
from mathvlm.generate.build import make_sample

# Pairs that render the same, so must normalize the same.
EQUIVALENT = [
    # whitespace inside math
    ("$x + 1 = 2$", "$x+1=2$"),
    ("$ x $", "$x$"),
    ("$\\angle  BAC$", "$\\angle BAC$"),
    # script braces
    ("$x^{2}$", "$x^2$"),
    ("$40^{\\circ}$", "$40^\\circ$"),
    ("$O_{1}$", "$O_1$"),
    ("$A^{\\prime}$", "$A'$"),
    ("$A^\\prime$", "$A'$"),
    # synonyms
    ("$\\dfrac{1}{2}$", "$\\frac{1}{2}$"),
    ("$\\tfrac{1}{2}$", "$\\frac{1}{2}$"),
    ("$x \\le 3$", "$x\\leq 3$"),
    ("$x \\ne 3$", "$x\\neq 3$"),
    ("$5\\mathrm{cm}$", "$5\\text{cm}$"),
    ("$40\\degree$", "$40^\\circ$"),
    # spacing and sizing commands
    ("$a\\,b\\;c\\quad d$", "$abcd$"),
    ("$\\left( x \\right)$", "$(x)$"),
    ("$\\left. x \\right|$", "$x|$"),
    ("$\\big( x \\big)$", "$(x)$"),
    # delimiters
    ("\\(x\\)", "$x$"),
    ("\\[x\\]", "$$x$$"),
    # unicode
    ("$∠ABC = 40°$", "$\\angle ABC=40^\\circ$"),
    ("$x − 1$", "$x-1$"),
    ("$2 × 3 ≤ 7$", "$2\\times 3\\leq 7$"),
    ("$\\triangle ABC$", "$△ABC$"),
    ("it’s", "it's"),
    # Markdown whitespace
    ("Find  x.", "Find x."),
    ("Find\nx.", "Find x."),
    ("  Find x.  \n", "Find x."),
    ("Solve:\n\n$$x=1$$\n\n", "Solve:\n\n\n\n$$x=1$$"),
    ("Solve: $$x=1$$", "Solve:\n\n$$x=1$$"),
    ("a\r\nb", "a b"),
    ("(a) one\n  \n(b) two", "(a) one\n\n(b) two"),
]

# Pairs that render differently, so must stay different.
DIFFERENT = [
    ("$x$", "$$x$$"),
    ("$\\alpha x$", "$\\alphax$"),
    ("$x^{23}$", "$x^23$"),
    ("$\\overline{AB}$", "$\\bar{AB}$"),
    ("$a\\cdot b$", "$a\\times b$"),
    ("$\\frac{1}{2}$", "$1/2$"),
    ("$\\text{two words}$", "$\\text{twowords}$"),
    ("$AB$", "$ab$"),
    ("Find x.", "Find x"),
    ("(a) one\n\n(b) two", "(a) one (b) two"),
    ("cost \\$5", "cost $5"),
    ("**Find** x.", "Find x."),
]


@pytest.mark.parametrize(("a", "b"), EQUIVALENT)
def test_equivalent_spellings_normalize_equal(a, b):
    assert normalize_text(a) == normalize_text(b)


@pytest.mark.parametrize(("a", "b"), DIFFERENT)
def test_different_renderings_stay_different(a, b):
    assert normalize_text(a) != normalize_text(b)


TRICKY = [
    "",
    "$",
    "$$",
    "$ $",
    "$$ $$",
    "unclosed $x + 1",
    "unclosed $$x",
    "\\(open",
    "a \\$ b $c$ d",
    "$\\text {a  b}$",
    "$x^{}$",
    "$x^$",
    "$\\$5$",
    "trailing backslash \\",
    "$\\left.$",
    "\n\n\n",
    "°²³ ∠△",
]


@pytest.mark.parametrize("s", [*TRICKY, *(a for a, _ in EQUIVALENT), *(a for a, _ in DIFFERENT)])
def test_normalization_is_idempotent(s):
    once = normalize_text(s)
    assert normalize_text(once) == once


def test_normalization_is_idempotent_on_generated_questions():
    for i in range(150):
        t = make_sample(0, "norm", i).transcription.text
        once = normalize_text(t)
        assert normalize_text(once) == once, t


def test_math_keeps_control_word_separation():
    assert normalize_math("\\angle   A") == "\\angle A"
    assert normalize_math("\\alpha2") == "\\alpha2"
    assert normalize_math("\\sqrt{2}") == "\\sqrt{2}"


@pytest.mark.parametrize(
    "label", ["45^\\circ", "$45^\\circ$", "45°", "$45^{\\circ}$", " 45 ^ \\circ "]
)
def test_label_spellings(label):
    assert normalize_label(label) == "45^\\circ"


def test_label_values_still_differ():
    assert normalize_label("45^\\circ") != normalize_label("46^\\circ")
    assert normalize_label("2x+1") != normalize_label("2x-1")


def test_ned():
    assert ned("abc", "abc") == 0.0
    assert ned("", "") == 0.0
    assert ned("", "abc") == 1.0
    assert ned("kitten", "sitting") == pytest.approx(3 / 7)
    assert ned("abcd", "abce") == ned("abce", "abcd") == pytest.approx(1 / 4)
