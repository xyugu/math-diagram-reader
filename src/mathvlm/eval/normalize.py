r"""Text normalization for the transcription metrics.

Two strings that render the same should normalize to the same string, so the text
metric counts transcription errors, not LaTeX spelling choices. The rules are
conservative: each one merges spellings that render the same, and nothing that renders
differently is merged. tests/test_normalize.py pins both directions.

What is normalized
------------------
- Unicode: a small map from math symbols to their LaTeX commands (``∠`` -> ``\angle``,
  ``°`` -> ``^\circ``, ``−`` -> ``-``), typographic quotes and dashes to ASCII, then NFKC.
- Math delimiters: ``\(...\)`` -> ``$...$`` and ``\[...\]`` -> ``$$...$$``. An escaped
  ``\$`` is a literal dollar sign, not a delimiter.
- Inside math:
  - whitespace is dropped (TeX ignores it), except the one space a control word needs
    before a letter (``\angle A``) and spaces inside ``\text{...}``, collapsed to one;
  - spacing commands (``\,``, ``\quad``, ``~``...) and delimiter sizing (``\left``,
    ``\big``...) are dropped;
  - braces around a one-token sub- or superscript go (``x^{2}`` -> ``x^2``,
    ``^{\circ}`` -> ``^\circ``), and ``^\prime`` becomes ``'``;
  - synonyms are unified (``\dfrac``/``\tfrac`` -> ``\frac``, ``\le`` -> ``\leq``,
    ``\mathrm`` -> ``\text``, ...).
- Outside math, Markdown whitespace rules: a single newline is a space, a blank line
  separates paragraphs, runs of spaces collapse. Display math is its own block, as it is
  when rendered.

What is not normalized
----------------------
Anything that changes the rendering: ``$`` vs ``$$``, ``\overline`` vs ``\bar``,
``\cdot`` vs ``\times``, ``\frac{1}{2}`` vs ``1/2``, Markdown emphasis, letter case.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# Applied before NFKC, which would otherwise turn "²" into a plain "2".
_UNICODE = {
    "−": "-",  # minus sign
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "′": "'",  # prime
    "″": "''",  # double prime
    " ": " ",  # no-break space
    "°": "^\\circ",
    "²": "^2",
    "³": "^3",
    "×": "\\times ",
    "÷": "\\div ",
    "·": "\\cdot ",
    "⋅": "\\cdot ",
    "≤": "\\leq ",
    "≥": "\\geq ",
    "≠": "\\neq ",
    "≈": "\\approx ",
    "≅": "\\cong ",
    "∼": "\\sim ",
    "∠": "\\angle ",
    "△": "\\triangle ",
    "∆": "\\triangle ",
    "⊥": "\\perp ",
    "∥": "\\parallel ",
    "√": "\\sqrt ",
    "π": "\\pi ",
    "θ": "\\theta ",
    "α": "\\alpha ",
    "β": "\\beta ",
    "γ": "\\gamma ",
}
_UNICODE_RE = re.compile("|".join(map(re.escape, _UNICODE)))

_SYNONYMS = {
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\leqslant": r"\leq",
    r"\geqslant": r"\geq",
    r"\to": r"\rightarrow",
    r"\gets": r"\leftarrow",
    r"\lt": "<",
    r"\gt": ">",
    r"\bigtriangleup": r"\triangle",
    r"\vartriangle": r"\triangle",
    r"\mathrm": r"\text",
    r"\textrm": r"\text",
    r"\textnormal": r"\text",
    r"\mbox": r"\text",
    r"\lbrace": r"\{",
    r"\rbrace": r"\}",
    r"\vert": "|",
    r"\lvert": "|",
    r"\rvert": "|",
}
_DEGREE = {r"\degree", r"\textdegree"}  # -> ^\circ
_SPACING = {r"\,", r"\;", r"\:", r"\!", "\\ ", r"\quad", r"\qquad", "~", r"\enspace"}
_SIZING = {
    r"\left", r"\right", r"\middle",
    r"\big", r"\Big", r"\bigg", r"\Bigg",
    r"\bigl", r"\bigr", r"\Bigl", r"\Bigr", r"\biggl", r"\biggr", r"\Biggl", r"\Biggr",
}  # fmt: skip

_TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|\s+|.", re.S)
_CONTROL_WORD_RE = re.compile(r"\\[A-Za-z]+")


def _unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", _UNICODE_RE.sub(lambda m: _UNICODE[m.group()], s))


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def _math_tokens(m: str) -> list[str]:
    """Tokens with whitespace, spacing and sizing removed and synonyms unified.
    Whitespace inside ``\\text{...}`` survives as single ``" "`` tokens."""
    raw = _TOKEN_RE.findall(m)
    out: list[str] = []
    i = 0
    while i < len(raw):
        t = raw[i]
        t = _SYNONYMS.get(t, t)
        if t == r"\text":
            # copy the {...} group verbatim, whitespace collapsed to one space
            j = i + 1
            while j < len(raw) and raw[j].isspace():
                j += 1
            if j < len(raw) and raw[j] == "{":
                depth, k, group = 0, j, []
                while k < len(raw):
                    tok = raw[k]
                    depth += tok == "{"
                    depth -= tok == "}"
                    group.append(" " if tok.isspace() else tok)
                    k += 1
                    if depth == 0:
                        break
                out.append(t)
                out.extend(group)
                i = k
                continue
        if t.isspace() or t in _SPACING:
            i += 1
            continue
        if t in (r"\left", r"\right") and i + 1 < len(raw) and raw[i + 1] == ".":
            i += 2  # the null delimiter
            continue
        if t in _SIZING:
            i += 1
            continue
        if t in _DEGREE:
            out.extend(("^", r"\circ"))
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def _strip_script_braces(tokens: list[str]) -> list[str]:
    """``^{x}`` -> ``^x`` and ``_{x}`` -> ``_x`` when x is a single token; ``^\\prime`` -> ``'``."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if (
            t in ("^", "_")
            and i + 3 < len(tokens)
            and tokens[i + 1] == "{"
            and tokens[i + 3] == "}"
            and tokens[i + 2] not in ("{", "}", "^", "_", " ")
        ):
            out.extend((t, tokens[i + 2]))
            i += 4
            continue
        out.append(t)
        i += 1
    merged: list[str] = []
    i = 0
    while i < len(out):
        if out[i] == "^" and i + 1 < len(out) and out[i + 1] == r"\prime":
            merged.append("'")
            i += 2
            continue
        merged.append(out[i])
        i += 1
    return merged


def normalize_math(m: str) -> str:
    """Normalize the body of a math span (no delimiters)."""
    tokens = _strip_script_braces(_math_tokens(m))
    out: list[str] = []
    for t in tokens:
        if out and _CONTROL_WORD_RE.fullmatch(out[-1]) and t[:1].isalpha():
            out.append(" ")  # "\angle A" must not become "\angleA"
        out.append(t)
    return "".join(out)


# ---------------------------------------------------------------------------
# Whole text
# ---------------------------------------------------------------------------


def _find_close(s: str, start: int, delim: str) -> int:
    """Index of the first unescaped ``delim`` at or after ``start``, or -1."""
    k = start
    while k < len(s):
        if s[k] == "\\":
            if s.startswith(delim, k) and delim.startswith("\\"):
                return k
            k += 2
            continue
        if s.startswith(delim, k):
            return k
        k += 1
    return -1


def _segments(s: str) -> list[tuple[str, str]]:
    """Split into ("text" | "inline" | "display", body). An unclosed delimiter is text."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0
    openers = (("$$", "$$", "display"), ("\\[", "\\]", "display"),
               ("$", "$", "inline"), ("\\(", "\\)", "inline"))  # fmt: skip
    while i < len(s):
        if s.startswith("\\$", i):
            buf.append("\\$")
            i += 2
            continue
        for op, cl, kind in openers:
            if s.startswith(op, i):
                j = _find_close(s, i + len(op), cl)
                if j >= 0:
                    out.append(("text", "".join(buf)))
                    buf = []
                    out.append((kind, s[i + len(op) : j]))
                    i = j + len(cl)
                    break
        else:
            buf.append(s[i])
            i += 1
    out.append(("text", "".join(buf)))
    return out


def normalize_text(s: str) -> str:
    """Normalize a whole question (Markdown with LaTeX). Idempotent."""
    s = _unicode(s.replace("\r\n", "\n").replace("\r", "\n"))
    parts: list[str] = []
    for kind, body in _segments(s):
        if kind == "text":
            parts.append(body)
            continue
        m = normalize_math(body)
        if not m:
            continue  # "$ $" is nothing; keeping "$$" would read as a display delimiter
        parts.append(f"${m}$" if kind == "inline" else f"\n\n$${m}$$\n\n")
    paragraphs = re.split(r"\n[ \t]*\n", "".join(parts))
    return "\n\n".join(p for p in (re.sub(r"\s+", " ", p).strip() for p in paragraphs) if p)


_SEP = r"(?:\s|\\[ ,;:!]|~)"  # whitespace or a TeX spacing command
_UNIT_RE = re.compile(
    rf"^(?P<value>.*?[^\s\\~])(?:{_SEP}*\\(?:text|mathrm)\{{\s*|{_SEP}+)"
    r"(?:in|ft|cm|mm|m|km|yd|mi|units?)\.?\s*\}?$"
)


def normalize_label(text: str) -> str:
    """Normalize a diagram label. Labels are math without delimiters (``45^\\circ``); a
    prediction that wraps one in ``$...$`` means the same thing.

    A trailing length unit set off by a space or ``\\text{}`` is dropped (``6 ft``,
    ``10 in.``, ``8\\text{ cm}`` -> the number): units are presentation, and datasets
    differ in whether they record them (PGDP5K does not). ``4m`` without a space is kept:
    it may be 4 times m."""
    s = _unicode(text).strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s.strip("$")
    m = _UNIT_RE.match(s)
    if m:
        s = m.group("value")
    return normalize_math(s)


def ned(a: str, b: str) -> float:
    """Normalized edit distance: Levenshtein distance / length of the longer string.
    0 is identical, 1 is nothing in common; two empty strings are identical."""
    return float(Levenshtein.normalized_distance(a, b))
