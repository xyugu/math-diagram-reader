r"""Raw model output -> validated ``Transcription``, with every leniency visible.

``schema.validate_output_json`` takes exactly one JSON value. Model output is messier,
so this layer accepts two forms, and reports which one it used:

1. ``direct``: the output, with surrounding whitespace stripped, is the JSON value.
2. ``fenced``: the whole output is one Markdown code fence (```` ```json ... ``` ````,
   language tag optional), and the fence body is the JSON value.

Nothing else is accepted. JSON inside prose, a reasoning preamble and text after the
fence are all parse failures. They are classified (``FAILURES``) so the report can say
*why* outputs fail, but never repaired, because a consumer of the model would not
repair them either. ``json_parse_rate`` counts both forms; ``json_parse_rate_strict``
counts only ``direct``.

A subtle failure that *does* parse: LaTeX written into a JSON string without doubling
its backslashes. ``"\frac"`` is legal JSON (``\f`` is a form feed), so it decodes to a
control character followed by ``rac``. ``escape_suspect`` flags decoded text containing
such characters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from mathvlm.schema import Transcription, ValidationReport, validate_output_json

FAILURES = {
    "missing": "no prediction for this sample",
    "empty": "empty output",
    "truncated": "output ends mid-JSON, or generation hit max_new_tokens",
    "invalid_escape": "a backslash escape JSON does not allow, usually unescaped LaTeX (\\sqrt)",
    "control_char": "a raw newline or tab inside a JSON string",
    "prose_wrapped": "a JSON object surrounded by other text",
    "no_json": "no JSON object at all",
    "syntax": "any other JSON syntax error",
}

_FENCE_RE = re.compile(r"\A```[ \t]*(?:json|JSON)?[ \t]*\n(.*?)\n?[ \t]*```\Z", re.S)
# Control characters that appear in decoded text when LaTeX backslashes were not
# escaped: \b (\beta), \f (\frac), \t (\theta, \triangle), \r (\rho).
_ESCAPE_SUSPECT_RE = re.compile("[\b\f\t\r]")

DiagramKind = Literal["diagram", "null", "failed"]


@dataclass(frozen=True)
class ParsedOutput:
    raw: str | None
    how: Literal["direct", "fenced"] | None  # None: did not parse
    report: ValidationReport  # stage "parse_error" exactly when how is None
    obj: Any  # the decoded JSON value, valid or not; None when it did not parse
    failure: str | None  # a FAILURES key when how is None

    @property
    def transcription(self) -> Transcription | None:
        """The validated output, or None if it is not schema-valid."""
        return self.report.value

    @property
    def text(self) -> str | None:
        """The ``text`` field whenever the output decoded to an object with a string
        there, even if the diagram is invalid: the text metric should not depend on the
        diagram being valid."""
        if isinstance(self.obj, dict) and isinstance(self.obj.get("text"), str):
            return self.obj["text"]
        return None

    @property
    def diagram_kind(self) -> DiagramKind:
        """What the output says about the figure: ``"null"`` or ``"diagram"`` (an object,
        valid or not), else ``"failed"`` (no parse, no ``diagram`` key, wrong type)."""
        if isinstance(self.obj, dict) and "diagram" in self.obj:
            d = self.obj["diagram"]
            if d is None:
                return "null"
            if isinstance(d, dict):
                return "diagram"
        return "failed"

    @property
    def escape_suspect(self) -> bool:
        return self.text is not None and _ESCAPE_SUSPECT_RE.search(self.text) is not None


def _attempt(candidate: str) -> tuple[ValidationReport, Any] | None:
    report = validate_output_json(candidate)
    if report.stage == "parse_error":
        return None
    try:  # plain decode, only to read text/diagram out of an invalid output
        obj = json.loads(candidate)
    except ValueError:
        obj = None
    return report, obj


def _extractable(s: str) -> bool:
    """True if some ``{`` in ``s`` starts a complete JSON object."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return True
    return False


def classify_failure(raw: str | None, truncated: bool = False) -> str:
    """Why an output that did not parse failed. One ``FAILURES`` key."""
    if raw is None:
        return "missing"
    s = raw.strip()
    if not s:
        return "empty"
    if truncated:
        return "truncated"
    m = _FENCE_RE.match(s)
    candidate = m.group(1) if m else s
    try:
        json.loads(candidate)
    except json.JSONDecodeError as e:
        if "Invalid \\escape" in e.msg or "Invalid \\uXXXX escape" in e.msg:
            return "invalid_escape"
        if e.msg.startswith("Invalid control character"):
            return "control_char"
        at_end = e.msg.startswith("Unterminated string") or e.pos >= len(candidate.rstrip())
        if at_end and candidate.lstrip().startswith(("{", "[")):
            return "truncated"
    except ValueError:
        pass
    if "{" not in s:
        return "no_json"
    if _extractable(s):
        return "prose_wrapped"
    return "syntax"


def parse_output(raw: str | None, truncated: bool = False) -> ParsedOutput:
    """Parse and validate one raw model output. ``truncated`` is what the generator
    reported (it hit max_new_tokens); it only affects how a failure is classified."""
    if raw is not None:
        s = raw.strip()
        got = _attempt(s)
        if got is not None:
            return ParsedOutput(raw, "direct", got[0], got[1], None)
        m = _FENCE_RE.match(s)
        if m:
            got = _attempt(m.group(1))
            if got is not None:
                return ParsedOutput(raw, "fenced", got[0], got[1], None)
    failure = classify_failure(raw, truncated)
    report = ValidationReport("parse_error", None, (f"{failure}: {FAILURES[failure]}",))
    return ParsedOutput(raw, None, report, None, failure)
