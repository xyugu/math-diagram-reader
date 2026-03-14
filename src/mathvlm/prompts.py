"""The single source of truth for the model-facing instruction.

Training and inference MUST use the identical string. Never inline a paraphrase
elsewhere in the codebase -- import ``INSTRUCTION`` from here.
"""

from __future__ import annotations

# Verbatim from the project brief. Do not reword: any change silently invalidates
# every checkpoint and every baseline number measured before the change.
INSTRUCTION = (
    'Transcribe this math question. Return JSON with keys "text" '
    '(Markdown with LaTeX) and "diagram" (null or a scene graph).'
)

# Qwen3.5 supports a thinking mode. It must stay off: a reasoning preamble ahead of
# the JSON breaks parsing. See README.md, design invariant 3.
ENABLE_THINKING = False


def build_messages(image, instruction: str = INSTRUCTION) -> list[dict]:
    """Return the chat-template message list for one image + the instruction.

    ``image`` is a PIL image (or anything the processor accepts). Kept here, next to
    the instruction, so the train-time and infer-time message shape cannot drift.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
