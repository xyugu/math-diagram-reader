"""Training examples for supervised fine-tuning, and the collator that batches them.

An example is exactly what the model sees at inference, followed by the answer:

    <inference prompt: image + INSTRUCTION, chat template, thinking off>
    <target JSON: schema.to_target_json(ground truth)><|im_end|>

and the loss covers only the target and its ``<|im_end|>``. The newline the template puts
after ``<|im_end|>`` is not trained: generation stops at ``<|im_end|>``.

Where the answer starts is found by searching each tokenized sequence for the last
occurrence of the generation prompt, the tokens of ``<|im_start|>assistant`` up to the
start of the answer. That avoids processing every image twice (once with the answer, once
without). tests/test_train_data.py checks the result against the direct definition: the
tokens before the answer equal the inference prompt token for token, and the labeled
tokens decode to the target plus ``<|im_end|>``.

This module does not import unsloth, so it can be tested without patching transformers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from mathvlm.prompts import ENABLE_THINKING, INSTRUCTION, build_messages
from mathvlm.schema import to_target_json
from mathvlm.splits import load_split

IGNORE_INDEX = -100  # label value the loss skips
END_OF_TURN = "<|im_end|>"


@dataclass(frozen=True)
class Record:
    id: str
    image: Path
    target: str  # schema.to_target_json of the ground truth


def load_records(data_dir: Path, limit: int | None = None) -> list[Record]:
    """A split's samples in index order, with validated, canonical targets."""
    return [
        Record(row["id"], data_dir / row["image"], to_target_json(gt))
        for row, gt in load_split(data_dir)[:limit]
    ]


def _messages(image: Image.Image, target: str | None) -> list[dict]:
    msgs = build_messages(image, INSTRUCTION)
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def prompt_ids(processor, image: Image.Image) -> list[int]:
    """Token ids of the inference prompt for one image, as infer.py builds it."""
    enc = processor.apply_chat_template(
        _messages(image, None),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=ENABLE_THINKING,
    )
    ids = enc["input_ids"]
    return list(ids[0] if ids and isinstance(ids[0], list) else ids)


def _find_last(seq: Sequence[int], sub: Sequence[int]) -> int:
    n = len(sub)
    for i in range(len(seq) - n, -1, -1):
        if list(seq[i : i + n]) == list(sub):
            return i
    return -1


class Collator:
    """Turns a list of ``Record`` into a padded batch with ``labels``.

    ``max_length`` guards against silently truncating a sequence: truncation would cut
    through the answer (or worse, the image tokens), so an over-long example raises.
    """

    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length
        tok = processor.tokenizer
        tok.padding_side = "right"
        self.pad_id = tok.pad_token_id
        self.end_id = tok.convert_tokens_to_ids(END_OF_TURN)
        # The generation prompt is what the template appends for add_generation_prompt:
        # the rendered prompt minus the rendered conversation without it.
        msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        kw = {"tokenize": False, "enable_thinking": ENABLE_THINKING}
        with_gen = processor.apply_chat_template(msgs, add_generation_prompt=True, **kw)
        without = processor.apply_chat_template(msgs, add_generation_prompt=False, **kw)
        if not with_gen.startswith(without):
            raise RuntimeError("chat template: generation prompt is not a suffix")
        self.gen_prompt = with_gen[len(without) :]
        self.gen_prompt_ids = tok.encode(self.gen_prompt, add_special_tokens=False)

    def __call__(self, records: Sequence[Record]) -> dict[str, Any]:
        images = [Image.open(r.image).convert("RGB") for r in records]
        batch = self.processor.apply_chat_template(
            [_messages(im, r.target) for im, r in zip(images, records, strict=True)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
            enable_thinking=ENABLE_THINKING,
        )
        ids = batch["input_ids"]
        if ids.shape[1] > self.max_length:
            raise ValueError(
                f"batch length {ids.shape[1]} exceeds max_length {self.max_length} "
                f"({[r.id for r in records]})"
            )
        labels = torch.full_like(ids, IGNORE_INDEX)
        for b, r in enumerate(records):
            row = ids[b].tolist()
            g = _find_last(row, self.gen_prompt_ids)
            if g < 0:
                raise ValueError(f"{r.id}: generation prompt not found in the sequence")
            start = g + len(self.gen_prompt_ids)
            try:
                end = row.index(self.end_id, start)
            except ValueError:
                raise ValueError(f"{r.id}: no {END_OF_TURN} after the answer") from None
            labels[b, start : end + 1] = ids[b, start : end + 1]
        batch["labels"] = labels
        return dict(batch)
