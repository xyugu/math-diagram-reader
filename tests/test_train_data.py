"""Tests for training examples (src/mathvlm/train/data.py), against the real processor.

The point of these tests: a training example must be the inference prompt, token for
token, followed by the target. Any drift (template, instruction, thinking block, a BPE
merge across the prompt/answer boundary) would train the model on a prompt it never sees.
"""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from mathvlm.config import apply_hf_env, get_config
from mathvlm.generate.build import build
from mathvlm.schema import to_target_json, validate_output_json
from mathvlm.train.data import END_OF_TURN, IGNORE_INDEX, Collator, load_records, prompt_ids

MAX_LEN = 2048


@pytest.fixture(scope="module")
def processor():
    apply_hf_env()
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(get_config().model.id, local_files_only=True)
    except OSError:
        pytest.skip("model processor not in the local HF cache")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    # The first 16 val samples: diagrams of every family, two without, varied image sizes.
    out = tmp_path_factory.mktemp("split")
    build("val", 16, 0, out)
    recs = load_records(out)
    assert any('"diagram":null' in r.target for r in recs)
    assert any('"diagram":{' in r.target for r in recs)
    return recs


@pytest.fixture(scope="module")
def batch(processor, records):
    return Collator(processor, MAX_LEN)(records)


def label_span(batch, b) -> tuple[int, int]:
    idx = (batch["labels"][b] != IGNORE_INDEX).nonzero().flatten()
    return int(idx[0]), int(idx[-1]) + 1


def test_records_hold_canonical_targets_in_index_order(records, tmp_path_factory):
    assert [r.id for r in records] == [f"val_{i:06d}" for i in range(16)]
    for r in records:
        gt = validate_output_json(r.image.with_suffix(".json").read_text()).value
        assert r.target == to_target_json(gt)
        json.loads(r.target)


def test_prompt_part_is_the_inference_prompt(processor, records, batch):
    for b, r in enumerate(records):
        start, _ = label_span(batch, b)
        expected = prompt_ids(processor, Image.open(r.image).convert("RGB"))
        assert batch["input_ids"][b, :start].tolist() == expected, r.id


def test_labels_are_the_target_and_end_of_turn(processor, records, batch):
    tok = processor.tokenizer
    end = tok.convert_tokens_to_ids(END_OF_TURN)
    for b, r in enumerate(records):
        start, stop = label_span(batch, b)
        labeled = batch["labels"][b, start:stop].tolist()
        # contiguous, equal to the inputs there, and no BPE merge across the boundary
        assert labeled == batch["input_ids"][b, start:stop].tolist()
        assert labeled == tok.encode(r.target, add_special_tokens=False) + [end], r.id
        assert tok.decode(labeled) == r.target + END_OF_TURN


def test_padding_and_the_trailing_newline_are_not_trained(batch):
    labels, mask = batch["labels"], batch["attention_mask"]
    assert bool((labels[mask == 0] == IGNORE_INDEX).all())
    for b in range(len(labels)):
        n = int(mask[b].sum())
        assert bool(mask[b, :n].all()) and not bool(mask[b, n:].any())  # right padding
        _, stop = label_span(batch, b)
        assert stop == n - 1  # the template's "\n" after <|im_end|> is the last real token


def test_image_tokens_match_the_image_grid(processor, batch):
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    merge = processor.image_processor.merge_size
    per_row = (batch["input_ids"] == image_id).sum(dim=1)
    expected = batch["image_grid_thw"].prod(dim=1) // (merge * merge)
    assert torch.equal(per_row, expected)
    assert batch["pixel_values"].shape[0] == int(batch["image_grid_thw"].prod(dim=1).sum())


def test_an_example_is_the_same_alone_and_in_a_batch(processor, records, batch):
    collate = Collator(processor, MAX_LEN)
    for b in (0, 5, 15):
        alone = collate([records[b]])
        n = int(batch["attention_mask"][b].sum())
        assert alone["input_ids"][0].tolist() == batch["input_ids"][b, :n].tolist()
        assert alone["labels"][0].tolist() == batch["labels"][b, :n].tolist()


def test_overlong_batches_raise_instead_of_truncating(processor, records):
    with pytest.raises(ValueError, match="exceeds max_length"):
        Collator(processor, 64)(records[:2])
