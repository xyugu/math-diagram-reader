"""Tests for the model-free parts of inference (src/mathvlm/infer.py). The model itself
is exercised by running the CLI, not by the test suite."""

from __future__ import annotations

import json

from mathvlm.infer import cut_at_stop, load_done, plan_batches
from mathvlm.splits import read_manifest


def test_cut_at_stop():
    assert cut_at_stop([5, 6, 9, 7, 9], {9}) == ([5, 6], False)
    assert cut_at_stop([5, 6, 7], {9}) == ([5, 6, 7], True)
    assert cut_at_stop([9, 1], {8, 9}) == ([], False)
    assert cut_at_stop([], {9}) == ([], True)


def test_plan_batches_groups_by_size_deterministically():
    rows = [{"id": f"s{i}", "index": i} for i in range(7)]
    areas = {"s0": 10, "s1": 50, "s2": 30, "s3": 50, "s4": 20, "s5": 40, "s6": 10}
    batches = plan_batches(rows, areas, 3)
    assert [[r["id"] for r in b] for b in batches] == [
        ["s1", "s3", "s5"],
        ["s2", "s4", "s0"],
        ["s6"],
    ]
    assert plan_batches(list(reversed(rows)), areas, 3) == batches
    assert sum(len(b) for b in plan_batches(rows, areas, 1)) == 7


def test_load_done_drops_a_line_cut_short(tmp_path):
    p = tmp_path / "predictions.jsonl"
    assert load_done(p) == {}
    good = [{"id": "a", "output": "x"}, {"id": "b", "output": "y"}]
    p.write_text("".join(json.dumps(r) + "\n" for r in good) + '{"id": "c", "out')
    assert set(load_done(p)) == {"a", "b"}
    assert [json.loads(line)["id"] for line in p.read_text().splitlines()] == ["a", "b"]


def test_read_manifest_orders_by_index(tmp_path):
    rows = [{"id": "x2", "index": 2}, {"id": "x0", "index": 0}, {"id": "x1", "index": 1}]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r["id"] for r in read_manifest(tmp_path)] == ["x0", "x1", "x2"]
