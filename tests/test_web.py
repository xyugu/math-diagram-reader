"""The viewer's data layer (web/runs.py) on a small fake run, and its HTTP routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mathvlm.schema import output_to_obj, validate_output
from mathvlm.web.runs import RunStore, check_output

GT = {
    "text": "In $\\triangle ABC$, find $x$.",
    "diagram": {
        "points": {"A": [10, 90], "B": [90, 90], "C": [50, 10]},
        "segments": [["A", "B"], ["A", "C"], ["B", "C"]],
        "circles": [],
        "marks": [],
        "labels": [{"on": ["A", "B"], "text": "x"}],
    },
}
UNKNOWN_POINT = {**GT, "diagram": {**GT["diagram"], "segments": [["A", "D"]]}}


def _jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r if isinstance(r, str) else json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    """A val split of 4 samples, run1 over it (with an eval), and a run on a split whose
    name has a slash and whose data is missing."""
    data = tmp_path / "data" / "val"
    manifest = []
    for i in range(4):
        sid = f"s{i}"
        (data / f"{sid}.json").parent.mkdir(parents=True, exist_ok=True)
        (data / f"{sid}.json").write_text(json.dumps(GT))
        Image.new("RGB", (8, 8), "white").save(data / f"{sid}.png")
        image = "../escape.png" if i == 3 else f"{sid}.png"
        manifest.append({"id": sid, "index": i, "family": "tri", "kind": "word",
                         "gt": f"{sid}.json", "image": image})  # fmt: skip
    _jsonl(data / "manifest.jsonl", manifest)

    out = tmp_path / "outputs"
    run = out / "infer" / "val" / "run1"
    # mathvlm.infer writes in batch order, not dataset order; the last line is cut short
    _jsonl(run / "predictions.jsonl", [
        {"id": "s2", "output": "I cannot read this.", "truncated": False, "n_tokens": 5},
        {"id": "s0", "output": json.dumps(GT), "truncated": False, "n_tokens": 90},
        {"id": "s1", "output": "```json\n" + json.dumps(GT) + "\n```", "truncated": False,
         "n_tokens": 95},
        {"id": "s3", "output": json.dumps(UNKNOWN_POINT), "truncated": False, "n_tokens": 80},
        '{"id": "s4", "outp',
    ])  # fmt: skip
    (run / "run.json").write_text(
        json.dumps({"split": "val", "name": "run1", "data": str(data), "adapter": "/a/adapter"})
    )
    _jsonl(out / "eval" / "val" / "run1" / "per_example.jsonl", [
        {"id": "s0", "text": {"exact": True},
         "structure": {"strict": {"points": [3, 0, 0], "segments": [2, 1, 1],
                                  "circles": [0, 0, 0], "marks": [0, 0, 0],
                                  "labels": [1, 0, 0], "marks.arrow": [0, 9, 9]}},
         "geometry": {"error_mean": 1.5}, "output": "raw text"},
    ])  # fmt: skip

    nested = out / "infer" / "pgdp5k" / "val" / "zs"
    _jsonl(nested / "predictions.jsonl", [{"id": "p0", "output": json.dumps(GT)}])
    (nested / "run.json").write_text(json.dumps({"data": str(tmp_path / "missing")}))
    return out


def test_runs_are_found_by_path_including_split_names_with_a_slash(outputs):
    store = RunStore(outputs)
    assert store.keys() == ["pgdp5k/val/zs", "val/run1"]
    nested, run1 = store.runs()
    assert (nested.split, nested.name, nested.has_eval) == ("pgdp5k/val", "zs", False)
    assert (run1.split, run1.name, run1.adapter, run1.has_eval) == ("val", "run1", "/a/adapter",
                                                                    True)  # fmt: skip
    assert run1.n == 4  # the line cut short is skipped


def test_samples_are_in_dataset_order_with_their_parse_stage(outputs):
    samples = RunStore(outputs).samples("val/run1")
    assert [s.id for s in samples] == ["s0", "s1", "s2", "s3"]
    assert [s.stage for s in samples] == ["ok", "ok", "parse_error", "invalid"]
    assert [s.pred for s in samples] == ["diagram", "diagram", "failed", "diagram"]
    s0 = samples[0]
    # pooled over the five categories only (2*6 / (2*6 + 1 + 1)); per-type rows ignored
    assert s0.text_exact is True
    assert s0.structure_f1 == pytest.approx(12 / 14)
    assert s0.point_error == 1.5
    assert samples[1].structure_f1 is None  # no eval row


def test_samples_without_a_manifest_keep_file_order(outputs):
    (s,) = RunStore(outputs).samples("pgdp5k/val/zs")
    assert (s.id, s.index, s.family, s.stage) == ("p0", 0, None, "ok")


def test_sample_detail(outputs):
    store = RunStore(outputs)
    s1 = store.sample("val/run1", "s1")
    assert (s1.how, s1.check.stage, s1.has_image) == ("fenced", "ok", True)
    assert s1.pred == GT and s1.gt == GT
    assert s1.check.canonical == output_to_obj(validate_output(GT).value)
    assert s1.eval is None

    s2 = store.sample("val/run1", "s2")
    assert (s2.how, s2.pred, s2.check.stage, s2.failure) == (None, None, "parse_error", "no_json")
    assert s2.raw == "I cannot read this."

    s3 = store.sample("val/run1", "s3")
    assert s3.check.stage == "invalid" and s3.check.canonical is None
    assert any("unknown point 'D'" in e for e in s3.check.errors)
    assert s3.pred == UNKNOWN_POINT  # the decoded output is kept, so the page can fix it

    s0 = store.sample("val/run1", "s0")
    assert s0.eval is not None and "output" not in s0.eval


def test_unknown_runs_and_samples_are_key_errors(outputs):
    store = RunStore(outputs)
    for key in ("val", "../outputs/infer/val/run1", "val/run1/", "nope"):
        with pytest.raises(KeyError):
            store.samples(key)
    with pytest.raises(KeyError):
        store.sample("val/run1", "s4")  # its line was cut short


def test_images_must_lie_inside_the_data_directory(outputs):
    store = RunStore(outputs)
    assert store.image_path("val/run1", "s0").name == "s0.png"
    with pytest.raises(KeyError):
        store.image_path("val/run1", "s3")  # manifest says ../escape.png
    with pytest.raises(KeyError):
        store.image_path("pgdp5k/val/zs", "p0")  # no manifest


def test_check_output_is_strict_and_reports_geometric_warnings():
    ok = check_output(json.dumps(GT))
    assert (ok.stage, ok.errors, ok.warnings) == ("ok", [], [])
    assert ok.canonical == output_to_obj(validate_output(GT).value)

    assert check_output('{"text": "a", "text": "b", "diagram": null}').stage == "invalid"
    assert check_output("```json\n" + json.dumps(GT) + "\n```").stage == "parse_error"

    # a right angle at A that a dragged point has turned into ~63 degrees
    bent = json.loads(json.dumps(GT))
    bent["diagram"]["marks"] = [{"type": "right_angle", "at": "A", "rays": ["B", "C"]}]
    res = check_output(json.dumps(bent))
    assert res.stage == "ok"
    assert any("right_angle at A" in w for w in res.warnings)


def test_routes(outputs):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from mathvlm.web.app import create_app

    client = TestClient(create_app(outputs_dir=outputs))
    assert [r["key"] for r in client.get("/api/runs").json()] == ["pgdp5k/val/zs", "val/run1"]
    meta = client.get("/api/meta").json()
    assert meta["grid"] == [0, 100] and "mark_size" in meta["render"]

    q = {"run": "val/run1", "id": "s0"}
    assert len(client.get("/api/samples", params={"run": "val/run1"}).json()) == 4
    assert client.get("/api/sample", params=q).json()["pred"] == GT
    img = client.get("/api/image", params=q)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get("/api/samples", params={"run": "nope"}).status_code == 404
    assert client.get("/api/image", params={**q, "id": "s3"}).status_code == 404

    assert client.post("/api/check", json={"text": json.dumps(GT)}).json()["stage"] == "ok"
    png = client.post("/api/render", json={"text": json.dumps(GT)})
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
    bad = client.post("/api/render", json={"text": json.dumps(UNKNOWN_POINT)})
    assert bad.status_code == 422

    page = client.get("/")
    assert page.status_code == 200 and "<title>" in page.text
