"""Evaluate predictions on a synthetic split.

    uv run python -m mathvlm.eval.run_eval --split val --oracle
    uv run python -m mathvlm.eval.run_eval --split val --pred <predictions.jsonl>

The predictions file is JSONL, one object per sample: ``"id"``, ``"output"`` (the raw
model output string) and optionally ``"truncated"`` (true if generation hit
max_new_tokens). A sample without a prediction is scored as a failure ("missing").

``--oracle`` feeds each ground truth back in as its own prediction, serialized exactly
like a training target. It must score perfectly; anything else is a bug in the metrics
or the pipeline.

Writes to ``<outputs_dir>/eval/<split>/<name>/``:

    summary.json       every aggregate (eval.metrics.summarize), plus per family
    summary.txt        the table printed to the terminal
    per_example.jsonl  one row per sample: parse, text, structure counts, geometry, output
    side_by_side/      input image | re-rendered prediction, first N samples by index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mathvlm.config import get_config
from mathvlm.eval.metrics import ExampleResult, evaluate_example, summarize, summarize_by
from mathvlm.eval.report import format_summary, write_json, write_jsonl, write_side_by_sides
from mathvlm.schema import Transcription, to_target_json
from mathvlm.splits import load_split, split_dir


def load_predictions(path: Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for k, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not (
            row.get("output") is None or isinstance(row.get("output"), str)
        ):
            raise ValueError(f"{path}:{k}: need a string 'id' and a string (or null) 'output'")
        if row["id"] in preds:
            raise ValueError(f"{path}:{k}: duplicate prediction for {row['id']}")
        preds[row["id"]] = row
    return preds


def evaluate_split(
    split: list[tuple[dict, Transcription]], preds: dict[str, dict]
) -> list[ExampleResult]:
    results = []
    for row, gt in split:
        p = preds.get(row["id"], {})
        results.append(
            evaluate_example(
                gt,
                p.get("output"),
                id=row["id"],
                meta={"family": row["family"], "kind": row["kind"]},
                truncated=bool(p.get("truncated", False)),
                unannotated=row.get("unannotated", ()),
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.eval.run_eval", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--data", type=Path, default=None, help="default: <data_dir>/synth/<split>")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", type=Path, help="predictions JSONL")
    src.add_argument("--oracle", action="store_true", help="ground truth as predictions")
    ap.add_argument("--name", default=None, help="run name (default: from --pred, or 'oracle')")
    ap.add_argument(
        "--out", type=Path, default=None, help="default: <outputs_dir>/eval/<split>/<name>"
    )
    ap.add_argument("--sbs", type=int, default=cfg.eval.side_by_side_limit,
                    help="side-by-side images to write (0: none)")  # fmt: skip
    args = ap.parse_args(argv)

    data = args.data or split_dir(args.split, cfg)
    split = load_split(data)
    if args.oracle:
        preds = {row["id"]: {"id": row["id"], "output": to_target_json(gt)} for row, gt in split}
        name = args.name or "oracle"
    else:
        preds = load_predictions(args.pred)
        stem = args.pred.parent.name if args.pred.name == "predictions.jsonl" else args.pred.stem
        name = args.name or stem
        unknown = sorted(set(preds) - {row["id"] for row, _ in split})
        if unknown:
            print(f"warning: {len(unknown)} predictions for ids not in the split, ignored "
                  f"(first: {unknown[0]})", file=sys.stderr)  # fmt: skip
    out = args.out or cfg.paths.resolved("outputs_dir") / "eval" / args.split / name
    out.mkdir(parents=True, exist_ok=True)

    results = evaluate_split(split, preds)
    summary = summarize(results)
    by_family = summarize_by(results, "family")
    text = format_summary(summary, by_family, title=f"eval {args.split} / {name}")

    write_json({"split": args.split, "name": name, **summary, "by_family": by_family},
               out / "summary.json")  # fmt: skip
    (out / "summary.txt").write_text(text + "\n")
    write_jsonl(results, out / "per_example.jsonl")
    images = {row["id"]: data / row["image"] for row, _ in split}
    sbs = write_side_by_sides(results, images, out / "side_by_side", args.sbs) if args.sbs else []

    print(text)
    print(f"\nwrote {out}/: summary.json, summary.txt, per_example.jsonl"
          + (f", side_by_side/ ({len(sbs)} images)" if sbs else ""))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
