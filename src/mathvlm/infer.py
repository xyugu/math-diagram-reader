"""Run the model over a synthetic split and save its raw outputs for the eval.

    uv run python -m mathvlm.infer --split val --name zeroshot
    uv run python -m mathvlm.infer --split val --adapter outputs/train/<run>/adapter
    uv run python -m mathvlm.eval.run_eval --split val --pred outputs/infer/val/zeroshot/predictions.jsonl

Writes to ``<outputs_dir>/infer/<split>/<name>/``:

    predictions.jsonl  one line per sample: id, output (the raw decoded text), truncated
                       (no stop token within max_new_tokens), n_tokens
    run.json           model, generation settings, library versions, GPU, timing

Outputs are saved raw. Parsing, fence stripping and failure classification belong to
the eval (eval/parse.py), so an eval change never needs a rerun of the model.

Generation: greedy, the instruction from prompts.py, thinking off (README.md, design
invariants 2 and 3). Generation stops at the end of the assistant turn (``<|im_end|>``)
or at end-of-text. The model config lists only end-of-text, and the model happens to
emit it right after ``<|im_end|>``; stopping on both does not rely on that habit.

Batching: samples are sorted by image area, largest first, so a batch holds images of
similar size (little padding) and memory peaks in the first batch. Prompts are padded
on the left, the side generation does not continue from; padded positions are zeroed
before the linear-attention layers. Greedy output can still differ in rare last-bit
cases between batch sizes (bf16 kernels are not batch-invariant), so the batch size is
recorded in run.json.

Fine-tuned models: ``--adapter`` loads a LoRA adapter (train/run.py) onto the base model
with peft and merges it into the weights, so a fine-tuned model runs through exactly the
same code as the baseline (no unsloth at inference). The run name defaults to the
training run's name (``<run>``, or ``<run>-checkpoint-N`` for a checkpoint).

Resumable: ids already in predictions.jsonl are skipped, so an interrupted run picks up
where it stopped. Resuming with a different adapter is refused. ``--overwrite`` starts over.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image

from mathvlm.config import Config, apply_hf_env, get_config
from mathvlm.prompts import ENABLE_THINKING, INSTRUCTION, build_messages
from mathvlm.splits import read_manifest, split_dir


def plan_batches(rows: Sequence[dict], areas: dict[str, int], batch_size: int) -> list[list[dict]]:
    """Largest images first, ties by dataset index, cut into batches. Deterministic."""
    ordered = sorted(rows, key=lambda r: (-areas[r["id"]], r["index"]))
    return [list(ordered[i : i + batch_size]) for i in range(0, len(ordered), batch_size)]


def cut_at_stop(ids: Sequence[int], stop: set[int]) -> tuple[list[int], bool]:
    """Generated ids before the first stop token, and whether generation was cut off
    (no stop token at all, i.e. it hit max_new_tokens)."""
    for k, t in enumerate(ids):
        if t in stop:
            return list(ids[:k]), False
    return list(ids), True


def load_done(path: Path) -> dict[str, dict]:
    """Predictions already written. A line cut short by a crash is dropped, and the file
    is rewritten without it so that it stays valid JSONL."""
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    lines = path.read_text().splitlines()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[row["id"]] = row
    if len(done) != sum(bool(line.strip()) for line in lines):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()))
    return done


def _versions() -> dict[str, str | None]:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "flash-linear-attention", "accelerate", "peft"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def run(
    cfg: Config,
    rows: list[dict],
    data_dir: Path,
    out_path: Path,
    batch_size: int,
    max_new_tokens: int,
    adapter: Path | None = None,
) -> dict:
    """Generate for every row not already in ``out_path``; append as batches finish."""
    import torch
    from tqdm import tqdm

    from mathvlm.modeling import load_model_and_processor

    torch.manual_seed(cfg.seed)
    t_load = time.perf_counter()
    model, processor, loader = load_model_and_processor(cfg)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.eval()
    load_s = time.perf_counter() - t_load
    tok = processor.tokenizer
    tok.padding_side = "left"
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    stop |= set(_as_list(model.generation_config.eos_token_id))
    stop.discard(None)

    areas = {}
    for r in rows:
        with Image.open(data_dir / r["image"]) as im:
            areas[r["id"]] = im.width * im.height
    batches = plan_batches(rows, areas, batch_size)

    n_tokens = n_truncated = 0
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with open(out_path, "a") as fh, tqdm(total=len(rows), unit="sample") as bar:
        for batch in batches:
            images = [Image.open(data_dir / r["image"]).convert("RGB") for r in batch]
            inputs = processor.apply_chat_template(
                [build_messages(im, INSTRUCTION) for im in images],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
                enable_thinking=ENABLE_THINKING,
            ).to(model.device)
            # left padding: every prompt must end at the last column
            assert bool(inputs["attention_mask"][:, -1].all()), "prompts are not left-padded"
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=cfg.model.do_sample,
                    eos_token_id=sorted(stop),
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, inputs["input_ids"].shape[1] :].tolist()
            lines = []
            for r, ids in zip(batch, gen, strict=True):
                kept, truncated = cut_at_stop(ids, stop)
                n_tokens += len(kept)
                n_truncated += truncated
                text = tok.decode(kept, skip_special_tokens=False)
                row = {"id": r["id"], "output": text, "truncated": truncated,
                       "n_tokens": len(kept)}  # fmt: skip
                lines.append(json.dumps(row, ensure_ascii=False) + "\n")
            fh.write("".join(lines))
            fh.flush()
            bar.update(len(batch))
            bar.set_postfix(tok_s=f"{n_tokens / (time.perf_counter() - t0):.0f}")

    elapsed = time.perf_counter() - t0
    return {
        "loader": loader,
        "load_s": round(load_s, 1),
        "generated": len(rows),
        "generate_s": round(elapsed, 1),
        "generated_tokens": n_tokens,
        "tokens_per_s": round(n_tokens / elapsed, 1) if elapsed > 0 else None,
        "truncated": n_truncated,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "stop_token_ids": sorted(stop),
    }


def _as_list(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, list | tuple) else [x]


def default_name(adapter: Path | None) -> str:
    """``zeroshot`` for the base model; otherwise named after the training run."""
    if adapter is None:
        return "zeroshot"
    if adapter.name.startswith("checkpoint-"):
        return f"{adapter.parent.parent.name}-{adapter.name}"
    return adapter.parent.name if adapter.name == "adapter" else adapter.name


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    apply_hf_env(cfg)
    ap = argparse.ArgumentParser(prog="python -m mathvlm.infer", description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir (train/run.py)")
    ap.add_argument("--name", default=None,
                    help="run name (default: zeroshot, or the training run's name)")  # fmt: skip
    ap.add_argument("--data", type=Path, default=None, help="default: <data_dir>/synth/<split>")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <outputs_dir>/infer/<split>/<name>")  # fmt: skip
    ap.add_argument("--batch", type=int, default=cfg.model.batch_size)
    ap.add_argument("--max-new-tokens", type=int, default=cfg.model.max_new_tokens)
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples by index")
    ap.add_argument("--overwrite", action="store_true", help="discard earlier predictions")
    args = ap.parse_args(argv)

    adapter = args.adapter.resolve() if args.adapter else None
    if adapter is not None and not (adapter / "adapter_config.json").exists():
        ap.error(f"{adapter} has no adapter_config.json")
    name = args.name or default_name(adapter)
    data = args.data or split_dir(args.split, cfg)
    out = args.out or cfg.paths.resolved("outputs_dir") / "infer" / args.split / name
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    if args.overwrite and pred_path.exists():
        pred_path.unlink()
    prev = out / "run.json"
    if pred_path.exists() and prev.exists():
        before = json.loads(prev.read_text()).get("adapter")
        if before != (str(adapter) if adapter else None):
            raise SystemExit(
                f"{pred_path} came from adapter {before}; pass --overwrite or another --name"
            )

    rows = read_manifest(data)[: args.limit]
    done = load_done(pred_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} samples, {len(rows) - len(todo)} already done, {len(todo)} to generate")

    stats = {}
    if todo:
        stats = run(cfg, todo, data, pred_path, args.batch, args.max_new_tokens, adapter)
    meta = {
        "split": args.split,
        "name": name,
        "adapter": str(adapter) if adapter else None,
        "data": str(data),
        "n": len(rows),
        "model": cfg.model.id,
        "dtype": cfg.model.dtype,
        "attn_implementation": cfg.model.attn_implementation,
        "batch_size": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": cfg.model.do_sample,
        "enable_thinking": ENABLE_THINKING,
        "instruction": INSTRUCTION,
        "versions": _versions(),
        "argv": sys.argv[1:] if argv is None else argv,
        **stats,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    if stats:
        print(f"{stats['generated']} samples in {stats['generate_s']}s "
              f"({stats['tokens_per_s']} tok/s, {stats['truncated']} truncated, "
              f"peak {stats['peak_vram_gb']} GB)")  # fmt: skip
    print(f"wrote {pred_path}")
    print(f"next: uv run python -m mathvlm.eval.run_eval --split {args.split} --pred {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
