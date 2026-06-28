"""LoRA fine-tuning with Unsloth + TRL (Phase 2).

    uv run python -m mathvlm.train.run --name overfit --train-split smoke --overfit
    uv run python -m mathvlm.train.run --name sft20k --train-split train

Writes to ``<outputs_dir>/train/<name>/``:

    adapter/      the final LoRA adapter (peft format) and processor files; run it with
                  ``python -m mathvlm.infer --split val --adapter <this dir>``
    checkpoints/  intermediate checkpoints (adapter + optimizer state); ``--resume``
    log.jsonl     one line per logging step and per eval: loss, learning rate, grad norm
    run.json      settings, data, LoRA coverage, versions, git commit, timing, memory

Model: unsloth's ``FastVisionModel`` in bf16 (16-bit LoRA, not QLoRA), with LoRA on the
language model only. The targets are unsloth's automatic selection; ``lora_coverage``
fails the run unless every linear layer of every language-model layer is adapted
(including the gated delta-net projections, which carry non-standard names) and nothing
in the vision tower is.

Examples and loss masking come from train/data.py: the inference prompt, token for token,
then the target, with the loss on the target only. Sequence packing stays off (unsloth
disables it for a custom collator), which matters here: packing would leak the recurrent
state of the linear-attention layers from one example into the next.

``--overfit`` trains on the first ``train.overfit.samples`` samples of the split for many
epochs, with no eval. Inference on those same samples should then be nearly perfect: a
check of train -> save -> load -> infer -> eval before spending hours on a real run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mathvlm.config import Config, apply_hf_env, get_config


def _env(cfg: Config) -> None:
    """Must run before unsloth or transformers is imported."""
    apply_hf_env(cfg)
    # unsloth writes generated trainer code to the working directory (the repo) otherwise
    cache = cfg.paths.resolved("outputs_dir") / "unsloth_cache"
    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(cache))


def lora_coverage(model) -> dict[str, int]:
    """LoRA-wrapped modules counted by kind. Raises unless every ``nn.Linear`` inside the
    language model's decoder layers is wrapped and no vision-tower module is."""
    import torch.nn as nn

    wrapped = [
        n for n, m in model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "base_layer")
    ]
    bare = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and ".language_model.layers." in n
        and not n.endswith(".base_layer")
        and ".lora_" not in n
    ]
    vision = [n for n in wrapped if ".visual." in n]
    if not wrapped or bare or vision:
        raise RuntimeError(
            f"LoRA coverage is wrong: {len(wrapped)} wrapped; language-model linears without "
            f"LoRA: {bare[:5]}{'...' if len(bare) > 5 else ''}; vision modules with LoRA: {vision[:5]}"
        )
    return dict(sorted(Counter(n.rsplit(".", 1)[-1] for n in wrapped).items()))


def _git() -> dict[str, str | bool | None]:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    head = run("rev-parse", "--short", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": head and head.strip(), "dirty": bool(status and status.strip())}


def _versions() -> dict[str, str | None]:
    out = {}
    for pkg in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "unsloth",
        "unsloth_zoo",
        "flash-linear-attention",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def train(cfg: Config, args: argparse.Namespace) -> Path:
    import torch  # noqa: I001 -- unsloth must be imported before trl and transformers
    from unsloth import FastVisionModel
    from transformers import TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    from mathvlm.splits import split_dir
    from mathvlm.train.data import Collator, as_dataset, load_records

    tc = cfg.train
    ov = tc.overfit
    out = cfg.paths.resolved("outputs_dir") / "train" / args.name
    adapter_dir = out / "adapter"
    if adapter_dir.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"{adapter_dir} exists; pass --overwrite to replace it or pick another --name"
        )
    out.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_dir = args.data or split_dir(args.train_split, cfg)
    train_records = load_records(train_dir, ov.samples if args.overfit else args.limit)
    eval_records = (
        [] if args.overfit else load_records(split_dir(args.val_split, cfg), tc.eval_samples)
    )

    # --- model ---
    set_seed(cfg.seed)
    t0 = time.perf_counter()
    model, processor = FastVisionModel.from_pretrained(
        cfg.model.id,
        load_in_4bit=False,
        dtype=getattr(torch, cfg.model.dtype),
        max_seq_length=tc.max_seq_length,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        lora_dropout=tc.lora_dropout,
        bias="none",
        random_state=cfg.seed,
    )
    coverage = lora_coverage(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    FastVisionModel.for_training(model)
    load_s = time.perf_counter() - t0
    print(f"LoRA on {sum(coverage.values())} modules {coverage}; "
          f"trainable {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")  # fmt: skip

    collator = Collator(processor, tc.max_seq_length)
    batch_size = ov.batch_size if args.overfit else tc.batch_size
    grad_accum = ov.grad_accum if args.overfit else tc.grad_accum
    log_path = out / "log.jsonl"
    if not args.resume and log_path.exists():
        log_path.unlink()

    class JsonlLog(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kw):
            if logs:
                with open(log_path, "a") as fh:
                    fh.write(
                        json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n"
                    )

    sft = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=ov.epochs if args.overfit else tc.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=ov.learning_rate if args.overfit else tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler,
        warmup_steps=tc.warmup,
        weight_decay=tc.weight_decay,
        max_grad_norm=tc.max_grad_norm,
        optim=tc.optim,
        bf16=True,
        logging_steps=tc.logging_steps,
        eval_strategy="steps" if eval_records else "no",
        eval_steps=tc.eval_every,
        save_strategy="no" if args.overfit else "steps",
        save_steps=tc.save_every,
        save_total_limit=tc.save_total_limit,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=tc.dataloader_workers,
        report_to="none",
        # examples are built by our collator, not by TRL
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=tc.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=as_dataset(train_records),
        eval_dataset=as_dataset(eval_records) if eval_records else None,
        data_collator=collator,
        processing_class=processor,
        callbacks=[JsonlLog()],
    )

    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_s = time.perf_counter() - t1
    final_eval = trainer.evaluate() if eval_records else {}

    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    meta = {
        "name": args.name,
        "overfit": args.overfit,
        "train_data": str(train_dir),
        "n_train": len(train_records),
        "val_split": None if args.overfit else args.val_split,
        "n_eval": len(eval_records),
        "model": cfg.model.id,
        "settings": {
            **tc.model_dump(exclude={"overfit"}),
            **({"overfit": ov.model_dump()} if args.overfit else {}),
            "effective_batch": batch_size * grad_accum,
            "max_steps": args.max_steps,
        },
        "lora": {"modules": coverage, "trainable": trainable, "total": total},
        "steps": trainer.state.global_step,
        "train_loss": result.training_loss,
        "final_eval": final_eval,
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 1),
        "samples_per_s": round(result.metrics.get("train_samples_per_second", 0.0), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "git": _git(),
        "versions": _versions(),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{trainer.state.global_step} steps in {train_s / 60:.1f} min "
          f"({meta['samples_per_s']} samples/s, peak {meta['peak_vram_gb']} GB); "
          f"train loss {result.training_loss:.4f}"
          + (f", eval loss {final_eval.get('eval_loss', float('nan')):.4f}" if final_eval else ""))  # fmt: skip
    print(f"adapter: {adapter_dir}")
    return adapter_dir


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.train.run", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--name", required=True, help="run name: outputs/train/<name>/")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val", help="for the periodic eval loss")
    ap.add_argument(
        "--data", type=Path, default=None, help="train split directory (overrides --train-split)"
    )
    ap.add_argument("--overfit", action="store_true", help="memorize a few samples (train.overfit)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N training samples")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run's adapter")
    args = ap.parse_args(argv)
    _env(cfg)
    adapter = train(cfg, args)
    split = args.train_split if args.overfit else args.val_split
    print(f"next: uv run python -m mathvlm.infer --split {split} --adapter {adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
