# mathvlm

Fine-tunes a small vision-language model, [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B),
to read an image of a math question and return one JSON object:

- `"text"`: the question as Markdown, with math in LaTeX (`$...$` inline, `$$...$$` display)
- `"diagram"`: `null` if there is no figure, otherwise a scene graph (points, segments, circles,
  marks, labels) that can be redrawn

Training is LoRA on the language layers (Unsloth + TRL) with the vision tower frozen, on images
from a synthetic generator whose ground truth is exact. Real textbook diagrams (PGDP5K) are used
for evaluation only.

## Results

Structure F1: each figure is split into its parts, the prediction is shifted and scaled onto the
ground truth, and matching parts are counted (F1 of precision and recall, pooled over diagrams).

| Model | Training data | Synthetic val | PGDP5K val (500 real) |
|-------|---------------|---------------|-----------------------|
| Zero-shot | none | 0.000 | 0.000 |
| `sft20k` | 20k synthetic | 0.989 | 0.489 |
| `sft_v4` | 20k synthetic, realism generator | 0.989 | 0.724 |

**Final score**, `sft_v4` on PGDP5K test (999 diagrams, scored once): structure F1 **0.729**,
schema-valid 90.3%, median point error 2.73 grid units (of 100). By part: points 0.88, segments
0.69, circles 0.66, labels 0.57, marks 0.46. `sft20k` on synthetic test: 0.988.

The zero-shot model never predicts a diagram. `sft20k` failed on real figures mostly on
conventions it had never seen: 42% of PGDP5K val diagrams have no letters, 53% of points are
unlabeled, and pages have no question text. The realism generator (`generate/realism.py`) adds
these, plus rays, numbered angles, dense labels, line names and textbook colors. Remaining
weaknesses: marks, labels on the wrong target, and point precision.

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Schema, renderer, synthetic generator, eval harness, zero-shot baseline | done |
| 2 | LoRA SFT on synthetic data + public math OCR data | synthetic done |
| 3 | Realism; evaluation on real textbook diagrams (PGDP5K) | paused, best model `sft_v4` |
| 4 | Optional GRPO with eval metrics as rewards; export for local inference | not started |

## Quickstart

```bash
uv sync --extra train --extra web
uv run python scripts/check_env.py    # CUDA + bf16 + model smoke test
uv run pytest
```

## Commands

```bash
# synthetic data (master seed 0; the split name is part of every sample's seed)
uv run python -m mathvlm.generate.build --split train --n 20000 --seed 0 --workers 12
uv run python -m mathvlm.generate.build --split val --n 200 --seed 0 --workers 12
# train, predict, score
uv run python -m mathvlm.train.run --name <run> --train-split train
uv run python -m mathvlm.infer --split val --adapter outputs/train/<run>/adapter
uv run python -m mathvlm.eval.run_eval --split val --pred outputs/infer/val/<run>/predictions.jsonl
# real textbook diagrams (PGDP5K, 85 MB, eval only)
uv run python -m mathvlm.real.pgdp5k --raw <path-to>/PGDP5K --splits val test
uv run python -m mathvlm.infer --split pgdp5k/val --adapter outputs/train/<run>/adapter
# prediction viewer: outputs beside their inputs; drag points to fix a diagram
uv run --extra web python -m mathvlm.web.serve   # http://127.0.0.1:8000
```

## Design invariants

Hard rules; code comments refer to them by number. Changing one is a deliberate, documented
decision.

1. **The diagram schema is frozen.** It lives only in `src/mathvlm/schema.py`
   (`SCHEMA_VERSION = "1.2"`); a change to its shape needs a version bump and a changelog entry
   there.
2. **The instruction string is one constant** in `src/mathvlm/prompts.py`, used identically by
   training and inference.
3. **Thinking mode is off**: a reasoning preamble breaks JSON parsing.
4. **Generation is deterministic given a seed**: sample `i` is identical alone or in a batch
   (bit-exact ground-truth JSON; PNG bytes only on the same matplotlib and fonts).
5. **Ground-truth coordinates are canonicalized** to a deterministic box in the 0-100 grid.
6. **Rotation augmentation is small (±4°)** and never changes the ground truth.
7. **Metrics are tested code**: a change in `src/mathvlm/eval/` needs a test in
   `tests/test_metrics.py`.

## Layout

```
configs/default.yaml   every setting and seed
src/mathvlm/
  schema.py            the diagram schema (frozen)
  prompts.py           the instruction string
  render.py            diagram JSON -> page image
  augment.py           blur, noise, paper texture, rotation, scale
  generate/            synthetic scenes, questions and realism; build.py is the CLI
  eval/                parsing, point matching, metrics, reports; run_eval.py is the CLI
  infer.py             batched generation (base model, or + a LoRA adapter)
  train/               training examples and the LoRA training CLI
  real/pgdp5k.py       PGDP5K -> our schema (eval only)
  web/                 prediction viewer; static/diagram.js ports render.py, change both together
tests/                 pytest; test_metrics.py matters most
data/, outputs/        generated data and run outputs (gitignored)
```

## Environment

- An NVIDIA GPU with CUDA 13 (developed on a 16 GB card). `torch` comes from the
  `pytorch-cu130` index in `pyproject.toml`, which is needed for Blackwell (sm_120) GPUs.
- If the repo sits on a slow filesystem (such as a Windows drive under WSL), keep `data/`,
  `outputs/` and `.venv` on a fast local disk and symlink them into the repo. Never write
  datasets into the repo tree.
- Always `uv run`; never install into another environment.

## Conventions

- Ruff for lint and format, line length 100: `uv run ruff check . --fix && uv run ruff format .`
- All tunables in `configs/default.yaml`, read through `mathvlm.config`; no magic numbers.
- Commit after each completed unit of work, Conventional Commits style; never leave the tests red.
