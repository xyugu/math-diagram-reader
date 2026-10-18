# mathvlm

Fine-tuning a small vision-language model to read images of math questions and turn them into
structured output.

**Input:** an image of a math question (printed or handwritten, possibly a phone photo, possibly
with a geometry diagram).

**Output:** a single JSON object with exactly two keys:

- `"text"`: the full question as Markdown, with math in LaTeX (`$...$` inline, `$$...$$` display)
- `"diagram"`: `null` if there is no figure, otherwise a scene graph that can be re-plotted

Base model: [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (natively multimodal,
post-trained; the `-Base` variant is not the one we want). Training stack: Unsloth + TRL, LoRA on
the language layers with the vision tower frozen to start.

## Quickstart

```bash
uv sync                                   # Phase 1 dependencies
uv run python scripts/check_env.py        # CUDA + bf16 + model smoke test
uv run pytest                             # test suite
```

All tunables live in `configs/default.yaml`.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Env, schema, renderer, synthetic generator, eval harness, zero-shot baseline | done |
| 2 | LoRA SFT on synthetic geometry + public math OCR data | in progress (synthetic done) |
| 3 | Realism augmentation; eval on real textbook diagrams (e.g. PGDP5K) | in progress (eval done) |
| 4 | Optional GRPO using eval metrics as rewards; export for local inference | not started |

## Results

Synthetic splits, master seed 0: val (200 samples, 177 with a diagram) for model selection,
test (500, 435 with a diagram) scored once per final model. Greedy, batch 8. `sft20k`: LoRA
r=16 on the language model, 1 epoch over 20k synthetic samples (1,250 steps, 42 min),
`outputs/train/sft20k/`.

| Metric | Zero-shot val | Zero-shot test | sft20k val | sft20k test |
|--------|---------------|----------------|------------|-------------|
| JSON parse rate (strict / with fence stripping) | 0.0% / 99.0% | 0.0% / 99.0% | 100% / 100% | 100% / 100% |
| Schema valid rate | 97.5% | 99.0% | 100% | 100% |
| Diagram detection accuracy | 10.5% | 13.0% | 100% | 100% |
| Text NED, normalized / exact match | 0.142 / 3.0% | 0.150 / 1.8% | 0.000 / 99.0% | 0.000 / 99.4% |
| Structure F1, strict / geometric | 0.000 / 0.000 | 0.000 / 0.000 | 0.989 / 0.989 | 0.988 / 0.988 |
| F1 marks / labels | 0.000 / 0.000 | 0.000 / 0.000 | 0.980 / 0.930 | 0.972 / 0.946 |
| Point error, mean / median (grid units) | n/a | n/a | 1.86 / 1.51 | 1.84 / 1.43 |

12 test samples (all text-only) share their answer with a training sample; leaving them out
changes no number above. These are in-distribution scores: the model is tested on the same
generator it was trained on, so they say nothing yet about real textbook images (Phase 3).

The zero-shot model reads the text well but formats it wrong (no `$` delimiters, Unicode
math, code fences) and never predicts a diagram. After fine-tuning, nearly all remaining
errors are a label's value put on a neighbouring angle or segment, and point imprecision
(~1.5 units on clean images, ~2.2 at the largest augmentation rotation).

### Real textbook diagrams (Phase 3)

PGDP5K val (500 diagram-only images from textbooks, converted by `mathvlm.real.pgdp5k`;
ground-truth text is `""`; arrow marks are ignored because PGDP5K does not annotate them).
Scores use the metric of `63d9cba` (angles compared by their rays, trailing units in
labels ignored); synthetic scores are unchanged by it, PGDP5K labels rose ~0.04-0.10.
`sft_v2`: same training as `sft20k`, on 20k samples from the realism generator
(`generate/realism.py`: rays with arrowheads, figure-only pages, hidden point names,
numbered angles; textbook colors and small screenshots), `outputs/train/sft_v2/`.

| Metric | Zero-shot | sft20k | sft_v2 | sft_v3 |
|--------|-----------|--------|--------|--------|
| JSON parse rate / schema valid rate | 90.0% / 88.4% | 95.0% / 55.4% | 99.8% / 84.8% | 99.8% / 88.0% |
| Diagram detection accuracy | 1.0% | 94.2% | 99.6% | 99.8% |
| Text exact (`""`) | 0% | 0% | 99.4% | 99.4% |
| Structure F1, all samples (strict) | 0.000 | 0.489 | 0.679 | 0.696 |
| F1 points / segments / marks / labels | 0 / 0 / 0 / 0 | 0.59 / 0.49 / 0.29 / 0.33 | 0.84 / 0.64 / 0.42 / 0.51 | 0.86 / 0.66 / 0.43 / 0.51 |
| Mean per-diagram F1: lettered (291) / no letters (209) | - | 0.67 / 0.02 | 0.77 / 0.50 | 0.78 / 0.55 |
| Point error, mean / median (grid units) | n/a | 5.18 / 2.46 | 5.83 / 2.68 | 6.11 / 2.95 |

On the synthetic val split `sft_v2` holds `sft20k`'s scores (structure F1 0.984 vs 0.989,
point error median 1.51 vs 1.51), and on `val_v2` (realism generator) it scores 0.922 vs
0.745.

What sft20k got wrong on real figures was mostly conventions its training data never
showed: 42% of PGDP5K val diagrams carry no letters at all (sft20k F1 ~0.02 on them), 53% of
points are unlabeled line ends and intersections (sft20k named them `bottom_left`, `1`, `B1`,
schema-invalid), and it never saw a page without question text. `sft_v2` fixes most of
that; what remained was mainly labels put on a neighbouring target and marks (F1 ~0.4-0.5),
point precision on real images, line-only figures (17% of PGDP5K) and arc labels.

Checkpoint (2026-10-26, branch `phase3-realism`): the generator since `sft_v2` adds dense
measured labels on figure-only pages (pieces of split segments, several sectors per vertex,
arcs, units) and a line-only family (`line_crossings`). `train_v3`/`val_v3` (20k/200) are
generated from `aa58051`, and `sft_v3` trained on them (structure F1 on the synthetic val
split 0.988, no regression). Diminishing returns on real figures: labels stayed at 0.51 and
point error rose a little. Lowercase point names (PGDP5K's line names `m`, `a`) still made 19
of sft_v3's 59 invalid outputs; `train_v4`/`val_v4` (from `140dc81`) add line names:

    uv run python -m mathvlm.train.run --name sft_v4 --train-split train_v4 --val-split val_v4

## Design invariants

These are hard rules. Code comments refer to them by number. Changing one is a deliberate,
documented decision, never a side effect.

1. **The diagram schema is frozen.** It lives in exactly one module, `src/mathvlm/schema.py`,
   currently `SCHEMA_VERSION = "1.2"` (1.2 added arc labels, free-text labels and arrowheads for real textbook figures). A change to its shape requires a version bump and a
   changelog entry in that file.
2. **The instruction string is a single constant** in `src/mathvlm/prompts.py`. Training and
   inference must use the identical string; never inline a paraphrase anywhere else.
3. **Thinking mode is off.** Qwen3.5 supports `enable_thinking`; a reasoning preamble breaks
   JSON parsing. It is pinned off in `prompts.py` and the config.
4. **Generation is deterministic given a seed.** Per-sample seeds are derived via
   `SeedSequence(master).spawn()`, so sample `i` is identical whether generated alone or in a
   batch. The guarantee is bit-exact *ground-truth JSON*; PNG bytes are only reproducible on
   the same pinned matplotlib and fonts.
5. **Ground-truth coordinates are canonicalized** to a deterministic bounding box in the 0-100
   grid before being written. The schema is only defined up to a similarity transform, so
   without this the model trains against an arbitrary target.
6. **Rotation augmentation is small (±4°) and never touches ground truth.** It is nuisance
   noise. Large rotations would make image-to-canonical-coordinates ill-posed.
7. **Metrics are tested code.** Every later phase is scored by `src/mathvlm/eval/`. Changes
   there require a corresponding test in `tests/test_metrics.py`.

## Layout

```
configs/default.yaml      the config file: all settings and seeds, nothing hardcoded
src/mathvlm/
  config.py               validated config loader + seed derivation
  prompts.py              the single instruction constant
  modeling.py             model + processor loading
  schema.py               FROZEN: pydantic models, validator, canonicalize()
  render.py               diagram JSON -> PNG; question-above-diagram page layout
  fixtures.py             hand-drawn reference diagrams for renderer checks and tests
  augment.py              blur, noise, paper texture, rotation, scale
  generate/
    primitives.py         intersect, foot, bisector, incenter, circumcenter, tangent
    scenes.py             scene families (triangles, circles, quadrilaterals, tangents)
    questions.py          templated LaTeX question consistent with the scene
    build.py              CLI: emit N samples as image + ground-truth JSON
  eval/
    normalize.py          LaTeX text normalization
    parse.py              raw output -> validated JSON (fence stripping, failure categories)
    matching.py           Hungarian alignment for unnamed "_P" points
    metrics.py            parse rate, NED, structure F1, geometry error
    report.py             per-example JSONL + side-by-side images + summary
    run_eval.py           CLI
  infer.py                batched generation -> raw outputs (base model, or + a LoRA adapter)
  splits.py               reading a split (synthetic `val`, or real `pgdp5k/val`): manifest + ground truth
  real/
    pgdp5k.py             PGDP5K real textbook diagrams -> our schema and split layout (eval only)
  train/
    data.py               training examples: the inference prompt + target, loss on the target
    run.py                CLI: LoRA fine-tuning (Unsloth + TRL)
  web/
    runs.py               saved predictions for the viewer: runs, samples, checking an edit
    app.py                FastAPI endpoints + the static page
    serve.py              CLI: serve the viewer on localhost
    static/diagram.js     render.py's drawing rules ported to SVG: change the two together
    static/app.js         the page: drag points and circle radii, edit the JSON, download it
scripts/check_env.py      CUDA/bf16/model smoke test
scripts/bench_generate.py generation speed benchmark (prefill, decode, batch size)
tests/                    pytest; test_metrics.py is the most important file here
data    -> ~/vlm_data     ext4 symlink (gitignored)
outputs -> ~/vlm_outputs  ext4 symlink (gitignored)
.venv   -> ~/.venvs/vlm_project   ext4 symlink (gitignored)
```

## Environment

- WSL2, Ubuntu 26.04. GPU: RTX 5070 Ti 16 GB, Blackwell **sm_120**, driver CUDA 13.4.
- `torch` must come from the `pytorch-cu130` index declared in `pyproject.toml`; the default
  PyPI wheel is not guaranteed to ship sm_120 kernels. Verify with `torch.cuda.get_arch_list()`.
- The repo is on `/mnt/e`, a 9p mount that is slow for many small files. Generated data, the
  Hugging Face cache and the virtualenv live on ext4 (the `data`/`outputs` symlinks, `HF_HOME`,
  and `.venv -> ~/.venvs/vlm_project`). Never write datasets into the repo tree. With the venv
  on 9p, `import transformers` alone took ~50 s (it scans every installed package's metadata);
  on ext4 it takes ~3 s. On a fresh clone, create the link before the first `uv sync`:
  `mkdir -p ~/.venvs/vlm_project && ln -s ~/.venvs/vlm_project .venv`.
- System RAM is 15 GB alongside 16 GB of VRAM. Pre-render datasets to disk; do not render
  matplotlib inside dataloader workers.

## Commands

```bash
uv sync                      # Phase 1 deps
uv sync --extra train        # adds unsloth/trl/peft for Phase 2
uv sync --extra train --extra web   # adds fastapi/uvicorn for the viewer (uv sync drops unlisted extras)
uv run pytest                # full test suite
uv run ruff check . --fix    # lint
uv run ruff format .         # format
uv run python scripts/check_env.py
uv run python -m mathvlm.augment --demo   # grids of escalating and random augmentation
# synthetic splits, master seed 0 (the split name is part of every sample's seed, so splits
# never share a figure; ~10% of text-only questions repeat because their template space is small)
uv run python -m mathvlm.generate.build --split val --n 200 --seed 0 --workers 12     # model selection
uv run python -m mathvlm.generate.build --split test --n 500 --seed 0 --workers 12    # final numbers only
uv run python -m mathvlm.generate.build --split train --n 20000 --seed 0 --workers 12 # ~5 min, 6 GB
uv run python -m mathvlm.eval.run_eval --split val --oracle   # GT as predictions: must be perfect
uv run python -m mathvlm.infer --split val --name zeroshot      # raw outputs, batch 8, ~2 min
uv run python -m mathvlm.eval.run_eval --split val --pred outputs/infer/val/zeroshot/predictions.jsonl
# Phase 2: prove train -> save -> infer -> eval on 32 memorized samples, then train for real
uv run python -m mathvlm.generate.build --split smoke --n 32 --seed 0
uv run python -m mathvlm.train.run --name overfit --train-split smoke --overfit   # ~4 min
uv run python -m mathvlm.infer --split smoke --adapter outputs/train/overfit/adapter
uv run python -m mathvlm.train.run --name <run> --train-split train
# Phase 3: real textbook diagrams (PGDP5K, 85 MB from Hugging Face, kept on ext4 only)
uv run python -m mathvlm.real.pgdp5k --raw ~/vlm_data/raw/pgdp5k/extracted/PGDP5K --splits val test
uv run python -m mathvlm.infer --split pgdp5k/val --adapter outputs/train/sft20k/adapter
# prediction viewer: each input beside its prediction, drawn in the browser; drag points and
# circle radii to fix a diagram, then download the JSON (nothing is written server-side)
uv run --extra web python -m mathvlm.web.serve   # http://127.0.0.1:8000
```

Always use `uv run`; do not activate the venv manually or install into another environment.

## Conventions

- `src/` layout, hatchling backend, absolute imports (`from mathvlm.schema import ...`).
- Ruff for lint and format, line length 100.
- All tunables go in `configs/default.yaml` and are read through `mathvlm.config`; no magic
  numbers in module bodies.
- Type-hint public functions. Pydantic models for anything crossing a module boundary.
- Commit after each completed unit of work, Conventional Commits style (`feat:`, `fix:`,
  `test:`, `chore:`, `docs:`). A commit should not leave the test suite red.
