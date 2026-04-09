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
| 1 | Env, schema, renderer, synthetic generator, eval harness, zero-shot baseline | in progress |
| 2 | LoRA SFT on synthetic geometry + public math OCR data | not started |
| 3 | Realism augmentation; eval on real textbook diagrams (e.g. PGDP5K) | not started |
| 4 | Optional GRPO using eval metrics as rewards; export for local inference | not started |

## Design invariants

These are hard rules. Code comments refer to them by number. Changing one is a deliberate,
documented decision, never a side effect.

1. **The diagram schema is frozen.** It lives in exactly one module, `src/mathvlm/schema.py`,
   currently `SCHEMA_VERSION = "1.1"`. A change to its shape requires a version bump and a
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
    matching.py           Hungarian alignment for unnamed "_P" points
    metrics.py            parse rate, NED, structure F1, geometry error
    report.py             per-example JSONL + side-by-side images + summary
    run_eval.py           CLI
  infer.py                load model, generate, parse/repair JSON
scripts/check_env.py      CUDA/bf16/model smoke test
scripts/bench_generate.py generation speed benchmark (prefill, decode, batch size)
tests/                    pytest; test_metrics.py is the most important file here
data    -> ~/vlm_data     ext4 symlink (gitignored)
outputs -> ~/vlm_outputs  ext4 symlink (gitignored)
```

## Environment

- WSL2, Ubuntu 26.04. GPU: RTX 5070 Ti 16 GB, Blackwell **sm_120**, driver CUDA 13.4.
- `torch` must come from the `pytorch-cu130` index declared in `pyproject.toml`; the default
  PyPI wheel is not guaranteed to ship sm_120 kernels. Verify with `torch.cuda.get_arch_list()`.
- The repo is on `/mnt/e`, a 9p mount that is slow for many small files. Generated data and the
  Hugging Face cache live on ext4 (the `data`/`outputs` symlinks and `HF_HOME`). Never write
  datasets into the repo tree.
- System RAM is 15 GB alongside 16 GB of VRAM. Pre-render datasets to disk; do not render
  matplotlib inside dataloader workers.

## Commands

```bash
uv sync                      # Phase 1 deps
uv sync --extra train        # adds unsloth/trl/peft for Phase 2
uv run pytest                # full test suite
uv run ruff check . --fix    # lint
uv run ruff format .         # format
uv run python scripts/check_env.py
uv run python -m mathvlm.generate.build --split val --n 16 --seed 0
uv run python -m mathvlm.eval.run_eval --split val
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
