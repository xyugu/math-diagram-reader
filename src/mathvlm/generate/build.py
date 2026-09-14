"""Build synthetic samples: image + ground-truth JSON.

    uv run python -m mathvlm.generate.build --split val --n 200 --seed 0

Writes, under ``<data_dir>/synth/<split>/`` (ext4 via the ``data`` symlink):

    <id>.png        the rendered, then augmented, question page
    <id>.json       the ground truth, a schema-valid {"text", "diagram"} object
    manifest.jsonl  one line per sample: id, family, kind, seeds, style, augmentation
                    parameters, question meta

and a contact sheet of the first samples to ``<outputs_dir>/generate/``.

Determinism (design invariant 4): sample ``index`` depends only on (master seed, split,
index), never on what else is generated, how many workers run, or in what order. Its
four concerns draw from independent RNG streams spawned from the sample seed (scene,
text, style, augmentation), so editing the question templates does not move the
geometry. Child streams are numbered in spawn order, so adding a stream at the end
leaves the earlier ones, and every existing sample, unchanged. The
ground-truth JSON is reproducible across machines with the same pinned dependencies;
in principle a last-bit difference in a platform's trig functions could flip an integer
rounding, though rejection sampling makes that vanishingly rare.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from mathvlm.augment import AugmentParams
from mathvlm.augment import apply as apply_augment
from mathvlm.augment import sample_params as sample_augment
from mathvlm.config import get_config, sample_seed
from mathvlm.generate.questions import compose, text_only_question
from mathvlm.generate.realism import add_rays, hide_names, number_angles
from mathvlm.generate.scenes import BUILDERS, Namer, chance, to_grid, validate_scene, wpick
from mathvlm.render import RenderStyle, contact_sheet, render_page, sample_style
from mathvlm.schema import (
    Transcription,
    consistency_issues,
    output_to_obj,
    renumber_unnamed,
    to_target_json,
    validate_output,
)


@dataclass
class Sample:
    id: str
    split: str
    index: int
    seed: int
    family: str
    kind: str
    transcription: Transcription
    style: RenderStyle
    augment: AugmentParams
    meta: dict

    def manifest_row(self) -> dict:
        return {
            "id": self.id,
            "split": self.split,
            "index": self.index,
            "seed": self.seed,
            "family": self.family,
            "kind": self.kind,
            "image": f"{self.id}.png",
            "gt": f"{self.id}.json",
            "style": asdict(self.style),
            "augment": self.augment.to_dict(),
            **self.meta,
        }

    def image(self):
        """The training image: the rendered page with this sample's augmentation."""
        return apply_augment(render_page(self.transcription, self.style).image, self.augment)


def _streams(seed: int) -> tuple[np.random.Generator, ...]:
    """(scene, text, style, augmentation, realism). Append new streams at the end only."""
    return tuple(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5))


def make_sample(master_seed: int, split: str, index: int) -> Sample:
    """Generate one sample. Pure function of (master_seed, split, index)."""
    gcfg = get_config().generate
    seed = sample_seed(master_seed, split, index)
    rng_scene, rng_text, rng_style, rng_aug, rng_real = _streams(seed)
    rcfg = gcfg.realism
    sid = f"{split}_{index:06d}"

    if chance(rng_scene, gcfg.no_diagram_fraction):
        text, kind = text_only_question(rng_text)
        report = validate_output({"text": text, "diagram": None})
        assert report.ok, report.errors
        return Sample(sid, split, index, seed, "no_diagram", kind, report.value,
                      sample_style(rng_style), sample_augment(rng_aug), {"attempts": 1})  # fmt: skip

    families = list(gcfg.families)
    family = wpick(rng_scene, families, [gcfg.families[f] for f in families])
    rejections: Counter = Counter()
    for attempt in range(1, gcfg.max_rejection_attempts + 1):
        namer = Namer(rng_scene, gcfg)
        raw = BUILDERS[family](rng_scene, namer, gcfg)
        n_rays = add_rays(raw, rng_real, namer, rcfg)
        scene = to_grid(raw, rng_scene, gcfg)
        problems = validate_scene(scene, gcfg)
        if problems:
            rejections[problems[0].split(" ")[0]] += 1
            continue
        composed = compose(scene, rng_text, gcfg)
        diagram = {
            "points": {k: [int(x), int(y)] for k, (x, y) in scene.points.items()},
            "segments": [list(s) for s in scene.segments],
            "circles": [{"center": c, "radius": r} for c, r in scene.circles],
            "marks": list(scene.marks),
            "labels": list(composed.labels),
        }
        text = composed.text
        real_meta: dict = {"rays": n_rays, "diagram_only": False}
        if chance(rng_real, rcfg.diagram_only_prob):  # the figure alone, as in PGDP5K
            text = ""
            real_meta["diagram_only"] = True
            real_meta["numbered_angles"] = number_angles(diagram, rng_real, rcfg)
            real_meta["hidden_names"] = hide_names(diagram, rng_real, rcfg)
            # with the question gone, only the givens written on the figure remain
            real_meta["givens"] = [{**g, "in_text": False}
                                   for g in composed.meta.get("givens", []) if g["in_figure"]]  # fmt: skip
        obj = {"text": text, "diagram": renumber_unnamed(diagram)}
        report = validate_output(obj)
        if not report.ok:
            rejections["schema"] += 1
            continue
        if consistency_issues(report.value.diagram):
            rejections["consistency"] += 1
            continue
        meta = {
            "attempts": attempt,
            "rejections": dict(rejections),
            "label_scheme": namer.scheme,
            **composed.meta,
            **real_meta,
        }
        return Sample(sid, split, index, seed, family, scene.kind, report.value,
                      sample_style(rng_style), sample_augment(rng_aug), meta)  # fmt: skip
    raise RuntimeError(
        f"{sid}: no valid {family} scene in {gcfg.max_rejection_attempts} attempts {dict(rejections)}"
    )


def _write(args: tuple[int, str, int, str, bool]) -> dict:
    master_seed, split, index, out, images = args
    s = make_sample(master_seed, split, index)
    out_dir = Path(out)
    gt = json.dumps(output_to_obj(s.transcription), indent=2, ensure_ascii=False) + "\n"
    (out_dir / f"{s.id}.json").write_text(gt)
    if images:
        s.image().save(out_dir / f"{s.id}.png")
    row = s.manifest_row()
    row["target_chars"] = len(to_target_json(s.transcription))
    return row


def build(
    split: str,
    n: int,
    master_seed: int,
    out: Path,
    start: int = 0,
    workers: int = 1,
    images: bool = True,
) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(master_seed, split, i, str(out), images) for i in range(start, start + n)]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            rows = list(ex.map(_write, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    else:
        rows = [_write(j) for j in jobs]
    with open(out / "manifest.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.generate.build", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=cfg.seed, help=f"master seed (default {cfg.seed})")
    ap.add_argument("--start", type=int, default=0, help="first sample index")
    ap.add_argument("--out", type=Path, default=None, help="default: <data_dir>/synth/<split>")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-images", action="store_true", help="ground truth only (fast)")
    ap.add_argument("--sheet", type=int, default=16, help="samples in the contact sheet (0: none)")
    args = ap.parse_args(argv)

    out = args.out or cfg.paths.resolved("data_dir") / "synth" / args.split
    t0 = time.time()
    rows = build(args.split, args.n, args.seed, out, args.start, args.workers, not args.no_images)
    dt = time.time() - t0

    fam = Counter(r["family"] for r in rows)
    attempts = [r["attempts"] for r in rows]
    print(f"wrote {len(rows)} samples to {out} in {dt:.1f}s ({len(rows) / dt:.1f}/s)")
    for f, c in sorted(fam.items()):
        print(f"  {f:26s} {c:5d}  ({c / len(rows):.0%})")
    print(f"  attempts per sample: mean {np.mean(attempts):.2f}, max {max(attempts)}")

    if args.sheet and not args.no_images:
        from PIL import Image

        ims = []
        for r in rows[: args.sheet]:
            im = Image.open(out / r["image"])
            ims.append(im.resize((im.width // 2, im.height // 2), Image.Resampling.LANCZOS))
        sheet_dir = cfg.paths.resolved("outputs_dir") / "generate"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        path = sheet_dir / f"{args.split}_contact.png"
        contact_sheet(ims, cols=4, pad=8).save(path)
        print(f"contact sheet: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
