"""Image-level augmentation: paper, lighting, rotation, scale, blur, noise, JPEG.

Applied to the rendered page after rendering. The ground truth is never touched
(design invariant 6), so rotation stays small: the model must still map the image to
coordinates in the canonical, unrotated frame.

Two halves:
- ``sample_params(rng)`` draws an ``AugmentParams`` from the config ranges. This is the
  only randomness, and its result is recorded in the dataset manifest.
- ``apply(image, params)`` is a pure function of the image and the parameters (the
  texture and noise fields are generated from ``params.seed``), so any augmented image
  can be reproduced exactly from what the manifest records.

Pipeline order mimics how a printed page becomes a photo or scan: ink on paper, then
lighting (brightness/contrast), then the camera's tilt and distance (rotation, scale),
then focus (blur), sensor noise, and finally JPEG compression.

    uv run python -m mathvlm.augment --demo     # writes a grid of escalating levels
"""

from __future__ import annotations

import argparse
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from mathvlm.config import get_config

_AC = get_config().augment


@dataclass(frozen=True)
class AugmentParams:
    rotation_deg: float = 0.0  # counter-clockwise
    scale: float = 1.0
    blur_sigma: float = 0.0  # pixels
    noise_std: float = 0.0  # fraction of the 0-255 range
    jpeg_quality: int | None = None  # None: no JPEG round trip
    paper_strength: float = 0.0
    paper_tint: tuple[int, int, int] = (255, 255, 255)
    brightness: float = 1.0
    contrast: float = 1.0
    seed: int = 0  # drives the paper texture and noise fields

    @property
    def is_identity(self) -> bool:
        return self == AugmentParams(seed=self.seed)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["paper_tint"] = list(self.paper_tint)
        return d


IDENTITY = AugmentParams()


def sample_params(rng: np.random.Generator) -> AugmentParams:
    """Draw augmentation parameters from the config. Draw order is fixed, so equal RNG
    states give equal parameters."""
    seed = int(rng.integers(0, 2**31 - 1))
    if not _AC.enabled or rng.random() < _AC.clean_prob:
        return AugmentParams(seed=seed)
    rotation = float(rng.uniform(-_AC.rotation_deg, _AC.rotation_deg))
    scale = float(rng.uniform(*_AC.scale_range))
    blur = float(rng.uniform(*_AC.blur_sigma_range))
    noise = float(rng.uniform(*_AC.gaussian_noise_std))
    jpeg = int(rng.integers(_AC.jpeg_quality_range[0], _AC.jpeg_quality_range[1] + 1))
    use_jpeg = bool(rng.random() < _AC.jpeg_prob)
    paper = float(rng.uniform(*_AC.paper_texture_strength))
    tint = tuple(_AC.paper_tints[int(rng.integers(len(_AC.paper_tints)))])
    brightness = float(rng.uniform(*_AC.brightness_range))
    contrast = float(rng.uniform(*_AC.contrast_range))
    # drawn last, so adding it left every earlier draw unchanged
    if rng.random() < _AC.small_prob:
        scale = float(rng.uniform(*_AC.small_scale_range))
    return AugmentParams(
        rotation_deg=rotation,
        scale=scale,
        blur_sigma=blur,
        noise_std=noise,
        jpeg_quality=jpeg if use_jpeg else None,
        paper_strength=paper,
        paper_tint=tint,
        brightness=brightness,
        contrast=contrast,
        seed=seed,
    )


def params_at_level(level: float, seed: int = 0) -> AugmentParams:
    """Deterministic parameters that move from identity (level 0) to the harsh end of
    every configured range (level 1). For demos, not for training."""
    lv = float(np.clip(level, 0.0, 1.0))
    q_lo, q_hi = _AC.jpeg_quality_range
    return AugmentParams(
        rotation_deg=lv * _AC.rotation_deg,
        scale=1.0 + lv * (_AC.scale_range[0] - 1.0),
        blur_sigma=lv * _AC.blur_sigma_range[1],
        noise_std=lv * _AC.gaussian_noise_std[1],
        jpeg_quality=None if lv == 0 else round(q_hi - lv * (q_hi - q_lo)),
        paper_strength=lv * _AC.paper_texture_strength[1],
        paper_tint=tuple(_AC.paper_tints[2]) if lv > 0 else (255, 255, 255),
        brightness=1.0 + lv * (_AC.brightness_range[0] - 1.0),
        contrast=1.0 + lv * (_AC.contrast_range[0] - 1.0),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _paper_field(h: int, w: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative paper/lighting field in (0, 1]: smooth, uneven shading plus fine
    grain. 1 means untouched. The shading comes from a 3x3 random grid upsampled to the
    image, which gives broad lighting gradients; finer grids read as stains."""
    coarse = rng.normal(0.0, 1.0, (3, 3)).astype(np.float32)
    low = np.asarray(Image.fromarray(coarse, mode="F").resize((w, h), Image.Resampling.BICUBIC))
    low = low / max(float(np.abs(low).max()), 1e-6)  # -> [-1, 1]
    shade = 1.0 - strength * (0.5 + 0.5 * low)  # in [1 - strength, 1]
    grain = rng.normal(0.0, 0.08 * strength, (h, w)).astype(np.float32)
    return np.clip(shade + grain, 0.0, 1.0)


def _border_color(arr: np.ndarray) -> tuple[int, int, int]:
    """Median color of the image border: what the paper looks like at the edges, used to
    fill the corners that rotation exposes."""
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def apply(image: Image.Image, p: AugmentParams) -> Image.Image:
    """Apply ``p`` to an RGB image. Pure: same image and params give the same pixels."""
    if p.is_identity:
        return image.copy()
    rng = np.random.default_rng(p.seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # 1. ink on paper: multiply blend keeps ink dark and turns white into paper
    if p.paper_strength > 0 or p.paper_tint != (255, 255, 255):
        field = _paper_field(h, w, p.paper_strength, rng)
        tint = np.asarray(p.paper_tint, dtype=np.float32) / 255.0
        arr = arr * field[..., None] * tint
    # 2. lighting
    arr = np.clip(((arr - 0.5) * p.contrast + 0.5) * p.brightness, 0.0, 1.0)
    out = Image.fromarray(np.round(arr * 255.0).astype(np.uint8), "RGB")

    # 3. camera tilt and distance
    if p.rotation_deg != 0:
        fill = _border_color(np.asarray(out))
        out = out.rotate(
            p.rotation_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill
        )
    if p.scale != 1.0:
        size = (max(1, round(out.width * p.scale)), max(1, round(out.height * p.scale)))
        resample = Image.Resampling.LANCZOS if p.scale < 1 else Image.Resampling.BICUBIC
        out = out.resize(size, resample)
    # 4. focus
    if p.blur_sigma > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(radius=p.blur_sigma))
    # 5. sensor noise
    if p.noise_std > 0:
        a = np.asarray(out, dtype=np.float32)
        a = a + rng.normal(0.0, p.noise_std * 255.0, a.shape).astype(np.float32)
        out = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8), "RGB")
    # 6. compression
    if p.jpeg_quality is not None:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=int(p.jpeg_quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


def augment(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, AugmentParams]:
    p = sample_params(rng)
    return apply(image, p), p


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _label(im: Image.Image, text: str) -> Image.Image:
    """Put a caption strip under an image (plain PIL, so it never touches matplotlib)."""
    from PIL import ImageDraw

    strip = Image.new("RGB", (im.width, 28), (255, 255, 255))
    ImageDraw.Draw(strip).text((8, 8), text, fill=(0, 0, 0))
    out = Image.new("RGB", (im.width, im.height + strip.height), (255, 255, 255))
    out.paste(im, (0, 0))
    out.paste(strip, (0, im.height))
    return out


def demo(out_dir: Path, seed: int, n_random: int = 8) -> list[Path]:
    from mathvlm.config import sample_rng
    from mathvlm.fixtures import PAGES
    from mathvlm.render import contact_sheet, render_page
    from mathvlm.schema import Transcription

    out_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(Transcription.model_validate(PAGES["page_bisector_display_math"])).image

    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    ims = [_label(apply(page, params_at_level(lv, seed)), f"level {lv:.2f}") for lv in levels]
    p1 = out_dir / "augment_levels.png"
    contact_sheet(ims, cols=len(levels), pad=8).save(p1)

    draws = []
    for i in range(n_random):
        params = sample_params(sample_rng(seed, "augment_demo", i))
        tag = "clean" if params.is_identity else (
            f"rot {params.rotation_deg:+.1f}  x{params.scale:.2f}  blur {params.blur_sigma:.1f}"
            f"  jpeg {params.jpeg_quality or '-'}"
        )  # fmt: skip
        draws.append(_label(apply(page, params), tag))
    p2 = out_dir / "augment_random.png"
    contact_sheet(draws, cols=4, pad=8).save(p2)
    return [p1, p2]


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.augment", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--demo", action="store_true", help="write demo grids for visual review")
    ap.add_argument("--out", type=Path, default=None, help="default: <outputs_dir>/augment")
    ap.add_argument("--seed", type=int, default=cfg.seed)
    args = ap.parse_args(argv)
    if not args.demo:
        ap.error("nothing to do: pass --demo")
    out = args.out or cfg.paths.resolved("outputs_dir") / "augment"
    for p in demo(out, args.seed):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
