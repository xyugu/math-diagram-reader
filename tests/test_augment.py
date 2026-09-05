"""Tests for image augmentation (src/mathvlm/augment.py) and its use in the build."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from mathvlm import augment as A
from mathvlm import schema as S
from mathvlm.config import get_config, sample_rng
from mathvlm.fixtures import PAGES
from mathvlm.generate import build as B
from mathvlm.render import render_page

AC = get_config().augment


@pytest.fixture(scope="module")
def page() -> Image.Image:
    return render_page(S.Transcription.model_validate(PAGES["page_right_triangle"])).image


def arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im, dtype=np.int16)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic():
    assert A.sample_params(sample_rng(0, "t", 1)) == A.sample_params(sample_rng(0, "t", 1))
    draws = {A.sample_params(sample_rng(0, "t", i)) for i in range(20)}
    assert len(draws) == 20


def test_sampled_params_stay_in_the_configured_ranges():
    draws = [A.sample_params(sample_rng(0, "ranges", i)) for i in range(400)]
    clean = [p for p in draws if p.is_identity]
    assert abs(len(clean) / len(draws) - AC.clean_prob) < 0.07
    for p in draws:
        if p.is_identity:
            continue
        assert abs(p.rotation_deg) <= AC.rotation_deg
        assert (AC.scale_range[0] <= p.scale <= AC.scale_range[1]) or (
            AC.small_scale_range[0] <= p.scale <= AC.small_scale_range[1]
        )
        assert AC.blur_sigma_range[0] <= p.blur_sigma <= AC.blur_sigma_range[1]
        assert AC.gaussian_noise_std[0] <= p.noise_std <= AC.gaussian_noise_std[1]
        assert AC.paper_texture_strength[0] <= p.paper_strength <= AC.paper_texture_strength[1]
        assert p.paper_tint in AC.paper_tints
        assert AC.brightness_range[0] <= p.brightness <= AC.brightness_range[1]
        assert AC.contrast_range[0] <= p.contrast <= AC.contrast_range[1]
        assert (
            p.jpeg_quality is None
            or AC.jpeg_quality_range[0] <= p.jpeg_quality <= AC.jpeg_quality_range[1]
        )
    jpeg_share = sum(p.jpeg_quality is not None for p in draws if not p.is_identity) / (
        len(draws) - len(clean)
    )
    assert abs(jpeg_share - AC.jpeg_prob) < 0.1
    augmented = [p for p in draws if not p.is_identity]
    small = sum(p.scale < AC.scale_range[0] for p in augmented) / len(augmented)
    assert abs(small - AC.small_prob) < 0.1


def test_disabled_config_gives_identity(monkeypatch):
    monkeypatch.setattr(A, "_AC", AC.model_copy(update={"enabled": False}))
    assert all(A.sample_params(sample_rng(0, "off", i)).is_identity for i in range(30))


def test_level_zero_is_identity_and_level_one_is_the_harsh_end():
    assert A.params_at_level(0.0).is_identity
    p = A.params_at_level(1.0)
    assert p.rotation_deg == AC.rotation_deg
    assert p.scale == pytest.approx(AC.scale_range[0])
    assert p.blur_sigma == AC.blur_sigma_range[1]
    assert p.jpeg_quality == AC.jpeg_quality_range[0]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def test_identity_leaves_pixels_unchanged(page):
    out = A.apply(page, A.IDENTITY)
    assert out.size == page.size
    assert np.array_equal(arr(out), arr(page))


def test_apply_is_pure(page):
    p = A.params_at_level(0.8, seed=5)
    assert np.array_equal(arr(A.apply(page, p)), arr(A.apply(page, p)))
    assert not np.array_equal(arr(A.apply(page, p)), arr(A.apply(page, replace(p, seed=6))))


def test_rotation_expands_and_fills_corners_with_paper(page):
    out = A.apply(page, A.AugmentParams(rotation_deg=4.0))
    assert out.width > page.width and out.height > page.height
    a = arr(out)
    for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert a[y, x].min() > 200  # filled with the paper color, not black


def test_scale_resizes(page):
    out = A.apply(page, A.AugmentParams(scale=0.85))
    assert out.size == (round(page.width * 0.85), round(page.height * 0.85))


def test_paper_tints_the_background_but_keeps_ink_dark(page):
    a0 = arr(page)
    ink = a0.sum(axis=2) < 150
    paper = a0.min(axis=2) == 255
    out = arr(
        A.apply(page, A.AugmentParams(paper_strength=0.25, paper_tint=(245, 240, 228), seed=1))
    )
    assert out[ink].max(axis=1).mean() < 100
    bg = out[paper]
    assert 170 < bg.mean() < 250
    assert bg[:, 2].mean() < bg[:, 0].mean()  # warm tint: less blue than red


def test_blur_removes_fine_detail(page):
    def sharpness(im):
        g = np.asarray(im.convert("L"), dtype=np.float32)
        return float(
            np.var(g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        )

    assert sharpness(A.apply(page, A.AugmentParams(blur_sigma=1.1))) < 0.6 * sharpness(page)


def test_noise_has_the_requested_strength():
    flat = Image.new("RGB", (200, 200), (128, 128, 128))
    out = arr(A.apply(flat, A.AugmentParams(noise_std=0.03, seed=3)))
    assert out.std() == pytest.approx(0.03 * 255, rel=0.15)


def test_jpeg_round_trip_changes_pixels_not_size(page):
    out = A.apply(page, A.AugmentParams(jpeg_quality=55))
    assert out.size == page.size and out.mode == "RGB"
    assert not np.array_equal(arr(out), arr(page))


# ---------------------------------------------------------------------------
# Build integration
# ---------------------------------------------------------------------------


def test_adding_the_augment_stream_left_earlier_streams_unchanged():
    """Child seed sequences are numbered in spawn order, so asking for 4 streams instead
    of 3 must not change the first three (scene, text, style)."""
    three = np.random.SeedSequence(42).spawn(3)
    four = np.random.SeedSequence(42).spawn(4)
    for a, b in zip(three, four[:3], strict=True):
        assert np.array_equal(
            np.random.default_rng(a).random(8), np.random.default_rng(b).random(8)
        )


def test_written_image_is_the_augmented_render(tmp_path):
    B.build("val", 6, 11, tmp_path)
    rows = [B.make_sample(11, "val", i) for i in range(6)]
    assert any(not s.augment.is_identity for s in rows)
    for s in rows:
        on_disk = np.asarray(Image.open(tmp_path / f"{s.id}.png").convert("RGB"))
        assert np.array_equal(on_disk, np.asarray(s.image()))
        clean = np.asarray(render_page(s.transcription, s.style).image)
        assert np.array_equal(on_disk, clean) == s.augment.is_identity


def test_manifest_records_augment_params(tmp_path):
    import json

    B.build("val", 3, 11, tmp_path, images=False)
    for line in (tmp_path / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        rebuilt = A.AugmentParams(
            **{**row["augment"], "paper_tint": tuple(row["augment"]["paper_tint"])}
        )
        assert rebuilt == B.make_sample(11, "val", row["index"]).augment


def test_demo_cli(tmp_path):
    assert A.main(["--demo", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "augment_levels.png").exists() and (tmp_path / "augment_random.png").exists()
