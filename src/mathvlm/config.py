"""Config loading. All settings come from ``configs/default.yaml``; nothing is hardcoded
in module bodies.

Also owns seed derivation. Per-sample seeds are *spawned* from the master seed rather
than drawn from a running RNG, so sample ``i`` is identical whether it was generated
alone or inside a batch (README.md, design invariant 4).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Base):
    data_dir: str
    outputs_dir: str
    hf_home: str

    def resolved(self, name: str) -> Path:
        raw = getattr(self, name)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p


class ModelConfig(_Base):
    id: str
    dtype: str
    device_map: str
    attn_implementation: str
    enable_thinking: bool
    max_new_tokens: int
    do_sample: bool
    batch_size: int


class SchemaConfig(_Base):
    grid_min: int
    grid_max: int
    unnamed_prefix: str
    consistency_collinear_tol: float
    consistency_angle_tol_deg: float
    consistency_length_tol: float


class RenderConfig(_Base):
    dpi: int
    figure_width_in: float
    question_width_in: float
    page_margin_in: float
    text_diagram_gap_in: float
    line_spacing: float
    diagram_pad: float
    usetex: bool
    mathtext_fontset: str
    fontsets: list[str]
    text_family: dict[str, str]
    ink: str
    inks: list[str]
    line_width: float
    line_width_jitter: tuple[float, float]
    point_marker_size: float
    point_marker_size_jitter: tuple[float, float]
    point_dot_prob: float
    label_fontsize: float
    label_fontsize_jitter: tuple[float, float]
    label_italic_prob: float
    label_offset_pt: float
    question_fontsize: float
    question_fontsize_jitter: tuple[float, float]
    mark_size: float
    mark_size_jitter: tuple[float, float]
    angle_arc_prob: float
    background: str


class AugmentConfig(_Base):
    enabled: bool
    clean_prob: float
    rotation_deg: float
    scale_range: tuple[float, float]
    blur_sigma_range: tuple[float, float]
    gaussian_noise_std: tuple[float, float]
    jpeg_quality_range: tuple[int, int]
    jpeg_prob: float
    paper_texture_strength: tuple[float, float]
    paper_tints: list[tuple[int, int, int]]
    brightness_range: tuple[float, float]
    contrast_range: tuple[float, float]


class QuestionConfig(_Base):
    max_givens: int
    max_length_givens: int
    figure_label_prob: float
    text_given_prob: float
    decimal_length_prob: float
    length_scale: tuple[float, float]
    variable_unknown_prob: float
    algebraic_prob: float
    family_ask_prob: float
    two_part_prob: float


class GenerateConfig(_Base):
    families: dict[str, float]
    no_diagram_fraction: float
    min_angle_deg: float
    min_side_len: float
    min_point_separation: float
    margin: float
    max_rejection_attempts: int
    incidence_tol: float
    horizontal_base_prob: float
    mark_prob: float
    unnamed_aux_prob: float
    prime_name_prob: float
    label_schemes: dict[str, float]
    questions: QuestionConfig


class EvalConfig(_Base):
    allow_rotation: bool
    allow_reflection: bool
    match_max_dist: float
    align_inlier_dist: float
    align_max_iters: int
    align_hypotheses_kept: int
    align_max_hypotheses: int
    side_by_side_limit: int

    @model_validator(mode="after")
    def _check(self) -> EvalConfig:
        if self.allow_reflection and not self.allow_rotation:
            raise ValueError("eval.allow_reflection requires eval.allow_rotation")
        return self


class Config(_Base):
    seed: int
    paths: PathsConfig
    model: ModelConfig
    schema_: SchemaConfig = Field(alias="schema")
    render: RenderConfig
    augment: AugmentConfig
    generate: GenerateConfig
    eval: EvalConfig

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the config file. ``extra="forbid"`` everywhere, so a typo in
    the YAML is a loud error rather than a silently ignored setting."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


def apply_hf_env(cfg: Config | None = None) -> Path:
    """Point the HF cache at ext4 before transformers is imported anywhere.

    The repo lives on a 9p mount; a multi-GB model cache there would be painfully slow.
    """
    cfg = cfg or get_config()
    hf_home = cfg.paths.resolved("hf_home")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    return hf_home


def sample_seed(master_seed: int, split: str, index: int) -> int:
    """Derive a stable, order-independent seed for one sample.

    Keyed on (master_seed, split, index) so regenerating sample 7 on its own gives
    byte-identical ground truth to generating it inside a batch of 200.
    """
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(_split_key(split), index))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def sample_rng(master_seed: int, split: str, index: int) -> np.random.Generator:
    """The per-sample numpy Generator. The only randomness source the generator may use."""
    return np.random.default_rng(sample_seed(master_seed, split, index))


def _split_key(split: str) -> int:
    """Stable small integer for a split name (hash() is salted per process, so not that)."""
    import hashlib

    digest = hashlib.sha256(split.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
