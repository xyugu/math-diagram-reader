"""Reading a generated split (``generate/build.py`` output): the manifest and the ground
truth. Shared by inference, evaluation and training, so all three see the same samples in
the same order.
"""

from __future__ import annotations

import json
from pathlib import Path

from mathvlm.config import Config, get_config
from mathvlm.schema import Transcription, validate_output_json


def split_dir(split: str, cfg: Config | None = None) -> Path:
    """Default location of a split: ``<data_dir>/synth/<split>``."""
    cfg = cfg or get_config()
    return cfg.paths.resolved("data_dir") / "synth" / split


def read_manifest(data_dir: Path) -> list[dict]:
    """Manifest rows in index order."""
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; generate the split first")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["index"])


def load_split(data_dir: Path) -> list[tuple[dict, Transcription]]:
    """Manifest rows (in index order) with their validated ground truth."""
    out = []
    for row in read_manifest(data_dir):
        report = validate_output_json((data_dir / row["gt"]).read_text())
        if not report.ok:
            raise ValueError(f"{row['id']}: ground truth is not schema-valid: {report.errors}")
        out.append((row, report.value))
    return out
