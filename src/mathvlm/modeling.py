"""Model + processor loading, in one place.

Why this exists: the correct auto class for Qwen3.5 depends on the transformers
version. On the pinned 5.17.0 it registers as ``Qwen3_5ForConditionalGeneration``
under ``AutoModelForImageTextToText``; newer releases expose the same thing as
``AutoModelForMultimodalLM`` (which is what the model card shows). Resolving that
once here keeps ``scripts/check_env.py`` and ``mathvlm.infer`` from drifting.
"""

from __future__ import annotations

from typing import Any

from mathvlm.config import Config, apply_hf_env, get_config

# Preference order. The first class that exists AND has the model's architecture
# registered wins.
_LOADER_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
)


def resolve_model_class(model_id: str) -> tuple[Any, str]:
    """Pick the auto class that can actually load ``model_id``.

    Returns ``(cls, name)``. Raises if none of the candidates can.
    """
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id)

    for name in _LOADER_CANDIDATES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        mapping = getattr(cls, "_model_mapping", None)
        if mapping is not None and type(config) in mapping:
            return cls, name

    raise RuntimeError(
        f"No auto class among {_LOADER_CANDIDATES} can load {model_id!r} "
        f"(config type {type(config).__name__}, transformers {transformers.__version__})."
    )


def load_model_and_processor(cfg: Config | None = None) -> tuple[Any, Any, str]:
    """Load the base model in the configured dtype, plus its processor.

    Returns ``(model, processor, loader_name)``.
    """
    import torch
    import transformers

    cfg = cfg or get_config()
    apply_hf_env(cfg)

    model_cls, loader_name = resolve_model_class(cfg.model.id)
    processor = transformers.AutoProcessor.from_pretrained(cfg.model.id)
    model = model_cls.from_pretrained(
        cfg.model.id,
        dtype=getattr(torch, cfg.model.dtype),
        device_map=cfg.model.device_map,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.eval()
    return model, processor, loader_name
