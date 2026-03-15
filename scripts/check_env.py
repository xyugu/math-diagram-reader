"""Phase 1, task 1: environment smoke test.

Checks, in order and failing loudly:
  1. CUDA is visible and the build carries sm_120 (Blackwell) kernels
  2. bf16 is supported and a bf16 matmul actually produces finite numbers
  3. the base model loads in bf16
  4. one image runs end-to-end through processor + generate

Run:  uv run python scripts/check_env.py
      uv run python scripts/check_env.py --no-model   # steps 1-2 only
"""

from __future__ import annotations

import argparse
import io
import sys
import time

# Must happen before transformers is imported: keeps the model cache off the 9p mount.
from mathvlm.config import apply_hf_env, get_config

CFG = get_config()
HF_HOME = apply_hf_env(CFG)

OK = "\x1b[32m  OK \x1b[0m"
BAD = "\x1b[31mFAIL \x1b[0m"
WARN = "\x1b[33mWARN \x1b[0m"

EXPECTED_ARCH = "sm_120"  # RTX 5070 Ti == compute capability 12.0


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def check_torch() -> bool:
    import torch

    section("1. torch / CUDA / Blackwell")
    print(f"  torch           {torch.__version__}")
    print(f"  built for CUDA  {torch.version.cuda}")

    ok = True
    if not torch.cuda.is_available():
        print(f"{BAD} torch.cuda.is_available() is False")
        return False

    arch_list = torch.cuda.get_arch_list()
    cap = torch.cuda.get_device_capability(0)
    print(f"  device          {torch.cuda.get_device_name(0)}")
    print(f"  capability      sm_{cap[0]}{cap[1]}")
    print(f"  arch_list       {' '.join(arch_list)}")

    if EXPECTED_ARCH in arch_list:
        print(f"{OK} {EXPECTED_ARCH} kernels present")
    else:
        print(
            f"{BAD} {EXPECTED_ARCH} missing from arch_list -> wrong wheel, expect "
            f"'no kernel image is available' at runtime"
        )
        ok = False

    try:
        import triton

        print(f"  triton          {triton.__version__}")
    except ImportError:
        print(f"{WARN} triton not importable (needed for Phase 2 training, not Phase 1)")

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"  VRAM            {total:.1f} GB total, {free:.1f} GB free")
    return ok


def check_bf16() -> bool:
    import torch

    section("2. bf16")
    if not torch.cuda.is_bf16_supported():
        print(f"{BAD} torch.cuda.is_bf16_supported() is False")
        return False
    print(f"{OK} bf16 reported supported")

    # Reporting support and actually computing correctly are different claims.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    out = (a @ b).float()
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        print(f"{BAD} bf16 matmul produced non-finite values")
        return False

    # Sanity-check it against an fp32 reference; bf16 has ~3 decimal digits so the
    # tolerance is deliberately loose.
    ref = a.float() @ b.float()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"{OK} bf16 matmul finite, relative error vs fp32 = {rel:.2e}")
    return rel < 1e-2


def make_probe_image():
    """A small synthetic 'math question' page, so step 4 exercises a realistic input.

    Deliberately hand-built here rather than imported from mathvlm.render: this script
    must stay runnable before the renderer exists.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, (ax_q, ax_d) = plt.subplots(
        2, 1, figsize=(5.0, 5.4), height_ratios=[1, 3], dpi=CFG.render.dpi
    )
    fig.patch.set_facecolor(CFG.render.background)

    ax_q.axis("off")
    ax_q.text(
        0.02,
        0.6,
        "In triangle $ABC$, $AB = 5$ and $\\angle ACB = 90^\\circ$.\nFind the length of $BC$.",
        fontsize=CFG.render.question_fontsize,
        va="center",
        ha="left",
        wrap=True,
    )

    ax_d.set_aspect("equal")
    ax_d.axis("off")
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(100, 0)  # y points down, matching the schema
    pts = {"A": (15, 80), "B": (85, 80), "C": (40, 25)}
    for p, q in [("A", "B"), ("B", "C"), ("C", "A")]:
        ax_d.plot(*zip(pts[p], pts[q], strict=True), color="black", lw=CFG.render.line_width)
    for name, (x, y) in pts.items():
        ax_d.plot([x], [y], "o", color="black", ms=CFG.render.point_marker_size)
        ax_d.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(6, -8),
            fontsize=CFG.render.label_fontsize,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def check_model() -> bool:
    import torch

    from mathvlm.prompts import INSTRUCTION, build_messages

    section("3. load base model")
    print(f"  model id        {CFG.model.id}")
    print(f"  HF_HOME         {HF_HOME}")

    import transformers

    from mathvlm.modeling import load_model_and_processor

    print(f"  transformers    {transformers.__version__}")

    t0 = time.time()
    model, processor, loader_name = load_model_and_processor(CFG)
    load_s = time.time() - t0
    print(f"  loader          {loader_name}")
    print(f"  architecture    {type(model).__name__}")

    n_params = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"{OK} loaded in {load_s:.1f}s | {n_params / 1e9:.2f}B params | {vram:.2f} GB VRAM")

    section("4. one image through the model")
    image = make_probe_image()
    print(f"  probe image     {image.size[0]}x{image.size[1]} px")

    messages = build_messages(image, INSTRUCTION)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=CFG.model.enable_thinking,
    ).to(model.device)
    print(f"  input tokens    {inputs['input_ids'].shape[-1]}")

    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,  # smoke test only; config value is for real runs
            do_sample=CFG.model.do_sample,
        )
    gen_s = time.time() - t0

    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    print(
        f"{OK} generated {len(new_tokens)} tokens in {gen_s:.1f}s "
        f"({len(new_tokens) / gen_s:.1f} tok/s) | peak VRAM {peak:.2f} GB"
    )

    print(f"\n--- raw model output (zero-shot, thinking={CFG.model.enable_thinking}) ---")
    print(text[:1500] or "<empty>")
    print("--- end ---")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="skip steps 3-4")
    args = ap.parse_args()

    results = {"torch/CUDA": check_torch(), "bf16": check_bf16()}
    if not all(results.values()):
        print(f"\n{BAD} hardware checks failed; not attempting the model load.")
        return 1

    if not args.no_model:
        results["model"] = check_model()

    section("summary")
    for name, passed in results.items():
        print(f"  {OK if passed else BAD} {name}")
    failed = [k for k, v in results.items() if not v]
    print(f"\n{'all checks passed' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
