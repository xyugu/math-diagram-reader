"""Generation speed benchmark for the base model.

Times prefill and decode separately on a rendered fixture page, after a warmup call
(the first call JIT-compiles any Triton kernels). Also saves the model's normal greedy
output, so runs with different kernels can be checked for identical results.

Run:  uv run python scripts/bench_generate.py --label fla --compare-to torch-fallback
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time

from mathvlm.config import apply_hf_env, get_config

CFG = get_config()
apply_hf_env(CFG)  # before transformers is imported


def kernel_status() -> dict[str, bool]:
    """Which optional fast-kernel packages are importable."""
    return {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="name for this run, e.g. torch-fallback, fla")
    ap.add_argument("--new-tokens", type=int, default=200, help="forced decode length")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--page", default="page_right_triangle", help="fixture page to use as input")
    ap.add_argument("--compare-to", default=None, help="label of an earlier run to compare with")
    args = ap.parse_args()

    import torch

    from mathvlm.fixtures import PAGES
    from mathvlm.modeling import load_model_and_processor
    from mathvlm.prompts import INSTRUCTION, build_messages
    from mathvlm.render import render_page
    from mathvlm.schema import Transcription

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    image = render_page(Transcription.model_validate(PAGES[args.page])).image
    model, processor, _ = load_model_and_processor(CFG)
    convs = [build_messages(image, INSTRUCTION) for _ in range(args.batch)]
    batch_kwargs = {"padding": True} if args.batch > 1 else {}
    inputs = processor.apply_chat_template(
        convs[0] if args.batch == 1 else convs,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=CFG.model.enable_thinking,
        processor_kwargs=batch_kwargs,
    ).to(model.device)
    n_in = inputs["input_ids"].shape[-1]
    n_new = args.new_tokens

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        _, first_call_s = timed(
            lambda: model.generate(**inputs, max_new_tokens=16, do_sample=False)
        )
        prefill = [timed(lambda: model(**inputs))[1] for _ in range(args.repeats)]
        totals = [
            timed(
                lambda: model.generate(
                    **inputs, min_new_tokens=n_new, max_new_tokens=n_new, do_sample=False
                )
            )[1]
            for _ in range(args.repeats)
        ]
        natural, natural_s = timed(
            lambda: model.generate(
                **inputs, max_new_tokens=CFG.model.max_new_tokens, do_sample=False
            )
        )

    prefill_s = statistics.median(prefill)
    total_s = statistics.median(totals)
    decode_s = total_s - prefill_s
    out_ids = natural[0][n_in:]
    result = {
        "label": args.label,
        "kernels": kernel_status(),
        "batch": args.batch,
        "input_tokens": n_in,
        "first_call_s": round(first_call_s, 2),
        "prefill_s": round(prefill_s, 3),
        "decode_tokens": n_new,
        "decode_s": round(decode_s, 2),
        "decode_tok_per_s": round(n_new * args.batch / decode_s, 1),
        "natural_tokens": len(out_ids),
        "natural_s": round(natural_s, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "output": processor.decode(out_ids, skip_special_tokens=True),
    }

    out_dir = CFG.paths.resolved("outputs_dir") / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.label}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(json.dumps({k: v for k, v in result.items() if k != "output"}, indent=2))
    print("--- greedy output ---\n" + result["output"] + "\n---")

    if args.compare_to:
        ref = json.loads((out_dir / f"{args.compare_to}.json").read_text())
        print(f"\nvs {args.compare_to}:")
        print(
            f"  prefill   {ref['prefill_s']:.3f}s -> {result['prefill_s']:.3f}s "
            f"({ref['prefill_s'] / result['prefill_s']:.1f}x)"
        )
        print(
            f"  decode    {ref['decode_tok_per_s']} -> {result['decode_tok_per_s']} tok/s "
            f"({result['decode_tok_per_s'] / ref['decode_tok_per_s']:.1f}x)"
        )
        print(
            f"  natural   {ref['natural_s']}s -> {result['natural_s']}s "
            f"({ref['natural_s'] / result['natural_s']:.1f}x)"
        )
        print(f"  identical greedy output: {ref['output'] == result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
