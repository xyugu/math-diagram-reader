"""Serve the prediction viewer.

    uv run --extra web python -m mathvlm.web.serve
    # open http://127.0.0.1:8000

Lists every run under ``<outputs_dir>/infer/`` (``mathvlm.infer``) and, where an eval has
been run, its per-example scores. Writes nothing: a fixed diagram is downloaded from the
page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mathvlm.config import get_config


def main(argv: list[str] | None = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        prog="python -m mathvlm.web.serve", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--host", default=cfg.web.host)
    ap.add_argument("--port", type=int, default=cfg.web.port)
    ap.add_argument("--outputs", type=Path, default=None, help="default: <outputs_dir>")
    args = ap.parse_args(argv)

    import uvicorn

    from mathvlm.web.app import create_app

    app = create_app(cfg, args.outputs)
    print(f"viewer at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
