#!/usr/bin/env python3
"""Regenerate prompts for the fixed 349-proposition Pander Score corpus."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from pathlib import Path

from pander_score.prompt_generation import generate_fixed_prompts
from pander_score.core.util import load_environment


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Generate fresh prompts for all 349 public v1 propositions using "
            "the two original elicitor models and prompt-generation protocol."
        )
    )
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "generated/v1",
        help="Output directory (default: generated/v1)",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    load_environment(ROOT)
    output = args.output.resolve()
    if output == (ROOT / "data/v1").resolve():
        raise SystemExit("Refusing to overwrite the frozen public benchmark directory")
    command = shlex.join(["uv", "run", "python", *sys.argv])
    print("This calls GPT-5.4-mini and Gemini 3 Flash and incurs API cost.", flush=True)
    print(f"Resume after interruption: {command}", flush=True)
    try:
        path, samples, retries = asyncio.run(
            generate_fixed_prompts(ROOT / "data/v1/propositions.csv", output)
        )
    except KeyboardInterrupt:
        print(f"Interrupted. Resume with: {command}", file=sys.stderr, flush=True)
        raise
    except Exception:
        print(f"Generation incomplete. Resume with: {command}", file=sys.stderr, flush=True)
        raise
    print(f"Complete: {samples} prompts; retries={retries}; output={path}")
    print(
        "Fresh generations are new corpora. The published benchmark remains "
        "data/v1/prompts.jsonl."
    )


if __name__ == "__main__":
    main()
