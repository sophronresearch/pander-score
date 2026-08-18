#!/usr/bin/env python3
"""Grade a freshly generated prompt corpus with the three published judgments."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

from pander_score.core.util import load_environment
from pander_score.prompt_grading import grade_prompts


ROOT = Path(__file__).resolve().parent


def main() -> None:
    load_environment(ROOT)
    prompts = ROOT / "generated/v1/prompts.jsonl"
    output = ROOT / "generated/v1/prompt_attributes"
    if not prompts.exists():
        raise SystemExit(f"No generated prompts found: {prompts}\nRun generate_prompts.py first.")
    command = shlex.join(["uv", "run", "python", *sys.argv])
    print(
        "This calls GPT-5.4-mini, Gemini 3 Flash, Claude Sonnet 5, and GPT-5.6 Terra "
        "and incurs API cost.",
        flush=True,
    )
    print(f"Resume after interruption: {command}", flush=True)
    try:
        asyncio.run(grade_prompts(prompts, output))
    except KeyboardInterrupt:
        print(f"Interrupted. Resume with: {command}", file=sys.stderr, flush=True)
        raise
    except Exception:
        print(f"Grading incomplete. Resume with: {command}", file=sys.stderr, flush=True)
        raise
    print(f"Complete: prompt judgments -> {output}")


if __name__ == "__main__":
    main()
