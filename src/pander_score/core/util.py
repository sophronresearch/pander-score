"""Shared helpers for judge prompting and JSONL inputs."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from dotenv import load_dotenv


TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_environment(root: Path) -> Path | None:
    """Load the nearest .env, preferring the repository over its parents."""
    for directory in (root, *root.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@functools.lru_cache(maxsize=None)
def load_template(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.txt").read_text(encoding="utf-8")


def format_interaction_log(messages: list[dict[str, str]]) -> str:
    parts = ["<interaction_log>"]
    for message in messages:
        if message["role"] == "user":
            parts.append(f"<user>{message['content']}</user>")
        elif message["role"] == "assistant":
            parts.append(f"<model>{message['content']}</model>")
    parts.append("</interaction_log>")
    return "\n\n".join(parts)
