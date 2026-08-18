#!/usr/bin/env python3
"""Run the complete Pander Score v1 benchmark for one target model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shlex
import sys
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig, get_model

from pander_score.core.config import (
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_TOKENS,
    resolve_model_config,
)
from pander_score.core.scorers import CREDENCE_JUDGES
from pander_score.core.task import pander_benchmark
from pander_score.core.util import load_environment, read_jsonl
from pander_score.data import BENCHMARK_PROMPTS, load_model_results, safe_model_filename
from pander_score.export import completed_sample_ids, export_model_results
from pander_score.core.metrics import compute_prompt_type_scores


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/v1"
RESULT_ROOT = ROOT / "results/v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run one model on all 11,172 frozen Pander Score v1 prompts, then "
            "score the results. Re-running the same command resumes automatically."
        )
    )
    result.add_argument("--target-model", required=True, help="Any Inspect-compatible model ID")
    result.add_argument("--reasoning-effort", help="Override reasoning effort")
    result.add_argument("--temperature", type=float, help="Override sampling temperature")
    result.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    result.add_argument("--max-connections", type=int, default=DEFAULT_MAX_CONNECTIONS)
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _recover_logs(log_dir: Path) -> None:
    from inspect_ai.log import recover_eval_log, recoverable_eval_logs

    for item in recoverable_eval_logs(str(log_dir)):
        print(
            f"Recovering {item.completed_samples} completed samples from "
            f"{item.log.name}...",
            flush=True,
        )
        try:
            recover_eval_log(item.log.name, overwrite=True)
        except Exception as error:
            print(f"Could not recover {item.log.name}: {error}", flush=True)


def _run_manifest(model_id: str, resolved: dict) -> dict:
    return {
        "benchmark_version": "v1",
        "target_model": model_id,
        "target_config": resolved,
        "credence_judges": {
            judge_id: {
                "reasoning_effort": config.reasoning_effort,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            for judge_id, config in CREDENCE_JUDGES.items()
        },
        "prompts_sha256": hashlib.sha256((DATA_ROOT / "prompts.jsonl").read_bytes()).hexdigest(),
        "prompt_count": BENCHMARK_PROMPTS,
    }


def _scientific_manifest(manifest: dict) -> dict:
    """Return the condition that must remain fixed when resuming a run.

    Concurrency affects throughput, not model outputs, so users may safely
    change it when resuming. Output-shaping settings such as max_tokens remain
    part of the comparison.
    """
    target_config = {
        key: value
        for key, value in manifest["target_config"].items()
        if key != "max_connections"
    }
    return {**manifest, "target_config": target_config}


def main() -> None:
    args = parser().parse_args()
    if args.max_tokens < 1 or args.max_connections < 1:
        raise SystemExit("--max-tokens and --max-connections must be positive")
    load_environment(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    resolved = resolve_model_config(
        args.target_model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
    )
    resolved_dict = resolved.to_dict()
    safe_name = safe_model_filename(args.target_model).removesuffix(".parquet")
    log_dir = RESULT_ROOT / "logs" / safe_name
    output_path = RESULT_ROOT / "per_model" / safe_model_filename(args.target_model)
    config_path = RESULT_ROOT / "run_configs" / f"{safe_name}.json"
    manifest = _run_manifest(args.target_model, resolved_dict)
    if config_path.exists() and _scientific_manifest(
        json.loads(config_path.read_text(encoding="utf-8"))
    ) != _scientific_manifest(manifest):
        raise SystemExit(
            f"Existing run configuration differs: {config_path}\n"
            "Use a fresh checkout/results directory rather than mixing experimental conditions."
        )
    _atomic_json(config_path, manifest)

    log_dir.mkdir(parents=True, exist_ok=True)
    _recover_logs(log_dir)
    completed = completed_sample_ids(log_dir, args.target_model, output_path)
    prompts = read_jsonl(DATA_ROOT / "prompts.jsonl")
    if len(prompts) != BENCHMARK_PROMPTS or len({row["id"] for row in prompts}) != BENCHMARK_PROMPTS:
        raise SystemExit(f"The fixed benchmark must contain exactly {BENCHMARK_PROMPTS} unique prompts")
    remaining = [row for row in prompts if row["id"] not in completed]

    print("Pander Score v1", flush=True)
    print(f"  Target: {args.target_model}", flush=True)
    print(f"  Resolved config: {json.dumps(resolved_dict, sort_keys=True)}", flush=True)
    print(f"  Completed: {len(completed)}/{BENCHMARK_PROMPTS}", flush=True)
    print(f"  Remaining: {len(remaining)}/{BENCHMARK_PROMPTS}", flush=True)

    if remaining:
        digest = hashlib.sha256(
            json.dumps(sorted(row["id"] for row in remaining)).encode()
        ).hexdigest()[:12]
        staged = RESULT_ROOT / "cache" / f"prompts.{safe_name}.{digest}.jsonl"
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.exists():
            temporary = staged.with_suffix(staged.suffix + ".tmp")
            temporary.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in remaining),
                encoding="utf-8",
            )
            temporary.replace(staged)

        target_config = GenerateConfig(
            reasoning_effort=resolved.reasoning_effort,
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
            max_connections=resolved.max_connections,
            max_retries=1,
            attempt_timeout=600,
            timeout=900,
        )
        command = shlex.join(["uv", "run", "python", *sys.argv])
        print(f"  Resume after interruption: {command}", flush=True)
        try:
            success, _logs = eval_set(
                tasks=[pander_benchmark(str(staged.resolve()))],
                model=[get_model(args.target_model, config=target_config)],
                log_dir=str(log_dir),
                log_dir_allow_dirty=True,
                retry_attempts=0,
                retry_immediate=False,
                max_samples=resolved.max_connections,
                retry_on_error=1,
            )
        except KeyboardInterrupt:
            print(f"Interrupted. Resume with: {command}", file=sys.stderr, flush=True)
            raise
        if not success:
            print("Some samples failed. Exporting completed work; re-run this command to retry.")

    _recover_logs(log_dir)
    now_completed = completed_sample_ids(log_dir, args.target_model, output_path)
    if not now_completed:
        raise SystemExit(
            "No scored samples completed. The run was interrupted or all requests failed; "
            f"resume with: {shlex.join(['uv', 'run', 'python', *sys.argv])}"
        )
    exported = export_model_results(log_dir, RESULT_ROOT, args.target_model, resolved_dict)
    rows = load_model_results(RESULT_ROOT, args.target_model, data_root=DATA_ROOT)
    unique_samples = rows["sample_id"].n_unique()
    print(f"  Exported: {unique_samples}/{BENCHMARK_PROMPTS} unique samples -> {exported}")
    if unique_samples != BENCHMARK_PROMPTS:
        raise SystemExit("Run the same command again to finish missing samples.")

    scores = compute_prompt_type_scores(rows)
    print(
        f"  Conversational: {100 * scores.conversational.score:+.1f} "
        f"[{100 * scores.conversational.ci_low:+.1f}, {100 * scores.conversational.ci_high:+.1f}]"
    )
    print(
        f"  Instructional:  {100 * scores.instructional.score:+.1f} "
        f"[{100 * scores.instructional.ci_low:+.1f}, {100 * scores.instructional.ci_high:+.1f}]"
    )


if __name__ == "__main__":
    main()
