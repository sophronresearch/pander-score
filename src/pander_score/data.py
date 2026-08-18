"""Load the fixed public inputs and normalized result layout."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pander_score.core.util import read_jsonl


BENCHMARK_PROPOSITIONS = 349
BENCHMARK_PROMPTS = 11_172


def safe_model_filename(model_id: str) -> str:
    return model_id.replace("/", "-") + ".parquet"


def load_prompt_attributes(data_root: Path) -> pl.DataFrame:
    """Build the prompt-side table from the three frozen public JSONLs."""
    prompts = read_jsonl(data_root / "prompts.jsonl")
    propositions = pl.read_csv(data_root / "propositions.csv")
    if (
        propositions.height != BENCHMARK_PROPOSITIONS
        or propositions["id"].n_unique() != BENCHMARK_PROPOSITIONS
    ):
        raise ValueError(
            f"Expected {BENCHMARK_PROPOSITIONS} unique propositions, "
            f"found {propositions.height} rows / "
            f"{propositions['id'].n_unique()} IDs"
        )
    domains = dict(propositions.select(["id", "domain"]).iter_rows())

    missing_domains = sorted(
        {
            str(row["metadata"]["proposition_id"])
            for row in prompts
            if str(row["metadata"]["proposition_id"]) not in domains
        }
    )
    if missing_domains:
        raise ValueError(
            "Prompts reference propositions absent from propositions.csv: "
            + ", ".join(missing_domains)
        )
    rows = {
        row["id"]: {
            "sample_id": row["id"],
            "prompt_text": row["input"],
            "target": row["target"],
            **row["metadata"],
            "domain": domains[str(row["metadata"]["proposition_id"])],
        }
        for row in prompts
    }
    if len(rows) != BENCHMARK_PROMPTS:
        raise ValueError(f"Expected {BENCHMARK_PROMPTS} prompts, found {len(rows)}")

    for name in ("author_valence", "new_evidence_cat", "truth_matters"):
        records = read_jsonl(data_root / "prompt_attributes" / f"{name}.jsonl")
        if len(records) != BENCHMARK_PROMPTS:
            raise ValueError(
                f"Expected {BENCHMARK_PROMPTS} {name} rows, found {len(records)}"
            )
        for record in records:
            sample_id = record.pop("sample_id")
            if sample_id not in rows:
                raise ValueError(f"{name} references unknown sample {sample_id}")
            rows[sample_id].update(record)

    return pl.DataFrame(list(rows.values()), infer_schema_length=None)


def load_model_results(
    result_root: Path,
    model_id: str,
    *,
    data_root: Path | None = None,
) -> pl.DataFrame:
    """Join one target model's outputs to the frozen prompt-side attributes."""
    model_path = result_root / "per_model" / safe_model_filename(model_id)
    if not model_path.exists():
        raise FileNotFoundError(f"No results for {model_id}: {model_path}")
    dependent = pl.read_parquet(model_path)

    attrs_path = result_root / "prompt_attributes.parquet"
    if attrs_path.exists():
        attributes = pl.read_parquet(attrs_path)
    elif data_root is not None:
        attributes = load_prompt_attributes(data_root)
    else:
        raise FileNotFoundError(
            f"{attrs_path} does not exist; pass data_root for a locally generated run"
        )

    attributes_for_join = attributes
    if "proposition_id" in dependent.columns and "proposition_id" in attributes.columns:
        attributes_for_join = attributes.drop("proposition_id")
    joined = dependent.join(attributes_for_join, on="sample_id", how="left")
    if joined.height != dependent.height:
        raise ValueError("Prompt-attribute join changed the number of result rows")
    return joined
