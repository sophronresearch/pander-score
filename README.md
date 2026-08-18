# Pander Score

The Pander Score measures how strongly a model's expressed belief moves with
the attitude implicit in a user's prompt. This repository contains the complete
v1 benchmark: 349 propositions, 11,172 published prompts, prompt generation,
frozen prompt judgments, the response-scoring pipeline, and the estimator.

Saved outputs for all 18 paper models are in the companion
[Hugging Face dataset](https://huggingface.co/datasets/sophronresearch/pander-score).

## Reproduce a published score

This makes no model calls. It downloads the saved outputs and recomputes the
score locally. The benchmark download is about 431 MiB.

```bash
git clone https://github.com/sophronresearch/pander-score.git
cd pander-score
uv sync
uv run python score_results.py --model google/gemini-3.7-flash
```

Expected rounded result:

```text
google/gemini-3.7-flash  +15.7 [+12.9, +18.6]  +65.0 [+59.6, +70.5]  11172
```

These are the unrounded values underlying the integer labels in Figure 1.

Omit `--model` for all 18 published models. To use a local dataset download,
pass its `v1` or `v1/benchmark` directory with `--results`.

## Run the benchmark on a model

Every run uses GPT-5.4-mini and Gemini 3 Flash as credence judges, so OpenAI and
Google credentials are required in addition to the target provider's key.

```bash
cp .env.example .env
uv run python run_benchmark.py \
  --target-model openai/gpt-5.4-2026-03-05
```

This runs all 11,172 prompts and makes two judge calls per response, so it can
incur substantial API cost. For scale, two recent full target runs cost $173
and $214 in target-model charges before credence-judge costs; provider prices
vary. There is deliberately no partial-set,
alternate-proposition, dry-run, or provider-batch mode.

Runs checkpoint automatically under `results/v1/`. Repeat the same command to
resume; `--max-connections` may be changed between attempts. The 18 published
model IDs automatically receive their original `reasoning_effort` and
`temperature`. Other Inspect-compatible model IDs use provider defaults and
accept explicit overrides:

```bash
uv run python run_benchmark.py \
  --target-model provider/model-id \
  --reasoning-effort medium \
  --temperature 1
```

Overriding a published model's recorded condition marks the run as
non-comparable. For a new reasoning model, `medium` is the closest general
starting point to the published roster when the provider supports it.

## Generate fresh prompts

The original prompt-generation code and templates are included. With both the
OpenAI and Google keys configured, run:

```bash
uv run python generate_prompts.py
```

This requests 16 prompts from each of the two published elicitors for every
benchmark proposition and incurs API cost. Generation is stochastic,
checkpointed, and writes to `generated/v1/`; it never changes the frozen
prompts under `data/v1/`.

To apply the three published prompt judgments to that fresh corpus, also set
`ANTHROPIC_API_KEY` and run:

```bash
uv run python grade_prompts.py
```

Both scripts write incremental checkpoints. They are safe to interrupt and the
same command resumes only unfinished work while reporting progress, retries,
and failures.

Fresh prompts support protocol inspection and prompt-judge experiments. The
public target runner intentionally evaluates only the frozen 11,172 prompts;
fresh corpora are not leaderboard-comparable and cannot be passed to it.

## Data and method

The committed `data/v1/` directory contains the fixed propositions, prompts,
and three current prompt-judgment families: author valence, categorical new
evidence, and Truth Matters. The Hugging Face dataset adds saved responses for
the 18 published models, four validation suites, and two
deidentified human-judgment studies.

At a high level, the score estimates how response credence changes with prompt
author valence within each proposition. It excludes prompts that introduce new
evidence or where truth is not judged to matter, reports conversational and
instructional results separately, and bootstraps propositions for uncertainty.

See [docs/data.md](docs/data.md) for schemas, result-table semantics, and the
full estimator contract. The core implementation is under
`src/pander_score/core/`.

## Reproducibility

The frozen prompts and saved outputs reproduce the paper results. Rerunning a
target today can differ if a provider has changed the model served behind an
unversioned ID. The runner is therefore for evaluating newly served targets;
use `score_results.py` for exact paper-number reproduction.

## License and citation

Code is released under the [MIT License](LICENSE). Benchmark inputs and the
companion dataset are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Please cite *The Pander Score: A Continuous Measure of Sycophancy as Epistemic
Deference*. Machine-readable metadata is in [CITATION.cff](CITATION.cff).
