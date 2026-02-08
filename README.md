# Cleo Bench

Cleo Bench is a reproducible benchmark pipeline for hard symbolic integration problems sourced from Cleo's answers on Mathematics Stack Exchange.

## What it does

1. Fetches Cleo's Math.SE answer corpus from Stack Exchange API.
2. Extracts integral targets and reference expressions (Cleo + accepted answer if present).
3. Computes high-precision numeric checks (`mpmath`, default `dps=80`).
4. Produces benchmark JSONL with two prompt tracks:
   - `prompt_full_question_sanitized`
   - `prompt_integral_only`
5. Exports Inspect-compatible datasets and task definitions.
6. Scores model outputs with deterministic numeric comparison first, then optional LLM judge fallback for unresolved cases.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## CLI

### 1) Fetch raw snapshot

```bash
cleo-bench fetch \
  --account-id 3364210 \
  --site math.stackexchange \
  --snapshot-date 2026-02-07
```

Artifacts:
- `data/raw/<snapshot-date>/bundle.json`
- `data/raw/<snapshot-date>/index.json`

### 2) Build processed dataset

```bash
cleo-bench build --snapshot-date 2026-02-07
```

Artifact:
- `data/processed/cleo_bench.jsonl`
- `reports/build_summary.json`

### 3) Validate numerics

```bash
cleo-bench validate \
  --input-path data/processed/cleo_bench.jsonl \
  --tolerance 1e-6 \
  --dps 80
```

Artifacts:
- `data/processed/cleo_bench.jsonl` (updated)
- `data/processed/cleo_bench_scorable.jsonl`
- `data/manual_queue/unresolved.jsonl`
- `reports/build_summary.json`

Notes:
- If direct quadrature fails for an item, validation can fall back to an explicit numeric approximation found in the original question text (e.g., `\approx ...`).

Optional parser-repair fallback:

```bash
cleo-bench validate --parser-repair-model <inspect_model_name>
```

### 3.5) Interactive manual review

```bash
cleo-bench review
```

By default this reads:
- `data/manual_queue/unresolved.jsonl`

And writes:
- `data/manual_queue/overrides.jsonl`

Use inside the reviewer:
- `i` edit `integral_latex`
- `c` edit `cleo_reference_latex`
- `a` edit `accepted_reference_latex`
- `x` mark non-integral (`is_integral_candidate=false`)
- `r` reset current item override
- `n`/`p` next/previous item
- `g <idx|item_id>` jump
- `s` save now
- `q` save and quit

Then apply overrides:

```bash
cleo-bench validate --overrides-path data/manual_queue/overrides.jsonl
```

### 4) Export Inspect datasets

```bash
cleo-bench export-inspect --input-path data/processed/cleo_bench.jsonl
```

Artifacts:
- `data/inspect/cleo_bench_full_question.jsonl`
- `data/inspect/cleo_bench_integral_only.jsonl`
- `data/inspect/task_config.json`

### 5) Run Inspect eval

```bash
inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model <model_name>

inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_integral_only \
  --model <model_name>
```

### 5.5) Run 5 random unseen Cleo items

```bash
python scripts/pick_random_untried.py --count 5
```

This command:
- reads prior tried sample IDs from `logs/*.eval`
- samples unseen items from `data/inspect/cleo_bench_full_question.jsonl`
- writes a subset JSONL under `data/inspect/subsets/`
- prints a ready-to-run `inspect eval ...` command

If a model run is truncated (`native_finish_reason: "length"`), increase completion budget, e.g.

```bash
python scripts/pick_random_untried.py --count 5 --max-tokens 65535
```

### 6) Summarize Inspect logs

```bash
cleo-bench summarize-eval --log-dir logs
```

Artifact:
- `reports/eval_summary.json`

## Scoring policy

Deterministic-first scoring:
- Parse candidate JSON output (`final_expression_latex`).
- Numerically evaluate candidate and reference values.
- Pass if `abs_err <= 1e-6` or `rel_err <= 1e-6`.

Judge fallback (optional, unresolved-only):
- Rubric criteria (0-5 each):
  - equivalence
  - constants_and_branches
  - task_completeness
  - expression_validity
- Pass threshold: equivalence >= 4 and total >= 14/20.

## Dataset schema

`data/processed/cleo_bench.jsonl` includes the full `CleoBenchItem` schema (33 fields):
- IDs/provenance/license fields
- raw title/body HTML fields
- two prompt tracks
- extracted integral/reference expressions
- numeric values and pass/fail deltas
- inclusion/status flags and review reason

## Tests

```bash
. .venv/bin/activate
python -m unittest discover -s tests -v
```

## License and attribution

Source content originates from Stack Exchange posts and retains original attribution and license metadata (`content_license` fields, typically CC BY-SA). Use and redistribution must preserve attribution requirements.
