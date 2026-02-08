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

Judge reliability knobs (useful when judge outputs are truncated):

```bash
inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model <model_name> \
  -T judge_model=<judge_model_name> \
  -T judge_api_key_env=OPENROUTER_API_KEY_JUDGE \
  -T judge_max_tokens=8192
```

Optional: SageMath MCP support (judge and/or solver)

1. Install optional MCP dependency and SageMath MCP server:

```bash
pip install -e .[mcp]
pip install "git+https://github.com/XBP-Europe/sagemath-mcp.git"
# Temporary compatibility pin (until upstream FastMCP fix lands):
pip install "fastmcp==2.13.3"
```

SageMath itself is not enough; you also need the MCP server package (`sagemath-mcp`) and the Python MCP client dependency (`.[mcp]`).

If `sagemath-mcp` is on your `PATH`, use `judge_sagemath_mcp_command=sagemath-mcp` (or `solver_sagemath_mcp_command=sagemath-mcp`) and omit args.
If you prefer running from a source checkout, use `*_sagemath_mcp_command=uv` and `*_sagemath_mcp_args=run,sagemath-mcp` (optionally set `*_sagemath_mcp_cwd`).
By default, Cleo Bench sets `SAGEMATH_MCP_EVAL_TIMEOUT=300` (5 minutes) for solver/judge MCP servers.

2. Run eval with SageMath tools enabled for the judge:

```bash
inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model <model_name> \
  -T judge_model=<judge_model_name> \
  -T judge_api_key_env=OPENROUTER_API_KEY_JUDGE \
  -T judge_use_sagemath_mcp=true \
  -T judge_sagemath_mcp_command=sagemath-mcp \
  -T judge_sagemath_mcp_eval_timeout_seconds=300
```

3. Run eval with SageMath tools enabled for the solver (agent/tool-use variant):

```bash
inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model <model_name> \
  -T solver_use_sagemath_mcp=true \
  -T solver_sagemath_mcp_command=sagemath-mcp \
  -T solver_sagemath_mcp_eval_timeout_seconds=300 \
  -T solver_sagemath_tool_choice=auto
```

4. Enable import statements in Sage executions (optional):

```bash
inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model <model_name> \
  -T solver_use_sagemath_mcp=true \
  -T solver_sagemath_mcp_command=sagemath-mcp \
  -T solver_sagemath_mcp_allow_imports=true \
  -T judge_use_sagemath_mcp=true \
  -T judge_sagemath_mcp_command=sagemath-mcp \
  -T judge_sagemath_mcp_allow_imports=true
```

Optional allowlist for imports:

```bash
-T solver_sagemath_mcp_allowed_imports=sympy,numpy \
-T judge_sagemath_mcp_allowed_imports=sympy,numpy
```

This maps to the Sage MCP env vars:
- `SAGEMATH_MCP_SECURITY_ALLOW_IMPORTS=1`
- `SAGEMATH_MCP_SECURITY_ALLOWED_IMPORTS=<comma-separated list>`
- `SAGEMATH_MCP_EVAL_TIMEOUT=300` (or your override)

5. Smoke-test MCP tool calls before a long eval run:

```bash
python scripts/verify_sagemath_mcp_solver.py \
  --model openrouter/deepseek/deepseek-v3.2 \
  --base-url https://openrouter.ai/api/v1 \
  --eval-timeout-seconds 300 \
  --allow-imports \
  --allowed-imports sympy,numpy
```

Important:
- Judge-side SageMath MCP runs only in fallback (`deterministic_status=unresolved`) when `judge_use_sagemath_mcp=true`.
- Judge fallback requires an extracted `final_expression_latex`; if the solver output does not include it, judge/MCP will not run.
- Solver-side MCP is controlled independently via `solver_use_sagemath_mcp=true`.
- Inspect score metadata now records `judge_tool_calls`, `judge_stop_reason`, `judge_error`, and `solver_tool_calls` so you can confirm tool usage in each sample.
- Solver retries one extra tool step when Sage errors look like syntax/symbol issues (e.g. `^` vs `**`, missing `*`, undefined symbols), then falls back to an explicit unresolved JSON payload if no visible answer text is produced.
- Recommended solver tool subset is `evaluate_sage,calculate_expression` (this is now the default in `scripts/pick_random_untried.py`).

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
python scripts/pick_random_untried.py --count 5 --max-tokens 65535 --judge-max-tokens 8192
```

To emit a run command with solver-side SageMath MCP enabled:

```bash
python scripts/pick_random_untried.py --count 5 --solver-use-sagemath-mcp --judge-use-sagemath-mcp
```

Set per-tool Sage timeout in the generated run command:

```bash
python scripts/pick_random_untried.py --count 5 --solver-use-sagemath-mcp --judge-use-sagemath-mcp --sagemath-eval-timeout-seconds 300
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
