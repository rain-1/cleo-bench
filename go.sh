OPENROUTER_API_KEY="${OPENROUTER_API_KEY_MATH}" \
inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model openrouter/deepseek/deepseek-v3.2-speciale \
  --model-base-url https://openrouter.ai/api/v1 \
  -M reasoning_enabled=true \
  --reasoning-effort medium \
  --max-tokens 163100 \
  -T judge_model=openrouter/openai/gpt-oss-120b:free \
  -T judge_api_key_env=OPENROUTER_API_KEY_JUDGE \
  -T judge_max_tokens=8192 \
  -T judge_use_sagemath_mcp=true \
  -T judge_sagemath_mcp_command=uv \
  -T judge_sagemath_mcp_args=run,sagemath-mcp \
  -T dataset_file=data/inspect/subsets/run5.jsonl \
  --sample-id q905653_a905723 \


OPENROUTER_API_KEY="${OPENROUTER_API_KEY_MATH}" inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question   --model openrouter/openai/gpt-oss-120b:free   --model-base-url https://openrouter.ai/api/v1   -M reasoning_enabled=true   --reasoning-effort medium   --max-tokens 130000   -T judge_model=openrouter/openai/gpt-oss-120b:free   -T judge_api_key_env=OPENROUTER_API_KEY_JUDGE   -T judge_max_tokens=8192   -T judge_use_sagemath_mcp=true   -T judge_sagemath_mcp_command=uv   -T judge_sagemath_mcp_args=run,sagemath-mcp   -T dataset_file=data/inspect/subsets/run5.jsonl   --sample-id q905653_a905723


set -a && source .env && set +a

OPENROUTER_API_KEY="${OPENROUTER_API_KEY_MATH}" .venv/bin/inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model openrouter/openai/gpt-oss-120b:free \
  --model-base-url https://openrouter.ai/api/v1 \
  -M reasoning_enabled=true \
  --reasoning-effort medium \
  --max-tokens 32768 \
  -T dataset_file=data/inspect/subsets/run5.jsonl \
  -T use_judge_when_unresolved=false \
  -T solver_use_sagemath_mcp=true \
  -T solver_require_sagemath_tool_call=true \
  -T solver_sagemath_mcp_command=uv \
  -T solver_sagemath_mcp_args=run,sagemath-mcp \
  -T solver_sagemath_mcp_allow_imports=true \
  --sample-id q710175_a711804




set -a && source .env && set +a

OPENROUTER_API_KEY="${OPENROUTER_API_KEY_MATH}" .venv/bin/inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question \
  --model openrouter/openrouter/pony-alpha \
  --model-base-url https://openrouter.ai/api/v1 \
  --reasoning-effort medium \
  --max-tokens 32768 \
  -T dataset_file=data/inspect/subsets/run5.jsonl \
  -T use_judge_when_unresolved=false \
  -T solver_use_sagemath_mcp=true \
  -T solver_require_sagemath_tool_call=true \
  -T solver_sagemath_mcp_command=uv \
  -T solver_sagemath_mcp_args=run,sagemath-mcp \
  -T solver_sagemath_mcp_allow_imports=true \
  -T solver_sagemath_mcp_tools=evaluate_sage,calculate_expression \
  --sample-id q710175_a711804


