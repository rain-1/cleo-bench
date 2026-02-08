"""Inspect task definitions for Cleo Bench."""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

from cleo_bench.scoring import score_with_optional_judge

SYSTEM_PROMPT = (
    "You are solving hard symbolic integration tasks. "
    "Return JSON only with key `final_expression_latex` and optional key `notes`."
)


def _resolve_dataset_path(dataset_file: str) -> str:
    path = Path(dataset_file)
    if path.is_absolute():
        return path.as_posix()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).as_posix()


@scorer(metrics=[mean(), stderr()])
def cleo_bench_scorer(
    use_judge_when_unresolved: bool = True,
    judge_model: str | None = None,
    dps: int = 80,
    tolerance: float = 1e-6,
):
    async def score(state: TaskState, target: Target) -> Score:
        _ = target  # Not used; metadata carries references.
        metadata = state.metadata or {}
        result = await score_with_optional_judge(
            output_text=state.output.completion,
            metadata=metadata,
            use_judge_when_unresolved=use_judge_when_unresolved,
            judge_model=judge_model,
            dps=dps,
            tolerance=tolerance,
        )
        value = result.get("final_score")
        if value is None:
            value = 0.0

        return Score(
            value=float(value),
            answer=state.output.completion,
            explanation=result.get("explanation"),
            metadata=result,
        )

    return score


@task(name="cleo_bench_full_question")
def cleo_bench_full_question(
    dataset_file: str = "data/inspect/cleo_bench_full_question.jsonl",
    use_judge_when_unresolved: bool = True,
    judge_model: str | None = None,
    dps: int = 80,
    tolerance: float = 1e-6,
) -> Task:
    dataset_path = _resolve_dataset_path(dataset_file)
    return Task(
        dataset=json_dataset(dataset_path),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=cleo_bench_scorer(
            use_judge_when_unresolved=use_judge_when_unresolved,
            judge_model=judge_model,
            dps=dps,
            tolerance=tolerance,
        ),
    )


@task(name="cleo_bench_integral_only")
def cleo_bench_integral_only(
    dataset_file: str = "data/inspect/cleo_bench_integral_only.jsonl",
    use_judge_when_unresolved: bool = True,
    judge_model: str | None = None,
    dps: int = 80,
    tolerance: float = 1e-6,
) -> Task:
    dataset_path = _resolve_dataset_path(dataset_file)
    return Task(
        dataset=json_dataset(dataset_path),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=cleo_bench_scorer(
            use_judge_when_unresolved=use_judge_when_unresolved,
            judge_model=judge_model,
            dps=dps,
            tolerance=tolerance,
        ),
    )
