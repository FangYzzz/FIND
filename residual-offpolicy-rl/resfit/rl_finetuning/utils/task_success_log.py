"""Load per-task initial-evaluation outcomes from a task-generator log."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence


_INITIAL_EVAL_MARKER = "evaluation after 0 steps"
_TASK_HEADER_RE = re.compile(r"^-+\s*task\s+(\d+):\s*(.*?)\s*-+$", re.IGNORECASE)
_PROGRESS_RE = re.compile(
    r"^Evaluating\s+(\d+)\s+episodes:\s*([✓✗.]*)$",
    re.IGNORECASE,
)


def _log_message(line: str) -> str:
    """Strip the timestamp/level prefix used by the task logger."""
    parts = line.rstrip("\n").split(" | ", maxsplit=2)
    return parts[-1].strip()


def load_initial_eval_outcomes_from_log(
    log_path: str | Path,
    candidate_tasks: Sequence[str],
    window_size: int,
) -> dict[str, list[int]]:
    """Parse the ordered, completed step-0 evaluation window for every task.

    The parser intentionally uses the final ``Evaluating N episodes: ✓✗...``
    line in each task section. It therefore preserves outcome order, which is
    required when subsequent training episodes evict the oldest window slots.
    """
    path = Path(log_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initial evaluation log does not exist: {path}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    tasks = list(candidate_tasks)
    if not tasks:
        raise ValueError("candidate_tasks must not be empty")

    outcomes_by_task: dict[str, list[int]] = {}
    current_task: str | None = None
    found_initial_eval = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            message = _log_message(raw_line)
            if not found_initial_eval:
                if _INITIAL_EVAL_MARKER in message.lower():
                    found_initial_eval = True
                continue

            header_match = _TASK_HEADER_RE.fullmatch(message)
            if header_match is not None:
                task_number = int(header_match.group(1))
                task_name = header_match.group(2).strip()
                expected_number = len(outcomes_by_task) + 1
                if task_number != expected_number:
                    raise ValueError(
                        f"Unexpected task number {task_number} at {path}:{line_number}; "
                        f"expected task {expected_number}"
                    )
                if task_number > len(tasks):
                    raise ValueError(
                        f"Log contains unexpected task {task_number} at "
                        f"{path}:{line_number}: {task_name!r}"
                    )
                expected_name = tasks[task_number - 1]
                if task_name != expected_name:
                    raise ValueError(
                        f"Task mismatch at {path}:{line_number}; expected "
                        f"{expected_name!r}, got {task_name!r}"
                    )
                current_task = task_name
                continue

            progress_match = _PROGRESS_RE.fullmatch(message)
            if progress_match is None or current_task is None:
                continue

            declared_size = int(progress_match.group(1))
            progress = progress_match.group(2)
            if declared_size != window_size or "." in progress:
                continue
            if len(progress) != window_size:
                raise ValueError(
                    f"Completed evaluation for {current_task!r} at "
                    f"{path}:{line_number} has {len(progress)} outcomes; "
                    f"expected {window_size}"
                )
            if current_task in outcomes_by_task:
                raise ValueError(
                    f"Duplicate completed initial evaluation for "
                    f"{current_task!r} at {path}:{line_number}"
                )

            outcomes_by_task[current_task] = [
                1 if symbol == "✓" else 0 for symbol in progress
            ]
            current_task = None
            if len(outcomes_by_task) == len(tasks):
                break

    if not found_initial_eval:
        raise ValueError(f"No step-0 evaluation marker found in {path}")

    missing_tasks = [task for task in tasks if task not in outcomes_by_task]
    if missing_tasks:
        raise ValueError(
            f"Initial evaluation in {path} is incomplete; missing a completed "
            f"{window_size}-episode window for: {missing_tasks}"
        )

    return {task: outcomes_by_task[task] for task in tasks}


def apply_initial_eval_outcome_overrides(
    initial_eval_outcomes: Mapping[str, Sequence[int]] | None,
    overrides: Mapping[str, Sequence[int]] | None,
    candidate_tasks: Sequence[str],
    window_size: int,
) -> dict[str, list[int]]:
    """Build complete ordered windows, applying configured values over a log.

    ``initial_eval_outcomes`` is normally parsed from the immutable step-0 log.
    ``overrides`` may replace any subset of those per-task windows. If there is
    no log-derived base, the overrides must provide every candidate task.
    """
    tasks = list(candidate_tasks)
    if not tasks:
        raise ValueError("candidate_tasks must not be empty")
    if len(set(tasks)) != len(tasks):
        raise ValueError("candidate_tasks must not contain duplicates")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    windows = {
        task: [int(value) for value in values]
        for task, values in (initial_eval_outcomes or {}).items()
    }
    if overrides is not None:
        if not overrides:
            raise ValueError(
                "initial_task_success_window_overrides must not be empty; "
                "use None to disable it"
            )
        unknown_tasks = set(overrides) - set(tasks)
        if unknown_tasks:
            raise ValueError(
                "Initial success-window overrides contain unknown tasks: "
                f"{sorted(unknown_tasks)}"
            )
        for task, raw_values in overrides.items():
            try:
                values = list(raw_values)
            except TypeError as exc:
                raise ValueError(
                    f"Initial success-window override for {task!r} must be a "
                    "sequence of 0/1 outcomes"
                ) from exc
            if len(values) != window_size:
                raise ValueError(
                    f"Initial success-window override for {task!r} must contain "
                    f"exactly {window_size} outcomes, got {len(values)}"
                )
            if any(value not in (0, 1) for value in values):
                raise ValueError(
                    f"Initial success-window override for {task!r} must contain "
                    "only 0 or 1"
                )
            windows[task] = [int(value) for value in values]

    missing_tasks = [task for task in tasks if task not in windows]
    if missing_tasks:
        raise ValueError(
            "Initial success windows are incomplete. Configure "
            "initial_task_success_log or provide overrides for: "
            f"{missing_tasks}"
        )

    return {task: windows[task] for task in tasks}
