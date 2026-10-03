from __future__ import annotations

from collections.abc import Mapping, Sequence


def compute_task_sampling_probabilities(
    tasks: Sequence[str],
    success_rates: Mapping[str, float],
    *,
    min_task_sample_probability: float,
    prioritize_low_success_tasks: bool,
) -> dict[str, float]:
    """Compute uniform-ablation or low-success-priority probabilities."""
    sampling_tasks = list(tasks)
    if not sampling_tasks:
        raise ValueError("At least one task is required for task sampling")

    if not prioritize_low_success_tasks:
        uniform_probability = 1.0 / len(sampling_tasks)
        return {task: uniform_probability for task in sampling_tasks}

    mastered_tasks = [
        task for task in sampling_tasks if success_rates[task] >= 1.0
    ]
    non_mastered_tasks = [
        task for task in sampling_tasks if task not in mastered_tasks
    ]

    if not non_mastered_tasks:
        uniform_probability = 1.0 / len(sampling_tasks)
        return {task: uniform_probability for task in sampling_tasks}

    mastered_probability_total = (
        len(mastered_tasks) * min_task_sample_probability
    )
    remaining_probability = 1.0 - mastered_probability_total
    difficulty_weights = {
        task: 1.0 - success_rates[task] for task in non_mastered_tasks
    }
    difficulty_total = sum(difficulty_weights.values())
    if difficulty_total <= 0.0:
        raise RuntimeError(
            "Non-mastered task difficulty weights must sum to a positive value"
        )

    probabilities = {
        task: min_task_sample_probability for task in mastered_tasks
    }
    probabilities.update({
        task: remaining_probability * difficulty_weights[task] / difficulty_total
        for task in non_mastered_tasks
    })
    return {task: probabilities[task] for task in sampling_tasks}
