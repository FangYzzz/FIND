from __future__ import annotations

import unittest

from resfit.rl_finetuning.utils.task_sampling import (
    compute_task_sampling_probabilities,
)


class TaskSamplingProbabilitiesTest(unittest.TestCase):
    tasks = ["easy", "medium", "hard"]
    success_rates = {"easy": 0.9, "medium": 0.5, "hard": 0.1}

    def test_priority_enabled_favors_lower_success_tasks(self):
        probabilities = compute_task_sampling_probabilities(
            self.tasks,
            self.success_rates,
            min_task_sample_probability=0.05,
            prioritize_low_success_tasks=True,
        )

        self.assertGreater(probabilities["hard"], probabilities["medium"])
        self.assertGreater(probabilities["medium"], probabilities["easy"])
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_priority_disabled_is_uniform_over_feasible_tasks(self):
        probabilities = compute_task_sampling_probabilities(
            ["easy", "hard"],
            self.success_rates,
            min_task_sample_probability=0.05,
            prioritize_low_success_tasks=False,
        )

        self.assertEqual(probabilities, {"easy": 0.5, "hard": 0.5})


if __name__ == "__main__":
    unittest.main()
