from __future__ import annotations

import unittest

from resfit.rl_finetuning.utils.task_success_log import (
    apply_initial_eval_outcome_overrides,
)


class ApplyInitialEvalOutcomeOverridesTest(unittest.TestCase):
    def test_partial_overrides_replace_log_window_and_preserve_task_order(self):
        tasks = ["task a", "task b"]
        base = {"task a": [0, 0, 0], "task b": [1, 1, 1]}

        windows = apply_initial_eval_outcome_overrides(
            base,
            {"task b": [1, 0, 1]},
            candidate_tasks=tasks,
            window_size=3,
        )

        self.assertEqual(list(windows), tasks)
        self.assertEqual(windows["task a"], [0, 0, 0])
        self.assertEqual(windows["task b"], [1, 0, 1])

    def test_overrides_without_log_must_cover_every_task(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            apply_initial_eval_outcome_overrides(
                None,
                {"task a": [1, 0, 1]},
                candidate_tasks=["task a", "task b"],
                window_size=3,
            )

    def test_override_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            apply_initial_eval_outcome_overrides(
                {"task a": [0, 0, 0]},
                {"task a": [1, 0]},
                candidate_tasks=["task a"],
                window_size=3,
            )

    def test_override_rejects_non_binary_value(self):
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            apply_initial_eval_outcome_overrides(
                {"task a": [0, 0, 0]},
                {"task a": [1, 2, 0]},
                candidate_tasks=["task a"],
                window_size=3,
            )

    def test_override_rejects_unknown_task(self):
        with self.assertRaisesRegex(ValueError, "unknown tasks"):
            apply_initial_eval_outcome_overrides(
                {"task a": [0, 0, 0]},
                {"task b": [1, 1, 1]},
                candidate_tasks=["task a"],
                window_size=3,
            )


if __name__ == "__main__":
    unittest.main()
