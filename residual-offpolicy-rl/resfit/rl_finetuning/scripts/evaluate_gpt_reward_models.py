#!/usr/bin/env python3
"""Compare multimodal models on saved task-generation and reward examples.

Fill ``EVALUATION_CASES`` below, then run this file.  The reward prompt and
stacking references intentionally mirror ``task_reward_generator.py``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_IMAGE_DIR = Path(
    "/home/yuan/self_vla/residual-offpolicy-rl/outputs/task_reward_generation/evaluate_task_reward_img"
)
IMAGE_FILENAME_PREFIX = "masked_image"

# API keys are read from .env: OPENAI_API_KEY, GEMINI_API_KEY,
# ANTHROPIC_API_KEY, and KIMI_API_KEY. Never put keys directly in this file.
# Override this list at runtime with, for example:
#   --models gpt-5.6-terra gemini-3.1-pro-preview
DEFAULT_MODELS = [
    "gemini-3.1-pro-preview",
    "claude-fable-5",
    "kimi-k3",
    "gpt-5.6-terra",
    "gpt-5.6-luna",

]

DEFAULT_MODEL_PROVIDERS = {
    "gemini-3.1-pro-preview": "gemini",
    "claude-fable-5": "anthropic",
    "kimi-k3": "kimi",
    "gpt-5.6-terra": "openai",
    "gpt-5.6-luna": "openai",
}

PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "kimi": "KIMI_API_KEY",
}

OPENAI_COMPATIBLE_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "kimi": "https://api.moonshot.ai/v1",
}

# USD per 1M tokens. Keep this table versioned with the result metadata so
# historical runs remain reproducible if providers change their prices.
MODEL_PRICING_USD_PER_MILLION = {
    "gemini-3.1-pro-preview": {"input": 2.0, "cached_input": 2.0, "output": 12.0},
    "claude-fable-5": {"input": 10.0, "cached_input": 10.0, "output": 50.0},
    "kimi-k3": {"input": 3.0, "cached_input": 0.3, "output": 15.0},
    "gpt-5.6-terra": {"input": 2.0, "cached_input": 0.2, "output": 12.0},
    "gpt-5.6-luna": {"input": 0.2, "cached_input": 0.02, "output": 1.2},
}
PRICING_AS_OF = "2026-09-16"

VALID_TASKS = {
    1: "Put a cube into the bowl",
    2: "Take the cube out of the bowl",
    3: "Stack one cube on the other cube",
    4: "Take the top cube off the other cube",
    5: "Open the drawer",
    6: "Close the drawer",
    7: "Hang the mug on the mug tree",
    8: "Take the mug off the mug tree",
}

# FILL GROUND TRUTH HERE.
# selectable_task_ids: all tasks that were selectable in the BEFORE scene.
# selected_task_id: the task actually selected/executed for this image pair.
# reward_ground_truth: 1 if that selected task succeeded, otherwise 0.
# Keeping these values as placeholders makes the script abort before spending
# API credits if a label has not been filled in.
EVALUATION_CASES = [
    {
        "case_id": 1,
        "selectable_task_ids": [2,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 2,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 3,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 4,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 1,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 5,
        "selectable_task_ids": [2,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 6,
        "selectable_task_ids": [2,5,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 7,
        "selectable_task_ids": [2,6,8],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 8,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 9,
        "selectable_task_ids": [4,5,8],
        "selected_task_id": 4,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 10,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 8,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 11,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 1,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 12,
        "selectable_task_ids": [2,5,8],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 13,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 14,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 15,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 16,
        "selectable_task_ids": [4,5,8],
        "selected_task_id":4 ,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 17,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 18,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 19,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 20,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 21,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 22,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
     {
        "case_id": 23,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 24,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 25,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 26,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id":1 ,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 27,
        "selectable_task_ids": [2,5,8],
        "selected_task_id":2 ,
        "reward_ground_truth": 1,
    },
     {
        "case_id": 28,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 1,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 29,
        "selectable_task_ids": [2,5,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 30,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 31,
        "selectable_task_ids": [2,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 32,
        "selectable_task_ids": [2,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 33,
        "selectable_task_ids": [2,6,8],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 34,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 35,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 36,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 37,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 38,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 39,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 40,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 41,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 42,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 43,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 44,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 45,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 46,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 47,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 48,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 49,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 50,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 51,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 52,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 53,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 1,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 54,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 55,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 56,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 57,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 58,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 59,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 1,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 60,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 61,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 4,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 62,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 4,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 63,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 64,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 65,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 66,
        "selectable_task_ids": [4,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 67,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 68,
        "selectable_task_ids": [4,5,8],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 69,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 70,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 71,
        "selectable_task_ids": [4,5,8],
        "selected_task_id": 4,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 72,
        "selectable_task_ids": [1,3,5,8],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 73,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 74,
        "selectable_task_ids": [2,5,7],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 75,
        "selectable_task_ids": [2,5,7],
        "selected_task_id": 7,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 76,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 1,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 77,
        "selectable_task_ids": [2,5,7],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 78,
        "selectable_task_ids": [2,6,7],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 79,
        "selectable_task_ids": [2,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 80,
        "selectable_task_ids": [2,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 81,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 82,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 1,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 83,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 84,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 4,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 85,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 4,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 86,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 4,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 87,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 4,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 88,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 89,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 8,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 90,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 7,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 91,
        "selectable_task_ids": [2,6,7],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 92,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 1,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 93,
        "selectable_task_ids": [1,3,5,7],
        "selected_task_id": 1,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 94,
        "selectable_task_ids": [1,3,6,7],
        "selected_task_id": 3,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 95,
        "selectable_task_ids": [2,6,7],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 96,
        "selectable_task_ids": [2,5,7],
        "selected_task_id": 5,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 97,
        "selectable_task_ids": [1,3,6,8],
        "selected_task_id": 3,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 98,
        "selectable_task_ids": [4,5,8],
        "selected_task_id": 4,
        "reward_ground_truth": 0,
    },
    {
        "case_id": 99,
        "selectable_task_ids": [2,6,8],
        "selected_task_id": 2,
        "reward_ground_truth": 1,
    },
    {
        "case_id": 100,
        "selectable_task_ids": [4,6,8],
        "selected_task_id": 6,
        "reward_ground_truth": 1,
    },
]

STACKING_TASKS = {
    "Stack one cube on the other cube",
    "Take the top cube off the other cube",
}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: int
    selectable_task_ids: tuple[int, ...]
    selectable_tasks: tuple[str, ...]
    selected_task_id: int
    selected_task: str
    reward_ground_truth: int
    before_path: Path
    after_path: Path


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str

    @property
    def display_name(self) -> str:
        return self.model


@dataclass
class EvaluationResult:
    provider: str
    model: str
    case_id: int
    selectable_task_ids: list[int]
    selectable_tasks: list[str]
    predicted_task_ids: list[int] | None
    predicted_tasks: list[str] | None
    task_generation_score: float
    task_raw_response: str | None
    task_error: str | None
    selected_task_id: int
    selected_task: str
    reward_ground_truth: int
    predicted_reward: int | None
    reward_correct: bool
    reward_reason: str | None
    reward_raw_response: str | None
    reward_error: str | None
    task_elapsed_seconds: float
    reward_elapsed_seconds: float
    elapsed_seconds: float
    task_input_tokens: int
    task_cached_input_tokens: int
    task_output_tokens: int
    task_cost_usd: float
    reward_input_tokens: int
    reward_cached_input_tokens: int
    reward_output_tokens: int
    reward_cost_usd: float
    total_tokens: int
    total_cost_usd: float


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multimodal model task-generation and reward success on "
            "saved image pairs."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=(
            "Directory containing masked_image_<id>_0_before.jpg and "
            "masked_image_<id>_1_after.jpg."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model IDs. Known defaults infer their provider; custom IDs use "
            "provider:model, e.g. anthropic:claude-sonnet-5."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (default: <image-dir>/model_evaluation_<timestamp>).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum API/response-parse attempts per model and case.",
    )
    parser.add_argument(
        "--case-start",
        type=int,
        default=None,
        help="Only evaluate cases whose ID is at least this value.",
    )
    parser.add_argument(
        "--case-end",
        type=int,
        default=None,
        help="Only evaluate cases whose ID is at most this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate labels, files, and model names without calling the API.",
    )
    return parser.parse_args()


def image_as_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


def load_cases(image_dir: Path) -> list[EvaluationCase]:
    problems: list[str] = []
    cases: list[EvaluationCase] = []
    seen_ids: set[int] = set()

    for item in EVALUATION_CASES:
        case_id = item.get("case_id")
        selectable_task_ids = item.get("selectable_task_ids")
        selected_task_id = item.get("selected_task_id")
        reward = item.get("reward_ground_truth")
        if type(case_id) is not int or case_id <= 0:
            problems.append(f"invalid case_id: {case_id!r}")
            continue
        if case_id in seen_ids:
            problems.append(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        selectable_ids_are_valid = (
            isinstance(selectable_task_ids, list)
            and bool(selectable_task_ids)
            and all(type(task_id) is int for task_id in selectable_task_ids)
            and len(selectable_task_ids) == len(set(selectable_task_ids))
            and all(task_id in VALID_TASKS for task_id in selectable_task_ids)
        )
        if not selectable_ids_are_valid:
            problems.append(
                f"case {case_id}: fill selectable_task_ids with one or more unique "
                f"task IDs from {sorted(VALID_TASKS)}"
            )
        if type(selected_task_id) is not int or selected_task_id not in VALID_TASKS:
            problems.append(
                f"case {case_id}: fill selected_task_id with one task ID from "
                f"{sorted(VALID_TASKS)}"
            )
        elif selectable_ids_are_valid and selected_task_id not in selectable_task_ids:
            problems.append(
                f"case {case_id}: selected_task_id={selected_task_id} is not in "
                f"selectable_task_ids={selectable_task_ids}"
            )
        if type(reward) is not int or reward not in (0, 1):
            problems.append(
                f"case {case_id}: fill reward_ground_truth with integer 0 or 1"
            )

        before_path = image_dir / (
            f"{IMAGE_FILENAME_PREFIX}_{case_id}_0_before.jpg"
        )
        after_path = image_dir / (
            f"{IMAGE_FILENAME_PREFIX}_{case_id}_1_after.jpg"
        )
        if not before_path.is_file():
            problems.append(f"case {case_id}: missing {before_path}")
        if not after_path.is_file():
            problems.append(f"case {case_id}: missing {after_path}")

        labels_are_valid = (
            selectable_ids_are_valid
            and type(selected_task_id) is int
            and selected_task_id in VALID_TASKS
            and selected_task_id in selectable_task_ids
            and type(reward) is int
            and reward in (0, 1)
        )
        if labels_are_valid:
            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    selectable_task_ids=tuple(selectable_task_ids),
                    selectable_tasks=tuple(
                        VALID_TASKS[task_id] for task_id in selectable_task_ids
                    ),
                    selected_task_id=selected_task_id,
                    selected_task=VALID_TASKS[selected_task_id],
                    reward_ground_truth=reward,
                    before_path=before_path,
                    after_path=after_path,
                )
            )

    if problems:
        formatted = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(f"Evaluation data is incomplete or invalid:\n{formatted}")
    return cases


def load_stacking_references() -> list[dict[str, Any]]:
    reference_dir = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "task_reward_references"
    )
    specs = [
        ("STACKED POSITIVE REFERENCE 1", "stacked_1.jpg", True),
        ("STACKED POSITIVE REFERENCE 2", "stacked_2.jpg", True),
        ("STACKED POSITIVE REFERENCE 3", "stacked_3.jpg", True),
        ("UNSTACKED NEGATIVE REFERENCE 1", "unstacked_1.jpg", False),
        ("UNSTACKED NEGATIVE REFERENCE 2", "unstacked_2.jpg", False),
    ]
    references = []
    for label, filename, is_stacked in specs:
        path = reference_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Stacking reference image not found: {path}")
        references.append(
            {
                "label": label,
                "is_stacked": is_stacked,
                "image_url": image_as_data_url(path),
            }
        )
    return references


def load_scene_state_reference() -> str:
    reference_path = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "task_reward_generation"
        / "prompt.jpg"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Drawer/mug scene-state reference image not found: {reference_path}"
        )
    return image_as_data_url(reference_path)


def make_geometry_detail_data_url(image_path: Path) -> str:
    """Mirror TaskRewardGenerator.make_geometry_detail_image()."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Task-generation evaluation requires OpenCV (cv2). "
            "Activate the project's Python environment."
        ) from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        luminance
    )
    detail = cv2.cvtColor(
        cv2.merge([luminance, channel_a, channel_b]),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(detail, (0, 0), 1.0)
    detail = cv2.addWeighted(detail, 1.5, blurred, -0.5, 0)
    ok, buffer = cv2.imencode(".jpg", detail, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to encode geometry detail image: {image_path}")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_task_generation_content(
    evaluation_case: EvaluationCase,
    scene_state_reference: str,
    stacking_references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the same scene-state classification used by task generation."""
    prompt = (
        "Classify the current robot scene from the labeled reference images and "
        "current-scene images. The scene contains exactly two cubes, one bowl, "
        "one lower handled drawer, one mug, and one mug tree.\n\n"
        "Return only one JSON object with exactly these fields:\n"
        '{"cube_count_in_bowl": 0, "cubes_stacked": false, '
        '"drawer_open": false, "mug_on_tree": false}\n\n'
        "cube_count_in_bowl must be integer 0 or 1. Decide it from direct visible "
        "evidence in the natural-color current image. If it is 1, "
        "cubes_stacked must be false. Never infer that an occluded cube is in the bowl.\n"
        "Only when cube_count_in_bowl=0, judge cubes_stacked. Stacked means true "
        "vertical 3D support: an elevated cube rests on and occludes the lower cube. "
        "Nearby cubes, touching side edges, or 2D alignment are not stacking. Compare "
        "against both positive and negative stacking references; offset stacks in the "
        "positive references are still stacked.\n"
        "drawer_open describes only the lower movable handled drawer. It is true only "
        "when the front panel is clearly moved outward or the interior is exposed. "
        "Compare natural color with the closed reference and ignore the upper panel.\n"
        "mug_on_tree is true only when the mug is visibly hanging on the mug tree.\n"
        "Use the geometry-enhanced current image only for cube support, occlusion, "
        "boundaries, and stacking. Do not use it for bowl, drawer, or mug judgments."
    )
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {
            "type": "input_text",
            "text": (
                "DRAWER/MUG REFERENCE (ground truth): drawer_open=false, "
                "mug_on_tree=true. Ignore this image for cube judgments."
            ),
        },
        {
            "type": "input_image",
            "image_url": scene_state_reference,
            "detail": "high",
        },
    ]
    for reference in stacking_references:
        stacked_value = str(reference["is_stacked"]).lower()
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"{reference['label']} (ground truth): "
                        f"cubes_stacked={stacked_value}. Ignore this image's "
                        "drawer, mug, and bowl states."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": reference["image_url"],
                    "detail": "high",
                },
            ]
        )
    content.extend(
        [
            {
                "type": "input_text",
                "text": "CURRENT SCENE IMAGE, natural-color view.",
            },
            {
                "type": "input_image",
                "image_url": image_as_data_url(evaluation_case.before_path),
                "detail": "high",
            },
            {
                "type": "input_text",
                "text": "CURRENT SCENE IMAGE, geometry-enhanced view.",
            },
            {
                "type": "input_image",
                "image_url": make_geometry_detail_data_url(
                    evaluation_case.before_path
                ),
                "detail": "high",
            },
        ]
    )
    return content


def parse_scene_state_response(output: str) -> dict[str, Any]:
    json_start = output.find("{")
    json_end = output.rfind("}")
    if json_start == -1 or json_end == -1:
        raise RuntimeError(f"Scene-state response is not valid JSON: {output}")
    try:
        result = json.loads(output[json_start : json_end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Scene-state response contains invalid JSON: {output}"
        ) from exc

    expected = {
        "cube_count_in_bowl",
        "cubes_stacked",
        "drawer_open",
        "mug_on_tree",
    }
    if not isinstance(result, dict) or set(result) != expected:
        raise RuntimeError(
            f"Scene-state response must contain exactly {sorted(expected)}: {output}"
        )
    cube_count = result["cube_count_in_bowl"]
    if type(cube_count) is not int or cube_count not in (0, 1):
        raise RuntimeError(f"cube_count_in_bowl must be integer 0 or 1: {output}")
    for field in ("cubes_stacked", "drawer_open", "mug_on_tree"):
        if type(result[field]) is not bool:
            raise RuntimeError(f"{field} must be a JSON boolean: {output}")
    if cube_count == 1:
        result["cubes_stacked"] = False
    return result


def scene_state_to_task_ids(scene_state: dict[str, Any]) -> list[int]:
    cube_count = scene_state["cube_count_in_bowl"]
    cubes_stacked = scene_state["cubes_stacked"]
    drawer_open = scene_state["drawer_open"]
    mug_on_tree = scene_state["mug_on_tree"]
    feasible = {
        1: cube_count == 0 and not cubes_stacked,
        2: cube_count == 1 and not cubes_stacked,
        3: cube_count == 0 and not cubes_stacked,
        4: cube_count == 0 and cubes_stacked,
        5: not drawer_open,
        6: drawer_open,
        7: not mug_on_tree,
        8: mug_on_tree,
    }
    return [task_id for task_id in VALID_TASKS if feasible[task_id]]


def task_generation_score(
    ground_truth_ids: tuple[int, ...] | list[int],
    predicted_ids: list[int],
) -> float:
    """Partial credit for a correct subset; any false positive makes it zero."""
    ground_truth = set(ground_truth_ids)
    predicted = set(predicted_ids)
    if not predicted.issubset(ground_truth):
        return 0.0
    return len(predicted) / len(ground_truth)


def build_reward_content(
    evaluation_case: EvaluationCase,
    stacking_references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Keep this prompt in sync with TaskRewardGenerator.reward_generator().
    prompt_reward = (
        "You are given two images:\n"
        "- BEFORE: the scene before the robot starts the task.\n"
        "- AFTER: the scene after the robot attempted the task.\n\n"
        "The robot was instructed to perform the following task:\n"
        f"{evaluation_case.selected_task}\n\n"
        "Instructions:\n"
        "1. Carefully inspect the BEFORE state of the task-relevant object(s).\n"
        "2. Pay attention to whether an object is inside a bowl, drawer, or container.\n"
        "3. Carefully inspect the AFTER state of the task-relevant object(s).\n"
        "4. Distinguish similar colors and shapes using both appearance and spatial context.\n"
        "5. Judge the physical task result, not merely a change in image pixels. "
        "Use clear direct evidence and be conservative.\n\n"
        "Return only one valid JSON object with exactly these fields:\n"
        '{"reward": 1, "before_state": "task-relevant object state before", '
        '"after_state": "task-relevant object state after", "task_completed": true}\n'
        "Set reward to integer 1 only if the task was visibly completed; otherwise set "
        "it to integer 0. Set task_completed=true exactly when reward=1, and false "
        "exactly when reward=0. Write before_state and after_state in concise English, "
        "describing only task-relevant visible state in the corresponding images. "
        "Do not provide step-by-step reasoning or discuss unrelated objects.\n"
    )

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt_reward}
    ]
    if evaluation_case.selected_task in STACKING_TASKS:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "The following labeled images are ground-truth stacking "
                    "references. Use them only to calibrate whether the two cubes "
                    "are physically stacked."
                ),
            }
        )
        for reference in stacking_references:
            stacked_value = str(reference["is_stacked"]).lower()
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            f"{reference['label']} (ground truth): "
                            f"cubes_stacked={stacked_value}."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": reference["image_url"],
                        "detail": "high",
                    },
                ]
            )

    content.extend(
        [
            {"type": "input_text", "text": "BEFORE"},
            {
                "type": "input_image",
                "image_url": image_as_data_url(evaluation_case.before_path),
                "detail": "high",
            },
            {"type": "input_text", "text": "AFTER"},
            {
                "type": "input_image",
                "image_url": image_as_data_url(evaluation_case.after_path),
                "detail": "high",
            },
        ]
    )
    return content


def parse_reward_response(output: str) -> tuple[int, str]:
    """Use the same strict response contract as the live reward generator."""
    json_start = output.find("{")
    json_end = output.rfind("}")
    if json_start == -1 or json_end == -1:
        raise RuntimeError(f"Reward response is not valid JSON: {output}")
    try:
        result = json.loads(output[json_start : json_end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Reward response contains invalid JSON: {output}") from exc

    expected_fields = {"reward", "before_state", "after_state", "task_completed"}
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise RuntimeError(
            f"Reward response must contain exactly {sorted(expected_fields)}: {output}"
        )
    reward = result["reward"]
    before_state = result["before_state"]
    after_state = result["after_state"]
    task_completed = result["task_completed"]
    if type(reward) is not int or reward not in (0, 1):
        raise RuntimeError(f"Reward must be integer 0 or 1: {output}")
    if not isinstance(before_state, str) or not before_state.strip():
        raise RuntimeError(f"before_state must be a non-empty string: {output}")
    if not isinstance(after_state, str) or not after_state.strip():
        raise RuntimeError(f"after_state must be a non-empty string: {output}")
    if type(task_completed) is not bool or task_completed != (reward == 1):
        raise RuntimeError(f"task_completed and reward are inconsistent: {output}")

    normalized_before = " ".join(before_state.split()).rstrip(" .;:,！。，；：")
    normalized_after = " ".join(after_state.split()).rstrip(" .;:,！。，；：")
    completion = "completed" if task_completed else "not completed"
    reason = (
        f"Before the attempt, {normalized_before}; afterward, {normalized_after}, "
        f"so the task was {completion} and reward={reward}."
    )
    return reward, reason


def resolve_model_target(value: str) -> ModelTarget:
    if value in DEFAULT_MODEL_PROVIDERS:
        return ModelTarget(DEFAULT_MODEL_PROVIDERS[value], value)
    if ":" in value:
        provider, model = value.split(":", 1)
        if provider not in PROVIDER_API_KEYS:
            raise ValueError(
                f"Unknown provider {provider!r}; choose from "
                f"{sorted(PROVIDER_API_KEYS)}"
            )
        if not model.strip():
            raise ValueError(f"Model ID is empty in {value!r}")
        return ModelTarget(provider, model)
    if value.startswith("gpt-"):
        return ModelTarget("openai", value)
    raise ValueError(
        f"Cannot infer provider for {value!r}. Use provider:model, for example "
        "gemini:gemini-3.1-pro-preview."
    )


def create_model_client(
    target: ModelTarget,
    openai_class: Any,
    anthropic_class: Any,
) -> Any:
    key_name = PROVIDER_API_KEYS[target.provider]
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(
            f"{key_name} is required for {target.provider}:{target.model}. "
            "Add it to the repository .env file."
        )
    if target.provider == "anthropic":
        return anthropic_class(api_key=api_key)
    if target.provider == "openai":
        return openai_class(api_key=api_key)
    return openai_class(
        api_key=api_key,
        base_url=OPENAI_COMPATIBLE_BASE_URLS[target.provider],
    )


def responses_content_to_chat_content(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted = []
    for item in content:
        if item["type"] == "input_text":
            converted.append({"type": "text", "text": item["text"]})
        elif item["type"] == "input_image":
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item["image_url"]},
                }
            )
        else:
            raise ValueError(f"Unsupported prompt item: {item['type']!r}")
    return converted


def responses_content_to_anthropic_content(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted = []
    for item in content:
        if item["type"] == "input_text":
            converted.append({"type": "text", "text": item["text"]})
        elif item["type"] == "input_image":
            data_url = item["image_url"]
            header, encoded = data_url.split(",", 1)
            media_type = header.removeprefix("data:").removesuffix(";base64")
            converted.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded,
                    },
                }
            )
        else:
            raise ValueError(f"Unsupported prompt item: {item['type']!r}")
    return converted


def call_model(
    client: Any,
    target: ModelTarget,
    content: list[dict[str, Any]],
) -> ModelResponse:
    started = time.monotonic()
    if target.provider == "openai":
        response = client.responses.create(
            model=target.model,
            input=[{"role": "user", "content": content}],
        )
        usage = response.usage
        details = getattr(usage, "input_tokens_details", None)
        return ModelResponse(
            text=response.output_text.strip(),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            elapsed_seconds=time.monotonic() - started,
        )
    elif target.provider == "anthropic":
        response = client.messages.create(
            model=target.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": responses_content_to_anthropic_content(content),
                }
            ],
        )
        text_blocks = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            raise RuntimeError("Anthropic response contained no text")
        usage = response.usage
        return ModelResponse(
            text="\n".join(text_blocks).strip(),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            cached_input_tokens=int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            elapsed_seconds=time.monotonic() - started,
        )
    else:
        response = client.chat.completions.create(
            model=target.model,
            messages=[
                {
                    "role": "user",
                    "content": responses_content_to_chat_content(content),
                }
            ],
        )
        output = response.choices[0].message.content
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError(f"{target.provider} response contained no text")
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        return ModelResponse(
            text=output.strip(),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            elapsed_seconds=time.monotonic() - started,
        )


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float:
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
    if pricing is None:
        raise ValueError(
            f"No pricing configured for {model!r}; add it to "
            "MODEL_PRICING_USD_PER_MILLION before making paid requests"
        )
    cached = min(input_tokens, cached_input_tokens)
    uncached = input_tokens - cached
    cost = (
        uncached * pricing["input"]
        + cached * pricing["cached_input"]
        + output_tokens * pricing["output"]
    ) / 1_000_000
    return cost


def evaluate_one(
    client: Any,
    target: ModelTarget,
    evaluation_case: EvaluationCase,
    scene_state_reference: str,
    stacking_references: list[dict[str, Any]],
    max_attempts: int,
) -> EvaluationResult:
    started = time.monotonic()
    task_started = time.monotonic()
    task_input_tokens = 0
    task_cached_input_tokens = 0
    task_output_tokens = 0
    task_raw_response: str | None = None
    task_error: str | None = None
    predicted_task_ids: list[int] | None = None
    task_content = build_task_generation_content(
        evaluation_case,
        scene_state_reference,
        stacking_references,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            response = call_model(client, target, task_content)
            task_input_tokens += response.input_tokens
            task_cached_input_tokens += response.cached_input_tokens
            task_output_tokens += response.output_tokens
            task_raw_response = response.text
            scene_state = parse_scene_state_response(task_raw_response)
            predicted_task_ids = scene_state_to_task_ids(scene_state)
            task_error = None
            break
        except Exception as exc:
            task_error = (
                f"attempt {attempt}/{max_attempts}: {type(exc).__name__}: {exc}"
            )
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 8))
    task_elapsed_seconds = time.monotonic() - task_started
    task_cost_usd = calculate_cost_usd(
        target.model,
        task_input_tokens,
        task_cached_input_tokens,
        task_output_tokens,
    )

    task_score = (
        0.0
        if predicted_task_ids is None
        else task_generation_score(
            evaluation_case.selectable_task_ids,
            predicted_task_ids,
        )
    )

    reward_raw_response: str | None = None
    reward_error: str | None = None
    predicted_reward: int | None = None
    reward_reason: str | None = None
    reward_content = build_reward_content(evaluation_case, stacking_references)
    reward_started = time.monotonic()
    reward_input_tokens = 0
    reward_cached_input_tokens = 0
    reward_output_tokens = 0

    for attempt in range(1, max_attempts + 1):
        try:
            response = call_model(client, target, reward_content)
            reward_input_tokens += response.input_tokens
            reward_cached_input_tokens += response.cached_input_tokens
            reward_output_tokens += response.output_tokens
            reward_raw_response = response.text
            predicted_reward, reward_reason = parse_reward_response(
                reward_raw_response
            )
            reward_error = None
            break
        except Exception as exc:
            reward_error = (
                f"attempt {attempt}/{max_attempts}: {type(exc).__name__}: {exc}"
            )
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 8))
    reward_elapsed_seconds = time.monotonic() - reward_started
    reward_cost_usd = calculate_cost_usd(
        target.model,
        reward_input_tokens,
        reward_cached_input_tokens,
        reward_output_tokens,
    )

    return EvaluationResult(
        provider=target.provider,
        model=target.display_name,
        case_id=evaluation_case.case_id,
        selectable_task_ids=list(evaluation_case.selectable_task_ids),
        selectable_tasks=list(evaluation_case.selectable_tasks),
        predicted_task_ids=predicted_task_ids,
        predicted_tasks=(
            None
            if predicted_task_ids is None
            else [VALID_TASKS[task_id] for task_id in predicted_task_ids]
        ),
        task_generation_score=task_score,
        task_raw_response=task_raw_response,
        task_error=task_error,
        selected_task_id=evaluation_case.selected_task_id,
        selected_task=evaluation_case.selected_task,
        reward_ground_truth=evaluation_case.reward_ground_truth,
        predicted_reward=predicted_reward,
        reward_correct=(
            predicted_reward == evaluation_case.reward_ground_truth
        ),
        reward_reason=reward_reason,
        reward_raw_response=reward_raw_response,
        reward_error=reward_error,
        task_elapsed_seconds=round(task_elapsed_seconds, 3),
        reward_elapsed_seconds=round(reward_elapsed_seconds, 3),
        elapsed_seconds=round(time.monotonic() - started, 3),
        task_input_tokens=task_input_tokens,
        task_cached_input_tokens=task_cached_input_tokens,
        task_output_tokens=task_output_tokens,
        task_cost_usd=task_cost_usd,
        reward_input_tokens=reward_input_tokens,
        reward_cached_input_tokens=reward_cached_input_tokens,
        reward_output_tokens=reward_output_tokens,
        reward_cost_usd=reward_cost_usd,
        total_tokens=(
            task_input_tokens
            + task_output_tokens
            + reward_input_tokens
            + reward_output_tokens
        ),
        total_cost_usd=task_cost_usd + reward_cost_usd,
    )


def build_summary(
    models: list[str], results: list[EvaluationResult], total_cases: int
) -> list[dict[str, Any]]:
    summary = []
    for model in models:
        model_results = [result for result in results if result.model == model]
        task_score_sum = sum(
            result.task_generation_score for result in model_results
        )
        reward_correct = sum(result.reward_correct for result in model_results)
        task_errors = sum(
            result.task_error is not None for result in model_results
        )
        reward_errors = sum(
            result.reward_error is not None for result in model_results
        )
        task_times = [result.task_elapsed_seconds for result in model_results]
        reward_times = [result.reward_elapsed_seconds for result in model_results]
        summary.append(
            {
                "model": model,
                "total": total_cases,
                "task_generation_score_sum": task_score_sum,
                "task_generation_success_rate": task_score_sum / total_cases,
                "reward_correct": reward_correct,
                "reward_success_rate": reward_correct / total_cases,
                "task_errors": task_errors,
                "reward_errors": reward_errors,
                "task_time_mean_seconds": statistics.fmean(task_times),
                "task_time_variance_seconds_squared": statistics.pvariance(task_times),
                "reward_time_mean_seconds": statistics.fmean(reward_times),
                "reward_time_variance_seconds_squared": statistics.pvariance(reward_times),
                "task_input_tokens": sum(r.task_input_tokens for r in model_results),
                "task_cached_input_tokens": sum(
                    r.task_cached_input_tokens for r in model_results
                ),
                "task_output_tokens": sum(r.task_output_tokens for r in model_results),
                "task_cost_usd": sum(r.task_cost_usd for r in model_results),
                "reward_input_tokens": sum(r.reward_input_tokens for r in model_results),
                "reward_cached_input_tokens": sum(
                    r.reward_cached_input_tokens for r in model_results
                ),
                "reward_output_tokens": sum(r.reward_output_tokens for r in model_results),
                "reward_cost_usd": sum(r.reward_cost_usd for r in model_results),
                "total_tokens": sum(r.total_tokens for r in model_results),
                "total_cost_usd": sum(r.total_cost_usd for r in model_results),
            }
        )
    return summary


def save_results(
    output_dir: Path,
    image_dir: Path,
    models: list[str],
    results: list[EvaluationResult],
    summary: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "image_dir": str(image_dir.resolve()),
        "models": models,
        "pricing_as_of": PRICING_AS_OF,
        "pricing_usd_per_million_tokens": {
            model: MODEL_PRICING_USD_PER_MILLION[model] for model in models
        },
        "time_variance_definition": "population variance (divide by N)",
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(asdict(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> int:
    args = parse_args()
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    model_values = list(dict.fromkeys(args.models))
    if not model_values or any(not model.strip() for model in model_values):
        raise ValueError("--models must contain at least one non-empty model ID")
    targets = [resolve_model_target(model) for model in model_values]
    missing_prices = [
        target.model
        for target in targets
        if target.model not in MODEL_PRICING_USD_PER_MILLION
    ]
    if missing_prices and not args.dry_run:
        raise ValueError(f"Missing pricing configuration for models: {missing_prices}")

    cases = load_cases(args.image_dir)
    if args.case_start is not None and args.case_end is not None:
        if args.case_start > args.case_end:
            raise ValueError("--case-start must be <= --case-end")
    cases = [
        case
        for case in cases
        if (args.case_start is None or case.case_id >= args.case_start)
        and (args.case_end is None or case.case_id <= args.case_end)
    ]
    if not cases:
        raise ValueError("No cases matched the requested case range")
    stacking_references = load_stacking_references()
    scene_state_reference = load_scene_state_reference()
    print(f"Validated {len(cases)} cases from {args.image_dir}")
    print("Task IDs:")
    for task_id, task_name in VALID_TASKS.items():
        print(f"  {task_id}: {task_name}")
    print(f"Models ({len(targets)}):")
    for target in targets:
        print(f"  {target.provider}: {target.model}")
    if args.dry_run:
        print("Dry run complete; no API requests were made.")
        return 0

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing API dependency. Activate the residual conda environment "
            "or install openai and python-dotenv."
        ) from exc

    Anthropic = None
    if any(target.provider == "anthropic" for target in targets):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude evaluation requires the anthropic package. "
                "Install it in the residual conda environment."
            ) from exc

    load_dotenv()
    results: list[EvaluationResult] = []
    for target in targets:
        client = create_model_client(target, OpenAI, Anthropic)
        print(f"\nEvaluating {target.provider}:{target.model}")
        for evaluation_case in cases:
            result = evaluate_one(
                client,
                target,
                evaluation_case,
                scene_state_reference,
                stacking_references,
                args.max_attempts,
            )
            results.append(result)
            reward_prediction = (
                "ERROR" if result.predicted_reward is None else result.predicted_reward
            )
            reward_mark = "OK" if result.reward_correct else "WRONG"
            task_prediction = (
                "ERROR"
                if result.predicted_task_ids is None
                else result.predicted_task_ids
            )
            print(
                f"  case {result.case_id:>2}: task predicted={task_prediction}, "
                f"ground_truth={result.selectable_task_ids}, "
                f"score={result.task_generation_score:.4f}; "
                f"reward predicted={reward_prediction}, "
                f"ground_truth={result.reward_ground_truth} [{reward_mark}]; "
                f"time={result.task_elapsed_seconds:.3f}s+"
                f"{result.reward_elapsed_seconds:.3f}s, "
                f"tokens={result.total_tokens}, cost=${result.total_cost_usd:.6f}"
            )
            if result.task_error:
                print(f"    task error: {result.task_error}", file=sys.stderr)
            if result.reward_error:
                print(f"    reward error: {result.reward_error}", file=sys.stderr)

    model_names = [target.display_name for target in targets]
    summary = build_summary(model_names, results, len(cases))
    print("\nModel success rates (API/parse errors count as zero/incorrect):")
    for item in summary:
        print(
            f"  {item['model']}: "
            f"task_generation={item['task_generation_success_rate']:.2%} "
            f"(score_sum={item['task_generation_score_sum']:.4f}/"
            f"{item['total']}, errors={item['task_errors']}), "
            f"reward={item['reward_success_rate']:.2%} "
            f"({item['reward_correct']}/{item['total']}, "
            f"errors={item['reward_errors']}), "
            f"task_time mean={item['task_time_mean_seconds']:.3f}s "
            f"var={item['task_time_variance_seconds_squared']:.3f}s^2, "
            f"reward_time mean={item['reward_time_mean_seconds']:.3f}s "
            f"var={item['reward_time_variance_seconds_squared']:.3f}s^2, "
            f"tokens={item['total_tokens']}, cost=${item['total_cost_usd']:.6f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        args.image_dir / f"model_evaluation_{timestamp}"
    )
    save_results(output_dir, args.image_dir, model_names, results, summary)
    print(f"Detailed JSON and CSV results: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
