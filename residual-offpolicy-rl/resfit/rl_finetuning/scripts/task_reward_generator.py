import os
import sys

import cv2
import time, threading, queue, random
import subprocess
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime
from pathlib import Path

import base64
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

import uvicorn
import zerorpc
import asyncio
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

import json_numpy
from resfit.rl_finetuning.utils.task_sampling import (
    compute_task_sampling_probabilities,
)

json_numpy.patch()

DEFAULT_CANDIDATE_TASKS = [
    "Put a cube into the bowl",
    "Take the cube out of the bowl",
    "Stack one cube on the other cube",
    "Take the top cube off the other cube",
    "Open the drawer",
    "Close the drawer",
    "Hang the mug on the mug tree",
    "Take the mug off the mug tree",
]

# Width of ``YYYY-MM-DD HH:mm:ss | INFO | `` in the file logger format.
# Loguru adds this prefix only to the first line of a multiline message, so it
# must be included when indenting continuation lines for visual alignment.
INFO_FILE_LOG_PREFIX_WIDTH = 29


def _format_aligned_info_items(
    message_label: str,
    items: list[str],
) -> str:
    """Put one item per line, aligned with the INFO message label."""
    item_indent = " " * INFO_FILE_LOG_PREFIX_WIDTH
    return (
        f"{message_label.rstrip()}\n"
        f"{item_indent}"
        + (", \n" + item_indent).join(items)
    )


def _resolve_candidate_tasks(candidate_tasks: Optional[list[str]]) -> list[str]:
    selected_tasks = (
        DEFAULT_CANDIDATE_TASKS if candidate_tasks is None else candidate_tasks
    )
    if not selected_tasks:
        raise ValueError("candidate_tasks must be null or a non-empty list")
    if any(not isinstance(task, str) or not task.strip() for task in selected_tasks):
        raise ValueError("candidate_tasks must contain only non-empty strings")
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("candidate_tasks must not contain duplicates")
    return list(selected_tasks)


def _parse_reward_response(output: str) -> tuple[str, str]:
    """Parse the reward model's compact, evidence-based JSON response."""
    json_start = output.find("{")
    json_end = output.rfind("}")
    if json_start == -1 or json_end == -1:
        raise RuntimeError(f"Reward response is not valid JSON: {output}")

    try:
        result = json.loads(output[json_start:json_end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Reward response contains invalid JSON: {output}"
        ) from exc

    expected_fields = {
        "reward",
        "before_state",
        "after_state",
        "task_completed",
    }
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise RuntimeError(
            "Reward response must contain exactly "
            f"{sorted(expected_fields)}: "
            f"{output}"
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
    if type(task_completed) is not bool:
        raise RuntimeError(f"task_completed must be a JSON boolean: {output}")
    if task_completed != (reward == 1):
        raise RuntimeError(
            "task_completed and reward are inconsistent in reward response: "
            f"{output}"
        )

    # Keep one logical record on one log line even if the model inserted
    # whitespace or newlines into the JSON string.
    normalized_before_state = " ".join(before_state.split()).rstrip(" .;:,！。，；：")
    normalized_after_state = " ".join(after_state.split()).rstrip(" .;:,！。，；：")
    completion_text = "completed" if task_completed else "not completed"
    normalized_reason = (
        f"Before the attempt, {normalized_before_state}. Afterward, "
        f"{normalized_after_state}, so the task was {completion_text} "
        f"and reward={reward}."
    )
    return str(reward), normalized_reason


class TaskRewardGenerator:
    def __init__(
        self,
        max_timesteps: int = 60,
        zed_stream_ip: str = "192.168.55.1",
        zed_stream_port: int = 30000,
        save_images: bool = True,
        candidate_tasks: Optional[list[str]] = None,
        success_rate_window_size: int = 20,
        min_task_sample_probability: float = 0.05,
        prioritize_low_success_tasks: bool = True,
    ):
        self.max_timesteps = max_timesteps
        self.save_images = bool(save_images)
        self.candidate_tasks = _resolve_candidate_tasks(candidate_tasks)
        self.success_rate_window_size = int(success_rate_window_size)
        if self.success_rate_window_size <= 0:
            raise ValueError("success_rate_window_size must be positive")
        self.min_task_sample_probability = float(min_task_sample_probability)
        self.prioritize_low_success_tasks = bool(prioritize_low_success_tasks)
        max_min_probability = 1.0 / len(self.candidate_tasks)
        if not 0.0 <= self.min_task_sample_probability <= max_min_probability:
            raise ValueError(
                "min_task_sample_probability must be in "
                f"[0, {max_min_probability}] for "
                f"{len(self.candidate_tasks)} candidate tasks, got "
                f"{self.min_task_sample_probability}"
            )
        self.current_scene = None
        self.next_scene = None

        # Camera frames are supplied by RobotEnv. Keep the stream arguments in
        # the public constructor for backward compatibility, but do not open a
        # separate (and previously unused) connection to port 30000 here.
        self.zed_stream_ip = zed_stream_ip
        self.zed_stream_port = int(zed_stream_port)
        self.camera = None
        self.image = None

        self.round_current = 0
        self.round_next = 1
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = Path("outputs/task_reward_generation") / self.timestamp
        self.output_dir = self.output_root
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.queue: "Queue[tuple[list[str], list[str]]]" = Queue()

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Fixed visual reference for the known drawer-closed and mug-on-tree states.
        self.scene_state_reference_path = (
            Path(__file__).resolve().parents[3]
            / "outputs"
            / "task_reward_generation"
            / "prompt.jpg"
        )
        if not self.scene_state_reference_path.is_file():
            raise FileNotFoundError(
                "Scene-state reference image not found at: "
                f"{self.scene_state_reference_path}"
            )
        self.scene_state_reference = base64.b64encode(
            self.scene_state_reference_path.read_bytes()
        ).decode("utf-8")

        # Same-camera few-shot references for the visually ambiguous cube
        # relation.  Keep these under source-controlled assets instead of a
        # timestamped training output directory so future runs do not lose the
        # prompt examples when old outputs are cleaned up.
        stacking_reference_dir = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "task_reward_references"
        )
        stacking_reference_specs = [
            ("STACKED POSITIVE REFERENCE 1", "stacked_1.jpg", True),
            ("STACKED POSITIVE REFERENCE 2", "stacked_2.jpg", True),
            ("STACKED POSITIVE REFERENCE 3", "stacked_3.jpg", True),
            ("UNSTACKED NEGATIVE REFERENCE 1", "unstacked_1.jpg", False),
            ("UNSTACKED NEGATIVE REFERENCE 2", "unstacked_2.jpg", False),
        ]
        self.stacking_references = []
        for label, filename, is_stacked in stacking_reference_specs:
            reference_path = stacking_reference_dir / filename
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"Stacking reference image not found at: {reference_path}"
                )
            self.stacking_references.append({
                "label": label,
                "is_stacked": is_stacked,
                "image": base64.b64encode(reference_path.read_bytes()).decode(
                    "utf-8"
                ),
            })


        self.round = 0
        self.scene_before = None
        self.scene_after = None
        self.selected_task = None
        self.last_reward_reason = None
        self.img_rgb = None

        # Each task has an independent fixed-size rolling window. A fresh
        # object uses a scalar fallback until an initial evaluation provides
        # the ordered binary outcomes that seed the real window.
        self.task_success_stats = {
            task: {
                "initial_success_rate": 0.5,
                "initial_eval_outcomes": None,
                "recent_outcomes": [],
                "attempts": 0,
            }
            for task in self.candidate_tasks
        }

    def get_task_success_rate(self, task: str) -> float:
        stats = self.task_success_stats[task]
        recent_outcomes = stats["recent_outcomes"]
        initial_eval_outcomes = stats.get("initial_eval_outcomes")
        if initial_eval_outcomes is None:
            baseline_slots = self.success_rate_window_size - len(recent_outcomes)
            total = (
                sum(recent_outcomes)
                + baseline_slots * stats["initial_success_rate"]
            )
        else:
            # Appending k training outcomes to the initial eval window evicts
            # its first k outcomes. Once k reaches the window size, the rate is
            # based entirely on training episodes.
            remaining_eval_outcomes = initial_eval_outcomes[len(recent_outcomes):]
            total = sum(remaining_eval_outcomes) + sum(recent_outcomes)
        return total / self.success_rate_window_size

    def get_task_attempts(self, task: str) -> int:
        return int(self.task_success_stats[task]["attempts"])

    def initialize_task_success_rates(
        self, initial_success_rates: dict[str, float]
    ) -> None:
        """Seed legacy scalar baselines when ordered outcomes are unavailable."""
        missing_tasks = set(self.candidate_tasks) - set(initial_success_rates)
        if missing_tasks:
            raise ValueError(
                "Initial success rates are missing configured tasks: "
                f"{sorted(missing_tasks)}"
            )

        initialized = {}
        for task in self.candidate_tasks:
            success_rate = float(initial_success_rates[task])
            if not 0.0 <= success_rate <= 1.0:
                raise ValueError(
                    f"Initial success rate for {task!r} must be in [0, 1], "
                    f"got {success_rate}"
                )
            initialized[task] = {
                "initial_success_rate": success_rate,
                "initial_eval_outcomes": None,
                "recent_outcomes": [],
                "attempts": 0,
            }

        # Mutate in place because the learner/checkpoint thread shares this
        # dictionary with the collector.
        self.task_success_stats.clear()
        self.task_success_stats.update(initialized)

    def initialize_task_success_windows(
        self, initial_eval_outcomes: dict[str, list[int]]
    ) -> None:
        """Seed each rolling window with ordered binary evaluation outcomes."""
        missing_tasks = set(self.candidate_tasks) - set(initial_eval_outcomes)
        if missing_tasks:
            raise ValueError(
                "Initial evaluation outcomes are missing configured tasks: "
                f"{sorted(missing_tasks)}"
            )

        initialized = {}
        for task in self.candidate_tasks:
            raw_outcomes = list(initial_eval_outcomes[task])
            if any(value not in (0, 1) for value in raw_outcomes):
                raise ValueError(
                    f"Initial evaluation outcomes for {task!r} must be 0 or 1"
                )
            outcomes = [int(value) for value in raw_outcomes]
            if len(outcomes) != self.success_rate_window_size:
                raise ValueError(
                    f"Initial evaluation for {task!r} must contain exactly "
                    f"{self.success_rate_window_size} outcomes, got {len(outcomes)}"
                )
            initialized[task] = {
                "initial_success_rate": sum(outcomes) / len(outcomes),
                "initial_eval_outcomes": outcomes,
                "recent_outcomes": [],
                "attempts": 0,
            }

        # Mutate in place because the learner/checkpoint thread shares this
        # dictionary with the collector.
        self.task_success_stats.clear()
        self.task_success_stats.update(initialized)

    def get_task_sampling_weight(self, task: str) -> float:
        """Return the all-candidate probability for legacy callers."""
        return self.get_task_sampling_probabilities()[task]

    def get_task_sampling_probabilities(
        self, tasks: Optional[list[str]] = None
    ) -> dict[str, float]:
        """Return normalized probabilities for the requested feasible tasks.

        When low-success prioritization is disabled, every feasible task has
        equal probability. Otherwise, every task at 100% success receives the
        configured fixed probability and the remaining mass is split between
        non-mastered tasks in proportion to ``1 - success_rate``. If every
        requested task is at 100%, the result is uniform.
        """
        sampling_tasks = list(self.candidate_tasks if tasks is None else tasks)
        if not sampling_tasks:
            raise ValueError("At least one task is required for task sampling")
        if len(set(sampling_tasks)) != len(sampling_tasks):
            raise ValueError("Sampling tasks must not contain duplicates")
        unknown_tasks = set(sampling_tasks) - set(self.candidate_tasks)
        if unknown_tasks:
            raise ValueError(f"Unknown tasks for sampling: {sorted(unknown_tasks)}")

        success_rates = {
            task: self.get_task_success_rate(task) for task in sampling_tasks
        }
        return compute_task_sampling_probabilities(
            sampling_tasks,
            success_rates,
            min_task_sample_probability=self.min_task_sample_probability,
            prioritize_low_success_tasks=self.prioritize_low_success_tasks,
        )

    def update_task_success_rate(self, task: str, reward: int) -> float:
        if task not in self.task_success_stats:
            raise ValueError(f"Unknown task for success-rate update: {task!r}")
        if reward not in (0, 1):
            raise ValueError(f"Reward must be 0 or 1, got {reward!r}")
        stats = self.task_success_stats[task]
        # Replace the per-task mapping atomically so the learner's checkpoint
        # thread sees either the complete old state or the complete new state.
        self.task_success_stats[task] = {
            "initial_success_rate": stats["initial_success_rate"],
            "initial_eval_outcomes": stats.get("initial_eval_outcomes"),
            "recent_outcomes": (
                stats["recent_outcomes"] + [int(reward)]
            )[-self.success_rate_window_size:],
            "attempts": stats["attempts"] + 1,
        }
        return self.get_task_success_rate(task)

    def restore_task_success_stats(self, saved_stats: dict) -> None:
        """Restore validated per-task rolling state from a checkpoint."""
        missing_tasks = set(self.candidate_tasks) - set(saved_stats)
        if missing_tasks:
            raise ValueError(
                "Checkpoint task-success stats are missing configured tasks: "
                f"{sorted(missing_tasks)}"
            )
        restored = {}
        for task in self.candidate_tasks:
            stats = saved_stats[task]
            if "initial_success_rate" not in stats:
                # Backward-compatible migration from the old Beta-count
                # checkpoint format. Exact recent outcomes were not recorded,
                # so preserve its rate as the baseline for the new window.
                successes = float(stats["successes"])
                failures = float(stats["failures"])
                if successes <= 0 or failures <= 0:
                    raise ValueError(
                        f"Task success pseudo-counts must be positive for {task!r}"
                    )
                restored[task] = {
                    "initial_success_rate": successes / (successes + failures),
                    "initial_eval_outcomes": None,
                    "recent_outcomes": [],
                    "attempts": max(0, int(successes + failures - 2.0)),
                }
                continue

            initial_success_rate = float(stats["initial_success_rate"])
            raw_initial_eval_outcomes = stats.get("initial_eval_outcomes")
            initial_eval_outcomes = (
                None
                if raw_initial_eval_outcomes is None
                else [int(value) for value in raw_initial_eval_outcomes]
            )
            recent_outcomes = [int(value) for value in stats["recent_outcomes"]]
            attempts = int(stats["attempts"])
            if not 0.0 <= initial_success_rate <= 1.0:
                raise ValueError(
                    f"Initial success rate for {task!r} must be in [0, 1]"
                )
            if any(value not in (0, 1) for value in recent_outcomes):
                raise ValueError(f"Recent outcomes for {task!r} must be 0 or 1")
            if initial_eval_outcomes is not None:
                if len(initial_eval_outcomes) != self.success_rate_window_size:
                    raise ValueError(
                        f"Initial evaluation window for {task!r} must contain "
                        f"{self.success_rate_window_size} outcomes"
                    )
                if any(value not in (0, 1) for value in initial_eval_outcomes):
                    raise ValueError(
                        f"Initial evaluation outcomes for {task!r} must be 0 or 1"
                    )
                eval_success_rate = (
                    sum(initial_eval_outcomes) / self.success_rate_window_size
                )
                if abs(eval_success_rate - initial_success_rate) > 1e-9:
                    raise ValueError(
                        f"Initial evaluation outcomes and success rate disagree "
                        f"for {task!r}"
                    )
            if len(recent_outcomes) > self.success_rate_window_size:
                raise ValueError(
                    f"Recent outcome window for {task!r} exceeds "
                    f"success_rate_window_size={self.success_rate_window_size}"
                )
            if attempts < len(recent_outcomes):
                raise ValueError(
                    f"Attempt count for {task!r} is smaller than its outcome window"
                )
            restored[task] = {
                "initial_success_rate": initial_success_rate,
                "initial_eval_outcomes": initial_eval_outcomes,
                "recent_outcomes": recent_outcomes,
                "attempts": attempts,
            }
        # Mutate in place so any checkpointing thread holding this dictionary
        # keeps seeing the restored and subsequently updated values.
        self.task_success_stats.clear()
        self.task_success_stats.update(restored)

    def setup_logger(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "log.txt"

        logger.remove()
        logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    def set_output_cycle(self, global_step: int) -> Path:
        """Route evaluation and following training artifacts to one step folder."""
        self.output_dir = self.output_root / f"step_{int(global_step)}"
        self.setup_logger()
        return self.output_dir

    def set_output_training(self, global_step: int) -> Path:
        """Append training logs to the most recent completed evaluation cycle."""
        latest_eval_step = 0
        for candidate in self.output_root.glob("step_*"):
            try:
                candidate_step = int(candidate.name.removeprefix("step_"))
            except ValueError:
                continue
            if candidate_step > int(global_step):
                continue
            progress_path = candidate / "evaluation_progress.json"
            if not progress_path.is_file():
                continue
            try:
                with progress_path.open("r", encoding="utf-8") as file:
                    progress = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            if progress.get("status") == "complete":
                latest_eval_step = max(latest_eval_step, candidate_step)

        self.output_dir = self.output_root / f"step_{latest_eval_step}"
        self.setup_logger()
        logger.info(
            "Training log resumed after the most recent completed evaluation "
            f"at step {latest_eval_step} (current global step: {int(global_step)})."
        )
        return self.output_dir

    def set_output_warmup(self) -> Path:
        """Route initial online replay-buffer collection artifacts to warmup/."""
        self.output_dir = self.output_root / "warmup"
        self.setup_logger()
        logger.info("---------------- warmup: collecting online replay buffer ----------------")
        return self.output_dir

    def restore_output_root(self, output_root: str | Path) -> None:
        """Reuse the artifact root stored in a resumed training checkpoint."""
        self.output_root = Path(output_root)
        self.output_dir = self.output_root

    def reduce_exposure_bgr(self, img_bgr, alpha=0.65, beta=-20):
        """Apply alpha * pixel + beta to reduce image exposure for GPT preprocessing."""
        return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


    def enhance_cube_visibility_bgr(self, img_bgr):
        """Enhance saturation and sharpen edges within candidate cube color regions."""

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        green_mask = (
            (h >= 35) & (h <= 90) &
            (s >= 20) &
            (v >= 50)
        )

        yellow_orange_wood_mask = (
            (h >= 8) & (h <= 40) &
            (s >= 15) &
            (v >= 50)
        )

        cube_mask = (green_mask | yellow_orange_wood_mask).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN, kernel)

        cube_mask = cv2.dilate(cube_mask, kernel, iterations=1)

        cube_mask = cv2.GaussianBlur(cube_mask, (9, 9), 0)

        mask_f = cube_mask.astype(np.float32) / 255.0
        mask_f = mask_f[..., None]

        s_enhanced = np.clip(s * 2.3, 0, 255)
        v_enhanced = np.clip(v * 1.08, 0, 255)

        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced]).astype(np.uint8)
        img_color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        blur = cv2.GaussianBlur(img_color_enhanced, (0, 0), 1.0)
        img_sharp = cv2.addWeighted(img_color_enhanced, 1.7, blur, -0.7, 0)

        out = img_bgr.astype(np.float32) * (1.0 - mask_f) + img_sharp.astype(np.float32) * mask_f

        return np.clip(out, 0, 255).astype(np.uint8)

    def mask_image(self, image):
        """Keep the tabletop polygon and fill the remaining pixels with the background color."""
        polygon_points = [
            [1500, 150],
            [1600, 545],
            [1310, 535],
            [1320, 715],
            [1280, 715],
            [1285, 830],
            [600, 820],
            [615, 715],
            [575, 715],
            [610, 520],
            [350, 510],
            [460, 135],
        ]
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

        masked_image = np.full_like(image, (0, 0, 0), dtype=image.dtype)
        masked_image[mask > 0] = image[mask > 0]

        return masked_image


    def process_img(self, img_rgb):
        img_rgb = np.asarray(img_rgb, dtype=np.uint8)
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(
                f"Expected an HWC RGB image with 3 channels, got shape={img_rgb.shape}"
            )

        # The input contract is RGB; OpenCV's JPEG encoder expects BGR.
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Keep a JPEG of the original PI camera frame before exposure changes,
        # visibility enhancement, or masking. capture_scene() saves it next to
        # the corresponding masked GPT image when save_images is enabled.
        ok, unmasked_buffer = cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("Failed to encode unmasked PI image to JPEG")
        unmasked_b64jpg = base64.b64encode(
            unmasked_buffer.tobytes()
        ).decode("utf-8")

        # Save a natural-color masked view in addition to the raw PI frame.
        masked_img_bgr = self.mask_image(img_bgr)
        ok, masked_buffer = cv2.imencode(".jpg", masked_img_bgr)
        if not ok:
            raise RuntimeError("Failed to encode natural-color masked image")
        masked_b64jpg = base64.b64encode(
            masked_buffer.tobytes()
        ).decode("utf-8")

        # Preserve the original GPT preprocessing: adjust exposure, enhance
        # colored-object saturation/brightness and sharpness, then apply the
        # workspace mask. This third image is the one sent to GPT.
        img_bgr_low_exp = self.reduce_exposure_bgr(
            img_bgr,
            alpha=1.0,
            beta=0,
        )
        img_bgr_processed = self.enhance_cube_visibility_bgr(img_bgr_low_exp)
        gpt_img_bgr = self.mask_image(img_bgr_processed)

        # GPT and the saved before/after artifacts must use the same image.
        ok, buffer = cv2.imencode(".jpg", gpt_img_bgr)
        if ok:
            b64jpg = base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            raise RuntimeError("Failed to encode masked image to JPEG")

        return gpt_img_bgr, b64jpg, unmasked_b64jpg, masked_b64jpg

    def make_geometry_detail_image(self, encoded_scene):
        """Create a second full-scene view that emphasizes object boundaries."""
        encoded_bytes = base64.b64decode(encoded_scene)
        image = cv2.imdecode(
            np.frombuffer(encoded_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError("Failed to decode scene for geometry enhancement")

        # CLAHE on luminance reveals the horizontal contact seam and cube side
        # faces without relying on a fixed cube location or changing hue.
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
            raise RuntimeError("Failed to encode geometry-enhanced scene")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def capture_scene(
        self,
        encoded_scene,
        unmasked_scene,
        masked_scene,
        before,
    ):
        """Optionally save matching PI, natural-mask, and enhanced GPT images."""
        if self.save_images:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "0_before" if before else "1_after"
            gpt_save_path = (
                self.output_dir / f"gpt_image_{self.round}_{suffix}.jpg"
            )
            pi_save_path = (
                self.output_dir / f"pi_image_{self.round}_{suffix}.jpg"
            )
            masked_save_path = (
                self.output_dir / f"masked_image_{self.round}_{suffix}.jpg"
            )
            gpt_save_path.write_bytes(base64.b64decode(encoded_scene))
            pi_save_path.write_bytes(base64.b64decode(unmasked_scene))
            masked_save_path.write_bytes(base64.b64decode(masked_scene))
        return encoded_scene

    def task_generator(self, scene):
        geometry_detail_scene = self.make_geometry_detail_image(scene)
        prompt_task = (
            "You are given labeled reference images followed by images of the current "
            "scene. Every label immediately before a reference image is a ground-truth fact.\n"
            "The DRAWER/MUG REFERENCE shows the lower handled drawer "
            "in the CLOSED state (drawer_open=false) and the blue mug HANGING ON the mug tree "
            "(mug_on_tree=true). These two labels are ground-truth facts about the reference image. "
            "Use it only as a visual reference for recognizing those two states; ignore its cubes "
            "and bowl.\n"
            "The STACKED POSITIVE REFERENCES have cubes_stacked=true. The UNSTACKED NEGATIVE "
            "REFERENCES have cubes_stacked=false. These references use the same overhead "
            "viewpoint as the current scene; compare them directly with the current view. "
            "Ignore the references' "
            "drawer, mug, and bowl states. In particular, "
            "the positive examples demonstrate that a real stack may be offset and may expose part "
            "of the lower cube's top or side faces.\n"
            "The current scene is shown once in natural color and once with geometry "
            "enhancement. "
            "All current images describe the same instant. Report every scene-state field from the current-scene "
            "images only. Do not copy any reference label automatically.\n"
            "The scene contains exactly two cubes, one bowl, one drawer, one mug, and one mug tree.\n"

            "Determine the cube state in this strict order. FIRST decide cube_count_in_bowl from "
            "direct visual evidence. If cube_count_in_bowl=1, immediately set cubes_stacked=false "
            "and do not evaluate or infer stacking. ONLY if cube_count_in_bowl=0 may you evaluate "
            "whether the cubes are stacked.\n"
            "For cube_count_in_bowl, use the natural-color current-scene image. Do NOT "
            "use geometry-enhanced current-scene images because their contrast enhancement can "
            "exaggerate reflections inside the "
            "bowl. Set cube_count_in_bowl=1 only when a distinct cube body, cube-shaped boundary, "
            "or cube-colored solid region is visibly inside or crossing the physical rim of the "
            "bowl. The white interior, cyan lighting, glare, reflections, shadows, and rim of an "
            "otherwise empty bowl are not a cube. If the bowl interior is empty or evidence of a "
            "cube inside it is ambiguous, set cube_count_in_bowl=0. The scene contains exactly two "
            "cubes, but this fact is context only: NEVER infer that a missing or occluded second cube "
            "must be in the bowl. Two stacked cubes can overlap and look like one tall object.\n"

            "Only after establishing cube_count_in_bowl=0, judge cubes_stacked conservatively. "
            "Here, stacked means vertical support "
            "in 3D: the upper cube's bottom face rests on the lower cube's top face. Position in the "
            "2D image is not physical height. Two cubes appearing one above the other in image pixels, "
            "touching side edges, being close together, or having a horizontal gap/seam does NOT mean "
            "they are stacked.\n"
            "Partial visibility of the lower cube's top or side faces does NOT imply that the cubes "
            "are separate. The STACKED POSITIVE REFERENCES are ground-truth examples of offset, "
            "imperfect stacks and must be treated as cubes_stacked=true. Set cubes_stacked=false "
            "only when both complete cube bodies independently rest on the tabletop at the same "
            "height, as in the UNSTACKED NEGATIVE REFERENCES. A clear tabletop gap and two "
            "independent ground-contact silhouettes are strong evidence for cubes_stacked=false.\n"
            "Set cubes_stacked=true only with clear 3D support and occlusion evidence: the upper cube "
            "is visibly elevated and rests on the lower cube, their projected footprints overlap enough "
            "for physical support, and the lower cube is partially occluded beneath it. Exact center "
            "alignment is NOT required. A color boundary or contact seam alone is insufficient. "
            "The overhead camera can make a real stack resemble one tall combined object, but require "
            "the support/occlusion evidence above and compare directly with both positive and negative "
            "references. If uncertain, choose the reference group that is visually most similar. "
            "For stacking, compare the current scene with its directly matching references. "
            "Geometry-enhanced views only enhance "
            "luminance and edges to make contact boundaries easier to see.\n"

            "For drawer_open, evaluate ONLY the lower movable drawer that has the visible handle. "
            "Ignore the upper handleless panel/compartment completely; it is fixed and must always "
            "be treated as closed, so its appearance must never affect drawer_open.\n"
            "For the drawer judgment, use ONLY the labeled CLOSED drawer reference and the natural-"
            "color current-scene image. Do NOT use "
            "geometry-enhanced current-scene images because "
            "its contrast enhancement exaggerates normal seams and bright edges. Compare the position "
            "of the lower drawer FRONT PANEL, not the handle position. The blue handle always protrudes even when "
            "the drawer is closed. White trim, rails, highlights, shadows, and narrow construction gaps "
            "that are also present in the CLOSED reference are normal closed-state features. They must "
            "never by themselves cause drawer_open=true.\n"
            "Set drawer_open=true only when the natural-color current view provides clear evidence "
            "that the lower drawer front panel has moved substantially outward relative to the CLOSED "
            "reference, producing a "
            "new and substantially wider opening or clearly exposing the drawer interior. If the front "
            "panel position is approximately the same as the CLOSED reference, or if the evidence is "
            "ambiguous, set drawer_open=false.\n"

            "Judge mug_on_tree only from the natural-color current-scene image. "
            "mug_on_tree is true only when the mug is currently hanging on the mug tree.\n"
            "mug_on_tree is false when the mug is not hanging on the mug tree.\n"

            "Do not evaluate tasks and do not return task names.\n"

            "Return only one valid JSON object with exactly these fields:\n"
            "{\"cube_count_in_bowl\": 0, "
            "\"cubes_stacked\": false, "
            "\"drawer_open\": false, "
            "\"mug_on_tree\": false}\n"
        )

        prompt_content = [
            {"type": "input_text", "text": prompt_task},
            {
                "type": "input_text",
                "text": (
                    "DRAWER/MUG REFERENCE (ground truth): drawer_open=false, "
                    "mug_on_tree=true. Ignore this image for cube judgments."
                ),
            },
            {
                "type": "input_image",
                "image_url": (
                    "data:image/jpeg;base64,"
                    f"{self.scene_state_reference}"
                ),
                "detail": "high",
            },
        ]
        for reference in self.stacking_references:
            stacked_value = str(reference["is_stacked"]).lower()
            prompt_content.extend([
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
                    "image_url": (
                        "data:image/jpeg;base64,"
                        f"{reference['image']}"
                    ),
                    "detail": "high",
                },
            ])
        prompt_content.extend([
            {
                "type": "input_text",
                "text": (
                    "CURRENT SCENE IMAGE, natural-color view: "
                    "classify every requested scene-state field from "
                    "this image. This is also the view used by the stacking references."
                ),
            },
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{scene}",
                "detail": "high",
            },
            {
                "type": "input_text",
                "text": (
                    "CURRENT SCENE IMAGE, geometry-enhanced view "
                    "of the exact same frame: use it only to inspect cube support, "
                    "occlusion, boundaries, and stacking. Ignore it for bowl, drawer, "
                    "and mug judgments."
                ),
            },
            {
                "type": "input_image",
                "image_url": (
                    f"data:image/jpeg;base64,{geometry_detail_scene}"
                ),
                "detail": "high",
            },
        ])
        response = self.client.responses.create(
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": prompt_content,
            }],
        )

        output = response.output_text.strip()
        json_start = output.find("{")
        json_end = output.rfind("}")
        if json_start == -1 or json_end == -1:
            raise RuntimeError(f"Scene-state response is not valid JSON: {output}")

        try:
            result = json.loads(output[json_start:json_end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Scene-state response contains invalid JSON: {output}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Scene-state response must be a JSON object: {output}")
        expected_fields = {
            "cube_count_in_bowl",
            "cubes_stacked",
            "drawer_open",
            "mug_on_tree",
        }
        if set(result) != expected_fields:
            raise RuntimeError(
                f"Scene-state response must contain exactly {sorted(expected_fields)}: "
                f"{output}"
            )

        cube_count_in_bowl = result.get("cube_count_in_bowl")
        cubes_stacked = result.get("cubes_stacked")
        drawer_open = result.get("drawer_open")
        mug_on_tree = result.get("mug_on_tree")
        if type(cube_count_in_bowl) is not int or cube_count_in_bowl not in (0, 1):
            raise RuntimeError(
                f"cube_count_in_bowl must be integer 0 or 1: {output}"
            )
        if type(cubes_stacked) is not bool:
            raise RuntimeError(f"cubes_stacked must be a JSON boolean: {output}")
        if type(drawer_open) is not bool:
            raise RuntimeError(f"drawer_open must be a JSON boolean: {output}")
        if type(mug_on_tree) is not bool:
            raise RuntimeError(f"mug_on_tree must be a JSON boolean: {output}")

        # Bowl occupancy has priority over stacking. Enforce the hierarchy in
        # code too, even if the model returns an inconsistent combination.
        if cube_count_in_bowl == 1:
            if cubes_stacked:
                logger.warning(
                    "Ignoring cubes_stacked=true because "
                    "cube_count_in_bowl=1 takes priority"
                )
            cubes_stacked = False

        evaluations = {
            "Put a cube into the bowl": {
                "feasible": cube_count_in_bowl == 0 and not cubes_stacked,
            },
            "Take the cube out of the bowl": {
                "feasible": cube_count_in_bowl == 1 and not cubes_stacked,
            },
            "Stack one cube on the other cube": {
                "feasible": cube_count_in_bowl == 0 and not cubes_stacked,
            },
            "Take the top cube off the other cube": {
                "feasible": cube_count_in_bowl == 0 and cubes_stacked,
            },
            "Open the drawer": {
                "feasible": not drawer_open,
            },
            "Close the drawer": {
                "feasible": drawer_open,
            },
            "Hang the mug on the mug tree": {
                "feasible": not mug_on_tree,
            },
            "Take the mug off the mug tree": {
                "feasible": mug_on_tree,
            },
        }
        feasible_tasks = [
            task
            for task in self.candidate_tasks
            if evaluations.get(task, {}).get("feasible") is True
        ]
        if not feasible_tasks:
            raise RuntimeError(
                "No candidate task is feasible for scene state "
                f"cube_count_in_bowl={cube_count_in_bowl}, "
                f"cubes_stacked={cubes_stacked}, "
                f"drawer_open={drawer_open}, "
                f"mug_on_tree={mug_on_tree}"
            )

        probabilities = self.get_task_sampling_probabilities(feasible_tasks)
        self.selected_task = random.choices(
            feasible_tasks,
            weights=[probabilities[task] for task in feasible_tasks],
            k=1,
        )[0]
        success_rate_items = [
            f"{task}: {self.get_task_success_rate(task):.6f}"
            for task in self.candidate_tasks
        ]
        logger.info(
            _format_aligned_info_items("Task success rates:", success_rate_items)
        )
        scene_state_items = [
            f"cube_count_in_bowl={cube_count_in_bowl}",
            f"cubes_stacked={cubes_stacked}",
            f"drawer_open={drawer_open}",
            f"mug_on_tree={mug_on_tree}",
        ]
        logger.info(
            _format_aligned_info_items("Scene state:", scene_state_items)
        )
        feasible_task_items = [
            f"{task}: {probabilities[task]:.6f}"
            for task in feasible_tasks
        ]
        logger.info(_format_aligned_info_items(
            "Feasible tasks(Sampling probability):", feasible_task_items
        ))

    def reward_generator(
        self,
        current_scene,
        next_scene,
        current_task,
    ):
        prompt_reward = (
            "You are given two images:\n"
            "- BEFORE: the scene before the robot starts the task.\n"
            "- AFTER: the scene after the robot attempted the task.\n\n"
            "The robot was instructed to perform the following task:\n"
            f"{current_task}\n\n"
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

        prompt_content = [{"type": "input_text", "text": prompt_reward}]
        if current_task in {
            "Stack one cube on the other cube",
            "Take the top cube off the other cube",
        }:
            prompt_content.append({
                "type": "input_text",
                "text": (
                    "The following labeled images are ground-truth "
                    "stacking references. Use them only to calibrate whether the "
                    "two cubes are physically stacked."
                ),
            })
            for reference in self.stacking_references:
                stacked_value = str(reference["is_stacked"]).lower()
                prompt_content.extend([
                    {
                        "type": "input_text",
                        "text": (
                            f"{reference['label']} (ground truth): "
                            f"cubes_stacked={stacked_value}."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/jpeg;base64,"
                            f"{reference['image']}"
                        ),
                        "detail": "high",
                    },
                ])

        prompt_content.extend([
            {
                "type": "input_text",
                "text": "BEFORE",
            },
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{current_scene}",
                "detail": "high",
            },
        ])
        prompt_content.extend([
            {
                "type": "input_text",
                "text": "AFTER",
            },
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{next_scene}",
                "detail": "high",
            },
        ])
        response = self.client.responses.create(
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": prompt_content,
            }],
        )

        reward, self.last_reward_reason = _parse_reward_response(
            response.output_text.strip()
        )
        return reward

    # task_0-(reward_0-task_1)-(reward_1-task_2)-...
    def start_task(self, img_rgb):
        """Capture the before-scene for an externally supplied evaluation task."""
        _, scene_gpt, scene_pi, scene_masked = self.process_img(img_rgb)
        self.round += 1
        self.scene_before = self.capture_scene(
            scene_gpt, scene_pi, scene_masked, before=True
        )

    def task_generation(self, img_rgb):  # image shape (1080, 1920, 3), uint8
        _, scene_gpt, scene_pi, scene_masked = self.process_img(img_rgb)
        last_error = None
        for attempt in range(1, 4):
            try:
                self.task_generator(scene_gpt)
                break
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    f"Scene-state recognition attempt {attempt}/3 failed: {exc}"
                )
        else:
            probabilities = self.get_task_sampling_probabilities()
            self.selected_task = random.choices(
                self.candidate_tasks,
                weights=[
                    probabilities[task] for task in self.candidate_tasks
                ],
                k=1,
            )[0]
            logger.error(
                "GPT could not identify a valid scene state after 3 attempts; "
                f"using probability-weighted fallback task "
                f"'{self.selected_task}'. Last error: {last_error}"
            )
            success_rate_items = [
                f"{task}: {self.get_task_success_rate(task):.6f}"
                for task in self.candidate_tasks
            ]
            logger.info(_format_aligned_info_items(
                "Task success rates:", success_rate_items
            ))
            logger.info(
                "Scene state: unavailable after 3 recognition attempts"
            )
            feasible_tasks_log_label = (
                "Feasible tasks(Sampling probability): scene feasibility "
                "unavailable; fallback over all candidates:"
            )
            feasible_task_items = [
                f"{task}: {probabilities[task]:.6f}"
                for task in self.candidate_tasks
            ]
            logger.info(_format_aligned_info_items(
                feasible_tasks_log_label, feasible_task_items
            ))

        self.round += 1
        self.scene_before = self.capture_scene(
            scene_gpt, scene_pi, scene_masked, before=True
        )

        logger.info(f"Selected task to execute: {self.selected_task}")

        return self.selected_task, self.round

    def reward_generation(
        self,
        img_rgb,
        task_prompt=None,
        log_separator=True,
    ):
        _, next_scene_gpt, next_scene_pi, next_scene_masked = self.process_img(
            img_rgb
        )
        self.scene_after = self.capture_scene(
            next_scene_gpt,
            next_scene_pi,
            next_scene_masked,
            before=False,
        )
        if task_prompt ==None:
            task = self.selected_task
        else:
            task = task_prompt
        reward = self.reward_generator(self.scene_before, self.scene_after, task)
        success_rate = self.update_task_success_rate(task, int(reward))
        logger.info(f"Reward: {reward}")
        logger.info(f"Reason: {self.last_reward_reason}")
        logger.info(
            f"Updated task success rate: {task}: {success_rate:.6f}"
        )
        if log_separator:
            logger.info("-------------------------------------------------------------------")


        return reward

# just for test
