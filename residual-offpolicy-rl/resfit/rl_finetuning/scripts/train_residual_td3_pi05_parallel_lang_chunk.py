# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os

# Cap all BLAS/OpenMP threadpools (critical to set before importing numpy/torch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Stop threads from spin-waiting
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import hashlib
import json
import logging
import pprint
import random
import shutil
import time
import dataclasses
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F
import queue
import copy
import hydra
import numpy as np
import tensordict
import torch
import torchrl
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm
import threading
import queue
import wandb
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent_lang import QAgentLang
from resfit.rl_finetuning.off_policy.rl.q_agent_rl_token import QAgentRLToken
from resfit.rl_finetuning.utils.rl_token_pipeline import (
    append_embedding_shard,
    artifact_id,
    bottleneck_id,
    embedding_store_lock,
    encode_all,
    initialize_embedding_store,
    train_or_load_rl_token,
)
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_franka import run_franka_evaluation
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.config.rlpd import (
    LanguageConfig,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.action_diagnostics import (
    save_episode_action_diagnostics,
)
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.task_success_log import (
    apply_initial_eval_outcome_overrides,
    load_initial_eval_outcomes_from_log,
)
from gpt_residual_robot import BasePolicy

class TrainingTimer:
    """Simple timing utility for measuring training stage proportions."""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()

    @contextmanager
    def time(self, stage_name: str):
        """Context manager to time a specific training stage."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[stage_name].append(elapsed)

    def get_timing_stats(self) -> dict[str, float]:
        """Get timing statistics as percentages of total time."""
        if not self.times:
            return {}

        total_time = sum(sum(times) for times in self.times.values())
        if total_time == 0:
            return {}

        stats = {}
        for stage_name, times_list in self.times.items():
            stage_total = sum(times_list)
            stage_avg = stage_total / len(times_list) if times_list else 0
            stage_percentage = (stage_total / total_time) * 100

            stats[f"timing/{stage_name}_percentage"] = stage_percentage
            stats[f"timing/{stage_name}_avg_ms"] = stage_avg * 1000  # Convert to ms
            stats[f"timing/{stage_name}_total_s"] = stage_total

        return stats

    def reset(self):
        """Reset all timing data."""
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()


class EpisodeCheckpointCoordinator:
    """Pause collection at an episode boundary until a checkpoint is durable."""

    def __init__(self):
        self._condition = threading.Condition()
        self._requested_step: int | None = None
        self._completed_step = -1
        self._failure: BaseException | None = None

    def request_and_wait(self, step: int, stop_event: threading.Event) -> bool:
        """Request a learner checkpoint and wait until it has been saved."""
        with self._condition:
            if self._requested_step is not None:
                raise RuntimeError(
                    "A synchronized checkpoint request is already pending at "
                    f"step {self._requested_step}"
                )
            self._requested_step = int(step)
            self._condition.notify_all()
            while (
                self._completed_step < step
                and self._failure is None
                and not stop_event.is_set()
            ):
                self._condition.wait(timeout=0.5)

            if self._failure is not None:
                raise RuntimeError(
                    f"Synchronized checkpoint at step {step} failed"
                ) from self._failure
            return self._completed_step >= step

    def requested_step(self) -> int | None:
        with self._condition:
            return self._requested_step

    def mark_complete(self, step: int) -> None:
        with self._condition:
            if self._requested_step != step:
                raise RuntimeError(
                    "Completed synchronized checkpoint does not match request: "
                    f"requested={self._requested_step}, completed={step}"
                )
            self._completed_step = int(step)
            self._requested_step = None
            self._condition.notify_all()

    def mark_failed(self, exc: BaseException) -> None:
        with self._condition:
            self._failure = exc
            self._condition.notify_all()

    def wake_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()


class LanguageEmbedder:
    """Compute a fixed sentence embedding for a task / prompt string.

    Frozen, lazy-loaded; the result is cached per string so we never re-run
    the LM during data collection.

    Default backend: ``sentence-transformers/all-MiniLM-L6-v2`` -> 384-dim
    embedding.  Swap ``model_name`` for any other HuggingFace sentence /
    text encoder; just keep ``cfg.agent.language.lang_emb_dim`` in sync.

    If the ``sentence-transformers`` package is missing, falls back to a
    deterministic hashed pseudo-embedding so the rest of the training
    pipeline still runs (useful for plumbing checks).
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        device: torch.device,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.emb_dim = emb_dim
        self.device = device
        self.model_name = model_name
        self._cache: dict[str, torch.Tensor] = {}
        self._model = None
        self._loaded = False

    def _maybe_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=str(self.device))
            self._model.eval()
            actual_dim = int(self._model.get_sentence_embedding_dimension())
            assert actual_dim == self.emb_dim, (
                f"LanguageEmbedder: model '{self.model_name}' produces {actual_dim}-dim "
                f"embeddings but cfg.agent.language.lang_emb_dim={self.emb_dim}. "
                f"Either change the model or update the config."
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"⚠️  LanguageEmbedder: could not load '{self.model_name}' ({e}); "
                f"falling back to deterministic hashed pseudo-embeddings. "
                f"Install sentence-transformers for real embeddings."
            )
            self._model = None

    def encode(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text]
        self._maybe_load()
        if self._model is not None:
            with torch.no_grad():
                vec = self._model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                ).to(self.device).float()
        else:
            # Python's built-in hash() is salted independently for every
            # process, so it cannot be used for an embedding that must remain
            # compatible with replay caches and resumed checkpoints. Derive a
            # stable seed from the UTF-8 task text instead.
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (
                2**31 - 1
            )
            g = torch.Generator(device="cpu").manual_seed(seed)
            vec = torch.randn(self.emb_dim, generator=g)
            vec = (vec / vec.norm()).to(self.device).float()
        self._cache[text] = vec
        return vec

    def __call__(self, text: str) -> torch.Tensor:
        return self.encode(text)

def _load_lerobot_task_prompts(dataset: LeRobotDataset, dataset_name: str) -> dict[int, str]:
    """Read ``meta/tasks.jsonl`` and return task_index -> prompt."""
    candidate_paths: list[Path] = []

    dataset_root = getattr(dataset, "root", None)
    if dataset_root is not None:
        candidate_paths.append(Path(dataset_root) / "meta" / "tasks.jsonl")

    dataset_meta_root = getattr(getattr(dataset, "meta", None), "root", None)
    if dataset_meta_root is not None:
        candidate_paths.append(Path(dataset_meta_root) / "tasks.jsonl")
        candidate_paths.append(Path(dataset_meta_root) / "meta" / "tasks.jsonl")

    name_path = Path(dataset_name).expanduser()
    candidate_paths.append(name_path / "meta" / "tasks.jsonl")
    candidate_paths.append(name_path / "tasks.jsonl")

    tasks_path = next((path for path in candidate_paths if path.exists()), None)
    if tasks_path is None:
        searched = "\n  ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Could not find LeRobot tasks.jsonl. Searched:\n  {searched}")

    task_prompts: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task_index" not in row or "task" not in row:
                raise KeyError(f"{tasks_path}:{line_no} must contain 'task_index' and 'task'")
            task_prompts[int(row["task_index"])] = str(row["task"])

    if not task_prompts:
        raise ValueError(f"No task prompts found in {tasks_path}")

    print(f"Loaded {len(task_prompts)} task prompts from {tasks_path}")
    return task_prompts

def _inject_task_emb(
    obs: dict[str, torch.Tensor], task_emb: torch.Tensor, key: str = "observation.task_emb"
) -> dict[str, torch.Tensor]:
    """Set ``obs[key]`` to ``task_emb`` (broadcast to whatever batch dim the
    rest of obs has).  Mutates ``obs`` in place and returns it for chaining.
    """
    if key in obs:
        return obs
    state = obs.get("observation.state", None)
    if state is not None and state.dim() == 2:
        emb = task_emb.unsqueeze(0).expand(state.size(0), -1).contiguous()
    else:
        emb = task_emb
    obs[key] = emb.to(state.device if state is not None else task_emb.device)
    return obs


def _inject_task_id(
    obs: dict[str, torch.Tensor], task_id: int, key: str = "observation.task_id"
) -> dict[str, torch.Tensor]:
    """Attach the discrete routing id using the observation's batch/device."""
    if obs is None:
        return obs
    state = obs.get("observation.state")
    device = state.device if isinstance(state, torch.Tensor) else None
    if isinstance(state, torch.Tensor) and state.ndim == 2:
        value = torch.full(
            (state.shape[0],), int(task_id), dtype=torch.long, device=device
        )
    else:
        value = torch.tensor(int(task_id), dtype=torch.long, device=device)
    obs[key] = value
    return obs


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _config_to_container(cfg) -> dict:
    """Return a plain config dict for both Hydra configs and dataclasses."""
    if OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True)
    elif dataclasses.is_dataclass(cfg):
        container = dataclasses.asdict(cfg)
    else:
        raise TypeError(f"Unsupported config type: {type(cfg).__name__}")
    if not isinstance(container, dict):
        raise TypeError(f"Expected config mapping, got {type(container).__name__}")
    return container


def _make_agent(experiment_cfg, **kwargs):
    if getattr(experiment_cfg, "rl_token", None) is not None and experiment_cfg.rl_token.enabled:
        return QAgentRLToken(
            **kwargs,
            token_dim=experiment_cfg.rl_token.token_dim,
            token_obs_key=experiment_cfg.rl_token.obs_key,
        )
    return QAgentLang(**kwargs)

OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
ONLINE_HF_REPO = os.environ.get("HF_ONLINE_BUFFER_REPO", None)

if OFFLINE_HF_REPO is not None:
    logger.info(f"Using offline buffer from {OFFLINE_HF_REPO}")
if ONLINE_HF_REPO is not None:
    logger.info(f"Using online buffer from {ONLINE_HF_REPO}")

# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()

# Dedicated sub-folders for the different cache types -----------------------
OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_buffer_cache"
ONLINE_CACHE_DIR = _CACHE_ROOT / "online_buffer_cache"


def process_image_batch(obs_dict, image_keys, enc_type= "vit", rb=False):
    """Resize images to CHW tensors, adding a batch dimension outside replay storage."""
    # RL tokens are already compact features, not pixels.
    if image_keys and all(k in obs_dict and obs_dict[k].ndim <= 2 for k in image_keys):
        return obs_dict
    imgs = []

    for k in image_keys:
        x = obs_dict[k]

        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise ValueError(f"{k} expected 3 dims [H,W,C], got shape={x.shape}")

        # HWC[224,224,3] -> CHW[3,224,224]
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        elif x.shape[0] == 3:
            pass
        else:
            raise ValueError(f"{k} is neither HWC nor CHW, got shape={x.shape}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        imgs.append(x)

    # [N,3,H,W]
    imgs = torch.stack(imgs, dim=0)

    out_size = 224 if enc_type == "siglip" else 84
    if imgs.shape[-2] != out_size or imgs.shape[-1] != out_size:
        imgs = F.interpolate(
            imgs,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )

    if rb:
        for i, k in enumerate(image_keys):
            obs_dict[k] = imgs[i].contiguous()
    else:  # train
        for i, k in enumerate(image_keys):
            obs_dict[k] = imgs[i].contiguous().unsqueeze(0)

    return obs_dict

def save_training_checkpoint(
    ckpt_path: Path,
    agent,
    online_rb,
    global_step: int,
    best_eval_success_rate: float,
    training_cum_time: float,
    episode_count: int,
    actor_updates: int,
    run_name: str,
    task_output_root: str,
    task_success_stats: dict,
    cfg,
    save_replay_buffer: bool = False,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    replay_size = len(online_rb)
    replay_dir = ckpt_path.parent / f"{ckpt_path.stem}_replay" / "online_rb"

    # A training checkpoint and its online replay data describe one logical
    # state.  Persist the replay data first so a model checkpoint is never
    # published while its matching buffer is still missing.  The generic
    # online cache may lag behind by save_online_rb_interval and therefore is
    # not safe for exact resume.
    if save_replay_buffer:
        if replay_dir.exists():
            shutil.rmtree(replay_dir)
        optimized_replay_buffer_dumps(online_rb, replay_dir)

    checkpoint = {
        "global_step": global_step,
        # Full agent state_dict captures every registered submodule:
        # encoders, actor, critic, actor_target, critic_target (and any new
        # submodule added later, e.g. lang_encoder).  This is the key fix
        # for the "load -> success_rate=0" bug: previously `encoders` was
        # never saved, so after loading we evaluated trained actor/critic
        # against a *freshly initialized* image encoder.
        "agent": agent.state_dict(),

        # Per-component dicts kept for backwards compatibility with eval
        # scripts that read them directly.
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "encoders": agent.encoders.state_dict(),
        "lang_encoder": agent.lang_encoder.state_dict() if agent.lang_encoder is not None else None,
        "actor_target": agent.actor_target.state_dict(),
        "critic_target": agent.critic_target.state_dict(),

        # All optimizers (encoder_opt was previously missing).
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),

        # LR schedulers (None-safe).
        "encoder_scheduler": (
            agent.encoder_scheduler.state_dict() if agent.encoder_scheduler is not None else None
        ),
        "critic_scheduler": (
            agent.critic_scheduler.state_dict() if agent.critic_scheduler is not None else None
        ),
        "actor_scheduler": (
            agent.actor_scheduler.state_dict() if agent.actor_scheduler is not None else None
        ),

        "best_eval_success_rate": best_eval_success_rate,
        "training_cum_time": training_cum_time,
        "episode_count": episode_count,
        "actor_updates": actor_updates,
        "run_name": run_name,
        "task_output_root": task_output_root,
        "task_success_stats": copy.deepcopy(task_success_stats),
        "online_replay_size": replay_size,
        "online_replay_dir": (
            str(replay_dir.relative_to(ckpt_path.parent))
            if save_replay_buffer
            else None
        ),
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        "cfg": _config_to_container(cfg),

        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    if save_replay_buffer:
        print(
            f"Saved matching online replay buffer to {replay_dir} "
            f"(size={replay_size} chunks)"
        )


def load_training_checkpoint(
    ckpt_path: str | Path,
    agent,
    online_rb,
    device: torch.device,
    load_replay_buffer: bool = False,
):
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if load_replay_buffer:
        replay_dir_value = checkpoint.get("online_replay_dir")
        if replay_dir_value is None:
            raise RuntimeError(
                "Checkpoint has no recorded online_replay_dir. Strict resume "
                "requires a checkpoint created by the replay-on-checkpoint "
                "code; generic caches and manually staged sidecars are not "
                "accepted."
            )
        replay_dir = ckpt_path.parent / replay_dir_value
        if not replay_dir.exists():
            raise FileNotFoundError(
                f"Matching online replay buffer not found: {replay_dir}"
            )
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, replay_dir)
        expected_replay_size = checkpoint.get("online_replay_size")
        if expected_replay_size is not None and len(online_rb) != expected_replay_size:
            raise RuntimeError(
                "Online replay buffer size mismatch for checkpoint "
                f"{ckpt_path}: expected {expected_replay_size}, loaded {len(online_rb)}"
            )
        print(
            f"Loaded matching online replay buffer from {replay_dir} "
            f"(size={len(online_rb)} chunks)"
        )

    # Prefer the full-agent state_dict if present (new format); this restores
    # encoders, actor, critic, actor_target, critic_target and any future
    # submodule in one shot.  Fall back to the old per-component layout for
    # backwards compatibility with checkpoints that pre-date this fix.
    loaded_modules: list[str] = []
    if "agent" in checkpoint:
        if (
            getattr(agent, "task_specific_actor", False)
            and not any(
                key.startswith("actor.actors.")
                for key in checkpoint["agent"]
            )
        ):
            raise RuntimeError(
                "This checkpoint contains one shared actor and cannot be "
                "resumed as an exact task-specific-actor run. Start a fresh run "
                "so the new task-id replay schema and optimizer state are built."
            )
        missing, unexpected = agent.load_state_dict(
            checkpoint["agent"],
            strict=getattr(agent, "task_specific_actor", False),
        )
        loaded_modules.append("agent (full)")
        if missing:
            print(f"  [load] missing keys (will stay at __init__ values): {len(missing)}")
        if unexpected:
            print(f"  [load] unexpected keys (ignored): {len(unexpected)}")
    else:
        if (
            getattr(agent, "task_specific_actor", False)
            and not any(key.startswith("actors.") for key in checkpoint["actor"])
        ):
            raise RuntimeError(
                "This legacy checkpoint contains one shared actor and cannot "
                "be exactly resumed with task-specific actors."
            )
        agent.actor.load_state_dict(checkpoint["actor"])
        loaded_modules.append("actor")
        agent.critic.load_state_dict(checkpoint["critic"])
        loaded_modules.append("critic")
        if "encoders" in checkpoint:
            agent.encoders.load_state_dict(checkpoint["encoders"])
            loaded_modules.append("encoders")
        else:
            print(
                "  ⚠️  Old checkpoint format: 'encoders' not found.  The image "
                "encoder will stay at __init__ (random) values, which is the "
                "exact reason eval success rate drops to 0%.  Re-train or "
                "continue training and re-save."
            )
        if "actor_target" in checkpoint:
            agent.actor_target.load_state_dict(checkpoint["actor_target"])
            loaded_modules.append("actor_target")
        else:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            loaded_modules.append("actor_target<-actor")
        if "critic_target" in checkpoint:
            agent.critic_target.load_state_dict(checkpoint["critic_target"])
            loaded_modules.append("critic_target")
        else:
            agent.critic_target.load_state_dict(agent.critic.state_dict())
            loaded_modules.append("critic_target<-critic")

    # ---------- Optimizers (continue-training only; safe to skip for eval) -
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "critic_opt" in checkpoint:
        agent.critic_opt.load_state_dict(checkpoint["critic_opt"])
    if "encoder_opt" in checkpoint:
        agent.encoder_opt.load_state_dict(checkpoint["encoder_opt"])

    for name in ("encoder_scheduler", "critic_scheduler", "actor_scheduler"):
        sched = getattr(agent, name, None)
        if sched is not None and checkpoint.get(name) is not None:
            sched.load_state_dict(checkpoint[name])

    print(f"  [load] restored: {', '.join(loaded_modules)}")

    global_step = checkpoint.get("global_step", 0)
    best_eval_success_rate = checkpoint.get("best_eval_success_rate", 0.0)
    training_cum_time = checkpoint.get("training_cum_time", 0.0)
    episode_count = checkpoint.get("episode_count", 0)
    actor_updates = checkpoint.get("actor_updates", 0)
    run_name = checkpoint.get("run_name", None)
    wandb_run_id = checkpoint.get("wandb_run_id", None)
    task_output_root = checkpoint.get("task_output_root", None)
    task_success_stats = checkpoint.get("task_success_stats", None)

    # Restore RNG states
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint and checkpoint["torch_random_state"] is not None:
        torch_rng_state = checkpoint["torch_random_state"]
        if not isinstance(torch_rng_state, torch.Tensor):
            torch_rng_state = torch.tensor(torch_rng_state, dtype=torch.uint8)
        else:
            torch_rng_state = torch_rng_state.to(dtype=torch.uint8, device="cpu")
        torch.set_rng_state(torch_rng_state)
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        cuda_rng_state = checkpoint["cuda_random_state"]
        fixed_cuda_states = []
        for s in cuda_rng_state:
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.uint8)
            else:
                s = s.to(dtype=torch.uint8, device="cpu")
            fixed_cuda_states.append(s)
        torch.cuda.set_rng_state_all(fixed_cuda_states)


    print(f"Loaded checkpoint from {ckpt_path}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return {
        "global_step": global_step,
        "best_eval_success_rate": best_eval_success_rate,
        "training_cum_time": training_cum_time,
        "episode_count": episode_count,
        "actor_updates": actor_updates,
        "run_name": run_name,
        "wandb_run_id": wandb_run_id,
        "task_output_root": task_output_root,
        "task_success_stats": task_success_stats,
    }


def _backfill_initial_eval_outcomes(
    saved_stats: dict,
    *,
    task_output_root: str | Path | None,
    candidate_tasks: list[str],
    window_size: int,
) -> tuple[dict, Path | None]:
    """Migrate scalar-baseline checkpoints using saved step-0 eval outcomes."""
    if all(
        saved_stats.get(task, {}).get("initial_eval_outcomes") is not None
        for task in candidate_tasks
    ):
        return saved_stats, None
    if task_output_root is None:
        return saved_stats, None

    progress_path = Path(task_output_root) / "step_0" / "evaluation_progress.json"
    if not progress_path.is_file():
        return saved_stats, None

    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    if progress.get("status") != "complete":
        return saved_stats, None
    if int(progress.get("eval_num_episode", -1)) != int(window_size):
        return saved_stats, None
    if progress.get("candidate_tasks") != candidate_tasks:
        return saved_stats, None

    outcomes_by_task = progress.get("successes_by_task", {})
    migrated = copy.deepcopy(saved_stats)
    for task in candidate_tasks:
        stats = migrated.get(task)
        raw_outcomes = outcomes_by_task.get(task)
        if stats is None or raw_outcomes is None or "initial_success_rate" not in stats:
            return saved_stats, None
        if len(raw_outcomes) != window_size or any(
            value not in (0, 1) for value in raw_outcomes
        ):
            return saved_stats, None
        outcomes = [int(value) for value in raw_outcomes]
        eval_rate = sum(outcomes) / window_size
        if abs(eval_rate - float(stats["initial_success_rate"])) > 1e-9:
            return saved_stats, None
        stats["initial_eval_outcomes"] = outcomes
    return migrated, progress_path


def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
    enc_type: str,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        curr_obs_i = process_image_batch(curr_obs_i, image_keys, enc_type, rb=True)
        next_obs_i = process_image_batch(next_obs_i, image_keys, enc_type, rb=True)
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        prev_residual_last = info.get("prev_residual_last")
        has_prev_residual = bool(info.get("has_prev_residual", False))
        if prev_residual_last is None:
            residual = torch.as_tensor(info["residual_action"], device=device)
            prev_residual_last = torch.zeros_like(residual.reshape(-1, residual.shape[-1])[-1])
        else:
            prev_residual_last = torch.as_tensor(
                prev_residual_last, dtype=torch.float32, device=device
            ).reshape(-1)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "prev_residual_last": prev_residual_last,
                "has_prev_residual": torch.tensor(
                    has_prev_residual, dtype=torch.bool, device=device
                ),
                "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
            },
            batch_size=[],
        ).unsqueeze(0)

        online_rb.add(td)


def collector_loop(
        cfg,
        lang_cfg,
        lang_embedder,
        device,
        base_policy,
        image_keys,
        enc_type,
        episode_queue,
        weights_queue,
        stop_event,
        checkpoint_sync,
        training_timer,
        run_name,
        outputs_dir,
        img_c,
        img_h,
        img_w,
        lowdim_dim,
        action_dim,
        initial_global_step=0,
        initial_episode_count=0,
        initial_best_eval_success_rate=-float("inf"),
        discard_initial_previous_episode=False,
    ):
        global_step = initial_global_step
        if (
            getattr(cfg, "skip_completed_eval_on_resume", False)
            and initial_global_step > 0
        ):
            global_step += cfg.send_transitions_len * cfg.chunk_len
            print(
                "[collector] periodic evaluation at resumed step "
                f"{initial_global_step} was already completed; continuing "
                f"collection at step {global_step}"
            )
        episode_count = initial_episode_count
        episode_idx = int(initial_episode_count)
        best_eval_success_rate = initial_best_eval_success_rate

        task_names = list(base_policy.task_reward_generator.candidate_tasks)
        task_to_id = {task: index for index, task in enumerate(task_names)}

        def _print_eval_success_rates(eval_metrics):
            for index, task in enumerate(task_names, start=1):
                success_rate = eval_metrics[f"eval/task_{index}/success_rate"]
                print(f"🎉 task {index} ({task}) success rate: {success_rate}")

        def _task_metrics(completed_task):
            """Build probability and rolling-rate metrics for one completed task."""
            probabilities = (
                base_policy.task_reward_generator.get_task_sampling_probabilities(
                    task_names
                )
            )

            task_metrics = {}
            for index, task in enumerate(task_names):
                task_metrics[f"tasks/task_{index}_probability"] = probabilities[task]
                if task == completed_task:
                    task_episode_count = (
                        base_policy.task_reward_generator.get_task_attempts(task)
                    )
                    task_metrics[f"tasks/task_{index}_success_rate"] = (
                        base_policy.task_reward_generator.get_task_success_rate(task)
                    )
                    task_metrics[f"tasks/task_{index}_episode_count"] = (
                        task_episode_count
                    )
                    # Keep the existing diagnostic metric for compatibility.
                    task_metrics[f"tasks/task_{index}_attempts"] = task_episode_count
            print(
                "[collector] task success stats: "
                f"{base_policy.task_reward_generator.task_success_stats}"
            )
            return task_metrics

        def _update_task_probabilities(completed_task):
            """Return metrics for the rate already updated by TaskRewardGenerator."""
            if completed_task not in base_policy.task_reward_generator.task_success_stats:
                logging.warning("Unknown generated task; not logging stats: %s", completed_task)
                return {}
            return _task_metrics(completed_task)

        def _attach_task_context(obs, task_prompt, task_emb=None):
            if obs is None:
                return obs
            if task_prompt not in task_to_id:
                raise KeyError(f"Unknown task prompt for routing: {task_prompt!r}")
            if lang_cfg.enabled:
                if task_emb is None:
                    raise RuntimeError(
                        "task_emb is required when language conditioning is enabled"
                    )
                _inject_task_emb(obs, task_emb, key=lang_cfg.lang_emb_obs_key)
            if cfg.agent.task_specific_actor:
                _inject_task_id(
                    obs, task_to_id[task_prompt], key=cfg.agent.task_id_obs_key
                )
            return obs
        initial_eval_completed = False
        run_initial_evaluation = bool(
            getattr(cfg, "eval_enabled", True)
            and cfg.eval_first
            and not getattr(cfg, "initial_task_success_log", None)
            and getattr(cfg, "initial_task_success_window_overrides", None) is None
            and global_step == 0
        )
        if not getattr(cfg, "eval_enabled", True):
            print("[collector] robot evaluation is disabled for this run")
        previous_residual_last = None
        save_episode_action_plots = bool(
            getattr(cfg, "save_episode_action_plots", False)
        )
        action_diagnostics_dir = Path(outputs_dir) / "action_diagnostics" / "training"
        diagnostic_base_actions = []
        diagnostic_residual_actions = []
        diagnostic_combined_actions = []
        diagnostic_start_global_step = None
        if run_initial_evaluation:
            # Initial evaluation performs its own task-specific resets. Avoid a
            # normal reset here, which would generate an unused random task.
            obs = None
            task_prompt = None
            task_emb = None
        else:
            base_policy.task_reward_generator.set_output_training(global_step)
            # A freshly collected replay warmup may end with an unfinished
            # random-policy episode.  Do not score that episode as the first
            # learned-policy result.  Cached warmup/replay and resumed runs do
            # not leave such a pending robot episode in this process.
            obs, task_prompt = base_policy.reset(
                evaluate_previous=not discard_initial_previous_episode
            )
            task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
            obs = _attach_task_context(obs, task_prompt, task_emb)
        agent = _make_agent(cfg, obs_shape=(img_c, img_h, img_w),
                            prop_shape=(lowdim_dim,),
                            action_dim=action_dim,
                            rl_cameras=image_keys,
                            cfg=cfg.agent,
                            residual_actor=True,  # Enable residual actor mode
                        ) 
        checkpoint_interval = int(cfg.checkpoint_interval)
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        next_checkpoint_threshold = (
            global_step // checkpoint_interval + 1
        ) * checkpoint_interval

        def _try_load_latest_weights():
            latest = None
            while True:
                try:
                    latest = weights_queue.get_nowait()
                except queue.Empty:
                    break

            if latest is not None:
                agent.actor.load_state_dict(latest["actor"])
                agent.encoders.load_state_dict(latest["encoders"])
                agent.critic.load_state_dict(latest["critic"])
                if agent.lang_encoder is not None and latest["lang_encoder"] is not None:
                    agent.lang_encoder.load_state_dict(latest["lang_encoder"])


        while (global_step <= cfg.algo.total_timesteps) and (not stop_event.is_set()):
            _try_load_latest_weights()

            episode_steps = []
            episode_done = False
            episode_task_prompt = task_prompt
            episode_wandb_metrics = {}

            while not episode_done and global_step <= cfg.algo.total_timesteps and (not stop_event.is_set()) and len(episode_steps)<cfg.send_transitions_len:
                if run_initial_evaluation and not initial_eval_completed:
                    pre_eval_task_success_stats = copy.deepcopy(
                        base_policy.task_reward_generator.task_success_stats
                    )
                    try:
                        with training_timer.time("evaluation"):
                            eval_metrics = run_franka_evaluation(
                                env=base_policy,  # todo eval?
                                agent=agent,
                                eval_num_episode=cfg.task_success_window_size,
                                device=device,
                                global_step=global_step,
                                save_video=cfg.save_video,
                                save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                                run_name=run_name,
                                output_dir=outputs_dir,
                                lang_cfg = lang_cfg,
                                lang_embedder= lang_embedder,
                                chunk_len=cfg.chunk_len,
                                initialize_task_success_metrics=True,
                            )

                            _print_eval_success_rates(eval_metrics)
                    finally:
                        base_policy.task_reward_generator.restore_task_success_stats(
                            pre_eval_task_success_stats
                        )
                    initial_eval_outcomes = eval_metrics["_eval_outcomes_by_task"]
                    base_policy.task_reward_generator.initialize_task_success_windows(
                        initial_eval_outcomes
                    )
                    print(
                        "[collector] initialized task success-rate rolling windows "
                        f"with the ordered outcomes from "
                        f"{cfg.task_success_window_size} evaluation episodes per task"
                    )
                    obs, task_prompt = base_policy.reset(evaluate_previous=False)
                    task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
                    obs = _attach_task_context(obs, task_prompt, task_emb)
                    episode_task_prompt = task_prompt
                    initial_eval_completed = True
                    previous_residual_last = None

                with torch.no_grad(), utils.eval_mode(agent):
                    stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                    print("stddev:", stddev)
                    # Attach the base-action chunk (H*dim) so actor/critic see chunk-level base.
                    obs["observation.base_action"] = base_policy.current_base_chunk(cfg.chunk_len)
                    obs_act = process_image_batch(copy.deepcopy(obs), image_keys, enc_type, rb=False)
                    action = agent.act(obs_act, eval_mode=False, stddev=stddev, cpu=False)  # (1, H*dim)

                if cfg.algo.progressive_clipping_steps > 0:
                    clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                    action = action * clip_factor

                # Open-loop execute the whole residual chunk; store ONE chunk macro-transition.
                next_obs, combined_chunk, reward, done, info = base_policy.step_chunk(action.reshape(1, -1))

                # BasePolicy returns a per-step base action (m,), but a chunk
                # transition requires next_obs to contain the next H*m chunk.
                next_obs["observation.base_action"] = base_policy.current_base_chunk(cfg.chunk_len)

                # The replayed transition carries the actually issued previous
                # residual boundary. Episode/task resets deliberately break the
                # chain so the first chunk is excluded from the boundary loss.
                info = dict(info)
                info["has_prev_residual"] = previous_residual_last is not None
                info["prev_residual_last"] = previous_residual_last
                current_residual = info["residual_action"].detach().reshape(
                    -1, info["residual_action"].shape[-1]
                )
                previous_residual_last = current_residual[-1].clone()

                if save_episode_action_plots:
                    if diagnostic_start_global_step is None:
                        diagnostic_start_global_step = global_step
                    diagnostic_base_actions.append(
                        info["executed_base_action"].detach().cpu().numpy()
                    )
                    diagnostic_residual_actions.append(
                        info["executed_residual_action"].detach().cpu().numpy()
                    )
                    diagnostic_combined_actions.append(
                        info["executed_combined_action"].detach().cpu().numpy()
                    )

                if done:
                    task_prompt = info["task_prompt"]
                    task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
                next_obs = _attach_task_context(next_obs, task_prompt, task_emb)
                if done.any():
                    episode_count += done.float().sum().item()
                    episode_done = True

                    terminal_reward = float(reward.sum().item())
                    episode_wandb_metrics.update(
                        _update_task_probabilities(
                            completed_task=episode_task_prompt,
                        )
                    )
                    previous_residual_last = None

                    # Reward is sparse/terminal -> only log it at episode end.
                    _log = {"training/reward": terminal_reward}
                    if getattr(agent, "last_rollout_gate_adopt_rate", None) is not None:
                        _log["rollout/gate_adopt_rate"] = agent.last_rollout_gate_adopt_rate
                    episode_wandb_metrics.update(_log)

                    if save_episode_action_plots:
                        try:
                            base_actions = np.concatenate(
                                diagnostic_base_actions, axis=0
                            )
                            residual_actions = np.concatenate(
                                diagnostic_residual_actions, axis=0
                            )
                            combined_actions = np.concatenate(
                                diagnostic_combined_actions, axis=0
                            )
                            executed_episode_steps = len(base_actions)
                            png_path, csv_path = save_episode_action_diagnostics(
                                output_dir=action_diagnostics_dir,
                                episode_number=int(episode_count),
                                task_prompt=episode_task_prompt,
                                start_global_step=int(diagnostic_start_global_step),
                                end_global_step=(
                                    int(diagnostic_start_global_step)
                                    + executed_episode_steps
                                    - 1
                                ),
                                base_actions=base_actions,
                                residual_actions=residual_actions,
                                combined_actions=combined_actions,
                            )
                            print(
                                "[collector] saved episode action diagnostics: "
                                f"{png_path} and {csv_path}"
                            )
                        except Exception:
                            logging.exception(
                                "Failed to save action diagnostics for training "
                                "episode %s",
                                int(episode_count),
                            )
                        finally:
                            diagnostic_base_actions.clear()
                            diagnostic_residual_actions.clear()
                            diagnostic_combined_actions.clear()
                            diagnostic_start_global_step = None

                step_payload = {
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": combined_chunk,  # combined action chunk (1, H*dim)
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
                episode_steps.append(step_payload)
                obs = next_obs

            # Count completed environment steps before publishing the payload.
            # This makes checkpoint/global_step mean "steps already processed",
            # so a resumed collector starts with the following step.
            global_step += len(episode_steps) * cfg.chunk_len

            ep_payload = {
                "episode_idx": episode_idx,
                "global_step_after_episode": global_step,
                "episode_count": episode_count,
                "steps": episode_steps,
                "wandb_metrics": episode_wandb_metrics,
            }


            episode_queue.put(ep_payload)
            print(f"[collector] pushed episode {episode_idx}, len={len(episode_steps)}, global_step={global_step}")

            episode_idx += 1
            if episode_done and global_step >= next_checkpoint_threshold:
                crossed_threshold = next_checkpoint_threshold
                print(
                    "[collector] checkpoint threshold "
                    f"{crossed_threshold} reached; episode ended at step "
                    f"{global_step}. Waiting for learner alignment and save."
                )
                checkpoint_saved = checkpoint_sync.request_and_wait(
                    global_step, stop_event
                )
                if not checkpoint_saved:
                    break
                print(
                    "[collector] synchronized checkpoint saved at step "
                    f"{global_step}; collection continuing"
                )
                next_checkpoint_threshold = (
                    global_step // checkpoint_interval + 1
                ) * checkpoint_interval

            periodic_eval_due = (
                getattr(cfg, "eval_enabled", True)
                and global_step > 0
                and global_step % cfg.eval_interval_every_steps == 0
            )
            if periodic_eval_due:
                pre_eval_task_success_stats = copy.deepcopy(
                    base_policy.task_reward_generator.task_success_stats
                )
                try:
                    with training_timer.time("evaluation"):
                        eval_metrics = run_franka_evaluation(
                            env=base_policy,  # todo eval?
                            agent=agent,
                            eval_num_episode=cfg.eval_num_episodes,
                            device=device,
                            global_step=global_step,
                            save_video=cfg.save_video,
                            save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                            run_name=run_name,
                            output_dir=outputs_dir,
                            lang_cfg = lang_cfg,
                            lang_embedder= lang_embedder,
                            chunk_len=cfg.chunk_len,
                        )

                        _print_eval_success_rates(eval_metrics)
                finally:
                    base_policy.task_reward_generator.restore_task_success_stats(
                        pre_eval_task_success_stats
                    )
                obs, task_prompt = base_policy.reset(evaluate_previous=False)
                task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
                obs = _attach_task_context(obs, task_prompt, task_emb)
                previous_residual_last = None

        stop_event.set()
        checkpoint_sync.wake_waiters()
        print("[collector] finished")


def learner_loop(
        cfg,
        device,
        agent,
        online_rb,
        offline_rb,
        image_keys,
        lowdim_keys,
        enc_type,
        episode_queue,
        weights_queue,
        stop_event,
        checkpoint_sync,
        model_save_dir,
        run_name,
        task_output_root,
        task_success_stats,
        online_cache_dir,
        online_cache_meta,
        online_cache_hash,
        initial_global_step=0,
        initial_actor_updates=0,
        initial_episode_count=0,
        initial_best_eval_success_rate=0.0,
        initial_training_cum_time=0.0,

    ):
        training_timer = TrainingTimer()

        global_step = initial_global_step
        actor_updates = initial_actor_updates
        episode_count = initial_episode_count
        best_eval_success_rate = initial_best_eval_success_rate
        training_cum_time = initial_training_cum_time
        train_start_time = time.time()

        metrics = {}

        online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
        offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)
        next_persist_threshold = len(online_rb) + cfg.save_online_rb_interval

        weights_queue.put({
            "actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
            "encoders": {k: v.detach().cpu() for k, v in agent.encoders.state_dict().items()},
            "critic": {k: v.detach().cpu() for k, v in agent.critic.state_dict().items()},
            "lang_encoder": ({k: v.detach().cpu() for k, v in agent.lang_encoder.state_dict().items()}
                             if agent.lang_encoder is not None else None)
        })

        def _publish_latest_actor():
            payload = {
                "actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
                "encoders": {k: v.detach().cpu() for k, v in agent.encoders.state_dict().items()},
                "critic": {k: v.detach().cpu() for k, v in agent.critic.state_dict().items()},
                "lang_encoder": ({k: v.detach().cpu() for k, v in agent.lang_encoder.state_dict().items()}
                                 if agent.lang_encoder is not None else None)
            }
            try:
                while True:
                    weights_queue.get_nowait()
            except Exception:
                pass
            weights_queue.put(payload)

        def _save_requested_checkpoint() -> bool:
            """Save only after learner and collector reach the same boundary."""
            requested_step = checkpoint_sync.requested_step()
            if requested_step is None or global_step < requested_step:
                return False
            if not episode_queue.empty():
                return False
            if global_step != requested_step:
                error = RuntimeError(
                    "Learner passed the synchronized checkpoint boundary: "
                    f"requested={requested_step}, learner={global_step}"
                )
                checkpoint_sync.mark_failed(error)
                stop_event.set()
                raise error

            ckpt_path = model_save_dir / f"checkpoint_{requested_step}.pt"
            try:
                save_training_checkpoint(
                    ckpt_path=ckpt_path,
                    agent=agent,
                    online_rb=online_rb,
                    global_step=requested_step,
                    best_eval_success_rate=best_eval_success_rate,
                    training_cum_time=training_cum_time,
                    episode_count=episode_count,
                    actor_updates=actor_updates,
                    run_name=run_name,
                    task_output_root=task_output_root,
                    task_success_stats=task_success_stats,
                    cfg=cfg,
                    save_replay_buffer=cfg.save_replay_on_checkpoint,
                )
            except BaseException as exc:
                checkpoint_sync.mark_failed(exc)
                stop_event.set()
                raise

            checkpoint_sync.mark_complete(requested_step)
            print(
                "[learner] collector/learner aligned and checkpoint saved at "
                f"episode boundary step {requested_step}"
            )
            return True

        len_rb = len(online_rb)
        while (not stop_event.is_set()) or (not episode_queue.empty()):
            iter_start = time.time()

            if _save_requested_checkpoint():
                continue

            got_episode = False
            try:
                ep_payload = episode_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            got_episode = True
            episode_count = ep_payload["episode_count"]
            global_step = max(global_step, ep_payload["global_step_after_episode"])
            wandb_step_metrics = dict(ep_payload.get("wandb_metrics") or {})

            steps = ep_payload["steps"]
            for st in steps:
                _add_transitions_to_buffer(
                    obs=st["obs"],
                    next_obs=st["next_obs"],
                    actions=st["action"].to(device),
                    reward=st["reward"].to(device),
                    done=st["done"].to(device),
                    info=st["info"],
                    device=device,
                    image_keys=image_keys,
                    lowdim_keys=lowdim_keys,
                    num_envs=cfg.num_envs,
                    online_rb=online_rb,
                    enc_type=enc_type,
                )
                len_rb+=1

            if len_rb >= next_persist_threshold:
                online_cache_dir.mkdir(parents=True, exist_ok=True)
                optimized_replay_buffer_dumps(online_rb, online_cache_dir)
                with open(online_cache_dir / "user_metadata.json", "w") as f:
                    json.dump(online_cache_meta, f, indent=2)
                if ONLINE_HF_REPO is not None:
                    _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
                print(f"Update online rb. Online buffer size = {len(online_rb)} transitions")

                if ONLINE_HF_REPO is not None:
                    _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)

                next_persist_threshold += cfg.save_online_rb_interval


            if not got_episode:
                time.sleep(0.01)
                continue

            # Checkpoints do not currently embed the online replay buffer. If
            # its separate cache is empty on resume, wait for the collector to
            # add data instead of sampling empty TorchRL storage.
            if online_batch_size > 0 and len(online_rb) == 0:
                print(
                    "[learner] online replay buffer is empty; waiting for "
                    "collector transitions before updating."
                )
                continue
            # (B) Updates
            if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:
                i = 0
                actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration

                while i < cfg.algo.num_updates_per_iteration:
                    with training_timer.time("batch_sampling"):
                        online_batch = online_rb.sample(online_batch_size)
                        online_batch = online_batch.to(device, non_blocking=True)

                        if cfg.algo.offline_fraction > 0.0:
                            offline_batch = offline_rb.sample(offline_batch_size)
                            offline_batch = offline_batch.to(device, non_blocking=True)
                            batch = torch.cat([online_batch, offline_batch], dim=0)
                        else:
                            batch = online_batch

                    update_actor = (i + 1) % actor_update_cadence == 0

                    stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)

                    if update_actor:
                        if cfg.algo.actor_lr_warmup_steps > 0:
                            warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                            current_lr = cfg.agent.actor_lr * warmup_progress
                            for param_group in agent.actor_opt.param_groups:
                                param_group["lr"] = current_lr
                        actor_updates += 1

                    with training_timer.time("gradient_update"):
                        metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                    if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                        td_errors = metrics["_td_errors"]
                        batch["_priority"] = td_errors

                        if cfg.algo.offline_fraction > 0.0:
                            online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                            if online_batch_size_actual > 0:
                                online_batch_subset = batch[:online_batch_size_actual]
                                online_rb.update_tensordict_priority(online_batch_subset)

                            if online_batch_size_actual < len(batch):
                                offline_batch_subset = batch[online_batch_size_actual:]
                                offline_rb.update_tensordict_priority(offline_batch_subset)
                        else:
                            online_rb.update_tensordict_priority(batch)

                    metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                    metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()
                    i += 1

            _publish_latest_actor()

            training_cum_time += time.time() - iter_start

            # (C) Logging
            if global_step > 0 and global_step % cfg.log_freq == 0 and metrics:
                sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

                log_dict = {
                    "training/SPS": sps,
                    "training/global_step": global_step,
                    "training/episode_count": episode_count,
                    "buffer/online_size": len(online_rb),
                    "buffer/online_size_chunks": len(online_rb),
                    "buffer/online_size_steps": len(online_rb) * cfg.chunk_len,
                    "buffer/online_capacity_chunks": max(
                        1, cfg.algo.buffer_size // cfg.chunk_len
                    ),
                    "buffer/online_capacity_steps": max(
                        1, cfg.algo.buffer_size // cfg.chunk_len
                    ) * cfg.chunk_len,
                    "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                    "timing/training_total_time": time.time() - train_start_time,
                    "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                    "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                }

                timing_stats = training_timer.get_timing_stats()
                log_dict.update(timing_stats)

                filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
                log_dict.update(filtered_metrics)

                if "_actions" in metrics:
                    actions = metrics["_actions"]
                    residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                    residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                    log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                    log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                    if actions.shape[-1] % cfg.chunk_len != 0:
                        raise ValueError(
                            f"Actor action dimension {actions.shape[-1]} is not "
                            f"divisible by chunk_len={cfg.chunk_len}"
                        )
                    action_per_step_dim = actions.shape[-1] // cfg.chunk_len
                    actions_per_step = actions.reshape(
                        -1, cfg.chunk_len, action_per_step_dim
                    )
                    log_dict["histograms/residual_actions_xyz"] = wandb.Histogram(
                        actions_per_step[..., :3].numpy().reshape(-1)
                    )
                    log_dict["histograms/residual_actions_quaternion"] = wandb.Histogram(
                        actions_per_step[..., 3:7].numpy().reshape(-1)
                    )

                if "_target_q" in metrics:
                    target_q = metrics["_target_q"]
                    log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

                last_sigma = getattr(agent.actor, "last_sigma", None)
                if last_sigma is not None:
                    log_dict["histograms/scale_sigma"] = wandb.Histogram(
                        last_sigma.detach().cpu().numpy().reshape(-1)
                    )

                wandb_step_metrics.update(log_dict)

                print(f"[learner {global_step}] critic_loss={metrics.get('train/critic_loss', -1):.4f}")

            # Collector-side episode metrics are written here so W&B steps
            # follow the same ordered stream that the learner consumes.
            if wandb_step_metrics:
                wandb.log(wandb_step_metrics, step=global_step)

            # The collector stops producing data after its episode-boundary
            # request, so this saves only once the learner has drained through
            # that exact step.
            _save_requested_checkpoint()

            if global_step >= cfg.algo.total_timesteps and episode_queue.empty():
                stop_event.set()
                break


        print("[learner] finished")

def main(cfg: ResidualTD3DexmgConfig):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    enc_type=cfg.agent.enc_type

    if cfg.rl_token.enabled and cfg.rl_token.offline_num_episodes is not None:
        if cfg.rl_token.offline_num_episodes <= 0:
            raise ValueError("rl_token.offline_num_episodes must be positive or null")
        cfg.offline_data.num_episodes = cfg.rl_token.offline_num_episodes
        print(
            "RL-token offline subset enabled: "
            f"using {cfg.offline_data.num_episodes} episode(s) for embedding collection, "
            "VAE training, and offline replay construction"
        )

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load the behaviour-cloning policy that will serve as the "base" policy
    # for residual learning.
    assert getattr(cfg, "base_policy", None) is not None, "Base policy configuration is required"


    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    dataset = LeRobotDataset(cfg.offline_data.name)
    print("stats keys:", list(dataset.meta.stats.keys()))


    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["eef_actions"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )

    # Create state standardizer from dataset statistics
    def concat_state_stats(stats_a: dict, stats_b: dict) -> dict:
        out = {}
        for k in ["mean", "std", "min", "max"]:
            if k in stats_a and k in stats_b:
                a = torch.as_tensor(stats_a[k], dtype=torch.float32)
                b = torch.as_tensor(stats_b[k], dtype=torch.float32)
                out[k] = torch.cat([a, b], dim=-1)
        return out

    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=concat_state_stats(
            dataset.meta.stats["eef_position"],
            dataset.meta.stats["gripper_position"],
        ),
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    base_policy = BasePolicy(
        main_host="127.0.0.1",
        main_port=8008,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        # RL-token embedding/VAE/offline replay stages use dataset images only.
        # Hardware is connected immediately before online warmup/rollout.
        connect_robot=not cfg.rl_token.enabled,
        save_images=cfg.save_images,
        candidate_tasks=cfg.candidate_tasks,
        success_rate_window_size=cfg.task_success_window_size,
        min_task_sample_probability=cfg.min_task_sample_probability,
        prioritize_low_success_tasks=cfg.prioritize_low_success_tasks,
    )
    task_names = list(base_policy.task_reward_generator.candidate_tasks)
    task_to_id = {task: index for index, task in enumerate(task_names)}
    if len(task_to_id) != len(task_names):
        raise ValueError("candidate_tasks must be unique for task-specific routing")
    print(
        "Task sampling mode: "
        + (
            "low-success priority"
            if cfg.prioritize_low_success_tasks
            else "uniform random (no low-success priority ablation)"
        )
    )
    cfg.agent.task_specific_actor = bool(
        getattr(cfg, "task_specific_actor", True)
    )
    cfg.agent.num_tasks = len(task_names)
    if cfg.agent.task_specific_actor and cfg.agent.num_tasks < 2:
        raise ValueError(
            "task-specific actor routing requires at least two candidate tasks"
        )
    print(
        "Task-specific actor routing with shared critic: "
        f"{'enabled' if cfg.agent.task_specific_actor else 'disabled'} "
        f"for {cfg.agent.num_tasks} tasks"
    )
    if cfg.candidate_tasks is not None:
        print(f"Configured candidate tasks: {cfg.candidate_tasks}")
    print(
        "Image artifact saving: "
        f"{'enabled' if cfg.save_images else 'disabled'} "
        "(evaluation/train/warmup)"
    )


    # Seeding (must be done before environment creation) ------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # CUDA seeding for multi-GPU reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    print(f"Set random seed to {cfg.seed}")

    # Environment setup ----------------------------------------------------
    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"

    cfg.eval_num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes)
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    cfg.eval_num_envs = min(num_cpus_available, cfg.eval_num_envs)


    # Observation / action dimensions -------------------------------------
    # Determine which image keys (camera observations) will be used. The
    # configuration can specify either a single camera name (str) or a list of
    # names.
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)


    lowdim_dim = 8  # 8+8
    img_c, img_h, img_w =3, 84, 84
    per_step_dim = 8
    chunk_len = cfg.chunk_len
    action_dim = per_step_dim * chunk_len  # actor/critic act_dim = chunk_len * per_step_dim
    cfg.agent.action_chunk_len = chunk_len
    # n-step is accumulated OVER CHUNKS, so bootstrap discount is gamma**H.
    gamma_chunk = cfg.algo.gamma ** chunk_len
    # Buffer/learning_starts counters are in STEP units; convert to chunk counts.
    online_buffer_size_chunks = max(1, cfg.algo.buffer_size // chunk_len)
    learning_starts_chunks = max(1, cfg.algo.learning_starts // chunk_len)
    lowdim_keys = ["observation.state", "observation.base_action"]

    # Language / task embedding -------------------------------------------
    lang_cfg: LanguageConfig = cfg.agent.language
    if lang_cfg.enabled:
        lang_embedder = LanguageEmbedder(
            emb_dim=lang_cfg.lang_emb_dim,
            device=device,
        )
        dataset_task_prompts = _load_lerobot_task_prompts(dataset, cfg.offline_data.name)
        lowdim_keys = ["observation.state", "observation.base_action", lang_cfg.lang_emb_obs_key]
    else:
        lang_embedder = None
        dataset_task_prompts = _load_lerobot_task_prompts(
            dataset, cfg.offline_data.name
        )
        lowdim_keys = ["observation.state", "observation.base_action"]

    if cfg.agent.task_specific_actor:
        lowdim_keys.append(cfg.agent.task_id_obs_key)

    # RL-Token stages must finish before any replay-cache lookup: old image
    # buffers and buffers produced by a different bottleneck are incompatible.
    offline_rl_tokens = None
    embedding_store = None
    rl_token_id = None
    if cfg.rl_token.enabled:
        dataset_task_prompts = _load_lerobot_task_prompts(dataset, cfg.offline_data.name)
        embedding_id = artifact_id(
            cfg.offline_data.name, cfg.offline_data.num_episodes, cfg.rl_token
        )
        rl_token_id = bottleneck_id(embedding_id, cfg.rl_token)
        bottleneck_root = _CACHE_ROOT / cfg.rl_token.checkpoint_dir / rl_token_id
        bottleneck_checkpoint = bottleneck_root / "rl_token.pt"
        load_trained_bottleneck = (
            bottleneck_checkpoint.exists()
            and not cfg.rl_token.force_retrain
            and not cfg.rl_token.force_recollect_embeddings
        )

        if load_trained_bottleneck:
            print(
                "Loading trained RL-token checkpoint without initializing the "
                f"offline embedding cache: {bottleneck_checkpoint}"
            )
            rl_token_encoder = train_or_load_rl_token(
                None, cfg.rl_token, bottleneck_root, device
            )
        else:
            artifact_root = _CACHE_ROOT / cfg.rl_token.embedding_cache_dir / embedding_id
            if cfg.offline_data.num_episodes is None:
                expected_embedding_frames = dataset.meta.total_frames
            else:
                expected_embedding_frames = sum(
                    dataset.meta.episodes[ep_idx]["length"]
                    for ep_idx in range(min(cfg.offline_data.num_episodes, dataset.meta.total_episodes))
                )
            # Hold the lock through both collection and VAE training because
            # the latter streams shards from this store. A concurrent reset
            # would otherwise invalidate data while the model is reading it.
            with embedding_store_lock(artifact_root):
                embedding_store = initialize_embedding_store(
                    artifact_root,
                    expected_embedding_frames,
                    cfg.rl_token.embedding_storage_dtype,
                    reset=cfg.rl_token.force_recollect_embeddings,
                )
                print(
                    f"Loaded disk embedding store: {len(embedding_store)}/{expected_embedding_frames} "
                    f"frames, complete={embedding_store.complete}"
                )
                if not embedding_store.complete:
                    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                    resume_index = len(embedding_store)
                    shard_embeddings, shard_masks, shard_episode_ids = [], [], []
                    completed = False
                    progress = tqdm(
                        total=expected_embedding_frames,
                        initial=resume_index,
                        desc="Collecting frozen pi0 embeddings",
                    )
                    try:
                        for sample_index, sample in enumerate(loader):
                            if sample_index < resume_index:
                                continue
                            ep_idx = int(sample["episode_index"].item())
                            if cfg.offline_data.num_episodes is not None and ep_idx >= cfg.offline_data.num_episodes:
                                break
                            raw_obs = {k: sample[k].to(device) for k in sample if any(
                                name in k for name in ("exterior_image_1_left", "exterior_image_2_left",
                                                       "wrist_image_left", "eef_position", "gripper_position")
                            )}
                            task_idx = int(sample["task_index"].item())
                            embedding, mask = base_policy.get_offline_vla_embedding(
                                raw_obs, dataset_task_prompts[task_idx]
                            )
                            shard_embeddings.append(embedding.cpu())
                            shard_masks.append(mask.cpu())
                            shard_episode_ids.append(ep_idx)
                            progress.update(1)
                            if len(shard_embeddings) >= cfg.rl_token.embedding_shard_size:
                                embedding_store = append_embedding_shard(
                                    artifact_root, shard_embeddings, shard_masks, shard_episode_ids
                                )
                                shard_embeddings, shard_masks, shard_episode_ids = [], [], []
                        completed = len(embedding_store) + len(shard_embeddings) == expected_embedding_frames
                    finally:
                        progress.close()
                        embedding_store = append_embedding_shard(
                            artifact_root,
                            shard_embeddings,
                            shard_masks,
                            shard_episode_ids,
                            complete=completed,
                        )
                    if not completed:
                        raise RuntimeError(
                            f"Embedding collection stopped at {len(embedding_store)}/{expected_embedding_frames} frames"
                        )
                rl_token_encoder = train_or_load_rl_token(
                    embedding_store, cfg.rl_token, bottleneck_root, device
                )
        # Do not eagerly encode the complete offline dataset here.  The
        # serialized replay buffer already contains ``observation.rl_token``;
        # when that buffer cache exists, encoding every VLA embedding again is
        # both redundant and expensive.  ``offline_rl_tokens`` is populated
        # lazily below only when the replay-buffer cache misses.
        base_policy.set_rl_token_encoder(rl_token_encoder, cfg.rl_token.obs_key)
        image_keys = [cfg.rl_token.obs_key]
        lowdim_keys = ["observation.state", "observation.base_action"]
        if cfg.agent.task_specific_actor:
            lowdim_keys.append(cfg.agent.task_id_obs_key)

    # Networks ------------------------------------------------------------
    agent = _make_agent(cfg,
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    ) 

    def _attach_task_context(obs, task_prompt, task_emb=None):
        if obs is None:
            return obs
        if task_prompt not in task_to_id:
            raise KeyError(f"Unknown task prompt for routing: {task_prompt!r}")
        if lang_cfg.enabled:
            if task_emb is None:
                raise RuntimeError("task_emb is required when language conditioning is enabled")
            _inject_task_emb(obs, task_emb, key=lang_cfg.lang_emb_obs_key)
        if cfg.agent.task_specific_actor:
            _inject_task_id(
                obs, task_to_id[task_prompt], key=cfg.agent.task_id_obs_key
            )
        return obs

    horizon = cfg.offline_data.horizon

    # Set up actor learning rate warmup
    actor_updates = 0
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup enabled: 0.0 -> {cfg.agent.actor_lr:.2e} "
            f"over {cfg.algo.actor_lr_warmup_steps} actor updates"
        )

    # Replay buffers -------------------------------------------------------
    # Use TensorDictPrioritizedReplayBuffer for unified PER support
    # For uniform sampling, we'll use alpha=0 and beta=0, and never update priorities
    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("Online-only training mode: offline_fraction=0.0")

    # Use TensorDictPrioritizedReplayBuffer with optimized prefetching
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=online_buffer_size_chunks, device="cpu"),alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        # One buffer entry = one chunk macro-transition; n-step is accumulated over
        # chunks with gamma_chunk = gamma**chunk_len.
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=gamma_chunk),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=online_batch_size,
    )

    # Caching layer for online replay buffer ----------------------------
    online_cache_meta = {
        "task": cfg.task,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "chunk_len": chunk_len,
        "gamma_chunk": gamma_chunk,
        # v2 guarantees that both obs and next_obs store H*m base-action chunks.
        # Bumping this schema also prevents loading older malformed caches whose
        # next observation still contained a single m-dimensional action.
        # v3 also guarantees RL tokens are stored as (token_dim,) per
        # transition. v2 online caches accidentally retained only token[0].
        "storage": "chunk_v6_discrete_task_id",
        "task_routing": {
            "enabled": cfg.agent.task_specific_actor,
            "task_id_obs_key": cfg.agent.task_id_obs_key,
            "candidate_tasks": task_names,
        },
        # Task sampling changes which warmup transitions enter this cache, so
        # the random-sampling ablation must not reuse an adaptive cache.
        "prioritize_low_success_tasks": cfg.prioritize_low_success_tasks,
        "temporal_boundary_context": "prev_residual_last_v1",
        # Online RL tokens are refreshed at the same macro-step cadence as the
        # residual actor. Old caches reused one token for the 35-step pi0 base
        # plan and are not temporally compatible.
        "rl_token_refresh": (
            "residual_chunk_boundary_v1" if cfg.rl_token.enabled else "disabled"
        ),
        "gripper_filter": {
            "close_threshold": base_policy.args.gripper_close_threshold,
            "open_threshold": base_policy.args.gripper_open_threshold,
            "confirm_steps": base_policy.args.gripper_confirm_steps,
        },
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        # Include random action noise scale to prevent mixing data from different noise levels
        "random_action_noise_scale": cfg.algo.random_action_noise_scale,
        "actor_action_scale": cfg.agent.actor.action_scale,
        # Normalization parameters for consistency
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "normalized_actions": True,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        online_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        online_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(online_cache_meta)
    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha1(_online_meta_str.encode()).hexdigest()[:8]  # noqa: S324
    # Base local path for the online buffer ------------------------------
    online_cache_dir = ONLINE_CACHE_DIR / online_cache_hash

    # Attempt to download/extract from HF every run (no-op if already cached)
    dl_dir = None
    if ONLINE_HF_REPO is not None:
        print(f"Attempting to download online buffer {online_cache_hash} from {ONLINE_HF_REPO}...")
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, ONLINE_CACHE_DIR)
    if dl_dir is not None:
        online_cache_dir = dl_dir

    loaded_online_from_cache = False
    resume_checkpoint = getattr(cfg, "resume_checkpoint", None)
    expected_replay_dir = None
    if resume_checkpoint:
        resume_checkpoint_path = Path(resume_checkpoint)
        expected_replay_dir = (
            resume_checkpoint_path.parent
            / f"{resume_checkpoint_path.stem}_replay"
            / "online_rb"
        )
    exact_replay_resume = bool(
        getattr(cfg, "resume", False)
        and cfg.save_replay_on_checkpoint
        and expected_replay_dir is not None
        and expected_replay_dir.exists()
    )
    if exact_replay_resume:
        print(
            "Exact checkpoint resume requested; skipping the generic online "
            "cache and loading the checkpoint-matched replay buffer later."
        )
    elif online_cache_dir.exists():
        print(f"{online_cache_dir} found on disk. Attempting to load...")
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, online_cache_dir)
        loaded_online_from_cache = len(online_rb) > 0
        if loaded_online_from_cache:
            print(f"Loaded online buffer from cache at {online_cache_dir} (size={len(online_rb)})")
        else:
            print(
                f"Online buffer cache at {online_cache_dir} is empty; "
                "waiting for newly collected transitions before training."
            )

    # Offline data is required for normalization. A null num_episodes means
    # use every episode in the dataset.
    assert cfg.offline_data is not None

    # Dataset and normalization already loaded above - use existing dataset

    # Use actual dataset metadata for precise buffer sizing
    if cfg.offline_data.num_episodes is not None:
        # Only use subset of episodes if specified
        selected_episode_count = min(cfg.offline_data.num_episodes, dataset.meta.total_episodes)
        total_frames = sum(
            dataset.meta.episodes[ep_idx]["length"]
            for ep_idx in range(selected_episode_count)
        )
        num_episodes = selected_episode_count
    else:
        # Use entire dataset
        total_frames = dataset.meta.total_frames
        num_episodes = dataset.meta.total_episodes

    # Calculate transitions: each episode contributes (episode_length - 1) transitions
    estimated_transitions = max(0, total_frames - num_episodes)

    print("Dataset buffer sizing:")
    print(f"  Total frames to process: {total_frames}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Estimated transitions: {estimated_transitions}")

    # Calculate buffer size for simplified approach (1 transition per frame pair)
    max_offline_transitions = (
        estimated_transitions if cfg.algo.offline_fraction > 0.0 else 1
    )  # Minimum size for online-only mode
    if cfg.algo.offline_fraction > 0.0:
        print(f"Offline buffer sized for GT-as-base approach: {max_offline_transitions} transitions")
    else:
        print("Online-only mode: creating minimal offline buffer (unused)")

    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        # Same MultiStep-over-chunks framework as the online buffer.
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=gamma_chunk),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=max(offline_batch_size, 1),  # Ensure batch_size is at least 1
    )

    # Normalization functions already defined above - use them

    # Convert offline dataset episodes into transitions and fill buffer
    def _populate_offline_buffer(
        dataset: LeRobotDataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        use_base_policy_for_base_actions: bool = False,
        base_policy: BasePolicy | None = None,
    ) -> int:
        """
        Iterates through *dataset* sequentially, converts consecutive frames
        into residual RL transitions and pushes them into *rb*.

        Two modes:
        1. GT-as-base (use_base_policy_for_base_actions=False):
           Uses GT actions as both the base action (in observations) and the target action
           (in transitions). Teaches residual policy to output zero: residual = GT - GT = 0

        2. Base-policy-as-base (use_base_policy_for_base_actions=True):
           Uses base policy to generate base actions and GT actions as targets.
           More consistent with online training: residual = GT - base_policy_action

        Returns the number of transitions added.
        """
        if (use_base_policy_for_base_actions or cfg.rl_token.enabled) and base_policy is None:
            raise ValueError(
                "base_policy must be provided when base actions or RL tokens "
                "are generated from the VLA policy"
            )

        # Populate buffer from pre-loaded dataset
        # Populate buffer with CHUNK macro-transitions (same format as online chunks):
        # Chunks are STRIDE-H (non-overlapping) so MultiStepTransform (gamma_chunk)
        # accumulates the n-step over-chunks return without double counting.
        H = chunk_len
        print("Populating offline buffer from dataset (chunk transitions)...")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)   # todo define the offline data :  dataset

        # 1) One per-frame pass -> per-episode frame records (kept on CPU).
        episodes: dict[int, list[dict]] = defaultdict(list)
        transitions = 0
        step_id = 0
        previous_ep_idx = None

        for sample in tqdm(loader, desc="Processing offline dataset"):
            ep_idx = int(sample["episode_index"].item())
            if num_episodes is not None and ep_idx >= num_episodes:
                break

            if "task_index" not in sample:
                raise KeyError(
                    "Dataset sample is missing 'task_index'; cannot route "
                    "task-specific actors"
                )
            dataset_task_index = int(sample["task_index"].item())
            if dataset_task_index not in dataset_task_prompts:
                raise KeyError(
                    f"task_index={dataset_task_index} was not found in tasks.jsonl "
                    f"(available: {sorted(dataset_task_prompts)})"
                )
            task_prompt = dataset_task_prompts[dataset_task_index]
            if task_prompt not in task_to_id:
                raise KeyError(
                    f"Offline task {task_prompt!r} is not in candidate_tasks; "
                    "task-specific routing requires one shared task ordering"
                )
            routed_task_id = task_to_id[task_prompt]

            if previous_ep_idx is None or ep_idx != previous_ep_idx:
                if use_base_policy_for_base_actions:
                    stale_actions = len(base_policy.base_action_buffer)
                    base_policy.base_action_buffer.clear()
                    base_policy._last_base_action = None
                    base_policy._last_rl_token = None
                    # Dataset trajectories are independent episodes. Let the
                    # first observation initialize the persistent gripper latch
                    # instead of carrying the previous trajectory's state.
                    base_policy.reset_gripper_filter_state()
                    base_policy.pi05_client.reset()
                    print(
                        f"[offline episode {ep_idx}] cleared {stale_actions} buffered "
                        "base action(s)"
                    )
                previous_ep_idx = ep_idx

            # Build observation and action directly for replay buffer ----------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            _gt_action: torch.Tensor = sample["eef_actions"].float().squeeze(0)
            gt_action_scaled = action_scaler.scale(_gt_action).reshape(-1).cpu()
            if "next.done" in sample:
                done_flag = bool(sample["next.done"].item())
            elif "done" in sample:
                done_flag = bool(sample["done"].item())
            else:
                done_flag = False

            if done_flag:
                ep_idx = int(sample["episode_index"].item())
                frame_idx = int(sample["frame_index"].item())
                print(f"episode {ep_idx} done at frame {frame_idx}")

            # Both base-action inference and streaming RL-token encoding need
            # the raw VLA observation. Build it once for the current frame.
            raw_obs = None
            if use_base_policy_for_base_actions or cfg.rl_token.enabled:
                raw_obs = {
                    k: sample[k].to(device)
                    for k in sample
                    if any(
                        name in k
                        for name in (
                            "exterior_image_1_left",
                            "exterior_image_2_left",
                            "wrist_image_left",
                            "eef_position",
                            "gripper_position",
                        )
                    )
                }

            # Generate base action based on the selected mode
            if use_base_policy_for_base_actions:
                # Get base action from base policy
                with torch.no_grad():
                    base_action = base_policy.get_offline_action_base(raw_obs, task_prompt)  # base_action shape:(8,)
                base_action = torch.as_tensor(base_action, dtype=torch.float32)
                base_action_scaled = action_scaler.scale(base_action.cpu()).reshape(-1)
            else:
                # Use GT action as base action (original behavior)

                # Use GT action as base action (residual target 0)
                base_action_scaled = gt_action_scaled.clone()
            state = torch.cat((sample["eef_position"], sample["gripper_position"].unsqueeze(-1)), dim=-1)
            frame = {
                "state": state_standardizer.standardize(state.float().squeeze(0)).cpu(),
                "base": base_action_scaled,          # (m,)
                "gt": gt_action_scaled,              # (m,)
                "reward": float(done_flag),         # sparse terminal reward
                "done": done_flag,
                "task_id": routed_task_id,
            }
            if cfg.rl_token.enabled:
                if offline_rl_tokens is not None:
                    if step_id >= len(offline_rl_tokens):
                        raise IndexError("RL-token cache is shorter than the offline dataset")
                    rl_token = offline_rl_tokens[step_id]
                else:
                    # A trained bottleneck can encode each VLA embedding as it
                    # arrives. Discarding the high-dimensional embedding here
                    # avoids recreating the full disk embedding cache merely
                    # to build a missing replay buffer.
                    embedding, valid = base_policy.get_offline_vla_embedding(
                        raw_obs, task_prompt
                    )
                    with torch.no_grad():
                        rl_token = rl_token_encoder.encode(
                            embedding.unsqueeze(0).to(device),
                            ~valid.unsqueeze(0).to(device),
                        ).squeeze(0).cpu()
                frame[cfg.rl_token.obs_key] = rl_token
                step_id += 1
            img = {}

            # Build observation dict directly in target format'
            for k in image_keys:
                if cfg.rl_token.enabled:
                    continue
                raw_k = k.replace("observation.images.", "")
                img[k] = sample[raw_k].squeeze(0)
            if not cfg.rl_token.enabled:
                img = process_image_batch(img, image_keys, enc_type, rb=True)
                to_uint8(img, image_keys)
            for k in image_keys:
                if not cfg.rl_token.enabled:
                    frame[k] = img[k]

            # Inject task embedding from the dataset's per-frame task_index.
            if lang_cfg.enabled:
                if lang_embedder is None:
                    raise RuntimeError("lang_embedder must be initialized when language is enabled")


                frame[lang_cfg.lang_emb_obs_key] = lang_embedder(task_prompt).detach().cpu()

            episodes[ep_idx].append(frame)

        # 2) Assemble non-overlapping (stride-H) chunk transitions per episode.
        transitions = 0
        for _ep, frames in episodes.items():
            T = len(frames)
            for t in range(0, T - 1, H):
                # executed steps in this chunk: up to H, stop right after first terminal
                n, term = 0, False
                while n < H and (t + n) < T:
                    is_done = frames[t + n]["done"]
                    n += 1
                    if is_done:
                        term = True
                        break
                last_valid = t + n - 1
                idxs = [min(t + h, last_valid) for h in range(H)]
                nt = min(t + H, T - 1)
                nidxs = [min(nt + h, T - 1) for h in range(H)]
                base_chunk = torch.stack([frames[i]["base"] for i in idxs], dim=0).reshape(-1)     # (H*m,)
                action_chunk = torch.stack([frames[i]["gt"] for i in idxs], dim=0).reshape(-1)     # (H*m,)
                next_base_chunk = torch.stack([frames[i]["base"] for i in nidxs], dim=0).reshape(-1)
                has_prev_residual = t > 0 and not frames[t - 1]["done"]
                prev_residual_last = (
                    frames[t - 1]["gt"] - frames[t - 1]["base"]
                    if has_prev_residual
                    else torch.zeros_like(frames[t]["gt"])
                )
                R = sum((cfg.algo.gamma ** h) * frames[t + h]["reward"] for h in range(n))
                cur_obs = {"observation.state": frames[t]["state"], "observation.base_action": base_chunk}
                next_obs = {"observation.state": frames[nt]["state"], "observation.base_action": next_base_chunk}
                if cfg.agent.task_specific_actor:
                    cur_obs[cfg.agent.task_id_obs_key] = torch.tensor(
                        frames[t]["task_id"], dtype=torch.long
                    )
                    next_obs[cfg.agent.task_id_obs_key] = torch.tensor(
                        frames[nt]["task_id"], dtype=torch.long
                    )
                for k in image_keys:
                    cur_obs[k] = frames[t][k]
                    next_obs[k] = frames[nt][k]
                if lang_cfg.enabled:
                    cur_obs[lang_cfg.lang_emb_obs_key] = frames[t][lang_cfg.lang_emb_obs_key]
                    next_obs[lang_cfg.lang_emb_obs_key] = frames[nt][lang_cfg.lang_emb_obs_key]

                transition = TensorDict(
                    {
                        "obs": TensorDict(cur_obs, batch_size=[]),
                        "action": action_chunk,
                        "next": TensorDict(
                            {
                                 "obs": TensorDict(next_obs, batch_size=[]),
                                "done": torch.tensor(bool(term), dtype=torch.bool),
                                "reward": torch.tensor(float(R), dtype=torch.float32),
                            },
                            batch_size=[],
                        ),
                        "prev_residual_last": prev_residual_last,
                        "has_prev_residual": torch.tensor(
                            has_prev_residual, dtype=torch.bool
                        ),
                        "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
                    },
                    batch_size=[],
                ).unsqueeze(0)
                rb.add(transition)
                transitions += 1
        # Log final statistics

        print(f"Added {transitions} offline chunk transitions")

        return transitions

    # Caching layer for offline replay buffer ---------------------------
    # Build a metadata dictionary that uniquely identifies the buffer
    offline_cache_meta = {
        "task": cfg.task,
        "dataset_name": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "chunk_len": chunk_len,
        "gamma_chunk": gamma_chunk,
        "storage": "chunk_v3_discrete_task_id",
        "task_routing": {
            "enabled": cfg.agent.task_specific_actor,
            "task_id_obs_key": cfg.agent.task_id_obs_key,
            "candidate_tasks": task_names,
        },
        "temporal_boundary_context": "prev_residual_last_v1",
        "gripper_filter": {
            "close_threshold": base_policy.args.gripper_close_threshold,
            "open_threshold": base_policy.args.gripper_open_threshold,
            "confirm_steps": base_policy.args.gripper_confirm_steps,
        },
        "rl_token_artifact": rl_token_id,
        "base_policy_wandb_id": cfg.base_policy.wandb_id,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "normalized_actions": True,
        "batch_size": offline_batch_size,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(offline_cache_meta)

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(offline_cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer ---------------------------------------
    cache_dir = OFFLINE_CACHE_DIR / cache_hash

    # Try to download/extract from the Hub (will no-op if file not there)
    downloaded_dir = None
    if OFFLINE_HF_REPO is not None:
        print(f"Attempting to download offline buffer {cache_hash} from {OFFLINE_HF_REPO}...")
        downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
    if downloaded_dir is not None:
        cache_dir = downloaded_dir  # use extracted location

    loaded_from_cache = False
    added = 0

    if cfg.algo.offline_fraction > 0.0:
        # Only populate offline buffer if we're using offline data
        if cache_dir.exists():
            print(f"{cache_dir} found on disk. Attempting to load...")
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(f"Loaded offline buffer from cache at {cache_dir} (size={len(offline_rb)})")

        if not loaded_from_cache:
            if cfg.rl_token.enabled:
                # A newly constructed buffer stores the compact RL token in
                # each observation (and next observation).  Persisting that
                # buffer makes subsequent runs independent of this full-dataset
                # encoding pass.
                if embedding_store is not None:
                    offline_rl_tokens = encode_all(
                        rl_token_encoder, embedding_store, device
                    )
                else:
                    print(
                        "Offline replay cache is missing; streaming VLA "
                        "embeddings through the trained RL-token encoder "
                        "without writing an embedding cache."
                    )
            added = _populate_offline_buffer(
                dataset=dataset,
                rb=offline_rb,
                image_keys=image_keys,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=(
                    base_policy
                    if cfg.offline_data.use_base_policy_for_base_actions or cfg.rl_token.enabled
                    else None
                ),
            )

            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")

            # Save buffer to disk for future runs + upload to Hub ----------------
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)

            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)

            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
        else:
            added = len(offline_rb)
    else:
        print("Skipping offline buffer population for online-only training")

    if cfg.rl_token.enabled:
        print(
            "RL-token offline stages complete; connecting Franka and the ZED "
            "reward camera for online rollout..."
        )
        base_policy.connect_online_resources(required=True)

    initial_eval_success_rates = None
    configured_initial_eval_log = getattr(cfg, "initial_task_success_log", None)
    configured_initial_eval_overrides = getattr(
        cfg, "initial_task_success_window_overrides", None
    )
    has_configured_initial_windows = (
        configured_initial_eval_log is not None
        or configured_initial_eval_overrides is not None
    )
    if has_configured_initial_windows and not getattr(cfg, "resume", False):
        if base_policy.task_reward_generator is None:
            raise RuntimeError(
                "Initial task success windows were configured before the task "
                "reward generator was connected"
            )
        candidate_tasks = list(
            base_policy.task_reward_generator.candidate_tasks
        )
        log_eval_outcomes = None
        if configured_initial_eval_log is not None:
            log_eval_outcomes = load_initial_eval_outcomes_from_log(
                configured_initial_eval_log,
                candidate_tasks=candidate_tasks,
                window_size=cfg.task_success_window_size,
            )
        initial_eval_outcomes = apply_initial_eval_outcome_overrides(
            log_eval_outcomes,
            configured_initial_eval_overrides,
            candidate_tasks=candidate_tasks,
            window_size=cfg.task_success_window_size,
        )
        base_policy.task_reward_generator.initialize_task_success_windows(
            initial_eval_outcomes
        )
        initial_eval_success_rates = {
            task: sum(outcomes) / len(outcomes)
            for task, outcomes in initial_eval_outcomes.items()
        }
        formatted_rates = ", ".join(
            f"{task}: {rate:.2%}"
            for task, rate in initial_eval_success_rates.items()
        )
        if configured_initial_eval_log is not None:
            print(
                f"Loaded per-task {cfg.task_success_window_size}-slot initial "
                "success windows from "
                f"{Path(configured_initial_eval_log).expanduser().resolve()}"
            )
        if configured_initial_eval_overrides is not None:
            overridden_tasks = list(configured_initial_eval_overrides)
            print(
                "Applied configured initial success-window overrides for: "
                f"{overridden_tasks}"
            )
        print(f"Initial task success rates: {formatted_rates}")
        if cfg.eval_first:
            print(
                "Skipping step-0 robot evaluation because "
                "initial task success windows are configured."
            )
    elif configured_initial_eval_overrides is not None:
        print(
            "Ignoring initial_task_success_window_overrides while resuming; "
            "the checkpoint's newer rolling-window state takes precedence."
        )

    configured_task_output_root = getattr(cfg, "task_output_root", None)
    if configured_task_output_root is not None:
        task_output_root_path = (
            Path(configured_task_output_root).expanduser().resolve()
        )
        if not task_output_root_path.is_dir():
            raise FileNotFoundError(
                "Configured task_output_root does not exist or is not a "
                f"directory: {task_output_root_path}"
            )
        if base_policy.task_reward_generator is None:
            raise RuntimeError(
                "task_output_root was configured before the task reward "
                "generator was connected"
            )
        base_policy.task_reward_generator.restore_output_root(
            task_output_root_path
        )
        print(f"Appending task artifacts in: {task_output_root_path}")

    # Warm-up phase (random policy) --------------------------------------

    # Match the cached-warmup path used by the RL-token entry point: random
    # warmup transitions belong in replay, but their task outcomes must not
    # advance the learned-policy success curves or adaptive sampling windows.
    # Restore this snapshot after an in-process warmup has been collected.
    pre_warmup_task_success_stats = copy.deepcopy(
        base_policy.task_reward_generator.task_success_stats
    )
    fresh_warmup_collected = False

    if len(online_rb) < learning_starts_chunks and not loaded_online_from_cache and not getattr(cfg, "resume", False):
        fresh_warmup_collected = True
        warmup_dir = base_policy.task_reward_generator.set_output_warmup()
        print(f"Warm-up: filling online buffer with {learning_starts_chunks - len(online_rb)} random chunks "
              f"({(learning_starts_chunks - len(online_rb)) * chunk_len} steps)…")
        print(f"Warm-up artifacts: {warmup_dir}")
        obs, task_prompt = base_policy.reset()
        task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
        obs = _attach_task_context(obs, task_prompt, task_emb)
        # Logging helper: print progress every 1 000 collected transitions
        next_log_threshold = 1000  # first threshold for progress message

        reward_sum = 0
        episode_count = 0

        while len(online_rb) < learning_starts_chunks:
            print(f"[warmup] len(online_rb) before step = {len(online_rb)}")

            # Attach the base-action chunk (H*dim) so the stored obs is chunk-level.
            obs["observation.base_action"] = base_policy.current_base_chunk(chunk_len)

            if cfg.algo.use_base_policy_for_warmup:
                # # Use base policy action + noise (residual exploration)
                # # Since the environment wrapper always adds base_action to residual_action,
                # # we just need to provide the noise as the residual action
                # Use base policy action + noise (residual exploration). The chunk client
                # adds the per-step base action, so the residual chunk is just the noise.
                rand_actions = (  # line2: Sample noise εt ∼ U (−noise scale, noise scale)
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
            else:
                # # Pure uniform random actions - need to cancel out the base policy action
                # # Since env does: combined = base_action + residual_action
                # # To get pure random: residual_action = random - base_action
                # Pure uniform random chunk - cancel out the base action chunk.
                # Since env does: combined = base_chunk + residual_chunk
                base_chunk = base_policy.current_base_chunk(chunk_len)

                pure_random = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
                rand_actions = pure_random - base_chunk

            # # line4: Observe next state st+1, reward rt, done flag dt


            # line3-4: open-loop execute the residual chunk; observe next chunk-start obs
            next_obs, combined_action, reward, done, info = base_policy.step_chunk(rand_actions)

            # Store a true chunk-level next observation. Without this assignment,
            # online next_obs has shape (m,) while offline next_obs has (H*m,),
            # causing torch.cat to fail during mixed critic warm-up.
            next_obs["observation.base_action"] = base_policy.current_base_chunk(chunk_len)

            task_prompt = info["task_prompt"]
            task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
            next_obs = _attach_task_context(next_obs, task_prompt, task_emb)
            reward_sum += reward.sum().item()
            episode_count += done.float().sum().item()
            if done:
                if reward>0:
                    res = "✓"
                else:
                    res = "✗"
                print(f"task: {task_prompt} : {res}")
            # Use the executed combined action returned by the environment
            # combined_action is the executed combined action chunk (1, H*dim)
            _add_transitions_to_buffer(  # line5: Add transition (st, atb, at, st+1, atb+1, rt, dt) to online replay buffer
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
                enc_type=enc_type
            )


            # Progress logging (every ~1 000 transitions) --------------
            if len(online_rb) >= next_log_threshold:
                success_rate = reward_sum / episode_count if episode_count > 0 else 0.0
                print(
                    f"[Warm-up] {len(online_rb)} / {learning_starts_chunks} "
                    f"chunks collected ({len(online_rb) * chunk_len} steps), reward_sum={reward_sum:.2f}, "
                    f"success_rate={success_rate:.3f} ({reward_sum}/{episode_count})"
                )
                next_log_threshold += 1000

            obs = next_obs  # roll state

        # Persist freshly-collected buffer (local + HF) --------------------
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        optimized_replay_buffer_dumps(online_rb, online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        base_policy.task_reward_generator.restore_task_success_stats(
            pre_warmup_task_success_stats
        )
        print(
            "Restored pre-warmup task success windows; random warmup outcomes "
            "are excluded from learned-policy episode counts and W&B curves."
        )

        loaded_online_from_cache = True  # treat as cached going forward

    global_step = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0
    actor_updates = 0
    resume_state = None

    if getattr(cfg, "resume", False) and getattr(cfg, "resume_checkpoint", None):
        print(f"Resuming training from checkpoint: {cfg.resume_checkpoint}")
        load_exact_replay = bool(
            cfg.save_replay_on_checkpoint
            and getattr(cfg, "require_exact_replay_on_resume", True)
        )
        if not load_exact_replay:
            if len(online_rb) == 0:
                raise RuntimeError(
                    "Exact replay resume was disabled, but no compatible "
                    "generic online replay cache was loaded."
                )
            print(
                "WARNING: exact checkpoint replay resume is disabled; using "
                f"the compatible generic online replay cache (size={len(online_rb)})."
            )
        resume_state = load_training_checkpoint(
            ckpt_path=cfg.resume_checkpoint,
            agent=agent,
            online_rb=online_rb,
            device=device,
            load_replay_buffer=load_exact_replay,
        )

        global_step = resume_state["global_step"]
        best_eval_success_rate = resume_state["best_eval_success_rate"]
        training_cum_time = resume_state["training_cum_time"]
        episode_count = resume_state["episode_count"]
        actor_updates = resume_state["actor_updates"]

        if configured_task_output_root is not None:
            print(
                "Using explicitly configured task artifact root instead of "
                "the root stored in the checkpoint: "
                f"{base_policy.task_reward_generator.output_root}"
            )
        elif resume_state.get("task_output_root"):
            base_policy.task_reward_generator.restore_output_root(
                resume_state["task_output_root"]
            )
            print(
                "Resuming task artifacts in: "
                f"{base_policy.task_reward_generator.output_root}"
            )
        else:
            print(
                "Checkpoint has no task_output_root; using a new task artifact "
                f"directory: {base_policy.task_reward_generator.output_root}"
            )

        saved_task_stats = resume_state.get("task_success_stats")
        if cfg.algo.reset_task_success_rates_on_restart:
            print("Task rolling success-rate state reset to baseline 0.5 after restart.")
        elif saved_task_stats is None:
            print(
                "Checkpoint has no task_success_stats; task success rates "
                "remain initialized at 0.5."
            )
        else:
            saved_task_stats, migrated_eval_path = _backfill_initial_eval_outcomes(
                saved_task_stats,
                task_output_root=resume_state.get("task_output_root"),
                candidate_tasks=list(
                    base_policy.task_reward_generator.candidate_tasks
                ),
                window_size=cfg.task_success_window_size,
            )
            if migrated_eval_path is not None:
                print(
                    "Recovered ordered initial evaluation outcomes for the "
                    f"rolling windows from {migrated_eval_path}"
                )
            base_policy.task_reward_generator.restore_task_success_stats(
                saved_task_stats
            )
            print("Restored per-task success rates from checkpoint.")

    base_policy.task_reward_generator.output_root.mkdir(parents=True, exist_ok=True)

    _hp_parts: list[str] = [
        cfg.task,  # e.g. "TwoArmBoxCleanup"
        f"n{cfg.algo.n_step}",  # n-step horizon
        f"utd{cfg.algo.num_updates_per_iteration}",  # updates-to-data ratio
        f"buf{cfg.algo.buffer_size}",  # replay buffer size
    ]

    # Offline dataset statistics (if any)
    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None and cfg.algo.offline_fraction > 0.0:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")
    elif cfg.algo.offline_fraction == 0.0:
        _hp_parts.append("online_only")

    # Learning-rate, expressed in scientific notation for brevity (e.g. 1e-4 → 1e-04)
    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")

    # Additional flags ---------------------------------------------------------
    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)

    if resume_state is not None and resume_state.get("run_name") is not None:
        run_name = resume_state["run_name"]
    else:
        run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

        if cfg.wandb.name is not None:
            run_name = f"{cfg.wandb.name}__{run_name}"

    wandb_run_id = cfg.wandb.continue_run_id
    resume_checkpoint_run = getattr(cfg.wandb, "resume_checkpoint_run", True)
    if wandb_run_id is None and resume_state is not None and resume_checkpoint_run:
        wandb_run_id = resume_state.get("wandb_run_id")
    elif wandb_run_id is None and resume_state is not None and not resume_checkpoint_run:
        print(
            "Loaded training state from checkpoint, but starting a new wandb run "
            "because wandb.resume_checkpoint_run=False. This is the right choice "
            "when the checkpoint step is behind the existing wandb history."
        )
        run_name = f"{run_name}__from_step{global_step}"

    _wandb_config = _config_to_container(cfg)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)

    if resume_state is not None and wandb_run_id is None and resume_checkpoint_run:
        print(
            "Checkpoint does not contain wandb_run_id; pass "
            "wandb.continue_run_id=<run_id> to resume the original wandb run."
        )

    if resume_state is not None and wandb_run_id is not None:
        print(
            f"Continuing wandb run {wandb_run_id} from checkpoint step {global_step}. "
            "WandB history is append-only: if that run already has logs after this "
            "step, they will remain. Use wandb.resume_checkpoint_run=False to load "
            "the checkpoint into a fresh wandb run instead."
        )

    wandb.init(
        id=wandb_run_id,
        resume=None if wandb_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )

    # Give every per-task success-rate chart its own episode-count x-axis.
    # Since collector logging omits a task's success_rate when another task is
    # executed, its curve advances only when that task completes an episode.
    task_metric_names = (
        list(base_policy.task_reward_generator.candidate_tasks)
        if base_policy.task_reward_generator is not None
        else list(cfg.candidate_tasks or [])
    )
    for task_index, _task in enumerate(task_metric_names):
        count_metric = f"tasks/task_{task_index}_episode_count"
        wandb.define_metric(count_metric)
        wandb.define_metric(
            f"tasks/task_{task_index}_success_rate",
            step_metric=count_metric,
        )

    if initial_eval_success_rates is not None:
        initial_curve_metrics = {}
        for task_index, task in enumerate(task_metric_names):
            initial_curve_metrics[f"tasks/task_{task_index}_episode_count"] = 0
            initial_curve_metrics[f"tasks/task_{task_index}_attempts"] = 0
            initial_curve_metrics[f"tasks/task_{task_index}_success_rate"] = (
                initial_eval_success_rates[task]
            )
        wandb.log(initial_curve_metrics, step=0)
        print(
            "Logged reused initial success-rate means at episode_count=0 "
            "for the new run."
        )

    # Log horizon to wandb summary

    # A fresh run gets one timestamped directory. Before the first checkpoint,
    # run_output_dir can reconnect a restarted process to that initial
    # directory. Exact checkpoint resume infers and reuses the same directory.
    configured_run_output_dir = getattr(cfg, "run_output_dir", None)
    reuse_run_dir = bool(
        resume_state is not None
        and cfg.reuse_run_dir_on_resume
        and getattr(cfg, "resume_checkpoint", None)
    )

    def _checkpoint_steps(directory: Path) -> list[int]:
        steps = []
        for candidate in directory.glob("checkpoint_*.pt"):
            try:
                steps.append(int(candidate.stem.removeprefix("checkpoint_")))
            except ValueError:
                continue
        return steps

    if configured_run_output_dir is not None:
        run_cache_dir = Path(configured_run_output_dir).expanduser().resolve()
        if not run_cache_dir.is_dir():
            raise FileNotFoundError(
                "Configured run_output_dir does not exist or is not a "
                f"directory: {run_cache_dir}"
            )
        model_save_dir = run_cache_dir / "models"
        existing_checkpoint_steps = _checkpoint_steps(model_save_dir)
        if resume_state is None and existing_checkpoint_steps:
            raise RuntimeError(
                f"run_output_dir already contains checkpoint step "
                f"{max(existing_checkpoint_steps)}. Resume from its latest "
                "checkpoint instead of starting from step 0."
            )
        if reuse_run_dir:
            resume_ckpt_path = Path(cfg.resume_checkpoint).expanduser().resolve()
            if resume_ckpt_path.parent != model_save_dir:
                raise ValueError(
                    "run_output_dir must contain resume_checkpoint; got "
                    f"run_output_dir={run_cache_dir}, "
                    f"resume_checkpoint={resume_ckpt_path}"
                )
            newer_steps = [
                step for step in existing_checkpoint_steps if step > global_step
            ]
            if newer_steps:
                raise RuntimeError(
                    f"Refusing in-place resume from step {global_step}: run "
                    f"directory already contains newer checkpoint step "
                    f"{max(newer_steps)}. Resume the latest checkpoint."
                )
        print(f"Reusing configured run directory: {run_cache_dir}")
    elif reuse_run_dir:
        resume_ckpt_path = Path(cfg.resume_checkpoint).expanduser().resolve()
        model_save_dir = resume_ckpt_path.parent
        if model_save_dir.name != "models":
            raise ValueError(
                "resume_checkpoint must be inside a models/ directory when "
                "reuse_run_dir_on_resume=True; got "
                f"{resume_ckpt_path}"
            )
        run_cache_dir = model_save_dir.parent

        checkpoint_steps = _checkpoint_steps(model_save_dir)
        newer_steps = [step for step in checkpoint_steps if step > global_step]
        if newer_steps:
            raise RuntimeError(
                f"Refusing in-place resume from step {global_step}: run directory "
                f"already contains newer checkpoint step {max(newer_steps)}. "
                "Resume the latest checkpoint or set "
                "reuse_run_dir_on_resume=False to create a separate directory."
            )
        print(f"Reusing original run directory: {run_cache_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"
        model_save_dir = run_cache_dir / "models"

    # Create subdirectories for models and outputs.
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)


    train_start_time = time.time()

    # Initialize timing utility
    training_timer = TrainingTimer()

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""

        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            with training_timer.time("batch_sampling"):
                # Sample batches from replay buffers
                online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(device, non_blocking=True)

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training
                    offline_batch = offline_rb.sample(offline_batch_size)
                    offline_batch = offline_batch.to(device, non_blocking=True)

                    batch = torch.cat([online_batch, offline_batch], dim=0)
                else:
                    # Online-only training
                    batch = online_batch

            # Only update critic during warmup (update_actor=False)
            with training_timer.time("gradient_update"):
                metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)

            # Update priorities for prioritized experience replay
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                # Update priorities in the batch for priority updates
                td_errors = metrics["_td_errors"]
                batch["_priority"] = td_errors

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training - update both buffers
                    online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                    # Update online buffer priorities
                    if online_batch_size_actual > 0:
                        online_batch_subset = batch[:online_batch_size_actual]
                        online_rb.update_tensordict_priority(online_batch_subset)

                    # Update offline buffer priorities
                    if online_batch_size_actual < len(batch):
                        offline_batch_subset = batch[online_batch_size_actual:]
                        offline_rb.update_tensordict_priority(offline_batch_subset)
                else:
                    # Online-only training - update only online buffer
                    online_rb.update_tensordict_priority(batch)

            # Progress logging
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )

    if cfg.algo.critic_warmup_steps > 0 and not cfg.resume:
        print(f"Critic warmup: running {cfg.algo.critic_warmup_steps} critic-only updates...")
        _run_critic_warmup(
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            cfg=cfg,
            device=device,
            training_timer=training_timer,
            online_batch_size=online_batch_size,
            offline_batch_size=offline_batch_size,
        )
        print("Critic warmup completed.")


    def launch_async_training():
        episode_queue = queue.Queue()
        weights_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        checkpoint_sync = EpisodeCheckpointCoordinator()
        collector_errors = []

        def _run_collector():
            try:
                collector_loop(
                    cfg,
                    lang_cfg,
                    lang_embedder,
                    device,
                    base_policy,
                    image_keys,
                    enc_type,
                    episode_queue,
                    weights_queue,
                    stop_event,
                    checkpoint_sync,
                    training_timer,
                    run_name,
                    outputs_dir,
                    img_c,
                    img_h,
                    img_w,
                    lowdim_dim,
                    action_dim,
                    global_step,
                    episode_count,
                    best_eval_success_rate,
                    fresh_warmup_collected,
                )
            except BaseException as exc:
                collector_errors.append(exc)
                checkpoint_sync.mark_failed(exc)
                stop_event.set()

        collector_thread = threading.Thread(
            target=_run_collector,
            daemon=True,
        )

        collector_thread.start()

        try:
            learner_loop( cfg,
                    device,
                    agent,
                    online_rb,
                    offline_rb,
                    image_keys,
                    lowdim_keys,
                    enc_type,
                    episode_queue,
                    weights_queue,
                    stop_event,
                    checkpoint_sync,
                    model_save_dir,
                    run_name,
                    str(base_policy.task_reward_generator.output_root),
                    base_policy.task_reward_generator.task_success_stats,
                    online_cache_dir,
                    online_cache_meta,
                    online_cache_hash,
                    global_step,
                    actor_updates,
                    episode_count,
                    best_eval_success_rate,
                    training_cum_time,)
        finally:
            stop_event.set()
            checkpoint_sync.wake_waiters()
            collector_thread.join()

        if collector_errors:
            raise RuntimeError("Collector thread failed") from collector_errors[0]

    launch_async_training()

    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Preserve checkpoints by default so this run remains resumable.
    if cfg.cleanup_run_dir_on_finish and run_cache_dir.exists():
        print(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        print("Run directory cleaned up successfully.")


# Hydra entry point -----------------------------------------------------------
@hydra.main(version_base=None, config_name="residual_td3_dexmg_config")
def hydra_entry(cfg: ResidualTD3DexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
