"""Per-episode action diagnostics for online residual-policy rollouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _as_xyz(actions, name: str) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"{name} must have shape [steps, action_dim>=3], got {values.shape}")
    return values[:, :3]


def _sign_flip_count(values: np.ndarray, tolerance: float = 1e-6) -> int:
    signs = np.sign(values[np.abs(values) > tolerance])
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _unused_output_stem(output_dir: Path, stem: str) -> Path:
    candidate = output_dir / stem
    duplicate_index = 1
    while candidate.with_suffix(".png").exists() or candidate.with_suffix(".csv").exists():
        candidate = output_dir / f"{stem}_restart{duplicate_index:02d}"
        duplicate_index += 1
    return candidate


def save_episode_action_diagnostics(
    *,
    output_dir: str | Path,
    episode_number: int,
    task_prompt: str,
    start_global_step: int,
    end_global_step: int,
    base_actions: np.ndarray,
    residual_actions: np.ndarray,
    combined_actions: np.ndarray,
) -> tuple[Path, Path]:
    """Save one complete episode's XYZ policy traces as PNG and CSV.

    ``base_actions`` and ``combined_actions`` are in physical action units.
    ``residual_actions`` contains the residual actor's normalized output. The
    applied physical correction is also written as ``combined - base``.
    """
    base_xyz = _as_xyz(base_actions, "base_actions")
    residual_xyz = _as_xyz(residual_actions, "residual_actions")
    combined_xyz = _as_xyz(combined_actions, "combined_actions")
    if not (len(base_xyz) == len(residual_xyz) == len(combined_xyz)):
        raise ValueError(
            "Action diagnostics must have matching step counts, got "
            f"base={len(base_xyz)}, residual={len(residual_xyz)}, "
            f"combined={len(combined_xyz)}"
        )
    if len(base_xyz) == 0:
        raise ValueError("Cannot plot an empty episode")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = _unused_output_stem(
        output_dir,
        f"episode_{int(episode_number):06d}_steps_{int(start_global_step):09d}"
        f"_{int(end_global_step):09d}",
    )

    applied_residual_xyz = combined_xyz - base_xyz
    table = np.column_stack([
        np.arange(len(base_xyz), dtype=np.int64),
        base_xyz,
        residual_xyz,
        combined_xyz,
        applied_residual_xyz,
    ])
    csv_path = output_stem.with_suffix(".csv")
    csv_tmp_path = csv_path.with_suffix(".csv.tmp")
    np.savetxt(
        csv_tmp_path,
        table,
        delimiter=",",
        fmt=["%d"] + ["%.9g"] * 12,
        header=(
            f"task={task_prompt}\n"
            "env_step,base_x_m,base_y_m,base_z_m,"
            "residual_x_normalized,residual_y_normalized,residual_z_normalized,"
            "combined_x_m,combined_y_m,combined_z_m,"
            "applied_residual_x_m,applied_residual_y_m,applied_residual_z_m"
        ),
    )
    csv_tmp_path.replace(csv_path)

    dimension_names = ("X", "Y", "Z")
    env_steps = np.arange(len(base_xyz))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for dimension, (axis, dimension_name) in enumerate(zip(axes, dimension_names)):
        axis.plot(
            env_steps,
            base_xyz[:, dimension],
            color="tab:blue",
            linewidth=1.5,
            label="base policy inference (m)",
        )
        axis.plot(
            env_steps,
            combined_xyz[:, dimension],
            color="tab:green",
            linewidth=1.2,
            alpha=0.85,
            label="base + residual (m)",
        )
        axis.set_ylabel(f"{dimension_name} position (m)")
        axis.grid(True, alpha=0.25)

        residual_axis = axis.twinx()
        residual_axis.plot(
            env_steps,
            applied_residual_xyz[:, dimension],
            color="tab:red",
            linewidth=1.0,
            alpha=0.8,
            label="residual policy correction (m)",
        )
        residual_axis.axhline(0.0, color="tab:red", linewidth=0.6, alpha=0.3)
        residual_axis.set_ylabel("residual correction (m)", color="tab:red")
        residual_axis.tick_params(axis="y", labelcolor="tab:red")
        residual_axis.text(
            0.99,
            0.95,
            f"residual sign flips: "
            f"{_sign_flip_count(applied_residual_xyz[:, dimension])}",
            transform=residual_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="tab:red",
        )

        if dimension == 0:
            handles, labels = axis.get_legend_handles_labels()
            residual_handles, residual_labels = residual_axis.get_legend_handles_labels()
            axis.legend(
                handles + residual_handles,
                labels + residual_labels,
                loc="best",
                fontsize=8,
            )

    axes[-1].set_xlabel("executed environment step")
    fig.suptitle(
        f"Training episode {int(episode_number)} | {task_prompt}\n"
        f"global steps {int(start_global_step)}-{int(end_global_step)}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = output_stem.with_suffix(".png")
    png_tmp_path = png_path.with_suffix(".png.tmp")
    fig.savefig(png_tmp_path, format="png", dpi=160)
    plt.close(fig)
    png_tmp_path.replace(png_path)
    return png_path, csv_path
