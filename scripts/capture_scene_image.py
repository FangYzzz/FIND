#!/usr/bin/env python3
"""Capture full-resolution scene images from the right ZEDX stream."""

from __future__ import annotations

import argparse
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DROID_ROOT = PROJECT_ROOT / "droid"
if str(DROID_ROOT) not in sys.path:
    sys.path.insert(0, str(DROID_ROOT))

from droid.camera_utils.camera_readers.zedx_camera import StreamZedCamera  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "scene_img"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Press a key, wait for the configured delay, and save an unmodified "
            "full-resolution frame from the right scene camera."
        )
    )
    parser.add_argument(
        "--key",
        default="s",
        help="Single key that triggers a photo (default: s).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Seconds to wait after the trigger key (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for captured JPEGs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--stream-ip", default="192.168.55.1")
    parser.add_argument("--stream-port", type=int, default=30002)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality in [1, 100] (default: 95).",
    )
    args = parser.parse_args()

    if len(args.key) != 1:
        parser.error("--key must be exactly one character")
    if args.key.lower() == "q":
        parser.error("--key cannot be q because q is reserved for quitting")
    if args.delay < 0:
        parser.error("--delay must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    return args


def read_key() -> str:
    """Read one key immediately, without requiring Enter."""
    if not sys.stdin.isatty():
        raise RuntimeError("This script must be run in an interactive terminal")

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(
            file_descriptor,
            termios.TCSADRAIN,
            previous_settings,
        )


def countdown(delay_seconds: float) -> None:
    deadline = time.monotonic() + delay_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(f"\r将在 {remaining:4.1f} 秒后拍照...", end="", flush=True)
        time.sleep(min(0.1, remaining))
    if delay_seconds > 0:
        print("\r正在拍照...              ", flush=True)


def capture_rgb_frame(camera: StreamZedCamera) -> np.ndarray:
    data = camera.read_camera()
    try:
        frame = data["image"]["right_cam"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("The right camera returned no image") from exc

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(f"Unexpected camera frame shape: {frame.shape}")
    return np.ascontiguousarray(frame[..., :3], dtype=np.uint8)


def save_rgb_jpeg(
    frame_rgb: np.ndarray,
    output_dir: Path,
    jpeg_quality: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = output_dir / f"scene_{timestamp}.jpg"
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    saved = cv2.imwrite(
        str(output_path),
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not saved:
        raise RuntimeError(f"Failed to save image: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    print(
        f"正在连接 right_cam ({args.stream_ip}:{args.stream_port})...",
        flush=True,
    )
    camera = StreamZedCamera(
        name="right_cam",
        stream_ip=args.stream_ip,
        stream_port=args.stream_port,
        is_hand_camera=False,
    )

    try:
        # Discard the first frame so the stream is warm before user capture.
        warmup_frame = capture_rgb_frame(camera)
        print(
            f"相机已连接，原始画面尺寸："
            f"{warmup_frame.shape[1]}x{warmup_frame.shape[0]}"
        )
        print(
            f"按 [{args.key}] 后等待 {args.delay:g} 秒拍照；按 [q] 退出。"
        )

        while True:
            key = read_key()
            if key.lower() == "q":
                print("退出。")
                break
            if key.lower() != args.key.lower():
                continue

            countdown(args.delay)
            frame_rgb = capture_rgb_frame(camera)
            output_path = save_rgb_jpeg(
                frame_rgb,
                output_dir,
                args.jpeg_quality,
            )
            print(
                f"已保存：{output_path} "
                f"({frame_rgb.shape[1]}x{frame_rgb.shape[0]})"
            )
            print(f"按 [{args.key}] 再拍一张；按 [q] 退出。")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
