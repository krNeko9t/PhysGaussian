"""Stage 5: Video compilation via ffmpeg."""

from __future__ import annotations

import os
import subprocess

from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def compile_video(
    output_dir: str,
    frame_dt: float,
    frame_num: int,
) -> None:
    """Compile rendered frames into an MP4 video using ffmpeg."""
    fps = int(1.0 / frame_dt)
    sample = os.path.join(output_dir, "0000.png")
    if not os.path.isfile(sample):
        LOGGER.warning("No rendered frames found, skip video compilation: %s", sample)
        return

    import cv2
    img = cv2.imread(sample)
    if img is None:
        LOGGER.error("Cannot read first frame for video compilation: %s", sample)
        return
    height = img.shape[0] // 2 * 2
    width = img.shape[1] // 2 * 2

    cmd = (
        f"ffmpeg -framerate {fps} -i {output_dir}/%04d.png "
        f"-c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p "
        f"{output_dir}/output.mp4"
    )
    LOGGER.info("Compiling video with ffmpeg: %s", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while compiling video. "
            f"exit_code={result.returncode}, stderr={result.stderr.strip()}"
        )
    LOGGER.info("Video saved to %s/output.mp4", output_dir)
