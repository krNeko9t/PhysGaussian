"""Stage 5: Video compilation via ffmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)
_DEBUG_LOG_PATH = "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/PhysGaussian/.cursor/debug-e9c2ff.log"
_DEBUG_SESSION_ID = "e9c2ff"


def _agent_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


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

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                ffmpeg_bin = candidate
                break
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg executable not found. Install ffmpeg or add it to PATH. "
            f"PATH={os.environ.get('PATH', '')}"
        )

    output_mp4 = os.path.join(output_dir, "output.mp4")
    cmd = [
        ffmpeg_bin,
        "-framerate",
        str(fps),
        "-i",
        os.path.join(output_dir, "%04d.png"),
        "-c:v",
        "libx264",
        "-s",
        f"{width}x{height}",
        "-y",
        "-pix_fmt",
        "yuv420p",
        output_mp4,
    ]
    # region agent log
    _agent_debug_log(
        run_id="run2",
        hypothesis_id="N5",
        location="physics_sim/stages/video.py:compile_video:before_ffmpeg",
        message="ffmpeg command prepared",
        data={
            "fps": int(fps),
            "width": int(width),
            "height": int(height),
            "ffmpeg_bin": ffmpeg_bin,
        },
    )
    # endregion
    LOGGER.info("Compiling video with ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    # region agent log
    _agent_debug_log(
        run_id="run2",
        hypothesis_id="N5",
        location="physics_sim/stages/video.py:compile_video:after_ffmpeg",
        message="ffmpeg finished",
        data={"returncode": int(result.returncode), "stderr_len": len(result.stderr or "")},
    )
    # endregion
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while compiling video. "
            f"exit_code={result.returncode}, stderr={result.stderr.strip()}"
        )
    LOGGER.info("Video saved to %s", output_mp4)
