"""Stage 5: Video compilation via ffmpeg."""

from __future__ import annotations

import os


def compile_video(
    output_dir: str,
    frame_dt: float,
    frame_num: int,
) -> None:
    """Compile rendered frames into an MP4 video using ffmpeg."""
    fps = int(1.0 / frame_dt)
    sample = os.path.join(output_dir, "0000.png")
    if not os.path.isfile(sample):
        print("No rendered frames found, skipping video compilation.")
        return

    import cv2
    img = cv2.imread(sample)
    if img is None:
        print(f"Cannot read {sample}, skipping video compilation.")
        return
    height = img.shape[0] // 2 * 2
    width = img.shape[1] // 2 * 2

    cmd = (
        f"ffmpeg -framerate {fps} -i {output_dir}/%04d.png "
        f"-c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p "
        f"{output_dir}/output.mp4"
    )
    print(f"Compiling video: {cmd}")
    os.system(cmd)
    print(f"Video saved to {output_dir}/output.mp4")
