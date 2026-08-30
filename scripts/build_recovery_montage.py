#!/usr/bin/env python3
"""Build a compact README GIF from three learned-recovery recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _video_frames(path: Path) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    try:
        return [Image.fromarray(frame).convert("RGB") for frame in reader]
    finally:
        reader.close()


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _panel(frame: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (18, 20, 24))
    fitted = ImageOps.contain(frame, size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, size[1] - 34, size[0], size[1]), fill=(0, 0, 0, 195))
    font = _font(14)
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(((size[0] - box[2] + box[0]) // 2, size[1] - 28), label, font=font, fill="white")
    return panel


def _sample(frames: list[Image.Image], index: int, output_frames: int) -> Image.Image:
    return frames[round(index / (output_frames - 1) * (len(frames) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="results/learned_recovery/videos")
    parser.add_argument("--method", default="safe_adaptive_ppo")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="media/demos/learned-recovery-montage.gif")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--strict-removal-labels", action="store_true",
        help="Use removed-goal labels only for captures that verify actual unavailability",
    )
    args = parser.parse_args()

    root = Path(args.videos)
    sources = [
        (
            "first_goal_removed",
            "Recover: first goal removed" if args.strict_removal_labels
            else "Sweeper targets first goal",
        ),
        (
            "second_goal_removed",
            "Recover: second goal removed" if args.strict_removal_labels
            else "Sweeper targets second goal",
        ),
        ("nominal", "Nominal: complete both goals"),
    ]
    videos = [
        (_video_frames(root / f"{args.method}_{args.seed}_{branch}.mp4"), label)
        for branch, label in sources
    ]
    panel_size = (320, 320)
    montage = []
    for index in range(args.frames):
        canvas = Image.new("RGB", (panel_size[0] * 3, panel_size[1]), (8, 10, 12))
        for panel_index, (frames, label) in enumerate(videos):
            canvas.paste(
                _panel(_sample(frames, index, args.frames), label, panel_size),
                (panel_index * panel_size[0], 0),
            )
        montage.append(canvas)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    montage[0].save(
        output, save_all=True, append_images=montage[1:],
        duration=round(1000 / args.fps), loop=0, optimize=True, disposal=2,
    )
    print(f"wrote {output} ({len(montage)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
