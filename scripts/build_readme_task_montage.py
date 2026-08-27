#!/usr/bin/env python3
"""Build the README montage from real Fetch and frozen-policy recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _gif_frames(path: Path) -> list[Image.Image]:
    image = Image.open(path)
    frames = []
    for index in range(image.n_frames):
        image.seek(index)
        frames.append(image.convert("RGB"))
    return frames


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
    overlay_height = 30
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, size[1] - overlay_height, size[0], size[1]), fill=(0, 0, 0, 190))
    font = _font(15)
    box = draw.textbbox((0, 0), label, font=font)
    x = (size[0] - (box[2] - box[0])) // 2
    draw.text((x, size[1] - overlay_height + 6), label, font=font, fill=(255, 255, 255, 255))
    return panel


def _index(output_index: int, output_frames: int, source_frames: int, hold_fraction: float) -> int:
    moving_frames = max(2, round(output_frames * (1.0 - hold_fraction)))
    if output_index >= moving_frames:
        return source_frames - 1
    return round(output_index / (moving_frames - 1) * (source_frames - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", default="media/demos/fetch-real-pick-and-place.gif")
    parser.add_argument("--videos", default="results/manipulation_ppo/videos")
    parser.add_argument("--output", default="media/demos/manipulation-task-montage.gif")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    video_root = Path(args.videos)
    fetch_frames = _gif_frames(Path(args.fetch))
    fetch_frames = [frame.crop((0, 0, frame.width, frame.height - 52)) for frame in fetch_frames]
    sources = [
        (fetch_frames, "Fetch: physical can pick/place", 0.0),
        (_video_frames(video_root / "PickCube.mp4"), "Panda: PickCube", 0.45),
        (_video_frames(video_root / "PickSingleYCB.mp4"), "Panda: randomized YCB", 0.2),
        (_video_frames(video_root / "UnitreeG1PlaceAppleInBowl.mp4"), "Unitree G1: apple in bowl", 0.4),
    ]
    panel_size = (360, 270)
    montage_frames = []
    for output_index in range(args.frames):
        canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * 2), (8, 10, 12))
        for source_index, (frames, label, hold_fraction) in enumerate(sources):
            frame = frames[_index(output_index, args.frames, len(frames), hold_fraction)]
            panel = _panel(frame, label, panel_size)
            canvas.paste(panel, ((source_index % 2) * panel_size[0], (source_index // 2) * panel_size[1]))
        montage_frames.append(canvas)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    montage_frames[0].save(
        output,
        save_all=True,
        append_images=montage_frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {output} ({len(montage_frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
