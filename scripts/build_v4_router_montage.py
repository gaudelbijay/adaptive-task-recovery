#!/usr/bin/env python3
"""Build the README montage of the router on LearnedRecovery-v4.

Each panel is one captured episode with the router's selected option drawn per
frame, so the deferral behaviour is visible rather than asserted: the router
commits almost immediately on the mechanism a one-frame model can identify, and
waits tens of steps before committing on the permanent/temporary pair.

Every source episode must be a verified safe success with its provenance
record; the builder refuses to assemble a montage from a failed episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL_BAND = 42
INK = (18, 18, 18)
MUTED = (95, 94, 88)
SURFACE = (252, 252, 251)
ACCENT = {"defer": (140, 138, 130), "committed": (42, 120, 214)}


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_episode(record_path: Path):
    record = json.loads(record_path.read_text())
    if not record["safe_success"]:
        raise SystemExit(
            f"{record_path} is not a safe success; refusing to build a montage "
            "from a failed episode"
        )
    frames = imageio.mimread(record["video"], memtest=False)
    options = [record["option_names"][o] for o in record["selected_option_by_step"]]
    return record, frames, options


def annotate(frame, title, option, step, total, width):
    image = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
    scale = width / image.width
    image = image.resize((width, int(image.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (width, image.height + LABEL_BAND), SURFACE)
    canvas.paste(image, (0, LABEL_BAND))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), title, font=_font(15), fill=INK)
    deferring = option == "defer"
    colour = ACCENT["defer"] if deferring else ACCENT["committed"]
    caption = "observing…" if deferring else f"committed: {option}"
    draw.text((10, 24), caption, font=_font(13), fill=colour)
    draw.text((width - 62, 24), f"t={step}/{total}", font=_font(12), fill=MUTED)
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", action="append", nargs=2, metavar=("RECORD", "TITLE"),
                        required=True, help="Repeat: capture json and its panel title.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    panels = []
    for record_path, title in args.panel:
        record, frames, options = load_episode(Path(record_path))
        panels.append((title, record, frames, options))

    length = min(len(frames) for _, _, frames, _ in panels)
    montage = []
    for index in range(0, length, args.stride):
        row = []
        for title, _, frames, options in panels:
            option = options[min(index, len(options) - 1)]
            row.append(annotate(
                frames[index], title, option, index, length - 1, args.panel_width,
            ))
        height = max(cell.shape[0] for cell in row)
        row = [
            np.pad(cell, ((0, height - cell.shape[0]), (0, 0), (0, 0)),
                   constant_values=252)
            for cell in row
        ]
        montage.append(np.concatenate(row, axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, montage, fps=args.fps, loop=0)
    print(f"wrote {args.output} ({len(montage)} frames, {len(panels)} panels)")
    for title, record, _, options in panels:
        commit = next(
            (i + 1 for i, o in enumerate(options) if o != "defer"), None
        )
        print(f"  {title}: commits at step {commit}, safe_success={record['safe_success']}")


if __name__ == "__main__":
    main()
