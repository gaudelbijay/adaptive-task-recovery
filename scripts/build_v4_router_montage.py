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

LABEL_BAND = 68
TRACK_H = 5
INK = (18, 18, 18)
MUTED = (95, 94, 88)
SURFACE = (252, 252, 251)
ACCENT = {"defer": (140, 138, 130), "committed": (42, 120, 214)}
TRACK_BG = (226, 226, 221)
ONSET = (179, 64, 31)      # the step the intervention fires
RESOLVED = (47, 107, 87)


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
    # The evaluator stops scoring at first resolution, so anything after it is
    # unmeasured. Hold the resolved frame rather than showing the policy
    # continuing to act with no task signal.
    resolution = record.get("resolution_step")
    if resolution is not None and resolution < len(frames):
        frames = frames[: resolution + 1] + [frames[resolution]] * (
            len(frames) - resolution - 1
        )
        options = options[:resolution] + [options[min(resolution, len(options) - 1)]] * (
            len(options) - resolution
        )
    return record, frames, options


def _goal_label(record):
    """Summarise how the episode resolved: placed, or written off."""
    goals = record.get("goals_at_resolution")
    if not goals:
        return None
    done = int(goals.get("goals_completed") or 0)
    gone = int(goals.get("goals_unavailable") or 0)
    return f"{done}/2 placed" if not gone else f"{done}/2 placed, {gone} gone"


def annotate(frame, title, option, step, total, width,
             resolved_at=None, commit_at=None, onset=None, goals=None):
    """Draw one panel: the render, its state caption, and a progress track.

    The track is what makes the panels comparable at a glance. It marks when
    the intervention fires, how long the router observes before committing,
    and where the episode resolves, so the deferral asymmetry is visible
    without reading the timestamps.
    """
    image = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
    scale = width / image.width
    image = image.resize((width, int(image.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (width, image.height + LABEL_BAND), SURFACE)
    canvas.paste(image, (0, LABEL_BAND))
    draw = ImageDraw.Draw(canvas)

    draw.text((10, 5), title, font=_font(15), fill=INK)
    deferring = option == "defer"
    colour = ACCENT["defer"] if deferring else ACCENT["committed"]
    shown = "temporary" if option == "temporary_recovery" else option
    caption = "observing\u2026" if deferring else f"committed: {shown}"
    draw.text((10, 24), caption, font=_font(13), fill=colour)

    if resolved_at is not None and step >= resolved_at:
        outcome = f"resolved t={resolved_at}"
        if goals:
            outcome += f"  \u00b7  {goals}"
        fill = RESOLVED
    else:
        outcome = f"step {step} of {total}"
        fill = MUTED
    draw.text((10, 41), outcome, font=_font(12), fill=fill)

    # Progress track with intervention, commit and resolution markers.
    x0, x1, y = 10, width - 10, LABEL_BAND - TRACK_H - 3
    span = max(total, 1)
    draw.rectangle([x0, y, x1, y + TRACK_H], fill=TRACK_BG)
    filled = x0 + int((x1 - x0) * min(step, span) / span)
    draw.rectangle([x0, y, filled, y + TRACK_H], fill=colour)
    for at, mark in ((onset, ONSET), (commit_at, ACCENT["committed"]),
                     (resolved_at, RESOLVED)):
        if at is None or at > span:
            continue
        mx = x0 + int((x1 - x0) * at / span)
        draw.rectangle([mx - 1, y - 3, mx + 1, y + TRACK_H + 3], fill=mark)
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", action="append", nargs=2, metavar=("RECORD", "TITLE"),
                        required=True, help="Repeat: capture json and its panel title.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--hold-frames", type=int, default=24,
        help="Frames to hold after the last panel resolves. Scoring stops at\n             resolution, so the montage ends there rather than showing\n             unmeasured post-resolution motion.",
    )
    args = parser.parse_args()

    panels = []
    for record_path, title in args.panel:
        record, frames, options = load_episode(Path(record_path))
        panels.append((title, record, frames, options))

    resolutions = [
        r.get("resolution_step") for _, r, _, _ in panels
        if r.get("resolution_step") is not None
    ]
    length = min(len(frames) for _, _, frames, _ in panels)
    if resolutions:
        length = min(length, max(resolutions) + args.hold_frames)
    montage = []
    for index in range(0, length, args.stride):
        row = []
        for title, record, frames, options in panels:
            option = options[min(index, len(options) - 1)]
            commit = next(
                (i + 1 for i, o in enumerate(options) if o != "defer"), None
            )
            row.append(annotate(
                frames[index], title, option, index, length - 1,
                args.panel_width, record.get("resolution_step"), commit,
                record.get("onset_step", 1), _goal_label(record),
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
