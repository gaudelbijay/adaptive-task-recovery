#!/usr/bin/env python3
"""Render a montage of the contact-rich PegInsertion recovery mechanisms.

The tabletop benchmark has a montage; this one did not, so the benchmark that
*clears* the audit was the one nobody could see. That is backwards: the clear is
what makes the flag informative, and a reader should be able to watch the task
whose held-out mechanism is genuinely unsolved.

One panel per mechanism, each labelled with what is happening rather than with
an internal identifier. No policy is loaded -- these are the mechanisms acting
on the nominal controller's starting state, so the panels show what the
environment does, not how well anything responds to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.peg_insertion_recovery  # noqa: F401

# Ordered so the two ejections sit together and the two blockages sit together,
# which is the contrast the audit turns on.
PANELS = (
    ("positive_lateral_peg_ejection", "peg knocked aside"),
    ("negative_lateral_peg_ejection", "peg knocked the other way"),
    ("permanent_hole_block", "hole blocked, stays blocked"),
    ("temporary_hole_block", "hole blocked, clears again"),
)


def render(env, index: int) -> np.ndarray:
    image = env.render()
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[min(index, image.shape[0] - 1)]
    return image.astype(np.uint8)


def label(frame: np.ndarray, text: str, subtitle: str) -> np.ndarray:
    """Caption a frame. Falls back to the bare frame if PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    pad = 46
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + pad), (255, 255, 255))
    canvas.paste(Image.fromarray(frame), (0, pad))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        title_font = small_font = ImageFont.load_default()
    draw.text((10, 6), text, fill=(20, 20, 20), font=title_font)
    draw.text((10, 27), subtitle, fill=(110, 110, 110), font=small_font)
    return np.asarray(canvas)


def rollout(kind: str, caption: str, args) -> list[np.ndarray]:
    env = gym.make(
        "PegInsertionRecovery-v1", num_envs=args.num_envs, reconfiguration_freq=1,
        max_episode_steps=args.steps + 40, obs_mode="state", render_mode="rgb_array",
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        intervention_probability=1.0, intervention_types=(kind,),
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True,
                             record_metrics=False)
    env.reset(seed=args.seed)
    base = env.unwrapped
    onset = int(base._onset_step[args.index])

    frames = []
    action = torch.zeros((args.num_envs,) + env.single_action_space.shape,
                         device=base.device)
    for step in range(args.steps):
        env.step(action)
        if step % args.stride:
            continue
        phase = "before the disturbance" if step < onset else f"step {step}, after onset"
        frames.append(label(render(env, args.index), caption, phase))
    env.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("media/demos/peg-recovery-montage.gif"))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2,
                        help="Keep every Nth frame; the GIF is long enough to read.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=511_000_000)
    args = parser.parse_args()

    columns = [rollout(kind, caption, args) for kind, caption in PANELS]
    width = min(len(c) for c in columns)
    # Two by two: the ejections on the top row, the blockages on the bottom.
    frames = []
    for i in range(width):
        top = np.concatenate([columns[0][i], columns[1][i]], axis=1)
        bottom = np.concatenate([columns[2][i], columns[3][i]], axis=1)
        frames.append(np.concatenate([top, bottom], axis=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(args.output, frames, fps=args.fps, loop=0)
    print(f"wrote {args.output}  frames={len(frames)}  size={frames[0].shape}")


if __name__ == "__main__":
    main()
