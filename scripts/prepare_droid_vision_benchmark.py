#!/usr/bin/env python3
"""Rebuild the DROID benchmark with vision, so the audit gets a fair test.

The proprioception-only version was reported as inconclusive: no rung reached
0.62 macro-AUROC, and a tie between two near-trivial models says nothing about
whether the task needs temporal structure. That weakness is ours, not DROID's --
whether an episode succeeds usually depends on the objects and the scene, which
proprioception cannot see.

This gives every rung the scene. A frozen image encoder embeds evenly spaced
frames from the exterior camera, and those embeddings are concatenated with the
proprioception features at the same timesteps. The ladder is unchanged: all
rungs receive the identical tensor and differ only in what they may extract from
it, so if the models now clear the competence precondition, the verdict becomes
interpretable.

A proprioception-only tensor at the same timesteps is written alongside, so any
change can be attributed to vision rather than to the shorter horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from huggingface_hub import HfFileSystem

REPO = "cadene/droid_1.0.1"
CAMERA = "observation.images.exterior_1_left"
COLUMNS = ["observation.state", "action", "building", "is_episode_successful"]


def build_encoder(name: str, device):
    import timm
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    return model, transform


def read_columns(path: str, frames: int, filesystem: HfFileSystem):
    try:
        table = pq.read_table(path, columns=COLUMNS, filesystem=filesystem)
    except Exception:
        return None
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or len(state) < frames or action.shape[0] != state.shape[0]:
        return None
    s, a = state[:frames], action[:frames, : state.shape[1]]
    ds = np.diff(s, axis=0, prepend=s[:1])
    da = np.diff(a, axis=0, prepend=a[:1])
    proprio = np.concatenate((s - s[:1], a - s[:1], a - s, ds + 0.5 * da), axis=1)
    return (str(table["building"][0].as_py()),
            bool(table["is_episode_successful"][0].as_py()),
            proprio.astype(np.float32))


def fetch_video(episode_path: str, cache: Path) -> Path | None:
    """Download one episode's exterior-camera clip. They are ~1.2 MB each."""
    rel = episode_path.split(f"datasets/{REPO}/", 1)[1]
    chunk = rel.split("/")[1]
    name = rel.rsplit("/", 1)[-1].replace(".parquet", ".mp4")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/videos/{chunk}/{CAMERA}/{name}"
    out = cache / f"{chunk}_{name}"
    if out.exists() and out.stat().st_size:
        return out
    try:
        with urllib.request.urlopen(url, timeout=120) as response, out.open("wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
        return out
    except Exception:
        return None


def sample_frames(path: Path, indices) -> np.ndarray | None:
    """Decode the requested frames.

    DROID's clips are AV1. The OpenCV build here cannot decode AV1 -- it reports
    "Failed to get pixel format" and returns nothing -- so decoding goes through
    the ffmpeg binary bundled with imageio-ffmpeg, which carries libdav1d.
    OpenCV remains as a fallback for any clip in a codec it does handle.
    """
    wanted = set(int(x) for x in indices)
    frames: dict[int, np.ndarray] = {}
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(str(path), "ffmpeg")
        for i, frame in enumerate(reader):
            if i in wanted:
                frames[i] = np.asarray(frame)
            if i >= max(wanted) or len(frames) == len(wanted):
                break
        reader.close()
    except Exception:
        frames = {}
    if frames:
        if len(frames) < len(wanted):
            last = frames[max(frames)]
            for k in wanted - set(frames):
                frames[k] = last
        return np.stack([frames[int(k)] for k in indices])
    return _sample_frames_opencv(path, indices)


def _sample_frames_opencv(path: Path, indices) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    wanted, frames, i = set(int(x) for x in indices), {}, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
        if len(frames) == len(wanted):
            break
    capture.release()
    if len(frames) < len(wanted):
        if not frames:
            return None
        last = frames[max(frames)]
        for k in wanted - set(frames):
            frames[k] = last          # pad short clips with their final frame
    return np.stack([frames[int(k)] for k in indices])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-prefix", default="results/droid/droid_vision_v1")
    parser.add_argument("--shards", type=int, default=6000)
    parser.add_argument("--frames", type=int, default=128, help="Prefix length in the parquet.")
    parser.add_argument("--steps", type=int, default=32, help="Frames embedded per episode.")
    parser.add_argument("--families", type=int, default=10)
    parser.add_argument("--min-per-class", type=int, default=15)
    # DROID's exterior clips are 180x320, so a 518-pixel DINOv2 would upsample
    # threefold at several times the cost for no extra detail. DINO ViT-S/16 at
    # 224 is self-supervised, scene-appropriate, and matched to the source.
    parser.add_argument("--encoder", default="vit_small_patch16_224.dino")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"datasets/{REPO}/data/**/*.parquet"))
    step = max(1, len(shards) // args.shards)
    sample = shards[::step][: args.shards]
    print(f"{len(shards)} shards; reading {len(sample)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        columns = list(pool.map(lambda p: read_columns(p, args.frames, filesystem), sample))
    usable = [(p, c) for p, c in zip(sample, columns) if c is not None]
    print(f"episodes with >= {args.frames} frames: {len(usable)}")

    counts = defaultdict(lambda: [0, 0])
    for _, (building, label, _) in usable:
        counts[building][int(label)] += 1
    eligible = [b for b, (bad, good) in counts.items()
                if bad >= args.min_per_class and good >= args.min_per_class]
    eligible.sort(key=lambda b: -sum(counts[b]))
    keep = eligible[: args.families]
    if len(keep) < 5:
        raise SystemExit(f"only {len(keep)} usable families")
    index = {b: i for i, b in enumerate(keep)}
    selected = [(p, c) for p, c in usable if c[0] in index]
    print(f"selected {len(selected)} episodes over {len(keep)} families")

    encoder, transform = build_encoder(args.encoder, device)
    picks = np.linspace(0, args.frames - 1, args.steps).round().astype(int)

    visual, proprio, labels, families, missing = [], [], [], [], 0
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            paths = list(pool.map(lambda pc: fetch_video(pc[0], cache), selected))
        for n, ((_, (building, label, prop)), video) in enumerate(zip(selected, paths)):
            arr = sample_frames(video, picks) if video else None
            if arr is None:
                missing += 1
                if missing >= 20 and not visual:
                    raise SystemExit(
                        f"{missing} clips fetched and none decoded; the decoder cannot "
                        "read this codec. Aborting before downloading the rest.")
                continue
            with torch.no_grad():
                # timm's transform pipeline starts from a PIL image: handing it a
                # uint8 tensor reaches Normalize as an integer type and raises.
                batch = torch.stack([transform(Image.fromarray(f)) for f in arr]).to(device)
                embedding = encoder(batch).float().cpu().numpy()
            visual.append(embedding.astype(np.float32))
            proprio.append(prop[picks])
            labels.append(int(label))
            families.append(index[building])
            if video and video.exists():
                os.remove(video)
            if (n + 1) % 250 == 0:
                print(f"  {n+1}/{len(selected)} encoded, {missing} without video")

    label = np.asarray(labels, dtype=np.int64)
    family = np.asarray(families, dtype=np.int64)
    episode = np.arange(len(labels), dtype=np.int64)
    prop_stack = np.stack(proprio)
    vis_stack = np.stack(visual)
    combined = np.concatenate([vis_stack, prop_stack], axis=2)

    for tag, tensor in (("", combined), ("_proprio_only", prop_stack)):
        out = Path(f"{args.out_prefix}{tag}.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, sequence=tensor, label=label,
                            object_id=family, episode_id=episode)
        print(f"wrote {out}  {tensor.shape}")

    audit = {
        "schema_version": 1, "benchmark": "DROID-success-vision",
        "protocol": "leave-one-building-out episode-success prediction from a visual prefix",
        "claim_boundary": (
            "Offline outcome prediction on externally collected real-robot data, "
            "using a frozen image encoder. Not a control result."
        ),
        "encoder": args.encoder, "camera": CAMERA,
        "steps": args.steps, "prefix_frames": args.frames,
        "episodes": int(len(label)), "episodes_without_video": missing,
        "failures": int((label == 0).sum()), "successes": int((label == 1).sum()),
        "visual_dim": int(vis_stack.shape[-1]), "proprio_dim": int(prop_stack.shape[-1]),
        "families": index,
        "objects": [b for b, _ in sorted(index.items(), key=lambda kv: kv[1])],
        "per_family": {b: {"failure": counts[b][0], "success": counts[b][1]} for b in keep},
    }
    Path(f"{args.out_prefix}.audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\n{audit['episodes']} episodes, visual {audit['visual_dim']}d + "
          f"proprio {audit['proprio_dim']}d over {args.steps} steps, {len(keep)} families")
    print(f"  {audit['failures']} failures / {audit['successes']} successes")


if __name__ == "__main__":
    main()
