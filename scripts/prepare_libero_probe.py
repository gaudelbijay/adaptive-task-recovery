#!/usr/bin/env python3
"""Extract features for the LIBERO identifiability audit.

LIBERO-Goal holds scene and objects fixed and varies only the goal, so task
identity should be recoverable only from the language instruction. This script
extracts what a policy could see *without* language:

  * the initial agentview frame (t=0), embedded with DINO ViT-S/16;
  * an order-free summary over an early prefix;
  * the demonstration length, which needs no vision at all.

libero_spatial is the positive control: its variation is spatial, so a visual
control is expected to identify it.
"""
from __future__ import annotations
import argparse, glob, json, os
import h5py, numpy as np, torch
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/atr-libero/data"))
    ap.add_argument("--suites", nargs="+", default=["libero_object", "libero_10"])
    ap.add_argument("--prefix-frames", type=int, default=8,
                    help="Frames from the episode start for the order-free rung.")
    ap.add_argument("--out", default=os.path.expanduser("~/atr-libero/features"))
    args = ap.parse_args()

    import timm
    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)
    model.eval().cuda()
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    os.makedirs(args.out, exist_ok=True)
    for suite in args.suites:
        files = sorted(glob.glob(os.path.join(args.root, suite, "*.hdf5")))
        first, summ, lens, labels, names = [], [], [], [], []
        for task_id, f in enumerate(files):
            names.append(os.path.basename(f).replace("_demo.hdf5", ""))
            with h5py.File(f, "r") as h:
                for demo in h["data"].keys():
                    rgb = h["data"][demo]["obs"]["agentview_rgb"]
                    T = rgb.shape[0]
                    idx = [0] + list(np.linspace(0, min(T - 1, 40), args.prefix_frames,
                                                 dtype=int))
                    # LIBERO stores frames bottom-up; flip so DINO sees them upright.
                    batch = torch.stack([tf(Image.fromarray(rgb[i][::-1].copy()))
                                         for i in idx]).cuda()
                    with torch.no_grad():
                        emb = model(batch).cpu().numpy()
                    first.append(emb[0])
                    pre = emb[1:]
                    summ.append(np.concatenate([pre.mean(0), pre.std(0)]))
                    lens.append(T)
                    labels.append(task_id)
            print(f"  {suite} {names[-1][:52]:<52} done", flush=True)
        np.savez_compressed(
            os.path.join(args.out, f"{suite}.npz"),
            first=np.array(first, dtype=np.float32),
            summary=np.array(summ, dtype=np.float32),
            length=np.array(lens, dtype=np.int32),
            label=np.array(labels, dtype=np.int32),
            task_names=np.array(names))
        print(f"{suite}: {len(labels)} demos, first={np.array(first).shape}")

if __name__ == "__main__":
    main()
