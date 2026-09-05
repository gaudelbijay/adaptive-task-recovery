"""Is the perfect identification real, or an artefact?

Three checks: whether initial states vary within a task at all, whether a
trivial pixel feature reproduces it, and whether the DINO result survives on
raw downsampled pixels rather than a learned embedding.
"""
import glob, os
import h5py, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

for suite in ("libero_object", "libero_goal"):
    files = sorted(glob.glob(os.path.expanduser(f"~/atr-libero/data/{suite}/*.hdf5")))
    X, y, within = [], [], []
    for tid, f in enumerate(files):
        with h5py.File(f, "r") as h:
            frames = []
            for demo in list(h["data"].keys()):
                img = h["data"][demo]["obs"]["agentview_rgb"][0].astype(np.float32)
                small = img[::8, ::8, :].reshape(-1) / 255.0   # 16x16x3 = 768 dims
                frames.append(small)
                X.append(small); y.append(tid)
            fr = np.stack(frames)
            within.append(fr.std(axis=0).mean())
    X = np.array(X); y = np.array(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    acc = cross_val_score(clf, X, y, cv=5).mean()
    print(f"\n=== {suite} ===")
    print(f"  mean within-task pixel std across 50 demos : {np.mean(within):.4f}")
    print(f"     (0 would mean every demo starts identically)")
    print(f"  per-task within std, first 5              : "
          f"{[round(float(w),4) for w in within[:5]]}")
    print(f"  10-way accuracy from RAW 16x16 pixels      : {acc:.3f}")
