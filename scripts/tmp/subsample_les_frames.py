"""Keep, per single-column step, the LES frame nearest that step (user 2026-10-30).

The SCM steps every ``--dt`` seconds (>= 10 min, PI 2026-10-28). An LES whose frames
are closer than that (cf06_q30_g01_ns: 108 s over 0.33 h) collides on one SCM step and
scoring refuses it. This writes a NEW artifact holding the t=0 frame plus, for each
step k = 1..round(t_last/dt), the frame nearest k*dt (offset must be <= dt/2, the
scoring's own nearest-step rule). Every kept array slice is byte-identical to the
source; the source file is not touched. Cumulative series stay valid (they integrate
from t=0). The record gains les_setup["frame_subsample"] (source, its sha256, kept
times).

usage: subsample_les_frames.py SRC DST --dt 600
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from legoesm.atmosphere.les_suite.bridge import load_artifact

p = argparse.ArgumentParser()
p.add_argument("src", type=Path)
p.add_argument("dst", type=Path)
p.add_argument("--dt", type=float, required=True)
a = p.parse_args()
if not (np.isfinite(a.dt) and a.dt > 0):
    raise SystemExit(f"--dt must be finite and positive, got {a.dt}")
if not __debug__:
    raise SystemExit("run without -O: the output checks below are asserts")
if a.dst.suffix != ".npz" or a.dst.resolve() == a.src.resolve():
    raise SystemExit("DST must be a new .npz path (numpy appends .npz otherwise)")
if a.dst.exists():
    raise SystemExit(f"{a.dst} exists; refusing to overwrite")

with np.load(a.src, allow_pickle=False) as z:
    arrays = {k: z[k] for k in z.files}
meta = json.loads(bytes(arrays["__meta__"]).decode())
t = arrays["times_s"].astype(np.float64)
nt, nz = t.size, arrays["heights_m"].size
if nt == nz:
    raise SystemExit("nt == nz: time-indexed arrays cannot be told apart by shape")
if t[0] != 0.0:
    raise SystemExit(f"first frame is at t={t[0]} s, not 0")

keep, offsets = [0], []
for k in range(1, int(round(t[-1] / a.dt)) + 1):
    i = int(np.argmin(np.abs(t - k * a.dt)))
    off = t[i] - k * a.dt
    if abs(off) > a.dt / 2 or i == 0:
        raise SystemExit(f"no frame within dt/2 of step {k} (nearest t={t[i]} s)")
    if int(round(t[i] / a.dt)) != k:      # scoring's own nearest-step rounding
        raise SystemExit(f"frame t={t[i]} s would not score on step {k}")
    keep.append(i)
    offsets.append(off)
if len(keep) < 2:
    raise SystemExit("no frame after t=0 would be kept")
if len(set(keep)) != len(keep):
    raise SystemExit(f"two steps share one frame: {keep}")

out = {}
for k, v in arrays.items():
    if k != "__meta__" and v.ndim >= 1 and v.shape[0] == nt:
        out[k] = v[keep]
    else:
        out[k] = v
setup = meta.get("les_setup")
if not isinstance(setup, dict) or "frame_subsample" in setup:
    raise SystemExit("record has no les_setup dict, or is already subsampled")
setup["frame_subsample"] = {
    "source": str(a.src.resolve()),
    "source_sha256": hashlib.sha256(a.src.read_bytes()).hexdigest(),
    "scm_dt_s": a.dt,
    "kept_times_s": [float(t[i]) for i in keep],
    "note": ("nearest LES frame to each SCM step (user 2026-10-30); the score is "
             "a nearest-step score on these frames only"),
}
out["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
np.savez_compressed(a.dst, **out)

load_artifact(a.dst)                       # the artifact's own validation
with np.load(a.dst, allow_pickle=False) as z:
    assert set(z.files) == set(arrays), set(z.files) ^ set(arrays)
    for k in arrays:
        if k == "__meta__":
            continue
        ref = out[k]
        assert z[k].dtype == ref.dtype and z[k].shape == ref.shape, k
        assert z[k].tobytes() == ref.tobytes(), k
    m2 = json.loads(bytes(z["__meta__"]).decode())
m0 = json.loads(bytes(arrays["__meta__"]).decode())
assert {k: v for k, v in m2.items() if k != "les_setup"} == {
    k: v for k, v in m0.items() if k != "les_setup"}
assert {k: v for k, v in m2["les_setup"].items() if k != "frame_subsample"} == m0["les_setup"]
sliced = sorted(k for k, v in arrays.items() if k != "__meta__" and v.ndim >= 1
                and v.shape[0] == nt)
print(f"wrote {a.dst}: frames {nt} -> {len(keep)}, kept t = "
      f"{[float(t[i]) for i in keep]} s, offsets from steps = {offsets} s; "
      f"time-sliced arrays {len(sliced)}: {sliced}")
