"""506b finding #1 (codex): the original issue says "C96 worse than C48".
The cross-grid closure used C48-only.  This runs the SAME off-(m=4) breakdown
metric on the cube at C48 AND C96 (and C192 if it fits) with the runner's own
resolution-scaled dt/diffusion policy, to test whether the cube has a
resolution-DEPENDENT seam/GCL pathology (C96 breaks markedly earlier) or just
the usual less-diffusion-at-higher-res shift shared by every grid.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "_atm_matrix", _ROOT / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
m = importlib.util.module_from_spec(_spec); sys.modules["_atm_matrix"] = m
_spec.loader.exec_module(m)
# reuse the cross-grid off-m4 metric (no duplicated numerics)
_spec2 = importlib.util.spec_from_file_location(
    "_cg", _ROOT / "scripts" / "tmp" / "diagnostic" / "diag_506b_crossgrid_case6.py")
cg = importlib.util.module_from_spec(_spec2); sys.modules["_cg"] = cg
_spec2.loader.exec_module(cg)

OUT = _ROOT / "results" / "diag_506b_cube_scaling"; OUT.mkdir(parents=True, exist_ok=True)
DAYS = 40.0
runs = [("C48", "C48"), ("C96", "C96")]

print(f"# Case-6 cube resolution scaling  (W6, {DAYS:.0f} d)")
for label, res in runs:
    odir = OUT / label; odir.mkdir(parents=True, exist_ok=True)
    tc = m.TestCase(equation_set="shallow_water", case="williamson6",
                    grid_type="cubed_sphere", resolution=res,
                    vertical_coord="none", duration_days=DAYS, quick_days=1.0,
                    run_kwargs={"test_num": 6})
    try:
        st, wall, notes = m.run_shallow_water(tc, odir, DAYS)
        print(f"  {label:6s} {st:6s} [{wall:.0f}s] {notes}")
    except Exception as e:
        print(f"  {label:6s} FAILED {type(e).__name__}: {e}")

print("\nlabel   day @ off-m4>1e-2   final off-m4")
curves = {}
for label, res in runs:
    npz = OUT / label / "snapshots_latlon.npz"
    if not npz.exists():
        print(f"  {label:6s} (no data)"); continue
    d = np.load(npz)
    key = "height" if "height" in d.files else [k for k in d.files
          if k not in ("lat", "lon", "times_days", "steps")][0]
    t = d["times_days"]; arr = d[key]
    fr = np.array([cg.off_m4_fraction(arr[i]) for i in range(arr.shape[0])])
    curves[label] = (t, fr)
    idx = np.argmax(fr > 1e-2) if (fr > 1e-2).any() else -1
    dc = f"{t[idx]:.1f}" if idx >= 0 else ">40 (none)"
    print(f"  {label:6s} {dc:>14s}      {fr[-1]:.3e}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for label, (t, fr) in curves.items():
        plt.semilogy(t, np.maximum(fr, 1e-16), "o-", label=f"cube_{label}")
    plt.axhline(1e-2, ls="--", c="k", lw=0.8, label="breakdown onset")
    plt.xlabel("day"); plt.ylabel("off-(m=4) power fraction")
    plt.title(f"Case-6 cube resolution scaling ({DAYS:.0f}d)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p = OUT / "cube_scaling_breakfrac.png"; plt.savefig(p, dpi=110)
    print(f"\nplot -> {p}")
except Exception as e:
    print(f"plot skipped: {e}")
