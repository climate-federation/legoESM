"""506b decisive test: is the Case-6 (Rossby-Haurwitz wave-4) symmetry
breakdown a CUBE artifact or the intrinsic barotropic instability?

If the wave-4 -> off-wave-4 power leak grows at a SIMILAR day on the
non-cube grids (spectral / icosahedral / lat-lon) as on the cube, the
breakdown is physical (literature: RH wave-4 is barotropically unstable,
Thuburn & Li 2000) and NO cube-specific rewrite can "fix" it without
suppressing real dynamics.  If the cube leaks markedly EARLIER, a residual
cube/GCL artifact remains.

Reuses the SAME ``run_shallow_water`` path on every grid (no duplicated
numerics); reads the saved ``snapshots_latlon.npz`` (181x360) and measures
the fraction of longitudinal power OUTSIDE multiples of m=4.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_MATRIX = _ROOT / "scripts" / "matrix" / "run_atmosphere_test_matrix.py"
_spec = importlib.util.spec_from_file_location("_atm_matrix", _MATRIX)
m = importlib.util.module_from_spec(_spec)
sys.modules["_atm_matrix"] = m
_spec.loader.exec_module(m)

OUT = _ROOT / "results" / "diag_506b_crossgrid"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = 40.0

# matched ~1.9-2.0 deg resolutions across grids
runs = [
    ("cube_C48", "cubed_sphere", "C48"),
    ("spectral_T63", "spectral", "T63"),
    ("ico_ico6", "icosahedral", "ico6"),
    ("latlon_96x192", "latlon", "96x192"),
]


def off_m4_fraction(h2d):
    """Fraction of longitudinal spectral power outside m in {0,4,8,12,...}
    of a (nlat, nlon) height field, area-weighted in latitude.

    Wave-4 RH solution has power ONLY at m multiples of 4; any growth of
    off-multiple-of-4 power == symmetry/structure breakdown.
    """
    nlat, nlon = h2d.shape
    lat = np.linspace(-90.0, 90.0, nlat)
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()
    f = np.fft.rfft(h2d - h2d.mean(axis=1, keepdims=True), axis=1)
    p = (np.abs(f) ** 2)                       # (nlat, nfreq)
    mm = np.arange(p.shape[1])
    on4 = (mm % 4 == 0)
    p_lat = (w[:, None] * p).sum(axis=0)       # latitude-weighted power per m
    tot = p_lat.sum()
    if tot <= 0:
        return 0.0
    return float(p_lat[~on4].sum() / tot)


def analyze(label):
    npz = OUT / label / "snapshots_latlon.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    key = "height" if "height" in d.files else (
        "h" if "h" in d.files else None)
    if key is None:
        flds = [k for k in d.files if k not in ("lat", "lon", "times_days", "steps")]
        key = flds[0] if flds else None
    if key is None:
        return None
    t = d["times_days"]
    arr = d[key]                               # (n_times, nlat, nlon)
    fr = np.array([off_m4_fraction(arr[i]) for i in range(arr.shape[0])])
    return t, fr, key


if __name__ == "__main__":
    print(f"# Case-6 cross-grid symmetry breakdown  (W6, {DAYS:.0f} d)")
    for label, grid, res in runs:
        odir = OUT / label
        odir.mkdir(parents=True, exist_ok=True)
        tc = m.TestCase(
            equation_set="shallow_water", case="williamson6",
            grid_type=grid, resolution=res, vertical_coord="none",
            duration_days=DAYS, quick_days=1.0, run_kwargs={"test_num": 6},
        )
        try:
            status, wall, notes = m.run_shallow_water(tc, odir, DAYS)
            print(f"  {label:16s} {status:6s} [{wall:.0f}s] {notes}")
        except Exception as e:
            print(f"  {label:16s} FAILED  {type(e).__name__}: {e}")

    print("\nlabel              day @ off-m4>1e-2   final off-m4   key")
    curves = {}
    for label, _, _ in runs:
        r = analyze(label)
        if r is None:
            print(f"  {label:16s}  (no data)")
            continue
        t, fr, key = r
        curves[label] = (t, fr)
        idx = np.argmax(fr > 1e-2) if (fr > 1e-2).any() else -1
        d_cross = f"{t[idx]:.1f}" if idx >= 0 else ">40 (none)"
        print(f"  {label:16s}  {d_cross:>14s}      {fr[-1]:.3e}   {key}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for label, (t, fr) in curves.items():
            plt.semilogy(t, np.maximum(fr, 1e-16), "o-", label=label)
        plt.axhline(1e-2, ls="--", c="k", lw=0.8, label="breakdown onset")
        plt.xlabel("day"); plt.ylabel("off-(m=4) power fraction")
        plt.title(f"Case-6 RH wave-4 breakdown: cube vs non-cube ({DAYS:.0f}d)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        p = OUT / "crossgrid_breakfrac.png"
        plt.savefig(p, dpi=110)
        print(f"\nplot -> {p}")
    except Exception as e:
        print(f"plot skipped: {e}")
