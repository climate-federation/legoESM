"""Does the virtual salt flux reproduce NEMO's surface dilution? (GLM falsifier,
2026-09-05, in the form the existing files allow.)

NEMO's non-linear free surface dilutes the top cell with the surface water
flux F; our ``virtual_salt_flux`` closure adds the salt flux -S_ref*F instead;
our ``real_freshwater`` closure (as shipped) adds NEITHER (the water only
stretches the column; both reviewers confirmed).  No file carries F, but the
two arms share forcing and state to first order, so

    x = dS(virtual) - dS(real)     ~ the dilution term the virtual closure applies
    y = dS(NEMO)    - dS(real)     ~ the dilution term NEMO applies

point by point on the shared eORCA1 mesh.  Slope ~1 at ice-free, open-ocean,
non-plume points confirms the linearisation; a slope < 1 where S -> 0 (river
plumes) is the runaway the Amazon clamp showed; ice points test the
(S - S_ice) vs S question.  dS is the change of the 0-``--ml-depth`` m mean
salinity between the early and late times (a mixed-layer mean, so the two
sides' different vertical mixing of the same input does not masquerade as a
closure difference).

Frames: NEMO output (331, 360) is mesh rows 0:331 / cols 1:361 of our
(332, 362) snapshot (box-budget probe, 2026-09-05, verified on nav_lat).
"""
from __future__ import annotations

import argparse

import numpy as np


def _ml_mean_S(S, zc, zmax):
    dz = np.gradient(zc)
    k = zc <= zmax
    w = dz[k][None, None, :] * np.isfinite(S[..., k])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(S[..., k] * dz[k][None, None, :], axis=-1) / w.sum(-1)


def _ours(path, zmax):
    z = np.load(path)
    S = np.asarray(z["S"], dtype=np.float64)
    zc = np.abs(np.asarray(z["z_center_ref"], dtype=np.float64))
    S = np.where(np.asarray(z["land_mask"])[..., None] > 0.5, S, np.nan)
    ice = np.asarray(z["ice_concentration"], dtype=np.float64) if "ice_concentration" in z else np.zeros(S.shape[:2])
    return (_ml_mean_S(S, zc, zmax)[0:331, 1:361], S[0:331, 1:361, 0], ice[0:331, 1:361],
            np.asarray(z["lat_T"])[0:331, 1:361], np.asarray(z["land_mask"])[0:331, 1:361] > 0.5)


def _nemo(path, rec, zmax):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        S = np.ma.filled(np.ma.masked_invalid(ds.variables["so"][rec]), np.nan).astype(np.float64)
        zc = np.asarray(ds.variables["deptht"][:], dtype=np.float64)
    finally:
        ds.close()
    S = np.moveaxis(S, 0, -1)                                      # (y, x, z)
    S = np.where(S == 0.0, np.nan, S)                              # NEMO land sentinel
    return _ml_mean_S(S, zc, zmax)


def _fit(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 20:
        return np.nan, np.nan, np.nan, int(x.size)
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = np.corrcoef(x, y)[0, 1]
    return float(m), float(b), float(r), int(x.size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--virtual-early", required=True); ap.add_argument("--virtual-late", required=True)
    ap.add_argument("--real-early", required=True); ap.add_argument("--real-late", required=True)
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--nemo-rec-early", type=int, required=True); ap.add_argument("--nemo-rec-late", type=int, required=True)
    ap.add_argument("--ml-depth", type=float, default=30.0)
    ap.add_argument("--plume-S", type=float, default=32.0, help="top-cell S below this = river plume")
    a = ap.parse_args()

    ve, _, ice_ve, lat, wet = _ours(a.virtual_early, a.ml_depth)
    vl, s_top_v, ice_vl, _, _ = _ours(a.virtual_late, a.ml_depth)
    re, _, ice_re, _, _ = _ours(a.real_early, a.ml_depth)
    rl, s_top_r, ice_rl, _, _ = _ours(a.real_late, a.ml_depth)
    ne = _nemo(a.nemo_gridt, a.nemo_rec_early, a.ml_depth)
    nl = _nemo(a.nemo_gridt, a.nemo_rec_late, a.ml_depth)
    if ne.shape != ve.shape:
        raise SystemExit(f"frame mismatch NEMO {ne.shape} vs ours {ve.shape}")
    x = (vl - ve) - (rl - re)
    y = (nl - ne) - (rl - re)
    # open ocean: every cell within 3 of this one is wet
    from scipy.ndimage import minimum_filter
    interior = minimum_filter(wet.astype(np.int8), size=7, mode="nearest") > 0
    ice_free = (np.maximum.reduce([ice_ve, ice_vl, ice_re, ice_rl]) < 0.01)
    icy = np.minimum.reduce([ice_ve, ice_vl, ice_re, ice_rl]) > 0.5
    plume = (np.minimum(s_top_v, s_top_r) < a.plume_S) & ~icy
    base = wet & interior & np.isfinite(x) & np.isfinite(y)
    print(f"dS = change of the 0-{a.ml_depth:.0f} m mean salinity, early->late; x = virtual - real, "
          f"y = NEMO - real [PSU].  y = m x + b")
    print(f"{'subset':>28} {'slope':>7} {'icpt':>8} {'r':>6} {'N':>7} {'<x>':>8} {'<y>':>8}")
    for name, m in (("ice-free open ocean, no plume", base & ice_free & ~plume & (np.abs(lat) < 60)),
                    ("  |lat|<30 only", base & ice_free & ~plume & (np.abs(lat) < 30)),
                    ("  30<|lat|<60", base & ice_free & ~plume & (np.abs(lat) >= 30) & (np.abs(lat) < 60)),
                    ("river plume (S_top<thr)", base & plume),
                    ("ice > 0.5 (both times)", base & icy),
                    ("ALL wet interior", base)):
        s, b, r, n = _fit(x[m], y[m])
        print(f"{name:>28} {s:7.3f} {b:8.4f} {r:6.3f} {n:7d} {np.nanmean(x[m]):8.4f} {np.nanmean(y[m]):8.4f}")
    print("\nREAD: slope ~1 (r high) on the first row = the virtual closure IS NEMO's dilution to first "
          "order; slope < 1 on plumes = virtual over-dilutes as S -> 0; the ice row's slope vs 1 measures "
          "the (S - S_ice) vs S question (0.89 expected if NEMO removes only S - S_ice; below that if the "
          "virtual arm double-counts S + S_ice).  Caveat: the real arm is the 'no dilution' baseline; "
          "any circulation difference between arms enters both x and y.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
