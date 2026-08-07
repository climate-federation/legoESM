"""Verdict scorer for the --tke-mxl-choice 3-vs-4 A/B (jobs 9332440/9332441).

PRE-REGISTERED METRIC, fixed before the arms were launched:
    Arctic >=66N fraction of area whose column never accumulates
    delta_sigma >= 0.01 kg/m3 below 10 m ("no-crossing" = lost stratification).
    CONFIRMS nn_mxl as the cause : choice-4 arm <= ~7%
    REFUTES                      : >= ~13%
    PARTIAL                      : 7-13%
Reference points measured earlier with the SAME estimator (job 9331814):
    tripole gwcorr d30 = 14.63%   |   NEMO d30 = 5.35%

SELF-VALIDATION (--validate): re-score the gwcorr d30 snapshot and require it to
reproduce 14.63% within tolerance. If the scorer cannot reproduce a number the
campaign already established, it is WRONG and must not be used for the verdict.
This is the "validate the instrument before quoting it" rule applied BEFORE the
result exists, rather than after a surprising number appears.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

R = "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo"
MESH = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"
DS, ZREF, ARCTIC_LAT = 0.01, 10.0, 66.0
GWCORR_D30_REF = 14.63     # job 9331814, area-weighted
NEMO_D30_REF = 5.35


def _mesh_area_lat():
    """Area, latitude AND the NEMO wet-column mask, all on the interior grid.

    The tmask term is LOAD-BEARING. Job 9331814 (which produced the established
    14.63% and NEMO's 5.35%) defined the Arctic as
        NEMO-mesh wet-column  AND  lat >= 66  AND  our-snapshot wet
    giving 4944 columns. Dropping the NEMO term gives 5164 and 15.85% -- the
    1.22-point discrepancy that failed --validate. Any consistent mask would do
    for an arm-vs-arm verdict, but the NEMO-intersected support is required to
    stay comparable with the 5.35% and 14.63% reference points.
    """
    import netCDF4
    with netCDF4.Dataset(MESH) as ds:
        def I(n):
            a = np.squeeze(np.asarray(ds[n][:], np.float64))
            return a[..., :331, 1:361]
        tmask = I("tmask")
        nemo_wetcol = np.any(np.moveaxis(tmask, 0, -1) > 0.5, axis=-1)
        return I("e1t") * I("e2t"), I("gphit"), nemo_wetcol


def no_crossing_fraction(snapshot: str):
    """Area fraction of Arctic columns with NO density crossing below 10 m."""
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.eos import wright_eos

    d = np.load(snapshot)
    T = np.asarray(d["T"], np.float64)
    S = np.asarray(d["S"], np.float64)
    if T.dtype != np.float64:
        raise SystemExit("FATAL: snapshot T is not float64")
    z = np.asarray(d["z_center_ref"], np.float64)
    Hb = np.asarray(d["H_bathy"], np.float64)
    lm = np.asarray(d["land_mask"], np.float64)
    A, glat, nemo_wetcol = _mesh_area_lat()
    NJ, NI = glat.shape

    wet = (z[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None]) & (lm[..., None] > 0.5)
    sig = np.asarray(wright_eos(jnp.asarray(T), jnp.asarray(S), jnp.asarray(0.0))) - 1000.0
    # sigma at the 10 m reference depth, bracketed PER COLUMN on WET levels.
    # An earlier draft used FIXED GLOBAL indices from `z` alone; on shallow
    # Arctic shelf columns the 10 m bracket level lies below bathymetry and is
    # DRY, so sigma was read from a dry cell and the column mis-classified.
    # That gave 15.85% where the established figure (job 9331814, wet-aware) is
    # 14.63% -- caught by --validate before the verdict depended on it.
    # diagnostics.py:336 is wet-aware for the same reason
    # (use_prev = (z_km1 > ref_depth_m) & (wet_km1 > 0.5)).
    zz = np.broadcast_to(z, sig.shape)
    above = (zz <= ZREF) & wet
    k_lo = np.where(above.any(-1),
                    above.shape[-1] - 1 - np.argmax(above[..., ::-1], -1), 0)
    k_hi = np.minimum(k_lo + 1, sig.shape[-1] - 1)
    tk = lambda a, k: np.take_along_axis(a, k[..., None], -1)[..., 0]
    z_lo, z_hi = tk(zz, k_lo), tk(zz, k_hi)
    s_lo, s_hi = tk(sig, k_lo), tk(sig, k_hi)
    den = np.where(np.abs(z_hi - z_lo) > 1e-9, z_hi - z_lo, 1.0)
    w = np.clip((ZREF - z_lo) / den, 0.0, 1.0)
    s_ref = (1 - w) * s_lo + w * s_hi
    dsig = sig - s_ref[..., None]
    searchable = wet & (z[(None,) * Hb.ndim + (slice(None),)] > ZREF)
    has = np.any((dsig >= DS) & searchable, axis=-1)

    # Support MUST match job 9331814: NEMO wet-column AND lat>=66 AND our wet.
    arc = nemo_wetcol & (glat >= ARCTIC_LAT) & np.any(wet, axis=-1)[:NJ, :NI]
    nc = arc & ~has[:NJ, :NI]
    if not arc.any():
        raise SystemExit("FATAL: empty Arctic mask -- estimator is broken")
    return dict(pct=float(A[nc].sum() / A[arc].sum() * 100.0),
                n_nc=int(nc.sum()), n_arc=int(arc.sum()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validate", action="store_true",
                   help="Re-score gwcorr d30 and require 14.63%% (instrument gate).")
    p.add_argument("--tol", type=float, default=0.25,
                   help="Absolute tolerance [pct points] for --validate.")
    a = p.parse_args()

    if a.validate:
        got = no_crossing_fraction(f"{R}/nemolev_trp_gwcorr_d90/snapshot_day0030.npz")
        d = abs(got["pct"] - GWCORR_D30_REF)
        print(f"[validate] gwcorr d30 no-crossing = {got['pct']:.2f}%  "
              f"(established {GWCORR_D30_REF}%, |diff| {d:.3f})")
        print(f"           {got['n_nc']}/{got['n_arc']} Arctic columns")
        if d > a.tol:
            print("FAIL: scorer does NOT reproduce the established number. "
                  "Do NOT use it for the verdict.")
            return 1
        print("PASS: scorer reproduces the established number.")
        return 0

    out = {}
    for tag, run in (("choice 4 (NEMO nn_mxl=2)", "nemolev_trp_mxl4_d30"),
                     ("choice 3 (control)", "nemolev_trp_mxl3ctl_d30")):
        snap = f"{R}/{run}/snapshot_day0030.npz"
        if not os.path.exists(snap):
            print(f"  {tag:26s} NOT READY ({run})"); continue
        out[tag] = no_crossing_fraction(snap)
        print(f"  {tag:26s} {out[tag]['pct']:6.2f}%  "
              f"({out[tag]['n_nc']}/{out[tag]['n_arc']} columns)")
    print(f"  {'NEMO d30 (reference)':26s} {NEMO_D30_REF:6.2f}%")
    print(f"  {'gwcorr d30 (old baseline)':26s} {GWCORR_D30_REF:6.2f}%  "
          f"[different commit -- reference only, NOT the control]")

    if len(out) == 2:
        c4 = out["choice 4 (NEMO nn_mxl=2)"]["pct"]
        c3 = out["choice 3 (control)"]["pct"]
        print(f"\n=== PRE-REGISTERED VERDICT ===")
        print(f"  control (choice 3) {c3:.2f}%   ->   choice 4 {c4:.2f}%   "
              f"(delta {c4 - c3:+.2f} pts)")
        if c4 <= 7.0:
            print("  CONFIRMS: nn_mxl mismatch was the dominant cause.")
        elif c4 >= 13.0:
            print("  REFUTES: nn_mxl is NOT the cause; the first-30-day gap "
                  "needs another explanation (next candidate: tracer advection,"
                  " superbee vs NEMO FCT2).")
        else:
            print("  PARTIAL: nn_mxl contributes but is not the whole gap.")
        print("  NB the verdict compares the two SAME-COMMIT arms; the 14.63%"
              " figure is 211 commits older and is context, not the bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
