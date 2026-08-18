"""Continuous Arctic stratification comparison vs NEMO (area-weighted).

WHY THIS EXISTS.  The binary no-crossing count (score_mxl_choice_verdict.py)
asks only "does ANY level below 10 m cross 0.01 kg/m3?".  It discards magnitude
and depth, so a broad distributed density shift leaves the class unchanged.
Codex (2026-08-07 r2) showed that is exactly what happened in the
--tke-mxl-choice A/B: the signed per-column shift was +0.008075 kg/m3 with 4048
columns positive vs 896 negative, yet only 6 columns net flipped, and concluded
the binary rule was "an invalid inference, not a conclusion forced by the A/B".

This reports instead:
  * the area-weighted ECDF of per-column max-dsigma below the reference depth,
    for each model and NEMO -- the whole distribution, not one cut;
  * a THRESHOLD SWEEP: no-crossing area fraction vs threshold, so no single
    arbitrary 0.01 decides anything;
  * a CONTINUOUS error vs NEMO: area-weighted mean |max-dsigma - NEMO| and the
    signed bias, which is what "moves toward NEMO" actually means.

The estimator, Arctic support and mesh handling are imported from the VALIDATED
scorer, which reproduces the campaign's established 14.63% at exact column
counts.  Do not re-derive them here.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_mxl_choice_verdict as SC


def _max_dsigma(sig, wet, z):
    """Per-column max density increase below the reference depth (continuous)."""
    zz = np.broadcast_to(z, sig.shape)
    above = (zz <= SC.ZREF) & wet
    k_lo = np.where(above.any(-1), above.shape[-1]-1-np.argmax(above[..., ::-1], -1), 0)
    k_hi = np.minimum(k_lo+1, sig.shape[-1]-1)
    tk = lambda a, k: np.take_along_axis(a, k[..., None], -1)[..., 0]
    z_lo, z_hi = tk(zz, k_lo), tk(zz, k_hi)
    s_lo, s_hi = tk(sig, k_lo), tk(sig, k_hi)
    den = np.where(np.abs(z_hi-z_lo) > 1e-9, z_hi-z_lo, 1.0)
    w = np.clip((SC.ZREF-z_lo)/den, 0.0, 1.0)
    ds = sig - ((1-w)*s_lo + w*s_hi)[..., None]
    return np.max(np.where(wet & (zz > SC.ZREF), ds, -np.inf), axis=-1)


def ours(snapshot):
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.eos import wright_eos
    d = np.load(snapshot)
    T = np.asarray(d["T"], np.float64); S = np.asarray(d["S"], np.float64)
    z = np.asarray(d["z_center_ref"], np.float64)
    Hb = np.asarray(d["H_bathy"], np.float64); lm = np.asarray(d["land_mask"], np.float64)
    wet = (z[(None,)*Hb.ndim+(slice(None),)] < Hb[..., None]) & (lm[..., None] > 0.5)
    sig = np.asarray(wright_eos(jnp.asarray(T), jnp.asarray(S), jnp.asarray(0.0))) - 1000.
    A, glat, nemo_wet = SC._mesh_area_lat()
    NJ, NI = glat.shape
    m = _max_dsigma(sig, wet, z)[:NJ, :NI]
    arc = nemo_wet & (glat >= SC.ARCTIC_LAT) & np.any(wet, axis=-1)[:NJ, :NI]
    return m, arc, A


def nemo(day=90):
    import jax.numpy as jnp, netCDF4
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.eos import wright_eos
    A, glat, nemo_wet = SC._mesh_area_lat()
    with netCDF4.Dataset(SC.MESH) as ds:
        def I(n):
            a = np.squeeze(np.asarray(ds[n][:], np.float64)); return a[..., :331, 1:361]
        e3t0, tmask = I("e3t_0"), I("tmask")
    tm = np.moveaxis(tmask, 0, -1) > 0.5
    e30 = np.moveaxis(e3t0, 0, -1)
    zc = np.cumsum(e30, axis=-1) - 0.5*e30
    wetcol = np.any(tm, axis=-1)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gw", "/burg-archive/glab/users/pg2328/legoESM/scripts/tmp/_score_gateway_verdicts.py")
    gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)
    Tn = np.moveaxis(gw._assemble_restart(gw.NEMO_RESTARTS[day], "tn", wetcol), 0, -1)
    Sn = np.moveaxis(gw._assemble_restart(gw.NEMO_RESTARTS[day], "sn", wetcol), 0, -1)
    sig = np.asarray(wright_eos(jnp.asarray(Tn), jnp.asarray(Sn), jnp.asarray(0.0))) - 1000.
    # NEMO z varies per column -> evaluate with its own zc via the same helper
    zz = zc
    above = (zz <= SC.ZREF) & tm
    k_lo = np.where(above.any(-1), above.shape[-1]-1-np.argmax(above[..., ::-1], -1), 0)
    k_hi = np.minimum(k_lo+1, sig.shape[-1]-1)
    tk = lambda a, k: np.take_along_axis(a, k[..., None], -1)[..., 0]
    z_lo, z_hi = tk(zz, k_lo), tk(zz, k_hi)
    s_lo, s_hi = tk(sig, k_lo), tk(sig, k_hi)
    den = np.where(np.abs(z_hi-z_lo) > 1e-9, z_hi-z_lo, 1.0)
    w = np.clip((SC.ZREF-z_lo)/den, 0.0, 1.0)
    ds = sig - ((1-w)*s_lo + w*s_hi)[..., None]
    m = np.max(np.where(tm & (zz > SC.ZREF), ds, -np.inf), axis=-1)
    return m, nemo_wet & (glat >= SC.ARCTIC_LAT) & wetcol, A


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="run dirs under results/omip_nemo (snapshot_final.npz)")
    p.add_argument("--nemo-day", type=int, default=90)
    a = p.parse_args()

    mn, arcn, A = nemo(a.nemo_day)
    def wq(m, arc, qs):
        w = A[arc]; v = m[arc]; o = np.argsort(v)
        c = np.cumsum(w[o])/np.sum(w)
        return [float(v[o][np.searchsorted(c, q)]) for q in qs]
    QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
    print(f"[NEMO d{a.nemo_day}] area-weighted quantiles of max-dsigma below "
          f"{SC.ZREF:.0f} m [kg/m3]:")
    print("   " + "  ".join(f"p{int(q*100)}={v:+.4f}" for q, v in zip(QS, wq(mn, arcn, QS))))

    print("\n  threshold sweep: no-crossing AREA % vs threshold")
    THR = (0.001, 0.003, 0.01, 0.03, 0.1)
    hdr = "    run".ljust(30) + "".join(f"{t:>9.3f}" for t in THR)
    print(hdr)
    def sweep(m, arc):
        return [float(A[arc & (m < t)].sum()/A[arc].sum()*100) for t in THR]
    print("    NEMO".ljust(30) + "".join(f"{v:9.2f}" for v in sweep(mn, arcn)))
    for run in a.runs:
        snap = None
        for n in ("snapshot_final.npz", "snapshot_day0030.npz"):
            q = f"{SC.R}/{run}/{n}"
            if os.path.exists(q): snap = q; break
        if snap is None:
            print(f"    {run}: NO SNAPSHOT"); continue
        m, arc, _ = ours(snap)
        print(f"    {run}".ljust(30) + "".join(f"{v:9.2f}" for v in sweep(m, arc)))
        common = arc & arcn
        w = A[common]
        err = float(np.sum(w*np.abs(m[common]-mn[common]))/np.sum(w))
        bias = float(np.sum(w*(m[common]-mn[common]))/np.sum(w))
        print(f"      vs NEMO on {int(common.sum())} common columns: "
              f"mean|err| {err:.4f}   signed bias {bias:+.4f} kg/m3")
        print("      quantiles: " + "  ".join(
            f"p{int(q*100)}={v:+.4f}" for q, v in zip(QS, wq(m, arc, QS))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
