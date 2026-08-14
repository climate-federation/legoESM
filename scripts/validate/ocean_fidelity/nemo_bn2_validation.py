"""Validate our N² against NEMO's OWN rn2, recovered from its restart.

THE PROBLEM.  NEMO writes no rn2/bn2 into its restart, and the only bn2 in the
output stream is an ANNUAL MEAN on grid_W while the T/S we can pair with it are
monthly — and the annual mean of a nonlinear function is not that function of
the annual mean.  Scoring against it would be a time-aggregation confound.

THE WAY IN.  The restart DOES carry ``dissl``, and zdftke defines

    dissl = SQRT(en) / zmxld            (zdftke.F90:717)
    zmxlm = SQRT( 2*en / rn2 )          (zdftke.F90:651, buoyancy length)

and ORCA1 runs nn_mxl=2, where NEMO sets zmxld = zmxlm (:680).  Eliminating the
length:

    rn2 = 2 * dissl**2

So NEMO's own stratification is recoverable at EXACTLY the instant its tn/sn
were written — no time alignment, no aggregation, no regridding.

WHAT THE INVERSION IS AND IS NOT.  zmxlm is ``MIN(buoyancy length, sweep
bounds)``: the |dl/dz| <= e3t sweeps and the rmxl_min floor can CAP it.  Where
a cap binds, zmxlm is SHORTER than the buoyancy length, so the inverted rn2
comes out TOO LARGE.  The inversion is therefore an UPPER BOUND on NEMO's rn2,
exact only where the buoyancy scale is what set the length.  It is used here as
a RELATIVE discriminator — which of our two N² forms sits closer to it on
identical state — not as an absolute truth field, and the comparison is
restricted to a mid-range band where caps are least likely to bind.

Usage (CPU, minutes; sbatch it per the login-node policy):
    python scripts/validate/ocean_fidelity/nemo_bn2_validation.py \
        --restart-npz .../restart_8760_full.npz \
        --mesh-mask data/grids/eORCA1.2_mesh_mask.nc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_REPO / "packages" / _p))
sys.path.insert(0, str(_REPO))

from global_tracer_content import (  # noqa: E402
    load_mesh_depth_1d, load_mesh_latitude, load_mesh_metrics,
)

_BANDS = (
    ("antarctic_S_of_45S", -91.0, -45.0),
    ("tropics_23S_23N", -23.0, 23.0),
    ("arctic_N_of_45N", 45.0, 91.0),
)


def _gdepw(mesh):
    import netCDF4 as nc
    ds = nc.Dataset(mesh)
    try:
        return np.asarray(ds.variables["gdepw_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--restart-npz", required=True)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--lo", type=float, default=1.0e-7,
                   help="lower edge of the comparison band on NEMO's inverted "
                        "rn2 [1/s2]; below this the rmxl_min floor and the "
                        "sweeps dominate and the inversion is not usable.")
    p.add_argument("--hi", type=float, default=1.0e-3,
                   help="upper edge; above this the buoyancy length is tiny "
                        "and e3w quantisation dominates.")
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

    rst = dict(np.load(a.restart_npz))
    for k in ("tn", "sn", "dissl"):
        if k not in rst:
            raise SystemExit(
                f"FATAL: restart npz lacks {k!r}. Rebuild with "
                "--fields tn sn un vn sshn en avm_k avt_k dissl.")

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    lat = load_mesh_latitude(a.mesh_mask)
    gdept = load_mesh_depth_1d(a.mesh_mask)
    gdepw_int = _gdepw(a.mesh_mask)[1:]

    T3, S3, dis3 = rst["tn"], rst["sn"], rst["dissl"]
    z, ny, nx = T3.shape
    ncol = ny * nx

    def cols(x):
        return np.transpose(x.reshape(z, ncol), (1, 0))

    T_c = np.nan_to_num(cols(T3), nan=0.0)
    S_c = np.where(np.isfinite(cols(S3)) & (cols(S3) > 0), cols(S3), 35.0)
    # dissl is stored on W-levels with index 0 = surface; the interior
    # interfaces our N2 returns are NEMO's jk = 2..jpk, i.e. python 1:.
    dissl_i = np.nan_to_num(cols(dis3)[:, 1:], nan=0.0)

    # NEMO's own rn2, eliminating zmxld between dissl and the buoyancy length.
    rn2_nemo = 2.0 * dissl_i ** 2

    ours = {}
    for form in ("seos", "teos10"):
        ours[form] = np.asarray(compute_buoyancy_frequency_nemo_bn2(
            jnp.asarray(T_c), jnp.asarray(S_c),
            jnp.asarray(gdept), jnp.asarray(gdepw_int), eos_form=form))

    wet_c = np.transpose(tmask.reshape(z, ncol), (1, 0)) > 0.5
    wet_i = wet_c[:, :-1] & wet_c[:, 1:]
    band = wet_i & (rn2_nemo > a.lo) & (rn2_nemo < a.hi)
    dV = np.transpose((e1t[None] * e2t[None] * e3t * (tmask > 0.5)
                       ).reshape(z, ncol), (1, 0))
    w_i = 0.5 * (dV[:, :-1] + dV[:, 1:])

    print(f"[inversion] rn2 = 2*dissl^2 from the restart; comparison band "
          f"{a.lo:g} < rn2 < {a.hi:g} covers "
          f"{100.0 * band.sum() / max(wet_i.sum(), 1):.1f}% of wet interfaces "
          f"({int(band.sum())} of {int(wet_i.sum())})")
    print("READ: the inversion is an UPPER BOUND on NEMO's rn2 (a bound-limited "
          "zmxlm makes it too large), so this ranks the two forms against each "
          "other; it is not an absolute truth field.")
    print()
    print("%-22s %9s %11s %11s %9s %9s" % (
        "band", "n", "NEMO rn2", "ours", "ratio", "log10 rms"))

    out = {"restart": a.restart_npz, "lo": a.lo, "hi": a.hi, "bands": {}}
    for bname, blo, bhi in _BANDS:
        inb = ((lat >= blo) & (lat < bhi)).reshape(ncol)
        m = band & inb[:, None]
        if not m.any():
            continue
        ww = w_i[m]
        den = ww.sum()
        nemo = float((rn2_nemo[m] * ww).sum() / den)
        row = {"n": int(m.sum()), "nemo_rn2": nemo}
        for form in ("seos", "teos10"):
            o = ours[form][m]
            val = float((o * ww).sum() / den)
            # log-space rms is the scale-free comparison; N2 spans decades.
            ok = (o > 0) & (rn2_nemo[m] > 0)
            lr = float(np.sqrt(np.mean(
                (np.log10(o[ok]) - np.log10(rn2_nemo[m][ok])) ** 2)))
            row[form] = {"ours": val, "ratio": val / nemo if nemo else float("nan"),
                         "log10_rms": lr, "n_positive": int(ok.sum())}
            print("%-22s %9d %11.4e %11.4e %9.3f %9.3f  <- %s" % (
                bname, m.sum(), nemo, val,
                val / nemo if nemo else float("nan"), lr, form))
        out["bands"][bname] = row
    print()
    print("VERDICT RULE, fixed before the numbers were read: the TEOS-10 form "
          "wins only if its log10 rms is LOWER than the S-EOS form's in every "
          "band. A closer volume-mean ratio with a worse log rms is a "
          "cancellation, not an improvement.")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[json] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
