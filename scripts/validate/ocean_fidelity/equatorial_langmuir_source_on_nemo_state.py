#!/usr/bin/env python
"""Our Langmuir TKE source evaluated on NEMO's own restart column, next to the
hand formula and to the TKE NEMO shows with and without ln_lc.

NEMO with Langmuir carries 1.6e-3 m2/s2 at 10 m against 2.2e-4 without; the
Axell source as written (us^3 (rn_lc sin(pi z/h_lc))^3 / h_lc) peaks near
1e-6 W/kg. This calls the production function ``nemo_langmuir_tke_source``
(vectorized path, the one the tripole card runs) on NEMO's hour-12 T/S and
stress, and prints the source profile, its column integral, and the
dissipation NEMO's LC-minus-noLC TKE difference would require.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equatorial_shear_ri_vs_nemo import _n2  # noqa: E402
from equatorial_tke_length_vs_nemo_restart import reassemble  # noqa: E402

RN_EBB, RHO0 = 67.83, 1026.0  # coeff-ok: rn_ebb namelist, phycst rho0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart-glob-lc", required=True)
    ap.add_argument("--restart-glob-nolc", required=True)
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=30)
    a = ap.parse_args()
    import jax.numpy as jnp
    import glob, netCDF4 as nc
    from legoesm.ocean.physics.vertical_mixing.tke import TKEConfig, nemo_langmuir_tke_source, _NEMO_TKE_LC_CSD

    nl = a.n_levels
    R = reassemble(a.restart_glob_lc, ("tn", "sn", "en", "avm_k", "dissl"))
    Rn = reassemble(a.restart_glob_nolc, ("en", "avm_k", "dissl"))
    zc = np.asarray(nc.Dataset(sorted(glob.glob(a.restart_glob_lc))[0]).variables["nav_lev"][:], float)[:nl]
    zw_int = 0.5 * (zc[:-1] + zc[1:])            # interior W levels (midpoint approx of gdepw_1d)
    dz_w = np.diff(zc)
    la, lo = R["nav_lat"], R["nav_lon"] % 360.0
    T = np.moveaxis(R["tn"], 0, -1)[..., :nl]; S = np.moveaxis(R["sn"], 0, -1)[..., :nl]
    T = np.where(np.abs(T) < 1e-6, np.nan, T)
    box = (np.abs(la) <= a.lat_halfwidth) & (lo >= a.lon_lo) & (lo < a.lon_hi) & np.isfinite(T[..., 0])
    Tf, Sf = T[box], S[box]
    N2 = np.asarray(_n2(np.nan_to_num(Tf), np.nan_to_num(Sf), zc, zw_int))      # (cols, nl-1)
    taum = R["en"][0][box] * RHO0 / RN_EBB
    cfg = TKEConfig(lc=True, lc_coeff=0.25)
    src = np.asarray(nemo_langmuir_tke_source(jnp.asarray(taum), jnp.asarray(N2), jnp.asarray(zw_int),
                                              jnp.asarray(np.broadcast_to(dz_w, N2.shape)), cfg))
    us3 = (2.0 * float(_NEMO_TKE_LC_CSD) * taum) ** 1.5
    print(f"zcsd={float(_NEMO_TKE_LC_CSD):.4f}; box cols {int(box.sum())}; taum mean {taum.mean():.4f} Pa; us mean {np.mean(us3**(1/3)):.4f} m/s; "
          f"hand column total (4/3pi) us^3 rn_lc^3 = {np.mean(us3)*0.25**3*4/(3*np.pi):.3e} m3/s3; "
          f"our source column integral {np.mean(np.sum(src*dz_w, axis=-1)):.3e} m3/s3")
    eL = np.moveaxis(R["en"], 0, -1)[box][:, 1:nl]; eN = np.moveaxis(Rn["en"], 0, -1)[box][:, 1:nl]
    dL = np.moveaxis(R["dissl"], 0, -1)[box][:, 1:nl]; dN = np.moveaxis(Rn["dissl"], 0, -1)[box][:, 1:nl]
    epsL = 0.7 * dL * eL; epsN = 0.7 * dN * eN     # rn_ediss * sqrt(e)/l_eps * e
    print("  k   z_w |  src ours (W/kg)  |  e LC     e noLC  |  eps LC    eps noLC   d(eps) | src/d(eps)")
    for k in range(min(16, nl - 1)):
        s_, el, en_, xl, xn = (np.nanmean(x[:, k]) for x in (src, eL, eN, epsL, epsN))
        print(f"{k+1:3d} {zw_int[k]:6.2f} | {s_:14.3e} | {el:8.2e} {en_:8.2e} | {xl:9.2e} {xn:9.2e} {xl-xn:9.2e} | {s_/max(xl-xn,1e-30):7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
