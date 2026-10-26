#!/usr/bin/env python
"""Arctic N² : ours vs NEMO, through the SAME routine, on a MATCHED window.

WHY.  The Arctic bias was shown to be a vertical redistribution, not an
inventory error (``arctic_salt_vs_volume.py``): our top 19 m varies by 0.018
psu where NEMO varies by 0.83, i.e. the column is mixed flat.  The surface TKE
boundary condition was then ELIMINATED as the cause -- our under-ice surface
value reproduces NEMO's ``rn_ebb*|tau|/rho0`` to four digits.  Since the TKE
mixing coefficient goes as ``l*sqrt(e)`` and ``e`` matches, what remains is the
LENGTH scale, whose dominant interior term is the buoyancy length
``l_b = sqrt(2e)/N``.

THIS PROBE measures ``N²`` for both models and the buoyancy length they imply,
which quantifies the suspected feedback: weak stratification -> long mixing
length -> more mixing -> weaker stratification.

MATCHED, deliberately:
  * both T/S go through the model's OWN ``compute_buoyancy_frequency_nemo_bn2``
    (NEMO ``eosbn2.F90`` bn2_t), so the instrument is identical on both sides;
  * NEMO's month-1 grid_T is used, NOT the grid_W ``bn2``, which is an ANNUAL
    MEAN -- Arctic winter mixing dwarfs summer, so an annual mean would flatter
    whichever side we wanted and prove nothing.

CAVEATS, stated rather than discovered later:
  * the ABSOLUTE N² depends on the S-EOS coefficient set; because the same
    routine and config are applied to both sides, the OURS-vs-NEMO contrast is
    the robust quantity, not the absolute value;
  * ours is an instantaneous day-30 state, NEMO's is a January monthly mean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_HERE.parents[3] / "packages" / _p))

from global_tracer_content import (  # noqa: E402
    _native, load_mesh_depth_1d, load_mesh_latitude, load_mesh_metrics,
)


def _mesh_gdepw(mesh_mask_path):
    import netCDF4 as nc
    ds = nc.Dataset(mesh_mask_path)
    try:
        g = np.asarray(ds.variables["gdepw_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    return g


def _n2(T, S, gdept, gdepw_int):
    """Model's own NEMO bn2. T/S level-LAST, per the function's contract."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(gdept), jnp.asarray(gdepw_int)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-month", type=int, default=1)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--arctic-lat", type=float, default=60.0)
    a = p.parse_args()

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    lat = load_mesh_latitude(a.mesh_mask)
    gdept = load_mesh_depth_1d(a.mesh_mask)
    gdepw = _mesh_gdepw(a.mesh_mask)
    gdepw_int = gdepw[1:]                      # interior interfaces, (nlev-1,)

    wet = tmask > 0.5
    arctic = (lat >= a.arctic_lat) & wet.any(axis=0)
    if not arctic.any():
        raise SystemExit("FATAL: empty Arctic region")

    z = dict(np.load(a.snapshot))
    T_our = np.transpose(_native(z["T"]), (1, 2, 0))     # -> (nj, ni, nlev)
    S_our = np.transpose(_native(z["S"]), (1, 2, 0))
    tke = _native(z["tke"])                              # (nlev-1, nj, ni)

    import netCDF4 as nc
    ds = nc.Dataset(a.nemo_gridt)
    try:
        it = a.nemo_month - 1
        def v(n):
            return np.ma.filled(np.ma.masked_invalid(ds.variables[n][it]),
                                0.0).astype(np.float64)
        S_nemo = np.transpose(v("so"), (1, 2, 0))        # (75,331,360)->(nj,ni,75)
        T_nemo = np.transpose(v("to"), (1, 2, 0))
    finally:
        ds.close()
    for nm, arr in (("so", S_nemo), ("to", T_nemo)):
        if np.abs(arr).max() > 1.0e6:
            raise SystemExit(f"FATAL: {nm} carries fill values")
    if S_nemo.shape != S_our.shape:
        raise SystemExit(f"NEMO {S_nemo.shape} vs ours {S_our.shape}")

    n2_our = _n2(T_our, S_our, gdept, gdepw_int)          # (nj, ni, nlev-1)
    n2_nemo = _n2(T_nemo, S_nemo, gdept, gdepw_int)

    # Interface volume weights, Arctic only.
    dV = e1t[None] * e2t[None] * e3t * (wet & arctic[None])
    dVi = 0.5 * (dV[:-1] + dV[1:])                        # (nlev-1, nj, ni)
    w = np.transpose(dVi, (1, 2, 0))
    gd_i = 0.5 * (gdept[:-1] + gdept[1:])

    print("iface depth_m     N2_ours      N2_nemo    ratio   "
          "l_b_ours_m  l_b_withNemoN_m")
    for k in range(n2_our.shape[-1]):
        den = w[..., k].sum()
        if den <= 0:
            continue
        a_our = float((n2_our[..., k] * w[..., k]).sum() / den)
        a_nem = float((n2_nemo[..., k] * w[..., k]).sum() / den)
        e_k = float((tke[k] * dVi[k]).sum() / dVi[k].sum())
        def lb(n2):
            return np.sqrt(2.0 * max(e_k, 0.0)) / np.sqrt(n2) if n2 > 0 else np.inf
        print(f"{k:3d} {gd_i[k]:9.2f} {a_our:+.4e} {a_nem:+.4e} "
              f"{(a_our / a_nem if a_nem != 0 else np.nan):7.3f} "
              f"{lb(a_our):11.2f} {lb(a_nem):11.2f}")
        if gd_i[k] > 120.0:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
