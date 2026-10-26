#!/usr/bin/env python
"""Pre-registered stripe-index A/B for the bilinear forcing fix.

Stripe index (GLM protocol, 2026-08-26): per snapshot field, regrid to 1
degree, take the zonal mean by row over 40S-40N, subtract a 5-row running
mean, SI = std of the residual.  Measures zonally coherent row-scale banding
(the nearest-neighbour forcing stairsteps) while ignoring smooth meridional
structure.  ACCEPTANCE (pre-registered before the run): SST SI reduced >=70%
and MLD SI >= 50% vs the nearest baseline; guards |global mean dSST| <= 0.1 C
and no new zonally coherent |dSST| > 0.3 C band.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))


def _si(field_g, ocean, tgt_lat):
    sel = np.abs(tgt_lat) <= 40.0
    zm = np.array([np.nanmean(np.where(ocean[j], field_g[j], np.nan))
                   for j in range(field_g.shape[0])])
    zm = zm[sel]
    k = np.ones(5) / 5.0
    run = np.convolve(zm, k, mode="same")
    r = (zm - run)[2:-2]
    return float(np.nanstd(r))


def _mld(L):
    from legoesm.ocean.diagnostics import mixed_layer_depth
    import jax.numpy as jnp
    z_c = np.asarray(L["z_center_ref"], np.float64)
    Hb = np.asarray(L["H_bathy"], np.float64)
    wet = ((z_c[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None])
           & (L["mask"][..., None] > 0.5)).astype(np.float64)
    return np.asarray(mixed_layer_depth(
        L["T3d"], L["S3d"], jnp.asarray(z_c), delta_sigma=0.01,
        wet_mask=wet, bottom_depth=Hb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="nearest-run snapshot")
    ap.add_argument("--fixed", required=True, help="bilinear-run snapshot")
    args = ap.parse_args()

    from compare_omip_nemo import _load_legoesm, regrid_curv_to_latlon

    tgt_lat = np.arange(-89.5, 90.0, 1.0)
    tgt_lon = np.arange(0.5, 360.0, 1.0)
    out = {}
    for tag, path in (("nearest", args.baseline), ("bilinear", args.fixed)):
        L = _load_legoesm(path)
        sst_g, oc = regrid_curv_to_latlon(L["sst"], L["lat"], L["lon"],
                                          L["mask"], tgt_lat, tgt_lon)
        mld = _mld(L)
        mld_g, ocm = regrid_curv_to_latlon(
            np.nan_to_num(mld, nan=0.0), L["lat"], L["lon"],
            np.isfinite(mld).astype(np.float64), tgt_lat, tgt_lon)
        out[tag] = dict(sst=sst_g, oc=oc > 0.5, mld=mld_g, ocm=ocm > 0.5)
        print(f"{tag}: SST SI = {_si(sst_g, oc > 0.5, tgt_lat):.5f} K   "
              f"MLD SI = {_si(mld_g, ocm > 0.5, tgt_lat):.4f} m")

    a, b = out["nearest"], out["bilinear"]
    si_sst = (_si(a['sst'], a['oc'], tgt_lat), _si(b['sst'], b['oc'], tgt_lat))
    si_mld = (_si(a['mld'], a['ocm'], tgt_lat), _si(b['mld'], b['ocm'], tgt_lat))
    red_sst = 100.0 * (1 - si_sst[1] / si_sst[0])
    red_mld = 100.0 * (1 - si_mld[1] / si_mld[0])
    both = a["oc"] & b["oc"]
    aw = np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :]
    d = np.where(both, b["sst"] - a["sst"], np.nan)
    g_dsst = float(np.nansum(d * aw) / np.nansum(np.where(both, aw, np.nan)))
    zm_d = np.array([np.nanmean(d[j]) for j in range(d.shape[0])])
    new_band = np.nanmax(np.abs(zm_d[(np.abs(tgt_lat) <= 40)]))
    print(f"\nSST SI reduction {red_sst:+.1f}% (need >=70)   "
          f"MLD SI reduction {red_mld:+.1f}% (need >=50)")
    print(f"guards: global mean dSST {g_dsst:+.4f} C (|.|<=0.1)   "
          f"max zonal-mean |dSST| 40S-40N {new_band:.3f} C (<0.3)")
    ok = red_sst >= 70 and red_mld >= 50 and abs(g_dsst) <= 0.1 and new_band < 0.3
    print("[VERDICT] " + ("ACCEPTED (pre-registered criteria met)" if ok
                          else "CRITERIA NOT MET — report, do not spin"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
