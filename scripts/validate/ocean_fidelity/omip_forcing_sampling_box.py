"""Cold-tongue forcing as each point-sampling method delivers it (F20 follow-up).

WHY.  Moving MPAS and FESOM from nearest-neighbour to bilinear forcing
sampling (base1013 -> base1014) warmed the nino3 day-30 SST on BOTH grids
(+0.10 / +0.08 K), and NEMO itself samples winds BICUBICALLY (its own SCRIP
weights), not bilinearly.  This asks, with no model run, what each method
puts into the box: wind speed, stress, turbulent heat, radiation.

Every method is fed the SAME forcing records (CORE-II NYF, centres inside
``[0, --days)``) and the SAME sea-surface temperature (the grid's own
``T_mean`` top level from ``--snapshot``), and the bulk is the production
one (``air_sea_fluxes`` with ``slp``, NCAR default -- the call made by the
scan-lane forcing), so the only thing that varies between rows is sampling.

Methods: ``nearest``, ``bilinear`` (all production cards), ``nemo_weights``
(our bicubic winds + bilinear rest, the point form of the oracle split) and,
on the tripole only, ``nemo_scrip`` (ORCA1's own weight files).

Weights: ``cell_area`` when the snapshot carries it (tripole); otherwise
equal weights (MPAS lloyd mesh is quasi-uniform; FESOM rows indicative).

Instrument check, measured 2026-10-13 on the tripole box: ``nemo_weights``
(our point bicubic) and ``nemo_scrip`` (ORCA1's own weight files) agree to
<1e-6 relative in every quantity, so the point form reproduces the oracle.

Usage (compute node; imports jax):
  omip_forcing_sampling_box.py --snapshot S.npz --label NAME [--days 30]
      [--forcing-dir D] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BOX = (-5.0, 5.0, 210.0, 270.0)            # nino3, lat0, lat1, lon0, lon1
METHODS = ("nearest", "bilinear", "nemo_weights")
QUANTITIES = ("wind_speed", "taux_ocean", "taum", "q_turb", "sw_down",
              "lw_down", "T_air", "q_air")


def box_points(lat, lon, ocean, box=BOX):
    la0, la1, lo0, lo1 = box
    lon = np.asarray(lon) % 360.0
    return ocean & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)


def record_quantities(forc, T_sfc_K):
    """Per-point quantities from one sampled record (dict of 1-D arrays)."""
    import jax.numpy as jnp
    from legoesm.ocean.bulk_flux_omip import air_sea_fluxes

    slp = forc.get("slp")
    tau_x, tau_y, sh, lh, _ = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]), v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]), q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if slp is None else jnp.asarray(slp))
    tau_x, tau_y = np.asarray(tau_x), np.asarray(tau_y)
    return {
        "wind_speed": np.hypot(forc["u10"], forc["v10"]),
        "taux_ocean": -tau_x,               # ocean feels -tau (atm convention)
        "taum": np.hypot(tau_x, tau_y),
        "q_turb": np.asarray(sh) + np.asarray(lh),   # + into the ocean
        "sw_down": np.asarray(forc["sw_down"]),
        "lw_down": np.asarray(forc["lw_down"]),
        "T_air": np.asarray(forc["T_air"]),
        "q_air": np.asarray(forc["q_air"]),
    }


def box_means(forcing, records, lat, lon, T_sfc_K, w, methods, scrip_sel=None):
    """``{method: {quantity: weighted time+box mean}}`` over ``records``.

    ``lat``/``lon``/``T_sfc_K``/``w`` are the box points only.  ``scrip_sel``
    (flat index of the box points in the full 332x362 tripole) enables the
    ``nemo_scrip`` method."""
    from legoesm.ocean.coupler.omip2_applicator import (
        _sample_forcing_points, sample_forcing_tripole_scrip)

    w = np.asarray(w, dtype=np.float64) / np.sum(w)
    out = {}
    for m in methods:
        acc = dict.fromkeys(QUANTITIES, 0.0)
        for idx in records:
            if m == "nemo_scrip":
                full = sample_forcing_tripole_scrip(forcing, idx)
                forc = {k: np.asarray(v).reshape(-1)[scrip_sel]
                        for k, v in full.items()}
            else:
                forc = {k: np.asarray(v) for k, v in _sample_forcing_points(
                    forcing, idx, lat, lon, method=m).items()}
            q = record_quantities(forc, T_sfc_K)
            for k in QUANTITIES:
                acc[k] += float(np.sum(q[k] * w))
        out[m] = {k: v / len(records) for k, v in acc.items()}
        bad = [k for k, v in out[m].items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite box mean for {m}: {bad}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--forcing-dir", default=None)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)

    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.forcing.core2 import load_core2_nyf

    set_policy(PrecisionPolicy.fp64())       # production cards run fp64

    z = np.load(a.snapshot)
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64)
    ocean = np.asarray(z["land_mask"]) > 0.5         # 1 = wet in our snapshots
    sst = np.asarray(z["T_mean"], dtype=np.float64)[..., 0]
    tripole = lat.ndim == 2
    if tripole and lat.shape != (332, 362):
        raise SystemExit(f"2-D grid {lat.shape} is not the (332, 362) eORCA1 "
                         "tripole the nemo_scrip weights encode")
    sel = box_points(lat, lon, ocean)
    if not np.all(np.isfinite(sst[sel])):
        raise SystemExit("non-finite SST at a wet box point: wrong mask sense?")
    sst_box = sst[sel]
    if not (20.0 < float(np.mean(sst_box)) < 32.0):
        raise SystemExit(f"box SST mean {np.mean(sst_box):.2f} C is not a "
                         "cold-tongue value: wrong mask sense or field")
    w = (np.asarray(z["cell_area"], dtype=np.float64)[sel] if "cell_area" in z
         else np.ones(int(sel.sum())))
    forcing = load_core2_nyf(allow_synthetic=False,
                             cache_dir=Path(a.forcing_dir) if a.forcing_dir
                             else None)
    t = np.asarray(forcing.time_s, dtype=np.float64)
    records = [int(i) for i in np.nonzero(t < a.days * 86400.0)[0]]
    methods = METHODS + (("nemo_scrip",) if tripole else ())
    print(f"[{a.label}] {int(sel.sum())} wet box points, SST mean "
          f"{np.mean(sst_box):.3f} C, {len(records)} records "
          f"(centres {t[records[0]] / 86400:.2f}-{t[records[-1]] / 86400:.2f} d)"
          f", weights={'cell_area' if 'cell_area' in z else 'equal'}")
    res = box_means(forcing, records, lat[sel], lon[sel] % 360.0,
                    sst_box + constants.T_freeze, w, methods,
                    scrip_sel=np.flatnonzero(sel.reshape(-1)) if tripole else None)
    ref = res["bilinear"]
    print(f"{'method':14s}" + "".join(f"{k:>12s}" for k in QUANTITIES))
    for m, r in res.items():
        print(f"{m:14s}" + "".join(f"{r[k]:12.5g}" for k in QUANTITIES))
    for m, r in res.items():
        if m != "bilinear":
            print(f"{m + '-bil':14s}" + "".join(
                f"{r[k] - ref[k]:+12.4g}" for k in QUANTITIES))
    if a.json:
        a.json.write_text(json.dumps({"label": a.label, "box": BOX,
                                      "n_points": int(sel.sum()),
                                      "records": len(records), "means": res},
                                     indent=1))
    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
