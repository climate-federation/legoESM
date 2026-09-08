#!/usr/bin/env python
"""Is the cold-tongue warm bias a FORCING problem or a DYNAMICS problem?

WHY.  nino3 is our largest error (+3.03 C at day 30, three times any other SST
band) and the lateral-viscosity hypothesis was refuted: cutting the
Smagorinsky closure 6.8x moved it 0.09 C and the Equatorial Undercurrent 5%.
Before proposing another mechanism, this splits the question the cheap way.

Two channels reach the box, and NEMO publishes both on the same 5-day file:

  HEAT      ``qt_oce`` = net heat into the ocean [W/m2].  If ours is more
            positive than NEMO's, the box is warm because we put heat in --
            a bulk/forcing defect, and no amount of circulation work fixes it.
  MOMENTUM  ``taum`` = wind-stress magnitude [N/m2].  The undercurrent is
            driven by the trades.  If our stress is weak, the EUC is weak for
            a forcing reason, not a viscous one, and that would also explain
            why removing viscosity did not restore it.

Ours is rebuilt through the SAME production bulk path the run used
(``air_sea_fluxes``), not a re-derived lookalike, at the forcing record whose
centre matches the snapshot day.  Sign conventions are the function's own:
``shflx``/``lhflx`` positive INTO the ocean, ``tau_x``/``tau_y`` in the
ATMOSPHERIC convention.

READS.  On matched days over the nino3 box (5S-5N, 150W-90W):
  * heat within ~10 W/m2 and stress within ~10%  => forcing is fine, the bias
    is DYNAMICS (upwelling/EUC), and the next work is the momentum balance.
  * our heat more positive by >20 W/m2           => FORCING; work the bulk
    formulation before any more circulation arms.
  * our stress weak by >20%                      => FORCING-driven weak EUC;
    the viscosity result is then explained and the trades are the target.

CAVEAT bounding every number: NEMO's is a 5-day mean, ours is one instant of
the matched window, and the probe refuses to run if the snapshot day falls
outside the NEMO averaging window it is scored against.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_EMISS = 0.98   # NEMO longwave emissivity (namsbc_blk)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

BOXES = {
    "nino3_5S5N_150W90W": (-5.0, 5.0, 210.0, 270.0),
    "nino4_5S5N_160E150W": (-5.0, 5.0, 160.0, 210.0),
    "eq_pacific_2S2N": (-2.0, 2.0, 160.0, 270.0),
}


def _aligner():
    spec = importlib.util.spec_from_file_location(
        "so_freshwater_budget", _HERE.parent / "so_freshwater_budget.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--nemo-sbc", required=True, type=Path)
    ap.add_argument("--nemo-recs", default="5:6")
    ap.add_argument("--day", type=float, default=30.0)
    ap.add_argument("--mesh", type=Path,
                    default=Path("data/grids/eORCA1.2_mesh_mask.nc"))
    ap.add_argument("--forcing-path", default=None)
    a = ap.parse_args()

    import netCDF4 as nc
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.forcing.core2 import load_core2_nyf
    from legoesm.ocean.coupler.omip2_applicator import _sample_forcing_points
    from legoesm.ocean.bulk_flux_omip import air_sea_fluxes

    mod = _aligner()
    lo, hi = (int(x) for x in a.nemo_recs.split(":"))

    ds = nc.Dataset(a.nemo_sbc)
    try:
        def rd(n):
            x = np.ma.filled(np.ma.masked_invalid(ds[n][lo:hi]), np.nan)
            return np.nanmean(x.astype(np.float64), axis=0)
        qt_n, taum_n = rd("qt_oce"), rd("taum")
        lat_n = np.asarray(ds["nav_lat"][:], dtype=np.float64)
        lon_n = np.asarray(ds["nav_lon"][:], dtype=np.float64) % 360.0
        tb = np.asarray(ds["time_centered_bounds"][:], dtype=np.float64)
    finally:
        ds.close()

    t0 = tb[0, 0]
    win = ((tb[lo, 0] - t0) / 86400.0, (tb[hi - 1, 1] - t0) / 86400.0)
    print(f"[window] NEMO recs {a.nemo_recs} = days {win[0]:.2f}-{win[1]:.2f}; "
          f"snapshot day {a.day:.2f}")
    if not (win[0] <= a.day <= win[1]):
        raise SystemExit(
            f"WINDOW MISMATCH: day {a.day} outside NEMO average "
            f"{win[0]:.2f}-{win[1]:.2f}. Refusing to compare.")

    d = nc.Dataset(a.mesh)
    try:
        area_m = (np.squeeze(d["e1t"][:]) * np.squeeze(d["e2t"][:])).astype(np.float64)
        lat_mesh = np.squeeze(d["gphit"][:]).astype(np.float64)
    finally:
        d.close()
    sj, si = mod.align_output_to_mesh(lat_mesh, lat_n)
    area_n = area_m[sj, si]

    z = np.load(a.snapshot)
    lat_o = np.asarray(z["lat_T"], dtype=np.float64)
    lon_o = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
    mask_o = np.asarray(z["land_mask"], dtype=np.float64) > 0.5

    forcing = load_core2_nyf(cache_dir=(Path(a.forcing_path)
                                        if a.forcing_path else None),
                             allow_synthetic=False)
    t = np.asarray(forcing.time_s, dtype=np.float64)
    idx = int(np.argmin(np.abs(t - a.day * 86400.0)))
    if not (win[0] <= t[idx] / 86400.0 <= win[1]):
        raise SystemExit(
            f"forcing record {idx} centre {t[idx] / 86400.0:.2f} d is outside "
            f"the NEMO window {win[0]:.2f}-{win[1]:.2f}")
    print(f"[forcing] record {idx}, centre {t[idx] / 86400.0:.2f} d")

    # CHANNEL NAMES ARE READ, NOT GUESSED. An earlier draft used swdown /
    # lwdown, which do not exist -- the real names are sw_down / lw_down -- so
    # a .get() fallback would have fed ZERO radiation in silently and produced
    # a confidently wrong net heat flux. Missing channels now raise.
    shp = lat_o.shape
    f = _sample_forcing_points(forcing, idx, lat_o.reshape(-1), lon_o.reshape(-1))
    f = {k: np.asarray(v).reshape(shp) for k, v in f.items()}
    for need in ("sw_down", "lw_down", "u10", "v10", "T_air", "q_air"):
        if need not in f:
            raise SystemExit(
                f"forcing channel {need!r} missing; got {sorted(f)}. Refusing "
                "to substitute a default -- a zero radiative term would give a "
                "plausible but wrong net heat flux.")
    T_sfc_K = np.asarray(z["T"], dtype=np.float64)[..., 0] + constants.T_freeze
    tau_x, tau_y, shflx, lhflx, _evap = air_sea_fluxes(
        u10=jnp.asarray(f["u10"]), v10=jnp.asarray(f["v10"]),
        T_air_K=jnp.asarray(f["T_air"]), q_air=jnp.asarray(f["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if f.get("slp") is None else jnp.asarray(f["slp"]),
    )
    taum_o = np.hypot(np.asarray(tau_x), np.asarray(tau_y))
    sw, lw = f["sw_down"], f["lw_down"]
    lwup = constants.sigma_sb * _EMISS * T_sfc_K ** 4
    qt_o = ((1.0 - constants.alpha_ocean_broadband) * sw + _EMISS * lw - lwup
            + np.asarray(shflx) + np.asarray(lhflx))

    def box_mean(fld, lat, lon, w, b):
        la, lb, oa, ob = b
        m = (lat >= la) & (lat <= lb) & (lon >= oa) & (lon <= ob) & np.isfinite(fld) & (w > 0)
        if not m.any():
            return np.nan, 0
        return float(np.nansum(fld[m] * w[m]) / np.nansum(w[m])), int(m.sum())

    w_o = area_m * mask_o
    w_n = area_n * np.isfinite(qt_n)
    print(f"\n{'box':22s}{'qt ours':>10s}{'qt NEMO':>10s}{'d(W/m2)':>10s}"
          f"{'taum ours':>11s}{'taum NEMO':>11s}{'ratio':>8s}")
    for name, b in BOXES.items():
        qo, no = box_mean(qt_o, lat_o, lon_o, w_o, b)
        qn, nn = box_mean(qt_n, lat_n, lon_n, w_n, b)
        to, _ = box_mean(taum_o, lat_o, lon_o, w_o, b)
        tn, _ = box_mean(taum_n, lat_n, lon_n, w_n, b)
        print(f"  {name:20s}{qo:10.1f}{qn:10.1f}{qo - qn:10.1f}"
              f"{to:11.4f}{tn:11.4f}{(to / tn if tn else np.nan):8.2f}")
    print("\nHeat within ~10 W/m2 AND stress ratio ~1 => forcing is fine, the "
          "bias is dynamics.\nHeat more positive by >20 W/m2 => forcing.  "
          "Stress ratio <0.8 => weak trades drive the weak EUC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
