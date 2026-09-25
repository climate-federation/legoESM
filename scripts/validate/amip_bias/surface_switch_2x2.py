#!/usr/bin/env python3
"""The 2x2 of the two surface-flux switches, replayed offline on stored states.

Both reviewers made the same P1 objection to the 30-day pair: the two switches
were moved TOGETHER, so the experiment identifies the package and neither
switch individually, and both named a fixed-state replay as the cheapest way
to separate them.  Codex: "the fixed-state replay ... using the shared restart
and all four switch combinations, before buying another 30-day branch."

The climate 2x2 costs 13 GPU-hours.  The FLUX 2x2 is free: the switches act
inside one routine, so feeding it one set of stored columns four times gives
each switch's contribution and their interaction exactly, with no integration
and no trajectory noise.  What it cannot give is the climate response -- the
boundary layer adjusts, and the adjusted flux change is smaller than this one.
This is the instrument reading, not the run.

  z_ref_model_level : tell the solver the level's TRUE height instead of 10 m
  ocean_q_sfc_saline: reduce the sea-surface saturation humidity by 2%

The state is the published 1000 hPa level, which the diagnostics clamp to the
lowest full model level, masked to ocean columns whose surface pressure is
above the level on BOTH sides so the sounding is not underground.

Usage: surface_switch_2x2.py <run> [--day N]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from surface_humidity_deficit import _prescribed_sst  # noqa: E402

SALINE = 0.98
BANDS = {"tropical ocean 20S-20N": (-20, 20, 0, 360),
         "trades 10-30N": (10, 30, 0, 360),
         "trades 10-30S": (-30, -10, 0, 360),
         "global ocean": (-90, 90, 0, 360)}


def lowest_level_height(run, T_low, p_low, p_s):
    """Height of the lowest full level [m], hypsometric from the run's own
    sigma coordinate.  Read from the run's checkpoint rather than assumed:
    the level placement changed between decks, and a wrong height here would
    silently rescale the very factor this probe exists to measure."""
    import glob
    cks = sorted(glob.glob(f"{rb.ROOT}/{run}/checkpoint_day_*.npz"))
    if not cks:
        raise SystemExit(f"FATAL: {run} has no checkpoint to read the grid from")
    vg = np.load(cks[-1], allow_pickle=True)["meta_vgrid"]
    from legoesm import constants
    # full-level sigma of the lowest layer, from the half levels
    sig = 0.5 * (vg[1][-1] + vg[1][-2])
    return (constants.R_d * T_low / constants.g) * np.log(1.0 / sig), float(sig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gustiness", type=float, nargs="*", default=None,
                    help="also sweep the COARE convective-gustiness depth z_i "
                         "[m] with BOTH corrections on (production 300, "
                         "COARE-native 600)")
    args = ap.parse_args(argv)
    from legoesm import constants
    from legoesm.core.bulk_flux import compute_most_fluxes
    from legoesm.thermo import saturation_specific_humidity

    fields = {v: rb._load_model(args.run, v) for v in ("ta", "hus", "ua", "va", "ps")}
    missing = [k for k, v in fields.items() if v is None]
    if missing:
        raise SystemExit(f"FATAL: {args.run} publishes no {missing}")
    d0 = fields["ta"]
    lat, lon = np.asarray(d0.lat), np.asarray(d0.lon)
    months = rb._month_labels(d0)
    plev = np.asarray(d0["plev"], dtype=np.float64)
    k = int(np.argmin(np.abs(plev - 100000.0)))

    def low(v):
        return np.asarray(fields[v][v]).mean(0)[k]

    T_a, q_a, u_a, v_a = low("ta"), low("hus"), low("ua"), low("va")
    p_s = np.asarray(fields["ps"]["ps"]).mean(0)
    sst = _prescribed_sst(args.run, months, lat, lon)

    # The land fraction is an fx field, not part of the monthly stream, so
    # the monthly loader returns None for it.  The previous version treated
    # that None as "no land" and silently scored LAND columns as ocean --
    # exactly the plausible-looking fallback this repo keeps getting caught
    # by.  Read the fx file, and fail loudly when it is absent.
    import glob
    _fs = sorted(glob.glob(f"{rb.ROOT}/{args.run}/cmor/fx/sftlf_fx_*.nc"))
    if not _fs:
        raise SystemExit(f"FATAL: {args.run} publishes no sftlf; refusing to "
                         "average ocean fluxes over an unknown land mask")
    import xarray as xr
    _d = xr.open_dataset(_fs[0])
    f_land = np.asarray(rb.bin_to_model(
        np.asarray(_d["sftlf"], dtype=np.float64) / 100.0,
        np.asarray(_d.lat), np.asarray(_d.lon) % 360.0, lat, lon,
        label="sftlf"))
    # Ocean columns only, and only where the 1000 hPa level is above ground:
    # a clamped underground sounding is a copy of the level above, not a
    # measurement, and it would enter the mean as a fictitious flux.
    valid = (f_land < 0.5) & (p_s > 100500.0) & np.isfinite(sst)
    print(f"{args.run}: {100.0 * valid.mean():.1f}% of cells are valid ocean "
          f"columns (ocean, 1000 hPa above ground, SST defined)")

    z_true, sig = lowest_level_height(args.run, T_a, 100000.0, p_s)
    rho = 100000.0 / (constants.R_d * T_a)
    wind = np.hypot(u_a, v_a)
    q_fresh = np.asarray(saturation_specific_humidity(sst, p_s))

    def lh(z_model_level, saline):
        z = np.where(valid, z_true, 100.0) if z_model_level else np.full_like(T_a, 10.0)
        q_sfc = q_fresh * (SALINE if saline else 1.0)
        _tx, _ty, _sh, lhf, _us = compute_most_fluxes(
            wind, np.zeros_like(wind), T_a, q_a, sst, q_sfc, rho,
            z_ref=z, scheme="coare3", stability_scheme="dyer1974")
        return np.asarray(lhf)

    arms = {"OFF (10 m, fresh)": lh(False, False),
            "height only": lh(True, False),
            "saline only": lh(False, True),
            "ON (both)": lh(True, True)}
    # Reference evaporation as a latent heat flux, same months, ERA5.
    ref = rb._ref_clim("evspsbl", months, lat, lon)
    ref_lh = np.asarray(ref) * constants.L_v if ref is not None else None

    print(f"lowest full level sigma {sig:.4f}, mean height "
          f"{float(np.nanmean(np.where(valid, z_true, np.nan))):.1f} m\n")
    print(f"{'band':<24}" + "".join(f"{a:>20}" for a in arms) + f"{'ERA5':>10}")
    for name, box in BANDS.items():
        cells = [rb.region_mean(a, lat, lon, box, valid=valid) for a in arms.values()]
        r = (f"{rb.region_mean(ref_lh, lat, lon, box, valid=valid):10.1f}"
             if ref_lh is not None else f"{'--':>10}")
        print(f"{name:<24}" + "".join(f"{c:20.1f}" for c in cells) + r)
    print("\nlatent heat flux [W/m2], area-weighted over valid ocean columns.\n")

    print(f"{'band':<24}{'height %':>10}{'saline %':>10}{'both %':>10}"
          f"{'interaction %':>15}")
    for name, box in BANDS.items():
        m = {k_: rb.region_mean(v, lat, lon, box, valid=valid)
             for k_, v in arms.items()}
        base = m["OFF (10 m, fresh)"]
        dh = 100.0 * (m["height only"] - base) / base
        ds = 100.0 * (m["saline only"] - base) / base
        db = 100.0 * (m["ON (both)"] - base) / base
        print(f"{name:<24}{dh:10.2f}{ds:10.2f}{db:10.2f}{db - dh - ds:15.2f}")
    if args.gustiness:
        # The production deck halves COARE's native gustiness depth (300 m
        # against 600), and that value is a CONVERGED tuning from a campaign
        # that ran with the surface flux inflated 16-18% by the height bug.
        # Lowering gustiness lowers the wind floor and so lowers evaporation,
        # which is the shape of a knob fitted to cancel the inflation.  With
        # the inflation gone it is the first thing that should move, so its
        # effect is measured here rather than assumed.
        from legoesm.atmosphere.physics.turbulence.config import (
            SurfaceLayerConfig)

        def lh_gust(zi):
            cfg = SurfaceLayerConfig(bulk_scheme="coare3",
                                     gustiness_w_zi=float(zi))
            _tx, _ty, _sh, lhf, _us = compute_most_fluxes(
                wind, np.zeros_like(wind), T_a, q_a, sst,
                q_fresh * SALINE, rho, z_ref=np.where(valid, z_true, 100.0),
                scheme="coare3", stability_scheme="dyer1974",
                gustiness_w_zi=float(zi))
            return np.asarray(lhf)

        print(f"\n{'band':<24}" + "".join(f"{'zi=' + str(int(z)):>12}"
                                          for z in args.gustiness))
        for name, box in BANDS.items():
            cells = [rb.region_mean(lh_gust(z), lat, lon, box, valid=valid)
                     for z in args.gustiness]
            print(f"{name:<24}" + "".join(f"{c:12.1f}" for c in cells))
        print("latent heat flux [W/m2] with BOTH corrections on, sweeping the "
              "gustiness depth.\nProduction runs 300 m; COARE 3.0's own value "
              "is 600 m.")

    print("\npercent change of the flux from the production default. The "
          "interaction column is the part of the package that is NOT the sum "
          "of the two switches; near zero means they are separable.")


if __name__ == "__main__":
    main()
