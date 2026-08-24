#!/usr/bin/env python3
"""Is the free-tropospheric COLD bias big enough to explain the clear-sky OLR
deficit, or is something else missing?

The 12-month AMIP is -19.6 W/m2 in `rlutcs` GLOBALLY (-8.4 even at the poles),
and `profile_rh_split.py` measures a cold bias peaking at -5.2 K at 700 hPa
together with a mid/upper-tropospheric moist bias. Both push clear-sky OLR
down. Before either is treated as the cause, the arithmetic has to close: a
temperature bias of that size and shape must actually be worth ~20 W/m2.

This measures it instead of asserting it. The model's own RRTMGP is run
clear-sky on real production columns three times:

    base            model state as it is
    +dT             temperature warmed to ERA5 by the measured bias profile
    +dq             humidity scaled to ERA5 by the measured ratio profile

and the two deltas are reported against the -19.6 W/m2 to be explained.

READ THIS BEFORE QUOTING THE NUMBER. This is a SENSITIVITY of the radiation
operator to a prescribed profile perturbation, NOT a prediction of what the
coupled model would do: correcting the temperature in a live run changes the
circulation, the humidity and the clouds too. The claim it can support is
narrow and it is the one in question — "a bias of this size and shape is / is
not worth ~20 W/m2 of clear-sky OLR".

Two deliberate choices, stated because they bound the answer:

* The perturbation profile is a GLOBAL MEAN, so any covariance between the
  bias and the local state is discarded. Running the profile and the columns
  from DIFFERENT runs was flagged in review as a category error; measured, it
  is not one here — cldF_fsd day 1825 gives +3.82 / +8.07 / +12.80 and
  ref1979's own day 360 gives +3.82 / +8.16 / +12.89, so the split is robust
  to the state it is evaluated on. Quote the same-run numbers anyway.
* The correction is held at the IDENTITY above the top measured level
  (100 hPa). The model is +11 K warm there, and applying ERA5's cooling would
  push OLR DOWN, so the temperature figure below is an UPPER bound on what a
  full temperature correction buys.
* Clear-sky means `include_clouds=False`, matching how `rsutcs`/`rlutcs` are
  produced. The cloud fields are still loaded and simply not passed.

Usage:
    rlutcs_kernel.py [--run <run dir name>] [--day N] [--ncol N]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_HARNESS = "/work/bd1083/b309178/diffESM/legoesm_pg/_albedo_hifi/harness_hifi.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bias_profiles(ref_run):
    """Global-mean (dT[K], q_model/q_ERA5) on the CMOR plev axis, below-ground
    cells masked — the same reduction ``profile_rh_split`` prints."""
    prs = _load(str(pathlib.Path(__file__).with_name("profile_rh_split.py")),
                "prs")
    rb = prs.rb
    mt, mq = rb._load_model(ref_run, "ta"), rb._load_model(ref_run, "hus")
    if mt is None or mq is None:
        raise SystemExit(f"{ref_run}: no ta/hus CMOR output to build the "
                         "perturbation profile from")
    months = rb._month_labels(mt)
    mlat = np.asarray(mt.lat, dtype=np.float64)
    mlon = np.asarray(mt.lon, dtype=np.float64) % 360.0
    plev_all = np.asarray(mt.plev, dtype=np.float64)
    keep = plev_all >= prs.P_FLOOR
    plev = plev_all[keep]
    Tm = np.asarray(mt["ta"]).mean(axis=0)[keep]
    qm = np.asarray(mq["hus"]).mean(axis=0)[keep]
    Te = prs._era5_on_model("ta", months, plev, mlat, mlon)
    qe = prs._to_mixing_ratio(prs._era5_on_model("hus", months, plev, mlat, mlon))

    ps_m = np.asarray(rb._load_model(ref_run, "ps")["ps"]).min(axis=0)
    ps_e = np.min([rb.bin_to_model(*prs._era5_surface_pressure([m]), mlat, mlon,
                                   label="ERA5 ps")
                   for m in sorted(set(months))], axis=0)
    below = plev[:, None, None] > np.minimum(ps_m, ps_e)[None, :, :]

    box = rb.REGIONS["GLOBAL"]
    dT = np.array([rb.region_mean(Te[k] - Tm[k], mlat, mlon, box, ~below[k])
                   for k in range(plev.size)])
    ratio = np.array([rb.region_mean(qe[k], mlat, mlon, box, ~below[k])
                      / rb.region_mean(qm[k], mlat, mlon, box, ~below[k])
                      for k in range(plev.size)])
    return plev, dT, ratio


def _on_columns(plev, prof, p_full, fill):
    """Interpolate a profile given on ``plev`` onto each column's p_full.

    Linear in log-p. OUTSIDE the plev range the profile is held at ``fill``
    (not extrapolated): above the top plev row the correction is unmeasured,
    and extrapolating a +11 K upper-level bias into the model's lid would
    manufacture the answer.
    """
    lp, lq = np.log(plev[::-1]), prof[::-1]
    out = np.interp(np.log(np.asarray(p_full)), lp, lq, left=fill, right=fill)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="Run whose columns are used (default: the harness's "
                         "own run, which is what it was reviewed against).")
    ap.add_argument("--ref-run", default="ref1979",
                    help="Run the bias PROFILE is measured from.")
    ap.add_argument("--day", type=int, default=1155)
    ap.add_argument("--ncol", type=int, default=2048)
    ap.add_argument("--n-tod", type=int, default=4)
    ap.add_argument("--target", type=float, default=-19.6,
                    help="The rlutcs error to be explained [W/m2].")
    args = ap.parse_args(argv)

    hh = _load(_HARNESS, "hh")
    if args.run:
        hh.RUN = f"{hh.RUN.rsplit('/', 1)[0]}/{args.run}"
    plev, dT, ratio = bias_profiles(args.ref_run)
    print(f"perturbation profile from {args.ref_run} (global mean, "
          f"below-ground masked):")
    for p, t, r in zip(plev, dT, ratio):
        print(f"   {p / 100:7.0f} hPa   dT(ERA5-model) {t:+6.2f} K   "
              f"q_ERA5/q_model {r:5.2f}")

    s = hh.load_state(args.day, ncol=args.ncol)
    w = np.asarray(s["w"], dtype=np.float64)
    p_full = np.asarray(s["p_full"], dtype=np.float64)

    # fill=0.0 for dT and 1.0 for the ratio: outside the measured range the
    # perturbation is the IDENTITY, never an extrapolation.
    dT_col = _on_columns(plev, dT, p_full, 0.0)
    r_col = _on_columns(plev, ratio, p_full, 1.0)

    def olrcs(state):
        flux, _ = hh.diurnal(state, n_tod=args.n_tod, include_clouds=False,
                            n_sub=1)
        return hh.gm(flux[2], w)          # rlut with clouds off = rlutcs

    import jax.numpy as jnp
    base = dict(s)
    warm = dict(s, T=jnp.asarray(np.asarray(s["T"]) + dT_col))
    dry = dict(s, q_v=jnp.asarray(np.asarray(s["q_v"]) * r_col))
    both = dict(s, T=warm["T"], q_v=dry["q_v"])

    o_base = olrcs(base)
    o_warm = olrcs(warm)
    o_dry = olrcs(dry)
    o_both = olrcs(both)

    print(f"\ncolumns: {args.ncol} from {hh.RUN.rsplit('/', 1)[-1]} "
          f"day {args.day}, {args.n_tod} times of day, clear sky")
    print(f"  base rlutcs                        {o_base:8.2f} W/m2")
    print(f"  + temperature correction to ERA5   {o_warm:8.2f}  "
          f"(d = {o_warm - o_base:+6.2f})")
    print(f"  + humidity correction to ERA5      {o_dry:8.2f}  "
          f"(d = {o_dry - o_base:+6.2f})")
    print(f"  + both                             {o_both:8.2f}  "
          f"(d = {o_both - o_base:+6.2f})")
    print(f"\nrlutcs error to explain: {args.target:+.1f} W/m2 "
          f"(model - obs), so a correction must supply {-args.target:+.1f}.")
    print(f"  temperature alone covers {100 * (o_warm - o_base) / -args.target:5.1f} %")
    print(f"  humidity alone covers    {100 * (o_dry - o_base) / -args.target:5.1f} %")
    print(f"  both together cover      {100 * (o_both - o_base) / -args.target:5.1f} %")
    print("\nSENSITIVITY OF THE RADIATION OPERATOR, not a coupled prediction: "
          "correcting T in a live run also moves the circulation, the humidity "
          "and the clouds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
