#!/usr/bin/env python3
"""How much of each layer's convective vapour sink is the RAIN DEBIT.

The Bechtold port removes the water that becomes convective rain from the
environment in proportion to each level's vapour mass (``q_v * dp``), not
at the levels the plume collected it (``bechtold.py``, the in-plume precip
water-budget coupling), and warms each debited level by ``L_v/c_p`` times
the debit.  So a band's share of the column's convective rain sink is just
its share of the column's vapour mass -- computable from a checkpoint alone,
no scheme call.  This prints that debit per sigma band next to the band's
share of column vapour, so the banded-ledger convection term can be split
into "rain debit" and "everything else" (mass-flux transport, detrainment,
downdraft) by subtraction.

Provenance: ``physstate_conv_precip`` is the checkpoint-INSTANT convective
rain source (after downdraft + sub-cloud evaporation), so the debit is an
instantaneous estimate; the ledger term it is compared to is a daily mean.
Usage: conv_rain_debit_by_band.py --day 135 <run> [<run> ...] [--sftlf-from <run>]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from cloud_layers import mesh_coords  # noqa: E402
from ledger_regional import BOXES, SEC_PER_DAY, _sftlf_on_mesh  # noqa: E402
from legoesm import constants  # noqa: E402

BANDS = ((0.967, 1.0), (0.91, 0.967), (0.83, 0.91), (0.70, 0.83), (0.0, 0.70))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--sftlf-from", default=None)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    for run in a.runs:
        rundir = f"{rb.ROOT}/{run}"
        exp = json.load(open(f"{rundir}/experiment_config.json"))
        lat, lon, area = mesh_coords(exp)
        fl = _sftlf_on_mesh(a.sftlf_from or run, lat, lon)
        z = np.load(f"{rundir}/checkpoint_day_{a.day:04d}.npz", allow_pickle=True)
        if "physstate_conv_precip" not in z:
            raise SystemExit(f"FATAL: {run} checkpoint has no physstate_conv_precip")
        q, ps = np.asarray(z["trc_q_v"]), np.asarray(z["p_s"])
        vg = np.asarray(z["meta_vgrid"])
        p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
        dp = p_half[:, 1:] - p_half[:, :-1]                       # top-down, > 0
        sig_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1]) / ps[:, None]
        vmass = np.maximum(q, 0.0) * dp                           # [kg/kg Pa]
        share = {b: (vmass * ((sig_full >= b[0]) & (sig_full < b[1]))).sum(1) / vmass.sum(1)
                 for b in BANDS}
        rain = np.asarray(z["physstate_conv_precip"]) * SEC_PER_DAY   # kg/m2/day
        print(f"\n=== {run}: convective rain debit by sigma band, checkpoint day {a.day} "
              f"[kg/m2/day; share = band's fraction of column vapour mass] ===")
        print(f"{'region':<26}{'conv rain':>10}" + "".join(
            f"{f'{b[0]:.2f}-{b[1]:.2f}':>16}" for b in BANDS))
        for name, (lo, hi) in BOXES.items():
            w = area * ((lat >= lo) & (lat <= hi)) * (1.0 - fl)
            if w.sum() <= 0.0:
                continue
            m = lambda f: float((f * w).sum() / w.sum())  # noqa: E731
            cells = [f"{m(rain * share[b]):7.2f} ({m(share[b]):4.2f})" for b in BANDS]
            print(f"{name + ' ocean':<26}{m(rain):10.2f}" + "".join(f"{c:>16}" for c in cells))
    print("\nA band's rain debit is the part of its ledger 'convection' term that is the "
          "vapour-mass-weighted rain sink;\nthe remainder is mass-flux transport, "
          "detrainment and downdraft. Debited water is warmed by L_v/c_p at the same levels.")


if __name__ == "__main__":
    main()
