#!/usr/bin/env python3
"""Native lowest-model-level q, RH and T from checkpoints.

The pressure-level diagnostics clamp "1000 hPa" to the lowest full level, so
the surface-layer moisture bias is scored here on the NATIVE level (sigma
0.9835, ~150 m) with the model's own saturation curve, area-weighted over the
tropical OCEAN (sftlf < 0.5) and land.  (The checkpoint's T_sfc override slot
is not the prescribed SST, so no sea-air deficit is derived here; use the
published hfls / evspsbl for the flux side.)

Usage: lowest_level_state.py --day 85 <run> [<run> ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from cloud_layers import mesh_coords  # noqa: E402
from ledger_regional import _sftlf_on_mesh  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--day", type=int, required=True)
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio

    print(f"{'run':<14}{'surface':<8}{'q_low g/kg':>11}{'RH_low':>8}{'T_low K':>9}{'p_low hPa':>10}")
    for run in args.runs:
        rundir = f"{rb.ROOT}/{run}"
        exp = json.load(open(f"{rundir}/experiment_config.json"))
        lat, lon, area = mesh_coords(exp)
        try:
            fl = _sftlf_on_mesh(run, lat, lon)
        except SystemExit:
            fl = _sftlf_on_mesh("dd_ctl", lat, lon)   # same mesh; fx not yet published
        z = np.load(f"{rundir}/checkpoint_day_{args.day:04d}.npz", allow_pickle=True)
        T, ps, q = np.asarray(z["T"]), np.asarray(z["p_s"]), np.asarray(z["trc_q_v"])
        vg = np.asarray(z["meta_vgrid"])
        p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
        p_low = 0.5 * (p_half[:, -1] + p_half[:, -2])
        q_low, T_low = q[:, -1], T[:, -1]
        qs_low = np.asarray(saturation_mixing_ratio(jnp.asarray(T_low), jnp.asarray(p_low)))
        trop = (lat >= -20.0) & (lat <= 20.0)
        for surf, w_s in (("ocean", (fl < 0.5)), ("land", (fl >= 0.5))):
            w = area * trop * w_s
            m = lambda f: float((f * w).sum() / w.sum())
            line = (f"{run:<14}{surf:<8}{m(q_low) * 1e3:11.2f}{m(q_low / qs_low):8.3f}{m(T_low):9.2f}"
                    f"{m(p_low) / 100:10.1f}")
            print(line)


if __name__ == "__main__":
    main()
