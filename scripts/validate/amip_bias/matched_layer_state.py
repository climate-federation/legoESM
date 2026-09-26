#!/usr/bin/env python3
"""Humidity and RH on matched PRESSURE layers from checkpoints, for runs on
different vertical grids.

The lowest model level sits at a different height on a refined grid, so the
surface-layer bias is scored on the mass-weighted mean over fixed pressure
layers (the lowest 100 hPa above the surface, and 925-1000 hPa), tropical
OCEAN (sftlf < 0.5 from dd_ctl's fx, same mesh), with the model's own
saturation curve.

Usage: matched_layer_state.py --day 90 <run> [<run> ...]
"""
from __future__ import annotations

import argparse
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

    exp = json.load(open(f"{rb.ROOT}/dd_ctl/experiment_config.json"))
    lat, lon, area = mesh_coords(exp)
    fl = _sftlf_on_mesh("dd_ctl", lat, lon)
    oc = (np.abs(lat) <= 20.0) & (fl < 0.5)
    w = area * oc
    print(f"day {args.day}, tropical ocean, mass-weighted over matched pressure layers")
    print(f"{'run':<12}{'lowest100 q':>12}{'RH':>7}{'T':>8}{'925-1000 q':>12}{'RH':>7}{'T':>8}{'p_low hPa':>10}")
    for run in args.runs:
        z = np.load(f"{rb.ROOT}/{run}/checkpoint_day_{args.day:04d}.npz", allow_pickle=True)
        T, ps, q = np.asarray(z["T"]), np.asarray(z["p_s"]), np.asarray(z["trc_q_v"])
        vg = np.asarray(z["meta_vgrid"])
        p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
        p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1]); dp = p_half[:, 1:] - p_half[:, :-1]
        qs = np.asarray(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p_full)))
        if not np.all(np.isfinite(q)):
            raise SystemExit(f"FATAL: non-finite q in {run}")
        row = f"{run:<12}"
        for lo, hi in ((ps[:, None] - 1.0e4, ps[:, None] + 1.0), (92500.0, 100000.0)):
            m = (p_full >= lo) & (p_full <= hi); ww = dp * m
            col = lambda f: (f * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
            g = lambda f: (col(f) * w).sum() / w.sum()
            row += f"{g(q) * 1e3:12.2f}{g(q / qs):7.3f}{g(T):8.2f}"
        row += f"{np.average(p_full[:, -1], weights=w) / 100:10.0f}"
        print(row)


if __name__ == "__main__":
    main()
