#!/usr/bin/env python3
"""Rain-evaporation timestep diagnostic on an MPAS checkpoint (codex R1 item 3).

Morrison's m2005 rain evaporation is explicit: dq = EPSR*dt*(q_sat-q_v)/AB per
call, so where EPSR*dt > 1 one call evaporates MORE than the deficit the layer
can hold (overshooting saturation, which the adjustment then condenses back
into cloud water). This calls the model's OWN kernel on the checkpoint state,
tropics 20S-20N, and reports the distribution of EPSR*dt in raining cells, the
fraction of raining cells that overshoot, and the evaporation rate summed over
the column and over sigma > 0.7, in kg/m2/day.

Usage: rain_evap_timestep.py <run> [--day N] [--dt 112.5]
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

SEC_PER_DAY = 86400.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--day", type=int, default=None)
    ap.add_argument("--dt", type=float, default=112.5)
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics._warm_rain import rain_evaporation_m2005
    from legoesm.atmosphere.physics.microphysics.integration import number_per_mass_to_per_volume

    rundir = f"{rb.ROOT}/{args.run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    cks = sorted(glob.glob(f"{rundir}/checkpoint_day_*.npz"))
    ck = cks[-1] if args.day is None else f"{rundir}/checkpoint_day_{args.day:04d}.npz"
    z = np.load(ck, allow_pickle=True)
    if str(z["number_convention"]) != "per_mass":
        raise SystemExit(f"FATAL: unexpected number convention {z['number_convention']}")
    T, ps = np.asarray(z["T"]), np.asarray(z["p_s"])
    vg = np.asarray(z["meta_vgrid"])
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    dp = p_half[:, 1:] - p_half[:, :-1]
    sig = p_full / ps[:, None]
    q_v, q_r, N_r_m = (np.asarray(z[k]) for k in ("trc_q_v", "trc_q_r", "trc_N_r"))
    T_v = T * (1.0 + (1.0 / constants.epsilon - 1.0) * q_v)
    rho = p_full / (constants.R_d * T_v)
    cfg = MorrisonConfig()
    q_sat = np.asarray(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p_full)))
    N_r = np.asarray(number_per_mass_to_per_volume(jnp.asarray(N_r_m), jnp.asarray(rho)))
    ev = np.asarray(rain_evaporation_m2005(
        jnp.asarray(q_v), jnp.asarray(q_r), jnp.asarray(N_r), jnp.asarray(q_sat),
        jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(rho), cfg, dt=None))
    ev_capped = np.minimum(ev, np.maximum(q_r, 0.0) / args.dt)
    dqsdt = constants.L_v * q_sat / (constants.R_v * T ** 2)
    ab = 1.0 + dqsdt * constants.L_v / constants.c_pd
    deficit = np.maximum(q_sat - q_v, 0.0)
    raining = (q_r > 1.0e-6) & (deficit > 1.0e-5)
    epsr_dt = np.where(raining, ev * args.dt * ab / np.maximum(deficit, 1e-30), np.nan)

    lat, lon, area = mesh_coords(exp)
    trop = (lat >= -20.0) & (lat <= 20.0)
    w = area * trop
    if not np.all(np.isfinite(ev)):
        raise SystemExit("FATAL: non-finite evaporation from the kernel")
    print(f"=== {args.run} {ck.rsplit('/', 1)[1]}: m2005 rain evaporation, tropics 20S-20N, dt={args.dt} s ===")
    for name, band in (("sigma>0.7", sig > 0.7), ("0.3<sigma<=0.7", (sig > 0.3) & (sig <= 0.7))):
        m = raining & band & trop[:, None]
        x = epsr_dt[m]
        n = x.size
        col_ev = ((ev_capped * band) * dp / constants.g).sum(1)
        col_un = ((ev * band) * dp / constants.g).sum(1)
        print(f"{name:<16} raining cells {n:8d}  EPSR*dt p50 {np.nanpercentile(x, 50):7.3f}  "
              f"p90 {np.nanpercentile(x, 90):7.3f}  p99 {np.nanpercentile(x, 99):7.3f}  "
              f"frac>1 {np.mean(x > 1.0):5.3f}  frac>0.5 {np.mean(x > 0.5):5.3f}")
        print(f"{'':<16} evap rate  capped {(col_ev * w).sum() / w.sum() * SEC_PER_DAY:7.3f}  "
              f"uncapped {(col_un * w).sum() / w.sum() * SEC_PER_DAY:7.3f}  kg/m2/day (tropical mean, instant)")
    # storage check: rain water path so the ledger's steady-state inference is honest
    rwp = ((np.maximum(q_r, 0.0) * dp / constants.g) * w[:, None]).sum() / w.sum()
    print(f"tropical-mean rain water path {rwp:7.4f} kg/m2 (storage negligible vs 3.6 kg/m2/day if << 1)")


if __name__ == "__main__":
    main()
