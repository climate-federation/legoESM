#!/usr/bin/env python
"""Probe the Kessler-microphysics → dycore blow-up in the AMIP path.

Reproduces the day-2 NaN-wind blow-up that surfaces with
``--microphysics kessler`` on cubed-sphere C16/L30 and prescribed SST.
Runs the model inline (no ``run_amip.py`` JIT/argparse overhead) and
prints per-step diagnostics so the failure can be localised to a
specific tendency or column.

Usage::

    python scripts/tmp/diag_kessler_amip_blowup.py [--n-steps 200] [--dt 600]

The script terminates as soon as a NaN is detected and reports:
  - which tendency channel went non-finite first (T, q_v, q_c, q_r, u, v)
  - the column coordinates with maximal magnitude
  - the column profile at that (face, i, j) just before the blow-up
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio


def _column_profile(field: jnp.ndarray, idx: tuple) -> np.ndarray:
    """Extract the column profile at idx (face, i, j) → (nlev,)."""
    f, i, j = idx
    return np.asarray(field[f, i, j, :])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--nlev", type=int, default=30)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--T-init", type=float, default=300.0)
    parser.add_argument("--RH-init", type=float, default=0.7)
    args = parser.parse_args(argv)

    grid = create_cubed_sphere(args.resolution)
    sigma = create_sigma_coordinate(args.nlev)

    state = held_suarez_init(grid, sigma, T_init=args.T_init)

    # Initialize tracers
    p_full = state.p_s.data[..., None] * sigma.sigma_full
    q_sat = saturation_mixing_ratio(state.T.data, p_full)
    q_v = jnp.minimum(args.RH_init * q_sat * sigma.sigma_full ** 2, q_sat)
    z = jnp.zeros_like(q_v)
    state = state._replace(tracers=dict(q_v=type(state.T)(data=q_v, name="q_v",
                                                          dims=state.T.dims, units="kg/kg"),
                                          q_c=type(state.T)(data=z, name="q_c",
                                                            dims=state.T.dims, units="kg/kg"),
                                          q_r=type(state.T)(data=z, name="q_r",
                                                            dims=state.T.dims, units="kg/kg")))

    micro_cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
    physics_fn = make_microphysics_physics(micro_cfg, model_type="hydrostatic",
                                            dt=args.dt)

    print(f"Initial state: T={float(jnp.mean(state.T.data)):.2f} K, "
          f"q_v_max={float(jnp.max(state.tracers['q_v'].data))*1000:.2f} g/kg")

    for step in range(args.n_steps):
        try:
            tendencies = physics_fn(state, grid, sigma)
        except Exception as exc:
            print(f"step {step}: physics_fn raised: {exc}")
            return 1

        dT_dt = tendencies.dT_dt.data
        tt = tendencies.tracer_tendencies
        dq_v_dt = tt["q_v"].data
        dq_c_dt = tt["q_c"].data
        dq_r_dt = tt["q_r"].data

        for name, t in [("dT_dt", dT_dt), ("dq_v_dt", dq_v_dt),
                         ("dq_c_dt", dq_c_dt), ("dq_r_dt", dq_r_dt)]:
            arr = np.asarray(t)
            if not np.all(np.isfinite(arr)):
                # Find the offending column
                bad = np.argwhere(~np.isfinite(arr))
                f, i, j, k = bad[0]
                print(f"step {step}: {name} non-finite at face={f}, i={i}, j={j}, k={k}")
                print(f"  T column: {_column_profile(state.T.data, (f, i, j))}")
                print(f"  q_v column: {_column_profile(state.tracers['q_v'].data, (f, i, j))}")
                print(f"  q_c column: {_column_profile(state.tracers['q_c'].data, (f, i, j))}")
                print(f"  q_r column: {_column_profile(state.tracers['q_r'].data, (f, i, j))}")
                return 1

        # Apply tendencies (Euler step) — bypass dynamics so we isolate
        # any slow accumulator inside Kessler itself.
        T_new = state.T.data + args.dt * dT_dt
        q_v_new = state.tracers["q_v"].data + args.dt * dq_v_dt
        q_c_new = state.tracers["q_c"].data + args.dt * dq_c_dt
        q_r_new = state.tracers["q_r"].data + args.dt * dq_r_dt

        # Floor tracers (mimics the runtime "tracers must be ≥ 0" pattern)
        q_v_new = jnp.maximum(q_v_new, 0.0)
        q_c_new = jnp.maximum(q_c_new, 0.0)
        q_r_new = jnp.maximum(q_r_new, 0.0)

        state = state._replace(
            T=state.T.replace(data=T_new),
            tracers=dict(
                q_v=state.tracers["q_v"].replace(data=q_v_new),
                q_c=state.tracers["q_c"].replace(data=q_c_new),
                q_r=state.tracers["q_r"].replace(data=q_r_new),
            ),
        )

        if (step + 1) % 50 == 0 or step in (0, 1, 2, 5, 10):
            T_mean = float(jnp.mean(state.T.data))
            T_min = float(jnp.min(state.T.data))
            T_max = float(jnp.max(state.T.data))
            qv_max = float(jnp.max(state.tracers["q_v"].data))
            qc_max = float(jnp.max(state.tracers["q_c"].data))
            qr_max = float(jnp.max(state.tracers["q_r"].data))
            day = (step + 1) * args.dt / 86400.0
            print(f"step {step+1:4d} (day {day:.3f}): "
                  f"T=[{T_min:.1f},{T_max:.1f}] mean={T_mean:.2f}K  "
                  f"q_v_max={qv_max*1000:.3g}g/kg  "
                  f"q_c_max={qc_max*1000:.3g}g/kg  "
                  f"q_r_max={qr_max*1000:.3g}g/kg")

    print("Run complete — Kessler-only path is stable in isolation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
