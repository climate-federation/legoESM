"""Iter-730 diagnostic: FB-chain W2 stability at C24 — does
iter-727's newly-wired d_sw1 mass-transport damping (`damp_c=damp_v`
through `transport_step` → `fv_tp_2d`) close the C24 blowup?

Task 3 of the iter-722 Path Forward: "Verify FB + duogrid stability
at C24 first, then C36."

Compares two FB runs:
  (A) damp_v=0.00 — baseline, matches pre-iter-727 behaviour.
  (B) damp_v=0.06 — iter-727 d_sw1 damp_c=damp_v path active +
      existing step-(9) del6_vt_flux both enabled.

For each run, records the step at which `h` goes non-finite (NaN/inf)
or reports STABLE if the run completes 1 day without blowup.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter730_fb_c24_stability.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig,
)

N = 24
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

# --- Williamson 2 IC (same as diag_williamson2_fb.py) ---
g = 9.80616
omega = 7.292e-5
u_0 = 38.61068276698372
h_0 = 29400.0 / g
R = cdgrid.radius

lat_c = cdgrid.base.lat
h_init = h_0 - (R * omega * u_0 + 0.5 * u_0 ** 2) * jnp.sin(lat_c) ** 2 / g
lat_ux = cdgrid.lat_edge_x
cos_ax = cdgrid.cos_angle_edge_x
sin_ax = cdgrid.sin_angle_edge_x
u_d = u_0 * jnp.cos(lat_ux) * cos_ax
lat_vy = cdgrid.lat_edge_y
sin_ay = cdgrid.sin_angle_edge_y
v_d = -u_0 * jnp.cos(lat_vy) * sin_ay
h_s = jnp.zeros_like(h_init)
state0 = FV3EdgeShallowWaterState(h=h_init, u_d=u_d, v_d=v_d, h_s=h_s)

dt = 300.0  # Conservative dt for C24.
n_steps = int(86400 / dt)

def run_case(damp_v, label):
    config = CDGridShallowWaterConfig(
        A_h=0.0, hyperdiff_coeff=0.0, div_damp=0.0,
        d4_bg=0.16, nord=1, dddmp=0.0,
        damp_v=damp_v, nord_v=-1,  # auto -> min(2,nord)=1
    )
    model = FV3FBShallowWaterModel(grid, config)
    model.set_initial_mass(state0)
    state = state0
    print(f"\n--- Case {label} (damp_v={damp_v}, nord_v=1) ---")
    last_printed = -1
    for i in range(n_steps):
        state = model.step(state, dt)
        if not bool(jnp.all(jnp.isfinite(state.h))):
            print(f"  BLOWUP at step {i + 1} / {n_steps} "
                  f"(t = {(i + 1) * dt:.0f} s)")
            return {"label": label, "damp_v": damp_v,
                    "blowup_step": i + 1, "total_steps": n_steps,
                    "stable": False}
        if (i + 1) % 48 == 0 and i != last_printed:
            dh = float(jnp.max(jnp.abs(state.h - state0.h)))
            maxu = float(jnp.max(jnp.abs(state.u_d)))
            maxv = float(jnp.max(jnp.abs(state.v_d)))
            print(f"  Step {i + 1}/{n_steps}: max|dh|={dh:.4f}, "
                  f"max|u_d|={maxu:.4f}, max|v_d|={maxv:.4f}")
            last_printed = i
    v_d_exact = -u_0 * jnp.cos(cdgrid.lat_edge_y) * cdgrid.sin_angle_edge_y
    v_err = float(jnp.max(jnp.abs(state.v_d - v_d_exact)))
    u_d_exact = u_0 * jnp.cos(cdgrid.lat_edge_x) * cdgrid.cos_angle_edge_x
    u_err = float(jnp.max(jnp.abs(state.u_d - u_d_exact)))
    print(f"  STABLE through 1 day: max|v_d_err|={v_err:.4f}, "
          f"max|u_d_err|={u_err:.4f}")
    return {"label": label, "damp_v": damp_v,
            "blowup_step": None, "total_steps": n_steps,
            "stable": True, "v_err_max": v_err, "u_err_max": u_err}

results = []
results.append(run_case(0.00, "A baseline (pre-iter-727 equiv)"))
results.append(run_case(0.06, "B iter-727 d_sw1 damp_v=0.06"))
results.append(run_case(0.12, "C high damp_v=0.12"))
results.append(run_case(0.30, "D very high damp_v=0.30"))

print("\n=== Iter-730 FB C24 W2 stability summary ===")
for r in results:
    if r["stable"]:
        print(f"  {r['label']}: STABLE through {r['total_steps']} steps "
              f"(max|v_err|={r['v_err_max']:.4f} m/s)")
    else:
        print(f"  {r['label']}: BLOWUP at step {r['blowup_step']} "
              f"/ {r['total_steps']}")
