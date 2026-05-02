"""Iter-794 diagnostic: W2 LEGACY div_damp sensitivity.

Per iter-793's finding that `dv div-damp` contributes ~37% of
the cube-vertex residual at t=0 (1.83e-5 out of 3.97e-5), iter-
794 sweeps `div_damp` on the LEGACY path and measures v_ll_Linf.

If v_ll_Linf is proportional to div_damp (as iter-793's t=0
analysis predicts), the divergence damping is directly amplifying
the cube-vertex mode-A.  A div_damp reduction would then ease the
artifact.  But div_damp is scientifically-motivated (it damps
grid-scale noise) and iter-761's 8× scaling was tuned for W2 L2
optimality — so reducing div_damp might trade smaller v_ll_Linf
for larger L2.  iter-794 measures both.

Scope: observational only.  div_damp is a config knob, no source-
code change.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_w2(n, div_damp_val, hours=24.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp_val,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0 = state.h
    for _ in range(n_steps):
        state = model.step(state, dt)

    h_mean = float(jnp.mean(jnp.abs(h0)))
    err = state.h - h0
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)) / h_mean)

    from legoesm.grids.cubed_sphere_cdgrid import (
        cell_centre_angles_from_4edge)
    from legoesm.grids.regridding import (
        get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))
    v_cc_linf = float(np.max(np.abs(v_north)))
    return {
        'L2': L2,
        'v_ll_linf': v_ll_linf,
        'v_cc_linf': v_cc_linf,
    }


n = 36
base = _div_damp_cube(n)
# iter-761 canonical: 8x base.  Sweep 0, 1x, 4x, 8x, 16x.

cases = [
    ('0 (none)', 0.0),
    ('1x base', 1.0 * base),
    ('4x base', 4.0 * base),
    ('8x (iter-761)', 8.0 * base),
    ('16x base', 16.0 * base),
]

print(f"Iter-794 W2 LEGACY div_damp sensitivity, C36, 24h, dt=300s")
print(f"Base _div_damp_cube(n) = {base:.3e}")
print()
print(f"{'div_damp':>18}  {'L2':>11}  {'v_ll_Linf':>11}  {'v_cc_Linf':>11}")
print("-" * 55)

results = {}
for label, dd in cases:
    r = _run_w2(n, dd)
    results[label] = r
    print(f"{label:>18}  {r['L2']:>11.3e}  {r['v_ll_linf']:>11.3e}  "
          f"{r['v_cc_linf']:>11.3e}")

print()
iter761 = results['8x (iter-761)']
for label, r in results.items():
    if label == '8x (iter-761)':
        continue
    dL2 = 100.0 * (r['L2'] - iter761['L2']) / iter761['L2']
    dvll = 100.0 * (r['v_ll_linf'] - iter761['v_ll_linf']) / iter761['v_ll_linf']
    print(f"  {label:>15} vs 8x: L2 {dL2:+.1f}%, v_ll_Linf {dvll:+.1f}%")

print()
print("Interpretation cues (observational only):")
print("- If v_ll_Linf drops substantially at div_damp=0:")
print("  div damp is the dominant W2 mode-A amplifier.  A smaller")
print("  div_damp coefficient could improve mode-A at the cost of")
print("  possibly larger grid-scale noise elsewhere.")
print("- If L2 grows substantially at div_damp=0:")
print("  div damp is needed for mass accuracy.  The trade-off must")
print("  be managed (iter-761 picked 8x for L2-optimality).")
print("- If a smaller div_damp (1x, 4x) gives smaller v_ll_Linf AND")
print("  comparable L2: a looser div_damp coefficient might be a")
print("  mild improvement over iter-761's 8x.")

print()
print("What iter-794 DOES measure (observational only):")
print("- W2 LEGACY 1-day L2, v_ll_Linf, v_cc_Linf across 5 div_damp")
print("  values (0, 1x, 4x, 8x_iter761, 16x).")
print("What it does NOT establish:")
print("- Whether a smaller div_damp breaks W5 or other harness cases.")
print("  iter-761's 8x was chosen for W2+W5 compatibility.")
print("- Whether the trade-off reported here at C36 holds at finer")
print("  grids (C48, C72).")
