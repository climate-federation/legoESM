"""Iter-809 diagnostic: W5 (mountain) LEGACY vs DUOGRID cross-
validation of the iter-808 sign-flip flux sync fix.

Per iter-808's iter-809+ candidate "Run W5 under the fixed
DUOGRID to measure impact", iter-809 runs Williamson case 5 at
C36 for 15 days under both LEGACY and the now-fixed DUOGRID
and compares the standard error metrics.

Expected: DUOGRID should be within ~10× LEGACY (as iter-808
showed for W2), not catastrophically broken.
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
    williamson_test5)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_w5(n, use_duogrid, days=3.0, dt=300.0):
    """Run W5 for `days` days.  Default 3 days (enough to see
    mountain-driven wave evolution without the 12-hour timeout)."""
    n_steps = int(round(days * 86400 / dt))
    div_damp = 8.0 * _div_damp_cube(n)

    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test5(grid)
    # W5 uses solid-body rotation with u0=20 m/s and mountain.
    u0 = 20.0
    # Project analytical winds to edge-midpoint D-grid.
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0 = state.h
    mass_init = float(jnp.sum(state.h * grid.area))
    try:
        for step in range(n_steps):
            state = model.step(state, dt)
    except Exception as e:
        return {'error': str(e)[:100]}

    h_new = np.asarray(state.h)
    u_new = np.asarray(state.u_d)
    v_new = np.asarray(state.v_d)

    mass_final = float(np.sum(h_new * np.asarray(grid.area)))
    mass_drift = (mass_final - mass_init) / mass_init
    h_min = float(np.min(h_new))
    h_max = float(np.max(h_new))
    # Since there's no analytical solution for W5, report field-
    # level stats.  h changes from the initial state indicate
    # mountain-driven wave propagation.
    h_change_rms = float(np.sqrt(np.mean((h_new - np.asarray(h0)) ** 2)))

    # Also compute v_north from the final state.
    from legoesm.grids.cubed_sphere_cdgrid import (
        cell_centre_angles_from_4edge)
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (u_new[:, :, :-1] + u_new[:, :, 1:])
    v_cc = 0.5 * (v_new[:, :-1, :] + v_new[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    v_cc_linf = float(np.max(np.abs(v_north)))

    return {
        'mass_drift': mass_drift,
        'h_min': h_min,
        'h_max': h_max,
        'h_change_rms': h_change_rms,
        'v_cc_linf': v_cc_linf,
    }


n = 36
days = 3.0

print(f"Iter-809 W5 mountain LEGACY vs DUOGRID at C{n}, {days} days")
print(f"(post-iter-808 sign-flip sync fix)")
print()

r_L = _run_w5(n, use_duogrid=False, days=days)
r_D = _run_w5(n, use_duogrid=True, days=days)

print(f"{'metric':>20}  {'LEGACY':>12}  {'DUOGRID':>12}  {'ratio':>10}")
print("-" * 60)
for k in ('mass_drift', 'h_min', 'h_max', 'h_change_rms', 'v_cc_linf'):
    if 'error' in r_L or 'error' in r_D:
        print(f"  error")
        break
    L = r_L[k]; D = r_D[k]
    if L == 0:
        ratio = 'N/A'
    elif L != 0 and abs(L) > 1e-30:
        ratio = f"{D / L:.3f}"
    else:
        ratio = 'N/A'
    print(f"  {k:>18}  {L:>12.3e}  {D:>12.3e}  {ratio:>10}")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID mass drift and h_change_rms are within ~10× of LEGACY:")
print("  the iter-808 sign-flip fix ALSO helps W5; DUOGRID path is")
print("  production-usable across multiple test cases.")
print("- If DUOGRID is still catastrophic on W5: the sign-flip fix")
print("  is specific to W2 symmetry; additional work needed for W5.")
