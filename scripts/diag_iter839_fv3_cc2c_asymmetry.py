"""Iter-839 reproducibility diagnostic: `fv3_cc2c` symmetric v_c
projection hypothesis test.

Codex iter-839 fidelity audit (agentId a5ae1319518698356) identified
`fv3_cc2c` in `src/legoesm/core/operators_cdgrid.py` as asymmetric:
u_c gets the face-normal projection `u_avg*sina_u − v_at_u*cosa_u`
but v_c gets only a plain 0.5-average.

iter-839 tested the symmetric-projection hypothesis by monkey-
patching `fv3_cc2c` to ALSO project v_c via
`v_c = v_avg*sina_v − u_at_v*cosa_v`, then running W2 C36 1d through
the standard model pipeline.  Measure L2 and v_ll_Linf vs baseline.

Scope: drop-in symmetric-projection test ONLY.  Does NOT test:
- whether a deeper refactor (consuming COVARIANT uc/vc +
  downstream contravariant conversion via `(uc − v*cosa_u)*rsin_u`)
  would work;
- whether the existing `compute_transport_quantities` +
  `transport_step` path on ut/vt could be wired in as a
  drop-in-faithful alternative.

Those are carried as iter-840+ candidates.
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

from legoesm.core import operators_cdgrid as ocd
from legoesm.core.operators_cdgrid import _broadcast_metric, _EPS
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


orig_fv3_cc2c = ocd.fv3_cc2c


def _fv3_cc2c_symmetric(u_cc, v_cc, cdgrid):
    """Patched fv3_cc2c with SYMMETRIC face-normal projection for
    BOTH u_c and v_c (iter-839 hypothesis)."""
    from legoesm.grids.halo import pad_halo_vector
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    u_avg = 0.5 * (u_pad[:, :-1, 1:-1] + u_pad[:, 1:, 1:-1])
    v_at_u = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])
    cosa_u = _broadcast_metric(cdgrid.cosa_u, u_avg)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
    u_c = u_avg * sina_u - v_at_u * cosa_u

    v_avg = 0.5 * (v_pad[:, 1:-1, :-1] + v_pad[:, 1:-1, 1:])
    u_at_v = 0.5 * (u_pad[:, 1:-1, :-1] + u_pad[:, 1:-1, 1:])
    cosa_v = _broadcast_metric(cdgrid.cosa_v, v_avg)
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v**2, _EPS))
    v_c = v_avg * sina_v - u_at_v * cosa_v
    return u_c, v_c


def _run_w2(n, variant, dt=300.0, days=1.0):
    """Run W2 at C{n} for `days` days.  variant ∈ {'baseline',
    'symmetric_vc'}."""
    if variant == 'baseline':
        ocd.fv3_cc2c = orig_fv3_cc2c
    elif variant == 'symmetric_vc':
        ocd.fv3_cc2c = _fv3_cc2c_symmetric
    else:
        raise ValueError(variant)

    n_steps = int(round(days * 86400 / dt))
    grid = create_cubed_sphere(n=n)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=1.0e15, div_damp=1.5e7 * (48 / n) ** 2,
        boundary_fix=True, fix_mass=True)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    crashed = None
    for step in range(n_steps):
        state = model.step(state, dt)
        if not np.all(np.isfinite(np.asarray(state.h))):
            crashed = step
            break

    if crashed is not None:
        return {'crashed_at_step': crashed}

    # Compute L2 error relative to analytical h
    h_np = np.asarray(state.h)
    h_exact = np.asarray(sw.h.data)
    area = np.asarray(grid.area)
    L2 = float(np.sqrt(
        np.sum(area * (h_np - h_exact) ** 2) / np.sum(area * h_exact ** 2)))
    return {'L2': L2, 'h_max': float(np.max(np.abs(h_np)))}


n = 36
days = 1.0

print(f"Iter-839 `fv3_cc2c` symmetric v_c projection hypothesis test")
print(f"W2 C{n} {days}d, iter-761 canonical config")
print()
print(f"{'variant':>15}  {'status':>18}  {'L2':>10}  {'h_max':>8}")
print("-" * 60)

try:
    for variant in ('baseline', 'symmetric_vc'):
        r = _run_w2(n, variant, days=days)
        if 'crashed_at_step' in r:
            print(f"{variant:>15}  CRASH step {r['crashed_at_step']:>6}  —  —")
        else:
            print(f"{variant:>15}  {'ok':>18}  {r['L2']:>10.3e}  "
                  f"{r['h_max']:>8.0f}")
finally:
    ocd.fv3_cc2c = orig_fv3_cc2c

print()
print("Interpretation (observational only):")
print("- If symmetric_vc L2 >> baseline L2 (by 100× or more):")
print("  the asymmetry is LOAD-BEARING under current fv3_cc2c +")
print("  cgrid_mass_flux_divergence semantics.  A drop-in symmetric")
print("  projection is NOT the Fortran-faithful fix.")
print("- Deeper-path fix candidates (iter-840+):")
print("  (a) use the existing compute_transport_quantities + transport_step")
print("      on covariant ut/vt (fv_tp_2d.py) instead of physical u_c/v_c.")
print("  (b) refactor cgrid_mass_flux_divergence to take covariant")
print("      uc/vc and derive contravariant ut/vt internally.")
