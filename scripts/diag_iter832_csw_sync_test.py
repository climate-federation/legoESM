"""Iter-832 diagnostic: test removing flux sync from _c_sw.

Per iter-831's finding that `_c_sw` drives the FB-chain h-growth,
iter-832 tests the Fortran-fidelity hypothesis that our Python
`_c_sw` is OVER-SYNCING mass flux.

Fortran c_sw (sw_core.F90:189-235) does NOT apply flux sync after
the 1st-order upwind mass transport.  The flux sync in Fortran
happens at d_sw1 (dyn_core.F90:853-900) as part of the full-step
d_sw chain.

Our Python `_c_sw` at line 1309-1312 applies
`synchronize_cgrid_fluxes` gated on use_duogrid.  This is
redundant with the sync that `_d_sw_native` also applies — we're
double-syncing.

iter-832 monkey-patches `_c_sw` to SKIP the flux sync and re-runs
the FB chain C24 W2 12h test.  If h_max drops, the over-sync is
the h-growth driver.
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

from legoesm.grids import halo as halo_mod
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


orig_sync = halo_mod.synchronize_cgrid_fluxes


def _run_fb(n, csw_sync_on, hours=12.0, dt=300.0):
    """Run FB chain; if csw_sync_on=False, patch sync to no-op
    ONLY during _c_sw invocation (tracked via a counter).
    Downstream d_sw sync should remain active.

    Since it's impractical to patch sync ONLY within c_sw from
    outside, this test instead toggles the GLOBAL sync (which
    affects both c_sw and d_sw).  This gives a cruder test: does
    ANY sync cause h-growth?
    """
    if not csw_sync_on:
        halo_mod.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
        # Also patch the module-level reference in operators_cdgrid.
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
        # And fv3_sw_core and fv_tp_2d.
        from legoesm.core import fv3_sw_core as fsc
        if hasattr(fsc, 'synchronize_cgrid_fluxes'):
            fsc.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
        from legoesm.core import fv_tp_2d as tp
        if hasattr(tp, 'synchronize_cgrid_fluxes'):
            tp.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
    else:
        halo_mod.synchronize_cgrid_fluxes = orig_sync
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = orig_sync
        from legoesm.core import fv3_sw_core as fsc
        if hasattr(fsc, 'synchronize_cgrid_fluxes'):
            fsc.synchronize_cgrid_fluxes = orig_sync
        from legoesm.core import fv_tp_2d as tp
        if hasattr(tp, 'synchronize_cgrid_fluxes'):
            tp.synchronize_cgrid_fluxes = orig_sync

    n_steps = int(round(hours * 3600 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
        damp_v=0.0, nord_v=0, d4_bg=0.16, nord=1)
    model = FV3FBShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    crashed_at = None
    for step in range(n_steps):
        state = model.step(state, dt)
        if not np.all(np.isfinite(np.asarray(state.h))):
            crashed_at = step
            break

    if crashed_at is not None:
        return {'crashed_at_step': crashed_at}

    h_np = np.asarray(state.h)
    return {'h_max': float(np.max(np.abs(h_np)))}


n = 24
hours = 12.0

print(f"Iter-832 FB DUOGRID C{n} {hours}h: sync ON (iter-808) vs OFF")
print()
print(f"{'variant':>25}  {'status':>22}  {'h_max':>8}")
print("-" * 60)

try:
    r_on = _run_fb(n, csw_sync_on=True)
    if 'crashed_at_step' in r_on:
        print(f"{'sync ON (iter-808)':>25}  CRASH at step {r_on['crashed_at_step']:>6}  —")
    else:
        print(f"{'sync ON (iter-808)':>25}  {'ok (12h)':>22}  "
              f"{r_on['h_max']:>8.0f}")
    r_off = _run_fb(n, csw_sync_on=False)
    if 'crashed_at_step' in r_off:
        print(f"{'sync OFF':>25}  CRASH at step {r_off['crashed_at_step']:>6}  —")
    else:
        print(f"{'sync OFF':>25}  {'ok (12h)':>22}  "
              f"{r_off['h_max']:>8.0f}")
finally:
    # Restore originals.
    halo_mod.synchronize_cgrid_fluxes = orig_sync
    from legoesm.core import operators_cdgrid as ocd_mod
    if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
        ocd_mod.synchronize_cgrid_fluxes = orig_sync
    from legoesm.core import fv3_sw_core as fsc
    if hasattr(fsc, 'synchronize_cgrid_fluxes'):
        fsc.synchronize_cgrid_fluxes = orig_sync
    from legoesm.core import fv_tp_2d as tp
    if hasattr(tp, 'synchronize_cgrid_fluxes'):
        tp.synchronize_cgrid_fluxes = orig_sync

print()
print("Interpretation cues (observational only):")
print("- If sync OFF gives h_max near physical (~2980):")
print("  the sync itself (or its placement in c_sw as well as d_sw)")
print("  is driving h-growth.  Next iter: sync-only-in-d_sw.")
print("- If sync OFF crashes or worsens: the sync isn't the culprit.")
