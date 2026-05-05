"""Iter-805 diagnostic: test whether DUOGRID h halo missing
interp_offsets is the dh/dt blowup cause.

Iter-804 ruled out both `cube_rmp_vectorized` and
`fill_corner_region` as individual causes of DUOGRID dh/dt
blowup.  Inspection shows that `_pad_halo_auto_h2` suppresses
interp_offsets when duogrid is active (operators_cdgrid.py:68-
69):
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets_h2

So the DUOGRID scalar halo for h is NEAREST-NEIGHBOR copy
(O(Δα) positional error at cube corners), while LEGACY uses
interpolated offsets.  When PPM then reconstructs face values,
this positional bias propagates into large face-value errors.

iter-805 tests:
  - DUOGRID but FORCE interp_offsets on the h halo (mimic LEGACY).

If dh/dt drops to LEGACY level: the nearest-neighbor h halo is
the culprit; fix is to restore interp_offsets in duogrid mode
for the scalar h halo (iter-631 previously added a guard that
pad_halo_mpi doesn't support offsets; the LOCAL path does).
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


orig_pad_halo = halo_mod.pad_halo


def _measure(label, use_duogrid, force_offsets, n=36):
    """Run t=0 W2 with optional force-on of interp_offsets on h halo."""
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)

    if force_offsets and grid.duogrid is not None:
        # Monkey-patch pad_halo to inject interp_offsets when
        # called with (halo=2, duogrid=dg) but offsets=None.
        offs_h2 = grid.halo_interp_offsets_h2
        offs_h1 = grid.halo_interp_offsets

        def _patched_pad_halo(data, halo=1, interp_offsets=None, duogrid=None):
            if duogrid is not None and interp_offsets is None:
                if halo == 1:
                    interp_offsets = offs_h1
                elif halo == 2:
                    interp_offsets = offs_h2
                # Note: pad_halo will error if duogrid AND offsets both set;
                # need to disable duogrid to use offsets.  Or, since duogrid
                # corner fill uses Lagrange on interior cells, we can still
                # use duogrid=None and just the offsets — matching LEGACY.
                return orig_pad_halo(data, halo=halo,
                                      interp_offsets=interp_offsets,
                                      duogrid=None)
            return orig_pad_halo(data, halo=halo,
                                  interp_offsets=interp_offsets,
                                  duogrid=duogrid)
        halo_mod.pad_halo = _patched_pad_halo
        # Also patch operators_cdgrid imports if needed
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'pad_halo'):
            ocd_mod.pad_halo = _patched_pad_halo
    else:
        halo_mod.pad_halo = orig_pad_halo
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'pad_halo'):
            ocd_mod.pad_halo = orig_pad_halo

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    dh_dt, du_dt, dv_dt = fv3_sw_tendencies(
        sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
        g=cfg.g, div_damp=cfg.div_damp,
        hyperdiff_coeff=cfg.hyperdiff_coeff,
        boundary_fix=cfg.boundary_fix,
    )

    return {
        'dh_dt_peak': float(np.max(np.abs(np.asarray(dh_dt)))),
        'du_dt_peak': float(np.max(np.abs(np.asarray(du_dt)))),
        'dv_dt_peak': float(np.max(np.abs(np.asarray(dv_dt)))),
    }


print(f"Iter-805 DUOGRID h halo interp_offsets test")
print(f"C36 W2 IC, iter-761 config")
print()
print(f"{'variant':>45}  {'dh/dt peak':>11}  {'du/dt peak':>11}  {'dv/dt peak':>11}")
print("-" * 85)

cases = [
    ('LEGACY (use_duogrid=False)', False, False),
    ('DUOGRID (baseline, no offsets)', True, False),
    ('DUOGRID (force offsets, LEGACY-like halo)', True, True),
]

results = {}
for label, use_dg, force_off in cases:
    r = _measure(label, use_dg, force_off)
    results[label] = r
    print(f"{label:>45}  {r['dh_dt_peak']:>11.3e}  "
          f"{r['du_dt_peak']:>11.3e}  {r['dv_dt_peak']:>11.3e}")

# Restore.
halo_mod.pad_halo = orig_pad_halo
from legoesm.core import operators_cdgrid as ocd_mod
if hasattr(ocd_mod, 'pad_halo'):
    ocd_mod.pad_halo = orig_pad_halo

print()
r_L = results['LEGACY (use_duogrid=False)']
for label, r in results.items():
    if 'LEGACY' in label:
        continue
    dh = r['dh_dt_peak'] / r_L['dh_dt_peak']
    du = r['du_dt_peak'] / r_L['du_dt_peak']
    dv = r['dv_dt_peak'] / r_L['dv_dt_peak']
    print(f"  {label}: dh {dh:.2f}x, du {du:.2f}x, dv {dv:.2f}x")
