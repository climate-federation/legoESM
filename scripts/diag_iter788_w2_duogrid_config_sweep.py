"""Iter-788 diagnostic: W2 DUOGRID config sweep to identify what
breaks in the full SW pipeline at C36.

Iter-787 found DUOGRID is 800× worse than LEGACY on W2 L2 and
630× worse on v_ll_Linf.  This iter systematically toggles config
knobs to narrow down which knob (or combination) drives the
DUOGRID blowup.

Config knobs tested:
  - boundary_fix (True / False)
  - damp_v (0.06 / 0.0)
  - nord_v (2 / 0)
  - div_damp (8x / 1x / 0)
  - hyperdiff_coeff (0 / small positive)

Strategy: run DUOGRID with each ALTERED config and compare to
LEGACY baseline (0.159 m/s) and DUOGRID default (99.96 m/s).
The point where v_ll_Linf drops near LEGACY level identifies the
offending knob.

Scope: observational only.  No source-code change.  Short runs
(6 hours) to limit cost; any config that DOESN'T blow up in 6
hours can be extended to 1 day for definitive comparison in a
subsequent iter.
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


def _run_w2(n, use_duogrid, cfg, hours=6.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
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
    try:
        for _ in range(n_steps):
            state = model.step(state, dt)
    except Exception as e:
        return {'error': str(e)[:100], 'L2': np.nan,
                'v_ll_linf': np.nan, 'v_cc_linf': np.nan}

    h_mean = float(jnp.mean(jnp.abs(h0)))
    err = state.h - h0
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)) / h_mean)

    from legoesm.grids.cubed_sphere_cdgrid import (
        cell_centre_angles_from_4edge)
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    v_cc_linf = float(np.max(np.abs(v_north)))

    return {
        'L2': L2,
        'v_cc_linf': v_cc_linf,
    }


n = 36
div_damp_base = _div_damp_cube(n)

# Configurations to sweep (DUOGRID path, 6-hour runs).
configs = [
    ('iter-761 canonical',
     dict(hyperdiff_coeff=0.0, div_damp=8.0*div_damp_base,
          boundary_fix=True, damp_v=0.06, nord_v=2)),
    ('boundary_fix=False',
     dict(hyperdiff_coeff=0.0, div_damp=8.0*div_damp_base,
          boundary_fix=False, damp_v=0.06, nord_v=2)),
    ('damp_v=0 nord_v=0',
     dict(hyperdiff_coeff=0.0, div_damp=8.0*div_damp_base,
          boundary_fix=True, damp_v=0.0, nord_v=0)),
    ('div_damp=0',
     dict(hyperdiff_coeff=0.0, div_damp=0.0,
          boundary_fix=True, damp_v=0.06, nord_v=2)),
    ('div_damp=base (1x)',
     dict(hyperdiff_coeff=0.0, div_damp=div_damp_base,
          boundary_fix=True, damp_v=0.06, nord_v=2)),
    ('all damping off',
     dict(hyperdiff_coeff=0.0, div_damp=0.0,
          boundary_fix=False, damp_v=0.0, nord_v=0)),
]

print(f"Iter-788 W2 DUOGRID config sweep at C36, 6-hour runs, dt=300s")
print()
print(f"LEGACY iter-761 canonical (for reference, 6h):")
legacy_cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, div_damp=8.0*div_damp_base,
    boundary_fix=True, damp_v=0.06, nord_v=2)
r_legacy = _run_w2(n, use_duogrid=False, cfg=legacy_cfg)
print(f"  L2={r_legacy['L2']:.3e}, v_cc_Linf={r_legacy['v_cc_linf']:.3e}")
print()
print(f"{'DUOGRID config':>25}  {'L2':>11}  {'v_cc_Linf':>11}  {'status':>10}")
print("-" * 70)
for label, kwargs in configs:
    cfg = CDGridShallowWaterConfig(**kwargs)
    r = _run_w2(n, use_duogrid=True, cfg=cfg)
    if 'error' in r:
        print(f"{label:>25}  {'NaN':>11}  {'NaN':>11}  ERROR: {r['error'][:40]}")
    else:
        status = 'OK' if r['v_cc_linf'] < 5.0 else (
            'drifting' if r['v_cc_linf'] < 30.0 else 'broken')
        print(f"{label:>25}  {r['L2']:>11.3e}  {r['v_cc_linf']:>11.3e}  "
              f"{status:>10}")

print()
print("Interpretation cues (observational only):")
print("- 'OK' = v_cc_Linf < 5 m/s (approaching LEGACY baseline).")
print("- 'drifting' = v_cc_Linf in [5, 30] m/s (visible artifact).")
print("- 'broken' = v_cc_Linf > 30 m/s (gross failure like iter-787).")
print("- If any DUOGRID config reaches 'OK', that knob toggle is a")
print("  candidate for the broken duogrid wiring.")

print()
print("What iter-788 DOES measure (observational only):")
print("- DUOGRID W2 6-hour L2 and v_cc_Linf under 6 config variants.")
print("What it does NOT establish:")
print("- Whether the identified failing config knob is FIXABLE (vs")
print("  reflecting a deeper wiring issue).")
print("- Whether 6-hour behaviour extrapolates to 1-day or longer.")
