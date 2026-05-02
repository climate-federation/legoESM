"""Iter-789 diagnostic: W2 with `_d2a2c_vect` REPLACED by analytical
ut/vt at every step.

Iter-787 observed W2 LEGACY gives v_ll_Linf = 0.159 m/s (mode-A
artifact at 8 cube vertices).  Iter-782/786 showed LEGACY
`_d2a2c_vect` has 11% relative error at cube vertices on the
solid-body rotation IC.

Iter-789 tests the direct causal hypothesis: if `_d2a2c_vect`
output were EXACT (analytical ut/vt everywhere), would W2 mode-A
drop near zero?

Method: monkey-patch `_d2a2c_vect` at the module level BEFORE the
model is created so that the FIRST JIT trace captures the patched
version.  The patched function computes ANALYTICAL ut/vt from the
solid-body rotation IC using grid angles + covariant→contravariant
formula, and returns those as (ut, vt) while still computing (ua,
va, uc, vc) from the original function (so other parts of the
pipeline are unchanged).

If analytical-ut-vt W2 has v_ll_Linf ≪ 0.159 m/s, `_d2a2c_vect` is
the root cause and the fix is to reduce its cube-vertex error.
If analytical-ut-vt W2 has v_ll_Linf ≈ 0.159 m/s, the root cause
is downstream of `_d2a2c_vect`.

Scope: observational only.  The monkey-patch is diagnostic-only.
Note: for W2 solid-body rotation, (u_d, v_d) stay constant (the
analytical solution), so the analytical ut/vt is ALSO constant in
time — making the override well-defined.  For general flows the
analytical would have to be time-varying.
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
from legoesm.core import fv3_sw_core as fv3_sw_core_mod


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _make_analytical_ut_vt(n, cdgrid, u0):
    """Compute analytical ut, vt at C-grid stagger for W2 solid-body
    rotation about the polar axis (alpha=0).  u_east = u0 * cos(lat),
    v_north = 0.

    Returns ut (shape matches cdgrid C-grid y-edge) and vt (x-edge).
    """
    # At y-edge (ut position):
    u_e_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_n_y = jnp.zeros_like(u_e_y)
    u_cov_y = (cdgrid.cos_angle_edge_y * u_e_y
               + cdgrid.sin_angle_edge_y * v_n_y)
    v_cov_y = (-cdgrid.sin_angle_edge_y * u_e_y
               + cdgrid.cos_angle_edge_y * v_n_y)
    ut_exact = (u_cov_y - v_cov_y * cdgrid.cosa_u) * cdgrid.rsin_u

    # At x-edge (vt position):
    u_e_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    v_n_x = jnp.zeros_like(u_e_x)
    u_cov_x = (cdgrid.cos_angle_edge_x * u_e_x
               + cdgrid.sin_angle_edge_x * v_n_x)
    v_cov_x = (-cdgrid.sin_angle_edge_x * u_e_x
               + cdgrid.cos_angle_edge_x * v_n_x)
    vt_exact = (v_cov_x - u_cov_x * cdgrid.cosa_v) * cdgrid.rsin_v

    return ut_exact, vt_exact


def _run_w2(n, hours, dt, override_ut_vt=False):
    n_steps = int(round(hours * 3600 / dt))
    div_damp = 8.0 * _div_damp_cube(n)

    grid = create_cubed_sphere(n=n, use_duogrid=False)  # LEGACY
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=div_damp,
        boundary_fix=True, damp_v=0.06, nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    # Monkey-patch _d2a2c_vect if requested.  Must patch BEFORE
    # first model.step() to affect the JIT trace.
    orig_d2a2c = fv3_sw_core_mod._d2a2c_vect
    if override_ut_vt:
        ut_exact, vt_exact = _make_analytical_ut_vt(n, cdgrid, u0)
        _patched_trace_hits = [0]
        def _patched(u_d, v_d, cdgrid_arg):
            _patched_trace_hits[0] += 1
            ua, va, uc, vc, _, _ = orig_d2a2c(u_d, v_d, cdgrid_arg)
            return ua, va, uc, vc, ut_exact, vt_exact
        fv3_sw_core_mod._d2a2c_vect = _patched
        # Also patch submodules that may import `_d2a2c_vect` by name.
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, "_d2a2c_vect"):
            ocd_mod._d2a2c_vect = _patched

    try:
        h0 = state.h
        for _ in range(n_steps):
            state = model.step(state, dt)
        if override_ut_vt:
            print(f"  (debug) _patched trace hits: {_patched_trace_hits[0]}")
    finally:
        fv3_sw_core_mod._d2a2c_vect = orig_d2a2c
        try:
            from legoesm.core import operators_cdgrid as ocd_mod
            if hasattr(ocd_mod, "_d2a2c_vect"):
                ocd_mod._d2a2c_vect = orig_d2a2c
        except Exception:
            pass

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

    return {
        'L2': L2,
        'v_ll_linf': float(np.max(np.abs(v_ll))),
        'v_cc_linf': float(np.max(np.abs(v_north))),
    }


n = 36
hours = 24.0
dt = 300.0

print(f"Iter-789 W2 alpha=0 C36 {hours}h (LEGACY path)")
print(f"Monkey-patched `_d2a2c_vect` to inject ANALYTICAL ut, vt")
print()
print(f"{'case':>22}  {'L2':>11}  {'v_ll_Linf':>11}  {'v_cc_Linf':>11}")
print("-" * 60)

baseline = _run_w2(n, hours, dt, override_ut_vt=False)
print(f"{'BASELINE':>22}  {baseline['L2']:>11.3e}  "
      f"{baseline['v_ll_linf']:>11.3e}  {baseline['v_cc_linf']:>11.3e}")

override = _run_w2(n, hours, dt, override_ut_vt=True)
dl2 = 100.0 * (override['L2'] - baseline['L2']) / baseline['L2']
dvll = 100.0 * (override['v_ll_linf'] - baseline['v_ll_linf']) / baseline['v_ll_linf']
dvcc = 100.0 * (override['v_cc_linf'] - baseline['v_cc_linf']) / baseline['v_cc_linf']
print(f"{'ANALYTICAL ut/vt':>22}  {override['L2']:>11.3e}  "
      f"{override['v_ll_linf']:>11.3e}  {override['v_cc_linf']:>11.3e}")
print()
print(f"{'Δ vs baseline':>22}  "
      f"{'L2 '+f'{dl2:+.1f}%':>11}  "
      f"{'vll '+f'{dvll:+.1f}%':>11}  "
      f"{'vcc '+f'{dvcc:+.1f}%':>11}")

print()
print("Interpretation cues (observational only):")
print("- If ANALYTICAL ut/vt drops v_ll_Linf to ≪ 0.159 m/s:")
print("  `_d2a2c_vect` IS the direct cause of the W2 mode-A artifact.")
print("  A Fortran-faithful fix to `_d2a2c_vect` corner handling would")
print("  eliminate the artifact.")
print("- If ANALYTICAL ut/vt leaves v_ll_Linf unchanged:")
print("  `_d2a2c_vect` is NOT the direct cause; the mode-A artifact")
print("  originates in another operator downstream.  Candidates:")
print("  momentum tendencies, vorticity flux, pressure gradient.")
print("- If ANALYTICAL ut/vt is larger than BASELINE:")
print("  the LEGACY `_d2a2c_vect` error is actually compensating for")
print("  other biases — hard to interpret but possible.")

print()
print("What iter-789 DOES measure (observational only):")
print("- W2 L2 and v_ll_Linf under LEGACY with and without an")
print("  analytical-ut/vt monkey-patch replacing `_d2a2c_vect` output.")
print("What it does NOT establish:")
print("- Whether a Fortran-faithful `_d2a2c_vect` fix that")
print("  INCREMENTALLY reduces (not eliminates) cube-vertex error")
print("  would produce proportional v_ll_Linf improvement.")
print("- Whether the override is physically consistent with other")
print("  operators (the analytical ut/vt may not be self-consistent")
print("  with the updated u_d, v_d fields that drift over time).")
