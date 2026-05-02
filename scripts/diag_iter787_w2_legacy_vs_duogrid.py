"""Iter-787 diagnostic: W2 v_ll_Linf under LEGACY vs DUOGRID at C36.

Iter-786 observed that the DUOGRID path has 3-4× LARGER cube-
vertex error in `_d2a2c_vect` than the LEGACY path.  The
production W2 sentinels run the LEGACY path (use_duogrid=False)
and measure v_ll_Linf ≈ 0.159 m/s at cube vertices.

iter-787 runs W2 under both paths for 1 day at C36 and compares:
  - L2 error in h at t=1 day
  - v_ll_Linf (mode-A artifact at cube vertices)

If DUOGRID v_ll_Linf is MUCH LARGER (ratio > 2):
  switching W2 to duogrid would WORSEN the mode-A artifact.
  iter-786's `_d2a2c_vect` cube-vertex error is directly
  manifesting in W2.

If DUOGRID v_ll_Linf is SIMILAR:
  the W2 artifact does not track `_d2a2c_vect` cube-vertex
  error alone; other components of the shallow-water chain
  (possibly compensating) dominate.

If DUOGRID v_ll_Linf is SMALLER:
  duogrid's halo-exchange improvements outweigh its corner-fill
  degradation, making it a net win for W2 despite iter-786.

Scope: observational only.  No source-code change.  Both runs use
the EXACT iter-761 canonical config (boundary_fix=True, damp_v=0.06,
nord_v=2, div_damp = 8 * _div_damp_cube(n), hyperdiff=0).
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


def _run_w2(n, use_duogrid, days=1.0, dt=300.0):
    n_steps = int(round(days * 86400 / dt))
    div_damp = 8.0 * _div_damp_cube(n)  # iter-761 canonical config

    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
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

    # v_ll_Linf — iter-765-documented mode-A metric.
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
print(f"Iter-787 W2 alpha=0 C36 1-day, LEGACY vs DUOGRID")
print(f"dt=300s (iter-761 canonical config)")
print()
print(f"{'path':>20}  {'L2 (h drift)':>13}  {'v_ll_Linf':>12}  "
      f"{'v_cc_Linf':>12}")
print("-" * 65)

legacy = _run_w2(n, use_duogrid=False)
print(f"{'LEGACY':>20}  {legacy['L2']:>13.3e}  "
      f"{legacy['v_ll_linf']:>12.3e}  {legacy['v_cc_linf']:>12.3e}")

duogrid = _run_w2(n, use_duogrid=True)
l2_ratio = duogrid['L2'] / legacy['L2']
vll_ratio = duogrid['v_ll_linf'] / legacy['v_ll_linf']
vcc_ratio = duogrid['v_cc_linf'] / legacy['v_cc_linf']
print(f"{'DUOGRID':>20}  {duogrid['L2']:>13.3e}  "
      f"{duogrid['v_ll_linf']:>12.3e}  {duogrid['v_cc_linf']:>12.3e}")
print()
print(f"{'DUOGRID / LEGACY':>20}  {l2_ratio:>13.3f}  "
      f"{vll_ratio:>12.3f}  {vcc_ratio:>12.3f}")

print()
print("Interpretation cues (observational only):")
print("- L2 (h drift): both should be small (~2e-4 for LEGACY iter-761")
print("  baseline).  DUOGRID may be larger due to _d2a2c_vect error.")
print("- v_ll_Linf: LEGACY baseline ~0.159 m/s (iter-765 documented).")
print("  If DUOGRID v_ll_Linf is >> 0.159 m/s, the duogrid cube-vertex")
print("  error in `_d2a2c_vect` directly drives the W2 artifact.")
print("- v_cc_Linf: raw v_north at cell centres (no lat-lon regrid)")
print("  provides a cleaner comparison unaffected by regrid interpol.")

print()
print("What iter-787 DOES measure (observational only):")
print("- W2 alpha=0 1-day C36 L2 and v_ll_Linf under LEGACY")
print("  (use_duogrid=False) and DUOGRID (use_duogrid=True) paths.")
print("What it does NOT establish:")
print("- WHICH part of the duogrid chain (halo exchange, corner fill,")
print("  _d2a2c_vect, tendencies) drives the difference — that requires")
print("  finer decomposition than iter-787 provides.")
print("- Whether fixing iter-786's duogrid cube-vertex error would")
print("  make DUOGRID better than LEGACY.")
