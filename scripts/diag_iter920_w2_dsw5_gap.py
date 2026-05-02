#!/usr/bin/env python
"""Iter-926 (file: diag_iter920_w2_dsw5_gap) — t=0 W2 dv_d_dt
decomposition + Fortran d_sw5 corner-damping pattern comparison.

Per user direction (iter-926 brief):

1. Build W2 C36 IC.
2. Compute one t=0 production `fv3_sw_tendencies` decomposition.
3. Print top 20 `dv_d_dt` hot spots with face/i/j, lat/lon, total
   dv, bare Coriolis+pressure contribution, div_damp contribution,
   boundary_fix contribution, and final D-grid projection
   contribution.
4. Compute the Fortran-style d_sw5 corner-damping increment using
   `_d_sw5_corner_divergence` with d2_bg/dddmp/d4_bg/nord matching
   `CDGridShallowWaterConfig`.
5. Compare its spatial pattern against production div_damp.  Report
   correlation and max-location mismatch.

The script is read-only — no production code change.  Output is a
text report; no PNGs.

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter920_w2_dsw5_gap.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.core.fv3_sw_core import _d2a2c_vect, _d_sw5_corner_divergence
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


N = 36
DT = 300.0


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _make_state(grid, cdgrid):
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    return state, sw


def _eval_tendencies(state, sw, cdgrid, *, div_damp, boundary_fix,
                     apply_fortran_xppm_boundary):
    # NB: `fv3_sw_tendencies` has signature (h, u_d, v_d, h_s, cdgrid,
    # g=..., div_damp=..., ...).  No `dt` positional; no `damp_v`/
    # `nord_v` (those are forwarded by `FV3EdgeShallowWaterModel.step`
    # to the post-step damp_v hook, NOT to the inner tendencies fn).
    dh, du, dv = fv3_sw_tendencies(
        state.h, state.u_d, state.v_d, state.h_s, cdgrid,
        g=9.80616,
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        dddmp=0.2,
        boundary_fix=boundary_fix,
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
    )
    return np.asarray(dh), np.asarray(du), np.asarray(dv)


def main() -> None:
    print(f"=== iter-926 W2 d_sw5 gap diagnostic at C{N}, t=0 ===")
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    state, sw = _make_state(grid, cdgrid)

    div_damp_prod = 8.0 * _div_damp_cube(N)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Production: full iter-893 matrix at t=0.
        dh_full, du_full, dv_full = _eval_tendencies(
            state, sw, cdgrid,
            div_damp=div_damp_prod, boundary_fix=True,
            apply_fortran_xppm_boundary=True,
        )

        # Bare A-L (no div_damp).
        dh_bare, du_bare, dv_bare = _eval_tendencies(
            state, sw, cdgrid,
            div_damp=0.0, boundary_fix=True,
            apply_fortran_xppm_boundary=True,
        )

        # No boundary_fix (production - boundary_fix).
        dh_nobf, du_nobf, dv_nobf = _eval_tendencies(
            state, sw, cdgrid,
            div_damp=div_damp_prod, boundary_fix=False,
            apply_fortran_xppm_boundary=True,
        )

    # Decompositions:
    dv_div_damp = dv_full - dv_bare        # div_damp's contribution
    dv_boundary_fix = dv_full - dv_nobf    # boundary_fix's contribution

    print()
    print(f"Coefficients (production iter-893 matrix):")
    da_min_c_val = float(jnp.min(cdgrid.area_corner))
    print(f"  div_damp                  = {div_damp_prod:.3e}")
    print(f"  da_min_c (min area_corner)= {da_min_c_val:.3e}")
    print(f"  d2_bg_eff = div_damp/da_min_c = {div_damp_prod/da_min_c_val:.3e}")
    print(f"  dddmp_prod                = 0.2")

    # --- Fortran d_sw5 corner damping computation ---
    # Compute ua, va, ut, vt for d_sw5 input.
    ua, va, uc, vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)

    # Match production's coefficient regime: d2_bg_eff = div_damp/da_min_c,
    # dddmp = 0.2, nord = 0 (since production uses del-2 not del-4).
    # Fortran would use these in `_d_sw5_corner_divergence(... d2_bg=..,
    # dddmp=.., nord=0)`.  Note: production divides div_damp by da_min_c
    # before using it; the helper expects the d2_bg as a unitless factor.
    d2_bg_eff = div_damp_prod / da_min_c_val
    ke_damping_fortran = _d_sw5_corner_divergence(
        state.u_d, state.v_d, ua, va, cdgrid, DT,
        d2_bg=d2_bg_eff, dddmp=0.2, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=False,
    )
    ke_damping_fortran = np.asarray(ke_damping_fortran)
    print(f"  ke_damping (Fortran d_sw5) shape = {ke_damping_fortran.shape}")
    print(f"  |ke_damping| max  = {np.abs(ke_damping_fortran).max():.3e}")
    print(f"  |ke_damping| mean = {np.abs(ke_damping_fortran).mean():.3e}")

    # Project ke_damping → equivalent dv_d_dt at D-grid v positions
    # (sw_core.F90:1942):
    #   v_d(i,j) = vt(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)
    # The d_sw5 contribution to dv per step is (ke_damp(i,j)-ke_damp(i,j+1))
    # / dy, divided by dt to get a tendency-equivalent.
    dy_v = np.asarray(cdgrid.dy_edge_x)  # shape (6, n+1, n)  [v-edges]
    # ke_damping is at corners (6, n+1, n+1).  v lives at (6, n+1, n)
    # with ∂y across j: ke_damping[:, :, j] - ke_damping[:, :, j+1].
    dv_dsw5_per_step = (
        ke_damping_fortran[:, :, :-1] - ke_damping_fortran[:, :, 1:]
    ) / np.maximum(dy_v, 1e-30)
    dv_dsw5_tendency = dv_dsw5_per_step / DT  # tendency-equivalent
    print(f"  |dv_dsw5| max (tendency)  = {np.abs(dv_dsw5_tendency).max():.3e}")

    # --- Top 20 production dv hot spots ---
    print()
    print("--- Top 20 production dv_d_dt hot spots (D-grid stagger) ---")
    # dv lives at v-edges (6, n+1, n).
    abs_dv = np.abs(dv_full)
    flat_idx = np.argsort(abs_dv.ravel())[::-1][:20]

    # Extract grid lat/lon at v-edge positions:
    # cdgrid has lat_edge_y, lon_edge_y at (6, n+1, n).
    lat_y = np.asarray(cdgrid.lat_edge_y) * 180.0 / np.pi
    lon_y = np.asarray(cdgrid.lon_edge_y) * 180.0 / np.pi

    print(
        f"{'idx':>3} {'face':>4} {'i':>3} {'j':>3} "
        f"{'lat°':>7} {'lon°':>7} "
        f"{'dv_full':>10} {'dv_bare':>10} "
        f"{'div_damp':>10} {'bdy_fix':>10} {'dsw5':>10}"
    )
    for k, idx in enumerate(flat_idx):
        f, i, j = np.unravel_index(idx, abs_dv.shape)
        print(
            f"{k:>3d} {f:>4d} {i:>3d} {j:>3d} "
            f"{lat_y[f, i, j]:>+7.2f} {lon_y[f, i, j]:>+7.2f} "
            f"{dv_full[f, i, j]:>+10.3e} {dv_bare[f, i, j]:>+10.3e} "
            f"{dv_div_damp[f, i, j]:>+10.3e} {dv_boundary_fix[f, i, j]:>+10.3e} "
            f"{dv_dsw5_tendency[f, i, j]:>+10.3e}"
        )

    # --- Spatial correlation: production div_damp vs Fortran d_sw5 ---
    print()
    print("--- Spatial pattern correlation (D-grid v stagger) ---")
    a = dv_div_damp.ravel()
    b = dv_dsw5_tendency.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
    print(f"  corr(production div_damp, Fortran d_sw5) = {corr:+.4f}")

    print(f"  |production div_damp| max = {np.abs(dv_div_damp).max():.3e}")
    print(f"  |Fortran d_sw5|     max = {np.abs(dv_dsw5_tendency).max():.3e}")
    print(f"  ratio max(d_sw5) / max(div_damp) = "
          f"{np.abs(dv_dsw5_tendency).max() / max(np.abs(dv_div_damp).max(), 1e-30):.4f}")

    # Hot-spot location overlap:
    f_div, i_div, j_div = np.unravel_index(
        np.abs(dv_div_damp).argmax(), dv_div_damp.shape)
    f_dsw, i_dsw, j_dsw = np.unravel_index(
        np.abs(dv_dsw5_tendency).argmax(), dv_dsw5_tendency.shape)
    print(f"  div_damp max @ face={f_div} (i,j)=({i_div},{j_div}) "
          f"lat,lon=({lat_y[f_div,i_div,j_div]:+.2f}, "
          f"{lon_y[f_div,i_div,j_div]:+.2f})°")
    print(f"  Fortran d_sw5 max @ face={f_dsw} (i,j)=({i_dsw},{j_dsw}) "
          f"lat,lon=({lat_y[f_dsw,i_dsw,j_dsw]:+.2f}, "
          f"{lon_y[f_dsw,i_dsw,j_dsw]:+.2f})°")

    # --- Decomposition summary at top hot spots ---
    print()
    print("--- Top hot spot decomposition (sum over top-20 |dv|) ---")
    top_dv_full = sum(abs(dv_full[
        np.unravel_index(idx, abs_dv.shape)]) for idx in flat_idx)
    top_dv_div = sum(abs(dv_div_damp[
        np.unravel_index(idx, abs_dv.shape)]) for idx in flat_idx)
    top_dv_bdy = sum(abs(dv_boundary_fix[
        np.unravel_index(idx, abs_dv.shape)]) for idx in flat_idx)
    top_dv_dsw5 = sum(abs(dv_dsw5_tendency[
        np.unravel_index(idx, abs_dv.shape)]) for idx in flat_idx)
    print(f"  Σ |dv_full|       = {top_dv_full:.3e}")
    print(f"  Σ |div_damp|      = {top_dv_div:.3e} "
          f"({100*top_dv_div/top_dv_full:.1f}% of full)")
    print(f"  Σ |boundary_fix|  = {top_dv_bdy:.3e} "
          f"({100*top_dv_bdy/top_dv_full:.1f}% of full)")
    print(f"  Σ |Fortran d_sw5| = {top_dv_dsw5:.3e} "
          f"({100*top_dv_dsw5/top_dv_full:.1f}% of full)")


if __name__ == "__main__":
    main()
