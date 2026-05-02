"""Iter-906b diagnostic: D-grid `dv_d_dt` decomposition matching the
iter-904 hot-spot stagger (Codex iter-906 stop-time fix).

Codex iter-906 stop-time review correctly flagged: "The iter-906
diagnostic does not measure the D-grid hot spots it claims to
explain".  iter-906 measured cell-centre `dv_cc` cancellation —
but iter-904's hot spots were on the D-grid v-edge stagger
(`dv_d_dt`, shape `(6, n+1, n)`), AFTER the cell-centre→D-grid
projection at `operators_cdgrid.py:2193-2214`:

  du_cc_pad, dv_cc_pad = pad_halo_vector(du_cc, dv_cc, ...)
  du_d_dt = 0.5 * (du_cc_pad[:, 1:-1, :-1] + du_cc_pad[:, 1:-1, 1:])
  dv_d_dt = 0.5 * (dv_cc_pad[:, :-1, 1:-1] + dv_cc_pad[:, 1:, 1:-1])

iter-906b extends iter-906's analysis to:
  (1) Reproduce the dv_cc cancellation at cell centres (iter-906's
      original finding) for verification.
  (2) Project dv_cc -> dv_d_dt via the same pad_halo_vector chain.
  (3) Identify the top-10 hot spots on the D-grid v-edge stagger
      (matching iter-904's identification at lat ±33.9° on faces 0/2).
  (4) For each D-grid hot spot, report the two cell-centre dv_cc
      contributors (the two cells averaged at that v-edge) and
      their cancellation quality.

This rigorously connects iter-906's cube-vertex finding to the
iter-904 D-grid hot spots through the actual production projection.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterState)
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, _arakawa_lamb_gradient,
    _interp_corner_to_center, dgrid_vorticity, fv3_sw_tendencies)
from legoesm.grids.halo import pad_halo_vector
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def main():
    n = 36
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    g = 9.80616
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    h = sw.h.data
    h_s = sw.h_s.data
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    # === Reproduce fv3_sw_tendencies up to the cell-centre dv_cc ===
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    base = cdgrid.base
    dg = base.duogrid
    offsets = None if dg is not None else base.halo_interp_offsets
    u_cc_pad, v_cc_pad = pad_halo_vector(
        u_cc, v_cc,
        base.cos_angle, base.sin_angle,
        base.cos_angle_padded, base.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                        + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
    v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                        + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + base.f
    dB_dy_cc = _interp_corner_to_center(dB_dy_perp)

    coriolis_term = -np.asarray(zeta_abs * u_cc)        # cell centre
    pressure_term = -np.asarray(dB_dy_cc)               # cell centre
    dv_cc_array = coriolis_term + pressure_term

    # === Project dv_cc to D-grid v-edges via pad_halo_vector ===
    # Use the SAME projection as fv3_sw_tendencies lines 2193-2214.
    # Need the corresponding du_cc to project as a vector.
    dB_dx_cc = _interp_corner_to_center(dB_dx)
    du_cc_array = np.asarray(zeta_abs * v_cc) - np.asarray(dB_dx_cc)

    # NOTE: production fv3_sw_tendencies adds div_damp + boundary_fix
    # contributions BEFORE projection.  For a clean iter-906b decomposition
    # we project the bare Coriolis-vorticity + pressure-gradient sum,
    # which suffices to test whether the cube-vertex cell-centre
    # cancellation breakdown PROPAGATES to the D-grid stagger.
    du_cc_jax = jnp.asarray(du_cc_array)
    dv_cc_jax = jnp.asarray(dv_cc_array)
    du_cc_pad, dv_cc_pad = pad_halo_vector(
        du_cc_jax, dv_cc_jax,
        base.cos_angle, base.sin_angle,
        base.cos_angle_padded, base.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    dv_d_dt = 0.5 * (dv_cc_pad[:, :-1, 1:-1] + dv_cc_pad[:, 1:, 1:-1])
    dv_d_dt = np.asarray(dv_d_dt)   # shape (6, n+1, n)

    # === Independently run production fv3_sw_tendencies to compare ===
    # (with the same iter-893 production matrix config knobs).
    div_damp = 8.0 * 1.5e7 * (48.0 / n) ** 2
    _, _, dv_d_dt_prod = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        g=g, div_damp=div_damp, boundary_fix=True, dddmp=0.2,
        apply_fortran_xppm_boundary=True)
    dv_d_dt_prod = np.asarray(dv_d_dt_prod)

    abs_dv_d = np.abs(dv_d_dt)
    abs_dv_d_prod = np.abs(dv_d_dt_prod)

    print("=" * 72)
    print("iter-906b: D-grid `dv_d_dt` decomposition (Codex iter-906 fix)")
    print("=" * 72)
    print(f"Cell-centre `dv_cc` stats (iter-906 reproduced):")
    print(f"  max |dv_cc|         = {np.max(np.abs(dv_cc_array)):.3e}")
    print(f"  max |coriolis term|  = {np.max(np.abs(coriolis_term)):.3e}")
    print(f"  max |pressure term|  = {np.max(np.abs(pressure_term)):.3e}")
    print()
    print(f"D-grid `dv_d_dt` stats (= projected bare Coriolis+pressure):")
    print(f"  max |dv_d_dt|         = {abs_dv_d.max():.3e}")
    print(f"  mean |dv_d_dt|        = {abs_dv_d.mean():.3e}")
    print()
    print(f"D-grid `dv_d_dt` from PRODUCTION fv3_sw_tendencies (incl. div_damp + boundary_fix):")
    print(f"  max |dv_d_dt_prod|    = {abs_dv_d_prod.max():.3e}")
    print(f"  mean |dv_d_dt_prod|   = {abs_dv_d_prod.mean():.3e}")
    print()

    # Top-10 D-grid v-edge hot spots from PRODUCTION tendency.
    flat_prod = abs_dv_d_prod.ravel()
    n_per_face_v = (n + 1) * n
    top10 = np.argsort(flat_prod)[-10:][::-1]
    print(f"Top 10 D-grid v-edge hot spots (production dv_d_dt):")
    print(f"{'face':>4}  {'i':>3}  {'j':>3}  {'lat':>8}  {'lon':>8}  "
          f"{'|dv_d_dt|':>12}")
    print("-" * 70)
    for idx in top10:
        face = idx // n_per_face_v
        rem = idx % n_per_face_v
        i = rem // n
        j = rem % n
        lat_deg = float(np.degrees(cdgrid.lat_edge_y[face, i, j]))
        lon_deg = float(np.degrees(cdgrid.lon_edge_y[face, i, j]))
        val = float(abs_dv_d_prod[face, i, j])
        print(f"{face:>4}  {i:>3}  {j:>3}  {lat_deg:>+8.2f}  "
              f"{lon_deg:>+8.2f}  {val:>+12.4e}")
    print()

    # For each D-grid v-edge hot spot, identify the TWO cell-centre cells
    # contributing via the projection 0.5 * (dv_cc_pad[:-1] + dv_cc_pad[1:])
    # in the i-axis.  Report cancellation quality at both contributing
    # cell-centre cells.
    print(f"Cell-centre dv_cc contributors at top-5 D-grid hot spots:")
    print(f"{'face':>4}  {'i_d':>3}  {'j_d':>3}  "
          f"{'cell_low(i,j)':>14}  {'dv_cc_low':>12}  "
          f"{'cell_high(i,j)':>14}  {'dv_cc_high':>12}  {'projected':>12}")
    print("-" * 110)
    for idx in top10[:5]:
        face = idx // n_per_face_v
        rem = idx % n_per_face_v
        i = rem // n
        j = rem % n
        # D-grid v-edge at (face, i, j) in interior i-axis [0, n-1] of dv_d_dt
        # is at i_d in [0, n], from `dv_d_dt[..., 1:-1]` slicing of cell-centre
        # halo-padded dv_cc_pad: cell-centre indices (i-1, j) and (i, j) in pad
        # -> after halo-pad it's pad index (i, j+1) and (i+1, j+1) in dv_cc_pad
        # which has shape (6, n+2, n+2). Interior cell-centre (i_cc, j_cc) is
        # pad index (i_cc+1, j_cc+1).  So the two contributors at v-edge
        # (i_d, j_d) interior are interior cells (i_d-1, j_d) and (i_d, j_d)
        # — when those are valid interior, otherwise halo cells.
        cells = []
        for cell_i in (i - 1, i):
            if 0 <= cell_i < n:
                cells.append((cell_i, j, dv_cc_array[face, cell_i, j]))
            else:
                cells.append((cell_i, j, np.nan))
        proj = 0.5 * (cells[0][2] + cells[1][2]) if not np.isnan(cells[0][2]) and not np.isnan(cells[1][2]) else float('nan')
        print(f"{face:>4}  {i:>3}  {j:>3}  "
              f"({cells[0][0]:>5}, {cells[0][1]:>5})  {cells[0][2]:>+12.4e}  "
              f"({cells[1][0]:>5}, {cells[1][1]:>5})  {cells[1][2]:>+12.4e}  "
              f"{proj:>+12.4e}")
    print()

    # Cancellation quality on the D-grid stagger.
    coriolis_cc_jax = jnp.asarray(coriolis_term)
    pressure_cc_jax = jnp.asarray(pressure_term)
    # Project each term via the same pad_halo (zero-vector for the unused
    # component since both terms are scalars per-cell to be averaged in i).
    cor_pad_zeros = jnp.zeros_like(coriolis_cc_jax)
    pre_pad_zeros = jnp.zeros_like(pressure_cc_jax)
    cor_pad_x, cor_pad_y = pad_halo_vector(
        cor_pad_zeros, coriolis_cc_jax,
        base.cos_angle, base.sin_angle,
        base.cos_angle_padded, base.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg)
    pre_pad_x, pre_pad_y = pad_halo_vector(
        pre_pad_zeros, pressure_cc_jax,
        base.cos_angle, base.sin_angle,
        base.cos_angle_padded, base.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg)
    coriolis_d_dt = 0.5 * (cor_pad_y[:, :-1, 1:-1] + cor_pad_y[:, 1:, 1:-1])
    pressure_d_dt = 0.5 * (pre_pad_y[:, :-1, 1:-1] + pre_pad_y[:, 1:, 1:-1])
    coriolis_d_dt = np.asarray(coriolis_d_dt)
    pressure_d_dt = np.asarray(pressure_d_dt)

    max_term_d = np.maximum(np.abs(coriolis_d_dt), np.abs(pressure_d_dt))
    cancel_d = np.where(max_term_d > 0,
                          abs_dv_d / np.maximum(max_term_d, 1e-30), 0.0)
    print(f"Cancellation quality on D-grid v-edge:")
    print(f"  hot-spot top-10 mean: {cancel_d.ravel()[top10].mean():.3e}")
    print(f"  global mean:           {cancel_d.mean():.3e}")
    print(f"  global max:            {cancel_d.max():.3e}")
    print()

    hot_d_cancel = cancel_d.ravel()[top10].mean()
    global_d_cancel = cancel_d.mean()
    if hot_d_cancel > 5.0 * global_d_cancel:
        print(f"VERDICT: D-grid hot spots have CANCELLATION "
              f"{hot_d_cancel/global_d_cancel:.1f}x WORSE than global avg.")
        print(f"         Confirms iter-906's cube-vertex cancellation finding")
        print(f"         propagates through the cell-centre -> D-grid projection")
        print(f"         to the iter-904 D-grid hot spots.  Targeting the cell-")
        print(f"         centre dv_cc cancellation at cube vertices is the")
        print(f"         correct path to reduce production W2 v_ll_Linf.")
    else:
        print(f"VERDICT: D-grid hot-spot cancellation ratio "
              f"({hot_d_cancel/global_d_cancel:.1f}x) is similar to global —")
        print(f"         cell-centre cube-vertex cancellation does NOT directly")
        print(f"         translate to D-grid hot spots.  iter-906's finding")
        print(f"         applies at cell centres but the D-grid hot spots are")
        print(f"         driven by a different mechanism (likely div_damp or")
        print(f"         boundary_fix in the production code path).")


if __name__ == "__main__":
    main()
