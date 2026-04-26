#!/usr/bin/env python
"""Iter-929 bare A-L operator decomposition at W2 hot spots.

iter-921 found that production v_ll_Linf hot spots at C36 1-day are
at face 1/3 i=0 corners (lat ±33-39° on the polar/equatorial cube
edge).  iter-907/908b/909 confirmed div_damp dominates the dv
magnitude at hot spots (~84%) but did not decompose what produces
the BARE A-L residual that div_damp is responding to.

This script computes the bare A-L tendency at t=0 W2 (no div_damp,
no boundary_fix) and decomposes `dv_cc` at the top hot spots into:

- Coriolis: `-zeta_abs * u_cc`
- Bernoulli-grad: `-dB_dy_cc`

The exact W2 alpha=0 geostrophic balance gives `dv = 0` everywhere
in geographic frame.  In grid-local frame the imperfect cancellation
between these two terms drives the bare A-L residual; the question
is which sub-operator carries the cube-vertex error.

Output: text report; no PNG.

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter929_w2_bare_al_decomposition.py
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
from legoesm.core.operators_cdgrid import (
    _arakawa_lamb_gradient,
    _interp_corner_to_center,
    dgrid_vorticity,
    fv3_d2cc,
    fv3_sw_tendencies,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo_vector
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


N = 36
DT = 300.0


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _bare_al_decomposition(state, cdgrid, g=9.80616):
    """Reproduce fv3_sw_tendencies bare A-L (no div_damp, no
    boundary_fix) AND return intermediates for decomposition.
    """
    h = state.h
    h_s = state.h_s
    u_d = state.u_d
    v_d = state.v_d

    # (a) Cell-centre velocities
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)

    # (c) Bernoulli function
    KE = 0.5 * (u_cc**2 + v_cc**2)
    B = KE + g * (h + h_s)

    # (d) Arakawa-Lamb gradient
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    # (e) Corner winds via halo + 4-point average
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_cc_pad, v_cc_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    u_corner = 0.25 * (
        u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
        + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:]
    )
    v_corner = 0.25 * (
        v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
        + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:]
    )

    # (f) Absolute vorticity at cell centres
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f

    # (g) Bare A-L tendencies
    dB_dx_cc = _interp_corner_to_center(dB_dx)
    dB_dy_cc = _interp_corner_to_center(dB_dy_perp)

    # Sub-operator contributions to dv_cc
    coriolis_dv = -zeta_abs * u_cc       # Coriolis term in dv
    bernoulli_dv = -dB_dy_cc             # B-gradient term in dv
    coriolis_du = zeta_abs * v_cc        # Coriolis term in du
    bernoulli_du = -dB_dx_cc             # B-gradient term in du

    return {
        "u_cc": np.asarray(u_cc),
        "v_cc": np.asarray(v_cc),
        "zeta": np.asarray(zeta),
        "zeta_abs": np.asarray(zeta_abs),
        "dB_dy_cc": np.asarray(dB_dy_cc),
        "dB_dx_cc": np.asarray(dB_dx_cc),
        "coriolis_dv": np.asarray(coriolis_dv),
        "bernoulli_dv": np.asarray(bernoulli_dv),
        "coriolis_du": np.asarray(coriolis_du),
        "bernoulli_du": np.asarray(bernoulli_du),
        "dv_cc_bare": np.asarray(coriolis_dv + bernoulli_dv),
        "du_cc_bare": np.asarray(coriolis_du + bernoulli_du),
    }


def main() -> None:
    print(f"=== iter-929 W2 bare A-L decomposition at C{N} t=0 ===")
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decomp = _bare_al_decomposition(state, cdgrid)
        # Also get the production dv_d_dt to find hot spot locations
        dh, du_d, dv_d = fv3_sw_tendencies(
            state.h, state.u_d, state.v_d, state.h_s, cdgrid,
            g=9.80616,
            div_damp=8.0 * _div_damp_cube(N),
            dddmp=0.2,
            boundary_fix=True,
            apply_fortran_xppm_boundary=True,
        )
        dv_d = np.asarray(dv_d)

    print()
    print(f"{'idx':>3} {'face':>4} {'i_cc':>4} {'j_cc':>4} "
          f"{'lat°':>7} {'lon°':>7} "
          f"{'dv_cc_bare':>12} {'coriolis_dv':>12} {'bernoulli_dv':>12} "
          f"{'|cor|/|bern|':>12}")

    # Find top 5 hot spots in dv_cc_bare (cell-centre) directly:
    abs_bare = np.abs(decomp["dv_cc_bare"])
    flat_idx = np.argsort(abs_bare.ravel())[::-1][:10]
    lat_cc = np.asarray(cdgrid.base.lat) * 180.0 / np.pi
    lon_cc = np.asarray(cdgrid.base.lon) * 180.0 / np.pi

    for k, idx in enumerate(flat_idx):
        f, i, j = np.unravel_index(idx, abs_bare.shape)
        c = decomp["coriolis_dv"][f, i, j]
        b = decomp["bernoulli_dv"][f, i, j]
        total = decomp["dv_cc_bare"][f, i, j]
        ratio = abs(c) / max(abs(b), 1e-30)
        print(
            f"{k:>3d} {f:>4d} {i:>4d} {j:>4d} "
            f"{lat_cc[f, i, j]:>+7.2f} {lon_cc[f, i, j]:>+7.2f} "
            f"{total:>+12.3e} {c:>+12.3e} {b:>+12.3e} {ratio:>12.4f}"
        )

    print()
    print("--- Magnitude summary (|·| over cell-centre interior) ---")
    print(f"  |dv_cc_bare| max  = {abs_bare.max():.3e}")
    print(f"  |coriolis_dv| max = {np.abs(decomp['coriolis_dv']).max():.3e}")
    print(f"  |bernoulli_dv| max= {np.abs(decomp['bernoulli_dv']).max():.3e}")
    print(f"  |coriolis_dv| mean= {np.abs(decomp['coriolis_dv']).mean():.3e}")
    print(f"  |bernoulli_dv| mean = {np.abs(decomp['bernoulli_dv']).mean():.3e}")

    # Pearson correlation between coriolis and -bernoulli (perfect cancel = +1):
    corr = np.corrcoef(decomp["coriolis_dv"].ravel(),
                        -decomp["bernoulli_dv"].ravel())[0, 1]
    print(f"  Pearson(coriolis_dv, -bernoulli_dv) = {corr:+.6f}")
    print("    (1.0 = perfect cancellation; 0 = no relation)")

    # |residual| / |coriolis|: relative imperfect-cancellation magnitude
    rel = abs_bare.ravel() / np.maximum(
        np.abs(decomp["coriolis_dv"]).ravel(), 1e-30)
    print(f"  median (|dv_bare|/|coriolis_dv|) = {np.median(rel):.3e}")
    print(f"  max    (|dv_bare|/|coriolis_dv|) = {rel.max():.3e}")
    f_max, i_max, j_max = np.unravel_index(rel.argmax(), abs_bare.shape)
    print(f"    max-ratio @ face={f_max} (i,j)=({i_max},{j_max}) "
          f"lat,lon=({lat_cc[f_max, i_max, j_max]:+.2f}, "
          f"{lon_cc[f_max, i_max, j_max]:+.2f})°")

    # --- Compare bare to production hot spots ---
    print()
    print("--- Production dv_d_dt hot spots (top 5) and bare-AL nearby ---")
    abs_dv_d = np.abs(dv_d)
    flat_d = np.argsort(abs_dv_d.ravel())[::-1][:5]
    lat_y = np.asarray(cdgrid.lat_edge_y) * 180.0 / np.pi
    lon_y = np.asarray(cdgrid.lon_edge_y) * 180.0 / np.pi
    for k, idx in enumerate(flat_d):
        f, i, j = np.unravel_index(idx, abs_dv_d.shape)
        # Production v_d edge at (i,j) corresponds to cell-centres
        # (i-1, j) and (i, j) — but indexing differs since u_cc/v_cc
        # are at (n,n).  v_d is (n+1, n).  v_d edge at (i, j) is
        # between cells (i-1, j) and (i, j) along the i-direction.
        i_cc_a = max(0, i - 1)
        i_cc_b = min(N - 1, i)
        c_a = decomp["coriolis_dv"][f, i_cc_a, j]
        b_a = decomp["bernoulli_dv"][f, i_cc_a, j]
        bare_a = decomp["dv_cc_bare"][f, i_cc_a, j]
        c_b = decomp["coriolis_dv"][f, i_cc_b, j]
        b_b = decomp["bernoulli_dv"][f, i_cc_b, j]
        bare_b = decomp["dv_cc_bare"][f, i_cc_b, j]
        print(
            f"[{k}] dv_d at f={f} (i,j)=({i},{j}) "
            f"lat,lon=({lat_y[f, i, j]:+.2f}, {lon_y[f, i, j]:+.2f})° "
            f"value={dv_d[f, i, j]:+.3e}"
        )
        print(
            f"     bare_AL @ cc[{i_cc_a},{j}] = {bare_a:+.3e} "
            f"(cor={c_a:+.3e}, bern={b_a:+.3e})"
        )
        print(
            f"     bare_AL @ cc[{i_cc_b},{j}] = {bare_b:+.3e} "
            f"(cor={c_b:+.3e}, bern={b_b:+.3e})"
        )


if __name__ == "__main__":
    main()
