"""Iter-931 sentinel: pin the bare A-L cube-vertex imperfect-
cancellation residual at C16/C24/C36/C48 to confirm the gap is
RESOLUTION-REFINABLE (not a structural floor).

iter-929 found bare A-L has ~1 % imperfect cancellation at the 8
cube vertices.  iter-931 measures the scaling across resolutions:

| N   | |coriolis_dv| max   | |bare residual| max  | imperfect % |
|-----|---------------------|----------------------|-------------|
| 16  | 3.029e-03           | 3.382e-05            | 1.117       |
| 24  | 3.040e-03           | 2.967e-05            | 0.976       |
| 36  | 3.045e-03           | 2.516e-05            | 0.826       |
| 48  | 3.047e-03           | 2.238e-05            | 0.735       |

Convergence: residual at C16 / C48 = 1.51× over 3× resolution
→ effective convergence rate p ≈ log(1.51)/log(3.0) ≈ 0.38.
This is slow but POSITIVE — refinement reduces the cube-vertex bias.

The integrated W2 v_ll_Linf scaling (iter-910) shows analogous
1st-order refinement (C16=0.354 → C36=0.132 → C48=0.125 m/s).
Both metrics confirm the cube-vertex bias is resolution-refinable,
NOT a structural floor.  Production is at C36 because of cost
constraints, not because higher resolution doesn't help.

This sentinel pins the four data points within ±5 % so future iters
that change the bare A-L cancellation structure (zeta, B-grad,
halo treatment) are caught — any rewrite of `fv3_d2cc`/
`_arakawa_lamb_gradient`/`pad_halo_vector` should preserve or
improve this scaling.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.operators_cdgrid import (
    _arakawa_lamb_gradient,
    _interp_corner_to_center,
    dgrid_vorticity,
    fv3_d2cc,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo_vector
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def _measure_bare_residual(N: int) -> dict:
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data
    h_s = sw.h_s.data

    # Bare A-L (no div_damp, no boundary_fix)
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    KE = 0.5 * (u_cc**2 + v_cc**2)
    B = KE + constants.g * (h + h_s)
    _, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    grid_obj = cdgrid.base
    dg = grid_obj.duogrid
    offsets = None if dg is not None else grid_obj.halo_interp_offsets
    u_cc_pad, v_cc_pad = pad_halo_vector(
        u_cc, v_cc,
        grid_obj.cos_angle, grid_obj.sin_angle,
        grid_obj.cos_angle_padded, grid_obj.sin_angle_padded,
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
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f
    dB_dy_cc = _interp_corner_to_center(dB_dy_perp)

    coriolis_dv = -np.asarray(zeta_abs * u_cc)
    bernoulli_dv = -np.asarray(dB_dy_cc)
    bare_dv = coriolis_dv + bernoulli_dv
    bg = max(np.abs(coriolis_dv).max(), np.abs(bernoulli_dv).max())
    res = np.abs(bare_dv).max()
    return {
        "background_max": bg,
        "residual_max": res,
        "imperfect_pct": 100.0 * res / bg,
    }


@pytest.mark.parametrize(
    "N, expected",
    [
        (16, {"residual_max": 3.382e-05, "imperfect_pct": 1.117}),
        (24, {"residual_max": 2.967e-05, "imperfect_pct": 0.976}),
        (36, {"residual_max": 2.516e-05, "imperfect_pct": 0.826}),
        (48, {"residual_max": 2.238e-05, "imperfect_pct": 0.735}),
    ],
    ids=["C16", "C24", "C36", "C48"],
)
def test_iter931_bare_al_residual_scaling_pinned(N, expected):
    """Pin the bare A-L imperfect-cancellation residual at each
    resolution within ±5 %.  Catches any rewrite of `fv3_d2cc`/
    `_arakawa_lamb_gradient`/`pad_halo_vector` that perturbs the
    cube-vertex behaviour.
    """
    measured = _measure_bare_residual(N)
    for metric, target in expected.items():
        assert np.isfinite(measured[metric])
        rel = abs(measured[metric] - target) / abs(target)
        assert rel < 0.05, (
            f"C{N} {metric} = {measured[metric]:.4e} drifted "
            f"{rel*100:.2f}% from iter-931's {target:.4e}."
        )


def test_iter931_residual_decreases_monotonically_with_resolution():
    """The bare cube-vertex residual must decrease monotonically
    from C16 → C48.  Confirms the gap is RESOLUTION-REFINABLE
    (not a structural floor).  Any future change that introduces
    a non-refinable cube-vertex artifact would fire this gate.
    """
    sequence = [_measure_bare_residual(N) for N in (16, 24, 36, 48)]
    residuals = [s["residual_max"] for s in sequence]
    pcts = [s["imperfect_pct"] for s in sequence]
    for i in range(len(residuals) - 1):
        assert residuals[i] > residuals[i + 1], (
            f"Bare residual not monotonic: C{16+i*8} = {residuals[i]:.3e}, "
            f"C{16+(i+1)*8} = {residuals[i+1]:.3e}.  "
            f"A non-refinable cube-vertex artifact has been introduced."
        )
        assert pcts[i] > pcts[i + 1], (
            f"Imperfect % not monotonic: C{16+i*8} = {pcts[i]:.3f}, "
            f"C{16+(i+1)*8} = {pcts[i+1]:.3f}."
        )
