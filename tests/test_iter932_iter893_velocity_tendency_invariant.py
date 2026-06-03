"""Iter-932 sentinel: pin the SCOPE of iter-893's PPM boundary fix.

iter-893 (`apply_fortran_xppm_boundary=True`) changes
`_ppm_reconstruct_1d` boundary cell face values for mass transport.
PPM is consumed by `cgrid_mass_flux_divergence` for the height
tendency (`dh_dt`); it is NOT involved in the velocity tendency
(`du_dt`, `dv_dt`) computation, which goes through the bare A-L
operator family (Coriolis + Bernoulli-grad).

Therefore at t=0, with `div_damp=0` and `boundary_fix=False` (so
no other production stabilizers fire), iter-893 should:
- CHANGE `dh_dt` (PPM mass transport is on the iter-893 path).
- Leave `du_dt` and `dv_dt` BIT-EXACT identical (no path).

iter-932 measured at C36 W2 t=0:
- max|du_off - du_on| = 0.0  (bit-exact)
- max|dv_off - dv_on| = 0.0  (bit-exact)
- max|dh_off - dh_on| = 8.57e-05  (iter-893 active on PPM)

This sentinel locks that SCOPE invariant.  Any future change that
makes iter-893 affect velocity tendencies (e.g., wiring PPM into
the velocity path) would fire this gate, requiring an explicit
accept/reject decision.

Complements iter-925 (rest-state bit-exact equality) which checks
the same property in the OPPOSITE direction (constants → no PPM
effect).  iter-932 verifies PPM-scoped change is FULLY scoped to
mass transport on a non-trivial state too.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


N = 36


def _w2_t0_state():
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    return cdgrid, sw, u_d, v_d


def _bare_al_tendencies(apply_xppm: bool):
    cdgrid, sw, u_d, v_d = _w2_t0_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dh, du, dv = fv3_sw_tendencies(
            sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
            g=constants.g,
            div_damp=0.0,
            boundary_fix=False,
            apply_fortran_xppm_boundary=apply_xppm,
        )
    return np.asarray(dh), np.asarray(du), np.asarray(dv)


@pytest.fixture(scope="module")
def _bare_off():
    return _bare_al_tendencies(apply_xppm=False)


@pytest.fixture(scope="module")
def _bare_on():
    return _bare_al_tendencies(apply_xppm=True)


def test_iter932_velocity_tendencies_bit_exact_under_iter893_toggle(
    _bare_off, _bare_on,
):
    """`du_dt` and `dv_dt` must be BIT-EXACT identical regardless of
    `apply_fortran_xppm_boundary`.  PPM is mass transport only; any
    drift here means iter-893 leaked into the velocity path.
    """
    _, du_off, dv_off = _bare_off
    _, du_on, dv_on = _bare_on
    np.testing.assert_array_equal(
        du_off, du_on,
        err_msg=(
            "iter-893 PPM toggle changed du_dt — PPM has leaked into "
            "the velocity tendency path."
        ),
    )
    np.testing.assert_array_equal(
        dv_off, dv_on,
        err_msg=(
            "iter-893 PPM toggle changed dv_dt — PPM has leaked into "
            "the velocity tendency path."
        ),
    )


def test_iter932_height_tendency_does_change_under_iter893_toggle(
    _bare_off, _bare_on,
):
    """`dh_dt` MUST differ between iter-893 off vs on at t=0 with a
    non-uniform W2 height field.  This is the positive sentinel —
    if iter-893 is silently no-op on `dh_dt`, the flag is dead.

    iter-932 baseline: max|dh_off - dh_on| ≈ 8.57e-05.
    """
    dh_off, _, _ = _bare_off
    dh_on, _, _ = _bare_on
    diff = float(np.abs(dh_off - dh_on).max())
    assert diff > 1e-6, (
        f"max|dh_off - dh_on| = {diff:.3e} suggests iter-893 has "
        f"become silently no-op on the height tendency.  Flag is "
        f"either dead or the W2 IC is no longer hitting boundary "
        f"cells."
    )


def test_iter932_height_tendency_drift_within_iter932_band(
    _bare_off, _bare_on,
):
    """Pin the magnitude of iter-893's effect on `dh_dt` at t=0.

    iter-932 measured 8.57e-05.  Pin within ±20 % to allow some
    iter-to-iter variability under unrelated changes (PPM internals,
    halo path, etc.) but catch order-of-magnitude drift.
    """
    dh_off, _, _ = _bare_off
    dh_on, _, _ = _bare_on
    diff = float(np.abs(dh_off - dh_on).max())
    expected = 8.57e-05
    rel = abs(diff - expected) / expected
    assert rel < 0.20, (
        f"max|dh_off - dh_on| = {diff:.3e} drifted {rel*100:.1f} % "
        f"from iter-932's {expected:.3e}.  iter-893's effect on "
        f"mass transport has changed order of magnitude — "
        f"investigate which PPM path edit caused the drift."
    )
