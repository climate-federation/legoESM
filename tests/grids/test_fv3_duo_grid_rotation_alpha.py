"""``create_fv3_duo_grid(rotation_alpha=...)``: the rotated-planet Coriolis
reaches both duo contexts, and alpha = 0 is byte-identical to the default."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.core.williamson_sw_analytic import rotated_coriolis
from legoesm.grids.factory import create_fv3_duo_grid
from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA

N = 8


@pytest.fixture(scope="module")
def grids():
    return (create_fv3_duo_grid(N), create_fv3_duo_grid(N, rotation_alpha=0.0),
            create_fv3_duo_grid(N, rotation_alpha=np.pi / 4))


def test_alpha_zero_is_byte_identical(grids):
    g0, gz, _ = grids
    for t in range(6):
        for k in ("f0", "fC"):
            assert np.asarray(g0.ctx_np["gs6"][t][k]).tobytes() == \
                np.asarray(gz.ctx_np["gs6"][t][k]).tobytes()


def test_rotated_coriolis_reaches_both_contexts(grids):
    g0, _, gr = grids
    sl = slice(gr.ng, gr.ng + N)
    for t in range(6):
        gs = gr.ctx_np["gs6"][t]
        ref = rotated_coriolis(np.asarray(gs["agrid_lon"])[sl, sl],
                               np.asarray(gs["agrid_lat"])[sl, sl],
                               omega=FV3_OMEGA, alpha=np.pi / 4)
        f0 = np.asarray(gs["f0"])[sl, sl]
        np.testing.assert_allclose(f0, ref, rtol=0, atol=1e-18)
        # non-vacuous: the unrotated grid differs
        assert np.abs(np.asarray(g0.ctx_np["gs6"][t]["f0"])[sl, sl] - f0).max() > 1e-5
        # B-node Coriolis fC on the full node lattice
        fc_ref = rotated_coriolis(np.asarray(gs["grid_lon"]),
                                  np.asarray(gs["grid_lat"]),
                                  omega=FV3_OMEGA, alpha=np.pi / 4)
        np.testing.assert_allclose(np.asarray(gs["fC"]), fc_ref,
                                   rtol=0, atol=1e-18)
        # the jitted stepper context carries the same rotated metrics
        for k in ("f0", "fC"):
            np.testing.assert_array_equal(np.asarray(gr.ctx_jax.gs6[t][k]),
                                          np.asarray(gs[k]))


def test_omega_reaches_both_contexts_and_none_is_byte_identical(grids):
    """``create_fv3_duo_grid(omega=...)``: 0.0 is the non-rotating planet
    (DCMIP 2008 3-1) in both contexts; None is the FV3_OMEGA default."""
    g0 = grids[0]
    gn = create_fv3_duo_grid(N, omega=None)
    g_still = create_fv3_duo_grid(N, omega=0.0)
    for t in range(6):
        for k in ("f0", "fC"):
            assert np.asarray(g0.ctx_np["gs6"][t][k]).tobytes() == \
                np.asarray(gn.ctx_np["gs6"][t][k]).tobytes()
            assert np.all(np.asarray(g_still.ctx_np["gs6"][t][k]) == 0.0)
            assert np.all(np.asarray(g_still.ctx_jax.gs6[t][k]) == 0.0)
            assert np.abs(np.asarray(g0.ctx_np["gs6"][t][k])).max() > 1e-5
