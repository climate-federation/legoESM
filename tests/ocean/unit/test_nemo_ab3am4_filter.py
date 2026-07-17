"""Direct unit gates for the NEMO nn_bt_flt=3 barotropic scheme
(barotropic_time_filter="nemo_ab3am4"): the AB3/AM4 coefficient arrays
(ts_bck_interp transcription), the ll_init ramp, edge cases, and the
validator guards."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _NEMO_AB3_ZA,
    _NEMO_BT_ALPHA,
    nemo_ab3am4_coeff_arrays,
)


def test_coefficient_rows_sum_to_one():
    """Consistency: extrapolation/interpolation exact on a constant field —
    every row of za and zb sums to 1 (NEMO's coefficients do)."""
    za, zb = nemo_ab3am4_coeff_arrays(50)
    np.testing.assert_allclose(np.asarray(za).sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(zb).sum(axis=1), 1.0, atol=1e-12)


def test_interior_coefficients_match_ts_bck_interp_alpha007():
    """The alpha=0.07 AM4 branch (dynspg_ts.F90 ts_bck_interp), NOT the
    alpha==0 published table (0.614/0.285/0.088/0.013)."""
    za, zb = nemo_ab3am4_coeff_arrays(10)
    a = _NEMO_BT_ALPHA
    eps = 0.00976186 - 0.13451357 * a
    gam = 0.08344500 - 0.51358400 * a
    zb0 = 0.5 + gam + 2.0 * a + 2.0 * eps
    np.testing.assert_allclose(
        np.asarray(zb)[5], [zb0, 1.0 - zb0 - gam - eps, gam, eps], atol=1e-12)
    np.testing.assert_allclose(np.asarray(za)[5], _NEMO_AB3_ZA, atol=1e-12)
    # dissipative: forward-weighted interpolation
    assert zb0 > 0.5


def test_ll_init_ramp_rows():
    """Per-window ramp: substep 0 forward/FB, substep 1 forward/AB2-AM3
    (dynspg_ts:536-543 + ts_bck_interp jn==1/2)."""
    za, zb = nemo_ab3am4_coeff_arrays(4)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(za)[1], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(
        np.asarray(zb)[1],
        [1.0833333333333, -0.1666666666666, 0.0833333333333, 0.0], atol=1e-12)


def test_n1_edge_is_forward():
    za, zb = nemo_ab3am4_coeff_arrays(1)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)


def test_wide_halo_rejected_and_filter_typo_raises():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    assert r.model_config.barotropic.barotropic_time_filter == "nemo_ab3am4"
    with pytest.raises(ValueError, match="wide_halo"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(
                barotropic=r.model_config.barotropic._replace(
                    barotropic_wide_halo=True)))
    with pytest.raises(ValueError, match="barotropic_time_filter"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(
                barotropic=r.model_config.barotropic._replace(
                    barotropic_time_filter="typo")))
