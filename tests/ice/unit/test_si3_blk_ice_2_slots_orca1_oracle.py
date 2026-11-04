"""ORCA1 oracle: blk_ice_2 returns NEMO's longwave and sensible-heat terms.

NEMO 5.0.1 ``blk_ice_2`` (sbcblk.F90) builds ``qns_ice = z_qlw - z_qsb - qla_ice``
and ``dqns_ice = -(z_dqlw + z_dqsb + dqla_ice)``.  The port computed all four
loop terms but returned none of them, so the ice-surface solver and the
ocean exchange could not read net longwave, sensible heat or their
temperature derivatives (tier-4 surface finding S2).

Fixture ``fixtures/orca1_t4/si3_blk_ice_2_slots.npz`` (+ ``.json``): seeded wet
ice and wet open-water cells of the tier-4 NEMO 5.0.1 ORCA1 blk_ice_2 record
(kt 17281, 32 ranks).  ``qlw_ice``/``qsb_ice`` are the recorded ``z_qlw``/``z_qsb``;
``dqlw_ice``/``dqsb_ice`` are NEMO's loop scalars rebuilt from recorded operands
(``4*emiss_i*stefan*t_su**3``, ``rhoa*wndm_ice*rCp_air*Ch_ice``; recording them
changes gfortran FMA contraction, tier-4 blkice V11).  Tolerance rel 1e-12,
atol 1e-12 * max|field| (vote V11).
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4" / "si3_blk_ice_2_slots.npz"
TOL = 1e-12
SLOTS = ("qlw_ice", "dqlw_ice", "qsb_ice", "dqsb_ice")


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


@pytest.fixture(scope="module")
def fx():
    z = np.load(FIX)
    return {k: z[k] for k in z.files}


def _blk(fx, T=None):
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.ice.sea_ice import _nemo_si3_blk_ice_2
    i = {k.split(":", 1)[1]: jnp.asarray(v) for k, v in fx.items() if k.startswith("blk2_in:")}
    return _nemo_si3_blk_ice_2(
        T_surface=i["t_su"] if T is None else T, h_ice=i["h_i"], h_snow=i["h_s"], cloud_fraction=i["cloud_fra"],
        theta_air=i["theta_air_zt"], q_air=i["q_air_zt"], p_surface=i["slp"], lw_down=i["qlw"],
        precip=i["precip"], snow=i["snow"], qsr=i["qsr"], rho_air=i["rhoa"], wind=i["wndm_ice"],
        Ch=i["Ch_ice"], Ce=i["Ce_ice"], ice_fraction_before=i["at_i_b"], category_fraction_before=i["a_i_b"],
        sst_celsius=i["sst_m"], qns_ocean=i["qns_oce"], qsr_ocean=i["qsr_oce"], emp_ocean_raw=i["emp"],
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG)


def test_fixture_is_wet_and_has_ice(fx):
    assert np.all(fx["blk2_in:smask0"] == 1.0)
    assert {"wet_ice", "wet_open"} <= set(fx["blk2_group"].tolist())


def test_slots_match_nemo(fx):
    out = _blk(fx)
    for k in SLOTS:
        p = np.asarray(out[k], np.float64); n = fx[f"blk2_nemo:{k}"]
        np.testing.assert_allclose(p, n, rtol=TOL, atol=TOL * float(np.max(np.abs(n))), err_msg=k)


def test_slots_close_qns_and_dqns(fx):
    out = {k: np.asarray(v) for k, v in _blk(fx).items()}
    np.testing.assert_array_equal(out["qns_ice"], out["qlw_ice"] - out["qsb_ice"] - out["qla_ice"])
    np.testing.assert_array_equal(out["dqns_ice"], -(out["dqlw_ice"] + out["dqsb_ice"] + out["dqla_ice"]))


def test_derivative_slots_are_the_temperature_derivatives(fx):
    """dqlw_ice = -d(qlw_ice)/dT and dqsb_ice = d(qsb_ice)/dT (NEMO sign convention), checked with jax.grad."""
    for c in np.flatnonzero(fx["blk2_group"] == "wet_ice")[:3]:
        sub = {k: v[c] if v.ndim else v for k, v in fx.items() if k.startswith("blk2_in:")}
        T0 = jnp.asarray(fx["blk2_in:t_su"][c])
        out = _blk(sub)
        g_lw = float(jax.grad(lambda T: _blk(sub, T)["qlw_ice"])(T0))
        g_sb = float(jax.grad(lambda T: _blk(sub, T)["qsb_ice"])(T0))
        np.testing.assert_allclose(-g_lw, float(out["dqlw_ice"]), rtol=1e-12)
        # NEMO's dqsb drops the Exner factor of theta_surface(T): d(qsb)/dT = dqsb * (p_ref/p)**kappa ~ 1
        assert abs(g_sb / float(out["dqsb_ice"]) - 1.0) < 0.05, (c, g_sb, float(out["dqsb_ice"]))
        assert np.isfinite(g_lw) and g_lw != 0.0
