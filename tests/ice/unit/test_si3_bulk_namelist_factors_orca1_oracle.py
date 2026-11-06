"""blk_ice_2 takes NEMO's rn_pfac, rn_efac and rn_snwblow instead of hard-wiring them.

NEMO 5.0.1 ``blk_ice_2`` (sbcblk.F90) scales total and solid precipitation by
``rn_pfac``, sublimation and its T derivative by ``rn_efac``, and spreads snow
with ``ice_var_snwblow``: ``1 - (1 - at_i_b)**rn_snwblow``.  The port hard-wired
1, 1 and 0.66 (exact at ORCA1 only; tier-4 surface finding S3).

Fixture ``fixtures/orca1_t4/si3_bulk_namelist_factors.npz`` (+ ``.json``): seeded
wet ice / open-water cells of the tier-4 NEMO 5.0.1 ORCA1 blk_ice_2 record
(kt 17281, 32 ranks), with NEMO's outputs at the recorded ORCA1 namelist
(1, 1, 0.66) and known answers at non-ORCA1 values (one combined case and one
case per factor, vote V14) from the tier-4 numpy NEMO transliteration
(tier4/blkice/nemo_ref.py, control C4: <= 1.8e-13 vs NEMO on these cells).
At ORCA1 values the oracle cannot tell the old code from the new; the
perturbed cases can.  Tolerance rel 1e-12, atol 1e-12 * max|field| (vote V11).
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4" / "si3_bulk_namelist_factors.npz"
TOL = 1e-12
OUT = ("qla_ice", "dqla_ice", "qns_ice", "dqns_ice", "evap_ice", "devap_ice", "tprecip", "sprecip", "emp_oce",
       "emp_ice", "emp_tot", "qemp_oce", "qemp_ice", "qns_tot", "qprec_ice", "qevap_ice")
CASES = ("orca1", "combined", "pfac", "efac", "snwblow")


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


def _blk(fx, pfac, efac, snwblow, at_i_b=None):
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.ice.config import SI3_HARD_SWITCHES
    from legoesm.ice.sea_ice import _nemo_si3_blk_ice_2
    i = {k.split(":", 1)[1]: jnp.asarray(v) for k, v in fx.items() if k.startswith("blk2_in:")}
    return _nemo_si3_blk_ice_2(
        T_surface=i["t_su"], h_ice=i["h_i"], h_snow=i["h_s"], cloud_fraction=i["cloud_fra"],
        theta_air=i["theta_air_zt"], q_air=i["q_air_zt"], p_surface=i["slp"], lw_down=i["qlw"],
        precip=i["precip"], snow=i["snow"], qsr=i["qsr"], rho_air=i["rhoa"], wind=i["wndm_ice"],
        Ch=i["Ch_ice"], Ce=i["Ce_ice"], ice_fraction_before=i["at_i_b"] if at_i_b is None else at_i_b,
        category_fraction_before=i["a_i_b"], sst_celsius=i["sst_m"], qns_ocean=i["qns_oce"],
        qsr_ocean=i["qsr_oce"], emp_ocean_raw=i["emp"], precip_factor=pfac, evap_factor=efac,
        snow_blow_exponent=snwblow, land_mask=i["smask0"], ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
        switches=SI3_HARD_SWITCHES)


def _close(p, n, name):
    p = np.asarray(p, np.float64)
    assert np.all(np.isfinite(p)), name
    np.testing.assert_allclose(p, n, rtol=TOL, atol=TOL * float(np.max(np.abs(n))), err_msg=name)


def test_orca1_namelist_matches_nemo(fx):
    pf, ef, sb = (float(fx[f"blk2_scalar:{k}"]) for k in ("rn_pfac", "rn_efac", "rn_snwblow"))
    assert (pf, ef, sb) == (1.0, 1.0, 0.66)
    from legoesm import constants
    assert (constants.precip_factor_orca1, constants.evap_factor_orca1,
            constants.snow_blow_exponent_orca1) == (pf, ef, sb)
    out = _blk(fx, pf, ef, sb)
    for k in OUT:
        _close(out[k], fx[f"blk2_nemo:{k}"], k)


@pytest.mark.parametrize("case", CASES)
def test_namelist_factors_match_known_answer(fx, case):
    pf, ef, sb = (float(x) for x in fx[f"blk2_case:{case}"])
    out = _blk(fx, pf, ef, sb)
    for k in OUT:
        _close(out[k], fx[f"blk2_ref_{case}:{k}"], f"{case}:{k}")


def test_perturbed_cases_differ_from_orca1(fx):
    """Control: each perturbed known answer differs from the ORCA1 one (the cases are not vacuous)."""
    for case, keys in (("pfac", "tprecip"), ("efac", "evap_ice"), ("snwblow", "emp_ice")):
        a, b = fx[f"blk2_ref_{case}:{keys}"], fx[f"blk2_ref_orca1:{keys}"]
        assert np.max(np.abs(a - b)) > 1e-6 * np.max(np.abs(b)), case


def test_snow_blow_exponent_is_differentiable(fx):
    """jax.grad vs central FD (rel step 1e-6) of emp_ice w.r.t. the exponent and at_i_b, three ice cells (V12)."""
    ice = np.flatnonzero((fx["blk2_group"] == "wet_ice") & (fx["blk2_in:snow"] > 0) & (fx["blk2_in:at_i_b"] < 0.95))[:3]
    assert ice.size == 3
    for c in ice:
        sub = {k: v[c] for k, v in fx.items() if k.startswith("blk2_in:")}
        f = lambda e: _blk(sub, 1.0, 1.0, e)["emp_ice"]  # noqa: E731
        g = float(jax.grad(f)(0.66)); h = 0.66e-6
        fd = (float(f(0.66 + h)) - float(f(0.66 - h))) / (2 * h)
        assert np.isfinite(g) and g != 0.0 and abs(g / fd - 1.0) < 1e-4, (c, g, fd)
        a0 = float(sub["blk2_in:at_i_b"])
        f = lambda a: _blk(sub, 1.0, 1.0, 0.66, at_i_b=a)["emp_ice"]  # noqa: E731
        g = float(jax.grad(f)(a0)); h = 1e-6 * max(a0, 1e-3)
        fd = (float(f(a0 + h)) - float(f(a0 - h))) / (2 * h)
        assert np.isfinite(g) and abs(g / fd - 1.0) < 1e-4, (c, g, fd)
