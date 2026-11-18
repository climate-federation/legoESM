"""Ice-shelf melt against COMPILED NEMO 5.0.1 (eORCA1 production build).

Fixture ``oracle_fixtures/nemo_5_0_1_isf.npz`` banks outputs of the compiled
NEMO ice-shelf routines (driver and votes:
``oracle_audit/shelf_npzd_bs_oracle``):

* three-equation cavity melt (``isfcav_mlt`` '3eq') on a 54-point (T, S, depth)
  grid for two exchange-velocity sets.  ``k0`` is the verbatim NEMO source
  rebuilt with ice conduction off (``rkappa = 0``): our port omits conduction,
  so that build is the exact algebra oracle.  The stock build keeps conduction
  and its gap is pinned per case as a KNOWN DIFFERENCE.
* the production 'spe' prescribed melt on six columns: compiled
  ``isf_tbl_ktop``, ``isf_tbl_lvl``, 3-D ``eos_fzp`` and ``isf_tbl_avg``, with
  literal copies of the private deposition glue.  The NEMO 5.0.1 3-D
  ``eos_fzp`` has an operator-precedence slip (``-5.87701e-2*S`` outside the
  polynomial; fixed in 5.0.2) making its TBL freezing point ~0.14 K colder;
  our port uses the corrected form, so the temperature tendency gap is pinned
  per case.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics import ice_shelf as isf

jax.config.update("jax_enable_x64", True)

FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "nemo_5_0_1_isf.npz")
RHO0, RCP, LFUS = 1026.0, 3991.86795711963, 0.334e6
LAMB = (-0.0573, 0.0832, -7.53e-8 * 9.80665 * RHO0)   # isfcav.F90 TEOS-10 branch
TOL, FLOOR = 1e-12, 1e-20
PIN_REL, PIN_ABS = 1e-6, 1e-12


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b) / np.maximum(np.abs(b), FLOOR)


def _sgap(a, b):
    return (np.asarray(a) - b) / np.maximum(np.abs(b), FLOOR)


def _three_eq(i):
    gt, gs = FIX["gamma_sets"][i]
    cfg = isf.IceShelfConfig(enabled=True, gamma_T=gt, gamma_S=gs, freeze_a=LAMB[0], freeze_b=LAMB[1],
                             freeze_c=LAMB[2], rho_w=RHO0, c_w=RCP, L_f=LFUS)
    return isf.three_equation_melt(jnp.asarray(FIX["T_C"]), jnp.asarray(FIX["S_psu"]),
                                   jnp.asarray(FIX["depth_m"]), config=cfg)


def _assert_pinned(got, pin):
    np.testing.assert_allclose(got, pin, rtol=PIN_REL, atol=PIN_ABS)


@pytest.mark.parametrize("i", [0, 1])
def test_three_equation_matches_nemo_without_conduction(i):
    r = _three_eq(i)
    assert _rel(r.freshwater_to_ocean, FIX[f"k0_fwf_{i}"]).max() <= TOL
    assert _rel(r.heat_extracted_from_ocean, -FIX[f"k0_qoce_{i}"]).max() <= TOL
    assert _rel(r.T_b_C, FIX[f"k0_tfrz_{i}"]).max() <= TOL


@pytest.mark.parametrize("i", [0, 1])
def test_three_equation_conduction_gap_pinned(i):
    r = _three_eq(i)
    _assert_pinned(_sgap(r.freshwater_to_ocean, FIX[f"stock_fwf_{i}"]), FIX[f"pin_stock_fwf_{i}"])
    _assert_pinned(_sgap(r.heat_extracted_from_ocean, -FIX[f"stock_qoce_{i}"]), FIX[f"pin_stock_qoce_{i}"])
    _assert_pinned(_sgap(r.T_b_C, FIX[f"stock_tfrz_{i}"]), FIX[f"pin_stock_tfrz_{i}"])


def _spe(j):
    zmin, zmax, fwf = FIX["spe_zmin_zmax_fwf"][j]
    nl = FIX["spe_E3"].size
    return isf.isf_prescribed_melt_tendencies(
        jnp.asarray(FIX["spe_SAL"]), jnp.asarray(FIX["spe_E3"]), jnp.ones(nl), jnp.asarray(fwf),
        jnp.asarray(zmin), jnp.asarray(zmax), rho_0=RHO0, c_sw=RCP, L_fus=LFUS)


@pytest.mark.parametrize("j", range(len(FIX["spe_names"])))
def test_spe_deposition_and_volume_match_nemo(j):
    dT, _, eta = _spe(j)
    e3 = FIX["spe_E3"]
    nem = FIX["spe_nemo_dtdt"][j]
    w_ours = np.asarray(dT) * e3 / np.sum(np.asarray(dT) * e3)
    w_nemo = nem * e3 / np.sum(nem * e3)
    assert np.abs(w_ours - w_nemo).max() <= TOL
    assert np.array_equal(np.asarray(dT) != 0, nem != 0)
    assert _rel(eta, FIX["spe_nemo_eta"][j]) <= TOL


@pytest.mark.parametrize("j", range(len(FIX["spe_names"])))
def test_spe_tendency_gap_pinned(j):
    dT, _, _ = _spe(j)
    nem = FIX["spe_nemo_dtdt"][j]
    got = np.where(nem != 0, _sgap(dT, nem), 0.0)
    _assert_pinned(got, FIX["pin_spe_dtdt"][j])
