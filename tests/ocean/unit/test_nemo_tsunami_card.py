"""The TSUNAMI card (round 1): transcription, explicit switches, refusals.

File-free only.  The oracle comparison waits for the acquisition record
(scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run.sh).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as CONSTANTS
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    TSUNAMI_NAMELIST,
    TSUNAMI_UNMEASURED,
    build_tsunami_zco_card,
    tsunami_horizontal_coordinates,
    validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
)
from legoesm.ocean.vertical import create_z_star_from_thicknesses


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(previous)


@pytest.fixture(scope="module")
def card():
    return build_tsunami_zco_card()


def test_construction_and_shapes(card):
    s = card.recipe.initial_state
    assert card.case == "TSUNAMI-zco"
    assert (card.dt_s, card.n_steps, card.dummy_bottom_records) == (1000.0, 100, 1)
    # usrdef_nam.F90:113-117: NINT(2000/10)+1 = 201 each way, kpk = 2.
    assert s.eta.data.shape == (201, 201)
    assert s.T.data.shape == (201, 201, 1)
    assert s.u.data.shape == (201, 202, 1) and s.v.data.shape == (202, 201, 1)
    assert s.uu_b.data.shape == (201, 202) and s.vv_b.data.shape == (202, 201)
    assert np.all(np.asarray(s.T.data) == 20.0)
    assert np.all(np.asarray(s.S.data) == 30.0)
    for f in (s.u, s.v, s.uu_b, s.vv_b):
        assert not np.any(np.asarray(f.data))
    assert np.all(np.asarray(card.recipe.land_mask) == 1.0)


def test_explicit_switch_values(card):
    nl = TSUNAMI_NAMELIST
    want = dict(
        rn_domszx_km=2000.0, rn_domszy_km=2000.0, rn_domszz_m=100.0,
        rn_dx_km=10.0, rn_dy_km=10.0, rn_0xratio=0.2, rn_0yratio=0.4,
        nn_fcase=0, rn_ppgphi0_deg=38.5, ln_Iperio=True, ln_Jperio=True,
        nn_itend=100, rn_Dt_s=1000.0, key_RK3=False, key_qco=True,
        key_vco_1d=True, ln_bt_fw=True, nn_bt_flt=1, rn_bt_alpha=0.0,
        ln_bt_auto=True, rn_bt_cmax=0.8, nn_e_resolved=6, ln_dynvor_een=True,
        nn_e3f_typ=0, rn_avm0_m2_s=1.2e-4, rn_avt0_m2_s=1.2e-5,
        ln_zad_Aimp=False, rn_shlat=0.0, ln_drg_OFF=True, ln_seos=True,
    )
    for k, v in want.items():
        assert getattr(nl, k) == v, k
    cfg = card.recipe.model_config
    assert cfg.barotropic.barotropic_time_filter == "nemo_boxcar1_ab3"
    assert cfg.barotropic.n_barotropic_substeps == 6
    assert cfg.barotropic.barotropic_coriolis == "een_metric"
    assert (cfg.A_v, cfg.K_v, cfg.K_h) == (1.2e-4, 1.2e-5, 0.0)
    assert cfg.lateral_viscosity.A_h == 0.0
    assert cfg.eos == "nemo_seos" and cfg.eos_nemo_seos.a0 == 1.6550e-1
    # dynspg_ts.F90:1223-1240 by hand: zcmax = sqrt(grav*H*(2/dx^2)).
    zcmax = math.sqrt(CONSTANTS.g * 100.0 * 2.0 / 1.0e8)
    assert math.ceil(1000.0 / 0.8 * zcmax) == 6


def test_f_plane(card):
    f0 = 2.0 * CONSTANTS.Omega * math.sin(math.pi / 180.0 * 38.5)
    src = tsunami_horizontal_coordinates()
    assert src["f0"] == f0
    assert np.all(np.asarray(card.recipe.grid.f_T) == src["f0"])
    assert (src["ii0"], src["ij0"]) == (40, 80)


def test_initial_ssh_against_formula_at_three_cells(card):
    eta = np.asarray(card.recipe.initial_state.eta.data)
    # Interior extremes of zti, ztj are 161 and 121 cells (i = j = 201).
    zmax = math.sqrt((10.0 * 161) ** 2 + (10.0 * 121) ** 2) / 20.0
    cells = {(79, 39): 0.0, (79, 45): 60.0, (85, 43): math.hypot(40.0, 60.0)}
    for (j, i), dist in cells.items():
        want = 0.1 * math.cos(dist / zmax * math.pi * 0.5)
        assert eta[j, i] == pytest.approx(want, rel=1e-14, abs=0.0)
    assert eta[79, 39] == 0.1
    assert eta[0, 0] == 0.0
    assert np.count_nonzero(eta) == 325


def test_execution_gate_refuses_with_named_blockers(card):
    assert len(TSUNAMI_UNMEASURED) == 5
    with pytest.raises(ValueError, match="B1:stp_mlf_external_mode_only"):
        validate_nemo_testcase_card_for_execution(card)


def test_validator_refuses_planted_drift(card):
    cfg = card.recipe.model_config
    bad_cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        n_barotropic_substeps=30))
    bad = card._replace(recipe=card.recipe._replace(model_config=bad_cfg))
    with pytest.raises(ValueError, match="filter/substeps"):
        validate_nemo_testcase_card(bad)
    with pytest.raises(ValueError, match="TSUNAMI_UNMEASURED"):
        validate_nemo_testcase_card(card._replace(unmeasured_features=()))


def test_single_level_is_opt_in_only():
    with pytest.raises(ValueError, match=">= 2 thicknesses"):
        create_z_star_from_thicknesses(np.array([100.0]))
    z = create_z_star_from_thicknesses(np.array([100.0]), allow_single_level=True)
    assert z.n_levels == 1
