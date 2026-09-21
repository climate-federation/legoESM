"""Oracle-bound construction controls for the ICE_RHEO rung-3.4 card."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import jax
import netCDF4
import numpy as np
import pytest
from legoesm.core.precision import get_policy
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    ICE_RHEO_TRACERS,
    build_ice_rheo_card,
    ice_rheo_air_stress,
    step_ice_rheo_card,
    validate_ice_rheo_card,
)

from legoesm import constants

_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
_ORACLE_GATE = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py"
)
_POINTWISE_BAR = 1.0e-15
# Shipped ICE_RHEO namelist_cfg:19-20,35 and usrdef_sbc.F90:112-117,122-127.
_TRACER_COUNT = 32
_SPACING_M = 2000.0
_STEP_DT_S = 30.0
_WIND_SPINUP_S = 21600.0
_WIND_MAX_M_S = 15.0
_DOMAIN_KM = 2000.0
_KM_TO_M = 1000.0
_WIND_RATIO = -0.8
_AIR_DENSITY_KG_M3 = 1.22
_AIR_DRAG = 1.4e-3


def _oracle_gate_module():
    spec = importlib.util.spec_from_file_location("rung34_oracle_gate", _ORACLE_GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle_card():
    gate = _oracle_gate_module()
    _, frame = gate.read_frame(_ROOT / "oracle_ice_step_entry_kt00000001.bin")
    names = (
        "e1t",
        "e2t",
        "e1u",
        "e2u",
        "e1v",
        "e2v",
        "e1f",
        "e2f",
        "tmask",
        "umask",
        "vmask",
    )
    with netCDF4.Dataset(_ROOT / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in names}
    with netCDF4.Dataset(_ROOT / "output.init_ice.nc") as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    return build_ice_rheo_card(frame, mesh, ocean_temperature_k), frame, mesh


def _mesh_xy(value: np.ndarray) -> np.ndarray:
    while value.ndim > 2:
        value = value[0]
    return value.T


def test_rung34_card_geometry_and_dtype_gate(oracle_card) -> None:
    card, _, mesh = oracle_card
    assert get_policy().transcendentals == "libm"
    assert card.case == "ICE_RHEO_OMIP_L3"
    assert card.dynamics_scheme == "si3_aevp"
    assert card.rheology_staggering == "si3_c_grid"
    assert card.transport_scheme == "si3_prather_xy_alternating"
    assert card.ridging_scheme == "si3_orca1_jpl1"
    assert card.jpl == 1 and card.nlay_i == 10 and card.nlay_s == 5
    assert card.thermodynamics is False and card.landfast is False
    assert tuple(ICE_RHEO_TRACERS[:4]) == ("v_i", "v_s", "a_i", "oa_i")
    assert len(ICE_RHEO_TRACERS) == _TRACER_COUNT
    # ato_i is reconstructed after each category-advection loop and owns no
    # Prather moments (icedyn_adv_pra.F90:423-429; module arrays :34-46).
    assert "ato_i" not in ICE_RHEO_TRACERS

    for name in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"):
        candidate = np.asarray(getattr(card.metrics, name))[2:-2, 2:-2]
        np.testing.assert_array_equal(candidate, _mesh_xy(mesh[name]))
    assert all(
        str(value.dtype) == "float64"
        for value in jax.tree_util.tree_leaves(card.initial_state)
    )

    planted = card._replace(
        metrics=card.metrics._replace(e1t=card.metrics.e1t.at[20, 20].add(1.0))
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            np.asarray(planted.metrics.e1t)[2:-2, 2:-2], _mesh_xy(mesh["e1t"])
        )


def test_rung34_initial_state_is_the_oracle_entry_frame(oracle_card) -> None:
    card, frame, _ = oracle_card
    index = ICE_RHEO_TRACERS.index
    for name in ("v_i", "v_s", "a_i", "oa_i", "a_ip", "v_ip", "v_il"):
        candidate = np.asarray(card.initial_state.contents[..., index(name)])
        np.testing.assert_array_equal(candidate, frame[name][..., 0])
    np.testing.assert_array_equal(card.initial_state.dynamics.u_ice_u, frame["u_ice"])
    np.testing.assert_array_equal(card.initial_state.dynamics.v_ice_v, frame["v_ice"])
    assert all(np.count_nonzero(np.asarray(value)) == 0 for value in card.initial_state.moments)


def test_rung34_shipped_wind_integer_exponent_branch(oracle_card) -> None:
    card, _, _ = oracle_card
    stress_u, stress_v = ice_rheo_air_stress(card.initial_state, 1)
    global_i = 3.0
    global_j = 3.0
    spinup = _STEP_DT_S / _WIND_SPINUP_S
    normalization = _WIND_MAX_M_S / math.sqrt(_DOMAIN_KM * _KM_TO_M)
    wind_u = normalization * (_DOMAIN_KM - 2.0 * global_i * 2.0) * spinup
    wind_v = (
        normalization * (_DOMAIN_KM - 2.0 * global_j * 2.0) * _WIND_RATIO * spinup
    )
    magnitude = math.sqrt(wind_u * wind_u + wind_v * wind_v)
    np.testing.assert_allclose(
        stress_u[2, 2], _AIR_DENSITY_KG_M3 * _AIR_DRAG * magnitude * wind_u
    )
    np.testing.assert_allclose(
        stress_v[2, 2], _AIR_DENSITY_KG_M3 * _AIR_DRAG * magnitude * wind_v
    )
    with pytest.raises(ValueError, match="must be positive"):
        ice_rheo_air_stress(card.initial_state, 0)


def test_rung34_card_rejects_frankenstein_selector(oracle_card) -> None:
    card, _, _ = oracle_card
    with pytest.raises(ValueError, match="selector composition"):
        validate_ice_rheo_card(card._replace(ridging_scheme="lipscomb2007"))


def test_rung34_kt1_full_dynall_gate(oracle_card) -> None:
    card, _, _ = oracle_card
    gate = _oracle_gate_module()
    _, oracle = gate.read_frame(
        _ROOT / "oracle_ice_step_entry_kt00000002.bin"
    )
    candidate = step_ice_rheo_card(card, completed_steps=0)
    index = ICE_RHEO_TRACERS.index
    fields = {
        name: np.asarray(candidate.contents[..., index(name)])
        for name in ICE_RHEO_TRACERS
    }
    fields.update(
        {
            "t_su": np.asarray(candidate.t_surface),
            "sv_i": np.asarray(candidate.bulk_salt_diagnostic),
            "u_ice": np.asarray(candidate.dynamics.u_ice_u),
            "v_ice": np.asarray(candidate.dynamics.v_ice_v),
            "stress1_i": np.asarray(candidate.dynamics.stress1_t),
            "stress2_i": np.asarray(candidate.dynamics.stress2_t),
            "stress12_i": np.asarray(candidate.dynamics.stress12_f),
        }
    )

    def oracle_field(name: str) -> np.ndarray:
        if name.startswith("e_s_l"):
            return oracle["e_s"][..., int(name[-2:]) - 1, 0]
        if name.startswith("e_i_l"):
            return oracle["e_i"][..., int(name[-2:]) - 1, 0]
        if name.startswith("szv_i_l"):
            return oracle["szv_i"][..., int(name[-2:]) - 1, 0]
        value = oracle[name]
        return value if value.ndim == 2 else value[..., 0]

    for name, value in fields.items():
        value = value[2:-2, 2:-2]
        expected = oracle_field(name)[2:-2, 2:-2]
        scale = max(float(np.max(np.abs(expected))), 1.0)
        error = float(np.max(np.abs(value - expected))) / scale
        assert error <= _POINTWISE_BAR, f"{name}: {error} > {_POINTWISE_BAR}"
