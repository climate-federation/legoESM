"""Non-vacuity controls for the Round-134 daily reset attribution."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
           / "ocean_fidelity" / "testcases")
YEAR = SCRIPTS / "nemo_testcase_l2_gyre_year_fromrest.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def year():
    return _load("round134_year", YEAR)


def _card():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return build_nemo_testcase_card("GYRE-zco")


def _payload(card):
    state = card.recipe.initial_state
    nlat, nlon, nlev = state.T.data.shape
    xyz = np.arange(nlat * nlon * nlev, dtype=np.float64).reshape(
        nlat, nlon, nlev)
    xy = np.arange(nlat * nlon, dtype=np.float64).reshape(nlat, nlon)
    return {
        "tn": xyz + 1.0, "sn": xyz + 2.0,
        "un": xyz + 3.0, "vn": xyz + 4.0,
        "en": xyz + 5.0, "avm_k": xyz + 6.0,
        "avt_k": xyz + 7.0, "dissl": xyz + 8.0,
        "uu_n": xy + 9.0, "vv_n": xy + 10.0,
        "sshn": xy + 11.0, "ssha": xy + 12.0,
        "ub_e": xy + 13.0, "ubb_e": xy + 14.0,
        "vb_e": xy + 15.0, "vbb_e": xy + 16.0,
        "sshb_e": xy + 17.0, "sshbb_e": xy + 18.0,
    }


@pytest.mark.parametrize("plant", ["daily-source-ulp", "daily-family-registry"])
def test_daily_reset_plants_exit_nonzero(plant):
    result = subprocess.run(
        [sys.executable, str(YEAR), "--daily-reset-self-check",
         "--plant", plant], capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"STATUS PLANT-FIRED: {plant}" in result.stdout
    assert f"REFUSE Round-134 {plant}" in result.stderr


def test_daily_reset_registry_is_exact_and_includes_live_velocity(year):
    year.validate_daily_reset_registry()
    assert tuple(year.DAILY_RESET_VARIABLES) == year.DAILY_RESET_FAMILIES
    assert year.DAILY_RESET_VARIABLES["vector"] == (
        "un", "vn", "uu_n", "vv_n", "ub_e", "ubb_e", "vb_e", "vbb_e")
    assert year.DAILY_RESET_VARIABLES["ssh"] == (
        "sshn", "ssha", "sshb_e", "sshbb_e")


def test_vector_and_ssh_resets_replace_only_the_registered_history_halves(year):
    import jax.numpy as jnp

    card = _card()
    state = card.recipe.initial_state
    assert state.uu_b is not None and state.vv_b is not None
    history = (
        jnp.full(state.uu_b.data.shape, 101.0),
        jnp.full(state.uu_b.data.shape, 102.0),
        jnp.full(state.vv_b.data.shape, 103.0),
        jnp.full(state.vv_b.data.shape, 104.0),
        jnp.full(state.eta.data.shape, 105.0),
        jnp.full(state.eta.data.shape, 106.0),
    )
    state = state._replace(bt_hist=history)
    payload = _payload(card)

    vector, pending = year.apply_daily_reset(state, payload, "vector", card)
    assert pending is None
    assert np.array_equal(np.asarray(vector.bt_hist[4]), np.asarray(history[4]))
    assert np.array_equal(np.asarray(vector.bt_hist[5]), np.asarray(history[5]))
    assert not np.array_equal(np.asarray(vector.bt_hist[0]),
                              np.asarray(history[0]))
    assert np.array_equal(np.asarray(vector.eta.data), np.asarray(state.eta.data))

    ssh, pending = year.apply_daily_reset(state, payload, "ssh", card)
    assert all(np.array_equal(np.asarray(ssh.bt_hist[index]),
                              np.asarray(history[index]))
               for index in range(4))
    assert not np.array_equal(np.asarray(ssh.bt_hist[4]),
                              np.asarray(history[4]))
    assert np.array_equal(np.asarray(ssh.eta.data), payload["sshn"])
    assert np.array_equal(np.asarray(pending), payload["ssha"])


def test_ssha_override_executes_inside_the_production_jitted_step(year):
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    card = _card()
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    gate, _ = year._gate_module()
    for kt in range(1, 7):
        freshwater, surface = gate._surface_forcings(card, state, kt)
        state = model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    freshwater, surface = gate._surface_forcings(card, state, 7)
    eta_a = jnp.asarray(state.eta.data)
    wet = np.argwhere(np.asarray(state.land_mask.data) > 0.5)
    j, i = (int(value) for value in wet[len(wet) // 2])
    eta_b = eta_a.at[j, i].add(jnp.asarray(1.0e-4, dtype=eta_a.dtype))
    left = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=eta_a)
    right = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=eta_b)
    moved = sum(
        int(np.count_nonzero(np.asarray(getattr(left, name).data).view(np.uint64)
                             != np.asarray(getattr(right, name).data).view(np.uint64)))
        for name in ("u", "v", "T", "S", "eta"))
    assert moved > 0


def test_attribution_bit_comparator_observes_one_ulp():
    owners = _load(
        "round134_owners",
        SCRIPTS / "nemo_testcase_l2_gyre_year_owners.py")
    original = np.asarray([1.0, 2.0], dtype=np.float64)
    moved = original.copy()
    moved[1] = np.nextafter(moved[1], np.inf)
    assert owners._bit_unequal(original, original) == 0
    assert owners._bit_unequal(original, moved) == 1
