"""Direct controls for the selectable SI3/ORCA1 single-category ridge arm."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice import ridging as ridging_module
from legoesm.ice.ridging import (
    SI3JPL1RidgingConfig,
    SI3JPL1RidgingState,
    apply_si3_jpl1_ridging,
)

_ORCA1_CATEGORY_COUNT = 5  # ORCA1 namelist_ice_cfg:15; outside this jpl=1 arm
_CASE_DT_S = 30.0  # tests/ICE_RHEO/EXPREF/namelist_cfg:35
_ACTIVE_DIVERGENCE_S1 = -1.0e-4  # synthetic active one-shift control
_ACTIVE_DEFORMATION_S1 = 3.0e-4  # synthetic active one-shift control


def _state(*, area: float = 0.8, open_water: float = 0.203) -> SI3JPL1RidgingState:
    def scalar(value):
        return jnp.asarray([value], dtype=jnp.float64)

    return SI3JPL1RidgingState(
        ice_area=scalar(area),
        open_water_area=scalar(open_water),
        ice_volume=scalar(1.6),
        snow_volume=scalar(0.08),
        age_content=scalar(0.24),
        pond_area=scalar(0.04),
        pond_volume=scalar(0.016),
        pond_lid_volume=scalar(0.008),
        snow_enthalpy=jnp.asarray([[3.0, 5.0]], dtype=jnp.float64),
        ice_enthalpy=jnp.asarray([[7.0, 11.0, 13.0]], dtype=jnp.float64),
        ice_salt_content=jnp.asarray([[17.0, 19.0, 23.0]], dtype=jnp.float64),
    )


def _independent_one_shift(state: SI3JPL1RidgingState) -> dict[str, float]:
    """Scalar replay of icedyn_rdgrft.F90:425-620,667-890, not the JAX helper."""

    area = float(state.ice_area[0])
    open_water = float(state.open_water_area[0])
    volume = float(state.ice_volume[0])
    divergence = -1.0e-4
    deformation = 3.0e-4
    dt = 30.0
    closing = 0.5 * 0.5 * (deformation - abs(divergence)) - min(divergence, 0.0)
    closing = max(closing, -divergence)
    opening = closing + divergence
    total = open_water + area
    g_open = open_water / total
    scale = 1.0 / (1.0 - math.exp(-1.0 / 0.03))
    transformed_minus_one = scale
    transformed_open = math.exp(-g_open / 0.03) * scale
    transformed_ice = math.exp(-1.0 / 0.03) * scale
    part_open = transformed_minus_one - transformed_open
    part_ice = transformed_open - transformed_ice
    thickness = volume / area
    ridge = 0.5 * (1.0 + math.tanh(5.0 * (thickness - 0.75))) * part_ice
    raft = part_ice - ridge
    mean_ridge = max(math.sqrt(25.0 * thickness), 1.1 * thickness)
    ridge_min = min(2.0 * thickness, 0.5 * (mean_ridge + thickness))
    ridge_exp = 3.0 * math.sqrt(thickness)
    phi = thickness / (ridge_min + ridge_exp)
    normalization = part_open + ridge * (1.0 - phi) + raft * (1.0 - 0.5)
    gross = closing / normalization
    ridge_removed = ridge * gross * dt
    raft_removed = raft * gross * dt
    frac_ridge = ridge_removed / area
    frac_raft = raft_removed / area
    retained = 1.0 - frac_ridge - frac_raft
    return {
        "area": area - ridge_removed - raft_removed + ridge_removed * phi + raft_removed * 0.5,
        "open": max(0.0, open_water + (opening - part_open * gross) * dt),
        "retained": retained,
        "ridge_fraction": frac_ridge,
        "raft_fraction": frac_raft,
        "phi": phi,
    }


def test_si3_jpl1_matches_independent_written_order_one_shift() -> None:
    state = _state()
    result, losses = apply_si3_jpl1_ridging(
        state,
        jnp.asarray([-1.0e-4], dtype=jnp.float64),
        jnp.asarray([3.0e-4], dtype=jnp.float64),
        30.0,
    )
    expected = _independent_one_shift(state)
    np.testing.assert_allclose(result.ice_area, [expected["area"]], rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(result.open_water_area, [expected["open"]], rtol=0.0, atol=2e-16)
    assert int(losses.iterations[0]) == 1

    # ln_icethd=F forces all snow/pond retention factors to one at
    # icedyn_rdgrft.F90:1244-1247, overriding the deck's 0.5 values.
    np.testing.assert_allclose(result.snow_volume, state.snow_volume)
    np.testing.assert_array_equal(losses.snow_volume, 0.0)
    np.testing.assert_allclose(result.pond_volume, state.pond_volume)
    np.testing.assert_array_equal(losses.pond_volume, 0.0)
    np.testing.assert_allclose(result.ice_volume, state.ice_volume, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(result.ice_enthalpy, state.ice_enthalpy, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        result.ice_salt_content, state.ice_salt_content, rtol=0.0, atol=4e-15
    )


def test_si3_jpl1_ledger_uses_source_rounding_under_jit(monkeypatch) -> None:
    """The donor/receiver ledger must reach the canonical arithmetic guard."""

    state = _state()

    def advance(current):
        return apply_si3_jpl1_ridging(
            current,
            jnp.asarray([_ACTIVE_DIVERGENCE_S1], dtype=jnp.float64),
            jnp.asarray([_ACTIVE_DEFORMATION_S1], dtype=jnp.float64),
            _CASE_DT_S,
        )[0]

    rounded = jax.jit(advance)(state)
    jax.block_until_ready(rounded)
    monkeypatch.setattr(ridging_module, "nemo_source_round", lambda value: value)

    def advance_unrounded(current):
        return apply_si3_jpl1_ridging(
            current,
            jnp.asarray([_ACTIVE_DIVERGENCE_S1], dtype=jnp.float64),
            jnp.asarray([_ACTIVE_DEFORMATION_S1], dtype=jnp.float64),
            _CASE_DT_S,
        )[0]

    unrounded = jax.jit(advance_unrounded)(state)
    jax.block_until_ready(unrounded)

    assert not np.array_equal(rounded.ice_enthalpy, unrounded.ice_enthalpy)


def test_si3_jpl1_divergence_without_shear_is_noop() -> None:
    state = _state(area=0.8, open_water=0.2)
    result, losses = apply_si3_jpl1_ridging(
        state,
        jnp.asarray([1.0e-4], dtype=jnp.float64),
        jnp.asarray([1.0e-4], dtype=jnp.float64),
        30.0,
    )
    for before, after in zip(state, result, strict=True):
        np.testing.assert_array_equal(after, before)
    assert int(losses.iterations[0]) == 0


def test_si3_jpl1_reports_excessive_removal_and_open_water_branches() -> None:
    state = _state(area=0.8, open_water=0.2)
    # This closing demand removes more than the available category area, so
    # icedyn_rdgrft.F90:600-612 must take its excessive-removal correction.
    _, losses = apply_si3_jpl1_ridging(
        state,
        jnp.asarray([-100.0], dtype=jnp.float64),
        jnp.asarray([100.0], dtype=jnp.float64),
        30.0,
    )
    assert bool(losses.excessive_removal_clamp[0])
    assert losses.open_water_correction.dtype == jnp.bool_


def test_si3_jpl1_preserves_nemo_open_water_and_lid_roundoff_exclusions() -> None:
    # Open water is not passed to ice_var_roundoff; icevar.F90:871 clips only
    # lid values in -epsi10 < v_il < 0, leaving larger negatives diagnostic.
    zero = jnp.asarray([0.0], dtype=jnp.float64)
    negative_open = _state(area=0.0, open_water=-2.0e-10)
    negative_open = negative_open._replace(
        ice_volume=zero, pond_lid_volume=jnp.asarray([-2.0e-10], dtype=jnp.float64)
    )
    result, _ = apply_si3_jpl1_ridging(negative_open, zero, zero, 30.0)
    np.testing.assert_array_equal(result.open_water_area, negative_open.open_water_area)
    np.testing.assert_array_equal(result.pond_lid_volume, negative_open.pond_lid_volume)

    tiny_lid = negative_open._replace(
        open_water_area=jnp.asarray([1.0], dtype=jnp.float64),
        pond_lid_volume=jnp.asarray([-0.5e-10], dtype=jnp.float64),
    )
    rounded, _ = apply_si3_jpl1_ridging(tiny_lid, zero, zero, 30.0)
    np.testing.assert_array_equal(rounded.pond_lid_volume, zero)


def test_si3_jpl1_rejects_unmeasured_selector() -> None:
    with pytest.raises(ValueError, match="no Frankenstein fallback"):
        apply_si3_jpl1_ridging(
            _state(),
            jnp.asarray([-1.0e-4], dtype=jnp.float64),
            jnp.asarray([3.0e-4], dtype=jnp.float64),
            30.0,
            config=SI3JPL1RidgingConfig(rafting=False),
        )


def test_si3_jpl1_rejects_a_state_category_axis() -> None:
    state = _state()
    scalar = tuple(
        jnp.broadcast_to(value.reshape(1, 1, 1), (1, 1, _ORCA1_CATEGORY_COUNT))
        for value in state[:8]
    )
    layered = tuple(
        jnp.broadcast_to(
            value.reshape(1, 1, 1, value.shape[-1]),
            (1, 1, _ORCA1_CATEGORY_COUNT, value.shape[-1]),
        )
        for value in state[8:]
    )
    multicategory = SI3JPL1RidgingState(*scalar, *layered)
    zero = jnp.zeros((1, 1, _ORCA1_CATEGORY_COUNT), dtype=jnp.float64)
    with pytest.raises(ValueError, match="category-collapsed state"):
        apply_si3_jpl1_ridging(multicategory, zero, zero, _CASE_DT_S)


def test_si3_jpl1_jit_and_gradient_are_finite() -> None:
    state = _state()
    divergence = jnp.asarray([-1.0e-4], dtype=jnp.float64)
    deformation = jnp.asarray([3.0e-4], dtype=jnp.float64)

    @jax.jit
    def objective(area: jnp.ndarray) -> jnp.ndarray:
        candidate = state._replace(ice_area=area)
        result, _ = apply_si3_jpl1_ridging(candidate, divergence, deformation, 30.0)
        return jnp.sum(result.ice_area + result.snow_volume)

    value = objective(state.ice_area)
    gradient = jax.grad(objective)(state.ice_area)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))

    zero = jnp.zeros_like(state.ice_area)
    ice_free = state._replace(
        ice_area=zero,
        open_water_area=jnp.ones_like(zero),
        ice_volume=zero,
    )

    def ice_free_objective(area: jnp.ndarray) -> jnp.ndarray:
        candidate = ice_free._replace(ice_area=area)
        result, _ = apply_si3_jpl1_ridging(candidate, zero, zero, 30.0)
        return jnp.sum(result.ice_area)

    ice_free_gradient = jax.grad(ice_free_objective)(zero)
    assert bool(jnp.all(jnp.isfinite(ice_free_gradient)))
