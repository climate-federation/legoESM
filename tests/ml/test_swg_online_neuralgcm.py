"""Direct tests for the NeuralGCM-style SWG losses: the lead-time low-pass,
the KE-spectrum term and the bias term (Kochkov et al. 2024, Methods 7.4)."""
from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.diagnostics import (
    isotropic_energy_spectrum,
    velocity_to_cell_centre,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    build_swg_card,
)
from legoesm.training import swg_online
from legoesm.training.swg_online import (
    BalancedNormalizers,
    BalancedTrajectory,
    build_closure_geometry,
    deterministic_trajectory_loss,
    ensemble_trajectory_scores,
    lead_filter,
    lead_filter_max_band,
    lead_time_cutoff_bands,
    lead_time_cutoff_table,
    neuralgcm_ensemble_scores,
    neuralgcm_trajectory_loss,
)

OBJECTIVE = "symmetric"


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def case():
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    eq7 = {"beta_m1_s1": card.resolved_namelist.rn_beta_m_s,
           "basin_length_m": card.resolved_namelist.rn_domsiz_m,
           "wall_ring": "weighted"}
    return card, geometry, eq7


def _mode(geometry, ky, kx):
    """The orthogonal DCT-II basis function (ky, kx) of the wet T box, zero
    elsewhere."""
    j0, j1, i0, i1 = geometry.wet_bounds
    ny, nx = j1 - j0 + 1, i1 - i0 + 1
    field = np.zeros(tuple(geometry.tmask.shape))
    field[j0:j1 + 1, i0:i1 + 1] = (
        np.cos(np.pi * ky * (np.arange(ny) + 0.5) / ny)[:, None]
        * np.cos(np.pi * kx * (np.arange(nx) + 0.5) / nx)[None, :])
    return jnp.asarray(field)


def _native(geometry, rng, windows=2, hours=8, scale=1.0e-2):
    shape = (windows, hours) + tuple(geometry.tmask.shape)
    return tuple(jnp.asarray(scale * rng.standard_normal(shape))
                 for _ in range(3))


def _with_u(native, increment):
    return (native[0], native[1] + increment, native[2])


def _u_pattern(geometry, records, hours, ky=40, kx=0, amplitude=1.0e-3):
    """A high-wavenumber U-velocity increment on the given records only."""
    pattern = amplitude * _mode(geometry, ky, kx) * geometry.umask
    keep = np.zeros(hours, dtype=bool)
    keep[list(records)] = True
    return jnp.where(jnp.asarray(keep)[None, :, None, None], pattern, 0.0)


# ------------------------------------------------------------ cutoff table

def test_cutoff_table_placeholder_refuses_and_a_table_steps_with_lead():
    with pytest.raises(ValueError, match="PENDING_TWIN_MEASUREMENT"):
        lead_time_cutoff_table(swg_online.LEAD_TIME_CUTOFF_PENDING)
    with pytest.raises(ValueError, match="PENDING_TWIN_MEASUREMENT"):
        lead_time_cutoff_bands("PENDING_TWIN_MEASUREMENT", n_records=4,
                               target_interval_hours=3)
    table = {"lead_hours": [0, 6, 24], "cutoff_band": [50, 20, 5.5]}
    # Left-continuous step: lead 3 -> entry 0, lead 6 -> entry 1, lead 24 -> 2.
    assert lead_time_cutoff_bands(
        table, n_records=12, target_interval_hours=3).tolist() == [
        50, 20, 20, 20, 20, 20, 20, 5.5, 5.5, 5.5, 5.5, 5.5]
    assert lead_time_cutoff_table(table) == ((0, 6, 24), (50.0, 20.0, 5.5))


@pytest.mark.parametrize("name", [
    "exp9_rigidlid_neuralgcm.json",
    "exp10_rigidlid_energy_score_neuralgcm_m8.json"])
def test_committed_table_rounds_a_lead_up_to_the_next_measured_day(name):
    root = pathlib.Path(__file__).resolve().parents[2] / "config/swg_online"
    table = json.loads((root / name).read_text())["loss"]["lead_time_cutoff"]
    # 3-hour records: lead 3 h is record 0, lead 24 h record 7, 27 h record 8.
    cut = lead_time_cutoff_bands(table, n_records=32 * 8,
                                 target_interval_hours=3)

    def at(lead_hours):
        return cut[lead_hours // 3 - 1]

    # Leads 3, 24 -> day 1; 27, 48 -> day 2 (the measured day-1/day-2 bands).
    assert [at(h) for h in (3, 24, 27, 48)] == [70, 70, 63, 63]
    # An unmeasured day takes the next measured day's band: day 5 -> day 6,
    # day 7 -> day 8, days 9-11 -> day 12; day 32 is the last.
    assert [at(24 * d) for d in (3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 25,
                                 32)] == [
        57, 54, 48, 48, 42, 42, 27, 27, 19, 19, 13, 13, 9, 9]
    assert at(24 * 4 + 3) == 48 and at(24 * 8 + 3) == 27
    # Lead 0 keeps every mode (the 80 x 80 wet box's largest radius is 111.7).
    assert lead_time_cutoff_table(table)[1][0] >= np.hypot(79, 79)


@pytest.mark.parametrize("table", [
    None, "full", [], {"lead_hours": [0]}, {"lead_hours": [0], "cutoff_band": [1],
                                            "extra": 1},
    {"lead_hours": [0, 6], "cutoff_band": [5]},
    {"lead_hours": [], "cutoff_band": []},
    {"lead_hours": [3, 6], "cutoff_band": [5, 4]},       # must start at 0
    {"lead_hours": [0, 6, 6], "cutoff_band": [5, 4, 3]},  # strictly increasing
    {"lead_hours": [0.0, 6], "cutoff_band": [5, 4]},      # integers
    {"lead_hours": [0, 6], "cutoff_band": [4, 5]},        # non-increasing
    {"lead_hours": [0, 6], "cutoff_band": [5, -1]},
    {"lead_hours": [0, 6], "cutoff_band": [5, float("nan")]},
    {"lead_hours": [0, 6], "cutoff_band": [5, True]}])
def test_cutoff_table_refuses_a_malformed_table(table):
    with pytest.raises(ValueError, match="lead_time_cutoff"):
        lead_time_cutoff_table(table)


# --------------------------------------------------------------- the filter

def test_lead_filter_keeps_modes_up_to_the_cutoff_radius_and_drops_the_rest(case):
    _, geometry, _ = case
    support = jnp.ones(tuple(geometry.tmask.shape), dtype=bool)
    low, high = _mode(geometry, 3, 4), _mode(geometry, 30, 40)    # radii 5, 50
    field = jnp.broadcast_to(low + high, (4,) + low.shape)
    big = lead_filter_max_band(geometry) + 1.0
    out = lead_filter(field, geometry, np.array([big, big, 20.0, 5.0]), support)
    # A record at the full band is untouched, bit for bit.
    assert np.array_equal(np.asarray(out[:2]), np.asarray(field[:2]))
    # Below it the high mode is gone to round-off; radius 5 sits on the cutoff
    # 5.0 and is kept (the test is <=).
    assert np.max(np.abs(np.asarray(out[2] - low))) < 1.0e-12
    assert np.max(np.abs(np.asarray(out[3] - low))) < 1.0e-12
    cut = lead_filter(field, geometry, np.array([big, big, 4.99, 4.99]), support)
    assert np.max(np.abs(np.asarray(cut[2]))) < 1.0e-12
    # Every record at the full band: the very same array comes back.
    assert lead_filter(field, geometry, np.full(4, big), support) is field
    # The result lives on the field's wet points only.
    half = jnp.zeros_like(support).at[:, :40].set(True)
    assert not np.any(np.asarray(
        lead_filter(field, geometry, np.zeros(4), half))[..., 40:])
    with pytest.raises(ValueError, match="one per record"):
        lead_filter(field, geometry, np.array([1.0, 2.0]), support)
    with pytest.raises(ValueError, match="non-negative"):
        lead_filter(field, geometry, np.array([1.0, 1.0, 1.0, -1.0]), support)
    with pytest.raises(ValueError, match="finite"):
        lead_filter(field, geometry, np.array([1.0, 1.0, 1.0, np.nan]), support)


# ------------------------------------------------- bit-for-bit at full band

def test_full_band_trajectory_loss_is_the_symmetric_loss_bit_for_bit(case):
    _, geometry, eq7 = case
    rng = np.random.default_rng(3)
    predicted, target = _native(geometry, rng), _native(geometry, rng)
    old = deterministic_trajectory_loss(
        predicted, target, geometry, objective=OBJECTIVE, **eq7)
    full = np.full(8, lead_filter_max_band(geometry))
    total, filtered, spectrum, bias = neuralgcm_trajectory_loss(
        predicted, target, geometry, objective=OBJECTIVE, cutoff_bands=full,
        spectrum_weight=0.0, bias_weight=0.0, **eq7)
    assert float(filtered) == float(old) > 0.0
    assert float(total) == float(old)          # + exact zeros
    assert float(spectrum) > 0.0 and float(bias) > 0.0
    weighted, *_ = neuralgcm_trajectory_loss(
        predicted, target, geometry, objective=OBJECTIVE, cutoff_bands=full,
        spectrum_weight=2.0, bias_weight=3.0, **eq7)
    assert float(weighted) == pytest.approx(
        float(old) + 2.0 * float(spectrum) + 3.0 * float(bias), rel=1.0e-13)
    # A narrower band is a different loss.
    narrow, *_ = neuralgcm_trajectory_loss(
        predicted, target, geometry, objective=OBJECTIVE,
        cutoff_bands=np.full(8, 5.0), spectrum_weight=0.0, bias_weight=0.0,
        **eq7)
    assert float(narrow) < float(old)


def test_full_band_energy_score_is_the_existing_score_bit_for_bit(case):
    _, geometry, _ = case
    rng = np.random.default_rng(5)
    shape = (2, 3, 4) + tuple(geometry.tmask.shape)
    predicted = tuple(jnp.asarray(1.0e-2 * rng.standard_normal(shape))
                      for _ in range(3))
    target = _native(geometry, rng, hours=4)
    normalizers = BalancedNormalizers(1.0e-10, 1.0e-6, 1.0e-6)
    old = ensemble_trajectory_scores(
        predicted, target, geometry, normalizers, objective=OBJECTIVE)
    full = np.full(4, lead_filter_max_band(geometry))
    total, distance, spread, energy, spectrum = neuralgcm_ensemble_scores(
        predicted, target, geometry, normalizers, objective=OBJECTIVE,
        cutoff_bands=full, spectrum_weight=0.0)
    assert (float(energy), float(distance), float(spread)) == tuple(
        float(x) for x in old)
    assert float(total) == float(old[0]) > 0.0
    assert float(spectrum) > 0.0
    weighted, *_ = neuralgcm_ensemble_scores(
        predicted, target, geometry, normalizers, objective=OBJECTIVE,
        cutoff_bands=full, spectrum_weight=4.0)
    assert float(weighted) == pytest.approx(
        float(old[0]) + 4.0 * float(spectrum), rel=1.0e-13)


# ------------------------------------------------- the lead-time cutoff acts

def test_a_planted_small_scale_difference_stops_counting_after_its_lead(case):
    _, geometry, eq7 = case
    hours = 8
    table = {"lead_hours": [0, 12], "cutoff_band": [1000, 6]}
    # 3-hour records: leads 3, 6, 9 keep every mode, leads 12..24 only radius<=6.
    cutoff = lead_time_cutoff_bands(table, n_records=hours,
                                    target_interval_hours=3)
    assert cutoff.tolist() == [1000] * 3 + [6] * 5
    full = np.full(hours, 1000.0)
    zeros = tuple(jnp.zeros((2, hours) + tuple(geometry.tmask.shape))
                  for _ in range(3))

    def eq7_term(increment, bands):
        return float(neuralgcm_trajectory_loss(
            _with_u(zeros, increment), zeros, geometry, objective=OBJECTIVE,
            cutoff_bands=bands, spectrum_weight=0.0, bias_weight=0.0,
            **eq7)[1])

    early = _u_pattern(geometry, range(0, 3), hours)
    late = _u_pattern(geometry, range(3, 8), hours)
    # Before its lead the difference counts in full, bit for bit.
    assert eq7_term(early, cutoff) == eq7_term(early, full) > 0.0
    # After it, almost nothing of it is left (the pattern's DCT leakage into
    # radius <= 6 is the remainder).
    unfiltered, filtered = eq7_term(late, full), eq7_term(late, cutoff)
    assert unfiltered > 0.0 and filtered < 0.05 * unfiltered
    # A large-scale difference is not removed by the same cutoff.
    large = _u_pattern(geometry, range(3, 8), hours, ky=2)
    assert eq7_term(large, cutoff) > 0.5 * eq7_term(large, full)


def test_the_energy_score_filter_acts_on_all_three_balanced_fields(case):
    _, geometry, _ = case
    hours = 4
    cutoff = np.array([1000.0, 1000.0, 6.0, 6.0])
    full = np.full(hours, 1000.0)
    shape = (2, 2, hours) + tuple(geometry.tmask.shape)
    members = tuple(jnp.zeros(shape) for _ in range(3))
    target = tuple(jnp.zeros((2, hours) + tuple(geometry.tmask.shape))
                   for _ in range(3))
    normalizers = BalancedNormalizers(1.0e-10, 1.0e-6, 1.0e-6)

    def energy(predicted, bands):
        return float(neuralgcm_ensemble_scores(
            predicted, target, geometry, normalizers, objective=OBJECTIVE,
            cutoff_bands=bands, spectrum_weight=0.0)[3])

    for field in (1, 2):           # rotational u, rotational v
        pattern = 1.0e-3 * jnp.stack([
            _mode(geometry, 40, 0), _mode(geometry, 0, 40)])[field - 1]
        pattern = pattern * (geometry.umask if field == 1 else geometry.vmask)
        late = jnp.where(jnp.asarray([False, False, True, True])
                         [None, None, :, None, None], pattern, 0.0)
        increment = jnp.broadcast_to(late, shape)
        predicted = tuple(m + (increment if i == field else 0.0)
                          for i, m in enumerate(members))
        assert energy(predicted, full) > 0.0
        assert energy(predicted, cutoff) < 0.1 * energy(predicted, full)
    # ... and the vorticity channel through a planted vorticity-only change
    # (the balanced u and v carry it too, but the filter must see it as well).
    pattern = _mode(geometry, 40, 0) * geometry.umask
    increment = jnp.broadcast_to(jnp.where(
        jnp.asarray([False, False, True, True])[None, None, :, None, None],
        1.0e-3 * pattern, 0.0), shape)
    predicted = (members[0], members[1] + increment, members[2])
    assert energy(predicted, cutoff) < 0.1 * energy(predicted, full)


# ------------------------------------------------------------ spectrum term

def test_ke_spectra_are_the_scorers_spectra_of_the_cell_centred_velocity(case):
    _, geometry, _ = case
    rng = np.random.default_rng(9)
    shape = (2, 3) + tuple(geometry.tmask.shape)
    trajectory = BalancedTrajectory(
        jnp.zeros(shape),
        jnp.asarray(rng.standard_normal(shape)) * geometry.umask,
        jnp.asarray(rng.standard_normal(shape)) * geometry.vmask)
    spectra = swg_online.kinetic_energy_spectra(trajectory, geometry)
    j0, j1, i0, i1 = geometry.wet_bounds
    assert spectra.shape == (2, 3, max(j1 - j0 + 1, i1 - i0 + 1) // 2)
    for window in range(2):
        for hour in range(3):
            u = np.asarray(trajectory.rotational_u_m_s[window, hour])
            v = np.asarray(trajectory.rotational_v_m_s[window, hour])
            u_c, v_c = velocity_to_cell_centre(
                jnp.concatenate([jnp.zeros_like(u[:, :1]), u], axis=1),
                jnp.concatenate([jnp.zeros_like(v[:1]), v], axis=0))
            _, expected = isotropic_energy_spectrum(
                u_c[j0:j1 + 1, i0:i1 + 1], v_c[j0:j1 + 1, i0:i1 + 1],
                geometry.spacing_m)
            np.testing.assert_allclose(
                spectra[window, hour], expected, rtol=1.0e-12, atol=0.0)


def test_spectrum_term_is_zero_for_equal_fields_and_positive_for_a_planted_change(
        case):
    _, geometry, eq7 = case
    rng = np.random.default_rng(11)
    target = _native(geometry, rng)
    hours = 8
    bands = np.full(hours, 1000.0)

    def terms(predicted, cutoff=bands):
        return [float(x) for x in neuralgcm_trajectory_loss(
            predicted, target, geometry, objective=OBJECTIVE,
            cutoff_bands=cutoff, spectrum_weight=1.0, bias_weight=1.0, **eq7)]

    assert terms(target)[2] == 0.0
    # A planted spectral change: one high-k mode added to U on two records.
    planted = _with_u(target, _u_pattern(geometry, (1, 2), hours))
    assert terms(planted)[2] > 0.0
    # The spectrum acts on the unfiltered velocity: a low cutoff leaves it as is.
    assert terms(planted, np.full(hours, 2.0))[2] == terms(planted)[2]
    # Finite gradient at identical fields (no 0/0 in the spectrum).
    grads = jax.grad(lambda u: neuralgcm_trajectory_loss(
        (target[0], u, target[2]), target, geometry, objective=OBJECTIVE,
        cutoff_bands=bands, spectrum_weight=1.0, bias_weight=1.0,
        **eq7)[0])(target[1])
    assert np.all(np.isfinite(np.asarray(grads)))


def test_ensemble_spectrum_term_compares_every_member_with_the_truth(case):
    _, geometry, _ = case
    rng = np.random.default_rng(13)
    hours = 4
    truth = _native(geometry, rng, hours=hours)
    members = tuple(jnp.stack([x, x], axis=1) for x in truth)
    normalizers = BalancedNormalizers(1.0e-10, 1.0e-6, 1.0e-6)
    full = np.full(hours, 1000.0)

    def spectrum(predicted):
        return float(neuralgcm_ensemble_scores(
            predicted, truth, geometry, normalizers, objective=OBJECTIVE,
            cutoff_bands=full, spectrum_weight=1.0)[4])

    assert spectrum(members) == 0.0
    # Only the second member is wrong: the term is half of the all-wrong one.
    bump = _u_pattern(geometry, (1, 2), hours)
    one_wrong = (members[0], members[1].at[:, 1].add(bump), members[2])
    both_wrong = (members[0], members[1] + bump[:, None], members[2])
    assert spectrum(one_wrong) == pytest.approx(
        0.5 * spectrum(both_wrong), rel=1.0e-12)
    assert spectrum(one_wrong) > 0.0


# ---------------------------------------------------------------- bias term

def test_bias_term_is_zero_for_zero_mean_errors_and_positive_for_an_offset(case):
    _, geometry, eq7 = case
    shape = (2, 4) + tuple(geometry.tmask.shape)
    pattern = 1.0e-6 * _mode(geometry, 3, 5) * geometry.vertex_interior
    truth = jnp.zeros(shape)
    # +pattern in window 0, -pattern in window 1: the batch mean error is 0.
    alternating = truth.at[0].add(pattern).at[1].add(-pattern)
    bias = swg_online.vorticity_bias_term(alternating, truth, geometry, **eq7)
    point = swg_online.balanced_trajectory_loss(
        BalancedTrajectory(alternating, truth, truth),
        BalancedTrajectory(truth, truth, truth), geometry, None, kind="point",
        **eq7)
    assert float(point) > 0.0
    assert float(bias) < 1.0e-12 * float(point)
    # The same sign in both windows: a mean offset, equal to Eq. 7 here
    # (a constant error is its own mean) and never above it.
    offset = truth + pattern
    bias = swg_online.vorticity_bias_term(offset, truth, geometry, **eq7)
    point = swg_online.balanced_trajectory_loss(
        BalancedTrajectory(offset, truth, truth),
        BalancedTrajectory(truth, truth, truth), geometry, None, kind="point",
        **eq7)
    assert float(bias) > 0.0
    assert float(bias) == pytest.approx(float(point), rel=1.0e-12)
    # Varying error: bias <= Eq. 7 (Jensen), strictly here.
    rng = np.random.default_rng(15)
    noisy = offset + 1.0e-6 * jnp.asarray(
        rng.standard_normal(shape)) * geometry.vertex_interior
    bias = swg_online.vorticity_bias_term(noisy, truth, geometry, **eq7)
    point = swg_online.balanced_trajectory_loss(
        BalancedTrajectory(noisy, truth, truth),
        BalancedTrajectory(truth, truth, truth), geometry, None, kind="point",
        **eq7)
    assert 0.0 < float(bias) < float(point)


def test_bias_term_acts_on_the_lead_filtered_vorticity(case):
    _, geometry, eq7 = case
    hours = 8
    table = {"lead_hours": [0, 12], "cutoff_band": [1000, 6]}
    cutoff = lead_time_cutoff_bands(table, n_records=hours,
                                    target_interval_hours=3)
    zeros = tuple(jnp.zeros((2, hours) + tuple(geometry.tmask.shape))
                  for _ in range(3))
    late = _u_pattern(geometry, range(3, 8), hours)       # same sign: a mean
    kwargs = dict(objective=OBJECTIVE, spectrum_weight=0.0, bias_weight=1.0,
                  **eq7)
    full_bias = float(neuralgcm_trajectory_loss(
        _with_u(zeros, late), zeros, geometry,
        cutoff_bands=np.full(hours, 1000.0), **kwargs)[3])
    cut_bias = float(neuralgcm_trajectory_loss(
        _with_u(zeros, late), zeros, geometry, cutoff_bands=cutoff,
        **kwargs)[3])
    assert full_bias > 0.0 and cut_bias < 0.05 * full_bias


# -------------------------------------------------------- AD and jit parity

def test_trajectory_loss_gradient_is_finite_matches_finite_differences_and_jit(
        case):
    _, geometry, eq7 = case
    rng = np.random.default_rng(17)
    predicted, target = _native(geometry, rng, hours=4), _native(
        geometry, rng, hours=4)
    bands = np.array([1000.0, 1000.0, 20.0, 6.0])

    def loss(u):
        return neuralgcm_trajectory_loss(
            (predicted[0], u, predicted[2]), target, geometry,
            objective=OBJECTIVE, cutoff_bands=bands, spectrum_weight=1.0e3,
            bias_weight=1.0, **eq7)[0]

    value, grad = jax.value_and_grad(loss)(predicted[1])
    assert np.all(np.isfinite(np.asarray(grad))) and np.any(np.asarray(grad))
    assert float(jax.jit(loss)(predicted[1])) == pytest.approx(
        float(value), rel=1.0e-12)
    direction = jnp.asarray(rng.standard_normal(predicted[1].shape)) \
        * geometry.umask
    eps = 1.0e-6          # the spectrum term is quartic in u: small step
    fd = (loss(predicted[1] + eps * direction)
          - loss(predicted[1] - eps * direction)) / (2.0 * eps)
    assert float(jnp.sum(grad * direction)) == pytest.approx(
        float(fd), rel=1.0e-6)
