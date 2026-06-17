"""Unit tests for :mod:`legoesm.training.compare_reanalysis`.

Pins the Stage 2 orchestration: model-vs-reference comparison on aligned
grid+sigma states, the precip-both-or-neither rule, pure-sigma pressure
construction, the SegmentCarry adapter, accumulated-precip conversion, and the
worst-column manifest emerging from a synthetic biased column.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.compare_reanalysis import (
    ColumnState,
    build_pressure_from_sigma,
    column_state_from_carry,
    column_state_from_hydrostatic,
    compare_state_to_reference,
    precip_mm_day_from_accum,
)


def _uniform_state(shape, nlev, *, T=250.0, q=1e-3, u=5.0, v=0.0, p_s=1.0e5,
                   precip=None, sst=None):
    full = shape + (nlev,)
    return ColumnState(
        T=jnp.full(full, T),
        q_v=jnp.full(full, q),
        u=jnp.full(full, u),
        v=jnp.full(full, v),
        p_s=jnp.full(shape, p_s),
        precip_mm_day=None if precip is None else jnp.full(shape, precip),
        sst_K=None if sst is None else jnp.full(shape, sst),
    )


def _sigma(nlev):
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    return sigma_full, sigma_half


def test_precip_accum_conversion():
    # 2 kg/m2 over 12 h -> 2 mm / 0.5 day = 4 mm/day
    rate = float(precip_mm_day_from_accum(jnp.asarray(2.0), 0.5 * 86400.0))
    assert rate == pytest.approx(4.0, abs=1e-9)


def test_build_pressure_from_sigma():
    nlev = 4
    sigma_full, sigma_half = _sigma(nlev)
    p_s = jnp.full((2, 3), 1.0e5)
    p_full, p_half = build_pressure_from_sigma(p_s, sigma_full, sigma_half)
    assert p_full.shape == (2, 3, nlev)
    assert p_half.shape == (2, 3, nlev + 1)
    assert float(p_half[0, 0, -1]) == pytest.approx(1.0e5)  # surface
    assert float(p_half[0, 0, 0]) == pytest.approx(0.0)  # model top


def test_compare_identical_states_zero_score():
    nlev = 5
    shape = (2, 3)
    sigma_full, sigma_half = _sigma(nlev)
    state = _uniform_state(shape, nlev, sst=300.0)
    lat = jnp.array([10.0, 20.0])
    lon = jnp.array([100.0, 110.0, 120.0])
    result = compare_state_to_reference(
        model=state, reference=state,
        sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, time_index=0, n_worst=2,
    )
    assert result.error_fields.combined_score.shape == shape
    assert float(jnp.max(result.error_fields.combined_score)) == pytest.approx(
        0.0, abs=1e-10
    )
    # Environment computed from the model column.
    assert result.environment.sst_K.shape == shape
    assert float(jnp.max(jnp.abs(result.environment.sst_K - 300.0))) == 0.0


def test_compare_flags_biased_column():
    nlev = 5
    shape = (2, 2)
    sigma_full, sigma_half = _sigma(nlev)
    model = _uniform_state(shape, nlev, T=250.0, sst=300.0)
    # Reference equals model except one column is much colder in the model.
    T_ref = np.full(shape + (nlev,), 250.0)
    T_model = np.full(shape + (nlev,), 250.0)
    T_model[1, 1] = 240.0  # 10 K cold bias in column (1,1)
    model = model._replace(T=jnp.asarray(T_model))
    reference = _uniform_state(shape, nlev, T=250.0)._replace(
        T=jnp.asarray(T_ref)
    )
    lat = jnp.array([0.0, 1.0])
    lon = jnp.array([0.0, 1.0])
    result = compare_state_to_reference(
        model=model, reference=reference,
        sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, time_index=4, n_worst=1,
    )
    assert len(result.manifest) == 1
    rec = result.manifest[0]
    assert rec.grid_index == (1, 1)
    assert rec.time_index == 4
    assert rec.T_rmse_K == pytest.approx(10.0, abs=1e-6)


def test_precip_dropped_when_only_one_side_has_it():
    nlev = 4
    shape = (2, 2)
    sigma_full, sigma_half = _sigma(nlev)
    model = _uniform_state(shape, nlev, precip=5.0, sst=300.0)
    reference = _uniform_state(shape, nlev, precip=None)  # ERA5 precip missing
    lat = jnp.array([0.0, 1.0])
    lon = jnp.array([0.0, 1.0])
    # Must NOT raise (the orchestration drops precip rather than passing one).
    result = compare_state_to_reference(
        model=model, reference=reference,
        sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, time_index=0, n_worst=1,
    )
    assert float(jnp.max(jnp.abs(result.error_fields.precip_err_mm_day))) == 0.0


def test_misaligned_states_raise():
    sigma_full, sigma_half = _sigma(4)
    model = _uniform_state((2, 2), 4)
    reference = _uniform_state((2, 2), 5)  # different nlev
    with pytest.raises(ValueError):
        compare_state_to_reference(
            model=model, reference=reference,
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2),
            time_index=0, n_worst=1,
        )


def test_bad_sigma_length_raises():
    model = _uniform_state((2, 2), 4)
    bad_sigma_full = jnp.linspace(0.1, 0.9, 5)  # wrong length
    bad_sigma_half = jnp.linspace(0.0, 1.0, 6)
    with pytest.raises(ValueError):
        compare_state_to_reference(
            model=model, reference=model,
            sigma_full=bad_sigma_full, sigma_half=bad_sigma_half,
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2),
            time_index=0, n_worst=1,
        )


def test_reversed_sigma_raises():
    """A reversed (surface-to-top) sigma_half must fail loudly, not corrupt."""
    nlev = 4
    sigma_full, sigma_half = _sigma(nlev)
    model = _uniform_state((2, 2), nlev)
    with pytest.raises(ValueError, match="increasing"):
        compare_state_to_reference(
            model=model, reference=model,
            sigma_full=sigma_full[::-1], sigma_half=sigma_half[::-1],
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2),
            time_index=0, n_worst=1,
        )


def test_partial_pressure_override_raises():
    nlev = 4
    sigma_full, sigma_half = _sigma(nlev)
    model = _uniform_state((2, 2), nlev)
    p_full = jnp.full((2, 2, nlev), 5.0e4)
    with pytest.raises(ValueError, match="both p_full and p_half"):
        compare_state_to_reference(
            model=model, reference=model,
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2),
            time_index=0, n_worst=1, p_full=p_full,  # p_half missing
        )


def test_reversed_pressure_override_raises():
    nlev = 4
    sigma_full, sigma_half = _sigma(nlev)
    model = _uniform_state((2, 2), nlev)
    p_full = jnp.broadcast_to(
        jnp.linspace(2.0e4, 9.0e4, nlev), (2, 2, nlev)
    )
    # Reversed half-pressure (surface-first) — correct shape, wrong orientation.
    p_half = jnp.broadcast_to(
        jnp.linspace(1.0e5, 0.0, nlev + 1), (2, 2, nlev + 1)
    )
    with pytest.raises(ValueError, match="monotonic"):
        compare_state_to_reference(
            model=model, reference=model,
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2),
            time_index=0, n_worst=1, p_full=p_full, p_half=p_half,
        )


def test_mismatched_surface_field_shape_raises():
    nlev = 4
    sigma_full, sigma_half = _sigma(nlev)
    model = _uniform_state((2, 2), nlev)
    bad = model._replace(p_s=jnp.full((2, 3), 1.0e5))  # wrong column shape
    with pytest.raises(ValueError, match="column shape"):
        compare_state_to_reference(
            model=bad, reference=_uniform_state((2, 2), nlev),
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2),
            time_index=0, n_worst=1,
        )


def test_nonfinite_score_not_selected_as_worst():
    """A NaN combined score must not be ranked as the worst column."""
    from legoesm.training.column_era5_metrics import rank_worst_columns

    score = jnp.array([0.5, jnp.nan, 0.3])
    idx, vals, valid = rank_worst_columns(score, 1)
    assert int(idx[0]) == 0  # 0.5 is worst finite; NaN routed to -inf
    assert bool(valid[0])


class _FakeCarry:
    """Minimal SegmentCarry-like duck for the adapter test."""

    def __init__(self, shape, nlev):
        full = shape + (nlev,)
        self.T = jnp.full(full, 250.0)
        self.q_v = jnp.full(full, 1e-3)
        self.u = jnp.full(full, 5.0)
        self.v = jnp.zeros(full)
        self.p_s = jnp.full(shape, 1.0e5)


def test_column_state_from_carry():
    carry = _FakeCarry((2, 3), 4)
    state = column_state_from_carry(
        carry, sst_K=jnp.full((2, 3), 301.0),
        precip_mm_day=jnp.full((2, 3), 2.0),
    )
    assert state.T.shape == (2, 3, 4)
    assert float(state.sst_K[0, 0]) == pytest.approx(301.0)
    assert float(state.precip_mm_day[0, 0]) == pytest.approx(2.0)
    assert state.p_s.shape == (2, 3)


def _hydro_state(shape, nlev):
    """A REAL driver HydrostaticState (Field-wrapped leaves; q_v separate)."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    full = shape + (nlev,)

    def _f(x):
        return Field(data=jnp.asarray(x))

    return HydrostaticState(
        u=_f(jnp.full(full, 5.0)), v=_f(jnp.zeros(full)),
        T=_f(jnp.full(full, 250.0)), p_s=_f(jnp.full(shape, 1.0e5)),
        phis=_f(jnp.zeros(shape)),
    )


def test_column_state_from_hydrostatic_unwraps_fields():
    """The driver state fields are Field wrappers — they must be unwrapped to
    raw arrays for the comparison."""
    shape, nlev = (2, 3), 5
    atm = _hydro_state(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 4e-3)
    cs = column_state_from_hydrostatic(atm, q_v, sst_K=jnp.full(shape, 300.0))
    # Raw arrays, not Field wrappers.
    for arr in (cs.T, cs.u, cs.v, cs.p_s):
        assert isinstance(arr, jnp.ndarray)
        assert not hasattr(arr, "data")
    assert cs.T.shape == shape + (nlev,)
    assert cs.p_s.shape == shape


def test_column_state_from_hydrostatic_amip_and_cmip():
    """One adapter feeds the compare for BOTH run modes; q_v is separate, only
    the SST source differs (prescribed for AMIP, coupled-ocean for CMIP) — and
    that difference propagates to the environment tag."""
    shape, nlev = (2, 3), 5
    atm = _hydro_state(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 4e-3)
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    ref = column_state_from_hydrostatic(atm, q_v)
    lat = jnp.array([10.0, 20.0])
    lon = jnp.array([100.0, 110.0, 120.0])

    for sst_value in (300.0, 298.5):  # AMIP prescribed vs CMIP coupled SST
        cs = column_state_from_hydrostatic(
            atm, q_v, sst_K=jnp.full(shape, sst_value))
        comp = compare_state_to_reference(
            model=cs, reference=ref, sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=lat, lon_deg=lon, time_index=0, n_worst=2,
        )
        # Identical atmosphere ⇒ zero combined score regardless of SST source.
        assert float(jnp.max(comp.error_fields.combined_score)) == pytest.approx(
            0.0, abs=1e-10)
        # ...but the SST environment tag reflects the per-mode SST.
        assert float(comp.environment.sst_K[0, 0]) == pytest.approx(sst_value)


def test_column_state_from_hydrostatic_sst_optional():
    atm = _hydro_state((2, 2), 4)
    q_v = jnp.full((2, 2, 4), 1e-3)
    cs = column_state_from_hydrostatic(atm, q_v)  # no SST (CMIP can omit)
    assert cs.sst_K is None
    assert cs.precip_mm_day is None


def test_column_state_from_hydrostatic_rejects_none_v():
    atm = _hydro_state((2, 2), 4)._replace(v=None)  # MPAS edge-velocity state
    with pytest.raises(ValueError, match="atm_state.v is None"):
        column_state_from_hydrostatic(atm, jnp.zeros((2, 2, 4)))
