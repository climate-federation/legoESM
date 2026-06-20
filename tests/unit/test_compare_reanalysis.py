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
    ReferenceBoundsConfig,
    assert_model_state_finite,
    build_pressure_from_sigma,
    column_state_from_carry,
    column_state_from_hydrostatic,
    compare_state_to_reference,
    owned_cell_valid_mask,
    precip_mm_day_from_accum,
    validate_reference_physical,
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


def test_validate_reference_physical_accepts_a_plausible_state():
    """A physically-plausible reference (K, Pa, kg/kg) passes (iter 99)."""
    ok = _uniform_state((4,), 3, T=285.0, q=6e-3, u=10.0, p_s=1.0e5, sst=290.0,
                        precip=3.0)
    assert validate_reference_physical(ok) is None


def test_validate_reference_physical_catches_celsius_temperature():
    """T in °C (≈ 12, far below 150 K) is caught with a units hint (iter 99)."""
    bad = _uniform_state((4,), 3, T=12.0)            # °C, not K
    with pytest.raises(ValueError, match=r"T outside.*K \(°C input\?\)"):
        validate_reference_physical(bad)


def test_validate_reference_physical_catches_hpa_surface_pressure():
    """p_s in hPa (≈ 1013, far below 3e4 Pa) is caught (iter 99)."""
    bad = _uniform_state((4,), 3, T=285.0, p_s=1013.0)   # hPa, not Pa
    with pytest.raises(ValueError, match=r"p_s outside.*Pa \(hPa input\?\)"):
        validate_reference_physical(bad)


def test_validate_reference_physical_catches_g_per_kg_humidity():
    """q_v in g/kg (≈ 8, far above 0.1 kg/kg) is caught (iter 99)."""
    bad = _uniform_state((4,), 3, T=285.0, q=8.0)    # g/kg, not kg/kg
    with pytest.raises(ValueError, match=r"q_v outside.*kg/kg \(g/kg input\?\)"):
        validate_reference_physical(bad)


def test_validate_reference_physical_catches_out_of_range_sst():
    """An out-of-range reference SST is caught (iter 282) — the SST-bounds check the
    other validate tests (T/p_s/q_v) don't cover.  This is the safety net for the
    ``era5_to_state`` loader's ZERO-FILL of a MISSING ERA5 skin_temperature: a 0 K SST
    (far below sst_min 250 K) fails LOUD here instead of a 0 K env tag silently driving
    the comparison.  Also catches a Celsius SST (20 °C read as 20 K)."""
    zero = _uniform_state((4,), 3, T=285.0, sst=0.0)       # the era5 zero-fill (no SST)
    with pytest.raises(ValueError, match=r"sst_K outside.*K"):
        validate_reference_physical(zero)
    celsius = _uniform_state((4,), 3, T=285.0, sst=20.0)   # 20 °C as 20 K, << 250 K
    with pytest.raises(ValueError, match=r"sst_K outside.*K"):
        validate_reference_physical(celsius)
    # a plausible SST (290 K) passes — the bound is not over-tight.
    assert validate_reference_physical(
        _uniform_state((4,), 3, T=285.0, sst=290.0)) is None


def test_validate_reference_physical_catches_absurd_wind():
    """A gross wind units/loader error (|u| or |v| far above the generous 200 m/s bound)
    is caught — the u/v check the other validate tests don't cover (iter 283)."""
    for bad in (_uniform_state((4,), 3, T=285.0, u=900.0),     # 900 m/s — a units bug
                _uniform_state((4,), 3, T=285.0, v=-900.0)):   # the v axis too
        with pytest.raises(ValueError, match=r"[uv] outside"):
            validate_reference_physical(bad)


def test_validate_reference_physical_catches_out_of_range_precip():
    """An out-of-range reference precip is caught — a NEGATIVE rate (a -999 fill / sign
    bug) or an absurd rate (mm/hr-as-mm/day or an accumulation, not a rate) — the precip
    check the other validate tests don't cover (iter 283)."""
    neg = _uniform_state((4,), 3, T=285.0, precip=-999.0)      # a fill-value / sign bug
    with pytest.raises(ValueError, match="precip_mm_day outside"):
        validate_reference_physical(neg)
    absurd = _uniform_state((4,), 3, T=285.0, precip=5000.0)   # >> 2000 mm/day
    with pytest.raises(ValueError, match="precip_mm_day outside"):
        validate_reference_physical(absurd)
    assert validate_reference_physical(                        # a plausible 3 mm/day passes
        _uniform_state((4,), 3, T=285.0, precip=3.0)) is None


def test_validate_reference_physical_catches_negative_humidity():
    """ERA5 q is non-negative — a meaningfully negative q_v (a fill/sign/loader bug)
    is rejected (the floor is a roundoff tolerance, NOT the model overshoot one;
    iter 99 Codex)."""
    bad = _uniform_state((4,), 3, T=285.0, q=-5.0e-4)
    with pytest.raises(ValueError, match="q_v outside"):
        validate_reference_physical(bad)


def test_validate_reference_physical_catches_non_finite():
    """A non-finite reference value is rejected before it can poison the bias."""
    base = _uniform_state((4,), 3, T=285.0)
    bad = base._replace(T=base.T.at[0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="non-finite"):
        validate_reference_physical(bad)


def test_validate_reference_physical_skips_absent_optional_fields():
    """precip/sst absent (None) ⇒ not checked; the present fields still validate."""
    ok = _uniform_state((4,), 3, T=285.0, q=6e-3, p_s=1.0e5)   # no precip/sst
    assert ok.precip_mm_day is None and ok.sst_K is None
    assert validate_reference_physical(ok) is None


def test_validate_reference_physical_respects_custom_bounds():
    """Custom bounds tighten/loosen the gate — a value legal by default fails a
    tightened bound, proving the config (not a hardcoded literal) drives it."""
    state = _uniform_state((4,), 3, T=285.0, p_s=1.0e5)
    tight = ReferenceBoundsConfig(T_max_K=284.0)     # 285 now out of range
    with pytest.raises(ValueError, match="T outside"):
        validate_reference_physical(state, bounds=tight)


def test_assert_model_state_finite_accepts_a_finite_run():
    """A finite model run passes — no false positive on the common case (iter 301)."""
    ok = _uniform_state((4,), 3, T=285.0, q=6e-3, u=10.0, sst=290.0, precip=3.0)
    assert assert_model_state_finite(ok) is None


def test_assert_model_state_finite_catches_a_diverged_run():
    """A NaN in the model T (a blown-up run) raises with the DIVERGED hint + the
    non-finite count — so a diverged BASELINE fails loud instead of flowing a garbage
    bias into the campaign and wasting a multi-day HPC run (iter 301). Finiteness ONLY:
    no physical-range check (the model's units are correct by construction)."""
    bad = _uniform_state((4,), 3)._replace(
        T=jnp.full((4, 3), 285.0).at[0, 0].set(jnp.nan))
    with pytest.raises(ValueError,
                       match=r"baseline model run\.T has 1/12 non-finite.*DIVERGED"):
        assert_model_state_finite(bad, name="baseline model run")


def test_assert_model_state_finite_checks_optional_fields_and_skips_absent():
    """An inf in an OPTIONAL present field (sst_K) is caught; absent optionals (sst/precip
    None) are skipped — symmetry with validate_reference_physical's optional handling."""
    bad = _uniform_state((4,), 3, sst=290.0)._replace(
        sst_K=jnp.full((4,), 290.0).at[1].set(jnp.inf))
    with pytest.raises(ValueError, match=r"sst_K has 1/4 non-finite"):
        assert_model_state_finite(bad)
    # absent sst/precip ⇒ not checked (no spurious raise on the no-optional state).
    plain = _uniform_state((4,), 3)
    assert plain.sst_K is None and plain.precip_mm_day is None
    assert assert_model_state_finite(plain) is None


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


def test_compare_sst_fallback_uses_surface_air_temperature_when_sst_absent():
    """In CMIP/uncoupled runs the driver may not thread an SST; the env SST tag
    then falls back to the SURFACE-air temperature ``model.T[..., -1]``.

    The documented fallback (`compare_reanalysis.py`) is a LIVE branch — every
    other compare test passes an explicit ``sst``, so the ``sst_K is None`` path is
    otherwise unexercised.  A bug there (using ``T[..., 0]`` = the model TOP, or
    crashing on ``None``) would silently mis-tag the CMIP environment SST and so
    mis-cluster / mis-deploy by climate.  A NON-uniform T profile (surface 295 K,
    top 220 K) makes the surface-vs-top choice decisive: the tag must be 295, the
    surface level, NOT 220, the top.
    """
    nlev = 5
    shape = (2, 2)
    sigma_full, sigma_half = _sigma(nlev)
    # Surface-last profile: T[..., 0] = 220 K (top), T[..., -1] = 295 K (surface).
    t_profile = jnp.linspace(220.0, 295.0, nlev)
    T = jnp.broadcast_to(t_profile, shape + (nlev,))
    model = _uniform_state(shape, nlev, sst=None)._replace(T=T)   # sst_K is None
    assert model.sst_K is None
    lat = jnp.array([0.0, 1.0])
    lon = jnp.array([0.0, 1.0])
    # The fallback WARNS (iter 280) so the operator learns the env tag is approximate.
    with pytest.warns(UserWarning, match="model.sst_K is None"):
        result = compare_state_to_reference(
            model=model, reference=model,                        # identical → zero score
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=lat, lon_deg=lon, time_index=0, n_worst=1,
        )
    # The fallback used the SURFACE-air temperature (295 K), not the top (220 K).
    assert result.environment.sst_K.shape == shape
    np.testing.assert_allclose(np.asarray(result.environment.sst_K), 295.0, atol=1e-5)
    assert result.manifest[0].environment.sst_K == pytest.approx(295.0, abs=1e-5)


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
    # have_precip is the precip-availability flag the campaign threads to the
    # per-variable bias (so a not-compared precip reads as NaN, not a spurious 0).
    assert result.have_precip is False
    # When BOTH states carry precip, the flag flips True (precip enters the score).
    ref_with_precip = _uniform_state(shape, nlev, precip=2.0)
    both = compare_state_to_reference(
        model=model, reference=ref_with_precip,
        sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, time_index=0, n_worst=1,
    )
    assert both.have_precip is True


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


def test_column_state_from_hydrostatic_rejects_sst_on_a_different_grid():
    """A coupled (CMIP) run with an ocean grid DIFFERENT from the atmosphere
    (make_base_driver_builder ocean_grid=...) yields ocean_state.T_sfc on the OCEAN
    grid.  The SST environment tag indexes by ATMOSPHERE column flat-index
    (column_manifest: sst[flat_i]), so a differently-shaped SST would silently misalign
    every column's SST (or index out of bounds).  It must fail LOUD at the bridge — like
    the MPAS edge/cell mesh checks — not silently corrupt the env tag.  (Non-vacuous: a
    grid-matched SST, AMIP prescribed or same-grid CMIP, still passes — covered above.)"""
    shape, nlev = (2, 3), 5
    atm = _hydro_state(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 4e-3)
    ocean_grid_sst = jnp.full((4, 5), 295.0)   # a DIFFERENT (ocean) grid shape than (2, 3)
    with pytest.raises(ValueError, match=r"sst_K shape .* != the atmosphere column grid"):
        column_state_from_hydrostatic(atm, q_v, sst_K=ocean_grid_sst)


def test_column_state_from_hydrostatic_rejects_precip_on_a_different_grid():
    """``precip_mm_day`` is the sibling optional per-column SURFACE field; it enters the
    precip SCORE term indexed by the same atmosphere column flat-index, so a wrong-grid
    precip has the IDENTICAL misalignment hazard as a wrong-grid SST and must fail LOUD
    the same way (the guard covers BOTH fields, not just SST)."""
    shape, nlev = (2, 3), 5
    atm = _hydro_state(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 4e-3)
    wrong_grid_precip = jnp.full((4, 5), 3.0)   # a DIFFERENT grid shape than (2, 3)
    with pytest.raises(ValueError,
                       match=r"precip_mm_day shape .* != the atmosphere column grid"):
        column_state_from_hydrostatic(atm, q_v, precip_mm_day=wrong_grid_precip)


def test_column_state_from_hydrostatic_accepts_both_grid_matched_surface_fields():
    """Positive complement to the wrong-grid guards (iters 329/330): a grid-matched sst_K
    (here Field-WRAPPED, exercising the shared `_checked_surface` ``_arr`` unwrap) AND a
    grid-matched precip_mm_day pass through TOGETHER and land as RAW arrays in the
    ColumnState — the guard must not false-reject a legitimate per-column surface field
    (the CMIP-with-precip case), and must unwrap a Field leaf like the atm-state fields."""
    from legoesm.core.field import Field
    shape, nlev = (2, 3), 5
    atm = _hydro_state(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 4e-3)
    cs = column_state_from_hydrostatic(
        atm, q_v,
        sst_K=Field(data=jnp.full(shape, 295.0)),     # Field-wrapped → must unwrap
        precip_mm_day=jnp.full(shape, 2.0))           # raw, grid-matched
    assert cs.sst_K.shape == shape and cs.precip_mm_day.shape == shape
    for arr in (cs.sst_K, cs.precip_mm_day):
        assert isinstance(arr, jnp.ndarray) and not hasattr(arr, "data")   # unwrapped raw
    assert float(cs.sst_K[0, 0]) == pytest.approx(295.0)
    assert float(cs.precip_mm_day[1, 2]) == pytest.approx(2.0)


def test_column_state_from_hydrostatic_rejects_none_v():
    atm = _hydro_state((2, 2), 4)._replace(v=None)  # MPAS edge-velocity state
    with pytest.raises(ValueError, match="atm_state.v is None"):
        column_state_from_hydrostatic(atm, jnp.zeros((2, 2, 4)))


# --------------------------------------------------------------------------- #
# MPAS/Voronoi compare-side handoff (iter 74): an edge-velocity state is
# reconstructed to cell-centered geographic winds so an MPAS run is rankable.
# --------------------------------------------------------------------------- #
def _mpas_state(mesh, nlev, *, bias_cell=None, bias_dT=0.0):
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    # θ uniform across cells (via exner-free constant T) except an optional cold
    # bias at one cell; u is the EDGE-normal velocity, v is None (the MPAS marker).
    T = np.full((mesh.nCells, nlev), 250.0)
    if bias_cell is not None:
        T[bias_cell] = 250.0 + bias_dT
    u_edge = (np.asarray(mesh.angleEdge)[:, None]
              * np.ones((1, nlev)))            # some non-uniform edge field
    u_edge = 6.0 * np.cos(u_edge)
    state = HydrostaticState(
        u=Field(data=jnp.asarray(u_edge)),
        T=Field(data=jnp.asarray(T)),
        p_s=Field(data=jnp.full((mesh.nCells,), 1.0e5)),
        phis=Field(data=jnp.zeros((mesh.nCells,))),
        v=None,
    )
    q_v = jnp.full((mesh.nCells, nlev), 1.0e-3)
    return state, q_v


def test_mpas_edge_state_reconstructs_cell_wind():
    from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity
    mesh = create_voronoi_mesh(2)
    nlev = 4
    state, q_v = _mpas_state(mesh, nlev)
    cs = column_state_from_hydrostatic(state, q_v, mesh=mesh)
    # Cell-centered geographic winds, (nCells, nlev), matching the Perot diagnostic.
    assert cs.u.shape == (mesh.nCells, nlev) and cs.v.shape == (mesh.nCells, nlev)
    u_ref, v_ref = reconstruct_cell_velocity(
        jnp.asarray(state.u.data, dtype=cs.T.dtype), mesh)
    np.testing.assert_allclose(np.asarray(cs.u), np.asarray(u_ref), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(cs.v), np.asarray(v_ref), rtol=1e-5)
    assert cs.u.dtype == cs.T.dtype          # condition-4 dtype cast
    assert np.all(np.isfinite(np.asarray(cs.u)))
    # The NATIVE edge velocity is carried for the LES-forcing extractor (iter 76).
    assert cs.u_edge is not None and cs.u_edge.shape == (mesh.nEdges, nlev)
    np.testing.assert_array_equal(
        np.asarray(cs.u_edge), np.asarray(state.u.data))


def test_mpas_state_without_mesh_raises():
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(2)
    state, q_v = _mpas_state(mesh, 4)
    with pytest.raises(ValueError, match="pass mesh="):
        column_state_from_hydrostatic(state, q_v)             # mesh omitted


def test_mpas_state_wrong_mesh_type_raises():
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(2)
    state, q_v = _mpas_state(mesh, 4)
    with pytest.raises(ValueError, match="not a VoronoiMesh"):
        column_state_from_hydrostatic(state, q_v, mesh=object())


def test_mpas_state_mismatched_mesh_size_raises():
    """A DIFFERENT-size Voronoi mesh would reconstruct wrong winds silently —
    rejected by the nEdges/nCells shape guards (Codex)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh2, mesh3 = create_voronoi_mesh(2), create_voronoi_mesh(3)
    state, q_v = _mpas_state(mesh2, 4)              # built for the level-2 mesh
    with pytest.raises(ValueError, match="nEdges|nCells"):
        column_state_from_hydrostatic(state, q_v, mesh=mesh3)  # wrong mesh


def test_latlon_state_passthrough_ignores_mesh():
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    nlev = 4
    state = HydrostaticState(
        u=Field(data=jnp.full((2, 3, nlev), 7.0)),
        T=Field(data=jnp.full((2, 3, nlev), 260.0)),
        p_s=Field(data=jnp.full((2, 3), 1.0e5)),
        phis=Field(data=jnp.zeros((2, 3))),
        v=Field(data=jnp.full((2, 3, nlev), -2.0)),
    )
    q_v = jnp.full((2, 3, nlev), 2.0e-3)
    cs = column_state_from_hydrostatic(state, q_v, mesh=object())  # mesh ignored
    np.testing.assert_array_equal(np.asarray(cs.u), 7.0)
    np.testing.assert_array_equal(np.asarray(cs.v), -2.0)
    assert cs.u_edge is None          # cell-wind grid → no native edge velocity


def test_mpas_compare_end_to_end_flags_worst_cell():
    """The compare machinery is grid-AGNOSTIC over the 1-D cell axis: an MPAS
    model state (reconstructed cell winds) vs a synthetic reference on the SAME
    mesh ranks the biased cell — so an MPAS run is rankable (the ERA5->MPAS-cell
    REGRIDDER for real reference data is the documented next step)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(2)
    nlev = 5
    bias_cell = 37
    model_state, q_v = _mpas_state(mesh, nlev, bias_cell=bias_cell, bias_dT=-12.0)
    model = column_state_from_hydrostatic(model_state, q_v, mesh=mesh)
    # Synthetic reference: identical but UNBIASED (the regridder would supply ERA5).
    ref_state, q_v_ref = _mpas_state(mesh, nlev)
    reference = column_state_from_hydrostatic(ref_state, q_v_ref, mesh=mesh)
    sigma_full, sigma_half = _sigma(nlev)
    rad2deg = 180.0 / np.pi
    result = compare_state_to_reference(
        model=model, reference=reference,
        sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=jnp.asarray(mesh.latCell) * rad2deg,
        lon_deg=jnp.asarray(mesh.lonCell) * rad2deg,
        time_index=0, n_worst=1,
    )
    assert len(result.manifest) == 1
    assert result.manifest[0].grid_index == (bias_cell,)   # the biased cell ranked
    assert result.manifest[0].T_rmse_K == pytest.approx(12.0, abs=1e-4)


def test_owned_cell_valid_mask_combines_owned_and_base():
    """The owned-cell mask helper: owned-only, owned-AND-base, layout duck-typing,
    and a shape-mismatch guard (the rank-local cell axes MUST align)."""
    from types import SimpleNamespace

    owned = jnp.array([True, True, True, False, False])     # 3 owned, 2 halo
    # raw-mask form and the VoronoiPartitionLayout duck-typed form agree.
    np.testing.assert_array_equal(np.asarray(owned_cell_valid_mask(owned)),
                                  np.asarray(owned))
    layout = SimpleNamespace(owned_mask_cells=owned)
    np.testing.assert_array_equal(
        np.asarray(owned_cell_valid_mask(layout)), np.asarray(owned))
    # AND with a base (e.g. land) mask: rankable only if owned AND base-valid.
    base = jnp.array([True, False, True, True, True])
    np.testing.assert_array_equal(
        np.asarray(owned_cell_valid_mask(layout, base_mask=base)),
        np.array([True, False, True, False, False]))
    # Misaligned (different-length) base mask must raise, not truncate/broadcast.
    with pytest.raises(ValueError, match="must align EXACTLY"):
        owned_cell_valid_mask(owned, base_mask=jnp.array([True, False]))
    # A non-1-D owned mask must raise (the cell axis is strictly (n_local_cells,)).
    with pytest.raises(ValueError, match="must be 1-D"):
        owned_cell_valid_mask(owned.reshape(5, 1))
    # A same-length but wrong-RANK base ((n,1) vs (n,)) must raise (not broadcast).
    with pytest.raises(ValueError, match="must align EXACTLY"):
        owned_cell_valid_mask(owned, base_mask=base.reshape(5, 1))


def test_mpas_owned_cell_mask_excludes_halo_from_ranking():
    """NON-VACUOUS distributed-MPAS guard: a HALO cell with the globally-LARGEST
    bias is excluded from the worst-column ranking when the owned-cell mask is
    passed as ``valid_mask`` — without it, that halo cell would be (wrongly) ranked
    + spun off on a rank that does NOT own it.  With the mask, the worst OWNED cell
    is ranked instead."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(2)
    nlev = 5
    n_owned = 120                          # owned = cells [0, 120); halo = [120, 162)
    halo_worst = 150                       # a HALO cell, the globally-worst bias
    owned_worst = 50                       # an OWNED cell, the second-worst bias

    state, q_v = _mpas_state(mesh, nlev)
    T = np.asarray(state.T.data).copy()
    T[halo_worst] = 250.0 - 12.0           # biggest model-vs-ref diff (12 K)
    T[owned_worst] = 250.0 - 8.0           # second biggest (8 K), but OWNED
    from legoesm.core.field import Field
    model_state = state._replace(T=Field(data=jnp.asarray(T)))
    model = column_state_from_hydrostatic(model_state, q_v, mesh=mesh)
    ref_state, q_v_ref = _mpas_state(mesh, nlev)
    reference = column_state_from_hydrostatic(ref_state, q_v_ref, mesh=mesh)
    sigma_full, sigma_half = _sigma(nlev)
    rad2deg = 180.0 / np.pi
    owned_mask = owned_cell_valid_mask(jnp.arange(mesh.nCells) < n_owned)

    def _rank(valid_mask):
        return compare_state_to_reference(
            model=model, reference=reference,
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=jnp.asarray(mesh.latCell) * rad2deg,
            lon_deg=jnp.asarray(mesh.lonCell) * rad2deg,
            time_index=0, n_worst=1, valid_mask=valid_mask)

    # Control (NO mask): the halo cell IS ranked worst — proving the bias is real
    # and the mask below is what changes the outcome (non-vacuous).
    assert _rank(None).manifest[0].grid_index == (halo_worst,)
    # With the owned mask: the halo cell is excluded; the worst OWNED cell is ranked.
    masked = _rank(owned_mask)
    assert masked.manifest[0].grid_index == (owned_worst,)
    assert masked.manifest[0].T_rmse_K == pytest.approx(8.0, abs=1e-4)


@pytest.mark.xfail(strict=True, reason=(
    "Hybrid-coordinate compare bug (iter 337, CODEX PENDING): compare_state_to_reference "
    "weights the bias by the SIGMA-layer thickness (diff(sigma_half), p_s-independent) and "
    "IGNORES the p_full/p_half override that carries the true layer pressures. "
    "vertical_coord defaults to 'hybrid' (p = A*p_ref + B*p_s), so over terrain (p_s != "
    "p_ref) the per-level bias is mis-weighted by up to ~276%. Fix: derive the mass weights "
    "from the layer PRESSURE thickness diff(p_half) (per-column, axis=-1 normalized) — "
    "byte-identical for pure-sigma (p_s cancels) and correct for hybrid — and thread the "
    "model coordinate through make_compare_fn so the campaign supplies the hybrid pressures."))
def test_compare_mass_weights_follow_the_pressure_profile_not_sigma():
    """Regression target for the hybrid mass-weight bug: the bias mass weighting must respond
    to the LAYER PRESSURE thickness (the p_half override), not the sigma thickness.  Two
    p_half overrides that put very different mass in the biased TOP layer must yield DIFFERENT
    combined scores; currently they are EQUAL (the weights come from sigma_half, ignoring the
    override) — the bug.  xpasses (then drop the xfail) once the codex-reviewed fix lands."""
    shape, nlev = (1, 1), 3
    ref = _uniform_state(shape, nlev, T=250.0)
    model = ref._replace(T=ref.T.at[0, 0, 0].set(260.0))   # 10 K bias in the TOP layer only
    sigma_half = jnp.array([0.0, 1.0 / 3, 2.0 / 3, 1.0])
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    lat, lon = jnp.array([0.0]), jnp.array([0.0])

    def _score(p_half_1d):
        p_half = jnp.broadcast_to(p_half_1d, shape + (nlev + 1,))
        p_full = 0.5 * (p_half[..., 1:] + p_half[..., :-1])
        comp = compare_state_to_reference(
            model=model, reference=ref, sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=lat, lon_deg=lon, time_index=0, n_worst=1,
            p_full=p_full, p_half=p_half)
        return float(jnp.max(comp.error_fields.combined_score))

    # A: the biased top layer is THICK (lots of mass); B: it is THIN (little mass).
    score_thick_top = _score(jnp.array([0.0, 40000.0, 70000.0, 1.0e5]))
    score_thin_top = _score(jnp.array([0.0, 2000.0, 60000.0, 1.0e5]))
    assert abs(score_thick_top - score_thin_top) > 1e-6   # mass weighting must follow p_half
