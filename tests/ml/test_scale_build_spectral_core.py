"""WB scale trainer: the spectral (semi-implicit) training core (#817 blocker 1).

The `--training-core spectral` selector routes the WB training rollout through
the differentiable Gaussian/spectral PE core with the Hoskins–Simmons
semi-implicit step (#829's ``_make_spectral_integrator``) instead of the
explicit lat-lon C-grid — the design fix for the exploding training adjoint
(~x1.3/step on the explicit core) and the pole-cell dt collapse.

These tests validate the WIRING at a tiny truncation (T10/L8):
  * ``spectral_state_to_carry`` inverts ``carry_to_spectral_state`` (smooth
    fields round-trip; tracers exactly);
  * ``build_mode_components(training_core='spectral')`` returns an SI-enabled
    core whose ``.raw`` runs finite and is reverse-mode differentiable;
  * the YAML ``spectral:`` block drives the SpectralPEConfig (semi_implicit
    default True — the point of the core — and overridable).

The adjoint-damping DEMONSTRATION deliberately lives at the target fine grid
(the #829 T10 dt-sweep proved SI gives no benefit at coarse truncation where
the gravity-wave CFL never binds), so these tests assert wiring, not damping.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate


def _synthetic_carry(grid, nlev):
    """Smooth, balanced-ish SegmentCarry on the Gaussian grid (ERA5 stand-in)."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry

    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    lat = np.asarray(grid.lat)[:, None]                      # (n_lat, 1)
    shape_3d = (n_lat, n_lon, nlev)
    shape_2d = (n_lat, n_lon)

    u = jnp.asarray(np.broadcast_to(
        20.0 * np.cos(lat)[..., None] * np.linspace(0.4, 1.0, nlev),
        shape_3d).copy())
    v = jnp.zeros(shape_3d)
    T = jnp.asarray(np.broadcast_to(
        250.0 + 40.0 * np.linspace(0.0, 1.0, nlev), shape_3d).copy())
    p_s = jnp.full(shape_2d, 1.0e5)
    phis = jnp.zeros(shape_2d)
    q_v = jnp.asarray(np.broadcast_to(
        5e-3 * np.linspace(0.1, 1.0, nlev), shape_3d).copy())

    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    state = HydrostaticState(
        u=Field(u, name="u", dims=d3, units="m/s"),
        v=Field(v, name="v", dims=d3, units="m/s"),
        T=Field(T, name="T", dims=d3, units="K"),
        p_s=Field(p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(phis, name="phis", dims=d2, units="m2/s2"),
    )
    return pack_carry(
        state, q_v=q_v, q_c=jnp.zeros(shape_3d), q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


def _yml(n_max=10, nlev=8, **spectral_over):
    spec = {"n_max": n_max, "dt": 1800.0}
    spec.update(spectral_over)
    return {
        "n_lat": 32, "n_lon": 64, "nlev": nlev,
        "dt": 300.0,               # lat-lon core dt; spectral core ignores it
        "spectral": spec,
        "loss": {},
    }


def _cfg(mode="neural_gcm"):
    return SimpleNamespace(
        mode=mode, training_core="spectral", smoke=False,
        multi_step_hours=(6, 12),
    )


def test_spectral_state_to_carry_roundtrip():
    """carry -> spectral -> carry recovers the physical fields (SH round-trip
    on band-limited-ish smooth inputs) and passes tracers through exactly."""
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_state_to_carry,
    )

    nlev = 8
    grid = create_gaussian_grid(10)
    sigma = create_sigma_coordinate(nlev)
    carry = _synthetic_carry(grid, nlev)

    state = carry_to_spectral_state(carry, grid)
    back = spectral_state_to_carry(state, grid, sigma)

    # Smooth (large-scale) fields survive the SH analysis/synthesis round-trip
    # to a small relative tolerance (T10 truncation of already-smooth input).
    assert np.allclose(np.asarray(back.T), np.asarray(carry.T), atol=0.5)
    assert np.allclose(np.asarray(back.u), np.asarray(carry.u), atol=0.5)
    assert np.allclose(np.asarray(back.p_s), np.asarray(carry.p_s), rtol=1e-3)
    # Tracers ride the grid (no spectral truncation) -> exact.
    assert np.array_equal(np.asarray(back.q_v), np.asarray(carry.q_v))
    assert np.all(np.isfinite(np.asarray(back.v)))


def test_spectral_pe_config_defaults_and_overrides():
    """The YAML spectral block drives SpectralPEConfig; semi_implicit defaults
    True (the adjoint-bounding point of the core) and is overridable."""
    from legoesm.training.scale_build import _spectral_pe_config

    pe = _spectral_pe_config(_yml())
    assert pe.semi_implicit is True
    assert pe.hyperdiff_coeff == 2.5e15 and pe.spectral_filter_strength == 0.01

    pe_off = _spectral_pe_config(_yml(semi_implicit=False, si_substeps=2))
    assert pe_off.semi_implicit is False and pe_off.si_substeps == 2


def test_spectral_pe_config_forwards_conservation_knobs():
    """The WB spectral core reaches the SAME conserved-quantity constraints as
    the AMIP/AIMIP lane: dry-mass anchor OFF by default (existing WB runs are
    byte-identical) but switchable from YAML, energy-conserving vertical
    transport + frictional heating ON by default and overridable."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.scale_build import _spectral_pe_config

    pe = _spectral_pe_config(_yml())
    assert pe.fix_mass is False and pe.anchor_mass_to_initial is False
    assert pe.vertical_advection_scheme == "sb_centered"
    assert pe.frictional_heating is True
    # Defaults track SpectralPEConfig, not a copy of its values.
    assert (pe.fix_mass, pe.vertical_advection_scheme) == (
        SpectralPEConfig().fix_mass,
        SpectralPEConfig().vertical_advection_scheme,
    )

    pe_on = _spectral_pe_config(_yml(
        fix_mass=True, anchor_mass_to_initial=True,
        vertical_advection_scheme="upwind", frictional_heating=False))
    assert pe_on.fix_mass is True and pe_on.anchor_mass_to_initial is True
    assert pe_on.vertical_advection_scheme == "upwind"
    assert pe_on.frictional_heating is False


@pytest.mark.parametrize("mode", ["neural_gcm", "physics"])
def test_spectral_core_raw_runs_finite_and_differentiable(mode):
    """build_mode_components(training_core='spectral'): the returned
    ``make_run_seg(params).raw(ic, n_steps, forcing)`` integrates the SI
    spectral core to a finite state AND reverse-mode differentiates to finite
    gradients — the two properties the exploding-adjoint blocker denies the
    explicit lat-lon core at scale."""
    import equinox as eqx

    from legoesm.training.losses import combined_loss
    from legoesm.training.scale_build import build_mode_components

    yml = _yml()
    cfg = _cfg(mode)
    model, grid, sigma, params, make_run_seg, loss_config, dt = (
        build_mode_components(cfg, yml))
    assert dt == 1800.0                       # spectral dt, no pole clamp
    assert int(grid.n_max) == 10

    nlev = int(yml["nlev"])
    ic = _synthetic_carry(grid, nlev)
    target = _synthetic_carry(grid, nlev)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    forcing = {
        "T_sfc": jnp.full((ncol,), 300.0),
        "sic": jnp.zeros((ncol,)),
        "day_of_year": jnp.asarray(80.0),
        "seconds_of_day": jnp.asarray(0.0),
    }
    n_steps = 3

    # Forward: finite rolled-out physical fields.
    pred = make_run_seg(params).raw(ic, n_steps, forcing)
    for name in ("u", "v", "T", "p_s", "q_v"):
        assert bool(jnp.all(jnp.isfinite(getattr(pred, name)))), name

    # Reverse mode: finite loss and finite, not-all-zero gradients.
    sigma_full = jnp.asarray(sigma.sigma_full)
    arr, static = eqx.partition(params, eqx.is_inexact_array)

    def loss_fn(a):
        trainable = eqx.combine(a, static)
        p = make_run_seg(trainable).raw(ic, n_steps, forcing)
        return combined_loss(p, target, sigma_full, grid=grid,
                             config=loss_config)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(arr)
    assert bool(jnp.isfinite(loss))
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in leaves)


def test_unknown_training_core_raises():
    from legoesm.training.scale_build import build_mode_components
    with pytest.raises(ValueError, match="unknown training_core"):
        build_mode_components(
            SimpleNamespace(mode="physics", training_core="bogus",
                            smoke=False, multi_step_hours=()),
            _yml())


def test_era5_forcing_calendar_convention():
    """codex #817 adversarial finding: spectral_rollout advances
    ``doy_eff = day_of_year + (t + seconds_of_day)/86400`` and wraps doy to
    [1, 366), so the sample forcing must carry the INTEGER 1-based day with the
    intra-day fraction in seconds_of_day alone.  The original 0-based
    fractional doy sent Jan 1 00Z to day 365 (a year off) and double-counted
    the hours."""
    from legoesm.training.scale_build import era5_time_to_forcing_calendar

    # Jan 1 00Z -> day 1, 0 s (NOT day 0 -> wrapped to 365).
    doy, sod = era5_time_to_forcing_calendar(np.datetime64("2010-01-01T00"), 2010)
    assert doy == 1.0 and sod == 0.0
    # Jan 1 06Z -> day 1 + 21600 s (fraction rides seconds, not doy).
    doy, sod = era5_time_to_forcing_calendar(np.datetime64("2010-01-01T06"), 2010)
    assert doy == 1.0 and sod == 21600.0
    # Feb 2 12Z -> day 33 + 43200 s.
    doy, sod = era5_time_to_forcing_calendar(np.datetime64("2010-02-02T12"), 2010)
    assert doy == 33.0 and sod == 43200.0
    # Consistency with the rollout's advance: doy + sod/86400 is the fractional
    # 1-based day the declination sees at step 0.
    assert doy + sod / 86400.0 == 33.5


# ---- scattered training windows (#1047 finding 2) ----

_SNAPS_PER_DAY = 4  # 6-hourly ERA5 cadence

def _fake_times():
    # 6-hourly datetimes across 2015-01-01 .. 2015-07-01 (well past day 90).
    return np.arange("2015-01-01", "2015-07-01", np.timedelta64(6, "h"),
                     dtype="datetime64[ns]")


def test_resolve_windows_precedence_and_validation():
    from legoesm.training.scale_build import _resolve_windows

    yml = {"train_windows": [(2015, 0, 1)]}
    # None everywhere -> consecutive mode
    assert _resolve_windows(SimpleNamespace(), {}) is None
    # YAML train_windows honoured
    assert _resolve_windows(SimpleNamespace(), yml) == [(2015, 0, 1)]
    # cfg.windows overrides YAML
    cfg = SimpleNamespace(windows=[(2016, 90, 2), (2016, 270, 2)])
    assert _resolve_windows(cfg, yml) == [(2016, 90, 2), (2016, 270, 2)]
    # validation: wrong arity / bad values raise
    with pytest.raises(ValueError, match="year, day_offset, n_days"):
        _resolve_windows(SimpleNamespace(windows=[(2015, 0)]), {})
    with pytest.raises(ValueError, match="day_offset>=0"):
        _resolve_windows(SimpleNamespace(windows=[(2015, -1, 1)]), {})
    with pytest.raises(ValueError, match="n_days>=1"):
        _resolve_windows(SimpleNamespace(windows=[(2015, 0, 0)]), {})
    # ICs must stay in-year (else the window silently samples the next year with a
    # wrong calendar label). 2015 is not a leap year -> 365 days.
    with pytest.raises(ValueError, match="must stay within 2015"):
        _resolve_windows(SimpleNamespace(windows=[(2015, 364, 2)]), {})
    # ...but a window ending exactly on Dec 31 is allowed (364 + 1 == 365).
    assert _resolve_windows(SimpleNamespace(windows=[(2015, 364, 1)]), {}) == [(2015, 364, 1)]
    # leap year gets the extra day (366): offset 365 + 1 day is valid in 2016.
    assert _resolve_windows(SimpleNamespace(windows=[(2016, 365, 1)]), {}) == [(2016, 365, 1)]


def test_training_sample_indices_consecutive_default():
    """No windows -> the first n_days of each train year (back-compat)."""
    from legoesm.training.scale_build import _training_sample_indices

    times = _fake_times()
    cfg = SimpleNamespace(n_days=2, smoke=False)
    yml = {"train_years": [2015]}
    idx = list(_training_sample_indices(cfg, yml, times, _SNAPS_PER_DAY, stride=1))
    # 2 days x 4 snaps = 8 samples starting at index 0, target = ic + stride.
    assert len(idx) == 2 * _SNAPS_PER_DAY
    assert idx[0] == (2015, 0, 1)
    assert [i_ic for _, i_ic, _ in idx] == list(range(8))
    assert all(i_tg == i_ic + 1 for _, i_ic, i_tg in idx)


def test_training_sample_indices_scattered_windows():
    """Windows scatter samples across the calendar at the given day offsets."""
    from legoesm.training.scale_build import _training_sample_indices

    times = _fake_times()
    cfg = SimpleNamespace(smoke=False, windows=[(2015, 0, 1), (2015, 90, 1)])
    idx = list(_training_sample_indices(cfg, {}, times, _SNAPS_PER_DAY, stride=2))
    assert len(idx) == 2 * _SNAPS_PER_DAY               # two 1-day windows
    first = [i_ic for _, i_ic, _ in idx[:4]]
    second = [i_ic for _, i_ic, _ in idx[4:]]
    assert first == [0, 1, 2, 3]                        # window at day 0
    assert second == [360, 361, 362, 363]              # window at day 90 (90*4)
    assert all(i_tg == i_ic + 2 for _, i_ic, i_tg in idx)  # stride honoured


def test_training_sample_indices_smoke_stops_after_one_window():
    from legoesm.training.scale_build import _training_sample_indices

    times = _fake_times()
    cfg = SimpleNamespace(smoke=True, windows=[(2015, 0, 1), (2015, 90, 1)])
    idx = list(_training_sample_indices(cfg, {}, times, _SNAPS_PER_DAY, stride=1))
    assert {y for y, _, _ in idx} == {2015}
    assert len(idx) == _SNAPS_PER_DAY                   # only the first window


def test_spectral_core_raw_two_leg_matches_raw():
    """``raw_two_leg(self, ...)`` is the same forecast as ``raw`` (the FD
    probe's self-check), and the wrapper itself runs: it once shipped calling
    an unimported ``jax`` and crashed only on the GPU."""
    from legoesm.training.scale_build import build_mode_components

    yml = _yml()
    model, grid, sigma, params, make_run_seg, loss_config, dt = (
        build_mode_components(_cfg("physics"), yml))
    ic = _synthetic_carry(grid, int(yml["nlev"]))
    ncol = int(grid.n_lat) * int(grid.n_lon)
    forcing = {
        "T_sfc": jnp.full((ncol,), 300.0),
        "sic": jnp.zeros((ncol,)),
        "land_frac": jnp.zeros((int(grid.n_lat), int(grid.n_lon))),
        "day_of_year": jnp.asarray(80.0),
        "seconds_of_day": jnp.asarray(0.0),
    }
    seg = make_run_seg(params)
    one = seg.raw(ic, 3, forcing)
    two = seg.raw_two_leg(seg, ic, 3, forcing, 1)
    for name in ("u", "v", "T", "p_s", "q_v"):
        np.testing.assert_allclose(np.asarray(getattr(two, name)),
                                   np.asarray(getattr(one, name)),
                                   rtol=1e-10, atol=1e-10, err_msg=name)


def _perturbed(params, scale=1e-2, seed=0):
    """Params with every float leaf nudged off its (possibly zero) init, so a
    learned arm's output actually depends on its inputs (incl. the clock)."""
    import equinox as eqx

    arr, static = eqx.partition(params, eqx.is_inexact_array)
    leaves, tdef = jax.tree_util.tree_flatten(arr)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    leaves = [x + scale * jax.random.normal(k, x.shape, x.dtype)
              for x, k in zip(leaves, keys)]
    return eqx.combine(jax.tree_util.tree_unflatten(tdef, leaves), static)


@pytest.mark.parametrize("mode", ["neural_gcm", "physics"])
def test_spectral_core_raw_multi_equals_one_rollout(mode):
    """``raw_multi`` chains segments EXACTLY: after k segments it equals one
    ``raw`` rollout of k*seg steps (clock, physics state and cached radiation
    carried on).  The IC sits at 21 UTC so a chained segment that restarted
    the clock would see a different sun."""
    from legoesm.training.scale_build import build_mode_components

    yml = _yml()
    model, grid, sigma, params, make_run_seg, loss_config, dt = (
        build_mode_components(_cfg(mode), yml))
    if mode == "neural_gcm":
        params = _perturbed(params)
    ic = _synthetic_carry(grid, int(yml["nlev"]))
    ncol = int(grid.n_lat) * int(grid.n_lon)
    forcing = {
        "T_sfc": jnp.full((ncol,), 300.0),
        "sic": jnp.zeros((ncol,)),
        "land_frac": jnp.zeros((int(grid.n_lat), int(grid.n_lon))),
        "day_of_year": jnp.asarray(80.0),
        "seconds_of_day": jnp.asarray(21 * 3600.0),
    }
    seg = make_run_seg(params)
    outs = seg.raw_multi(ic, 2, 2, forcing)
    assert len(outs) == 2
    for got, n in ((outs[0], 2), (outs[1], 4)):
        ref = seg.raw(ic, n, forcing)
        for name in ("u", "v", "T", "p_s", "q_v"):
            np.testing.assert_allclose(np.asarray(getattr(got, name)),
                                       np.asarray(getattr(ref, name)),
                                       rtol=1e-10, atol=1e-10,
                                       err_msg=f"{name} after {n} steps")


def test_supervised_leads_and_curriculum():
    from legoesm.training.scale_build import curriculum_n_segs, supervised_leads

    assert supervised_leads({"loss": {}}) == ()
    yml = {"loss": {"train_leads_hours": [6, 12, 18, 24],
                    "train_curriculum_hours": [6, 12, 24]}}
    leads = supervised_leads(yml)
    assert leads == (6, 12, 18, 24)
    assert curriculum_n_segs(yml, leads, 12) == (1, 2, 4)
    assert curriculum_n_segs({"loss": {}}, leads, 5) == (4,)
    with pytest.raises(ValueError, match="equal curriculum"):
        curriculum_n_segs(yml, leads, 10)
    with pytest.raises(ValueError, match="consecutive multiples"):
        supervised_leads({"loss": {"train_leads_hours": [6, 12, 24]}})
    with pytest.raises(ValueError, match="increasing"):
        curriculum_n_segs({"loss": {"train_curriculum_hours": [12, 6, 24]}}, leads, 12)
    with pytest.raises(ValueError, match="must end at the last lead"):
        curriculum_n_segs({"loss": {"train_curriculum_hours": [6, 12]}}, leads, 12)


def test_multi_lead_loss_is_mean_over_scored_leads():
    """``multi_lead_loss`` = (1/N) * sum of the per-lead carry losses of one
    chained forecast, N = the stage's lead count; near-surface terms ride
    along when the target carries the fields."""
    from legoesm.training.losses import LossConfig, combined_loss
    from legoesm.training.scale_build import multi_lead_loss

    grid = create_gaussian_grid(10, dealiasing="quadratic")
    sigma = create_sigma_coordinate(8)
    sf = jnp.asarray(sigma.sigma_full)
    base = _synthetic_carry(grid, 8)
    preds = tuple(base._replace(T=base.T + d) for d in (0.5, 1.0, 2.0))
    tgts = tuple((base, {}) for _ in preds)

    class _Seg:
        def raw_multi(self, ic, seg_steps, n, forcing, adjoint_steps=0):
            return preds[:n]

    cfg = LossConfig()
    kw = dict(seg_steps=1, sigma_full=sf, grid=grid, sigma=sigma,
              loss_config=cfg, diagnose_fn=None)
    for n in (1, 2, 3):
        got = multi_lead_loss(_Seg(), base, tgts, None, n_segs=n, **kw)
        want = sum(float(combined_loss(p, base, sf, grid=grid, config=cfg))
                   for p in preds[:n]) / n
        np.testing.assert_allclose(float(got), want, rtol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        multi_lead_loss(_Seg(), base, tgts, None, n_segs=4, **kw)


def test_curriculum_without_leads_is_refused():
    from legoesm.training.scale_build import supervised_leads

    with pytest.raises(ValueError, match="needs loss.train_leads_hours"):
        supervised_leads({"loss": {"train_curriculum_hours": [6]}})


def test_multi_lead_loss_real_rollout_with_surface_targets():
    """Real chained forecast + the WB2 eval diagnosis: targets equal to the
    forecast's own states and diagnosed near-surface fields score ~0; moving
    one lead's t2m target by 1 K adds exactly (1/s)^2 * w / N to the loss."""
    import importlib
    import pathlib
    import sys as _sys

    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
    from legoesm.training.scale_build import build_mode_components, multi_lead_loss

    root = str(pathlib.Path(__file__).resolve().parents[2])
    if root not in _sys.path:
        _sys.path.insert(0, root)
    diagnose = importlib.import_module(
        "evaluations.wb_forecast").diagnose_headline_fields

    yml = _yml()
    _, grid, sigma, params, mk, _, dt = build_mode_components(_cfg("neural_gcm"), yml)
    seg = mk(_perturbed(params))
    ic = _synthetic_carry(grid, int(yml["nlev"]))
    ncol = int(grid.n_lat) * int(grid.n_lon)
    f = {"T_sfc": jnp.full((ncol,), 300.0), "sic": jnp.zeros((ncol,)),
         "day_of_year": jnp.asarray(80.0), "seconds_of_day": jnp.asarray(0.0)}
    outs = seg.raw_multi(ic, 2, 2, f)
    sfc = [{k: diagnose(carry_to_spectral_state(o, grid), grid, sigma).fields[k]
            for k in ("t2m", "u10", "v10", "mslp")} for o in outs]
    cfg = LossConfig(w_t2m=1.0, t2m_resid_scale=2.0, w_u10=0.1, u10_resid_scale=3.0)
    kw = dict(n_segs=2, seg_steps=2, sigma_full=jnp.asarray(sigma.sigma_full),
              grid=grid, sigma=sigma, loss_config=cfg, diagnose_fn=diagnose,
              terms=True)
    l0, t0 = multi_lead_loss(seg, ic, tuple(zip(outs, sfc)), f, **kw)
    assert {"mse_t2m", "mse_u10"} <= set(t0)
    assert float(l0) < 1e-16
    moved = [dict(sfc[0]), sfc[1]]
    moved[0]["t2m"] = moved[0]["t2m"] + 1.0
    l1, t1 = multi_lead_loss(seg, ic, tuple(zip(outs, moved)), f, **kw)
    np.testing.assert_allclose(float(l1) - float(l0), 0.25 / 2, rtol=1e-9)


def test_lead_hours_must_be_integers():
    from legoesm.training.scale_build import supervised_leads

    with pytest.raises(ValueError, match="integer hours"):
        supervised_leads({"loss": {"train_leads_hours": [6.9, 12.9]}})


def test_legacy_multi_step_hours_train_every_lead_on_spectral_only():
    from legoesm.training.scale_build import supervised_leads

    old = {"loss": {"multi_step_hours": [6, 12], "multi_step_weights": [1.0, 1.0]}}
    assert supervised_leads(old) == (6, 12)
    assert supervised_leads(old, "latlon") == ()
    assert supervised_leads({"loss": {"multi_step_hours": [6]}}) == ()
    with pytest.raises(ValueError, match="weighted"):
        supervised_leads({"loss": {"multi_step_hours": [6, 12],
                                   "multi_step_weights": [1.0, 0.5]}})
