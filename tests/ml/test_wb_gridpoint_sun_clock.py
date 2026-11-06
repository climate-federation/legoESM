"""The WB grid-point segments place the sun at each sample's own date and time
(user decision Q3, 2026-11-06), through the trainer's own build.

Every solar-geometry call the rollout makes (the network's insolation input;
the physics arm's radiation on the per-step clock) is recorded by wrapping the
solar module, for two samples six months and twelve hours apart.  Before the
fix both saw Jan 1 00 UTC + elapsed steps.  The segment's second clock site
(the radiation refresh of the un-fused radiation path) runs only under
ExperimentConfig.unfused_radiation, which no WB deck selects; it is not
reached here (mutation-checked).
Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import legoesm.atmosphere.physics.radiation.solar as solar  # noqa: E402
from legoesm.forcing.time_utils import day_to_calendar  # noqa: E402
from tests.ml.test_wb_dycore_swap import _YML, _carry, _forcing, _mod  # noqa: E402

N_STEPS = 2
TIMES = ("2020-01-01T00", "2020-07-01T12")


@pytest.fixture
def seen(monkeypatch):
    calls = []

    def rec(doy, hour):
        jax.debug.callback(lambda d, h: calls.append((float(d), float(h))),
                           jnp.asarray(doy), jnp.asarray(hour))

    cz, dm = solar.cos_zenith_angle, solar.daily_mean_insolation

    def cos_zenith_angle(lat, lon, day_of_year, hour, *a, **k):
        rec(day_of_year, hour)
        return cz(lat, lon, day_of_year, hour, *a, **k)

    def daily_mean_insolation(lat, day_of_year, *a, **k):
        rec(day_of_year, -1.0)
        return dm(lat, day_of_year, *a, **k)

    monkeypatch.setattr(solar, "cos_zenith_angle", cos_zenith_angle)
    monkeypatch.setattr(solar, "daily_mean_insolation", daily_mean_insolation)
    return calls


@pytest.mark.parametrize("core,mode", [("latlon", "neural_gcm"),
                                       ("cubed_sphere", "neural_gcm"),
                                       ("latlon", "physics")])
def test_sun_is_the_samples_own(core, mode, seen):
    from legoesm.training.scale_build import (
        build_mode_components, era5_time_to_forcing_calendar)

    cfg = _mod.build_scale_config_from_args(
        ["--mode", mode, "--smoke", "--training-core", core])
    # physics arm: radiation every 2nd step (fused path: per-step clock)
    # (the synthetic carry seeds no ice skin, so the physics arm runs without it)
    yml = {**_YML, "rad_update_steps": 2 if mode == "physics" else 1,
           **({"ice_skin_prognostic": False} if mode == "physics" else {})}
    _, grid, _, params, mk, _, dt = build_mode_components(cfg, yml)
    ic = _carry(core, grid, 8)
    f0 = _forcing(core, cfg, yml, grid, 0)
    for t in TIMES:
        doy, sod = era5_time_to_forcing_calendar(np.datetime64(t, "ns"), 2020)
        seen.clear()
        f = f0._replace(day_of_year=jnp.asarray(doy),
                        seconds_of_day=jnp.asarray(sod))
        jax.block_until_ready(mk(params).raw(ic, N_STEPS, f).T)
        assert seen, "no solar-geometry call recorded"
        start = doy - 1.0 + sod / 86400.0
        # the clock of the step boundaries 0..N_STEPS (end-of-step k+1 per
        # step; the cadence refresh at the completed-step count)
        want = {tuple(round(float(x), 9) for x in day_to_calendar(
            start + k * dt / 86400.0)) for k in range(N_STEPS + 1)}
        for d, h in seen:
            if h < 0:                                   # daily mean: day only
                assert any(abs(d - w[0]) < 1e-9 for w in want), (t, d)
            else:
                assert (round(d, 9), round(h * 3600.0, 9)) in want, (t, d, h)
        if mode == "neural_gcm":
            # the network's input: one call per step, at end-of-step k+1
            got = [(round(d, 9), round(h * 3600.0, 9)) for d, h in seen]
            assert got == [tuple(round(float(x), 9) for x in day_to_calendar(
                start + (k + 1) * dt / 86400.0)) for k in range(N_STEPS)], t


@pytest.mark.parametrize("core", ["latlon", "mpas"])
def test_loader_packs_each_samples_own_date_and_time(core, monkeypatch):
    """The grid-point / MPAS loader packs the sample's own initial time as a
    1-based day of year plus seconds of day (the packing the segments
    unpack), written out here from the calendar (codex/GLM r1: the clock
    test above packs its own forcing)."""
    from scripts.data.make_synthetic_era5 import build_synthetic_era5_dataset
    from legoesm.training import era5_to_state
    from legoesm.training.scale_build import build_mode_components, load_era5_samples

    ds = build_synthetic_era5_dataset(nlat=8, nlon=12, ntime=2)
    ds = ds.assign_coords(time=np.array(["2019-07-01T12", "2019-07-01T18"],
                                       dtype="datetime64[ns]"))
    monkeypatch.setattr(era5_to_state, "open_era5_zarr", lambda _: ds)
    cfg = _mod.build_scale_config_from_args(
        ["--mode", "neural_gcm", "--smoke", "--training-core", core])
    yml = {**_YML, "era5_zarr": "synthetic", "train_years": [2019],
           "eval_years": [2021], "era5_cadence_hours": 6,
           "n_training_days": 1, "era5_surface_fluxes": False}
    _, grid, sigma, *_ = build_mode_components(cfg, yml)
    samples = load_era5_samples(cfg, yml, grid, sigma)
    assert len(samples) == 1
    f = samples[0][2]
    # 2019-07-01 = day 182 of a non-leap year, 12 UTC
    assert float(f.day_of_year) == 182.0
    assert float(f.seconds_of_day) == 43200.0
