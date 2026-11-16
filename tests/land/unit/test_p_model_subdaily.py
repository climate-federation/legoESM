"""Subdaily (noon-window) P-model acclimation, Mengoli et al. (2022).

Properties pinned here (land/p_model.py, acclimation='subdaily_noon'):

* the 11:00-13:00 local-solar window holds 3 hourly / 5 half-hourly steps and
  only the noon step at a 3-hourly step, and closes exactly once per day;
* the first completed window SEEDS the realised traits with that day's
  optimum; afterwards real <- real + alpha * (opt - real) (pyrealm order);
* the traits do not change between windows, and the 24 h running means keep
  evolving exactly as under ema_24h;
* misuse raises (unknown scheme, missing solar hour, P-hydro soil means,
  ema state handed to subdaily consumers);
* finite gradients and eager/jit parity through a full day;
* the land restart round-trips the subdaily state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.p_model import (
    PModelConfig, acclim_means, acclimated_capacities, acclimated_capacities_c4,
    advance_pmodel_acclim, capacities_at, capacities_c4_at, in_noon_window,
    init_pmodel_acclim, local_solar_hour,
)

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _ENTRY_X64)


SUB = PModelConfig(acclimation="subdaily_noon")
NCOL = 2
T_DAY = constants.T_freeze + 22.0
T_NIGHT = constants.T_freeze + 12.0


def _init(cfg=SUB):
    return init_pmodel_acclim(NCOL, t_init_K=T_DAY, ps_init_pa=101325.0,
                              cfg=cfg, co2_init_ppm=410.0, dtype=jnp.float64)


def _hours(dt):
    return np.arange(0.0, 24.0, dt / 3600.0)


def _drivers(h, light_scale=1.0):
    """Daytime-shaped drivers at solar hour h (scalar), broadcast to NCOL."""
    day = max(np.sin(np.pi * (h - 6.0) / 12.0), 0.0)
    v = lambda x: jnp.full((NCOL,), x, dtype=jnp.float64)  # noqa: E731
    return dict(T_K=v(T_NIGHT + (T_DAY - T_NIGHT) * day),
                ppfd=v(light_scale * 900.0 * day), vpd_pa=v(500.0 + 1000.0 * day),
                co2_ppm=v(410.0), ps_pa=v(101325.0))


def _run_day(state, dt, cfg=SUB, light_scale=1.0, start_h=0.0):
    for h in _hours(dt) + start_h:
        state = advance_pmodel_acclim(state, **_drivers(h % 24.0, light_scale),
                                      cfg=cfg, dt=dt,
                                      solar_hour=jnp.full((NCOL,), h % 24.0))
    return state


@pytest.mark.parametrize("dt,n_expected", [(1800.0, 5), (3600.0, 3), (10800.0, 1)])
def test_window_step_count_and_one_close_per_day(dt, n_expected):
    h = jnp.asarray(_hours(dt))
    inw = np.asarray(in_noon_window(h, dt, SUB))
    assert int(inw.sum()) == n_expected
    # the window is centred on noon
    assert float(np.asarray(h)[inw].mean()) == pytest.approx(12.0)
    nxt = np.asarray(in_noon_window(h + dt / 3600.0, dt, SUB))
    assert int((inw & ~nxt).sum()) == 1


def test_local_solar_hour_uses_longitude():
    # 12:00 UTC at 90 E is 18:00 mean solar time; at 90 W it is 06:00.
    assert float(local_solar_hour(43200.0, 90.0)) == pytest.approx(18.0)
    assert float(local_solar_hour(43200.0, -90.0)) == pytest.approx(6.0)


def test_first_window_seeds_traits_with_that_days_optimum():
    dt = 1800.0
    s1 = _run_day(_init(), dt)
    assert np.all(np.asarray(s1.n_days) == 1.0)
    # the optimum at the five noon-window means, evaluated independently
    win = [h for h in _hours(dt) if 11.0 <= h <= 13.0]
    m = {k: jnp.mean(jnp.stack([_drivers(h)[k] for h in win]), axis=0)
         for k in ("T_K", "ppfd", "vpd_pa", "co2_ppm", "ps_pa")}
    opt = capacities_at(m["T_K"], m["vpd_pa"], m["co2_ppm"], m["ps_pa"],
                        m["ppfd"], SUB)
    np.testing.assert_allclose(s1.vcmax25_real, opt.vcmax25_leaf, rtol=1e-12)
    np.testing.assert_allclose(s1.jmax25_real, opt.vcmax25_leaf * opt.rjv25,
                               rtol=1e-12)
    np.testing.assert_allclose(s1.xi_real, opt.xi_sqrt_pa, rtol=1e-12)
    opt4 = capacities_c4_at(m["T_K"], m["vpd_pa"], m["co2_ppm"], m["ps_pa"],
                            m["ppfd"], SUB)
    np.testing.assert_allclose(s1.vcmax25_c4_real, opt4.vcmax25_c4_leaf,
                               rtol=1e-12)
    np.testing.assert_allclose(s1.day_t_K, m["T_K"], rtol=1e-12)


def test_memory_after_seed_follows_alpha():
    """Day 2 at twice the light: Vcmax is proportional to absorbed light at
    fixed T/VPD/CO2, so opt2 = 2 opt1 and the realised value moves by alpha."""
    dt = 3600.0
    s1 = _run_day(_init(), dt)
    s2 = _run_day(s1, dt, light_scale=2.0)
    alpha = SUB.subdaily_alpha
    np.testing.assert_allclose(s2.vcmax25_real,
                               s1.vcmax25_real * (1.0 + alpha), rtol=1e-10)
    # xi does not depend on light: unchanged by the second day
    np.testing.assert_allclose(s2.xi_real, s1.xi_real, rtol=1e-12)
    assert np.all(np.asarray(s2.n_days) == 2.0)


def test_traits_frozen_between_windows_and_ema_unchanged():
    dt = 1800.0
    s0 = _init()
    s = s0
    for h in _hours(dt)[:20]:            # 00:00-09:30, no window step
        s = advance_pmodel_acclim(s, **_drivers(h), cfg=SUB, dt=dt,
                                  solar_hour=jnp.full((NCOL,), h))
    for f in ("xi_real", "vcmax25_real", "jmax25_real", "n_days", "win_count"):
        np.testing.assert_array_equal(getattr(s, f), getattr(s0, f))
    # the 24 h running means advance exactly as under ema_24h
    e = _init(PModelConfig())
    for h in _hours(dt)[:20]:
        e = advance_pmodel_acclim(e, **_drivers(h), cfg=PModelConfig(), dt=dt)
    for f in ("iabs_mean", "t_mean_K", "vpd_mean_pa", "co2_mean_ppm", "ps_ema"):
        np.testing.assert_array_equal(getattr(s, f), getattr(e, f))


def test_capacities_read_realised_traits_and_window_means():
    s = _run_day(_init(), 3600.0)
    caps = acclimated_capacities(s, SUB)
    np.testing.assert_array_equal(caps.vcmax25_leaf, s.vcmax25_real)
    np.testing.assert_allclose(caps.rjv25 * caps.vcmax25_leaf, s.jmax25_real,
                               rtol=1e-12)
    assert np.all((np.asarray(caps.chi) > 0) & (np.asarray(caps.chi) < 1))
    c4 = acclimated_capacities_c4(s, SUB)
    np.testing.assert_array_equal(c4.vcmax25_c4_leaf, s.vcmax25_c4_real)
    view = acclim_means(s, SUB)
    np.testing.assert_array_equal(view.t_mean_K, s.day_t_K)
    np.testing.assert_array_equal(view.vpd_mean_pa, s.day_vpd_pa)
    # ema_24h view is the state itself
    e = _init(PModelConfig())
    assert acclim_means(e, PModelConfig()) is e


def test_default_state_has_no_subdaily_fields():
    e = _init(PModelConfig())
    assert e.xi_real is None and e.win_count is None and e.n_days is None
    assert PModelConfig().acclimation == "ema_24h"


def test_off_window_nan_does_not_reach_the_window_sums():
    dt = 3600.0
    s = _init()
    for h in _hours(dt):
        d = _drivers(h)
        if not 11.0 <= h <= 13.0:          # poison every off-window driver
            d = {k: v * jnp.nan for k, v in d.items()}
        s = advance_pmodel_acclim(s, **d, cfg=SUB, dt=dt,
                                  solar_hour=jnp.full((NCOL,), h))
    for f in ("xi_real", "vcmax25_real", "jmax25_real", "day_t_K"):
        assert np.all(np.isfinite(np.asarray(getattr(s, f)))), f


def test_invalid_window_or_memory_is_refused():
    with pytest.raises(ValueError, match="subdaily_alpha"):
        _init(PModelConfig(acclimation="subdaily_noon", subdaily_alpha=1.5))
    with pytest.raises(ValueError, match="window"):
        _init(PModelConfig(acclimation="subdaily_noon", window_start_h=13.0,
                           window_end_h=11.0))


def test_refusals():
    bad = PModelConfig(acclimation="noon")
    with pytest.raises(ValueError, match="unknown PModelConfig.acclimation"):
        _init(bad)
    s = _init()
    with pytest.raises(ValueError, match="solar hour"):
        advance_pmodel_acclim(s, **_drivers(12.0), cfg=SUB, dt=1800.0)
    with pytest.raises(ValueError, match="P-hydro"):
        init_pmodel_acclim(NCOL, t_init_K=T_DAY, ps_init_pa=101325.0, cfg=SUB,
                           soil_init=(-0.5, 0.1))
    with pytest.raises(ValueError, match="lacks"):
        acclimated_capacities(_init(PModelConfig()), SUB)


def test_eager_jit_parity_over_a_day():
    dt = 3600.0
    hours = jnp.asarray(_hours(dt))
    drv = [_drivers(float(h)) for h in _hours(dt)]
    xs = {k: jnp.stack([d[k] for d in drv]) for k in drv[0]}
    sh = jnp.broadcast_to(hours[:, None], (hours.size, NCOL))

    def run(s0, xs, sh):
        def body(s, x):
            d, h = x
            return advance_pmodel_acclim(s, **d, cfg=SUB, dt=dt, solar_hour=h), None
        return jax.lax.scan(body, s0, (xs, sh))[0]

    eager = _run_day(_init(), dt)
    jitted = jax.jit(run)(_init(), xs, sh)
    for f in ("xi_real", "vcmax25_real", "jmax25_real", "day_t_K"):
        np.testing.assert_allclose(getattr(jitted, f), getattr(eager, f),
                                   rtol=1e-12)


def test_gradient_through_a_day_is_finite_and_nonzero():
    dt = 3600.0
    hours = _hours(dt)

    def vcmax_after_day(t_offset):
        s = _init()
        for h in hours:
            d = _drivers(h)
            d["T_K"] = d["T_K"] + t_offset
            s = advance_pmodel_acclim(s, **d, cfg=SUB, dt=dt,
                                      solar_hour=jnp.full((NCOL,), h))
        return jnp.sum(s.vcmax25_real)

    g = jax.grad(vcmax_after_day)(0.0)
    assert np.isfinite(float(g)) and float(g) != 0.0
    from jax.test_util import check_grads
    check_grads(vcmax_after_day, (0.0,), order=1, modes=("rev",),
                atol=1e-6, rtol=1e-4)


def test_restart_round_trips_subdaily_state(tmp_path):
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import init_multilayer_land_state
    from legoesm.land.restart import load_land_restart, save_land_restart
    from legoesm.land.surface_scheme.two_leaf_canopy import TwoLeafCanopyConfig

    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            stomatal_model="medlyn", capacity_scheme="p_model",
            g1_source="p_model").validate(),
        p_model=SUB)
    st = init_multilayer_land_state(NCOL, cfg, T_init=T_DAY)
    st = st._replace(pmodel_acclim=_run_day(st.pmodel_acclim, 3600.0))
    path = save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                             t_end_s=86400.0, n_steps_completed=24)
    loaded, _ = load_land_restart(path, expected_land_mode="multilayer",
                                  expected_ncol=NCOL)
    for f, a in zip(st.pmodel_acclim._fields, st.pmodel_acclim):
        b = getattr(loaded.pmodel_acclim, f)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
