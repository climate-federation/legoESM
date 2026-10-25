"""The land receives the atmosphere's OWN rain/snow partition (review
2026-10-10 F29).

The production column loop used to re-split total precipitation with a hard
0 C step on the lowest-level temperature, discarding the microphysics' phase:
wherever the two disagreed, the latent heat of fusion the atmosphere released
(or did not) was not matched at the land.  Now each scheme publishes the frozen
part of its surface flux (``precip_frozen``), the combined physics sums it like
``precip`` (unknown -- never partial -- when a precipitating module does not
publish it), it rides sfc-diag slot 14, and the land takes it as snowfall.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import (  # noqa: E402
    MPAS_SFC_DIAG_BASE_KEYS,
    MPAS_SFC_DIAG_EXTRA_KEYS,
)
from legoesm.thermo import saturation_specific_humidity  # noqa: E402


def _f(x):
    return Field(data=jnp.asarray(x), name="x", dims=("nCells",), units="kg/m^2/s")


def _sfc_diag(precip, frozen):
    n = len(MPAS_SFC_DIAG_BASE_KEYS) + len(MPAS_SFC_DIAG_EXTRA_KEYS)
    sd = [None] * n
    sd[2] = None if precip is None else _f(precip)
    sd[len(MPAS_SFC_DIAG_BASE_KEYS)
       + MPAS_SFC_DIAG_EXTRA_KEYS.index("precip_frozen")] = (
        None if frozen is None else _f(frozen))
    return tuple(sd)


def test_land_takes_the_published_frozen_flux_not_a_temperature_split():
    from legoesm.driver.model_driver import land_snowfall_from_sfc_diag
    precip = jnp.asarray([2e-4, 3e-4, 0.0, 1e-4])
    frozen = jnp.asarray([5e-5, 0.0, 0.0, 2e-4])        # last: > precip -> clipped
    snow = land_snowfall_from_sfc_diag(_sfc_diag(precip, frozen), precip)
    np.testing.assert_array_equal(np.asarray(snow), [5e-5, 0.0, 0.0, 1e-4])
    # nothing published -> no snow
    z = land_snowfall_from_sfc_diag(_sfc_diag(None, None), jnp.zeros(4))
    np.testing.assert_array_equal(np.asarray(z), np.zeros(4))
    # precip WITHOUT its frozen part -> refused, never re-split by T
    with pytest.raises(ValueError, match="precip_frozen"):
        land_snowfall_from_sfc_diag(_sfc_diag(precip, None), precip)


def _column(T_sfc, *, snow=0.0, rain=0.0, rh=0.7, ncol=3, nlev=20):
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, nlev + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_half = jnp.broadcast_to((sh * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (ncol, nlev))
    T = jnp.broadcast_to(jnp.maximum(T_sfc * jnp.clip(sf, 0.01, None) ** 0.19,
                                     200.0)[None, :], (ncol, nlev))
    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0, None)))
    q_v = rh * saturation_specific_humidity(T, p_full)
    z = jnp.zeros((ncol, nlev))
    lo = lambda v: z.at[:, -6:].set(v)  # noqa: E731
    hydro = HydrometeorState(q_c=z, q_r=lo(rain), q_i=z, q_s=lo(snow), q_g=z,
                             N_c=z, N_r=lo(1e4 if rain else 0.0), N_i=z)
    return T, q_v, hydro, p_full, p_half, rho, dz


def _run(name, col, dt=300.0):
    from legoesm.atmosphere.physics.microphysics import config as mc
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    from legoesm.atmosphere.physics.microphysics.seifert_beheng import (
        seifert_beheng_microphysics)
    from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
    fn, cfg = {"kessler": (kessler_microphysics, mc.KesslerConfig()),
               "sundqvist": (sundqvist_microphysics, mc.SundqvistConfig()),
               "seifert_beheng": (seifert_beheng_microphysics,
                                  mc.SeifertBehengConfig()),
               "morrison": (morrison_microphysics, mc.MorrisonConfig()),
               "thompson": (thompson_microphysics, mc.ThompsonConfig())}[name]
    return fn(*col, dt, config=cfg)


@pytest.mark.parametrize("name", ["morrison", "thompson"])
def test_ice_schemes_publish_their_frozen_fall(name):
    """Cold column holding only snow: everything that lands is frozen.
    Warm column holding only rain: nothing that lands is frozen.  (The old
    land split was blind to both: it keyed on the lowest-level T only.)"""
    cold = _run(name, _column(262.0, snow=1e-3))
    assert float(cold.precipitation.min()) > 0.0        # non-vacuous
    np.testing.assert_allclose(np.asarray(cold.precip_frozen),
                               np.asarray(cold.precipitation), rtol=1e-10)
    # saturated: Thompson caps sedimentation by the evaporation sink
    warm = _run(name, _column(295.0, rain=1e-3, rh=1.0))
    assert float(warm.precipitation.min()) > 0.0
    np.testing.assert_array_equal(np.asarray(warm.precip_frozen), 0.0)


@pytest.mark.parametrize("name", ["kessler", "seifert_beheng"])
def test_lv_only_schemes_publish_zero_frozen(name):
    """L_v-only heating: every kg they precipitate is liquid energetically."""
    out = _run(name, _column(262.0, rain=1e-3))
    np.testing.assert_array_equal(np.asarray(out.precip_frozen), 0.0)
    assert out.precip_frozen.shape == out.precipitation.shape


def test_zm_publishes_its_convective_snow_flux():
    from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
    from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
        zhang_mcfarlane_convection)
    ncol, nlev = 2, 16
    sig = jnp.linspace(5e3 / 1e5, 1.0, nlev)
    p_full = sig[None, :] * jnp.full((ncol, 1), 1e5)
    p_half = jnp.concatenate([jnp.full((ncol, 1), 2.5e3),
                              0.5 * (p_full[:, :-1] + p_full[:, 1:]),
                              jnp.full((ncol, 1), 1e5)], axis=1)
    zf = -8500.0 * jnp.log(p_full / 1e5)
    T = jnp.maximum(300.0 - 7e-3 * zf, 300.0 - 7.0 * 12.0)
    q = 16e-3 * jnp.exp(-zf / 3000.0)
    u = jnp.full((ncol, nlev), 5.0)
    out, _ = zhang_mcfarlane_convection(
        T, q, p_full, p_half, u, jnp.zeros_like(u), jnp.zeros((ncol, nlev)),
        dt=300.0, config=ZhangMcFarlaneConfig(land_fraction="none"),
        pblh=jnp.zeros(ncol))  # CAM's cold-start PBL height (ZM needs one)
    snow = np.asarray(out.precip_frozen_sfc)
    prec = np.asarray(jnp.sum(out.dq_r_conv_dt * (p_half[:, 1:] - p_half[:, :-1]),
                              axis=-1) / constants.g)
    assert snow.shape == (ncol,)
    assert float(snow.max()) > 0.0, snow                # non-vacuous: ZM snows here
    assert np.all(snow >= 0.0) and np.all(snow <= np.maximum(prec, 0.0) + 1e-12)


def test_combined_sums_frozen_and_never_reports_a_partial_sum():
    from legoesm.atmosphere.physics.combined import _FROZEN_UNKNOWN, _frozen_add
    from legoesm.core.state import HydrostaticTendencies

    def mod(precip, frozen):
        z = _f(jnp.zeros(2))
        return HydrostaticTendencies(
            du_dt=z, dT_dt=z, dp_s_dt=z, dphis_dt=z,
            precip=None if precip is None else _f(precip),
            precip_frozen=None if frozen is None else _f(frozen))

    acc = _frozen_add(None, mod(None, None))             # radiation-like
    assert acc is None
    acc = _frozen_add(acc, mod([2.0, 1.0], [1.0, 0.0]))  # microphysics
    acc = _frozen_add(acc, mod([1.0, 1.0], [0.5, 0.5]))  # ZM
    np.testing.assert_array_equal(np.asarray(acc), [1.5, 0.5])
    # a precipitating module without the frozen part poisons the sum
    assert _frozen_add(acc, mod([1.0, 1.0], None)) is _FROZEN_UNKNOWN
    assert _frozen_add(_FROZEN_UNKNOWN, mod([1.0, 1.0], [0.1, 0.1])) is _FROZEN_UNKNOWN
