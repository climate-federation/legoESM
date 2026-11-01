"""Unit tests for the Bitz-Lipscomb (1999) sea-ice thermodynamic core.

Pins the physics of ``ice/bitz_lipscomb.py`` (finding F-ICE-1,
the multi-layer enthalpy replacement for the Semtner-0 single skin
node), ahead of its prognostic-state integration:

* enthalpy ↔ temperature inversion is exact;
* salinity-dependent conductivity (brine lowers k) and effective heat
  capacity (brine latent storage spikes near the melting point);
* **exact column energy conservation** over a conduction step — the
  defining BL99 property;
* finite gradients (AD-safe) through the implicit solve.
"""

from __future__ import annotations

import base64
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ice.bitz_lipscomb import (
    freezing_temperature,
    ice_enthalpy,
    ice_temperature_from_enthalpy,
    ice_thermal_conductivity,
    ice_specific_heat,
    bitz_lipscomb_conduction_step,
)

_Tf = constants.T_freeze


def test_enthalpy_temperature_roundtrip():
    """T → q → T recovers the temperature (exact quadratic inverse)."""
    T = jnp.array([_Tf - 0.5, _Tf - 1.0, _Tf - 5.0, _Tf - 15.0, _Tf - 30.0])
    S = jnp.full_like(T, 4.0)
    q = ice_enthalpy(T, S)
    assert jnp.all(q < 0.0)                       # ice enthalpy is negative
    T_rt = ice_temperature_from_enthalpy(q, S)
    assert float(jnp.max(jnp.abs(T_rt - T))) < 1e-6


def test_conductivity_brine_lowers_k():
    """k(S=0) = k_0; brine (S>0) lowers conductivity (Untersteiner)."""
    T = jnp.full((2,), _Tf - 2.0)
    k = ice_thermal_conductivity(T, jnp.array([0.0, 8.0]))
    assert float(k[0]) == pytest.approx(constants.k_ice_default, rel=1e-6)
    assert float(k[1]) < float(k[0])


def test_specific_heat_brine_spikes_near_melt():
    """Effective heat capacity exceeds fresh-ice c_pi and grows toward
    the melting point (brine latent-heat storage)."""
    c = ice_specific_heat(jnp.array([_Tf - 10.0, _Tf - 1.0]), jnp.full((2,), 4.0))
    assert float(c[0]) > constants.c_pi
    assert float(c[1]) > float(c[0])             # larger near melting


def test_freezing_point_depression():
    """T_m = −μS depresses below 0 °C with salinity."""
    Tm = freezing_temperature(jnp.array([0.0, 4.0]))
    assert float(Tm[0]) == 0.0
    assert float(Tm[1]) == pytest.approx(-constants.mu_ice_freeze * 4.0)


@pytest.mark.parametrize("F_top", [[-30.0, 0.0, 50.0], [-100.0, 20.0, 5.0]])
def test_conduction_step_conserves_energy(F_top):
    """Column enthalpy change equals the net boundary flux to machine
    precision: Σ(q_new − q_old)·dz = (F_top − F_bottom)·dt."""
    n, ncol = 4, 3
    T0 = jnp.stack([jnp.linspace(_Tf - 12.0, _Tf - 2.0, n)] * ncol)
    S = jnp.full((ncol,), 4.0)
    h = jnp.full((ncol,), 2.0)
    q0 = ice_enthalpy(T0, S[:, None])
    F = jnp.asarray(F_top)
    T_bottom = jnp.full((ncol,), constants.T_freeze_ocean)
    dt = 3600.0

    q_new, T_new, F_bottom = bitz_lipscomb_conduction_step(q0, h, S, F, T_bottom, dt)
    dz = h / n
    column_denthalpy = jnp.sum((q_new - q0) * dz[:, None], axis=1)
    net_flux = (F - F_bottom) * dt
    rel = jnp.abs(column_denthalpy - net_flux) / (jnp.abs(net_flux) + 1.0)
    assert float(jnp.max(rel)) < 1e-9
    assert jnp.all(jnp.isfinite(T_new))


def test_conduction_step_directional():
    """A warming surface flux warms the top layer; a cooling flux cools it."""
    n, ncol = 4, 2
    T0 = jnp.stack([jnp.linspace(_Tf - 12.0, _Tf - 2.0, n)] * ncol)
    S = jnp.full((ncol,), 4.0)
    h = jnp.full((ncol,), 2.0)
    q0 = ice_enthalpy(T0, S[:, None])
    T_bottom = jnp.full((ncol,), constants.T_freeze_ocean)
    _, T_new, _ = bitz_lipscomb_conduction_step(
        q0, h, S, jnp.array([80.0, -80.0]), T_bottom, 3600.0,
    )
    assert float(T_new[0, 0]) > float(T0[0, 0])   # warmed
    assert float(T_new[1, 0]) < float(T0[1, 0])   # cooled


def test_conduction_step_ad_safe():
    """Gradients through the implicit conduction solve are finite."""
    n, ncol = 4, 2
    T0 = jnp.stack([jnp.linspace(_Tf - 10.0, _Tf - 2.0, n)] * ncol)
    S = jnp.full((ncol,), 4.0)
    h = jnp.full((ncol,), 1.5)
    q0 = ice_enthalpy(T0, S[:, None])
    T_bottom = jnp.full((ncol,), constants.T_freeze_ocean)

    def loss(flux_scalar):
        F = jnp.full((ncol,), flux_scalar)
        q_new, _, _ = bitz_lipscomb_conduction_step(q0, h, S, F, T_bottom, 3600.0)
        return jnp.sum(q_new ** 2)

    g = jax.grad(loss)(10.0)
    assert jnp.isfinite(g)


def test_si3_column_step_reverse_mode_finite_on_open_water_and_bare_ice():
    """Review F3: open water (no ice, no lead freezing) and snow-free ice hit
    0/0 in DISCARDED where-branches of the SI3 chain (new-ice thickness,
    layer-enthalpy remap); the reverse pass must stay finite there."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        ice_enthalpy_from_temperature,
        option2_salinity_profile,
        si3_column_step_arrays,
        snow_enthalpy_from_temperature,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    si3 = SI3ThermoConfig()

    def a(*v):
        return jnp.asarray(v, dtype=jnp.float64)

    # columns: open water | snow-free ice | snow-covered ice
    conc, h_i, h_s = a(0.0, 0.9, 0.9), a(0.0, 2.0, 2.0), a(0.0, 0.0, 0.2)
    bulk = a(0.0, 6.3, 6.3)
    sal = option2_salinity_profile(bulk, a(34.7, 34.7, 34.7))
    ti = jnp.stack([a(271.0, 271.0, 271.0), a(257.0, 261.0, 266.0),
                    a(257.0, 261.0, 266.0)])
    ts = jnp.stack([a(271.0, 271.0, 271.0)] * 2 + [a(255.0, 260.0, 266.0)])
    e_i = ice_enthalpy_from_temperature(ti, sal, c) * (h_i > 0)[:, None]
    e_s = snow_enthalpy_from_temperature(ts, c) * (h_s > 0)[:, None]
    state = SI3ColumnArrays(conc, h_i, h_s, a(271.0, 260.0, 260.0), e_i, e_s,
                            bulk, sal, jnp.zeros(3))
    z3 = jnp.zeros(3)
    forcing = SI3SurfaceForcing(
        qns_ice=a(-60.0, -60.0, -60.0), qsr_ice=z3, dqns_ice=a(-7.0, -7.0, -7.0),
        qtr_ice_top=z3, t_bottom=a(271.27, 271.27, 271.27),
        sss=a(34.7, 34.7, 34.7), evaporation=z3, snow_precipitation=z3,
        qprec_ice=z3, qcn_ice_bottom=z3, qsb_ice_bottom=z3, fhld=z3,
        qlead=z3)   # qlead >= 0: no lead freezing, so open water stays ice-free

    def loss(qns, h):
        out = si3_column_step_arrays(state._replace(h_ice=h),
                                     forcing._replace(qns_ice=qns),
                                     3600.0, c, si3=si3).exit
        return (jnp.sum(out.h_ice) + jnp.sum(out.concentration)
                + 1e-9 * jnp.sum(out.e_ice) + 1e-9 * jnp.sum(out.e_snow))

    out = si3_column_step_arrays(state, forcing, 3600.0, c, si3=si3).exit
    assert float(out.concentration[0]) == 0.0 and float(out.h_snow[1]) == 0.0
    g_q, g_h = jax.grad(loss, argnums=(0, 1))(forcing.qns_ice, state.h_ice)
    assert bool(jnp.all(jnp.isfinite(g_q))), g_q
    assert bool(jnp.all(jnp.isfinite(g_h))), g_h
    assert float(jnp.abs(g_h[1:]).max()) > 0.0     # ice columns stay live

    def state_loss(value):
        result = si3_column_step_arrays(
            value, forcing, 3600.0, c, si3=si3
        ).exit
        return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(result))

    state_gradient = jax.grad(state_loss)(state)
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree.leaves(state_gradient)
    ), state_gradient


def test_si3_column_step_shared_forcing_gradients_finite_with_open_water_batch():
    """All seven shared inputs remain finite with 8 open-water columns."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        ice_enthalpy_from_temperature,
        option2_salinity_profile,
        si3_column_step_arrays,
        snow_enthalpy_from_temperature,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    ncol = 64
    live = jnp.arange(ncol) >= 8
    zero = jnp.zeros(ncol, dtype=jnp.float64)
    h_ice = jnp.where(live, 2.0, 0.0)
    h_snow = jnp.where(live, 0.2, 0.0)
    S_bulk = jnp.where(live, 6.3, 0.0)
    sss = jnp.full((ncol,), 34.7, dtype=jnp.float64)
    S_layers = option2_salinity_profile(S_bulk, sss)
    e_ice = ice_enthalpy_from_temperature(
        jnp.full((ncol, 3), 260.0, dtype=jnp.float64), S_layers, c
    ) * live[:, None]
    e_snow = snow_enthalpy_from_temperature(
        jnp.full((ncol, 3), 260.0, dtype=jnp.float64), c
    ) * live[:, None]

    def loss(conc_live, qns, qsr, dqns, qtr, t_bottom, snow_precip):
        concentration = jnp.where(live, conc_live, 0.0)
        state = SI3ColumnArrays(
            concentration, h_ice, h_snow, jnp.where(live, 260.0, c.T0),
            e_ice, e_snow, S_bulk, S_layers, zero,
        )
        forcing = SI3SurfaceForcing(
            qns_ice=jnp.full((ncol,), qns),
            qsr_ice=jnp.full((ncol,), qsr),
            dqns_ice=jnp.full((ncol,), dqns),
            qtr_ice_top=jnp.full((ncol,), qtr),
            t_bottom=jnp.full((ncol,), t_bottom),
            sss=sss,
            evaporation=zero,
            snow_precipitation=jnp.full((ncol,), snow_precip),
            qprec_ice=zero,
            qcn_ice_bottom=zero,
            qsb_ice_bottom=zero,
            fhld=zero,
            qlead=zero,
        )
        out = si3_column_step_arrays(
            state, forcing, 3600.0, c, si3=SI3ThermoConfig()
        ).exit
        return jnp.sum(out.h_ice * out.concentration)

    args = (0.9, -60.0, 0.0, -7.0, 0.0, 271.27, 0.0)
    gradients = jax.grad(loss, argnums=tuple(range(7)))(*args)
    assert all(bool(jnp.isfinite(value)) for value in gradients), gradients


def _degenerate_four_column_step(*, compiled=False):
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    concentration = jnp.asarray([0.0, 0.0, 5.0e-11, 5.0e-11])
    h_ice = jnp.asarray([0.0, 0.0, 0.5, 0.5])
    qlead = jnp.asarray([0.0, -40.0, 0.0, -40.0])
    zero = jnp.zeros((4,), dtype=jnp.float64)
    zero_layers = jnp.zeros((4, 3), dtype=jnp.float64)
    state = SI3ColumnArrays(
        concentration, h_ice, zero, jnp.full((4,), c.T0),
        zero_layers, zero_layers, zero, zero_layers, zero,
    )
    forcing = SI3SurfaceForcing(
        qns_ice=zero, qsr_ice=zero, dqns_ice=zero, qtr_ice_top=zero,
        t_bottom=jnp.full((4,), 271.27), sss=jnp.full((4,), 34.7),
        evaporation=zero, snow_precipitation=zero, qprec_ice=zero,
        qcn_ice_bottom=zero, qsb_ice_bottom=zero, fhld=zero, qlead=qlead,
    )
    step = lambda value: si3_column_step_arrays(  # noqa: E731
        value, forcing, 3600.0, c, si3=SI3ThermoConfig()
    ).exit
    if compiled:
        step = jax.jit(step)
    first = step(state)
    return first, step(first)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_degenerate_four_column_two_step_forward_bits_match_lane(compiled):
    """Pin 920c61d884 on open-water and sub-epsilon columns exactly."""
    expected = (
        {
            "concentration": (0.0, 1.0477509710156091e-05, 5e-11,
                              1.0477559710156091e-05),
            "h_ice": (0.0, 0.05, 0.0, 0.050002145361805514),
            "h_snow": (0.0, 0.0, 0.0, 0.0),
            "T_surface": (273.15, 273.15, 273.14510067184534,
                          273.14510067184534),
            "e_ice": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      305700631.51037043, 305769255.1803446,
                      306405736.09880507, 305700631.51037043,
                      305769255.1803446, 306405736.09880507),
            "e_snow": (0.0,) * 12,
            "S_bulk": (0.1, 0.1, 0.1, 0.1),
            "S_layers": (0.1,) * 12,
            "age_volume": (0.0, 0.037719034956561925, 1.8e-07,
                           0.03771921495656193),
        },
        {
            "concentration": (0.0, 2.0955019420312182e-05, 0.0,
                              2.0955069420312182e-05),
            "h_ice": (0.0, 0.02500000000000001, 0.0,
                      0.04984625211138035),
            "h_snow": (0.0, 0.0, 0.0, 0.0),
            "T_surface": (273.15, 273.1446003491123, 273.15,
                          271.59905056567794),
            "e_ice": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      307618229.39976406, 307803557.08903724,
                      308172027.764088),
            "e_snow": (0.0,) * 12,
            "S_bulk": (0.1, 0.1, 0.1, 0.1),
            "S_layers": (0.1,) * 12,
            "age_volume": (0.0, 0.11315710486968578, 1.8e-07,
                           0.11315746486968578),
        },
    )
    if compiled:
        expected[0].update({
            "concentration": (0.0, 1.0477509710156093e-05, 5e-11,
                              1.0477559710156093e-05),
            "h_ice": (0.0, 0.049999999999999996, 0.0,
                      0.05000214536180551),
            "age_volume": (0.0, 0.03771903495656193, 1.8e-07,
                           0.037719214956561935),
        })
        expected[1].update({
            "concentration": (0.0, 2.0955019420312186e-05, 0.0,
                              2.0955069420312186e-05),
            "h_ice": (0.0, 0.025, 0.0, 0.04984625211138034),
            "age_volume": (0.0, 0.1131571048696858, 1.8e-07,
                           0.11315746486968581),
        })

    for actual, wanted in zip(_degenerate_four_column_step(compiled=compiled),
                              expected, strict=True):
        for name, leaf in zip(actual._fields, actual, strict=True):
            np.testing.assert_array_equal(
                np.asarray(leaf).ravel(), np.asarray(wanted[name]))


def test_ice_forming_open_water_gradients_match_main_and_lane_difference():
    """Formation gradients follow the lane forward and agreed main values."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    zero = jnp.zeros((1,), dtype=jnp.float64)
    zero_layers = jnp.zeros((1, 3), dtype=jnp.float64)

    def run(concentration, h_ice, qlead, t_bottom, sss):
        state = SI3ColumnArrays(
            jnp.asarray([concentration]), jnp.asarray([h_ice]), zero,
            jnp.asarray([c.T0]), zero_layers, zero_layers,
            zero, zero_layers, zero,
        )
        forcing = SI3SurfaceForcing(
            qns_ice=zero, qsr_ice=zero, dqns_ice=zero,
            qtr_ice_top=zero, t_bottom=jnp.asarray([t_bottom]),
            sss=jnp.asarray([sss]), evaporation=zero,
            snow_precipitation=zero, qprec_ice=zero,
            qcn_ice_bottom=zero, qsb_ice_bottom=zero, fhld=zero,
            qlead=jnp.asarray([qlead]),
        )
        return si3_column_step_arrays(
            state, forcing, 3600.0, c, si3=SI3ThermoConfig()
        ).exit

    def ice_volume(qlead, t_bottom, sss):
        out = run(0.0, 0.0, qlead, t_bottom, sss)
        return out.h_ice[0] * out.concentration[0]

    args = (-40.0, 271.27, 34.7)
    gradient = jax.grad(ice_volume, argnums=(0, 1, 2))(*args)
    main_gradient = np.asarray((
        -1.3096887137695114e-08,
        8.220414129866561e-07,
        4.470021523873651e-08,
    ))
    np.testing.assert_allclose(np.asarray(gradient), main_gradient, rtol=2e-15)

    for index, epsilon in enumerate((1.0e-3, 1.0e-4, 1.0e-4)):
        plus = list(args)
        minus = list(args)
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference = (
            ice_volume(*plus) - ice_volume(*minus)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            float(gradient[index]), float(finite_difference), rtol=1e-7
        )

    concentration_gradient = jax.grad(
        lambda concentration: run(concentration, 0.5, *args).concentration[0]
    )(5.0e-11)
    # Stay wholly below epsi10 while avoiding subtractive cancellation in the
    # source-rounded forward expression.
    epsilon = 4.0e-11
    concentration_difference = (
        run(5.0e-11 + epsilon, 0.5, *args).concentration[0]
        - run(5.0e-11 - epsilon, 0.5, *args).concentration[0]
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        float(concentration_gradient), float(concentration_difference),
        rtol=5e-11, atol=0.0,
    )

    # The dh stage clears a genuinely empty column, so equality of its zero
    # output with the zero entry value must not manufacture an identity path.
    def empty_outputs(concentration):
        out = run(concentration, 0.0, 0.0, args[1], args[2])
        return jnp.stack((out.concentration[0], out.age_volume[0]))

    np.testing.assert_array_equal(
        np.asarray(jax.jacrev(empty_outputs)(0.0)), np.zeros(2),
    )

    age_identity = jax.grad(
        lambda age: si3_column_step_arrays(
            SI3ColumnArrays(
                zero, zero, zero, jnp.asarray([c.T0]), zero_layers,
                zero_layers, zero, zero_layers, jnp.asarray([age]),
            ),
            SI3SurfaceForcing(
                qns_ice=zero, qsr_ice=zero, dqns_ice=zero,
                qtr_ice_top=zero, t_bottom=jnp.asarray([args[1]]),
                sss=jnp.asarray([args[2]]), evaporation=zero,
                snow_precipitation=zero, qprec_ice=zero,
                qcn_ice_bottom=zero, qsb_ice_bottom=zero, fhld=zero,
                qlead=zero,
            ),
            3600.0, c, si3=SI3ThermoConfig(),
        ).exit.age_volume[0]
    )(0.0)
    assert float(age_identity) == 1.0

    thickness_gradient = jax.grad(
        lambda h_ice: run(5.0e-11, h_ice, *args).h_ice[0]
    )(0.5)
    thickness_difference = (
        run(5.0e-11, 0.5 + 1.0e-5, *args).h_ice[0]
        - run(5.0e-11, 0.5 - 1.0e-5, *args).h_ice[0]
    ) / 2.0e-5
    np.testing.assert_allclose(
        float(thickness_gradient), float(thickness_difference), rtol=1e-6,
    )


def test_si3_column_step_mixed_dtypes_match_lane_and_differentiate():
    """The reverse-only guards preserve 920c61d884 dtype promotion."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    zero64 = jnp.zeros((1,), dtype=jnp.float64)
    zero_layers64 = jnp.zeros((1, 3), dtype=jnp.float64)
    state = SI3ColumnArrays(
        jnp.asarray([5.0e-11], dtype=jnp.float64),
        jnp.asarray([0.5], dtype=jnp.float64),
        zero64,
        jnp.asarray([c.T0], dtype=jnp.float64),
        zero_layers64,
        zero_layers64,
        zero64,
        zero_layers64,
        zero64,
    )

    def forcing(qlead_dtype):
        return SI3SurfaceForcing(
            qns_ice=zero64, qsr_ice=zero64, dqns_ice=zero64,
            qtr_ice_top=zero64, t_bottom=jnp.asarray([271.27]),
            sss=jnp.asarray([34.7]), evaporation=zero64,
            snow_precipitation=zero64, qprec_ice=zero64,
            qcn_ice_bottom=zero64, qsb_ice_bottom=zero64, fhld=zero64,
            qlead=jnp.asarray([-40.0], dtype=qlead_dtype),
        )

    def step(value, surface_forcing):
        return si3_column_step_arrays(
            value, surface_forcing, 3600.0, c, si3=SI3ThermoConfig(),
        )

    mixed = step(state, forcing(jnp.float32))
    all64 = step(state, forcing(jnp.float64))
    for mixed_stage, all64_stage in (
        (mixed.post_do, all64.post_do), (mixed.exit, all64.exit),
    ):
        for mixed_leaf, all64_leaf in zip(mixed_stage, all64_stage, strict=True):
            np.testing.assert_array_equal(
                np.asarray(mixed_leaf), np.asarray(all64_leaf),
            )

    state32_concentration = state._replace(
        concentration=state.concentration.astype(jnp.float32),
    )
    gradient = jax.grad(
        lambda concentration: step(
            state32_concentration._replace(concentration=concentration),
            forcing(jnp.float64),
        ).exit.concentration[0]
    )(state32_concentration.concentration)
    assert gradient.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(gradient)))


_SI3_GRADIENT_SWEEP_ORACLE_B85 = (
    "c-rmTZA?>V6bEoDlWj&Y_>wV|DP3?1NTIkbkb2u9O&}YN31*Ik3KIrSWW4N+$QWzDz(k;eL#hw~1!RtGWTHs#EnuAT&Hx!TfKUsHAVSI<3p&S#-8wK0b73tP{=Y9LCGfOOp7PH*Cp9tx0Dy&Sgg;;3nLGD$OV6#NgdML!82;!x;%War$HT_oEJI4+jsO7YH+O>E8xB`_mhjc7m(P6V_u*6LRPfFvB4Ph_#X`wDm|Yu_B3}H~zUYbjBD^3U8tF2xnErD~O#gxJog0Pp<CMjR+#XpJ^XYp2(_Tk~pM%08<g;9dk~506udLN{e;qXI_fc#i-J)LUR`o}1LP-S6s%q-%mhz_m{-5%kCCjY$UZNk@<fo4(nO&=-Yv?WhGrx!U3Vg|y!9uSF!A8-s+_YC$YtQY8C~NO=d`GD3^$hPR<t69xE9r`x1^?OlvjS%JJOBUl&UU}k`C8VRFpXM(ONZP+?f`O!iTw@$04z+{@BG_-2LJ%{lRMb&fFWVdhB+I|G1%|ut>0-VkL%d4Q43ZCMK*`#MHB8n=h|h4zdz%-0b*&ML!nxmyL+~^ZHYQ`M^4uk-!IP|JRQzEY8rDdJ8T~rJr;G3XC6s$go&*;BRSM!StrldI!kBYm9n%<W^*x$r0YY@!Ojc&mlN~liX(TBJFxI_2m2iW0ANzzVJO(|001yEcd*|9L&BU5b2gY`u;0;Jzw>5uyHS8fEie;EZgiWp5z3Ai0&X_RiK+ij9G$11<9gB9G>a|Je$ErC*#(1gH<GSCZRH=!Shc*v6KTH$hFvB-wng6Ra;c{ZyDr@=l_pZ_Y@Mo&Piu7cU4Ff;>y#~xBwb5)m~eO3g<`*h+(GUDV{-@l9RL7eQr}@H*zW)UFf(_s-vL9yoDFj}m}9Wt(ObXcdTxEcw?-}aB*pSXhl!eKuh}WS-c&#1xgARGq71f|_H$l9HE!yY`;c^fz<OiH5!Q9y6*IY|IP4l3`$aHO@6t)F94$=EmmZ_CpWmDP@RYI6Ipv0lr~^$c-6UP#O5erttc$^Z2f2gX0mkMI_B#Loz@)yzP_W+t0AObBV88SDMN7}+6GHZ?aSzrAKVD=jwRrB=8Q+H__GD#E-MiX(h;{9a+~#zDimsgMOv(9L^;Yh<FI8gIOXalgvt6ENuRT{)@p7D1$g)aZ=Wh=DKt<-?rsz7~ZYbt#n6tqggZ+-)`kjOT<qyspwP1K~_o{l6F~TcDwd-nA=d|zdy#2)YcM8&E14(Ru?dSYG71yyt9!SzP;be@dgms74bM>V%udqI{{k~+V$YqGyppdo3ON%I_XOH*jsTDfsltH4~PV$yvlCDS1IX0W>lCj@G?jU!7vAKi&4gdf!sqZip>~{bFn3+4+?@WdLj(%azhB+I|G1%|ut=~}%NBs0rm4;}~e=y-RnLD8}`;qZyALq{B`yH3*=OhnY@h<pB?r?d-UFDN`L)S&s#=prXb0<3Po7XcZb4R4wObz^K#~<UYZi^i+riwXr;`_n6bBC;0v-HGd-q2Oy6Mwhwcq;Zg$Q|SkFgAB)yWjZ>Wj4Sg"
)


def test_si3_column_step_full_gradient_sweep_matches_main_or_lane_difference():
    """All exit/input leaf pairs match the reviewed main/FD oracle, eager/JIT."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    concentrations = (0.0, 1e-12, 5e-11, 1e-10, 2e-10, 0.5)
    thicknesses = (0.0, 0.5)
    qleads = (0.0, -40.0)

    def vector(concentration, h_ice, qlead):
        return jnp.asarray((
            concentration, h_ice, 0.0, c.T0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 271.27, 34.7, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, qlead, 1.0,
        ), dtype=jnp.float64)

    cases = tuple(
        (concentration, h_ice, qlead)
        for concentration in concentrations
        for h_ice in thicknesses
        for qlead in qleads
    )
    vectors = jnp.stack([vector(*case) for case in cases])

    def step(values):
        scalar = lambda index: values[index:index + 1]
        layers = lambda index: jnp.full((1, 3), values[index])
        state = SI3ColumnArrays(
            scalar(0), scalar(1), scalar(2), scalar(3), layers(4), layers(5),
            scalar(6), layers(7), scalar(8),
        )
        forcing = SI3SurfaceForcing(
            qns_ice=scalar(9), qsr_ice=scalar(10), dqns_ice=scalar(11),
            qtr_ice_top=scalar(12), t_bottom=scalar(13), sss=scalar(14),
            evaporation=scalar(15), snow_precipitation=scalar(16),
            qprec_ice=scalar(17), qcn_ice_bottom=scalar(18),
            qsb_ice_bottom=scalar(19), fhld=scalar(20), qlead=scalar(21),
            frq_m=scalar(22), sst_m=None,
        )
        result = si3_column_step_arrays(
            state, forcing, 3600.0, c, si3=SI3ThermoConfig(),
        ).exit
        return jnp.stack([jnp.sum(leaf) for leaf in result])

    oracle = np.frombuffer(
        zlib.decompress(base64.b85decode(_SI3_GRADIENT_SWEEP_ORACLE_B85)),
        dtype="<f8",
    ).reshape(24, 9, 23)
    eager = jax.vmap(jax.jacrev(step))(vectors)
    compiled = jax.jit(jax.vmap(jax.jacrev(step)))(vectors)
    for mode, actual in (("eager", eager), ("jit", compiled)):
        actual = np.asarray(actual)
        mode_oracle = oracle.copy()
        if mode == "eager":
            # The source-rounded h(concentration) slope at this active point is
            # exactly one eager float64 ulp; XLA's fused evaluation gives the
            # separately reviewed compiled value stored in the oracle.
            mode_oracle[18, 1, 0] = 2.0 ** -21
        relative = np.abs(actual - mode_oracle) / np.maximum(
            np.maximum(np.abs(actual), np.abs(mode_oracle)),
            np.finfo(np.float64).tiny,
        )
        for case, worst in zip(cases, np.max(relative, axis=(1, 2)), strict=True):
            print(
                f"gradient-sweep {mode} conc={case[0]:.12g} "
                f"h_ice={case[1]:.1f} qlead={case[2]:.1f} worst_rel={worst:.3e}"
            )
        assert bool(np.all(np.isfinite(actual))), mode
        assert float(np.max(relative)) <= 1.0e-6, mode
