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


def test_inactive_column_guard_stops_h_ice_cotangent():
    """The helper-level plant remains bound to its cotangent mask."""
    from legoesm.ice import bitz_lipscomb as bl

    active = jnp.asarray([False] * 8 + [True] * 56)
    h_ice = jnp.linspace(0.0, 2.0, 64, dtype=jnp.float64)
    gradient = jax.grad(
        lambda h: jnp.sum(bl._guard_inactive_columns(h, active, h.dtype))
    )(h_ice)
    assert bool(jnp.all(gradient[:8] == 0.0)), gradient
    assert bool(jnp.all(gradient[8:] == 1.0)), gradient


def test_cumulative_remap_reverse_mode_finite_at_zero_thickness():
    """A discarded sub-epsilon remap has a finite zero cotangent."""
    from legoesm.ice import bitz_lipscomb as bl

    enthalpy = jnp.asarray([3.0e8, 2.9e8, 2.8e8])

    def remap(thickness):
        h_old = list(thickness)
        content_old = list(enthalpy * thickness)
        return bl._cumulative_remap(
            h_old, content_old, 3, bl.nemo_source_round,
        )

    gradient = jax.grad(lambda h: jnp.sum(remap(h)))(
        jnp.asarray([0.0, 0.0, 1.0e-300], dtype=jnp.float64)
    )
    assert bool(jnp.all(jnp.isfinite(gradient))), gradient
    assert bool(jnp.all(gradient == 0.0)), gradient


def test_si3_column_step_remap_reverse_mode_finite_at_tiny_snow():
    """The live production remap keeps an active tiny-snow gradient finite."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    zero = jnp.zeros((1,), dtype=jnp.float64)
    zero_layers = jnp.zeros((1, 3), dtype=jnp.float64)
    forcing = SI3SurfaceForcing(
        qns_ice=zero, qsr_ice=zero, dqns_ice=zero, qtr_ice_top=zero,
        t_bottom=jnp.asarray([271.27]), sss=jnp.asarray([34.7]),
        evaporation=zero, snow_precipitation=zero, qprec_ice=zero,
        qcn_ice_bottom=zero, qsb_ice_bottom=zero, fhld=zero, qlead=zero,
    )

    def step_exit(thickness):
        state = SI3ColumnArrays(
            jnp.asarray([0.5]), jnp.asarray([0.5]), thickness,
            jnp.asarray([c.T0]), zero_layers,
            jnp.asarray([[0.0, 0.0, -1.0e8]]), zero, zero_layers, zero,
        )
        return si3_column_step_arrays(
            state, forcing, 3600.0, c, si3=SI3ThermoConfig(),
        ).exit

    jacobian = jax.jacrev(step_exit)(jnp.asarray([3.0e-200]))
    for field, leaf in zip(jacobian._fields, jacobian):
        assert bool(jnp.all(jnp.isfinite(leaf))), f"{field}: {leaf}"
    np.testing.assert_array_equal(np.asarray(jacobian.h_snow), np.asarray([[1.0]]))
    for field, leaf in zip(jacobian._fields, jacobian):
        if field != "h_snow":
            np.testing.assert_array_equal(
                np.asarray(leaf), np.zeros_like(np.asarray(leaf)),
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
    expected_post_do = (
        [1.0477559710156091e-05], [0.050002145361805514], [0.0],
        [273.14510067184534],
        [[305700631.51037043, 305769255.1803446, 306405736.09880507]],
        [[0.0, 0.0, 0.0]], [0.1], [[0.1, 0.1, 0.1]], [0.0],
    )
    expected_exit = (*expected_post_do[:-1], [0.03771921495656193])
    for mixed_stage, all64_stage in (
        (mixed.post_do, all64.post_do), (mixed.exit, all64.exit),
    ):
        for mixed_leaf, all64_leaf in zip(mixed_stage, all64_stage, strict=True):
            np.testing.assert_array_equal(
                np.asarray(mixed_leaf), np.asarray(all64_leaf),
            )
    for actual, expected in zip(mixed.post_do, expected_post_do, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for actual, expected in zip(mixed.exit, expected_exit, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

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
    np.testing.assert_array_equal(
        np.asarray(gradient), np.asarray([1.0], dtype=np.float32),
    )

    state32_energy = state._replace(
        e_ice=state.e_ice.astype(jnp.float32),
        age_volume=jnp.asarray([1.25], dtype=jnp.float64),
    )
    dtype_result = step(state32_energy, forcing(jnp.float64))
    assert dtype_result.exit.age_volume.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(dtype_result.exit.age_volume),
        np.asarray([1.2877188920974731], dtype=np.float32),
    )
    age_gradient = jax.grad(
        lambda age: step(
            state32_energy._replace(age_volume=age), forcing(jnp.float64),
        ).exit.age_volume[0]
    )(jnp.asarray([1.25], dtype=jnp.float32))
    assert age_gradient.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(age_gradient), np.asarray([1.0], dtype=np.float32),
    )
