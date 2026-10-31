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


def test_ice_forming_open_water_gradients_match_main_and_finite_difference():
    """Open-water formation keeps every gradient present on main edd34c2446."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    zero = jnp.zeros((1,), dtype=jnp.float64)
    zero_layers = jnp.zeros((1, 3), dtype=jnp.float64)

    def run(concentration, qlead, t_bottom, sss):
        state = SI3ColumnArrays(
            jnp.asarray([concentration]), zero, zero,
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
        out = run(0.0, qlead, t_bottom, sss)
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
        lambda concentration: run(concentration, *args).concentration[0]
    )(5.0e-11)
    assert float(concentration_gradient) == 1.0


def test_cumulative_remap_reverse_mode_finite_at_zero_thickness():
    """Review F3 on the lane's live remap: a zero-total column must have a
    finite zero cotangent without reviving main's deleted duplicate remap."""
    from legoesm.ice import bitz_lipscomb as bl

    en = jnp.asarray([3.0e8, 2.9e8, 2.8e8])

    def remap(thickness):
        h_old = list(thickness)
        content_old = list(en * thickness)
        return bl._cumulative_remap(h_old, content_old, 3, bl.nemo_source_round)

    # A sub-epsilon remap with material confined to the last old layer is the
    # discarded-column case that exposes NaN cotangents when the guard is
    # removed.  Evenly splitting a shared total (the former fixture) was
    # vacuous: its unguarded gradient was already finite.
    g = jax.grad(lambda h: jnp.sum(remap(h)))(
        jnp.asarray([0.0, 0.0, 1.0e-300], dtype=jnp.float64)
    )
    assert bool(jnp.all(jnp.isfinite(g))), g
    assert bool(jnp.all(g == 0.0)), g
