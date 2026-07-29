from __future__ import annotations

import jax.numpy as jnp
from legoesm.diagnostics.water_budget import (
    area_integral,
    atm_moisture_residual,
    ice_water_content,
    land_water_content_multilayer,
    land_water_content_slab,
    runoff_conservation_residual,
    water_inventory_residual,
)

from legoesm import constants as _const

_DT = 3600.0
_E0 = 4.0e-5   # evap [kg/m^2/s], +up
_P0 = 3.5e-5   # precip [kg/m^2/s], +down


def _balanced_cwv_fields(area):
    # The atmosphere truly drains at (E - P): <CWV> rises by (E0 - P0)*dt.
    cwv_prev = jnp.full(area.shape, 25.0)
    cwv_now = cwv_prev + (_E0 - _P0) * _DT
    return cwv_prev, cwv_now


def test_atm_moisture_residual_balanced_is_zero():
    area = jnp.ones((4, 8))
    cwv_prev, cwv_now = _balanced_cwv_fields(area)
    evap = jnp.full((4, 8), _E0)
    precip = jnp.full((4, 8), _P0)
    r = atm_moisture_residual(cwv_now, cwv_prev, evap, precip, area, _DT)
    assert abs(float(r)) < 1e-12


def test_atm_moisture_residual_flags_inflated_precip_c1():
    # C1: the atm rained the TRUE P0 (CWV fields unchanged), but the coupler
    # is handed a precip rate 4e5x too large. The independent CWV witness no
    # longer matches the delivered precip -> residual explodes.
    area = jnp.ones((4, 8))
    cwv_prev, cwv_now = _balanced_cwv_fields(area)
    evap = jnp.full((4, 8), _E0)
    precip_bug = jnp.full((4, 8), _P0 * 4.0e5)
    r = atm_moisture_residual(cwv_now, cwv_prev, evap, precip_bug, area, _DT)
    # ~ P0*(4e5 - 1) ~ 14 kg/m^2/s, >> any sane ~1e-3 tripwire threshold.
    assert abs(float(r)) > 1.0


def test_atm_moisture_residual_ignores_land_storage_no_false_flag():
    # Rejection-test analogue: a cell where the land is STORING water (soil
    # moistening) must NOT perturb the atmosphere-only budget. Encode it as a
    # spatially heterogeneous E/P whose AREA MEANS match the CWV drain; the
    # residual keys off <E>-<P>, never any storage term, so it stays ~0.
    area = jnp.ones((2, 2))
    cwv_prev = jnp.full((2, 2), 30.0)
    # heterogeneous surface fluxes (a 'land' cell with P>E, an 'ocean' cell
    # with E>P) whose means are E0, P0:
    evap = jnp.array([[2.0e-5, 6.0e-5], [_E0, _E0]])
    precip = jnp.array([[6.0e-5, 1.0e-5], [_P0, _P0]])
    cwv_now = cwv_prev + (float(jnp.mean(evap)) - float(jnp.mean(precip))) * _DT
    r = atm_moisture_residual(cwv_now, cwv_prev, evap, precip, area, _DT)
    assert abs(float(r)) < 1e-12


def test_runoff_conservation_residual_balanced_is_zero():
    atm_area = jnp.ones((3,))
    ocean_area = jnp.ones((3,))
    river = jnp.array([1.0, 2.0, 3.0])   # exported on atm grid
    owet = jnp.array([1.0, 1.0, 1.0])    # all-wet: nothing discarded
    applied = river * owet
    r = runoff_conservation_residual(river, atm_area, applied, ocean_area)
    assert abs(float(r)) < 1e-12


def test_runoff_conservation_residual_flags_wet_mask_discard_m2():
    # M2: interior cell (index 1) is fully-dry ocean; its runoff is gated out
    # by the wet mask and silently lost. Residual = the discarded amount.
    atm_area = jnp.ones((3,))
    ocean_area = jnp.ones((3,))
    river = jnp.array([1.0, 2.0, 3.0])
    owet = jnp.array([1.0, 0.0, 1.0])
    applied = river * owet
    r = runoff_conservation_residual(river, atm_area, applied, ocean_area)
    assert abs(float(r) - 2.0) < 1e-12


def test_area_integral_matches_hand_sum():
    field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    area = jnp.array([[0.5, 0.5], [2.0, 1.0]])
    r = area_integral(field, area)
    assert abs(float(r) - (0.5 + 1.0 + 6.0 + 4.0)) < 1e-12


_A = 1.0e10  # cell area [m^2]


def test_water_inventory_residual_balanced_is_zero():
    # Atm rains f_water*P*dt over one cell; that water is delivered to the ocean
    # (F_ocean = f_water*P*A). The store integral drops by exactly that flux*dt,
    # so R = d/dt(store) + F_ocean ~ 0.
    dt = 3600.0
    p = 3.0e-5
    f_water = 0.7
    flux_into_ocean = f_water * p * _A          # [kg/s], +into ocean
    store_prev = 5.0e14
    store_now = store_prev - flux_into_ocean * dt
    r = water_inventory_residual(store_now, store_prev, flux_into_ocean, dt)
    assert abs(float(r)) < 1e-3


def test_water_inventory_residual_flags_h2_ice_precip_leak():
    # H2: the atm rained f_ice*P over the ice fraction (W_atm dropped), but the
    # frozen precip entered NEITHER W_ice NOR F_ocean (dropped). The store
    # integral falls with no compensating ocean flux -> R = -(discarded rate).
    dt = 3600.0
    p = 3.0e-5
    f_ice = 0.4
    discarded = f_ice * p * _A                  # [kg/s] silently lost
    store_prev = 5.0e14
    store_now = store_prev - discarded * dt      # atm drained, nothing gained
    r = water_inventory_residual(store_now, store_prev, 0.0, dt)
    assert abs(float(r) - (-discarded)) < 1e-3
    assert abs(float(r)) > 1.0


def test_water_inventory_residual_land_storage_no_false_flag():
    # A fully-land cell legitimately STORES water (soil moistens 4 kg/m2): W_atm
    # falls by 4, W_land rises by 4, no ocean flux. Because storage tendencies
    # are IN the budget, the store INTEGRAL is unchanged -> R ~ 0 (no false
    # land-storage flag). Exercises land_water_content_slab + the residual.
    dt = 3600.0
    area = jnp.array([_A, _A])
    f_land = jnp.array([1.0, 1.0])
    w_atm_prev = jnp.array([10.0, 10.0])
    w_land_prev = land_water_content_slab(
        jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), f_land)
    store_prev = area_integral(w_atm_prev + w_land_prev, area)
    w_atm_now = jnp.array([6.0, 10.0])           # cell 0 rained 4 kg/m2
    w_land_now = land_water_content_slab(
        jnp.array([4.0, 0.0]), jnp.array([0.0, 0.0]), f_land)
    store_now = area_integral(w_atm_now + w_land_now, area)
    r = water_inventory_residual(store_now, store_prev, 0.0, dt)
    assert abs(float(r)) < 1e-3


def test_ice_water_content_fraction_of_water_and_snow():
    # concentration is ice-fraction-OF-WATER; whole-cell weight is f_water.
    h_ice = jnp.array([2.0])
    conc = jnp.array([0.5])
    f_water = jnp.array([0.5])
    w = ice_water_content(h_ice, conc, f_water)
    assert abs(float(w[0]) - _const.rho_ice * 2.0 * 0.5 * 0.5) < 1e-6
    w2 = ice_water_content(h_ice, conc, f_water, h_snow=jnp.array([0.1]))
    assert abs(float(w2[0]) - (_const.rho_ice * 2.0 * 0.5 * 0.5
                               + _const.rho_snow * 0.1 * 0.5 * 0.5)) < 1e-6


def test_ice_water_content_sums_itd_categories():
    # Per-category thickness/concentration (trailing cat axis) collapse to a
    # single per-cell volume; f_water scalar per cell.
    h_ice = jnp.array([[1.0, 3.0]])
    conc = jnp.array([[0.2, 0.1]])
    f_water = jnp.array([1.0])
    w = ice_water_content(h_ice, conc, f_water)
    assert abs(float(w[0]) - _const.rho_ice * (1.0 * 0.2 + 3.0 * 0.1)) < 1e-6


def test_land_water_content_multilayer_matches_hand_integral():
    theta = jnp.array([[0.3, 0.3, 0.3]])
    dz = jnp.array([0.1, 0.2, 0.3])
    snow = jnp.array([5.0])
    f_land = jnp.array([1.0])
    w = land_water_content_multilayer(
        theta, dz, snow, f_land, surface_water=jnp.array([0.01]))
    expect = (0.3 * (0.1 + 0.2 + 0.3) * _const.rho_water
              + 5.0 + 0.01 * _const.rho_water)
    assert abs(float(w[0]) - expect) < 1e-6


def test_land_water_content_slab_fraction_weight():
    # Half-land cell: only f_land of the per-land-area bucket+snow counts.
    w = land_water_content_slab(
        jnp.array([20.0]), jnp.array([5.0]), jnp.array([0.5]))
    assert abs(float(w[0]) - 0.5 * 25.0) < 1e-6


def test_area_integral_routes_through_global_sum_when_distributed(monkeypatch):
    # Distributed WIRING (non-vacuous): area_integral must route its SUM through
    # the AD-safe global_sum_if_distributed collective. Simulate a 2-shard
    # reduction by patching the collective (in the water_budget namespace) to
    # DOUBLE its input; the returned integral must then be 2x the rank-local sum.
    import legoesm.diagnostics.water_budget as wb
    monkeypatch.setattr(wb, "global_sum_if_distributed", lambda x: x * 2.0)
    field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    area = jnp.array([[0.5, 0.5], [2.0, 1.0]])
    local = 0.5 + 1.0 + 6.0 + 4.0
    r = wb.area_integral(field, area)
    assert abs(float(r) - 2.0 * local) < 1e-9


def test_area_integral_serial_is_rank_local_byte_identical():
    # Serial path: the real global_sum_if_distributed is identity on a single
    # process (is_multi_process() False), so the wrap is a no-op and the value
    # is byte-identical to the rank-local integral.
    import legoesm.diagnostics.water_budget as wb
    field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    area = jnp.array([[0.5, 0.5], [2.0, 1.0]])
    assert float(wb.area_integral(field, area)) == float(jnp.sum(field * area))


def test_atm_moisture_residual_reduces_globally_when_distributed(monkeypatch):
    # Tripwire A WIRING (non-vacuous): must form its area means from GLOBALLY
    # reduced sums. Patch is_multi_process True and the collective to a call-
    # recording identity; assert it is invoked (numerator + denominator of the
    # weighted means) and the identity collective reproduces the balanced (~0)
    # serial residual (no serial-value regression).
    import legoesm.diagnostics.water_budget as wb
    calls = {"n": 0}
    def spy(x):
        calls["n"] += 1
        return x
    monkeypatch.setattr(wb, "is_multi_process", lambda: True)
    monkeypatch.setattr(wb, "global_sum_if_distributed", spy)
    area = jnp.ones((4, 8))
    cwv_prev, cwv_now = _balanced_cwv_fields(area)
    evap = jnp.full((4, 8), _E0)
    precip = jnp.full((4, 8), _P0)
    r = atm_moisture_residual(cwv_now, cwv_prev, evap, precip, area, _DT)
    assert calls["n"] > 0
    assert abs(float(r)) < 1e-12
