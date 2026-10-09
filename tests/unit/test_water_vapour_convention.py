"""The water-tracer CONVENTION (thermo.py "Conventions", 2026-09-28):
specific quantities on total air mass, kept as is from the reanalysis;
mixing-ratio schemes convert once at entry and once at exit.

Identities pinned here, each exact to float precision:
  * r <-> q maps and their tendency / finite-increment maps round-trip
  * T_v(q) == T (1 + r/eps) / (1 + r) for r = r(q): the exact vapour-only
    identity, so the specific-humidity formula the dycores use is exact
    for the specific tracer (and 0.6 q^2 off if fed a mixing ratio)
  * q_sat == r_sat / (1 + r_sat) at the same (T, p): a saturated parcel in
    the specific convention reads RH = 1 against q_sat and 1/(1+r_sat)
    against r_sat (the tripwire for a scheme still on the old reference)
  * column water sum(q dp)/g equals ERA5's own integral of q on the
    source column (the loader-plus-bookkeeping lock GLM asked for)
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.thermo import (  # noqa: E402
    mixing_ratio_increment_to_specific_humidity,
    mixing_ratio_tendency_to_specific_humidity_tendency,
    mixing_ratio_to_specific_humidity,
    saturation_mixing_ratio,
    saturation_specific_humidity,
    specific_humidity_tendency_to_mixing_ratio_tendency,
    specific_humidity_to_mixing_ratio,
    virtual_temperature,
)
from legoesm.training.era5_to_state import era5_terrain_product

Q = jnp.asarray([1e-4, 2e-3, 1e-2, 2e-2, 3e-2])


def test_r_q_maps_round_trip_and_tendency_maps_are_inverse():
    r = specific_humidity_to_mixing_ratio(Q)
    np.testing.assert_allclose(np.asarray(mixing_ratio_to_specific_humidity(r)), np.asarray(Q),
                               rtol=1e-15, atol=0)
    dq = jnp.asarray(1e-6)
    dr = specific_humidity_tendency_to_mixing_ratio_tendency(Q, dq)
    np.testing.assert_allclose(np.asarray(mixing_ratio_tendency_to_specific_humidity_tendency(r, dr)),
                               1e-6, rtol=1e-14, atol=0)
    # finite increment: exact against the map, and the derivative form is
    # only first order (differs at O(dr) relative)
    dr_fin = jnp.asarray(2e-3)
    exact = mixing_ratio_to_specific_humidity(r + dr_fin) - Q
    np.testing.assert_allclose(np.asarray(mixing_ratio_increment_to_specific_humidity(r, dr_fin)),
                               np.asarray(exact), rtol=1e-13, atol=0)
    first_order = mixing_ratio_tendency_to_specific_humidity_tendency(r, dr_fin)
    assert np.abs(np.asarray(first_order - exact) / np.asarray(exact)).min() > 1e-4


def test_virtual_temperature_specific_form_is_the_exact_identity():
    T = jnp.asarray(300.0)
    r = specific_humidity_to_mixing_ratio(Q)
    eps = constants.epsilon
    exact = T * (1.0 + r / eps) / (1.0 + r)
    np.testing.assert_allclose(np.asarray(virtual_temperature(T, Q)), np.asarray(exact),
                               rtol=1e-14, atol=0)
    # feeding the MIXING ratio to the specific formula is off by ~(1/eps-1) q^2
    wrong = virtual_temperature(T, r)
    err = np.asarray(jnp.abs(wrong - exact) / T)
    assert err[-1] > 3e-4 and err[-1] < 1e-3


def test_saturation_references_and_the_saturated_parcel_tripwire():
    T, p = jnp.asarray(300.0), jnp.asarray(1.0e5)
    r_sat = saturation_mixing_ratio(T, p)
    q_sat = saturation_specific_humidity(T, p)
    np.testing.assert_allclose(float(q_sat), float(r_sat / (1.0 + r_sat)), rtol=1e-14)
    assert float(r_sat) > 0.02             # a moist parcel: the two differ by > 2 %
    q_parcel = q_sat                       # saturated, in the tracer convention
    assert abs(float(q_parcel / q_sat) - 1.0) < 1e-14
    rh_old_reference = float(q_parcel / r_sat)
    # the old reference reads it sub-saturated by exactly 1/(1+r_sat)
    np.testing.assert_allclose(rh_old_reference, 1.0 / (1.0 + float(r_sat)), rtol=1e-14)


def test_column_water_through_the_carry_matches_the_reanalysis_integral():
    """The REAL loader (spectral carry) and the REAL column-integral helper:
    sum(q dp)/g on the loaded state == the trapezoid integral of ERA5's own
    q profile between the same pressures.  Restoring the loader's old
    r = q/(1-q) conversion makes this fail by > 0.5 % (codex 2026-09-28:
    the first draft called neither path)."""
    from legoesm.diagnostics.column_integrals import column_water_vapor
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import ERA5Slice, era5_to_spectral_carry
    grid = create_gaussian_grid(n_max=10)
    n_lat, n_lon = int(np.asarray(grid.lat).shape[0]), int(np.asarray(grid.lon).shape[0])
    plev = np.array([1.0e4, 3.0e4, 5.0e4, 7.0e4, 8.5e4, 1.0e5])
    q_prof = np.array([1e-5, 2e-4, 2e-3, 6e-3, 1.2e-2, 1.8e-2])
    sh = (n_lat, n_lon, plev.size)
    era5 = ERA5Slice(
        T=np.full(sh, 285.0), u=np.zeros(sh), v=np.zeros(sh),
        q=np.broadcast_to(q_prof, sh).copy(),
        p_s=np.full((n_lat, n_lon), 1.0e5), sst=np.full((n_lat, n_lon), 290.0),
        phis=np.zeros((n_lat, n_lon)),
        lat=np.asarray(grid.lat), lon=np.asarray(grid.lon), plev_Pa=plev)
    sigma = create_sigma_coordinate(40, dtype=jnp.float64)
    carry = era5_to_spectral_carry(era5, grid, sigma,
                                   target_phis=era5_terrain_product(era5, grid))
    q_m = jnp.asarray(carry.q_v)
    p_s = jnp.asarray(carry.p_s)
    tpw = np.asarray(column_water_vapor(q_m, p_s, sigma.dsigma))
    p_dense = np.linspace(float(sigma.sigma_half[0]) * 1.0e5, 1.0e5, 20001)
    q_dense = np.interp(np.log(p_dense), np.log(plev), q_prof)
    tpw_ref = float(np.trapezoid(q_dense, p_dense) / constants.g)
    np.testing.assert_allclose(tpw, tpw_ref, rtol=3e-3)
    # the old loader (r = q/(1-q)) would read > 0.5 % high on this column
    tpw_r = np.asarray(column_water_vapor(q_m / (1.0 - q_m), p_s, sigma.dsigma))
    assert float(((tpw_r - tpw_ref) / tpw_ref).min()) > 5e-3


def test_finite_increment_never_yields_negative_water():
    r = jnp.asarray(0.01)
    got = float(mixing_ratio_increment_to_specific_humidity(r, jnp.asarray(-0.02)))
    q0 = float(mixing_ratio_to_specific_humidity(r))
    np.testing.assert_allclose(got, -q0, rtol=1e-15)     # depletes to exactly zero
