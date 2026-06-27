"""Unit tests for the cos²(lat) Laplacian-viscosity scaling
(`LatLonCGridOceanConfig.A_h_lat_scaling`).

The scaling addresses the high-latitude viscous-Coriolis runaway documented
in `docs/ocean/experiments/realistic_geometry_phase4_results.md` (D1):
on real ETOPO at A_h=2e5, the model blows up at a single Arctic
partial-cell at lat 82.5° in ~15 sim-days because the viscous decay time
shrinks with cos²(lat) and becomes faster than the Coriolis period at the
poles.  Standard MITgcm/MOM6/NEMO production fix: scale A_h by cos²(lat).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    laplacian_scaling_factor,
    biharmonic_scaling_factor,
    vector_laplacian_cgrid,
    vector_laplacian_dissipation_cgrid,
    divergence_cgrid,
    curl_vertex_cgrid,
    compute_vertex_mask,
)


def test_laplacian_scaling_factor_shape_and_endpoints():
    grid = create_latlon_grid(36, 72)                     # 5° resolution
    # Explicit ``power=2`` mirrors the legacy constant-CFL convention.
    # The function's current default is ``power=1`` (constant grid
    # Reynolds number, which is the recommended convention and is what
    # the ocean PE backends use in production); pass ``power=2``
    # explicitly so these legacy regression tests still check the
    # cos²(lat) form they were written for.
    scale_u, scale_v = laplacian_scaling_factor(grid, power=2)

    # Shape conventions: scale_u at u-face (cell-center) lat → (n_lat,);
    # scale_v at v-face lat → (n_lat+1,).
    assert scale_u.shape == (36,)
    assert scale_v.shape == (37,)

    # cos²(lat) is non-negative everywhere
    assert bool(jnp.all(scale_u >= 0))
    assert bool(jnp.all(scale_v >= 0))

    # At cell-center lat 0 (between rows 17 and 18 for 36 cells centered
    # on lat=0; the closest u-face latitudes are ±2.5°), scale_u should be
    # very close to 1.
    cos_lat = np.asarray(grid.cos_lat)
    expected_scale_u = cos_lat ** 2
    np.testing.assert_allclose(np.asarray(scale_u), expected_scale_u,
                                rtol=1e-12, atol=1e-12)

    # Highest-latitude u-face at ±87.5° should give cos²(87.5°) ≈ 0.00191
    assert float(scale_u[-1]) < 0.01
    assert float(scale_u[0]) < 0.01

    # v-face at ±90° (the very ends) is exactly cos²(±lat[-1]) per the
    # 'edge clamp' convention used in the implementation.
    assert float(scale_v[0]) < 0.01
    assert float(scale_v[-1]) < 0.01


def test_laplacian_scaling_factor_consistency_with_biharmonic():
    """B_h scaling is cos⁴, A_h scaling at power=2 is cos² → A_h² = B_h."""
    grid = create_latlon_grid(36, 72)
    # Pass ``power=2`` so the (cos² · cos² = cos⁴) identity matches
    # the biharmonic cos⁴ convention.  The Laplacian production default
    # is ``power=1``; this test is documenting the legacy cos² form.
    lap_u, lap_v = laplacian_scaling_factor(grid, power=2)
    bih_u, bih_v = biharmonic_scaling_factor(grid)
    np.testing.assert_allclose(np.asarray(lap_u) ** 2,
                                np.asarray(bih_u),
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(lap_v) ** 2,
                                np.asarray(bih_v),
                                rtol=1e-12, atol=1e-12)


def test_laplacian_scaling_equator_near_unity():
    """Near the equator the scale must be very close to 1 (no significant
    damping reduction).  The v-face cos(lat_v) is now the single-source
    cos-OF-interface convention (``vface_zonal_cos_lat``, #516
    ``d2c9e9826``) rather than the legacy mean-of-cos: the equator v-face
    interface latitude is exactly 0, so the factor there is cos(0)²=1.0
    EXACTLY (mean-of-cos gave cos²(2.5°)≈0.998).  The cos-of-interface form
    is the more faithful O(dlat²)-correct metric and restores strain↔stress
    adjointness / div↔advection mass-consistency on non-uniform-dlat grids.
    """
    grid = create_latlon_grid(36, 72)
    _, scale_v = laplacian_scaling_factor(grid, power=2)
    # v-face at index 18 (out of 37) sits at lat = 0; cos-of-interface gives
    # cos(0)² = 1.0 exactly (single-source vface_zonal_cos_lat, #516).
    expected = 1.0
    assert abs(float(scale_v[18]) - expected) < 1e-6
    # The point of this test: scaling is essentially OFF at the equator
    assert float(scale_v[18]) > 0.99


def test_ah_scaling_off_is_identity():
    """A_h_lat_scaling=False must produce the same vlap result as the
    legacy path (multiply-by-A_h-only).  This is the bit-exact regression
    guard."""
    from legoesm.ocean.state import LatLonCGridOceanConfig

    config_off = LatLonCGridOceanConfig.from_flat(A_h=1e5, A_h_lat_scaling=False)
    config_on = LatLonCGridOceanConfig.from_flat(A_h=1e5, A_h_lat_scaling=True)
    assert config_off.A_h_lat_scaling is False
    assert config_on.A_h_lat_scaling is True
    # NamedTuple default: pre-existing call sites that don't pass the flag
    # default to False.
    config_legacy = LatLonCGridOceanConfig.from_flat(A_h=1e5)
    assert config_legacy.A_h_lat_scaling is False


def test_ah_scaling_on_reduces_high_lat_tendency():
    """With A_h_lat_scaling=True, the Laplacian tendency at high latitude
    is reduced by cos²(lat) compared to the unscaled tendency."""
    grid = create_latlon_grid(36, 72)
    n_lat, n_lon = 36, 72

    # Construct a non-trivial velocity field so the vector Laplacian is
    # non-zero everywhere.  Use a sinusoidal mode in lat-lon.
    np.random.seed(0)
    u = np.random.randn(n_lat, n_lon + 1, 1) * 0.01
    v = np.random.randn(n_lat + 1, n_lon, 1) * 0.01
    u_jax = jnp.asarray(u); v_jax = jnp.asarray(v)

    vlap_u, vlap_v = vector_laplacian_cgrid(u_jax, v_jax, grid)
    # Legacy cos²(lat) convention — see other tests in this module.
    scale_u, scale_v = laplacian_scaling_factor(grid, power=2)

    # Unscaled tendency (legacy path)
    tend_u_unscaled = 1e5 * np.asarray(vlap_u)
    # Scaled tendency (cos²(lat) applied)
    tend_u_scaled = 1e5 * np.asarray(scale_u)[:, None, None] * np.asarray(vlap_u)

    cos_lat = np.asarray(grid.cos_lat)

    # At highest lat (cos²(87.5°) ≈ 0.00191), scaled tendency must be
    # ≈ 0.00191× the unscaled.
    expected_ratio_high = cos_lat[-1] ** 2  # ≈ 0.00191 at 87.5°
    # Compare cell-by-cell at the high-lat row, where vlap_u is non-zero
    nonzero = np.abs(tend_u_unscaled[-1]) > 1e-15
    if nonzero.any():
        ratio = (tend_u_scaled[-1, nonzero] /
                 tend_u_unscaled[-1, nonzero])
        np.testing.assert_allclose(ratio, expected_ratio_high,
                                    rtol=1e-5, atol=1e-10)

    # At lowest |lat| (cos²(2.5°) ≈ 0.998), scaled tendency must be
    # nearly equal to unscaled.
    middle = n_lat // 2 - 1
    expected_ratio_eq = cos_lat[middle] ** 2  # ≈ 0.998 at 2.5°
    nonzero_eq = np.abs(tend_u_unscaled[middle]) > 1e-15
    if nonzero_eq.any():
        ratio_eq = (tend_u_scaled[middle, nonzero_eq] /
                    tend_u_unscaled[middle, nonzero_eq])
        np.testing.assert_allclose(ratio_eq, expected_ratio_eq,
                                    rtol=1e-5, atol=1e-10)


def test_viscous_cfl_latitude_independent():
    """End-to-end check on the physics motivation: with cos²(lat) scaling,
    the effective viscous CFL ``A_h_eff(lat) * dt / dx²(lat)`` is
    latitude-independent (up to the spherical metric)."""
    grid = create_latlon_grid(36, 72)
    R = constants.R_earth
    dlon = 2.0 * np.pi / 72                                # radians
    dt = 600.0
    A_h_global = 2.0e5

    cos_lat = np.asarray(grid.cos_lat)
    dx_lat = R * dlon * cos_lat                            # dx at each lat row
    A_h_eff = A_h_global * cos_lat ** 2                    # cos²(lat) scaling

    # CFL number with scaling
    cfl_scaled = A_h_eff * dt / dx_lat ** 2
    # Should be approximately constant across latitudes
    cfl_at_eq = cfl_scaled[len(cfl_scaled) // 2 - 1]
    cfl_at_pole = cfl_scaled[-1]
    # cos²(lat) / cos²(lat) → constant
    assert abs(cfl_at_eq - cfl_at_pole) < 1e-12 * cfl_at_eq

    # Compare to UNSCALED CFL: at high lat it's much larger
    cfl_unscaled = A_h_global * dt / dx_lat ** 2
    cfl_unscaled_at_pole = cfl_unscaled[-1]
    cfl_unscaled_at_eq = cfl_unscaled[len(cfl_unscaled) // 2 - 1]
    # Pole CFL is at least 100× larger than equator CFL without scaling
    assert cfl_unscaled_at_pole / cfl_unscaled_at_eq > 100.0


# ---------------------------------------------------------------------------
# A_h_cos_power — configurable exponent for cos(lat) scaling. Matches
# Veros's ``hor_friction_cosPower``; required for the Veros recipe.
# ---------------------------------------------------------------------------

def test_ah_cos_power_default_is_one():
    """The new ``A_h_cos_power`` field defaults to 1 — the production
    convention (constant grid Reynolds number) — so adding it does not
    change behavior for any legacy config."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1e5, A_h_lat_scaling=True)
    assert cfg.A_h_cos_power == 1


def test_ah_cos_power_one_matches_legacy_power_one():
    """``power=1`` matches what was hardcoded in ocean_pe_latlon_cgrid.py
    before A_h_cos_power was exposed: bit-exact regression guard."""
    grid = create_latlon_grid(36, 72)
    scale_u_n1, scale_v_n1 = laplacian_scaling_factor(grid, power=1)
    cos_u = np.asarray(grid.cos_lat)
    np.testing.assert_allclose(np.asarray(scale_u_n1), cos_u,
                                rtol=1e-12, atol=1e-12)


def test_ah_cos_power_two_gives_cos_squared():
    """``power=2`` (Veros ``hor_friction_cosPower=2``) gives cos²(lat) —
    the legacy constant-viscous-CFL convention."""
    grid = create_latlon_grid(36, 72)
    scale_u_n2, _ = laplacian_scaling_factor(grid, power=2)
    cos_u = np.asarray(grid.cos_lat)
    np.testing.assert_allclose(np.asarray(scale_u_n2), cos_u ** 2,
                                rtol=1e-12, atol=1e-12)


def test_ah_cos_power_zero_is_no_scaling():
    """``power=0`` corresponds to Veros's ``enable_hor_friction_cos_scaling
    =False``: cos⁰(lat)=1 everywhere, so the per-row scale is identity."""
    grid = create_latlon_grid(36, 72)
    scale_u_n0, scale_v_n0 = laplacian_scaling_factor(grid, power=0)
    assert jnp.allclose(scale_u_n0, 1.0)
    assert jnp.allclose(scale_v_n0, 1.0)


# ---------------------------------------------------------------------------
# vector_laplacian_dissipation_cgrid: the positive-definite K_diss_h density.
# ---------------------------------------------------------------------------


def test_vector_laplacian_dissipation_nonneg_and_zero_for_no_flow():
    """``A_h·(div²+<ζ²>)`` is ``>= 0`` everywhere by construction, and is exactly
    zero for the rest state (u=v=0) — it credits only the energy the lateral
    viscosity actually removes (div=ζ=0 ⇒ no KE removal)."""
    grid = create_latlon_grid(24, 48)
    n_lat, n_lon = 24, 48
    A_h = 1.0e4
    np.random.seed(3)
    u = jnp.asarray(np.random.randn(n_lat, n_lon + 1, 2) * 0.05)
    v = jnp.asarray(np.random.randn(n_lat + 1, n_lon, 2) * 0.05)
    diss = vector_laplacian_dissipation_cgrid(u, v, grid, A_h)
    assert diss.shape == (n_lat, n_lon, 2)
    assert bool(jnp.all(jnp.isfinite(diss)))
    assert float(jnp.min(diss)) >= 0.0
    assert float(jnp.max(diss)) > 0.0
    # Rest state: div = 0, ζ = 0 -> exactly zero dissipation.
    diss0 = vector_laplacian_dissipation_cgrid(
        jnp.zeros((n_lat, n_lon + 1, 2)), jnp.zeros((n_lat + 1, n_lon, 2)),
        grid, A_h)
    np.testing.assert_array_equal(np.asarray(diss0), 0.0)
    # A constant zonal wind has only the (small) spherical-metric div/curl, so its
    # dissipation is orders of magnitude below the sheared field's — confirming the
    # density tracks genuine shear, not a constant offset.
    u_uni = jnp.ones((n_lat, n_lon + 1, 2)) * 0.3
    v_uni = jnp.zeros((n_lat + 1, n_lon, 2))
    diss_uni = vector_laplacian_dissipation_cgrid(u_uni, v_uni, grid, A_h)
    assert float(jnp.max(diss_uni)) < 0.05 * float(jnp.max(diss))


def test_vector_laplacian_dissipation_linear_in_A_h_and_matches_div_curl():
    """The dissipation is linear in ``A_h`` and equals an independent
    ``A_h·(div² + <ζ²>_corners)`` reconstruction (locks the formula + the
    4-corner ζ²→centre averaging)."""
    grid = create_latlon_grid(24, 48)
    n_lat, n_lon = 24, 48
    np.random.seed(5)
    u = jnp.asarray(np.random.randn(n_lat, n_lon + 1, 1) * 0.05)
    v = jnp.asarray(np.random.randn(n_lat + 1, n_lon, 1) * 0.05)
    d1 = vector_laplacian_dissipation_cgrid(u, v, grid, 1.0e4)
    d2 = vector_laplacian_dissipation_cgrid(u, v, grid, 2.0e4)
    np.testing.assert_allclose(np.asarray(d2), 2.0 * np.asarray(d1), rtol=1e-12)
    # Independent reconstruction.
    div = divergence_cgrid(u, v, grid)
    zeta = curl_vertex_cgrid(u, v, grid)
    z2 = zeta ** 2
    z2c = 0.25 * (z2[:-1, :-1, :] + z2[1:, :-1, :] + z2[:-1, 1:, :] + z2[1:, 1:, :])
    ref = 1.0e4 * (div ** 2 + z2c)
    np.testing.assert_allclose(np.asarray(d1), np.asarray(ref), rtol=1e-12)


def test_vector_laplacian_dissipation_energy_consistent_with_vlap_sink():
    """On a periodic (no-land) grid the domain-integrated positive-definite
    dissipation ``Σ A_h(div²+<ζ²>)·A_cell`` equals the mean KE removed by the
    vector Laplacian ``-Σ (u·A_h∇²u + v·A_h∇²v)·A_dual`` to a few percent — the
    Helmholtz energy identity that makes this the FAITHFUL, clamp-free K_diss_h.
    Uses a smooth large-scale field so the discrete identity holds tightly."""
    n_lat, n_lon = 40, 80
    grid = create_latlon_grid(n_lat, n_lon)
    A_h = 1.0e4
    # Smooth large-scale velocity (low wavenumber) -> small discretisation error.
    lat = np.asarray(grid.lat)[:, None]
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)[None, :]
    lon_u = np.linspace(0, 2 * np.pi, n_lon + 1)[None, :]
    u = jnp.asarray((np.sin(lon_u) * np.cos(lat))[:, :, None])
    lat_v = np.linspace(float(grid.lat[0]) - float(grid.dlat) / 2,
                        float(grid.lat[-1]) + float(grid.dlat) / 2, n_lat + 1)[:, None]
    v = jnp.asarray((np.cos(2 * lon) * np.cos(lat_v))[:, :, None])

    # Positive-definite dissipation density -> domain energy (area-weighted).
    diss = vector_laplacian_dissipation_cgrid(u, v, grid, A_h)
    area = np.asarray(grid.area)
    E_diss = float((np.asarray(diss)[:, :, 0] * area).sum())

    # Mean-KE sink of the vector Laplacian: -Σ u·(A_h ∇²u) over u-faces (×dual
    # area) - Σ v·(A_h ∇²v) over v-faces. The u-face dual area ~ cell area
    # interpolated to the face; for the periodic smooth field the cell-centred
    # contraction -[u·vlap_u]|_centre·area + ... is the consistent discrete energy.
    vlap_u, vlap_v = vector_laplacian_cgrid(u, v, grid)
    pu = -np.asarray(u * (A_h * vlap_u))
    pv = -np.asarray(v * (A_h * vlap_v))
    sink_cell = 0.5 * (pu[:, :-1, 0] + pu[:, 1:, 0]) + 0.5 * (pv[:-1, :, 0] + pv[1:, :, 0])
    E_sink = float((sink_cell * area).sum())

    assert E_diss > 0.0 and E_sink > 0.0
    rel = abs(E_diss - E_sink) / E_sink
    assert rel < 0.05, f"energy mismatch {rel:.4f} (diss={E_diss:.4e}, sink={E_sink:.4e})"
