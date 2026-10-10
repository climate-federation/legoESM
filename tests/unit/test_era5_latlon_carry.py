"""Mechanical validation of era5_to_latlon_carry on a synthetic ERA5 slice.

No real ERA5 data is required: a fabricated ERA5Slice on a small lat-lon
grid exercises the regrid + vertical-interp + specific-humidity->mixing-
ratio conversion and asserts shape, finiteness, and physical bounds on
the model lat-lon grid.  (Full physical validation against real ERA5 is
deferred to when a zarr path is provided.)
"""

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.factory import create_grid
from legoesm.training.era5_to_state import (
    ERA5Slice,
    era5_to_latlon_carry,
    prognostic_carry_seeds,
)


def _synthetic_era5(n_lat=37, n_lon=72, n_plev=8):
    lat = np.linspace(-np.pi / 2 * 0.98, np.pi / 2 * 0.98, n_lat)
    lon = np.linspace(0.0, 2 * np.pi * (1 - 1.0 / n_lon), n_lon)
    plev = np.array([1000, 2000, 5000, 10000, 25000, 50000, 85000, 100000.0])
    latg = lat[:, None, None]
    # Earth-like: warm tropics, cold poles + a pressure-consistent lapse
    # (warm at high pressure / surface, cold at low pressure / top) +
    # a zonal jet in u.  Altitude proxy z(plev) = (1 - plev/p_s)*12 km
    # so the lapse decreases T with HEIGHT, not with array index.
    z = (1.0 - plev / 1.0e5) * 12000.0          # (n_plev,) m, 0 at surface
    T_sfc = 300.0 - 40.0 * np.sin(latg) ** 2    # (n_lat,1,1)
    T = T_sfc - 6.5e-3 * z[None, None, :]
    T = np.broadcast_to(T, (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    u = (30.0 * np.cos(latg) * np.sin(2 * latg)).astype(np.float32)
    u = np.broadcast_to(u, (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    v = np.zeros((n_lat, n_lon, n_plev), np.float32)
    q = np.broadcast_to(
        (0.018 * np.cos(latg) ** 2), (n_lat, n_lon, n_plev)
    ).astype(np.float32).copy()
    p_s = np.full((n_lat, n_lon), 1.0e5, np.float32)
    sst = (300.0 - 40.0 * np.sin(lat[:, None]) ** 2) * np.ones((n_lat, n_lon))
    phis = np.zeros((n_lat, n_lon), np.float32)
    return ERA5Slice(
        T=T, u=u, v=v, q=q, p_s=p_s, sst=sst.astype(np.float32),
        phis=phis, lat=lat, lon=lon, plev_Pa=plev,
    )


def test_latlon_carry_shapes_and_physical():
    grid = create_grid("latlon", 24)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(30)
    carry = era5_to_latlon_carry(_synthetic_era5(), grid, sigma,
                                 target_phis=jnp.zeros(np.shape(grid.grid_lat)))
    # SegmentCarry exposes T/u/v/p_s and q_v via the packed layout.
    T = np.asarray(carry.T.data if hasattr(carry.T, "data") else carry.T)
    u = np.asarray(carry.u.data if hasattr(carry.u, "data") else carry.u)
    assert T.shape == (grid.n_lat, grid.n_lon, 30)
    assert u.shape == (grid.n_lat, grid.n_lon, 30)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(u))
    # Physical bounds: T in a sane atmospheric range, jet present.
    assert 180.0 < T.min() and T.max() < 320.0
    assert np.abs(u).max() > 5.0, "zonal jet did not survive the regrid"
    qv = np.asarray(carry.q_v.data if hasattr(carry, "q_v") and hasattr(carry.q_v, "data")
                    else getattr(carry, "q_v"))
    assert np.all(qv >= 0.0) and qv.max() < 0.05, "mixing ratio unphysical"


def test_latlon_carry_loads_q_as_specific_humidity():
    # ERA5 specific humidity is loaded AS IS (the tracer convention on
    # every lane, 2026-09-28); the former r = q/(1-q) = 0.0204 must FAIL.
    base = _synthetic_era5()
    era5 = base._replace(q=np.full_like(base.q, 0.02))
    grid = create_grid("latlon", 12)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(20)
    carry = era5_to_latlon_carry(era5, grid, sigma,
                                 target_phis=jnp.zeros(np.shape(grid.grid_lat)))
    qv = np.asarray(getattr(carry, "q_v").data
                    if hasattr(getattr(carry, "q_v"), "data") else getattr(carry, "q_v"))
    # Near the surface (sigma~1) where interpolation is well-posed.
    np.testing.assert_allclose(float(qv[..., -1].mean()), 0.02, rtol=1e-3)


def _mountain_era5(n_lat=37, n_lon=72, n_plev=8, peak_phis=5.6e4):
    """Synthetic ERA5 slice with a single steep orographic peak in phis."""
    base = _synthetic_era5(n_lat=n_lat, n_lon=n_lon, n_plev=n_plev)
    phis = np.zeros((n_lat, n_lon), np.float32)
    phis[n_lat // 3, n_lon // 2] = peak_phis      # ~5600 m ERA5-like spike
    phis[n_lat // 3, n_lon // 2 + 1] = peak_phis * 0.6
    return base._replace(phis=phis)


def _max_abs_grad(arr):
    di = np.abs(np.diff(arr, axis=0)).max()
    dj = np.abs(arr - np.roll(arr, 1, axis=1)).max()
    return float(max(di, dj))


# --- T2.2: shared hydrostatic-adjustment helper (cube + lat-lon) -------------

def test_phis_adjust_identity_when_no_smoothing_nonhybrid():
    # phis_raw == phis_smooth and non-hybrid ⇒ p_s and phis unchanged.
    from legoesm.training.era5_to_state import _apply_phis_hydrostatic_adjustment
    phis = jnp.asarray(np.full((4, 6), 2.0e4))
    p_s = jnp.asarray(np.full((4, 6), 9.0e4))
    T_sfc = jnp.asarray(np.full((4, 6), 288.0))
    phis_adj, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis, phis, p_s, T_sfc, sigma=None, is_hybrid=False)
    np.testing.assert_allclose(np.asarray(phis_adj), np.asarray(phis), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(p_s_adj), np.asarray(p_s), rtol=1e-6)


def test_phis_adjust_lowering_terrain_raises_ps_sign():
    # Smoothing LOWERS a peak (phis_smooth < phis_raw) ⇒ p_s must INCREASE
    # (descend from higher raw surface to lower smoothed surface).
    from legoesm.training.era5_to_state import _apply_phis_hydrostatic_adjustment
    phis_raw = jnp.asarray(np.array([[5.0e4]]))
    phis_smooth = jnp.asarray(np.array([[3.0e4]]))   # terrain cut by smoothing
    p_s = jnp.asarray(np.array([[6.0e4]]))
    T_sfc = jnp.asarray(np.array([[270.0]]))
    _, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis_raw, phis_smooth, p_s, T_sfc, sigma=None, is_hybrid=False)
    assert float(p_s_adj[0, 0]) > float(p_s[0, 0]), "lowering terrain must raise p_s"
    # Magnitude matches the barometric formula exactly.
    from legoesm import constants
    expect = 6.0e4 * np.exp((5.0e4 - 3.0e4) / (constants.R_d * 270.0))
    np.testing.assert_allclose(float(p_s_adj[0, 0]), expect, rtol=1e-6)


def test_phis_adjust_hybrid_floor_refuses_the_column():
    # Decision C: a p_s below the degenerate-layer floor is REFUSED (the
    # terrain product is never lowered); at the floor it passes untouched.
    import pytest
    from legoesm.training.era5_to_state import (
        _apply_phis_hydrostatic_adjustment, _hybrid_p_s_floor)
    from legoesm.grids.vertical import make_hybrid_levels
    sigma = make_hybrid_levels(20)
    floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
    phis_s = jnp.asarray(np.full((3, 3), 4.0e4))
    p_s = jnp.asarray(np.full((3, 3), floor * 0.5))   # below the floor
    T_sfc = jnp.asarray(np.full((3, 3), 260.0))
    with pytest.raises(ValueError, match="positive-thickness floor"):
        _apply_phis_hydrostatic_adjustment(
            phis_s, phis_s, p_s, T_sfc, sigma=sigma, is_hybrid=True)
    phis_adj, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis_s, phis_s, jnp.full((3, 3), floor * 1.5), T_sfc, sigma=sigma, is_hybrid=True)
    np.testing.assert_array_equal(np.asarray(phis_adj), np.asarray(phis_s))
    np.testing.assert_allclose(np.asarray(p_s_adj), floor * 1.5)


# --- T3.2: lat-lon carry with real (non-flat) orography ----------------------

def _product_from_raw(grid, raw, passes=4):
    """The grid's terrain product stand-in: the masked diffusion of a raw
    field with an all-land mask (what load_real_topography does to the binned
    elevation), so carry tests exercise the real target path."""
    from legoesm.grids.topography import neighbour_table, masked_diffusion
    nb, area = neighbour_table(grid)
    return jnp.asarray(masked_diffusion(np.asarray(raw).ravel(), np.ones(area.shape),
                                        nb, area, passes=passes).reshape(np.shape(raw)))


def test_latlon_carry_starts_on_the_target_and_stays_finite():
    # Decision C: the carry's phis IS the target product (smoother than the
    # raw regridded ERA5 orography), p_s moved to it, everything finite.
    from legoesm.training.era5_to_state import era5_to_latlon_carry, regrid_2d_to_gaussian
    from legoesm.grids.vertical import create_sigma_coordinate
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = create_sigma_coordinate(20)
    raw = np.asarray(regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid))
    target = _product_from_raw(grid, raw)
    carry = era5_to_latlon_carry(era5, grid, sigma, target_phis=target)
    phis = np.asarray(carry.phis)
    np.testing.assert_array_equal(phis, np.asarray(target))
    assert np.all(np.isfinite(phis)) and np.all(np.isfinite(np.asarray(carry.p_s)))
    assert _max_abs_grad(phis) < _max_abs_grad(raw), "the product is not smoother than raw"
    assert phis.max() <= raw.max() + 1e-6, "the product must not amplify the peak"
    T = np.asarray(carry.T)
    assert np.all(np.isfinite(T)) and 150.0 < T.min() and T.max() < 340.0


def test_latlon_carry_hybrid_orography_respects_ps_floor():
    from legoesm.training.era5_to_state import era5_to_latlon_carry, _hybrid_p_s_floor
    from legoesm.grids.vertical import make_hybrid_levels
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = make_hybrid_levels(20)
    # a flat target: p_s only ever moves UP from the raw mountain, so the
    # floor never binds (the refusal itself is gated in the helper test)
    carry = era5_to_latlon_carry(era5, grid, sigma,
                                 target_phis=jnp.zeros(np.shape(grid.grid_lat)))
    floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
    p_s = np.asarray(carry.p_s)
    assert np.all(np.isfinite(np.asarray(carry.T)))
    assert np.all(p_s >= floor - 1.0), "hybrid p_s floor not honored on lat-lon"


def test_latlon_carry_analytic_temperature_value():
    # Value-based check (not just bounds): a latitude-only ERA5 T field
    # must interpolate to the model grid with the right equator-pole
    # structure — equatorial column warmer than polar at the surface.
    era5 = _synthetic_era5(n_lat=73, n_lon=144)
    grid = create_grid("latlon", 24)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(30)
    carry = era5_to_latlon_carry(era5, grid, sigma,
                                 target_phis=jnp.zeros(np.shape(grid.grid_lat)))
    T = np.asarray(carry.T.data if hasattr(carry.T, "data") else carry.T)
    lat = np.asarray(grid.lat)
    i_eq = int(np.argmin(np.abs(lat)))
    i_pole = int(np.argmax(np.abs(lat)))
    T_eq_sfc = float(T[i_eq, :, -1].mean())
    T_pole_sfc = float(T[i_pole, :, -1].mean())
    assert T_eq_sfc > T_pole_sfc + 15.0, (
        f"equator-pole gradient lost: eq={T_eq_sfc:.1f} pole={T_pole_sfc:.1f}")
    # Equatorial surface T near the synthetic 300 K equator value.
    assert 285.0 < T_eq_sfc < 305.0


# --- T3: spectral carry gets the SAME phis treatment as the lat-lon carry ----
# (smooth_phis_gaussian + shared _apply_phis_hydrostatic_adjustment; it was the
#  odd grid out — raw phis, no barometric p_s reconciliation, no hybrid floor)

def test_spectral_carry_truncates_the_target_and_reconciles_ps():
    from legoesm import constants
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import (
        era5_to_spectral_carry,
        regrid_2d_to_gaussian,
        regrid_latlon_to_gaussian,
    )
    era5 = _mountain_era5()
    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(20)
    raw = np.asarray(regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid))
    target = _product_from_raw(grid, raw)
    carry = era5_to_spectral_carry(era5, grid, sigma, target_phis=target)
    phis = np.asarray(carry.phis)
    assert np.all(np.isfinite(phis)) and np.all(np.isfinite(np.asarray(carry.p_s)))
    assert _max_abs_grad(phis) < _max_abs_grad(raw), "the product is not smoother than raw"
    # p_s is RECONCILED to the terrain the SPECTRAL DYNAMICS feel — the
    # ROUND-TRIPPED smoothed phis (Gibbs ringing included), not the
    # grid-space smoothed field.  Reconciling to the grid-space field left
    # every ingested state ~850 Pa RMS off the model's balanced manifold
    # (measured 2026-08-26, corr +0.996 with the barometric response to the
    # truncation mismatch), and that standing gap was 87% of the WB training
    # loss.  Non-hybrid: p_s_adj = p_s * exp((phis_raw - phis_rt)/(R_d T)).
    from legoesm.grids.gaussian import sh_analysis, sh_synthesis
    T_ll, _, _, _, p_s_ll = regrid_latlon_to_gaussian(era5, grid)
    smooth_rt = np.asarray(sh_synthesis(grid, sh_analysis(
        grid, jnp.asarray(target, jnp.float64))))
    T_sfc = np.asarray(T_ll)[..., -1]
    expected_ps = np.asarray(p_s_ll) * np.exp(
        (raw - smooth_rt) / (constants.R_d * T_sfc))
    np.testing.assert_allclose(np.asarray(carry.p_s), expected_ps, rtol=1e-4)
    # The carry's phis IS the round-tripped field (idempotent under the
    # core's own sh_analysis, so carry and dynamics share one surface).
    np.testing.assert_allclose(np.asarray(carry.phis), smooth_rt, rtol=1e-6,
                               atol=1e-6)
    # Sign: where the EFFECTIVE terrain was cut, lowering it must RAISE p_s.
    cut = (raw - smooth_rt) > 1.0
    assert cut.any()
    assert np.all(np.asarray(carry.p_s)[cut] > np.asarray(p_s_ll)[cut])


def test_spectral_carry_hybrid_orography_respects_ps_floor():
    # Mirror of test_latlon_carry_hybrid_orography_respects_ps_floor.
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import make_hybrid_levels
    from legoesm.training.era5_to_state import (
        _hybrid_p_s_floor, era5_to_spectral_carry)
    era5 = _mountain_era5()
    grid = create_gaussian_grid(21)
    sigma = make_hybrid_levels(20)
    carry = era5_to_spectral_carry(era5, grid, sigma,
                                   target_phis=jnp.zeros(np.shape(grid.grid_lat)))
    floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
    p_s = np.asarray(carry.p_s)
    assert np.all(np.isfinite(np.asarray(carry.T)))
    assert np.all(p_s >= floor - 1.0), "hybrid p_s floor not honored on spectral"


# ---------------------------------------------------------------------------
# Conditional prognostic-carry seeding (all-scheme AIMIP training support)
# ---------------------------------------------------------------------------

def _latlon_carry(microphysics, turbulence, n=12, nlev=20):
    grid = create_grid("latlon", n)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(nlev)
    return era5_to_latlon_carry(
        _synthetic_era5(), grid, sigma,
        target_phis=jnp.zeros(np.shape(grid.grid_lat)),
        microphysics=microphysics, turbulence=turbulence,
    )


def test_warm_rain_carry_leaves_extras_none():
    """kessler + diagnostic turbulence ⇒ q_i…N_i / tke / qke stay None.

    The warm-rain default path's carry pytree must be byte-identical to
    the legacy carry — seeding non-None would flip the lax.scan carry
    structure and the _dm_upd(None tendency) path.
    """
    carry = _latlon_carry("kessler", "louis")
    for name in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i", "tke", "qke"):
        assert getattr(carry, name) is None, (
            f"{name} should be None for warm-rain kessler/louis, "
            f"got {type(getattr(carry, name))}")
    # The three warm-rain microphysics slots are still real arrays.
    assert carry.q_v is not None and carry.q_c is not None
    assert carry.q_r is not None


def test_double_moment_carry_seeds_zero_tracers():
    """morrison ⇒ q_i…N_i seeded as zero arrays of the 3-D field shape."""
    carry = _latlon_carry("morrison", "louis")
    nlev = 20
    grid = create_grid("latlon", 12)
    expected = (grid.n_lat, grid.n_lon, nlev)
    for name in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
        arr = getattr(carry, name)
        assert arr is not None, f"{name} should be seeded for morrison"
        arr = np.asarray(arr)
        assert arr.shape == expected, (
            f"{name} shape {arr.shape} != {expected}")
        assert np.all(arr == 0.0), f"{name} should seed to zero, got nonzero"
    # Diagnostic turbulence ⇒ tke / qke still None.
    assert carry.tke is None and carry.qke is None


def test_stateful_turbulence_carry_seeds_tke():
    """tke turbulence ⇒ tke seeded as a zero (ncol, nlev) array; qke None.

    The tke/qke carry is FLATTENED per-column (ncol, nlev) — not the
    grid-shaped (n_lat, n_lon, nlev) layout the microphysics tracers use —
    because the turbulence scheme reads it per-column (#405/#413)."""
    carry = _latlon_carry("kessler", "tke")
    nlev = 20
    grid = create_grid("latlon", 12)
    expected = (grid.n_lat * grid.n_lon, nlev)
    assert carry.tke is not None, "tke should be seeded for the tke scheme"
    tke = np.asarray(carry.tke)
    assert tke.shape == expected, f"tke shape {tke.shape} != {expected}"
    assert np.all(tke == 0.0), "tke should seed to zero"
    # tke (not mynn25) ⇒ the qke slot stays None.
    assert carry.qke is None
    # Warm-rain microphysics still leaves the double-moment slots None.
    assert carry.q_i is None and carry.N_c is None


def test_mynn25_seeds_qke_not_tke():
    """mynn25 ⇒ the qke slot is seeded, tke stays None (distinct moment)."""
    carry = _latlon_carry("kessler", "mynn25")
    assert carry.qke is not None, "mynn25 should seed qke"
    assert np.all(np.asarray(carry.qke) == 0.0)
    assert carry.tke is None, "mynn25 carries q²=qke, not tke"


def test_prognostic_carry_seeds_helper_warm_rain_empty():
    """The shared helper returns {} for warm-rain (no carry-structure flip)."""
    seeds = prognostic_carry_seeds("kessler", "louis", (4, 4, 5))
    assert seeds == {}
    # sundqvist is also warm-rain (3 slots).
    assert prognostic_carry_seeds("sundqvist", "smagorinsky", (4, 4, 5)) == {}
    # SDM default (condensation-only, 2 slots) ⇒ no double-moment seed.
    assert "q_i" not in prognostic_carry_seeds("sdm", "louis", (4, 4, 5))


def test_prognostic_carry_seeds_helper_double_moment_and_stateful():
    """The helper seeds the right keys for a double-moment + stateful combo."""
    seeds = prognostic_carry_seeds("seifert_beheng", "tke", (3, 3, 6))
    # microphysics tracers are grid-shaped (n_lat, n_lon, nlev)...
    for name in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
        assert name in seeds, f"{name} missing from seeds"
        assert np.asarray(seeds[name]).shape == (3, 3, 6)
    # ...but the turbulence energy carry is FLATTENED (ncol, nlev) = (9, 6).
    assert "tke" in seeds
    assert np.asarray(seeds["tke"]).shape == (3 * 3, 6)
    assert "qke" not in seeds  # tke scheme uses the tke slot




def test_phis_adjust_refuses_a_zero_filled_raw_geopotential():
    """A store without surface geopotential is zero-filled by the loader;
    the hydrostatic move onto a real product refuses it."""
    from legoesm.training.era5_to_state import _apply_phis_hydrostatic_adjustment
    sigma = np.linspace(0.05, 0.95, 10)
    with pytest.raises(ValueError, match="all zero"):
        _apply_phis_hydrostatic_adjustment(
            np.zeros((4, 8)), np.full((4, 8), 5000.0), np.full((4, 8), 1.0e5),
            np.full((4, 8), 280.0), sigma, False)
