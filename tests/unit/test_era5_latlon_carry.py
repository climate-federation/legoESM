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
    carry = era5_to_latlon_carry(_synthetic_era5(), grid, sigma)
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


def test_latlon_carry_specific_to_mixing_ratio():
    # A column with q_specific=0.02 must map to r = q/(1-q) ≈ 0.0204.
    # Build a fresh slice with the desired q (no in-place mutation).
    base = _synthetic_era5()
    era5 = base._replace(q=np.full_like(base.q, 0.02))
    grid = create_grid("latlon", 12)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(20)
    carry = era5_to_latlon_carry(era5, grid, sigma)
    qv = np.asarray(getattr(carry, "q_v").data
                    if hasattr(getattr(carry, "q_v"), "data") else getattr(carry, "q_v"))
    # Near the surface (sigma~1) where interpolation is well-posed.
    assert 0.0200 < float(qv[..., -1].mean()) < 0.0210


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


def test_phis_adjust_hybrid_floor_raises_ps_and_lowers_phis():
    # A p_s well below the degenerate-layer floor must be raised to the floor,
    # and phis lowered by the barometric equivalent (split-PGF consistency).
    from legoesm.training.era5_to_state import (
        _apply_phis_hydrostatic_adjustment, _hybrid_p_s_floor)
    from legoesm.grids.vertical import make_hybrid_levels
    sigma = make_hybrid_levels(20)
    floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
    phis_s = jnp.asarray(np.full((3, 3), 4.0e4))
    p_s = jnp.asarray(np.full((3, 3), floor * 0.5))   # below the floor
    T_sfc = jnp.asarray(np.full((3, 3), 260.0))
    phis_adj, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis_s, phis_s, p_s, T_sfc, sigma=sigma, is_hybrid=True)
    assert np.all(np.asarray(p_s_adj) >= floor - 1.0), "p_s not raised to floor"
    assert np.all(np.asarray(phis_adj) < np.asarray(phis_s)), "phis not lowered to match"


# --- T3.2: lat-lon carry with real (non-flat) orography ----------------------

def test_latlon_carry_smooths_orography_and_stays_finite():
    # With a steep ERA5 peak, the carry's phis must be SMOOTHER (smaller max
    # gradient) than the raw regridded orography — the fix for the blow-up.
    from legoesm.training.era5_to_state import era5_to_latlon_carry, regrid_2d_to_gaussian
    from legoesm.grids.vertical import create_sigma_coordinate
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = create_sigma_coordinate(20)
    carry = era5_to_latlon_carry(era5, grid, sigma)
    raw = np.asarray(regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid))
    phis = np.asarray(carry.phis)
    assert np.all(np.isfinite(phis)) and np.all(np.isfinite(np.asarray(carry.p_s)))
    assert _max_abs_grad(phis) < _max_abs_grad(raw), "carry did not smooth phis"
    assert phis.max() <= raw.max() + 1e-6, "smoothing must not amplify the peak"
    T = np.asarray(carry.T)
    assert np.all(np.isfinite(T)) and 150.0 < T.min() and T.max() < 340.0


def test_latlon_carry_hybrid_orography_respects_ps_floor():
    from legoesm.training.era5_to_state import era5_to_latlon_carry, _hybrid_p_s_floor
    from legoesm.grids.vertical import make_hybrid_levels
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = make_hybrid_levels(20)
    carry = era5_to_latlon_carry(era5, grid, sigma)
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
    carry = era5_to_latlon_carry(era5, grid, sigma)
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

def test_spectral_carry_smooths_phis_and_reconciles_ps():
    from legoesm import constants
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.topography import smooth_phis_gaussian
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import (
        era5_to_spectral_carry,
        regrid_2d_to_gaussian,
        regrid_latlon_to_gaussian,
    )
    era5 = _mountain_era5()
    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(20)
    carry = era5_to_spectral_carry(era5, grid, sigma)
    raw = np.asarray(regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid))
    phis = np.asarray(carry.phis)
    assert np.all(np.isfinite(phis)) and np.all(np.isfinite(np.asarray(carry.p_s)))
    # phis is SMOOTHED: it actually changed, gradients reduced, ridge amplitude cut.
    assert np.abs(phis - raw).max() > 0.0, "spectral carry left phis raw"
    assert _max_abs_grad(phis) < _max_abs_grad(raw), "carry did not smooth phis"
    assert phis.max() < raw.max(), "smoothing must reduce the ridge amplitude"
    # p_s is RECONCILED to the smoothed phis by the exact barometric relation
    # (non-hybrid): p_s_adj = p_s * exp((phis_raw - phis_smooth)/(R_d T_sfc)).
    T_ll, _, _, _, p_s_ll = regrid_latlon_to_gaussian(era5, grid)
    smooth = np.asarray(smooth_phis_gaussian(raw))
    T_sfc = np.asarray(T_ll)[..., -1]
    expected_ps = np.asarray(p_s_ll) * np.exp(
        (raw - smooth) / (constants.R_d * T_sfc))
    np.testing.assert_allclose(np.asarray(carry.p_s), expected_ps, rtol=1e-4)
    # Sign: where the peak was cut, lowering terrain must RAISE p_s.
    cut = (raw - smooth) > 1.0
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
    carry = era5_to_spectral_carry(era5, grid, sigma)
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


# --- audit 2026-07-17 T5: carry smoothing params are driver-tunable ----------

def test_latlon_carry_smoothing_passes_take_effect():
    """More Laplacian passes → smoother carry phis (the driver now threads
    cfg.topo_smoothing; the value was previously hard-coded)."""
    from legoesm.training.era5_to_state import era5_to_latlon_carry
    from legoesm.grids.vertical import create_sigma_coordinate
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = create_sigma_coordinate(20)
    few = np.asarray(era5_to_latlon_carry(
        era5, grid, sigma, smoothing_passes=1).phis)
    many = np.asarray(era5_to_latlon_carry(
        era5, grid, sigma, smoothing_passes=8).phis)
    assert _max_abs_grad(many) < _max_abs_grad(few), (
        "more passes must reduce the phis gradient")


def test_spectral_carry_smoothing_passes_take_effect():
    from legoesm.training.era5_to_state import era5_to_spectral_carry
    from legoesm.grids.vertical import create_sigma_coordinate
    era5 = _mountain_era5()
    grid = create_grid("gaussian", 21)
    sigma = create_sigma_coordinate(20)
    few = np.asarray(era5_to_spectral_carry(
        era5, grid, sigma, smoothing_passes=1).phis)
    many = np.asarray(era5_to_spectral_carry(
        era5, grid, sigma, smoothing_passes=8).phis)
    assert _max_abs_grad(many) < _max_abs_grad(few)


def test_cube_smooth_phis_edge_blend_width_defaults_to_two():
    """smooth_phis_cubed_sphere now blends with width=2 (TopographyConfig's
    default and the static-topography path), not the old silent width=1 — the
    docstring's 'same pipeline' promise (audit 2026-07-17)."""
    from legoesm.grids.topography import (
        smooth_phis_cubed_sphere, blend_scalar_cube_edges_2d,
    )
    from legoesm.grids.topography import _laplacian_smooth_cubed_sphere
    rng = np.random.default_rng(0)
    phis = jnp.asarray(rng.normal(size=(6, 8, 8)) * 1.0e4)
    got = np.asarray(smooth_phis_cubed_sphere(phis, smoothing_passes=2))
    lap = _laplacian_smooth_cubed_sphere(np.asarray(phis), passes=2)
    want_w2 = np.asarray(blend_scalar_cube_edges_2d(
        jnp.asarray(lap), strength=0.3, width=2))
    want_w1 = np.asarray(blend_scalar_cube_edges_2d(
        jnp.asarray(lap), strength=0.3, width=1))
    np.testing.assert_allclose(got, want_w2, atol=1e-9)
    assert not np.allclose(got, want_w1), "default must be width=2, not width=1"
