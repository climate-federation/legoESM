"""Unit tests for the Fox-Kemper mixed-layer-eddy (MLE) restratification.

Covers BOTH the shared grid-agnostic core
(:mod:`legoesm.ocean.physics.lateral_mixing.mle`) and the lat-lon C-grid adapter
(:mod:`legoesm.ocean.physics.lateral_mixing.mle_latlon_cgrid`):

(a) ``mle_vertical_structure``: mu = 0 at the surface (gdepw/H = 0) and the ML
    base (= 1), with a mid-mixed-layer peak.
(b) ``mle_coefficient``: rc_f > 0 at the reference latitude; equator guard.
(c) ``mle_mld_and_buoyancy``: a 2-layer column gives the expected MLD and the
    correct ML-mean-buoyancy sign.
(d) EXACT tracer conservation: a buoyancy front on a small flat-bottom C-grid
    integrates to sum(dT·area·dz) ~ 0 (and sum(dS·area·dz) ~ 0) to roundoff.
(e) Restratifying flux sign: a light cell next to a dense cell gets a tendency
    that moves tracer DOWN the buoyancy gradient (flattening isopycnals).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

# Float64 everywhere: ``create_latlon_grid`` / ``create_ocean_z_star`` read the
# active precision policy (default fp32), and an exact-conservation telescoping
# check needs float64 metrics (fp32 would leave a ~1e-6 residual).
from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64())

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.physics.lateral_mixing.mle import (
    MLEConfig,
    face_mld,
    mle_coefficient,
    mle_mld_and_buoyancy,
    mle_vertical_structure,
)
from legoesm.ocean.physics.lateral_mixing.mle_latlon_cgrid import (
    mle_tracer_tendency_latlon_cgrid,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.vertical import create_ocean_z_star


# ---------------------------------------------------------------------------
# (a) Vertical structure function mu(z)
# ---------------------------------------------------------------------------
def test_vertical_structure_endpoints_and_peak() -> None:
    """mu(0) = mu(1) = 0 with a positive interior peak near mid-depth."""
    frac = jnp.linspace(0.0, 1.0, 21)
    mu = np.asarray(mle_vertical_structure(frac))
    # Endpoints: zeta = 1 (surface) and zeta = -1 (ML base) -> (1 - zeta^2) = 0.
    assert mu[0] == pytest.approx(0.0, abs=1e-12)
    assert mu[-1] == pytest.approx(0.0, abs=1e-12)
    # Strictly positive in the interior.
    assert np.all(mu[1:-1] > 0.0)
    # The peak is at the centre (gdepw/H = 0.5 -> zeta = 0 -> mu = 1).
    k_peak = int(np.argmax(mu))
    assert frac[k_peak] == pytest.approx(0.5, abs=0.05)
    assert mu[k_peak] == pytest.approx(1.0, rel=1e-6)


def test_vertical_structure_nonnegative_clamped() -> None:
    """mu is clamped to >= 0 even for gdepw/H outside [0, 1] (below the ML)."""
    frac = jnp.array([-0.2, 1.2, 2.0])
    mu = np.asarray(mle_vertical_structure(frac))
    assert np.all(mu >= 0.0)


# ---------------------------------------------------------------------------
# (b) MLE coefficient rc_f
# ---------------------------------------------------------------------------
def test_mle_coefficient_positive() -> None:
    """rc_f = rn_ce / (5 km · 2Ω sin lat) is positive at the NEMO ref latitude."""
    rc_f = mle_coefficient(ce=0.06, lat_ref_deg=20.0)
    assert rc_f > 0.0
    # Closed-form cross-check against constants (no magic numbers).
    f0 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
    assert rc_f == pytest.approx(0.06 / (5.0e3 * f0), rel=1e-10)


def test_mle_coefficient_scales_with_ce() -> None:
    """rc_f is linear in the efficiency coefficient ce."""
    a = mle_coefficient(ce=0.06, lat_ref_deg=20.0)
    b = mle_coefficient(ce=0.12, lat_ref_deg=20.0)
    assert b == pytest.approx(2.0 * a, rel=1e-10)


def test_mle_coefficient_equator_guard() -> None:
    """A reference latitude at the equator (f0 -> 0) must raise, not blow up."""
    with pytest.raises(ValueError):
        mle_coefficient(ce=0.06, lat_ref_deg=0.0)
    with pytest.raises(ValueError):
        mle_coefficient(ce=0.06, lat_ref_deg=0.5)


def test_mle_coefficient_differentiable_wrt_ce() -> None:
    """``ce`` is the registered tunable (MLEConfig.ce, tier-2, SPEC_MODULES);
    ``mle_coefficient`` must be differentiable wrt a TRACED ce.  A prior
    ``float(ce)`` cast raised ConcretizationTypeError under extended-tier
    training (violating the differentiable=True contract).  Production still
    gets a Python float in -> Python float out (constant-folding preserved).
    (Physics review: ocean/mle faithfulness, 2026-06-29.)"""
    # Production path: Python-float ce -> Python float (no tracer leakage).
    rc_f = mle_coefficient(ce=0.06, lat_ref_deg=20.0)
    assert isinstance(rc_f, float)
    # Training path: traced ce -> finite gradient, no ConcretizationTypeError.
    g = jax.grad(lambda ce: mle_coefficient(ce, 20.0))(0.06)
    assert jnp.isfinite(g)
    # rc_f is linear in ce, so d(rc_f)/d(ce) = rc_f / ce.
    assert float(g) == pytest.approx(rc_f / 0.06, rel=1e-8)


def test_face_mld_modes() -> None:
    """face_mld picks min / avg / max of the two neighbour MLDs; unknown raises."""
    a = jnp.array([10.0, 50.0])
    b = jnp.array([30.0, 20.0])
    assert np.allclose(np.asarray(face_mld(a, b, "min")), [10.0, 20.0])
    assert np.allclose(np.asarray(face_mld(a, b, "avg")), [20.0, 35.0])
    assert np.allclose(np.asarray(face_mld(a, b, "max")), [30.0, 50.0])
    with pytest.raises(ValueError):
        face_mld(a, b, "median")


# ---------------------------------------------------------------------------
# (c) MLE mixed-layer depth + ML-mean buoyancy on a 2-layer column
# ---------------------------------------------------------------------------
def test_mld_and_buoyancy_two_layer_column() -> None:
    """A column with a sharp pycnocline at level 2 yields MLD = top-2 thickness
    and a positive ML-mean buoyancy (light water -> rho < rho0)."""
    nlev = 4
    # Uniform 100 m layers; level centres at 50, 150, 250, 350 m.
    dz = jnp.full((1, 1, nlev), 100.0)
    z_faces = jnp.array([0.0, 100.0, 200.0, 300.0, 400.0])
    wet = jnp.ones((1, 1, nlev))
    # Mixed layer = first two levels (light, rho = 1024); jump of +1 kg/m^3 at
    # level 2 exceeds the 0.01 criterion referenced to the ~10 m level (level 0).
    rho = jnp.array([[[1024.0, 1024.0, 1025.0, 1025.2]]])
    zmld, bm, _in_ml = mle_mld_and_buoyancy(
        rho, dz, wet,
        z_faces=z_faces, rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=1025.0, grav=constants.g,
    )
    # zmld / bm are 2-D (n_lat, n_lon) for a (1,1,nlev) column.
    assert np.asarray(zmld).shape == (1, 1)
    # First level denser than rho_ref + 0.01 is level 2 -> ML = levels {0, 1}.
    assert float(zmld[0, 0]) == pytest.approx(200.0, rel=1e-6)
    # bm = g·mean[(rho0 - rho)/rho0] over the ML.  rho0 = 1025 so the ML water
    # (1024) is LIGHTER than reference -> positive buoyancy.
    expected_bm = constants.g * ((1025.0 - 1024.0) / 1025.0)
    assert float(bm[0, 0]) == pytest.approx(expected_bm, rel=1e-6)


def test_mld_fully_mixed_column() -> None:
    """A column with no density jump is treated as fully mixed (MLD = full
    wet-column depth)."""
    nlev = 3
    dz = jnp.full((1, 1, nlev), 50.0)
    z_faces = jnp.array([0.0, 50.0, 100.0, 150.0])
    wet = jnp.ones((1, 1, nlev))
    rho = jnp.array([[[1024.0, 1024.0, 1024.0]]])
    zmld, _, _ = mle_mld_and_buoyancy(
        rho, dz, wet, z_faces=z_faces, rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=1025.0, grav=constants.g,
    )
    assert float(zmld[0, 0]) == pytest.approx(150.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Shared C-grid front fixture (d) + (e)
# ---------------------------------------------------------------------------
def _front_setup(n_lat=4, n_lon=5, nlev=6):
    """Build a small flat-bottom lat-lon C-grid with a zonal buoyancy front.

    Returns ``(T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg)``.
    The front is a smooth east-west SST gradient (warm/light west, cold/dense
    east) confined to the upper mixed layer; all cells are wet (flat bottom).
    """
    grid = create_latlon_grid(n_lat, n_lon, radius=constants.R_earth)
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=600.0, dz_surface=50.0, dz_deep=150.0,
    )
    mask = jnp.ones((n_lat, n_lon))
    u_mask, v_mask = compute_face_masks(mask, grid)
    J = jnp.ones((n_lat, n_lon))  # eta = 0 -> Jacobian 1 (flat bottom z*)

    # Mixed layer = top 3 levels.  Warm (light) in the west, cold (dense) east:
    # a monotone zonal SST gradient drives a single-signed bolus streamfunction.
    lon_frac = jnp.linspace(0.0, 1.0, n_lon)              # 0 (west) .. 1 (east)
    T_surf = 20.0 - 6.0 * lon_frac                         # 20 degC west -> 14 east
    T = jnp.zeros((n_lat, n_lon, nlev))
    for k in range(nlev):
        if k < 3:
            T = T.at[:, :, k].set(T_surf[jnp.newaxis, :])  # mixed (front) layers
        else:
            T = T.at[:, :, k].set(4.0)                     # cold deep water
    S = jnp.full((n_lat, n_lon, nlev), 35.0)

    # In-situ density via the model EOS (linear-ish; warm -> light).
    from legoesm.ocean.eos import wright_eos
    p = jnp.zeros_like(T)
    rho = wright_eos(T, S, p)
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, J)
    cfg = MLEConfig(ce=0.06, lat_ref_deg=20.0)
    return T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg


# ---------------------------------------------------------------------------
# (d) EXACT tracer conservation
# ---------------------------------------------------------------------------
def test_tracer_conservation_front() -> None:
    """The bolus tendency integrates to zero heat and salt: it only
    REDISTRIBUTES tracer (sum(dT·area·dz) ~ 0 to roundoff)."""
    T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg = _front_setup()
    dT, dS = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg,
    )
    n_lat, n_lon, nlev = T.shape
    area = np.asarray(grid.area)[:, :, None]              # (n_lat, n_lon, 1)
    dz = np.asarray(z_coord.dz_ref)[None, None, :] * np.asarray(J)[:, :, None]
    vol = area * dz                                       # (n_lat, n_lon, nlev)

    heat = float(np.sum(np.asarray(dT) * vol))
    salt = float(np.sum(np.asarray(dS) * vol))
    # Scale tolerance to the magnitude of the (unsigned) per-cell flux budget:
    # the bolus flux only redistributes tracer, so the signed sum telescopes to
    # zero while the unsigned sum is O(the actual transport).  Float64
    # telescoping leaves a ~1e-12 relative residual; 1e-9 is a comfortable
    # ceiling (matches the bbl_adv conservation gate).
    scale = float(np.sum(np.abs(np.asarray(dT)) * vol)) + 1e-30
    assert abs(heat) / scale < 1e-9, f"heat not conserved: {heat} (scale {scale})"
    scale_s = float(np.sum(np.abs(np.asarray(dS)) * vol)) + 1e-30
    assert abs(salt) / scale_s < 1e-9, f"salt not conserved: {salt}"


def test_tendency_is_nontrivial_on_front() -> None:
    """Sanity: the front actually produces a non-zero bolus tendency (so the
    conservation test above is not vacuously passing on an all-zero field)."""
    T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg = _front_setup()
    dT, _ = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg,
    )
    assert float(np.max(np.abs(np.asarray(dT)))) > 0.0


def test_no_front_no_tendency() -> None:
    """A horizontally uniform column (no buoyancy gradient) gives zero MLE
    tendency: dbm/dx = dbm/dy = 0 -> Psi = 0."""
    n_lat, n_lon, nlev = 4, 5, 6
    grid = create_latlon_grid(n_lat, n_lon, radius=constants.R_earth)
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=600.0, dz_surface=50.0, dz_deep=150.0)
    mask = jnp.ones((n_lat, n_lon))
    u_mask, v_mask = compute_face_masks(mask, grid)
    J = jnp.ones((n_lat, n_lon))
    T = jnp.zeros((n_lat, n_lon, nlev))
    for k in range(nlev):
        T = T.at[:, :, k].set(18.0 if k < 3 else 4.0)     # uniform horizontally
    S = jnp.full((n_lat, n_lon, nlev), 35.0)
    from legoesm.ocean.eos import wright_eos
    rho = wright_eos(T, S, jnp.zeros_like(T))
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, J)
    cfg = MLEConfig()
    dT, dS = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg)
    assert float(np.max(np.abs(np.asarray(dT)))) < 1e-12
    assert float(np.max(np.abs(np.asarray(dS)))) < 1e-12


# ---------------------------------------------------------------------------
# (e) Restratifying sign
# ---------------------------------------------------------------------------
def test_restratifying_warms_dense_side_at_surface() -> None:
    """Across a warm(light)/cold(dense) front the bolus advects light water
    over dense — it should COOL the warm (light) surface side and WARM the cold
    (dense) surface side, i.e. flatten the SST front (restratify).

    Concretely: with warm water to the west and cold to the east, the surface
    bolus tendency should be NEGATIVE on the warm (west) cells and POSITIVE on
    the cold (east) cells of the upper mixed layer, reducing the front.
    """
    T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg = _front_setup()
    dT, _ = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg,
    )
    dT = np.asarray(dT)
    # Surface (k=0), a mid-latitude row.  West end is the warm/light extreme,
    # east end the cold/dense extreme.
    row = dT.shape[0] // 2
    warm_west = dT[row, 0, 0]
    cold_east = dT[row, -1, 0]
    # Restratification flattens the front: cool the warm side, warm the cold side.
    assert warm_west < 0.0, f"warm (west) surface cell should cool, got {warm_west}"
    assert cold_east > 0.0, f"cold (east) surface cell should warm, got {cold_east}"


# ---------------------------------------------------------------------------
# (f) CLOSED overturning cell: uniform tracers are invariant
# ---------------------------------------------------------------------------
def test_uniform_tracer_zero_tendency_with_active_front() -> None:
    """REGRESSION (2026-08-11): a spatially UNIFORM tracer must have zero MLE
    tendency even where the streamfunction is active.

    The horizontal-only flux divergence conserved the GLOBAL sum while pumping
    tracer at transport-convergence cells (measured as -54 psu / -31 degC
    extremes at equatorial river-plume fronts after 30 days,
    results/omip_nemo/mle_psi_diag_d30).  The vertical continuity branch
    (NEMO zw_mle = -di[psi_uw] - dj[psi_vw]) closes the overturning cell, so a
    uniform field sees a divergence-free transport and is exactly invariant.
    """
    T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg = _front_setup()
    # Control: the operator is ACTIVE on this front (else the test is vacuous).
    dT_f, _ = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg,
    )
    scale = float(np.max(np.abs(np.asarray(dT_f))))
    assert scale > 1e-12, "front produced no tendency; test would be vacuous"
    # Uniform tracers against the SAME active rho front.
    Tu = jnp.full_like(T, 12.0)
    Su = jnp.full_like(S, 35.0)
    dTu, dSu = mle_tracer_tendency_latlon_cgrid(
        Tu, Su, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg,
    )
    assert float(np.max(np.abs(np.asarray(dTu)))) < 1e-9 * scale, (
        "uniform T gained a tendency: the bolus transport is not "
        "divergence-free per cell (missing/broken vertical branch)")
    assert float(np.max(np.abs(np.asarray(dSu)))) < 1e-9 * scale


def test_uniform_salinity_untouched_by_thermal_front() -> None:
    """S is uniform in the front fixture, so dS must vanish identically while
    dT carries the restratification -- the per-tracer face of the same
    closed-cell property (this is exactly the field the horizontal-only bug
    corrupted in production)."""
    T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg = _front_setup()
    dT, dS = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, N2, mask, u_mask, v_mask, z_coord, J, grid, cfg,
    )
    scale = float(np.max(np.abs(np.asarray(dT)))) + 1e-30
    assert float(np.max(np.abs(np.asarray(dS)))) < 1e-9 * scale
