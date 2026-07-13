"""Flux-form continuity validation for the cubed-sphere + MPAS PE dycores.

The surface-pressure tendency and the vertical mass flux / sigma-dot must be
built from the FLUX-FORM layer-mass divergence div(dp_k·v) — NOT the advective
form div(v)·dp_k (dp pulled out of the divergence).  The two differ wherever
∇p_s ≠ 0; only the flux form telescopes (face/edge fluxes cancel in pairs), so
the area-integrated surface-pressure tendency vanishes to machine precision
without any global fixer.  Reference construction:
``primitive_eq_latlon_cgrid.py`` sec 9/10 (dp to faces → div(dp·v) → cumsum →
dp_s_dt + shared-closure mass_flux / sigma_dot).

Measured residuals |∫dp_s/dt·dA| / ∫p_s·dA on a Held-Suarez-topo state
(2000 m DCMIP mountain, 10 spin steps, CPU x64, this fix):

  ==================  ===========  ======================
  config              flux form    advective (same state)
  ==================  ===========  ======================
  cube σ duogrid      9.0e-25      2.3e-11
  cube hyb duogrid    7.0e-26      4.1e-11
  cube σ no-duogrid   1.6e-11      7.0e-12
  MPAS σ              9.8e-25      1.5e-10
  MPAS hyb            3.0e-25      1.8e-10
  ==================  ===========  ======================

Non-duogrid cube: the panel-seam fluxes are built from LAGRANGE-INTERPOLATED
cross-face halos, so the two panels sharing an edge disagree at interpolation
precision — the residual is seam quadrature, not the advective-form error
(with duogrid the seam fluxes are synchronized and the integral telescopes
exactly).  15-step fix_mass=False dry-mass drift (measured): cube duogrid
2.1e-15 (σ) / 8.1e-16 (hyb); MPAS 1.4e-14 (both) — the advective form left an
O(v·∇p_s) integral no discrete cancellation removes.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    make_hybrid_levels,
    dp_from_hybrid,
    compute_sigma_dot,
    compute_sigma_dot_from_cumsum,
)

NLEV = 8
N_SPIN = 10          # steps to develop winds from the mountain-balanced IC
N_DRIFT = 15         # fix_mass=False drift horizon
DT_CUBE = 300.0      # [s] C12
DT_MPAS = 300.0      # [s] ico3

# Flux-form budget gates.  Measured values (docstring table) sit 12-13 orders
# below the 1e-12 gate on the exact-telescoping paths; the non-duogrid cube
# seam-quadrature path gets a 1e-9 gate (measured 1.6e-11).
BUDGET_TOL = 1e-12
BUDGET_TOL_SEAM = 1e-9
DRIFT_TOL = 1e-12    # measured ≤ 2.1e-15 (cube duogrid) / 1.4e-14 (MPAS)


def _coord(name):
    return (create_sigma_coordinate(NLEV) if name == "sigma"
            else make_hybrid_levels(NLEV))


def _rel_budget(dp_s_dt, p_s, area):
    """|∫ dp_s/dt dA| / ∫ p_s dA  [1/s] — fractional dry-mass tendency."""
    num = abs(float(jnp.sum(
        dp_s_dt.astype(jnp.float64) * area.astype(jnp.float64))))
    den = float(jnp.sum(
        p_s.astype(jnp.float64) * area.astype(jnp.float64)))
    return num / den


def _advective_dps_dt(div_v, p_s, coord, name):
    """The OLD advective closure on the same state (for the same-state
    comparison): sigma: -p_s·Σ(div·Δσ)/(1-σ_top); hybrid: -Σ(div·dp)/B_range."""
    if name == "sigma":
        sigma_range = 1.0 - coord.sigma_half[0]
        return -p_s * jnp.sum(div_v * coord.dsigma, axis=-1) / sigma_range
    dp = dp_from_hybrid(coord, p_s)
    return -jnp.sum(div_v * dp, axis=-1) / coord.B_range


# ---------------------------------------------------------------------------
# Shared closure unit identity: flux-form σ̇ == advective σ̇ when ∇p_s = 0
# (div_dp = p_s·Δσ·div(v) exactly), and hard σ̇=0 boundaries.
# ---------------------------------------------------------------------------

def test_sigma_dot_from_cumsum_uniform_ps_identity():
    # Explicit f64 coordinate: without a precision policy the builder
    # defaults its arrays to f32, which would put the comparison at f32 eps.
    coord = create_sigma_coordinate(NLEV, dtype=jnp.float64)
    rng = np.random.default_rng(7)
    div_v = jnp.asarray(rng.standard_normal((4, 5, NLEV)) * 1e-5)
    p_s = jnp.full((4, 5), 1.0e5)

    sd_adv = compute_sigma_dot(div_v, coord)
    div_dp = div_v * (p_s[..., None] * coord.dsigma)
    cumsum = jnp.cumsum(div_dp, axis=-1)
    sd_flux = compute_sigma_dot_from_cumsum(
        cumsum, cumsum[..., -1:], p_s, coord)

    assert sd_flux.shape == (4, 5, NLEV + 1)
    np.testing.assert_allclose(
        np.asarray(sd_flux), np.asarray(sd_adv), rtol=1e-10, atol=1e-18)
    # Hard boundary closure: σ̇ exactly zero at top + surface.
    assert float(jnp.max(jnp.abs(sd_flux[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(sd_flux[..., -1]))) == 0.0


# ---------------------------------------------------------------------------
# Cubed-sphere (FV3 C-D grid) PE
# ---------------------------------------------------------------------------

def _cube_model_state(coord_name, use_duogrid):
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.idealized.held_suarez_topo import (
        held_suarez_topo_init,
    )

    grid = create_cubed_sphere(12, use_duogrid=use_duogrid)
    coord = _coord(coord_name)
    cfg = CDGridPrimitiveEquationConfig(
        fix_mass=False,
        use_conservation_fixer=False,
        zero_mean_ps_tendency=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    state = held_suarez_topo_init(grid, coord)
    return grid, coord, model, state


def _cube_budget_residuals(coord_name, use_duogrid):
    from legoesm.core.operators_cdgrid import (
        dgrid_to_cgrid, cgrid_divergence, center_to_dgrid_vector,
    )

    grid, coord, model, state = _cube_model_state(coord_name, use_duogrid)
    for _ in range(N_SPIN):
        state = model.step(state, DT_CUBE)
    assert bool(jnp.isfinite(state.p_s.data).all())
    p_s = state.p_s.data

    tend = model.tendencies(state)
    res_flux = _rel_budget(tend.dp_s_dt.data, p_s, grid.area)

    # Advective closure on the SAME state (the pre-fix formula).
    u_d, v_d = center_to_dgrid_vector(state.u.data, state.v.data, model.cdgrid)
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, model.cdgrid)
    div_v = cgrid_divergence(u_c, v_c, model.cdgrid)
    res_adv = _rel_budget(
        _advective_dps_dt(div_v, p_s, coord, coord_name), p_s, grid.area)
    return res_flux, res_adv


@pytest.mark.parametrize("coord_name", ["sigma", "hybrid"])
def test_cube_flux_form_ps_budget_duogrid(coord_name):
    """Duogrid cube: seam fluxes synchronized → the budget telescopes to
    machine zero (measured 9.0e-25 σ / 7.0e-26 hyb) while the advective
    closure on the SAME state leaves an O(v·∇p_s) integral (~1e-11)."""
    res_flux, res_adv = _cube_budget_residuals(coord_name, use_duogrid=True)
    assert res_flux < BUDGET_TOL, (
        f"cube {coord_name} duogrid flux-form budget {res_flux:.3e} "
        f"exceeds {BUDGET_TOL:.0e}")
    assert res_flux < 1e-3 * res_adv, (
        f"flux-form residual {res_flux:.3e} not ≪ advective {res_adv:.3e}")


def test_cube_flux_form_ps_budget_nonduogrid():
    """Non-duogrid cube: panel-seam fluxes come from Lagrange-interpolated
    halos, so the budget closes only to seam-interpolation precision
    (measured 1.6e-11 — cube edge quadrature, reported per task spec)."""
    res_flux, _ = _cube_budget_residuals("sigma", use_duogrid=False)
    assert res_flux < BUDGET_TOL_SEAM, (
        f"cube sigma non-duogrid flux-form budget {res_flux:.3e} exceeds "
        f"{BUDGET_TOL_SEAM:.0e}")


@pytest.mark.parametrize("coord_name", ["sigma", "hybrid"])
def test_cube_mass_drift_no_fixer_duogrid(coord_name):
    """15 steps, fix_mass=False, duogrid: dry mass must hold at the fp64
    floor purely from the discrete flux telescoping (measured 2.1e-15 σ /
    8.1e-16 hyb; the advective form drifted at fixer-masked O(v·∇p_s))."""
    grid, _, model, state = _cube_model_state(coord_name, use_duogrid=True)

    def _mass(s):
        return float(jnp.sum(
            s.p_s.data.astype(jnp.float64) * grid.area.astype(jnp.float64)))

    m0 = _mass(state)
    for _ in range(N_DRIFT):
        state = model.step(state, DT_CUBE)
    assert bool(jnp.isfinite(state.p_s.data).all())
    drift = abs(_mass(state) - m0) / m0
    assert drift < DRIFT_TOL, (
        f"cube {coord_name} duogrid {N_DRIFT}-step no-fixer drift "
        f"{drift:.3e} exceeds {DRIFT_TOL:.0e}")


def test_cube_flux_equals_advective_uniform_ps():
    """Operator identity: with UNIFORM p_s the two closures coincide
    (div(dp·v) = dp·div(v) when dp is horizontally constant) — guards the
    flux-form wiring against a normalization/sign slip."""
    from legoesm.core.operators_cdgrid import (
        dgrid_to_cgrid, cgrid_divergence, cgrid_flux_divergence_sync,
        cgrid_interp_cc_to_faces_local, pad_halo_auto,
    )
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    grid = create_cubed_sphere(12)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = create_sigma_coordinate(NLEV)
    rng = np.random.default_rng(3)
    u_d = jnp.asarray(rng.standard_normal((6, 13, 13, NLEV)))
    v_d = jnp.asarray(rng.standard_normal((6, 13, 13, NLEV)))
    p_s = jnp.full((6, 12, 12), 1.0e5)

    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    dp = p_s[..., None] * coord.dsigma
    dp_u, dp_v = cgrid_interp_cc_to_faces_local(pad_halo_auto(dp, cdgrid))
    div_dp = cgrid_flux_divergence_sync(dp_u, dp_v, u_c, v_c, cdgrid)
    sigma_range = 1.0 - coord.sigma_half[0]
    dps_flux = -jnp.sum(div_dp, axis=-1) / sigma_range

    div_v = cgrid_divergence(u_c, v_c, cdgrid)
    dps_adv = _advective_dps_dt(div_v, p_s, coord, "sigma")
    np.testing.assert_allclose(
        np.asarray(dps_flux), np.asarray(dps_adv), rtol=2e-12, atol=0)


# ---------------------------------------------------------------------------
# MPAS (Voronoi / TRiSK) PE
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fp64_policy():
    """fp64 storage policy for EVERY test here (autouse).

    ``jax_enable_x64`` alone is NOT enough: the repo's PrecisionPolicy
    defaults to fp32 storage for the FV dycores, and the cube drift /
    flux-vs-advective identity gates below sit at fp64 machine precision
    (measured cube drift 4e-7 under fp32 vs 2e-15 under fp64).
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy

    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _mpas_model_state(coord_name):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.idealized.held_suarez_topo import (
        held_suarez_topo_init_mpas,
    )

    mesh = create_voronoi_mesh(3)
    coord = _coord(coord_name)
    cfg = MPASPrimitiveEquationConfig(fix_mass=False)
    model = MPASPrimitiveEquationModel(mesh, coord, cfg)
    state = held_suarez_topo_init_mpas(mesh, coord)
    return mesh, coord, model, state


@pytest.mark.parametrize("coord_name", ["sigma", "hybrid"])
def test_mpas_flux_form_ps_budget(coord_name, fp64_policy):
    """TRiSK edge fluxes appear ± in exactly the two cells sharing the edge,
    so the flux-form budget telescopes exactly on the closed sphere
    (measured 9.8e-25 σ / 3.0e-25 hyb vs advective 1.5e-10 / 1.8e-10)."""
    from legoesm.core.operators_voronoi import divergence_cell_3d

    mesh, coord, model, state = _mpas_model_state(coord_name)
    for _ in range(N_SPIN):
        state = model.step(state, DT_MPAS)
    assert bool(jnp.isfinite(state.p_s.data).all())
    p_s = state.p_s.data

    tend = model.tendencies(state)
    res_flux = _rel_budget(tend.dp_s_dt.data, p_s, mesh.areaCell)

    div_v = divergence_cell_3d(state.u.data, mesh)
    res_adv = _rel_budget(
        _advective_dps_dt(div_v, p_s, coord, coord_name), p_s, mesh.areaCell)

    assert res_flux < BUDGET_TOL, (
        f"MPAS {coord_name} flux-form budget {res_flux:.3e} exceeds "
        f"{BUDGET_TOL:.0e}")
    assert res_flux < 1e-3 * res_adv, (
        f"flux-form residual {res_flux:.3e} not ≪ advective {res_adv:.3e}")


@pytest.mark.parametrize("coord_name", ["sigma", "hybrid"])
def test_mpas_mass_drift_no_fixer(coord_name, fp64_policy):
    """15 steps, fix_mass=False: dry mass holds at the fp64 floor purely from
    the discrete edge-flux telescoping (measured 1.4e-14 both coords)."""
    mesh, _, model, state = _mpas_model_state(coord_name)

    def _mass(s):
        return float(jnp.sum(
            s.p_s.data.astype(jnp.float64)
            * mesh.areaCell.astype(jnp.float64)))

    m0 = _mass(state)
    for _ in range(N_DRIFT):
        state = model.step(state, DT_MPAS)
    assert bool(jnp.isfinite(state.p_s.data).all())
    drift = abs(_mass(state) - m0) / m0
    assert drift < DRIFT_TOL, (
        f"MPAS {coord_name} {N_DRIFT}-step no-fixer drift {drift:.3e} "
        f"exceeds {DRIFT_TOL:.0e}")
