"""Sanity tests for the DCMIP 2012 §2-0-0 rest-state-with-topography IC."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh

from tests.test_cases.dcmip2012.rest_state_topography import (
    rest_state_topography_init,
    rest_state_topography_init_latlon,
    rest_state_topography_init_mpas,
    rest_state_topography_init_spectral,
)


@pytest.fixture(scope="module")
def sigma():
    return standard_hybrid_levels(8)


def _check_rest_state_invariants(state, *, h_0=2000.0, T_0=300.0,
                                  is_spectral=False):
    """Common sanity asserts for any rest-state-topography IC."""
    if is_spectral:
        # Spectral state stores complex coefficients; check finiteness
        # and rest-condition via vor=div=0.
        assert float(jnp.max(jnp.abs(state.vor_hat.data))) < 1e-10
        assert float(jnp.max(jnp.abs(state.div_hat.data))) < 1e-10
        return

    # Grid-space states: u=v=0 exactly, T = T_0 everywhere.
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    if state.v is not None:
        assert float(jnp.max(jnp.abs(state.v.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.T.data - T_0))) < 1e-6

    # Surface geopotential: 0 ≤ phis ≤ g·h_0
    assert float(jnp.min(state.phis.data)) >= 0.0
    assert float(jnp.max(state.phis.data)) <= constants.g * h_0 * 1.001

    # Surface pressure: hydrostatically reduced from p_ref by mountain
    # height — p_s_min = p_ref·exp(-g·h_0/(R_d·T_0))
    p_min_expected = constants.p_ref * jnp.exp(
        -constants.g * h_0 / (constants.R_d * T_0))
    assert float(jnp.min(state.p_s.data)) >= float(p_min_expected) * 0.99
    assert float(jnp.max(state.p_s.data)) <= constants.p_ref + 1e-6


def test_rest_state_cubed_sphere(sigma):
    grid = create_cubed_sphere(8)
    state = rest_state_topography_init(grid, sigma, h_0=2000.0)
    _check_rest_state_invariants(state)


def test_rest_state_latlon(sigma):
    grid = create_latlon_grid(24, 48)
    state = rest_state_topography_init_latlon(grid, sigma, h_0=2000.0)
    _check_rest_state_invariants(state)


def test_rest_state_mpas(sigma):
    mesh = create_voronoi_mesh(4)
    state = rest_state_topography_init_mpas(mesh, sigma, h_0=2000.0)
    # MPAS state has u at edges only, no v
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.T.data - 300.0))) < 1e-6
    assert float(jnp.min(state.phis.data)) >= 0.0


def test_rest_state_spectral(sigma):
    grid = create_gaussian_grid(21)
    state = rest_state_topography_init_spectral(grid, sigma, h_0=2000.0)
    _check_rest_state_invariants(state, is_spectral=True)


def test_lapse_rate_spectral_ic_survives_first_step_projection():
    """The spectral model truncates lnps to the 2/3 band every step.  With
    the terrain band-limited first, the p_s change over one step is bounded
    by 10 Pa (T21 matrix config, fp64: 1.06 Pa; on the raw-terrain IC the
    first step removes 2168 Pa at the mountain)."""
    from legoesm.grids.gaussian import sh_synthesis
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel)
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_spectral)

    grid = create_gaussian_grid(21, omega=0.0)
    coord = make_cam6_l32_levels()
    cfg = SpectralPEConfig(hyperdiff_coeff=2.338e15, spectral_filter_order=8,
                           spectral_filter_strength=0.01, fix_mass=True,
                           anchor_mass_to_initial=True,
                           T_damping_pressure_surfaces=True)
    s0 = lapse_rate_state_topography_init_spectral(
        grid, coord, h_0=2000.0, dealiasing_fraction=cfg.dealiasing_fraction)
    s1 = SpectralPrimitiveEquationModel(grid, coord, cfg).step(s0, 600.0)
    ps = lambda s: jnp.exp(sh_synthesis(grid, s.lnps_hat.data))
    assert float(jnp.max(jnp.abs(ps(s1) - ps(s0)))) < 10.0


def test_matrix_spectral_rest_state_topo_row_ps_perturbation(tmp_path):
    """The matrix row itself (run_baroclinic, spectral T21, 6 h): its
    ps_perturbation diagnostic (max |p_s - ps_init_spec|: dynamics plus the
    ~1 Pa residual projection of the IC's off-band lnps) is bounded by 10 Pa
    at 6 h (before the fix: 2126 Pa, the first-step projection of the raw-
    terrain IC; after: ~1.6 Pa), and the anchored dry mass holds."""
    import numpy as np
    from scripts.matrix.run_atmosphere_test_matrix import (
        _DYCORE_MASS_DRIFT_TOL, TestCase as MatrixCase, run_baroclinic)
    tc = MatrixCase("hydrostatic", "rest_state_topo", "spectral", "T21",
                    "hybrid", 7, 1, {"h_0": 2000.0})
    status, _, _ = run_baroclinic(tc, tmp_path, 0.25)
    assert status == "PASS"
    csv = next(tmp_path.rglob("mean_timeseries.csv"))
    with open(csv) as f:
        header = f.readline().strip().split(",")
    rows = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert rows[-1, header.index("time_days")] == 0.25
    assert rows[-1, header.index("ps_perturbation")] < 10.0
    # Row 0 is the diagnostic of the IC state itself (step 0, before any
    # step), so mass[0] is the initialised state's integrated mass.
    assert rows[0, header.index("step")] == 0
    mass = rows[:, header.index("mass")]
    assert np.max(np.abs(mass / mass[0] - 1.0)) < _DYCORE_MASS_DRIFT_TOL
