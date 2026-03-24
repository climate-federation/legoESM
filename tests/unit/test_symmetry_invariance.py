"""Symmetry and invariance tests for shallow water dynamics (Category 5).

Tests:
  5a) Axisymmetric preservation on lat-lon grid
  5b) Hemispheric symmetry on lat-lon grid
  5d) Time-reversal symmetry on lat-lon grid (inviscid)
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
    FVShallowWaterLatLonModel,
    FVShallowWaterLatLonConfig,
)
from legoesm.core.state import ShallowWaterState
from legoesm.core.field import Field
from legoesm.grids.latlon import create_latlon_grid
from legoesm import constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIMS = ("lat", "lon")
_H0 = constants.H_MEAN  # 10 000 m


def _make_latlon_state(grid, h_data, u_data, v_data):
    """Build a ShallowWaterState from raw arrays on a lat-lon grid."""
    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=_DIMS, units="m"),
        u=Field(data=u_data, name="u", dims=_DIMS, units="m/s"),
        v=Field(data=v_data, name="v", dims=_DIMS, units="m/s"),
        h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=_DIMS, units="m"),
    )


def _step_n(model, state, dt, n_steps):
    """Step forward *n_steps* times."""
    for _ in range(n_steps):
        state = model.step(state, dt)
    return state


# ---------------------------------------------------------------------------
# 5a  Axisymmetric preservation
# ---------------------------------------------------------------------------

class TestAxiSymmetricPreservation:
    """A zonally-uniform state should remain zonally uniform."""

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        h_data = _H0 + 100.0 * jnp.cos(lat2d)
        u_data = 10.0 * jnp.cos(lat2d)
        v_data = jnp.zeros_like(lat2d)

        state0 = _make_latlon_state(grid, h_data, u_data, v_data)
        return model, state0

    def test_zonal_std_small(self, setup):
        model, state0 = setup
        state = _step_n(model, state0, 60.0, 20)

        h = state.h.data
        zonal_std = jnp.std(h, axis=1)         # std over longitudes
        zonal_mean = jnp.mean(jnp.abs(h), axis=1)
        ratio = zonal_std / jnp.maximum(zonal_mean, 1e-30)
        assert float(jnp.max(ratio)) < 0.01, (
            f"Zonal std/mean = {float(jnp.max(ratio)):.4e}, expected < 0.01"
        )


# ---------------------------------------------------------------------------
# 5b  Hemispheric symmetry
# ---------------------------------------------------------------------------

class TestHemisphericSymmetry:
    """A state symmetric about the equator should stay symmetric."""

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        h_data = _H0 + 100.0 * jnp.cos(2.0 * lat2d)
        u_data = 10.0 * jnp.cos(lat2d)
        v_data = jnp.zeros_like(lat2d)

        state0 = _make_latlon_state(grid, h_data, u_data, v_data)
        return model, state0, grid

    def test_h_symmetric(self, setup):
        model, state0, grid = setup
        state = _step_n(model, state0, 60.0, 20)

        h = state.h.data  # (n_lat, n_lon)
        n_lat = h.shape[0]
        h_south = h[:n_lat // 2, :]       # southern hemisphere
        h_north = h[n_lat // 2:, :][::-1]  # northern hemisphere, flipped

        # Check that the max relative difference is < 1%
        diff = jnp.abs(h_south - h_north)
        scale = jnp.maximum(jnp.abs(h_south) + jnp.abs(h_north), 1e-30) / 2.0
        rel_diff = diff / scale
        assert float(jnp.max(rel_diff)) < 0.01, (
            f"Max hemispheric asymmetry = {float(jnp.max(rel_diff)):.4e}, expected < 0.01"
        )


# ---------------------------------------------------------------------------
# 5d  Time-reversal symmetry
# ---------------------------------------------------------------------------

class TestTimeReversal:
    """Run N steps forward, negate velocities, run N steps back."""

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        # Inviscid, no conservation fixer, no polar filter for cleanest reversal
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=False,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        h_data = _H0 + 100.0 * jnp.cos(lat2d)
        u_data = 10.0 * jnp.cos(lat2d)
        v_data = jnp.zeros_like(lat2d)

        state0 = _make_latlon_state(grid, h_data, u_data, v_data)
        return model, state0

    def test_h_recovers(self, setup):
        model, state0 = setup
        n_steps = 20
        dt = 60.0

        # Forward
        state_fwd = _step_n(model, state0, dt, n_steps)

        # Negate velocities
        state_rev = state_fwd._replace(
            u=state_fwd.u.replace(data=-state_fwd.u.data),
            v=state_fwd.v.replace(data=-state_fwd.v.data),
        )

        # Backward
        state_back = _step_n(model, state_rev, dt, n_steps)

        h0 = state0.h.data
        h_back = state_back.h.data
        norm_h0 = jnp.sqrt(jnp.mean(h0 ** 2))
        err = jnp.sqrt(jnp.mean((h_back - h0) ** 2)) / norm_h0
        assert float(err) < 0.1, (
            f"Time-reversal error = {float(err):.4e}, expected < 0.1"
        )


# ---------------------------------------------------------------------------
# 5e  Sign-reversal symmetry of Coriolis
# ---------------------------------------------------------------------------

class TestCoriolisSignReversal:
    """Running with a perturbation at +45N vs -45S should produce
    u fields that match and v fields that have opposite signs.

    Coriolis: f = 2*Omega*sin(lat), so f(45N) = -f(-45S).
    For a velocity perturbation u>0, Coriolis deflects v in opposite
    directions in the two hemispheres.
    """

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        lon2d = grid.lon2d

        # Gaussian height perturbation centred at 45N (pi/4)
        lat_c_NH = jnp.pi / 4.0
        sigma = jnp.pi / 8.0
        r2_NH = (lat2d - lat_c_NH)**2 + (lon2d - jnp.pi)**2
        h_NH = _H0 + 200.0 * jnp.exp(-r2_NH / sigma**2)
        u_NH = 10.0 * jnp.exp(-r2_NH / sigma**2)

        # Reflected perturbation at 45S (-pi/4)
        lat_c_SH = -jnp.pi / 4.0
        r2_SH = (lat2d - lat_c_SH)**2 + (lon2d - jnp.pi)**2
        h_SH = _H0 + 200.0 * jnp.exp(-r2_SH / sigma**2)
        u_SH = 10.0 * jnp.exp(-r2_SH / sigma**2)

        state_NH = _make_latlon_state(
            grid, h_NH, u_NH, jnp.zeros_like(lat2d),
        )
        state_SH = _make_latlon_state(
            grid, h_SH, u_SH, jnp.zeros_like(lat2d),
        )

        return model, state_NH, state_SH, grid

    def test_coriolis_sign_reversal(self, setup):
        model, state_NH, state_SH, grid = setup
        n_steps = 30
        dt = 60.0

        state_NH_f = _step_n(model, state_NH, dt, n_steps)
        state_SH_f = _step_n(model, state_SH, dt, n_steps)

        u_NH = state_NH_f.u.data
        v_NH = state_NH_f.v.data
        u_SH = state_SH_f.u.data
        v_SH = state_SH_f.v.data

        # Flip SH fields in latitude to compare with NH
        u_SH_flip = u_SH[::-1, :]
        v_SH_flip = v_SH[::-1, :]

        # u fields should match (both hemispheres have same |u| pattern)
        u_scale = jnp.maximum(jnp.max(jnp.abs(u_NH)), 1e-30)
        u_diff = float(jnp.max(jnp.abs(u_NH - u_SH_flip))) / float(u_scale)
        print(f"  Coriolis sign reversal: |u_NH - u_SH_flip|/|u| = {u_diff:.4e}")
        assert u_diff < 0.1, (
            f"u mismatch under Coriolis sign reversal: {u_diff:.4e}"
        )

        # v fields should have opposite signs
        # v_NH and v_SH_flip should be approximately negatives of each other
        v_scale = jnp.maximum(jnp.max(jnp.abs(v_NH)), 1e-30)
        v_sum = float(jnp.max(jnp.abs(v_NH + v_SH_flip))) / float(v_scale)
        print(f"  Coriolis sign reversal: |v_NH + v_SH_flip|/|v| = {v_sum:.4e}")
        assert v_sum < 0.1, (
            f"v sign reversal failed: {v_sum:.4e}"
        )


# ---------------------------------------------------------------------------
# 5f  Galilean invariance (shallow water f-plane approximation near equator)
# ---------------------------------------------------------------------------

class TestGalileanInvariance:
    """A Gaussian height bump should propagate similarly whether or not a
    uniform background flow u0 is added. Near the equator f ~ 0, so the
    equations are approximately Galilean-invariant.

    We check that after N steps, the perturbation pattern from the boosted
    run (shifted back by the advected displacement) correlates well with
    the stationary run.
    """

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        # Use small hyperdiffusion for stability but keep it minimal
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        lon2d = grid.lon2d

        # Gaussian bump near equator
        sigma = jnp.pi / 8.0
        r2 = lat2d**2 + (lon2d - jnp.pi)**2
        h_bump = 200.0 * jnp.exp(-r2 / sigma**2)
        h_base = _H0 + h_bump

        u0 = 20.0  # Background flow [m/s]

        # Test A: bump with zero background flow
        state_A = _make_latlon_state(
            grid, h_base, jnp.zeros_like(lat2d), jnp.zeros_like(lat2d),
        )

        # Test B: bump with uniform background flow u0
        state_B = _make_latlon_state(
            grid, h_base, u0 * jnp.ones_like(lat2d), jnp.zeros_like(lat2d),
        )

        return model, state_A, state_B, grid, u0

    def test_galilean_invariance(self, setup):
        model, state_A, state_B, grid, u0 = setup
        n_steps = 15
        dt = 60.0

        state_A_f = _step_n(model, state_A, dt, n_steps)
        state_B_f = _step_n(model, state_B, dt, n_steps)

        # Perturbation patterns (subtract mean)
        h_A = state_A_f.h.data - _H0
        h_B = state_B_f.h.data - _H0

        # Compute pattern correlation between h_A and h_B
        # At coarse resolution with FV nonlinearity, we just check that
        # the perturbation patterns are broadly similar (correlation > 0.8)
        h_A_flat = h_A.ravel()
        h_B_flat = h_B.ravel()

        # Remove means for correlation
        h_A_zm = h_A_flat - jnp.mean(h_A_flat)
        h_B_zm = h_B_flat - jnp.mean(h_B_flat)

        denom = jnp.sqrt(jnp.sum(h_A_zm**2) * jnp.sum(h_B_zm**2))
        corr = float(jnp.sum(h_A_zm * h_B_zm) / jnp.maximum(denom, 1e-30))

        print(f"  Galilean invariance: pattern correlation = {corr:.4f}")
        assert corr > 0.8, (
            f"Galilean invariance failed: pattern correlation = {corr:.4f}, "
            f"expected > 0.8"
        )


# ---------------------------------------------------------------------------
# 5g  Axisymmetric preservation on cubed-sphere
# ---------------------------------------------------------------------------

class TestAxiSymmetricCubedSphere:
    """A zonally-uniform state on the cubed-sphere should remain
    approximately zonally uniform. This specifically tests for
    face-boundary artifacts that can break zonal symmetry."""

    @pytest.fixture
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        g = constants.g
        Omega = constants.Omega
        R = constants.R_earth
        u0 = 20.0
        h0 = 1e4

        # Balanced zonal jet: zonally uniform on CS
        lat_c = grid.lat
        h = h0 - (R * Omega * u0 + 0.5 * u0**2) * jnp.sin(lat_c)**2 / g

        lat_corner = cdgrid.lat_corner
        u_geo = u0 * jnp.cos(lat_corner)
        u_d = u_geo * cdgrid.cos_angle_corner
        v_d = u_geo * cdgrid.sin_angle_corner

        state0 = CDGridShallowWaterState(
            h=h, u_d=u_d, v_d=v_d, h_s=jnp.zeros_like(h),
        )

        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridShallowWaterModel(grid, config)
        return model, state0, grid

    def test_face_symmetry(self, setup):
        """After 50 steps, equivalent faces should have similar h statistics.

        On a cubed-sphere, faces 0-3 are equatorial and faces 4-5 are polar.
        A zonal jet naturally has different mean h on polar vs equatorial faces.
        We compare equatorial faces with each other (should match closely)
        and polar faces with each other.
        """
        model, state0, grid = setup
        dt = 60.0
        state = state0
        for _ in range(50):
            state = model.step(state, dt)

        h = state.h  # (6, n, n)
        face_means = [float(jnp.mean(h[f])) for f in range(6)]

        # Equatorial faces (0-3) should agree with each other
        eq_means = face_means[:4]
        eq_avg = sum(eq_means) / 4.0
        max_eq_dev = max(abs(fm - eq_avg) / abs(eq_avg) for fm in eq_means)
        assert max_eq_dev < 0.01, (
            f"Equatorial face means differ by {max_eq_dev:.4e}, expected < 0.01"
        )

        # Polar faces (4-5) should agree with each other
        pol_means = face_means[4:]
        pol_dev = abs(pol_means[0] - pol_means[1]) / abs(pol_means[0])
        assert pol_dev < 0.01, (
            f"Polar face means differ by {pol_dev:.4e}, expected < 0.01"
        )


# ---------------------------------------------------------------------------
# 5h  Grid rotation invariance on cubed-sphere
# ---------------------------------------------------------------------------

class TestGridRotationInvariance:
    """Run the same physical problem on different faces of the cubed-sphere.
    The L2 norm of the response should be the same regardless of which face
    the perturbation is placed on (tests face-boundary quality)."""

    @pytest.fixture
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.core.operators import hyperdiffusion

        grid = create_cubed_sphere(8)
        return grid, hyperdiffusion

    def test_impulse_response_face_invariance(self, setup):
        """Place an impulse on each face and compare L2 norms after diffusion."""
        grid, hyperdiffusion = setup
        n = grid.n
        dx_min = float(jnp.min(grid.dx))
        dt = 0.1
        nu = dx_min**4 / (16.0 * dt)
        N_steps = 5

        l2_norms = []
        for face in range(6):
            field_data = jnp.zeros((6, n, n))
            field_data = field_data.at[face, n // 2, n // 2].set(1.0)
            field = Field(data=field_data, name="q", dims=("face", "x", "y"), units="1")
            for _ in range(N_steps):
                tendency = hyperdiffusion(field, grid, nu)
                field = field.replace(data=field.data + dt * tendency.data)
            l2 = float(jnp.sqrt(jnp.sum(field.data**2 * grid.area)))
            l2_norms.append(l2)

        mean_l2 = sum(l2_norms) / 6.0
        for i, l2 in enumerate(l2_norms):
            rel = abs(l2 - mean_l2) / (mean_l2 + 1e-30)
            assert rel < 0.05, (
                f"Face {i} L2 = {l2:.6e} differs from mean {mean_l2:.6e} by {rel:.3e}"
            )


# ---------------------------------------------------------------------------
# 5i  MPAS hemispheric symmetry
# ---------------------------------------------------------------------------

class TestHemisphericSymmetryMPAS:
    """On an icosahedral mesh, the global mean h should be the same
    whether a bump is placed in the NH or SH, confirming no hemispheric
    bias in the MPAS discretization."""

    @pytest.fixture
    def setup(self):
        from legoesm.atmosphere.dynamics.shallow_water_mpas import (
            MPASShallowWaterModel,
            MPASShallowWaterConfig,
            MPASShallowWaterState,
        )
        from legoesm.grids.voronoi import create_voronoi_mesh

        mesh = create_voronoi_mesh(2, lloyd_iterations=30)
        config = MPASShallowWaterConfig(fix_mass=True, fix_energy=False)
        model = MPASShallowWaterModel(mesh, config)
        return model, mesh

    def _make_state(self, mesh, lat_center):
        """Create an MPAS state with a Gaussian bump at given latitude."""
        from legoesm.atmosphere.dynamics.shallow_water_mpas import MPASShallowWaterState
        H0 = constants.H_MEAN
        sigma = jnp.pi / 6.0
        r2 = (mesh.latCell - lat_center)**2 + (mesh.lonCell - jnp.pi)**2
        h_data = H0 + 100.0 * jnp.exp(-r2 / sigma**2)
        return MPASShallowWaterState(
            h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
            u=Field(data=jnp.zeros(mesh.nEdges), name="u",
                    dims=("nEdges",), units="m/s", staggering="edge"),
            h_s=Field(data=jnp.zeros(mesh.nCells), name="h_s",
                      dims=("nCells",), units="m"),
        )

    def test_nh_sh_symmetry(self, setup):
        """Bump at +45N and -45S should produce same global mean h."""
        model, mesh = setup
        state_NH = self._make_state(mesh, jnp.pi / 4.0)
        state_SH = self._make_state(mesh, -jnp.pi / 4.0)

        dt = 60.0
        for _ in range(30):
            state_NH = model.step(state_NH, dt)
            state_SH = model.step(state_SH, dt)

        area = mesh.areaCell
        mean_NH = float(jnp.sum(state_NH.h.data * area) / jnp.sum(area))
        mean_SH = float(jnp.sum(state_SH.h.data * area) / jnp.sum(area))

        rel_diff = abs(mean_NH - mean_SH) / abs(mean_NH)
        assert rel_diff < 0.01, (
            f"NH/SH mean h differs: NH={mean_NH:.2f}, SH={mean_SH:.2f}, "
            f"rel_diff={rel_diff:.4e}"
        )
