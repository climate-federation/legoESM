"""Symmetry and invariance tests for dynamics (Category 5).

Tests:
  5g) Axisymmetric preservation on cubed-sphere
  5h) Grid rotation invariance on cubed-sphere
  5i) MPAS hemispheric symmetry
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm import constants


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
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
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
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import MPASShallowWaterState
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
