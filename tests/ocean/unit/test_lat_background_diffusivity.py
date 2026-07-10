"""Unit tests for the latitude-dependent background vertical diffusivity
(MED-2: Gregg et al. 2003 / CVMix ``bkgnd`` Henyey-Wright-Flatte scaling).

Two layers are exercised directly (import + call the real functions):

  * :func:`...vertical_mixing._shared.latitude_background_diffusivity` — the
    scaling kernel: equatorial reduction (K_bg(equator) < K_bg(midlat) for the
    SAME N), monotonicity in ``|lat|``, the ``N -> 0`` / equator floors (no
    NaN), bounded in ``[K_bg_eq, K_bg_pole]``, and differentiability;
  * :func:`...vertical_mixing.k_profiles.compute_vertical_K_profiles` — the
    wiring into the ``constant`` scheme: ``lat_dependent=False`` is BYTE-
    IDENTICAL to the legacy constant background, ``lat_dependent=True`` produces
    a latitude-varying field with the configured Prandtl ratio ``A_v/K_v``, and
    a missing ``lat_deg`` raises.

Plus the fail-loud guard on the EXPLICIT ``constant_vertical_mixing`` path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing._shared import (
    latitude_background_diffusivity,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    ConstantVerticalMixingConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.constant import (
    constant_vertical_mixing,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def grid_z_state():
    grid = create_latlon_grid(n_lat=8, n_lon=12)
    z = create_ocean_z_star(n_levels=5, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    return grid, z, state


def _base(**over):
    return dict(
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(scheme="none"),
        **over,
    )


# ---------------------------------------------------------------------------
# Direct kernel tests
# ---------------------------------------------------------------------------
class TestLatitudeScalingKernel:
    CFG = ConstantVerticalMixingConfig(K_bg_eq=1e-5, K_bg_pole=1e-4)

    def test_equator_weaker_than_midlat_same_N(self):
        # Same stratification at both columns; only latitude differs.
        N2 = jnp.full((2, 4), 1e-5)
        lat = jnp.array([0.0, 45.0])          # equator, midlat (leading-axis)
        K = latitude_background_diffusivity(lat, N2, self.CFG)
        assert K.shape == (2, 4)
        assert bool(jnp.all(K[0] < K[1]))                    # equator reduced
        assert bool(jnp.all(K >= self.CFG.K_bg_eq - 1e-15))  # >= equatorial floor
        assert bool(jnp.all(K <= self.CFG.K_bg_pole + 1e-15))  # <= polar cap

    def test_interpolates_strictly_between_for_weak_N(self):
        # Weak N so the midlat point does NOT saturate at the polar cap.
        N2 = jnp.full((1, 3), 1e-7)
        K = latitude_background_diffusivity(jnp.array([45.0]), N2, self.CFG)
        assert bool(jnp.all(K > self.CFG.K_bg_eq))
        assert bool(jnp.all(K < self.CFG.K_bg_pole))

    def test_monotone_nondecreasing_when_well_stratified(self):
        # Well-stratified column (|f| << N): the Gregg shape is monotone in
        # |lat| up to the K_bg_pole cap.
        lat = jnp.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 90.0])
        K = latitude_background_diffusivity(lat, jnp.full((8, 3), 1e-5), self.CFG)
        col = K[:, 0]
        assert bool(jnp.all(jnp.diff(col) >= -1e-18))   # non-decreasing
        assert bool(col[0] < col[-1])                   # grows overall

    def test_weak_stratification_stays_bounded(self):
        # For weak N ~ |f| the raw Henyey shape turns over at high latitude
        # (a known Gregg property) — not asserted monotone, but MUST stay
        # bounded in [K_bg_eq, K_bg_pole] and finite.
        lat = jnp.array([0.0, 10.0, 30.0, 60.0, 80.0, 90.0])
        K = latitude_background_diffusivity(lat, jnp.full((6, 3), 5e-8), self.CFG)
        assert bool(jnp.all(jnp.isfinite(K)))
        assert bool(jnp.all(K >= self.CFG.K_bg_eq - 1e-15))
        assert bool(jnp.all(K <= self.CFG.K_bg_pole + 1e-15))

    def test_symmetric_in_hemisphere(self):
        N2 = jnp.full((2, 3), 1e-6)
        K = latitude_background_diffusivity(jnp.array([-40.0, 40.0]), N2, self.CFG)
        assert jnp.allclose(K[0], K[1])                 # depends on |lat| only

    @pytest.mark.parametrize("n2_val", [0.0, -1.0e-6, 1.0e-12])
    def test_N_to_zero_or_negative_no_nan(self, n2_val):
        # N -> f floor / unstratified / convecting columns must not NaN, and
        # must stay within [K_bg_eq, K_bg_pole]; the equator worst case (|f|->0).
        lat = jnp.array([0.0, 30.0, 60.0, 90.0])
        K = latitude_background_diffusivity(lat, jnp.full((4, 3), n2_val), self.CFG)
        assert bool(jnp.all(jnp.isfinite(K)))
        assert bool(jnp.all(K >= self.CFG.K_bg_eq - 1e-15))
        assert bool(jnp.all(K <= self.CFG.K_bg_pole + 1e-15))

    def test_equator_returns_floor(self):
        # Exactly at the equator, |f|->0 so L->0 and K -> K_bg_eq.
        K = latitude_background_diffusivity(
            jnp.array([0.0]), jnp.full((1, 3), 1e-5), self.CFG)
        assert jnp.allclose(K, self.CFG.K_bg_eq, atol=1e-8)

    def test_two_dim_lat_broadcasts(self):
        # A full (n_lat, n_lon) latitude field must broadcast like a 1-D one.
        lat2d = jnp.array([[0.0, 0.0], [60.0, 60.0]])       # (2, 2)
        N2 = jnp.full((2, 2, 3), 1e-5)                       # (2, 2, nlev-1)
        K = latitude_background_diffusivity(lat2d, N2, self.CFG)
        assert K.shape == (2, 2, 3)
        assert bool(jnp.all(K[0] < K[1]))                    # equator < 60 deg

    def test_differentiable_no_nan_grad(self):
        cfg = self.CFG
        lat = jnp.array([0.0, 30.0, 60.0, 90.0])

        def loss_n2(N2):
            return jnp.sum(latitude_background_diffusivity(lat, N2, cfg))

        # Normal + the N2=0 floor both give finite gradients.
        assert bool(jnp.all(jnp.isfinite(jax.grad(loss_n2)(jnp.full((4, 3), 1e-5)))))
        assert bool(jnp.all(jnp.isfinite(jax.grad(loss_n2)(jnp.zeros((4, 3))))))

        def loss_lat(latd):
            return jnp.sum(
                latitude_background_diffusivity(latd, jnp.full((4, 3), 1e-5), cfg))

        # Gradient wrt latitude finite even at the equator (|f| floor).
        assert bool(jnp.all(jnp.isfinite(jax.grad(loss_lat)(lat))))


# ---------------------------------------------------------------------------
# Wiring into compute_vertical_K_profiles (implicit path)
# ---------------------------------------------------------------------------
class TestConstantSchemeWiring:
    def test_lat_dependent_false_is_byte_identical(self, grid_z_state):
        grid, z, state = grid_z_state
        # Non-default lat params but the flag is OFF -> must reproduce the
        # legacy spatially-constant field EXACTLY.
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(
                    A_v=3e-3, K_v=7e-4, lat_dependent=False,
                    K_bg_eq=2e-5, K_bg_pole=5e-4, N_ref=6e-3),
            ),
            **_base())
        # Threading lat_deg must not perturb the OFF path either.
        K0, A0 = compute_vertical_K_profiles(state, z, None, cfg)
        K1, A1 = compute_vertical_K_profiles(
            state, z, None, cfg, lat_deg=jnp.degrees(grid.lat))
        assert bool(jnp.all(K0 == 7e-4)) and bool(jnp.all(A0 == 3e-3))
        assert bool(jnp.all(K1 == 7e-4)) and bool(jnp.all(A1 == 3e-3))

    def test_lat_dependent_true_varies_with_latitude(self, grid_z_state):
        grid, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(
                    A_v=1e-3, K_v=1e-4, lat_dependent=True,
                    K_bg_eq=1e-5, K_bg_pole=1e-4),
            ),
            **_base())
        K, A = compute_vertical_K_profiles(
            state, z, None, cfg, lat_deg=jnp.degrees(grid.lat))
        nlev = state.T.data.shape[-1]
        assert K.shape == state.T.data.shape[:-1] + (nlev - 1,)
        assert bool(jnp.all(jnp.isfinite(K))) and bool(jnp.all(jnp.isfinite(A)))
        # Rows (S->N): idx 3 ~ -11.25 deg (near equator), idx 0 ~ -78.75 deg.
        assert bool(jnp.all(K[3] < K[0]))               # equatorward reduced
        # Bounded within the configured [eq, pole] range.
        assert bool(jnp.all(K >= 1e-5 - 1e-15))
        assert bool(jnp.all(K <= 1e-4 + 1e-15))

    def test_viscosity_preserves_prandtl_ratio(self, grid_z_state):
        grid, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(
                    A_v=1e-3, K_v=1e-4, lat_dependent=True,
                    K_bg_eq=1e-5, K_bg_pole=1e-4),
            ),
            **_base())
        K, A = compute_vertical_K_profiles(
            state, z, None, cfg, lat_deg=jnp.degrees(grid.lat))
        # A_v = K_v * (A_v_cfg / K_v_cfg) -> constant Prandtl ratio everywhere.
        assert jnp.allclose(A / K, 1e-3 / 1e-4)

    def test_lat_dependent_true_requires_lat_deg(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(lat_dependent=True),
            ),
            **_base())
        with pytest.raises(ValueError, match="lat_deg"):
            compute_vertical_K_profiles(state, z, None, cfg, lat_deg=None)


class TestExplicitPathFailsLoud:
    def test_explicit_constant_path_raises_on_lat_dependent(self):
        z = create_ocean_z_star(n_levels=4, H_max=1000.0)
        q = jnp.zeros((2, 3, 4))
        jac = jnp.ones((2, 3))
        cfg = ConstantVerticalMixingConfig(lat_dependent=True)
        with pytest.raises(ValueError, match="lat_dependent"):
            constant_vertical_mixing(q, q, q, q, z, jac, cfg)

    def test_explicit_constant_path_ok_when_off(self):
        # Sanity: the guard does not disturb the legacy (flag off) explicit path.
        z = create_ocean_z_star(n_levels=4, H_max=1000.0)
        q = jnp.zeros((2, 3, 4))
        jac = jnp.ones((2, 3))
        out = constant_vertical_mixing(
            q, q, q, q, z, jac, ConstantVerticalMixingConfig())
        assert bool(jnp.all(jnp.isfinite(out.K_v)))
