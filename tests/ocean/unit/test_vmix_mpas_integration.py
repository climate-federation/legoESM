"""Direct unit tests for the MPAS Voronoi-mesh vertical-mixing adapters.

``ocean/physics/vertical_mixing/mpas_integration.py`` bridges KPP onto the MPAS
C-grid (T/S at cells, edge-normal u at edges).  These tests exercise its public
factories and the two private kernels directly on a small ``subdivision_level=1``
Voronoi mesh (42 cells / 120 edges) plus tiny synthetic arrays:

  * ``make_kpp_physics_mpas`` — returns ``(du_dt_edge, dT_dt_cell, dS_dt_cell)``
    with the right shapes, all finite, land/sub-seafloor-masked.
  * ``make_kpp_profiles_mpas`` — returns ``(A_v_cells, K_v_cells)`` at half
    levels, both non-negative and finite (the implicit-solver inputs).
  * ``_vertical_diffusion_edge_partial`` — second-order vertical-diffusion
    stencil with per-edge thicknesses: zero on a uniform field, volume-
    conserving on a sheared field, zero on a single level.
  * ``_mpas_surface_buoyancy_flux`` — the MPAS surface-forcing convention
    (real salt feeds buoyancy only); ``None`` when no forcing channel is present.

The KPP boundary-layer numerics themselves are validated by the KPP suite; here
we pin the MPAS *adapter* contract (shapes, masking, finiteness, sign).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig, VerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
    _mpas_surface_buoyancy_flux,
    _vertical_diffusion_edge_partial,
    make_kpp_physics_mpas,
    make_kpp_profiles_mpas,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=1)


@pytest.fixture(scope="module")
def z_coord():
    return create_ocean_z_star(n_levels=6, H_max=4000.0)


@pytest.fixture(scope="module")
def state(mesh, z_coord):
    return rest_state_mpas_ocean(
        mesh, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=85.0,
    )


@pytest.fixture
def kpp_cfg():
    return VerticalMixingConfig(scheme="kpp", kpp=KPPConfig())


class TestKPPPhysicsMPAS:
    def test_shapes_and_finite(self, mesh, z_coord, state, kpp_cfg):
        fn = make_kpp_physics_mpas(kpp_cfg)
        du_dt_edge, dT_dt, dS_dt = fn(state, mesh, z_coord, None)
        nCells, nlev = state.T.data.shape
        nEdges = state.u.data.shape[0]
        assert du_dt_edge.shape == (nEdges, nlev)
        assert dT_dt.shape == (nCells, nlev)
        assert dS_dt.shape == (nCells, nlev)
        assert bool(jnp.all(jnp.isfinite(du_dt_edge)))
        assert bool(jnp.all(jnp.isfinite(dT_dt)))
        assert bool(jnp.all(jnp.isfinite(dS_dt)))

    def test_land_cells_zero_tracer_tendency(self, mesh, z_coord, state, kpp_cfg):
        """KPP tracer tendencies must be exactly zero on land cells."""
        fn = make_kpp_physics_mpas(kpp_cfg)
        _, dT_dt, dS_dt = fn(state, mesh, z_coord, None)
        land = state.land_mask.data < 0.5
        if bool(jnp.any(land)):
            assert jnp.allclose(dT_dt[land], 0.0)
            assert jnp.allclose(dS_dt[land], 0.0)

    def test_callable_factory(self, kpp_cfg):
        assert callable(make_kpp_physics_mpas(kpp_cfg))


def _constant_rho_eos(T, S, p):
    """Stability-FLIPPING EOS: uniform density ⇒ N²=0 everywhere, so the KPP
    Richardson / boundary-layer decision is unambiguously different from the
    stratified Wright profile (warm surface / cold deep ⇒ N²>0).  If a KPP
    builder silently fell back to Wright, the K_v / tendency would be identical
    — so a measurable difference proves ``eos_fn`` actually reaches the density.
    """
    return jnp.full_like(T, 1026.0)


class TestKPPEosThreading:
    """#M1 codex-HIGH regression: the MPAS KPP/convection density must use the
    model-selected EOS (``eos_fn``), not silently fall back to Wright."""

    def test_profiles_eos_fn_reaches_density(self, mesh, z_coord, state, kpp_cfg):
        pf_default = make_kpp_profiles_mpas(kpp_cfg)            # None ⇒ Wright
        pf_const = make_kpp_profiles_mpas(kpp_cfg, eos_fn=_constant_rho_eos)
        _, Kv_w = pf_default(state, mesh, z_coord, None)
        _, Kv_c = pf_const(state, mesh, z_coord, None)
        assert not bool(jnp.allclose(Kv_w, Kv_c)), (
            "make_kpp_profiles_mpas ignored eos_fn (Wright fallback): a "
            "stability-flipping EOS left K_v unchanged.")

    def test_profiles_default_is_wright(self, mesh, z_coord, state, kpp_cfg):
        """eos_fn=None must be BIT-IDENTICAL to explicitly passing Wright."""
        from legoesm.ocean.eos import wright_eos
        pf_default = make_kpp_profiles_mpas(kpp_cfg)
        pf_wright = make_kpp_profiles_mpas(kpp_cfg, eos_fn=wright_eos)
        Av_d, Kv_d = pf_default(state, mesh, z_coord, None)
        Av_w, Kv_w = pf_wright(state, mesh, z_coord, None)
        assert jnp.allclose(Kv_d, Kv_w) and jnp.allclose(Av_d, Av_w)

    def test_physics_eos_fn_reaches_density(self, mesh, z_coord, state, kpp_cfg):
        pf_default = make_kpp_physics_mpas(kpp_cfg)
        pf_const = make_kpp_physics_mpas(kpp_cfg, eos_fn=_constant_rho_eos)
        _, dT_w, _ = pf_default(state, mesh, z_coord, None)
        _, dT_c, _ = pf_const(state, mesh, z_coord, None)
        assert not bool(jnp.allclose(dT_w, dT_c)), (
            "make_kpp_physics_mpas ignored eos_fn (Wright fallback).")


class TestKPPProfilesMPAS:
    def test_shapes_nonneg_finite(self, mesh, z_coord, state, kpp_cfg):
        pf = make_kpp_profiles_mpas(kpp_cfg)
        A_v, K_v = pf(state, mesh, z_coord, None)
        nCells, nlev = state.T.data.shape
        assert A_v.shape == (nCells, nlev - 1)   # half levels
        assert K_v.shape == (nCells, nlev - 1)
        # Diffusivity / viscosity inputs to the implicit solve are >= 0, finite.
        assert bool(jnp.all(K_v >= 0.0))
        assert bool(jnp.all(A_v >= 0.0))
        assert bool(jnp.all(jnp.isfinite(K_v)))
        assert bool(jnp.all(jnp.isfinite(A_v)))

    def test_land_cells_zeroed(self, mesh, z_coord, state, kpp_cfg):
        pf = make_kpp_profiles_mpas(kpp_cfg)
        A_v, K_v = pf(state, mesh, z_coord, None)
        land = state.land_mask.data < 0.5
        if bool(jnp.any(land)):
            assert jnp.allclose(A_v[land], 0.0)
            assert jnp.allclose(K_v[land], 0.0)


class TestEdgePartialDiffusion:
    def test_uniform_field_zero_tendency(self):
        nE, nlev = 4, 5
        h_e = jnp.full((nE, nlev), 100.0)
        K = jnp.full((nE, nlev - 1), 1e-3)
        field = jnp.full((nE, nlev), 0.5)
        tend = _vertical_diffusion_edge_partial(field, h_e, K)
        assert tend.shape == (nE, nlev)
        assert jnp.allclose(tend, 0.0)

    def test_sheared_field_conserves_column_momentum(self):
        nE, nlev = 4, 5
        h_e = jnp.full((nE, nlev), 100.0)
        K = jnp.full((nE, nlev - 1), 1e-3)
        sheared = jnp.broadcast_to(jnp.linspace(1.0, -1.0, nlev), (nE, nlev))
        tend = _vertical_diffusion_edge_partial(sheared, h_e, K)
        # Zero-flux BC => volume-integrated tendency is ~0.
        col = jnp.sum(tend * h_e, axis=-1)
        assert jnp.allclose(col, 0.0, atol=1e-12)
        assert float(jnp.max(jnp.abs(tend))) > 0.0
        # Diffusion damps the shear: top (largest u) gets a negative tendency.
        assert float(tend[0, 0]) < 0.0

    def test_single_level_returns_zeros(self):
        h_e = jnp.full((3, 1), 100.0)
        K = jnp.zeros((3, 0))
        field = jnp.full((3, 1), 0.7)
        tend = _vertical_diffusion_edge_partial(field, h_e, K)
        assert tend.shape == (3, 1)
        assert jnp.allclose(tend, 0.0)

    def test_finite_with_zero_thickness_subseafloor(self):
        """Sub-seafloor levels (h_e = 0) must stay finite (1 m floor)."""
        nE, nlev = 3, 4
        h_e = jnp.array([[100.0, 100.0, 0.0, 0.0]] * nE)
        K = jnp.zeros((nE, nlev - 1)).at[:, 0].set(1e-3)
        field = jnp.broadcast_to(jnp.linspace(1.0, 0.0, nlev), (nE, nlev))
        tend = _vertical_diffusion_edge_partial(field, h_e, K)
        assert bool(jnp.all(jnp.isfinite(tend)))


class TestMPASSurfaceBuoyancyFlux:
    def test_returns_none_when_no_forcing(self):
        T3 = jnp.full((6, 4, 1), 20.0)
        S3 = jnp.full((6, 4, 1), 35.0)
        B_f, Q_sfc_T, Q_sfc_S = _mpas_surface_buoyancy_flux(None, None, None, T3, S3)
        assert B_f is None
        assert Q_sfc_T is None
        assert Q_sfc_S is None

    def test_heat_flux_drives_buoyancy(self):
        """Surface cooling (q_net < 0) is destabilising -> B_f > 0."""
        T3 = jnp.full((6, 4, 1), 20.0)
        S3 = jnp.full((6, 4, 1), 35.0)
        q_net = jnp.full((6, 4), -100.0)   # ocean losing heat
        B_f, Q_sfc_T, Q_sfc_S = _mpas_surface_buoyancy_flux(q_net, None, None, T3, S3)
        assert B_f is not None and B_f.shape == (6, 4)
        assert bool(jnp.all(jnp.isfinite(B_f)))
        assert float(jnp.min(B_f)) > 0.0       # cooling destabilises
        assert Q_sfc_T is not None
        # MPAS convention: real salt feeds buoyancy only, so a heat-only
        # forcing leaves the non-local salinity flux absent.
        assert Q_sfc_S is None

    def test_real_salt_buoyancy_only_no_nonlocal_salt(self):
        """Real salt_flux feeds B_f but NOT Q_sfc_S (real_salt_in_qs=False)."""
        T3 = jnp.full((6, 4, 1), 20.0)
        S3 = jnp.full((6, 4, 1), 35.0)
        salt = jnp.full((6, 4), 1e-4)
        B_f, _, Q_sfc_S = _mpas_surface_buoyancy_flux(None, None, salt, T3, S3)
        assert B_f is not None and bool(jnp.all(jnp.isfinite(B_f)))
        # Real salt is buoyancy-only: the non-local salinity flux is an
        # explicit ZERO (not the freshwater term) so it injects no second
        # real-salt contribution (no-double-count contract).
        assert Q_sfc_S is not None
        assert jnp.allclose(Q_sfc_S, 0.0)


def _mpas_ice_forcing(nCells, ice=1.0, q_net=-200.0, tau=0.1):
    """Surface-cooling + wind forcing WITH sea-ice cover on the MPAS cell grid.

    Cooling (q_net<0) is destabilising -> a convective KPP boundary layer;
    wind (tau) drives the shear velocity scale.  ``ice_concentration`` (nCells,)
    is what the under-ice attenuation reads.
    """
    return OceanSurfaceForcing(
        q_net=jnp.full((nCells,), q_net),
        tau_x=jnp.full((nCells,), tau),
        tau_y=jnp.zeros((nCells,)),
        ice_concentration=jnp.full((nCells,), ice),
    )


class TestKPPUnderIceMPAS:
    """MPAS KPP under-ice attenuation (``KPPConfig.eice``; NEMO ``nn_eice``).

    The lat-lon eice lever now reaches the MPAS bridge: ``_run_mpas_kpp`` reads
    ``surface_forcing.ice_concentration`` under the shared static eice gate and
    threads it to ``kpp_vertical_mixing`` -> ``_kpp_ice_attenuation`` (the same
    grid-agnostic kernel the C-grid paths use).  These pin the ADAPTER contract
    (the attenuation kernel itself is covered by ``test_kpp_under_ice.py``).
    """

    def test_eice3_builds_and_runs_no_raise(self, mesh, z_coord, state):
        """eice=3 must no longer raise NotImplementedError on the MPAS bridge —
        it is wired.  Build + run returns finite, non-negative K_v."""
        cfg = VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(eice=3))
        pf = make_kpp_profiles_mpas(cfg)
        nCells = state.T.data.shape[0]
        A_v, K_v = pf(state, mesh, z_coord, _mpas_ice_forcing(nCells))
        assert bool(jnp.all(jnp.isfinite(K_v))) and bool(jnp.all(K_v >= 0.0))
        assert bool(jnp.all(jnp.isfinite(A_v))) and bool(jnp.all(A_v >= 0.0))

    def test_full_ice_attenuates_diffusivity(self, mesh, z_coord, state):
        """Under full ice (fi=1), eice=3 gives eff=min(4·1,1)=1 -> attenuation
        max(0,1-1)=0 -> the KPP velocity scales floor -> the boundary-layer
        diffusivity collapses toward background vs the un-attenuated eice=0 run
        on the SAME forcing.  Proves ice_concentration reaches the scales."""
        nCells = state.T.data.shape[0]
        forcing = _mpas_ice_forcing(nCells, ice=1.0)
        pf0 = make_kpp_profiles_mpas(
            VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(eice=0)))
        pf3 = make_kpp_profiles_mpas(
            VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(eice=3)))
        _, Kv0 = pf0(state, mesh, z_coord, forcing)
        _, Kv3 = pf3(state, mesh, z_coord, forcing)
        ocean = state.land_mask.data >= 0.5     # (nCells,)
        sum0 = float(jnp.sum(Kv0[ocean]))
        sum3 = float(jnp.sum(Kv3[ocean]))
        assert sum0 > 0.0, ("non-vacuous guard: eice=0 must produce a KPP "
                            "boundary layer to attenuate")
        assert sum3 < sum0, (
            "eice=3 under full ice did not reduce KPP diffusivity — ice "
            "concentration is not reaching the MPAS velocity-scale attenuation.")

    def test_eice0_ignores_ice_bit_identical(self, mesh, z_coord, state):
        """eice=0 must NOT read ice_concentration: forcing WITH ice gives the
        bit-identical K_v to forcing without it (no silent under-ice coupling on
        the default path — the eice=0 -> ice_frac=None gate)."""
        nCells = state.T.data.shape[0]
        pf0 = make_kpp_profiles_mpas(
            VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(eice=0)))
        f_ice = _mpas_ice_forcing(nCells, ice=1.0)
        f_noice = f_ice._replace(ice_concentration=None)
        _, Kv_ice = pf0(state, mesh, z_coord, f_ice)
        _, Kv_noice = pf0(state, mesh, z_coord, f_noice)
        assert jnp.allclose(Kv_ice, Kv_noice), (
            "eice=0 path read ice_concentration (should be inert).")

    def test_unknown_eice_raises(self, mesh, z_coord, state):
        """An out-of-set eice (dispatch footgun) raises rather than silently
        running an undefined attenuation — validated at the call, like the
        C-grid paths."""
        cfg = VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(eice=2))
        pf = make_kpp_profiles_mpas(cfg)
        nCells = state.T.data.shape[0]
        with pytest.raises(ValueError, match="eice"):
            pf(state, mesh, z_coord, _mpas_ice_forcing(nCells))
