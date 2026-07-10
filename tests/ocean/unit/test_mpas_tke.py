"""Direct unit tests for TKE vertical mixing on the MPAS Voronoi mesh.

``ocean/physics/vertical_mixing/mpas_integration.py::make_tke_profiles_mpas``
wires the grid-agnostic Gaspar (1990) / Burchard (2002) TKE closure
(``tke.py::tke_vertical_mixing``) onto the MPAS TRiSK C-grid: it reconstructs
cell-centred (u, v) for the shear (the SAME reconstruction KPP uses), builds
N^2 from cell T/S/rho, runs the DIAGNOSTIC quasi-steady closure per cell, and
applies the KPP MPAS CFL cap on the returned (A_v = K_M, K_v = K_H) profiles.

These tests pin the MPAS-TKE *adapter* contract on a small
``subdivision_level=1`` Voronoi mesh (42 cells / 120 edges):

  * dispatch — ``make_mpas_ocean_physics`` + ``make_tke_profiles_mpas`` BUILD
    for ``scheme="tke"`` under ``implicit_vertical_mixing=True`` (a); and the
    profiles function returns finite, non-negative, land-masked K profiles;
  * dispatch-hardening — ``catke`` / ``richardson`` on MPAS STILL raise, ``tke``
    without implicit vmix raises, and the un-plumbed prognostic / adiabatic
    options raise (b);
  * grid-agnostic-core equivalence — for a single wet column, the MPAS TKE
    profile equals a DIRECT ``tke_vertical_mixing`` call on that column's
    reconstructed inputs (no cross-cell leakage, no lat-lon assumption) (c).

The TKE closure numerics themselves are validated by the TKE suite; here we pin
the MPAS *bridge* contract (dispatch, masking, CFL cap, finiteness, sign,
grid-agnostic equivalence).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.eos import rho_0
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
    _TKE_DIAGNOSTIC_DT_S,
    _TKE_DIAGNOSTIC_N_ITER,
    _reconstruct_mpas_cell_fields,
    make_tke_profiles_mpas,
)
from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
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
def tke_cfg():
    return VerticalMixingConfig(scheme="tke", tke=TKEConfig())


def _wind_forcing(state, tau_x_pa=0.1):
    """Uniform eastward wind stress (Pa) over all cells — spins up TKE.

    ``tau_x_pa`` defaults to a light 0.1 Pa (enough for the finiteness / CFL /
    equivalence checks, which are magnitude-agnostic).  The spin-up check passes
    a stronger stress — see ``test_wind_spins_up_tke_mixing`` for why the coarse
    fixture needs it.
    """
    nCells = state.T.data.shape[0]
    tau_x = jnp.full((nCells,), tau_x_pa, dtype=state.T.data.dtype)
    tau_y = jnp.zeros((nCells,), dtype=state.T.data.dtype)
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


# ---------------------------------------------------------------------------
# (a) Dispatch: TKE builds + produces valid profiles under implicit vmix.
# ---------------------------------------------------------------------------
class TestTKEDispatchBuilds:
    def test_make_mpas_ocean_physics_accepts_tke(self, tke_cfg):
        """scheme='tke' + implicit_vertical_mixing=True builds without raising."""
        config = OceanPhysicsConfig(
            vertical_mixing=tke_cfg,
            surface_forcing=SurfaceForcingConfig(scheme="none"),
        )
        fn = make_mpas_ocean_physics(config, implicit_vertical_mixing=True)
        assert callable(fn)

    def test_make_tke_profiles_mpas_builds(self, tke_cfg):
        profiles_fn = make_tke_profiles_mpas(tke_cfg)
        assert callable(profiles_fn)

    def test_profiles_shapes_finite_nonneg_masked(
            self, mesh, z_coord, state, tke_cfg):
        """(A_v, K_v) at half-levels: right shape, finite, >= 0, land zeroed."""
        profiles_fn = make_tke_profiles_mpas(tke_cfg)
        A_v, K_v = profiles_fn(state, mesh, z_coord, _wind_forcing(state))
        nCells, nlev = state.T.data.shape
        assert A_v.shape == (nCells, nlev - 1)
        assert K_v.shape == (nCells, nlev - 1)
        assert bool(jnp.all(jnp.isfinite(A_v)))
        assert bool(jnp.all(jnp.isfinite(K_v)))
        # Prognostic TKE >= tke_min floor => K_M, K_H >= 0.
        assert float(jnp.min(A_v)) >= 0.0
        assert float(jnp.min(K_v)) >= 0.0
        # Land cells must be exactly zero (whole column).
        mask = state.land_mask.data
        land = mask < 0.5
        if bool(jnp.any(land)):
            assert float(jnp.max(jnp.abs(A_v[land]))) == 0.0
            assert float(jnp.max(jnp.abs(K_v[land]))) == 0.0

    def test_profiles_respect_cfl_cap(self, mesh, z_coord, state, tke_cfg):
        """A_v, K_v never exceed the KPP MPAS CFL cap 0.25 min(dz)^2/cfl_dt."""
        profiles_fn = make_tke_profiles_mpas(tke_cfg)
        A_v, K_v = profiles_fn(state, mesh, z_coord, _wind_forcing(state))
        dz = z_coord.dz_ref
        Av_max = 0.25 * jnp.minimum(dz[:-1], dz[1:]) ** 2 / tke_cfg.tke.cfl_cap_dt_s
        # allow a tiny fp slack on the <= comparison
        assert bool(jnp.all(A_v <= Av_max[None, :] + 1e-9))
        assert bool(jnp.all(K_v <= Av_max[None, :] + 1e-9))

    def test_wind_spins_up_tke_mixing(self, mesh, z_coord, state, tke_cfg):
        """Surface wind stress drives near-surface A_v above the calm value.

        The surface TKE flux (|tau|/rho_0)^{3/2} is injected at the topmost
        interior interface, so the wind's effect is strongest there.  More
        surface TKE -> larger K_M (monotone), so the forced top-interface
        viscosity must strictly exceed the unforced one.

        A STRONG stress (10 Pa) is used deliberately: this coarse 6-level /
        4000 m fixture has a ~63 m top layer, so the diagnostic surface TKE
        from a light 0.1 Pa wind lands at/below the ``tke_surface_min`` floor
        (1e-4 m^2/s^2) and is bit-identical to calm — the wind signal only
        clears the floor above ~a few Pa here (verified: 0.1/1 Pa sit on the
        floor; >=~10 Pa lift K_M[:, 0] to ~8x calm).  This probes the surface-
        flux WIRING (correctly threaded through the shared bridge), not a
        realistic wind magnitude.
        """
        profiles_fn = make_tke_profiles_mpas(tke_cfg)
        A_v_forced, _ = profiles_fn(
            state, mesh, z_coord, _wind_forcing(state, tau_x_pa=10.0))
        A_v_calm, _ = profiles_fn(state, mesh, z_coord, None)
        assert float(jnp.max(A_v_forced[:, 0])) > float(jnp.max(A_v_calm[:, 0]))


# ---------------------------------------------------------------------------
# (b) Dispatch-hardening: unsupported schemes / options STILL raise on MPAS.
# ---------------------------------------------------------------------------
class TestTKEDispatchHardening:
    def test_catke_still_raises_on_mpas(self):
        from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig
        config = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="catke", catke=CATKEConfig()),
        )
        with pytest.raises(ValueError, match="catke"):
            make_mpas_ocean_physics(config, implicit_vertical_mixing=True)

    def test_richardson_still_raises_on_mpas(self):
        config = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="richardson"),
        )
        with pytest.raises(NotImplementedError, match="richardson"):
            make_mpas_ocean_physics(config, implicit_vertical_mixing=True)

    def test_tke_without_implicit_raises(self, tke_cfg):
        """TKE is implicit-only on MPAS; explicit vmix must fail-fast."""
        config = OceanPhysicsConfig(vertical_mixing=tke_cfg)
        with pytest.raises(ValueError, match="implicit_vertical_mixing"):
            make_mpas_ocean_physics(config, implicit_vertical_mixing=False)

    def test_prognostic_tke_rejected_on_mpas(self):
        """The prognostic carry is not wired on MPAS -> reject loudly."""
        cfg = VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(prognostic=True))
        with pytest.raises(NotImplementedError, match="prognostic"):
            make_tke_profiles_mpas(cfg)

    def test_adiabatic_n2_rejected_on_mpas(self):
        cfg = VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(n2_mode="adiabatic"))
        with pytest.raises(NotImplementedError, match="n2_mode"):
            make_tke_profiles_mpas(cfg)


# ---------------------------------------------------------------------------
# (c) Grid-agnostic-core equivalence: the MPAS bridge is a faithful wrapper
#     around the SAME tke_vertical_mixing closure the lat-lon path calls — a
#     single wet column matches a direct closure call (no cross-cell leakage,
#     no lat-lon assumption leaked into the shared core).
# ---------------------------------------------------------------------------
class TestTKEGridAgnosticEquivalence:
    def test_single_column_matches_direct_closure(
            self, mesh, z_coord, state, tke_cfg):
        sf = _wind_forcing(state)

        # Bridge output over the whole mesh.
        profiles_fn = make_tke_profiles_mpas(tke_cfg)
        A_v, K_v = profiles_fn(state, mesh, z_coord, sf)

        # Reconstruct the SAME cell inputs the bridge builds internally, then
        # run the closure on ONE wet column and compare to the bridge at that
        # cell.  On z-star every wet cell is full-depth (no sub-seafloor mask),
        # so the only bridge post-step is the CFL cap.
        u_e, v_n, T_w, S_w, rho, J, mask = _reconstruct_mpas_cell_fields(
            state, mesh, z_coord)
        c = int(jnp.argmax(mask.astype(jnp.int32)))  # first wet cell
        assert float(mask[c]) > 0.5

        dz_half_c = z_coord.dz_half_ref[None, :] * J[c:c + 1, None]
        out_c = tke_vertical_mixing(
            u_e[c:c + 1], v_n[c:c + 1], T_w[c:c + 1], S_w[c:c + 1],
            rho[c:c + 1], dz_half_c,
            tke_old=None,
            tau_x_surface=sf.tau_x[c:c + 1], tau_y_surface=sf.tau_y[c:c + 1],
            dt=_TKE_DIAGNOSTIC_DT_S, cfg=tke_cfg.tke,
            rho_0=rho_0, g=constants.g,
            n_iterations=_TKE_DIAGNOSTIC_N_ITER,
            z_interface=z_coord.z_half_ref[1:-1],
        )

        # Replay the bridge's CFL cap for this column (no mask: c is wet+full).
        dz = z_coord.dz_ref
        Av_max = 0.25 * jnp.minimum(dz[:-1], dz[1:]) ** 2 / tke_cfg.tke.cfl_cap_dt_s
        expected_A = jnp.minimum(out_c.K_M[0], Av_max)
        expected_K = jnp.minimum(out_c.K_H[0], Av_max)

        assert jnp.allclose(A_v[c], expected_A, rtol=1e-9, atol=1e-12)
        assert jnp.allclose(K_v[c], expected_K, rtol=1e-9, atol=1e-12)

    def test_tke_positive_after_diagnostic_solve(
            self, mesh, z_coord, state, tke_cfg):
        """The closure floors prognostic TKE => the derived K_M, K_H stay >= 0.

        (Guards the sign contract on the actual reconstructed-mesh inputs, not
        just synthetic columns.)
        """
        u_e, v_n, T_w, S_w, rho, J, mask = _reconstruct_mpas_cell_fields(
            state, mesh, z_coord)
        dz_half = z_coord.dz_half_ref[None, :] * J[:, None]
        out = tke_vertical_mixing(
            u_e, v_n, T_w, S_w, rho, dz_half,
            tke_old=None, tau_x_surface=None, tau_y_surface=None,
            dt=_TKE_DIAGNOSTIC_DT_S, cfg=tke_cfg.tke,
            rho_0=rho_0, g=constants.g,
            n_iterations=_TKE_DIAGNOSTIC_N_ITER,
            z_interface=z_coord.z_half_ref[1:-1],
        )
        assert bool(jnp.all(out.tke_new >= 0.0))
        assert bool(jnp.all(out.K_M >= 0.0))
        assert bool(jnp.all(out.K_H >= 0.0))
