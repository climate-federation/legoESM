"""Direct unit tests for TKE vertical mixing on the MPAS Voronoi mesh.

``ocean/physics/vertical_mixing/mpas_integration.py::make_tke_profiles_mpas``
wires the grid-agnostic Gaspar (1990) / Burchard (2002) TKE closure
(``tke.py::tke_vertical_mixing``) onto the MPAS TRiSK C-grid: it reconstructs
cell-centred (u, v) for the shear (the SAME reconstruction KPP uses), builds
N^2 from cell T/S/rho, runs the DIAGNOSTIC quasi-steady closure per cell, and
returns the raw closure (A_v = K_M, K_v = K_H) profiles — NO CFL post-cap
(implicit-only path; the closure's kappaM_max is the ceiling).

These tests pin the MPAS-TKE *adapter* contract on a small
``subdivision_level=1`` Voronoi mesh (42 cells / 120 edges):

  * dispatch — ``make_mpas_ocean_physics`` + ``make_tke_profiles_mpas`` BUILD
    for ``scheme="tke"`` under ``implicit_vertical_mixing=True`` (a); and the
    profiles function returns finite, non-negative, land-masked K profiles;
  * dispatch-hardening — ``catke`` / ``richardson`` on MPAS STILL raise, ``tke``
    without implicit vmix raises, and the un-plumbed adiabatic-N² / Veros-slot
    options raise (b); the PROGNOSTIC carry builds and is exercised in
    ``TestPrognosticTKECarryOnMPAS``;
  * grid-agnostic-core equivalence — for a single wet column, the MPAS TKE
    profile equals a DIRECT ``tke_vertical_mixing`` call on that column's
    reconstructed inputs (no cross-cell leakage, no lat-lon assumption) (c).

The TKE closure numerics themselves are validated by the TKE suite; here we pin
the MPAS *bridge* contract (dispatch, masking, no-post-cap, finiteness, sign,
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

    def test_profiles_have_no_cfl_post_cap(self, mesh, z_coord, state,
                                           tke_cfg):
        """The TKE bridge applies NO explicit-diffusion CFL post-cap (codex):
        it is implicit-only (unconditionally stable backward-Euler), the
        C-grid zdftke path and NEMO cap nothing — a 0.25·dz²/300s cap would
        bind in convective columns (~0.08 m²/s at 10-m cells vs closure K of
        O(1-10)) and break closure equivalence across grids.  The closure's
        own kappaM_max remains the only ceiling."""
        profiles_fn = make_tke_profiles_mpas(tke_cfg)
        A_v, K_v = profiles_fn(state, mesh, z_coord, _wind_forcing(state))
        assert bool(jnp.all(jnp.isfinite(A_v))) and bool(jnp.all(A_v >= 0.0))
        assert bool(jnp.all(A_v <= tke_cfg.tke.kappaM_max + 1e-9))
        # source-level tripwire: the cap code must not silently return
        import inspect
        from legoesm.ocean.physics.vertical_mixing import mpas_integration
        src = inspect.getsource(mpas_integration.make_tke_profiles_mpas)
        assert "_Av_max" not in src, "CFL post-cap re-appeared on the TKE bridge"

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

    def test_prognostic_tke_builds_on_mpas(self):
        """The prognostic carry IS wired on MPAS now — the bridge and the
        physics factory must both ACCEPT prognostic=True (was a reject until
        the MPASOceanState.tke carry landed)."""
        cfg = VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(prognostic=True))
        assert callable(make_tke_profiles_mpas(cfg))
        config = OceanPhysicsConfig(
            vertical_mixing=cfg,
            surface_forcing=SurfaceForcingConfig(scheme="none"),
        )
        assert callable(
            make_mpas_ocean_physics(config, implicit_vertical_mixing=True))

    def test_prognostic_without_dt_raises(self, mesh, z_coord, state):
        """Mode A needs the model dt — calling profiles_fn without dt_tke
        must fail loud (never silently fall back to the diagnostic dt)."""
        pf = make_tke_profiles_mpas(VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(prognostic=True)))
        with pytest.raises(ValueError, match="dt_tke"):
            pf(state, mesh, z_coord, _wind_forcing(state))

    def test_prognostic_bottom_tke_bc_rejected_on_mpas(self):
        """The Veros T15 bottom Dirichlet row is not threaded on this bridge
        -> reject rather than silently ignore."""
        cfg = VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(prognostic=True, bottom_tke_bc=True))
        with pytest.raises(NotImplementedError, match="bottom_tke_bc"):
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
        # so the bridge applies NO post-step (raw closure output).
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

        # DIRECT closure equivalence — the bridge applies NO post-cap (codex:
        # replaying the removed CFL cap here would silently re-legitimize it;
        # the bridge output must equal the raw kernel output exactly).
        assert jnp.allclose(A_v[c], out_c.K_M[0], rtol=1e-9, atol=1e-12)
        assert jnp.allclose(K_v[c], out_c.K_H[0], rtol=1e-9, atol=1e-12)

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


class TestNemoSurfaceTermsOnMPAS:
    """The ORCA1 zdftke card's surface terms (lc / etau nn_htau=1 / eice) now
    RUN on the MPAS bridge: profiles_fn threads lat_deg=degrees(mesh.latCell)
    and (under the eice gate) surface_forcing.ice_concentration into the
    grid-agnostic kernel — the same inputs the C-grid k_profiles path passes.
    The closure numerics are covered by the TKE suite; these pin the ADAPTER
    contract (threading, gating, fail-fast)."""

    def _card(self, **over):
        from scripts.run.run_omip_core2 import orca1_zdftke_config
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        # prognostic=False: these adapter tests pin the DIAGNOSTIC Mode-B
        # path (stateless per-call closure — no carry threading needed); the
        # prognostic Mode-A carry has its own suite
        # (TestPrognosticTKECarryOnMPAS).
        return VerticalMixingConfig(
            scheme="tke",
            tke=orca1_zdftke_config(prognostic=False)._replace(**over))

    def _ice_wind_forcing(self, state, ice=1.0, tau_x_pa=0.15):
        n = state.T.data.shape[0]
        return OceanSurfaceForcing(
            tau_x=jnp.full((n,), tau_x_pa), tau_y=jnp.zeros((n,)),
            ice_concentration=jnp.full((n,), ice))

    def test_orca1_card_runs_on_mpas(self, mesh, z_coord, state):
        """lc=True + etau latitude profile + eice=3 must BUILD AND RUN (no
        NotImplementedError, no lat_deg-missing ValueError — the kernel raises
        'lat_deg' if the latitude profile is requested without it, so a clean
        run PROVES lat_deg reaches the kernel)."""
        pf = make_tke_profiles_mpas(self._card())
        A_v, K_v = pf(state, mesh, z_coord, self._ice_wind_forcing(state))
        assert bool(jnp.all(jnp.isfinite(A_v))) and bool(jnp.all(A_v >= 0.0))
        assert bool(jnp.all(jnp.isfinite(K_v))) and bool(jnp.all(K_v >= 0.0))

    def test_eice_full_ice_attenuates_vs_eice0(self, mesh, z_coord, state):
        """Full ice + eice=3 must reduce the wind-driven mixing vs eice=0 on
        the SAME forcing (proves ice_concentration reaches the kernel)."""
        f = self._ice_wind_forcing(state, ice=1.0)
        _, K0 = make_tke_profiles_mpas(self._card(eice=0))(
            state, mesh, z_coord, f)
        _, K3 = make_tke_profiles_mpas(self._card(eice=3))(
            state, mesh, z_coord, f)
        ocean = state.land_mask.data >= 0.5
        assert float(jnp.sum(K0[ocean])) > 0.0
        assert float(jnp.sum(K3[ocean])) < float(jnp.sum(K0[ocean]))

    def test_eice_without_ice_fails_fast(self, mesh, z_coord, state):
        """eice!=0 with NO ice field must raise (the KPP-bridge contract) —
        never silently run un-attenuated."""
        pf = make_tke_profiles_mpas(self._card(eice=3))
        f_noice = self._ice_wind_forcing(state)._replace(
            ice_concentration=None)
        with pytest.raises(ValueError, match="ice_concentration"):
            pf(state, mesh, z_coord, f_noice)
        with pytest.raises(ValueError, match="ice_concentration"):
            pf(state, mesh, z_coord, None)

    def test_unknown_eice_raises(self, mesh, z_coord, state):
        pf = make_tke_profiles_mpas(self._card(eice=2))
        with pytest.raises(ValueError, match="eice"):
            pf(state, mesh, z_coord, self._ice_wind_forcing(state))

    def test_eice0_no_ice_is_fine_and_default_card_unchanged(
            self, mesh, z_coord, state, tke_cfg):
        """eice=0 ignores ice entirely (no read, no raise without ice), and
        the plain default TKEConfig path stays valid (regression: the new
        gating must not disturb the pre-existing diagnostic bridge)."""
        pf = make_tke_profiles_mpas(self._card(eice=0, lc=False,
                                               etau_mode="none"))
        A_v, K_v = pf(state, mesh, z_coord, _wind_forcing(state))
        assert bool(jnp.all(jnp.isfinite(K_v)))
        pf_default = make_tke_profiles_mpas(tke_cfg)
        A_v2, K_v2 = pf_default(state, mesh, z_coord, _wind_forcing(state))
        assert bool(jnp.all(jnp.isfinite(K_v2)))

    def test_kernel_receives_degrees_and_e3t_inputs(
            self, mesh, z_coord, state, monkeypatch):
        """Capture-level pin of the bridge->kernel input contract (codex MED
        2026-07-27): a clean run only proves lat_deg/dz_ref are non-None —
        RADIANS would also run silently (with an h_tau profile wrong by
        180/pi), and a wrong dz_ref would silently mis-size the nn_mxl=3
        |dl/dz|<=e3t sweeps. Spy on the kernel and assert the exact arrays."""
        from legoesm.ocean.physics.vertical_mixing import (
            mpas_integration as mi,
        )
        captured = {}
        real_kernel = mi.tke_vertical_mixing

        def spy(*a, **kw):
            captured.update(kw)
            return real_kernel(*a, **kw)

        monkeypatch.setattr(mi, "tke_vertical_mixing", spy)
        pf = mi.make_tke_profiles_mpas(self._card())
        pf(state, mesh, z_coord, self._ice_wind_forcing(state, ice=0.0))
        # nn_htau=1 needs DEGREES; mesh.latCell is radians (Coriolis input).
        assert bool(jnp.allclose(captured["lat_deg"],
                                 jnp.degrees(mesh.latCell)))
        assert float(jnp.max(jnp.abs(captured["lat_deg"]))) > 4.0  # not rad
        # nn_mxl=3 e3t inputs: reference thicknesses + the land-safe J
        # (land cells J=1.0 so dz_ref*J never divides by zero).
        assert bool(jnp.array_equal(captured["dz_ref"], z_coord.dz_ref))
        jac = captured["jacobian"]
        assert jac.shape == (state.T.data.shape[0],)
        assert bool(jnp.all(jnp.isfinite(jac)))
        land = state.land_mask.data < 0.5
        if bool(jnp.any(land)):
            assert bool(jnp.all(jac[land] == 1.0))

    def test_eice3_quarter_ice_maps_to_full_attenuation(
            self, mesh, z_coord, state):
        """NEMO nn_eice=3 maps fi -> min(4*fi, 1): QUARTER ice must attenuate
        exactly like mode-1 FULL ice (effective fraction 1.0), and differ
        from mode-1 quarter ice (raw 0.25). Full-ice-only tests cannot see a
        broken mapping — fi=1 is a fixed point of min(4*fi,1) (codex MED
        2026-07-27)."""
        f_q = self._ice_wind_forcing(state, ice=0.25)
        f_full = self._ice_wind_forcing(state, ice=1.0)
        _, K3q = make_tke_profiles_mpas(self._card(eice=3))(
            state, mesh, z_coord, f_q)
        _, K1f = make_tke_profiles_mpas(self._card(eice=1))(
            state, mesh, z_coord, f_full)
        _, K1q = make_tke_profiles_mpas(self._card(eice=1))(
            state, mesh, z_coord, f_q)
        assert bool(jnp.allclose(K3q, K1f, rtol=1e-12, atol=0.0))
        assert not bool(jnp.allclose(K3q, K1q))

    def test_partial_cell_zeroes_subseafloor_interfaces(self, mesh, z_coord):
        """Partial-cell geometry: profiles at interfaces below each column's
        deepest active level must be EXACTLY zero. Every other fixture here
        is full-depth z*, so the bottom_level masking branch was untested
        (codex MED 2026-07-27)."""
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean as _rest
        n = mesh.latCell.shape[0]
        # Mid-column ridge over half the cells; keep a few full columns.
        H = jnp.where(jnp.arange(n) % 2 == 0, 4000.0, 1500.0)
        pc = create_partial_cell_coordinate(z_coord, H)
        st = _rest(mesh, pc, T_water_init_C=20.0, T_deep=2.0,
                   S_uniform=35.0, H_max=4000.0, land_lat_threshold=85.0,
                   bathymetry=H)
        pf = make_tke_profiles_mpas(self._card())
        A_v, K_v = pf(st, mesh, pc, self._ice_wind_forcing(st, ice=0.0))
        assert bool(jnp.all(jnp.isfinite(A_v)))
        assert bool(jnp.all(jnp.isfinite(K_v)))
        # Interface k sits below the column's deepest active level when
        # k >= bottom_level (interfaces are between full levels k and k+1).
        nlev_half = K_v.shape[1]
        k_idx = jnp.arange(nlev_half)[None, :]
        bot = jnp.asarray(pc.bottom_level)[:, None]
        below = k_idx >= bot
        ocean = st.land_mask.data >= 0.5
        assert bool(jnp.all(K_v[ocean][below[ocean]] == 0.0))
        assert bool(jnp.all(A_v[ocean][below[ocean]] == 0.0))
        # Non-vacuous: the ridge columns really do cut the column, and the
        # open interfaces above the ridge still mix.
        assert bool(jnp.any(below[ocean]))
        assert float(jnp.sum(K_v[ocean])) > 0.0


class TestPrognosticTKECarryOnMPAS:
    """Mode-A (NEMO prognostic en) on the Voronoi bridge: the 3-tuple
    contract, carry evolution, dead-cell masking, and the model-step
    seed/carry cycle (pytree-stable scan carry)."""

    def _card(self, **over):
        from scripts.run.run_omip_core2 import orca1_zdftke_config
        return VerticalMixingConfig(
            scheme="tke", tke=orca1_zdftke_config()._replace(**over))

    def _forcing(self, state, ice=0.0, tau_x_pa=0.15):
        n = state.T.data.shape[0]
        return OceanSurfaceForcing(
            tau_x=jnp.full((n,), tau_x_pa), tau_y=jnp.zeros((n,)),
            ice_concentration=jnp.full((n,), ice))

    def test_carry_evolves_and_feeds_back(self, mesh, z_coord, state):
        """One Mode-A call must ADVANCE the seed (wind injects TKE), and a
        second call from tke_new must differ from the first (the carry feeds
        back) — the discriminator against a silently-diagnostic path."""
        assert self._card().tke.prognostic is True  # the #1326 card default
        pf = make_tke_profiles_mpas(self._card())
        f = self._forcing(state)
        A1, K1, tke1 = pf(state, mesh, z_coord, f, dt_tke=150.0)
        assert tke1.shape == (state.T.data.shape[0],
                              state.T.data.shape[1] - 1)
        assert bool(jnp.all(jnp.isfinite(tke1))) and bool(jnp.all(tke1 >= 0))
        bg = self._card().tke.tke_background
        ocean = state.land_mask.data >= 0.5
        assert float(jnp.max(jnp.abs(tke1[ocean] - bg))) > 0.0
        from legoesm.core.field import Field
        st2 = state._replace(tke=Field(
            data=tke1, name="tke", dims=("nCells", "level"),
            units="m^2/s^2"))
        A2, K2, tke2 = pf(st2, mesh, z_coord, f, dt_tke=150.0)
        assert not bool(jnp.allclose(tke2, tke1))

    def test_carry_masked_on_land(self, mesh, z_coord, state):
        pf = make_tke_profiles_mpas(self._card())
        _, _, tke1 = pf(state, mesh, z_coord, self._forcing(state),
                        dt_tke=150.0)
        land = state.land_mask.data < 0.5
        if bool(jnp.any(land)):
            assert bool(jnp.all(tke1[land] == 0.0))

    def test_model_seed_and_step_carry(self, mesh, z_coord):
        """MPASOceanModel.seed_tke seeds background-on-wet once (idempotent),
        and step() advances the carry with an UNCHANGED pytree structure —
        the lax.scan stability contract."""
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig
        st = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=85.0)
        config = MPASOceanConfig(
            physics=OceanPhysicsConfig(
                vertical_mixing=self._card(),
                surface_forcing=SurfaceForcingConfig(scheme="none"),
            ),
            implicit_vertical_mixing=True,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        assert st.tke is None
        st = model.seed_tke(st)
        assert st.tke is not None
        bg = self._card().tke.tke_background
        wet = st.land_mask.data >= 0.5
        assert bool(jnp.all(st.tke.data[wet] == bg))
        assert bool(jnp.all(st.tke.data[~wet] == 0.0))
        st_again = model.seed_tke(st)
        assert st_again.tke is st.tke  # idempotent no-op when seeded
        st1 = model.step(st, 150.0, surface_forcing=self._forcing(st))
        assert st1.tke is not None
        assert jax.tree_util.tree_structure(st1) == (
            jax.tree_util.tree_structure(st))
        st2 = model.step(st1, 150.0, surface_forcing=self._forcing(st1))
        assert not bool(jnp.allclose(st2.tke.data, st1.tke.data))

    def test_step_without_seed_raises(self, mesh, z_coord):
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig
        st = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=85.0)
        config = MPASOceanConfig(
            physics=OceanPhysicsConfig(
                vertical_mixing=self._card(),
                surface_forcing=SurfaceForcingConfig(scheme="none"),
            ),
            implicit_vertical_mixing=True,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        with pytest.raises(ValueError, match="seed_tke"):
            model.step(st, 150.0, surface_forcing=self._forcing(st))

    def test_diagnostic_mode_unchanged_two_tuple(self, mesh, z_coord, state):
        """prognostic=False keeps the two-tuple diagnostic contract and needs
        no dt (regression: the Mode-A plumbing must not disturb Mode-B)."""
        pf = make_tke_profiles_mpas(self._card(prognostic=False))
        out = pf(state, mesh, z_coord, self._forcing(state))
        assert len(out) == 2


class TestPrognosticCarryHardening:
    """codex 2026-07-27 round-2: forced-land masking, partial-cell tke_new,
    lax.scan pytree/dtype stability, and the restore paths that must
    reconstruct the carry as a Field."""

    def _card(self):
        from scripts.run.run_omip_core2 import orca1_zdftke_config
        return VerticalMixingConfig(scheme="tke", tke=orca1_zdftke_config())

    def _forcing(self, n, dtype, ice=0.0):
        return OceanSurfaceForcing(
            tau_x=jnp.full((n,), 0.15, dtype=dtype),
            tau_y=jnp.zeros((n,), dtype=dtype),
            ice_concentration=jnp.full((n,), ice, dtype=dtype))

    def test_carry_land_masking_forced_land(self, mesh, z_coord):
        """land_lat_threshold=60 GUARANTEES land cells on the level-1 mesh
        (polar vertices sit at +-90) — the 85-degree fixture could be
        all-ocean, making the land assert vacuous (codex MED)."""
        st = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=60.0)
        land = st.land_mask.data < 0.5
        assert bool(jnp.any(land)), "fixture must contain land"
        pf = make_tke_profiles_mpas(self._card())
        n = st.T.data.shape[0]
        _, _, tke1 = pf(st, mesh, z_coord,
                        self._forcing(n, st.T.data.dtype), dt_tke=150.0)
        assert bool(jnp.all(tke1[land] == 0.0))
        assert float(jnp.sum(tke1[~land])) > 0.0

    def test_carry_partial_cell_subseafloor_zeroed(self, mesh, z_coord):
        """Prognostic tke_new at interfaces below bottom_level must be
        EXACTLY zero (the diagnostic partial-cell test does not cover the
        Mode-A return; codex MED)."""
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        n = mesh.latCell.shape[0]
        H = jnp.where(jnp.arange(n) % 2 == 0, 4000.0, 1500.0)
        pc = create_partial_cell_coordinate(z_coord, H)
        st = rest_state_mpas_ocean(
            mesh, pc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=85.0, bathymetry=H)
        pf = make_tke_profiles_mpas(self._card())
        _, _, tke1 = pf(st, mesh, pc,
                        self._forcing(n, st.T.data.dtype), dt_tke=150.0)
        k_idx = jnp.arange(tke1.shape[1])[None, :]
        below = k_idx >= jnp.asarray(pc.bottom_level)[:, None]
        ocean = st.land_mask.data >= 0.5
        assert bool(jnp.any(below[ocean]))
        assert bool(jnp.all(tke1[ocean][below[ocean]] == 0.0))

    def test_scan_carry_stable_treedef_dtype_shape(self, mesh, z_coord):
        """The production host loop is step-per-day, but the carry contract
        is lax.scan-grade: run the step INSIDE lax.scan for 3 steps under
        the PRODUCTION (uniform-f64) policy and assert the carry keeps an
        identical treedef and identical leaf dtypes/shapes.  A SPLIT
        f32-storage/f64-compute policy is a pre-existing model-wide scan
        incompatibility (cast_pytree's storage cast does not downcast, so
        u/T/S/eta ALL return f64 — nothing tke-specific); the tke pin under
        that policy is covered by test_split_policy_tke_dtype_pinned."""
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig
        st = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=85.0)
        config = MPASOceanConfig(
            physics=OceanPhysicsConfig(
                vertical_mixing=self._card(),
                surface_forcing=SurfaceForcingConfig(scheme="none"),
            ),
            implicit_vertical_mixing=True,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        st = model.seed_tke(st)
        sf = self._forcing(st.T.data.shape[0], st.T.data.dtype)

        def body(carry, _):
            return model._step_impl(carry, 150.0, surface_forcing=sf), None

        out, _ = jax.lax.scan(body, st, None, length=3)
        assert jax.tree_util.tree_structure(out) == (
            jax.tree_util.tree_structure(st))
        for a, b in zip(jax.tree_util.tree_leaves(st),
                        jax.tree_util.tree_leaves(out)):
            assert a.dtype == b.dtype and a.shape == b.shape
        assert not bool(jnp.allclose(out.tke.data, st.tke.data))

    def test_split_policy_tke_dtype_pinned(self, mesh, z_coord):
        """Under f32-storage/f64-compute, ONE step must return the tke carry
        in the SEED dtype (the pre-compute-cast pin — codex r3 RED: pinning
        to the post-cast dtype froze the carry at f64).  Single step only:
        the split policy is scan-incompatible model-wide (u/T/S/eta return
        f64 from the no-downcast storage cast — pre-existing, not carry-
        specific), so the pin is asserted directly on the step output."""
        import jax.numpy as _jnp
        from legoesm.core.precision import (
            PrecisionPolicy, get_policy, set_policy,
        )
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig
        _orig = get_policy()
        set_policy(PrecisionPolicy(
            storage=_jnp.float32, compute=_jnp.float64,
            accumulate=_jnp.float64, control=_jnp.float64))
        try:
            st = rest_state_mpas_ocean(
                mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
                S_uniform=35.0, H_max=4000.0, land_lat_threshold=85.0)
            config = MPASOceanConfig(
                physics=OceanPhysicsConfig(
                    vertical_mixing=self._card(),
                    surface_forcing=SurfaceForcingConfig(scheme="none"),
                ),
                implicit_vertical_mixing=True,
            )
            model = MPASOceanModel(mesh, z_coord, config)
            st = model.seed_tke(st)
            seed_dtype = st.tke.data.dtype
            sf = self._forcing(st.T.data.shape[0], st.T.data.dtype)
            out = model._step_impl(st, 150.0, surface_forcing=sf)
            assert out.tke.data.dtype == seed_dtype
        finally:
            set_policy(_orig)

    def test_restart_npz_reconstructs_none_carry_as_field(
            self, mesh, z_coord, state, tmp_path):
        """load_restart on a template with tke=None must reconstruct the
        saved carry as a Field, not drop it (codex MED — a dropped carry
        re-spins turbulence from background on restart)."""
        import numpy as np
        from legoesm.core.field import Field
        from legoesm.ocean.restart import load_restart
        n, nlev = state.T.data.shape
        tke_arr = jnp.full((n, nlev - 1), 2.5e-4, dtype=state.T.data.dtype)
        seeded = state._replace(tke=Field(
            data=tke_arr, name="tke", dims=("nCells", "level"),
            units="m^2/s^2"))
        p = tmp_path / "restart.npz"
        np.savez(p, **{f: np.asarray(getattr(seeded, f).data)
                       for f in ("u", "T", "S", "eta", "tke")})
        restored = load_restart(p, state)  # template carry is None
        assert restored.tke is not None
        assert bool(jnp.allclose(restored.tke.data, tke_arr))


class TestNemoBn2OnMPAS:
    """MPAS must EXECUTE the card's stratification, not merely be allowed it.

    The ORCA1 card sets n2_mode="nemo_bn2" with n2_eos_form="teos10", and MPAS
    shares that card. Codex 9408814 #5 named this as the gap the other tests
    miss: source inspection cannot catch bad kwargs, wrong shapes, or a JIT
    failure on the bridge. This runs the profile function.
    """

    def test_profiles_run_with_nemo_bn2_and_teos10(self, mesh, z_coord, state):
        """The card's exact stratification settings, executed on MPAS."""
        import numpy as np
        pf = make_tke_profiles_mpas(VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(n2_mode="nemo_bn2", n2_eos_form="teos10")))
        A_v, K_v = pf(state, mesh, z_coord)
        nc, nl = state.T.data.shape[0], state.T.data.shape[1] - 1
        for name, arr in (("A_v", A_v), ("K_v", K_v)):
            a = np.asarray(arr)
            assert a.shape == (nc, nl), f"{name} shape {a.shape} != {(nc, nl)}"
            assert np.all(np.isfinite(a)), f"{name} has non-finite entries"
            assert np.all(a >= 0.0), f"{name} went negative"

    def test_prognostic_profiles_run_with_nemo_bn2(self, mesh, z_coord, state):
        """The ORCA1 card runs PROGNOSTIC TKE; that is a different return
        arity and a different code path from the diagnostic one above."""
        import numpy as np
        pf = make_tke_profiles_mpas(VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(prognostic=True, n2_mode="nemo_bn2",
                          n2_eos_form="teos10")))
        A_v, K_v, tke_new = pf(state, mesh, z_coord, _wind_forcing(state),
                               dt_tke=3600.0)
        for name, arr in (("A_v", A_v), ("K_v", K_v), ("tke", tke_new)):
            a = np.asarray(arr)
            assert np.all(np.isfinite(a)), f"{name} has non-finite entries"
            assert np.all(a >= 0.0), f"{name} went negative"

    def test_nemo_bn2_changes_the_answer_on_mpas(self, mesh, z_coord, state):
        """Non-vacuity: if the ladders never reached the kernel the two
        stratification modes would return the same array and the test above
        would pass while proving nothing."""
        import numpy as np
        base = make_tke_profiles_mpas(VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(n2_mode="insitu")))(
                state, mesh, z_coord)[1]
        bn2 = make_tke_profiles_mpas(VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(n2_mode="nemo_bn2", n2_eos_form="teos10")))(
                state, mesh, z_coord)[1]
        # `not allclose` is satisfied by NaN, so a broken kernel would pass
        # this "non-vacuity" check vacuously. Require finite first.
        assert np.all(np.isfinite(np.asarray(bn2)))
        assert not np.allclose(np.asarray(base), np.asarray(bn2)), (
            "nemo_bn2 gave the insitu answer on MPAS -- the depth ladders "
            "are not reaching the kernel")

    def test_adiabatic_still_refused_on_mpas(self):
        """The narrowing must not have opened the mode that genuinely needs
        the cell-centre hydrostatic pressure this bridge does not compute."""
        import pytest as _pytest
        with _pytest.raises(NotImplementedError, match="adiabatic"):
            make_tke_profiles_mpas(VerticalMixingConfig(
                scheme="tke", tke=TKEConfig(n2_mode="adiabatic")))
