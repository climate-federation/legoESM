"""Cross-grid coverage for the 5 new convection schemes (PR6).

The previous PR0 audit confirmed that ``zhang_mcfarlane``, ``kain_fritsch``,
``emanuel``, ``tiedtke``, and ``bechtold`` route through the unified
``_make_hydrostatic_convection`` bridge for ``model_type="hydrostatic"``
(used for cubed-sphere) AND ``"mpas"`` (the MPAS combined orchestrator
passes its own ``model_type="mpas"`` through).  Pre-PR6 the bridge had
three cubed-sphere assumptions that broke the lat-lon and MPAS paths:

1. ``ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]`` — IndexError on
   lat-lon ``(n_lat, n_lon)`` and MPAS ``(nCells,)``.
2. Hardcoded ``dims_3d = ("face", "x", "y", "level")`` and
   ``dims_2d = ("face", "x", "y")`` — wrong metadata on lat-lon /
   MPAS Fields.
3. Unconditional ``state.u.data.reshape(ncol, nlev)`` for CMT-capable
   schemes — IndexError on MPAS where ``state.u`` lives on edges
   ``(nEdges, nlev)`` not cells.

PR6 fixed all three plus added an explicit ``model_type="mpas"`` branch
to ``make_convection_physics`` (was a hidden ValueError).

These tests pin the fixes by exercising each of the 5 schemes on a
lat-lon C-grid Held-Suarez state and an MPAS Voronoi Held-Suarez
state and asserting:
- The bridge call succeeds (no IndexError / ValueError).
- ``dT_dt`` has the same shape as the input ``T``.
- All Field metadata uses the input state's grid-native ``dims``.
- ``dv_dt`` is ``None`` for MPAS (since ``state.v is None``) and a
  Field with the right shape for lat-lon.
- All output is finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_init_latlon,
    held_suarez_init_mpas,
)
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    ZhangMcFarlaneConfig,
    KainFritschConfig,
    EmanuelConfig,
    BechtoldConfig,
)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def latlon_grid():
    return create_latlon_grid(n_lat=12, n_lon=24)


@pytest.fixture(scope="module")
def mpas_mesh():
    # Smallest viable mesh: subdivision_level=1 → 42 cells.  Level 2
    # (162 cells) is needed for some dycore operators but the column-
    # wise convection bridge is fine at level 1.
    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(8, sigma_top=0.05)


@pytest.fixture(scope="module")
def latlon_state(latlon_grid, sigma_coord):
    return held_suarez_init_latlon(
        latlon_grid, sigma_coord, T_init=290.0,
        perturbation_amplitude=0.0,  # deterministic fixture
    )


@pytest.fixture(scope="module")
def mpas_state(mpas_mesh, sigma_coord):
    return held_suarez_init_mpas(
        mpas_mesh, sigma_coord, T_init=290.0,
        perturbation_amplitude=0.0,
    )


# ---------------------------------------------------------------------------
# Per-scheme configs (one per scheme — keep defaults conservative)
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="module",
    params=[
        ("zhang_mcfarlane", "zhang_mcfarlane", ZhangMcFarlaneConfig),
        ("kain_fritsch", "kain_fritsch", KainFritschConfig),
        ("emanuel", "emanuel", EmanuelConfig),
        ("bechtold", "bechtold", BechtoldConfig),
    ],
    ids=["zhang_mcfarlane", "kain_fritsch", "emanuel", "bechtold"],
)
def scheme_config(request):
    """Yield a (name, ConvectionConfig) pair for each of the 5 new schemes."""
    name, scheme_field, scheme_cls = request.param
    # The test grids carry no land fraction: ZM gets the explicit aquaplanet choice.
    kwargs = {scheme_field: (scheme_cls(land_fraction="none")
                             if name == "zhang_mcfarlane" else scheme_cls())}
    return name, ConvectionConfig(scheme=name, **kwargs)


def _zm_ps(scheme_name, state):
    """ZM needs the carried PBL height: a fresh PhysicsState (pblh = 0, CAM's
    cold start) for ZM, no carry for the other schemes (as before)."""
    if scheme_name != "zhang_mcfarlane":
        return None
    from tests._zm_cold_start import cold_phys_state
    return cold_phys_state(state.T.data)


# ---------------------------------------------------------------------------
# Lat-lon C-grid coverage
# ---------------------------------------------------------------------------

class TestConvectionLatLon:
    def test_bridge_call_succeeds(
        self, scheme_config, latlon_grid, latlon_state, sigma_coord,
    ):
        """The bridge runs end-to-end on a lat-lon HydrostaticState."""
        scheme_name, cfg = scheme_config
        physics_fn = make_convection_physics(
            cfg, model_type="hydrostatic", dt=300.0,
        )
        # Returns (tendencies, conv_prog_out).
        tend, _ = physics_fn(latlon_state, latlon_grid, sigma_coord,
                             phys_state=_zm_ps(scheme_name, latlon_state))
        # Basic structure.
        assert tend.dT_dt.data.shape == latlon_state.T.data.shape, (
            f"{scheme_name}: dT_dt shape mismatch"
        )
        assert tend.du_dt.data.shape == latlon_state.u.data.shape
        assert tend.dv_dt is not None, (
            f"{scheme_name}: lat-lon has v → dv_dt should be a Field"
        )
        assert tend.dv_dt.data.shape == latlon_state.v.data.shape
        assert tend.dp_s_dt.data.shape == latlon_state.p_s.data.shape

    def test_field_dims_match_grid_native_metadata(
        self, scheme_config, latlon_grid, latlon_state, sigma_coord,
    ):
        """Returned Field metadata uses lat-lon dims (``("lat","lon",...)``)
        rather than the legacy hardcoded ``("face","x","y",...)``."""
        scheme_name, cfg = scheme_config
        physics_fn = make_convection_physics(
            cfg, model_type="hydrostatic", dt=300.0,
        )
        tend, _ = physics_fn(latlon_state, latlon_grid, sigma_coord,
                             phys_state=_zm_ps(scheme_name, latlon_state))
        assert tend.dT_dt.dims == latlon_state.T.dims, (
            f"{scheme_name}: dT_dt dims {tend.dT_dt.dims} should match "
            f"state.T.dims {latlon_state.T.dims}"
        )
        assert tend.dp_s_dt.dims == latlon_state.p_s.dims
        assert tend.du_dt.dims == latlon_state.u.dims

    def test_finite_output(
        self, scheme_config, latlon_grid, latlon_state, sigma_coord,
    ):
        """All tendency fields are finite."""
        scheme_name, cfg = scheme_config
        physics_fn = make_convection_physics(
            cfg, model_type="hydrostatic", dt=300.0,
        )
        tend, _ = physics_fn(latlon_state, latlon_grid, sigma_coord,
                             phys_state=_zm_ps(scheme_name, latlon_state))
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
            f"{scheme_name}: dT_dt has NaN/Inf"
        )
        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dv_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))


# ---------------------------------------------------------------------------
# MPAS Voronoi coverage
# ---------------------------------------------------------------------------

class TestConvectionMPAS:
    def test_mpas_dispatch_no_value_error(self, scheme_config):
        """``make_convection_physics(model_type="mpas")`` must NOT raise.

        Pre-PR6 the dispatcher only accepted "hydrostatic",
        "nonhydrostatic", "spectral_pe" — passing "mpas" (which the
        MPAS combined orchestrator does) raised ValueError.
        """
        scheme_name, cfg = scheme_config
        # Should resolve to the hydrostatic factory under the hood.
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        assert callable(physics_fn)

    def test_bridge_call_succeeds(
        self, scheme_config, mpas_mesh, mpas_state, sigma_coord,
    ):
        """The bridge runs end-to-end on an MPAS HydrostaticState."""
        scheme_name, cfg = scheme_config
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord,
                             phys_state=_zm_ps(scheme_name, mpas_state))
        assert tend.dT_dt.data.shape == mpas_state.T.data.shape, (
            f"{scheme_name}: dT_dt shape mismatch"
        )
        # u stays on edges.
        assert tend.du_dt.data.shape == mpas_state.u.data.shape
        # MPAS has v=None → dv_dt should also be None.
        assert tend.dv_dt is None, (
            f"{scheme_name}: MPAS state.v is None, but bridge emitted "
            f"dv_dt={tend.dv_dt}"
        )
        assert tend.dp_s_dt.data.shape == mpas_state.p_s.data.shape

    def test_field_dims_match_grid_native_metadata(
        self, scheme_config, mpas_mesh, mpas_state, sigma_coord,
    ):
        """Returned Field metadata uses MPAS dims (``("nCells","level")``,
        ``("nEdges","level")``)."""
        scheme_name, cfg = scheme_config
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord,
                             phys_state=_zm_ps(scheme_name, mpas_state))
        assert tend.dT_dt.dims == mpas_state.T.dims, (
            f"{scheme_name}: dT_dt dims {tend.dT_dt.dims} should match "
            f"state.T.dims {mpas_state.T.dims}"
        )
        assert tend.dp_s_dt.dims == mpas_state.p_s.dims
        assert tend.du_dt.dims == mpas_state.u.dims

    def test_finite_output(
        self, scheme_config, mpas_mesh, mpas_state, sigma_coord,
    ):
        """All tendency fields are finite on MPAS."""
        scheme_name, cfg = scheme_config
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord,
                             phys_state=_zm_ps(scheme_name, mpas_state))
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
            f"{scheme_name}: dT_dt has NaN/Inf"
        )
        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))

    def test_kf_wrf_variant_mpas_needs_reconstructed_winds(
        self, mpas_mesh, mpas_state, sigma_coord,
    ):
        """The opt-in WRF KF-Eta variant on MPAS: refused with mpas_cmt off
        (it would run on zero winds), finite with the edge-normal
        reconstruction on.  The smooth default runs either way (above)."""
        wrf = KainFritschConfig(variant="wrf_kfeta")
        off = make_convection_physics(
            ConvectionConfig(scheme="kain_fritsch", kain_fritsch=wrf),
            model_type="mpas", dt=300.0)
        with pytest.raises(ValueError, match="mpas_cmt"):
            off(mpas_state, mpas_mesh, sigma_coord)
        # With the reconstruction on, the port receives exactly the
        # reconstructed cell winds (spy on the port's entry; codex r1).
        import numpy as np

        from legoesm.atmosphere.physics.convection import kain_fritsch_wrf
        from legoesm.grids.voronoi import reconstruct_cell_velocity
        real = kain_fritsch_wrf.kain_fritsch_wrf_convection
        seen = {}
        # the fixture starts at rest; give the edges a non-trivial wind
        ne, nl = mpas_state.u.data.shape
        st = mpas_state._replace(u=mpas_state.u.replace(
            data=5.0 * jnp.sin(jnp.arange(ne, dtype=mpas_state.u.data.dtype))[:, None]
            * jnp.ones((1, nl), dtype=mpas_state.u.data.dtype)))

        def spy(*a, **kw):
            seen["u"], seen["v"] = kw["u"], kw["v"]
            return real(*a, **kw)

        mp = pytest.MonkeyPatch()
        mp.setattr(kain_fritsch_wrf, "kain_fritsch_wrf_convection", spy)
        try:
            on = make_convection_physics(
                ConvectionConfig(scheme="kain_fritsch", kain_fritsch=wrf,
                                 mpas_cmt=True),
                model_type="mpas", dt=300.0)
            tend, _ = on(st, mpas_mesh, sigma_coord)
        finally:
            mp.undo()
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        u_ref, v_ref = reconstruct_cell_velocity(st.u.data, mpas_mesh)
        assert float(jnp.max(jnp.abs(u_ref))) > 0.0  # non-trivial winds
        np.testing.assert_allclose(np.asarray(seen["u"]), np.asarray(u_ref),
                                   rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(np.asarray(seen["v"]), np.asarray(v_ref),
                                   rtol=1e-12, atol=0.0)

    def test_mpas_cmt_is_zero_documented_limitation(
        self, scheme_config, mpas_mesh, mpas_state, sigma_coord,
    ):
        """Documented PR6 limitation: convective momentum transport
        (CMT) on MPAS is currently zero because the bridge does not
        interpolate edge winds to cells.  This test pins the
        limitation so that a future PR adding edge→cell interpolation
        will trip and prompt explicit reconsideration of this gate."""
        scheme_name, cfg = scheme_config
        # Only CMT-capable schemes have a non-trivial CMT path.
        cmt_capable = {"zhang_mcfarlane", "bechtold"}
        if scheme_name not in cmt_capable:
            pytest.skip(f"{scheme_name} produces no CMT")
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord,
                             phys_state=_zm_ps(scheme_name, mpas_state))
        # CMT was disabled because state.u is on edges (shape mismatch
        # vs ncol=nCells).  du_dt should be exactly zero.
        max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
        assert max_du == 0.0, (
            f"{scheme_name}: MPAS du_dt should be zero (CMT disabled "
            f"due to edge-vs-cell wind staggering), got max |du_dt| "
            f"= {max_du}.  If the bridge now interpolates edge→cell "
            f"winds, update or remove this test."
        )


# ---------------------------------------------------------------------------
# Sibling-physics MPAS dispatch (PR6 also fixed microphysics; turbulence
# and GWD remain intentionally unsupported with a clear error message).
# ---------------------------------------------------------------------------

class TestMicrophysicsMPAS:
    """The microphysics bridge needs the same grid-portability fixes
    as convection (generic ncol formula, dim metadata derived from
    state, MPAS-aware u/v shaping).  Microphysics doesn't read v
    so MPAS support is straightforward."""

    def test_mpas_dispatch_succeeds(self):
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig, KesslerConfig,
        )
        from legoesm.atmosphere.physics.microphysics.integration import (
            make_microphysics_physics,
        )
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        fn = make_microphysics_physics(cfg, model_type="mpas", dt=300.0)
        assert callable(fn)

    def test_mpas_bridge_returns_finite_tendencies(
        self, mpas_mesh, mpas_state, sigma_coord,
    ):
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig, KesslerConfig,
        )
        from legoesm.atmosphere.physics.microphysics.integration import (
            make_microphysics_physics,
        )
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        fn = make_microphysics_physics(cfg, model_type="mpas", dt=300.0)
        tend = fn(mpas_state, mpas_mesh, sigma_coord)
        # Grid-native shapes preserved.
        assert tend.dT_dt.data.shape == mpas_state.T.data.shape
        assert tend.du_dt.data.shape == mpas_state.u.data.shape
        # MPAS state.v is None → dv_dt must be None to match the
        # HydrostaticTendencies contract.
        assert tend.dv_dt is None
        # All output finite.
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))

    def test_mpas_dim_metadata_matches_state(
        self, mpas_mesh, mpas_state, sigma_coord,
    ):
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig, KesslerConfig,
        )
        from legoesm.atmosphere.physics.microphysics.integration import (
            make_microphysics_physics,
        )
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        fn = make_microphysics_physics(cfg, model_type="mpas", dt=300.0)
        tend = fn(mpas_state, mpas_mesh, sigma_coord)
        assert tend.dT_dt.dims == mpas_state.T.dims
        assert tend.dp_s_dt.dims == mpas_state.p_s.dims
        assert tend.du_dt.dims == mpas_state.u.dims

    def test_lat_lon_bridge_returns_finite_tendencies(
        self, latlon_grid, latlon_state, sigma_coord,
    ):
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig, KesslerConfig,
        )
        from legoesm.atmosphere.physics.microphysics.integration import (
            make_microphysics_physics,
        )
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        fn = make_microphysics_physics(cfg, model_type="hydrostatic", dt=300.0)
        tend = fn(latlon_state, latlon_grid, sigma_coord)
        assert tend.dT_dt.data.shape == latlon_state.T.data.shape
        assert tend.dv_dt is not None  # lat-lon has v
        assert tend.dT_dt.dims == latlon_state.T.dims
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))


class TestTurbulenceGWDMPASStatus:
    """Status of turbulence / GWD dispatch on MPAS Voronoi meshes.

    Gravity-wave drag is supported: the MPAS bridge reconstructs
    cell-centered winds from the prognostic edge-normal velocity
    (:func:`legoesm.grids.voronoi.reconstruct_cell_velocity`), runs the
    column GWD backend on cell quantities, and projects the
    cell-centered wind tendencies back to edge-normal form via
    ``angleEdge``.  GWD produces only momentum and temperature
    tendencies — no tracer or stateful TKE outputs — so the MPAS step
    can consume the resulting ``HydrostaticTendencies`` without
    additional plumbing (audit 2026-05-12 MEDIUM #10).

    Turbulence is now supported on MPAS (UNBLOCKED): the factory builds an
    MPAS turbulence ``physics_fn`` (``_make_mpas_turbulence``) that reconstructs
    cell-centered winds, runs the column backend, and projects the wind
    tendencies back to edge-normal form — and ``MPASPrimitiveEquationModel.step``
    now carries ``state.tracers`` and threads ``PhysicsState`` so the q_v
    diffusion tendency and any prognostic TKE are no longer silently dropped (the
    earlier no-ship concern). ``make_turbulence_physics(..., model_type="mpas")``
    therefore returns a callable rather than raising. (The companion bridge tests
    above exercise the dispatch end-to-end.)"""

    def test_turbulence_mpas_produces_finite_nonzero_tendencies(
        self, mpas_mesh, mpas_state, sigma_coord,
    ):
        """Behavioural proof that MPAS turbulence is genuinely unblocked — not a
        stub that merely returns a callable. The bridge fn runs end-to-end on an
        MPAS state and returns finite, non-trivial tendencies (this was a
        deliberate ``NotImplementedError`` before the MPAS step gained tracer +
        ``PhysicsState`` plumbing). Also checks the pipeline wiring
        (``_wants_forcing``) so the MPAS step threads forcing and consumes the
        returned (du/dT + q_v tracer + TKE) tendencies."""
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.turbulence.integration import (
            make_turbulence_physics,
        )
        from legoesm.core.field import Field
        turb_fn = make_turbulence_physics(
            TurbulenceConfig(scheme="louis"), model_type="mpas", dt=300.0,
        )
        assert callable(turb_fn)
        assert getattr(turb_fn, "_wants_forcing", False) is True, (
            "MPAS turbulence physics_fn must be forcing-aware so the MPAS step "
            "threads forcing + consumes its tracer/TKE tendencies."
        )
        # The held_suarez_init_mpas fixture is a RESTING state (u_edge ≡ 0).
        # Louis turbulence is shear-driven, so a zero-wind column yields
        # exactly-zero tendencies — physically correct, but it cannot prove
        # the bridge is "not a stub".  Impose a sheared edge-normal wind
        # (linear 0→20 m/s in the vertical, mirroring the lat-lon validator
        # column) so the resolved gradient actually drives vertical mixing.
        nedges, nlev = mpas_state.u.data.shape
        u_shear = jnp.broadcast_to(
            jnp.linspace(0.0, 20.0, nlev)[None, :], (nedges, nlev),
        ).astype(mpas_state.u.data.dtype)
        mpas_state = mpas_state._replace(
            u=Field(
                data=u_shear, name="u", dims=mpas_state.u.dims,
                units="m/s",
                staggering=getattr(mpas_state.u, "staggering", "edge"),
            ),
        )
        # Invoke the bridge end-to-end (forcing defaults to None and is handled).
        result = turb_fn(mpas_state, mpas_mesh, sigma_coord)
        # MPAS turbulence returns ``(HydrostaticTendencies, tke_out)``.
        tend = result[0] if isinstance(result, tuple) else result
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
        # Not a no-op stub: turbulence actually moves the column (vertical
        # diffusion of momentum/heat from the resolved gradients).
        moved = (
            float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0
            or float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
        )
        assert moved, "MPAS turbulence produced all-zero tendencies (stub?)."

    def test_gwd_mpas_returns_callable(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            make_gwd_physics,
        )
        cfg = GravityWaveDragConfig(scheme="lindzen")
        fn = make_gwd_physics(cfg, model_type="mpas", dt=300.0)
        assert callable(fn)


# ---------------------------------------------------------------------------
# WRF Tiedtke (own contract: dynamics q_v tendency + lagged turbulence
# carries, omega, cell-centre winds; so not in the generic parametrisation)
# ---------------------------------------------------------------------------

def _tiedtke_ps(T_native, *, carries=True):
    import math
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    shape = jnp.shape(T_native)
    ncol, nlev = math.prod(shape[:-1]), shape[-1]
    ps = init_physics_state(
        ncol, nlev, PhysicsConfig(convection=ConvectionConfig(scheme="tiedtke")),
        dtype=jnp.asarray(T_native).dtype)
    if carries:
        ps = ps._replace(dyn_tendency_qv=jnp.zeros(shape, ps.evap_sfc.dtype))
    return ps


def test_tiedtke_latlon_bridge_refuses_without_dynamics_tendency(
        latlon_grid, latlon_state, sigma_coord):
    fn = make_convection_physics(ConvectionConfig(scheme="tiedtke"),
                                 model_type="hydrostatic", dt=300.0)
    with pytest.raises(ValueError, match="needs PhysicsState"):
        fn(latlon_state, latlon_grid, sigma_coord,
           phys_state=_tiedtke_ps(latlon_state.T.data, carries=False))


def test_tiedtke_latlon_bridge_runs_with_carries(
        latlon_grid, latlon_state, sigma_coord):
    fn = make_convection_physics(ConvectionConfig(scheme="tiedtke"),
                                 model_type="hydrostatic", dt=300.0)
    from legoesm.core.field import Field
    T = latlon_state.T
    st = latlon_state._replace(tracers={
        **(latlon_state.tracers or {}),
        "q_i": Field(jnp.zeros_like(T.data), name="q_i", dims=T.dims)})
    tend, _ = fn(st, latlon_grid, sigma_coord,
                 phys_state=_tiedtke_ps(T.data))
    assert tend.dT_dt.data.shape == T.data.shape
    assert tend.dv_dt.data.shape == latlon_state.v.data.shape
    assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
    assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))


def test_tiedtke_mpas_bridge_refuses_without_cell_winds(
        mpas_mesh, mpas_state, sigma_coord):
    """MPAS without mpas_cmt hands schemes zero winds; WRF Tiedtke's
    cumulus friction reads u, v, so it refuses (vote V10)."""
    fn = make_convection_physics(ConvectionConfig(scheme="tiedtke"),
                                 model_type="mpas", dt=300.0)
    with pytest.raises(ValueError, match="cell-centred winds"):
        fn(mpas_state, mpas_mesh, sigma_coord,
           phys_state=_tiedtke_ps(mpas_state.T.data))
