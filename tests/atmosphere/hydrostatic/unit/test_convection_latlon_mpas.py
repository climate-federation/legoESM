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
    TiedtkeConfig,
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
        ("tiedtke", "tiedtke", TiedtkeConfig),
        ("bechtold", "bechtold", BechtoldConfig),
    ],
    ids=["zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold"],
)
def scheme_config(request):
    """Yield a (name, ConvectionConfig) pair for each of the 5 new schemes."""
    name, scheme_field, scheme_cls = request.param
    # The test grids carry no land fraction: ZM gets the explicit aquaplanet choice.
    kwargs = {scheme_field: (scheme_cls(land_fraction="none")
                             if name == "zhang_mcfarlane" else scheme_cls())}
    return name, ConvectionConfig(scheme=name, **kwargs)


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
        tend, _ = physics_fn(latlon_state, latlon_grid, sigma_coord)
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
        tend, _ = physics_fn(latlon_state, latlon_grid, sigma_coord)
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
        tend, _ = physics_fn(latlon_state, latlon_grid, sigma_coord)
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
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord)
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
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord)
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
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord)
        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
            f"{scheme_name}: dT_dt has NaN/Inf"
        )
        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))

    def test_tiedtke_mpas_passes_none_not_zeros_for_missing_mc(
        self, mpas_mesh, mpas_state, sigma_coord,
    ):
        """MPAS state has ``state.v is None`` and no ``q_v`` tracer,
        so the bridge cannot synthesize a moisture convergence
        diagnostic.  Pre-fix the bridge passed ``mc_col = zeros`` to
        the leaf, which silently bypassed Tiedtke's saturation-deficit
        proxy (the leaf gates the proxy on
        ``moisture_convergence is None``).  After the fix the bridge
        passes ``None`` and the proxy fires.

        ``smooth_positive_part`` is strictly positive everywhere
        (it is a softplus), so a "carry > 0" assertion would pass
        even with ``mc_col = zeros``.  Instead we exercise the leaf
        directly twice — once with ``moisture_convergence=None``
        (proxy path) and once with ``moisture_convergence=zeros``
        (no-proxy path) — and assert the bridge's dT/dt matches the
        proxy call and *differs* from the zeros call.  This pins the
        actual semantic difference the fix is meant to deliver.

        For the proxy path to differ measurably from the
        ``moisture_convergence=zeros`` path the column needs *some*
        moisture: the proxy is ``column-integrated max(q_v - RH_crit
        q_sat, 0)`` and vanishes identically when ``q_v = 0``.  Attach
        a near-saturated ``q_v`` field to ``mpas_state.tracers`` so
        the proxy fires (~2.7e-14 magnitude under q_v=0 collapses to
        identical noise floor in both paths and the differential
        signal vanishes).
        """
        from legoesm.atmosphere.physics.convection.tiedtke import (
            tiedtke_convection,
        )
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.grids.vertical import pressure_from_sigma
        from legoesm.core.field import Field

        # Build a near-saturated q_v field consistent with the state's T,p_s
        # profile, then wrap in tracers so the bridge's q_v_col path fires.
        nlev = sigma_coord.n_levels
        ncol = mpas_state.T.data.shape[0]
        T_col = mpas_state.T.data.reshape(ncol, nlev)
        _state_dtype = T_col.dtype
        p_s_arr = mpas_state.p_s.data
        p_full_col = pressure_from_sigma(
            sigma_coord.sigma_full, p_s_arr,
        ).reshape(ncol, nlev)
        p_half_col = pressure_from_sigma(
            sigma_coord.sigma_half, p_s_arr,
        ).reshape(ncol, nlev + 1)
        q_sat_col = saturation_mixing_ratio(T_col, p_full_col)
        # 0.95 RH near the surface, ramping to 0.50 aloft — mirrors the
        # spectral-PE test fixture.  The proxy uses RH_crit ~ 0.7, so
        # 0.95 RH layers register as super-critical and the column-MC
        # proxy produces a non-trivial value.
        sigma_full = sigma_coord.sigma_full
        rh_profile = jnp.where(sigma_full > 0.7, 0.95, 0.50)
        q_v_col = (rh_profile[None, :] * q_sat_col).astype(_state_dtype)
        q_v_grid = q_v_col.reshape(mpas_state.T.data.shape)
        mpas_state = mpas_state._replace(
            tracers={
                "q_v": Field(
                    data=q_v_grid, name="q_v",
                    dims=mpas_state.T.dims, units="kg/kg",
                ),
            },
        )

        cfg = ConvectionConfig(scheme="tiedtke", tiedtke=TiedtkeConfig())

        # Bridge call.
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend_bridge, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord)

        # Replicate the column inputs the bridge constructs for the
        # MPAS Tiedtke path.  ``state.v is None`` and edge-vs-cell
        # mismatch → u/v are zeros (this is the bridge's MPAS CMT
        # graceful-degrade branch).  ``phys_state is None`` →
        # conv_prog_profile is zeros at first call.
        # Build column inputs first so we can attach a non-zero q_v
        # both to the in-test ``mpas_state.tracers`` (bridge sees it)
        # and to the proxy/zeros leaf calls (so the proxy path actually
        # has sat_excess > 0 — without this, q_v=0 makes the proxy
        # collapse to the same near-zero ``mc_gate(0)`` floor as the
        # zeros-MC bypass, and the differential test cannot
        # discriminate the bridge fix.  Pre-fix the test exposed
        # ``zeros_diff/proxy_scale ≈ 0/2.6e-14``).
        nlev = sigma_coord.n_levels
        ncol = mpas_state.T.data.shape[0]
        T_col = mpas_state.T.data.reshape(ncol, nlev)
        _state_dtype = T_col.dtype
        p_s = mpas_state.p_s.data
        from legoesm.grids.vertical import pressure_from_sigma
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.core.field import Field
        p_full_col = pressure_from_sigma(
            sigma_coord.sigma_full, p_s,
        ).reshape(ncol, nlev)
        p_half_col = pressure_from_sigma(
            sigma_coord.sigma_half, p_s,
        ).reshape(ncol, nlev + 1)

        q_sat_col = saturation_mixing_ratio(T_col, p_full_col)
        sigma_full = sigma_coord.sigma_full
        rh_profile = jnp.where(sigma_full > 0.7, 0.95, 0.5)
        q_v_col = (rh_profile[None, :] * q_sat_col).astype(_state_dtype)
        q_v_grid = q_v_col.reshape(mpas_state.T.data.shape)
        q_v_field = Field(
            data=q_v_grid, name="q_v",
            dims=mpas_state.T.dims, units="kg/kg",
        )
        mpas_state = mpas_state._replace(tracers={"q_v": q_v_field})

        u_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Bridge call (must come AFTER the tracer attachment).
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend_bridge, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord)

        # Proxy path: moisture_convergence=None → leaf engages the
        # saturation-deficit proxy.
        out_proxy, _ = tiedtke_convection(
            T=T_col, q_v=q_v_col,
            p_full=p_full_col, p_half=p_half_col,
            u=u_col, v=v_col,
            conv_prog_profile=prog_in,
            dt=300.0, config=cfg.tiedtke,
            moisture_convergence=None,
        )
        # No-proxy path: moisture_convergence=zeros((ncol, nlev)) →
        # leaf computes column_MC = 0 and bypasses the proxy.
        mc_zeros = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        out_zeros, _ = tiedtke_convection(
            T=T_col, q_v=q_v_col,
            p_full=p_full_col, p_half=p_half_col,
            u=u_col, v=v_col,
            conv_prog_profile=prog_in,
            dt=300.0, config=cfg.tiedtke,
            moisture_convergence=mc_zeros,
        )
        proxy_dT = out_proxy.dT_dt.reshape(ncol, nlev)
        zeros_dT = out_zeros.dT_dt.reshape(ncol, nlev)
        bridge_dT = tend_bridge.dT_dt.data.reshape(ncol, nlev)

        # The bridge must have called the leaf with
        # ``moisture_convergence=None`` — i.e. its dT/dt matches the
        # proxy call to round-off (we use a relative tolerance so
        # this is robust to dtype promotion in the bridge).
        npt_max_abs = float(jnp.max(jnp.abs(bridge_dT - proxy_dT)))
        proxy_scale = float(jnp.max(jnp.abs(proxy_dT))) + 1e-30
        assert npt_max_abs / proxy_scale < 1e-6, (
            "Tiedtke MPAS bridge dT/dt should match the proxy-path "
            "leaf call (moisture_convergence=None).  "
            f"max |bridge - proxy| / max |proxy| = "
            f"{npt_max_abs / proxy_scale:.3e}.  Bridge may be "
            "passing zeros for moisture_convergence instead of None."
        )
        # And the proxy and zero-MC paths must differ meaningfully.
        # ``smooth_positive_part(0 - threshold)`` is small but
        # non-zero, so this lower bound (1% of proxy magnitude) is
        # comfortably above numerical noise but well within the
        # pre/post-fix gap.
        zeros_diff = float(jnp.max(jnp.abs(proxy_dT - zeros_dT)))
        assert zeros_diff / proxy_scale > 1e-2, (
            "Proxy path and zeros-MC path produce indistinguishable "
            "Tiedtke tendencies — the differential test cannot "
            "discriminate the bridge fix.  "
            f"max |proxy - zeros| / max |proxy| = "
            f"{zeros_diff / proxy_scale:.3e}.  Tighten the test "
            "fixture (e.g. raise T_init or add a CAPE-positive "
            "profile) so the proxy contribution becomes more "
            "distinct from the zero-MC bypass."
        )

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
        cmt_capable = {"zhang_mcfarlane", "tiedtke", "bechtold"}
        if scheme_name not in cmt_capable:
            pytest.skip(f"{scheme_name} produces no CMT")
        physics_fn = make_convection_physics(
            cfg, model_type="mpas", dt=300.0,
        )
        tend, _ = physics_fn(mpas_state, mpas_mesh, sigma_coord)
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
