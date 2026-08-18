"""Unit tests for MPAS Voronoi-mesh AMOC + SSS apply helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.spinup import (
    atlantic_basin_mask_mpas,
    compute_amoc_from_state_mpas,
    compute_acc_from_state_mpas,
    compute_mht_from_state_mpas,
)
from legoesm.ocean.coupler.sss_apply import (
    apply_sss_restoring_step_mpas,
)
from legoesm.ocean.forcing.sss_restoring import (
    SSSRestoringConfig,
    DEFAULT_OMIP2_REGIONS,
)


# ==============================================================================
# Fake VoronoiMesh — minimal connectivity for tests
# ==============================================================================

class _FakeMesh:
    """Tiny stub with the fields ``compute_amoc_from_state_mpas`` reads.

    Builds a 4-cell mesh with 3 edges (a single row of cells at
    different latitudes) so we can drive edge-normal velocities and
    verify the meridional flux + cumulative-depth integral.
    """

    def __init__(self):
        # 4 cells in a north-south chain at lon=0.
        self.nCells = 4
        self.nEdges = 3
        self.latCell = np.deg2rad(np.array([20.0, 25.0, 30.0, 35.0]))
        self.lonCell = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0]))
        # Edges connect adjacent cells (north–south).
        self.latEdge = np.deg2rad(np.array([22.5, 27.5, 32.5]))
        self.lonEdge = np.deg2rad(np.array([0.0, 0.0, 0.0]))
        # Edge normal points north — angleEdge = π/2 (from east).
        self.angleEdge = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        # Edge length: 5° lat × 111 km/deg ≈ 555 km
        self.dvEdge = np.array([5.55e5, 5.55e5, 5.55e5])
        # Connectivity (c1 south of c2).
        self.cellsOnEdge = np.array([
            [0, 1, 2],  # c1: south cell
            [1, 2, 3],  # c2: north cell
        ])


class _FakeMeshTwoBasins:
    """6-cell mesh — 3 cells at Atlantic longitudes + 3 at Pacific.

    Used to verify the Atlantic basin mask filters out Pacific
    contributions.  All cells at the same latitude (26.5 °N) so a
    constant edge flux maps to one band.
    """

    def __init__(self):
        self.nCells = 6
        self.nEdges = 4
        # Three Atlantic cells at lon=0°, three Pacific at lon=180°.
        self.latCell = np.deg2rad(np.array([
            26.0, 27.0, 28.0,    # Atlantic
            26.0, 27.0, 28.0,    # Pacific
        ]))
        self.lonCell = np.deg2rad(np.array([
            0.0, 0.0, 0.0,       # Atlantic (within -75 → 15 band)
            180.0, 180.0, 180.0, # Pacific
        ]))
        self.latEdge = np.deg2rad(np.array([26.5, 27.5, 26.5, 27.5]))
        self.lonEdge = np.deg2rad(np.array([0.0, 0.0, 180.0, 180.0]))
        self.angleEdge = np.array([np.pi / 2] * 4)
        self.dvEdge = np.array([1e5, 1e5, 1e5, 1e5])
        # Edge 0,1: Atlantic; Edge 2,3: Pacific.
        self.cellsOnEdge = np.array([
            [0, 1, 3, 4],
            [1, 2, 4, 5],
        ])


# ==============================================================================
# Atlantic basin mask on MPAS
# ==============================================================================

class TestAtlanticBasinMaskMPAS:

    def test_default_band(self):
        mesh = _FakeMeshTwoBasins()
        mask = atlantic_basin_mask_mpas(mesh)
        # Atlantic cells (lon=0°) → True.
        assert bool(mask[0]) and bool(mask[1]) and bool(mask[2])
        # Pacific cells (lon=180°) → False.
        assert not bool(mask[3])
        assert not bool(mask[4])
        assert not bool(mask[5])

    def test_wraparound_band(self):
        mesh = _FakeMeshTwoBasins()
        mask = atlantic_basin_mask_mpas(
            mesh, lon_min_deg=170.0, lon_max_deg=-170.0,
        )
        # Now Pacific cells (lon=180°) → True; Atlantic cells (lon=0°) → False.
        assert bool(mask[3]) and bool(mask[4]) and bool(mask[5])
        assert not bool(mask[0])


# ==============================================================================
# MPAS AMOC computation
# ==============================================================================

class TestComputeAMOCFromStateMPAS:

    def test_zero_velocity_returns_zero(self):
        mesh = _FakeMesh()
        nlev = 4
        u = np.zeros((mesh.nEdges, nlev))
        h = np.full((mesh.nCells, nlev), 100.0)
        amoc = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=22.5, basin="global",
        )
        assert amoc == 0.0

    def test_northward_upper_transport_gives_positive_amoc(self):
        """Synthetic state with positive v at surface levels +
        negative v at depth (overturning cell) must produce a
        positive RAPID-style AMOC magnitude at the target band."""
        mesh = _FakeMesh()
        nlev = 6
        u = np.zeros((mesh.nEdges, nlev))
        # Northward (positive) at top 3 levels, southward (return) at bottom 3.
        u[:, :3] = 0.05    # 5 cm/s northward
        u[:, 3:] = -0.05
        h = np.full((mesh.nCells, nlev), 200.0)  # uniform 200-m layers
        amoc = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=22.5, basin="global",
            lat_band_width_deg=1.0,
        )
        assert amoc > 0.0, f"AMOC should be positive, got {amoc}"

    def test_reversed_cell_reports_same_magnitude(self):
        """SIGN-AGNOSTIC semantics (2026-08-11): the sign of the ψ peak
        depends on grid orientation / cumsum direction, NOT reliably on the
        physical circulation direction — the signed convention reported
        -0.13 Sv on a tripole state whose true AMOC (NEMO amoc_core,
        cross-validated) is 9.8 Sv.  The diagnostic now reports peak
        MAGNITUDE, matching amoc_core; a reversed cell therefore reports
        the SAME positive strength as the forward cell, and direction is
        not inferable from this scalar."""
        mesh = _FakeMesh()
        nlev = 6
        u = np.zeros((mesh.nEdges, nlev))
        u[:, :3] = 0.05
        u[:, 3:] = -0.05
        h = np.full((mesh.nCells, nlev), 200.0)
        fwd = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=22.5, basin="global",
            lat_band_width_deg=1.0,
        )
        rev = compute_amoc_from_state_mpas(
            -u, h, mesh, target_lat_deg=22.5, basin="global",
            lat_band_width_deg=1.0,
        )
        assert fwd > 0.0
        assert rev == fwd, f"magnitude must be orientation-invariant: {fwd} vs {rev}"

    def test_partial_cell_min_rule_zeros_dry_edge(self):
        """An interior edge against a fully-DRY cell carries no meridional flux
        (min-rule edge thickness -> 0), so AMOC at that band is zero.  A centred
        0.5·(h_wet+0) average would leak phantom flux and give a nonzero AMOC."""
        mesh = _FakeMesh()
        nlev = 6
        u = np.zeros((mesh.nEdges, nlev))
        u[0, :3] = 0.05      # overturning cell on edge 0 (cells 0<->1)
        u[0, 3:] = -0.05
        h = np.full((mesh.nCells, nlev), 200.0)
        h[1] = 0.0           # north cell of edge 0 is land (dry)
        amoc = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=22.5, basin="global",
            lat_band_width_deg=1.0,
        )
        assert abs(amoc) < 1e-9

    def test_target_lat_outside_grid_returns_nan(self):
        mesh = _FakeMesh()
        nlev = 4
        u = np.full((mesh.nEdges, nlev), 0.05)
        h = np.full((mesh.nCells, nlev), 100.0)
        amoc = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=80.0, basin="global",
            lat_tol_deg=2.0,
        )
        assert np.isnan(amoc)

    def test_atlantic_basin_suppresses_pacific_signal(self):
        mesh = _FakeMeshTwoBasins()
        nlev = 4
        u = np.zeros((mesh.nEdges, nlev))
        # Strong northward flow ONLY at Pacific edges (index 2, 3).
        u[2:, :2] = 0.1
        u[2:, 2:] = -0.1
        h = np.full((mesh.nCells, nlev), 200.0)
        amoc_global = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=26.5, basin="global",
            lat_band_width_deg=1.5,
        )
        amoc_atlantic = compute_amoc_from_state_mpas(
            u, h, mesh, target_lat_deg=26.5, basin="atlantic",
            lat_band_width_deg=1.5,
        )
        # Atlantic filter must hide the Pacific signal.
        assert abs(amoc_atlantic) < 1e-6, (
            f"Atlantic mask should suppress Pacific signal; got "
            f"amoc_atlantic={amoc_atlantic}"
        )
        # Global picks the Pacific signal.
        assert abs(amoc_global) > abs(amoc_atlantic)

    def test_unknown_basin_raises(self):
        mesh = _FakeMesh()
        u = np.full((mesh.nEdges, 2), 0.05)
        h = np.full((mesh.nCells, 2), 100.0)
        with pytest.raises(ValueError):
            compute_amoc_from_state_mpas(u, h, mesh, basin="southern")

    def test_shape_mismatch_raises(self):
        mesh = _FakeMesh()
        u = np.zeros((mesh.nEdges, 4))
        h = np.zeros((mesh.nCells, 6))  # different nlev
        with pytest.raises(ValueError):
            compute_amoc_from_state_mpas(u, h, mesh)


# ==============================================================================
# MPAS ACC@Drake (section transport)
# ==============================================================================

class _FakeDrakeMesh:
    """East-west cell chain across the Drake meridian (lon=-68) at lat=-55.

    Cells at lon [-70, -66, -62, -58]; only edge 0 (cells 0↔1) straddles the
    -68 meridian.  Optionally appends an ANTIPODAL edge (cells near +112, the
    antipode of -68) carrying flow, to verify the |d|<90° wrap guard excludes it.
    """

    def __init__(self, antipodal=False):
        lon = [-70.0, -66.0, -62.0, -58.0]
        lat = [-55.0, -55.0, -55.0, -55.0]
        c1 = [0, 1, 2]
        c2 = [1, 2, 3]
        lat_edge = [-55.0, -55.0, -55.0]
        if antipodal:
            # Two extra cells straddling +112° (= -68 + 180), an edge between
            # them: opposite SIGNED circular-longitude but a wrap artefact.
            lon += [110.0, 114.0]
            lat += [-55.0, -55.0]
            c1 += [4]
            c2 += [5]
            lat_edge += [-55.0]
        self.nCells = len(lon)
        self.nEdges = len(c1)
        self.lonCell = np.deg2rad(np.array(lon))
        self.latCell = np.deg2rad(np.array(lat))
        self.latEdge = np.deg2rad(np.array(lat_edge))
        self.dvEdge = np.full(self.nEdges, 1.0e5)        # 100 km edges
        self.cellsOnEdge = np.array([c1, c2])


class TestComputeACCFromStateMPAS:

    def test_eastward_flow_positive_analytic(self):
        """Eastward u on the one straddling edge -> +ve ACC = u·dv·Σh."""
        mesh = _FakeDrakeMesh()
        nlev = 4
        u = np.full((mesh.nEdges, nlev), 0.1)            # 0.1 m/s eastward
        h = np.full((mesh.nCells, nlev), 200.0)
        acc = compute_acc_from_state_mpas(u, h, mesh, drake_lon_deg=-68.0)
        # only edge 0 straddles: s_e=+1, F = 0.1·1e5·200 per level × 4 = 8e6 m³/s
        assert abs(acc - 8.0) < 1e-9

    def test_sign_preserved_westward_negative(self):
        mesh = _FakeDrakeMesh()
        nlev = 4
        h = np.full((mesh.nCells, nlev), 200.0)
        pos = compute_acc_from_state_mpas(
            np.full((mesh.nEdges, nlev), 0.1), h, mesh, drake_lon_deg=-68.0)
        neg = compute_acc_from_state_mpas(
            np.full((mesh.nEdges, nlev), -0.1), h, mesh, drake_lon_deg=-68.0)
        assert pos > 0.0 and neg < 0.0
        assert abs(pos + neg) < 1e-9                      # equal magnitude

    def test_only_straddling_edge_counts(self):
        """Flow on non-straddling edges (both cells east of -68) is ignored."""
        mesh = _FakeDrakeMesh()
        nlev = 4
        h = np.full((mesh.nCells, nlev), 200.0)
        u_all = np.full((mesh.nEdges, nlev), 0.1)        # every edge flows
        u_one = np.zeros((mesh.nEdges, nlev)); u_one[0] = 0.1   # only edge 0
        acc_all = compute_acc_from_state_mpas(u_all, h, mesh, drake_lon_deg=-68.0)
        acc_one = compute_acc_from_state_mpas(u_one, h, mesh, drake_lon_deg=-68.0)
        assert abs(acc_all - acc_one) < 1e-9             # extras don't count

    def test_antipodal_meridian_excluded(self):
        """The |d|<90° guard rejects the antipode-of-Drake edge (wrap artefact)."""
        mesh = _FakeDrakeMesh(antipodal=True)
        nlev = 4
        h = np.full((mesh.nCells, nlev), 200.0)
        u = np.full((mesh.nEdges, nlev), 0.1)            # incl. the antipodal edge
        acc = compute_acc_from_state_mpas(u, h, mesh, drake_lon_deg=-68.0)
        assert abs(acc - 8.0) < 1e-9                      # antipodal edge excluded

    def test_partial_cell_uses_min_rule_not_centered(self):
        """A bottom step at the straddling edge -> min-rule cross-section (the
        shallower cell), NOT a centred average that would count phantom area."""
        mesh = _FakeDrakeMesh()
        nlev = 4
        u = np.zeros((mesh.nEdges, nlev)); u[0] = 0.1
        h = np.full((mesh.nCells, nlev), 200.0)
        h[1, -1] = 0.0                                # cell 1 bottom level = rock
        acc = compute_acc_from_state_mpas(u, h, mesh, drake_lon_deg=-68.0)
        # min(200,0)=0 at the bottom -> only 3 full levels: 0.1·1e5·200·3 = 6 Sv
        # (a centred 0.5·(200+0) would wrongly give 7 Sv).
        assert abs(acc - 6.0) < 1e-9

    def test_coastline_dry_cell_zero_transport(self):
        """A straddling edge against a fully-dry (land) cell carries no flux
        (min-rule -> 0 cross-section, no phantom into-land transport)."""
        mesh = _FakeDrakeMesh()
        nlev = 4
        u = np.full((mesh.nEdges, nlev), 0.1)
        h = np.full((mesh.nCells, nlev), 200.0)
        h[1] = 0.0                                    # cell 1 is land
        acc = compute_acc_from_state_mpas(u, h, mesh, drake_lon_deg=-68.0)
        assert abs(acc) < 1e-9

    def test_no_section_edges_returns_nan(self):
        mesh = _FakeDrakeMesh()
        nlev = 4
        u = np.full((mesh.nEdges, nlev), 0.1)
        h = np.full((mesh.nCells, nlev), 200.0)
        acc = compute_acc_from_state_mpas(u, h, mesh, drake_lon_deg=120.0)
        assert np.isnan(acc)

    def test_out_of_band_returns_nan(self):
        mesh = _FakeDrakeMesh()
        nlev = 4
        u = np.full((mesh.nEdges, nlev), 0.1)
        h = np.full((mesh.nCells, nlev), 200.0)
        acc = compute_acc_from_state_mpas(
            u, h, mesh, drake_lon_deg=-68.0,
            drake_lat_south_deg=-30.0, drake_lat_north_deg=-20.0)
        assert np.isnan(acc)

    def test_shape_mismatch_raises(self):
        mesh = _FakeDrakeMesh()
        u = np.zeros((mesh.nEdges, 4))
        h = np.zeros((mesh.nCells, 6))
        with pytest.raises(ValueError):
            compute_acc_from_state_mpas(u, h, mesh)


# ==============================================================================
# MPAS MHT (global meridional heat transport)
# ==============================================================================

class TestComputeMHTFromStateMPAS:

    def test_northward_warm_positive(self):
        mesh = _FakeMesh()                 # edges at 22.5/27.5/32.5 N (all NH)
        nlev = 4
        u = np.full((mesh.nEdges, nlev), 0.05)        # northward (angle=pi/2)
        th = np.full((mesh.nCells, nlev), 10.0)       # 10 degC
        h = np.full((mesh.nCells, nlev), 200.0)
        r = compute_mht_from_state_mpas(u, th, h, mesh)
        assert r["nh_peak_PW"] > 0.0
        assert r["nh_peak_lat"] > 0.0

    def test_sign_flips(self):
        mesh = _FakeMesh()
        nlev = 4
        th = np.full((mesh.nCells, nlev), 10.0)
        h = np.full((mesh.nCells, nlev), 200.0)
        north = compute_mht_from_state_mpas(
            np.full((mesh.nEdges, nlev), 0.05), th, h, mesh)
        south = compute_mht_from_state_mpas(
            np.full((mesh.nEdges, nlev), -0.05), th, h, mesh)
        # _FakeMesh edges are all NH; northward warm -> NH +, southward -> NH -.
        # (Empty lat bins pad with 0, so compare north vs south rather than
        # asserting south's absolute sign at the padded peak.)
        assert north["nh_peak_PW"] > 0.0
        assert north["nh_peak_PW"] > south["nh_peak_PW"]
        # the flow latitudes (22-33 N) carry the negative southward transport:
        assert float(np.min(south["mht_PW"])) < 0.0

    def test_zero_velocity_zero(self):
        mesh = _FakeMesh()
        nlev = 4
        r = compute_mht_from_state_mpas(
            np.zeros((mesh.nEdges, nlev)),
            np.full((mesh.nCells, nlev), 10.0),
            np.full((mesh.nCells, nlev), 200.0), mesh)
        assert abs(r["nh_peak_PW"]) < 1e-12

    def test_level_mismatch_raises(self):
        mesh = _FakeMesh()
        with pytest.raises(ValueError):
            compute_mht_from_state_mpas(
                np.zeros((mesh.nEdges, 4)),
                np.zeros((mesh.nCells, 6)),
                np.zeros((mesh.nCells, 4)), mesh)


# ==============================================================================
# MPAS SSS apply
# ==============================================================================

class _FakeMPASState:
    """Minimal MPAS-state stub: ``S`` and ``land_mask`` only."""

    def __init__(self, n_cells=6, nlev=3, S_init=34.0, mask_init=1.0):
        self.S = Field(
            jnp.full((n_cells, nlev), S_init, dtype=jnp.float64),
            name="S", dims=("nCells", "nlev"), units="PSU",
        )
        self.land_mask = Field(
            jnp.full((n_cells,), mask_init, dtype=jnp.float64),
            name="land_mask", dims=("nCells",), units="1",
        )

    def _replace(self, **kw):
        new = _FakeMPASState.__new__(_FakeMPASState)
        new.S = self.S
        new.land_mask = self.land_mask
        for k, v in kw.items():
            setattr(new, k, v)
        return new


class TestApplySSSRestoringMPAS:

    def test_disabled_is_noop(self):
        mesh = _FakeMeshTwoBasins()
        state = _FakeMPASState(n_cells=mesh.nCells)
        out = apply_sss_restoring_step_mpas(
            state,
            S_target=np.full(mesh.nCells, 35.0),
            ice_concentration=None,
            config=SSSRestoringConfig(enabled=False),
            mesh=mesh,
            dt=3600.0,
        )
        assert out is state

    def test_salty_bias_freshens_surface(self):
        mesh = _FakeMeshTwoBasins()
        state = _FakeMPASState(n_cells=mesh.nCells, S_init=35.5)
        out = apply_sss_restoring_step_mpas(
            state,
            S_target=np.full(mesh.nCells, 34.7),
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=30.0, regions=(),
            ),
            mesh=mesh,
            dt=86400.0,
        )
        S_top_new = np.asarray(out.S.data)[..., 0]
        assert np.all(S_top_new < 35.5)

    def test_only_surface_layer_modified(self):
        mesh = _FakeMeshTwoBasins()
        state = _FakeMPASState(n_cells=mesh.nCells, S_init=35.5, nlev=3)
        out = apply_sss_restoring_step_mpas(
            state,
            S_target=np.full(mesh.nCells, 34.7),
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=10.0, regions=(),
            ),
            mesh=mesh,
            dt=86400.0,
        )
        S_new = np.asarray(out.S.data)
        assert np.allclose(S_new[..., 1], 35.5)
        assert np.allclose(S_new[..., 2], 35.5)
        assert np.all(S_new[..., 0] < 35.5)

    def test_default_omip2_regions_on_1d_mesh(self):
        """The DEFAULT_OMIP2_REGIONS lat/lon masks (used by run_omip_core2
        --sss-restore) must work on a 1-D Voronoi mesh (the region-mask builder
        is elementwise, so it broadcasts over (nCells,)) and still freshen a
        salty bias toward the target."""
        mesh = _FakeMeshTwoBasins()
        state = _FakeMPASState(n_cells=mesh.nCells, S_init=35.5)
        out = apply_sss_restoring_step_mpas(
            state,
            S_target=np.full(mesh.nCells, 34.7),
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=365.0,
                regions=DEFAULT_OMIP2_REGIONS,
            ),
            mesh=mesh,
            dt=86400.0,
        )
        S_top_new = np.asarray(out.S.data)[..., 0]
        assert np.all(np.isfinite(S_top_new))
        assert np.all(S_top_new < 35.5)          # restored toward 34.7

    def test_land_cells_untouched(self):
        mesh = _FakeMeshTwoBasins()
        state = _FakeMPASState(n_cells=mesh.nCells, S_init=35.5)
        mask_arr = np.asarray(state.land_mask.data).copy()
        mask_arr[:3] = 0.0
        state.land_mask = Field(
            jnp.asarray(mask_arr), name=state.land_mask.name,
            dims=state.land_mask.dims, units=state.land_mask.units,
        )
        out = apply_sss_restoring_step_mpas(
            state,
            S_target=np.full(mesh.nCells, 34.7),
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=10.0, regions=(),
            ),
            mesh=mesh,
            dt=86400.0,
        )
        S_top_new = np.asarray(out.S.data)[..., 0]
        assert np.allclose(S_top_new[:3], 35.5)
        assert np.all(S_top_new[3:] < 35.5)
