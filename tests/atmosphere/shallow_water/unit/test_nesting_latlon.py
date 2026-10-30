"""One-way (parent -> child) lat-lon nesting: geometry, interpolation, capability.

Covers the substrate grid construct (:mod:`legoesm.grids.nesting`), its
parent->child interpolation operator, and the capability wiring
(``instantiate(..., nesting=True)``).  The shallow-water nest stepping driver
that used to be tested here was deleted (test-only, never wired).

Run (x64, CPU):
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/atmosphere/shallow_water/unit/test_nesting_latlon.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.nesting import (
    NestedLatLonGrid,
    create_nested_latlon_grid,
    apply_boundary_interp,
)
from legoesm.grids.capability import instantiate, capability_matrix
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    williamson_test2_cgrid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nest() -> NestedLatLonGrid:
    return create_nested_latlon_grid(
        parent_n_lat=32,
        refinement_ratio=3,
        lat_south_deg=10.0,
        lat_north_deg=50.0,
        lon_west_deg=40.0,
        lon_east_deg=120.0,
        n_halo=2,
        n_relax=4,
    )


@pytest.fixture
def fp64_policy():
    """Temporarily set fp64 STORAGE so mass is conserved to ~round-off.

    The mass-conservation guarantee is fundamentally storage-precision bound: the
    fixer's uniform-depth correction is rounded to the state dtype each step, so
    fp32 storage caps relative conservation at ~1e-7 (N*eps over the interior
    cells), while fp64 reaches ~1e-15.  The 1e-12 target therefore requires fp64
    storage (consistent with JAX_ENABLE_X64=1).  Restored after the test.
    """
    from legoesm.core.precision import get_policy, set_policy, PrecisionPolicy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


@pytest.fixture
def nest_fp64(fp64_policy) -> NestedLatLonGrid:
    return create_nested_latlon_grid(
        parent_n_lat=32, refinement_ratio=3,
        lat_south_deg=10.0, lat_north_deg=50.0,
        lon_west_deg=40.0, lon_east_deg=120.0, n_halo=2, n_relax=4,
    )


# ---------------------------------------------------------------------------
# Geometry: exact integer refinement, child nested inside parent
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_refinement_ratio(self, nest):
        r = nest.refinement_ratio
        assert r == 3
        # Child cell spacing is exactly parent/r in BOTH directions.
        assert nest.child.dlat == pytest.approx(nest.parent.dlat / r, rel=1e-12)
        assert nest.child.dlon == pytest.approx(nest.parent.dlon / r, rel=1e-12)

    def test_child_counts_are_multiples_of_r(self, nest):
        r = nest.refinement_ratio
        assert nest.child.n_lat % r == 0
        assert nest.child.n_lon % r == 0

    def test_child_inside_parent_lat_band(self, nest):
        # Child latitudes lie strictly inside the parent cell-centre band so the
        # bilinear boundary stencil never extrapolates past a pole.
        plat = np.asarray(nest.parent.lat)
        clat = np.asarray(nest.child.lat)
        assert clat.min() > plat.min()
        assert clat.max() < plat.max()

    def test_child_cell_centres_nest_in_parent(self, nest):
        # Each parent cell in the footprint splits into r child cells whose
        # centres average to the parent centre (exact centre nesting).
        r = nest.refinement_ratio
        clat = np.asarray(nest.child.lat)
        # The r-cell running means of the child centres reproduce a subset of
        # parent centres (to round-off).
        grouped = clat.reshape(-1, r).mean(axis=1)
        plat = np.asarray(nest.parent.lat)
        # every grouped value must coincide with SOME parent centre.  Tolerance
        # is float32-storage-bound (the default precision policy stores grid
        # axes as float32; the refinement itself is computed in float64).
        atol = 1e-9 if clat.dtype == np.float64 else 1e-6
        for g in grouped:
            assert np.min(np.abs(plat - g)) < atol

    def test_interior_boundary_masks_partition(self, nest):
        m_int = np.asarray(nest.interior_mask)
        m_bnd = np.asarray(nest.boundary_mask)
        assert np.allclose(m_int + m_bnd, 1.0)
        # interior is a contiguous block inset by n_halo
        nh = nest.n_halo
        assert m_int[nh:-nh, nh:-nh].all()
        assert m_int[:nh, :].sum() == 0
        assert m_int[:, :nh].sum() == 0

    def test_rejects_pole_adjacent_box(self):
        with pytest.raises(ValueError, match="margin north and south"):
            create_nested_latlon_grid(
                parent_n_lat=16, refinement_ratio=2,
                lat_south_deg=-89.9, lat_north_deg=89.9,
                lon_west_deg=0.0, lon_east_deg=90.0,
            )

    def test_rejects_bad_ratio(self):
        with pytest.raises(ValueError, match="refinement_ratio"):
            create_nested_latlon_grid(
                parent_n_lat=16, refinement_ratio=1,
                lat_south_deg=10.0, lat_north_deg=40.0,
                lon_west_deg=0.0, lon_east_deg=60.0,
            )


# ---------------------------------------------------------------------------
# Boundary interpolation operator
# ---------------------------------------------------------------------------


class TestInterp:
    def test_constant_field_exact(self, nest):
        # Bilinear interpolation of a constant parent field is exact everywhere.
        c = 7.25
        parent_field = jnp.full((nest.parent.n_lat, nest.parent.n_lon), c)
        out = apply_boundary_interp(parent_field, nest.interp_centers)
        assert out.shape == (nest.child.n_lat, nest.child.n_lon)
        assert jnp.allclose(out, c, atol=1e-10)

    def test_weights_sum_to_one(self, nest):
        for w in (nest.interp_centers, nest.interp_uface, nest.interp_vface):
            s = np.asarray(w.weights).sum(axis=-1)
            assert np.allclose(s, 1.0, atol=1e-10)

    def test_linear_lon_field_reproduced(self, nest):
        # A field LINEAR in longitude index is reproduced by bilinear interp at
        # child cell centres (bilinear is exact on bilinear data away from the
        # periodic seam, which this interior box avoids).
        plon = np.asarray(nest.parent.lon)
        field = jnp.asarray(np.broadcast_to(plon[None, :], (nest.parent.n_lat, nest.parent.n_lon)).copy())
        out = np.asarray(apply_boundary_interp(field, nest.interp_centers))
        expect = np.asarray(nest.child.lon2d)
        # interior of the child (away from any wrap) matches the child lon grid
        nh = nest.n_halo
        assert np.allclose(out[nh:-nh, nh:-nh], expect[nh:-nh, nh:-nh], atol=1e-6)

    def test_face_prolongation_reproduces_parent_face(self, nest_fp64):
        # BUG 1 regression: the u/v-face operators must source DIRECTLY from the
        # parent FACE fields (stagger-correct), so a child face coincident with a
        # parent face REPRODUCES the parent face value to ~round-off — not a
        # smoothed blend of adjacent parent-face cell-centre averages.  Under
        # integer refinement (r=3) every r-th child face lines up with a parent
        # face exactly.
        nest = nest_fp64
        p, c = nest.parent, nest.child
        r = nest.refinement_ratio

        # --- u-faces (lon interfaces, at cell-centre latitudes) ---
        pu = jnp.asarray(
            np.random.default_rng(0).standard_normal((p.n_lat, p.n_lon + 1))
        )
        cu = np.asarray(apply_boundary_interp(pu, nest.interp_uface))
        # parent u-face longitudes and child u-face longitudes
        pu_lon = float(np.asarray(p.lon)[0]) - 0.5 * p.dlon + np.arange(p.n_lon + 1) * p.dlon
        cu_lon = float(np.asarray(c.lon)[0]) - 0.5 * c.dlon + np.arange(c.n_lon + 1) * c.dlon
        plat = np.asarray(p.lat)
        clat = np.asarray(c.lat)
        # match aligned child u-faces (col) and child rows to parent faces/rows
        n_matched_u = 0
        for kc, lon in enumerate(cu_lon):
            kp = int(np.argmin(np.abs(pu_lon - lon)))
            if abs(pu_lon[kp] - lon) > 1e-9:
                continue
            for jc, la in enumerate(clat):
                jp = int(np.argmin(np.abs(plat - la)))
                if abs(plat[jp] - la) > 1e-9:
                    continue
                assert abs(cu[jc, kc] - float(pu[jp, kp])) < 1e-12
                n_matched_u += 1
        assert n_matched_u > 0  # at least some faces align (r-fold)

        # --- v-faces (lat interfaces, at cell-centre longitudes) ---
        pv = jnp.asarray(
            np.random.default_rng(1).standard_normal((p.n_lat + 1, p.n_lon))
        )
        cv = np.asarray(apply_boundary_interp(pv, nest.interp_vface))
        pv_lat = float(np.asarray(p.lat)[0]) - 0.5 * p.dlat + np.arange(p.n_lat + 1) * p.dlat
        cv_lat = float(np.asarray(c.lat)[0]) - 0.5 * c.dlat + np.arange(c.n_lat + 1) * c.dlat
        plon = np.asarray(p.lon)
        clon = np.asarray(c.lon)
        n_matched_v = 0
        for jc, la in enumerate(cv_lat):
            jp = int(np.argmin(np.abs(pv_lat - la)))
            if abs(pv_lat[jp] - la) > 1e-9:
                continue
            for ic, lo in enumerate(clon):
                ip = int(np.argmin(np.abs(plon - lo)))
                if abs(plon[ip] - lo) > 1e-9:
                    continue
                assert abs(cv[jc, ic] - float(pv[jp, ip])) < 1e-12
                n_matched_v += 1
        assert n_matched_v > 0

    def test_balanced_w2_parent_gives_zero_boundary_v(self, nest_fp64):
        # BUG 1: a BALANCED Williamson-2 C-grid parent (v-face wind == 0
        # everywhere by construction) must prolong to ~0 v-wind on the child v
        # faces in the prescribed boundary band + relaxation zone — the
        # stagger-correct face-to-face operator does NOT manufacture spurious
        # meridional wind there.  (The old cell-centre round-trip could, because
        # averaging u-faces and re-interpolating breaks the discrete balance.)
        nest = nest_fp64
        parent_ic = williamson_test2_cgrid(nest.parent)
        # W2 is a zonal geostrophic flow: parent v faces are identically 0.
        assert float(jnp.max(jnp.abs(parent_ic.v))) < 1e-10
        bc_v = apply_boundary_interp(parent_ic.v, nest.interp_vface)
        # child v faces over the band + relaxation zone must stay ~0.
        wv = np.asarray(nest.relax_weight_vface)  # >0 on band+relax v faces
        band = wv > 0.0
        assert band.any()
        assert float(np.max(np.abs(np.asarray(bc_v)[band]))) < 1e-10


# ---------------------------------------------------------------------------
# Capability wiring
# ---------------------------------------------------------------------------


class TestCapability:
    def test_matrix_reports_latlon_nesting(self):
        mat = capability_matrix()
        assert "regional" in mat["latlon"]["nesting_extents"]
        # other grids still report no nesting
        assert mat["cubed_sphere"]["nesting_extents"] == []

    def test_instantiate_builds_nest(self):
        nest = instantiate(
            "latlon", extent="regional", nesting=True,
            parent_n_lat=24, refinement_ratio=2,
            lat_south_deg=0.0, lat_north_deg=40.0,
            lon_west_deg=0.0, lon_east_deg=80.0,
        )
        assert isinstance(nest, NestedLatLonGrid)
        assert nest.refinement_ratio == 2

    def test_instantiate_rejects_nesting_other_grid(self):
        with pytest.raises(NotImplementedError, match="nesting"):
            instantiate("cubed_sphere", extent="regional", nesting=True, n=12)

    def test_instantiate_rejects_operators_with_nesting(self):
        with pytest.raises(ValueError, match="pair"):
            instantiate(
                "latlon", extent="regional", nesting=True, operators=True,
                parent_n_lat=24, refinement_ratio=2,
                lat_south_deg=0.0, lat_north_deg=40.0,
                lon_west_deg=0.0, lon_east_deg=80.0,
            )
