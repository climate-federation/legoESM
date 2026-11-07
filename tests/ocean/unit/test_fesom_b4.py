"""Stage-B4 FESOM harmonization units that need NO fesom_jax install:

* ``build_node_neighbor_table`` (mesh.edges -> MPAS-convention adjacency)
  against a hand-built table, and through ``laplacian_smooth_voronoi``
  against a hand-computed neighbour average;
* area-conservative runoff renorm on a node cloud (the smoother + renorm
  pair conserves the source area-integral exactly);
* the NEMO monthly-init VERTICAL interpolation (``target_depths=``):
  pure-kernel exactness on linear profiles + the end-to-end loader on a
  synthetic 75-level NetCDF pair regridded onto a coarser node-cloud
  ladder.

Run with ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.bathymetry import laplacian_smooth_voronoi
from legoesm.ocean.dynamics.ocean_model_fesom import build_node_neighbor_table
from legoesm.ocean.forcing.nemo_native_fields import (
    _interp_columns_to_depths,
    load_nemo_monthly_init_ts,
)


# ===========================================================================
# Node neighbour table (runoff coastal spread topology)
# ===========================================================================

# Tiny synthetic mesh: 5 nodes, edges of a "bowtie"
#   0-1, 1-2, 0-2 (triangle), 2-3, 3-4  (tail)
_EDGES = np.array([[0, 1], [1, 2], [0, 2], [2, 3], [3, 4]])
_N = 5
# Hand-built adjacency: 0:{1,2} 1:{0,2} 2:{1,0,3} 3:{2,4} 4:{3}
_EXPECT_NBRS = {0: {1, 2}, 1: {0, 2}, 2: {0, 1, 3}, 3: {2, 4}, 4: {3}}


class TestNodeNeighborTable:
    def test_matches_hand_built_adjacency(self):
        tbl, deg = build_node_neighbor_table(_EDGES, _N)
        assert tbl.shape == (3, _N)          # max degree = 3 (node 2)
        assert deg.tolist() == [2, 2, 3, 2, 1]
        for c in range(_N):
            got = {int(x) for x in tbl[: deg[c], c]}
            assert got == _EXPECT_NBRS[c], f"node {c}"
            # padding slots are exactly -1
            assert (tbl[deg[c]:, c] == -1).all()

    def test_rejects_bad_shapes_and_ids(self):
        with pytest.raises(ValueError, match="node-id pairs"):
            build_node_neighbor_table(np.arange(6), _N)
        with pytest.raises(ValueError, match="outside"):
            build_node_neighbor_table(np.array([[0, 5]]), _N)

    def test_rejects_duplicate_and_self_loop_edges(self):
        # A directed export (both orderings present) would double-count
        # every neighbour in the smoother's mean — refuse, never mis-weight.
        with pytest.raises(ValueError, match="duplicate"):
            build_node_neighbor_table(np.array([[0, 1], [1, 0]]), _N)
        with pytest.raises(ValueError, match="self-loop"):
            build_node_neighbor_table(np.array([[2, 2]]), _N)

    def test_smoother_matches_hand_computed_average(self):
        """One pass of laplacian_smooth_voronoi on the synthetic table must
        equal the hand-computed 0.5*x + 0.5*mean(self+neighbours)."""
        tbl, deg = build_node_neighbor_table(_EDGES, _N)
        x = np.array([1.0, 0.0, 0.0, 0.0, 10.0])
        got = np.asarray(laplacian_smooth_voronoi(x, tbl, deg, 1))
        expect = np.empty(_N)
        for c in range(_N):
            vals = [x[c]] + [x[n] for n in sorted(_EXPECT_NBRS[c])]
            expect[c] = 0.5 * x[c] + 0.5 * np.mean(vals)
        np.testing.assert_allclose(got, expect, rtol=1e-15)


class TestRunoffRenormOnNodeCloud:
    def test_spread_plus_renorm_conserves_area_integral(self):
        """The runoff pipeline on an unstructured node cloud = smoother
        passes (which do NOT conserve) followed by the area-conservative
        renorm — the pair must return the source total exactly, and only
        on wet nodes."""
        from scripts.run.run_omip_core2 import _area_conservative_scale

        tbl, deg = build_node_neighbor_table(_EDGES, _N)
        area = np.array([1.0, 2.0, 3.0, 4.0, 5.0])       # m^2, non-uniform
        wet = np.array([1.0, 1.0, 1.0, 1.0, 0.0])        # node 4 dry
        R = np.array([0.0, 7.0, 0.0, 0.0, 0.0])          # point discharge
        target = float((R * area * wet).sum())           # source total [kg/s]
        Rm = R.copy()
        for _ in range(3):
            Rm = np.where(wet > 0.5,
                          np.asarray(laplacian_smooth_voronoi(Rm, tbl, deg, 1)),
                          0.0)
        Rm = _area_conservative_scale(Rm, area, wet > 0.5, target)
        got = float((Rm * area * wet).sum())
        np.testing.assert_allclose(got, target, rtol=1e-14)
        assert (Rm[wet > 0.5] >= 0.0).all()
        assert Rm[4] == 0.0                               # dry node stays dry
        # the spread actually moved water off the discharge node
        assert Rm[0] > 0.0 and Rm[2] > 0.0


# ===========================================================================
# NEMO monthly-init vertical interpolation (target_depths=)
# ===========================================================================

class TestVerticalInterpKernel:
    def test_exact_on_linear_profiles(self):
        src = np.linspace(0.5, 100.0, 21)                # positive-down
        tgt = np.array([1.0, 17.3, 42.0, 99.9])
        field = 2.0 + 0.31 * src                          # linear in depth
        field = np.broadcast_to(field, (3, 21)).copy()    # 3 columns
        got = _interp_columns_to_depths(field, src, tgt)
        expect = np.broadcast_to(2.0 + 0.31 * tgt, (3, tgt.size))
        np.testing.assert_allclose(got, expect, rtol=1e-14)

    def test_end_clamp_matches_np_interp(self):
        src = np.array([10.0, 20.0, 30.0])
        tgt = np.array([1.0, 25.0, 99.0])                # out of range both ends
        col = np.array([5.0, 7.0, -1.0])
        got = _interp_columns_to_depths(col[None, :], src, tgt)[0]
        np.testing.assert_allclose(got, np.interp(tgt, src, col), rtol=1e-14)


def _write_synthetic_nemo_init(path, var, lat2d, lon2d, deptht, field):
    """(12, nlev, y, x) synthetic NEMO monthly-init file."""
    import netCDF4

    with netCDF4.Dataset(path, "w") as ds:
        nt, nz, ny, nx = field.shape
        ds.createDimension("time_counter", nt)
        ds.createDimension("deptht", nz)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        v = ds.createVariable(var, "f8",
                              ("time_counter", "deptht", "y", "x"))
        v[:] = field
        d = ds.createVariable("deptht", "f8", ("deptht",))
        d[:] = deptht
        la = ds.createVariable("nav_lat", "f8", ("y", "x"))
        la[:] = lat2d
        lo = ds.createVariable("nav_lon", "f8", ("y", "x"))
        lo[:] = lon2d


class TestMonthlyInitLoaderTargetDepths:
    NZ_SRC = 75

    @pytest.fixture()
    def synthetic_files(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")  # noqa: F841
        ny, nx = 3, 4
        lat2d, lon2d = np.meshgrid(np.array([-10.0, 0.0, 10.0]),
                                   np.array([100.0, 110.0, 120.0, 130.0]),
                                   indexing="ij")
        deptht = np.linspace(0.5, 5000.0, self.NZ_SRC)   # positive-down
        # T linear in depth, offset by latitude; S linear with other slope.
        month_off = np.arange(12.0)[:, None, None, None]
        T = (month_off + lat2d[None, None] * 0.1
             + 0.002 * deptht[None, :, None, None]
             + np.zeros((12, self.NZ_SRC, ny, nx)))
        S = 35.0 - 0.0004 * deptht[None, :, None, None] \
            + np.zeros((12, self.NZ_SRC, ny, nx))
        tp = tmp_path / "temp.nc"
        sp = tmp_path / "salt.nc"
        _write_synthetic_nemo_init(tp, "contemp", lat2d, lon2d, deptht, T)
        _write_synthetic_nemo_init(sp, "presalt", lat2d, lon2d, deptht, S)
        return str(tp), str(sp), lat2d, lon2d, deptht

    def test_interp_onto_coarser_ladder_exact_on_linear(self,
                                                        synthetic_files):
        tp, sp, lat2d, lon2d, deptht = synthetic_files
        # 1-D node-cloud target AT the source points (nearest-wet regrid is
        # then exact), coarser 10-level ladder inside the source range.
        lat_pts = lat2d.ravel()
        lon_pts = lon2d.ravel()
        tgt_depths = np.linspace(5.0, 4500.0, 10)
        month = 3
        T, S = load_nemo_monthly_init_ts(
            tp, sp, lat_pts, lon_pts, n_levels=10, month=month,
            target_depths=tgt_depths)
        assert T.shape == (lat_pts.size, 10)
        expect_T = ((month - 1) + lat_pts[:, None] * 0.1
                    + 0.002 * tgt_depths[None, :])
        expect_S = 35.0 - 0.0004 * tgt_depths[None, :] \
            + np.zeros_like(expect_T)
        np.testing.assert_allclose(T, expect_T, rtol=1e-12)
        np.testing.assert_allclose(S, expect_S, rtol=1e-12)

    def test_exact_ladder_contract_unchanged_without_target_depths(
            self, synthetic_files):
        tp, sp, lat_pts, lon_pts, _ = synthetic_files
        with pytest.raises(ValueError, match="exact-ladder"):
            load_nemo_monthly_init_ts(
                tp, sp, np.asarray(lat_pts).ravel(),
                np.asarray(lon_pts).ravel(), n_levels=10)

    def test_target_depths_must_be_positive_down(self, synthetic_files):
        tp, sp, lat2d, lon2d, _ = synthetic_files
        with pytest.raises(ValueError, match="POSITIVE-DOWN"):
            load_nemo_monthly_init_ts(
                tp, sp, lat2d.ravel(), lon2d.ravel(), n_levels=3,
                target_depths=np.array([-10.0, -5.0, -1.0]))
