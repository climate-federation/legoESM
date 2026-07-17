"""Direct unit tests for the multi-category state builders in
``legoesm.ice.state``: the grid-rank generalisation of
``init_dynamic_ice_state`` and the single->multi lift
``distribute_dynamic_state_to_categories``.

Before this, ``init_dynamic_ice_state`` hard-assumed the cubed-sphere rank-4
multi-category shape ``(6, n, n, n_cat)``: it REFUSED the lat-lon
``(n_lat, n_lon, n_cat)`` / MPAS ``(nCells, n_cat)`` shapes outright
(``len(shape) < 4``), and its ``shape[:3]`` velocity slice would have leaked
the category axis into ``u_ice`` on those grids — while ``step_sea_ice``'s
shape validation (``_base_spatial_ndim``) is grid-aware and accepts them.
That mismatch is what kept multi-category ice unreachable from the OMIP
runner (scheme-reachability audit #13).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.ice.itd import aggregate_state, category_bounds, upper_bounds
from legoesm.ice.state import (
    distribute_dynamic_state_to_categories,
    init_dynamic_ice_state,
)

from legoesm import constants


class TestInitDynamicIceStateGridRanks:
    @pytest.mark.parametrize("shape,spatial", [
        ((6, 4, 4, 5), (6, 4, 4)),   # cubed-sphere (the old-only case)
        ((8, 12, 5), (8, 12)),       # lat-lon / tripole
        ((30, 5), (30,)),            # MPAS
    ])
    def test_multicat_shapes_on_every_grid_rank(self, shape, spatial):
        st = init_dynamic_ice_state(shape, n_categories=5)
        assert st.h_ice.data.shape == shape
        assert st.h_snow.data.shape == shape
        # Velocity/stress are spatial-only: the category axis must NOT leak
        # (the old shape[:3] slice put it into u_ice on lat-lon shapes).
        assert st.u_ice.data.shape == spatial
        assert st.sigma_12.data.shape == spatial
        assert "category" in st.h_ice.dims

    def test_single_category_unchanged(self):
        st = init_dynamic_ice_state((8, 12))
        assert st.h_ice.data.shape == (8, 12)
        assert st.u_ice.data.shape == (8, 12)
        assert "category" not in st.h_ice.dims

    def test_mismatched_trailing_axis_still_raises(self):
        with pytest.raises(ValueError, match="inconsistent with"):
            init_dynamic_ice_state((8, 12, 4), n_categories=5)

    @pytest.mark.parametrize("spatial,n_cat,expected", [
        ((30,), 1, ("nCells",)),                       # MPAS single-cat
        ((30,), 5, ("nCells", "category")),            # MPAS multi-cat
        ((8, 12), 1, ("lat", "lon")),                  # lat-lon single-cat
        ((8, 12), 5, ("lat", "lon", "category")),      # lat-lon multi-cat
        ((6, 4, 4), 1, ("face", "x", "y")),            # cube single-cat
        ((6, 4, 4), 5, ("face", "x", "y", "category")),  # cube multi-cat
    ])
    def test_spatial_dim_names_match_grid_rank(self, spatial, n_cat, expected):
        # The dim NAMES must follow the spatial rank, not the cubed-sphere
        # ("face","x","y") for every grid — a (nCells, n_cat) field was
        # mislabelled with four dims before this. Velocity stays spatial-only.
        shape = spatial + ((n_cat,) if n_cat > 1 else ())
        st = init_dynamic_ice_state(shape, n_categories=n_cat)
        assert st.h_ice.dims == expected
        assert st.u_ice.dims == expected[:len(spatial)]  # no trailing category

    @pytest.mark.parametrize("n_cat", [0, -1, -3])
    def test_nonpositive_categories_refused(self, n_cat):
        with pytest.raises(ValueError, match=r"n_categories.*>= 1|positive"):
            init_dynamic_ice_state((8, 12), n_categories=n_cat)

    @pytest.mark.parametrize("shape,n_cat", [
        ((), 1),                 # rank-0 spatial (scalar) — no grid
        ((2, 3, 4, 5), 1),       # rank-4 spatial — no grid has 4 spatial dims
        ((2, 3, 4, 5, 6), 6),    # rank-4 spatial after stripping the category
    ])
    def test_unsupported_spatial_rank_refused(self, shape, n_cat):
        with pytest.raises(ValueError, match="unsupported spatial rank"):
            init_dynamic_ice_state(shape, n_categories=n_cat)


class TestDistributeDynamicStateToCategories:
    def _seeded_single_cat(self, shape=(4, 6)):
        st = init_dynamic_ice_state(shape, S_ice_init=0.0)
        h = jnp.zeros(shape).at[0, 0].set(0.3).at[1, 1].set(2.5).at[2, 2].set(9.0)
        conc = jnp.zeros(shape).at[0, 0].set(0.8).at[1, 1].set(0.9).at[2, 2].set(1.0)
        snow = jnp.zeros(shape).at[0, 0].set(0.1).at[1, 1].set(0.25)
        sal = jnp.zeros(shape).at[0, 0].set(4.0).at[1, 1].set(6.0)
        t_srf = jnp.full(shape, 255.0)
        return st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            h_snow=st.h_snow.replace(data=snow),
            S_ice=st.S_ice.replace(data=sal),
            T_ice=st.T_ice.replace(data=t_srf),
        )

    def test_aggregate_of_lift_recovers_the_input(self):
        """Round trip: aggregate_state(distribute(x)) == x for thickness,
        concentration and (with the delta seeding) temperature over icy cells
        — the conservation contract of the lift."""
        n_cat = 5
        st1 = self._seeded_single_cat()
        st = distribute_dynamic_state_to_categories(st1, n_cat)
        h_agg, t_agg, a_agg = aggregate_state(
            st.h_ice.data, st.T_ice.data, st.concentration.data)
        icy = st1.concentration.data > 0
        assert jnp.allclose(jnp.where(icy, h_agg, 0.0),
                            jnp.where(icy, st1.h_ice.data, 0.0))
        assert jnp.allclose(a_agg, st1.concentration.data)
        assert jnp.allclose(jnp.where(icy, t_agg, 0.0),
                            jnp.where(icy, st1.T_ice.data, 0.0))

    def test_each_cell_lands_in_its_thickness_bin(self):
        n_cat = 5
        st = distribute_dynamic_state_to_categories(
            self._seeded_single_cat(), n_cat)
        lo = category_bounds(n_cat)
        hi = upper_bounds(n_cat)
        h_mc = st.h_ice.data
        occupied = jnp.sum((h_mc > 0).astype(int), axis=-1)
        assert int(occupied[0, 0]) == 1 and int(occupied[1, 1]) == 1
        # 9.0 m exceeds the last upper bound -> last bin
        assert float(h_mc[2, 2, -1]) == 9.0
        # 0.3 m sits in the bin whose bounds contain it
        k = int(jnp.argmax(h_mc[0, 0] > 0))
        assert float(lo[k]) <= 0.3 < float(hi[k])

    def test_snow_and_salt_ride_in_the_occupied_bin_and_conserve(self):
        n_cat = 5
        st1 = self._seeded_single_cat()
        st = distribute_dynamic_state_to_categories(st1, n_cat)
        conc_mc = st.concentration.data
        # Aggregate snow VOLUME sum_k a_k h_snow_k == a * h_snow of the input.
        vol_in = st1.concentration.data * st1.h_snow.data
        vol_out = jnp.sum(conc_mc * st.h_snow.data, axis=-1)
        assert jnp.allclose(vol_out, vol_in)
        # Salinity sits only in the occupied bin.
        occ = (st.h_ice.data > 0) | (conc_mc > 0)
        assert jnp.all(jnp.where(occ, True, st.S_ice.data == 0.0))
        assert float(jnp.max(st.S_ice.data[1, 1])) == 6.0

    def test_ponds_zero_velocity_passthrough_dims_tagged(self):
        st1 = self._seeded_single_cat()
        u = jnp.full((4, 6), 0.07)
        st1 = st1._replace(u_ice=st1.u_ice.replace(data=u))
        st = distribute_dynamic_state_to_categories(st1, 3)
        assert jnp.all(st.pond_area.data == 0.0)
        assert jnp.all(st.pond_depth.data == 0.0)
        assert st.u_ice.data.shape == (4, 6)
        assert jnp.allclose(st.u_ice.data, 0.07)
        assert st.h_ice.dims[-1] == "category"

    def test_double_lift_refused(self):
        st = distribute_dynamic_state_to_categories(
            self._seeded_single_cat(), 5)
        with pytest.raises(ValueError, match="double-lift"):
            distribute_dynamic_state_to_categories(st, 5)

    def test_fewer_than_two_bins_refused(self):
        with pytest.raises(ValueError, match="n_categories"):
            distribute_dynamic_state_to_categories(
                self._seeded_single_cat(), 1)

    def test_icy_cells_empty_bins_keep_the_itd_placeholder_temperature(self):
        """For an ICY cell, distribute_to_categories fills the unoccupied bins
        with T_freeze_ocean; the lift must carry that through unchanged (the
        thermo ignores empty bins, but a stray 260 K default there would be a
        silent seam with itd.py).  Ice-free cells are excluded: the ITD's
        h=0-in-bin-0 convention parks the input T there, which is itd.py's
        contract, not the lift's."""
        st1 = self._seeded_single_cat()
        st = distribute_dynamic_state_to_categories(st1, 4)
        icy = (st1.concentration.data > 0)[..., jnp.newaxis]
        empty_bin = (st.h_ice.data == 0) & (st.concentration.data == 0)
        sel = icy & empty_bin
        t_out = st.T_ice.data
        assert jnp.allclose(jnp.where(sel, t_out, constants.T_freeze_ocean),
                            constants.T_freeze_ocean)


def test_distribute_conserves_pond_water():
    """Pond area/depth follow the SAME occupied-bin distribution as snow and
    salinity: pond volume (area*depth) is prognostic liquid water carrying
    mass and enthalpy — the earlier zeroing silently deleted it on any lift
    of a ponded state (restart / ponded IC / mid-run)."""
    state = init_dynamic_ice_state((3,))
    state = state._replace(
        h_ice=state.h_ice.replace(data=jnp.array([1.0, 0.5, 0.0])),
        concentration=state.concentration.replace(
            data=jnp.array([0.9, 0.4, 0.0])),
        pond_area=state.pond_area.replace(data=jnp.array([0.3, 0.2, 0.1])),
        pond_depth=state.pond_depth.replace(data=jnp.array([0.05, 0.02, 0.4])),
    )
    mc = distribute_dynamic_state_to_categories(state, n_categories=5)
    # GRID-CELL pond volume = concentration * pond_area * pond_depth (the
    # ITD-remap measure); conserved where ice exists.
    vol_in = (state.concentration.data * state.pond_area.data
              * state.pond_depth.data)
    vol_out = jnp.sum(
        mc.concentration.data * mc.pond_area.data * mc.pond_depth.data,
        axis=-1)
    assert jnp.allclose(vol_out[:2], vol_in[:2], rtol=1e-12), (
        f"pond volume lost in lift: {vol_in[:2]} -> {vol_out[:2]}")
    # ice-free cell has no occupied bin: ponds zero there (nothing to sit on)
    assert float(jnp.sum(mc.pond_area.data[2])) == 0.0
