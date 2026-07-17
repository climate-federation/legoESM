"""Unit tests for driver-side runoff + ice-shelf apply helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.coupler.runoff_apply import (
    apply_runoff_step,
    apply_runoff_step_mpas,
)
from legoesm.ocean.coupler.ice_shelf_apply import (
    _ice_base_layer_index,
    apply_ice_shelf_basal_step,
    apply_ice_shelf_basal_step_mpas,
)
from legoesm.ocean.physics.ice_shelf import IceShelfConfig


class _FakeZCoord:
    dz_ref = np.array([10.0, 50.0, 100.0])


class _FakeState:
    """LatLonCGridOceanState stub with S, T, eta, land_mask only."""

    def __init__(
        self, n_lat=4, n_lon=4, nlev=3,
        T_init=5.0, S_init=34.7, mask_init=1.0,
    ):
        self.S = Field(
            jnp.full((n_lat, n_lon, nlev), S_init, dtype=jnp.float64),
            name="S", dims=("lat", "lon", "lev"), units="PSU",
        )
        self.T = Field(
            jnp.full((n_lat, n_lon, nlev), T_init, dtype=jnp.float64),
            name="T", dims=("lat", "lon", "lev"), units="degC",
        )
        self.eta = Field(
            jnp.zeros((n_lat, n_lon), dtype=jnp.float64),
            name="eta", dims=("lat", "lon"), units="m",
        )
        self.land_mask = Field(
            jnp.full((n_lat, n_lon), mask_init, dtype=jnp.float64),
            name="land_mask", dims=("lat", "lon"), units="1",
        )

    def _replace(self, **kw):
        new = _FakeState.__new__(_FakeState)
        for attr in ("S", "T", "eta", "land_mask"):
            setattr(new, attr, getattr(self, attr))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


# ==============================================================================
# _ice_base_layer_index
# ==============================================================================

class TestIceBaseLayerIndex:

    def test_zero_draft_picks_layer_0(self):
        dz = np.array([10.0, 50.0, 100.0])
        assert int(_ice_base_layer_index(np.array(0.0), dz)) == 0

    def test_shallow_within_top_layer(self):
        # 5 m draft within layer 0 (0–10 m).
        dz = np.array([10.0, 50.0, 100.0])
        assert int(_ice_base_layer_index(np.array(5.0), dz)) == 0

    def test_mid_layer(self):
        # 30 m draft → between interfaces 10 + 60 → layer 1.
        dz = np.array([10.0, 50.0, 100.0])
        assert int(_ice_base_layer_index(np.array(30.0), dz)) == 1

    def test_at_interface_picks_lower_layer(self):
        # Exactly at interface 60 m → layer 2 (side='right').
        dz = np.array([10.0, 50.0, 100.0])
        assert int(_ice_base_layer_index(np.array(60.0), dz)) == 2

    def test_deeper_than_column_clips_to_deepest(self):
        dz = np.array([10.0, 50.0, 100.0])
        assert int(_ice_base_layer_index(np.array(9999.0), dz)) == 2

    def test_2d_broadcast(self):
        dz = np.array([10.0, 50.0, 100.0])
        drafts = np.array([[0.0, 5.0, 30.0], [70.0, 200.0, 9.9]])
        k = _ice_base_layer_index(drafts, dz)
        assert k.shape == drafts.shape
        np.testing.assert_array_equal(k, np.array([[0, 0, 1], [2, 2, 0]]))

    def test_1d_dz_required(self):
        with pytest.raises(ValueError):
            _ice_base_layer_index(np.array(5.0), np.array([[10.0, 50.0]]))


# ==============================================================================
# Runoff apply
# ==============================================================================

class TestApplyRunoffStep:

    def test_zero_runoff_no_change(self):
        st = _FakeState()
        R = np.zeros((4, 4))
        new = apply_runoff_step(st, R_kg_m2_s=R, z_coord=_FakeZCoord(),
                                dt=3600.0)
        assert np.allclose(np.asarray(new.eta.data), 0.0)
        assert np.allclose(
            np.asarray(new.S.data), np.asarray(st.S.data),
        )

    def test_positive_runoff_raises_eta(self):
        st = _FakeState()
        R = np.full((4, 4), 1.0e-3)   # 1 mm/s into ocean
        new = apply_runoff_step(st, R_kg_m2_s=R, z_coord=_FakeZCoord(),
                                dt=3600.0)
        # eta increases by R / rho_0 · dt.
        rho_0 = 1025.0
        expected = 1.0e-3 / rho_0 * 3600.0
        assert np.allclose(np.asarray(new.eta.data), expected, rtol=1e-3)

    def test_positive_runoff_dilutes_top_salinity(self):
        st = _FakeState(S_init=34.7)
        R = np.full((4, 4), 1.0e-3)
        new = apply_runoff_step(st, R_kg_m2_s=R, z_coord=_FakeZCoord(),
                                dt=3600.0)
        S_new_top = np.asarray(new.S.data)[..., 0]
        assert np.all(S_new_top < 34.7)

    def test_deeper_layers_untouched(self):
        st = _FakeState(S_init=34.7, nlev=3)
        R = np.full((4, 4), 1.0e-3)
        new = apply_runoff_step(st, R_kg_m2_s=R, z_coord=_FakeZCoord(),
                                dt=3600.0)
        S_new = np.asarray(new.S.data)
        assert np.allclose(S_new[..., 1], 34.7)
        assert np.allclose(S_new[..., 2], 34.7)

    def test_land_cells_untouched(self):
        st = _FakeState()
        mask = np.ones((4, 4), dtype=np.float64)
        mask[0, 0] = 0.0
        st.land_mask = Field(
            jnp.asarray(mask), name=st.land_mask.name,
            dims=st.land_mask.dims, units=st.land_mask.units,
        )
        R = np.full((4, 4), 1.0e-3)
        new = apply_runoff_step(st, R_kg_m2_s=R, z_coord=_FakeZCoord(),
                                dt=3600.0)
        # Land cell preserves S, eta.
        assert np.asarray(new.S.data)[0, 0, 0] == 34.7
        assert np.asarray(new.eta.data)[0, 0] == 0.0
        # Ocean cells modified.
        assert np.asarray(new.S.data)[1, 1, 0] < 34.7
        assert np.asarray(new.eta.data)[1, 1] > 0.0


class _FakeMPASRunoffState:
    """MPAS ocean-state stub for runoff: S (nCells, nlev), eta / land_mask
    (nCells,) -- the 1-D-spatial shapes apply_runoff_step_mpas requires."""

    def __init__(self, n_cells=4, nlev=3, S_init=34.7, mask_init=1.0):
        self.S = Field(
            jnp.full((n_cells, nlev), S_init, dtype=jnp.float64),
            name="S", dims=("nCells", "lev"), units="PSU",
        )
        self.eta = Field(
            jnp.zeros((n_cells,), dtype=jnp.float64),
            name="eta", dims=("nCells",), units="m",
        )
        self.land_mask = Field(
            jnp.full((n_cells,), mask_init, dtype=jnp.float64),
            name="land_mask", dims=("nCells",), units="1",
        )

    def _replace(self, **kw):
        new = _FakeMPASRunoffState.__new__(_FakeMPASRunoffState)
        for attr in ("S", "eta", "land_mask"):
            setattr(new, attr, getattr(self, attr))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


class TestApplyRunoffStepMPAS:
    """apply_runoff_step_mpas was implemented but UNTESTED and unreachable --
    run_centennial_spinup warned '--runoff not yet supported on grid=mpas'.
    Same virtual-salt + eta-rise convention as the lat-lon apply, on (nCells,).
    """

    def test_zero_runoff_no_change(self):
        st = _FakeMPASRunoffState()
        new = apply_runoff_step_mpas(
            st, R_kg_m2_s=np.zeros(4), z_coord=_FakeZCoord(), dt=3600.0)
        assert np.allclose(np.asarray(new.eta.data), 0.0)
        assert np.allclose(np.asarray(new.S.data), np.asarray(st.S.data))

    def test_positive_runoff_raises_eta(self):
        st = _FakeMPASRunoffState()
        new = apply_runoff_step_mpas(
            st, R_kg_m2_s=np.full(4, 1.0e-3), z_coord=_FakeZCoord(), dt=3600.0)
        rho_0 = 1025.0
        expected = 1.0e-3 / rho_0 * 3600.0
        assert np.allclose(np.asarray(new.eta.data), expected, rtol=1e-3)

    def test_positive_runoff_dilutes_top_only(self):
        st = _FakeMPASRunoffState(S_init=34.7, nlev=3)
        new = apply_runoff_step_mpas(
            st, R_kg_m2_s=np.full(4, 1.0e-3), z_coord=_FakeZCoord(), dt=3600.0)
        S_new = np.asarray(new.S.data)
        assert np.all(S_new[..., 0] < 34.7)      # top diluted
        assert np.allclose(S_new[..., 1], 34.7)  # deeper untouched
        assert np.allclose(S_new[..., 2], 34.7)

    def test_land_cells_untouched(self):
        st = _FakeMPASRunoffState()
        mask = np.array([0.0, 1.0, 1.0, 1.0])
        st = st._replace(land_mask=Field(
            jnp.asarray(mask), name="land_mask", dims=("nCells",), units="1"))
        new = apply_runoff_step_mpas(
            st, R_kg_m2_s=np.full(4, 1.0e-3), z_coord=_FakeZCoord(), dt=3600.0)
        assert np.asarray(new.S.data)[0, 0] == 34.7   # land cell preserved
        assert np.asarray(new.eta.data)[0] == 0.0
        assert np.asarray(new.S.data)[1, 0] < 34.7    # ocean cell modified
        assert np.asarray(new.eta.data)[1] > 0.0


# ==============================================================================
# Ice-shelf apply
# ==============================================================================

class TestApplyIceShelfBasalStep:

    def test_disabled_is_noop(self):
        st = _FakeState()
        mask = np.zeros((4, 4))
        draft = np.full((4, 4), 500.0)
        new, diag = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=False),
        )
        assert new is st
        assert np.all(np.asarray(diag["m_dot_m_s"]) == 0.0)

    def test_warm_cavity_melts_and_freshens(self):
        st = _FakeState(T_init=0.0, S_init=34.7)
        # Single cavity cell at (1, 1) with 500 m draft.  dz_ref =
        # [10, 50, 100] → z_iface = [0, 10, 60, 160].  500 m > 160
        # → draft layer clipped to deepest (k=2).
        mask = np.zeros((4, 4))
        mask[1, 1] = 1.0
        draft = np.full((4, 4), 500.0)
        new, diag = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        m_dot = np.asarray(diag["m_dot_m_s"])
        # Cavity cell melts.
        assert m_dot[1, 1] > 0.0
        # Non-cavity cells unchanged.
        assert np.all(m_dot[0, :] == 0.0)
        assert np.all(m_dot[2:, :] == 0.0)
        # Draft-layer (k=2) S dilutes + T cools at cavity cell.
        S_arr = np.asarray(new.S.data)
        T_arr = np.asarray(new.T.data)
        assert S_arr[1, 1, 2] < 34.7
        assert T_arr[1, 1, 2] < 0.0
        # Top + middle layers untouched.
        assert S_arr[1, 1, 0] == 34.7
        assert S_arr[1, 1, 1] == 34.7
        assert T_arr[1, 1, 0] == 0.0
        assert T_arr[1, 1, 1] == 0.0
        # Diagnostic reports the layer index.
        assert int(diag["ice_base_layer_idx"][1, 1]) == 2

    def test_basal_melt_raises_eta(self):
        """Meltwater entering the ocean must raise the free surface
        by exactly ``fw / rho_0 * dt`` at the cavity cell.  Guards
        against missing the ``eta`` update in the state replace."""
        st = _FakeState(T_init=0.0, S_init=34.7)
        mask = np.zeros((4, 4))
        mask[1, 1] = 1.0
        draft = np.full((4, 4), 500.0)
        new, diag = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        fw = np.asarray(diag["freshwater_kg_m2_s"])
        eta_new = np.asarray(new.eta.data)
        # eta_new should equal fw / rho_0 * dt at the cavity cell.
        rho_0 = 1025.0
        dt = 3600.0
        expected = fw[1, 1] / rho_0 * dt
        assert eta_new[1, 1] == pytest.approx(expected, rel=1e-6)
        # Non-cavity cells stay at zero.
        assert eta_new[0, 0] == 0.0

    def test_only_draft_layer_modified(self):
        st = _FakeState(T_init=0.0, S_init=34.7, nlev=3)
        mask = np.zeros((4, 4))
        mask[1, 1] = 1.0
        # Draft 30 m → z_iface=[0,10,60,160], 30 > 10 but < 60 → k=1.
        draft = np.full((4, 4), 30.0)
        new, diag = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        S_new = np.asarray(new.S.data)
        T_new = np.asarray(new.T.data)
        # Mid-layer (k=1) modified; top + deepest untouched.
        assert S_new[1, 1, 1] < 34.7
        assert T_new[1, 1, 1] < 0.0
        assert S_new[1, 1, 0] == 34.7
        assert S_new[1, 1, 2] == 34.7
        assert T_new[1, 1, 0] == 0.0
        assert T_new[1, 1, 2] == 0.0
        assert int(diag["ice_base_layer_idx"][1, 1]) == 1

    def test_shallow_draft_picks_top_layer(self):
        """Shallow shelf (draft < dz_ref[0]=10 m) → layer 0."""
        st = _FakeState(T_init=0.0, S_init=34.7, nlev=3)
        mask = np.zeros((4, 4))
        mask[1, 1] = 1.0
        draft = np.full((4, 4), 5.0)
        new, diag = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        S_new = np.asarray(new.S.data)
        T_new = np.asarray(new.T.data)
        assert S_new[1, 1, 0] < 34.7
        assert T_new[1, 1, 0] < 0.0
        assert S_new[1, 1, 1] == 34.7
        assert S_new[1, 1, 2] == 34.7
        assert int(diag["ice_base_layer_idx"][1, 1]) == 0

    def test_land_cells_skipped(self):
        st = _FakeState(T_init=0.0)
        mask_l = np.ones((4, 4))
        mask_l[0, 0] = 0.0
        st.land_mask = Field(
            jnp.asarray(mask_l), name=st.land_mask.name,
            dims=st.land_mask.dims, units=st.land_mask.units,
        )
        # Cavity request includes a land cell.
        cavity_mask = np.zeros((4, 4))
        cavity_mask[0, 0] = 1.0   # land + cavity → skip
        cavity_mask[1, 1] = 1.0   # ocean + cavity → apply
        draft = np.full((4, 4), 500.0)
        new, diag = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=cavity_mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True),
        )
        m = np.asarray(diag["m_dot_m_s"])
        # Land cavity cell skipped.
        assert m[0, 0] == 0.0
        # Ocean cavity cell melts.
        assert m[1, 1] > 0.0


# ==============================================================================
# MPAS ice-shelf apply
# ==============================================================================

class _FakeMPASState:
    """MPAS Voronoi state stub: 1-D cell axis."""

    def __init__(self, n_cells=8, nlev=3, T_init=0.0, S_init=34.7,
                 mask_init=1.0):
        self.S = Field(
            jnp.full((n_cells, nlev), S_init, dtype=jnp.float64),
            name="S", dims=("nCells", "lev"), units="PSU",
        )
        self.T = Field(
            jnp.full((n_cells, nlev), T_init, dtype=jnp.float64),
            name="T", dims=("nCells", "lev"), units="degC",
        )
        self.eta = Field(
            jnp.zeros((n_cells,), dtype=jnp.float64),
            name="eta", dims=("nCells",), units="m",
        )
        self.land_mask = Field(
            jnp.full((n_cells,), mask_init, dtype=jnp.float64),
            name="land_mask", dims=("nCells",), units="1",
        )

    def _replace(self, **kw):
        new = _FakeMPASState.__new__(_FakeMPASState)
        for attr in ("S", "T", "eta", "land_mask"):
            setattr(new, attr, getattr(self, attr))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


class TestApplyIceShelfBasalStepMPAS:

    def test_disabled_is_noop(self):
        st = _FakeMPASState()
        mask = np.zeros(8)
        draft = np.full(8, 500.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=False),
        )
        assert new is st
        assert np.all(np.asarray(diag["m_dot_m_s"]) == 0.0)

    def test_warm_cavity_melts_and_freshens(self):
        st = _FakeMPASState(T_init=0.0, S_init=34.7)
        mask = np.zeros(8)
        mask[3] = 1.0
        # Draft 500 m → deepest layer k=2 (dz_ref=[10,50,100],
        # z_iface=[0,10,60,160], 500 > 160 → clipped to nlev-1).
        draft = np.full(8, 500.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        m_dot = np.asarray(diag["m_dot_m_s"])
        assert m_dot[3] > 0.0
        assert np.all(np.delete(m_dot, 3) == 0.0)
        S_arr = np.asarray(new.S.data)
        T_arr = np.asarray(new.T.data)
        # Draft layer (k=2) modified; top layer untouched.
        assert S_arr[3, 2] < 34.7
        assert T_arr[3, 2] < 0.0
        assert S_arr[3, 0] == 34.7
        assert T_arr[3, 0] == 0.0
        assert int(diag["ice_base_layer_idx"][3]) == 2

    def test_basal_melt_raises_eta(self):
        st = _FakeMPASState(T_init=0.0, S_init=34.7)
        mask = np.zeros(8)
        mask[3] = 1.0
        draft = np.full(8, 500.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        fw = np.asarray(diag["freshwater_kg_m2_s"])
        eta_new = np.asarray(new.eta.data)
        rho_0 = 1025.0
        dt = 3600.0
        expected = fw[3] / rho_0 * dt
        assert eta_new[3] == pytest.approx(expected, rel=1e-6)
        assert eta_new[0] == 0.0

    def test_only_draft_layer_modified(self):
        st = _FakeMPASState(T_init=0.0, S_init=34.7, nlev=3)
        mask = np.zeros(8)
        mask[3] = 1.0
        # Draft 30 m → k=1 (10 < 30 < 60).
        draft = np.full(8, 30.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        S_new = np.asarray(new.S.data)
        T_new = np.asarray(new.T.data)
        # Mid-layer modified; top + deepest untouched.
        assert S_new[3, 1] < 34.7
        assert T_new[3, 1] < 0.0
        assert S_new[3, 0] == 34.7
        assert S_new[3, 2] == 34.7
        assert T_new[3, 0] == 0.0
        assert T_new[3, 2] == 0.0
        assert int(diag["ice_base_layer_idx"][3]) == 1

    def test_shallow_draft_picks_top_layer_mpas(self):
        st = _FakeMPASState(T_init=0.0, S_init=34.7, nlev=3)
        mask = np.zeros(8)
        mask[3] = 1.0
        draft = np.full(8, 5.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        S_new = np.asarray(new.S.data)
        T_new = np.asarray(new.T.data)
        assert S_new[3, 0] < 34.7
        assert T_new[3, 0] < 0.0
        assert S_new[3, 1] == 34.7
        assert S_new[3, 2] == 34.7
        assert int(diag["ice_base_layer_idx"][3]) == 0

    def test_land_cells_skipped(self):
        st = _FakeMPASState(T_init=0.0)
        mask_l = np.ones(8)
        mask_l[0] = 0.0
        st.land_mask = Field(
            jnp.asarray(mask_l), name=st.land_mask.name,
            dims=st.land_mask.dims, units=st.land_mask.units,
        )
        cavity_mask = np.zeros(8)
        cavity_mask[0] = 1.0  # land + cavity → skip
        cavity_mask[3] = 1.0  # ocean + cavity → apply
        draft = np.full(8, 500.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=cavity_mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True),
        )
        m = np.asarray(diag["m_dot_m_s"])
        assert m[0] == 0.0
        assert m[3] > 0.0

    def test_shape_mismatch_raises(self):
        st = _FakeMPASState(n_cells=8)
        mask = np.zeros(7)  # wrong size
        draft = np.zeros(7)
        with pytest.raises(ValueError):
            apply_ice_shelf_basal_step_mpas(
                st,
                ice_shelf_mask=mask, ice_draft_m=draft,
                z_coord=_FakeZCoord(), dt=3600.0,
                config=IceShelfConfig(enabled=True),
            )

    def test_draft_shape_mismatch_raises(self):
        st = _FakeMPASState(n_cells=8)
        mask = np.zeros(8)
        draft = np.zeros(7)  # mismatched draft
        with pytest.raises(ValueError):
            apply_ice_shelf_basal_step_mpas(
                st,
                ice_shelf_mask=mask, ice_draft_m=draft,
                z_coord=_FakeZCoord(), dt=3600.0,
                config=IceShelfConfig(enabled=True),
            )

    def test_2d_mask_raises(self):
        st = _FakeMPASState(n_cells=8)
        mask = np.zeros((2, 4))
        draft = np.zeros((2, 4))
        with pytest.raises(ValueError):
            apply_ice_shelf_basal_step_mpas(
                st,
                ice_shelf_mask=mask, ice_draft_m=draft,
                z_coord=_FakeZCoord(), dt=3600.0,
                config=IceShelfConfig(enabled=True),
            )

    def test_linear_scheme(self):
        """Mirror of the lat-lon ``scheme='linear'`` path on Voronoi cells:
        positive melt + freshwater, draft-layer-only T/S response."""
        st = _FakeMPASState(T_init=1.0, S_init=34.7, nlev=3)
        mask = np.zeros(8)
        mask[4] = 1.0
        # 500 m draft → k=2 (deepest layer).
        draft = np.full(8, 500.0)
        new, diag = apply_ice_shelf_basal_step_mpas(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="linear"),
        )
        m_dot = np.asarray(diag["m_dot_m_s"])
        fw = np.asarray(diag["freshwater_kg_m2_s"])
        assert m_dot[4] > 0.0
        assert fw[4] > 0.0
        assert np.all(np.delete(m_dot, 4) == 0.0)
        S_new = np.asarray(new.S.data)
        T_new = np.asarray(new.T.data)
        # Deepest layer modified; top + middle untouched.
        assert S_new[4, 2] < 34.7
        assert T_new[4, 2] < 1.0
        assert S_new[4, 0] == 34.7
        assert S_new[4, 1] == 34.7
        assert T_new[4, 0] == 1.0
        assert T_new[4, 1] == 1.0
        assert int(diag["ice_base_layer_idx"][4]) == 2
