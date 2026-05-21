"""Unit tests for driver-side runoff + ice-shelf apply helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.coupler.runoff_apply import apply_runoff_step
from legoesm.ocean.coupler.ice_shelf_apply import apply_ice_shelf_basal_step
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
        # Single cavity cell at (1, 1) with 500 m draft.
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
        # Top-layer S at cavity cell dilutes; T cools.
        S_arr = np.asarray(new.S.data)
        T_arr = np.asarray(new.T.data)
        assert S_arr[1, 1, 0] < 34.7
        assert T_arr[1, 1, 0] < 0.0

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

    def test_only_top_layer_modified(self):
        st = _FakeState(T_init=0.0, S_init=34.7, nlev=3)
        mask = np.zeros((4, 4))
        mask[1, 1] = 1.0
        draft = np.full((4, 4), 500.0)
        new, _ = apply_ice_shelf_basal_step(
            st,
            ice_shelf_mask=mask, ice_draft_m=draft,
            z_coord=_FakeZCoord(), dt=3600.0,
            config=IceShelfConfig(enabled=True, scheme="three_equation"),
        )
        S_new = np.asarray(new.S.data)
        T_new = np.asarray(new.T.data)
        # Layers 1 + 2 untouched at cavity cell.
        assert S_new[1, 1, 1] == 34.7
        assert S_new[1, 1, 2] == 34.7
        assert T_new[1, 1, 1] == 0.0

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
