"""The virtual-salt closure must dilute TEMPERATURE by the rain/evaporation/
restoring water whose heat content the surface heat flux already carries
(NEMO linear free surface: trasbc adds emp*sst back).  Without it the
production closure carried a spurious -(E-P) cp SST surface flux of a few
W/m2 (codex + GLM, 2026-09-05).  Runoff and ice melt water have no heat-flux
term in this coupling and stay out of the twin.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.freshwater import FreshwaterForcing, virtual_closure_temperature_twin


def _fw(**kw):
    d = dict(precip=0.0, evap=0.0, runoff=0.0, ice_fw=0.0)
    d.update(kw)
    return FreshwaterForcing(**{k: np.array([v]) for k, v in d.items()})


def test_twin_channels_and_sign():
    rho0, T, h = 1025.0, np.array([20.0]), np.array([1.0])
    # evaporation removes water; the heat flux already removed its heat at
    # SST, so the twin WARMS the top cell (concentration of T)
    assert float(virtual_closure_temperature_twin(_fw(evap=2e-4), T, h, rho0)[0]) == pytest.approx(2e-4 / rho0 * 20.0, rel=1e-12)
    # rain cools it (rain heat at air temperature lives in the heat flux)
    assert float(virtual_closure_temperature_twin(_fw(precip=2e-4), T, h, rho0)[0]) == pytest.approx(-2e-4 / rho0 * 20.0, rel=1e-12)
    # runoff and ice melt: excluded
    assert float(virtual_closure_temperature_twin(_fw(runoff=2e-4, ice_fw=2e-4), T, h, rho0)[0]) == 0.0
    # restoring water: included (sbcssr subtracts its heat content)
    fw = FreshwaterForcing(precip=np.zeros(1), evap=np.zeros(1), runoff=np.zeros(1),
                           ice_fw=np.zeros(1), restoring=np.array([1e-4]))
    assert float(virtual_closure_temperature_twin(fw, T, h, rho0)[0]) == pytest.approx(-1e-4 / rho0 * 20.0, rel=1e-12)
    # land masked
    assert float(virtual_closure_temperature_twin(_fw(evap=2e-4), T, h, rho0, mask=np.zeros(1))[0]) == 0.0


class TestVirtualStep:
    @staticmethod
    def _run(F_kg=3.0e-5, dt=300.0):
        import jax.numpy as jnp
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())
        from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import compute_layer_thickness, create_ocean_z_star

        g = create_beta_plane_cgrid_geometry(8, 8, dx_m=50e3, f0=1e-4, beta=0.0)
        z = create_ocean_z_star(3, H_max=300.0)
        st = rest_state_latlon_cgrid_ocean(g, z, land_lat_threshold=90.0)
        st = st._replace(T=st.T.replace(data=jnp.full_like(st.T.data, 10.0)),
                         S=st.S.replace(data=jnp.full_like(st.S.data, 35.0)))
        cfg = LatLonCGridOceanConfig.from_flat(
            freshwater_closure="virtual_salt_flux", normalize_freshwater=False,
            use_conservation_fixer=False, fix_salt=False, fix_volume=False,
            fix_eta_drift=True, enable_runtime_checks=False, n_barotropic_substeps=20)
        model = LatLonCGridOceanModel(g, z, cfg)
        shp = st.eta.data.shape
        fw = FreshwaterForcing(precip=jnp.full(shp, F_kg), evap=jnp.zeros(shp),
                               runoff=jnp.zeros(shp), ice_fw=jnp.zeros(shp))
        h0 = compute_layer_thickness(st.eta.data, st.H_bathy.data, z,
                                     min_water_column_m=cfg.min_water_column_m)
        s1 = model.step(st, dt, freshwater=fw)
        return np.asarray(st.T.data), np.asarray(s1.T.data), np.asarray(h0), cfg.rho_0

    def test_rain_dilutes_top_cell_temperature_under_the_virtual_closure(self):
        T0, T1, h0, rho0 = self._run()
        F_dt = 3.0e-5 * 300.0 / rho0
        sl = (slice(1, -1), slice(1, -1))
        # twin (-T F dt/h1) plus the virtual closure's uniform z-star stretch
        # of the whole column (-T F dt/H, the volume goes to eta), which NEMO's
        # linear free surface does not have: the known small residual of the
        # virtual closure (3% here, ~1e-3 in production).
        H = h0[sl].sum(-1)
        expected = -T0[sl][..., 0] * F_dt * (1.0 / h0[sl][..., 0] + 1.0 / H)
        got = T1[sl][..., 0] - T0[sl][..., 0]
        # rtol=0: numpy's default relative tolerance would hide a 1e-7 K signal
        assert np.allclose(got, expected, rtol=0.0, atol=1e-3 * abs(expected).max())
        assert np.all(got < 0.0)
        # layers below: only the uniform stretch
        below = -T0[sl][..., 1:] * (F_dt / H)[..., None]
        assert np.allclose(T1[sl][..., 1:] - T0[sl][..., 1:], below, rtol=0.0, atol=1e-3 * abs(below).max())
