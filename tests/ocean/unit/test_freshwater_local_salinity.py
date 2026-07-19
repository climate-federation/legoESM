"""Local-salinity virtual-salt closure (NEMO tra_sbc: sfx = emp * sss).

2026-07-18 Arctic halocline-erosion audit: the fixed S_ref=35 virtual-salt
closure over-salinifies ice growth on fresh shelves (Siberian ~27 PSU) by
~(35-4)/(27-4) and over-dilutes rivers.  ``freshwater_salinity="local"``
threads the LOCAL top-cell salinity through the SAME closure functions
(broadcasting; no new numerics).  These tests pin:

* array-S == scalar-S bit-equivalence when the array is constant at S_ref;
* the local/s_ref tendency ratio equals S_local/S_ref;
* column-integral conservation of the runoff-spread closure with array S;
* dispatch hardening: unknown ``freshwater_salinity`` raises in the lat-lon
  model constructor and in the MPAS tendency gate.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    runoff_spread_virtual_salt_tendency_3d,
    virtual_salt_flux,
    virtual_salt_flux_from_net,
)

jax.config.update("jax_enable_x64", True)  # tolerances below are f64-scaled


def _fw(shape, precip=2.0e-5, evap=1.0e-5, runoff=3.0e-5):
    z = jnp.zeros(shape)
    return FreshwaterForcing(
        precip=z + precip, evap=z + evap, runoff=z + runoff, ice_fw=z,
    )


class TestArrayVsScalarSRef:
    def test_from_net_bit_equal_when_array_constant(self):
        F = jnp.asarray(np.random.default_rng(0).normal(0, 1e-5, (4, 5)))
        dz0 = jnp.full((4, 5), 2.0)
        a = virtual_salt_flux_from_net(F, 35.0, dz0, 1026.0)
        b = virtual_salt_flux_from_net(F, jnp.full((4, 5), 35.0), dz0, 1026.0)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_virtual_salt_flux_local_ratio(self):
        fw = _fw((3, 3))
        dz0 = jnp.full((3, 3), 1.5)
        S_local = jnp.full((3, 3), 27.0)
        d_ref = virtual_salt_flux(fw, 35.0, dz0, 1026.0)
        d_loc = virtual_salt_flux(fw, S_local, dz0, 1026.0)
        np.testing.assert_allclose(
            np.asarray(d_loc), np.asarray(d_ref) * (27.0 / 35.0), rtol=1e-12)

    def test_spatially_varying_local_S(self):
        # Each cell's tendency scales with ITS salinity (the whole point).
        fw = _fw((2, 2))
        dz0 = jnp.full((2, 2), 1.0)
        S_local = jnp.asarray([[27.0, 35.0], [30.0, 34.0]])
        d = virtual_salt_flux(fw, S_local, dz0, 1026.0)
        d_ref = virtual_salt_flux(fw, 1.0, dz0, 1026.0)
        np.testing.assert_allclose(
            np.asarray(d), np.asarray(d_ref) * np.asarray(S_local), rtol=1e-12)


class TestRunoffSpreadArrayS:
    def test_column_integral_conservation_with_local_S(self):
        # sum_k dS[k]*h_k == -S_local*(F_top+R)/rho_0 exactly (the documented
        # conservation identity, now per-cell with the local S).
        ny, nx, nlev = 3, 4, 6
        rng = np.random.default_rng(1)
        h_k = jnp.asarray(rng.uniform(5.0, 40.0, (ny, nx, nlev)))
        mask = jnp.ones((ny, nx))
        fw = _fw((ny, nx))
        S_local = jnp.asarray(rng.uniform(25.0, 35.0, (ny, nx)))
        dS = runoff_spread_virtual_salt_tendency_3d(
            fw, S_local, h_k, 1026.0, mask, runoff_spread_m=60.0)
        col = np.asarray(jnp.sum(dS * h_k, axis=-1))
        F_top = np.asarray(fw.precip - fw.evap + fw.ice_fw)
        R = np.asarray(fw.runoff)
        expected = -np.asarray(S_local) * (F_top + R) / 1026.0
        np.testing.assert_allclose(col, expected, rtol=1e-10)


class TestSignConvention:
    def test_freezing_salinifies_runoff_dilutes_local(self):
        # ABSOLUTE sign pin (a shared sign flip would slip past ratio tests):
        # net freshwater REMOVAL (evap/freezing, F<0) must SALINIFY (dS/dt>0),
        # net freshwater INPUT (runoff, F>0) must DILUTE (dS/dt<0) — with the
        # LOCAL S array exactly as with the scalar.
        dz0 = jnp.full((2, 2), 2.0)
        S_local = jnp.full((2, 2), 27.0)
        d_freeze = virtual_salt_flux_from_net(
            jnp.full((2, 2), -1.0e-5), S_local, dz0, 1026.0)
        d_runoff = virtual_salt_flux_from_net(
            jnp.full((2, 2), +1.0e-5), S_local, dz0, 1026.0)
        assert float(jnp.min(d_freeze)) > 0.0
        assert float(jnp.max(d_runoff)) < 0.0


class TestDispatchHardening:
    def test_latlon_config_rejects_unknown(self):
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        cfg = LatLonCGridOceanConfig(freshwater_salinity="sss_typo")
        with pytest.raises(ValueError, match="freshwater_salinity"):
            LatLonCGridOceanModel._validate_config(cfg)

    def test_latlon_rejects_local_plus_normalize(self):
        # local-S breaks the zero-global-salt promise of the normalization
        # (∫S_local·F' dA covariance) — the combination must raise, not
        # silently drift the salt budget (codex HIGH 2026-07-18).
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        cfg = LatLonCGridOceanConfig(freshwater_salinity="local",
                                     normalize_freshwater=True)
        with pytest.raises(ValueError, match="normalize_freshwater"):
            LatLonCGridOceanModel._validate_config(cfg)
        # each alone is fine
        LatLonCGridOceanModel._validate_config(
            LatLonCGridOceanConfig(freshwater_salinity="local"))
        LatLonCGridOceanModel._validate_config(
            LatLonCGridOceanConfig(normalize_freshwater=True))

    def test_mpas_gate_source_tripwire(self):
        # The MPAS raise lives inside mpas_ocean_baroclinic_tendencies (a
        # full Voronoi mesh is integration-suite territory).  Source-level
        # tripwire in the dispatch-hardening style: the gate, its
        # normalize-incompatibility branch, and the local-S selection must
        # all be present — deleting any of them goes red here even though
        # the numeric path is not executed.
        import inspect
        from legoesm.ocean.dynamics import ocean_pe_mpas
        src = inspect.getsource(ocean_pe_mpas.mpas_ocean_baroclinic_tendencies)
        assert 'getattr(config, "freshwater_salinity", "s_ref")' in src
        assert "incompatible with" in src            # normalize+local raise
        assert "freshwater_salinity must be" in src  # unknown-selection raise
        assert "S_3d[:, 0]" in src                   # local top-cell S source


def test_runner_prescribed_sic_handoff_tripwire():
    # The prescribed-ice SIC handoff (codex HIGH round-1) lives in the OMIP
    # host loop; source tripwire so removing the sf._replace silently cannot
    # pass (the blend-branch attach alone must NOT satisfy this — match the
    # prescribed-branch guard condition specifically).
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    src = (root / "scripts" / "run" / "run_omip_core2.py").read_text()
    assert "if _sic is not None and ice_config is None:" in src
    assert src.count("sf = sf._replace(ice_concentration=") >= 2  # both branches


def test_config_default_is_bit_compat():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.mpas_config import MPASOceanConfig
    assert LatLonCGridOceanConfig().freshwater_salinity == "s_ref"
    assert MPASOceanConfig().freshwater_salinity == "s_ref"
