"""atm_owns_surface + WOA 3-D ocean: one land/sea partition.

Driver methods exercised on a bare instance (no setup, no WOA file): the
geometry cache is per grid, ocean-dry cells become land, and an ocean-wet cell
the atmosphere has as full land is refused.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.ocean.init_latlon_cgrid as icg
import legoesm.ocean.init_woa as iw
from legoesm.driver.coupled_esm_driver import CoupledESMDriver


def _bare(wet):
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = SimpleNamespace(woa_t_path="t.nc", woa_s_path="s.nc",
                                    ocean_nlev=4, ocean_H_max_m=4000.0)
    d._woa_geometry = lambda grid: (None, None, wet)
    return d


def test_dry_cells_become_land_and_wet_cells_keep_the_mask():
    wet = np.array([[1.0, 0.0], [1.0, 0.0]])
    f = jnp.array([0.0, 0.3, 0.5, 0.0])
    atm = SimpleNamespace(_latlon_native_grid=object(), grid=None)
    out = _bare(wet)._owned_land_fraction_hook(atm, f)
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.5, 1.0])


def test_ocean_wet_full_land_cell_is_refused():
    wet = np.array([[1.0, 1.0]])
    atm = SimpleNamespace(_latlon_native_grid=object(), grid=None)
    with pytest.raises(ValueError, match="wet for the 3-D ocean but full land"):
        _bare(wet)._owned_land_fraction_hook(atm, jnp.array([0.2, 1.0]))


def test_geometry_cache_is_per_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(iw, "woa_ocean_mask", lambda g, p: calls.append(g) or g)
    monkeypatch.setattr(iw, "woa_ocean_bathymetry", lambda g, p, H_max: g)
    monkeypatch.setattr(icg, "make_partial_cell_latlon",
                        lambda z, H, m: ("z", "H", m))
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = SimpleNamespace(woa_t_path="t.nc", woa_s_path="s.nc",
                                    ocean_nlev=4, ocean_H_max_m=4000.0)
    g1, g2 = object(), object()
    assert d._woa_geometry(g1)[2] is g1
    assert d._woa_geometry(g1)[2] is g1
    assert d._woa_geometry(g2)[2] is g2
    assert calls == [g1, g2]


def test_explicit_ocean_grid_is_refused_before_the_atmosphere_sets_up():
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = SimpleNamespace(co2_tracer=False, atm_owns_surface=True,
                                    ocean_mode="dynamic", ocean_ic="woa")
    d._ocean_grid_arg = object()
    d._atm = SimpleNamespace(setup=lambda: pytest.fail("atmosphere set up"))
    with pytest.raises(ValueError, match="explicit ocean grid"):
        d.setup()


def test_wet_cell_a_roundoff_below_full_land_is_refused():
    wet = np.array([[1.0, 1.0]])
    atm = SimpleNamespace(_latlon_native_grid=object(), grid=None)
    with pytest.raises(ValueError, match="wet for the 3-D ocean but full land"):
        _bare(wet)._owned_land_fraction_hook(
            atm, jnp.array([0.2, 1.0 - 1e-14], dtype=jnp.float64))
