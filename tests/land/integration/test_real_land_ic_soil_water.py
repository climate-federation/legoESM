"""The shipped mpas6 land IC keeps its deep soil water on the AMIP soil.

Before the conversion, the Richards step on the AMIP (van Genuchten) soil,
started from the spin-up's Clapp-Hornberger matric potential, dried the deep
root zone from 0.289 to 0.151 in one day.  Needs the gitignored IC and the CLM
surfdata; skipped where they are absent.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_IC = pathlib.Path(os.environ.get(
    "LEGOESM_LAND_IC_MPAS6", _REPO / "data/lmip_soil_ic/soil_ic_mpas6.npz"))
_CLM = pathlib.Path(os.environ.get(
    "LEGOESM_CLM_SURFDATA",
    "/work/bd1083/b309178/diffESM/legoesm_ap/data/clm/"
    "surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (_IC.is_file() and _CLM.is_file()),
                       reason="needs the mpas6 land IC and the CLM surfdata"),
]


def test_one_day_of_soil_water_keeps_the_ic_deep_root_zone():
    import jax
    import jax.numpy as jnp
    from legoesm.grids.factory import create_grid
    from legoesm.land.clm_surface_map import (
        clm_hydraulics_config, load_clm_surface)
    from legoesm.land.restart import (
        HYDRAULICS_SOURCE_CLM_MAP, convert_ic_soil_water, load_land_restart,
        soil_hydraulics_stamp)
    from legoesm.land.richards import RichardsConfig, solve_richards
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid

    g = create_grid("mpas", resolution=6)
    lat = np.rad2deg(np.asarray(g.latCell))
    lon = np.rad2deg(np.asarray(g.lonCell))
    h = clm_hydraulics_config(load_clm_surface(str(_CLM), lat, lon))
    grid = make_soil_grid(SoilGridConfig(n_layers=10, total_depth=3.0,
                                         growth_factor=2.0))
    st, meta = load_land_restart(_IC, expected_land_mode="multilayer",
                                 expected_ncol=lat.size)
    st = st._replace(surface_water=jnp.zeros(lat.size))
    conv, rep = convert_ic_soil_water(
        st, meta, h, soil_hydraulics_stamp(h.retention_curve,
                                           HYDRAULICS_SOURCE_CLM_MAP, _CLM),
        grid.dz)
    assert rep is not None

    z = jnp.zeros(lat.size)
    sink = jnp.zeros_like(st.theta_soil)

    @jax.jit
    def day(psi, theta, sw):
        def body(c, _):
            o = solve_richards(c[0], c[1], grid, h, RichardsConfig(), z, sink,
                               300.0, surface_water=c[2])
            return (o.psi_new, o.theta_new, o.surface_water), None
        return jax.lax.scan(body, (psi, theta, sw), None, length=288)[0]

    deep0 = float(np.asarray(st.theta_soil)[:, -1].mean())
    deep_fixed = float(np.asarray(
        day(conv.psi_soil, conv.theta_soil, conv.surface_water)[1])[:, -1].mean())
    deep_carried = float(np.asarray(
        day(st.psi_soil, st.theta_soil, st.surface_water)[1])[:, -1].mean())
    assert abs(deep_fixed - deep0) < 0.005, (deep0, deep_fixed)
    assert deep0 - deep_carried > 0.1, (deep0, deep_carried)
