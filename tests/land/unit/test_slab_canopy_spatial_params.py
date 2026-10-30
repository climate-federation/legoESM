"""Slab land + two-leaf canopy on a gridded state with gridded land params.

The coupler hands slab land ``reshape_params`` output (every land-param leaf on
the state's grid shape).  The canopy step flattens the state to columns; it must
flatten every spatial param the same way, so a gridded step equals the columnar
step on the same columns.  Before the fix the gridded step raised
``Incompatible shapes for broadcasting: (ncol,), <grid shape>`` (coupled 3-D
CMIP deck smoke, PR #1946).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field
from legoesm.land.carbon import CarbonConfig, init_carbon_state
from legoesm.land.clm_surface_map import clm_provider_rebuild
from legoesm.land.config import LandConfig
from legoesm.land.slab_land import step_land
from legoesm.land.state import LandState
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5, reshape_params
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _params(ncol):
    """Column-varying slab params: forest -> crop cover across the columns."""
    idx = {n: i for i, n in enumerate(CLM5_PFT_NAMES)}
    a = np.linspace(0.0, 0.9, ncol)
    c = np.zeros((ncol, N_PFT_CLM5))
    c[:, idx["broadleaf_evergreen_tropical"]] = 0.9 - a
    c[:, idx["crop_c3"]] = a
    c[:, 0] = 0.1
    o = jnp.ones(ncol)
    sm = dict(theta_wp=0.12 * o, theta_fc=0.30 * o, glacier_frac=0.0 * o,
              soil_albedo=0.15 * o, lai=jnp.linspace(0.5, 4.0, ncol))
    return clm_provider_rebuild(sm, variant="slab", include_soil_albedo=False)(
        jnp.asarray(c))()


def _state(shape, ncol):
    w = jnp.linspace(20.0, 140.0, ncol).reshape(shape)  # spans the bucket range
    f = lambda d, n, u: Field(data=d, name=n, dims=tuple(f"d{i}" for i in range(len(shape))), units=u)  # noqa: E731
    return LandState(T_soil=f(jnp.full(shape, 290.0), "T_soil", "K"),
                     W_bucket=f(w, "W_bucket", "kg/m2"),
                     snow_depth=f(jnp.zeros(shape), "snow_depth", "kg/m2"),
                     snow_age=f(jnp.zeros(shape), "snow_age", "s"))


def _forcing(shape):
    full = lambda v: jnp.full(shape, v)  # noqa: E731
    return AtmToSurface(
        sw_down=full(700.0), lw_down=full(380.0), precip_total=full(1e-5),
        precip_snow=full(0.0), T_lowest=full(295.0), q_lowest=full(0.012),
        u_lowest=full(3.0), v_lowest=full(0.5), p_lowest=full(98000.0),
        p_surface=full(101325.0), rho_lowest=full(1.18), cos_zenith=full(0.8),
        co2_ppmv=full(420.0), has_radiation=True, has_precipitation=True)


@pytest.mark.parametrize("carbon", ["none", "differland"])
@pytest.mark.parametrize("shape", [(6, 2, 2), (3, 8)])  # cubed-sphere, lat-lon
def test_gridded_canopy_step_equals_columnar_step(shape, carbon):
    ncol = math.prod(shape)
    cfg = LandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=30),
                     carbon=CarbonConfig(scheme=carbon))
    p = _params(ncol)
    assert float(jnp.ptp(p.W_max)) > 0.0  # the params really vary by column

    def _run(shp, lp):
        c0 = None if carbon == "none" else init_carbon_state(shp, cfg.carbon)
        return step_land(_state(shp, ncol), _forcing(shp), cfg, U_min=1.0,
                         dt=1800.0, lat=jnp.zeros(shp), carbon_state=c0,
                         doy=180.0, land_params=lp)

    col = _run((ncol,), p)
    grd = _run(shape, reshape_params(p, shape))

    assert grd[1].T_sfc.shape == shape and grd[0].W_bucket.data.shape == shape
    for name, a, b in zip(("state", "response", "carbon"), col, grd):
        la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
        assert len(la) == len(lb), name
        for x, y in zip(la, lb):
            np.testing.assert_allclose(np.asarray(y).reshape(-1),
                                       np.asarray(x).reshape(-1),
                                       rtol=1e-10, atol=0, err_msg=name)
