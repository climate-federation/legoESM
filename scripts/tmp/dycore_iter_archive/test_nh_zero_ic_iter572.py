"""FV3_3D iter 572: NH with truly zero IC.

iter-571 found PE on Held-Suarez has ~1 mK δT noise from dynamics
alone.  Compare: NH with truly zero IC (no winds at all).  What's
the dynamics-only noise growth?

Tests
-----

1. ``test_nh_zero_ic_growth``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_zero_state(n=16):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float64),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0), dtype=jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_nh_zero_ic_growth(capsys):
    """NH from rest (all zeros) — measures dynamics-only noise."""
    grid, hc, tm, state = _build_zero_state(n=16)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        heat_source_del2_iters=8,
        heat_source_del2_coeff=0.20,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)
    s = state
    for _ in range(10):
        s = step(s, 10.0)
    with capsys.disabled():
        print(
            f"\n[iter-572 NH from rest (zero IC) @ C16, 10 steps]"
        )
        for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
            d = np.asarray(getattr(s, fld).data)
            print(
                f"  {fld:12s}: |max|={np.abs(d).max():.3e}, "
                f"std={d.std():.3e}"
            )
    # Verify result remains finite — bounds will likely all be near 0
    for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
        d = getattr(s, fld).data
        assert jnp.all(jnp.isfinite(d))
