"""Colliding-modons SW test case (issue #521) — IC + short prognostic run."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel, iter1009_dual_target_config)
from tests.test_cases.modons import (
    colliding_modons_cubesphere, modon_zonal_wind, _UBAR, _H0)


def test_modon_ic_fields():
    """Uniform free surface, peak zonal wind = ubar, antisymmetric twin jets,
    Coriolis zeroed (non-rotating)."""
    grid = create_cubed_sphere(24)
    cd = create_cubed_sphere_cdgrid(grid)
    state, cd_nr = colliding_modons_cubesphere(grid, cd)

    # uniform h = 5000 m
    assert np.allclose(np.asarray(state.h), _H0)
    # non-rotating: f zeroed
    assert float(jnp.max(jnp.abs(cd_nr.f_corner))) == 0.0
    assert float(jnp.max(jnp.abs(cd.f_corner))) > 0.0   # original was rotating

    # peak geographic wind at the two centres has the right sign + magnitude
    R = grid.radius
    u_w = float(modon_zonal_wind(jnp.array(jnp.pi * 0.5), jnp.array(0.0), R))
    u_e = float(modon_zonal_wind(jnp.array(jnp.pi * 1.5), jnp.array(0.0), R))
    assert u_w == pytest.approx(_UBAR, abs=0.1)     # westerly +50
    assert u_e == pytest.approx(-_UBAR, abs=0.1)    # easterly -50
    # antisymmetry of the twin (centres are mirror images): u(lon+pi) = -u(lon)
    lon = jnp.linspace(0.0, 2 * jnp.pi, 37)
    u_a = modon_zonal_wind(lon, jnp.zeros_like(lon), R)
    u_b = modon_zonal_wind(lon + jnp.pi, jnp.zeros_like(lon), R)
    assert float(jnp.max(jnp.abs(u_a + u_b))) < 1e-9
    # projected D-grid winds never exceed the geographic peak
    assert float(jnp.max(jnp.abs(state.u_d))) <= _UBAR + 1e-6


def test_modon_prognostic_run_stable_and_conserves_mass():
    """Short non-rotating SW integration: finite, mass-conserving, modons evolve,
    no cube-corner blow-up."""
    n = 24
    grid = create_cubed_sphere(n)
    cd = create_cubed_sphere_cdgrid(grid)
    model = FV3EdgeShallowWaterModel(grid, iter1009_dual_target_config(n))
    state, cd_nr = colliding_modons_cubesphere(grid, model.cdgrid)
    model.cdgrid = cd_nr                       # non-rotating planet (FV3 f0=fC=0)
    model.set_initial_mass(state)

    area = model.cdgrid.base.area
    mass0 = float(jnp.sum(state.h * area))
    u0 = state.u_d

    step = jax.jit(lambda s: model.step(s, 300.0))
    s = state
    for _ in range(96):                        # 8 h at dt=300 s
        s = step(s)
    jax.block_until_ready(s.h)

    assert np.all(np.isfinite(np.asarray(s.h)))
    assert np.all(np.isfinite(np.asarray(s.u_d)))
    mass1 = float(jnp.sum(s.h * area))
    assert abs(mass1 / mass0 - 1.0) < 1e-6           # mass conserved
    assert float(jnp.max(jnp.abs(s.u_d))) < 200.0    # bounded, no blow-up
    assert float(jnp.max(jnp.abs(s.u_d - u0))) > 1e-3  # modons actually evolved
    # free surface stays physical (no negative depth)
    assert float(jnp.min(s.h)) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
