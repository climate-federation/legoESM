"""Per-column (spatial) van-Genuchten hydraulics in the Richards solver."""
import jax
# Enable x64 BEFORE jax.numpy / the module-level GRID is built, else GRID.dz is
# float32 and the float32-soil dtype regression below would pass vacuously (the
# body never upcasts, so there is no carry mismatch to catch).
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig, psi_from_theta)
from legoesm.land.richards import solve_richards, RichardsConfig
from legoesm.land.soil_texture import SOIL_TEXTURE_VG

GRID = make_soil_grid(SoilGridConfig(n_layers=6, total_depth=2.0))
RCFG = RichardsConfig()


def _per_col_config(textures):
    """SoilHydraulicsConfig with (ncol,1) VG fields for the named textures."""
    pick = lambda k: jnp.asarray([[SOIL_TEXTURE_VG[t][k]] for t in textures])
    return SoilHydraulicsConfig(theta_r=pick("theta_r"), theta_sat=pick("theta_sat"),
                                alpha_vg=pick("alpha_vg"), n_vg=pick("n_vg"),
                                K_sat=pick("K_sat"))


def _run(cfg, ncol, steps=20, dt=1800.0):
    theta = jnp.full((ncol, GRID.n_layers), 0.30)
    psi = psi_from_theta(theta, cfg)
    flux = jnp.full((ncol,), 5e-6)        # infiltration [m/s]
    sink = jnp.zeros((ncol, GRID.n_layers))
    for _ in range(steps):
        out = solve_richards(psi, theta, GRID, cfg, RCFG, flux, sink, dt)
        # the boundary fluxes must stay (ncol,) under a per-column (ncol,1) config
        assert out.runoff_subsurface.shape == (ncol,)
        assert out.runoff_surface.shape == (ncol,)
        psi, theta = out.psi_new, out.theta_new
    return np.asarray(theta)


def test_heterogeneous_columns_diverge():
    """Sand vs clay columns under the SAME forcing develop DIFFERENT moisture
    (sand drains/holds less than clay), and both stay physical."""
    cfg = _per_col_config(["sand", "clay"])
    theta = _run(cfg, ncol=2)
    assert np.all(np.isfinite(theta))
    sand, clay = theta[0], theta[1]
    # clay retains more water than sand at the surface (higher field capacity)
    assert clay[0] > sand[0]
    # both within their own [theta_r, theta_sat]
    for i, t in enumerate(["sand", "clay"]):
        lo, hi = SOIL_TEXTURE_VG[t]["theta_r"], SOIL_TEXTURE_VG[t]["theta_sat"]
        assert np.all(theta[i] >= lo - 1e-6) and np.all(theta[i] <= hi + 1e-6)


def test_uniform_array_matches_scalar():
    """A per-column config with all columns = loam reproduces the scalar loam
    config exactly (the array path is the same solver, just broadcast)."""
    scalar = SoilHydraulicsConfig()                      # VG loam scalar default
    arr = _per_col_config(["loam", "loam"])
    t_scalar = _run(scalar, ncol=2)
    t_arr = _run(arr, ncol=2)
    assert np.allclose(t_scalar, t_arr, atol=1e-9)


def test_float32_soil_state_survives_scan_carry_under_x64():
    """A float32 soil state (e.g. downcast between coupled segments) must not
    trip the Picard ``fori_loop`` "scan carry input/output type mismatch" under
    x64: the Thomas solve promotes ``psi + dpsi`` to dz's float64 working
    precision, so a float32 input carry vs a float64 output carry would crash at
    compile.  ``solve_richards`` now promotes the initial carry to that working
    dtype.  Before the fix this call raised TypeError at trace time.  (No-op
    when x64 is off: dz is then float32 and nothing is promoted.)"""
    cfg = _per_col_config(["sand", "clay"])
    ncol = 2
    theta = jnp.full((ncol, GRID.n_layers), 0.30, dtype=jnp.float32)
    psi = psi_from_theta(theta, cfg).astype(jnp.float32)
    flux = jnp.full((ncol,), 5e-6, dtype=jnp.float32)
    sink = jnp.zeros((ncol, GRID.n_layers), dtype=jnp.float32)

    out = solve_richards(psi, theta, GRID, cfg, RCFG, flux, sink, 1800.0)  # no raise

    work = jnp.result_type(psi, GRID.dz)
    assert out.psi_new.dtype == work
    assert out.theta_new.dtype == work
    assert np.all(np.isfinite(np.asarray(out.theta_new)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
