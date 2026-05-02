"""Iter-743 diagnostic: does enabling duogrid on the production
A-L+RK3 path reduce the iter-717 W2 v-wind mode-4 artifact?

Per iter-722 user directive: "implement the exact FV3 duogrid."
The test matrix currently calls `create_cubed_sphere(n)` with
the default `use_duogrid=False`, so production runs without
duogrid and takes the `interp_offsets` halo path inside
`pad_halo_vector`.  Iter-743 runs the same W2 C36 1-day case
with `use_duogrid=True` and reports the v_ll_Linf sentinel
(iter-742) for direct before/after comparison.

If duogrid reduces the artifact significantly, the test matrix
should be flipped to `use_duogrid=True` for production.  If it
doesn't help or worsens, iter-744+ attacks the non-duogrid
production path directly.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter743_production_duogrid.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig)
from tests.test_cases.williamson import (
    williamson_test2, williamson_test2_exact)


def run_case(use_duogrid, label):
    n = 36
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    # Production settings (mirroring scripts/run_atmosphere_test_matrix.py
    # L1178-1181).
    from scripts.run_atmosphere_test_matrix import (
        _hyperdiff_cube, _div_damp_cube)
    dt = 300.0
    config = CDGridShallowWaterConfig(
        hyperdiff_coeff=_hyperdiff_cube(n),
        div_damp=_div_damp_cube(n),
        boundary_fix=True)
    model = FV3EdgeShallowWaterModel(grid, config)
    cdgrid = model.cdgrid

    # W2 IC identical to matrix.
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    # Step 1 day.
    n_steps = int(86400 / dt)
    for _ in range(n_steps):
        state = model.step(state, dt)
        if not bool(jnp.all(jnp.isfinite(state.h))):
            print(f"  {label}: BLOWUP!")
            return None

    # Compute iter-742 v_ll_Linf.
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    ca_4edge_np = np.asarray(ca_4edge, dtype=np.float64)
    sa_4edge_np = np.asarray(sa_4edge, dtype=np.float64)
    u_cc = 0.5 * (np.asarray(state.u_d, dtype=np.float64)[:, :, :-1]
                  + np.asarray(state.u_d, dtype=np.float64)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d, dtype=np.float64)[:, :-1, :]
                  + np.asarray(state.v_d, dtype=np.float64)[:, 1:, :])
    v_north_face = sa_4edge_np * u_cc + ca_4edge_np * v_cc

    # Regrid to lat-lon (matches matrix's coord_kind="cube").
    from scripts.run_atmosphere_test_matrix import _regrid_2d
    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, "cube")

    v_ll_linf = float(np.max(np.abs(v_ll)))
    v_north_linf = float(np.max(np.abs(v_north_face)))

    # Also compute h L2/Linf for reference.
    exact = williamson_test2_exact(grid, 86400.0)
    h_err = state.h - exact.h.data
    area = grid.area
    h_l2 = float(jnp.sqrt(jnp.sum(h_err**2 * area) / jnp.sum(
        exact.h.data**2 * area)))
    h_linf = float(jnp.max(jnp.abs(h_err)) / jnp.max(jnp.abs(exact.h.data)))

    print(f"  {label:12s}: h_L2={h_l2:.3e}  h_Linf={h_linf:.3e}  "
          f"v_ll_Linf={v_ll_linf:.3e}  v_north_Linf={v_north_linf:.3e}")
    return {"v_ll_linf": v_ll_linf, "v_north_linf": v_north_linf,
            "h_l2": h_l2, "h_linf": h_linf}


print("=== Iter-743 W2 C36 1-day production duogrid comparison ===\n")
print("  (same hyperdiff, div_damp, boundary_fix=True, dt=300s)\n")
r_off = run_case(use_duogrid=False, label="duogrid=OFF")
r_on = run_case(use_duogrid=True, label="duogrid=ON")

print("\n=== Summary ===")
if r_off and r_on:
    dv = r_on["v_ll_linf"] - r_off["v_ll_linf"]
    pct = 100.0 * dv / r_off["v_ll_linf"]
    print(f"  v_ll_Linf delta: {r_off['v_ll_linf']:.3e} -> "
          f"{r_on['v_ll_linf']:.3e} ({dv:+.3e}, {pct:+.1f}%)")
    if abs(pct) < 1.0:
        print("  Duogrid has NEGLIGIBLE effect on the visible artifact.")
    elif pct < -10.0:
        print("  Duogrid MATERIALLY REDUCES the artifact — flip to on.")
    elif pct > 10.0:
        print("  Duogrid MATERIALLY WORSENS the artifact — keep off.")
    else:
        print("  Duogrid has small but measurable effect.")
