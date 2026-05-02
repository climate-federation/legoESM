"""Iter-754 diagnostic: enable Fortran-faithful `del6_vt_flux`-based
vorticity damping in `fv3_sw_tendencies` and measure effect on the
W2 polar v-wind peak.

Iter-745 established baseline v_ll_Linf = 0.303 m/s at face 4 (19,
17), lat +86°.  Iter-745→751 traced the mechanism to scalar-
Laplacian-of-vector-component in the hyperdiff block.  Iter-752→753b
built the Fortran-faithful del-n-on-vorticity helper.  Iter-754
wires it into `fv3_sw_tendencies` behind the `damp_v>0` gate.

This diagnostic tests:
(A) Baseline: damp_v=0 (use existing hyperdiff).  Reproduces
    iter-745's v_ll_Linf = 0.303 m/s.
(B) del6 only: damp_v>0, hyperdiff_coeff=0.  Uses ONLY the new
    Fortran-faithful path.  Reports the polar peak.
(C) del6 + hyperdiff: both active.  Tests whether del6 stacks with
    hyperdiff or if one dominates.

If (B) gives v_ll_Linf < 0.15 m/s (half the baseline), the port is
working as expected and the mechanism thesis is confirmed.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter754_del6_polar_peak.py
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
from tests.test_cases.williamson import williamson_test2
from scripts.run_atmosphere_test_matrix import (
    _hyperdiff_cube, _div_damp_cube, _regrid_2d)


def run_case(label, damp_v, nord_v, hyperdiff_coeff):
    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    dt = 300.0
    config = CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff_coeff,
        div_damp=_div_damp_cube(n),
        boundary_fix=True,
        damp_v=damp_v,
        nord_v=nord_v,
    )
    model = FV3EdgeShallowWaterModel(grid, config)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    n_steps = int(86400 / dt)
    for step in range(n_steps):
        state = model.step(state, dt)
        if not bool(jnp.all(jnp.isfinite(state.h))):
            print(f"  {label}: BLOWUP at step {step}!")
            return None

    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    ca_np = np.asarray(ca_4edge, dtype=np.float64)
    sa_np = np.asarray(sa_4edge, dtype=np.float64)
    u_cc = 0.5 * (np.asarray(state.u_d, dtype=np.float64)[:, :, :-1]
                  + np.asarray(state.u_d, dtype=np.float64)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d, dtype=np.float64)[:, :-1, :]
                  + np.asarray(state.v_d, dtype=np.float64)[:, 1:, :])
    v_north_face = sa_np * u_cc + ca_np * v_cc

    per_face_linf = np.max(np.abs(v_north_face).reshape(6, -1), axis=1)
    face_lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    face_lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi

    lon_deg = face_lon
    lat_deg = face_lat
    v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, "cube")
    v_ll_linf = float(np.max(np.abs(v_ll)))
    v_north_linf = float(np.max(np.abs(v_north_face)))

    # h error
    from tests.test_cases.williamson import williamson_test2_exact
    exact = williamson_test2_exact(grid, 86400.0)
    h_err = state.h - exact.h.data
    area = grid.area
    h_l2 = float(jnp.sqrt(jnp.sum(h_err**2 * area)
                          / jnp.sum(exact.h.data**2 * area)))

    print(f"  {label:30s}: "
          f"v_ll_Linf={v_ll_linf:.3e}  "
          f"h_L2={h_l2:.3e}  "
          f"face4={per_face_linf[4]:.2e}  face0={per_face_linf[0]:.2e}")
    return {"v_ll_linf": v_ll_linf, "h_l2": h_l2,
            "face4_linf": float(per_face_linf[4]),
            "face0_linf": float(per_face_linf[0])}


hyp = _hyperdiff_cube(36)

print("=== Iter-754 del6-based vorticity damping vs hyperdiff ===\n")

cases = [
    # label                        damp_v  nord_v  hyperdiff
    ("baseline (hyperdiff only)",     0.0,    1,   hyp),
    ("del6 nord_v=1 small",          0.06,    1,   0.0),
    ("del6 nord_v=1 medium",         0.12,    1,   0.0),
    ("del6 nord_v=1 large",          0.20,    1,   0.0),
    ("del6 nord_v=2 small",          0.06,    2,   0.0),
    ("del6 nord_v=2 medium",         0.12,    2,   0.0),
    ("del6 + hyperdiff (both)",      0.12,    1,   hyp),
    ("no damping at all",             0.0,    1,   0.0),
]

results = {}
for (label, dv, nv, hc) in cases:
    r = run_case(label, dv, nv, hc)
    results[label] = r
    print()

print("\n=== Summary ===")
baseline = results.get("baseline (hyperdiff only)")
if baseline is not None:
    base_v = baseline["v_ll_linf"]
    for label, r in results.items():
        if r is None:
            continue
        dv = r["v_ll_linf"] - base_v
        pct = 100.0 * dv / base_v
        print(f"  {label:30s}: v_ll_Linf={r['v_ll_linf']:.3e}  "
              f"({dv:+.3e}, {pct:+.1f}%)  h_L2={r['h_l2']:.3e}")
