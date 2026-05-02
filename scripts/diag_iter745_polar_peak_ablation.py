"""Iter-745 diagnostic: ablate each stabilizer / term in the W2
production path to isolate what drives the ±86° near-pole v-wind
peak.

Iter-744 localised the W2 v_ll artifact to latitude ±86°
(face 4 / face 5 interior near the pole), longitude ±90° mod 180°,
|peak| = 0.303 m/s.  Iter-745 tests which of the following reduces
or eliminates the peak:

  (A) boundary_fix=False       — the non-FV3 cube-edge smoother.
  (B) div_damp=0                — adaptive Smagorinsky-style damp.
  (C) hyperdiff_coeff=0         — biharmonic hyperdiffusion.
  (D) all three above off       — pure A-L + RK3 core.
  (E) face-native peak location — which face? which (i,j)?  this
      also tells us whether the peak is really on face 4/5 or if
      it's an equatorial-face cell projected to lat ±86° by the
      lat-lon regrid.

No source changes — diagnostic only.  Results go to the iter-745
section of docs/fv3_fortran_fidelity_review.md.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter745_polar_peak_ablation.py
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


def run_case(boundary_fix, div_damp_coeff, hyperdiff_coeff, label):
    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    dt = 300.0
    config = CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff_coeff,
        div_damp=div_damp_coeff,
        boundary_fix=boundary_fix)
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

    # Compute face-native v_north (4-edge-average angle convention).
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    ca_np = np.asarray(ca_4edge, dtype=np.float64)
    sa_np = np.asarray(sa_4edge, dtype=np.float64)
    u_cc = 0.5 * (np.asarray(state.u_d, dtype=np.float64)[:, :, :-1]
                  + np.asarray(state.u_d, dtype=np.float64)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d, dtype=np.float64)[:, :-1, :]
                  + np.asarray(state.v_d, dtype=np.float64)[:, 1:, :])
    v_north_face = sa_np * u_cc + ca_np * v_cc

    # Per-face L_inf and argmax.
    per_face_linf = np.max(np.abs(v_north_face).reshape(6, -1), axis=1)
    face_lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    face_lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    argmax_per_face = []
    for f in range(6):
        idx = int(np.argmax(np.abs(v_north_face[f])))
        i, j = np.unravel_index(idx, v_north_face[f].shape)
        argmax_per_face.append((f, i, j, face_lat[f, i, j],
                                face_lon[f, i, j],
                                float(v_north_face[f, i, j])))

    # Post-regrid Linf (what PNG shows).
    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, "cube")
    v_ll_linf = float(np.max(np.abs(v_ll)))
    v_north_linf = float(np.max(np.abs(v_north_face)))

    print(f"  {label:30s}: "
          f"v_ll_Linf={v_ll_linf:.3e}  "
          f"v_north_Linf={v_north_linf:.3e}")
    print(f"    per-face |v_north| Linf: "
          f"{[f'{x:.2e}' for x in per_face_linf]}")
    top_faces = np.argsort(per_face_linf)[::-1][:2]
    for f in top_faces:
        f_, i_, j_, lat_, lon_, val_ = argmax_per_face[f]
        print(f"    face{f_} argmax: (i={i_},j={j_}) "
              f"lat={lat_:+.2f}° lon={lon_:+.2f}° "
              f"v_north={val_:+.3e}")
    return {
        "v_ll_linf": v_ll_linf,
        "v_north_linf": v_north_linf,
        "per_face_linf": per_face_linf.tolist(),
        "top_face_argmax": argmax_per_face,
    }


print("=== Iter-745 W2 C36 1-day ablation of production stabilizers ===\n")
print("Baseline: boundary_fix=True, div_damp=_div_damp_cube(36), "
      "hyperdiff=_hyperdiff_cube(36)\n")

hyp = _hyperdiff_cube(36)
dd = _div_damp_cube(36)

cases = [
    ("baseline (all on)",       True,  dd,   hyp),
    ("boundary_fix=OFF",        False, dd,   hyp),
    ("div_damp=OFF",            True,  0.0,  hyp),
    ("hyperdiff=OFF",           True,  dd,   0.0),
    ("all stabilizers OFF",     False, 0.0,  0.0),
    ("only div_damp ON",        False, dd,   0.0),
    ("only hyperdiff ON",       False, 0.0,  hyp),
    ("only boundary_fix ON",    True,  0.0,  0.0),
]

results = {}
for (label, bf, dd_c, hy_c) in cases:
    r = run_case(bf, dd_c, hy_c, label)
    results[label] = r
    print()

print("\n=== Summary (v_ll_Linf) ===")
baseline = results["baseline (all on)"]["v_ll_linf"]
for label, r in results.items():
    if r is None:
        continue
    delta = r["v_ll_linf"] - baseline
    pct = 100.0 * delta / baseline if baseline != 0 else 0.0
    print(f"  {label:30s}: {r['v_ll_linf']:.3e}  ({delta:+.3e} / "
          f"{pct:+.1f}%)")

print("\nInterpretation key:")
print("  - If 'boundary_fix=OFF' differs significantly from baseline,")
print("    boundary_fix is either helping (less OFF) or creating the")
print("    peak (less baseline).")
print("  - If 'all stabilizers OFF' is close to baseline, the peak is")
print("    intrinsic to the A-L core and stabilizers do not create it.")
print("  - Face-native argmax lets us verify the peak is on face 4/5")
print("    (polar) vs bleeding into equatorial faces near their pole-")
print("    adjacent cells.")
