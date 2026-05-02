"""Iter-745b diagnostic: does using the non-orthogonal
`pad_halo_vector` rotation path (Fortran-faithful) reduce the W2
polar v-wind peak?

Iter-745 self-review noted that the production `fv3_sw_tendencies`
calls `pad_halo_vector(..., cos_theta=None, sin_theta=None)` which
activates the ORTHOGONAL rotation branch (halo.py:1650-1652).  The
Fortran-faithful option (halo.py:1642-1648) uses the non-orthogonal
metric (`cos_theta=cosa_cell, sin_theta=sina_cell`) and should match
Fortran's cubed-sphere metric convention more exactly.

This script monkey-patches `pad_halo_vector` for the duration of the
W2 C36 1-day run so that every caller (including the two in
`fv3_sw_tendencies`) takes the non-orthogonal branch.  Compares:
  - baseline (orthogonal) v_ll_Linf
  - non-orthogonal v_ll_Linf

No source-code change committed unless the delta is favorable.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter745b_nonorthogonal_halo_rotation.py
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

# We'll install the patch by intercepting pad_halo_vector and
# injecting cos_theta/sin_theta from a module-level closure when
# none is passed.  This covers all call sites uniformly.
from legoesm.grids import halo as halo_mod

_original_pad_halo_vector = halo_mod.pad_halo_vector
_patch_cosa_cell = None
_patch_sina_cell = None


def _patched_pad_halo_vector(
    u_data, v_data, cos_angle, sin_angle,
    cos_angle_padded, sin_angle_padded,
    interp_offsets=None, halo=1, duogrid=None,
    cos_theta=None, sin_theta=None,
):
    """Wrapper that injects cosa_cell/sina_cell as cos_theta/sin_theta
    when the caller didn't specify them."""
    if cos_theta is None and sin_theta is None \
            and _patch_cosa_cell is not None \
            and _patch_sina_cell is not None:
        # Shape guard: the non-orthogonal path requires (6, n, n)
        # metrics that match u_data / v_data cell-centre shape.
        if u_data.shape == _patch_cosa_cell.shape:
            cos_theta = _patch_cosa_cell
            sin_theta = _patch_sina_cell
    return _original_pad_halo_vector(
        u_data, v_data, cos_angle, sin_angle,
        cos_angle_padded, sin_angle_padded,
        interp_offsets=interp_offsets, halo=halo, duogrid=duogrid,
        cos_theta=cos_theta, sin_theta=sin_theta,
    )


def run_case(use_nonorthogonal, label):
    global _patch_cosa_cell, _patch_sina_cell

    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Install / clear patch.
    if use_nonorthogonal:
        _patch_cosa_cell = np.asarray(cdgrid.cosa_cell)
        _patch_sina_cell = np.asarray(cdgrid.sina_cell)
        halo_mod.pad_halo_vector = _patched_pad_halo_vector
        # Also update the import in operators_cdgrid.
        from legoesm.grids import halo as _halo_ref
        _halo_ref.pad_halo_vector = _patched_pad_halo_vector
    else:
        _patch_cosa_cell = None
        _patch_sina_cell = None
        halo_mod.pad_halo_vector = _original_pad_halo_vector
        from legoesm.grids import halo as _halo_ref
        _halo_ref.pad_halo_vector = _original_pad_halo_vector

    dt = 300.0
    config = CDGridShallowWaterConfig(
        hyperdiff_coeff=_hyperdiff_cube(n),
        div_damp=_div_damp_cube(n),
        boundary_fix=True)
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
            # Restore.
            halo_mod.pad_halo_vector = _original_pad_halo_vector
            from legoesm.grids import halo as _halo_ref
            _halo_ref.pad_halo_vector = _original_pad_halo_vector
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
    argmax_per_face = []
    for f in range(6):
        idx = int(np.argmax(np.abs(v_north_face[f])))
        i, j = np.unravel_index(idx, v_north_face[f].shape)
        argmax_per_face.append((f, i, j, face_lat[f, i, j],
                                face_lon[f, i, j],
                                float(v_north_face[f, i, j])))

    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, "cube")
    v_ll_linf = float(np.max(np.abs(v_ll)))
    v_north_linf = float(np.max(np.abs(v_north_face)))

    # Also compute h L2 for comparison.
    from tests.test_cases.williamson import williamson_test2_exact
    exact = williamson_test2_exact(grid, 86400.0)
    h_err = state.h - exact.h.data
    area = grid.area
    h_l2 = float(jnp.sqrt(jnp.sum(h_err**2 * area)
                          / jnp.sum(exact.h.data**2 * area)))

    print(f"  {label:24s}: "
          f"v_ll_Linf={v_ll_linf:.3e}  v_north_Linf={v_north_linf:.3e}  "
          f"h_L2={h_l2:.3e}")
    print(f"    per-face |v_north|: "
          f"{[f'{x:.2e}' for x in per_face_linf]}")
    top_faces = np.argsort(per_face_linf)[::-1][:2]
    for f in top_faces:
        f_, i_, j_, lat_, lon_, val_ = argmax_per_face[f]
        print(f"    face{f_} argmax: (i={i_},j={j_}) "
              f"lat={lat_:+.2f}° lon={lon_:+.2f}°  v_north={val_:+.3e}")

    # Restore.
    halo_mod.pad_halo_vector = _original_pad_halo_vector
    from legoesm.grids import halo as _halo_ref
    _halo_ref.pad_halo_vector = _original_pad_halo_vector
    return {"v_ll_linf": v_ll_linf, "v_north_linf": v_north_linf,
            "h_l2": h_l2}


print("=== Iter-745b non-orthogonal pad_halo_vector test ===\n")
print("  Baseline: pad_halo_vector(cos_theta=None) — orthogonal branch.")
print("  Test:     pad_halo_vector(cos_theta=cosa_cell, sin_theta=sina_cell)"
      " — non-orthogonal, Fortran-faithful.\n")

r_base = run_case(False, "orthogonal  (baseline)")
print()
r_no = run_case(True,  "non-orthogonal   (test)")

print("\n=== Summary ===")
if r_base and r_no:
    dv = r_no["v_ll_linf"] - r_base["v_ll_linf"]
    pct = 100.0 * dv / r_base["v_ll_linf"]
    print(f"  v_ll_Linf: {r_base['v_ll_linf']:.3e} -> "
          f"{r_no['v_ll_linf']:.3e} ({dv:+.3e}, {pct:+.1f}%)")
    dh = r_no["h_l2"] - r_base["h_l2"]
    hpct = 100.0 * dh / r_base["h_l2"]
    print(f"  h_L2:      {r_base['h_l2']:.3e} -> {r_no['h_l2']:.3e} "
          f"({dh:+.3e}, {hpct:+.1f}%)")
    print()
    if pct < -10.0 and hpct < 10.0:
        print("  → Non-orthogonal path MATERIALLY REDUCES polar peak.  "
              "Commit the source change.")
    elif pct > 10.0 or r_no["v_ll_linf"] > 1e0:
        print("  → Non-orthogonal path WORSENS or blows up.  "
              "Do not commit.")
    else:
        print("  → Effect is small.  Review both metrics before committing.")
