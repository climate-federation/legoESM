"""FV3_3D iter-190 helper: run HS C36 cube hybrid 5-day with FV3-
faithful corner-divergence damping ON, regrid to lat-lon, save
snapshot PNGs.

Targeted alternative to ``run_atmosphere_test_matrix.py`` which
runs the full 3-case sweep.  This script runs ONLY the held_suarez
cubed_sphere C36 hybrid case with the iter-19 production damping
config (``corner_div_damp_d2_bg = 0.0005``,
``corner_div_damp_d4_bg = 0.02``, ``corner_div_damp_nord = 1``).
Exercises iter-187 smag_vort + iter-189 dt-actual + iter-190 dedup.

Output: ``results/iter190_hs_c36_hybrid/snapshots_{u,v,wind_speed,p_s}.png``

Usage::

    JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/tmp/_iter190_hs_c36_hybrid_snapshot.py
"""
from __future__ import annotations

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
)
from legoesm.atmosphere.held_suarez import (
    held_suarez_forcing, held_suarez_init,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from legoesm.grids.vertical import standard_hybrid_levels


_REPO = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO / "results" / "iter190_hs_c36_hybrid"


def _regrid_2d(field_face_n_n: np.ndarray, n: int) -> np.ndarray:
    """Regrid (6, n, n) cube → (181, 360) lat-lon."""
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
    return apply_cubedsphere_to_latlon(field_face_n_n, weights)


def main():
    n = 36
    nlev = 6
    dt = 200.0
    n_days = 5.0
    n_steps = int(n_days * 86400.0 / dt)
    snapshot_steps = sorted(set(
        int(k) for k in np.linspace(0, n_steps, 6).tolist()
    ))

    print(f"[iter-190] HS C{n} cube hybrid: {n_days}d, dt={dt}s, "
          f"{n_steps} steps, {len(snapshot_steps)} snapshots")
    print(f"[iter-190] FV3-faithful damping: d2_bg=0.0005, "
          f"d4_bg=0.02, nord=1, smag_vort cap (iter-187), "
          f"dt-actual=dt (iter-189), dedup'd zeta a2b (iter-190)")

    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    # Matrix-correct per-resolution coefficients at C36 (mirrors
    # ``_laplacian_visc_cube(36)``, ``_hyperdiff_cube(36, ref_n=48,
    # ref_coeff=1e16)``, and ``_div_damp_cube(36, ref_n=48,
    # ref_coeff=1.5e7)`` in ``run_atmosphere_test_matrix.py``).
    cfg = CDGridPrimitiveEquationConfig(
        A_h=4.079e6,
        # iter-19 production setting at C36 (validated 30d stable;
        # mid_std reduction 45 % vs no corner damping)
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02,
        corner_div_damp_nord=1,
        # iter-188 dt proxy (default 200.0; iter-189 will use dt_actual=dt)
        corner_div_damp_dt_proxy=dt,
        # iter-187/189 wirings exercised when nord >= 1 + d4_bg > 0.
        # iter-170 (use_fv3_a2b_zeta_corner) and iter-12 damp_v left
        # OFF here to match the matrix iter-19 production reference
        # exactly — adding both at C36 over-damps and goes NaN ~day 4.
        # Matrix per-resolution damping (ref_n=48, scaled to C36)
        hyperdiff_coeff=3.161e16,        # 1e16 * (48/36)**4
        hyperdiff_ps_coeff=3.161e16,
        div_damp_coeff=2.667e7,          # 1.5e7 * (48/36)**2
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    physics_fn = held_suarez_forcing

    # Match the matrix exactly: pass the cell-centre HydrostaticState
    # directly (model.step routes through _step_cell_centre which
    # handles the FV3 conversion internally with the right physics
    # tendency wiring).
    state = held_suarez_init(grid, coord)

    snapshots: dict[int, dict[str, np.ndarray]] = {}
    t_start = time.time()
    for step in range(n_steps + 1):
        if step in snapshot_steps:
            mid_lev = nlev // 2
            u_mid = np.asarray(state.u.data[..., mid_lev], dtype=np.float64)
            v_mid = np.asarray(state.v.data[..., mid_lev], dtype=np.float64)
            ws_mid = np.sqrt(u_mid ** 2 + v_mid ** 2)
            p_s = np.asarray(state.p_s.data, dtype=np.float64)
            snapshots[step] = {
                "u": u_mid, "v": v_mid, "wind_speed": ws_mid, "p_s": p_s,
            }
            print(
                f"  step {step:5d} day {step*dt/86400.0:5.2f} "
                f"max|u|={float(np.max(np.abs(u_mid))):.2f} "
                f"max|v|={float(np.max(np.abs(v_mid))):.2f} "
                f"wall={time.time() - t_start:6.1f}s"
            )
        if step < n_steps:
            state = model.step_with_physics(state, dt, physics_fn=physics_fn)
            # Force materialization periodically so the iteration
            # actually proceeds rather than building a huge JAX trace.
            if step % 50 == 0:
                jax.block_until_ready(state.u.data)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[iter-190] {len(snapshots)} snapshots collected.  "
          f"Regridding cube→lat-lon and saving to {_OUT_DIR}")

    field_specs = [
        ("v", "Meridional wind v [m/s]", "RdBu_r"),
        ("u", "Zonal wind u [m/s]", "RdBu_r"),
        ("wind_speed", "Wind speed |v| [m/s]", "viridis"),
        ("p_s", "Surface pressure [Pa]", "viridis"),
    ]
    for field, label, cmap in field_specs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.ravel()
        sample_vals = []
        for step in snapshot_steps:
            arr = snapshots[step][field]
            sample_vals.append(_regrid_2d(arr, n).ravel())
        all_vals = np.concatenate(sample_vals)
        all_vals = all_vals[np.isfinite(all_vals)]
        if field in ("u", "v"):
            v_lim = float(np.max(np.abs(all_vals))) if len(all_vals) else 1.0
            vmin, vmax = -v_lim, v_lim
        else:
            vmin = float(all_vals.min()) if len(all_vals) else 0.0
            vmax = float(all_vals.max()) if len(all_vals) else 1.0

        for k, step in enumerate(snapshot_steps):
            ax = axes[k]
            arr = snapshots[step][field]
            grid_ll = _regrid_2d(arr, n)
            im = ax.imshow(
                grid_ll, origin="lower", aspect="auto", cmap=cmap,
                extent=[-180, 180, -90, 90], vmin=vmin, vmax=vmax,
            )
            day = step * dt / 86400.0
            ax.set_title(f"day {day:5.2f}", fontsize=10)
            ax.set_xlabel("Longitude [°]")
            ax.set_ylabel("Latitude [°]")
        for k in range(len(snapshot_steps), len(axes)):
            axes[k].set_visible(False)
        cbar = fig.colorbar(
            im, ax=axes.tolist(), orientation="vertical",
            fraction=0.02, pad=0.02, label=label,
        )
        fig.suptitle(
            f"FV3_3D iter-187/189/190: HS C{n} cube hybrid {n_days}d, "
            f"FV3-faithful damping ON ({field})",
            fontsize=12, fontweight="bold",
        )
        out = _OUT_DIR / f"snapshots_{field}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    print(f"[iter-190] Done.  Wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
