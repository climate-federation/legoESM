"""Run Williamson TC2 and TC5 with time-series snapshots.

Compares baseline (fv3_sw_tendencies) vs consistent (fv3_sw_tendencies_consistent)
models at C36, saving u_east, v_north, wind_speed, h snapshots at regular intervals.
"""
import os, sys, time
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2, williamson_test5, williamson_test2_exact,
    compute_error_norms,
)


def make_edge_ic(grid, cdgrid, test_num):
    if test_num == 2:
        sw = williamson_test2(grid)
        u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    else:
        sw = williamson_test5(grid)
        u0 = 20.0
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y
    return FV3EdgeShallowWaterState(h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


def geo_winds(state, grid):
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1] + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :] + np.asarray(state.v_d)[:, 1:, :])
    ca, sa = np.asarray(grid.cos_angle), np.asarray(grid.sin_angle)
    return ca * u_cc - sa * v_cc, sa * u_cc + ca * v_cc


def save_snapshot_grid(fields, grid, filename, suptitle):
    """Save a multi-panel figure: rows=fields, cols=snapshot times."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_fields = len(fields)
    n_times = len(fields[0]["data"])
    fig, axes = plt.subplots(n_fields, n_times, figsize=(4 * n_times, 3.5 * n_fields),
                             squeeze=False)

    for row, fspec in enumerate(fields):
        vmin, vmax = fspec.get("vmin"), fspec.get("vmax")
        sym = fspec.get("symmetric", True)
        cmap = "RdBu_r" if sym else "viridis"
        if vmin is None:
            all_vals = np.concatenate([d.ravel() for d in fspec["data"]])
            if sym:
                vmax = max(abs(float(np.nanmin(all_vals))), abs(float(np.nanmax(all_vals))))
                vmax = max(vmax, 0.01)
                vmin = -vmax
            else:
                vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        for col, (arr, t_day) in enumerate(zip(fspec["data"], fspec["times"])):
            ax = axes[row, col]
            # Flatten 6 faces into a 2x3 mosaic
            mosaic = np.zeros((2 * grid.n, 3 * grid.n))
            for f in range(6):
                r, c = f // 3, f % 3
                mosaic[r * grid.n:(r + 1) * grid.n, c * grid.n:(c + 1) * grid.n] = arr[f].T
            im = ax.imshow(mosaic, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Day {t_day:.1f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(fspec["label"], fontsize=10)

        # One colorbar per row
        cb = fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label(fspec.get("units", ""), fontsize=8)

    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def run_case(test_num, n, duration_days, dt, outdir):
    print(f"\n{'='*60}")
    print(f"  Williamson TC{test_num}  C{n}  {duration_days} days  dt={dt}s")
    print(f"{'='*60}")

    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    dx = float(grid.radius) * np.pi / (2 * n)
    hyperdiff = dx**4 / (86400.0 * 10)
    div_damp = 0.15 * dx**2 / dt

    n_steps = int(duration_days * 86400 / dt)
    snap_every = max(1, n_steps // 8)  # ~8 snapshots

    models = {
        "Baseline": FV3EdgeShallowWaterModel(
            grid, CDGridShallowWaterConfig(hyperdiff_coeff=hyperdiff, div_damp=div_damp)),
    }

    for label, model in models.items():
        print(f"\n  [{label}]")
        state = make_edge_ic(grid, model.cdgrid, test_num)
        model.set_initial_mass(state)
        mass0 = float(jnp.sum(state.h * model.cdgrid.base.area))

        snaps_u, snaps_v, snaps_wspd, snaps_h = [], [], [], []
        snap_times = []

        t0 = time.time()
        for i in range(n_steps + 1):
            if i % snap_every == 0 or i == n_steps:
                u_e, v_n = geo_winds(state, grid)
                snaps_u.append(u_e)
                snaps_v.append(v_n)
                snaps_wspd.append(np.sqrt(u_e**2 + v_n**2))
                snaps_h.append(np.asarray(state.h))
                snap_times.append(i * dt / 86400.0)

            if i < n_steps:
                state = model.step(state, dt)

            if i % 100 == 0:
                ok = bool(jnp.all(jnp.isfinite(state.h)))
                if not ok:
                    print(f"    NaN at step {i}!")
                    break

        wall = time.time() - t0
        mass1 = float(jnp.sum(state.h * model.cdgrid.base.area))
        drift = abs(mass1 - mass0) / abs(mass0)
        maxv = float(np.max(np.abs(snaps_v[-1])))

        if test_num == 2:
            ref = williamson_test2_exact(grid, duration_days * 86400)
            h_err = np.asarray(state.h) - np.asarray(ref.h.data)
            area = np.asarray(grid.area)
            l2 = float(np.sqrt(np.sum(h_err**2 * area) / np.sum(ref.h.data**2 * area)))
            print(f"    L2={l2:.4e}  max|v|={maxv:.4f}  mass_drift={drift:.2e}  wall={wall:.1f}s")
        else:
            print(f"    max|v|={maxv:.2f}  mass_drift={drift:.2e}  wall={wall:.1f}s")

        tag = label.lower()
        tc_dir = f"{outdir}/williamson{test_num}/cubed_sphere"
        os.makedirs(tc_dir, exist_ok=True)

        for field_name, field_data, sym in [
            ("u_east", snaps_u, False),
            ("v_north", snaps_v, True),
            ("wind_speed", snaps_wspd, False),
            ("height", snaps_h, False),
        ]:
            save_snapshot_grid([
                {"label": field_name, "units": "m/s" if field_name != "height" else "m",
                 "data": field_data, "times": snap_times, "symmetric": sym},
            ], grid, f"{tc_dir}/{field_name}_{tag}_C{n}.png",
                f"{label}: {field_name}  TC{test_num}  C{n}")


def main():
    outdir = "results/atmosphere/shallow_water"
    os.makedirs(outdir, exist_ok=True)

    n = 36
    dt = 300.0

    # TC2: 5-day steady-state geostrophic flow
    run_case(test_num=2, n=n, duration_days=5.0, dt=dt, outdir=outdir)

    # TC5: 15-day mountain flow
    run_case(test_num=5, n=n, duration_days=15.0, dt=dt, outdir=outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
