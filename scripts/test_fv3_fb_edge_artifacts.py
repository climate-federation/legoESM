"""Test FV3 forward-backward model for edge artifacts.

Runs Williamson TC2 at C36 for 1 day with the new FV3ForwardBackwardModel
and compares against the baseline CDGrid model.  Checks:
1. Stability (all fields finite)
2. Mass conservation
3. L2 error norm vs analytic solution
4. Edge artifact ratio (boundary vs interior error)
5. Visual v-wind snapshot saved for inspection
"""
import os
import sys
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3ForwardBackwardModel,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
    williamson_test2_exact,
    compute_error_norms,
)


def cube_edge_artifact_ratio(error_field, n):
    """Ratio of max error at face boundaries vs interior."""
    # Face boundary: first/last 2 rows and columns on each face
    bw = max(2, n // 8)
    boundary_mask = np.zeros((6, n, n), dtype=bool)
    boundary_mask[:, :bw, :] = True
    boundary_mask[:, -bw:, :] = True
    boundary_mask[:, :, :bw] = True
    boundary_mask[:, :, -bw:] = True
    interior_mask = ~boundary_mask

    err = np.abs(np.asarray(error_field))
    bdy_max = float(np.max(err[boundary_mask])) if np.any(boundary_mask) else 0.0
    int_max = float(np.max(err[interior_mask])) if np.any(interior_mask) else 1e-30
    return bdy_max / max(int_max, 1e-30)


def make_edge_ic(grid, cdgrid, test_num=2):
    """Create edge-midpoint D-grid initial condition for Williamson tests."""
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    # Analytic winds at edge-midpoint positions
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x

    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y

    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


def run_model(model, state, dt, n_steps, label="", verbose=False):
    """Run model for n_steps, return final state and diagnostics."""
    model.set_initial_mass(state)
    mass0 = float(jnp.sum(state.h * model.cdgrid.base.area))

    print(f"  [{label}] Running {n_steps} steps (dt={dt}s)...")
    for i in range(n_steps):
        state = model.step(state, dt)
        if i % 10 == 0:
            ok = bool(jnp.all(jnp.isfinite(state.h)))
            max_u = float(jnp.max(jnp.abs(state.u_d)))
            max_v = float(jnp.max(jnp.abs(state.v_d)))
            max_h = float(jnp.max(state.h))
            min_h = float(jnp.min(state.h))
            if verbose or not ok or max_u > 100:
                print(f"    step {i}: h=[{min_h:.1f}, {max_h:.1f}] "
                      f"|u|={max_u:.2f} |v|={max_v:.2f} ok={ok}")
            if not ok:
                print(f"  [{label}] NaN at step {i}!")
                return state, {"stable": False}

    mass1 = float(jnp.sum(state.h * model.cdgrid.base.area))
    mass_drift = abs(mass1 - mass0) / abs(mass0)

    return state, {"stable": True, "mass_drift": mass_drift}


def compute_geo_winds(state, grid):
    """Extract geographic (u_east, v_north) at cell centers from edge-midpoint D-grid."""
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                  + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                  + np.asarray(state.v_d)[:, 1:, :])
    ca = np.asarray(grid.cos_angle)
    sa = np.asarray(grid.sin_angle)
    u_east = ca * u_cc - sa * v_cc
    v_north = sa * u_cc + ca * v_cc
    return u_east, v_north


def compute_vwind_field(state, grid):
    """Extract geographic v-wind at cell centers from edge-midpoint D-grid."""
    _, v_north = compute_geo_winds(state, grid)
    return v_north


def save_wind_panel_plot(field, grid, filename, title="", label="m/s",
                         symmetric=True):
    """Save a cell-center field as a 6-face panel PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = grid.n
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        if symmetric:
            vmax = max(abs(float(np.nanmin(field))),
                       abs(float(np.nanmax(field))))
            vmax = max(vmax, 0.01)
            kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            kw = dict(cmap="viridis")
        for face in range(6):
            ax = axes[face // 3, face % 3]
            im = ax.imshow(field[face].T, origin="lower", **kw)
            ax.set_title(f"Face {face}")
        plt.colorbar(im, ax=axes.ravel().tolist(), label=label)
        fig.suptitle(title, fontsize=14)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {filename}")
    except Exception as e:
        print(f"  Could not save plot: {e}")


def main():
    n = 36  # C36
    dt = 300.0
    duration_days = 1.0
    n_steps = int(duration_days * 86400 / dt)

    print(f"=== FV3 Forward-Backward Edge Artifact Test ===")
    print(f"Resolution: C{n}, dt={dt}s, duration={duration_days} day(s)")
    print(f"Steps: {n_steps}")
    print()

    # Grid
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Hyperdiffusion and divergence damping (same as test matrix)
    dx = float(grid.radius) * np.pi / (2 * n)
    hyperdiff = dx**4 / (86400.0 * 10)
    div_damp = 0.15 * dx**2 / dt

    # ---- Baseline: FV3EdgeShallowWaterModel (RK3 + Arakawa-Lamb) ----
    config_baseline = CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff,
        div_damp=div_damp,
    )
    model_baseline = FV3EdgeShallowWaterModel(grid, config_baseline)
    state0 = make_edge_ic(grid, model_baseline.cdgrid)
    state_bl, diag_bl = run_model(
        model_baseline, state0, dt, n_steps, label="Baseline RK3")

    # ---- New: FV3ForwardBackwardModel (d2a2c-consistent RK3) ----
    config_fb = CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff,
        div_damp=div_damp,
    )
    model_fb = FV3ForwardBackwardModel(grid, config_fb)
    state0_fb = make_edge_ic(grid, model_fb.cdgrid)
    state_fb, diag_fb = run_model(
        model_fb, state0_fb, dt, n_steps, label="d2a2c-consistent")

    # ---- Error norms ----
    ref = williamson_test2_exact(grid, duration_days * 86400)

    print()
    print("=== Results ===")
    print()

    for label, state, diag in [("Baseline RK3", state_bl, diag_bl),
                                 ("FV3 FB", state_fb, diag_fb)]:
        if not diag["stable"]:
            print(f"  {label}: UNSTABLE (NaN)")
            continue

        # Compute error from h field (cell-center)
        h_err = np.asarray(state.h) - np.asarray(ref.h.data)
        area = np.asarray(grid.area)
        l2 = float(np.sqrt(np.sum(h_err**2 * area) / np.sum(ref.h.data**2 * area)))

        # v-wind for edge artifact analysis
        v_north = compute_vwind_field(state, grid)
        # For TC2, exact v_north = 0, so v_north IS the error
        edge_ratio = cube_edge_artifact_ratio(v_north, n)
        v_max = float(np.max(np.abs(v_north)))

        print(f"  {label}:")
        print(f"    Stable: {diag['stable']}")
        print(f"    Mass drift: {diag['mass_drift']:.2e}")
        print(f"    L2 error:   {l2:.4e}")
        print(f"    Max |v|:    {v_max:.4f} m/s (should be ~0 for TC2)")
        print(f"    Edge ratio: {edge_ratio:.2f} (boundary/interior max |v|)")
        print()

    # ---- Save diagnostic panels ----
    outdir = "output/fv3_fb_test"
    os.makedirs(outdir, exist_ok=True)

    for label_tag, state_s, diag_s, tag in [
        ("baseline", state_bl, diag_bl, "Baseline RK3"),
        ("consistent", state_fb, diag_fb, "Consistent"),
    ]:
        if not diag_s["stable"]:
            continue
        u_geo, v_geo = compute_geo_winds(state_s, grid)
        wspd = np.sqrt(u_geo**2 + v_geo**2)
        h_err = np.asarray(state_s.h) - np.asarray(ref.h.data)

        save_wind_panel_plot(
            u_geo, grid,
            f"{outdir}/u_east_{label_tag}_C{n}.png",
            f"{tag}: u_east  C{n}  {duration_days}-day", "m/s")
        save_wind_panel_plot(
            v_geo, grid,
            f"{outdir}/v_north_{label_tag}_C{n}.png",
            f"{tag}: v_north  C{n}  {duration_days}-day", "m/s")
        save_wind_panel_plot(
            wspd, grid,
            f"{outdir}/wspd_{label_tag}_C{n}.png",
            f"{tag}: wind speed  C{n}  {duration_days}-day", "m/s",
            symmetric=False)
        save_wind_panel_plot(
            h_err, grid,
            f"{outdir}/h_err_{label_tag}_C{n}.png",
            f"{tag}: h error  C{n}  {duration_days}-day", "m")

    print(f"Snapshots saved to {outdir}/")
    print()

    # ---- Pass/fail ----
    if diag_fb["stable"]:
        fb_h_err = np.asarray(state_fb.h) - np.asarray(ref.h.data)
        fb_l2 = float(np.sqrt(np.sum(fb_h_err**2 * area) / np.sum(ref.h.data**2 * area)))
        fb_v = compute_vwind_field(state_fb, grid)
        fb_ratio = cube_edge_artifact_ratio(fb_v, n)
        fb_vmax = float(np.max(np.abs(fb_v)))

        passed = True
        if fb_l2 > 0.01:
            print(f"FAIL: L2 error {fb_l2:.4e} > 0.01")
            passed = False
        if diag_fb["mass_drift"] > 1e-4:
            print(f"FAIL: Mass drift {diag_fb['mass_drift']:.2e} > 1e-4")
            passed = False
        if fb_ratio > 3.0:
            print(f"WARN: Edge ratio {fb_ratio:.2f} > 3.0 (artifacts likely)")
        if fb_vmax > 0.3:
            print(f"WARN: Max |v| {fb_vmax:.4f} > 0.3 m/s (residual artifacts)")

        if passed:
            print("PASS: Forward-backward model passes basic checks")
    else:
        print("FAIL: FV3 FB model is unstable")


if __name__ == "__main__":
    main()
