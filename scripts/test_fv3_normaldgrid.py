"""Test FV3 normal D-grid shallow water model for edge artifacts.

Runs Williamson TC2 and TC5 at C36 with the new FV3NormalDGridModel
and compares against the baseline FV3EdgeShallowWaterModel.

Checks:
1. Stability (all fields finite)
2. Mass conservation
3. L2 error norm vs analytic solution
4. Edge artifact ratio (boundary vs interior error in v-wind)
5. Visual v-wind snapshots saved for inspection
6. TC5 stability over 15 days
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
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig,
)
from legoesm.atmosphere.dynamics.shallow_water_fv3_normaldgrid import (
    FV3NormalDGridModel,
    NormalDGridShallowWaterState,
    NormalDGridShallowWaterConfig,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
    williamson_test2_exact,
    williamson_test5,
    compute_error_norms,
)


# ==============================================================================
# Helpers
# ==============================================================================

def cube_edge_artifact_ratio(error_field, n):
    """Ratio of max error at face boundaries vs interior."""
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


def make_edge_ic(grid, cdgrid):
    """Create edge-midpoint D-grid IC for Williamson TC2."""
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x

    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y

    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


def make_normal_dgrid_ic(grid, cdgrid, test_case="tc2"):
    """Create normal D-grid IC for Williamson TC2 or TC5.

    Normal D-grid layout:
      u_xi  at x-face positions (6, n+1, n) = edge_y positions
      v_eta at y-face positions (6, n, n+1) = edge_x positions
    """
    if test_case == "tc2":
        sw = williamson_test2(grid)
        u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    elif test_case == "tc5":
        sw = williamson_test5(grid)
        u0 = 20.0  # TC5 uses u0=20 m/s
    else:
        raise ValueError(f"Unknown test case: {test_case}")

    # u_xi at edge_y positions (6, n+1, n):
    #   u_east = u0 * cos(lat), v_north = 0
    #   u_xi = cos_angle * u_east + sin_angle * v_north = cos_angle * u_east
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    u_xi = cdgrid.cos_angle_edge_y * u_east_y   # (6, n+1, n)

    # v_eta at edge_x positions (6, n, n+1):
    #   v_eta = -sin_angle * u_east + cos_angle * v_north = -sin_angle * u_east
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    v_eta = -cdgrid.sin_angle_edge_x * u_east_x  # (6, n, n+1)

    return NormalDGridShallowWaterState(
        h=sw.h.data, u_xi=u_xi, v_eta=v_eta, h_s=sw.h_s.data)


def make_edge_ic_tc5(grid, cdgrid):
    """Create edge-midpoint D-grid IC for Williamson TC5."""
    sw = williamson_test5(grid)
    u0 = 20.0

    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x

    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y

    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


def run_model(model, state, dt, n_steps, label="", verbose=False):
    """Run model for n_steps, return final state and diagnostics."""
    model.set_initial_mass(state)
    area = model.cdgrid.base.area
    mass0 = float(jnp.sum(state.h * area))

    print(f"  [{label}] Running {n_steps} steps (dt={dt}s)...")
    for i in range(n_steps):
        state = model.step(state, dt)
        if i % 50 == 0 or i == n_steps - 1:
            ok = bool(jnp.all(jnp.isfinite(state.h)))
            if hasattr(state, 'u_xi'):
                max_u = float(jnp.max(jnp.abs(state.u_xi)))
                max_v = float(jnp.max(jnp.abs(state.v_eta)))
            else:
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

    mass1 = float(jnp.sum(state.h * area))
    mass_drift = abs(mass1 - mass0) / abs(mass0)

    return state, {"stable": True, "mass_drift": mass_drift}


def compute_geo_winds_edge(state, grid):
    """Geographic winds from edge-midpoint D-grid state."""
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                  + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                  + np.asarray(state.v_d)[:, 1:, :])
    ca = np.asarray(grid.cos_angle)
    sa = np.asarray(grid.sin_angle)
    u_east = ca * u_cc - sa * v_cc
    v_north = sa * u_cc + ca * v_cc
    return u_east, v_north


def compute_geo_winds_normal(state, grid):
    """Geographic winds from normal D-grid state."""
    # Cell-centre averages
    u_cc = 0.5 * (np.asarray(state.u_xi)[:, :-1, :]
                  + np.asarray(state.u_xi)[:, 1:, :])
    v_cc = 0.5 * (np.asarray(state.v_eta)[:, :, :-1]
                  + np.asarray(state.v_eta)[:, :, 1:])
    ca = np.asarray(grid.cos_angle)
    sa = np.asarray(grid.sin_angle)
    u_east = ca * u_cc - sa * v_cc
    v_north = sa * u_cc + ca * v_cc
    return u_east, v_north


def save_wind_panel_plot(field, grid, filename, title="", label="m/s",
                         symmetric=True):
    """Save a cell-center field as a 6-face panel PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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


# ==============================================================================
# Main
# ==============================================================================

def main():
    n = 36   # C36
    dt = 300.0

    print("=" * 70)
    print("FV3 Normal D-Grid Shallow Water Test")
    print("=" * 70)
    print(f"Resolution: C{n}, dt={dt}s")
    print()

    # Grid
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Diffusion parameters (same as test matrix)
    dx = float(grid.radius) * np.pi / (2 * n)
    hyperdiff = dx**4 / (86400.0 * 10)
    div_damp = 0.15 * dx**2 / dt

    outdir = "results/atmosphere/shallow_water/williamson2/cubed_sphere"
    os.makedirs(outdir, exist_ok=True)

    # ==================================================================
    # TC2: 1-day integration
    # ==================================================================
    print("-" * 50)
    print("Williamson TC2: 1-day integration")
    print("-" * 50)
    n_steps_1d = int(86400 / dt)

    # Baseline: FV3EdgeShallowWaterModel (RK3)
    config_bl = CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff,
        div_damp=div_damp,
    )
    model_bl = FV3EdgeShallowWaterModel(grid, config_bl)
    state0_bl = make_edge_ic(grid, model_bl.cdgrid)
    state_bl_1d, diag_bl_1d = run_model(
        model_bl, state0_bl, dt, n_steps_1d, label="Baseline (edge-D)")

    # Normal D-grid model (no hyperdiffusion — C-A-C filter controls modes)
    config_nd = NormalDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
    )
    model_nd = FV3NormalDGridModel(grid, config_nd)
    state0_nd = make_normal_dgrid_ic(grid, cdgrid)
    state_nd_1d, diag_nd_1d = run_model(
        model_nd, state0_nd, dt, n_steps_1d, label="Normal D-grid")

    # Error analysis
    ref_1d = williamson_test2_exact(grid, 86400)
    area = np.asarray(grid.area)

    print()
    print("=== TC2 1-day Results ===")

    results = []
    for label, state, diag, geo_fn in [
        ("Baseline (edge-D)", state_bl_1d, diag_bl_1d, compute_geo_winds_edge),
        ("Normal D-grid", state_nd_1d, diag_nd_1d, compute_geo_winds_normal),
    ]:
        if not diag["stable"]:
            print(f"  {label}: UNSTABLE (NaN)")
            results.append((label, None, None, None, diag))
            continue

        h_err = np.asarray(state.h) - np.asarray(ref_1d.h.data)
        l2 = float(np.sqrt(np.sum(h_err**2 * area) / np.sum(ref_1d.h.data**2 * area)))

        u_east, v_north = geo_fn(state, grid)
        edge_ratio = cube_edge_artifact_ratio(v_north, n)
        v_max = float(np.max(np.abs(v_north)))

        print(f"  {label}:")
        print(f"    Stable:     {diag['stable']}")
        print(f"    Mass drift: {diag['mass_drift']:.2e}")
        print(f"    L2(h) err:  {l2:.4e}")
        print(f"    Max |v|:    {v_max:.4f} m/s (exact=0 for TC2)")
        print(f"    Edge ratio: {edge_ratio:.2f}")
        print()

        results.append((label, l2, v_max, edge_ratio, diag))

    # Save v-wind snapshots (1-day)
    for tag, state_s, diag_s, geo_fn, prefix in [
        ("baseline", state_bl_1d, diag_bl_1d, compute_geo_winds_edge, "baseline"),
        ("normal_dgrid", state_nd_1d, diag_nd_1d, compute_geo_winds_normal, "normal_dgrid"),
    ]:
        if not diag_s["stable"]:
            continue
        u_east, v_north = geo_fn(state_s, grid)
        save_wind_panel_plot(
            v_north, grid,
            f"{outdir}/v_north_{prefix}_1day_C{n}.png",
            f"{tag}: v_north TC2 C{n} 1-day", "m/s")

    # ==================================================================
    # TC2: 5-day integration
    # ==================================================================
    print("-" * 50)
    print("Williamson TC2: 5-day integration")
    print("-" * 50)
    n_steps_5d = int(5 * 86400 / dt)

    # Reset and run baseline 5 days
    model_bl2 = FV3EdgeShallowWaterModel(grid, config_bl)
    state0_bl2 = make_edge_ic(grid, model_bl2.cdgrid)
    state_bl_5d, diag_bl_5d = run_model(
        model_bl2, state0_bl2, dt, n_steps_5d, label="Baseline (edge-D) 5d")

    # Reset and run normal D-grid 5 days
    model_nd2 = FV3NormalDGridModel(grid, config_nd)
    state0_nd2 = make_normal_dgrid_ic(grid, cdgrid)
    state_nd_5d, diag_nd_5d = run_model(
        model_nd2, state0_nd2, dt, n_steps_5d, label="Normal D-grid 5d")

    ref_5d = williamson_test2_exact(grid, 5 * 86400)

    print()
    print("=== TC2 5-day Results ===")
    for label, state, diag, geo_fn in [
        ("Baseline (edge-D)", state_bl_5d, diag_bl_5d, compute_geo_winds_edge),
        ("Normal D-grid", state_nd_5d, diag_nd_5d, compute_geo_winds_normal),
    ]:
        if not diag["stable"]:
            print(f"  {label}: UNSTABLE (NaN)")
            continue

        h_err = np.asarray(state.h) - np.asarray(ref_5d.h.data)
        l2 = float(np.sqrt(np.sum(h_err**2 * area) / np.sum(ref_5d.h.data**2 * area)))

        u_east, v_north = geo_fn(state, grid)
        edge_ratio = cube_edge_artifact_ratio(v_north, n)
        v_max = float(np.max(np.abs(v_north)))

        print(f"  {label}:")
        print(f"    Stable:     {diag['stable']}")
        print(f"    Mass drift: {diag['mass_drift']:.2e}")
        print(f"    L2(h) err:  {l2:.4e}")
        print(f"    Max |v|:    {v_max:.4f} m/s")
        print(f"    Edge ratio: {edge_ratio:.2f}")
        print()

    # Save v-wind snapshots (5-day)
    for tag, state, diag, geo_fn, prefix in [
        ("baseline", state_bl_5d, diag_bl_5d, compute_geo_winds_edge, "baseline"),
        ("normal_dgrid", state_nd_5d, diag_nd_5d, compute_geo_winds_normal, "normal_dgrid"),
    ]:
        if not diag["stable"]:
            continue
        u_east, v_north = geo_fn(state, grid)
        save_wind_panel_plot(
            v_north, grid,
            f"{outdir}/v_north_{prefix}_5day_C{n}.png",
            f"{tag}: v_north TC2 C{n} 5-day", "m/s")

    # ==================================================================
    # TC5: 15-day stability test
    # ==================================================================
    print("-" * 50)
    print("Williamson TC5: 15-day stability test")
    print("-" * 50)
    n_steps_tc5 = int(15 * 86400 / dt)

    # Baseline TC5
    model_bl_tc5 = FV3EdgeShallowWaterModel(grid, config_bl)
    state0_bl_tc5 = make_edge_ic_tc5(grid, model_bl_tc5.cdgrid)
    state_bl_tc5, diag_bl_tc5 = run_model(
        model_bl_tc5, state0_bl_tc5, dt, n_steps_tc5,
        label="Baseline TC5 15d")

    # Normal D-grid TC5
    model_nd_tc5 = FV3NormalDGridModel(grid, config_nd)
    state0_nd_tc5 = make_normal_dgrid_ic(grid, cdgrid, test_case="tc5")
    state_nd_tc5, diag_nd_tc5 = run_model(
        model_nd_tc5, state0_nd_tc5, dt, n_steps_tc5,
        label="Normal D-grid TC5 15d")

    print()
    print("=== TC5 15-day Results ===")
    for label, state, diag, geo_fn in [
        ("Baseline (edge-D)", state_bl_tc5, diag_bl_tc5, compute_geo_winds_edge),
        ("Normal D-grid", state_nd_tc5, diag_nd_tc5, compute_geo_winds_normal),
    ]:
        if not diag["stable"]:
            print(f"  {label}: UNSTABLE (NaN)")
            continue

        u_east, v_north = geo_fn(state, grid)
        v_max = float(np.max(np.abs(v_north)))
        wspd = np.sqrt(u_east**2 + v_north**2)
        wspd_max = float(np.max(wspd))

        print(f"  {label}:")
        print(f"    Stable:      {diag['stable']}")
        print(f"    Mass drift:  {diag['mass_drift']:.2e}")
        print(f"    Max |v|:     {v_max:.4f} m/s")
        print(f"    Max wspd:    {wspd_max:.4f} m/s")
        print()

    # Save TC5 snapshots
    tc5dir = "results/atmosphere/shallow_water/williamson5/cubed_sphere"
    os.makedirs(tc5dir, exist_ok=True)
    for tag, state, diag, geo_fn, prefix in [
        ("baseline", state_bl_tc5, diag_bl_tc5, compute_geo_winds_edge, "baseline"),
        ("normal_dgrid", state_nd_tc5, diag_nd_tc5, compute_geo_winds_normal, "normal_dgrid"),
    ]:
        if not diag["stable"]:
            continue
        u_east, v_north = geo_fn(state, grid)
        wspd = np.sqrt(u_east**2 + v_north**2)
        save_wind_panel_plot(
            v_north, grid,
            f"{tc5dir}/v_north_{prefix}_15day_C{n}.png",
            f"{tag}: v_north TC5 C{n} 15-day", "m/s")
        save_wind_panel_plot(
            wspd, grid,
            f"{tc5dir}/wspd_{prefix}_15day_C{n}.png",
            f"{tag}: wind speed TC5 C{n} 15-day", "m/s",
            symmetric=False)

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True

    if diag_nd_1d["stable"]:
        _, l2_nd, vmax_nd, ratio_nd, _ = results[1] if len(results) > 1 else (None, None, None, None, None)
        if l2_nd is not None:
            if l2_nd > 0.01:
                print(f"FAIL: 1-day L2(h) error {l2_nd:.4e} > 0.01")
                all_pass = False
            if diag_nd_1d["mass_drift"] > 1e-4:
                print(f"FAIL: 1-day mass drift {diag_nd_1d['mass_drift']:.2e} > 1e-4")
                all_pass = False
            if ratio_nd is not None and ratio_nd > 3.0:
                print(f"WARN: 1-day edge artifact ratio {ratio_nd:.2f} > 3.0")
            if vmax_nd is not None and vmax_nd > 0.3:
                print(f"WARN: 1-day max |v| {vmax_nd:.4f} > 0.3 m/s")
    else:
        print("FAIL: Normal D-grid model unstable (1-day TC2)")
        all_pass = False

    if not diag_nd_5d["stable"]:
        print("FAIL: Normal D-grid model unstable (5-day TC2)")
        all_pass = False

    if not diag_nd_tc5["stable"]:
        print("FAIL: Normal D-grid model unstable (15-day TC5)")
        all_pass = False
    else:
        print(f"PASS: Normal D-grid TC5 stable for 15 days "
              f"(mass drift: {diag_nd_tc5['mass_drift']:.2e})")

    if all_pass:
        print()
        print("PASS: All normal D-grid checks passed")
    print()


if __name__ == "__main__":
    main()
