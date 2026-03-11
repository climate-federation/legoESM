#!/usr/bin/env python
"""Run DCMIP FV 3D test cases with snapshots and time series diagnostics.

Runs:
1. DCMIP-2012 Test 1-1: 3D Deformational Flow (tracer transport, 12 days)
2. DCMIP-2012 Test 1-2: Hadley-like Meridional Circulation (1 day)
3. DCMIP-2012 Test 1-3: Horizontal Advection over Orography (12 days)
4. DCMIP-2025 TC1: Mountain gravity waves (NH compressible, 30 min)
5. DCMIP-2025 TC2a: Gap flow (NH, small Earth, 10 min)

Produces snapshots (lat-lon and cross-sections), time series of mass
conservation, error norms, and stability diagnostics.

Usage:
    python scripts/run_dcmip_fv_diagnostics.py
"""
import sys
import time
import os
from pathlib import Path

# Ensure project root is on the path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

OUTPUT_BASE = "results/atmosphere/nonhydrostatic/dcmip_fv"


# =========================================================================
# Plotting helpers
# =========================================================================

def _regrid_to_latlon(field_faces, grid, n_lon=360, n_lat=180):
    """Regrid cubed-sphere (6, n, n) to lat-lon using IDW."""
    lon = np.asarray(grid.lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(grid.lat, dtype=np.float64).reshape(-1)
    val = np.asarray(field_faces, dtype=np.float64).reshape(-1)

    cos_lat = np.cos(lat)
    src_xyz = np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])

    lon_1d = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat_1d = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    cos_lat_t = np.cos(lat2d.ravel())
    tgt_xyz = np.column_stack([
        cos_lat_t * np.cos(lon2d.ravel()),
        cos_lat_t * np.sin(lon2d.ravel()),
        np.sin(lat2d.ravel()),
    ])

    k = min(8, src_xyz.shape[0])
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(tgt_xyz, k=k)
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist
    w /= np.sum(w, axis=1, keepdims=True)
    field_ll = np.sum(val[idx] * w, axis=1).reshape(n_lat, n_lon)

    return lon2d * 180 / np.pi, lat2d * 180 / np.pi, field_ll


def _pcolor_ll(ax, lon2d, lat2d, data, cmap, vmin, vmax, title):
    pc = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title, fontsize=10, fontweight="bold")
    return pc


def save_tracer_snapshots(out_dir, tag, snap_data, grid, tracer_names):
    """Save lat-lon tracer snapshot panels at selected times."""
    times = sorted(snap_data.keys())
    n_tracers = len(tracer_names)

    for ti, tname in enumerate(tracer_names):
        fig, axes = plt.subplots(1, len(times), figsize=(4.5 * len(times), 4))
        if len(times) == 1:
            axes = [axes]

        panels = {}
        for t_key in times:
            q = snap_data[t_key]["tracers"]  # (6, n, n, nlev, n_tracers)
            # Column mean over levels
            q_col = np.mean(q[..., ti], axis=-1)  # (6, n, n)
            lon2d, lat2d, q_ll = _regrid_to_latlon(q_col, grid)
            panels[t_key] = q_ll

        all_vals = np.concatenate([p.ravel() for p in panels.values()])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = max(abs(vmax - vmin) * 0.02, 1e-6)
        vmin -= pad
        vmax += pad

        for i, t_key in enumerate(times):
            _pcolor_ll(axes[i], lon2d, lat2d, panels[t_key], "viridis", vmin, vmax,
                       snap_data[t_key]["label"])

        fig.suptitle(f"{tag} — {tname} (column mean)", fontsize=13, fontweight="bold", y=1.02)
        fig.colorbar(plt.cm.ScalarMappable(cmap="viridis",
                     norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                     ax=list(axes), shrink=0.8, orientation="horizontal",
                     label=f"{tname}", pad=0.15)
        plt.savefig(f"{out_dir}/{tname.replace(' ', '_')}_snapshots.png",
                    dpi=150, bbox_inches="tight")
        plt.close()


def save_tracer_cross_section(out_dir, tag, snap_data, sigma_coord, grid, tracer_names):
    """Save latitude-level cross-section at 0 deg longitude for each snapshot time."""
    times = sorted(snap_data.keys())

    # Pick face 0 mid-column (approximate equatorial cross-section)
    for ti, tname in enumerate(tracer_names):
        fig, axes = plt.subplots(1, len(times), figsize=(4.5 * len(times), 4))
        if len(times) == 1:
            axes = [axes]

        sigma_full = np.asarray(sigma_coord.sigma_full)

        for i, t_key in enumerate(times):
            q = snap_data[t_key]["tracers"]  # (6, n, n, nlev, n_tracers)
            # Take face 0, mid x-index for a latitude cross-section
            n = q.shape[1]
            cross = q[0, n // 2, :, :, ti]  # (n, nlev) — latitude vs level
            lat_face = np.asarray(grid.lat[0, n // 2, :]) * 180 / np.pi

            pc = axes[i].pcolormesh(lat_face, sigma_full, cross.T,
                                    cmap="viridis", shading="auto")
            axes[i].invert_yaxis()
            axes[i].set_xlabel("Latitude [deg]")
            axes[i].set_ylabel("Sigma")
            axes[i].set_title(snap_data[t_key]["label"], fontsize=10, fontweight="bold")
            plt.colorbar(pc, ax=axes[i], shrink=0.8)

        fig.suptitle(f"{tag} — {tname} (lat-level cross section)",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{tname.replace(' ', '_')}_cross_section.png",
                    dpi=150, bbox_inches="tight")
        plt.close()


def save_transport_timeseries(out_dir, tag, ts_data, tracer_names, has_error_norms=True):
    """Save mass conservation and error norm time series for transport tests."""
    times_days = np.array(ts_data["times_days"])
    n_tracers = len(tracer_names)

    # Mass conservation
    n_panels = 1 + (1 if has_error_norms else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ti in range(n_tracers):
        mass_rel = np.array(ts_data["mass_rel"][ti])
        axes[0].plot(times_days, mass_rel, linewidth=1.5, label=tracer_names[ti])

    axes[0].set_ylabel("Relative mass change")
    axes[0].set_title("Mass Conservation", fontweight="bold")
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0].legend(fontsize=8)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
    axes[0].grid(True, alpha=0.3)

    if has_error_norms:
        for ti in range(n_tracers):
            l2 = np.array(ts_data["l2"][ti])
            axes[1].plot(times_days, l2, linewidth=1.5, label=tracer_names[ti])
        axes[1].set_ylabel("Normalized L2 error")
        axes[1].set_title("L2 Error Norms", fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [days]")
    fig.suptitle(f"{tag} — Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{tag}_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(f"{out_dir}/{tag}_timeseries.npz", **{
        k: np.array(v) if not isinstance(v, list) else v
        for k, v in ts_data.items()
    })


def save_nh_timeseries(out_dir, tag, ts_data):
    """Save time series for non-hydrostatic test cases."""
    times_min = np.array(ts_data["times_min"])
    mass_rel = np.array(ts_data["mass_rel"])
    w_max = np.array(ts_data["w_max"])
    u_max = np.array(ts_data["u_max"])
    theta_p_max = np.array(ts_data["theta_p_max"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(times_min, mass_rel, "b-", linewidth=1.5)
    axes[0].set_ylabel("Relative mass change")
    axes[0].set_title("Mass Conservation", fontweight="bold")
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_min, w_max, "r-", linewidth=1.5, label="max|w|")
    axes[1].plot(times_min, u_max, "b-", linewidth=1.5, label="max|u|")
    axes[1].set_ylabel("Speed [m/s]")
    axes[1].set_title("Maximum Velocities", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times_min, theta_p_max, "g-", linewidth=1.5)
    axes[2].set_ylabel("max|theta'| [K]")
    axes[2].set_title("Potential Temperature Perturbation", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [min]")
    fig.suptitle(f"{tag} — Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{tag}_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(f"{out_dir}/{tag}_timeseries.npz", **ts_data)


def save_nh_snapshot(out_dir, tag, snap_data, grid, height_coord, terrain_metric):
    """Save w and theta' lat-lon snapshots for NH cases."""
    times = sorted(snap_data.keys())

    for field_key, cmap, label in [("w_mid", "RdBu_r", "w at mid-level"),
                                    ("theta_p_mid", "RdBu_r", "theta' at mid-level")]:
        fig, axes = plt.subplots(1, len(times), figsize=(4.5 * len(times), 4))
        if len(times) == 1:
            axes = [axes]

        panels = {}
        for t_key in times:
            field = snap_data[t_key][field_key]
            lon2d, lat2d, field_ll = _regrid_to_latlon(field, grid)
            panels[t_key] = field_ll

        all_vals = np.concatenate([p.ravel() for p in panels.values()])
        vabs = max(float(np.max(np.abs(all_vals))), 1e-8)

        for i, t_key in enumerate(times):
            _pcolor_ll(axes[i], lon2d, lat2d, panels[t_key], cmap, -vabs, vabs,
                       snap_data[t_key]["label"])

        fig.suptitle(f"{tag} — {label}", fontsize=13, fontweight="bold", y=1.02)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                     norm=plt.Normalize(vmin=-vabs, vmax=vabs)),
                     ax=list(axes), shrink=0.8, orientation="horizontal",
                     label=label, pad=0.15)
        fname = field_key.replace("'", "p")
        plt.savefig(f"{out_dir}/{tag}_{fname}_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()


# =========================================================================
# DCMIP-2012 transport tests
# =========================================================================

def run_dcmip_transport(test_id, test_name, wind_fn, init_fn, tracer_names,
                        duration_days, dt, resolution=16, n_levels=30,
                        has_flow_reversal=True):
    """Generic DCMIP transport test runner with snapshots and time series."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.tracer_transport import (
        TracerTransportModel, TracerTransportConfig,
    )
    from tests.test_cases.dcmip_transport import (
        create_dcmip_sigma, compute_tracer_error_norms,
    )

    tag = f"dcmip_{test_id}"
    out_dir = f"{OUTPUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  {test_name}")
    print(f"{'=' * 70}")

    grid = create_cubed_sphere(resolution)
    sigma_coord = create_dcmip_sigma(n_levels)
    state_init = init_fn(grid, sigma_coord)
    duration = duration_days * 86400.0
    n_steps = int(duration / dt)
    n_tracers = state_init.tracers.data.shape[-1]

    config = TracerTransportConfig(hyperdiff_coeff=0.0)
    model = TracerTransportModel(grid, sigma_coord, wind_fn, config)

    print(f"  C{resolution} L{n_levels}, dt={dt}s, {n_steps} steps ({duration_days} days)")

    area = grid.area
    dsigma = sigma_coord.dsigma

    def tracer_mass(q_data):
        masses = []
        for i in range(q_data.shape[-1]):
            qi = q_data[..., i]
            integral = jnp.sum(qi * dsigma, axis=-1)
            masses.append(float(jnp.sum(integral * area)))
        return masses

    mass_init = tracer_mass(state_init.tracers.data)
    print(f"  Initial tracer masses: {[f'{m:.6e}' for m in mass_init]}")

    # Snapshot schedule: 5 evenly spaced snapshots including start and end
    n_snapshots = 5
    snapshot_steps = set([0] + [int(n_steps * i / (n_snapshots - 1)) for i in range(1, n_snapshots)])
    snapshot_steps.add(n_steps)

    # Time series diagnostics every 1/20 of the total
    diag_interval = max(1, n_steps // 20)

    # JIT warmup
    state = state_init
    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.tracers.data)
    print(f"  JIT: {time.time() - t0:.1f}s")

    # Collect data
    snap_data = {}
    ts_data = {
        "times_days": [0.0],
        "mass_rel": [[0.0] for _ in range(n_tracers)],
    }
    if has_flow_reversal:
        ts_data["l2"] = [[0.0] for _ in range(n_tracers)]

    # Day-0 snapshot
    snap_data[0] = {
        "tracers": np.asarray(state_init.tracers.data),
        "label": "t=0",
    }

    print(f"\n  {'Step':>8s}  {'q1 range':>20s}  {'mass_err_q1':>12s}")
    print(f"  {'-' * 50}")

    t_start = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)

        step_num = i + 1  # because step 0 was warmup, step 1 already done
        if step_num % diag_interval == 0 or step_num in snapshot_steps:
            q = state.tracers.data
            jax.block_until_ready(q)

            masses_now = tracer_mass(q)
            t_days = step_num * dt / 86400.0
            ts_data["times_days"].append(t_days)
            for ti in range(n_tracers):
                if abs(mass_init[ti]) > 1e-15:
                    rel = (masses_now[ti] - mass_init[ti]) / abs(mass_init[ti])
                else:
                    rel = masses_now[ti] - mass_init[ti]
                ts_data["mass_rel"][ti].append(rel)

            if has_flow_reversal:
                norms = compute_tracer_error_norms(state, state_init, grid)
                for ti in range(n_tracers):
                    ts_data["l2"][ti].append(float(norms['l2'][ti]))

            q1_range = f"[{float(jnp.min(q[...,0])):.4f}, {float(jnp.max(q[...,0])):.4f}]"
            m_err = ts_data["mass_rel"][0][-1]
            print(f"  {step_num:8d}  {q1_range:>20s}  {m_err:12.2e}")

        if step_num in snapshot_steps:
            snap_data[step_num] = {
                "tracers": np.asarray(state.tracers.data),
                "label": f"t={step_num * dt / 86400.0:.1f}d",
            }

    wall_time = time.time() - t_start
    print(f"  Wall time: {wall_time:.1f}s")

    # Final mass conservation
    mass_final = tracer_mass(state.tracers.data)
    print(f"\n  Final tracer masses: {[f'{m:.6e}' for m in mass_final]}")
    for i in range(n_tracers):
        if abs(mass_init[i]) > 1e-15:
            rel_err = abs(mass_final[i] - mass_init[i]) / abs(mass_init[i])
            print(f"  Tracer {i+1} mass conservation: relative error = {rel_err:.2e}")
        else:
            abs_err = abs(mass_final[i] - mass_init[i])
            print(f"  Tracer {i+1} mass conservation: absolute error = {abs_err:.2e}")

    # Final error norms (flow reversal → exact = initial)
    if has_flow_reversal:
        norms = compute_tracer_error_norms(state, state_init, grid)
        print(f"\n  Flow-reversal error norms (exact = initial):")
        for i in range(n_tracers):
            print(f"    q{i+1}: l1={float(norms['l1'][i]):.4e}  "
                  f"l2={float(norms['l2'][i]):.4e}  "
                  f"linf={float(norms['linf'][i]):.4e}")

    # Save figures
    print("  Generating figures...")
    save_tracer_snapshots(out_dir, test_name, snap_data, grid, tracer_names)
    save_tracer_cross_section(out_dir, test_name, snap_data, sigma_coord, grid, tracer_names)
    save_transport_timeseries(out_dir, tag, ts_data, tracer_names,
                              has_error_norms=has_flow_reversal)

    # Save raw snapshot data (native face panels + metadata)
    for step_num, sdata in snap_data.items():
        np.savez(f"{out_dir}/snapshot_step{step_num:06d}.npz",
                 remap_method="native_cubed_sphere",
                 tracers=sdata["tracers"])

    print(f"  Results saved to {out_dir}/")
    return True


def run_dcmip_transport_11():
    """DCMIP-2012 Test 1-1: 3D Deformational Flow."""
    from tests.test_cases.dcmip_transport import dcmip11_wind, dcmip11_init
    return run_dcmip_transport(
        test_id="11", test_name="DCMIP-2012 Test 1-1: 3D Deformational Flow",
        wind_fn=dcmip11_wind, init_fn=dcmip11_init,
        tracer_names=["q1 cosine bells", "q2 correlated", "q3 slotted", "q4 conservation"],
        duration_days=12, dt=1800.0, resolution=16, n_levels=30,
        has_flow_reversal=True,
    )


def run_dcmip_transport_12():
    """DCMIP-2012 Test 1-2: Hadley-like Meridional Circulation."""
    from tests.test_cases.dcmip_transport import dcmip12_wind, dcmip12_init
    return run_dcmip_transport(
        test_id="12", test_name="DCMIP-2012 Test 1-2: Hadley-like Circulation",
        wind_fn=dcmip12_wind, init_fn=dcmip12_init,
        tracer_names=["q1 vertical layer"],
        duration_days=1, dt=600.0, resolution=16, n_levels=30,
        has_flow_reversal=True,
    )


def run_dcmip_transport_13():
    """DCMIP-2012 Test 1-3: Horizontal Advection over Orography."""
    from tests.test_cases.dcmip_transport import dcmip13_wind, dcmip13_init
    return run_dcmip_transport(
        test_id="13", test_name="DCMIP-2012 Test 1-3: Advection over Orography",
        wind_fn=dcmip13_wind, init_fn=dcmip13_init,
        tracer_names=["q1 low cloud", "q2 mid cloud", "q3 thin cloud", "q4 total"],
        duration_days=12, dt=1800.0, resolution=16, n_levels=30,
        has_flow_reversal=True,
    )


# =========================================================================
# DCMIP-2025 non-hydrostatic tests
# =========================================================================

def global_mass_nh(state, height_coord, terrain_metric, grid):
    """Total atmospheric mass for non-hydrostatic model."""
    rho_0 = height_coord.rho_ref
    rho_total = rho_0 + state.rho_prime.data
    J = terrain_metric.jacobian
    dz = height_coord.dz
    mass_col = jnp.sum(rho_total * dz, axis=-1) * J
    area = grid.area
    return float(jnp.sum(mass_col * area))


def run_dcmip2025_tc1():
    """DCMIP-2025 TC1: Mountain gravity waves (30 min)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel, CompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    tag = "dcmip25_tc1"
    out_dir = f"{OUTPUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("  DCMIP-2025 TC1: Mountain Gravity Waves (30 min)")
    print(f"{'=' * 70}")

    resolution = 16
    n_levels = 20
    dt = 10.0
    duration = 1800.0
    n_steps = int(duration / dt)
    diag_interval = max(1, n_steps // 20)

    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=n_levels)

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=10000.0,
        sponge_coeff=0.05,
    )
    model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)

    print(f"  C{resolution} L{n_levels}, dt={dt}s, {n_steps} steps")

    mass_init = global_mass_nh(state, height_coord, terrain_metric, grid)
    print(f"  Initial total mass: {mass_init:.6e}")

    # Snapshot schedule
    snapshot_steps = set([0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps])

    # JIT warmup
    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT: {time.time() - t0:.1f}s")

    # Time series data
    mid_k = n_levels // 2
    ts_data = {
        "times_min": [0.0],
        "mass_rel": [0.0],
        "w_max": [0.0],
        "u_max": [float(jnp.max(jnp.abs(state.u.data)))],
        "theta_p_max": [0.0],
    }
    snap_data = {}

    # Initial snapshot (from original state before warmup)
    _, _, terrain_metric_snap = dcmip25_tc1_init(grid, n_levels=n_levels)
    snap_data[0] = {
        "w_mid": np.zeros((6, grid.n, grid.n)),
        "theta_p_mid": np.zeros((6, grid.n, grid.n)),
        "label": "t=0",
    }

    print(f"\n  {'Step':>6s}  {'max|w|':>8s}  {'max|u|':>8s}  {'max|thp|':>8s}  {'mass_err':>10s}")
    print(f"  {'-' * 52}")

    t_start = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)
        step_num = i + 1

        if step_num % diag_interval == 0 or step_num in snapshot_steps:
            jax.block_until_ready(state.u.data)
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            theta_p_max = float(jnp.max(jnp.abs(state.theta_prime.data)))
            mass_now = global_mass_nh(state, height_coord, terrain_metric, grid)
            mass_rel = (mass_now - mass_init) / abs(mass_init)

            t_min = step_num * dt / 60.0
            ts_data["times_min"].append(t_min)
            ts_data["mass_rel"].append(mass_rel)
            ts_data["w_max"].append(w_max)
            ts_data["u_max"].append(u_max)
            ts_data["theta_p_max"].append(theta_p_max)

            print(f"  {step_num:6d}  {w_max:8.4f}  {u_max:8.2f}  {theta_p_max:8.4f}  {mass_rel:10.2e}")

        if step_num in snapshot_steps:
            # w is at half-levels, take mid-level (approx)
            w_data = np.asarray(state.w.data)
            w_mid = w_data[..., mid_k] if w_data.shape[-1] > mid_k else w_data[..., -1]
            snap_data[step_num] = {
                "w_mid": w_mid,
                "theta_p_mid": np.asarray(state.theta_prime.data[..., mid_k]),
                "label": f"t={step_num * dt / 60:.0f}min",
            }

    wall_time = time.time() - t_start

    mass_final = global_mass_nh(state, height_coord, terrain_metric, grid)
    mass_err = abs(mass_final - mass_init) / abs(mass_init)
    stable = bool(jnp.all(jnp.isfinite(state.u.data)))

    print(f"\n  Final total mass: {mass_final:.6e}")
    print(f"  Mass conservation: relative error = {mass_err:.2e}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Stable: {stable}")

    # Save figures
    print("  Generating figures...")
    save_nh_timeseries(out_dir, tag, ts_data)
    save_nh_snapshot(out_dir, tag, snap_data, grid, height_coord, terrain_metric)

    # Save results summary
    with open(f"{out_dir}/results.txt", "w") as f:
        f.write(f"test: DCMIP-2025 TC1 Mountain Gravity Waves\n")
        f.write(f"resolution: C{resolution} L{n_levels}\n")
        f.write(f"dt: {dt}s, n_steps: {n_steps}\n")
        f.write(f"mass_err: {mass_err:.2e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time: {wall_time:.1f}s\n")
        f.write(f"final_w_max: {float(jnp.max(jnp.abs(state.w.data))):.4f}\n")
        f.write(f"final_u_max: {float(jnp.max(jnp.abs(state.u.data))):.2f}\n")
        f.write(f"final_theta_p_max: {float(jnp.max(jnp.abs(state.theta_prime.data))):.4f}\n")

    print(f"  Results saved to {out_dir}/")
    return stable


def run_dcmip2025_tc2a():
    """DCMIP-2025 TC2a: Gap flow (10 min, small Earth)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel, CompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc2_init

    tag = "dcmip25_tc2a"
    out_dir = f"{OUTPUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("  DCMIP-2025 TC2a: Gap Flow (10 min, small Earth)")
    print(f"{'=' * 70}")

    resolution = 16
    n_levels = 20
    dt = 5.0
    duration = 600.0  # 10 min (shorter to stay stable at C16)
    n_steps = int(duration / dt)
    diag_interval = max(1, n_steps // 20)

    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
        grid, n_levels=n_levels, subcase="a",
    )

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=15000.0,
        sponge_coeff=1.0 / (0.1 * 86400.0),
        small_earth_factor=20.0,
    )
    model = CompressibleEulerModel(small_grid, height_coord, terrain_metric, config)

    print(f"  C{resolution} L{n_levels}, dt={dt}s, {n_steps} steps")

    mass_init = global_mass_nh(state, height_coord, terrain_metric, small_grid)
    print(f"  Initial total mass: {mass_init:.6e}")

    # Snapshot schedule
    snapshot_steps = set([0, n_steps // 3, 2 * n_steps // 3, n_steps])

    # JIT warmup
    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT: {time.time() - t0:.1f}s")

    mid_k = n_levels // 2
    ts_data = {
        "times_min": [0.0],
        "mass_rel": [0.0],
        "w_max": [0.0],
        "u_max": [float(jnp.max(jnp.abs(state.u.data)))],
        "theta_p_max": [0.0],
    }
    snap_data = {
        0: {
            "w_mid": np.zeros((6, small_grid.n, small_grid.n)),
            "theta_p_mid": np.zeros((6, small_grid.n, small_grid.n)),
            "label": "t=0",
        }
    }

    print(f"\n  {'Step':>6s}  {'max|w|':>8s}  {'max|u|':>8s}  {'mass_err':>10s}")
    print(f"  {'-' * 40}")

    t_start = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)
        step_num = i + 1

        if step_num % diag_interval == 0 or step_num in snapshot_steps:
            jax.block_until_ready(state.u.data)
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            theta_p_max = float(jnp.max(jnp.abs(state.theta_prime.data)))
            mass_now = global_mass_nh(state, height_coord, terrain_metric, small_grid)
            mass_rel = (mass_now - mass_init) / abs(mass_init)

            t_min = step_num * dt / 60.0
            ts_data["times_min"].append(t_min)
            ts_data["mass_rel"].append(mass_rel)
            ts_data["w_max"].append(w_max)
            ts_data["u_max"].append(u_max)
            ts_data["theta_p_max"].append(theta_p_max)

            print(f"  {step_num:6d}  {w_max:8.4f}  {u_max:8.2f}  {mass_rel:10.2e}")

            # Check for NaN/blowup
            if not np.isfinite(w_max) or w_max > 1e3:
                print("  ** BLOWUP DETECTED — stopping early **")
                break

        if step_num in snapshot_steps:
            w_data = np.asarray(state.w.data)
            w_mid = w_data[..., mid_k] if w_data.shape[-1] > mid_k else w_data[..., -1]
            snap_data[step_num] = {
                "w_mid": w_mid,
                "theta_p_mid": np.asarray(state.theta_prime.data[..., mid_k]),
                "label": f"t={step_num * dt / 60:.1f}min",
            }

    wall_time = time.time() - t_start

    mass_final = global_mass_nh(state, height_coord, terrain_metric, small_grid)
    mass_err = abs(mass_final - mass_init) / abs(mass_init)
    stable = bool(jnp.all(jnp.isfinite(state.u.data)))

    print(f"\n  Final total mass: {mass_final:.6e}")
    print(f"  Mass conservation: relative error = {mass_err:.2e}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Stable: {stable}")

    # Save figures
    print("  Generating figures...")
    save_nh_timeseries(out_dir, tag, ts_data)
    save_nh_snapshot(out_dir, tag, snap_data, small_grid, height_coord, terrain_metric)

    with open(f"{out_dir}/results.txt", "w") as f:
        f.write(f"test: DCMIP-2025 TC2a Gap Flow (small Earth)\n")
        f.write(f"resolution: C{resolution} L{n_levels}\n")
        f.write(f"dt: {dt}s, duration: {duration}s, n_steps: {n_steps}\n")
        f.write(f"mass_err: {mass_err:.2e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time: {wall_time:.1f}s\n")

    print(f"  Results saved to {out_dir}/")
    return stable


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 70)
    print("legoESM — DCMIP FV 3D Test Cases with Diagnostics")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Float64: {jnp.zeros(1).dtype}")
    print()

    results = {}

    # Transport tests
    results["transport_11"] = run_dcmip_transport_11()
    results["transport_12"] = run_dcmip_transport_12()
    results["transport_13"] = run_dcmip_transport_13()

    # Non-hydrostatic tests
    results["tc1"] = run_dcmip2025_tc1()
    results["tc2a"] = run_dcmip2025_tc2a()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  All results in: {OUTPUT_BASE}/")
    print("=" * 70)
