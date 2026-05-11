"""Eady advection scheme comparison — standalone experiment runner.

Systematically tests tracer and momentum advection schemes on the Eady
baroclinic instability setup, tracking instability growth, equilibration,
and spin-down across potentially very long runs.

Two forcing regimes:
  strong  (U=0.8 m/s, tau~5d, default 60d)  — ageostrophic, fast instability
  weak    (U=0.2 m/s, tau~20d, default 600d) — balanced ocean, realistic

Defaults to the implicit Crank-Nicolson barotropic solver to eliminate
2dt aliasing noise from the explicit substep solver (PR #218).

Diagnostics (computed inline as time series):
  - Standard: mean_T, mean_S, max_speed, mean_ke, mean_eta
  - EKE:      eddy kinetic energy (KE minus zonal-mean KE)
  - Var(T):   volume-weighted temperature variance (implicit diffusion metric)

Snapshot-derived diagnostics (computed at snapshot times, recomputable):
  - Zonal wavenumber power spectra of u, v

Usage
-----
Quick smoke test:
    JAX_ENABLE_X64=1 python scripts/run_eady_advection_comparison.py \\
        --quick --tracer-schemes tvd,weno5

Full strong-forcing comparison (60 days):
    JAX_ENABLE_X64=1 python scripts/run_eady_advection_comparison.py

Weak-forcing long run (600 days):
    JAX_ENABLE_X64=1 python scripts/run_eady_advection_comparison.py \\
        --regime weak

Continue from checkpoint:
    JAX_ENABLE_X64=1 python scripts/run_eady_advection_comparison.py \\
        --restart-from results/eady_advection/strong_implicit_cn/tvd_vector_invariant \\
        --days 200

Recompute diagnostics from saved snapshots:
    python scripts/run_eady_advection_comparison.py --recompute \\
        --output results/eady_advection
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Regime presets
# ---------------------------------------------------------------------------

REGIMES = {
    "strong": dict(U_surface=0.8, default_days=60),
    "weak": dict(U_surface=0.2, default_days=600),
}

ALL_TRACER_SCHEMES = [
    "upwind", "tvd", "dst3", "dst3_multidim", "ppm_fct", "som",
    "weno5", "weno7",
]

ALL_MOMENTUM_SCHEMES = ["vector_invariant", "weno5", "weno7"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eady advection scheme comparison runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--regime", choices=["strong", "weak"], default="strong",
        help="Forcing regime: strong (U=0.8, 60d) or weak (U=0.2, 600d)")
    p.add_argument(
        "--barotropic-solver", choices=["explicit_substep", "implicit_cn"],
        default="implicit_cn",
        help="Barotropic solver (default: implicit_cn)")
    p.add_argument(
        "--tracer-schemes", type=str,
        default=",".join(ALL_TRACER_SCHEMES),
        help="Comma-separated tracer advection schemes")
    p.add_argument(
        "--momentum-schemes", type=str, default="vector_invariant",
        help="Comma-separated momentum advection schemes")
    p.add_argument(
        "--grid", choices=["latlon_channel", "mpas_channel"],
        default="latlon_channel",
        help="Grid type (default: latlon_channel)")
    p.add_argument(
        "--resolution", type=str, default="24x72",
        help="Grid resolution (default: 24x72)")
    p.add_argument("--days", type=float, default=None,
                   help="Override duration (otherwise set by regime)")
    p.add_argument("--dt", type=float, default=300.0,
                   help="Timestep in seconds (default: 300)")
    p.add_argument("--output", type=str, default="results/eady_advection",
                   help="Base output directory")
    p.add_argument("--restart-from", type=str, default=None,
                   help="Continue from a previous run directory")
    p.add_argument("--checkpoint-days", type=float, default=30.0,
                   help="Save intermediate restarts every N days")
    p.add_argument("--quick", action="store_true",
                   help="Short 10-day runs for smoke testing")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute diagnostics from saved snapshots (no simulation)")
    p.add_argument("--B-h", type=float, default=None,
                   help="Override biharmonic viscosity [m^4/s]")
    p.add_argument("--C-smag", type=float, default=None,
                   help="Override Smagorinsky coefficient")
    p.add_argument("--no-sponge", action="store_true",
                   help="Disable sponge relaxation")
    p.add_argument("--no-kpp", action="store_true",
                   help="Disable KPP vertical mixing")
    return p


# ---------------------------------------------------------------------------
# Enhanced scalar function (adds EKE and Var(T) to base diagnostics)
# ---------------------------------------------------------------------------

def _make_eady_scalar_fn(grid_type, grid, z_coord):
    """Scalar diagnostics including EKE and Var(T)."""
    from ocean_test_matrix.extraction import _make_scalar_fn
    base_fn = _make_scalar_fn(grid_type, grid, z_coord)

    _area = grid.area if (grid is not None and hasattr(grid, 'area')) else None
    _z_coord = z_coord

    def scalar_fn(s):
        base = base_fn(s)

        # Need ocean mask and volume for EKE and Var(T)
        if not (hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data')):
            base["eke"] = 0.0
            base["var_T"] = 0.0
            return base

        ocean_mask = s.land_mask.data > 0.5

        if _area is None or _z_coord is None:
            base["eke"] = 0.0
            base["var_T"] = 0.0
            return base

        from legoesm.ocean.vertical import compute_layer_thickness
        h_k = compute_layer_thickness(
            s.eta.data, s.H_bathy.data, _z_coord, min_water_column_m=0.5)
        vol = _area[..., None] * h_k * ocean_mask[..., None]
        vol_total = jnp.sum(vol)

        # --- Var(T): volume-weighted temperature variance ---
        mean_T = base["mean_T"]
        T_anom = s.T.data - mean_T
        var_T = float(jnp.sum(T_anom**2 * vol) / jnp.maximum(vol_total, 1e-10))
        base["var_T"] = var_T

        # --- EKE: eddy kinetic energy (total KE minus zonal-mean KE) ---
        u = s.u.data
        v = s.v.data

        if grid_type == "latlon_channel":
            # C-grid: u (n_lat, n_lon+1, nlev), v (n_lat+1, n_lon, nlev)
            # Interpolate to cell centers
            u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
            v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])

            # Zonal mean (axis=1 is longitude for cell-center arrays)
            u_bar = jnp.mean(u_cc, axis=1, keepdims=True)
            v_bar = jnp.mean(v_cc, axis=1, keepdims=True)

            u_prime = u_cc - u_bar
            v_prime = v_cc - v_bar

            eke_field = 0.5 * (u_prime**2 + v_prime**2)
            eke = float(jnp.sum(eke_field * vol) / jnp.maximum(vol_total, 1e-10))
        else:
            # MPAS: EKE not straightforward without zonal averaging
            # Fall back to 0 for now
            eke = 0.0

        base["eke"] = eke
        return base

    return scalar_fn


# ---------------------------------------------------------------------------
# Zonal wavenumber spectra (computed from snapshots)
# ---------------------------------------------------------------------------

def _compute_zonal_spectra(snapshots, grid_type):
    """Compute zonal wavenumber power spectra from saved 3D velocity snapshots.

    Returns dict mapping snapshot step -> {"k": array, "Pu": array, "Pv": array}
    where Pu/Pv are power spectral densities averaged over latitude and depth.
    """
    if grid_type != "latlon_channel":
        return {}  # skip for MPAS (needs regridding)

    spectra = {}
    for step, fields in snapshots.items():
        u3d = fields.get("u_3d")
        v3d = fields.get("v_3d")
        if u3d is None or v3d is None:
            continue

        u3d = np.asarray(u3d)
        v3d = np.asarray(v3d)

        # u3d, v3d are (n_lat, n_lon, n_lev) at cell centers
        n_lon = u3d.shape[1]

        # FFT along zonal axis (axis=1), average power over lat and lev
        u_hat = np.fft.rfft(u3d, axis=1)
        v_hat = np.fft.rfft(v3d, axis=1)

        # Power spectral density (one-sided)
        Pu = np.mean(np.abs(u_hat)**2, axis=(0, 2)) / n_lon
        Pv = np.mean(np.abs(v_hat)**2, axis=(0, 2)) / n_lon
        k = np.arange(Pu.shape[0])

        spectra[step] = {"k": k, "Pu": Pu, "Pv": Pv}

    return spectra


def _save_spectra(spectra, output_dir, dt):
    """Save zonal spectra to CSV and PNG."""
    if not spectra:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "zonal_spectra.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "day", "wavenumber", "Pu", "Pv"])
        for step in sorted(spectra.keys()):
            day = step * dt / 86400.0
            sp = spectra[step]
            for i, k in enumerate(sp["k"]):
                writer.writerow([step, f"{day:.2f}", int(k),
                                 f"{sp['Pu'][i]:.6e}", f"{sp['Pv'][i]:.6e}"])

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        steps_sorted = sorted(spectra.keys())
        for step in steps_sorted:
            day = step * dt / 86400.0
            sp = spectra[step]
            k = sp["k"][1:]  # skip k=0
            axes[0].semilogy(k, sp["Pu"][1:], label=f"day {day:.0f}")
            axes[1].semilogy(k, sp["Pv"][1:], label=f"day {day:.0f}")
        axes[0].set_title("Zonal power spectrum: u")
        axes[1].set_title("Zonal power spectrum: v")
        for ax in axes:
            ax.set_xlabel("Zonal wavenumber k")
            ax.set_ylabel("Power")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "zonal_spectra.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------

def _run_single(
    tracer_scheme: str,
    momentum_scheme: str,
    grid_type: str,
    resolution: str,
    barotropic_solver: str,
    eu_config,
    days: float,
    dt: float,
    output_dir: Path,
    checkpoint_days: float,
    restart_from: str | None = None,
    no_sponge: bool = False,
    no_kpp: bool = False,
):
    """Run one Eady experiment and save all diagnostics.

    Returns (status, wall_time, notes_dict).
    """
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    from ocean_test_matrix.timeloop import _run_timeloop
    from ocean_test_matrix.extraction import (
        _make_check_fn, _make_extract_fn, _key_array_fn,
    )
    from ocean_test_matrix.diagnostic_io import (
        _write_results_txt, _save_case_diagnostics,
        _save_velocity_profiles, _save_cross_sections, save_restart,
    )
    from legoesm.ocean.experiments.eady_uniform import (
        EadyUniformConfig, create_initial_conditions as eu_ic,
        create_forcings as eu_forcings, compute_sponge_mask,
    )
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.core.field import Field

    # Build a TestCase for _create_ocean_setup
    if grid_type == "latlon_channel":
        res_parts = resolution.split("x")
        tc = TestCase(
            case="eady_advection", grid_type=grid_type,
            resolution=resolution, duration_days=days, quick_days=10,
            run_kwargs={
                "lat_south": eu_config.lat_south,
                "lat_north": eu_config.lat_north,
                "lon_west": eu_config.lon_west,
                "lon_east": eu_config.lon_east,
            },
        )
    else:
        tc = TestCase(
            case="eady_advection", grid_type=grid_type,
            resolution=resolution, duration_days=days, quick_days=10,
            run_kwargs={
                "lat_south": eu_config.lat_south,
                "lat_north": eu_config.lat_north,
                "lon_west": eu_config.lon_west,
                "lon_east": eu_config.lon_east,
            },
        )

    # Physics
    physics = eu_forcings(grid_type, None, eu_config)
    if grid_type == "latlon_channel" and not no_kpp:
        from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
        physics = physics._replace(
            vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        )

    # Barotropic solver only applies to latlon C-grid
    bt_solver = barotropic_solver if grid_type == "latlon_channel" else None

    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(
            tc, nlev=20, physics=physics,
            A_h=eu_config.A_h, B_h=eu_config.B_h,
            C_smag=eu_config.C_smag,
            K_h=eu_config.K_h, K_bih=eu_config.K_bih,
            A_v=eu_config.A_v, K_v=eu_config.K_v,
            bottom_drag_r=eu_config.bottom_drag_coeff,
            eos="linear",
            eos_linear=LinearEOSConfig(
                alpha_T=eu_config.alpha_T, rho_ref=eu_config.rho_0,
                T_ref=eu_config.T_ref, S_ref=eu_config.S_uniform,
            ),
            barotropic_diffusion_alpha=eu_config.barotropic_diffusion_alpha,
            barotropic_div_damp=eu_config.barotropic_div_damp,
            tracer_advection=tracer_scheme,
            momentum_advection=momentum_scheme,
            barotropic_solver=bt_solver,
        ))

    state = eu_ic(grid_type, grid, z_coord, eu_config)

    # Handle restart
    start_step = 0
    start_day = 0.0
    if restart_from is not None:
        restart_path = Path(restart_from)
        restart_file = restart_path / "restart.npz"
        if not restart_file.exists():
            print(f"  ERROR: restart file not found: {restart_file}")
            return "ERROR", 0.0, {"error": "restart not found"}
        restart = np.load(restart_file, allow_pickle=True)
        start_day = float(restart["time_days"])
        start_step = int(restart["step"])
        # Rebuild state from restart
        state = _state_from_restart(restart, state, grid_type)
        print(f"  Restarting from day {start_day:.1f} (step {start_step})")

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)
    checkpoint_every = int(checkpoint_days * 86400 / dt)

    # Sponge setup
    gamma = compute_sponge_mask(grid, eu_config)
    T_init = np.array(state.T.data)

    if grid_type == "latlon_channel":
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_cell_to_uface, interp_cell_to_vface)
        decay_T = jnp.array(np.exp(-dt * gamma)[..., np.newaxis])
        T_init_jnp = jnp.array(T_init)
        decay_u = jnp.array(np.exp(-dt * np.array(
            interp_cell_to_uface(jnp.array(gamma))))[..., np.newaxis])
        decay_v = jnp.array(np.exp(-dt * np.array(
            interp_cell_to_vface(jnp.array(gamma))))[..., np.newaxis])
    else:
        decay_T = jnp.array(np.exp(-dt * gamma)[:, np.newaxis])
        T_init_jnp = jnp.array(T_init)
        c1 = np.asarray(grid.cellsOnEdge[0])
        c2 = np.asarray(grid.cellsOnEdge[1])
        gamma_edge = 0.5 * (gamma[c1] + gamma[c2])
        decay_u = jnp.array(np.exp(-dt * gamma_edge)[:, np.newaxis])
        decay_v = None

    check_fn = _make_check_fn(grid_type)
    scalar_fn = _make_eady_scalar_fn(grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(grid_type, grid, lon_deg, lat_deg,
                                  include_velocity_3d=True)

    use_sponge = not no_sponge

    def step_fn(s, dt_):
        s_new = model.step(s, dt_)
        if not use_sponge:
            return s_new
        T_new = s_new.T.data * decay_T + T_init_jnp * (1.0 - decay_T)
        u_new = s_new.u.data * decay_u
        sponge_kw = dict(
            u=Field(u_new, name="u", dims=s_new.u.dims, units=s_new.u.units),
            T=Field(T_new, name="T", dims=s_new.T.dims, units=s_new.T.units))
        if hasattr(s_new, 'v') and decay_v is not None:
            v_new = s_new.v.data * decay_v
            sponge_kw["v"] = Field(v_new, name="v", dims=s_new.v.dims,
                                   units=s_new.v.units)
        if getattr(s_new, 'T_som', None) is not None:
            sponge_kw["T_som"] = s_new.T_som.replace(
                data=s_new.T_som.data * decay_T[..., jnp.newaxis])
        if getattr(s_new, 'S_som', None) is not None:
            sponge_kw["S_som"] = s_new.S_som.replace(
                data=s_new.S_som.data * decay_T[..., jnp.newaxis])
        s_new = s_new._replace(**sponge_kw)
        return s_new

    # --- Time loop with intermediate checkpoints ---
    total_days = days + start_day
    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, grid_type),
        label=f"Eady {tracer_scheme}/{momentum_scheme}",
        total_days=total_days,
        blowup_threshold=float("inf"),
        max_speed_threshold=float("inf"))

    # Shift times if restarting
    if start_day > 0:
        diag["times"] = [t + start_day for t in diag["times"]]

    # --- Compute snapshot-derived diagnostics ---
    spectra = _compute_zonal_spectra(snapshots, grid_type)
    _save_spectra(spectra, output_dir, dt)

    # --- Summary metrics ---
    T_vals = diag.get("mean_T", [])
    T_drift = abs(T_vals[-1] - T_vals[0]) if len(T_vals) >= 2 else 0
    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eke_final = diag["eke"][-1] if diag.get("eke") else 0
    var_T_vals = diag.get("var_T", [])
    if len(var_T_vals) >= 2 and abs(var_T_vals[0]) > 1e-30:
        var_T_drift = (var_T_vals[-1] - var_T_vals[0]) / abs(var_T_vals[0])
    else:
        var_T_drift = 0.0

    notes = {
        "status": "PASS" if ok else "FAIL",
        "tracer": tracer_scheme,
        "momentum": momentum_scheme,
        "barotropic_solver": barotropic_solver,
        "max_speed": max_speed,
        "T_drift": T_drift,
        "eke_final": eke_final,
        "var_T_drift_pct": var_T_drift * 100,
        "wall_time": wall,
        "Ld_km": eu_config.Ld_km,
        "tau_days": eu_config.efolding_days,
    }

    notes_str = (f"max_speed={max_speed:.4f}m/s, T_drift={T_drift:.2e}, "
                 f"EKE={eke_final:.2e}, Var(T)_drift={var_T_drift*100:.2f}%, "
                 f"Ld={eu_config.Ld_km:.0f}km, tau={eu_config.efolding_days:.0f}d")

    # --- Save outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full
    case_label = f"Eady {tracer_scheme}/{momentum_scheme} ({grid_type})"

    _write_results_txt(output_dir, {
        "test": "eady_advection", "grid": grid_type,
        "resolution": resolution,
        "days": days + start_day, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes_str,
        "wall_time": f"{wall:.1f}s",
        "tracer_advection": tracer_scheme,
        "momentum_advection": momentum_scheme,
        "barotropic_solver": barotropic_solver,
    })

    eady_extent = (eu_config.lon_west, eu_config.lon_east,
                   eu_config.lat_south, eu_config.lat_north)
    _save_case_diagnostics(
        output_dir, case_label,
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("speed_sfc", "Surface Speed (m/s)", "plasma"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU",
                      "mean_ke": "m2/s2", "eke": "m2/s2",
                      "var_T": "K2"},
        domain_extent=eady_extent,
        mesh=grid if coord_kind == "mpas" else None)

    for fkey in ("u_3d", "speed_3d"):
        _save_cross_sections(
            output_dir, case_label, snapshots, dt, fkey,
            coord_kind, lon_deg, lat_deg, depth, "Depth (m)")
    _save_velocity_profiles(output_dir, case_label, snapshots, dt,
                            depth, "Depth (m)")

    end_step = start_step + n_steps
    end_day = start_day + days
    save_restart(state, output_dir, grid_type, end_step, end_day)

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# State reconstruction from restart
# ---------------------------------------------------------------------------

def _state_from_restart(restart, state_template, grid_type):
    """Rebuild ocean state from a restart NPZ file."""
    from legoesm.core.field import Field

    replacements = {}
    for name in ("u", "v", "T", "S", "eta", "w"):
        key = name
        if key in restart and hasattr(state_template, name):
            template_field = getattr(state_template, name)
            replacements[name] = Field(
                jnp.array(restart[key]),
                name=template_field.name,
                dims=template_field.dims,
                units=template_field.units,
            )

    # SOM moments
    for name in ("T_som", "S_som"):
        if name in restart and getattr(state_template, name, None) is not None:
            template_field = getattr(state_template, name)
            replacements[name] = template_field.replace(
                data=jnp.array(restart[name]))

    return state_template._replace(**replacements)


# ---------------------------------------------------------------------------
# Recompute mode
# ---------------------------------------------------------------------------

def _recompute_diagnostics(output_base: Path, dt: float):
    """Walk output directory and recompute diagnostics from saved snapshots."""
    print(f"Recomputing diagnostics from: {output_base}")

    run_dirs = []
    for solver_dir in sorted(output_base.iterdir()):
        if not solver_dir.is_dir() or solver_dir.name.startswith("."):
            continue
        for run_dir in sorted(solver_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            # Check for snapshot data
            snap_file = run_dir / "snapshots_latlon.npz"
            snap_nc = run_dir / "snapshots_latlon.nc"
            if snap_file.exists() or snap_nc.exists():
                run_dirs.append(run_dir)

    if not run_dirs:
        print("  No runs with snapshots found.")
        return

    for run_dir in run_dirs:
        print(f"\n  Recomputing: {run_dir.relative_to(output_base)}")

        # Load snapshots
        snap_file = run_dir / "snapshots_latlon.npz"
        if snap_file.exists():
            data = np.load(snap_file, allow_pickle=True)
            # Reconstruct snapshots dict from NPZ arrays
            times = data.get("snapshot_times", np.array([]))
            snapshots = {}
            for i, t in enumerate(times):
                step = int(t)  # approximate
                snap = {}
                for key in data.files:
                    if key == "snapshot_times":
                        continue
                    arr = data[key]
                    if arr.ndim >= 2 and arr.shape[0] == len(times):
                        snap[key] = arr[i]
                if snap:
                    snapshots[step] = snap
        else:
            print(f"    Skipping (no NPZ snapshots)")
            continue

        # Recompute zonal spectra
        spectra = _compute_zonal_spectra(snapshots, "latlon_channel")
        _save_spectra(spectra, run_dir, dt)
        print(f"    Saved zonal spectra ({len(spectra)} snapshots)")

    print("\nRecompute complete.")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_results, regime, barotropic_solver, output_base):
    """Print and save a comparison summary table."""
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  EADY ADVECTION COMPARISON — {regime.upper()} regime, "
          f"{barotropic_solver}")
    print(f"{sep}")
    print(f"  {'Tracer':<16} {'Momentum':<18} {'Status':>6} "
          f"{'max_spd':>8} {'T_drift':>10} {'EKE':>10} "
          f"{'Var(T)%':>8} {'Wall':>8}")
    print("-" * 100)

    for r in all_results:
        n = r["notes"] if isinstance(r["notes"], dict) else {}
        print(f"  {n.get('tracer','?'):<16} {n.get('momentum','?'):<18} "
              f"{r['status']:>6} "
              f"{n.get('max_speed',0):>8.4f} "
              f"{n.get('T_drift',0):>10.2e} "
              f"{n.get('eke_final',0):>10.2e} "
              f"{n.get('var_T_drift_pct',0):>7.2f}% "
              f"{n.get('wall_time',0):>7.1f}s")
    print(sep)

    # Save to file
    summary_path = output_base / "summary.txt"
    output_base.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"Eady Advection Comparison — {regime} regime, "
                f"{barotropic_solver}\n")
        f.write(f"{'Tracer':<16} {'Momentum':<18} {'Status':>6} "
                f"{'max_spd':>8} {'T_drift':>10} {'EKE':>10} "
                f"{'Var(T)%':>8} {'Wall':>8}\n")
        f.write("-" * 100 + "\n")
        for r in all_results:
            n = r["notes"] if isinstance(r["notes"], dict) else {}
            f.write(
                f"{n.get('tracer','?'):<16} {n.get('momentum','?'):<18} "
                f"{r['status']:>6} "
                f"{n.get('max_speed',0):>8.4f} "
                f"{n.get('T_drift',0):>10.2e} "
                f"{n.get('eke_final',0):>10.2e} "
                f"{n.get('var_T_drift_pct',0):>7.2f}% "
                f"{n.get('wall_time',0):>7.1f}s\n")
    print(f"\n  Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    from legoesm.core.precision import set_policy, PrecisionPolicy
    set_policy(PrecisionPolicy.fp64())

    output_base = Path(args.output)

    # Recompute mode: skip simulation entirely
    if args.recompute:
        _recompute_diagnostics(output_base, args.dt)
        return

    # Resolve regime and duration
    regime = REGIMES[args.regime]
    if args.quick:
        days = 10.0
    elif args.days is not None:
        days = args.days
    else:
        days = regime["default_days"]

    tracer_schemes = [s.strip() for s in args.tracer_schemes.split(",")]
    momentum_schemes = [s.strip() for s in args.momentum_schemes.split(",")]

    # Build EadyUniformConfig for this regime, applying CLI overrides
    from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
    eu_kwargs = dict(U_surface=regime["U_surface"])
    if args.B_h is not None:
        eu_kwargs["B_h"] = args.B_h
    if args.C_smag is not None:
        eu_kwargs["C_smag"] = args.C_smag
    eu_config = EadyUniformConfig(**eu_kwargs)

    # Print header
    print("=" * 78)
    print("  Eady Advection Comparison")
    print("=" * 78)
    print(f"  Regime:       {args.regime} (U={regime['U_surface']} m/s)")
    print(f"  Duration:     {days} days")
    print(f"  Grid:         {args.grid} ({args.resolution})")
    print(f"  BT solver:    {args.barotropic_solver}")
    print(f"  B_h:          {eu_config.B_h:.2e}")
    print(f"  C_smag:       {eu_config.C_smag}")
    print(f"  Sponge:       {'OFF' if args.no_sponge else 'ON'}")
    print(f"  KPP:          {'OFF' if args.no_kpp else 'ON'}")
    print(f"  Tracer:       {', '.join(tracer_schemes)}")
    print(f"  Momentum:     {', '.join(momentum_schemes)}")
    print(f"  Ld:           {eu_config.Ld_km:.0f} km")
    print(f"  tau:          {eu_config.efolding_days:.1f} days")
    print(f"  Checkpoints:  every {args.checkpoint_days} days")
    print(f"  Output:       {output_base}")
    print("=" * 78)

    # MPAS warning
    if args.grid == "mpas_channel":
        print("\n  WARNING: MPAS channel has known TRiSK stability issues "
              "at U=0.8 m/s.")
        print("  The implicit barotropic solver is NOT available for MPAS.")
        if args.regime == "strong":
            print("  Consider --regime weak for MPAS runs.\n")

    # Run each combination
    all_results = []
    total_combos = len(tracer_schemes) * len(momentum_schemes)
    combo_idx = 0

    t_start_all = time.time()

    for tracer in tracer_schemes:
        for momentum in momentum_schemes:
            combo_idx += 1
            run_label = f"{tracer}_{momentum}"
            run_dir = output_base / args.regime / run_label

            print(f"\n[{combo_idx}/{total_combos}] {run_label}")
            print("-" * 60)

            try:
                status, wall, notes = _run_single(
                    tracer_scheme=tracer,
                    momentum_scheme=momentum,
                    grid_type=args.grid,
                    resolution=args.resolution,
                    barotropic_solver=args.barotropic_solver,
                    eu_config=eu_config,
                    days=days,
                    dt=args.dt,
                    output_dir=run_dir,
                    checkpoint_days=args.checkpoint_days,
                    restart_from=args.restart_from,
                    no_sponge=args.no_sponge,
                    no_kpp=args.no_kpp,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                status = "ERROR"
                wall = 0.0
                notes = {"error": str(e)[:200], "tracer": tracer,
                         "momentum": momentum}

            all_results.append({
                "status": status, "wall": wall, "notes": notes})

    total_wall = time.time() - t_start_all

    # Summary
    _print_summary(all_results, args.regime, args.barotropic_solver,
                   output_base)
    print(f"\n  Total wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")

    # Exit code
    n_fail = sum(1 for r in all_results if r["status"] != "PASS")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
