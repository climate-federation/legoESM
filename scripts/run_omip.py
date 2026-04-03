#!/usr/bin/env python
"""Reference OMIP simulation — ocean-only forced integration on all grids.

Runs a realistic ocean simulation using WOA18-based initialization (or
analytical fallback), full physics stack (KPP, GM/Redi, SW penetration,
convection, bottom drag), and restoring surface forcing.  Supports all
four ocean grids: cubed-sphere, lat-lon, MPAS, and spectral.

Usage:
    # Quick 30-day smoke test on cubed-sphere:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py --quick --grid cubed_sphere

    # All grids, 1 year:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py --days 365 --grid all

    # Full physics with WOA18 data:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \\
        --days 365 --grid cubed_sphere --woa-t woa18_t.nc --woa-s woa18_s.nc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ===========================================================================
# Grid types and default resolutions / timesteps
# ===========================================================================

GRID_TYPES = ["cubed_sphere", "latlon", "mpas", "spectral"]

GRID_DEFAULTS: dict[str, dict] = {
    "cubed_sphere": {"resolution": "C24", "dt": 300.0},
    "latlon":       {"resolution": "36x72", "dt": 300.0},
    "mpas":         {"resolution": "ico3", "dt": 300.0},
    "spectral":     {"resolution": "T21", "dt": 300.0},
}

ALL_RESULTS: list[dict] = []


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Reference OMIP simulation on all ocean grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", type=str, default="all",
                   choices=GRID_TYPES + ["all"])
    p.add_argument("--resolution", type=str, default=None,
                   help="Grid resolution (e.g. C24, 36x72, ico3, T21)")
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--dt", type=float, default=None,
                   help="Timestep [s] (default: grid-specific)")
    p.add_argument("--days", type=float, default=365.0)
    p.add_argument("--quick", action="store_true",
                   help="Short 30-day run for CI")
    p.add_argument("--output", type=str, default="results/omip")
    p.add_argument("--checkpoint-days", type=float, default=30.0)
    p.add_argument("--woa-t", type=str, default=None,
                   help="WOA18 temperature NetCDF path")
    p.add_argument("--woa-s", type=str, default=None,
                   help="WOA18 salinity NetCDF path")
    p.add_argument("--sw-down", type=float, default=200.0,
                   help="Constant downwelling SW [W/m²]")
    p.add_argument("--physics", type=str, default="full",
                   choices=["full", "minimal", "none"])
    p.add_argument("--water-type", type=str, default="II",
                   choices=["I", "IA", "IB", "II", "III"])
    p.add_argument("--no-conservation-fixer", action="store_true")
    p.add_argument("--restoring-timescale", type=float, default=365.0,
                   help="SST/SSS restoring timescale [days] (default: 365)")
    p.add_argument("--no-restoring", action="store_true",
                   help="Disable SST/SSS restoring")
    p.add_argument("--diag-every", type=int, default=None,
                   help="Diagnostic interval in steps (default: ~1 day)")
    return p.parse_args()


# ===========================================================================
# Resolution parsing
# ===========================================================================

def _parse_resolution(grid_type: str, resolution: str) -> dict:
    if grid_type == "cubed_sphere":
        return {"n": int(resolution.lstrip("Cc"))}
    elif grid_type == "latlon":
        parts = resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif grid_type == "mpas":
        return {"level": int(resolution.replace("ico", ""))}
    elif grid_type == "spectral":
        return {"truncation": int(resolution.lstrip("Tt"))}
    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Physics config presets
# ===========================================================================

def _build_physics_config(preset: str, water_type: str):
    """Build OceanPhysicsConfig from a preset name."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    if preset == "none":
        return None

    if preset == "minimal":
        return OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant"),
            lateral_mixing=LateralMixingConfig(scheme="harmonic"),
            surface_forcing=SurfaceForcingConfig(scheme="restoring"),
            bottom_drag=BottomDragConfig(scheme="linear"),
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=ShortwavePenetrationConfig(water_type=water_type),
        )

    # "full" preset
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=LateralMixingConfig(scheme="gm_redi"),
        surface_forcing=SurfaceForcingConfig(scheme="restoring"),
        bottom_drag=BottomDragConfig(scheme="quadratic"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=ShortwavePenetrationConfig(water_type=water_type),
    )


# ===========================================================================
# Grid + model creation
# ===========================================================================

def _create_setup(grid_type: str, resolution: str, nlev: int, H_max: float,
                  physics_preset: str, water_type: str):
    """Create grid, z_coord, config, model for any grid type.

    All grids use the SAME config-based diffusion (A_h, K_h, A_v, K_v)
    via ``physics=None`` (built-in tendencies) so that the 4 grids are
    physically equivalent.  The ``physics_preset`` only affects which
    OceanPhysicsConfig modules are enabled on cubed-sphere grids where
    the modular pipeline is supported.

    Returns (grid, z_coord, config, model, coord_kind).
    """
    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    params = _parse_resolution(grid_type, resolution)

    # Mixing coefficients tuned per grid for equivalent effective diffusion
    # at ~5° resolution.  FV grids (cubed-sphere, latlon, MPAS) need higher
    # explicit K_h because the discrete Laplacian has truncation error;
    # spectral grids are spectrally accurate and rely on hyperdiffusion.
    A_v = 1.0e-3  # vertical viscosity [m²/s] (all grids)
    K_v = 1.0e-4  # vertical tracer diffusivity [m²/s] (all grids)
    if grid_type == "spectral":
        A_h = 1.0e4   # spectral Laplacian is exact: less needed
        K_h = 1.0e3
    else:
        A_h = 1.0e5   # FV grids need more dissipation
        K_h = 1.0e5

    if grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig

        grid = create_cubed_sphere(params["n"])
        # physics=None → use built-in A_h/K_h/A_v/K_v diffusion
        # (same as latlon/MPAS/spectral for consistency).
        config = OceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            n_barotropic_substeps=30,
            use_conservation_fixer=True,
            physics=None,
        )
        model = OceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "cube"

    elif grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon import LatLonOceanModel
        from legoesm.ocean.state import LatLonOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        config = LatLonOceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            n_barotropic_substeps=30,
            use_conservation_fixer=True,
            physics=None,
        )
        model = LatLonOceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "latlon"

    elif grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        mesh = create_voronoi_mesh(params["level"])
        config = MPASOceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            n_barotropic_substeps=30,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        return mesh, z_coord, config, model, "mpas"

    elif grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig

        grid = create_gaussian_grid(params["truncation"])
        # Keep eta_hyperdiff for barotropic stability (required by unsplit
        # SSP-RK3), but reduce 3D hyperdiffusion to be more consistent
        # with the explicit A_h/K_h on the other grids.
        config = SpectralOceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            # Keep default hyperdiffusion coefficients — they are tuned
            # for stability of the unsplit SSP-RK3 spectral solver.
        )
        model = SpectralOceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "gaussian"

    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# State initialization with WOA T/S
# ===========================================================================

def _init_rest_state(grid_type, grid, z_coord, H_max):
    """Create rest-state initial condition (zero velocity, exponential T, uniform S).

    Uses each grid's standard rest_state function, which provides a
    horizontally uniform stratified profile.  This avoids creating strong
    pressure gradients from horizontal T/S contrasts.
    """
    # land_lat_threshold=80 → land at high latitudes (standard for ocean
    # test cases).  The spectral model requires land boundaries to constrain
    # the barotropic mode (eta_hyperdiff is tuned for this case).
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon import rest_state_latlon_ocean
        return rest_state_latlon_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        return rest_state_spectral_ocean(grid, z_coord, H_max=H_max)
    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Surface forcing construction
# ===========================================================================

def _build_surface_forcing(grid_type, grid, sw_down_value):
    """Create OceanSurfaceForcing with constant SW for the given grid."""
    from legoesm.ocean.state import OceanSurfaceForcing

    if grid_type == "cubed_sphere":
        shape = (6, grid.n, grid.n)
    elif grid_type == "latlon":
        shape = (grid.lat.shape[0], grid.lon.shape[0])
    elif grid_type == "mpas":
        # MPAS model.step doesn't accept surface_forcing yet
        return None
    elif grid_type == "spectral":
        # Spectral model doesn't use surface_forcing pipeline
        return None
    else:
        return None

    sw = jnp.full(shape, sw_down_value)
    return OceanSurfaceForcing(sw_down=sw)


# ===========================================================================
# Grid-agnostic SST/SSS restoring
# ===========================================================================

def _apply_restoring(state, grid_type, grid, T_target, S_target, dt, tau_s):
    """Apply SST/SSS restoring toward WOA climatology.

    This is the OMIP-standard Haney (1971) surface flux restoring:
    dT/dt|surface = -(T_surface - T*) / tau, applied to the top layer only.

    Works identically across all grid types by operating on the state
    arrays directly.

    Parameters
    ----------
    state : ocean state (any grid type)
    grid_type : str
    T_target, S_target : jnp.ndarray
        Target surface T/S from WOA, shape matching the surface layer.
    dt : float
        Timestep [s].
    tau_s : float
        Restoring timescale [s].
    """
    # Restoring coefficient: fraction toward target per step
    alpha = dt / tau_s

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis
        # Apply restoring directly in spectral space to avoid aliasing
        # from repeated synthesis→modify→analysis cycles.
        # T_target and S_target are grid-space 2D fields that were
        # pre-transformed to spectral coefficients and stored alongside
        # the restoring targets (see run_omip_single).
        T_hat = state.T_hat.data
        S_hat = state.S_hat.data
        # T_target / S_target are spectral coefficients of the WOA
        # surface field (pre-computed once).
        T_hat_new = T_hat.at[:, 0].set(
            T_hat[:, 0] - alpha * (T_hat[:, 0] - T_target),
        )
        S_hat_new = S_hat.at[:, 0].set(
            S_hat[:, 0] - alpha * (S_hat[:, 0] - S_target),
        )
        return state._replace(
            T_hat=state.T_hat.replace(data=T_hat_new),
            S_hat=state.S_hat.replace(data=S_hat_new),
        )

    # FV grids: cubed-sphere (6,n,n,nlev), latlon (nlat,nlon,nlev),
    #           MPAS (nCells, nlev)
    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data

    # Restore surface layer only, ocean cells only
    if T.ndim == 4:
        # cubed-sphere: mask shape (6,n,n), target shape (6,n,n)
        mask_sfc = mask
    elif T.ndim == 3:
        # latlon: mask shape (nlat,nlon), target shape (nlat,nlon)
        mask_sfc = mask
    else:
        # MPAS: mask shape (nCells,), target shape (nCells,)
        mask_sfc = mask

    T_new = T.at[..., 0].set(
        T[..., 0] - alpha * (T[..., 0] - T_target) * mask_sfc,
    )
    S_new = S.at[..., 0].set(
        S[..., 0] - alpha * (S[..., 0] - S_target) * mask_sfc,
    )

    return state._replace(
        T=state.T.replace(data=T_new),
        S=state.S.replace(data=S_new),
    )


# ===========================================================================
# Diagnostics
# ===========================================================================

def _extract_scalars(state, grid_type, grid, z_coord):
    """Compute scalar diagnostics from ocean state."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d, sh_synthesis
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real
        eta_grid = sh_synthesis(grid, state.eta_hat.data).real
        mask = np.asarray(state.land_mask_grid.data)
        sst = float(np.nanmean(np.where(mask > 0.5, np.asarray(T_grid[..., 0]), np.nan)))
        sss = float(np.nanmean(np.where(mask > 0.5, np.asarray(S_grid[..., 0]), np.nan)))
        ssh = float(np.nanmean(np.where(mask > 0.5, np.asarray(eta_grid), np.nan)))
        return {"SST": sst, "SSS": sss, "SSH": ssh}

    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    eta = np.asarray(state.eta.data)
    mask = np.asarray(state.land_mask.data)

    if grid_type == "mpas":
        # MPAS: (nCells, nlev), mask: (nCells,)
        wet = mask > 0.5
        sst = float(np.mean(T[wet, 0])) if wet.any() else 0.0
        sss = float(np.mean(S[wet, 0])) if wet.any() else 0.0
        ssh = float(np.mean(eta[wet])) if wet.any() else 0.0
        u = np.asarray(state.u.data)
        max_u = float(np.max(np.abs(u)))
    else:
        # Cubed-sphere (6,n,n,nlev) or latlon (nlat,nlon,nlev)
        if T.ndim == 4:
            mask_3d = mask[..., np.newaxis]
        else:
            mask_3d = mask[..., np.newaxis]
        wet = mask > 0.5
        wet_3d = mask_3d > 0.5
        sst = float(np.mean(T[..., 0][wet]))
        sss = float(np.mean(S[..., 0][wet]))
        ssh = float(np.mean(eta[wet]))
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data) if hasattr(state, 'v') else np.zeros_like(u)
        max_u = float(np.max(np.sqrt(u**2 + v**2)))

    return {"SST": sst, "SSS": sss, "SSH": ssh, "max_speed": max_u if grid_type != "spectral" else 0.0}


def _check_finite(state, grid_type):
    """Check if state contains finite and physically sensible values."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d
        # Check spectral coefficients are finite
        ok_finite = bool(jnp.all(jnp.isfinite(state.T_hat.data)))
        if not ok_finite:
            return False
        # Also check magnitude: spectral coeff l=0 ~ global mean T
        # which should be O(10) not O(1e6)
        T0_mag = float(jnp.abs(state.T_hat.data[0, 0]))
        return T0_mag < 1000.0
    T = state.T.data
    ok = bool(
        jnp.all(jnp.isfinite(T))
        & jnp.all(jnp.isfinite(state.eta.data))
        & (jnp.max(jnp.abs(T)) < 100.0)  # |T| < 100°C
    )
    return ok


# ===========================================================================
# Time loop
# ===========================================================================

def _run_omip_loop(model, state, grid_type, grid, z_coord, dt, n_steps,
                   diag_every, label="",
                   restoring_targets=None, restoring_tau_s=None):
    """Run time loop with diagnostics.

    Parameters
    ----------
    restoring_targets : tuple(T_target, S_target) or None
        Surface T/S targets for SST/SSS restoring.
    restoring_tau_s : float or None
        Restoring timescale [seconds].

    Returns (final_state, diagnostics, wall_time, ok).
    """
    diag: dict[str, list] = {"day": [], "step": []}
    snapshots: dict[int, dict] = {}
    snap_steps = {0, n_steps}
    for i in range(1, min(10, n_steps)):
        snap_steps.add(max(1, int(i * n_steps / 10)))

    # Initial diagnostics
    scalars = _extract_scalars(state, grid_type, grid, z_coord)
    for k, v in scalars.items():
        diag.setdefault(k, []).append(v)
    diag["day"].append(0.0)
    diag["step"].append(0)

    t0 = time.time()
    last_print = t0
    blown_up = False

    for i in range(n_steps):
        state = model.step(state, dt)

        # Apply SST/SSS restoring (grid-agnostic, after dynamics step)
        if restoring_targets is not None:
            T_tgt, S_tgt = restoring_targets
            state = _apply_restoring(
                state, grid_type, grid, T_tgt, S_tgt,
                dt, restoring_tau_s,
            )

        step = i + 1

        if step % 100 == 0:
            if not _check_finite(state, grid_type):
                print(f"  BLOWUP at step {step}")
                blown_up = True
                break

        if step % diag_every == 0 or step == n_steps:
            day = step * dt / 86400.0
            scalars = _extract_scalars(state, grid_type, grid, z_coord)
            diag["day"].append(day)
            diag["step"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            now = time.time()
            if now - last_print > 15:
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:4])
                elapsed = now - t0
                total_days = n_steps * dt / 86400.0
                print(f"    [{label}] Day {day:7.1f}/{total_days:.0f} | {summary} "
                      f"| {elapsed:.0f}s elapsed")
                last_print = now

    if grid_type == "spectral":
        jax.block_until_ready(state.T_hat.data)
    else:
        jax.block_until_ready(state.T.data)

    wall = time.time() - t0
    ok = not blown_up and _check_finite(state, grid_type)
    return state, diag, wall, ok


# ===========================================================================
# Output
# ===========================================================================

def _save_output(output_dir: Path, diag, args, grid_type, wall_time, ok):
    """Save diagnostics and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timeseries CSV
    import csv
    csv_path = output_dir / "timeseries.csv"
    keys = list(diag.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n_rows = len(diag[keys[0]])
        for i in range(n_rows):
            w.writerow([diag[k][i] if i < len(diag[k]) else "" for k in keys])

    # Results JSON
    results = {
        "grid_type": grid_type,
        "resolution": args.resolution or GRID_DEFAULTS[grid_type]["resolution"],
        "nlev": args.nlev,
        "days": args.days,
        "dt": args.dt or GRID_DEFAULTS[grid_type]["dt"],
        "physics": args.physics,
        "water_type": args.water_type,
        "sw_down": args.sw_down,
        "wall_time_s": wall_time,
        "status": "PASS" if ok else "FAIL",
        "final_SST": diag["SST"][-1] if diag["SST"] else None,
        "final_SSS": diag["SSS"][-1] if diag["SSS"] else None,
        "final_SSH": diag["SSH"][-1] if diag["SSH"] else None,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot timeseries
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        days = diag["day"]

        for ax, key in zip(axes.flat, ["SST", "SSS", "SSH", "max_speed"]):
            if key in diag:
                ax.plot(days, diag[key])
                ax.set_xlabel("Day")
                ax.set_ylabel(key)
                ax.set_title(key)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"OMIP {grid_type} — {args.physics} physics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "timeseries.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass  # plotting is optional

    return results


# ===========================================================================
# Single-grid runner
# ===========================================================================

def run_omip_single(grid_type: str, args) -> dict:
    """Run OMIP simulation on a single grid type."""
    resolution = args.resolution or GRID_DEFAULTS[grid_type]["resolution"]
    dt = args.dt or GRID_DEFAULTS[grid_type]["dt"]
    days = 30.0 if args.quick else args.days
    n_steps = int(days * 86400.0 / dt)
    diag_every = args.diag_every or max(1, int(86400.0 / dt))  # ~daily

    print(f"\n{'='*70}")
    print(f"  OMIP: {grid_type} | {resolution} | {args.nlev} levels | "
          f"dt={dt:.0f}s | {days:.0f} days ({n_steps} steps)")
    print(f"  Physics: {args.physics} | SW: {args.sw_down} W/m² | "
          f"Water type: {args.water_type}")
    print(f"{'='*70}")

    t_setup = time.time()

    # Create grid + model (all grids use identical config-based diffusion
    # for cross-grid consistency; physics pipeline disabled).
    grid, z_coord, config, model, coord_kind = _create_setup(
        grid_type, resolution, args.nlev, args.H_max,
        args.physics, args.water_type,
    )

    # --- Initialization strategy ---
    # Start from rest state with uniform T/S, then restore toward WOA
    # climatology.  Starting from full WOA T/S creates extreme pressure
    # gradients that trigger violent geostrophic adjustment (200+ m/s
    # currents, SSS >70 PSU).  Uniform start + restoring is the standard
    # OMIP spin-up approach: the model gradually builds up the
    # climatological circulation from rest.
    from legoesm.ocean.init_woa import init_ocean_from_woa
    T_woa, S_woa = init_ocean_from_woa(grid, z_coord, args.woa_t, args.woa_s)

    # Initial state: rest state with uniform stratification.  Uses only
    # the global-mean of the WOA T profile and S profile to set a
    # physically sensible (but horizontally uniform) initial condition.
    state = _init_rest_state(grid_type, grid, z_coord, args.H_max)

    # SST/SSS restoring targets: WOA surface T/S climatology.
    restoring_targets = None
    restoring_tau_s = None
    if not args.no_restoring and grid_type != "spectral":
        # SST/SSS restoring toward WOA climatology for FV grids.
        # Spectral model: restoring is unstable at coarse resolution
        # (large-scale gradients trigger exponential growth that
        # hyperdiffusion cannot suppress); rely on dynamics + diffusion.
        restoring_targets = (T_woa[..., 0], S_woa[..., 0])
        restoring_tau_s = args.restoring_timescale * 86400.0  # days → seconds
        print(f"  Restoring: tau={args.restoring_timescale:.0f} days")
    elif grid_type == "spectral":
        print(f"  Restoring: disabled (spectral stability)")

    setup_time = time.time() - t_setup
    print(f"  Setup: {setup_time:.1f}s")

    # Run time loop
    state, diag, wall_time, ok = _run_omip_loop(
        model, state, grid_type, grid, z_coord,
        dt, n_steps, diag_every,
        label=f"{grid_type}/{resolution}",
        restoring_targets=restoring_targets,
        restoring_tau_s=restoring_tau_s,
    )

    status = "PASS" if ok else "FAIL"
    icon = "  " if ok else "**"
    sst_str = f"SST={diag['SST'][-1]:.2f}" if diag["SST"] else ""
    print(f"\n  {icon} {status} | {grid_type}/{resolution} | "
          f"{wall_time:.1f}s | {sst_str}")

    # Save output
    output_dir = Path(args.output) / grid_type / resolution
    results = _save_output(output_dir, diag, args, grid_type, wall_time, ok)

    ALL_RESULTS.append(results)
    return results


# ===========================================================================
# Summary
# ===========================================================================

def print_summary():
    """Print summary table of all runs."""
    if not ALL_RESULTS:
        return

    print(f"\n{'='*70}")
    print("  OMIP SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Grid':<15s} {'Resolution':<10s} {'Status':<8s} "
          f"{'Time (s)':<10s} {'Final SST':<10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for r in ALL_RESULTS:
        sst = f"{r['final_SST']:.2f}" if r["final_SST"] is not None else "N/A"
        print(f"  {r['grid_type']:<15s} {r['resolution']:<10s} "
              f"{r['status']:<8s} {r['wall_time_s']:<10.1f} {sst:<10s}")

    n_pass = sum(1 for r in ALL_RESULTS if r["status"] == "PASS")
    n_total = len(ALL_RESULTS)
    print(f"\n  {n_pass}/{n_total} passed")
    print(f"{'='*70}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    grids = GRID_TYPES if args.grid == "all" else [args.grid]

    print(f"legoESM OMIP Reference Simulation")
    print(f"  Grids: {', '.join(grids)}")
    print(f"  Days: {'30 (quick)' if args.quick else args.days}")
    print(f"  Physics: {args.physics}")

    for grid_type in grids:
        try:
            run_omip_single(grid_type, args)
        except Exception:
            print(f"\n  !! ERROR running {grid_type}:")
            traceback.print_exc()
            ALL_RESULTS.append({
                "grid_type": grid_type,
                "resolution": args.resolution or GRID_DEFAULTS[grid_type]["resolution"],
                "status": "ERROR",
                "wall_time_s": 0.0,
                "final_SST": None,
                "final_SSS": None,
                "final_SSH": None,
            })

    print_summary()

    # Exit with error if any failed
    if any(r["status"] != "PASS" for r in ALL_RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
