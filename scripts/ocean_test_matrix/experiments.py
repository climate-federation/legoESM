"""Ocean test matrix experiment runners.

Each ``run_*`` function sets up an experiment, runs the timeloop,
computes diagnostics, writes output files, and returns (status, wall_time, notes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from ocean_test_matrix import config
from ocean_test_matrix.setup import _create_ocean_setup
from ocean_test_matrix.timeloop import _run_timeloop, _compute_drift
from ocean_test_matrix.extraction import (
    _make_check_fn, _make_scalar_fn, _make_extract_fn, _key_array_fn,
    _make_baroclinic_scalar_fn,
)
from ocean_test_matrix.diagnostic_io import (
    _write_results_txt, _save_case_diagnostics, _save_velocity_profiles,
)
from ocean_test_matrix.testcase import TestCase


def run_rest_state(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Rest state adjustment: model should remain near initial condition."""
    from legoesm.ocean.experiments.rest_state import RestStateConfig, create_initial_conditions as rest_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = rest_ic(tc.grid_type, grid, z_coord, RestStateConfig())

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State ({tc.grid_type})", total_days=days)

    # Check drift is small - all grids now use same physical units
    # Use absolute eta drift in meters rather than relative drift since
    # initial mean_eta is ~0 in rest state, making relative drift meaningless (division by ~0).
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    # Spectral land-leakage diagnostic: check that eta stays near zero in land cells
    if tc.grid_type == "spectral" and hasattr(state, 'land_mask_grid'):
        from legoesm.grids.gaussian import sh_synthesis
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
        eta_grid = np.asarray(sh_synthesis(grid, state.eta_hat.data).real,
                              dtype=np.float64)
        land_leakage = np.max(np.abs(eta_grid * (1.0 - mask)))
        notes += f", land_leakage={land_leakage:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full  # positive downward for plotting

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Rest State {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_rest_state_no_land(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Rest state adjustment with no land: pure ocean should remain near initial condition."""
    from legoesm.ocean.experiments.rest_state import RestStateConfig, create_initial_conditions as rest_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = rest_ic(tc.grid_type, grid, z_coord, RestStateConfig(include_land=False))

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State No Land ({tc.grid_type})", total_days=days)

    # Check drift is small - all grids now use same physical units
    # Use absolute eta drift in meters rather than relative drift since
    # initial mean_eta is ~0 in rest state, making relative drift meaningless (division by ~0).
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full  # positive downward for plotting

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Rest State No Land {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_rest_state_uniform_ts(tc: TestCase, output_dir: Path, days: float
                              ) -> tuple[str, float, str]:
    """Rest state with land but uniform T/S — diagnostic for baroclinic PGF hypothesis.

    If the stratified rest_state (with land) blows up but this test stays
    stable, it confirms the baroclinic pressure gradient is the primary
    trigger for land-boundary instabilities.
    """
    from legoesm.ocean.experiments.rest_state import RestStateConfig, create_initial_conditions as rest_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = rest_ic(tc.grid_type, grid, z_coord, RestStateConfig(uniform_ts=True))

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State Uniform T/S ({tc.grid_type})", total_days=days)

    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Rest State Uniform T/S {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_rest_state_uniform_ts_no_land(tc: TestCase, output_dir: Path, days: float
                                      ) -> tuple[str, float, str]:
    """Rest state with uniform T/S and no land — control for uniform_ts diagnostic."""
    from legoesm.ocean.experiments.rest_state import RestStateConfig, create_initial_conditions as rest_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = rest_ic(tc.grid_type, grid, z_coord, RestStateConfig(uniform_ts=True, include_land=False))

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State Uniform T/S No Land ({tc.grid_type})", total_days=days)

    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Rest State Uniform T/S No Land {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Barotropic Wave
# ===========================================================================

def run_barotropic_wave(tc: TestCase, output_dir: Path, days: float
                        ) -> tuple[str, float, str]:
    """Barotropic gravity wave: Gaussian SSH perturbation propagation."""
    if tc.grid_type == "spectral":
        raise NotImplementedError(
            "Barotropic wave skipped for spectral grid (land masking issues)")
    from legoesm.ocean.experiments.barotropic_wave import create_initial_conditions as bw_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = bw_ic(tc.grid_type, grid, z_coord)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Barotropic Wave ({tc.grid_type})", total_days=days)

    # All grids now use same physical units
    eta_max = diag["max_abs_eta"][-1] if diag.get("max_abs_eta") else 0
    notes = f"max|eta|={eta_max:.4f} m"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Barotropic Wave {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_abs_eta": "m",
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Wind-Driven Gyre
# ===========================================================================

def _run_gyre_experiment(tc: TestCase, output_dir: Path, days: float,
                         wind_profile: str, label: str,
                         wind_buffer_deg: float = 0.0,
                         ) -> tuple[str, float, str]:
    """Shared runner for barotropic gyre experiments."""
    _supported = ("cubed_sphere", "latlon", "mpas", "mpas_regional",
                   "latlon_regional", "cs_regional")
    if tc.grid_type not in _supported:
        raise NotImplementedError(
            f"{label} not implemented for {tc.grid_type} grid "
            f"(no surface forcing support)")

    from legoesm.ocean.experiments.regional_gyre import (
        RegionalGyreConfig, create_initial_conditions as gyre_ic, create_forcings as gyre_forcings)
    gyre_config = RegionalGyreConfig(
        wind_profile=wind_profile, wind_buffer_deg=wind_buffer_deg)
    physics = gyre_forcings(tc.grid_type, None, gyre_config)
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=gyre_config.A_h))
    state = gyre_ic(tc.grid_type, grid, z_coord, gyre_config)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg,
                                  include_velocity_3d=True)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"{label} ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    notes = f"max speed={max_speed:.4f} m/s, eta drift={eta_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})

    field_specs = [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("speed_sfc", "Surface speed (m/s)", "magma"),
    ]

    case_label = f"{label} {tc.grid_type} {tc.resolution}"
    # For regional grids, pass domain extent so plots are cropped correctly
    # (the auto-crop heuristic fails on unstructured meshes because
    # nearest-neighbour regridding bleeds beyond the domain).
    is_regional = tc.grid_type in ("mpas_regional", "latlon_regional",
                                    "cs_regional")
    extent = (0.0, 120.0, 15.0, 75.0) if is_regional else None
    _save_case_diagnostics(
        output_dir, case_label,
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=field_specs,
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU"},
        domain_extent=extent,
        mesh=grid if coord_kind == "mpas" else None)

    _save_velocity_profiles(output_dir, case_label, snapshots, dt,
                            depth, level_label="Depth (m)")

    return "PASS" if ok else "FAIL", wall, notes


def run_barotropic_gyre(tc: TestCase, output_dir: Path, days: float
                        ) -> tuple[str, float, str]:
    """Wind-driven single barotropic gyre (Stommel 1948, Munk 1950)."""
    return _run_gyre_experiment(tc, output_dir, days,
                                wind_profile="single_gyre",
                                label="Barotropic Gyre")


def run_barotropic_double_gyre(tc: TestCase, output_dir: Path, days: float
                               ) -> tuple[str, float, str]:
    """Wind-driven barotropic double gyre (Holland & Lin 1975) — cosine wind."""
    return _run_gyre_experiment(tc, output_dir, days,
                                wind_profile="double_gyre",
                                label="Barotropic Double Gyre")


def run_barotropic_double_gyre_sin2(tc: TestCase, output_dir: Path, days: float
                                    ) -> tuple[str, float, str]:
    """Wind-driven barotropic double gyre with sin² wind profile."""
    return _run_gyre_experiment(tc, output_dir, days,
                                wind_profile="double_gyre_sin2",
                                wind_buffer_deg=5.0,
                                label="Barotropic Double Gyre sin2")


def _run_baroclinic_gyre(tc: TestCase, output_dir: Path, days: float,
                         gyre_config=None, label: str = "Baroclinic Gyre",
                         ) -> tuple[str, float, str]:
    """Shared runner for baroclinic gyre experiments with different wind profiles."""
    if tc.grid_type not in ("mpas_regional", "latlon_regional"):
        raise NotImplementedError(
            f"Baroclinic gyre only implemented for regional grids, not {tc.grid_type}")

    from legoesm.ocean.experiments.baroclinic_gyre import (
        BaroclinicGyreConfig, create_initial_conditions, create_forcings)

    if gyre_config is None:
        gyre_config = BaroclinicGyreConfig()
    physics = create_forcings(tc.grid_type, None, gyre_config)

    grid, z_coord, ocean_config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=gyre_config.A_h))

    state = create_initial_conditions(tc.grid_type, grid, z_coord, gyre_config)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_baroclinic_scalar_fn(tc.grid_type, grid, z_coord, gyre_config)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg,
                                  include_velocity_3d=True)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"{label} ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_list = diag.get("mean_T", [])
    T_drift = (abs(T_list[-1] - T_list[0])
               if len(T_list) >= 2 else 0.0)

    # Extract baroclinic diagnostics
    dT_ns_surface = diag.get("dT_ns_surface", [0])[-1] if diag.get("dT_ns_surface") else 0
    dT_ns_thermocline = diag.get("dT_ns_thermocline", [0])[-1] if diag.get("dT_ns_thermocline") else 0
    T_spatial_std_surface = diag.get("T_spatial_std_surface", [0])[-1] if diag.get("T_spatial_std_surface") else 0
    T_spatial_std_thermocline = diag.get("T_spatial_std_thermocline", [0])[-1] if diag.get("T_spatial_std_thermocline") else 0

    notes = (f"max_speed={max_speed:.4f}m/s, eta_drift={eta_drift:.2e}, "
             f"T_drift={T_drift:.3f}degC, dT_NS_sfc={dT_ns_surface:.6f}degC, "
             f"dT_NS_thermo={dT_ns_thermocline:.6f}degC")

    # Save results
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full
    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "max_speed": max_speed, "eta_drift": eta_drift, "T_drift": T_drift,
        "depth": depth.tolist(), "notes": notes,
    })

    # Regional extent for proper plotting
    extent = (gyre_config.lon_west, gyre_config.lon_east,
              gyre_config.lat_south, gyre_config.lat_north)
    _save_case_diagnostics(
        output_dir, f"{label} {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("speed_sfc", "Surface speed (m/s)", "magma"),
            ("u_sfc", "Zonal velocity (m/s)", "RdBu_r"),
            ("v_sfc", "Meridional velocity (m/s)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("w_133m", "Vertical velocity at 134m (m/s)", "RdBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU",
                      "dT_ns_surface": "degC", "dT_ns_thermocline": "degC",
                      "T_north_surface": "degC", "T_south_surface": "degC",
                      "T_north_thermocline": "degC", "T_south_thermocline": "degC",
                      "T_spatial_std_surface": "degC", "T_spatial_std_thermocline": "degC"},
        domain_extent=extent,
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_baroclinic_gyre(tc: TestCase, output_dir: Path, days: float
                       ) -> tuple[str, float, str]:
    """Regional baroclinic gyre with sin² wind profile (default)."""
    return _run_baroclinic_gyre(tc, output_dir, days,
                                label="Baroclinic Gyre")


def run_baroclinic_gyre_cos(tc: TestCase, output_dir: Path, days: float
                            ) -> tuple[str, float, str]:
    """Regional baroclinic gyre with cosine wind profile (no taper)."""
    from legoesm.ocean.experiments.baroclinic_gyre import BaroclinicGyreConfig
    config_ = BaroclinicGyreConfig(wind_profile="double_gyre", wind_buffer_deg=0.0)
    return _run_baroclinic_gyre(tc, output_dir, days,
                                gyre_config=config_,
                                label="Baroclinic Gyre cos")

# ===========================================================================
# Runner: Global Wind-Driven Circulation
# ===========================================================================

def run_global_barotropic_wind(tc: TestCase, output_dir: Path, days: float
                                ) -> tuple[str, float, str]:
    """Global barotropic wind-driven circulation with simplified continent.

    Tests the barotropic response to a global 3-belt wind stress
    (trades, westerlies, polar easterlies) in a basin with:
    - One meridional continent (30-90°E) from the north polar cap to 55°S
    - Open Drake Passage south of 55°S → circumpolar current
    - Polar caps (land poleward of ±80°)

    Expected features: subtropical/subpolar gyres in Atlantic-like and
    Pacific-like basins, western boundary currents, and ACC-like flow.
    """
    if tc.grid_type not in ("cubed_sphere", "latlon", "mpas"):
        raise NotImplementedError(
            f"Global wind not implemented for {tc.grid_type}")

    from legoesm.ocean.experiments.global_barotropic_wind import (
        GlobalBarotropicWindConfig, create_initial_conditions as gbw_ic,
        create_forcings as gbw_forcings)
    gbw_config = GlobalBarotropicWindConfig()
    physics = gbw_forcings(tc.grid_type, None, gbw_config)
    nlev_override = tc.run_kwargs.get("nlev", None)
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=gbw_config.A_h, nlev=nlev_override))
    state = gbw_ic(tc.grid_type, grid, z_coord, gbw_config)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Global Wind ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    notes = f"max speed={max_speed:.4f} m/s, eta drift={eta_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})

    _save_case_diagnostics(
        output_dir, f"Global Wind {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("speed_sfc", "Surface speed (m/s)", "magma"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Geostrophic Adjustment
# ===========================================================================

def run_geostrophic_adjustment(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Geostrophic adjustment: meridional temperature front relaxation."""
    from legoesm.ocean.experiments.geostrophic_adjustment import create_initial_conditions as ga_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = ga_ic(tc.grid_type, grid, z_coord)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Geostrophic Adj ({tc.grid_type})", total_days=days)

    # All grids now use same physical units
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Geostrophic Adj {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("SSS", "SSS (PSU)", "YlGnBu"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Phillips Two-Layer
# ===========================================================================

def run_phillips_two_layer(tc: TestCase, output_dir: Path, days: float
                           ) -> tuple[str, float, str]:
    """Phillips two-layer baroclinic test with zonal-mean relaxation."""
    from legoesm.ocean.experiments.phillips_two_layer import create_initial_conditions as p2l_ic
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=2, H_max=3500.0))
    state = p2l_ic(tc.grid_type, grid, z_coord)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    # Relaxation forcing toward target temperature profiles
    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            sh_analysis, sh_analysis_3d, sh_synthesis_3d)
        lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        _, lat_2d = np.meshgrid(lon, lat, indexing='xy')
        T_star_upper = 16.0 - 10.0 * np.sin(np.radians(lat_2d)) ** 2
        T_star_lower = 8.0 - 4.0 * np.sin(np.radians(lat_2d)) ** 2
        T_star_3d = np.stack([T_star_upper, T_star_lower], axis=-1)
        T_star_hat = sh_analysis_3d(grid, jnp.array(T_star_3d))
        tau_relax = 15.0 * 86400.0
        drag_factor = float(jnp.exp(-dt / (25.0 * 86400.0)))

        def forcing_fn(s, dt_):
            from legoesm.core.field import Field
            T_hat = s.T_hat.data
            dT_hat = -(T_hat - T_star_hat) / tau_relax
            new_T_hat = T_hat + dt_ * dT_hat
            # Spectral ocean uses vor_hat/div_hat, not u_hat/v_hat
            new_vor_hat = s.vor_hat.data * drag_factor
            new_div_hat = s.div_hat.data * drag_factor
            return s._replace(
                T_hat=Field(new_T_hat),
                vor_hat=Field(new_vor_hat),
                div_hat=Field(new_div_hat))

    else:
        if tc.grid_type == "mpas":
            lat_rad = np.asarray(grid.latCell, dtype=np.float64)
        elif tc.grid_type == "latlon":
            # lat is 1D (n_lat,) — broadcast to (n_lat, n_lon)
            lat_1d = np.asarray(grid.lat, dtype=np.float64)
            n_lon = grid.n_lon if hasattr(grid, "n_lon") else grid.lon.shape[0]
            lat_rad = np.broadcast_to(lat_1d[:, None], (lat_1d.size, n_lon))
        else:
            lat_rad = np.asarray(grid.lat, dtype=np.float64)
        T_star_upper = jnp.array(16.0 - 10.0 * np.sin(lat_rad) ** 2)
        T_star_lower = jnp.array(8.0 - 4.0 * np.sin(lat_rad) ** 2)
        tau_relax = 15.0 * 86400.0
        drag_factor = float(jnp.exp(-dt / (25.0 * 86400.0)))

        _is_mpas = (tc.grid_type == "mpas")

        def forcing_fn(s, dt_):
            from legoesm.core.field import Field
            T_data = s.T.data
            mask = s.land_mask.data
            dT0 = -(T_data[..., 0] - T_star_upper) / tau_relax * mask
            dT1 = -(T_data[..., 1] - T_star_lower) / tau_relax * mask
            T_new = T_data.at[..., 0].set(T_data[..., 0] + dt_ * dT0)
            T_new = T_new.at[..., 1].set(T_data[..., 1] + dt_ * dT1)
            # For MPAS, u is on edges — can't multiply by cell mask,
            # and there is no separate v field.
            u_new = s.u.data * drag_factor
            if _is_mpas:
                return s._replace(
                    T=Field(T_new),
                    u=Field(u_new))
            mask_3d = mask[..., jnp.newaxis]
            u_new = u_new * mask_3d
            v_new = s.v.data * drag_factor * mask_3d
            return s._replace(
                T=Field(T_new),
                u=Field(u_new),
                v=Field(v_new))

    def step_fn(s, dt_):
        s = model.step(s, dt_)
        return forcing_fn(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Phillips 2-layer ({tc.grid_type})", total_days=days)

    # All grids now use same physical units
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Phillips 2-Layer {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Helper: cell-center lat/lon for any grid type
# ===========================================================================

def _get_cell_latlon_rad(grid_type, grid):
    """Return (lat, lon) in radians, broadcast to match cell shape."""
    if grid_type == "mpas":
        return (np.asarray(grid.latCell, dtype=np.float64),
                np.asarray(grid.lonCell, dtype=np.float64))
    elif grid_type in ("latlon", "spectral"):
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        return lat_2d, lon_2d
    else:  # cubed_sphere
        return (np.asarray(grid.lat, dtype=np.float64),
                np.asarray(grid.lon, dtype=np.float64))


# ===========================================================================
# Runner: Inertia-Gravity Wave (Bishnu et al. 2024)
# ===========================================================================
# Reference: Bishnu et al. (2024), "A Verification Suite of Test Cases for
# the Barotropic Solver of Ocean Models", JAMES.
# DOI: 10.1029/2022MS003545
#
# Sinusoidal inertia-gravity (Poincare) wave on the sphere.
# Analytical dispersion: omega^2 = f^2 + g*H*(kx^2 + ky^2)
# Tests the barotropic pressure-gradient and Coriolis terms.
# ===========================================================================

def run_inertia_gravity_wave(tc: TestCase, output_dir: Path, days: float
                              ) -> tuple[str, float, str]:
    """Bishnu et al. 2024: inertia-gravity (Poincare) wave propagation.

    Single-level (SW-equivalent) ocean model with sinusoidal IGW initial
    condition. Measures L2 error against analytical solution and checks
    dispersion properties.
    """
    from legoesm.ocean.experiments.inertia_gravity_wave import create_initial_conditions as igw_ic
    H_max = 1000.0  # equivalent depth (m)
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=2, H_max=H_max))
    state = igw_ic(tc.grid_type, grid, z_coord)

    # Store initial eta for error computation
    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis
        eta_init = np.asarray(
            sh_synthesis(grid, state.eta_hat.data), dtype=np.float64)
    else:
        eta_init = np.asarray(state.eta.data, dtype=np.float64)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"IGW ({tc.grid_type})", total_days=days)

    # Compute analytical solution at t_final
    t_final = days * 86400.0
    f0 = 1.0e-4
    kx, ky = 2.0, 2.0
    k_phys = kx / config._A_EARTH
    l_phys = ky / config._A_EARTH
    omega = np.sqrt(f0**2 + config._G_EARTH * H_max * (k_phys**2 + l_phys**2))
    lat, lon = _get_cell_latlon_rad(tc.grid_type, grid)
    eta_exact = np.cos(kx * lon + ky * lat - omega * t_final)

    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis
        eta_final = np.asarray(
            sh_synthesis(grid, state.eta_hat.data), dtype=np.float64)
    else:
        eta_final = np.asarray(state.eta.data, dtype=np.float64)

    l2_err = float(np.sqrt(np.mean((eta_final - eta_exact)**2)) /
                   max(np.sqrt(np.mean(eta_exact**2)), 1e-30))
    max_eta = float(np.max(np.abs(eta_final)))
    notes = f"L2={l2_err:.4f}, max|eta|={max_eta:.3f}m, omega={omega:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "H_max": H_max,
        "reference": "Bishnu et al. 2024, DOI:10.1029/2022MS003545",
        "L2_error": l2_err, "omega_analytical": omega,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"IGW Bishnu {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[("eta", "SSH (m)", "RdBu_r")],
        vol_key="mean_eta",
        heat_key="mean_T",
        scalar_units={"mean_eta": "m"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Lock Exchange (NEMO / Petersen et al. 2015)
# ===========================================================================
# Reference: Petersen et al. (2015), Ocean Modelling 86, 93-113.
# DOI: 10.1016/j.ocemod.2014.12.004
# Also: Ilicak et al. (2012), Ocean Modelling 45-46, 37-49.
# NEMO test cases: https://sites.nemo-ocean.io/user-guide/tests.html
#
# Two fluids of different densities separated by a vertical front.
# Dense cold water on one side, light warm water on the other.
# Gravity currents form when the "lock" is removed (t=0).
# Key diagnostic: Reference Potential Energy (RPE) measures spurious mixing.
# RPE(t) = g * integral(rho * z_star dV) where z_star is the equilibrium
# parcel height in a minimum-energy sorted state.
# ===========================================================================

def _compute_rpe(state, grid_type, grid, z_coord):
    """Compute Reference Potential Energy (Ilicak et al. 2012).

    RPE = g * sum(rho_sorted * z_ref * dz * area)
    Approximation: sort density profile at each column and compute
    domain-integrated rho * z.
    """
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d
        T = np.asarray(sh_synthesis_3d(grid, state.T_hat.data), dtype=np.float64)
        S = np.asarray(sh_synthesis_3d(grid, state.S_hat.data), dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)
    elif grid_type == "mpas":
        T = np.asarray(state.T.data, dtype=np.float64)
        S = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.areaCell, dtype=np.float64)
    else:
        T = np.asarray(state.T.data, dtype=np.float64)
        S = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)

    # Compute density at each point using linearized EOS
    from legoesm.ocean.eos import linear_eos
    rho = np.asarray(linear_eos(
        jnp.array(T), jnp.array(S), jnp.zeros_like(jnp.array(T)),
        rho_ref=1025.0, alpha_T=2.0e-4, beta_S=0.0, T_ref=15.0,
    ), dtype=np.float64)

    # Potential energy: PE = g * sum(rho * z * dz * area)
    # For RPE, we'd sort density globally, but as approximation compute PE
    spatial_shape = T.shape[:-1]
    area_bc = area.reshape(spatial_shape)
    pe = 0.0
    for k in range(len(z_full)):
        pe += float(np.nansum(rho[..., k] * z_full[k] * dz[k] * area_bc))
    return config._G_EARTH * pe


def run_lock_exchange(tc: TestCase, output_dir: Path, days: float
                      ) -> tuple[str, float, str]:
    """Lock exchange: density-driven gravity currents (Petersen et al. 2015).

    Cold dense water in western hemisphere, warm light in eastern.
    Monitors potential energy evolution as a proxy for spurious mixing.
    """
    from legoesm.ocean.experiments.lock_exchange import (
        LockExchangeConfig, create_initial_conditions as le_ic)
    le_config = LockExchangeConfig()
    H_max = le_config.H_max  # 500 m shallow basin
    nlev = le_config.nlev     # 20 levels
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=nlev, H_max=H_max))
    state = le_ic(tc.grid_type, grid, z_coord, le_config)

    # Compute initial PE
    pe_init = _compute_rpe(state, tc.grid_type, grid, z_coord)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    # Custom scalar function that includes PE
    base_scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)

    def scalar_fn(s):
        scalars = base_scalar_fn(s)
        pe = _compute_rpe(s, tc.grid_type, grid, z_coord)
        scalars["PE"] = pe
        if abs(pe_init) > 1e-30:
            scalars["PE_rel"] = (pe - pe_init) / abs(pe_init)
        else:
            scalars["PE_rel"] = 0.0
        return scalars

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Lock Exchange ({tc.grid_type})", total_days=days,
        blowup_threshold=200.0)

    pe_drift = _compute_drift(diag.get("PE", []))
    pe_rel_final = diag["PE_rel"][-1] if diag.get("PE_rel") else 0.0
    notes = f"PE drift={pe_drift:.2e}, PE_rel_final={pe_rel_final:.4e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": nlev, "H_max": H_max,
        "reference": "Petersen et al. 2015, DOI:10.1016/j.ocemod.2014.12.004",
        "PE_drift": pe_drift, "PE_rel_final": pe_rel_final,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Lock Exchange {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "PE": "J",
                      "PE_rel": ""},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Overflow (NEMO / Petersen et al. 2015)
# ===========================================================================
# Reference: Petersen et al. (2015), Ocean Modelling 86, 93-113.
# DOI: 10.1016/j.ocemod.2014.12.004
# NEMO test cases: https://sites.nemo-ocean.io/user-guide/tests.html
#
# Dense water on a shallow shelf overflows and descends a continental slope.
# Tests numerical mixing near sloping topography.
# Adapted to global grids: cold dense water at high latitudes flows
# equatorward over a mid-latitude bathymetric ridge.
# ===========================================================================

def run_overflow(tc: TestCase, output_dir: Path, days: float
                 ) -> tuple[str, float, str]:
    """Overflow: dense water descending a bathymetric slope (Petersen et al. 2015).

    Cold dense water at high latitudes flows equatorward over a mid-latitude
    ridge. Monitors PE evolution and plume descent.
    """
    from legoesm.ocean.experiments.overflow import (
        OverflowConfig, create_initial_conditions as ov_ic)
    ov_config = OverflowConfig()
    H_max = ov_config.H_max  # 2000 m
    nlev = ov_config.nlev     # 20 levels
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=nlev, H_max=H_max))
    state = ov_ic(tc.grid_type, grid, z_coord, ov_config)

    pe_init = _compute_rpe(state, tc.grid_type, grid, z_coord)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 30)

    check_fn = _make_check_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)
    base_scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)

    def scalar_fn(s):
        scalars = base_scalar_fn(s)
        pe = _compute_rpe(s, tc.grid_type, grid, z_coord)
        scalars["PE"] = pe
        if abs(pe_init) > 1e-30:
            scalars["PE_rel"] = (pe - pe_init) / abs(pe_init)
        else:
            scalars["PE_rel"] = 0.0
        return scalars

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Overflow ({tc.grid_type})", total_days=days,
        blowup_threshold=200.0)

    pe_drift = _compute_drift(diag.get("PE", []))
    pe_rel_final = diag["PE_rel"][-1] if diag.get("PE_rel") else 0.0
    # All grids now use same physical units
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = (f"PE drift={pe_drift:.2e}, PE_rel={pe_rel_final:.4e}, "
             f"T drift={T_drift:.2e}")

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": nlev, "H_max": H_max,
        "reference": "Petersen et al. 2015, DOI:10.1016/j.ocemod.2014.12.004",
        "PE_drift": pe_drift, "PE_rel_final": pe_rel_final,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Overflow {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "PE": "J"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Stommel Gyre Tracer (Hecht et al. 2000)
# ===========================================================================
# Reference: Hecht, Wingate, Kasahara (2000), "A better, more discriminating
# test problem for ocean tracer transport", Ocean Modelling 2, 1-15.
# DOI: 10.1016/S1463-5003(00)00004-4
#
# Wind-driven Stommel gyre with a passive tracer (salinity field).
# The tracer blob is advected through the highly sheared western boundary
# current, which is a severe test of advection scheme accuracy.
# Based on MITgcm barotropic gyre setup:
#   - Domain: global (~1200 km effective gyre scale)
#   - Wind: tau_x = -tau0 * cos(pi * y / L_y)
#   - Viscosity: A_h to resolve Munk layer
#   - Linear bottom drag
# ===========================================================================

def run_stommel_gyre_tracer(tc: TestCase, output_dir: Path, days: float
                             ) -> tuple[str, float, str]:
    """Stommel gyre with passive tracer (Hecht et al. 2000).

    Wind-driven gyre with a salinity blob advected through the western
    boundary current. Monitors tracer conservation (integral, min, max)
    and transport through the sheared flow.
    """
    if tc.grid_type not in ("cubed_sphere", "latlon", "mpas"):
        raise NotImplementedError(
            f"Stommel gyre tracer not implemented for {tc.grid_type} grid "
            f"(no surface forcing support)")

    from legoesm.ocean.experiments.regional_gyre import (
        RegionalGyreConfig, create_forcings as gyre_forcings)
    from legoesm.ocean.experiments.stommel_gyre_tracer import create_initial_conditions as sgt_ic
    gyre_config = RegionalGyreConfig(wind_profile="single_gyre")
    physics = gyre_forcings(tc.grid_type, None, gyre_config)
    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics))
    state = sgt_ic(tc.grid_type, grid, z_coord)

    # Store initial tracer integral for conservation check
    area = np.asarray(grid.grid_area, dtype=np.float64)
    S_init_sfc = np.asarray(state.S.data[..., 0], dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    S_integral_init = float(np.sum(S_init_sfc * area * mask))
    S_min_init = float(np.min(S_init_sfc[mask > 0.5]))
    S_max_init = float(np.max(S_init_sfc[mask > 0.5]))

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)
    base_scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)

    def scalar_fn(s):
        scalars = base_scalar_fn(s)
        S_sfc = np.asarray(s.S.data[..., 0], dtype=np.float64)
        ocean = mask > 0.5
        scalars["S_min"] = float(np.min(S_sfc[ocean]))
        scalars["S_max"] = float(np.max(S_sfc[ocean]))
        scalars["S_integral"] = float(np.sum(S_sfc * area * mask))
        if abs(S_integral_init) > 1e-30:
            scalars["S_integral_rel"] = (
                (scalars["S_integral"] - S_integral_init) / abs(S_integral_init))
        else:
            scalars["S_integral_rel"] = 0.0
        return scalars

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Stommel Tracer ({tc.grid_type})", total_days=days)

    S_int_drift = _compute_drift(diag.get("S_integral", []))
    S_min_final = diag["S_min"][-1] if diag.get("S_min") else 0
    S_max_final = diag["S_max"][-1] if diag.get("S_max") else 0
    # Check for new extrema (overshoots/undershoots)
    overshoot = max(0, S_max_final - S_max_init)
    undershoot = max(0, S_min_init - S_min_final)
    notes = (f"S integral drift={S_int_drift:.2e}, "
             f"overshoot={overshoot:.3f}, undershoot={undershoot:.3f}")

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "reference": "Hecht et al. 2000, DOI:10.1016/S1463-5003(00)00004-4",
        "S_integral_drift": S_int_drift,
        "S_overshoot": overshoot, "S_undershoot": undershoot,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})

    field_specs = [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("SSS", "SSS (PSU)", "YlGnBu"),
    ]
    if tc.grid_type != "mpas":
        field_specs.append(("speed_sfc", "Surface speed (m/s)", "magma"))

    _save_case_diagnostics(
        output_dir, f"Stommel Tracer {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=field_specs,
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU",
                      "S_min": "PSU", "S_max": "PSU", "S_integral": "PSU*m^2"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Eady Baroclinic Instability
# ===========================================================================

def run_eady_instability(tc: TestCase, output_dir: Path, days: float
                         ) -> tuple[str, float, str]:
    """Eady baroclinic instability with meridional temperature front.

    Regional channel setup with thermal-wind-balanced initial velocity,
    no wind forcing, and a small SSH perturbation to seed instability.
    """
    if tc.grid_type not in ("mpas_channel", "latlon_channel"):
        raise NotImplementedError(
            f"Eady instability only for channel grids, not {tc.grid_type}")

    from legoesm.ocean.experiments.eady_instability import (
        EadyInstabilityConfig, create_initial_conditions as eady_ic,
        create_forcings as eady_forcings)

    eady_config = EadyInstabilityConfig()
    physics = eady_forcings(tc.grid_type, None, eady_config)

    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=eady_config.A_h))

    state = eady_ic(tc.grid_type, grid, z_coord, eady_config)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Eady Instability ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    T_vals = diag.get("mean_T", [])
    T_drift = abs(T_vals[-1] - T_vals[0]) if len(T_vals) >= 2 else 0
    notes = f"max_speed={max_speed:.4f}m/s, T_drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})

    _save_case_diagnostics(
        output_dir, f"Eady Instability {tc.grid_type} {tc.resolution}",
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
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Classical Eady (uniform N², linear shear)
# ===========================================================================

def run_eady_uniform(tc: TestCase, output_dir: Path, days: float
                     ) -> tuple[str, float, str]:
    """Classical Eady instability: uniform N², linear shear, linear EOS."""
    if tc.grid_type not in ("mpas_channel", "latlon_channel"):
        raise NotImplementedError(
            f"eady_uniform only for channel grids, not {tc.grid_type}")

    from legoesm.ocean.experiments.eady_uniform import (
        EadyUniformConfig, create_initial_conditions as eu_ic,
        create_forcings as eu_forcings)
    from legoesm.ocean.eos import LinearEOSConfig

    eu_config = EadyUniformConfig()
    physics = eu_forcings(tc.grid_type, None, eu_config)

    grid, z_coord, config_, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(
            tc, nlev=20, physics=physics,
            A_h=eu_config.A_h, B_h=eu_config.B_h,
            eos="linear",
            eos_linear=LinearEOSConfig(
                alpha_T=eu_config.alpha_T,
                rho_ref=eu_config.rho_0,
                T_ref=eu_config.T_ref,
                S_ref=eu_config.S_uniform,
            ),
            barotropic_diffusion_alpha=eu_config.barotropic_diffusion_alpha,
        ))

    state = eu_ic(tc.grid_type, grid, z_coord, eu_config)

    dt = config.DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Eady Uniform ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    T_vals = diag.get("mean_T", [])
    T_drift = abs(T_vals[-1] - T_vals[0]) if len(T_vals) >= 2 else 0
    notes = (f"max_speed={max_speed:.4f}m/s, T_drift={T_drift:.2e}, "
             f"Ld={eu_config.Ld_km:.0f}km, tau={eu_config.efolding_days:.0f}d")

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})

    _save_case_diagnostics(
        output_dir, f"Eady Uniform {tc.grid_type} {tc.resolution}",
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
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner dispatch
# ===========================================================================

RUNNERS: dict[str, Callable] = {
    "rest_state_stratified_with_land": run_rest_state,
    "rest_state_uniform_with_land": run_rest_state_uniform_ts,
    "rest_state_stratified_no_land": run_rest_state_no_land,
    "rest_state_uniform_no_land": run_rest_state_uniform_ts_no_land,
    "barotropic_wave": run_barotropic_wave,
    "barotropic_gyre": run_barotropic_gyre,
    "barotropic_double_gyre": run_barotropic_double_gyre,
    "barotropic_double_gyre_sin2": run_barotropic_double_gyre_sin2,
    "baroclinic_gyre": run_baroclinic_gyre,
    "baroclinic_gyre_cos": run_baroclinic_gyre_cos,
    "global_barotropic_wind": run_global_barotropic_wind,
    "global_barotropic_wind_1lev": run_global_barotropic_wind,
    "geostrophic_adjustment": run_geostrophic_adjustment,
    "phillips_two_layer": run_phillips_two_layer,
    "inertia_gravity_wave": run_inertia_gravity_wave,
    "lock_exchange": run_lock_exchange,
    "overflow": run_overflow,
    "stommel_gyre_tracer": run_stommel_gyre_tracer,
    "eady_instability": run_eady_instability,
    "eady_uniform": run_eady_uniform,
}
