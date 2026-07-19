"""Continue an Eady uniform run from a saved restart.npz.

Loads the state from a prior run's ``restart.npz``, rebuilds the
grid/model/physics with matching parameters, and integrates for a
further ``--days`` days. Writes snapshots, diagnostics, a new restart,
and the standard set of plots to a new output directory.

Example (chain 600→1200→1800 ...):

    python scripts/continue_eady_uniform.py \
        --restart-from results/ocean/eady_uniform/latlon_channel/100x50/som_U02_Bh2.3e11_Cs0.2_600d \
        --days 600 \
        --tag som_U02_600to1200d

    python scripts/continue_eady_uniform.py \
        --restart-from results/ocean/eady_uniform/latlon_channel/100x50/som_U02_600to1200d \
        --days 600 \
        --tag som_U02_1200to1800d

Assumes the physics parameters match the original 600 d weak-forcing run
(U_surface=0.2, B_h=2.3e11, C_smag=0.2, tracer_advection=som, no sponge,
dt=300, 100x50x20). Use ``--override`` for any changes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import set_policy, PrecisionPolicy
set_policy(PrecisionPolicy.fp64())

from legoesm.core.field import Field
from legoesm.grids.operators_latlon_cgrid import (
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig, compute_sponge_mask,
    create_forcings as eu_forcings,
)
from legoesm.core.field import Field as CoreField  # alias just for clarity
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig

# Test-matrix helpers for running the loop and plotting.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matrix"))
from ocean_test_matrix.timeloop import _run_timeloop
from ocean_test_matrix.extraction import (
    _make_check_fn, _make_scalar_fn, _make_extract_fn, _key_array_fn,
)
from ocean_test_matrix.diagnostic_io import (
    save_restart, _save_case_diagnostics, _save_cross_sections,
    _save_velocity_profiles, _write_results_txt,
)
from ocean_test_matrix.setup import _create_ocean_setup
from ocean_test_matrix.testcase import TestCase
from legoesm.ocean.eos import LinearEOSConfig


def _field(arr, name, dims, units=""):
    return Field(jnp.asarray(arr, dtype=jnp.float64), name=name, dims=dims, units=units)


def _state_from_restart(restart, grid, z_coord) -> LatLonCGridOceanState:
    """Rebuild LatLonCGridOceanState from a restart.npz file."""
    kw = dict(
        u=_field(restart["u"], "u", ("lat", "lon+1", "z"), "m/s"),
        v=_field(restart["v"], "v", ("lat+1", "lon", "z"), "m/s"),
        T=_field(restart["T"], "T", ("lat", "lon", "z"), "degC"),
        S=_field(restart["S"], "S", ("lat", "lon", "z"), "PSU"),
        eta=_field(restart["eta"], "eta", ("lat", "lon"), "m"),
        H_bathy=_field(restart["H_bathy"], "H_bathy", ("lat", "lon"), "m"),
        land_mask=_field(restart["land_mask"], "land_mask", ("lat", "lon")),
        u_mask=_field(restart["u_mask"], "u_mask", ("lat", "lon+1")),
        v_mask=_field(restart["v_mask"], "v_mask", ("lat+1", "lon")),
        w=_field(restart["w"], "w", ("lat", "lon", "z"), "m/s"),
    )
    if "T_som" in restart.files:
        kw["T_som"] = _field(restart["T_som"], "T_som", ("lat", "lon", "z", "mom"))
    if "S_som" in restart.files:
        kw["S_som"] = _field(restart["S_som"], "S_som", ("lat", "lon", "z", "mom"))
    return LatLonCGridOceanState(**kw)


def _build_config_and_model(args):
    """Build the EadyUniformConfig + grid + model for continuation.

    Reuses the test-matrix `_create_ocean_setup` so physics coupling and
    barotropic setup exactly match the original 600-day run.
    """
    cfg = EadyUniformConfig(
        U_surface=args.U_surface,
        B_h=args.B_h,
        C_smag=args.C_smag,
        A_h=args.A_h,
        K_h=args.K_h,
        K_bih=args.K_bih,
        tracer_advection=args.tracer_advection,
    )
    # Physics forcing (wind, heat flux, etc.) for Eady is empty, but the
    # call matches how run_eady_uniform builds it, then KPP is layered on.
    physics = eu_forcings("latlon_channel", None, cfg)
    physics = physics._replace(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
    )

    tc = TestCase(
        case="eady_uniform",
        grid_type="latlon_channel",
        resolution=f"{args.n_lat}x{args.n_lon}",
        duration_days=args.days,
        quick_days=args.days,
        run_kwargs=dict(
            lat_south=cfg.lat_south, lat_north=cfg.lat_north,
            lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        ),
    )

    grid, z_coord, model_cfg, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(
            tc, nlev=args.levels, physics=physics,
            A_h=cfg.A_h, B_h=cfg.B_h, C_smag=cfg.C_smag,
            K_h=cfg.K_h, K_bih=cfg.K_bih,
            A_v=cfg.A_v, K_v=cfg.K_v,
            bottom_drag_r=cfg.bottom_drag_coeff,
            eos="linear",
            eos_linear=LinearEOSConfig(
                alpha_T=cfg.alpha_T, rho_ref=cfg.rho_0,
                T_ref=cfg.T_ref_C, S_ref=cfg.S_uniform,
            ),
            barotropic_diffusion_alpha=cfg.barotropic_diffusion_alpha,
            barotropic_div_damp=cfg.barotropic_div_damp,
            tracer_advection=cfg.tracer_advection,
            momentum_advection=args.momentum_advection,
            weno_d_term=False if args.no_weno_d_term else None,
        )
    )
    return cfg, grid, z_coord, model, lon_deg, lat_deg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restart-from", type=Path, required=True,
                   help="Directory containing restart.npz to continue from.")
    p.add_argument("--days", type=float, required=True,
                   help="Number of ADDITIONAL days to integrate.")
    p.add_argument("--tag", type=str, required=True,
                   help="Output tag (directory name).")
    p.add_argument("--dt", type=float, default=300.0)
    p.add_argument("--levels", type=int, default=20)
    p.add_argument("--n-lat", type=int, default=100)
    p.add_argument("--n-lon", type=int, default=50)
    p.add_argument("--U-surface", type=float, default=0.2)
    p.add_argument("--B-h", type=float, default=2.3e11)
    p.add_argument("--C-smag", type=float, default=0.2)
    p.add_argument("--A-h", type=float, default=0.0)
    p.add_argument("--K-h", type=float, default=0.0)
    p.add_argument("--K-bih", type=float, default=0.0)
    p.add_argument("--tracer-advection", type=str, default="som")
    p.add_argument("--momentum-advection", type=str, default=None,
                   choices=[None, "vector_invariant", "weno5", "weno7"],
                   help="Override momentum advection scheme")
    p.add_argument("--no-weno-d-term", action="store_true",
                   help="Disable the WENO D-term in momentum advection")
    p.add_argument("--no-sponge", action="store_true", default=True,
                   help="Weak-forcing runs use no sponge (default).")
    p.add_argument("--n-snaps", type=int, default=20,
                   help="Number of snapshots to save.")
    p.add_argument("--out-base", type=Path,
                   default=Path("results/ocean/eady_uniform/latlon_channel/100x50"))
    args = p.parse_args()

    out_dir = args.out_base / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    cfg, grid, z_coord, model, lon_deg, lat_deg = _build_config_and_model(args)

    restart = np.load(args.restart_from / "restart.npz", allow_pickle=True)
    start_day = float(restart["time_days"])
    start_step = int(restart["step"])
    print(f"Loaded restart from {args.restart_from} — start_day={start_day}, step={start_step}")

    state = _state_from_restart(restart, grid, z_coord)
    dt = args.dt
    n_steps = int(args.days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    # Sponge (disabled for weak-forcing runs)
    if not args.no_sponge:
        gamma = compute_sponge_mask(grid, cfg)
        T_init = np.array(state.T.data)
        decay_T = jnp.array(np.exp(-dt * gamma)[..., np.newaxis])
        T_init_jnp = jnp.array(T_init)
        decay_u = jnp.array(np.exp(-dt * np.array(
            interp_cell_to_uface(jnp.array(gamma))))[..., np.newaxis])
        decay_v = jnp.array(np.exp(-dt * np.array(
            interp_cell_to_vface(jnp.array(gamma))))[..., np.newaxis])

    def step_fn(s, dt_):
        s_new = model.step(s, dt_)
        if args.no_sponge:
            return s_new
        T_new = s_new.T.data * decay_T + T_init_jnp * (1.0 - decay_T)
        u_new = s_new.u.data * decay_u
        v_new = s_new.v.data * decay_v
        sponge_kw = dict(
            u=Field(u_new, name="u", dims=s_new.u.dims, units=s_new.u.units),
            v=Field(v_new, name="v", dims=s_new.v.dims, units=s_new.v.units),
            T=Field(T_new, name="T", dims=s_new.T.dims, units=s_new.T.units),
        )
        if getattr(s_new, "T_som", None) is not None:
            sponge_kw["T_som"] = s_new.T_som.replace(
                data=s_new.T_som.data * decay_T[..., jnp.newaxis])
        if getattr(s_new, "S_som", None) is not None:
            sponge_kw["S_som"] = s_new.S_som.replace(
                data=s_new.S_som.data * decay_T[..., jnp.newaxis])
        return s_new._replace(**sponge_kw)

    check_fn = _make_check_fn("latlon_channel")
    scalar_fn = _make_scalar_fn("latlon_channel", grid, z_coord)
    extract_fn = _make_extract_fn("latlon_channel", grid, lon_deg, lat_deg,
                                  include_velocity_3d=True)

    state_out, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, "latlon_channel"),
        label=f"Eady Uniform continuation +{args.days:.0f}d from day {start_day:.0f}",
        total_days=args.days, n_snaps=args.n_snaps,
    )

    # Shift times in snapshots/diagnostics to absolute model-days
    if "times_days" in snapshots:
        snapshots["times_days"] = np.asarray(snapshots["times_days"]) + start_day
    if "steps" in snapshots:
        snapshots["steps"] = np.asarray(snapshots["steps"]) + start_step
    if "time_days" in diag:
        diag["time_days"] = [t + start_day for t in diag["time_days"]]
    if "step" in diag:
        diag["step"] = [s + start_step for s in diag["step"]]

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    T_vals = diag.get("mean_T", [])
    T_drift = abs(T_vals[-1] - T_vals[0]) if len(T_vals) >= 2 else 0
    notes = (f"max_speed={max_speed:.4f}m/s, T_drift={T_drift:.2e}, "
             f"Ld={cfg.Ld_km:.0f}km, tau={cfg.efolding_days:.0f}d "
             f"(continuation from day {start_day:.0f})")

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full
    case_label = f"Eady Uniform latlon_channel {args.n_lat}x{args.n_lon} " \
                 f"(+{args.days:.0f}d from {start_day:.0f})"

    _write_results_txt(out_dir, {
        "test": "eady_uniform", "grid": "latlon_channel",
        "resolution": f"{args.n_lat}x{args.n_lon}",
        "days": args.days, "dt": dt, "levels": args.levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s",
        "start_day": start_day, "end_day": start_day + args.days,
    })

    _save_case_diagnostics(
        out_dir, case_label,
        dt, diag, snapshots, "latlon", lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("speed_sfc", "Surface Speed (m/s)", "plasma"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth, level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=None,
    )
    for fkey in ("u_3d", "speed_3d"):
        _save_cross_sections(
            out_dir, case_label, snapshots, dt, fkey,
            "latlon", lon_deg, lat_deg, depth, "Depth (m)",
        )
    _save_velocity_profiles(out_dir, case_label, snapshots, dt,
                            depth, "Depth (m)")

    end_step = start_step + n_steps
    end_day = start_day + args.days
    save_restart(state_out, out_dir, "latlon_channel", end_step, end_day)
    print(f"Continuation done: {start_day}→{end_day} d, status={'PASS' if ok else 'FAIL'}, "
          f"{wall:.1f}s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
