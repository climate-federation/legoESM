#!/usr/bin/env python
"""DINO (Diabatic Neverworld Ocean) standalone production script.

Replicates the Kamm, Deshayes & Madec (2025, GMD) DINO 1° R1
configuration on the lat-lon Mercator grid OR on an MPAS regional
Voronoi mesh, with full surface forcing (wind + T/S restoring +
Q_sr split + Jerlov SW penetration) and physics (KPP + GM/Redi +
enhanced-diffusion convection).

Quick start::

    JAX_ENABLE_X64=1 python scripts/run_dino.py --days 10
    JAX_ENABLE_X64=1 python scripts/run_dino.py --grid mpas --days 10
    JAX_ENABLE_X64=1 python scripts/plot/plot_dino.py results/dino   # visualize

For the full list of options::

    python scripts/run_dino.py --help

This script is **portable** by design: no project-internal CI hooks,
no test-matrix integration. The output directory is self-contained
(NPZ snapshots + a JSON config dump) so it can be moved to a GPU
machine for production runs.

See ``docs/dev-notes/ocean_experiments_reference.md`` for a 1-minute
orientation and the full scientific configuration, decisions log, and
stability investigation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.experiments.dino import (
    DINO_L2_RECIPES,
    DINO_RECIPES,
    DINOConfig,
    apply_dino_lat_lon_surface_forcing,
    apply_dino_mpas_surface_forcing,
    create_dino_z_star,
    dino_config_for_recipe,
    dino_lat_lon_grid,
    dino_lat_lon_vertical,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
    dino_lat_lon_surface_forcing_arrays,
    nemo_faithful_dino_config,
    dino_mpas_model_config,
    dino_mpas_state,
    dino_mpas_surface_forcing_arrays,
)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", type=Path, default=None,
        help="YAML file of run parameters (keys = the long flag names with "
             "dashes->underscores, e.g. grid, days, mpas_eq_visc_boost). Loaded "
             "as DEFAULTS so any explicit CLI flag still overrides it. Use the "
             "committed scripts/experiment/dino/*.yaml to reproduce a run.",
    )
    p.add_argument(
        "--grid", choices=("latlon", "mpas"), default="latlon",
        help="Horizontal grid: 'latlon' (Mercator) or 'mpas' (regional "
             "Voronoi with periodic_x=True + seam wall).",
    )
    p.add_argument(
        "--recipe", choices=sorted(DINO_RECIPES), default=None,
        help="Select a named MODEL RECIPE (pure config overlay of another "
             "ocean model's DINO numerics — EOS, advection/limiter, vertical "
             "mixing, momentum/Coriolis/integrator blocks). L1 cards: "
             "'legoesm_default' (Wright+KPP, the identity), 'nemo_paper' "
             "(Kamm et al. 2025: nemo_seos + TKE). L2 cards: 'veros' "
             "(Vallis EOS + TKE + superbee + streamfunction/AB2), 'mitgcm' "
             "(JMD95-ish EOS + KPP + DST3 + flux-form/AB2/unsplit-FS), "
             "'oceananigans' (TEOS-10 + CATKE + WENO + AB2). Applied FIRST; "
             "explicit --eos/--vmix/--barotropic-solver still override it. "
             "L2 cards are lat-lon-only (the paper R1 comparison grid).",
    )
    p.add_argument(
        "--n-lon", type=int, default=50,
        help="Zonal cell count for the Mercator grid (lat-lon only; "
             "50 = 1° R1, default).",
    )
    p.add_argument(
        "--mpas-resolution-km", type=float, default=97.0,
        help="Cell-spacing target for the MPAS Voronoi mesh "
             "(MPAS only). Default 97 km gives 9686 cells, matching "
             "the 50-col Mercator basin's 9900 cells within 2%% -- cross-"
             "grid comparison is at equivalent mean cell area. "
             "(Theoretical area-equivalent is ~82 km but the regional "
             "Voronoi generator has quantization gaps below ~85 km; "
             "97 km is the closest working value to 9900-cell match.)",
    )
    p.add_argument(
        "--mpas-eq-visc-boost", type=float, default=None,
        help="MPAS-only equatorial A_h boost factor (DINOConfig."
             "mpas_equatorial_visc_boost, default 8.0). Boosts lateral "
             "viscosity near the equator (tight Gaussian, sigma=5deg) to "
             "constrain the forced f->0 equatorial jet that otherwise runs "
             "away on the implicit-CN MPAS path; ignored on lat-lon.",
    )
    p.add_argument(
        "--vmix", choices=("kpp", "tke", "constant", "richardson", "catke"),
        default=None,
        help="Vertical-mixing closure (DINOConfig.vmix_scheme): 'kpp' "
             "(multi-year-stable default), 'tke' (paper's NEMO scheme — the "
             "default 5e-4 momentum floor fixes the day-39 instability so it "
             "runs to ~day 226, but a 2nd viscosity-insensitive SW-corner mode "
             "NaNs it ~day 230; multi-year needs kpp), or 'constant' "
             "(background-only; also NaNs ~day 230). Both grids.",
    )
    p.add_argument(
        "--eos", choices=("wright", "nemo_seos"), default=None,
        help="Equation of state (DINOConfig.eos): 'wright' (legoESM default, "
             "Wright 1997 full nonlinear EOS) or 'nemo_seos' (the paper/NEMO "
             "simplified S-EOS, Roquet et al. 2015, with the DINO coefficients "
             "— the oracle EOS for the thermocline comparison). Both grids.",
    )
    p.add_argument(
        "--nemo-faithful-grid", action="store_true",
        help="Build the lat-lon grid on NEMO's EXACT DINO R1 mesh "
             "(DINOConfig.nemo_faithful_grid): 48×195 with the equator on a "
             "T-point and faces [1,49] (matches our NEMO 5.0.2 build cell-for-"
             "cell to 3e-6°; 100%% wet/dry-domain agreement), instead of the "
             "legoESM [-50,0]/198×50 default. Co-sets the bathymetry lon frame "
             "+ sill anchor via nemo_faithful_dino_config. Lat-lon only.",
    )
    p.add_argument(
        "--tke-momentum-visc-bg", type=float, default=None,
        help="TKE-only background vertical viscosity FLOOR [m²/s] "
             "(DINOConfig.tke_momentum_visc_bg, default 5e-4 = 4× the paper "
             "1.2e-4). Damps the SW channel-corner surface-momentum instability "
             "that NaNs our TKE at the paper value; applied (max with A_v_bg) "
             "only when --vmix tke. kpp/constant ignore it. Lower it (e.g. "
             "1.2e-4) to run TKE at the unstable paper viscosity.",
    )
    p.add_argument(
        "--evd-momentum", choices=("on", "off"), default=None,
        help="Enhanced vertical diffusion on MOMENTUM (DINOConfig."
             "evd_on_momentum; NEMO nn_evdm=1, the DINO namelist setting). "
             "Default on (paper-faithful); effective only for --vmix "
             "tke/constant (kpp carries its own convective viscosity — the "
             "combination is rejected). Lat-lon only; MPAS is tracer-only.",
    )
    p.add_argument(
        "--gm-kappa-scheme", choices=("visbeck", "treguier"), default=None,
        help="Adaptive kappa_GM scaling (DINOConfig.gm_kappa_scheme): "
             "'visbeck' (Visbeck 1997, historical default) or 'treguier' "
             "(Treguier 1997 / NEMO nn_aei_ijk_t=21 — the DINO oracle "
             "scaling, cap aei0=rn_Ue*rn_Le=3000 m2/s). Lat-lon only.",
    )
    p.add_argument(
        "--allow-multiyear", action="store_true",
        help="Opt out of the 1-year local-machine cap on --days (use inside "
             "SLURM GPU jobs; the multi-year DINO_R1 comparison runs).",
    )
    p.add_argument(
        "--preset", choices=("r1_exact",), default=None,
        help="Config preset: 'r1_exact' = the DINO_R1 exactness preset "
             "(dino_r1_exact_config: S-EOS, TKE, nemo_quadratic drag, EIV "
             "off, stabilizer floors off, ppm_fct — see "
             "docs/ocean/fidelity/dino_l1_exactness_audit.md).  Individual "
             "flags still override on top.",
    )
    p.add_argument(
        "--bottom-drag-scheme",
        choices=("legacy", "nemo_quadratic", "nemo_loglayer"), default=None,
        help="Bottom-drag law (DINOConfig.bottom_drag_scheme): 'legacy' = "
             "historical MOM6 quadratic-with-floor; 'nemo_quadratic' = "
             "zdfdrg np_non_lin, the DINO reference's namdrg selection "
             "(Cd0*sqrt(u^2+v^2+ke0), Cd0=C_d_bottom); 'nemo_loglayer' = "
             "zdfdrg np_loglayer.",
    )
    p.add_argument(
        "--treguier-aei0", type=float, default=None,
        help="Treguier kappa cap aei0 [m2/s] (DINOConfig.treguier_aei0, "
             "default 3000 = the DINO namelist rn_Ue*rn_Le). Ignored unless "
             "--gm-kappa-scheme treguier.",
    )
    p.add_argument(
        "--barotropic-solver",
        choices=("implicit_cn", "explicit_substep", "rigid_lid",
                 "implicit_unsplit"),
        default=None,
        help="Barotropic (free-surface / rigid-lid) solver "
             "(DINOConfig.barotropic_solver, default 'implicit_cn'). "
             "'rigid_lid' is the lat-lon-C-grid island-streamfunction solver — "
             "the natural ACC solver for the re-entrant channel (the southern "
             "continent below the channel is a topological island), so the "
             "depth-integrated transport is solved directly as the island "
             "circulation instead of riding the free-surface eta solve's null "
             "mode (which under-/over-shoots + oscillates the ACC). "
             "LAT-LON ONLY (MPAS supports implicit_cn / explicit_substep).",
    )
    p.add_argument(
        "--momentum-advection",
        choices=("vector_invariant", "flux_form", "weno5", "weno7", "weno9"),
        default=None,
        help="Horizontal momentum-advection scheme "
             "(DINOConfig.momentum_advection, default 'vector_invariant' = the "
             "AL81 PV-flux vector-invariant form; 'weno7' = the Oceananigans "
             "WENOVectorInvariant card block; 'flux_form' = MITgcm). Overrides "
             "the recipe card — the L2 bisect lever the cards could not "
             "previously isolate from their time-integration blocks.",
    )
    p.add_argument(
        "--coriolis-scheme",
        choices=("matsuno_split", "explicit_ab2"),
        default=None,
        help="Coriolis time-stepping placement (DINOConfig.coriolis_scheme). "
             "'matsuno_split' (default): unconditionally-neutral FB rotation "
             "sub-step. 'explicit_ab2': f×u enters du_dt (requires "
             "--outer ab2 via the recipe card; model validation rejects "
             "unsupported pairings loudly).",
    )
    p.add_argument(
        "--barotropic-slow-forcing-ab2", choices=("on", "off"), default=None,
        help="AB2 time-centering of the barotropic slow forcing F_slow "
             "(DINOConfig.barotropic_slow_forcing_ab2; Oceananigans Gᵁ "
             "convention). REQUIRED for stability when coriolis_scheme="
             "explicit_ab2 pairs with barotropic_solver=implicit_cn (without "
             "it the barotropic-mode Coriolis integrates forward-Euler — the "
             "diagnosed 'oceananigans'-card barotropic blowup). Default: the "
             "recipe card's value ('oceananigans' card = on).",
    )
    p.add_argument(
        "--rigid-lid-dt-mom-ratio", type=float, default=None,
        help="dt_mom under-relaxation ratio for the rigid_lid faithful stack "
             "(DINOConfig.rigid_lid_dt_mom_ratio, default 9.0 = Veros "
             "dt_mom=4800/dt_tracer=43200). dt_mom = dt/ratio accelerates the "
             "ACC spin-up; ignored unless --barotropic-solver rigid_lid.",
    )
    p.add_argument(
        "--days", type=float, default=10.0,
        help="Total simulated duration in days (default 10; capped at "
             "365 by local-machine policy — see plan).",
    )
    p.add_argument(
        "--snapshot-every-days", type=float, default=1.0,
        help="Snapshot interval in days (default 1).",
    )
    p.add_argument(
        "--dt", type=float, default=None,
        help="Baroclinic timestep [s]. Default uses DINOConfig.dt = 2700.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/dino"),
        help="Directory to write snapshots + log. Default 'results/dino' "
             "(relative to CWD). Note: `results/` should be gitignored — "
             "snapshots can be many MB.",
    )
    p.add_argument(
        "--no-forcing", action="store_true",
        help="Run dycore-only from rest (no wind, no restoring) — for "
             "shake-down tests of the model + bathymetry combination.",
    )
    p.add_argument(
        "--physics-off", action="store_true",
        help="Disable KPP / GM-Redi / convection — dycore only.",
    )
    # Two-pass: if --config is given, load the YAML as argparse DEFAULTS, then
    # re-parse so any explicit CLI flag overrides the file. Unknown YAML keys
    # are rejected (typo guard) — only argparse dests are accepted.
    args, _ = p.parse_known_args()
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in p._actions} - {"help"}
        unknown = set(cfg) - valid
        if unknown:
            raise SystemExit(
                f"--config {args.config}: unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(valid - {'config'})}")
        # set_defaults bypasses each action's ``type=``, so coerce Path-typed
        # keys (e.g. output_dir) from their YAML string form ourselves.
        path_dests = {a.dest for a in p._actions if a.type is Path}
        for k in list(cfg):
            if k in path_dests and cfg[k] is not None:
                cfg[k] = Path(cfg[k])
        p.set_defaults(**cfg)
    return p.parse_args()


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def _diagnose(state, grid, grid_kind: str):
    """Compact per-snapshot diagnostics. ``grid_kind`` ∈ {"latlon","mpas"}."""
    cell_mask = np.asarray(state.land_mask.data) > 0.5
    u = np.asarray(state.u.data)
    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    eta = np.asarray(state.eta.data)

    u_max = float(np.max(np.abs(u)))
    eta_max = float(np.max(np.abs(eta)))
    T_max = float(np.max(T[cell_mask, :])) if cell_mask.any() else float("nan")
    T_min = float(np.min(T[cell_mask, :])) if cell_mask.any() else float("nan")
    S_max = float(np.max(S[cell_mask, :])) if cell_mask.any() else float("nan")
    S_min = float(np.min(S[cell_mask, :])) if cell_mask.any() else float("nan")

    if grid_kind == "latlon":
        v = np.asarray(state.v.data)
        v_max = float(np.max(np.abs(v)))
        # u: (n_lat, n_lon+1, nlev), v: (n_lat+1, n_lon, nlev) → cell centers
        u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
        v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        ke = 0.5 * (u_cc ** 2 + v_cc ** 2)
        area = np.asarray(grid.area)            # (n_lat, n_lon)
        ke_total = float(np.sum(ke * area[..., None] * cell_mask[..., None]))
    else:  # mpas
        # u is edge-normal (nEdges, nlev); approximate per-cell KE as
        # the mean of |u|² over the cell's edges. Cheap and good enough
        # for a monitor.
        v_max = float("nan")
        ke_edge = 0.5 * u ** 2                  # (nEdges, nlev)
        area = np.asarray(grid.areaCell)        # (nCells,)
        ke_per_cell = float(np.mean(ke_edge) * np.sum(area * cell_mask))
        ke_total = ke_per_cell

    return {
        "u_max": u_max, "v_max": v_max, "eta_max": eta_max,
        "T_max": T_max, "T_min": T_min,
        "S_max": S_max, "S_min": S_min,
        "ke_total": ke_total,
    }


def _save_snapshot(state, t_seconds, snapshot_dir: Path, idx: int, grid_kind: str):
    """Save a snapshot as NPZ. Handles both lat-lon (has v) and MPAS (no v)."""
    fields = {
        "time_seconds": t_seconds,
        "time_days": t_seconds / 86400.0,
        "eta": np.asarray(state.eta.data),
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data),
        "land_mask": np.asarray(state.land_mask.data),
        "H_bathy": np.asarray(state.H_bathy.data),
    }
    if grid_kind == "latlon":
        fields["v"] = np.asarray(state.v.data)
    np.savez_compressed(snapshot_dir / f"snapshot_{idx:05d}.npz", **fields)


def _save_run_metadata(args, cfg: DINOConfig, grid, z, output_dir: Path,
                       grid_kind: str):
    """Write a JSON file with the run config + grid info."""
    if grid_kind == "latlon":
        grid_info = {
            "kind": "latlon-mercator",
            "n_lat": int(grid.n_lat),
            "n_lon": int(grid.n_lon),
            "radius": float(grid.radius),
            "lat_min_deg": float(np.degrees(np.min(grid.lat))),
            "lat_max_deg": float(np.degrees(np.max(grid.lat))),
            "dx_eq_km": float(np.max(grid.dx) / 1000.0),
            "dx_pole_km": float(np.min(grid.dx) / 1000.0),
        }
    else:
        grid_info = {
            "kind": "mpas-regional-voronoi",
            "nCells": int(grid.nCells),
            "nEdges": int(grid.nEdges),
            "radius": float(grid.radius),
            "median_dcEdge_km": float(np.median(grid.dcEdge) / 1000.0),
        }

    metadata = {
        "args": vars(args),
        "config": dataclasses.asdict(cfg),
        "grid": grid_info,
        "vertical": {
            "n_levels": int(z.n_levels),
            "H_max": float(z.H_max),
            "dz_top": float(z.dz_ref[0]),
            "dz_bot": float(z.dz_ref[-1]),
        },
    }
    metadata["config"]["wind_tau_lats_deg"] = list(cfg.wind_tau_lats_deg)
    metadata["config"]["wind_tau_values"] = list(cfg.wind_tau_values)
    # JSON-serialize every Path-valued arg (output_dir, --config, ...), not just
    # output_dir — a new Path flag must not break the metadata dump.
    metadata["args"] = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in metadata["args"].items()
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = _parse_args()

    # JAX x64 sanity check: DINO uses Wright EOS + barotropic split;
    # both need 64-bit precision to avoid silent eta drift and EOS noise.
    import jax
    if not jax.config.x64_enabled:
        import warnings
        warnings.warn(
            "JAX_ENABLE_X64 is OFF. DINO needs 64-bit for the Wright EOS "
            "and barotropic-baroclinic split — silent precision artifacts "
            "will appear at multi-day integration. Re-run with "
            "`JAX_ENABLE_X64=1 python scripts/run_dino.py ...`.",
            stacklevel=2,
        )

    # Local-machine policy: cap at 1 yr (decision logged in plan).
    # --allow-multiyear is the explicit opt-out for GPU/SLURM jobs (the
    # multi-year DINO_R1 comparison runs).
    if args.days > 365.0 and not args.allow_multiyear:
        raise SystemExit(
            f"--days={args.days} exceeds the 1-year local-machine cap. "
            "Long spin-ups should run on a GPU machine (pass "
            "--allow-multiyear inside a SLURM job) — see plan."
        )

    # Base config: the r1_exact preset (NEMO-DINO exact stack) when selected,
    # else the legoESM defaults.
    if getattr(args, "preset", None) == "r1_exact":
        from legoesm.ocean.experiments.dino import dino_r1_exact_config
        cfg = dino_r1_exact_config()
    else:
        cfg = DINOConfig()
    # A named model recipe (pure config overlay) is applied FIRST; the explicit
    # scheme flags below still override it. L2 cards (veros/mitgcm/oceananigans)
    # select lat-lon-C-grid-only blocks (flux-form/WENO momentum, AB2 outer,
    # rigid-lid / unsplit free surface), so they are rejected on the MPAS mesh —
    # the intercomparison runs on the paper's lat-lon R1 grid.
    if args.recipe is not None:
        if args.recipe in DINO_L2_RECIPES and args.grid != "latlon":
            raise SystemExit(
                f"--recipe {args.recipe} is a lat-lon-only L2 card (it needs "
                f"dycore blocks the MPAS Voronoi mesh does not provide); rerun "
                f"with --grid latlon, or select an L1 card "
                f"({sorted(set(DINO_RECIPES) - DINO_L2_RECIPES)}) on MPAS.")
        cfg = dino_config_for_recipe(args.recipe, base=cfg)
    if args.dt is not None:
        cfg = dataclasses.replace(cfg, dt=args.dt)
    if args.vmix is not None:
        cfg = dataclasses.replace(cfg, vmix_scheme=args.vmix)
    if args.eos is not None:
        cfg = dataclasses.replace(cfg, eos=args.eos)
    if args.tke_momentum_visc_bg is not None:
        cfg = dataclasses.replace(
            cfg, tke_momentum_visc_bg=args.tke_momentum_visc_bg)
    if args.evd_momentum is not None:
        cfg = dataclasses.replace(
            cfg, evd_on_momentum=(args.evd_momentum == "on"))
    if args.gm_kappa_scheme is not None:
        cfg = dataclasses.replace(cfg, gm_kappa_scheme=args.gm_kappa_scheme)
    if args.bottom_drag_scheme is not None:
        cfg = dataclasses.replace(
            cfg, bottom_drag_scheme=args.bottom_drag_scheme)
    if args.treguier_aei0 is not None:
        cfg = dataclasses.replace(cfg, treguier_aei0=args.treguier_aei0)
    if args.mpas_eq_visc_boost is not None:
        cfg = dataclasses.replace(
            cfg, mpas_equatorial_visc_boost=args.mpas_eq_visc_boost)
    if args.barotropic_solver is not None:
        cfg = dataclasses.replace(cfg, barotropic_solver=args.barotropic_solver)
    if args.momentum_advection is not None:
        cfg = dataclasses.replace(
            cfg, momentum_advection=args.momentum_advection)
    if args.coriolis_scheme is not None:
        cfg = dataclasses.replace(cfg, coriolis_scheme=args.coriolis_scheme)
    if args.barotropic_slow_forcing_ab2 is not None:
        cfg = dataclasses.replace(
            cfg,
            barotropic_slow_forcing_ab2=(
                args.barotropic_slow_forcing_ab2 == "on"),
        )
    if args.rigid_lid_dt_mom_ratio is not None:
        cfg = dataclasses.replace(
            cfg, rigid_lid_dt_mom_ratio=args.rigid_lid_dt_mom_ratio)
    if args.nemo_faithful_grid:
        if args.grid != "latlon":
            p.error("--nemo-faithful-grid is lat-lon only (NEMO's Mercator DINO "
                    "mesh); rerun with --grid latlon.")
        # Applied LAST: co-sets the bathymetry lon frame + sill anchor onto
        # whatever recipe/overrides preceded it (must not be clobbered after).
        cfg = nemo_faithful_dino_config(base=cfg)
    dt = cfg.dt
    grid_kind = args.grid

    # The MPAS model validates ONLY {explicit_substep, implicit_cn}; rigid_lid
    # (lat-lon island streamfunction) and implicit_unsplit have no Voronoi-mesh
    # implementation.  Reject any unsupported solver up front with a clear
    # message instead of letting MPASOceanModel raise after building the mesh
    # (dispatch-hardening: a clearer error, earlier).
    _MPAS_BAROTROPIC = ("implicit_cn", "explicit_substep")
    if grid_kind == "mpas" and cfg.barotropic_solver not in _MPAS_BAROTROPIC:
        raise SystemExit(
            f"--barotropic-solver {cfg.barotropic_solver} is not supported on "
            f"the MPAS Voronoi mesh (lat-lon-C-grid-only). On MPAS use one of "
            f"{_MPAS_BAROTROPIC}; rigid_lid / implicit_unsplit require "
            f"--grid latlon.")
    # The MPAS implicit-vertical-mixing path only builds K-profiles for KPP
    # (ocean_model_mpas.py:251); tke/catke/richardson have no MPAS profile
    # builder and would NaN after the mesh is built. Fail loud + early
    # (dispatch-hardening) -- catches e.g. `--recipe nemo_paper --grid mpas`
    # (nemo_paper sets vmix_scheme="tke") and `--grid mpas --vmix tke`.
    _MPAS_VMIX = ("kpp", "constant")
    if grid_kind == "mpas" and cfg.vmix_scheme not in _MPAS_VMIX:
        raise SystemExit(
            f"--vmix {cfg.vmix_scheme} (from --recipe {args.recipe!r} / --vmix) "
            f"is not supported on the MPAS Voronoi mesh: only {_MPAS_VMIX} have "
            f"an MPAS K-profile builder. Use --grid latlon, or override with "
            f"--vmix kpp on MPAS.")

    # Build grid, state, model — branch on grid type
    if grid_kind == "latlon":
        grid = dino_lat_lon_grid(cfg, n_lon=args.n_lon)
        # zstar OR masked_zco (NEMO ln_zco full-cell masking) per
        # cfg.vertical_coordinate — the coordinate drives state + model.
        z = dino_lat_lon_vertical(grid, cfg)
        state = dino_lat_lon_state(grid, z, cfg)
        model_cfg, _ = dino_lat_lon_model_config(
            grid, cfg, physics=not args.physics_off,
        )
        model = LatLonCGridOceanModel(grid, z, model_cfg)
        forcing = (None if args.no_forcing
                   else dino_lat_lon_surface_forcing_arrays(grid, cfg))
        apply_forcing = apply_dino_lat_lon_surface_forcing
        from legoesm.ocean.experiments.dino import dino_step_surface_forcing
        sf_step = (
            dino_step_surface_forcing(forcing)
            if forcing is not None
            and getattr(cfg, "wind_through_step", False)
            else None)
        grid_desc = f"{grid.n_lat}x{grid.n_lon} lat-lon Mercator"
    else:  # mpas
        z = create_dino_z_star(cfg)
        grid = create_regional_voronoi_mesh(
            lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
            lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
            resolution_km=args.mpas_resolution_km,
            periodic_x=True,
        )
        state = dino_mpas_state(grid, z, cfg)
        model_cfg, _ = dino_mpas_model_config(
            grid, cfg, physics=not args.physics_off,
        )
        model = MPASOceanModel(grid, z, model_cfg)
        forcing = (None if args.no_forcing
                   else dino_mpas_surface_forcing_arrays(grid, cfg))
        apply_forcing = apply_dino_mpas_surface_forcing
        if getattr(cfg, "wind_through_step", False):
            raise SystemExit(
                "wind_through_step is wired on the lat-lon DINO path only")
        sf_step = None
        grid_desc = f"{grid.nCells} cells MPAS regional Voronoi"

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.output_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    _save_run_metadata(args, cfg, grid, z, args.output_dir, grid_kind)

    # Time loop
    n_steps_total = int(round(args.days * 86400.0 / dt))
    snapshot_every_steps = max(1, int(round(args.snapshot_every_days * 86400.0 / dt)))

    print(f"DINO {grid_desc}: {z.n_levels} levels, dt={dt}s")
    print(f"Run: {n_steps_total} steps = {args.days:.2f} days, "
          f"snapshot every {snapshot_every_steps} steps "
          f"= {snapshot_every_steps * dt / 86400.0:.2f} days")
    print(f"Forcing: {'OFF (dycore only)' if args.no_forcing else 'wind + T/S restoring (Q_sr split) + Jerlov-I SW penetration'}")
    print(f"Physics: "
          f"{'OFF' if args.physics_off else cfg.vmix_scheme.upper() + ' + GM/Redi + enhanced-diffusion convection'}")
    print(f"Output:  {args.output_dir}")
    print()
    print(f"{'step':>6} {'day':>7} {'|u|':>10} {'|v|':>10} {'|eta|':>10} "
          f"{'T_max':>7} {'T_min':>7} {'KE':>10}")

    # The rigid_lid faithful stack uses the ab2 outer integrator + a
    # streamfunction (ψ, dψ, dψ_prev, dpsin, dpsin_prev) carry; seed them (and
    # the rigid-lid island cache + dtype reconciliation) once from the concrete
    # initial state before the eager step loop, so the first step has a complete
    # carry. lat-lon only (rigid_lid is rejected on MPAS upstream).
    # barotropic_slow_forcing_ab2 (the 'oceananigans' card's Gᵁ AB2
    # time-centering) likewise needs its F_slow_{u,v}_prev carry seeded before
    # step 1 (_step_impl raises on an unseeded prev); seed_scan_carry now does
    # that too. Other stacks keep the eager-loop path byte-identical.
    if grid_kind == "latlon" and (
            cfg.barotropic_solver == "rigid_lid"
            or model_cfg.flat_get("barotropic_slow_forcing_ab2")):
        state = model.seed_scan_carry(state, dt)

    t_wall_start = time.time()
    snapshot_idx = 0
    _save_snapshot(state, 0.0, snapshot_dir, snapshot_idx, grid_kind)

    for k in range(n_steps_total):
        if forcing is not None:
            # NEMO time convention: step k (0-based) ends at t=(k+1)*dt —
            # drives the seasonal forcing phases when forcing_annual_cycle.
            state = apply_forcing(state, forcing, z, cfg, dt,
                                  t_seconds=(k + 1) * dt)

        state = model.step(
            state, dt=dt,
            surface_forcing=(sf_step if getattr(cfg, "wind_through_step",
                                                False) else None))

        is_last = (k == n_steps_total - 1)
        if (k + 1) % snapshot_every_steps == 0 or is_last:
            snapshot_idx += 1
            t_seconds = (k + 1) * dt
            _save_snapshot(state, t_seconds, snapshot_dir, snapshot_idx, grid_kind)
            d = _diagnose(state, grid, grid_kind)
            print(f"{k+1:6d} {t_seconds/86400.0:7.2f} "
                  f"{d['u_max']:10.4e} {d['v_max']:10.4e} {d['eta_max']:10.4e} "
                  f"{d['T_max']:7.2f} {d['T_min']:7.2f} {d['ke_total']:10.4e}")

    wall = time.time() - t_wall_start
    print()
    print(f"Done. Wall time: {wall:.1f}s ({wall/n_steps_total*1000:.1f} ms/step). "
          f"Snapshots: {snapshot_idx + 1}")


if __name__ == "__main__":
    main()
