#!/usr/bin/env python
"""DINO (Diabatic Neverworld Ocean) standalone production script.

Replicates the Kamm, Deshayes & Madec (2025, GMD) DINO 1° R1
configuration on the lat-lon Mercator grid OR on an MPAS regional
Voronoi mesh, with full surface forcing (wind + T/S restoring +
Q_sr split + Jerlov SW penetration) and physics (KPP + GM/Redi +
enhanced-diffusion convection).

Quick start::

    JAX_ENABLE_X64=1 python scripts/run/run_dino.py --days 10
    JAX_ENABLE_X64=1 python scripts/run/run_dino.py --grid mpas --days 10
    JAX_ENABLE_X64=1 python scripts/plot/plot_dino.py results/dino   # visualize

Multi-GPU SPMD (lat-lon only; the state is sharded into latitude bands
across N devices — ``make_sharded_ocean_step``)::

    # Single controller, 4 local GPUs:
    JAX_ENABLE_X64=1 python scripts/run/run_dino.py --n-devices 4

    # One process per GPU (SLURM/srun, NCCL across processes):
    srun -n 8 python scripts/run/run_dino.py --n-devices 8 \
        --multicontroller --allow-multiyear

``n_lat % n_devices`` must be 0: the default R1 grid (198 rows)
divides only by 1 and 2; the R2 grid (``--n-lon 100`` -> 396 rows)
divides by 4. No exact DINO Mercator row count divides by 8 — wider
device counts belong on the synthetic-grid scaling benches
(``scripts/bench/bench_ocean_latlon_spmd_scaling.py``).

Restart from a previous snapshot (lat-lon only; the NPZ written by this
script's snapshot cadence)::

    python scripts/run/run_dino.py --restart-from \
        results/dino/snapshots/snapshot_00010.npz

Ensemble initial-condition perturbation (``--seed``; 0 = the canonical
rest-state IC, identical to the unseeded run)::

    python scripts/run/run_dino.py --seed 7

For the full list of options::

    python scripts/run/run_dino.py --help

This script is **portable** by design: no project-internal CI hooks,
no test-matrix integration. The output directory is self-contained
(NPZ snapshots + a JSON config dump) so it can be moved to a GPU
machine for production runs.

See ``docs/dev-notes/ocean_experiments_reference.md`` for a 1-minute
orientation and the full scientific configuration, decisions log, and
stability investigation.

Numerics note (decision 2026-09-09, NEMO-fidelity campaign): on CPU the XLA
compiler contracts ``a - b*c`` into a fused multiply-add, which NEMO's
reference build never does. The two sites where that broke bit-exactness
(the tracer and momentum implicit-solve sweeps) round the multiply
separately by hand. A global switch exists — ``XLA_FLAGS=--xla_cpu_max_isa=AVX``
(also ``SSE4_2``, or ``--xla_backend_optimization_level=0``) — and was
measured: with the hand fixes in place it changes no operator result, moves
GYRE trajectory rows by <= 7e-13 and no first-failing step, costs ~3% speed.
It was deliberately NOT adopted. If a new fused-multiply-add mismatch appears
in a NEMO-identity comparison, try that flag first to confirm the class
before hand-fixing the site.
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
        "--coriolis-placement", choices=("cell_average", "face_latitude"),
        default=None,
        help="Where the vertex Coriolis is EVALUATED (#1455; "
             "DINOConfig.coriolis_placement). 'cell_average' (default) is the "
             "mean of the two adjacent tracer rows; 'face_latitude' evaluates "
             "f = 2*Omega*sin(phi_face) AT the v-face latitude, which is "
             "NEMO's own ff_f convention. NB 'face_latitude' is 10x less "
             "consistent with the discrete curl of solid-body rotation "
             "(planetary-vorticity/Kelvin consistency) and is intended for "
             "oracle-matching only -- see create_latlon_geometry's docstring.",
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
             "(DINOConfig.nemo_faithful_grid): 52 columns x 199 rows, equator "
             "on a T-point, U-faces [0,51], instead of the legoESM "
             "[-50,0]/198x50 default. The mesh is transcribed from NEMO's own "
             "namelist -- no NEMO file is read -- and is BIT-EXACT against "
             "NEMO 5.0.2's mesh_mask on every field except the two Coriolis "
             "arrays, which differ by <=3 ulp because the NEMO binary "
             "vectorised its own sine (gated by scripts/validate/"
             "ocean_fidelity/dino_1226/nemo_dino_mesh_gate.py). Co-sets the "
             "bathymetry lon frame, sill anchor, vertical coordinate, omega "
             "and metric convention via nemo_faithful_dino_config, and returns "
             "the same domain the certified NEMO twin runs on. Lat-lon only.",
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
        "--tke-prandtl-ri", choices=("on", "off"), default=None,
        help="TKE Richardson-dependent Prandtl (DINOConfig.tke_prandtl_ri; "
             "NEMO zdftke nn_pdl=1). on: interior tracer diffusivity avt drops "
             "toward 0.1·avm in stratified water (Pr=clamp(Ri/ri_cri,1,10)); "
             "off (default): constant Pr=10. Effective only for --vmix tke. "
             "The nemo_dino_kamm recipe sets it on.",
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
             "scaling, cap aei0=0.5*rn_Ue*rn_Le=1500 m2/s, ldftra.F90:332). "
             "Lat-lon only.",
    )
    p.add_argument(
        "--gm-redi-mld-criterion", choices=("rho_c", "n2_integral"),
        default=None,
        help="Mixed-layer-depth criterion for the NEMO ldfslp slope ramp / "
             "native slopes (DINOConfig.gm_redi_mld_criterion): 'rho_c' "
             "(default, potential-density difference) or 'n2_integral' (NEMO "
             "zdfmxl exact integral(MAX(N^2,0) dz) >= g*rho_c/rho0; the "
             "nemo_dino_kamm card selects it). Only affects runs with the ML "
             "ramp / native slopes active.",
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
             "default 1500 = 0.5*rn_Ue*rn_Le, ldftra.F90:332). Ignored unless "
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
        "--barotropic-coriolis",
        choices=("avg", "een", "een_metric"),
        default=None,
        help="In-substep barotropic Coriolis discretization "
             "(DINOConfig.barotropic_coriolis; explicit_substep only). "
             "'avg' (legacy 4-pt average) annihilates the 2Δx zonal "
             "checkerboard — the C-grid barotropic Coriolis null mode that "
             "drives the spurious deep-equatorial jet. 'een' = NEMO "
             "dyn_spg_ts::dyn_cor_2D enstrophy-conserving EEN (ln_dynvor_een), "
             "which restores the velocity null mode. 'een_metric' = METRIC-"
             "COMPLETE EEN (folds NEMO's e1v/r1_e1u + e2u/r1_e2v scale factors "
             "into ffu/ffv exactly, matching NEMO dyn_cor_2D) — the more NEMO-"
             "faithful choice (node 16; DINO default). NB it is a ~1% high-lat "
             "correction and does NOT cure the |lat|~68deg 2dx eta runaway.",
    )
    p.add_argument(
        "--barotropic-coriolis-split",
        choices=("frozen", "live"),
        default=None,
        help="Barotropic-Coriolis split (DINOConfig.barotropic_coriolis_split; "
             "explicit_substep only). 'frozen' (default) holds the planetary "
             "Coriolis frozen in F_slow across the substep window; 'live' "
             "removes the pre-step 2D barotropic Coriolis from F_slow and "
             "re-applies it LIVE each substep on the evolving transport (NEMO "
             "dyn_spg_ts:296-300 + dyn_cor_2D). Under vorticity_scheme="
             "'een_total' 'live' requires --barotropic-coriolis een; it is what "
             "unblocks the nemo_dino_kamm_mlf leapfrog at dt=2700 (node 16).",
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
        "--outer-integrator",
        choices=("forward_euler", "ab2", "leapfrog"),
        default=None,
        help="Outer (baroclinic) time integrator (DINOConfig.outer_integrator). "
             "'forward_euler' (default), 'ab2' (Veros/MITgcm), or 'leapfrog' = "
             "NEMO Modified Leap-Frog (stp_MLF): three time levels + the plain "
             "Robert-Asselin filter. leapfrog REQUIRES coriolis_scheme="
             "explicit_ab2 (Coriolis in the RHS) — pair with --vorticity-scheme "
             "een_total (the nemo_dino_kamm_mlf recipe sets all three).",
    )
    p.add_argument(
        "--vorticity-scheme",
        choices=("al81", "een_total"),
        default=None,
        help="Vector-invariant vorticity flux (DINOConfig.vorticity_scheme). "
             "'al81' (default): relative-vorticity EEN triad (planetary f in the "
             "separate face-f path). 'een_total': NEMO ln_dynvor_een — the "
             "ABSOLUTE vorticity (f+zeta)/e3f rides the EEN triad (Coriolis IN "
             "the RHS); requires --coriolis-scheme explicit_ab2.",
    )
    p.add_argument(
        "--asselin-gamma", type=float, default=None,
        help="Robert-Asselin filter coefficient rn_atfp for --outer-integrator "
             "leapfrog (DINOConfig.asselin_gamma; NEMO default 0.1, plain RA not "
             "Williams). Ignored for other integrators.",
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
    p.add_argument(
        "--n-devices", type=int, default=1,
        help="Number of devices for lat-band SPMD sharding (lat-lon only; "
             "1 = the plain single-device step). Requires "
             "n_lat %% n_devices == 0 — the default R1 grid (198 rows) "
             "divides only by 1 and 2; --n-lon 100 (396 rows) divides by "
             "4. Global build runs on the host CPU when n > 1 (set "
             "JAX_PLATFORMS=cuda,cpu to enable that).",
    )
    p.add_argument(
        "--multicontroller", action="store_true",
        help="Multi-process mode: one process per device via "
             "jax.distributed (route-B). MUST be launched with one "
             "process per device (srun -n N / mpiexec -n N); "
             "init_multicontroller_distributed runs before any other "
             "JAX use. Requires --n-devices == the GLOBAL device count.",
    )
    p.add_argument(
        "--coordinator", type=str, default=None,
        help="Coordinator host:port for --multicontroller bootstrap "
             "(mpiexec path; SLURM auto-detects when omitted).",
    )
    p.add_argument(
        "--restart-from", type=Path, default=None,
        help="Load the initial state from a snapshot NPZ written by this "
             "script (lat-lon only). Restarts eta/T/S/u/v; the vertical "
             "grid and geometry are rebuilt from the config, so the "
             "restart MUST come from the same --n-lon/grid settings. "
             "Restarts the CLOCK at 0 (the snapshot's time metadata is "
             "recorded in the run metadata, not replayed).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Initial-condition perturbation seed for ensemble runs "
             "(0 = no perturbation, the canonical rest-state IC; "
             "identical to the unseeded run). Adds small masked Gaussian "
             "noise to u/v/eta (lat-lon) or u/eta (MPAS).",
    )
    p.add_argument(
        "--timing-jsonl", type=Path, default=None,
        help="Append a JSON line of per-step timing (compile/warmup/"
             "steady-step wall times, SYPD, Mcells/s) to this file "
             "(default: <output-dir>/timing.jsonl; multicontroller: "
             "rank 0 writes, other ranks skip).",
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
# Restart / IC perturbation
# ---------------------------------------------------------------------

def _load_restart(snapshot_path: Path):
    """Load a snapshot NPZ written by ``_save_snapshot`` as a dict."""
    print(f"[restart] Loading {snapshot_path}")
    data = np.load(snapshot_path)
    return {k: data[k] for k in data.files}


def _apply_restart_arrays(state, restart_arrays, grid_kind: str):
    """Replace the prognostic fields with the restart snapshot's arrays."""
    import jax.numpy as jnp

    state = state._replace(
        eta=state.eta.replace(data=jnp.asarray(restart_arrays["eta"])),
        T=state.T.replace(data=jnp.asarray(restart_arrays["T"])),
        S=state.S.replace(data=jnp.asarray(restart_arrays["S"])),
        u=state.u.replace(data=jnp.asarray(restart_arrays["u"])),
    )
    if grid_kind == "latlon":
        if "v" not in restart_arrays:
            raise SystemExit(
                f"restart NPZ {restart_arrays.get('time_days', '?')} lacks "
                "'v' — not a lat-lon DINO snapshot.")
        state = state._replace(
            v=state.v.replace(data=jnp.asarray(restart_arrays["v"])))
    return state


def _perturb_state(state, seed: int, grid_kind: str):
    """Add small masked Gaussian noise to the IC (ensemble runs).

    ``seed=0`` is the canonical un-perturbed IC (returns the state
    unchanged). The velocity noise is projected through the face masks so
    a seeded IC stays a VALID model state (the same convention as the
    SPMD scaling bench's IC recipe).
    """
    if seed == 0:
        return state

    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    u = np.asarray(state.u.data)
    u_mask = np.asarray(state.u_mask.data)[..., None] if hasattr(state, "u_mask") else None
    u_noise = 0.02 * rng.standard_normal(u.shape)
    if u_mask is not None:
        u_noise = u_noise * u_mask
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u + u_noise)))
    if grid_kind == "latlon":
        v = np.asarray(state.v.data)
        v_mask = np.asarray(state.v_mask.data)[..., None] if hasattr(state, "v_mask") else None
        v_noise = 0.02 * rng.standard_normal(v.shape)
        if v_mask is not None:
            v_noise = v_noise * v_mask
        state = state._replace(
            v=state.v.replace(data=jnp.asarray(v + v_noise)))
    eta = np.asarray(state.eta.data)
    eta_noise = 0.005 * rng.standard_normal(eta.shape)
    state = state._replace(
        eta=state.eta.replace(data=jnp.asarray(eta + eta_noise)))
    print(f"[seed] IC perturbed with seed={seed} "
          "(u,v: 0.02 m/s; eta: 0.005 m std)")
    return state


# ---------------------------------------------------------------------
# Per-step timing
# ---------------------------------------------------------------------

def _write_timing_jsonl(args, record: dict):
    """Append the timing record (rank 0 only under multicontroller)."""
    import jax

    if args.multicontroller and jax.process_index() != 0:
        return
    out_path = args.timing_jsonl or (args.output_dir / "timing.jsonl")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    if not args.multicontroller or jax.process_index() == 0:
        print(f"  Timing record: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _is_nemo_fidelity_run(args) -> bool:
    """The NEMO-identity DINO path: NEMO's exact mesh and/or a ``nemo_*`` card."""
    return bool(args.nemo_faithful_grid) or str(args.recipe or "").startswith("nemo_")


def _force_fp64_for_nemo_fidelity(args) -> None:
    """NEMO-fidelity runs are ALWAYS fp64, and not by a flag (user decision
    2026-09-09).  ``JAX_ENABLE_X64=1`` only permits float64; legoESM's
    constructors cast to ``get_policy().control``, which defaults to float32,
    so without this the NEMO-exact mesh and initial state are rebuilt in
    single precision and stop being bit-exact.  Must run before any array is
    built.  Other cards keep the default policy.
    """
    if not _is_nemo_fidelity_run(args):
        return
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64())
    policy = get_policy()
    if policy.control != PrecisionPolicy.fp64().control:
        raise SystemExit(
            f"NEMO-fidelity run requires the fp64 precision policy, got {policy}")
    print(f"[run_dino] NEMO-fidelity run: precision policy forced to fp64 "
          f"(control={policy.control}, storage={policy.storage})", flush=True)


def main():
    args = _parse_args()
    _force_fp64_for_nemo_fidelity(args)

    # Multi-controller init MUST run before any other JAX use (backend
    # init) — shared helper with the ocean/atm SPMD benches.
    if args.multicontroller:
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(args.coordinator)

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
    if args.coriolis_placement is not None:
        cfg = dataclasses.replace(
            cfg, coriolis_placement=args.coriolis_placement)
    if args.tke_momentum_visc_bg is not None:
        cfg = dataclasses.replace(
            cfg, tke_momentum_visc_bg=args.tke_momentum_visc_bg)
    if args.evd_momentum is not None:
        cfg = dataclasses.replace(
            cfg, evd_on_momentum=(args.evd_momentum == "on"))
    if args.tke_prandtl_ri is not None:
        cfg = dataclasses.replace(
            cfg, tke_prandtl_ri=(args.tke_prandtl_ri == "on"))
    if args.barotropic_coriolis is not None:
        cfg = dataclasses.replace(
            cfg, barotropic_coriolis=args.barotropic_coriolis)
    if args.barotropic_coriolis_split is not None:
        cfg = dataclasses.replace(
            cfg, barotropic_coriolis_split=args.barotropic_coriolis_split)
    if args.gm_kappa_scheme is not None:
        cfg = dataclasses.replace(cfg, gm_kappa_scheme=args.gm_kappa_scheme)
    if args.gm_redi_mld_criterion is not None:
        cfg = dataclasses.replace(
            cfg, gm_redi_mld_criterion=args.gm_redi_mld_criterion)
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
    if args.outer_integrator is not None:
        cfg = dataclasses.replace(cfg, outer_integrator=args.outer_integrator)
    if args.vorticity_scheme is not None:
        cfg = dataclasses.replace(cfg, vorticity_scheme=args.vorticity_scheme)
    if args.asselin_gamma is not None:
        cfg = dataclasses.replace(cfg, asselin_gamma=args.asselin_gamma)
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
            raise SystemExit(
                "--nemo-faithful-grid is lat-lon only (NEMO's Mercator DINO "
                "mesh); rerun with --grid latlon.")
        # Applied LAST: co-sets the bathymetry lon frame + sill anchor onto
        # whatever recipe/overrides preceded it (must not be clobbered after).
        cfg = nemo_faithful_dino_config(base=cfg)

    # #1492 footgun guard, evaluated on the RESOLVED config (NOT on args):
    # leapfrog + the post-step ("applied_now") surface applier discards ~56%
    # of every applied surface flux (the increment cancels in the leap-frog
    # combine; retention (1-2*gamma)/(2*(1-gamma)) = 4/9 at gamma=0.1,
    # measured twice on independent grids).  Both knobs have a recipe-card
    # source as well as (for the integrator) a CLI flag, so this MUST read
    # cfg -- an earlier version sat inside `if args.outer_integrator is not
    # None:` and could only fire when --outer-integrator was passed
    # EXPLICITLY, which meant a card-only run (`--recipe nemo_dino_kamm_mlf`,
    # which sets outer_integrator="leapfrog" internally) sailed straight past
    # it.  Placed after every cfg mutation above so no later override can
    # reintroduce the combination.
    if (cfg.outer_integrator == "leapfrog"
            and getattr(cfg, "surface_tendency_placement", "applied_now")
            == "applied_now"):
        raise SystemExit(
            "outer_integrator='leapfrog' with "
            "surface_tendency_placement='applied_now' discards ~56% of "
            "the applied surface flux (#1492). Use a card that sets "
            "surface_tendency_placement='leapfrog_rhs' (e.g. "
            "nemo_dino_kamm_mlf), or choose forward_euler/ab2 (both "
            "retain 1.000000).")

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

    # --- SPMD / restart / seed guards (dispatch-hardening, before build) ---
    nd = args.n_devices
    if nd > 1 and grid_kind != "latlon":
        raise SystemExit(
            "--n-devices (lat-band SPMD) is lat-lon only; rerun with "
            "--grid latlon or --n-devices 1.")
    if args.multicontroller and nd < 2:
        raise SystemExit(
            "--multicontroller needs --n-devices >= 2 (it federates one "
            "process per device; single-device runs need no federation).")
    if args.restart_from is not None and grid_kind != "latlon":
        raise SystemExit(
            "--restart-from is lat-lon only (the snapshot NPZ carries "
            "lat-lon C-grid fields).")
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(
            f"need {nd} devices, have {avail}. For a CPU smoke test set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=<N>.")
    if args.multicontroller and nd != avail:
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")

    # Build grid, state, model — branch on grid type. For nd > 1 the GLOBAL
    # build runs on the host CPU backend (rest-state init at global shape
    # would materialise the whole domain per device; only the per-band
    # shards reach the accelerator via shard_state_latlon — the #1370
    # convention shared with the ocean SPMD bench / run_omip lane).
    import contextlib

    _build_ctx = contextlib.nullcontext()
    if nd > 1:
        try:
            _build_ctx = jax.default_device(
                jax.local_devices(backend="cpu")[0])
        except RuntimeError:
            print("[SPMD] WARNING: no cpu backend — global init will "
                  "materialise on the accelerator (set "
                  "JAX_PLATFORMS=cuda,cpu to enable the host-side build)",
                  flush=True)

    with _build_ctx:
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

        if nd > 1 and grid.n_lat % nd != 0:
            raise SystemExit(
                f"--n-devices {nd}: n_lat ({grid.n_lat}) not divisible by "
                f"the device count; pick --n-devices dividing n_lat (the "
                f"default R1 grid's 198 rows divide only by 1 and 2; "
                f"--n-lon 100 -> 396 rows divides by 4). No exact DINO "
                f"Mercator row count divides by 8 — wider device counts "
                f"belong on the synthetic-grid SPMD benches.")

        # --- Restart from a previous snapshot (lat-lon only) ---
        if args.restart_from is not None:
            restart = _load_restart(args.restart_from)
            state = _apply_restart_arrays(state, restart, grid_kind)
            print(f"[restart] Applied restart at "
                  f"t={float(restart['time_days']):.2f} days "
                  f"(clock restarts at 0)")

        # --- IC perturbation (ensemble runs; seed=0 = canonical IC) ---
        state = _perturb_state(state, args.seed, grid_kind)

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

    # --- Lat-band SPMD sharding (lat-lon only, nd > 1) ---
    # The rigid-lid island streamfunction solver and the SPMD band step
    # are mutually exclusive lanes (rigid_lid has no sharded step); refuse
    # the combination loudly instead of sharding a solver that cannot run.
    mesh = None
    if nd > 1:
        if cfg.barotropic_solver == "rigid_lid":
            raise SystemExit(
                "--n-devices with --barotropic-solver rigid_lid is not "
                "supported (the island-streamfunction solver has no "
                "lat-band SPMD step); use implicit_cn (default) or "
                "explicit_substep.")
        if sf_step is not None:
            raise SystemExit(
                "--n-devices with wind_through_step is not supported "
                "(the in-step OceanSurfaceForcing lane has no sharded "
                "path here); run with the pre-step applicator "
                "(wind_through_step=False).")
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            make_sharded_ocean_step,
            shard_forcing_latlon,
            shard_state_latlon,
        )

        # Prime the build-once caches from the CONCRETE state so the
        # wrapper can build the per-band vertex masks host-side.
        model.prime_step_caches(state)
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:nd]), axis_names=("lat",))
        step_fn = make_sharded_ocean_step(model, mesh)
        state = shard_state_latlon(state, mesh)
        if forcing is not None:
            forcing = shard_forcing_latlon(forcing, mesh)
        print(f"[SPMD] Sharded across {nd} devices "
              f"(lat bands of {grid.n_lat // nd} rows)")
    else:
        step_fn = model.step

    # Under multicontroller, ONLY rank 0 writes snapshots / metadata /
    # stdout (every rank would otherwise write identical copies to the
    # shared path, or corrupt each other).
    _is_writer = (not args.multicontroller) or (jax.process_index() == 0)

    t_wall_start = time.time()
    snapshot_idx = 0
    # Under SPMD the initial state is the sharded v_lower carrier — save
    # the GLOBAL gathered state (the snapshot/restart contract is the
    # single-device layout).
    if _is_writer:
        if mesh is not None:
            from legoesm.ocean.dynamics.sharded_ocean_step import (
                gather_state_latlon,
            )
            _save_snapshot(gather_state_latlon(state, mesh, to_host=True),
                           0.0, snapshot_dir, snapshot_idx, grid_kind)
        else:
            _save_snapshot(state, 0.0, snapshot_dir, snapshot_idx, grid_kind)

    # #1492: NEMO-faithful surface-tracer-tendency placement (lat-lon only —
    # the MPAS applicator has no return_rate= mode / model.step has no
    # external_tracer_rate hook there). "applied_now" (default) keeps the
    # legacy pre-step state mutation; "leapfrog_rhs" folds the tendency into
    # the leap-frog Nnn RHS instead (see DINOConfig.surface_tendency_placement).
    _sf_placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    if _sf_placement not in ("applied_now", "leapfrog_rhs"):
        raise SystemExit(
            f"Unknown DINOConfig.surface_tendency_placement {_sf_placement!r}: "
            "expected 'applied_now' or 'leapfrog_rhs'.")
    if _sf_placement == "leapfrog_rhs" and grid_kind != "latlon":
        raise SystemExit(
            "surface_tendency_placement='leapfrog_rhs' is only wired for "
            "--grid latlon (apply_dino_mpas_surface_forcing has no "
            "return_rate= mode).")

    # Per-step timing: compile (step 1), warmup, steady state (the
    # hackathon-measured R1 plateau is ~4-5 steps on H100; 5 warmup steps
    # are discarded before the steady mean).
    step_times_ms = []

    for k in range(n_steps_total):
        t_step_start = time.perf_counter()
        _ext_rate = None
        if forcing is not None:
            # NEMO time convention: step k (0-based) ends at t=(k+1)*dt —
            # drives the seasonal forcing phases when forcing_annual_cycle.
            if _sf_placement == "leapfrog_rhs":
                state, _ext_rate = apply_forcing(
                    state, forcing, z, cfg, dt, t_seconds=(k + 1) * dt,
                    return_rate=True)
            else:
                state = apply_forcing(state, forcing, z, cfg, dt,
                                      t_seconds=(k + 1) * dt)

        if nd > 1:
            # The sharded step rides the pre-step applicator (EAGER, host
            # side); the in-step surface_forcing lane is refused above.
            state = step_fn(state, dt=dt)
        else:
            state = step_fn(
                state, dt=dt,
                surface_forcing=(sf_step if getattr(
                    cfg, "wind_through_step", False) else None),
                external_tracer_rate=_ext_rate)

        # Block until ready (per-step wall time is only meaningful when
        # the async dispatch is synchronized).
        jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
                               if leaf is not None])

        ms = (time.perf_counter() - t_step_start) * 1000.0
        step_times_ms.append(ms)
        if k == 0:
            print(f"[step 1] compile+step = {ms:.1f} ms (JIT compile)")
        elif (k + 1) % 10 == 0 or k == n_steps_total - 1:
            print(f"[step {k+1}] {ms:.1f} ms")

        is_last = (k == n_steps_total - 1)
        if _is_writer and ((k + 1) % snapshot_every_steps == 0 or is_last):
            snapshot_idx += 1
            t_seconds = (k + 1) * dt
            # Under SPMD the live state is the sharded v_lower carrier —
            # gather the GLOBAL state (re-appends the pole-wall v-row) for
            # the snapshot + diagnostics, then re-shard the live carrier.
            if mesh is not None:
                from legoesm.ocean.dynamics.sharded_ocean_step import (
                    gather_state_latlon,
                )
                state_global = gather_state_latlon(state, mesh, to_host=True)
                _save_snapshot(state_global, t_seconds, snapshot_dir,
                               snapshot_idx, grid_kind)
                d = _diagnose(state_global, grid, grid_kind)
                del state_global
            else:
                _save_snapshot(state, t_seconds, snapshot_dir,
                               snapshot_idx, grid_kind)
                d = _diagnose(state, grid, grid_kind)
            print(f"{k+1:6d} {t_seconds/86400.0:7.2f} "
                  f"{d['u_max']:10.4e} {d['v_max']:10.4e} {d['eta_max']:10.4e} "
                  f"{d['T_max']:7.2f} {d['T_min']:7.2f} {d['ke_total']:10.4e}")

    wall = time.time() - t_wall_start
    n_warmup = min(5, len(step_times_ms))
    steady_times = step_times_ms[n_warmup:]
    steady_mean = float(np.mean(steady_times)) if steady_times else 0.0
    steady_std = float(np.std(steady_times)) if steady_times else 0.0
    compile_ms = step_times_ms[0] if step_times_ms else 0.0
    sypd = (dt / (steady_mean / 1000.0)) / 365.25 if steady_mean > 0 else 0.0
    if grid_kind == "latlon":
        n_cells = grid.n_lat * grid.n_lon * z.n_levels
    else:
        n_cells = grid.nCells * z.n_levels
    mcells_s = (n_cells / (steady_mean / 1000.0) / 1e6) if steady_mean > 0 else 0.0

    print()
    print("=== TIMING SUMMARY ===")
    print(f"  Steps:        {n_steps_total}")
    print(f"  Compile (s1): {compile_ms:.1f} ms")
    print(f"  Steady mean:  {steady_mean:.1f} ms (steps {n_warmup+1}–{n_steps_total})")
    print(f"  Steady std:   {steady_std:.1f} ms")
    print(f"  SYPD:         {sypd:.1f}")
    print(f"  Mcells/s:     {mcells_s:.1f}")
    print(f"Done. Wall time: {wall:.1f}s. Snapshots: {snapshot_idx + 1}")

    _write_timing_jsonl(args, {
        "script": "run_dino",
        "resolution_label": f"n_lon={args.n_lon}",
        "n_lat": int(grid.n_lat) if grid_kind == "latlon" else None,
        "n_lon": int(grid.n_lon) if grid_kind == "latlon" else None,
        "n_cells": int(n_cells),
        "n_levels": int(z.n_levels),
        "dt": dt,
        "days": args.days,
        "n_steps": n_steps_total,
        "n_devices": nd,
        "multicontroller": args.multicontroller,
        "n_processes": int(jax.process_count()),
        "precision": "float64" if jax.config.jax_enable_x64 else "float32",
        "recipe": args.recipe,
        "preset": args.preset,
        "vmix": cfg.vmix_scheme,
        "barotropic_solver": cfg.barotropic_solver,
        "forcing": not args.no_forcing,
        "physics": not args.physics_off,
        "seed": args.seed,
        "compile_ms": compile_ms,
        "steady_mean_ms": steady_mean,
        "steady_std_ms": steady_std,
        "n_warmup": n_warmup,
        "sypd": sypd,
        "mcells_s": mcells_s,
        "total_wall_s": wall,
        "step_times_ms": step_times_ms,
    })


if __name__ == "__main__":
    main()
