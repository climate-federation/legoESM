"""Long RCE run under MPI — 12 ranks, 132x132, 100-day capable.

Production-ready harness for RCE on the plane CRM with:

* Full physics: gray radiation, Kessler microphysics, Smagorinsky
  LES, surface flux, tracer positivity, mean-wind filter (MPI),
  moist-mass fixer (MPI).
* MPI: rank 0 stepd dycore + broadcasts state; all ranks reduce
  for mean wind + moist mass.
* Outputs (all on rank 0):
  - `<output>/log.txt` — per-step CSV with diagnostics.
  - `<output>/snapshots/snap_day_NNN.npz` — every 24 hrs (sim
    time), x-y surface fields (CWV, MSE, surface T_atm, surface
    q_v, |U|_sfc, precip).
  - `<output>/profiles/prof_day_NNN.npz` — every 5 days,
    horizontally-meaned vertical profiles (T, q_v, q_c, q_r,
    cloud fraction).
  - `<output>/progress.txt` — overwritten each step with
    "current day = X.YYY" for external monitoring.

CLI
---
.. code-block:: bash

   mpirun -np 12 .venv/bin/python \\
       scripts/run_rce_mpi_long.py \\
       --days 100 --dt 2.0 --output results/rce100d

Wall-clock estimate (M5 Pro, 12 CPUs):
   100 days × 86400 s / dt=2 s = 4.32M steps
   ~1.5 s/step (rank 0 bound) → ~75 days wall clock.
   Requires cluster for full 100-day run.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from legoesm import constants
from legoesm.atmosphere.dynamics.shared.cfl_diagnostic import (
    compute_courant_numbers_plane,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    HORIZONTAL_ADVECTION_HALO_REQUIREMENT as _ADV_HALO_REQ,
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
    cloud_fraction_profile_plane, column_moist_static_energy_plane,
    column_water_vapor_plane, moist_static_energy_3d_plane,
    precipitation_rate_proxy_plane, temperature_3d_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    compute_dry_mass_plane_mpi,
    compute_total_water_mass_plane_mpi,
    fix_mass_nonhydrostatic_plane_mpi,
    fix_moist_mass_plane_mpi,
    remove_horizontal_mean_wind_plane_mpi,
)
from legoesm.atmosphere.dynamics.crm.rce_surface_flux import (
    compose_rce_surface_scalar_tendencies,
    wind_speed_at_lowest_level_plane,
)
from legoesm.atmosphere.dynamics.shared.tracer_positivity import (
    apply_positive_filter_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    build_smooth_k1_pattern,
    make_wing2018_qv_ref_fn, make_wing2018_theta_ref_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import (
    create_height_coordinate, create_stretched_height_coordinate,
)
from legoesm.parallel.plane_mpi import (
    gather_plane_field,
    make_plane_pencil_grid,
    make_plane_pencil_layout,
    scatter_plane_field,
)

# PRECISION: x64 is toggled at IMPORT (before argparse), so f32 is selected via
# an env var (mirrors run_rcemip_plane's LEGOESM_RCEMIP_PLANE_FP32). Set
# LEGOESM_RCE_MPI_FP32=1 BEFORE launch → x64 stays OFF → float64 array requests
# canonicalize to float32 (dynamics + mpi4jax halo run in f32). Default = fp64.
# main() cross-checks --precision against this env var so a mismatch fails loud.
if os.environ.get("LEGOESM_RCE_MPI_FP32") != "1":
    jax.config.update("jax_enable_x64", True)


T_SFC_K = 300.0
Q_SFC_FRAC = 0.0224          # Wing saturation q_v at SST
Q_SFC_IC_FRAC = 0.018        # IC sub-saturated (~80% RH) — avoids
                              # instant-condensation blowup at start.
GAMMA_TROP = 6.7e-3
Z_T = 15_000.0
SEC_PER_DAY = 86400.0


# iter-208: ``build_smooth_k1_pattern`` lives in
# ``legoesm.atmosphere.idealized.rcemip_initial_conditions`` so the
# iter-207 unit test imports it without loading the entire driver
# (mpi4jax / jax-MPI / argparse / etc.). The driver consumes it via
# the package-path import block below.

# Radiation call-frequency convention (literature):
# - SAM (Khairoutdinov-Randall): 600 s
# - CM1 (Bryan-Fritsch): 60-120 s
# - WRF-LES: 300-600 s
# - RCEMIP1 spec: model-default but typically 600 s
# Default here: 600 s. Tendency held constant between calls.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=132)
    p.add_argument("--ny", type=int, default=132)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2_000.0)
    p.add_argument("--dt", type=float, default=20.0,
                   help="Outer dt [s]. Bare-dycore stability bound is "
                        "BUBBLE-dependent: with the legacy 0.5 K warm "
                        "bubble at z<1 km (F1) dt was limited to ≤1 s; "
                        "with the F8/F10 clean Wing IC (no bubble + no "
                        "qv noise) dt=10 s is bit-stable bare-dycore + "
                        "dt=5 s is stable through 864 steps of full "
                        "physics (smoke iter-9, 2026-05). Production "
                        "default set to 5 s — 5× speedup of every "
                        "30-day run vs the iter-2 conservative 1 s "
                        "default.")
    p.add_argument("--days", type=float, default=100.0)
    p.add_argument("--H", type=float, default=33_000.0,
                   help="Model top [m]. RCEMIP1 33 km.")
    p.add_argument("--dz-sfc", type=float, default=100.0,
                   help="Stretched-grid surface dz [m] (ignored if "
                        "--vertical-grid=uniform).")
    p.add_argument("--vertical-grid", choices=["uniform", "stretched"],
                   default="uniform",
                   help="uniform = no stretching, dz=H/nlev (~1100m for "
                        "default H=33km, nlev=30). Stable at dt=6. "
                        "stretched = ~50-100m surface layer (CFL-tight, "
                        "needs dt<=1).")
    p.add_argument("--c-h", type=float, default=1.5e-3)
    p.add_argument("--n-acoustic-substeps", type=int, default=12,
                   help="Number of acoustic-mode substeps per outer "
                        "dycore step. iter-14 + iter-38 measured N=12 "
                        "at dt=5 s on the clean Wing IC = 132x132 "
                        "1-sim-hour PASS (max|w|=6.1e-3 m/s, MSE drift="
                        "1.7e-4 over 725 steps). iter-183 kept N=12 "
                        "while lifting outer dt to 20 s (Van Leer + "
                        "beta=0.2 absorb the extra inner stiffness; "
                        "Ca_substep=0.87 at the new contract vs ~0.22 "
                        "at dt=5). The historical default 24 was set "
                        "for the iter-1 dt=1.0 s config; iter-59 "
                        "refreshed to 12 to match production.")
    p.add_argument("--n-physics-substeps", type=int, default=10,
                   help="Operator-split physics N x per dycore step. "
                        "Stiff sources (microphysics, surface flux) "
                        "sub-stepped at dt_outer/N.")
    p.add_argument("--rad-call-interval-s", type=float, default=600.0,
                   help="Radiation call interval [s]. Literature: SAM "
                        "600s, CM1 60-120s, WRF-LES 300-600s. "
                        "Tendency held constant between calls.")
    p.add_argument("--no-radiation", action="store_true", default=False,
                   help="Fully disable the gray radiation tendency "
                        "(skips slow_physics_fn entirely; cached_rad_tend "
                        "stays None). Use for dycore-isolation smokes. "
                        "Codex iter-39 review: --rad-call-interval-s=1e9 "
                        "does NOT disable radiation — it still fires once "
                        "at step 1 and caches the tendency for the full "
                        "run. This flag is the only way to truly run "
                        "without radiation.")
    p.add_argument("--no-mass-fixer", action="store_true", default=False,
                   help="iter-95: disable fix_moist_mass_plane which "
                        "rescales all moist tracers to maintain "
                        "IC-time total water mass. Necessary for "
                        "RCE spinup runs: surface flux must be "
                        "allowed to NET ADD moisture until "
                        "precipitation balances at equilibrium. "
                        "The fixer is appropriate for gravity-wave "
                        "smokes where total water IS conserved; "
                        "it is FATAL for RCE because it removes "
                        "the surface-flux moisture every step, "
                        "pinning CWV at IC value and preventing "
                        "convection initiation.")
    p.add_argument("--smag-cs", type=float, default=0.2)
    p.add_argument("--semi-implicit-acoustic", action="store_true",
                   help="Use per-column Thomas tridiagonal solve for "
                        "the acoustic substep (lifts vertical-CFL "
                        "limit; main benefit on stretched grids).")
    p.add_argument("--hyperdiff", type=float, default=5.0e6,
                   help="Biharmonic horizontal diffusion on u, v, theta', "
                        "rho', w [m^4/s]. Production default 5e6 — the "
                        "previous 1e6 default could not damp the 2-Δz "
                        "vertical mode seeded by the warm-bubble IC at "
                        "dt=1 s, max|w| growing to >20 m/s by step 250 "
                        "(see CRM_implementation.md F6).")
    p.add_argument("--bubble-theta-pert", type=float, default=0.0,
                   help="Warm-bubble convection seed amplitude [K]. "
                        "Default 0 = no bubble (standard RCEMIP1 "
                        "protocol: spin up convection from surface "
                        "fluxes + qv noise alone). Set 0.5 to "
                        "reproduce the legacy single-bubble IC. The "
                        "0.5 K bubble seeds a 2-Δz vertical mode at "
                        "dz~1.1 km that no amount of hyperdiff fully "
                        "damps — only suitable for short rising-"
                        "thermal smokes, not 30-day runs.")
    p.add_argument("--qv-noise-amp", type=float, default=0.0,
                   help="Amplitude of zero-mean random qv perturbations "
                        "[kg/kg] applied to the lowest 4 model levels "
                        "of the IC. Default 0 = clean Wing 2018 IC; "
                        "convection spins up from radiative cooling + "
                        "surface flux (~hours). Set 1e-5 - 5e-5 (0.01 - "
                        "0.05 g/kg) for a gentle stochastic seed. "
                        "Amplitudes ≥ 2.5e-4 trigger immediate Kessler "
                        "condensation hotspots that destabilise the "
                        "dycore in <5 min sim time (smoke verified — "
                        "see CRM_implementation.md F7).")
    p.add_argument("--qv-noise-seed", type=int, default=0,
                   help="RNG seed for qv noise perturbation. Same seed "
                        "→ bit-identical IC across reruns.")
    p.add_argument("--theta-noise-amp", type=float, default=0.0,
                   help="Amplitude of zero-mean random theta' "
                        "perturbations [K] applied to the lowest 4 "
                        "model levels of the IC (RCEMIP / Wing 2018 "
                        "standard symmetry-breaker; +/-0.1 K is the "
                        "Wing 2018 RCEMIP1 protocol value). iter-181 "
                        "added this flag as a PROPOSED alternative to "
                        "the qv-noise symmetry-breaker; iter-182 then "
                        "traced F11 to its root cause (inherent "
                        "radiative-convective initiation at dx=4 km "
                        "that the dycore cannot resolve once convection "
                        "nucleates) and confirmed BOTH theta' AND qv "
                        "noise hit the same wall: any nonzero "
                        "amplitude + gray radiation -> blowup in "
                        "25-50 outer steps. Default 0 = clean Wing "
                        "IC; set nonzero only on the F11 fix paths "
                        "(LES dx=1km, subgrid convection scheme, "
                        "smooth wing perturbation, adaptive dt). "
                        "See CRM_implementation.md F11.")
    p.add_argument("--theta-noise-seed", type=int, default=0,
                   help="RNG seed for theta' noise perturbation. Same "
                        "seed → bit-identical IC across reruns. "
                        "iter-207 Codex LOW: only consumed by "
                        "--theta-noise-mode=white. The smooth_k1 mode "
                        "is deterministic (cos pattern depends only "
                        "on grid dims) so this seed is silently "
                        "ignored when mode=smooth_k1.")
    p.add_argument("--theta-noise-mode",
                   choices=["white", "smooth_k1"], default="white",
                   help="theta' perturbation pattern (iter-203). "
                        "'white' (default) = RCEMIP / Wing 2018 "
                        "uniform random ±amp K per cell — the iter-181 "
                        "behaviour. 'smooth_k1' = single cosine wave "
                        "at the lowest non-trivial wavenumber in x + y "
                        "with peak amplitude amp K — F11 fix-path-3 "
                        "candidate. Hypothesis: white noise nucleates "
                        "sub-resolved convective cells at dx=4 km; a "
                        "smooth large-scale perturbation may break "
                        "column symmetry without triggering the "
                        "iter-182 radiative-convective initiation "
                        "runaway.")
    p.add_argument("--use-dd", action="store_true", default=False,
                   help="Switch from the legacy rank-0-dycore + "
                        "broadcast pattern to true per-rank domain "
                        "decomposition via step_halo + owned_mask. "
                        "Activates the R7 MPI mass fixer + MPI mean "
                        "wind. Required for any real MPI scaling claim. "
                        "Defaults False to preserve the F8-stable "
                        "config from iter-2; flip to True once you have "
                        "verified the DD path on the smoke run.")
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--sponge-width", type=float, default=10_000.0,
                   help="Sponge layer width from model top [m]. "
                        "Default 10 km matches CompressibleEulerConfig. "
                        "5 km is too thin for H=33 km when gravity-wave "
                        "wavelengths exceed sponge depth.")
    p.add_argument("--acoustic-off-centering", type=float, default=0.2,
                   help="Skamarock-Klemp off-centering parameter beta "
                        "in [0, 1). 0 = neutral forward-backward. "
                        "0.05-0.1 damps acoustic modes. Try 0.1 if "
                        "instability appears as growing rho_prime / w "
                        "oscillations.")
    p.add_argument("--vertical-theta-diffusion", type=float, default=0.0,
                   help="Explicit vertical Laplacian diffusivity on "
                        "theta_prime [m^2/s]. 0 = off. Try 1e4-5e4 to "
                        "damp the buoyancy/PG feedback that destabilises "
                        "the dycore at dt > 0.5 s on coarse vertical grids.")
    # iter-192 Codex MEDIUM#2: derive choices from the shared
    # HORIZONTAL_ADVECTION_HALO_REQUIREMENT map so argparse +
    # halo-dispatch + driver can never disagree on which scheme
    # names are valid. A future fourth scheme automatically
    # surfaces in --help once it's added to the map.
    p.add_argument("--advection",
                   choices=sorted(_ADV_HALO_REQ),
                   default="van_leer",
                   help="Horizontal advection scheme for theta/u/v/w. "
                        "upwind1: 1st-order, cheap, dispersive (legacy). "
                        "van_leer: 2nd-order TVD, monotone, stencil width 4 "
                        "— the iter-183 production choice (3x wall-time "
                        "speedup vs dt=10+upwind1 at the same sim time "
                        "after iter-180 dt=20 + beta=0.2). "
                        "weno5: 5th-order WENO-Z, much less grid-scale "
                        "dispersion but stencil 6 + ~3x per-step cost — "
                        "iter-183 wall-time measurement showed WENO5 at "
                        "dt=20 actually runs SLOWER than upwind1 at dt=10 "
                        "(2x fewer steps but 2.4x cost per step). Use only "
                        "for sharp-front problems where dispersion matters.")
    p.add_argument("--vertical-tracer-advection",
                   choices=["centered", "van_leer"],
                   default="van_leer",
                   help="VERTICAL tracer advection. van_leer (default) = "
                        "monotone TVD, positive-definite (matches the serial "
                        "run_rcemip_plane default + SAM's monotone scalar "
                        "transport); centered = 2nd-order, can overshoot into "
                        "negative tracer at sharp convective gradients. Now "
                        "honored on the MPI halo path (codex CRM-dycore "
                        "review) — previously the halo silently used centered.")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64",
                   help="Floating-point precision. float32 REQUIRES env "
                        "LEGOESM_RCE_MPI_FP32=1 set BEFORE launch (x64 is a "
                        "module-import toggle); main() refuses a mismatch. "
                        "f32 runs the dynamics + mpi4jax halo in single "
                        "precision for GPU/accelerator throughput.")
    p.add_argument("--adaptive-dt", action="store_true", default=False,
                   help="iter-228 F11 fix-path-4 STUB: opt-in flag "
                        "for runtime CFL monitoring + dt shrinkage. "
                        "When implemented (not yet wired into the "
                        "time-integration loop — see iter-225 "
                        "feasibility note in CRM_implementation.md "
                        "F11), monitor advective Courant ``Ca_adv = "
                        "max|w| * dt / dx`` after each outer step. "
                        "Halve dt for the next step if Ca > 0.5; "
                        "restore to nominal --dt when Ca < 0.1. "
                        "iter-233 REPLACEMENT: the FV3-style "
                        "``--n-outer-split`` flag below picks a "
                        "static substep count from a conservative "
                        "max-wind estimate at trace time — "
                        "scan-friendly + fully AD-safe + no "
                        "while-loop refactor. ``--adaptive-dt`` is "
                        "kept as a parse-only stub for back-compat "
                        "but raises NotImplementedError; prefer "
                        "``--n-outer-split auto`` for new code.")
    p.add_argument("--n-outer-split", default="1",
                   help="iter-233 FV3-style trace-time outer "
                        "subcycle count. ``1`` (default): no outer "
                        "subcycling, preserves iter-183 production "
                        "contract bit-for-bit. ``N`` (positive int): "
                        "static N inner steps per outer step, each "
                        "of size ``--dt / N``. ``auto``: call "
                        "``select_n_outer_split(dt, dx, "
                        "max_wind_safe=--max-wind-safe, "
                        "cfl_safe=--cfl-safe)`` to pick N at parse "
                        "time from a conservative max-wind estimate. "
                        "For iter-183 production (dt=20, dx=2000, "
                        "wind=300, cfl=0.4) ``auto`` selects N=8. "
                        "The chosen integer is a STATIC Python int "
                        "— no XLA retrace, no traced control flow, "
                        "fully reverse-differentiable through "
                        "eqx.filter_value_and_grad.")
    p.add_argument("--max-wind-safe", type=float, default=300.0,
                   help="Conservative upper-bound max|w|/max|u| for "
                        "``--n-outer-split auto`` [m/s]. Default 300 "
                        "covers the iter-223 F11 cascade ceiling "
                        "(max|w|=225 m/s before NaN) with 33%% "
                        "margin. Ignored when ``--n-outer-split`` "
                        "is a numeric value.")
    p.add_argument("--cfl-safe", type=float, default=0.4,
                   help="Target advective CFL for "
                        "``--n-outer-split auto`` (dimensionless). "
                        "Default 0.4 = SK08/FV3 standard (2.5x "
                        "margin under the formal CFL=1 limit). "
                        "Ignored when ``--n-outer-split`` is a "
                        "numeric value.")
    p.add_argument("--implicit-buoyancy", action="store_true", default=False,
                   help="Klemp-Wilhelmson 1978 implicit-buoyancy treatment "
                        "of the w-equation in the SI acoustic substep. Adds "
                        "three nearest-neighbour bands proportional to "
                        "dtheta_ref/dz to the tridiagonal solve. Closes the "
                        "w<->theta gravity-wave feedback that destabilises "
                        "the dycore at coarse dz (~1 km) with stratified "
                        "ICs. Only active when --semi-implicit-acoustic.")
    p.add_argument("--snapshot-hours", type=float, default=24.0)
    p.add_argument("--snapshot-3d-hours", type=float, default=0.0,
                   help="If > 0, dump full 3D MSE/qv/T volumes "
                        "(ny,nx,nlev) as compressed float32 NPZ "
                        "every N simulation hours. 0 disables.")
    p.add_argument("--profile-days", type=float, default=5.0)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--output", type=str, default="results/rce_long")
    return p.parse_args()


def build_height_coord_and_state(args, grid):
    # This RCE driver is a near-EQUILIBRIUM smoke: its SST (T_SFC_K), qv
    # profile, and radiation are tuned for a quiescent ~300 K start, so the
    # theta reference deliberately uses T_v0=T_SFC_K — NOT the strict-RCEMIP
    # fixed 295 K, which sits ~8 K below the SST and floods the column with
    # convection (CWV/max|w| blow up). Strict-RCEMIP runs (test_rcemip_plane_smoke,
    # run_rcemip_plane) keep the 295 K default. The surface-temp param was
    # renamed T_sfc -> T_v0 (the old T_sfc= kwarg raised TypeError).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=T_SFC_K, q_sfc=Q_SFC_FRAC, z_t=Z_T, Gamma=GAMMA_TROP,
    )
    qv_fn = make_wing2018_qv_ref_fn(q_sfc=Q_SFC_IC_FRAC, z_t=Z_T)
    # iter-95: pass p_sfc=101480 (Wing 2018 Tab A1) to enable the
    # bottom-up hydrostatic BC. Without this, the legacy top-down BC
    # produces 12 K too-hot T at the lowest model level, breaking
    # surface-flux coupling and preventing convection initiation.
    if args.vertical_grid == "uniform":
        hc = create_height_coordinate(
            n_levels=args.nlev, H=args.H, theta_ref_fn=theta_fn,
            p_sfc=101480.0,
        )
    else:
        hc = create_stretched_height_coordinate(
            n_levels=args.nlev, H=args.H, dz_sfc=args.dz_sfc,
            theta_ref_fn=theta_fn,
            p_sfc=101480.0,
        )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    z = hc.z_full
    nlev = z.shape[0]
    qv_profile = qv_fn(z)
    new_tracers = jnp.zeros((grid.ny, grid.nx, nlev, 3), dtype=jnp.float64)
    qv_3d = qv_profile[None, None, :] * jnp.ones(
        (grid.ny, grid.nx, nlev), dtype=jnp.float64,
    )
    new_tracers = new_tracers.at[..., 0].set(qv_3d)

    # Convection seed.
    # --------------------------------------------------------------
    # Default (--bubble-theta-pert 0): NO bubble. Convection spins
    # up from --qv-noise-amp boundary-layer qv perturbations + the
    # surface flux. This matches the RCEMIP1 protocol and avoids
    # the 2-Δz vertical mode that the single-bubble IC seeds at
    # the coarse dz~1.1 km production grid (see CRM_implementation.md
    # F6: hyperdiff up to 5e6 only delays — does not stop — the mode
    # that grows from the bubble's single-level perturbation).
    #
    # Optional (--bubble-theta-pert > 0): legacy warm-bubble IC,
    # cosine-tapered horizontally over ~10·dx and vertically over
    # z<1 km. Density set so the initial pressure is unperturbed
    # (ρ'/ρ_ref = −θ'/θ_ref). Use ONLY for short rising-thermal
    # smokes.
    bubble_amp = float(args.bubble_theta_pert)
    if bubble_amp > 0.0:
        ny, nx = grid.ny, grid.nx
        jj = jnp.arange(ny)
        ii = jnp.arange(nx)
        yy, xx = jnp.meshgrid(jj, ii, indexing="ij")
        yc, xc = (ny - 1) / 2.0, (nx - 1) / 2.0
        r_cells = jnp.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        r0 = 10.0  # bubble radius in cell units
        horiz = jnp.where(
            r_cells < r0,
            0.5 * (1.0 + jnp.cos(jnp.pi * r_cells / r0)),
            0.0,
        )
        z_top_bubble = 1000.0
        vert = jnp.where(z < z_top_bubble,
                         0.5 * (1.0 + jnp.cos(jnp.pi * z / z_top_bubble)),
                         0.0)
        bubble_theta = bubble_amp * horiz[:, :, None] * vert[None, None, :]
        bubble_rho = -hc.rho_ref * bubble_theta / hc.theta_ref
        state = state._replace(
            theta_prime=state.theta_prime.replace(data=bubble_theta),
            rho_prime=state.rho_prime.replace(data=bubble_rho),
        )

    # RCEMIP1 qv noise seed in the lowest 4 levels.
    qv_noise_amp = float(args.qv_noise_amp)
    if qv_noise_amp > 0.0:
        key = jax.random.PRNGKey(int(args.qv_noise_seed))
        n_seed_lev = min(4, nlev)
        noise = jax.random.uniform(
            key, shape=(grid.ny, grid.nx, n_seed_lev),
            minval=-qv_noise_amp, maxval=qv_noise_amp, dtype=jnp.float64,
        )
        noise = noise - jnp.mean(noise, axis=(0, 1), keepdims=True)
        # Lowest n_seed_lev model levels = LAST indices (k goes top→bottom).
        new_qv = new_tracers[..., 0]
        new_qv = new_qv.at[..., -n_seed_lev:].add(noise)
        new_qv = jnp.maximum(new_qv, 0.0)
        new_tracers = new_tracers.at[..., 0].set(new_qv)

    state = state._replace(
        tracers=state.tracers.replace(data=new_tracers),
    )

    # iter-181 theta' noise seed in the lowest 4 levels (RCEMIP /
    # Wing 2018 standard symmetry-breaker). iter-212 update: the
    # original iter-181 claim that theta' was safer than qv noise
    # because it doesn't enter the LW optical depth was REFUTED at
    # the iter-212 smoke — theta' nonzero amplitudes blow up via a
    # DIFFERENT mechanism (direct buoyancy injection, not radiation
    # feedback). Same F11 dx=4 km wall, just from a different
    # physics pathway.
    # iter-203: added --theta-noise-mode to pick the perturbation
    # pattern. "white" (default) is RCEMIP / Wing 2018 uniform
    # random; "smooth_k1" is the F11 fix-path-3 candidate — a
    # single-cosine smooth perturbation at the lowest non-trivial
    # wavenumber (kx=1, ky=1) with amplitude theta_noise_amp.
    # Both modes blow up at dx=4 km regardless of radiation status
    # (iter-212 smoke at amp=0.01 + --no-radiation: NaN at step 20).
    # Helper retained for LES-regime (dx<=1 km) experiments where
    # smaller dx may resolve convective cells before Kessler
    # saturates.
    theta_noise_amp = float(args.theta_noise_amp)
    if theta_noise_amp > 0.0:
        n_seed_lev = min(4, nlev)
        mode = args.theta_noise_mode
        if mode == "white":
            key_t = jax.random.PRNGKey(int(args.theta_noise_seed))
            theta_noise = jax.random.uniform(
                key_t, shape=(grid.ny, grid.nx, n_seed_lev),
                minval=-theta_noise_amp, maxval=theta_noise_amp,
                dtype=jnp.float64,
            )
            # Subtract horizontal mean so total energy is conserved
            # at IC (mirrors the qv-noise mean-removal pattern).
            theta_noise = theta_noise - jnp.mean(
                theta_noise, axis=(0, 1), keepdims=True,
            )
        elif mode == "smooth_k1":
            # iter-207: defer to the module-level
            # build_smooth_k1_pattern helper so unit tests exercise
            # the same code path the driver uses. The helper also
            # subtracts the horizontal mean explicitly so degenerate
            # grids (nx=1 or ny=1) don't violate the zero-mean
            # contract.
            pattern = build_smooth_k1_pattern(grid.ny, grid.nx)
            theta_noise = theta_noise_amp * pattern[:, :, None] * jnp.ones(
                (1, 1, n_seed_lev), dtype=jnp.float64,
            )
        else:
            raise SystemExit(
                f"error: --theta-noise-mode rejected: {mode!r}. "
                f"Expected 'white' or 'smooth_k1'."
            )
        new_theta_p = state.theta_prime.data
        new_theta_p = new_theta_p.at[..., -n_seed_lev:].add(theta_noise)
        state = state._replace(
            theta_prime=state.theta_prime.replace(data=new_theta_p),
        )

    return hc, state


def _surface_flux_physics_fn(grid, hc, tm):
    def physics_fn(state, grid_in, hc_in, tm_in):
        wspd = wind_speed_at_lowest_level_plane(state)
        T_sfc = jnp.full(wspd.shape, T_SFC_K)
        q_sfc = jnp.full(wspd.shape, Q_SFC_FRAC)
        dtheta_sfc, dq_sfc = compose_rce_surface_scalar_tendencies(
            state, hc_in, T_sfc, q_sfc, wspd,
            C_h=1.5e-3, gustiness_floor=5.0,
        )
        k_sfc = -1
        dtheta_data = jnp.zeros_like(state.theta_prime.data).at[
            ..., k_sfc
        ].set(dtheta_sfc)
        dtr_data = jnp.zeros_like(state.tracers.data).at[
            ..., k_sfc, 0
        ].set(dq_sfc)
        return PlaneNonHydrostaticTendencies(
            du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
            dv_dt=state.v.replace(data=jnp.zeros_like(state.v.data)),
            dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
            dtheta_prime_dt=state.theta_prime.replace(data=dtheta_data),
            drho_prime_dt=state.rho_prime.replace(
                data=jnp.zeros_like(state.rho_prime.data),
            ),
            dphis_dt=state.phis.replace(
                data=jnp.zeros_like(state.phis.data),
            ),
            dtracers_dt=state.tracers.replace(data=dtr_data),
        )
    return physics_fn


def _sum_tendencies(*tendencies):
    out_kwargs = {}
    fields = (
        "du_dt", "dv_dt", "dw_dt", "dtheta_prime_dt",
        "drho_prime_dt", "dphis_dt", "dtracers_dt",
    )
    for fld in fields:
        sum_data = sum(getattr(t, fld).data for t in tendencies)
        out_kwargs[fld] = getattr(tendencies[0], fld).replace(data=sum_data)
    return PlaneNonHydrostaticTendencies(**out_kwargs)


def build_fast_physics_fn(args, grid, hc, tm):
    """Per-step physics: surface flux + microphysics. Stiff sources
    that must respond to dycore evolution each step."""
    fns = [_surface_flux_physics_fn(grid, hc, tm)]
    fns.append(make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler"),
        model_type="plane", dt=args.dt / args.n_physics_substeps,
    ))

    def physics_fn(state, grid_in, hc_in, tm_in):
        tendencies = [fn(state, grid_in, hc_in, tm_in) for fn in fns]
        return _sum_tendencies(*tendencies)
    return physics_fn


def build_slow_physics_fn(args, grid, hc, tm):
    """Slow physics: gray radiation. Called every
    ``--rad-call-interval-s`` seconds; tendency held between calls."""
    rad_fn = make_radiation_physics(
        RadiationConfig(scheme="gray"), model_type="plane",
    )
    # RAD-7 (iter-48): pin the surface radiative boundary to the fixed SST
    # (SAM uses the SST for σ·ε·T_sfc⁴, not the drifting lowest-level air T =
    # the default T[..., -1]). Local ncol per rank = grid.ny·grid.nx.
    rad_fn.set_T_sfc_override(
        jnp.full((int(grid.ny * grid.nx),), float(T_SFC_K), dtype=jnp.float64))

    def physics_fn(state, grid_in, hc_in, tm_in):
        return rad_fn(state, grid_in, hc_in, tm_in)
    return physics_fn


def _broadcast_state(state, comm, root=0):
    field_names = (
        "u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers",
    )
    new_fields = {}
    for name in field_names:
        fld = getattr(state, name)
        arr = np.asarray(fld.data)
        if comm.Get_rank() != root:
            arr = np.empty(arr.shape, dtype=arr.dtype)
        comm.Bcast(arr, root=root)
        new_fields[name] = fld.replace(data=jnp.asarray(arr))
    return state._replace(**new_fields)


_DD_FIELD_NAMES = (
    "u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers",
)


def _scatter_state(state, layout):
    """Build rank-local slab from a globally-replicated state.

    Each rank independently calls :func:`scatter_plane_field` (pure
    slicing — no MPI). Assumes every rank has built the same global
    IC deterministically before this call. Avoids the bcast cost of
    a "rank-0 owns; everyone else receives" pattern at IC time."""
    new_fields = {}
    for name in _DD_FIELD_NAMES:
        fld = getattr(state, name)
        local_data = scatter_plane_field(fld.data, layout)
        new_fields[name] = fld.replace(data=local_data)
    return state._replace(**new_fields)


def _gather_state(state_local, layout):
    """Gather rank-local slabs to a single global state on rank 0.

    Per-field :func:`gather_plane_field` (uses ``comm.gather`` under
    the hood — a COLLECTIVE that every rank must enter). Returns the
    gathered global state on rank 0 and ``None`` on all other ranks.

    Bug avoided: previously this returned early on the first
    ``gather_plane_field == None`` (i.e. on non-rank-0 after the
    first field), which left rank 0 blocked on the next gather
    waiting for a peer that had already exited. We now ALWAYS
    participate in every collective regardless of rank.
    """
    rank0_arrays = []
    for name in _DD_FIELD_NAMES:
        fld = getattr(state_local, name)
        global_data = gather_plane_field(fld.data, layout)
        rank0_arrays.append(global_data)
    # Non-rank-0 has None in every slot — return None to signal the
    # caller this isn't the rank that owns the gathered state.
    if rank0_arrays[0] is None:
        return None
    rank0_fields = {}
    for name, arr in zip(_DD_FIELD_NAMES, rank0_arrays):
        fld = getattr(state_local, name)
        rank0_fields[name] = fld.replace(data=arr)
    return state_local._replace(**rank0_fields)


def save_snapshot(out_dir, day_idx, t_sim, state, hc):
    """x-y surface fields snapshot."""
    cwv = np.asarray(column_water_vapor_plane(state, hc))
    mse = np.asarray(column_moist_static_energy_plane(state, hc))
    precip = np.asarray(precipitation_rate_proxy_plane(state, hc))
    theta_sfc = np.asarray(
        hc.theta_ref[-1] + state.theta_prime.data[..., -1]
    )
    T_sfc = theta_sfc * float(hc.exner_ref[-1])
    qv_sfc = np.asarray(state.tracers.data[..., -1, 0])
    qc_sfc = np.asarray(state.tracers.data[..., -1, 1])
    qr_sfc = np.asarray(state.tracers.data[..., -1, 2])
    u_sfc = np.asarray(state.u.data[..., -1])
    v_sfc = np.asarray(state.v.data[..., -1])
    wind_sfc = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    snap_path = out_dir / "snapshots" / f"snap_day_{day_idx:04d}.npz"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snap_path,
        t_sim=t_sim, day=t_sim / SEC_PER_DAY,
        cwv=cwv, mse=mse, precip=precip,
        T_sfc=T_sfc, qv_sfc=qv_sfc, qc_sfc=qc_sfc, qr_sfc=qr_sfc,
        u_sfc=u_sfc, v_sfc=v_sfc, wind_sfc=wind_sfc,
    )


def save_snapshot_3d(out_dir, hr_idx, t_sim, state, hc):
    """Full 3D MSE/q_v/T/condensate volumes [float32, compressed].

    Stored at ``snapshots_3d/snap_hr_NNNN.npz`` with arrays ``mse``, ``qv``,
    ``T``, ``cond`` each shape (ny, nx, nlev); plus ``z`` (nlev,) and scalars
    ``t_sim``, ``day``, ``hour``. ``cond`` is the total condensate mixing
    ratio (all non-vapor tracer mass slots, i.e. q_c + q_r + ... [kg/kg]) so
    the volume is directly consumable as an SCM-RCE campaign reference (which
    reads ``z``/``T``/``mse``/``cond``; q_v is inverted from ``mse``).
    """
    mse_3d = np.asarray(
        moist_static_energy_3d_plane(state, hc), dtype=np.float32,
    )
    qv_3d = np.asarray(state.tracers.data[..., 0], dtype=np.float32)
    T_3d = np.asarray(temperature_3d_plane(state, hc), dtype=np.float32)
    # Total condensate = sum of the non-vapor tracer MASS slots (slot 0 is
    # q_v). For the warm-rain Kessler CRM these are q_c (slot 1) and q_r
    # (slot 2); summing all slots >= 1 stays correct if more hydrometeor
    # mass tracers are added. Clipped to >= 0 (advection can leave tiny
    # negatives without the mass fixer).
    n_tracers = state.tracers.data.shape[-1]
    if n_tracers > 1:
        cond_3d = np.asarray(
            jnp.clip(state.tracers.data[..., 1:], 0.0).sum(axis=-1),
            dtype=np.float32,
        )
    else:
        cond_3d = np.zeros_like(qv_3d)
    z = np.asarray(hc.z_full, dtype=np.float32)
    snap_path = out_dir / "snapshots_3d" / f"snap_hr_{hr_idx:04d}.npz"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snap_path,
        t_sim=t_sim, day=t_sim / SEC_PER_DAY, hour=t_sim / 3600.0,
        z=z, mse=mse_3d, qv=qv_3d, T=T_3d, cond=cond_3d,
    )


def save_profile(out_dir, day_idx, t_sim, state, hc):
    """Horizontally-averaged vertical profiles."""
    theta_tot = hc.theta_ref + state.theta_prime.data
    T = theta_tot * hc.exner_ref
    T_prof = np.asarray(jnp.mean(T, axis=(0, 1)))
    qv_prof = np.asarray(jnp.mean(state.tracers.data[..., 0], axis=(0, 1)))
    qc_prof = np.asarray(jnp.mean(state.tracers.data[..., 1], axis=(0, 1)))
    qr_prof = np.asarray(jnp.mean(state.tracers.data[..., 2], axis=(0, 1)))
    cf_prof = np.asarray(cloud_fraction_profile_plane(state, hc))
    w_var = np.asarray(jnp.var(state.w.data, axis=(0, 1)))
    z = np.asarray(hc.z_full)
    z_half = np.asarray(hc.z_half)
    prof_path = out_dir / "profiles" / f"prof_day_{day_idx:04d}.npz"
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prof_path,
        t_sim=t_sim, day=t_sim / SEC_PER_DAY,
        z=z, z_half=z_half,
        T=T_prof, qv=qv_prof, qc=qc_prof, qr=qr_prof,
        cloud_fraction=cf_prof, w_variance=w_var,
    )


def write_progress(out_dir, t_sim, total_t, step, total_steps,
                   wall_elapsed):
    day = t_sim / SEC_PER_DAY
    target_day = total_t / SEC_PER_DAY
    progress = step / max(total_steps, 1)
    eta_sec = (wall_elapsed / max(step, 1)) * (total_steps - step)
    p_path = out_dir / "progress.txt"
    p_path.write_text(
        f"current_day = {day:.3f}\n"
        f"target_day  = {target_day:.3f}\n"
        f"progress    = {progress * 100:.2f}%\n"
        f"step        = {step}/{total_steps}\n"
        f"wall_elapsed_s = {wall_elapsed:.1f}\n"
        f"eta_wall_s     = {eta_sec:.1f}\n"
        f"eta_wall_hours = {eta_sec / 3600:.2f}\n"
    )


def main():
    args = parse_args()
    # PRECISION cross-check (codex): x64 is a module-IMPORT toggle set by BOTH
    # JAX_ENABLE_X64 and our LEGOESM_RCE_MPI_FP32, so verify the ACTUAL runtime
    # x64 state matches --precision — checking our env var alone would miss
    # JAX_ENABLE_X64=1 + --precision float32 (which would silently run float64)
    # and the reverse. Fail LOUD on a mismatch.
    _x64_on = bool(jax.config.jax_enable_x64)
    _want_x64 = args.precision == "float64"
    if _x64_on != _want_x64:
        raise SystemExit(
            f"error: --precision {args.precision} but jax_enable_x64={_x64_on} "
            "(float64 needs x64 ON, float32 needs x64 OFF). x64 is set at import "
            "from JAX_ENABLE_X64 and LEGOESM_RCE_MPI_FP32. For float32 launch "
            "with LEGOESM_RCE_MPI_FP32=1 and WITHOUT JAX_ENABLE_X64=1; for "
            "float64 leave LEGOESM_RCE_MPI_FP32 unset."
        )
    if args.implicit_buoyancy and not args.semi_implicit_acoustic:
        raise SystemExit(
            "error: --implicit-buoyancy rejected: requires "
            "--semi-implicit-acoustic (the Klemp-Wilhelmson 1978 "
            "substitution lives inside the column tridiagonal solve)."
        )
    if args.adaptive_dt:
        # iter-228: flag parses but isn't wired through the time loop
        # yet (~80 LOC restructure to switch the for-loop to a
        # ``while t_sim < target_t`` loop with per-step Ca_adv
        # diagnostic + dt shrinkage). Refuse explicitly rather than
        # silently no-op'ing the user's expectation.
        raise SystemExit(
            "error: --adaptive-dt is a parse-only stub today. The "
            "runtime CFL-monitoring + dt-shrinkage loop is reserved "
            "for a follow-on PR (see CRM_implementation.md F11 "
            "fix-path-4 feasibility note). iter-233 added the "
            "FV3-style replacement: use ``--n-outer-split auto`` "
            "(or a specific integer) to subcycle the outer step "
            "with a static, scan-friendly substep count."
        )
    # iter-233: resolve --n-outer-split (str CLI arg) to a Python
    # int. Three forms:
    #   "1"   -> 1 (default, no outer subcycling, preserves
    #            iter-183 production contract)
    #   "N"   -> int(N), user-pinned static count (must be >= 1)
    #   "auto" -> select_n_outer_split(dt, dx, max_wind_safe,
    #            cfl_safe) at parse time from a conservative
    #            max-wind estimate (FV3-style).
    # The output ``n_outer_split`` is a STATIC Python int — used
    # as ``range(n_outer_split)`` in the outer loop below; no
    # tracing, no XLA retrace, fully AD-safe.
    if args.n_outer_split == "auto":
        from legoesm.timestepping.split_explicit import (
            select_n_outer_split,
        )
        # iter-235 (Codex iter-234 round-2 LOW#1): catch raw
        # ValueError from select_n_outer_split (bad
        # --max-wind-safe / --cfl-safe / --dt / --dx) and re-raise
        # as clean SystemExit. The earlier finite/range CLI
        # validator runs AFTER this block, so auto-mode would
        # otherwise crash with a raw stack trace from
        # math.ceil(nan) / divide-by-zero.
        try:
            n_outer_split = select_n_outer_split(
                args.dt, args.dx,
                max_wind_safe=args.max_wind_safe,
                cfl_safe=args.cfl_safe,
            )
        except ValueError as exc:
            raise SystemExit(
                f"error: --n-outer-split=auto failed: {exc}. Check "
                f"--dt, --dx, --max-wind-safe, --cfl-safe are all "
                f"finite positive (cfl-safe in (0,1])."
            )
    else:
        try:
            n_outer_split = int(args.n_outer_split)
        except ValueError:
            raise SystemExit(
                f"error: --n-outer-split={args.n_outer_split!r} "
                f"must be 'auto' or a positive integer (1 = no "
                f"subcycling, default)."
            )
        if n_outer_split < 1:
            raise SystemExit(
                f"error: --n-outer-split={n_outer_split} must be "
                f">= 1. Use '1' (default) for no subcycling, "
                f"'auto' for FV3-style trace-time selection, or a "
                f"specific integer for user-pinned subcycling."
            )
    dt_inner = args.dt / n_outer_split
    # iter-67/68: validate numeric CLI args reject NaN/inf with
    # concise SystemExit. iter-67 hardcoded 17 arg names; iter-68
    # auto-detects via vars(args) so a future ``--new-coeff``
    # added to parse_args is automatically validated.
    #
    # Skip ``rad_call_interval_s``: validated by
    # physics_schedule.radiation_call_every_steps (iter-43) which
    # iter-65 wrapped to also produce a clean SystemExit. Including
    # it here would double-validate but the helper error message is
    # more specific (cites the schedule constraints), so let the
    # helper handle it.
    import math as _math
    _SKIP_FINITE_CHECK = {"rad_call_interval_s"}
    for _attr, _val in vars(args).items():
        if _attr in _SKIP_FINITE_CHECK:
            continue
        # Only validate float-typed args (bool is a subclass of int,
        # so `isinstance(True, float)` is False — safe).
        if not isinstance(_val, float):
            continue
        if not _math.isfinite(_val):
            _flag = "--" + _attr.replace("_", "-")
            raise SystemExit(
                f"error: {_flag} rejected: must be finite, got {_val!r}"
            )
    # iter-70 Codex HIGH coverage: extend iter-67/68/69 validation to
    # ranges, not just finiteness/integer-positivity. Without these,
    # bad-but-finite values silently corrupt the run:
    # * --days -1 → total_steps = -17280 → empty loop → "Done. 0 steps"
    # * --snapshot-hours 0 → division-by-zero in snap_dt → save every step
    # * --dx <= 0, --H <= 0, --dz-sfc <= 0 → grid creation fails downstream
    # * --bubble-theta-pert -1, --qv-noise-amp -1 → silent skip of seed
    # * --acoustic-off-centering -0.1 → physically invalid; > 1.0 → unstable
    _POSITIVE_FLOATS = {  # must be > 0 (zero meaningless or div-by-zero risk)
        "--dt": args.dt,
        "--days": args.days,
        "--dx": args.dx,
        "--H": args.H,
        "--dz-sfc": args.dz_sfc,
        "--snapshot-hours": args.snapshot_hours,
        "--profile-days": args.profile_days,
    }
    for _flag, _val in _POSITIVE_FLOATS.items():
        if _val <= 0.0:
            raise SystemExit(
                f"error: {_flag} rejected: must be positive, got {_val!r}"
            )
    _NONNEG_FLOATS = {  # must be >= 0 (0 is a meaningful "disabled" sentinel)
        "--snapshot-3d-hours": args.snapshot_3d_hours,
        "--smag-cs": args.smag_cs,
        "--hyperdiff": args.hyperdiff,
        "--sponge-coeff": args.sponge_coeff,
        "--sponge-width": args.sponge_width,
        "--vertical-theta-diffusion": args.vertical_theta_diffusion,
        "--bubble-theta-pert": args.bubble_theta_pert,
        "--qv-noise-amp": args.qv_noise_amp,
        # iter-188 Codex gap: --theta-noise-amp was added in iter-181
        # but not wired into the non-negative validator. A negative
        # amplitude was silently treated as "no noise" by the
        # ``if theta_noise_amp > 0.0`` gate at line 405, masking a
        # caller-side typo. Reject explicitly.
        "--theta-noise-amp": args.theta_noise_amp,
        "--c-h": args.c_h,
    }
    for _flag, _val in _NONNEG_FLOATS.items():
        if _val < 0.0:
            raise SystemExit(
                f"error: {_flag} rejected: must be non-negative, "
                f"got {_val!r}"
            )
    # --acoustic-off-centering beta must be in [0, 1) per Skamarock-Klemp.
    if not (0.0 <= args.acoustic_off_centering < 1.0):
        raise SystemExit(
            f"error: --acoustic-off-centering rejected: must be in "
            f"[0, 1), got {args.acoustic_off_centering!r}"
        )
    # iter-69: positive-int guards for grid dims + substep counts.
    # Without these, ``--nx 0`` would crash deep in plane_mpi
    # ``make_plane_pencil_layout`` with an opaque
    # ``ValueError: n_ranks_x=1 exceeds nx_global=0``. argparse
    # type=int accepts 0/negative for these args.
    _POSITIVE_INTS = {
        "--nx": args.nx,
        "--ny": args.ny,
        "--nlev": args.nlev,
        "--n-acoustic-substeps": args.n_acoustic_substeps,
        "--n-physics-substeps": args.n_physics_substeps,
        "--log-every-steps": args.log_every_steps,
    }
    for _flag, _val in _POSITIVE_INTS.items():
        if _val <= 0:
            raise SystemExit(
                f"error: {_flag} rejected: must be positive integer, "
                f"got {_val!r}"
            )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if n_ranks == 12:
        n_ranks_y, n_ranks_x = 3, 4
    elif n_ranks == 4:
        n_ranks_y, n_ranks_x = 2, 2
    elif n_ranks == 1:
        n_ranks_y, n_ranks_x = 1, 1
    else:
        n_ranks_y, n_ranks_x = 1, n_ranks

    if args.ny % n_ranks_y != 0 or args.nx % n_ranks_x != 0:
        if rank == 0:
            print(f"ERROR: grid not divisible by {n_ranks_y}x{n_ranks_x}.")
        return

    # iter-186 Codex HIGH: the halo width must match the advection
    # scheme's stencil reach. Pre-iter-186 the layout always
    # defaulted to halo=1 so --use-dd --advection van_leer (the
    # iter-183 production combo) crashed at the first halo
    # slow-tendency call. iter-187 promoted the per-scheme halo
    # requirements to a public map in compressible_euler_plane so
    # caller (driver) + callee (halo dispatch) consult the SAME
    # source of truth — no more silent drift. iter-192 reuses the
    # _ADV_HALO_REQ alias bound during parse_args() so the lookup
    # cannot reference a different module / map snapshot.
    _required_halo = _ADV_HALO_REQ[args.advection]
    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=args.ny, nx_global=args.nx,
        halo=_required_halo,
    )
    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    hc, state = build_height_coord_and_state(args, grid)
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=args.semi_implicit_acoustic,
        acoustic_off_centering=args.acoustic_off_centering,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=args.n_acoustic_substeps,
        vertical_theta_diffusion=args.vertical_theta_diffusion,
        horizontal_advection_scheme=args.advection,
        vertical_tracer_advection=args.vertical_tracer_advection,
        implicit_buoyancy=args.implicit_buoyancy,
    )
    # Save the GLOBAL state for snapshot dumping (needed on every rank
    # in legacy mode; saved on rank 0 in DD mode after gather).
    state_global_ic = state

    # ----------------------------------------------------------------- #
    # Domain-decomposition: scatter IC + rebuild local grid / model /   #
    # physics so every rank steps its own slab via step_halo.           #
    # ----------------------------------------------------------------- #
    if args.use_dd and n_ranks > 1:
        state = _scatter_state(state, layout)
        grid_local = make_plane_pencil_grid(
            layout, dx=args.dx, dy=args.dx, nlev=args.nlev,
            dtype=jnp.float64,
        )
        terrain_local = make_flat_plane_terrain_metric(grid_local, hc)
        model = PlaneCompressibleEulerModel(
            grid_local, hc, terrain_local, config=cfg,
        )
        fast_physics_fn = build_fast_physics_fn(
            args, grid_local, hc, terrain_local,
        )
        slow_physics_fn = build_slow_physics_fn(
            args, grid_local, hc, terrain_local,
        )
        grid = grid_local
        terrain = terrain_local
    else:
        model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)
        fast_physics_fn = build_fast_physics_fn(args, grid, hc, terrain)
        slow_physics_fn = build_slow_physics_fn(args, grid, hc, terrain)
    # iter-55 Codex MEDIUM#1 fix: skip the schedule helper entirely
    # when --no-radiation is set. Otherwise a bogus --rad-call-
    # interval-s (NaN/inf, rejected by physics_schedule's iter-43
    # validation) crashes the driver even though the value is
    # unused. With --no-radiation the schedule is irrelevant —
    # cached_rad_tend stays None for the full run.
    if args.no_radiation:
        rad_call_every_steps = 1  # harmless sentinel; never consulted
    else:
        from legoesm.driver.physics_schedule import (
            radiation_call_every_steps as _rad_every,
        )
        # iter-65 Codex iter-55 LOW#2 fix: convert raw ValueError from
        # the schedule helper (iter-43 NaN/inf/dt<=0 guard) into a
        # SystemExit with a concise CLI-style message. A user typing
        # ``--rad-call-interval-s nan`` no longer gets a Python
        # traceback; they get a clear ``error: --rad-call-interval-s
        # rejected: ...`` and exit code 2.
        try:
            rad_call_every_steps = _rad_every(
                args.rad_call_interval_s, args.dt,
            )
        except ValueError as exc:
            raise SystemExit(
                f"error: --rad-call-interval-s rejected: {exc}"
            ) from None
    cached_rad_tend = [None]  # mutable closure for the cache

    def apply_physics_substep(state, dt_sub):
        """Forward-Euler physics tendency applied for dt_sub.
        Combines fast-physics tendency + cached radiation tendency."""
        tend = fast_physics_fn(state, grid, hc, terrain)
        if cached_rad_tend[0] is not None:
            tend = _sum_tendencies(tend, cached_rad_tend[0])
        new_u = state.u.data + dt_sub * tend.du_dt.data
        new_v = state.v.data + dt_sub * tend.dv_dt.data
        new_w = state.w.data + dt_sub * tend.dw_dt.data
        new_theta = (
            state.theta_prime.data + dt_sub * tend.dtheta_prime_dt.data
        )
        new_rho = state.rho_prime.data + dt_sub * tend.drho_prime_dt.data
        new_tr = state.tracers.data + dt_sub * tend.dtracers_dt.data
        return state._replace(
            u=state.u.replace(data=new_u),
            v=state.v.replace(data=new_v),
            w=state.w.replace(data=new_w),
            theta_prime=state.theta_prime.replace(data=new_theta),
            rho_prime=state.rho_prime.replace(data=new_rho),
            tracers=state.tracers.replace(data=new_tr),
        )

    def physics_split(state, dt_outer, n_sub):
        dt_sub = dt_outer / n_sub
        for _ in range(n_sub):
            state = apply_physics_substep(state, dt_sub)
            state = apply_positive_filter_state(
                state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
            )
        return state
    # owned_mask + target water mass.
    if args.use_dd and n_ranks > 1:
        # Per-rank local mask is all-ones (every cell of the local slab
        # is owned by this rank; halo is added by step_halo's exchange).
        owned_mask = jnp.ones(
            (layout.ny_local, layout.nx_local), dtype=jnp.float64,
        )
        target_water = compute_total_water_mass_plane_mpi(
            state, hc, grid, layout, owned_mask,
        )
    else:
        # Legacy rank-0-broadcast path: owned_mask is in GLOBAL coords
        # so the MPI helpers can mark only this rank's slice of the
        # replicated full state.
        owned_mask = jnp.zeros((args.ny, args.nx), dtype=jnp.float64)
        owned_mask = owned_mask.at[
            layout.iy_start:layout.iy_end,
            layout.ix_start:layout.ix_end,
        ].set(1.0)
        target_water = compute_total_water_mass_plane(state, hc, grid)

    total_t = args.days * SEC_PER_DAY
    total_steps = int(total_t / args.dt)
    # iter-75: catch the silent-pass class where ``args.dt >>
    # total_t`` (e.g. ``--dt 1e10`` + ``--days 0.0005``) makes the
    # int division round to 0 and the outer loop skips entirely,
    # producing "Done. 0 steps" silently. iter-70 caught --days <= 0
    # at parse time, but a finite positive dt > total_t is a
    # different path. Reject loudly.
    if total_steps < 1:
        raise SystemExit(
            f"error: --dt rejected: total_steps={total_steps} "
            f"(computed from --days={args.days} × 86400 s / "
            f"--dt={args.dt}); must be >= 1. Either --dt is too "
            f"large or --days is too small for the chosen --dt to "
            f"advance at least one outer step."
        )
    snap_dt = args.snapshot_hours * 3600.0
    snap3d_dt = args.snapshot_3d_hours * 3600.0
    snap3d_enabled = args.snapshot_3d_hours > 0.0
    prof_dt = args.profile_days * SEC_PER_DAY

    out_dir = Path(args.output)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "snapshots").mkdir(exist_ok=True)
        (out_dir / "profiles").mkdir(exist_ok=True)
        if snap3d_enabled:
            (out_dir / "snapshots_3d").mkdir(exist_ok=True)
        log_path = out_dir / "log.txt"
        log_f = open(log_path, "w", buffering=1)
        # iter-235 (Codex iter-234 round-2 LOW#2): log dt_inner
        # alongside the outer dt so a post-run analyst doesn't see
        # ``dt=20`` beside a Ca_substep that was actually computed
        # at dt_inner=2.5 (n_outer_split=8). For the default
        # n_outer_split=1 path, dt_inner==dt and the field is
        # redundant; logging it unconditionally keeps the schema
        # stable.
        log_f.write(
            f"# RCE MPI LONG  n_ranks={n_ranks} grid={args.ny}x{args.nx} "
            f"nlev={args.nlev} dx={args.dx} dt={args.dt} "
            f"dt_inner={dt_inner} "
            f"days={args.days} total_steps={total_steps}\n"
        )
        log_f.write(
            f"# physics: "
            f"{'NO radiation' if args.no_radiation else 'gray radiation'}"
            f" + Kessler microphysics + "
            f"Smagorinsky LES (cs={args.smag_cs})\n"
        )
        # iter-231 (Codex round-2 MEDIUM#1): the contract fingerprint
        # at run_rce_mpi_long.py:1051 dropped advection +
        # acoustic_off_centering + mass-fixer mode, so a regression
        # test could not prove those parts of the iter-183 contract
        # actually ran. Logging them on a discrete ``# config:`` line
        # makes the contract grep-able post-run and gives the iter-183
        # envelope test a hard fingerprint to assert against.
        log_f.write(
            f"# config: advection={args.advection} "
            f"acoustic_off_centering={args.acoustic_off_centering} "
            f"mass_fixer={'off' if args.no_mass_fixer else 'on'} "
            f"si_acoustic={'on' if args.semi_implicit_acoustic else 'off'} "
            f"n_acoustic_substeps={args.n_acoustic_substeps} "
            f"hyperdiff={args.hyperdiff} "
            f"n_outer_split={n_outer_split}\n"
        )
        log_f.write(
            f"# step,day,CWV_mean,CWV_max,MSE_mean,max|w|,"
            f"max(qc),max(qr),max(precip_mm_day),Ca_substep\n"
        )

        # Save IC snapshot + profile. In DD mode the `state` variable
        # holds the rank-local slab; dump from the pre-scatter global
        # IC so snapshots stay full-domain shape.
        ic_state_for_io = state_global_ic if args.use_dd else state
        save_snapshot(out_dir, 0, 0.0, ic_state_for_io, hc)
        save_profile(out_dir, 0, 0.0, ic_state_for_io, hc)
        if snap3d_enabled:
            save_snapshot_3d(out_dir, 0, 0.0, ic_state_for_io, hc)
        write_progress(out_dir, 0.0, total_t, 0, total_steps, 0.0)

    wall_start = time.time()
    next_snap_t = snap_dt
    next_snap3d_t = snap3d_dt
    next_prof_t = prof_dt
    # iter-40: explicit counter for radiation tendency refreshes. Used
    # by test_plane_crm_production_scale_132x132_with_radiation /
    # ..._envelope (iter-38/39) to assert the radiation tick branch
    # actually fires the expected number of times. Codex iter-39
    # MEDIUM#1 / MEDIUM#3 gap: without an observable count the test
    # could pass under a broken rad_call_every_steps arithmetic.
    rad_call_count = 0
    # iter-55 Codex MEDIUM#2 fix: init step + t_sim BEFORE the loop
    # so that ``--days 0`` (used by some smoke-tests as a CLI-parse-
    # only dry-run) doesn't crash the final ``Done.`` print with a
    # NameError. With total_steps=0 the loop body is skipped and the
    # report shows ``Done. 0 steps...``.
    step = 0
    t_sim = 0.0

    def _maybe_fire_radiation(step_idx):
        """Fire gray-radiation slow tendency on schedule. Returns
        True if it fired this step (caller increments the counter)."""
        if args.no_radiation:
            return False
        if (step_idx - 1) % rad_call_every_steps != 0:
            return False
        cached_rad_tend[0] = slow_physics_fn(state, grid, hc, terrain)
        return True

    for step in range(1, total_steps + 1):
        if args.use_dd and n_ranks > 1:
            # ----------------------------------------------------- #
            # True per-rank DD path.                                #
            # ----------------------------------------------------- #
            if _maybe_fire_radiation(step):
                rad_call_count += 1
            # iter-233 FV3-style outer subcycling: ``n_outer_split``
            # is a STATIC Python int chosen at parse_args time
            # (auto: from a conservative max-wind CFL; explicit:
            # user-pinned). The Python ``for`` loop unrolls at
            # trace time so the inner ``step_halo`` JIT cache is
            # hit once per outer step.
            for _sub in range(n_outer_split):
                state = model.step_halo(
                    state, dt=dt_inner, layout=layout,
                    owned_mask=owned_mask,
                )
            state = physics_split(state, args.dt, args.n_physics_substeps)
            state = apply_positive_filter_state(
                state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
            )
            state = remove_horizontal_mean_wind_plane_mpi(
                state, layout, owned_mask,
            )
            if not args.no_mass_fixer:
                state = fix_moist_mass_plane_mpi(
                    state, hc, grid, layout, owned_mask,
                    target_total_water=target_water,
                )
        else:
            # ----------------------------------------------------- #
            # Legacy rank-0-dycore + broadcast path (replicated).   #
            # ----------------------------------------------------- #
            if rank == 0:
                if _maybe_fire_radiation(step):
                    rad_call_count += 1
                # iter-233 FV3-style outer subcycling (legacy path).
                for _sub in range(n_outer_split):
                    state = model.step(
                        state, dt=dt_inner, physics_fn=None,
                    )
                state = physics_split(
                    state, args.dt, args.n_physics_substeps,
                )
            state = _broadcast_state(state, comm, root=0)
            state = apply_positive_filter_state(
                state, tracer_slots_to_filter=(0, 1, 2), mode="clip",
            )
            state = remove_horizontal_mean_wind_plane_mpi(
                state, layout, owned_mask,
            )
            _ = compute_total_water_mass_plane_mpi(
                state, hc, grid, layout, owned_mask,
            )
            if not args.no_mass_fixer:
                state = fix_moist_mass_plane(
                    state, hc, grid, target_total_water=target_water,
                )
        t_sim = step * args.dt
        wall_elapsed = time.time() - wall_start

        # In DD mode, diagnostics + snapshots need a globally-assembled
        # state on rank 0. Gather lazily — only on log/snapshot/profile
        # ticks, never every step.
        need_gather = args.use_dd and n_ranks > 1 and (
            step % args.log_every_steps == 0 or step == 1
            or t_sim >= next_snap_t
            or (snap3d_enabled and t_sim >= next_snap3d_t)
            or t_sim >= next_prof_t
        )
        state_for_io = (
            _gather_state(state, layout) if need_gather else state
        )

        if rank == 0:
            # Live progress every step.
            write_progress(
                out_dir, t_sim, total_t, step, total_steps, wall_elapsed,
            )
            # Per-step diagnostics. In DD mode `state_for_io` is the
            # gathered global state on rank 0; in legacy mode it is the
            # same as `state` (rank-0 already holds the full domain).
            if step % args.log_every_steps == 0 or step == 1:
                state_io = state_for_io
                # Build a global grid for CFL diagnostics in DD mode
                # (cn.acoustic depends only on rho_ref + dx + dt).
                grid_io = grid
                if args.use_dd and n_ranks > 1:
                    grid_io = create_plane_grid(
                        nx=args.nx, ny=args.ny, nlev=args.nlev,
                        dx=args.dx, dy=args.dx, dtype=jnp.float64,
                    )
                cwv = column_water_vapor_plane(state_io, hc)
                mse = column_moist_static_energy_plane(state_io, hc)
                precip = precipitation_rate_proxy_plane(state_io, hc)
                # iter-233 Codex MEDIUM: under outer subcycling
                # (n_outer_split>1) the dycore actually steps at
                # ``dt_inner = args.dt / n_outer_split``, not
                # ``args.dt``. Passing args.dt inflates the reported
                # Ca_substep by n_outer_split and would trigger
                # false-alarm CFL warnings even though the physics
                # is correct (each subcycle uses dt_inner internally
                # for acoustic substepping).
                cn = compute_courant_numbers_plane(
                    state_io, hc, grid_io, dt_inner,
                    n_acoustic_substeps=args.n_acoustic_substeps,
                )
                max_w = float(jnp.max(jnp.abs(state_io.w.data)))
                max_qc = float(jnp.max(state_io.tracers.data[..., 1]))
                max_qr = float(jnp.max(state_io.tracers.data[..., 2]))
                precip_mmday = float(jnp.max(precip)) * SEC_PER_DAY
                log_f.write(
                    f"{step},{t_sim / SEC_PER_DAY:.6f},"
                    f"{float(jnp.mean(cwv)):.4e},"
                    f"{float(jnp.max(cwv)):.4e},"
                    f"{float(jnp.mean(mse)):.4e},"
                    f"{max_w:.4e},{max_qc:.4e},{max_qr:.4e},"
                    f"{precip_mmday:.4e},"
                    f"{float(cn.acoustic):.4f}\n"
                )
                if not bool(jnp.all(jnp.isfinite(cwv))):
                    log_f.write(f"# BAIL: NaN at step {step}\n")
                    break

            # Snapshot every snapshot_hours. state_for_io is the
            # gathered global state on rank 0 in DD mode, or `state`
            # in legacy mode (rank 0 already holds the full domain).
            if t_sim >= next_snap_t:
                day_idx = int(round(t_sim / SEC_PER_DAY))
                save_snapshot(out_dir, day_idx, t_sim, state_for_io, hc)

            # 3D snapshot every snapshot_3d_hours.
            if snap3d_enabled and t_sim >= next_snap3d_t:
                hr_idx = int(round(t_sim / 3600.0))
                save_snapshot_3d(
                    out_dir, hr_idx, t_sim, state_for_io, hc,
                )

            # Profile every profile_days.
            if t_sim >= next_prof_t:
                day_idx = int(round(t_sim / SEC_PER_DAY))
                save_profile(out_dir, day_idx, t_sim, state_for_io, hc)

        # Keep gather/snapshot/profile timers identical on every rank.
        # These thresholds feed need_gather, which gates collective gathers.
        if t_sim >= next_snap_t:
            next_snap_t += snap_dt
        if snap3d_enabled and t_sim >= next_snap3d_t:
            next_snap3d_t += snap3d_dt
        if t_sim >= next_prof_t:
            next_prof_t += prof_dt

    if rank == 0:
        log_f.close()
        write_progress(
            out_dir, t_sim, total_t, total_steps, total_steps,
            time.time() - wall_start,
        )
        print(
            f"Done. {step} steps, {t_sim / SEC_PER_DAY:.3f} days sim. "
            f"Wall: {(time.time() - wall_start) / 60:.1f} min. "
            f"rad_calls={rad_call_count}."
        )


if __name__ == "__main__":
    main()
