"""Component factory for the legoESM driver.

Resolves configured atmosphere, ocean, land, ice, and coupler
components into concrete model instances.  The driver delegates all
component construction here, so that ``ModelDriver`` never hardcodes
a particular dynamical core or surface component class.

Design
------
* The atmosphere factory uses the two-axis resolver already in
  ``atmosphere.dynamics`` (``resolve_solver_name`` / ``create_model``),
  but wraps it with driver-specific logic:
  - Computes physical diffusion coefficients from grid properties.
  - Validates that the grid type is compatible with the chosen solver.
  - Returns the model instance ready for time-stepping.
* ``create_ocean_component``, ``create_land_component``, and
  ``create_ice_component`` delegate to the canonical component
  constructors in ``legoesm.ocean``, ``legoesm.land``, and
  ``legoesm.ice``.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import jax.numpy as jnp

from legoesm.driver.config import ExperimentConfig, DycoreConfig

logger = logging.getLogger("legoesm.driver.factory")


# =========================================================================
# Supported-combination matrix
# =========================================================================

# (model_type, discretization, grid_type) -> flat solver name
#
# Only combinations listed here are supported through the driver.  Any
# other combination fails fast with a precise diagnostic message.

_DRIVER_SUPPORTED: dict[tuple[str, str, str], str] = {
    # --- Cubed-sphere C-D grid ---
    ("shallow_water", "centered",       "cubed_sphere"): "cdgrid_shallow_water",
    ("shallow_water", "cdgrid",         "cubed_sphere"): "cdgrid_shallow_water",
    ("hydrostatic",   "centered",       "cubed_sphere"): "cdgrid_primitive_equations",
    ("hydrostatic",   "cdgrid",         "cubed_sphere"): "cdgrid_primitive_equations",
    ("nonhydrostatic","centered",       "cubed_sphere"): "cdgrid_compressible_euler",
    ("nonhydrostatic","cdgrid",         "cubed_sphere"): "cdgrid_compressible_euler",

    # --- Cubed-sphere finite-volume (alias) ---
    ("shallow_water", "finite_volume",  "cubed_sphere"): "cdgrid_shallow_water",
    ("hydrostatic",   "finite_volume",  "cubed_sphere"): "cdgrid_primitive_equations",
    ("nonhydrostatic","finite_volume",  "cubed_sphere"): "cdgrid_compressible_euler",

    # --- Spectral (Gaussian grid) ---
    ("shallow_water", "spectral",       "gaussian"):     "spectral_shallow_water",
    ("hydrostatic",   "spectral",       "gaussian"):     "spectral_primitive_equations",
    ("nonhydrostatic","spectral",       "gaussian"):     "spectral_compressible_euler",

    # --- Lat-lon C-grid ---
    ("shallow_water", "finite_volume",  "latlon"):       "latlon_cgrid_shallow_water",
    ("hydrostatic",   "finite_volume",  "latlon"):       "latlon_cgrid_primitive_equations",
    ("shallow_water", "centered",       "latlon"):       "latlon_cgrid_shallow_water",
    ("hydrostatic",   "centered",       "latlon"):       "latlon_cgrid_primitive_equations",
    ("shallow_water", "latlon_cgrid",   "latlon"):       "latlon_cgrid_shallow_water",
    ("hydrostatic",   "latlon_cgrid",   "latlon"):       "latlon_cgrid_primitive_equations",

    # --- MPAS / SCVT Voronoi mesh + TRiSK discretization
    # (Ringler 2010, Thuburn 2009).  Canonical grid_type = "mpas"
    # (matching the ocean side); legacy aliases voronoi /
    # icosahedral / mpas_voronoi are normalised at the config
    # boundary via driver.config.normalize_grid_type. ---
    ("hydrostatic",   "mpas",           "mpas"):        "mpas_primitive_equations",
    ("nonhydrostatic","mpas",           "mpas"):        "mpas_compressible_euler",

    # --- FV3 six-face duo cube (certified fv_dynamics JAX lane) ---
    # ONE solver serves both arms: the certified lane is a single program
    # with a static ``hydrostatic`` switch.  Dry dynamics, fp64; the ONLY
    # physics is the certified Held-Suarez step (hydrostatic arm,
    # ``held_suarez_forcing``) — the branch below refuses everything
    # else loudly.
    ("hydrostatic",   "fv3_duo",        "cubed_sphere"): "fv3_duo_primitive_equations",
    ("nonhydrostatic","fv3_duo",        "cubed_sphere"): "fv3_duo_primitive_equations",

    # --- Doubly-periodic plane (CRM rollout, PR2c) ---
    # Plane only supports the non-hydrostatic compressible Euler dycore.
    # All other (model_type, plane) combinations fall through to
    # ``_fail_unsupported`` so users see a clear error pointing at the
    # PR2c roadmap rather than a quiet construction crash.
    ("nonhydrostatic","plane",          "plane"):        "plane_compressible_euler",

    # --- SFNO data-driven ---
    # SFNO is a spherical-harmonic operator: it reads ``grid.n_sh`` and
    # runs SH synthesis/analysis, so it can only construct on the Gaussian
    # spectral grid.  (These rows previously advertised ``cubed_sphere``,
    # which has no SH transform — the build crashed with AttributeError.)
    ("shallow_water", "sfno",           "gaussian"):     "sfno_shallow_water",
    ("hydrostatic",   "sfno",           "gaussian"):     "sfno_primitive_equations",

    # --- U-Cast data-driven (convolutional U-Net emulator) ---
    # The U-Net itself is grid-agnostic, but the PE bridge packs/unpacks
    # the spectral primitive-equation state (see ucast_pe.py docstring:
    # ``grid : GaussianGrid``), so it is Gaussian-only too.
    ("hydrostatic",   "u_cast",         "gaussian"):     "ucast_primitive_equations",
}


def supported_matrix() -> list[dict[str, str]]:
    """Return the driver-supported (model_type, discretization, grid) matrix."""
    return [
        {"model_type": k[0], "discretization": k[1], "grid_type": k[2],
         "solver": v}
        for k, v in sorted(_DRIVER_SUPPORTED.items())
    ]


# =========================================================================
# Diffusion coefficient helpers
# =========================================================================

class DiffusionCoeffs(NamedTuple):
    """Physical diffusion coefficients computed from grid properties."""
    A_h: float         # Laplacian viscosity [m^2/s]
    hyperdiff: float   # Biharmonic hyperdiffusion [m^4/s]
    div_damp: float    # Divergence damping [m^2/s]
    # Horizontal THERMAL diffusivity [m^2/s].  Historically locked to A_h;
    # k_h_scale=None keeps that (byte-identical).  A separate scale decouples
    # the momentum-viscosity circulation lever (a_h_scale) from the thermal
    # smoothing that stabilizes vertical computational modes (the cldG abs-145
    # tropical sawtooth died with BOTH scaled 0.25).
    K_h_A: float = None
    # Divergence-SELECTIVE biharmonic damping [m^4/s] (CAM-FV ldiv4).  Only
    # the MPAS hydrostatic lane consumes it; 0.0 = off, byte-identical.
    div_damp4: float = 0.0


def _grid_min_dx(grid) -> float:
    """Minimum grid spacing [m] used to scale diffusion coefficients.

    MPAS/Voronoi reports the cell-to-cell edge distance directly; lat-lon's
    smallest cell is the pole cell (``min(dx)/2``).  Falls back to 1e5 m for
    grids exposing neither attribute.
    """
    if hasattr(grid, 'dcEdge'):
        # Voronoi/MPAS: dcEdge is the cell-to-cell distance along each edge
        return float(jnp.min(jnp.asarray(grid.dcEdge)))
    if hasattr(grid, 'dx'):
        # Lat-lon: min(dx)/2 is the POLE-cell spacing.  This bounds the scalar
        # Laplacian A_h to be CFL-stable at the pole (A_h = 0.05*dx_pole^2/dt);
        # an equatorial-dx A_h is ~3000x larger and makes the EXPLICIT diffusion
        # operator CFL-UNSTABLE at the pole (the polar filter truncates wave
        # modes, not the diffusion stencil) -> immediate blow-up.  NOTE: this
        # pole-bounded scalar A_h is too weak to damp midlatitude grid-scale
        # noise at fine (<=2deg) resolution; a latitude-dependent viscosity
        # (A_h ~ local dx^2) or biharmonic hyperdiffusion in the latlon_cgrid
        # dycore is needed for stable 2deg global runs (TODO).
        return float(jnp.min(jnp.asarray(grid.dx))) / 2.0
    return 1e5


def _grid_cell_area(grid, dx_min: float) -> float:
    """Smallest cell area [m^2], the ``L^2`` of CAM-FV's divergence damping.

    MPAS/Voronoi carries ``areaCell`` directly; every other grid falls back
    to ``dx_min**2``, which is the same quantity for a quasi-uniform mesh.
    """
    if hasattr(grid, 'areaCell'):
        return float(jnp.min(jnp.asarray(grid.areaCell)))
    return float(dx_min) ** 2


def compute_diffusion(grid, dc: DycoreConfig) -> DiffusionCoeffs:
    """Compute physical diffusion coefficients from grid and config.

    Parameters
    ----------
    grid : Grid object
        Must expose a ``dx`` attribute (minimum grid spacing in metres).
    dc : DycoreConfig
        Dycore configuration with ``dt``, ``hyperdiff_scale``,
        ``div_damp_scale``.

    Returns
    -------
    DiffusionCoeffs
    """
    dx_min = _grid_min_dx(grid)
    DT = dc.dt

    # Laplacian viscosity (2nd-order, scale-NON-selective).  The earlier
    # default 0.05*dx^2/DT was ~5-10x too strong: it damped a 3000 km
    # baroclinic eddy in ~5 h (faster than its ~1-2 day growth), crushing the
    # midlatitude eddy-driven jets and producing spurious equatorial
    # super-rotation (dry Held-Suarez jet 2.9 vs spectral 37 m/s).  Reduced
    # ~16x to a level whose eddy-scale (~4000 km) damping time is ~days while
    # still damping 2*dx grid noise in a few hours; the scale-selective
    # 4th-order hyperdiff below is the primary grid-noise control.  a_h_scale
    # exposes it for tuning (0 = rely on hyperdiff alone).
    A_h = dc.a_h_scale * 3.0e-3 * dx_min ** 2 / DT

    # Biharmonic: e-folding time for grid-scale noise
    tau_efold = 24.0 * 3600.0
    hyperdiff = dc.hyperdiff_scale * dx_min ** 4 / tau_efold

    # Divergence damping: damps external gravity wave mode
    c_grav = 300.0
    div_damp = dc.div_damp_scale * c_grav * dx_min / (2.0 * 3.14159)

    _khs = getattr(dc, "k_h_scale", None)
    K_h_A = A_h if _khs is None else _khs * 3.0e-3 * dx_min ** 2 / DT

    # Divergence-selective biharmonic damping, scaled from CAM-FV's ``ldiv4``
    # (cd_core.F90:620-684): CAM sets tau4 = 0.01/dt and multiplies by the
    # SQUARE of the cell area, so nu_div4 = 0.01 * L^4 / dt with L^2 the cell
    # area.  The Earth-radius factors are consistent (CAM's angular divergence
    # supplies one ae, its angular Laplacian two, cdtau4 the fourth), verified
    # in review -- no metric factor is missing.
    #
    # CAM-INSPIRED, NOT IDENTICAL, and the difference is not a detail:
    #  * CAM applies this once per ACOUSTIC substep, we apply it once per
    #    dycore step, so matching the nondimensional 0.01 matches the per-
    #    APPLICATION damping fraction, never the damping per simulated second.
    #  * CAM's coefficient is per-cell and sits INSIDE the final gradient;
    #    ours is one global scalar outside it, built from the SMALLEST cell.
    #    On the res6 mesh areaCell spans a factor 1.60, so the coarsest cells
    #    receive 1/1.60^2 = 39% of the nominal rate.  Quasi-uniform meshes
    #    only; a variable-resolution mesh needs the per-cell form.
    #  * The discrete eigenvalues and the SSP-RK54 stability function differ
    #    from CAM's, so the realised per-step damping differs too (measured
    #    21.8% of the strongest scalar-Laplacian mode, against CAM's nominal
    #    1% per application before its own integrator).
    # The scale is therefore a CALIBRATION KNOB whose 1.0 means "CAM's own
    # nondimensional rate", not "CAM's behaviour".  res6 at dt=112.5 s gives
    # 7.235e15 m^4/s at scale 1.0.
    nu_div4_cam = 0.01 * _grid_cell_area(grid, dx_min) ** 2 / DT
    div_damp4 = getattr(dc, "mpas_div_damp4_scale", 0.0) * nu_div4_cam
    return DiffusionCoeffs(A_h=A_h, hyperdiff=hyperdiff, div_damp=div_damp,
                           K_h_A=K_h_A, div_damp4=div_damp4)


# Solvers that apply the EXPLICIT biharmonic hyperdiff / divergence damping
# (forward the raw ``diff.hyperdiff`` / ``diff.div_damp`` to a finite-volume
# operator with no implicit/spectral inversion).  Spectral dycores recompute
# their own implicit-stable hyperdiff (and zero the FV one); the plane dycore
# zeroes hyperdiff; lat-lon clamps A_h itself — none belong here, so the guard
# never cries wolf about a coefficient the selected solver discards.
# NB: cube SW (``cdgrid_shallow_water``) is intentionally ABSENT — it now
# builds FV3EdgeShallowWaterModel with a validated FIXED preset
# (williamson_cli_calibration) that owns its own hyperdiff/div_damp, so the
# generic ``diff.hyperdiff`` guard would cry wolf about a coefficient the
# model discards (codex M2 review).  The cube-SW factory branch rejects
# non-default diffusion scales directly instead.
_EXPLICIT_HYPERDIFF_SOLVERS = frozenset({
    "cdgrid_primitive_equations",
    "cdgrid_compressible_euler",
    "mpas_primitive_equations",
})
_EXPLICIT_DIVDAMP_SOLVERS = frozenset({"cdgrid_primitive_equations"})


def warn_if_diffusion_unstable(solver_name: str, diff: DiffusionCoeffs,
                               grid, dt: float) -> None:
    """Warn (no clamp) if an explicit diffusion coeff exceeds the stable max.

    Only the lat-lon C-grid clamps its Laplacian A_h; the biharmonic hyperdiff
    and divergence damping reach the cubed-sphere / MPAS finite-volume dycores
    with no runtime guard, so an over-large ``hyperdiff_scale`` / ``div_damp_scale``
    (or ``dt``) blows up downstream as a silent NaN.  This converts that into a
    clear init-time diagnostic.

    The threshold is the model's OWN ``adaptive_hyperdiff_coeff(..., safety=1.0)``
    — the largest coefficient legoESM considers stable for the grid+dt.  Using
    that (rather than an idealized forward-Euler Cartesian bound) keeps the guard
    consistent with the validated coefficient picker and the multi-stage SSP-RK
    integrators these dycores use, so it fires only on gross over-specification,
    not on marginal calibration (which depends on the integrator stability region
    and cube metric and is validated empirically, not knowable at config time).
    Coefficients are NOT clamped — that would silently alter the user's tuned
    scale-selective damping.
    """
    from legoesm.core.cfl import adaptive_hyperdiff_coeff
    dx_min = _grid_min_dx(grid)
    checks = []
    if solver_name in _EXPLICIT_HYPERDIFF_SOLVERS:
        checks.append(("hyperdiff (biharmonic)", diff.hyperdiff, 4))
    if solver_name in _EXPLICIT_DIVDAMP_SOLVERS:
        checks.append(("div_damp", diff.div_damp, 2))
    for name, coeff, order in checks:
        coeff_max = adaptive_hyperdiff_coeff(dx_min, dt, order=order, safety=1.0)
        if coeff > coeff_max:
            logger.warning(
                "Diffusion %s=%.3e for solver %s exceeds the max stable "
                "coefficient %.3e (grid dx_min=%.1f m, dt=%.1f s) by %.1fx; "
                "the run may go unstable. Reduce hyperdiff_scale/div_damp_scale "
                "or dt.",
                name, coeff, solver_name, coeff_max, dx_min, dt,
                coeff / coeff_max,
            )


# =========================================================================
# Atmosphere factory
# =========================================================================

# =========================================================================
# fv3_duo slice-1 DEFAULT-DENY wall
# =========================================================================
#
# The certified duo lane consumes ONLY the fields allow-listed below (read
# from ``_run_fv3_duo`` + this factory branch); every OTHER ExperimentConfig
# field is inert on this lane, so a non-default value is a request the run
# would silently ignore -- the "successful wrong experiment" failure mode
# (codex 2026-08-18: moisture flags, land/surface schemes, forcing decks and
# IC selectors all sailed past the enumerated deny-list).  The specific
# contract-citing refusals in the branch below stay as the fast path; this
# wall is the backstop that makes the deny-list exhaustive by construction.

_FV3_DUO_ALLOWED_NONDEFAULT: frozenset[str] = frozenset({
    # Grid selection (nlev is additionally pinned to {5, 10} in-branch).
    "grid.grid_type", "grid.resolution", "grid.nlev",
    # Lane selection + timestep (dt is the only dynamic deck quantity).
    "dycore.model_type", "dycore.discretization", "dycore.dt",
    # Integration span + reproducibility bookkeeping (manifest-recorded).
    "days", "start_day", "seed",
    # Pinned to 'fp64' by the specific guard (the ExperimentConfig default
    # is 'fp32', so every valid duo config differs here).
    "precision",
    # The five scheme selectors are pinned to 'none' by the specific guard.
    "radiation", "convection", "microphysics", "turbulence",
    "gravity_wave_drag",
    # The ONLY physics this lane runs: the certified 3-pass Held-Suarez
    # step (apply_held_suarez_step, gated at 1.7645e-8 vs the Fortran
    # oracle), applied by _run_fv3_duo after each dynamics step.  Every
    # other physics/forcing selector stays refused.
    "held_suarez_forcing",
    # SPMD: the duo lane accepts --distributed --distributed-mode spmd
    # (single-process, face axis over the local devices; the specific
    # guard above refuses every other mode). The three engineering knobs
    # it selects are dual-reviewed and parity-gated (PR #1656).
    "distributed", "distributed_mode",
    "dycore.fv3_duo_windows", "dycore.fv3_duo_window_pad",
    # Output cadence + destination -- the OutputConfig fields the lane's
    # snapshot + checkpoint writers read (checkpoint_days: slice-2
    # restart, the shared cube/MPAS cadence field -> fv3duo_ckpt_v1).
    "output.output_dir", "output.diag_days", "output.checkpoint_days",
    # CLI-default drift that CANNOT affect the duo dynamics (measured on a
    # stock ``run_amip --discretization fv3_duo`` config, job 9433540):
    # ``--clouds`` defaults to 'xu_randall' at the argparse layer. The duo
    # EXECUTION LOOP never evaluates physics or cloud diagnostics (the one
    # other consumer, _create_diagnostics' clt at model_driver.py:3146, is
    # not collected by this lane — codex 2026-08-18 corrected the earlier
    # "only radiation optics" claim); ``--use-polar-filter`` (BooleanOptionalAction, default None =
    # "no choice") gates a lat-lon-C-grid-only Fourier filter this
    # cubed-sphere lane never builds.  Refusing either would refuse every
    # stock CLI launch.  Still inert with held_suarez_forcing on: the HS
    # step (apply_held_suarez_step) consumes pt/ua/va/delp/peln/pkz/pe/lat
    # only — no cloud field, no filter (codex 2026-08-24 re-raised; the
    # measured-inert justification survives the HS wiring by read).
    "cloud_scheme", "dycore.use_polar_filter",
})


def _flatten_config_fields(cfg, prefix: str = ""):
    """Yield ``("a.b.c", value)`` fields of a nested NamedTuple config.

    Recurses ONLY into NamedTuples; sequences/mappings are ATOMIC values
    compared whole (a changed tuple is still refused as one field). A
    frozen-surface test pins both this shape and the defaults it is
    diffed against.
    """
    for name in cfg._fields:
        val = getattr(cfg, name)
        if hasattr(val, "_fields"):  # nested NamedTuple sub-config
            yield from _flatten_config_fields(val, f"{prefix}{name}.")
        else:
            yield f"{prefix}{name}", val


def resolve_fv3_duo_layout(*, world: int, n_local: int, n_global: int,
                           distributed: bool) -> str:
    """Pure fv3_duo layout policy -- returns 'shard' or 'single', or
    raises ValueError.  Split out of ``create_atmosphere_dycore`` so the
    auto-adapt + refusal rules are unit-testable with plain ints (no jax
    devices, no model construction).

    * *world*      = max(jax.process_count(), launcher_world_size()).
    * *n_local*    = this process's local device count.
    * *n_global*   = the global device count (== n_local single-process).
    * *distributed*= config.distributed (the explicit --distributed flag).

    Rules:
    - A multi-process launch (world>1) WITHOUT --distributed is refused
      (it would run the single-process I/O path across ranks).
    - The device set is the GLOBAL set under --distributed, else this
      process's LOCAL set.
    - 2/3/6 devices -> 'shard' (face-shard + face-batch).
    - Otherwise (1, or 4/5/7...): 'single'.  Under an EXPLICIT
      --distributed that is a hard error (the user asked to distribute an
      unshardable count); on the auto path it is a legitimate fall-back
      the caller logs loudly.
    """
    if world > 1 and not distributed:
        raise ValueError(
            f"fv3_duo: this looks like a MULTI-PROCESS launch (world "
            f"size {world}) but --distributed was not set. Multi-process "
            f"SPMD is explicit (it changes the checkpoint/snapshot I/O "
            f"path); pass --distributed --distributed-mode spmd. A "
            f"genuinely single-process run inside a multi-task allocation: "
            f"launch it with srun --ntasks=1. Single-process runs auto-"
            f"adapt to the local device count with no flag.")
    n = n_global if distributed else n_local
    if n >= 2 and 6 % n == 0:
        return "shard"
    if distributed:
        raise ValueError(
            f"fv3_duo spmd: {n} device(s) cannot face-shard (need 2, 3 "
            f"or 6 to divide the 6 cube faces). Provide 2/3/6 devices "
            f"(GPUs, or xla_force_host_platform_device_count), or drop "
            f"--distributed to run single-device.")
    return "single"


def _ledger_level_weight(config, sigma):
    """(nlev,) band weight for the budget ledger, or None for the full column.

    Shared by the physics factory and the dycore so the two halves of the
    ledger table cannot be banded differently.
    """
    band = getattr(getattr(config, "output", None), "budget_ledger_sigma_band", None)
    if band is None:
        return None
    if not hasattr(sigma, "sigma_half"):
        raise ValueError(
            "budget_ledger_sigma_band needs a sigma vertical coordinate; this "
            "run's coordinate has no sigma_half, so a single band does not "
            "select one pressure range.")
    from legoesm.diagnostics.process_ledger import sigma_band_weight
    return sigma_band_weight(sigma.sigma_half, float(band[0]), float(band[1]))


def _refuse_fv3_duo_non_default(config: ExperimentConfig) -> None:
    """Refuse EVERY non-default, non-allow-listed field, all at once.

    Diffs the incoming config field-by-field against a freshly
    constructed default instance and raises ONE error naming every
    offending path and value, so a mis-built launch script is fixed in
    one round-trip instead of field-by-field.
    """
    defaults = dict(_flatten_config_fields(type(config)()))
    offending = [
        (path, val)
        for path, val in _flatten_config_fields(config)
        if path not in _FV3_DUO_ALLOWED_NONDEFAULT
        and not (val == defaults[path])
    ]
    if offending:
        listing = ", ".join(f"{p}={v!r}" for p, v in offending)
        raise ValueError(
            f"fv3_duo (slice 1) runs ONLY the certified dry-dynamics deck; "
            f"the driver lane consumes no other configuration, so each field "
            f"below would be SILENTLY inert -- a successful wrong experiment. "
            f"Non-default unsupported fields ({len(offending)}): {listing}. "
            f"Allowed non-default fields: "
            f"{sorted(_FV3_DUO_ALLOWED_NONDEFAULT)}. Reset the offenders or "
            f"choose a lane that supports them.")


def _create_fv3_duo_column_model(config: ExperimentConfig, gc, model_type,
                                 *, bundle=None):
    """The FV3 duo as a COLUMN model for the MPAS lane (route A).

    The MPAS lane consumes the whole ExperimentConfig surface, so the
    closed lane's default-deny wall does not apply; what the column lane
    cannot honour yet is refused HERE, by name, so nothing is silently
    inert (the wall's failure mode):
      * the MPAS-lane numerics knobs that edit the state after the step
        (top sponge, q_v del2/del4 smoothing): the columns are a VIEW of
        the duo bundle and the model refuses wind edits; q_v smoothing
        is OFF on the duo by decision (user 2026-09-26);
      * held_suarez_forcing: the closed lane's FV3 hswf is the certified
        HS on this dycore; the MPAS lane's HS drops the meridional drag;
      * distributed / windows / NH / km outside {5, 10} (rung 7 / M4).
    """
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel,
    )
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig,
        FV3DuoDynamicsModel,
    )

    refused = []
    if getattr(config, "sponge_enabled", False):
        refused.append("sponge_enabled=True (post-step wind edit)")
    for k in ("mpas_qv_smooth_del2_m2s", "mpas_qv_smooth_del4_m4s"):
        if float(getattr(config, k, 0.0) or 0.0) != 0.0:
            refused.append(f"{k}={getattr(config, k)!r} (q_v smoothing is "
                           "OFF on the duo, decision 2026-09-26)")
    if config.held_suarez_forcing:
        refused.append("held_suarez_forcing=True (use the closed duo lane's "
                       "certified FV3 hswf)")
    if config.distributed:
        refused.append("distributed=True (single-process; windows are rung 7)")
    if model_type != "hydrostatic":
        refused.append(f"model_type={model_type!r} (hydrostatic only: the "
                       "column increments rebuild the hydrostatic pressures)")
    if config.precision not in ("fp64", "float64", "mixed_fp64_storage"):
        refused.append(f"precision={config.precision!r} (fp64 only)")
    # the vertical table follows nlev: the certified analytic branch at
    # 5/10, CAM6's L32 table at 32 (user decision 2026-09-26: CAM L32)
    eta = {5: "analytic", 10: "analytic", 32: "cam6_l32"}.get(gc.nlev)
    if eta is None:
        refused.append(f"grid.nlev={gc.nlev} (analytic set_eta km in {{5, 10}} "
                       "or the CAM6 L32 table at 32)")
    # Setup-time regrids (topography, SST/SIC, land, ozone) land on
    # ``self.grid``, which the driver sets to the duo's own column mesh
    # at grid-creation time (M6, ``ModelDriver._create_grid``) -- the
    # standard cubed-sphere centres are NOT the duo's A-grid (MEASURED
    # 2026-09-26, C12: 1.6 deg offsets).  The factory is handed that same
    # grid bundle so the model's mesh is the driver's mesh.
    if bundle is None:
        refused.append("the driver did not build the duo grid at grid "
                       "creation (ModelDriver._create_grid, M6); setup-time "
                       "regrids would land on the standard cubed sphere")
    # The column lane's DYNAMICS terrain is the ERA5 IC's own phis (the
    # grid is rebuilt with it, del-2 filtered -- the MPAS lane's choice);
    # a topography file / analytic mountain feeds land fraction and CMOR
    # orog only.  Under the analytic baroclinic-wave IC (flat-terrain
    # balanced) a mountain would be placed in the diagnostics and never
    # reach dynamics -- refused rather than silently flat (codex M6 r1).
    if config.topography != "flat" and config.ic != "era5":
        refused.append(f"topography={config.topography!r} with ic="
                       f"{config.ic!r} (only ic='era5' carries a dynamics "
                       "terrain on this lane; the file/mountain would be "
                       "diagnostic-only while dynamics stayed flat)")
    # The column model's mass block is FV3's nwat block on the slot list
    # the driver's tracer registry names (3 warm-rain species, or the six
    # water species + numbers for an ice scheme): every registered tracer
    # takes its tendency and moves the layer mass (fv_update_phys
    # :324/:335/:352, moist_cp case(6)).  A scheme whose tendencies
    # target a tracer the registry does not carry is refused by the
    # registry's own slot validation below.
    # Convection schemes that READ A GRID OPERATOR are refused by trait:
    # the column mesh carries no edge topology, so the moisture-
    # convergence operator and the resolved-w diagnostic return None,
    # which Kuo turns into a zero source (an inert scheme that runs,
    # codex 2026-09-30) and Kain-Fritsch needs concretely.  Tiedtke /
    # Bechtold engage their internal saturation-deficit proxy on None
    # (the unified pipeline's own degrade path) and are admitted.
    if config.convection != "none":
        from legoesm.atmosphere.physics.convection.integration import (
            convection_scheme_traits)
        _tr = convection_scheme_traits(config.convection)
        if _tr.is_simple_mc_consumer or _tr.is_w_grid_consumer:
            refused.append(
                f"convection={config.convection!r} (reads a grid operator "
                f"-- moisture convergence / resolved w -- the column mesh "
                f"has no edge topology for; it would run inert)")
    # MPAS-dycore numerics knobs (dycore.mpas_*) are the MPAS model's; the
    # duo reads none of them except the grid-general positivity knob, so a
    # non-default value would be inert
    d_def = type(config.dycore)()
    inert = [f for f in config.dycore._fields
             if f.startswith("mpas_")
             and f != "mpas_conservative_tracer_clamp"
             and getattr(config.dycore, f) != getattr(d_def, f)]
    if inert:
        refused.append("dycore." + ", dycore.".join(inert)
                       + " (MPAS-dycore knobs the duo does not read)")
    if getattr(config.output, "budget_ledger", False):
        refused.append("output.budget_ledger=True (the column model exports "
                       "no per-step ledger)")
    if refused:
        raise ValueError(
            "fv3_duo column lane cannot honour: " + "; ".join(refused))
    moist = any(getattr(config, k) != "none"
                for k in ("microphysics", "convection", "turbulence"))
    from legoesm.driver.physics_pipeline import (
        moisture_registry_for, validate_microphysics_tracer_slots)
    registry = moisture_registry_for(config.microphysics)
    validate_microphysics_tracer_slots(
        config.microphysics, registry.n_tracers,
        context="fv3_duo column lane tracer slots")
    if bundle.ctx_np.get("ectx") is None:
        raise ValueError("fv3_duo column lane needs the duo ext bundle "
                         "(ctx['ectx'] with amat6) for the c2l column winds")
    dyn = FV3DuoDynamicsModel(
        bundle, FV3DuoConfig(km=gc.nlev, hydrostatic=True,
                             storage_dtype="float64", moist=moist, eta=eta,
                             fill=config.dycore.fv3_duo_fill))
    return FV3DuoColumnModel(
        dyn, tracer_names=registry.names,
        conservative_tracer_clamp=config.dycore.mpas_conservative_tracer_clamp,
        energy_consistent_moisture_clip=config.energy_consistent_moisture_clip)


def create_atmosphere_dycore(
    config: ExperimentConfig,
    grid,
    sigma,
    coeff_grid=None,
    fv3_duo_bundle=None,
) -> Any:
    """Resolve and instantiate the configured atmosphere dynamical core.

    ``fv3_duo_bundle``: the duo grid the driver built at grid creation
    (column lane only); the column model is built ON it so its mesh is
    the grid every forcing was regridded onto.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    grid
        Horizontal grid (cubed-sphere, Gaussian, lat-lon, etc.).
    sigma
        Vertical coordinate.
    coeff_grid
        Grid the diffusion coefficients are derived from.  Under a cell
        partition ``grid`` is the rank-local mesh; its min(dcEdge) /
        min(areaCell) are rank-dependent, so the GLOBAL mesh must be passed
        here or every rank integrates a different viscosity.  ``None`` means
        ``grid``.

    Returns
    -------
    model
        The instantiated dynamical core, ready for ``step_with_physics``.

    Raises
    ------
    ValueError
        If the (model_type, discretization, grid_type) combination is
        not supported.
    """
    dc = config.dycore
    gc = config.grid

    model_type = dc.model_type
    discretization = dc.discretization
    # Defensive canonical-name normalization at the factory entry: callers
    # that bypass run_amip's argparse postprocessor (direct test fixtures,
    # ad-hoc scripts, older YAML loaders) might still pass ``voronoi`` /
    # ``icosahedral`` / ``mpas_voronoi`` for the SCVT mesh.  The dispatch
    # table below speaks only the canonical ``mpas`` so we normalise
    # here too — the cost is one dict lookup.
    from legoesm.driver.config import normalize_grid_type
    grid_type = normalize_grid_type(gc.grid_type)

    key = (model_type, discretization, grid_type)

    if key not in _DRIVER_SUPPORTED:
        _fail_unsupported(model_type, discretization, grid_type)

    solver_name = _DRIVER_SUPPORTED[key]
    if coeff_grid is None:
        coeff_grid = grid
    diff = compute_diffusion(coeff_grid, dc)
    warn_if_diffusion_unstable(solver_name, diff, coeff_grid, dc.dt)

    logger.info(
        "Atmosphere: model_type=%s, discretization=%s, grid=%s -> %s",
        model_type, discretization, grid_type, solver_name,
    )

    def _reject_unselectable_time_integrator(path: str) -> None:
        """Raise unless ``dc.time_integrator`` is left at the grid-aware default.

        These solver branches build their config with the scheme's OWN outer
        integrator and never read ``dc.time_integrator``, so an explicit choice
        was SILENTLY IGNORED and the run used a different integrator than the
        user asked for.  That is reachable from ``run_amip --time-integrator``
        and (since the nested-YAML boundary began mapping the key) from
        ``atmosphere.time_integrator`` too.

        Mirrors the spectral-hydrostatic guard below: reject an unsupported
        EXPLICIT selection with a clear message rather than overriding it in
        silence.  ``auto`` and the DycoreConfig default mean "no deliberate
        choice" and are tolerated.
        """
        _default = type(dc)().time_integrator
        if dc.time_integrator not in ("auto", _default):
            raise ValueError(
                f"{path} does not implement a selectable time_integrator (its "
                f"outer integrator is fixed by the scheme); got "
                f"dycore.time_integrator={dc.time_integrator!r}. Use 'auto' "
                f"(the default), or select a solver that supports it."
            )

    # ----- Cubed-sphere C-D grid solvers -----
    if solver_name == "cdgrid_shallow_water":
        # FV3 single-implementation program M2 (2026-07-13, codex-reviewed):
        # the FV3-faithful cube SW core is `FV3EdgeShallowWaterModel`
        # (edge-midpoint split-D winds + `fv3_sw_tendencies`) — the model the
        # Williamson matrix validates.  The legacy corner-corner
        # `CDGridShallowWaterModel` is a DIFFERENT discretization that cannot
        # stabilize the cube W5/W6 wave class (it lacks the `damp_v`
        # del6_vt_flux vorticity sink, which needs edge-midpoint winds; W5
        # peaks 550+ m/s at every supported tuning while Edge stays ~45 m/s).
        # So the driver now advertises the validated core + its
        # resolution-robust preset (`williamson_cli_calibration`: matched
        # hyperdiff + div_damp + damp_v, stable C24-C48).  See
        # docs/architecture/fv3_single_implementation_program.md (Phase-1 M2).
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel, CDGridShallowWaterConfig,
            williamson_cli_calibration,
        )
        # The cube SW core ships a VALIDATED fixed damping preset
        # (williamson_cli_calibration).  The generic driver diffusion knobs
        # do NOT apply here: FV3EdgeShallowWaterModel has no A_h consumer, and
        # hyperdiff/div_damp are owned by the preset.  Reject non-default
        # scales LOUDLY rather than silently ignoring them (codex M2 review).
        for _knob, _val in (("hyperdiff_scale", dc.hyperdiff_scale),
                            ("a_h_scale", dc.a_h_scale),
                            ("div_damp_scale", dc.div_damp_scale)):
            if _val != 1.0:
                raise ValueError(
                    f"cube shallow-water uses the validated fixed preset "
                    f"`williamson_cli_calibration`; dycore.{_knob}={_val!r} "
                    f"(non-default) would be silently ignored.  Remove it, or "
                    f"tune the preset coefficients directly.")
        # Calibrate on the ACTUAL grid resolution (grid.n), not the config
        # field, and refuse a grid/config mismatch (codex M2 review): a direct
        # caller passing a C48 grid with gc.resolution=24 would otherwise get
        # C24 damping on a C48 grid.
        n = int(grid.n)
        if n != gc.resolution:
            raise ValueError(
                f"cube SW grid resolution (grid.n={n}) does not match config "
                f"resolution (gc.resolution={gc.resolution}); refusing to "
                f"build a model with mismatched damping calibration.")
        cfg = williamson_cli_calibration(n)._replace(
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
            # "auto" -> this dycore's own default; explicit names verbatim.
            time_integrator=(CDGridShallowWaterConfig().time_integrator
                             if dc.time_integrator == "auto"
                             else dc.time_integrator),
        )
        return FV3EdgeShallowWaterModel(grid, cfg)

    if solver_name == "cdgrid_primitive_equations":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        )
        cfg = CDGridPrimitiveEquationConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff, hyperdiff_ps_coeff=diff.hyperdiff,
            div_damp_coeff=diff.div_damp,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
            conservative_tracer_clamp=dc.mpas_conservative_tracer_clamp,  # #1354/#1515 borrow (grid-general knob)
            energy_consistent_moisture_clip=config.energy_consistent_moisture_clip,  # #1354/#1515 (hard-floor path only)
            # Issue #273 Phase 3: forward the implicit gravity-wave
            # damping switches from the canonical driver config.
            # Default off (both 0/False) keeps the explicit path
            # bit-exact for existing call sites.
            implicit_grav_wave_use_pcg=dc.implicit_grav_wave_use_pcg,
            implicit_grav_wave_damping=dc.implicit_grav_wave_damping,
            # "auto" -> this dycore's own default; explicit names verbatim.
            time_integrator=(CDGridPrimitiveEquationConfig().time_integrator
                             if dc.time_integrator == "auto"
                             else dc.time_integrator),
            # #771: flux-form moisture transport (needs moisture attached via
            # ExperimentConfig.moisture_advection; default off = advective).
            moisture_flux_form=getattr(dc, "moisture_flux_form", False),
        )
        return CDGridPrimitiveEquationModel(grid, sigma, cfg)

    if solver_name == "cdgrid_compressible_euler":
        _reject_unselectable_time_integrator("cubed-sphere non-hydrostatic (compressible Euler)")
        from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
        )
        # Forward the driver-level mass fixer (mirrors the plane / NH
        # branches).  Previously dropped: DycoreConfig.fix_mass=True was
        # silently ignored on the cubed-sphere NH path.
        # conservation_fixer=False overrides fix_mass=True (same contract
        # as the lat-lon branch below).
        _nh_fix_mass = dc.fix_mass and dc.conservation_fixer
        cfg = CDGridCompressibleEulerConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff,
            fix_mass=_nh_fix_mass,
            anchor_mass_to_initial=_nh_fix_mass,
        )
        # Non-hydrostatic requires height coordinate and terrain metric.
        # Expect grid to provide these or construct defaults.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                # Build flat (zero topography) height coordinate.
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                H_top = 30_000.0  # metres
                height_coord = create_height_coordinate(nlev, H_top)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.lat)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        return CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg)

    # ----- Spectral solvers (Gaussian grid) -----
    if solver_name == "spectral_shallow_water":
        _reject_unselectable_time_integrator("spectral shallow water")
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralShallowWaterModel
        return SpectralShallowWaterModel(grid=grid)

    if solver_name == "spectral_primitive_equations":
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
        )
        # Compute hyperdiffusion from truncation: 0.5-hour e-folding at max wavenumber
        n_max = grid.n_max
        a = grid.radius
        eig_max = n_max * (n_max + 1) / (a * a)
        hyperdiff = 1.0 / (0.5 * 3600.0 * eig_max ** 2)
        # Spectral hydrostatic PE runs the 5-stage SSP-RK54 by default (spectral
        # stability).  It ALSO supports the semi-implicit LEAPFROG path, which
        # evaluates physics ONCE per step and is the integrator #405 requires to
        # thread a prognostic PhysicsState carry (ssp_rk54 evaluates physics
        # per-RK-stage, where a single-step carry is ill-defined).  Any OTHER
        # explicit integrator is still rejected with a clear message.
        _spectral_leapfrog = str(dc.time_integrator).lower() in (
            "leapfrog", "leapfrog_si")
        if not _spectral_leapfrog and dc.time_integrator not in (
                "ssp_rk54", "auto", type(dc)().time_integrator):
            raise ValueError(
                f"spectral hydrostatic PE supports time_integrator='ssp_rk54' "
                f"(default) or 'leapfrog_si' (#405 prognostic-physics path); got "
                f"dycore.time_integrator={dc.time_integrator!r} — that integrator "
                f"is not implemented for the spectral path."
            )
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=hyperdiff,
            hyperdiff_order=2,
            time_integrator="leapfrog_si" if _spectral_leapfrog else "ssp_rk54",
            # The leapfrog path is semi-implicit (needs the SI matrices the
            # euler/leapfrog SI steps consume); ssp_rk54 stays explicit.
            semi_implicit=_spectral_leapfrog,
            p_floor=200.0,
            dealiasing_fraction=0.667,
            # Forward the driver-level mass fixer (mirrors the plane / NH
            # branches).  Previously dropped: DycoreConfig.fix_mass=True was
            # silently ignored on the spectral hydrostatic path.
            # conservation_fixer=False overrides fix_mass=True (lat-lon
            # branch contract).
            fix_mass=dc.fix_mass and dc.conservation_fixer,
            anchor_mass_to_initial=dc.fix_mass and dc.conservation_fixer,
            conservative_tracer_clamp=dc.mpas_conservative_tracer_clamp,  # #1354/#1515 borrow (grid-general knob)
        )
        return SpectralPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=pe_config,
        )

    if solver_name == "spectral_compressible_euler":
        _reject_unselectable_time_integrator("spectral non-hydrostatic (compressible Euler)")
        from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
            SpectralCompressibleEulerModel, SpectralNHConfig,
        )
        # Non-hydrostatic uses a height (z-star) coordinate, not sigma:
        # build the flat-topography defaults exactly like the CDGrid-NH
        # branch above.  z_s must be (n_lat, n_lon) on the Gaussian grid
        # (grid.lat is 1-D), hence zeros_like(lat2d).
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                height_coord = create_height_coordinate(nlev, 30_000.0)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.lat2d)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        # conservation_fixer=False overrides fix_mass=True (lat-lon contract).
        nh_cfg = SpectralNHConfig(
            fix_mass=dc.fix_mass and dc.conservation_fixer,
            anchor_mass_to_initial=dc.fix_mass and dc.conservation_fixer,
        )
        return SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, nh_cfg,
        )

    # ----- MPAS icosahedral -----
    if solver_name == "mpas_primitive_equations":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
        )
        # Integrator: ``"auto"`` (run_amip CLI default) resolves to the
        # MPAS dycore's own default (ssp_rk54_scan).  The MPAS
        # biharmonic hyperdiffusion eigenvalues at production time
        # steps fall OUTSIDE ssp_rk3's stability region
        # (primitive_eq_mpas.py stability notes: ssp_rk3 diverges
        # within ~3 steps at dt=600 with hyperdiff ON, ssp_rk54 stays
        # bounded) — silently forwarding the global "ssp_rk3" default
        # was the "MPAS hidden CFL constraint" that forced the AMIP
        # smoke test down to dt=60 (AMIP.md Known issues #2).  Any
        # explicit scheme name (including ssp_rk3, for deliberate
        # integrator-sensitivity runs) is forwarded verbatim.
        _ti = dc.time_integrator
        if _ti == "auto":
            _ti = MPASPrimitiveEquationConfig().time_integrator
        cfg = MPASPrimitiveEquationConfig(
            nu_del2=diff.A_h,
            nu_del4=diff.hyperdiff,
            nu_del4_ps=diff.hyperdiff,
            K_h=diff.K_h_A,
            # conservation_fixer=False overrides fix_mass=True (lat-lon
            # contract; codex 2026-07-12 round 2 — this branch predates
            # the audit but had the same gap).
            fix_mass=dc.fix_mass and dc.conservation_fixer,
            time_integrator=_ti,
            # #930 cure: vertical biharmonic damping of the 2Δσ T checkerboard.
            nu_vert4_T=dc.mpas_nu_vert4_T,
            # Mass-conserving tracer positivity clamp (see DycoreConfig).
            conservative_tracer_clamp=dc.mpas_conservative_tracer_clamp,
            energy_consistent_moisture_clip=config.energy_consistent_moisture_clip,  # #1354/#1515 (no-op under the borrow)
            # Sigma-lane vertical advection scheme (see DycoreConfig).
            vert_advection_scheme=dc.mpas_vert_advection_scheme,
            sponge_del2_top_layers=int(dc.mpas_sponge_del2_top_layers),
            sponge_del2_top_factor=float(dc.mpas_sponge_del2_top_factor),
            # Divergence-selective biharmonic damping (CAM-FV ldiv4).  The
            # vector del2/del4 above damp rotational and divergent modes
            # alike; this one is the piece CAM applies at every level and we
            # had computed (``div_damp``) but never handed to this core.
            nu_div4=diff.div_damp4,
            # Budget-ledger vertical band. The dycore owns the SNAPSHOT-derived
            # dynamics and clips rows, so it needs the SAME weight the physics
            # rows use; without it those two rows stay full-column while the
            # rest are banded and the table does not partition. Measured that
            # way once: dynamics read 1.1504 identically in a full-column, a
            # free-troposphere and a boundary-layer run.
            budget_ledger_level_weight=_ledger_level_weight(config, sigma),
        )
        return MPASPrimitiveEquationModel(mesh=grid, sigma_coord=sigma, config=cfg)

    if solver_name == "mpas_compressible_euler":
        _reject_unselectable_time_integrator("MPAS non-hydrostatic (compressible Euler)")
        from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
            MPASCompressibleEulerModel, MPASCompressibleEulerConfig,
        )
        # Non-hydrostatic uses a height (z-star) coordinate, not sigma:
        # flat-topography defaults as in the CDGrid-NH branch; the MPAS
        # surface field is 1-D over cells.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                height_coord = create_height_coordinate(nlev, 30_000.0)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.latCell)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        # conservation_fixer=False overrides fix_mass=True (lat-lon contract).
        nh_cfg = MPASCompressibleEulerConfig(
            nu_del2=diff.A_h,
            nu_del4=diff.hyperdiff,
            K_h=diff.K_h_A,
            fix_mass=dc.fix_mass and dc.conservation_fixer,
            anchor_mass_to_initial=dc.fix_mass and dc.conservation_fixer,
        )
        return MPASCompressibleEulerModel(grid, height_coord, terrain_metric, nh_cfg)

    # ----- FV3 six-face duo cube (certified fv_dynamics JAX lane) -----
    if (solver_name == "fv3_duo_primitive_equations"
            and getattr(config.dycore, "fv3_duo_column_lane", False)):
        return _create_fv3_duo_column_model(config, gc, model_type,
                                            bundle=fv3_duo_bundle)
    if solver_name == "fv3_duo_primitive_equations":
        # Slice 1 contract, enforced LOUDLY. The core SUPPORTS moist
        # coupling (zvir != 0) on both arms now, but this lane never
        # passes it and never routes tracers, so slice 1 stays dry by
        # construction; consv_te != 0 still raises in the core, and
        # the dtype-uniformity gate checks every leaf.
        # This lane never routes physics tendencies, so any active scheme
        # would be SILENTLY inert — the exact failure mode dispatch
        # hardening exists to prevent.
        _physics_on = {
            name: getattr(config, name)
            for name in ("radiation", "convection", "microphysics",
                         "turbulence", "gravity_wave_drag")
            if getattr(config, name) != "none"
        }
        # The ONE routed scheme: Kessler warm rain, the shared column core
        # applied by _run_fv3_duo._fv3_duo_apply_kessler after each step
        # (apply_kessler_step_sixface_jax).  Alone -- with Held-Suarez the
        # ordering of two operator-split forcings is an uncertified choice.
        kessler_alone = _physics_on == {"microphysics": "kessler"}
        if kessler_alone and config.held_suarez_forcing:
            raise ValueError(
                "fv3_duo: microphysics='kessler' together with "
                "held_suarez_forcing is not certified (two operator-split "
                "forcings, unmeasured ordering); choose one.")
        if kessler_alone and model_type != "hydrostatic":
            raise ValueError(
                "fv3_duo Kessler is hydrostatic-only: the bridge reads "
                "pt as temperature on the hydrostatic post-remap state; "
                f"model_type={model_type!r} + kessler is uncertified.")
        if _physics_on and not kessler_alone:
            raise ValueError(
                f"fv3_duo runs DRY dynamics: the certified fv_dynamics "
                f"lane refuses moist coupling (fv3_dynamics.py:301-311) "
                f"and the driver lane routes no scheme tendencies (the "
                f"only physics it runs is the certified Held-Suarez step, "
                f"--held-suarez-forcing, or Kessler microphysics ALONE), "
                f"so these active schemes would "
                f"be silently inert: {_physics_on}. Set them all to "
                f"'none' (with --allow-disabled-physics in run_amip).")
        if config.held_suarez_forcing and model_type != "hydrostatic":
            # The certified HS orchestration is hydrostatic-only (the
            # parity runner refuses --physics held_suarez with --nh; the
            # oracle HS deck is hydrostatic), so the NH arm would run an
            # uncertified combination — refuse rather than extrapolate.
            raise ValueError(
                "fv3_duo Held-Suarez forcing is hydrostatic-only: the "
                "certified 3-pass HS step (full_step_oracle_parity) is "
                "gated on the hydrostatic arm; model_type="
                f"{model_type!r} + held_suarez_forcing is uncertified. "
                "Use model_type='hydrostatic' or drop "
                "--held-suarez-forcing.")
        # Precision -> FV3DuoConfig.storage_dtype (coarse policy,
        # 2026-08-28). fp64 is the certified default. fp32/mixed are
        # accepted here and mapped; the MODEL then gives the authoritative
        # "runtime not yet wired" message (the dtype-uniformity gates + the
        # config field are in place, but the in-phase fp64 workspaces + grid
        # metrics are not yet threaded). Mapping rather than refusing here
        # keeps the single source of truth in FV3DuoDynamicsModel.__init__.
        _PRECISION_TO_DTYPE = {
            "fp64": "float64", "float64": "float64",
            # mixed_fp64_storage keeps fp64 STORAGE (only compute/accumulate
            # differ in the general policy) -> uniform fp64 here = the
            # certified path, runnable today.
            "mixed_fp64_storage": "float64",
            "fp32": "float32", "float32": "float32",
            # "mixed" resolves to fp32 storage; a true per-op mixed split
            # (fp64 pressure column / energy fixer) is a later increment,
            # so the model refuses it with that note until then.
            "mixed": "float32",
        }
        _storage_dtype = _PRECISION_TO_DTYPE.get(config.precision)
        if _storage_dtype is None:
            raise ValueError(
                f"fv3_duo: unsupported precision={config.precision!r}; "
                f"expected one of {sorted(_PRECISION_TO_DTYPE)}.")
        if _storage_dtype != "float64":
            # fp32 / mixed (fp32-storage) map through, but the RUNTIME is
            # not yet wired (the ~57 in-phase fp64 workspace allocations +
            # the fp64 grid metrics/halo tables must be threaded to the
            # storage dtype, and the pressure column / energy fixer kept
            # fp64 for a true mixed mode). Refuse at the FACTORY (config
            # level) so a driver run fails on the config, not deep in the
            # model -- keeping "fp64" in the message as the supported
            # value. The model carries an independent NotImplementedError
            # guard for a DIRECT FV3DuoConfig(storage_dtype=...)
            # construction that bypasses this factory. Tracked: fv3
            # fp32/mixed increment 2.
            raise ValueError(
                f"fv3_duo currently runs precision='fp64' only "
                f"(mixed_fp64_storage also resolves to uniform fp64); "
                f"precision={config.precision!r} -> storage_dtype="
                f"{_storage_dtype!r} needs the fp32/mixed RUNTIME which is "
                f"not yet wired. Use fp64.")
        if config.distributed and config.distributed_mode != "spmd":
            raise ValueError(
                f"fv3_duo distributed runs are SPMD-only "
                f"(--distributed-mode spmd): the duo lane shards the six-"
                f"face axis under one jitted program (ring exchanges + "
                f"face-batched phases, PR #1656); an mpi4jax-style rank "
                f"decomposition is not wired. Got distributed_mode="
                f"{config.distributed_mode!r}.")
        # DEFAULT-DENY backstop: anything else non-default is refused,
        # all offenders listed at once (see _refuse_fv3_duo_non_default).
        _refuse_fv3_duo_non_default(config)
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoConfig,
            FV3DuoDynamicsModel,
        )
        from legoesm.grids.factory import create_fv3_duo_grid

        km = gc.nlev
        if km not in (5, 10):
            raise ValueError(
                f"fv3_duo runs the analytic set_eta branch only "
                f"(fv_eta.F90:334-344, km in {{5, 10}}); got grid.nlev="
                f"{km}. The other km are hand-tabulated in the oracle and "
                f"are not ported.")
        # The duo lane steps its own six-face bundle, not the driver's
        # standard cubed-sphere grid (which stays for lat/lon metadata /
        # topography accessors) — discretization-keyed wiring, no driver
        # grid dispatch (L1).
        bundle = create_fv3_duo_grid(gc.resolution)
        if config.held_suarez_forcing and bundle.ctx_np.get("ectx") is None:
            # create_fv3_duo_grid builds with use_ext_bundle=True, so this
            # is unreachable today; it fails CLOSED at construction (not
            # mid-run) if the grid factory ever stops building the ext
            # bundle the HS step's c2l Earth-frame winds need.
            raise ValueError(
                "fv3_duo held_suarez_forcing needs the duo ext bundle "
                "(ctx['ectx'] with amat6) for the c2l Earth-frame winds; "
                "the grid bundle was built without it "
                "(build_six_face_duo_context use_ext_bundle=False?).")
        cfg = FV3DuoConfig(
            km=km,
            hydrostatic=(model_type == "hydrostatic"),
            storage_dtype=_storage_dtype,
            # Kessler => MOIST dynamics (user 2026-09-24): a moist scheme
            # on the adiabatic core is the misleading configuration GLM
            # flagged, so the coupling follows the scheme, never a knob.
            moist=(config.microphysics == "kessler"),
        )
        # AUTO-ADAPT the execution layout to the VISIBLE devices (user
        # 2026-08-28: "adjust automatically to the number of devices").
        # The face sharding + face-batched step is near-bitwise the
        # single-device loop (tolerance-gated: spmd_multiprocess_parity
        # asserts rtol/atol 1e-12 at 2/3/6 devices; step_face_batched
        # batched==loop asserts rtol 1e-13/atol 1e-12) -- it changes
        # PERFORMANCE and the last bits only, NOT the physics, which is
        # why auto-selecting it on device count is not a hidden SCIENTIFIC
        # choice. The resolved layout is LOGGED (below) so it appears in
        # the run record, never sits silently in a default.
        #
        # SCOPE: multi-PROCESS stays EXPLICIT behind --distributed -- it
        # changes the driver's checkpoint/snapshot I/O (gather/broadcast/
        # reshard) and the refusal gates key off config.distributed;
        # auto-enabling it would run the single-process I/O path across
        # ranks (each np.asarray sees only its own shards -> corrupt).
        # Single-process auto-sharding is safe (every shard is host-local,
        # so the single-process I/O path is correct as-is).
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        from legoesm.parallel.early_init import launcher_world_size

        # A multi-process launch WITHOUT --distributed never initialised
        # jax.distributed, so jax.process_count() returns 1 even though N
        # copies of this program are running -- each would then auto-shard
        # over its LOCAL devices and clobber the others' output (GLM
        # mechanism review 2026-08-28). Take the max of jax.process_count()
        # and the launcher's declared world size (SLURM step / MPI / PMI /
        # SLURM_NTASKS) -- REFUSE-SAFE, a false positive only costs the
        # user a --distributed flag while a false negative corrupts. The
        # decision is a PURE function of the counts (resolve_fv3_duo_layout),
        # unit-tested in test_fv3_duo_layout_policy.
        multiprocess = config.distributed
        devs = jax.devices() if multiprocess else jax.local_devices()
        kt = config.dycore.fv3_duo_windows
        if kt is not None:
            # EXPLICIT window SPMD (M6 in the driver): 6*kt*kt devices, one
            # window each, on a (face, tile_i, tile_j) mesh.  No auto-
            # selection and no tolerance on the count: a mismatch is a
            # mis-built launch, refused (user call 2026-09-21).
            pad = config.dycore.fv3_duo_window_pad
            need = 6 * kt * kt
            if len(devs) != need:
                raise ValueError(
                    f"fv3_duo_windows={kt} needs exactly {need} "
                    f"{'global' if multiprocess else 'local'} devices "
                    f"(6*kt*kt, one window each); found {len(devs)}. Launch "
                    f"{need} ranks with --distributed --distributed-mode "
                    f"spmd, or drop --fv3-duo-windows for the face layout.")
            mesh = Mesh(np.array(devs).reshape(6, kt, kt),
                        ("face", "tile_i", "tile_j"))
            logger.info(
                "  fv3_duo layout: WINDOW-sharded, kt=%d pad=%d over %d %s "
                "device(s) (one window each) + face-batched%s", kt, pad,
                len(devs), "global" if multiprocess else "local",
                " [multi-process SPMD]" if multiprocess else "")
            return FV3DuoDynamicsModel(
                bundle, cfg, step_spmd_mesh=mesh, step_windows=(kt, pad),
                step_face_batched=True)
        layout = resolve_fv3_duo_layout(
            world=max(jax.process_count(), launcher_world_size()),
            n_local=jax.local_device_count(),
            n_global=jax.device_count(),
            distributed=config.distributed)

        if layout == "shard":
            mesh = Mesh(np.array(devs), ("face",))
            logger.info(
                "  fv3_duo layout: face-sharded over %d %s device(s) "
                "(1/2/3 faces each) + face-batched%s", len(devs),
                "global" if multiprocess else "local",
                " [multi-process SPMD]" if multiprocess else "")
            return FV3DuoDynamicsModel(
                bundle, cfg,
                step_out_shardings=NamedSharding(mesh,
                                                 PartitionSpec("face")),
                step_spmd_mesh=mesh,
                step_face_batched=True)

        # 'single': 1 device, or an auto-path count that does not divide 6
        # (4/5/7...). The latter is a LOUD fall-back (never a silent
        # behaviour substitution) so the wasted devices are visible; the
        # explicit-distributed unshardable case already raised inside
        # resolve_fv3_duo_layout.
        if len(devs) >= 2:
            logger.warning(
                "  fv3_duo layout: %d local devices do not divide the 6 "
                "cube faces (need 2/3/6); running SINGLE-DEVICE on %s. "
                "Set CUDA_VISIBLE_DEVICES to 2/3/6 devices to face-shard.",
                len(devs), devs[0])
        else:
            logger.info("  fv3_duo layout: single-device (%s)", devs[0])
        return FV3DuoDynamicsModel(bundle, cfg)

    # ----- Doubly-periodic plane -----
    if solver_name == "plane_compressible_euler":
        _reject_unselectable_time_integrator("plane non-hydrostatic (compressible Euler)")
        from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
            CompressibleEulerConfig,
        )
        from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
            PlaneCompressibleEulerModel,
            make_flat_plane_terrain_metric,
        )
        # PR2c MVP: every plane setup is dry and uses flat terrain. The
        # plane grid carries its own ``f_y`` (Coriolis) field built at
        # ``create_plane_grid`` time; the driver does not override it.
        # Build a flat terrain metric if the grid did not preattach one,
        # and a default height coordinate if the driver did not pass a
        # vertical coordinate that exposes ``z_full`` / ``z_half``.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None:
            from legoesm.grids.vertical import create_height_coordinate
            nlev = grid.nlev
            # 30 km model top is the same default used by the
            # cubed-sphere NH branch above.
            height_coord = create_height_coordinate(nlev, H=30_000.0)
        if terrain_metric is None:
            terrain_metric = make_flat_plane_terrain_metric(grid, height_coord)
        # PR2c keeps the driver path strict: hyperdiff / sponge knobs
        # come from the dycore config but the plane dycore rejects
        # them per ``validate_plane_config`` until PR3. The driver
        # therefore constructs a minimal config that is safe for
        # ``PR2c`` use; users wanting sponge enabled can construct
        # ``PlaneCompressibleEulerModel`` directly with a custom
        # ``CompressibleEulerConfig``.
        cfg = CompressibleEulerConfig(
            sponge_coeff=0.0,
            hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0,
            hyperdiff_w_coeff=0.0,
            semi_implicit_acoustic=False,
            use_coriolis=False,
            # conservation_fixer=False overrides fix_mass=True (lat-lon
            # contract; codex 2026-07-12 round 2).
            fix_mass=dc.fix_mass and dc.conservation_fixer,
            anchor_mass_to_initial=dc.fix_mass and dc.conservation_fixer,
        )
        return PlaneCompressibleEulerModel(grid, height_coord, terrain_metric, cfg)

    # ----- Lat-lon C-grid solvers -----
    if solver_name in ("latlon_cgrid_shallow_water",
                       "latlon_cgrid_primitive_equations"):
        # The lat-lon C-grid configs support A_h and fix_mass but not
        # The lat-lon C-grid solver uses Laplacian viscosity (A_h) only —
        # it has no biharmonic hyperdiffusion operator.  However, the
        # driver still uses compute_diffusion().hyperdiff for moisture
        # smoothing, so hyperdiff_scale is NOT rejected here.
        #
        # Divergence damping is not used anywhere on lat-lon, so values
        # > 1.0 (requesting amplified damping) are rejected.
        if dc.div_damp_scale > 1.0:
            raise ValueError(
                f"Lat-lon C-grid solver does not support divergence "
                f"damping. div_damp_scale={dc.div_damp_scale} was "
                f"requested but this mechanism is not implemented. "
                f"Set div_damp_scale=1.0 (default) or 0.0 (disabled)."
            )

        # Honor conservation_fixer: when explicitly False, disable fix_mass
        # even if dc.fix_mass is True.
        _fix_mass = dc.fix_mass
        if dc.conservation_fixer is False:
            _fix_mass = False
            if dc.fix_mass:
                logger.info(
                    "Lat-lon C-grid solver: conservation_fixer=False "
                    "overrides fix_mass=True → mass fixer disabled",
                )

        # ---- Pole-cell CFL safeguards ----
        # The explicit C-grid solver on a lat-lon grid has its smallest
        # cell at the poles: dx_pole = R * dlon * cos(π/2 - dlat/2).
        # Both the advective CFL (dt < dx / c_grav) and the diffusive
        # CFL (A_h < 0.4 * dx² / dt) must be satisfied there.
        #
        # Stage 3-E: when ``dc.use_polar_filter`` is True, the Fourier
        # polar filter truncates Fourier modes in longitude that would
        # violate CFL at high latitudes, so the dynamics is stable at
        # ``dt`` set by the equatorial CFL instead of the pole CFL.
        # ``dx_equator`` = R * dlon ≫ dx_pole at all but the lowest
        # resolutions, so this typically lifts the clamp by ~100x at
        # n_lat=180 and ~10x at n_lat=90 — enough to make a 100-y AMIP
        # at 1° feasible within a chained 72-h SLURM budget.
        from legoesm.core.cfl import (
            pole_cell_dx, cfl_max_dt, max_laplacian_viscosity,
        )
        dx_pole = pole_cell_dx(grid)
        c_grav = 300.0  # gravity wave speed [m/s]
        dt_max_advective_pole = cfl_max_dt(
            dx_pole, c_grav, cfl_number=0.8, ndim=1,
        )
        _effective_dt = dc.dt

        if dc.use_polar_filter:
            # Filter on → equatorial CFL is the effective limit.
            # dx_equator = R * dlon = circumference / n_lon.
            import math as _math
            dx_equator = float(2.0 * _math.pi * grid.radius / grid.n_lon)
            dt_max_advective = cfl_max_dt(
                dx_equator, c_grav, cfl_number=0.8, ndim=1,
            )
            dx_for_diffusion = dx_equator
            _clamp_dx_label = "equatorial"
        else:
            dt_max_advective = dt_max_advective_pole
            dx_for_diffusion = dx_pole
            _clamp_dx_label = "pole-cell"

        if _effective_dt > dt_max_advective:
            logger.warning(
                "Lat-lon C-grid: dt=%.1f s exceeds %s advective "
                "CFL limit (%.1f s); clamping to %.1f s. "
                "Set dycore.dt <= %.1f for this grid.",
                _effective_dt, _clamp_dx_label, dt_max_advective,
                dt_max_advective, dt_max_advective,
            )
            _effective_dt = dt_max_advective

        A_h_max = max_laplacian_viscosity(dx_for_diffusion, _effective_dt)
        _A_h = min(diff.A_h, A_h_max)
        if diff.A_h > A_h_max:
            logger.warning(
                "Lat-lon C-grid: A_h=%.2e exceeds %s diffusive "
                "CFL limit (%.2e); clamping.",
                diff.A_h, _clamp_dx_label, A_h_max,
            )

    if solver_name == "latlon_cgrid_shallow_water":
        _reject_unselectable_time_integrator("lat-lon C-grid shallow water")
        from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterModel, CGridLatLonShallowWaterConfig,
        )
        cfg = CGridLatLonShallowWaterConfig(
            A_h=_A_h,
            fix_mass=_fix_mass,
            # Stage 3-E: pass polar-filter parameters through (mirrors the
            # PE branch below).  Previously dropped: the dt/CFL relaxation
            # above already assumed the filter was ON (dt lifted to the
            # equatorial CFL) while the model silently ran WITHOUT it.
            use_polar_filter=dc.use_polar_filter,
            polar_filter_cutoff_deg=dc.polar_filter_cutoff_deg,
            polar_filter_max_wave_speed=dc.polar_filter_max_wave_speed,
        )
        model = CGridLatLonShallowWaterModel(grid, cfg, dt=_effective_dt)
        model.effective_dt = _effective_dt
        return model

    if solver_name == "latlon_cgrid_primitive_equations":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        )
        cfg = CGridLatLonPrimitiveEquationConfig(
            A_h=_A_h,
            fix_mass=_fix_mass,
            conservative_tracer_clamp=dc.mpas_conservative_tracer_clamp,  # #1354/#1515 borrow (grid-general knob)
            energy_consistent_moisture_clip=config.energy_consistent_moisture_clip,  # #1354/#1515 (hard-floor path only)
            # Stage 3-E: pass polar-filter parameters through.  When
            # use_polar_filter is False (default) the model's filter
            # mask is None and no FFT is applied — bit-identical to
            # pre-Stage-3-E behaviour.
            # Direct attribute access (these are guaranteed DycoreConfig fields);
            # getattr-with-literal fallbacks would mask a rename behind a stale
            # default instead of failing (CLAUDE.md).
            use_polar_filter=dc.use_polar_filter,
            polar_filter_cutoff_deg=dc.polar_filter_cutoff_deg,
            polar_filter_max_wave_speed=dc.polar_filter_max_wave_speed,
            # #836 top sponge (Rayleigh damping toward the lid).  Default
            # sponge_coeff=0 -> OFF -> the tendency is bit-identical to the
            # pre-#836 dycore.  Direct DycoreConfig attribute access (guaranteed
            # fields; no getattr-literal fallback that would mask a rename).
            sponge_coeff=dc.sponge_coeff,
            sponge_width_m=dc.sponge_width_m,
            sponge_shape=dc.sponge_shape,
            sponge_scale_height_m=dc.sponge_scale_height_m,
            # #1029 ω-side SB81 conversion (opt-in, default OFF —
            # bit-identical legacy arithmetic form when False).
            sb81_omega_conversion=dc.sb81_omega_conversion,
            # Task #25: time integrator (default ssp_rk3, opt into
            # ssp_rk3_scan for ~1.5× JIT compile speedup at scale).
            # "auto" -> this dycore's own default; explicit names verbatim.
            time_integrator=(CGridLatLonPrimitiveEquationConfig().time_integrator
                             if dc.time_integrator == "auto"
                             else dc.time_integrator),
        )
        model = CGridLatLonPrimitiveEquationModel(
            grid, sigma, cfg, dt=_effective_dt)
        model.effective_dt = _effective_dt
        return model

    # ----- SFNO data-driven -----
    if solver_name == "sfno_shallow_water":
        _reject_unselectable_time_integrator("SFNO shallow water")
        from legoesm.atmosphere.dynamics.neural.sfno_sw import SFNOShallowWaterModel
        return SFNOShallowWaterModel(grid=grid)

    if solver_name == "sfno_primitive_equations":
        _reject_unselectable_time_integrator("SFNO primitive equations")
        from legoesm.atmosphere.dynamics.neural.sfno_pe import (
            SFNOPrimitiveEquationConfig,
            SFNOPrimitiveEquationModel,
        )
        # Do NOT take the bare config default here: it leaves
        # spectral_filter_strength at 0.0, and an undamped state_update rollout
        # is known-divergent (|u850| 33 -> 969 m/s by macro step 4, NaN by step
        # 12 on a trained T63 checkpoint; measured 2026-07-28). A user picking
        # this solver from a config must not get the divergent variant by
        # default. 0.01 / order 8 are the dycore's own filter settings, verified
        # to hold a 240 h rollout physical.
        return SFNOPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma,
            config=SFNOPrimitiveEquationConfig(
                spectral_filter_strength=0.01, spectral_filter_order=8),
        )

    # ----- U-Cast data-driven (convolutional U-Net emulator) -----
    if solver_name == "ucast_primitive_equations":
        _reject_unselectable_time_integrator("u_cast primitive equations")
        from legoesm.atmosphere.dynamics.neural.ucast_pe import (
            UCastPrimitiveEquationConfig,
            UCastPrimitiveEquationModel,
        )
        from legoesm.ml.channel_packing import PE3DChannelSpec

        # The default UCastConfig is sized for the 13-level ERA5 pressure
        # set (4*13+2 = 54 channels).  The driver knows the experiment's
        # actual nlev, so size the network to the PE channel count here —
        # the model ctor fail-fasts on any mismatch.
        base = UCastPrimitiveEquationConfig()
        n_ch = PE3DChannelSpec(nlev=sigma.n_levels).n_channels
        cfg = base._replace(
            ucast_config=base.ucast_config._replace(
                in_channels=n_ch, out_channels=n_ch,
            )
        )
        return UCastPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=cfg)

    # Should be unreachable — the key check above guarantees this.
    raise RuntimeError(f"Internal error: unhandled solver {solver_name!r}")


def _fail_unsupported(model_type: str, discretization: str, grid_type: str):
    """Raise a precise diagnostic for an unsupported combination."""
    # Gather what IS available for each partial match.
    available_for_grid = sorted({
        (k[0], k[1]) for k in _DRIVER_SUPPORTED if k[2] == grid_type
    })
    available_for_model = sorted({
        (k[1], k[2]) for k in _DRIVER_SUPPORTED if k[0] == model_type
    })

    msg_parts = [
        f"Unsupported atmosphere configuration: "
        f"model_type={model_type!r}, discretization={discretization!r}, "
        f"grid_type={grid_type!r}.",
    ]
    if available_for_grid:
        combos = ", ".join(f"({m}, {d})" for m, d in available_for_grid)
        msg_parts.append(
            f"  On grid_type={grid_type!r}, supported (model_type, discretization) "
            f"pairs are: {combos}."
        )
    else:
        all_grids = sorted({k[2] for k in _DRIVER_SUPPORTED})
        msg_parts.append(
            f"  grid_type={grid_type!r} has no driver-supported solvers. "
            f"Supported grid types: {all_grids}."
        )
    if available_for_model:
        combos = ", ".join(f"({d}, {g})" for d, g in available_for_model)
        msg_parts.append(
            f"  For model_type={model_type!r}, supported (discretization, grid_type) "
            f"pairs are: {combos}."
        )
    raise ValueError("\n".join(msg_parts))


# =========================================================================
# Ocean, land, ice, and coupler component factories
# =========================================================================

def create_ocean_component(
    config: ExperimentConfig,
    grid,
    vertical_coord=None,
    *,
    ocean_config=None,
    sst_map=None,
):
    """Create the configured ocean component.

    This factory supports two modes:

    1. **Simple ocean** (default): returns a step-function created by
       ``legoesm.ocean.simple_ocean.make_ocean``.  Pass an optional
       ``SimpleOceanConfig`` via *ocean_config*; if omitted, the
       default fixed-SST configuration is used.
    2. **Full ocean model**: if *ocean_config* is an ``OceanConfig``,
       instantiates a full ``OceanModel`` from ``legoesm.ocean``.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object.
    vertical_coord
        Vertical coordinate (needed for the full ocean model; ignored
        for the simple ocean).
    ocean_config
        A ``SimpleOceanConfig``, an ``OceanConfig``, or an
        ``OceanComplexity`` rung.  If *None*, defaults to
        ``SimpleOceanConfig()`` (fixed SST at 300 K).  An
        ``OceanComplexity`` selects the simple-ocean *mode* via the
        components taxonomy (``fixed_sst``/``slab``/``slab_multilayer``);
        ``OceanComplexity.FULL_3D`` is rejected here because the full 3D
        model needs an explicit ``OceanConfig``.
    sst_map : array-like, optional
        Spatial SST map for fixed/slab modes.

    Returns
    -------
    step_fn or OceanModel
        A callable step function (simple ocean) or an ``OceanModel``
        instance (full ocean).
    """
    from legoesm.components import OceanComplexity, ocean_simple_mode
    from legoesm.ocean.simple_ocean import SimpleOceanConfig, make_ocean
    from legoesm.ocean.state import OceanConfig

    if ocean_config is None:
        ocean_config = SimpleOceanConfig()

    # An OceanComplexity rung selects the simple-ocean mode via the components
    # taxonomy (the single source for fixed_sst/slab/slab_multilayer -> mode).
    # full_3d is a prognostic model needing a full OceanConfig, not a rung.
    if isinstance(ocean_config, OceanComplexity):
        if ocean_config is OceanComplexity.FULL_3D:
            raise ValueError(
                "full_3d ocean is a prognostic 3D model that needs its "
                "parameters: pass ocean_config=OceanConfig(...) rather than "
                "the OceanComplexity.FULL_3D rung."
            )
        ocean_config = SimpleOceanConfig(mode=ocean_simple_mode(ocean_config))

    if isinstance(ocean_config, SimpleOceanConfig):
        logger.info(
            "Ocean: simple mode=%s (h_mix=%.0f m)",
            ocean_config.mode, ocean_config.h_mix,
        )
        return make_ocean(ocean_config, sst_map=sst_map)

    if isinstance(ocean_config, OceanConfig):
        from legoesm.ocean import OceanModel
        # full_3d OceanModel is cubed-sphere-only (it calls create_cubed_sphere_cdgrid).
        # Guard the grid family with a clear message instead of an AttributeError
        # deep inside cdgrid construction (mirrors the resolve_model_complexity guard).
        from legoesm.driver.config import normalize_grid_type
        from legoesm.grids.cubed_sphere import CubedSphereGrid
        # full_3d OceanModel is cubed-sphere-only (it calls
        # create_cubed_sphere_cdgrid).  Detect the family from the GRID OBJECT,
        # not ``config.grid`` — ``config`` is logging-only here and is legitimately
        # ``None`` on the complexity-builder path (test_component_complexity passes
        # config=None with a real grid).  Mirrors the isinstance(grid,
        # CubedSphereGrid) convention used throughout ocean/atmosphere physics.
        if not isinstance(grid, CubedSphereGrid):
            _gt = (
                normalize_grid_type(config.grid.grid_type)
                if config is not None and getattr(config, "grid", None) is not None
                else type(grid).__name__
            )
            raise ValueError(
                f"full_3d (OceanConfig) ocean is implemented only for grid_type in "
                f"{sorted(_FULL_OCEAN_GRID_TYPES)} (the cubed-sphere OceanModel); got "
                f"{_gt!r}. Lat-lon and MPAS full ocean require their grid-specific "
                f"models/configs — that wiring is not yet implemented here."
            )
        if vertical_coord is None:
            raise ValueError(
                "full OceanModel needs a vertical_coord (an ocean z-star "
                "coordinate from legoesm.ocean.vertical.create_ocean_z_star)."
            )
        logger.info("Ocean: full OceanModel")
        # OceanModel's vertical-coordinate parameter is named ``z_coord``.
        return OceanModel(grid=grid, z_coord=vertical_coord, config=ocean_config)

    raise TypeError(
        f"ocean_config must be SimpleOceanConfig or OceanConfig, "
        f"got {type(ocean_config).__name__!r}"
    )


def create_land_component(config: ExperimentConfig, grid, *, land_config=None):
    """Create the configured land surface model.

    Returns a ``step_land`` callable (slab land) or a
    ``step_multilayer_land`` callable, depending on the config type.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object (used for determining spatial shape).
    land_config
        A ``LandConfig``, a ``MultiLayerLandConfig``, or a
        ``LandComplexity`` rung.  If *None*, defaults to ``LandConfig()``.
        A ``LandComplexity`` selects the default config of that complexity
        (``slab`` -> ``LandConfig``, ``multilayer``/column ->
        ``MultiLayerLandConfig``).

    Returns
    -------
    step_fn : callable
        The land surface step function.
    """
    from legoesm.components import LandComplexity
    from legoesm.land import (
        LandConfig, MultiLayerLandConfig,
        step_land, step_multilayer_land,
        TwoLeafCanopyConfig,
    )

    if land_config is None:
        land_config = LandConfig()

    # A LandComplexity rung selects the default config of that complexity.
    # Land has no string "mode" (the config *type* is the complexity), so the
    # rung -> config resolution lives here in the factory (which owns the
    # concrete land configs) rather than in the pure components taxonomy.
    if isinstance(land_config, LandComplexity):
        # Explicit, exhaustive rung dispatch with a final raise (CLAUDE.md: never a
        # silent else:default — a new LandComplexity rung must fail loudly here,
        # not silently degrade to the slab LandConfig).  Mirrors ice_complexity_config.
        if land_config is LandComplexity.MULTILAYER:
            land_config = MultiLayerLandConfig()
        elif land_config is LandComplexity.SLAB:
            land_config = LandConfig()
        else:
            raise ValueError(
                f"unhandled LandComplexity rung: {land_config!r} — add its "
                f"config mapping in create_land_component (not yet implemented)."
            )

    if isinstance(land_config, MultiLayerLandConfig):
        scheme_name = type(land_config.surface_scheme).__name__
        logger.info(
            "Land: multilayer model (n_layers=%d, surface_scheme=%s)",
            land_config.soil_grid.n_layers, scheme_name,
        )
        return step_multilayer_land

    if isinstance(land_config, LandConfig):
        scheme_name = type(land_config.surface_scheme).__name__
        logger.info("Land: slab model (surface_scheme=%s)", scheme_name)
        return step_land

    raise TypeError(
        f"land_config must be LandConfig or MultiLayerLandConfig, "
        f"got {type(land_config).__name__!r}"
    )


class SeaIceComponent(NamedTuple):
    """A built sea-ice brick: the step function bound to its resolved config.

    Unlike the ocean (whose step closure binds its config) and land (whose two
    distinct step functions encode the rung), the sea-ice model has a *single*
    step function that takes its config at call time.  Returning the resolved
    ``SeaIceConfig`` alongside ``step`` keeps the selected complexity from being
    silently dropped: the caller wires ``component.step`` with
    ``component.config`` and a dynamic rung cannot degrade to thermodynamic
    defaults downstream.
    """

    step: object  # step_sea_ice(state, forcing, config, dt) -> SeaIceState
    config: object  # the resolved SeaIceConfig to pass to step


def create_ice_component(config: ExperimentConfig, grid, *, ice_config=None):
    """Create the configured sea-ice model.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object (needed when ice dynamics are enabled).
    ice_config
        A ``SeaIceConfig`` instance, or an ``IceComplexity`` rung.  If *None*,
        defaults to ``SeaIceConfig()`` (thermodynamic slab).  An
        ``IceComplexity`` is resolved to a ``SeaIceConfig`` via
        :func:`ice_complexity_config`.

    Returns
    -------
    SeaIceComponent
        ``(step, config)`` — ``step_sea_ice`` and the *resolved* ``SeaIceConfig``
        the caller must pass to it (so the selected rung is never dropped).
    """
    from legoesm.components import IceComplexity
    from legoesm.ice import SeaIceConfig, step_sea_ice

    if ice_config is None:
        ice_config = SeaIceConfig()
    if isinstance(ice_config, IceComplexity):
        ice_config = ice_complexity_config(ice_config)

    logger.info(
        "Ice: dynamics=%s, n_categories=%d",
        ice_config.dynamics, ice_config.n_categories,
    )
    return SeaIceComponent(step=step_sea_ice, config=ice_config)


def ice_complexity_config(complexity):
    """Resolve an :class:`~legoesm.components.IceComplexity` rung to a SeaIceConfig.

    ``thermodynamic`` -> the default single-category slab (``SeaIceConfig()``,
    ``dynamics="none"``); ``dynamic`` -> EVP rheology + 5-category CICE ITD
    (``dynamics="evp", n_categories=5``).  Orthogonal sub-physics
    (snow/brine/ridging/ponds) stay at their opt-in defaults — they are gates,
    not rungs.  Lives here (not the pure components taxonomy) because it builds a
    concrete ``SeaIceConfig``, which the driver owns.  Raises ``ValueError`` on an
    unknown rung.
    """
    from legoesm.components import IceComplexity
    from legoesm.ice import SeaIceConfig

    c = IceComplexity(complexity)  # raises ValueError on an unknown rung
    if c is IceComplexity.THERMODYNAMIC:
        return SeaIceConfig()  # default single-category slab (dynamics="none")
    if c is IceComplexity.DYNAMIC:
        return SeaIceConfig(dynamics="evp", n_categories=5)
    # Explicit branches (no implicit default): a future IceComplexity rung added
    # without a mapping here must fail loudly, not silently fall back to slab.
    raise ValueError(f"unhandled IceComplexity rung: {c!r}")


class ModelComplexitySpec(NamedTuple):
    """Concrete, factory-ready inputs for every component at one complexity level.

    The pure components taxonomy (``model_complexity_rungs``) yields complexity
    *enums*; this driver-level resolver turns them into values the component
    factories actually accept — crucially a concrete ``OceanConfig`` for the
    ``full_3d`` ocean (``create_ocean_component`` rejects the bare ``FULL_3D``
    rung because the 3-D model needs its parameters).  So a driver builds every
    component at a chosen complexity through one resolution, full ocean included.
    """

    atmosphere_model_type: str  # -> ExperimentConfig.dycore.model_type
    ocean_config: object  # create_ocean_component accepts: OceanComplexity rung | OceanConfig
    land_config: object  # create_land_component accepts: LandComplexity rung
    ice_config: object  # the resolved SeaIceConfig (pass to step_sea_ice)


#: Grid families whose full-3-D ocean is wired through ``create_ocean_component``
#: today (``OceanConfig`` -> cubed-sphere ``OceanModel``).  Lat-lon C-grid and
#: MPAS full ocean use their own grid-specific models/configs
#: (``LatLonCGridOceanModel`` / MPAS) that are not yet routed through this
#: resolver — see :func:`resolve_model_complexity`.
_FULL_OCEAN_GRID_TYPES: frozenset[str] = frozenset({"cubed_sphere"})


def resolve_model_complexity(level, *, grid_type: str = "cubed_sphere"):
    """Resolve a model-wide complexity *level* to concrete, factory-ready configs.

    ``resolve_model_complexity("full")`` -> a :class:`ModelComplexitySpec` whose
    ``ocean_config`` is an ``OceanConfig`` (the default 3-D ocean; pass an
    explicit ``OceanConfig`` for custom parameters), so every field can be handed
    straight to its factory:

        spec = resolve_model_complexity(level, grid_type=cfg.grid.grid_type)
        dycore   = create_atmosphere_dycore(cfg_with(spec.atmosphere_model_type), grid, sigma)
        ocean    = create_ocean_component(cfg, grid, z_coord, ocean_config=spec.ocean_config)
        land     = create_land_component(cfg, grid, land_config=spec.land_config)
        ice      = create_ice_component(cfg, grid, ice_config=spec.ice_config)

    The simple-ocean and land fields stay as their complexity *rungs* (which the
    factories already resolve), so this resolver does not duplicate that logic; it
    only synthesises the one concrete config the factory cannot derive from a bare
    rung (the full 3-D ocean).

    *grid_type* selects which full-3-D ocean to synthesise.  The full ocean MODEL
    is grid-dependent: only ``"cubed_sphere"`` is wired through
    ``create_ocean_component`` (``OceanConfig`` -> ``OceanModel``, the cdgrid
    ocean).  Lat-lon C-grid and MPAS full ocean use their own grid-specific
    models/configs not yet routed here, so ``full`` complexity on those grids
    raises ``ValueError`` rather than silently synthesising a cubed-sphere
    ``OceanConfig`` that would build the wrong ocean class.  Simple ocean rungs
    are grid-agnostic, so ``idealized``/``intermediate`` work on any grid.  Also
    raises ``ValueError`` on an unknown *level*.
    """
    from legoesm.components import (
        OceanComplexity,
        atmosphere_model_type,
        model_complexity_rungs,
    )
    from legoesm.ocean.state import OceanConfig

    rungs = model_complexity_rungs(level)
    if rungs.ocean is OceanComplexity.FULL_3D:
        if grid_type not in _FULL_OCEAN_GRID_TYPES:
            raise ValueError(
                f"full_3d ocean via resolve_model_complexity is currently wired "
                f"only for grid_type in {sorted(_FULL_OCEAN_GRID_TYPES)} "
                f"(OceanConfig -> cubed-sphere OceanModel); got {grid_type!r}. "
                f"Lat-lon and MPAS full ocean use their grid-specific "
                f"models/configs — build them explicitly."
            )
        # Full-3D cube ocean uses the FV3 C-D grid (the only cube backend) with
        # the FV3-faithful fv3sw barotropic — NOT the default a_grid barotropic
        # (forbidden by the never-A-grid / FV3-faithfulness directive).  Import
        # the SW core provider so its registry entry exists before construction.
        import legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid  # noqa: F401
        ocean_config = OceanConfig(barotropic_staggering="fv3sw")
    else:
        ocean_config = rungs.ocean  # simple rung — grid-agnostic, factory resolves
    return ModelComplexitySpec(
        atmosphere_model_type=atmosphere_model_type(rungs.atmosphere),
        ocean_config=ocean_config,
        land_config=rungs.land,
        ice_config=ice_complexity_config(rungs.ice),
    )
