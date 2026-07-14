"""Composable model driver for legoESM.

Orchestrates grid creation, vertical coordinate, dycore, physics,
forcing, state initialization, time-stepping, diagnostics, and
checkpointing into a single reusable class.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.forcing.time_utils import daily_forcing_bucket, day_to_calendar

from legoesm.core.conservation import (
    compute_global_moisture, fix_moisture_hydrostatic,
    energy_consistent_moisture_floor,
)
from legoesm.core.tracers import (
    TracerRegistry,
    init_tracers,
    make_full_moisture_registry,
    make_moisture_registry,
)
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import (
    build_physics_pipeline,
    required_microphysics_tracer_slots,
    turbulence_config_for,
    validate_microphysics_tracer_slots,
)
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.driver.restart import save_restart, load_restart

logger = logging.getLogger("legoesm.driver")


def _meshes_compatible(a, b) -> bool:
    """Return True when JAX device meshes *a* and *b* match enough to
    safely share the SPMD halo backend.

    Two meshes are compatible when they have the same axis names in the
    same order and the same device ids in the same flat layout.  Object
    identity is the fast path; semantic equality is the fallback so two
    independently-constructed but equivalent ``Mesh`` objects compare
    equal (Codex review for issue #275 fix A).
    """
    if a is None or b is None:
        return a is b
    if a is b:
        return True
    try:
        if tuple(a.axis_names) != tuple(b.axis_names):
            return False
        # ``mesh.devices`` is a NumPy array of JAX devices.  Flatten
        # and compare device ids — identity of the Device objects is
        # itself a fast check, but ``int(d.id)`` works across both
        # local and distributed runtimes.
        ids_a = tuple(int(d.id) for d in a.devices.flat)
        ids_b = tuple(int(d.id) for d in b.devices.flat)
        return ids_a == ids_b
    except Exception:  # noqa: BLE001 — opaque mesh objects must not crash
        return False


def _wallclock_exhausted(elapsed_s: float, max_s: float, buffer_s: float) -> bool:
    """True if the run should checkpoint and exit to fit the wallclock budget.

    ``max_s <= 0`` disables the check.  Otherwise fire once the elapsed time is
    within ``buffer_s`` of the budget, leaving time to write the checkpoint
    before SLURM kills the job (so a dependency chain can resume).
    """
    return max_s > 0.0 and elapsed_s >= (max_s - buffer_s)


def _standalone_cloud_config(cfg, cloud_scheme: str):
    """Tuned ``CloudConfig`` for the standalone (MPAS/spectral) radiation path.

    Mirrors the FV pipeline's ``build_cloud_config`` call (#689) so the tuned
    experiment-level cloud scalars (``cloud_rh_crit`` / ``cloud_q_c_diagnostic``
    / Xu-Randall knobs) reach the standalone backends too (#870 Phase 1) —
    previously these paths silently ran ``CloudConfig`` defaults.

    ``convective_cloud`` stays OFF here: the standalone radiation call
    (``radiation/integration.py``) does not thread ``conv_precip`` into
    ``compute_cloud_properties``, and ``convective_cloud=True`` without it
    trips that function's loud misconfiguration guard by design.  Returns
    ``None`` (=> scheme-default config) when the scheme is "none".
    """
    if cloud_scheme == "none":
        return None
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config

    # LOUD, not silent (repo doctrine): a user/YAML requesting
    # convective_cloud=True on a standalone backend would otherwise get
    # different physics with no trace (pre-merge codex review).  The lane
    # still runs (the production YAML sets it true for the FV path); the
    # forced drop is now visible in the log.
    if bool(getattr(cfg, "convective_cloud", False)):
        logger.warning(
            "convective_cloud=True is FORCED OFF on the standalone "
            "(MPAS/spectral) radiation path: it does not thread conv_precip, "
            "and convective_cloud without it trips compute_cloud_properties' "
            "misconfiguration guard. The FV (cubed-sphere/latlon) pipeline "
            "honours the setting."
        )

    return build_cloud_config(
        cloud_scheme,
        convective_cloud=False,
        rh_crit=getattr(cfg, "cloud_rh_crit", None),
        q_c_diagnostic=getattr(cfg, "cloud_q_c_diagnostic", None),
        conv_cloud_max=getattr(cfg, "cloud_conv_cloud_max", None),
        conv_cloud_condensate=getattr(cfg, "cloud_conv_cloud_condensate", None),
        p_xr=getattr(cfg, "cloud_p_xr", None),
        alpha_xr=getattr(cfg, "cloud_alpha_xr", None),
    )


class ModelDriver:
    """Top-level simulation driver.

    Encapsulates the full AMIP integration workflow: grid creation,
    physics setup, state initialization, time loop, diagnostics,
    and checkpoint/restart.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration.
    output_dir : str or Path, optional
        Override output directory. If None, auto-generates.
    """

    def __init__(self, config: ExperimentConfig, output_dir: str | Path | None = None):
        # Defensive canonical-name normalization at the driver entry:
        # callers that bypass run_amip's argparse postprocessor (direct
        # test fixtures, ad-hoc scripts, older YAML loaders) might still
        # pass ``voronoi`` / ``icosahedral`` / ``mpas_voronoi`` as
        # grid_type for the SCVT mesh.  The dispatch sites below speak
        # only the canonical ``mpas`` — normalise once here so every
        # downstream branch is consistent.
        from legoesm.driver.config import normalize_grid_type
        canonical_grid_type = normalize_grid_type(config.grid.grid_type)
        if canonical_grid_type != config.grid.grid_type:
            config = config._replace(
                grid=config.grid._replace(grid_type=canonical_grid_type),
            )
        self.config = config
        # Snapshot the INPUT config (post grid-normalization, which is
        # idempotent) before setup() mutates self.config in place — e.g. the
        # CFL-driven dt reduction in _create_dycore.  The run manifest records
        # THIS so `reproduce` replays the identical setup path; recording the
        # post-mutation config (dt already reduced) would take a different path
        # and fail to reproduce.  NamedTuple._replace rebinds self.config to new
        # objects, so this reference stays the original input.
        self._input_config = config
        self.grid = None
        self.sigma = None
        self.model = None
        self.physics = None
        self.state = None
        self.tracers: dict[str, jax.Array] = {}
        warm_registry = make_moisture_registry()
        required_slots = required_microphysics_tracer_slots(config.microphysics)
        if required_slots > warm_registry.n_tracers:
            self.tracer_registry: TracerRegistry = make_full_moisture_registry()
        else:
            self.tracer_registry: TracerRegistry = warm_registry
        validate_microphysics_tracer_slots(
            config.microphysics,
            self.tracer_registry.n_tracers,
            context="ModelDriver tracer registry",
        )
        self.get_sst_sic = None
        # Seasonal insolation offset (days): align model day 0 to
        # config.insolation_start_doy for the radiation day_of_year ONLY, so an
        # AMIP run from a non-January ERA5 date can run the matching solar season
        # without shifting start_day (which the relative-indexed SST forcing
        # depends on). None => 0.0 => legacy (day 0 -> Jan 1). Static Python
        # float: the calendar is computed host-side per step, so this is a
        # constant, never a traced leaf. See config.insolation_start_doy.
        _insol_doy = getattr(config, "insolation_start_doy", None)
        self._insolation_day_offset: float = (
            0.0 if _insol_doy is None else float(_insol_doy) - 1.0
        )
        # Optional per-segment surface-property feedback hook.  A coupled
        # driver sets this to a callable ``day -> (sfc_albedo, sfc_T)`` (each
        # grid-shaped or None) returning the coupler's tile-blended dynamic
        # surface albedo / skin temperature; threaded into radiation as traced
        # SegmentForcing so the sea-ice/ocean/land albedo + skin-T feedbacks
        # reach the atmosphere.  None (AMIP / standalone) ⇒ static blend.
        self.get_sfc_override = None
        # Optional per-segment SHARED surface-flux feedback hook.  A coupled
        # driver sets this to a callable ``day -> (sfc_shflx, sfc_lhflx)``
        # (each grid-shaped [W/m2, +up] or None) returning the coupler's
        # tile-blended sensible / latent heat flux; threaded into the surface
        # tendency as traced SegmentForcing so the atmosphere consumes the SAME
        # surface fluxes the coupler feeds the ocean (air-sea budget closure).
        # None (AMIP / standalone) means the atmosphere computes its own bulk
        # surface fluxes (byte-identical to the pre-shared-flux behaviour).
        self.get_sfc_flux_override = None
        self.diagnostics = None
        self._phis_data = None
        self._f_land = None
        # Prognostic multilayer (Richards) land state, carried in SegmentCarry.land_ml
        # and persisted across segments.  None ⇒ slab-land (scalar T_land) path.
        self._land_ml_state = None
        self._fric_decay = None
        self._qv_smooth_coeff = None
        self._hyperdiffusion_3d_fn = None
        self._hs_newtonian_relax = None  # precomputed HS relaxation fn
        self._ensemble_size = 1
        self._device_config = None
        self._carry_aux: dict = {}  # held radiation + carry metadata for checkpoint
        # (step, day) exactly as the last load_checkpoint returned them —
        # lets the run loops distinguish the production restart convention
        # (callers pass the CHECKPOINT day straight back into run()) from
        # a caller-supplied EPOCH day (the legacy contract).  See the
        # START_DAY normalization in _prepare_run_context / _run_spectral
        # (FIX_RESTART_TIME).
        self._loaded_checkpoint_step_day: tuple | None = None
        self._last_checkpoint_step: int | None = None

        # MPI distributed state (populated by _setup_parallel)
        self._mpi_rank: int | None = None
        self._mpi_world_size: int | None = None
        self._owned_face_ids: jax.Array | None = None  # shape (n_local_faces,)
        self._layout = None  # DistributedLayout for scatter/gather
        # MPAS/Voronoi cell-partition MPI: layout carries the partition
        # (owned+halo cell/edge index maps), local mesh, and halo-exchange
        # handle.  The diagnostics / checkpoint gather distinguish a
        # cell-partitioned run from cubed-sphere faces by checking
        # ``_voronoi_layout is not None`` and read owned-cell ids straight
        # off ``_voronoi_layout.partition``.
        self._voronoi_layout = None
        self._grid_global = None  # global grid preserved under band/cell MPI
        self._physics_lat = None  # rank-local lat for physics
        self._physics_lon = None  # rank-local lon for physics
        # Global owned-cell count for the MPAS diag (partition-static:
        # allreduced ONCE on first use by _mpas_global_diag, then cached).
        self._mpas_g_n_cells = None

        # SPMD halo backend lifecycle.  When the driver activates the
        # explicit SPMD halo backend for multi-GPU single-node cubed-
        # sphere runs, it records the previous backend here so that
        # ``_restore_halo_backend()`` can return the process-global
        # halo dispatch to its prior state on teardown.  Keeps test
        # isolation when several drivers are instantiated in one
        # process and prevents leaked state from affecting later
        # single-rank runs (issue #275 follow-up).
        self._spmd_halo_activated: bool = False
        self._previous_halo_backend: str | None = None

        if output_dir is not None:
            self._output_dir = Path(output_dir)
        elif config.output.output_dir:
            self._output_dir = Path(config.output.output_dir)
        else:
            N = config.grid.resolution
            NLEV = config.grid.nlev
            N_DAYS = config.days
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = Path(f"results/amip/C{N}_L{NLEV}_{N_DAYS}d_{run_id}")

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def fric_decay(self) -> jax.Array:
        """Per-level Rayleigh-friction decay factors ``exp(-k_f dt)``.

        Computed by ``setup()`` (``_create_friction``). Public accessor so
        external training-segment builders (WB scale trainer) reuse the
        driver's boundary-layer damping instead of re-deriving it — an
        undamped training rollout produces NaN gradients (#797 bug 11).
        """
        return self._fric_decay

    # Backward-compatible accessors for individual tracers.
    @property
    def q_v(self) -> jax.Array:
        return self.tracers.get("q_v")

    @q_v.setter
    def q_v(self, value):
        self.tracers["q_v"] = value

    @property
    def q_c(self) -> jax.Array:
        return self.tracers.get("q_c")

    @q_c.setter
    def q_c(self, value):
        self.tracers["q_c"] = value

    @property
    def q_r(self) -> jax.Array:
        return self.tracers.get("q_r")

    @q_r.setter
    def q_r(self, value):
        self.tracers["q_r"] = value

    @property
    def q_i(self) -> jax.Array | None:
        return self.tracers.get("q_i")

    @q_i.setter
    def q_i(self, value):
        self.tracers["q_i"] = value

    @property
    def q_s(self) -> jax.Array | None:
        return self.tracers.get("q_s")

    @q_s.setter
    def q_s(self, value):
        self.tracers["q_s"] = value

    @property
    def q_g(self) -> jax.Array | None:
        return self.tracers.get("q_g")

    @q_g.setter
    def q_g(self, value):
        self.tracers["q_g"] = value

    # --- Double-moment hydrometeor plumbing (shared by the compiled-segment
    #     and per-step physics paths) -------------------------------------
    _DOUBLE_MOMENT_TRACERS = ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")

    def _double_moment_step_inputs(self) -> dict:
        """Current ice/snow/graupel + droplet/rain/ice-number columns to feed
        ``step_unified`` so coupled radiation gets number-aware r_eff and a
        double-moment microphysics evolves its full state. Empty (legacy
        warm-rain) unless the moisture registry carries these tracers."""
        if not isinstance(self.tracers, dict):
            return {}
        return {k: self.tracers[k] for k in self._DOUBLE_MOMENT_TRACERS
                if self.tracers.get(k) is not None}

    def _apply_double_moment_tendencies(self, phys_out, dt) -> None:
        """Integrate the ice/snow/graupel + number tracers one step from the
        matching ``PhysicsOutput`` tendencies, clipped non-negative. No-op
        unless the full-moisture registry is active (q_i present)."""
        if not (isinstance(self.tracers, dict)
                and self.tracer_registry.has("q_i")):
            return
        _upd = {
            "q_i": phys_out.dq_i_dt, "q_s": phys_out.dq_s_dt,
            "q_g": phys_out.dq_g_dt, "N_c": phys_out.dN_c_dt,
            "N_r": phys_out.dN_r_dt, "N_i": phys_out.dN_i_dt,
        }
        for k, tend in _upd.items():
            if self.tracers.get(k) is not None:
                self.tracers[k] = jnp.maximum(self.tracers[k] + dt * tend, 0.0)

    def _checkpoint_carry_aux(self) -> dict | None:
        """``self._carry_aux`` augmented with the evolved double-moment tracers
        (namespaced ``dmtr_*``) so a checkpoint persists them — otherwise a
        restart would silently reinitialize q_i/q_s/q_g/N_c/N_r/N_i to the
        setup zeros (the held-radiation carry_aux is round-tripped by the npz /
        per-rank backends, so this rides the same mechanism). No-op for
        warm-rain runs. Returns a NEW dict (does not mutate ``self._carry_aux``).
        Only valid for FULL-LOCAL state (serial / replicated / per-rank
        distributed); band-gathered lat-lon-MPI checkpoints must guard instead,
        since these tracers are rank-local bands (mirrors the T_land limitation).
        """
        base = dict(self._carry_aux) if self._carry_aux else {}
        dm = self._double_moment_step_inputs()
        for k, v in dm.items():
            base[f"dmtr_{k}"] = v
        # Tag the convection carry with its scheme (codex round 8): a
        # scheme change that keeps the carry SHAPE (mass_flux<->edmf
        # both carry (ncol,); the profile-prognostic schemes all carry
        # (ncol, nlev)) would otherwise silently feed one scheme's
        # memory to another on restart.  Stored as a plain string; the
        # npz loader passes it through un-cast.
        if "conv_prog" in base:
            base["conv_prog_scheme"] = np.asarray(
                str(getattr(self.config, "convection", "none")))
        # Multilayer (Richards) land state (#730 chain-enable): the prognostic
        # soil/snow/carbon columns ride carry_aux (namespaced land_ml_*) so a
        # chained C48 SOTA restart resumes the deep-soil spin-up instead of
        # cold-starting. No-op for slab-land runs (_land_ml_state is None).
        if self._land_ml_state is not None:
            for _f, _v in self._land_ml_state._asdict().items():
                # Optional fields (TgC, surface_water) may be None — np.asarray
                # would pickle a 0-d object array into the npz and crash the
                # load-side jnp.asarray. Skip; restore only replaces saved keys.
                if _v is not None:
                    base[f"land_ml_{_f}"] = np.asarray(_v)
        return base if base else None

    def _restore_dm_tracers_from_carry_aux(self) -> None:
        """Pop any ``dmtr_*`` entries restored into ``self._carry_aux`` back into
        the tracer dict, so a double-moment restart resumes the evolved ice /
        snow / graupel / number state rather than setup zeros."""
        _tracers = getattr(self, "tracers", None)
        if not (isinstance(self._carry_aux, dict)
                and isinstance(_tracers, dict)):
            return
        for key in [k for k in self._carry_aux if k.startswith("dmtr_")]:
            _tracers[key[len("dmtr_"):]] = self._carry_aux.pop(key)

    @staticmethod
    def _carry_has_unscatterable_land_ml(carry_aux) -> bool:
        """True iff ``carry_aux`` carries multilayer-land (``land_ml_*``) fields.

        Pure predicate behind the lat-lon band-MPI restart fail-fast guard: a
        gathered checkpoint's ``land_ml_*`` are the writer's GLOBAL soil/snow/
        carbon columns, which cannot be band-scattered (multilayer land is
        single-rank-only, #769).  Extracted so the guard — which only fires on
        the distributed restart path never reached by the single-process
        tests — is unit-testable directly; a future rename of the ``land_ml_``
        prefix then breaks the test loudly instead of silently disabling the
        guard.  Operates on the post-``bcast`` in-memory ``carry_aux`` whose
        keys are bare (the ``carry_``/``diag_`` serialization prefixes are
        stripped on load), so it matches with ``startswith`` not a substring."""
        return isinstance(carry_aux, dict) and any(
            k.startswith("land_ml_") for k in carry_aux)

    def _restore_land_ml_from_carry_aux(self) -> None:
        """Rebuild ``self._land_ml_state`` from any ``land_ml_*`` entries restored
        into ``carry_aux``, so a chained multilayer-land restart resumes the
        prognostic soil/snow/carbon columns instead of cold-starting (#730).
        No-op for slab-land runs (``_land_ml_state`` is None).

        Fail-loud on a partial or version-skewed checkpoint: the save side
        (:meth:`_checkpoint_carry_aux`) emits every non-None field, so the
        restored set must exactly match the current template's non-None field
        set.  A missing field would leave that prognostic column at its
        cold-start value — a silent mixed restart, exactly what #730 exists to
        prevent — and an unknown or wrong-shaped field signals a schema /
        resolution skew.  Mirrors the MPAS phys-state load, which likewise
        refuses a field-set mismatch rather than silently dropping columns."""
        if not isinstance(self._carry_aux, dict):
            return
        keys = [k for k in self._carry_aux if k.startswith("land_ml_")]
        if not keys:
            return
        # Pop the namespaced keys regardless of land type so a stray land_ml_*
        # (e.g. a slab run chained off a multilayer checkpoint) is never left to
        # leak forward into the next _checkpoint_carry_aux() re-save.
        popped = {k[len("land_ml_"):]: self._carry_aux.pop(k) for k in keys}
        template = self._land_ml_state
        if template is None:
            return  # slab run: keys removed above, nothing to restore
        import jax.numpy as jnp
        valid = set(template._fields)
        unknown = set(popped) - valid
        if unknown:
            raise ValueError(
                f"land_ml checkpoint has unknown field(s) {sorted(unknown)}; "
                f"current MultiLayerLandState fields are {sorted(valid)}")
        expected = {f for f in template._fields
                    if getattr(template, f) is not None}
        got = set(popped)
        if got != expected:
            raise ValueError(
                "land_ml checkpoint field set does not match the current "
                "MultiLayerLandState: "
                f"missing={sorted(expected - got)}, "
                f"unexpected={sorted(got - expected)}. Refusing to build a "
                "mixed restart state (the missing prognostic columns would "
                "silently stay at cold-start values).")
        fields = {}
        for name, val in popped.items():
            arr = jnp.asarray(val)
            ref = getattr(template, name)
            if arr.shape != ref.shape:
                raise ValueError(
                    f"land_ml checkpoint field '{name}' has shape "
                    f"{tuple(arr.shape)}, expected {tuple(ref.shape)} "
                    "(resolution / soil-layer-count skew)")
            fields[name] = arr
        self._land_ml_state = template._replace(**fields)

    def _validate_microphysics_tracer_state(
        self,
        *,
        context: str = "ModelDriver tracer state",
    ) -> int:
        """Validate that the live tracer dict can hold scheme tendencies."""
        tracers = getattr(self, "tracers", None)
        have_slots = 0
        if isinstance(tracers, dict):
            for name in self.tracer_registry.names:
                if tracers.get(name) is None:
                    break
                have_slots += 1
        return validate_microphysics_tracer_slots(
            self.config.microphysics,
            have_slots,
            context=context,
        )

    def _reject_shallow_water_unrunnable(self) -> None:
        """Shallow-water is not a runnable ModelDriver equation set.

        ``_init_state`` builds a hydrostatic primitive-equation state
        (``held_suarez_init`` / ``isothermal_rest_state_spectral``), never a
        shallow-water state, so a SW dycore would be handed a PE state and
        crash cryptically at the first step.  Reject LOUDLY at the public
        entry points (setup/run) and as an _init_state backstop (codex M2
        review).  The component factory still builds the correct
        ``FV3EdgeShallowWaterModel`` for component-registry / build-time use.
        """
        if self.config.dycore.model_type == "shallow_water":
            raise NotImplementedError(
                "shallow-water is not runnable via ModelDriver: it builds a "
                "hydrostatic primitive-equation state, not a shallow-water "
                "state.  Use `legoesm test williamson` or "
                "`scripts/matrix/run_atmosphere_test_matrix.py --only sw` "
                "(both construct the SW model + initial state directly).")

    def setup(self) -> None:
        """Initialize grid, dycore, physics, forcing, and state."""
        # SW is not a runnable ModelDriver equation set — reject before any
        # dycore/state construction so the failure is clear, not a downstream
        # scale-guard or shape crash (codex M2 review).
        self._reject_shallow_water_unrunnable()
        # Strict validation — abort early on invalid parameters
        self.config.validate_strict()

        # Bootstrap runtime: precision, backend, devices, and (optionally) MPI.
        # This is the canonical single entry point — handles everything before
        # any JAX array creation.
        self._bootstrap_runtime()

        # Config cross-validation
        config_warnings = self.config.validate()
        for w in config_warnings:
            logger.warning(f"  Config: {w}")

        # Only rank 0 creates output directory (or single-rank)
        if self._mpi_rank is None or self._mpi_rank == 0:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        self._create_grid()
        self._create_topography()
        self._create_dycore()
        self._create_forcing()
        self._init_state()
        self._create_ensemble()
        self._create_physics()
        self._setup_external_forcing()
        self._create_diagnostics()
        self._create_friction()
        # Manifest guard runs BEFORE _save_config: an invalid or different-config
        # existing manifest must abort setup *before* experiment_config.json is
        # (over)written, so the two run-start provenance files never disagree.
        self._write_run_manifest()
        self._save_config()
        self._setup_parallel()

    def static_topography_phis(self):
        """The model's STATIC surface geopotential ``phis = g·z_s`` on the model
        grid, built WITHOUT running the full :meth:`setup` — NO filesystem writes
        (no output dir, run manifest, or ``experiment_config.json``).

        Runs only the minimal construction chain ``validate_strict → runtime
        bootstrap → grid → topography`` needed to populate the topography field;
        it does NOT build the dycore, state, physics, or parallel halos.  This
        exposes the model's OWN orographic field cheaply — e.g. as the CONSISTENT
        ``phis`` source for the orographic LES-forcing term — so a launch-time
        probe need not write a phantom run directory.

        Idempotent: returns the already-built field if :meth:`setup` (or a prior
        call) ran.  A flat model yields ``jnp.zeros(grid_shape_2d)`` (the caller
        decides whether all-zero means "no orography").

        NOTE: not purely side-effect-free — ``_bootstrap_runtime`` sets the global
        precision/runtime singleton (the same one the subsequent run uses), so the
        probe MUST be built from the SAME config that drives the run.
        """
        if self._phis_data is None:
            self.config.validate_strict()
            self._bootstrap_runtime()
            self._create_grid()
            self._create_topography()
        return self._phis_data

    def static_land_fraction(self):
        """The model's STATIC land fraction on the model grid, built WITHOUT the full
        :meth:`setup` — the same minimal construction chain as
        :meth:`static_topography_phis` (``_create_topography`` populates BOTH ``phis``
        and ``f_land`` in every branch: flat → zeros, idealised → from topography, real
        → loaded, ``land_mask_path`` → from file).

        Exposes the model's OWN land/sea distribution cheaply so a launch-time probe can
        build an OCEAN-only ranking mask (:func:`legoesm.training.compare_reanalysis.
        ocean_valid_mask`) WITHOUT writing a phantom run directory.  Grid-shaped — flatten
        ROW-MAJOR to the worst-column column order.  An aquaplanet/flat model yields an
        all-zero field (all ocean).  Same runtime-singleton caveat as
        :meth:`static_topography_phis` (build from the SAME config that drives the run).
        """
        # static_topography_phis runs the chain that sets BOTH _phis_data and _f_land,
        # guarded on _phis_data; after it returns, _f_land is populated too.
        self.static_topography_phis()
        return self._f_land

    def _create_grid(self) -> None:
        """Create horizontal grid and vertical coordinate."""
        gc = self.config.grid

        # All global grids go through the one component-agnostic factory
        # (legoesm.grids.factory.create_grid) — the driver owns no grid
        # constructor dispatch of its own.  Two grid types need driver-local
        # handling the factory cannot do: ``latlon`` is sliced to this rank's
        # MPI band post-construction, and ``plane`` takes (nx, ny, nlev, dx,
        # dy) rather than a single resolution.
        from legoesm.grids.factory import create_grid

        if gc.grid_type == "latlon":
            global_grid = create_grid("latlon", gc.resolution)
            # Stage 3-B: under lat-lon band MPI, slice the global grid
            # to this rank's lat band.  The runtime bootstrap (Stage
            # 3-A) already activated set_halo_backend("mpi", layout)
            # via initialize_distributed_latlon — query that layout
            # via the standard ``get_mpi_topology`` accessor.
            #
            # The global grid is preserved as ``self._grid_global``
            # for downstream code that needs it (state init at the
            # global grid, output gather to rank 0 — Stage 3-D).
            from legoesm.grids.halo import get_mpi_topology
            from legoesm.parallel.latlon_mpi import (
                LatLonBandLayout, slice_latlon_grid_to_band,
            )
            _layout = get_mpi_topology()
            if (self.config.distributed
                    and isinstance(_layout, LatLonBandLayout)):
                self._grid_global = global_grid
                self.grid = slice_latlon_grid_to_band(global_grid, _layout)
                logger.info(
                    "  Lat-lon MPI: rank %d owns rows [%d:%d) of %d "
                    "(n_lat_local=%d, n_lon=%d).  total_area set to "
                    "global sphere area via allreduce.",
                    _layout.rank, _layout.lat_start, _layout.lat_end,
                    _layout.n_lat_global, _layout.n_lat_local,
                    _layout.n_lon_global,
                )
            else:
                self.grid = global_grid
        elif gc.grid_type == "plane":
            # PR2c MVP: square ``resolution x resolution`` doubly-
            # periodic plane with default ``dx = dy = 10 km``. Users
            # that need full control (non-square domains, finer dx,
            # Coriolis, custom ``lat0`` / ``lon0``) should build a
            # ``PlaneGrid`` via ``create_plane_grid`` directly and pass
            # the resulting model into the simulation harness instead
            # of going through ``ExperimentConfig``.
            from legoesm.grids.plane import create_plane_grid
            self.grid = create_plane_grid(
                nx=gc.resolution, ny=gc.resolution, nlev=gc.nlev,
                dx=10_000.0, dy=10_000.0,
            )
        else:
            # cubed_sphere / gaussian / mpas.  mpas keeps the driver's
            # 50-iteration Lloyd relaxation default; unknown grid types raise
            # ValueError inside create_grid (with the available list).  Legacy
            # mpas aliases (voronoi, icosahedral, mpas_voronoi) are normalised
            # to "mpas" at the config boundary (driver.config.normalize_grid_type).
            kwargs = {"lloyd_iterations": 50} if gc.grid_type == "mpas" else {}
            global_grid = create_grid(gc.grid_type, gc.resolution, **kwargs)
            # MPAS / Voronoi cell-partition MPI: partition the global mesh
            # and slice this rank's (owned + halo) local mesh.  Deferred to
            # here — not the runtime bootstrap — because the partition needs
            # the actual mesh (geometric / METIS graph partition + halo-ring
            # expansion).  The global mesh is preserved as ``self._grid_global``
            # for global-cell state init and the diagnostics / checkpoint
            # gather to rank 0 (mirrors the lat-lon band-MPI pattern, but the
            # slice is an unstructured owned-cell index set, not a lat band).
            if (gc.grid_type == "mpas" and self.config.distributed
                    and self._device_config is not None
                    and self._device_config.is_distributed):
                from legoesm.parallel.voronoi_mpi import initialize_voronoi_mpi
                rank, n_ranks, vlayout = initialize_voronoi_mpi(global_grid)
                self._grid_global = global_grid
                self._voronoi_layout = vlayout
                self._mpi_rank = rank
                self._mpi_world_size = n_ranks
                self.grid = vlayout.local_mesh
                logger.info(
                    "  MPAS MPI: rank %d/%d owns %d cells (+%d halo), "
                    "%d local edges of %d global",
                    rank, n_ranks,
                    vlayout.partition.n_owned_cells,
                    vlayout.partition.n_local_cells
                    - vlayout.partition.n_owned_cells,
                    vlayout.partition.n_local_edges,
                    vlayout.partition.nEdges_global,
                )
            else:
                self.grid = global_grid

        if gc.vertical_coord == "hybrid":
            from legoesm.grids.vertical import make_hybrid_levels
            self.sigma = make_hybrid_levels(
                gc.nlev, p_top_Pa=gc.p_top_Pa, stretching=gc.stretching,
            )
        else:
            from legoesm.grids.vertical import create_sigma_coordinate
            self.sigma = create_sigma_coordinate(gc.nlev)

        logger.info(f"  Grid: {gc.grid_type} {gc.resolution}, "
              f"{gc.nlev} levels ({gc.vertical_coord})")

        # Cache lat/lon accessors via GridProtocol for grid-agnostic use
        self._grid_lat = self.grid.grid_lat
        self._grid_lon = self.grid.grid_lon

    def _create_topography(self) -> None:
        """Load or generate topography and land-sea mask."""
        from legoesm.grids.topography import (
            TopographyConfig, load_real_topography,
            gaussian_mountain, phis_from_topography,
            land_mask_from_topography,
        )

        topo = self.config.topography
        shape_2d = self.grid.grid_shape_2d

        from legoesm.core.precision import get_policy
        _sd = get_policy().storage
        if topo == "flat":
            self._phis_data = jnp.zeros(shape_2d, dtype=_sd)
            self._f_land = jnp.zeros(shape_2d, dtype=_sd)
        elif topo == "gaussian":
            z_s = gaussian_mountain(self.grid)
            self._phis_data = phis_from_topography(z_s)
            self._f_land = land_mask_from_topography(z_s)
        else:
            topo_config = TopographyConfig(
                source="file", path=topo,
                smoothing_passes=self.config.topo_smoothing,
                edge_blend_strength=self.config.topo_edge_blend,
            )
            self._phis_data, self._f_land = load_real_topography(
                self.grid, config=topo_config
            )

        # Real land-sea mask overrides the elevation-derived land fraction
        # (works with any ``topography`` setting, including "flat").
        land_mask_path = getattr(self.config, "land_mask_path", "")
        if land_mask_path:
            from legoesm.grids.topography import load_land_fraction
            self._f_land = load_land_fraction(
                self.grid, land_mask_path
            ).astype(_sd)
            logger.info(
                f"  Land-sea mask: {land_mask_path} "
                f"(land fraction mean={float(jnp.mean(self._f_land)):.3f})"
            )

    def _create_dycore(self) -> None:
        """Create the dynamical core model via the component factory.

        The factory resolves ``(model_type, discretization, grid_type)``
        from :attr:`config` and instantiates the correct solver with
        physically derived diffusion coefficients.
        """
        from legoesm.driver.component_factory import (
            create_atmosphere_dycore, compute_diffusion,
        )

        self.model = create_atmosphere_dycore(self.config, self.grid, self.sigma)

        # Stage 3-B: under lat-lon band MPI the dycore model needs its
        # config's ``pole_v_bc`` flags set per this rank's pole-touch
        # state — only the boundary rank zeros the actual global pole
        # row; interior ranks leave their band-edge v-row alone (it's
        # an interior v-face shared with the neighbour rank).  Apply
        # the override after the factory built the model with default
        # serial flags ``(True, True)``.
        if self.config.distributed and self.config.grid.grid_type == "latlon":
            from legoesm.grids.halo import get_mpi_topology
            from legoesm.parallel.latlon_mpi import (
                LatLonBandLayout, pole_v_bc_for_layout,
            )
            _layout = get_mpi_topology()
            if isinstance(_layout, LatLonBandLayout):
                # Model.config is a CGridLatLonPrimitiveEquationConfig
                # NamedTuple — use ``_replace`` to set the rank-aware
                # flags.  ``pole_v_bc_offset`` stays 0 because the
                # backend-aware operator path operates on rank-local
                # (unpadded) state, not on a pre-padded array.
                self.model.config = self.model.config._replace(
                    pole_v_bc=pole_v_bc_for_layout(_layout),
                    pole_v_bc_offset=0,
                )
                logger.info(
                    "  Lat-lon MPI: pole_v_bc=%s on rank %d "
                    "(south_pole=%s, north_pole=%s).",
                    self.model.config.pole_v_bc, _layout.rank,
                    _layout.south_rank is None,
                    _layout.north_rank is None,
                )

        # The component factory may clamp dt for pole-cell CFL on lat-lon
        # grids.  Propagate the clamped value back into the driver config
        # so all downstream code (friction, radiation sub-cycling, etc.)
        # uses the actual timestep.
        _eff_dt = getattr(self.model, 'effective_dt', None)
        if _eff_dt is not None and _eff_dt < self.config.dycore.dt:
            logger.info(
                f"  Factory clamped dt from {self.config.dycore.dt:.1f}s "
                f"to {_eff_dt:.1f}s (pole-cell CFL)"
            )
            self.config = self.config._replace(
                dycore=self.config.dycore._replace(dt=_eff_dt),
            )

        # When conservation_fixer=False, propagate to fix_mass=False in
        # the driver config so the compiled-segment driver-level mass fixer
        # (fix_ps_mass_target in build_segment_fn) is also disabled.
        # The component factory already disables the dycore-internal fixer,
        # but the driver reads cfg.dycore.fix_mass independently.
        if not self.config.dycore.conservation_fixer and self.config.dycore.fix_mass:
            logger.info(
                "  conservation_fixer=False → disabling driver-level "
                "fix_mass to match dycore config"
            )
            self.config = self.config._replace(
                dycore=self.config.dycore._replace(fix_mass=False),
            )

        # Keep hyperdiffusion coefficient for moisture smoothing later.
        diff = compute_diffusion(self.grid, self.config.dycore)
        self._hyperdiff = diff.hyperdiff

        dc = self.config.dycore

        # --- CFL validation (uses cfl module, warns and adjusts if unsafe) ---
        from legoesm.core.cfl import cfl_check_and_adjust
        gc = self.config.grid
        model_type_map = {
            "shallow_water": "shallow_water",
            "hydrostatic": "primitive_eq",
            "nonhydrostatic": "compressible",
        }
        cfl_model = model_type_map.get(dc.model_type, "primitive_eq")
        dt_safe = cfl_check_and_adjust(
            dc.dt, gc.resolution, model_type=cfl_model,
            radius=getattr(self.grid, 'radius', constants.R_earth),
            grid_type=gc.grid_type,
            # Stage 3-E: when the polar filter is on, the equatorial
            # CFL is the actual stability limit (the filter handles
            # the pole CFL).  Without this kwarg the driver clamp
            # would override the factory clamp's polar-filter aware
            # value back down to ~5 s at 1° lat-lon.
            use_polar_filter=getattr(dc, "use_polar_filter", False),
        )
        if dt_safe < dc.dt:
            logger.warning(
                f"  CFL: reducing dt from {dc.dt:.0f}s to {dt_safe:.0f}s "
                f"for {gc.grid_type} C{gc.resolution}"
            )
            self.config = self.config._replace(
                dycore=dc._replace(dt=dt_safe),
            )
            dc = self.config.dycore

        logger.info(
            f"  Dycore: {dc.model_type}/{dc.discretization} on "
            f"{self.config.grid.grid_type}, dt={dc.dt}s"
        )

    def _create_forcing(self) -> None:
        """Load SST/SIC forcing data."""
        cfg = self.config

        # Under lat-lon band MPI ``_create_grid`` has already replaced
        # ``self.grid`` / ``self._grid_lat`` with this rank's band, but the
        # SST/SIC band-slicing wrapper installed later in ``_setup_parallel``
        # (``_band_get_sst_sic``) slices ``sst[lat_start:lat_end]`` of a
        # *global* field.  Building the forcing function on the rank-local
        # grid here would slice the band twice → an empty ``(0, n_lon)`` SST
        # on every non-zero rank (the radiation column adapter then fails to
        # reshape size 0 into the rank-local column count).  Build the forcing
        # on the preserved GLOBAL grid so the wrapper slices exactly once.
        # This applies ONLY to the lat-lon band path, which installs that
        # re-slicing wrapper.  MPAS cell-partition MPI also sets
        # ``_grid_global`` but has NO such wrapper — its state, physics, and
        # forcing all live on this rank's local (owned+halo) cells, so the
        # forcing must be built on the LOCAL mesh (``self.grid``) to match the
        # local state; using the global mesh there would yield an
        # ``(nCells_global,)`` SST that mismatches the local column count.
        # Serial and cubed-sphere paths have no ``_grid_global`` → local grid.
        _use_global_forcing = (
            getattr(self, "_grid_global", None) is not None
            and cfg.grid.grid_type == "latlon"
        )
        forcing_grid = self._grid_global if _use_global_forcing else self.grid
        forcing_lat = forcing_grid.grid_lat

        if cfg.dataset == "analytical":
            from legoesm.forcing.analytical import analytical_sst_sic
            lat_deg = np.degrees(np.asarray(forcing_lat))
            T_ice = cfg.T_ice

            def get_sst_sic(day):
                return analytical_sst_sic(lat_deg, day, T_ice=T_ice)

            self.get_sst_sic = get_sst_sic
        else:
            from legoesm.forcing.amip import (
                AMIPForcingConfig, get_amip_preset,
                load_amip_forcing, get_forcing_at_time,
            )
            if cfg.dataset == "custom":
                forcing_config = AMIPForcingConfig(
                    dataset="custom", path=cfg.forcing_path,
                    sst_var=cfg.sst_var or "sst",
                    sic_var=cfg.sic_var or "sic",
                    time_var=cfg.time_var or "time",
                    lat_var=cfg.lat_var or "lat",
                    lon_var=cfg.lon_var or "lon",
                    sst_offset=cfg.sst_offset, sic_scale=cfg.sic_scale,
                    sic_path=getattr(cfg, 'sic_path', ''),
                    # T_ice is the SST freezing floor (applied post-interp); wire
                    # the run's value so --t-ice-k reaches it (was left default).
                    T_ice=cfg.T_ice,
                )
            else:
                # Forward the run's SST/SIC unit conversions: run_amip defaults
                # these to the preset's own values (so a bare ``--dataset cobe``
                # keeps sic_scale=0.01), and an explicit --sic-scale/--sst-offset
                # overrides them — e.g. ``--sic-scale 0`` for a no-sea-ice run,
                # which the bare ``_replace(path, T_ice)`` used to silently drop.
                forcing_config = get_amip_preset(cfg.dataset)._replace(
                    path=cfg.forcing_path, T_ice=cfg.T_ice,
                    sst_offset=cfg.sst_offset, sic_scale=cfg.sic_scale,
                )

            # Anchor the SST/SIC time axis to the run's start year so a model
            # day indexes the file by real calendar date (AMIP-II): a 1979 run
            # reads the 1979 records of a 1870-2022 input4MIPs file, not 1870.
            forcing = load_amip_forcing(
                forcing_config, forcing_grid,
                start_year=getattr(cfg, "start_year", None),
            )
            self._forcing = forcing

            def get_sst_sic(day):
                return get_forcing_at_time(forcing, day)

            self.get_sst_sic = get_sst_sic

    def _apply_standard_atmosphere_ic(self) -> None:
        """Override the scaffold temperature with a realistic standard
        atmosphere (constant-lapse-rate troposphere + isothermal stratosphere +
        equator-pole surface gradient).  Grid-agnostic core; the per-grid work
        is only fetching the horizontal latitude field in the state's layout.

        The equator surface temperature is taken from ``config.T_init`` so the
        existing ``--t-init`` flag stays meaningful; the gradient, lapse rate
        and stratospheric floor use Earth-like defaults.
        """
        from legoesm.atmosphere.standard_atmosphere import (
            StandardAtmosphereConfig,
            standard_atmosphere_temperature,
            standard_atmosphere_zonal_wind,
        )

        cfg = self.config
        gt = cfg.grid.grid_type
        # Horizontal latitude in the state's native layout.  The realistic
        # T / p_s overlay below is GRID-AGNOSTIC (standard_atmosphere_temperature
        # accepts any lat shape); only the balanced zonal jet (step 5) is
        # lat-lon-specific (the A-grid u is geographic-east) and is skipped on
        # the cube.  spectral/MPAS are rejected in ExperimentConfig.validate_strict
        # (no grid-space T Field), so the else here is defensive.
        if gt == "latlon":
            lat_h = self.grid.lat2d                   # (n_lat, n_lon)
        elif gt == "cubed_sphere":
            lat_h = self.grid.lat                     # (6, n, n) geographic lat [rad]
        else:
            raise NotImplementedError(
                f"ic='standard' not yet wired for grid_type={gt!r} "
                f"(discretization={cfg.dycore.discretization!r}); 'latlon' and "
                "'cubed_sphere' are supported. Use ic='default' or 'era5'."
            )

        sa_cfg = StandardAtmosphereConfig(T_sfc_equator_K=cfg.T_init)
        lat_h = jnp.asarray(lat_h)

        # (1) Surface temperature (sigma=1 limit) — needed for the p_s reduction;
        #     independent of the vertical coordinate.
        T_sfc_std = standard_atmosphere_temperature(
            lat_h, jnp.ones((1,), dtype=self.sigma.sigma_full.dtype), sa_cfg,
        )[..., 0]

        # (2) Make surface pressure consistent with the NEW temperature over
        #     topography FIRST.  The scaffold reduced p_s against the uniform
        #     T_init column (p_s = p0*exp(-phis/(R_d*T_init))); rescale to the
        #     standard surface temperature, p_s = p0*exp(-phis/(R_d*T_sfc)),
        #     as a relative correction (exact no-op where phis == 0).
        phis = self.state.phis.data
        p_s_old = self.state.p_s.data
        p_s_new = p_s_old * jnp.exp(
            -phis / constants.R_d * (1.0 / T_sfc_std - 1.0 / cfg.T_init)
        )
        self.state = self.state._replace(
            p_s=self.state.p_s.replace(data=p_s_new.astype(p_s_old.dtype)),
        )

        # (3) Vertical coordinate: use sigma_full — the SAME pressure convention
        #     the production physics/radiation/saturation pipeline uses
        #     (p_full = p_s * sigma_full; physics_pipeline.py, compiled_segments).
        #     The model treats sigma_full (= A_full + B_full on the hybrid
        #     coordinate) as the effective level coordinate everywhere, so the IC
        #     MUST match it: initializing T/q on the "true" hybrid pressure
        #     (A*p_ref + B*p_s) while the physics evaluates on p_s*sigma_full
        #     would hand the first radiation/convection/saturation step a column
        #     on a different pressure grid (spurious condensation over terrain).
        #     The topography-adjusted p_s above is what makes the columns
        #     physical; the level coordinate stays consistent with downstream.
        sigma_full = self.sigma.sigma_full

        # (4) Temperature.
        T_new = standard_atmosphere_temperature(
            lat_h, sigma_full, sa_cfg,
        ).astype(self.state.T.data.dtype)
        self.state = self.state._replace(T=self.state.T.replace(data=T_new))

        # (5) Thermal-wind-balanced zonal wind so the imposed equator-pole
        #     temperature gradient does not launch a geostrophic-adjustment shock
        #     at startup.  v stays zero (the balance is zonal).  Reuses the grid's
        #     own radius/rotation (constants fallback per the audit rule).
        #
        #     LAT-LON ONLY: the A-grid u is geographic-east, so the balanced jet
        #     is assigned directly.  On the cubed-sphere u/v are cube-LOCAL
        #     components, so a geographic-east jet would need a per-cell
        #     grid-angle rotation; a coupled CLIMATE spin-up grows its own
        #     circulation from the realistic T gradient within a few days (damped
        #     by hyperdiffusion), so the balanced-jet IC is not required (unlike a
        #     baroclinic-wave test).  Keep the scaffold winds on the cube.
        if gt == "latlon":
            radius = getattr(self.grid, "radius", constants.R_earth)
            omega = getattr(self.grid, "omega", constants.Omega)
            u_new = standard_atmosphere_zonal_wind(
                lat_h, sigma_full, radius, omega, sa_cfg,
            ).astype(self.state.u.data.dtype)
            self.state = self.state._replace(
                u=self.state.u.replace(data=jnp.broadcast_to(u_new, self.state.u.data.shape)),
            )
        else:
            logger.info(
                "  ic='standard' on %s: applied realistic T + p_s; balanced "
                "zonal jet skipped (cube-local winds need grid-angle rotation) "
                "— circulation spins up from the T gradient.", gt,
            )

    def _init_state(self) -> None:
        """Initialize atmospheric state and moisture."""
        from legoesm.diagnostics.column_integrals import column_water_vapor

        cfg = self.config
        # Backstop: SW is not a runnable ModelDriver equation set (the public
        # setup()/run() entries reject it first; this covers a direct
        # _init_state() call).  See _reject_shallow_water_unrunnable.
        self._reject_shallow_water_unrunnable()
        N = cfg.grid.resolution
        NLEV = cfg.grid.nlev

        if cfg.grid.grid_type == "mpas":
            from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
            shape_3d = (self.grid.nCells, NLEV)
            self.state = held_suarez_init_mpas(
                self.grid, self.sigma, T_init=cfg.T_init,
            )
            if jnp.any(self._phis_data != 0):
                self.state = self.state._replace(
                    phis=self.state.phis.replace(data=self._phis_data),
                )
        elif cfg.dycore.discretization == "spectral":
            from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
            shape_3d = (self.grid.n_lat, self.grid.n_lon, NLEV)
            phis_arg = self._phis_data if jnp.any(self._phis_data != 0) else None
            self.state = isothermal_rest_state_spectral(
                self.grid, self.sigma, T_init=cfg.T_init, phis=phis_arg,
            )
        else:
            if cfg.grid.grid_type == "cubed_sphere":
                from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
                shape_3d = (6, N, N, NLEV)
                self.state = held_suarez_init(
                    self.grid, self.sigma, T_init=cfg.T_init, phis=self._phis_data
                )
            else:
                # Lat-lon and Gaussian grids use (n_lat, n_lon, nlev) layout.
                # Pass phis so p_s is hydrostatically reduced over topography
                # (p_s = p_ref*exp(-phis/(R_d*T_init))) — matching the
                # cubed-sphere branch above.  Previously phis was patched in
                # *after* construction, leaving p_s flat over terrain; that is
                # the reference state the ic='standard' p_s recompute corrects
                # relative to, and is also more correct for ic='default'.
                from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
                shape_3d = (self.grid.n_lat, self.grid.n_lon, NLEV)
                self.state = held_suarez_init_latlon(
                    self.grid, self.sigma, T_init=cfg.T_init,
                    phis=self._phis_data,
                )

        # Physically-realistic "standard atmosphere" override: replace the
        # uniform-T_init scaffold temperature with a constant-lapse-rate
        # troposphere + isothermal stratosphere + equator-pole gradient, BEFORE
        # the moisture init below so the q_v column integral comes out Earth-like
        # (~15-30 kg/m^2) instead of ~80 kg/m^2.  ERA5 IC (handled later) takes
        # precedence and fully overwrites the state.
        if cfg.ic == "standard":
            self._apply_standard_atmosphere_ic()

        # Initialize all tracers via registry
        self.tracers = init_tracers(self.tracer_registry, shape_3d)
        self._validate_microphysics_tracer_state(
            context="ModelDriver initialized tracer state",
        )

        # Moisture initialization (spectral and MPAS use dry physics)
        if hasattr(self.state, 'p_s') and hasattr(self.state.p_s, 'data'):
            # Build p_full as p_s * sigma_full — the SAME convention the
            # production physics/radiation/saturation pipeline uses
            # (physics_pipeline.py, compiled_segments.py).  Initializing q_sat
            # and the vertical humidity taper on this grid keeps the moisture
            # consistent with the temperature state AND with the first physics
            # step, so a topography+hybrid run does not start supersaturated on a
            # mismatched pressure grid.  (Do NOT switch to pressure_at_full here
            # unless the whole physics pipeline is migrated to it too.)
            p_full_init = self.state.p_s.data[..., None] * self.sigma.sigma_full
            q_sat_init = saturation_mixing_ratio(self.state.T.data, p_full_init)
            self.tracers["q_v"] = cfg.rh_init * q_sat_init * self.sigma.sigma_full ** 2
            self.tracers["q_v"] = jnp.minimum(self.tracers["q_v"], q_sat_init)

            # Fuse the two diagnostic means into one host transfer.
            _stats = jnp.stack([
                jnp.mean(self.tracers["q_v"]),
                jnp.mean(column_water_vapor(
                    self.tracers["q_v"], self.state.p_s.data,
                    self.sigma.dsigma,
                )),
            ])
            _h = np.asarray(_stats)
            mean_qv = float(_h[0]) * 1000.0
            cwv = float(_h[1])
            logger.info(f"  State init: T={cfg.T_init}K, q_v={mean_qv:.2f} g/kg, CWV={cwv:.1f} kg/m2")
        else:
            logger.info(f"  State init: T={cfg.T_init}K (dry spectral)")

        # MPAS Phase B: attach the moisture tracers to the dycore STATE so the
        # primitive-equation step advects them (mass-consistently, via the
        # shared tracer kernel + the dycore's own wind/mass-flux) and the column
        # physics (convection/microphysics) can read + update them.  Gated on
        # moist physics being active, so a dry radiative-dynamical MPAS run
        # (convection=microphysics=turbulence=none) keeps tracers=None and is
        # byte-for-byte unchanged (e.g. the 100-yr gray radiative AMIP).
        if cfg.grid.grid_type == "mpas":
            _moist = (cfg.microphysics != "none" or cfg.convection != "none"
                      or cfg.turbulence != "none")
            if _moist:
                from legoesm.core.field import Field
                self.state = self.state._replace(tracers={
                    k: Field(data=v, name=k, dims=("nCells", "nlev"),
                             units="kg/kg")
                    for k, v in self.tracers.items()
                })
                logger.info(
                    f"  MPAS moisture ON: dycore advects "
                    f"{sorted(self.tracers.keys())} (q_v mean "
                    f"{float(jnp.mean(self.tracers['q_v'])) * 1000:.2f} g/kg)"
                )

        # Spectral moist AMIP (CMIP6 deck on gaussian/spectral): the
        # spectral state keeps prognostics as SH coefficients, so the
        # generic ``hasattr(state, 'p_s')`` moisture init above never
        # fires and ``self.tracers['q_v']`` is still zeros here.
        # Initialize q_v on the Gaussian grid from the (transformed)
        # T/p_s rest state with the SAME rh_init * q_sat * sigma**2
        # taper as the gridpoint paths, then attach the tracers to the
        # dycore state so the spectral PE step advects them and the
        # combined physics (convection/microphysics) can read + update
        # them.  Gated on moist physics — a dry spectral run (e.g. the
        # AIMIP gray path) keeps tracers=None, byte-for-byte unchanged.
        if cfg.dycore.discretization == "spectral":
            _moist = (cfg.microphysics != "none" or cfg.convection != "none"
                      or cfg.turbulence != "none")
            if _moist:
                from legoesm.core.field import Field
                from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
                    spectral_pe_to_grid,
                )
                _f0 = spectral_pe_to_grid(self.state, self.grid, self.sigma)
                _p_full0 = _f0['p_s'][..., None] * self.sigma.sigma_full
                _q_sat0 = saturation_mixing_ratio(_f0['T'], _p_full0)
                _q_v0 = jnp.minimum(
                    cfg.rh_init * _q_sat0 * self.sigma.sigma_full ** 2,
                    _q_sat0,
                )
                self.tracers["q_v"] = _q_v0
                self.state = self.state._replace(tracers={
                    k: Field(data=v, name=k,
                             dims=("n_lat", "n_lon", "nlev"), units="kg/kg")
                    for k, v in self.tracers.items()
                })
                _cwv0 = float(jnp.mean(column_water_vapor(
                    _q_v0, _f0['p_s'], self.sigma.dsigma)))
                logger.info(
                    f"  Spectral moisture ON: dycore advects "
                    f"{sorted(self.tracers.keys())} (q_v mean "
                    f"{float(jnp.mean(_q_v0)) * 1000:.2f} g/kg, "
                    f"CWV={_cwv0:.1f} kg/m2)"
                )

        # ERA5 IC override — replace held-suarez rest state with ERA5 reanalysis.
        # Applied after the default moisture init so the Field metadata (dims,
        # units, names) from held_suarez_init is preserved as the template.
        if cfg.ic == "era5" and cfg.ic_path:
            from legoesm.training.era5_to_state import (
                load_era5_ic,
                era5_to_cubedsphere_carry,
                era5_to_spectral_carry,
                era5_to_latlon_carry,
                era5_to_mpas_carry,
            )
            logger.info(
                f"  IC: loading ERA5 from {cfg.ic_path} "
                f"(year={cfg.start_year})"
            )
            era5_slice = load_era5_ic(cfg.ic_path, cfg.start_year)

            if cfg.grid.grid_type == "cubed_sphere":
                _target_phis = (
                    self._phis_data
                    if self._phis_data is not None and jnp.any(self._phis_data != 0)
                    else None
                )
                carry = era5_to_cubedsphere_carry(
                    era5_slice, self.grid, self.sigma,
                    target_phis=_target_phis,
                )
            elif cfg.dycore.discretization == "spectral":
                carry = era5_to_spectral_carry(
                    era5_slice, self.grid, self.sigma
                )
            elif cfg.grid.grid_type == "latlon":
                carry = era5_to_latlon_carry(
                    era5_slice, self.grid, self.sigma
                )
            elif cfg.grid.grid_type == "mpas":
                # MPAS carries the wind as the edge-normal component on mesh
                # edges (no cell-centred v); era5_to_mpas_carry regrids ERA5
                # to cells/edges and projects the winds via angleEdge.
                carry = era5_to_mpas_carry(
                    era5_slice, self.grid, self.sigma,
                    smoothing_passes=cfg.topo_smoothing,
                )
            else:
                raise NotImplementedError(
                    f"ERA5 IC not yet supported for "
                    f"grid_type={cfg.grid.grid_type!r} / "
                    f"discretization={cfg.dycore.discretization!r}. "
                    "Supported: cubed_sphere, latlon, mpas, gaussian/spectral."
                )

            if cfg.dycore.discretization == "spectral":
                # The spectral state holds prognostics as SH coefficients,
                # not grid-point Fields.  Forward-transform the grid-space
                # ERA5 carry: (u, v) -> (vor_hat, div_hat) via the validated
                # ``vordiv_from_uv_3d`` (inverse of the dycore's
                # ``uv_from_vordiv_3d``); T/ln(p_s)/phis via ``sh_analysis``.
                from legoesm.grids.gaussian import (
                    vordiv_from_uv_3d, sh_analysis_3d, sh_analysis,
                )
                vor_hat, div_hat = vordiv_from_uv_3d(
                    self.grid, carry.u, carry.v
                )
                T_hat = sh_analysis_3d(self.grid, carry.T)
                lnps_hat = sh_analysis(self.grid, jnp.log(carry.p_s))
                phis_hat = sh_analysis(self.grid, carry.phis)
                self.state = self.state._replace(
                    vor_hat=self.state.vor_hat.replace(data=vor_hat),
                    div_hat=self.state.div_hat.replace(data=div_hat),
                    T_hat=self.state.T_hat.replace(data=T_hat),
                    lnps_hat=self.state.lnps_hat.replace(data=lnps_hat),
                    phis_hat=self.state.phis_hat.replace(data=phis_hat),
                )
                self.tracers["q_v"] = jnp.asarray(carry.q_v)
                # Re-attach the ERA5 q_v into the dycore-advected tracer
                # state (the spectral moist-init block above seeded a
                # rest-state q_v; override with ERA5).
                if (self.state.tracers is not None
                        and "q_v" in self.state.tracers):
                    self.state = self.state._replace(tracers={
                        **self.state.tracers,
                        "q_v": self.state.tracers["q_v"].replace(
                            data=jnp.asarray(carry.q_v)),
                    })
                # Stats from the grid-space reconstruction.
                from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
                    spectral_pe_to_grid,
                )
                _fg = spectral_pe_to_grid(self.state, self.grid, self.sigma)
                _stats_era5 = jnp.stack([
                    jnp.mean(self.tracers["q_v"]),
                    jnp.mean(column_water_vapor(
                        self.tracers["q_v"], _fg['p_s'], self.sigma.dsigma,
                    )),
                    jnp.mean(_fg['T']),
                ])
            elif cfg.grid.grid_type == "mpas":
                # MPAS state: edge-normal u (nEdges, nlev), no v; scalars at
                # cells.  q_v lives in the dycore-advected tracer dict.
                self.state = self.state._replace(
                    u=self.state.u.replace(data=carry.u),
                    T=self.state.T.replace(data=carry.T),
                    p_s=self.state.p_s.replace(data=carry.p_s),
                    phis=self.state.phis.replace(data=carry.phis),
                )
                self.tracers["q_v"] = jnp.asarray(carry.q_v)
                _stats_era5 = jnp.stack([
                    jnp.mean(self.tracers["q_v"]),
                    jnp.mean(column_water_vapor(
                        self.tracers["q_v"], self.state.p_s.data,
                        self.sigma.dsigma,
                    )),
                    jnp.mean(self.state.T.data),
                ])
            else:
                self.state = self.state._replace(
                    u=self.state.u.replace(data=carry.u),
                    v=self.state.v.replace(data=carry.v),
                    T=self.state.T.replace(data=carry.T),
                    p_s=self.state.p_s.replace(data=carry.p_s),
                    phis=self.state.phis.replace(data=carry.phis),
                )
                self.tracers["q_v"] = jnp.asarray(carry.q_v)

                _stats_era5 = jnp.stack([
                    jnp.mean(self.tracers["q_v"]),
                    jnp.mean(column_water_vapor(
                        self.tracers["q_v"], self.state.p_s.data,
                        self.sigma.dsigma,
                    )),
                    jnp.mean(self.state.T.data),
                ])
            _h2 = np.asarray(_stats_era5)
            logger.info(
                f"  State init (ERA5 {cfg.start_year}): "
                f"T_mean={_h2[2]:.1f}K, "
                f"q_v={_h2[0]*1000:.2f} g/kg, "
                f"CWV={_h2[1]:.1f} kg/m2"
            )

            # ERA5 phis is kept for dynamics — substituting ETOPO phis at t=0
            # while keeping ERA5 winds creates a pressure-gradient imbalance
            # (local delta can reach ~15 % p_s over Tibet/Andes) that drives
            # a gravity wave transient exceeding CFL within the first day.
            # ETOPO is used for CMOR orog and for f_land (passive surface tile);
            # the dynamics run with the ERA5 orography that is consistent with
            # the ERA5 initial wind field.  Proper ETOPO-in-dynamics requires
            # building a balanced IC on ETOPO from the outset (future work).

    def _create_ensemble(self) -> None:
        """Create ensemble members if ensemble_size > 1.

        Perturbs the initial state and tracers to create *ensemble_size*
        members.  Each leaf array gains a leading ensemble dimension:
        ``(n_members, ...)``.  Diagnostics later use ``ensemble_mean``
        before collecting.
        """
        self._ensemble_size = self.config.ensemble_size
        # Save a single-member template for unpack_carry during ensemble runs
        self._state_template = self.state
        if self._ensemble_size <= 1:
            return

        from legoesm.parallel.ensemble import perturb_initial_conditions
        from legoesm.runtime.rng import split_key

        # Central RNG: derive the ensemble-IC key from the run's master seed
        # (config.seed) rather than a hardcoded magic seed, so the perturbation
        # is reproducible from the seed recorded in the run manifest.
        key = split_key(self.config.seed, "ensemble_ic")
        self.state = perturb_initial_conditions(
            self.state, key, self._ensemble_size, scale=0.01,
        )
        # Tile tracers: each tracer (S,...) -> (n_members, S,...)
        for name, arr in self.tracers.items():
            if arr is not None:
                self.tracers[name] = jnp.broadcast_to(
                    arr[None], (self._ensemble_size,) + arr.shape
                ).copy()  # copy so each member can diverge

        logger.info(f"  Ensemble: {self._ensemble_size} members (IC perturbation scale=0.01)")

    def _create_physics(self) -> None:
        """Build the physics pipeline."""
        self.physics = build_physics_pipeline(self.grid, self.sigma, self.config)
        rad_str = self.config.radiation or "none"
        conv_str = self.config.convection or "none"
        logger.info(f"  Physics: radiation={rad_str}, convection={conv_str}")

        # Activate land surface when f_land was loaded in _create_topography.
        # Two modes, distinguished by whether an explicit --land-mask-file was
        # given (full slab, slab_land_active=True) or only --topography was
        # given (passive: albedo + T_sfc blend, T_land carried but not stepped).
        _has_land = (
            self._f_land is not None
            and bool(jnp.any(self._f_land > 0))
        )
        if _has_land:
            from legoesm.surface_albedo import land_vegetation_albedo
            from legoesm.core.precision import get_policy
            _sd = get_policy().storage
            self.physics.f_land = self._f_land.astype(_sd)
            # Land albedo, in precedence order:
            #   1. ``albedo_land_path`` — a static satellite NetCDF (ICON-extpar
            #      ALB); ``albedo_land_month`` (1-12) picks a month, 0 -> annual
            #      mean (main behaviour, unchanged / byte-identical when set).
            #   2. ``surfdata_path`` — harmonized legoesm_surfdata (per-column
            #      soil-colour + PFT-vegetation blend + glacier override).
            #   3. latitude-vegetation default (ocean/uncovered fallback).
            # The two file paths are alternative products; ``albedo_land_path``
            # wins when both are set (direct satellite albedo over the derived
            # blend).  ``lat_albedo`` is always the uncovered/fallback field.
            _alb_path = getattr(self.config, "albedo_land_path", "")
            surfdata_path = getattr(self.config, "surfdata_path", "")
            lat_albedo = land_vegetation_albedo(self.grid.grid_lat)
            if _alb_path:
                from legoesm.grids.topography import load_land_albedo
                _alb_month = getattr(self.config, "albedo_land_month", 0) or None
                self.physics.albedo_land = load_land_albedo(
                    self.grid, _alb_path, month=_alb_month
                ).astype(_sd)
                logger.info(
                    f"  Land albedo: {_alb_path} "
                    f"(month={_alb_month or 'annual mean'}, "
                    f"mean={float(jnp.mean(self.physics.albedo_land)):.3f})"
                )
            elif surfdata_path:
                self.physics.albedo_land = self._surfdata_land_albedo(
                    surfdata_path, lat_albedo
                ).astype(_sd)
            else:
                self.physics.albedo_land = lat_albedo.astype(_sd)
            # Tiled (mosaic) surface fluxes + the radiation cadence apply to ANY
            # active land tile — slab OR multilayer (Richards).  Thread them at the
            # _has_land level so use_multilayer_land (topography-derived f_land, no
            # slab, no mask) actually runs the tiled turbulent-flux path: the flux
            # injection in physics_pipeline is gated on physics.surface_tiled, which
            # was previously only set inside the slab/mask branch below -> multilayer
            # validated but silently no-op'd the tiled surface.  Slab-specific bucket/
            # stomata/snow stay in the slab-activation branch.
            self.physics.rad_update_steps = self.config.rad_update_steps
            # ocean bulk scheme on the ocean tile, land Monin-Obukhov on the land
            # tile (validate_strict requires louis + an active land tile).
            self.physics.surface_tiled = bool(
                getattr(self.config, "surface_tiled", False)
            )
            self.physics.surface_z0_land = float(
                getattr(self.config, "surface_z0_land", 0.1)
            )
            _activate = bool(getattr(self.config, "land_mask_path", "")) or bool(
                getattr(self.config, "slab_land_active", False)
            )
            if _activate:
                self.physics.slab_land_active = True
                # Prognostic soil-water bucket: soil-moisture-limited land
                # evaporation (beta) instead of a saturated wet surface.
                _bucket = bool(getattr(self.config, "land_soil_bucket", False))
                self.physics.land_soil_bucket = _bucket
                self.physics.land_bucket_w_max = float(
                    getattr(self.config, "land_bucket_w_max", 150.0)
                )
                self.physics.land_beta_min = float(
                    getattr(self.config, "land_beta_min", 0.1)
                )
                self.physics.land_bucket_w_init_frac = float(
                    getattr(self.config, "land_bucket_w_init_frac", 0.5)
                )
                # Bucket runoff partition (Green-Ampt infiltration + saturation excess)
                self.physics.land_K_infiltration = float(
                    getattr(self.config, "land_K_infiltration", 1.0e-5)
                )
                self.physics.land_infil_suction_boost = float(
                    getattr(self.config, "land_infil_suction_boost", 2.0)
                )
                self.physics.land_infiltration_excess = bool(
                    getattr(self.config, "land_infiltration_excess", True)
                )
                # Stomatal soil-water limitation: route beta_soil through the
                # shared land Jarvis model (legoesm.land.stomata).
                _stomatal = bool(getattr(self.config, "land_stomatal_beta", False))
                self.physics.land_stomatal_beta = _stomatal
                if _stomatal:
                    from legoesm.land.stomata import StomataConfig
                    self.physics.stomata_config = StomataConfig(
                        gs_max=self.config.land_gs_max,
                    )
                # Prognostic snow + snow-albedo feedback on the slab tile.
                self.physics.snow_albedo_feedback = bool(
                    getattr(self.config, "snow_albedo_feedback", False))
                logger.info(
                    f"  Land tile: ACTIVE (slab land, C_land="
                    f"{self.physics.C_land:.1e} J/m2/K, "
                    f"f_land mean={float(jnp.mean(self._f_land)):.3f}, "
                    f"tiled_surface={self.physics.surface_tiled}"
                    + (f", z0_land={self.physics.surface_z0_land:g}m"
                       if self.physics.surface_tiled else "")
                    + (f", soil_bucket(W_max={self.physics.land_bucket_w_max:g}"
                       f" kg/m2, beta_min={self.physics.land_beta_min:g})"
                       if _bucket else "")
                    + (", stomatal_beta(Jarvis)" if _stomatal else "")
                    + ")"
                )
            else:
                logger.info(
                    f"  Land tile: PASSIVE (albedo + T_sfc blend, "
                    f"f_land mean={float(jnp.mean(self._f_land)):.3f}, "
                    "T_land carried but not stepped)"
                )

            # MULTILAYER (Richards) override: when use_multilayer_land, the
            # differentiable segment advances a per-column MultiLayerLandState
            # (SegmentCarry.land_ml) in place of the scalar slab T_land, using the
            # faithful CLM default (PFT veg params + reference-soil thermal/hydro).
            # Reuses the land mask (f_land) loaded above; the slab knobs become
            # unused (the land/ocean blend reads the multilayer surface T+albedo).
            if getattr(self.config, "use_multilayer_land", False):
                self._setup_multilayer_land(_sd)

    def _setup_multilayer_land(self, storage_dtype) -> None:
        """Activate the differentiable multilayer (Richards) land tile.

        Loads the CLM reference surface map onto the model columns, builds the
        per-column ``LandSurfaceParams`` + ``MultiLayerLandConfig`` (faithful CLM
        default via :func:`clm_multilayer_setup`), wires them onto the physics
        pipeline (``land_ml_*`` attrs consumed by ``compute_radiation_core``), and
        seeds an initial ``MultiLayerLandState`` warm-started from the near-surface
        air temperature (no tropical cold spin-up).  The state then rides
        ``SegmentCarry.land_ml`` and persists across segments (see the run loop).

        Raises if the CLM surfdata cannot be mapped — ``use_multilayer_land`` must
        NOT silently degrade to the slab (that would run different land physics
        silently, the issue-#405 bug class)."""
        import numpy as _np
        from legoesm.land import init_multilayer_land_state, aridity_theta_init
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.land.clm_surface_map import (
            load_clm_surface, download_clm_surfdata, clm_multilayer_setup,
        )
        from legoesm.land.stomata import StomataConfig
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.soil_grid import SoilGridConfig
        from legoesm.land.surface_scheme import (
            SimpleSEBConfig, TwoLeafCanopyConfig,
        )
        # Land surface-scheme dispatch (#730). "simple_seb" (default) = bulk SEB
        # with the beta_soil moisture path; "two_leaf" = DifferBESS two-leaf canopy
        # (Kelvin h_r bare-soil + two-leaf stomatal transpiration), which limits
        # land ET below potential and breaks the over-evaporation wet loop.
        _scheme_name = self.config.land_surface_scheme
        if _scheme_name == "simple_seb":
            _surface_scheme = SimpleSEBConfig()
        elif _scheme_name == "two_leaf":
            _surface_scheme = TwoLeafCanopyConfig()
        else:
            raise ValueError(
                f"Unknown land_surface_scheme {_scheme_name!r}; "
                "expected 'simple_seb' or 'two_leaf'."
            )

        ad = self.physics.adapter
        # column-order latitude / longitude in RADIANS, flattened to (ncol,).
        # Cubed-sphere stores per-cell (6,n,n) lat/lon; the LAT-LON grid stores
        # 1-D axes (lat (n_lat,), lon (n_lon,)) — flatten_2d on those raised
        # "cannot reshape (n_lat,) into ncol" and killed every latlon
        # use_multilayer_land run at setup (#837 follow-up / #869 lane).
        # Prefer the grid's own 2-D fields when present, else broadcast the
        # 1-D axes to the (n_lat, n_lon) cell grid (same convention as
        # run_lmip_smoke.grid_latlon_rad).  The CLM map regrids onto DEGREE
        # coordinates; the land tile consumes radians.
        _glat = _np.asarray(getattr(self.grid, "lat2d", self.grid.lat))
        _glon = _np.asarray(getattr(self.grid, "lon2d", self.grid.lon))
        if _glat.ndim == 1 and _glon.ndim == 1:
            _glon, _glat = _np.meshgrid(_glon, _glat)   # -> (n_lat, n_lon)
        lat_rad = _np.asarray(ad.flatten_2d(_glat)).reshape(-1)
        lon_rad = _np.asarray(ad.flatten_2d(_glon)).reshape(-1)
        lat_deg = _np.degrees(lat_rad)
        lon_deg = _np.degrees(lon_rad)
        # download_clm_surfdata caches to /tmp (one-time); load_clm_surface regrids
        # the CLM reference surfdata onto the model columns.  A pre-staged path
        # (``clm_surfdata_path``) is required on compute nodes with no outbound
        # internet (the default fetches from UCAR, which fails on such nodes).
        surfdata_path = self.config.clm_surfdata_path
        if surfdata_path:
            # A staged path is set: use it, and FAIL LOUDLY if absent rather than
            # attempting a (compute-node-blocked) network fetch to that exact path.
            if not os.path.exists(surfdata_path):
                raise FileNotFoundError(
                    f"clm_surfdata_path={surfdata_path!r} does not exist; stage the "
                    "CLM surfdata NetCDF there (compute nodes have no outbound "
                    "internet to download it).")
            surfdata_file = surfdata_path
        else:
            surfdata_file = download_clm_surfdata()
        surface_map = load_clm_surface(surfdata_file, lat_deg, lon_deg)

        # Non-spatial defaults from the config; clm_multilayer_setup overwrites only
        # hydraulics/thermal.  Thread the active snow-albedo + stomata features the
        # slab land already uses (snow_albedo_feedback, land_stomatal_beta) so
        # ``use_multilayer_land`` is a strict UPGRADE (adds Richards multilayer soil
        # + CLM texture/PFT maps) rather than a partial regression to a
        # no-snow-albedo / no-stomata surface.  The soil SEB uses land MOST (with
        # the CLM per-cell z0) to MATCH the atmospheric tiled LAND tile, which
        # hard-codes land "most"; surface_bulk_scheme (coare3/large_yeager) is the
        # OCEAN surface-layer scheme and must NOT drive the land skin-T (it would
        # evolve T_sfc with ocean-roughness logic, inconsistent with the atmosphere
        # land-flux exchange law — the T_sfc now feeds T_land, so consistency here
        # matters).
        base = MultiLayerLandConfig(
            soil_grid=SoilGridConfig(
                n_layers=self.config.multilayer_n_layers,
                total_depth=self.config.multilayer_soil_depth,
            ),
            bulk_scheme="most",
            snow_albedo_feedback=self.config.snow_albedo_feedback,
            surface_scheme=_surface_scheme,
            stomata=StomataConfig(
                enabled=self.config.land_stomatal_beta,
                gs_max=self.config.land_gs_max,
            ),
        )
        params, cfg = clm_multilayer_setup(surface_map, base_config=base)

        self.physics.land_ml_cfg = cfg
        self.physics.land_ml_params = params
        self.physics.land_ml_lat = jnp.asarray(lat_rad, dtype=storage_dtype)
        self.physics.land_ml_doy = 0.0

        # Transient land-use cover (LULC): load the annual cover series onto the SAME
        # model columns (identical _nearest_regrid targets => cell-for-cell aligned
        # with surface_map) and build the per-year vegetation-param rebuild (soil /
        # LAI frozen).  ``params`` above is the year=None fallback baked on the
        # pipeline; the run loop re-weights per segment via
        # ``_transient_land_ml_params`` and passes the result as a TRACED
        # SegmentForcing arg so the jitted step reads the evolving cover (not the
        # closure-baked attribute).  ``include_soil_albedo=True`` mirrors
        # clm_multilayer_setup (which weights the soil-colour albedo), so the
        # rebuild at the base cover reproduces ``params`` exactly.
        self._land_cover_transient = None
        if getattr(self.config, "transient_land_cover", False):
            from legoesm.land.clm_surface_map import (
                load_transient_cover_on_columns, clm_provider_rebuild,
            )
            _cover, _years = load_transient_cover_on_columns(
                self.config.land_cover_surfdata, lat_deg, lon_deg)
            _rebuild = clm_provider_rebuild(
                surface_map, variant="multilayer", include_soil_albedo=True)
            self._land_cover_transient = (_cover, _years, _rebuild)
            logger.info(
                "  Land tile: TRANSIENT cover ACTIVE (%d cover years, %d..%d) "
                "from %s", int(_years.shape[0]), int(_years[0]), int(_years[-1]),
                self.config.land_cover_surfdata,
            )

        ncol = lat_deg.shape[0]
        T_init = ad.flatten_2d(self.state.T.data[..., -1]).reshape(-1).astype(
            storage_dtype)
        # Aridity-aware cold-start soil moisture (#730 / #837).  Seed theta from
        # the near-surface RH of the IC atmosphere, mapped into the per-column
        # plant-available range [theta_wp, theta_fc] the tile's beta reads
        # (params.theta_wp/theta_fc from clm_multilayer_setup).  The legacy
        # moisture-uniform 0.5*theta_sat seed leaves subtropical deserts
        # rainforest-wet, so a hot bare-soil skin drives a runaway
        # potential-evaporation blowup (~day 8).  RH here uses the model's own
        # saturation_mixing_ratio -- the SAME law the land bulk flux uses -- so it
        # is consistent with the running physics (q_v is the atmospheric lowest
        # level; p_s is a close proxy for the lowest-level pressure).  Fall back to
        # the frac*theta_sat uniform seed (``land_soil_moisture_init_frac``) only
        # when the IC carries no q_v tracer (the aridity map needs RH).
        _qv = self.q_v  # canonical tracer store: raw (...,nlev) array, same column
        # layout as self.state.T.data; populated by both the analytical and ERA5 IC.
        if _qv is not None:
            q_v_low = ad.flatten_2d(
                getattr(_qv, "data", _qv)[..., -1]).reshape(-1).astype(storage_dtype)
            p_s = ad.flatten_2d(
                getattr(self.state.p_s, "data", self.state.p_s)).reshape(-1).astype(
                    storage_dtype)
            rh_low = q_v_low / jnp.maximum(
                saturation_mixing_ratio(T_init, p_s), 1e-12)
            theta_wp = jnp.asarray(getattr(params, "theta_wp", cfg.theta_wp))
            theta_fc = jnp.asarray(getattr(params, "theta_fc", cfg.theta_fc))
            theta_init = aridity_theta_init(
                rh_low, theta_wp, theta_fc).reshape(-1, 1).astype(storage_dtype)
        else:
            theta_init = (self.config.land_soil_moisture_init_frac
                          * cfg.hydraulics.theta_sat)
            logger.warning(
                "  Land tile: IC has no q_v tracer; multilayer soil seeded at "
                "land_soil_moisture_init_frac*theta_sat (aridity-aware skipped).")
        # Canonical cold-start template — always built so the spun-up land-IC
        # path (below) has the correct pytree STRUCTURE to graft onto (the
        # restart round-trips only the core prognostic fields; the optional
        # structural fields would otherwise be None and break the segment scan).
        _template = init_multilayer_land_state(
            ncol, cfg, T_init=T_init, theta_init=theta_init)
        _land_ic_path = getattr(self.config, "land_ic_path", "")
        if _land_ic_path:
            # #746 item 1: a spun-up land IC (offline run_land_spinup restart)
            # REPLACES the cold-start soil column with an equilibrated one, so
            # the coupled run doesn't start from the day-0 cold-start shock that
            # drives the land cloud-albedo cold trap.  Shapes are validated
            # against this run's grid (ncol / n_layers) on load; a mismatch or a
            # slab-mode restart raises rather than silently reshaping.
            from legoesm.land.restart import (
                load_land_restart, merge_land_restart_into_template)
            _ic_state, _ic_meta = load_land_restart(
                _land_ic_path, expected_land_mode="multilayer",
                expected_ncol=ncol, expected_n_layers=cfg.soil_grid.n_layers)
            # Graft the restart's prognostic columns onto the canonical template
            # (fixes the pytree structure), then cast the array leaves to the
            # run's storage precision (the restart deserialises float64).
            _merged = merge_land_restart_into_template(_ic_state, _template)
            self._land_ml_state = jax.tree_util.tree_map(
                lambda a: (a.astype(storage_dtype)
                           if hasattr(a, "dtype")
                           and jnp.issubdtype(a.dtype, jnp.floating) else a),
                _merged)
            logger.info(
                "  Land tile: MULTILAYER IC from spin-up restart %s "
                "(%d soil layers, %d columns; t_end=%.0f s, cold-start seed "
                "SKIPPED)",
                _land_ic_path, cfg.soil_grid.n_layers, ncol,
                float(_ic_meta.get("t_end_s", 0.0)),
            )
        else:
            self._land_ml_state = _template
            logger.info(
                "  Land tile: MULTILAYER override ACTIVE (%d soil layers, %d columns)",
                cfg.soil_grid.n_layers, ncol,
            )

    def _transient_land_ml_params(self, day: float):
        """Per-segment multilayer ``LandSurfaceParams`` at the segment's calendar
        year, or ``None`` when transient cover is inactive.

        ``cover_year = start_year + day/365`` selects the annual cover
        (``interp_annual``, clamped to the series endpoints), which re-weights the
        PFT-dependent vegetation params (soil/LAI frozen).  Host-side + concrete
        (cheap, like ``_precompute_external_forcing``); the returned pytree has a
        STABLE structure across segments, so feeding it as a traced ``SegmentForcing``
        leaf changes only leaf *values* — no retrace.  ``None`` (transient cover off)
        lets the jitted step fall back to the closure-baked
        ``pipeline.land_ml_params`` (byte-identical static path)."""
        _trans = getattr(self, "_land_cover_transient", None)
        if _trans is None:
            return None
        from legoesm.land.global_surface_data import interp_annual
        cover, years, rebuild = _trans
        cover_year = float(self._start_year) + day / 365.0
        fracs = interp_annual(cover, years, jnp.asarray(cover_year))
        return rebuild(fracs)()

    def _surfdata_land_albedo(self, surfdata_path: str, lat_albedo):
        """Static land albedo field from harmonized surface data.

        Loads + regrids the surfdata to ``self.grid`` (once, host-side), adapts it
        to a slab SimpleSEB land config, fills mask-land gaps with bare soil, and
        maps the per-column ``albedo_veg`` (soil-colour + PFT-vegetation blend with
        a glacier override) onto the model grid.  Cells the driver considers ocean
        (``f_land == 0``) keep ``lat_albedo`` — physically irrelevant there (the
        radiation step blends land albedo by ``f_land``) but keeps the field finite
        and smooth everywhere.

        Static snapshot at ``config.start_day`` day-of-year and the run's
        ``config.start_year`` cover slice (a transient LUH2/HYDE/... surfdata is
        sampled at the start year, not collapsed to a nonsensical multi-century
        year-mean).  Within-run seasonal-LAI / soil-wetness / transient-cover
        evolution of the albedo is a follow-up (it would rebuild ``land_params``
        per segment, as the GHG hook already does for ``current_year``).
        """
        from legoesm.land.config import LandConfig
        from legoesm.land.boundary_data import (
            init_land_surface_data, fill_land_param_gaps,
        )

        # LandConfig defaults to SimpleSEBConfig -> LandSurfaceParams with albedo_veg.
        land_cfg = LandConfig()
        # Sample transient cover at the run start year; a config without start_year
        # (or a static single-year surfdata) falls back to the legacy year-mean.
        _start_year = getattr(self.config, "start_year", None)
        _, land_params, gsd = init_land_surface_data(
            surfdata_path, self.grid, land_cfg, float(self.config.start_day),
            year=None if _start_year is None else float(_start_year),
        )
        # Reconcile to the driver's AUTHORITATIVE land mask (not surfdata's own
        # cover): surfdata properties are kept only where _f_land > 0, so the
        # land params never disagree with the ocean tile (weighted by 1-f_land)
        # or preexisting AMIP runs.  Ravel matches the loader's column order
        # since grid_shape_2d == grid_lat.shape (verified for latlon/gaussian/
        # cubed-sphere).
        land_params = fill_land_param_gaps(
            land_params, gsd, f_land=jnp.asarray(self._f_land).reshape(-1),
        )

        albedo_col = jnp.asarray(land_params.albedo_veg)          # (ncol,)
        grid_shape = tuple(int(s) for s in jnp.asarray(self.grid.grid_lat).shape)
        n_grid = int(np.prod(grid_shape))
        if albedo_col.shape[0] != n_grid:
            raise ValueError(
                f"surfdata albedo has {albedo_col.shape[0]} columns but grid "
                f"{type(self.grid).__name__} has {n_grid} (shape {grid_shape}); "
                "column/grid layout mismatch."
            )
        # Loader flattens grid.grid_lat in C-order (_target_latlon_flat -> ravel),
        # so reshape aligns cell-for-cell with grid_lat / physics.albedo_land.
        surf_albedo = albedo_col.reshape(grid_shape)
        is_land = jnp.asarray(self._f_land) > 0.0
        on_land = jnp.where(is_land, surf_albedo, jnp.nan)
        logger.info(
            f"  Land albedo: SURFDATA ({surfdata_path}); "
            f"{int(jnp.sum(is_land))} land cells, on-land albedo range "
            f"[{float(jnp.nanmin(on_land)):.3f}, {float(jnp.nanmax(on_land)):.3f}]"
        )
        return jnp.where(is_land, surf_albedo, lat_albedo)

    def _setup_external_forcing(self) -> None:
        """Configure external forcing: solar, ozone, aerosol, GHG."""
        from legoesm.forcing.external import (
            SolarConfig, OzoneConfig, AerosolConfig, GHGConfig,
            get_solar_forcing_at_time,
            get_ghg_at_time,
        )

        cfg = self.config
        self._solar_config = SolarConfig(
            S_0=cfg.S_0, source=cfg.solar_source,
            path=cfg.solar_file, tsi_var=cfg.solar_tsi_var,
            spectral_var=cfg.solar_spectral_var,
            spectral_band_order=getattr(
                cfg, "solar_spectral_band_order", "auto",
            ),
            start_year=cfg.start_year,
        )
        self._use_solar_spectral = (cfg.solar_source == "spectral_file")

        # Ozone external forcing
        self._ozone_ext_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                                  and cfg.ozone_forcing == "external")
        self._ozone_ext_config = OzoneConfig(
            enabled=self._ozone_ext_active,
            source="climatology", path=cfg.ozone_file,
            use_reference_if_missing=True,
            # Required for the non-cyclic branch in ``get_ozone_at_time``
            # so that CMIP6 multi-year ozone files (>12 months, e.g. the
            # UReading 1850–2014 vmro3 file) are sampled at the actual
            # simulation calendar year instead of falling back to a
            # 1850 climatology.
            start_year=cfg.start_year,
        )

        # Aerosol external forcing
        self._aerosol_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                                and cfg.aerosol_forcing == "external")
        # Volcanic stratospheric LONGWAVE aerosol (gap #9): active only on a
        # gas-radiation scheme (gray ignores aerosol) with the LW switch and
        # a volcanic file present.  Default OFF ⇒ no LW aerosol path.
        self._aerosol_lw_active = (
            cfg.radiation in ("rrtmg", "rrtmgp")
            and bool(cfg.volcanic_aerosol_lw)
            and bool(cfg.volcanic_aerosol_file)
        )
        self._aerosol_lw_od = None
        self._aerosol_config = AerosolConfig(
            enabled=self._aerosol_active,
            source="climatology", path=cfg.aerosol_file,
            use_reference_if_missing=True,
            reference_aod_550=cfg.aerosol_reference_aod,
            volcanic_enabled=bool(cfg.volcanic_aerosol_file),
            volcanic_path=cfg.volcanic_aerosol_file,
            volcanic_scale=cfg.volcanic_aerosol_scale,
            # Volcanic stratospheric LONGWAVE aerosol (gap #9): only when
            # the LW switch is set AND a volcanic file is present.  Default
            # OFF ⇒ ``get_aerosol_lw_at_time`` returns None ⇒ zeros LW od
            # ⇒ byte-identical (RRTMGP no-op).
            volcanic_lw_enabled=(bool(cfg.volcanic_aerosol_lw)
                                 and bool(cfg.volcanic_aerosol_file)),
            # Calendar anchor for the non-cyclic dispatch in
            # ``get_aerosol_at_time``.  Multi-year volcanic time-series
            # (e.g. 1850–2014 CMIP6 ``bc_aeropt_cmip6_volc_*``) are
            # sampled at their actual eruption calendars instead of
            # being collapsed onto a 12-month cycle.
            start_year=cfg.start_year,
        )

        # Solar init
        solar_init = get_solar_forcing_at_time(self._solar_config, cfg.start_day)
        if self._use_solar_spectral:
            self._solar_weights_template = jnp.asarray(
                solar_init["solar_fraction_by_gpt"]
            )
        else:
            self._solar_weights_template = jnp.array([], dtype=jnp.float32)

        # CMIP experiment GHG override
        self._experiment = cfg.experiment
        self._start_year = cfg.start_year
        if self._experiment:
            from legoesm.forcing.experiments import ghg_at_year
            co2, ch4, n2o = ghg_at_year(self._experiment, self._start_year)
            self.config = cfg._replace(co2_ppmv=co2, ch4_ppbv=ch4, n2o_ppbv=n2o)
            cfg = self.config
            logger.info(f"  CMIP: {self._experiment} (year {self._start_year}), "
                  f"CO2={co2:.1f} ppmv")

        # GHG forcing config
        self._ghg_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                            and cfg.ghg_forcing == "external")
        if self._ghg_active:
            self._ghg_config = GHGConfig(
                co2_ppmv=cfg.co2_ppmv,
                ch4_ppbv=cfg.ch4_ppbv,
                n2o_ppbv=cfg.n2o_ppbv,
                source="annual_file",
                path=cfg.ghg_file,
                start_year=cfg.start_year,
            )
            # Log initial GHG values
            ghg_init = get_ghg_at_time(self._ghg_config, cfg.start_day)
            logger.info(
                f"  GHG external: CO2={ghg_init['co2_ppmv']:.1f}ppmv, "
                f"CH4={ghg_init['ch4_ppbv']:.0f}ppbv, "
                f"N2O={ghg_init['n2o_ppbv']:.1f}ppbv, "
                f"CFC11={ghg_init['cfc11_pptv']:.0f}pptv, "
                f"CFC12={ghg_init['cfc12_pptv']:.0f}pptv"
            )
        else:
            self._ghg_config = GHGConfig(
                co2_ppmv=cfg.co2_ppmv,
                ch4_ppbv=cfg.ch4_ppbv,
                n2o_ppbv=cfg.n2o_ppbv,
                source="constant",
            )

    def _owned_p_s_and_lat(self):
        """Return (p_s, lat) for physics — rank-local if MPI, global otherwise."""
        if self._owned_face_ids is not None:
            p_s = self.state.p_s.data[self._owned_face_ids]
            lat = self._physics_lat
        else:
            p_s = self.state.p_s.data
            lat = self._grid_lat
            # Multi-controller SPMD: ``p_s`` is GLOBALLY sharded across
            # processes, but the external-forcing consumers downstream
            # (``get_ozone_at_time`` & co) are host/NumPy interpolators —
            # ``np.asarray`` on a process-spanning array raises "spans
            # non-addressable devices".  Gather to a process-local replicated
            # array (collective; every call site runs on all processes —
            # context build + per-segment forcing refresh).  No-op for
            # fully-addressable arrays, so serial / single-GPU / mpi4jax
            # lanes are byte-unchanged.  First hit by the 2-node Levante
            # receipt run (#693): the CPU parity smokes run gray radiation
            # with no external forcing and never reach this path.
            p_s = self._gather_spmd_tree_to_host(p_s)
        return p_s, lat

    def _precompute_external_forcing(self, day, p_s, lat):
        """Pre-compute ozone/aerosol/GHG fields outside JIT boundary."""
        from legoesm.forcing.external import (
            get_ozone_at_time, get_aerosol_at_time, get_aerosol_lw_at_time,
            get_ghg_at_time, ghg_concentrations_to_vmr,
        )
        from legoesm.forcing.surface_utils import distribute_column_aod_to_layers

        nlev = self.sigma.sigma_full.shape[0]
        shape_2d = p_s.shape
        ncol = int(np.prod(np.array(shape_2d)))

        p_full = p_s[..., None] * self.sigma.sigma_full
        p_half = p_s[..., None] * self.sigma.sigma_half
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        lat_col = lat.reshape(ncol)

        if self._ozone_ext_active:
            o3_vmr = jnp.asarray(get_ozone_at_time(
                self._ozone_ext_config, day,
                lat_grid=lat_col, p_grid=p_full_col,
            ))
        else:
            # External ozone forcing inactive — fall back to the
            # climatological ozone-VMR column (US Std Atm 1976 fit) that
            # RRTMGP uses internally when ``o3_vmr=None``.  Previously
            # ``o3_vmr`` was initialised to zeros and threaded into the
            # solver, which then clipped it to 1e-10 and effectively
            # disabled stratospheric ozone heating (audit 2026-05-12
            # HIGH #3).  Building the profile here keeps the radiation
            # JIT signature stable (it always sees a real array) while
            # still producing physical stratospheric heating in
            # gray/RRTMGP runs without an external ozone file.
            from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
                standard_o3_profile,
            )
            o3_vmr = standard_o3_profile(p_full_col).astype(p_s.dtype)

        aerosol_od = jnp.zeros((ncol, nlev), dtype=p_s.dtype)
        if self._aerosol_active:
            aerosol_col = get_aerosol_at_time(
                self._aerosol_config, day, lat_grid=lat_col,
            )
            aerosol_od = distribute_column_aod_to_layers(
                jnp.asarray(aerosol_col), p_half_col,
            )

        # Volcanic stratospheric LONGWAVE aerosol (gap #9): same (ncol,
        # nlev) shape as ``aerosol_od``, default zeros so the SegmentForcing
        # / forcing-dict leaf is a concrete fixed-shape array (no retrace)
        # and a run without volcanic LW aerosol is byte-identical (zeros LW
        # od is a RRTMGP no-op).  Distributed to layers by the SAME
        # pressure-thickness helper used for the SW aerosol.  Stored as an
        # instance attribute (NOT added to the 3-tuple return) so the five
        # existing unpack call sites keep their arity.
        aerosol_lw_od = jnp.zeros((ncol, nlev), dtype=p_s.dtype)
        if self._aerosol_lw_active:
            aerosol_lw_col = get_aerosol_lw_at_time(
                self._aerosol_config, day, lat_grid=lat_col,
            )
            if aerosol_lw_col is not None:
                aerosol_lw_od = distribute_column_aod_to_layers(
                    jnp.asarray(aerosol_lw_col), p_half_col,
                )
        self._aerosol_lw_od = aerosol_lw_od

        # GHG VMR override (None for gray radiation / constant forcing)
        ghg_vmr = None
        if self._ghg_active:
            ghg_conc = get_ghg_at_time(self._ghg_config, day)
            ghg_vmr = ghg_concentrations_to_vmr(ghg_conc)

        # CMIP experiment transient GHG override — applies even when
        # ghg_forcing is "constant" so that built-in experiment
        # trajectories (historical, SSP, 1pctCO2) drive radiation.
        cfg = self.config
        if (self._experiment
                and cfg.radiation in ("rrtmg", "rrtmgp")
                and ghg_vmr is None):
            from legoesm.forcing.experiments import (
                ghg_at_year, EXPERIMENT_TEMPLATES,
            )
            tmpl = EXPERIMENT_TEMPLATES.get(self._experiment)
            if tmpl is not None and tmpl.forcing_type == "transient":
                current_year = self._start_year + day / 365.0
                co2, ch4, n2o = ghg_at_year(self._experiment, current_year)
                ghg_vmr = ghg_concentrations_to_vmr({
                    "co2_ppmv": co2,
                    "ch4_ppbv": ch4,
                    "n2o_ppbv": n2o,
                })

        # Interactive carbon-radiation coupling (#3 / C4MIP): when the coupled
        # driver runs a prognostic CO2 tracer it sets ``self._co2_vmr_override``
        # (a global-mean CO2 mole fraction) each segment; inject it into the
        # radiation GHG so emitted / absorbed CO2 actually changes radiative
        # forcing.  Gated on a gas-radiation scheme (gray ignores GHG) and on the
        # override being present, so fixed-CO2 runs are byte-identical.
        co2_vmr_override = getattr(self, "_co2_vmr_override", None)
        if (co2_vmr_override is not None
                and cfg.radiation in ("rrtmg", "rrtmgp")):
            ghg_vmr = {**(ghg_vmr or {}), "co2": co2_vmr_override}

        return o3_vmr, aerosol_od, ghg_vmr

    def _create_diagnostics(self) -> None:
        """Set up diagnostic collection."""
        # Cloud config for the total-cloud-cover (clt) diagnostic.  Built via
        # the SAME shared ``build_cloud_config`` the physics pipeline uses, from
        # the same ``ExperimentConfig`` fields (``cloud_scheme`` +
        # ``cloud_rh_crit`` / ``cloud_q_c_diagnostic``), so clt derives from the
        # model's OWN large-scale/stratiform cloud fraction (issue #689).
        #
        # The opt-in ``convective_cloud`` add-on is deliberately NOT counted in
        # clt: it is a bounded (cap 0.15) radiative-TUNING term combined by
        # maximum overlap, not a physical cloud-AREA fraction, and matching it
        # faithfully would require radiation's lagged, per-rank ``conv_precip``
        # carry (not worth threading for a diagnostic — codex review).  Where a
        # column actually convects the RH-based stratiform fraction is already
        # high, so excluding the add-on changes clt little.  clt is therefore
        # the model's stratiform cloud cover with maximum-random overlap.
        # ``cloud_scheme == 'none'`` => ``None`` => clt not published.
        _cloud_scheme = getattr(self.config, "cloud_scheme", "none")
        diag_cloud_config = None
        if _cloud_scheme != "none":
            from legoesm.atmosphere.physics.clouds.config import (
                build_cloud_config,
            )
            diag_cloud_config = build_cloud_config(
                _cloud_scheme,
                convective_cloud=False,  # stratiform-only clt (see above)
                rh_crit=getattr(self.config, "cloud_rh_crit", None),
                q_c_diagnostic=getattr(self.config, "cloud_q_c_diagnostic", None),
            )
        self.diagnostics = DiagnosticCollector(
            nlev=self.config.grid.nlev,
            sigma_full=self.sigma.sigma_full,
            dsigma=self.sigma.dsigma,
            experiment_id=self.config.experiment or "amip",
            monthly_means=self.config.output.monthly_means,
            cmip_output=self.config.output.cmip_output,
            clear_sky_diag=self.config.output.clear_sky_diag,
            n_days=self.config.days,
            output_dir=self._output_dir,
            cmip_resolution_deg=self.config.output.cmip_resolution_deg,
            start_year=self.config.start_year,
            cloud_config=diag_cloud_config,
        )
        # Register per-cell horizontal areas so every global-mean diagnostic
        # (<R_TOA>, <SST>, <CWV>, ...) is area-weighted.  On a lat-lon grid an
        # unweighted ``jnp.mean`` over-weights the polar rows (each cell counts
        # equally despite spanning ~cos(lat) less area), which biased <rsdt> to
        # ~281 W/m² and faked a -36 W/m² "cold drift" where the area-weighted
        # TOA budget is near balance.  ``grid_area`` is grid-agnostic (lat-lon
        # (n_lat,n_lon), cube (6,n,n)); grids without it keep the plain mean.
        self.diagnostics.set_area_weights(getattr(self.grid, "grid_area", None))
        # Configure CMIP spatial regridding weights
        if self.config.output.cmip_output:
            self.diagnostics.set_cmip_grid_info(
                grid_type=self.config.grid.grid_type,
                grid=self.grid,
                start_year=self.config.start_year,
            )
            # Register time-invariant fields for the CMIP6 ``fx`` file.
            # _phis_data is the ETOPO field; dynamics run with ERA5 phis but
            # CMOR orog reports the ETOPO field (the intended mountain mask).
            self.diagnostics.set_fixed_fields(
                phis=np.asarray(self._phis_data),
                land_fraction=np.asarray(self._f_land),
            )

    def _sync_and_collect_diagnostics(self, **kwargs) -> dict:
        """Synchronize device computation and collect diagnostics.

        When running in multi-device mode, gathers the state from all
        devices before diagnostic collection.  Consolidates the
        ``jax.block_until_ready`` + ``diagnostics.collect`` pattern
        into a single method to avoid scattered sync points.

        In ``perf_mode``, uses ``collect_lightweight()`` which computes
        only scalar reductions (``jnp.mean``, ``jnp.max``) directly on
        the sharded arrays — no full-state gather, no host materialization
        via ``np.asarray()``.  JAX handles the cross-device reductions
        internally for SPMD-sharded arrays, so global means and max-wind
        are still correct.  Snapshots, profiles, and monthly means are
        skipped.
        """
        perf_mode = kwargs.pop("perf_mode", None)
        if perf_mode is None:
            # Auto-detect: use perf_mode when distributed to avoid
            # expensive allgather on every diagnostic interval.
            pm_setting = getattr(
                self.config,
                'output', None,
            )
            pm_flag = getattr(pm_setting, 'diagnostics_perf_mode', 'auto')
            if pm_flag == "always":
                perf_mode = True
            elif pm_flag == "never":
                perf_mode = False
            elif pm_flag == "auto":
                # auto: use perf_mode when running distributed MPI, and
                # ALWAYS under multi-controller SPMD (cs_spmd step 5b):
                # collect_lightweight's jnp reductions are SPMD-global on
                # the face-sharded arrays and their replicated scalar
                # results are fully addressable on every process, while
                # the full collect() would np.asarray non-fully-
                # addressable global arrays (crash).
                perf_mode = (
                    (self._device_config is not None
                     and self._device_config.is_distributed)
                    or self._is_spmd_multiprocess()
                )
            else:
                raise ValueError(
                    "diagnostics_perf_mode must be one of "
                    f"('auto', 'always', 'never'), got {pm_flag!r}"
                )

        # CMIP output requires full collect() for spatial/monthly
        # accumulation — perf mode skips those, producing zero files.
        cmip_on = self.config.output.cmip_output
        if cmip_on and perf_mode:
            perf_mode = False

        # Lat-lon band MPI: each rank holds its own latitude band as an
        # ordinary (non-SPMD-sharded) array, so ``collect_lightweight``'s
        # ``jnp.mean`` would reduce over this rank's band only — a wrong
        # "global" mean — and skip the spatial gather, so snapshots/profiles
        # would capture a single band.  Force the full gather+collect path
        # (correct global means, profiles, and snapshots).  The gather runs
        # only at the diagnostic cadence, negligible against the steps
        # between intervals.  Distinguished from cubed-sphere replicated MPI
        # by ``_owned_face_ids is None`` (the lat-lon path never sets it).
        _latlon_mpi = (
            self._layout is not None
            and self._owned_face_ids is None
            and self._device_config is not None
            and self._device_config.is_distributed
            and self.config.grid.grid_type == "latlon"
        )
        if _latlon_mpi and perf_mode:
            perf_mode = False

        # Multi-controller SPMD full collect (cmip_output or an explicit
        # diagnostics_perf_mode='never'/'auto'-overridden request): gather
        # the sharded state + every array kwarg to a host replica on EVERY
        # process (process_allgather is collective), then run the standard
        # single-process collect on each process — identical host inputs
        # give identical accumulators on every rank, and the flush/save
        # sites are already root-gated via _mpi_rank=process_index.
        if not perf_mode and self._is_spmd_multiprocess():
            state = kwargs.get('state', self.state)
            jax.block_until_ready(state.u.data)
            kwargs['state'] = self._gather_spmd_tree_to_host(state)
            for _k, _v in list(kwargs.items()):
                if _k != 'state' and isinstance(_v, jax.Array):
                    kwargs[_k] = self._gather_spmd_tree_to_host(_v)
            return self.diagnostics.collect(**kwargs)

        if perf_mode:
            # Lightweight path: scalar reductions only, no gather.
            state = kwargs.get('state', self.state)
            jax.block_until_ready(state.u.data)
            return self.diagnostics.collect_lightweight(
                elapsed_day=kwargs.get('elapsed_day', 0.0),
                state=state,
                q_v=kwargs.get('q_v', None),
                sst=kwargs.get('sst', None),
                sic=kwargs.get('sic', None),
                precip_total=kwargs.get('precip_total', None),
                sw_up_toa=kwargs.get('sw_up_toa', None),
                lw_up_toa=kwargs.get('lw_up_toa', None),
                sw_net_sfc=kwargs.get('sw_net_sfc', None),
                lw_net_sfc=kwargs.get('lw_net_sfc', None),
            )

        if self._device_config is not None:
            if self._device_config.is_distributed:
                # MPI replicated dynamics: state is full (6, n, n) on each
                # rank but non-owned faces are stale.  Gather owned-face
                # data from all ranks into a correct global state on rank 0.
                state = kwargs.get('state', self.state)
                jax.block_until_ready(state.u.data)
                if _latlon_mpi:
                    # Lat-lon band MPI: concatenate each rank's latitude
                    # band into the global field on rank 0.  The
                    # HydrostaticState stores u/v/T cell-centred (identical
                    # shapes — confirmed: v is NOT a face array here, unlike
                    # the raw C-grid dynamics state the checkpoint path
                    # gathers with the v-face trim), so every field uses the
                    # same scalar concatenation.  All ranks must participate
                    # in each MPI gather; non-root ranks then bail with the
                    # sentinel the run loop ignores.
                    from legoesm.parallel.latlon_mpi import gather_field_latlon
                    from legoesm.core.field import Field
                    from legoesm.core.state import HydrostaticState

                    def _g(arr):
                        if arr is None:
                            return None
                        return gather_field_latlon(arr, self._layout)

                    u_g = _g(state.u.data)
                    v_g = _g(state.v.data)
                    T_g = _g(state.T.data)
                    ps_g = _g(state.p_s.data)
                    phis_g = _g(state.phis.data)
                    for _tname in (
                        'q_v', 'q_c', 'q_r', 'q_i', 'q_s', 'q_g',
                        'sst', 'sic', 'precip_total',
                        'sw_up_toa', 'lw_up_toa', 'sw_net_sfc', 'lw_net_sfc',
                        'sw_down_toa', 'shflx', 'lhflx', 'lat_deg_grid',
                        't_low_mean', 'sw_up_toa_clr', 'lw_up_toa_clr',
                    ):
                        if kwargs.get(_tname) is not None:
                            kwargs[_tname] = _g(kwargs[_tname])

                    if self._mpi_rank != 0:
                        return {'mean_T': 0.0, 'max_v': 0.0}

                    gathered = HydrostaticState(
                        u=Field(u_g, name="u", dims=state.u.dims, units=state.u.units),
                        v=Field(v_g, name="v", dims=state.v.dims, units=state.v.units),
                        T=Field(T_g, name="T", dims=state.T.dims, units=state.T.units),
                        p_s=Field(ps_g, name="p_s", dims=state.p_s.dims, units=state.p_s.units),
                        phis=Field(phis_g, name="phis", dims=state.phis.dims, units=state.phis.units),
                    )
                    kwargs['state'] = gathered
                    return self.diagnostics.collect(**kwargs)
                if self._owned_face_ids is not None and self._layout is not None:
                    from legoesm.parallel.layout import gather
                    from legoesm.core.field import Field
                    from legoesm.core.state import HydrostaticState
                    # Only rank 0 needs the full state for diagnostics,
                    # so use root_only=True to save 50% MPI bandwidth
                    # (MPI.Gather instead of MPI.Allgather).
                    _ofi = list(self._owned_face_ids)
                    fields_local = {
                        'u': state.u.data[_ofi], 'v': state.v.data[_ofi],
                        'T': state.T.data[_ofi], 'p_s': state.p_s.data[_ofi],
                        'phis': state.phis.data[_ofi],
                    }
                    # All ranks must participate in MPI.Gather
                    fields_global = {}
                    for name, arr in fields_local.items():
                        fields_global[name] = gather(arr, self._layout, root_only=True)

                    # Gather tracers (all ranks participate)
                    for tname in ('q_v', 'q_c', 'q_r', 'q_i', 'q_s', 'q_g'):
                        arr = kwargs.get(tname)
                        if arr is not None:
                            kwargs[tname] = gather(arr[_ofi], self._layout, root_only=True)

                    # Gather the segment-accumulated 2-D diagnostics too:
                    # these are accumulated at OWNED faces only
                    # (``.at[_ofi].add`` in the segment scan), so without
                    # this gather rank 0 would write zeros / stale values
                    # on its five non-owned faces into the timeseries and
                    # CMOR output.  All ranks must participate (collective).
                    for tname in ('precip_total', 'shflx', 'lhflx',
                                  'sw_up_toa', 'lw_up_toa', 'sw_net_sfc',
                                  'lw_net_sfc', 'sw_down_toa', 't_low_mean',
                                  'sw_up_toa_clr', 'lw_up_toa_clr'):
                        arr = kwargs.get(tname)
                        if arr is not None:
                            kwargs[tname] = gather(arr[_ofi], self._layout, root_only=True)

                    # Only rank 0 collects full diagnostics
                    if self._mpi_rank != 0:
                        return {'mean_T': 0.0, 'max_v': 0.0}

                    gathered = HydrostaticState(
                        u=Field(fields_global['u'], name="u", dims=state.u.dims, units=state.u.units),
                        v=Field(fields_global['v'], name="v", dims=state.v.dims, units=state.v.units),
                        T=Field(fields_global['T'], name="T", dims=state.T.dims, units=state.T.units),
                        p_s=Field(fields_global['p_s'], name="p_s", dims=state.p_s.dims, units=state.p_s.units),
                        phis=Field(fields_global['phis'], name="phis", dims=state.phis.dims, units=state.phis.units),
                    )
                    kwargs['state'] = gathered

                return self.diagnostics.collect(**kwargs)
            elif self._device_config.mesh is not None:
                # Multi-GPU single-node: replicate sharded → full
                from legoesm.parallel.sharded_dynamics import gather_state
                gathered = gather_state(kwargs.get('state', self.state), self._device_config)
                kwargs['state'] = gathered
        jax.block_until_ready(kwargs.get('state', self.state).u.data)
        return self.diagnostics.collect(**kwargs)

    def _create_friction(self) -> None:
        """Precompute Rayleigh friction decay factors."""
        cfg = self.config
        DT = cfg.dycore.dt
        sigma_full = self.sigma.sigma_full

        k_f_max = cfg.k_BL_max_per_day / 86400.0
        k_free = cfg.k_free_per_day / 86400.0
        # Sign/units: k_f >= 0 [1/s], DT [s] -> fric_decay = exp(-k_f*DT) in
        # (0, 1]; this Rayleigh term is a NON-CONSERVATIVE momentum SINK relaxing
        # u, v toward rest (never amplifies).  The BL/free-tropo drag is the
        # Held-Suarez DRY-CORE surrogate for surface friction.  A real turbulence
        # scheme already applies the PHYSICAL surface stress as the boundary-layer
        # bottom BC (louis.py implicit diffusion of u, v with sflx_u = tau_x, i.e.
        # momentum handed to the ocean/land), so keeping k_f here DOUBLE-COUNTS
        # surface drag -- a spurious second, momentum-to-nowhere sink that
        # ~halves the low-level trades (#931).  Gate it to an exact no-op
        # (k_f = 0 -> decay = 1.0) whenever a real BL scheme owns surface
        # momentum; keep it only when NO BL scheme does -- i.e.
        # turbulence == "none".  That single condition is sufficient: a pure
        # Held-Suarez dry core runs turbulence="none" (the config default, and
        # the HS test matrix sets it explicitly), so it still gets its defining
        # Rayleigh friction here.  We must NOT additionally keep k_f on
        # held_suarez_forcing: HS is ADDITIVE to the physics pipeline, so a
        # held_suarez_forcing + louis config would apply BOTH the Louis surface
        # stress AND this Rayleigh drag -- the very double-count this fix removes
        # (codex #931).  The HS *thermal* Newtonian relaxation is applied
        # separately below and is unaffected.  cfg.turbulence is a STATIC Python
        # config field -> compile-time feature gate (NOT jnp.where),
        # constant-folds, no retrace/AD impact.  NOTE: turbulence != "none" is
        # the proxy for "a BL scheme owns surface momentum" -- correct for all
        # stock schemes (nonzero drag); a degenerate Cd_neutral=0 override would
        # give zero surface stress yet still gate k_f off (an undamped BL), a
        # user misconfiguration outside this fix's scope.
        if cfg.turbulence == "none":
            k_f = k_free + k_f_max * jnp.maximum(
                0.0, (sigma_full - cfg.sigma_b) / (1.0 - cfg.sigma_b)
            )
        else:
            k_f = jnp.zeros_like(sigma_full)  # decay = 1.0, exact no-op
        # Top-of-atmosphere sponge (#836): a Rayleigh damping increasing toward
        # the model lid (sigma -> 0), ADDED to the surface-drag k_f so the
        # existing fric_decay tail (applied to u, v every step) absorbs
        # upward-propagating wave energy the hydrostatic latlon-cgrid dycore
        # otherwise reflects off the rigid ~35 hPa top.  sin^2 taper from 0 at
        # sigma = sponge_sigma_top to sponge_coeff_per_day at the top (the same
        # shape as dynamics.compressible_euler.sponge_profile, expressed in
        # sigma).  Config-gated -> byte-identical when sponge_enabled is False.
        if getattr(cfg, "sponge_enabled", False):
            k_sp_max = cfg.sponge_coeff_per_day / 86400.0
            _sig_top = max(cfg.sponge_sigma_top, 1e-6)  # coeff-ok: /~0 guard
            frac = jnp.clip((_sig_top - sigma_full) / _sig_top, 0.0, 1.0)
            k_f = k_f + k_sp_max * jnp.sin(0.5 * jnp.pi * frac) ** 2
        self._fric_decay = jnp.exp(-k_f * DT)
        self._qv_smooth_coeff = self._hyperdiff * 0.5

        # Select grid-appropriate hyperdiffusion operator
        if cfg.grid.grid_type == "cubed_sphere":
            from legoesm.core.operators_3d import hyperdiffusion_3d
        else:
            from legoesm.core.operators_latlon_3d import hyperdiffusion_3d
        self._hyperdiffusion_3d_fn = hyperdiffusion_3d

        # Held-Suarez Newtonian temperature relaxation (precomputed coefficients)
        if cfg.held_suarez_forcing:
            from legoesm.atmosphere.forcing.idealized.held_suarez import (
                held_suarez_equilibrium_temperature,
                K_A, K_S, SIGMA_B,
            )
            _hs_sigma_b = SIGMA_B
            _hs_k_a = K_A
            _hs_k_s = K_S

            def _newtonian_relax(T, p_s, lat):
                """Compute dT/dt from HS Newtonian relaxation [K/s].

                Handles arbitrary lat shapes: (6,n,n) for cubed-sphere,
                (n_lat, n_lon) for lat-lon, (n_lat,) for Gaussian.
                """
                p_full = p_s[..., None] * sigma_full
                # Expand lat to broadcast with (... , nlev)
                n_expand = p_full.ndim - lat.ndim
                lat_exp = lat
                for _ in range(n_expand):
                    lat_exp = lat_exp[..., None]
                T_eq = held_suarez_equilibrium_temperature(lat_exp, p_full)
                sigma_factor = jnp.maximum(
                    0.0, (sigma_full - _hs_sigma_b) / (1.0 - _hs_sigma_b))
                cos_lat_4 = jnp.cos(lat_exp) ** 4
                k_T = _hs_k_a + (_hs_k_s - _hs_k_a) * sigma_factor * cos_lat_4
                return -k_T * (T - T_eq)

            self._hs_newtonian_relax = _newtonian_relax

    def _save_config(self) -> None:
        """Save experiment config to output directory (rank 0 only)."""
        if self._mpi_rank is not None and self._mpi_rank != 0:
            return
        from legoesm.driver.config import save_experiment_config
        save_experiment_config(self.config, self._output_dir / "experiment_config.json")

    def _write_run_manifest(self) -> None:
        """Write the reproducibility run manifest (Stage A1; rank 0 only).

        Captures the *resolved* runtime config (``self.config`` — after grid-type
        normalization and any setup-time overrides) plus environment/git
        provenance, so an interrupted or archived run is reconstructible.  Writes
        beside ``experiment_config.json`` in the run's output directory.

        * Rank-0 guarded (like ``_save_config``) so MPI ranks do not race on the
          one file.
        * **Write-once:** an existing *valid, same-config* manifest is preserved
          (a legitimate resume/retry of the same run). An existing manifest that
          is invalid, or written for a *different* config, is fatal — one output
          directory holds one run; mixing two runs' provenance is refused.
        * **Required, not best-effort:** a write failure aborts ``setup()``.
          This is cheap — setup() has not yet entered the time loop — and upholds
          the A1 guarantee that a run which proceeds is always reconstructible.

        NOTE: this is a provenance guard, not a mutual-exclusion lock.  The
        atomic-exclusive create stops two starts from both *creating* the
        manifest, and a different-config reuse is fatal, but two *same-config*
        concurrent invocations sharing one explicit output directory are treated
        as a resume and both proceed — they would then collide on checkpoints /
        diagnostics / results, exactly as the driver's other output writes
        already do without a run lock.  Serialising concurrent same-config
        writers needs a general active-run lease (acquire / heartbeat / stale
        detection), which is a driver-wide feature, not the manifest's job, and
        must not break legitimate checkpoint-restart into the same directory.
        Tracked as a follow-up; default output dirs are uniquely timestamped, so
        this only bites on deliberate explicit-dir reuse.
        """
        if self._mpi_rank is not None and self._mpi_rank != 0:
            return
        from legoesm.driver.restart import (
            RUN_MANIFEST_FILENAME,
            compute_config_hash,
            dataset_provenance_entry,
            read_run_manifest,
            validate_run_manifest,
            write_run_manifest,
        )
        manifest_file = self._output_dir / RUN_MANIFEST_FILENAME
        # Dataset provenance: every path-typed ExperimentConfig field that
        # names an input dataset. Membership-checked against _fields (legacy
        # config kinds may lack some) — never getattr-with-default, which
        # would silently drop a renamed field.
        dataset_path_fields = (
            "forcing_path",
            "sic_path",
            "ozone_file",
            "ghg_file",
            "solar_file",
            "aerosol_file",
            "volcanic_aerosol_file",
            "land_mask_path",
            "ic_path",
        )
        config_fields = type(self._input_config)._fields
        datasets = [
            dataset_provenance_entry(getattr(self._input_config, name), dataset_id=name)
            for name in dataset_path_fields
            if name in config_fields and getattr(self._input_config, name)
        ]
        # Atomic-exclusive create: wins the race against a concurrent start into
        # the same directory.  The winner creates the manifest; everyone else
        # (this run on retry/resume, or a racing process) takes the validate path.
        try:
            write_run_manifest(
                self._output_dir,
                self._input_config,
                exclusive=True,
                rng_seeds={"master": self._input_config.seed},
                dataset_provenance=datasets,
            )
            return
        except FileExistsError:
            pass
        # An existing manifest must be valid AND for this exact config; otherwise
        # fail CLOSED rather than proceed with mismatched/unreconstructable
        # provenance.  Runs before _save_config so experiment_config.json is never
        # overwritten by a start that is about to be rejected.
        try:
            existing = read_run_manifest(manifest_file)
            validate_run_manifest(existing)
        except Exception as exc:
            raise RuntimeError(
                f"Existing run manifest at {manifest_file} is invalid ({exc}); "
                f"refusing to start a run without valid run-start provenance. "
                f"Remove or repair it to continue."
            ) from exc
        if existing["config"]["config_hash"] != compute_config_hash(self._input_config):
            raise RuntimeError(
                f"Output directory {self._output_dir} already holds a run "
                f"manifest written for a DIFFERENT config; refusing to mix two "
                f"runs' provenance in one directory. Use a fresh output "
                f"directory, or remove the existing run_manifest.json."
            )
        logger.info(
            f"Run manifest already present at {manifest_file}; preserving it "
            f"(same config — resume/retry)."
        )

    def _record_final_state_digest(self) -> None:
        """Record the final-state SHA-256 digest into the run manifest (rank 0).

        Fills ``result.state_digest`` so ``legoesm reproduce --check`` has a
        reference to compare a rerun against.  Best-effort: the run has already
        completed successfully, so a digest/layout hiccup (e.g. a backend whose
        state pytree differs from the checkpoint layout) must not turn a good run
        into a failure — it just leaves reproduce --check without a reference,
        which it reports honestly.
        """
        if self._mpi_rank is not None and self._mpi_rank != 0:
            return
        try:
            from legoesm.driver.restart import (
                RUN_MANIFEST_FILENAME,
                pytree_state_digest,
                record_state_digest,
            )
            manifest_file = self._output_dir / RUN_MANIFEST_FILENAME
            if not manifest_file.exists():
                return
            # Backend-agnostic digest of the full final state (prognostic state +
            # tracers + carry), so spectral/MPAS layouts are covered too.
            digest = pytree_state_digest(
                self.state, self.tracers, self._carry_aux
            )
            record_state_digest(manifest_file, digest)
        except Exception as exc:  # pragma: no cover - provenance best-effort
            logger.warning(f"Could not record final state digest: {exc}")


    def _maybe_build_tiled_step(self, dt):
        """Build the sub-face-tiled dynamics step (P4 increment 1b) or None.

        Returns ``make_tiled_cc_step`` over this driver's model + device
        mesh when ``config.enable_tiled_dycore`` is on and the device
        layout is sub-face tiled; ``None`` (the default) leaves the
        compiled segment on ``_dynamics_model.step``.  The model copy
        mirrors ``build_segment_fn``'s inner dynamics copy under the SAME
        predicate (outer ``cfg.dycore.fix_mass`` AND model-config
        ``fix_mass`` -> disable inner fixer + per-stage zero-mean; the
        segment applies the target-anchored fixer OUTSIDE the dynamics),
        so the tiled numerics match the untiled inner model exactly; a
        config whose EFFECTIVE inner model still applies per-stage
        ``zero_mean_ps_tendency`` is refused (the tiled base cut omits
        that term).  Flag-on with no tiled layout is a LOUD error, never
        a silent untiled fallback, and the whole path is gated behind
        ``LEGOESM_TILED_DYCORE_EXPERIMENTAL=1`` until the outer segment
        sharding composition is device-validated (increment 1c).
        """
        if not getattr(self.config, "enable_tiled_dycore", False):
            return None
        dc = self._device_config
        if (dc is None or getattr(dc, "mesh", None) is None
                or tuple(getattr(dc, "tiling", (1, 1))) == (1, 1)):
            raise ValueError(
                "enable_tiled_dycore=True requires a sub-face-tiled device "
                "layout (n_devices = 6*kt^2 > 6); got "
                f"tiling={getattr(dc, 'tiling', None)!r}. Disable the flag "
                "or launch with a tiled device count."
            )
        kt_i, kt_j = dc.tiling
        if kt_i != kt_j:
            raise ValueError(
                f"enable_tiled_dycore: tiling must be square, got {dc.tiling}")
        import os as _os
        if _os.environ.get("LEGOESM_TILED_DYCORE_EXPERIMENTAL") != "1":
            raise NotImplementedError(
                "enable_tiled_dycore: the OUTER compiled-segment sharding "
                "composition around the tiled core is not yet device-"
                "validated (the segment currently runs with "
                "device_config=None under sub-face tiling — codex round-14 "
                "HIGH; increment 1c is the real-device full-segment parity "
                "lane).  Set LEGOESM_TILED_DYCORE_EXPERIMENTAL=1 to run "
                "anyway."
            )
        import copy as _copy
        from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
            make_tiled_cc_step,
        )
        _m = _copy.copy(self.model)
        _mc = getattr(_m, "config", None)
        # Mirror build_segment_fn's inner-copy predicate EXACTLY (codex
        # round-14 Medium): the outer target-anchored fixer path
        # (cfg.dycore.fix_mass True) disables the inner fixer + per-stage
        # zero-mean; when the outer fixer is OFF the untiled inner model
        # KEEPS zero_mean_ps_tendency active — a per-stage global-mean
        # term the tiled base cut does not implement, so that case is
        # refused rather than silently dropped.
        _outer_fix_mass = bool(getattr(self.config.dycore, "fix_mass", False))
        if (
            _outer_fix_mass
            and getattr(_mc, "fix_mass", False)
            and hasattr(_mc, "_replace")
        ):
            _kw = {"fix_mass": False}
            if hasattr(_mc, "zero_mean_ps_tendency"):
                _kw["zero_mean_ps_tendency"] = False
            _m.config = _mc._replace(**_kw)
            _mc = _m.config
        # The EFFECTIVE inner model (post-mirror) must not apply the
        # per-RK-stage zero-mean (gate in primitive_eq_cdgrid:
        # ``zm and not (ucf and fm)``): the tiled base cut integrates the
        # RAW dp_s/dt, and silently dropping the term would change the
        # untiled-vs-tiled numerics.  (The tiled psum primitive
        # ``make_tiled_zero_mean_tendency_stage_2d`` exists but is not
        # wired into the step stage — increment 1c+.)
        _zm_active = (
            bool(getattr(_mc, "zero_mean_ps_tendency", False))
            and not (bool(getattr(_mc, "use_conservation_fixer", False))
                     and bool(getattr(_mc, "fix_mass", False)))
        )
        if _zm_active:
            raise NotImplementedError(
                "enable_tiled_dycore: this config leaves per-RK-stage "
                "zero_mean_ps_tendency ACTIVE on the inner model, which "
                "the tiled base cut does not implement — enable the outer "
                "mass fixer (conservation_fixer + fix_mass) or set "
                "zero_mean_ps_tendency=False."
            )
        logger.info(
            "Tiled dycore step ROUTED into the compiled segment "
            "(P4 increment 1b, experimental): kt=%d, dt=%.1f s, "
            "mesh axes %s.", int(kt_i), float(dt),
            getattr(dc.mesh, "axis_names", None),
        )
        return make_tiled_cc_step(_m, dc.mesh, kt=int(kt_i), dt=float(dt))

    def _bootstrap_runtime(self) -> None:
        """Bootstrap the full runtime: precision, backend, devices, MPI.

        Uses the canonical ``legoesm.runtime.bootstrap()`` entry point
        so that all initialisation (XLA flags, x64, precision policy,
        device mesh, MPI topology) goes through one place.
        """
        from legoesm.runtime import bootstrap

        rc = bootstrap(
            precision=self.config.precision,
            distributed=self.config.distributed,
            distributed_mode=getattr(
                self.config, "distributed_mode", "mpi",
            ),
            grid_type=self.config.grid.grid_type,
            n_devices=self.config.n_devices,
            allow_level_fallback=getattr(
                self.config, "allow_level_fallback", False,
            ),
            # For lat-lon MPI: pass n_lat so initialize_distributed_latlon
            # can build the rank's LatLonBandLayout.  Other grids ignore
            # this — cubed-sphere uses the per-face N via its own path.
            grid_n=self.config.grid.resolution,
        )
        self._device_config = rc.device_config

        # Multi-controller SPMD (distributed_mode='spmd'): rank/world
        # come from jax.distributed — no mpi4jax topology exists (and
        # must never be armed in this mode).  The DeviceConfig keeps
        # is_distributed=False so _setup_parallel routes the SPMD shard
        # branch; only the process-0 output guards need the rank.
        if (rc.distributed and getattr(
                self.config, "distributed_mode", "mpi") == "spmd"):
            import jax as _jax
            self._mpi_rank = _jax.process_index()
            self._mpi_world_size = _jax.process_count()
            logger.info(
                "  Runtime: multi-controller SPMD — process %d/%d, "
                "%d global devices",
                self._mpi_rank, self._mpi_world_size,
                getattr(rc.device_config, "n_devices", 1),
            )
        # Detect MPI rank early for output guards and logging.
        # Two topology shapes coexist in the codebase:
        #   - cubed-sphere ``MPITopology``       → ``.n_processes``
        #   - lat-lon band ``LatLonBandLayout``  → ``.n_ranks``
        # Use ``getattr`` so this early hook works for both without
        # needing to import either type here.  ``_setup_parallel``
        # later overwrites these values with the type-specific path.
        elif rc.distributed:
            from legoesm.parallel.distributed import get_active_topology
            topo = get_active_topology()
            if topo is not None:
                self._mpi_rank = topo.rank
                self._mpi_world_size = getattr(
                    topo, "n_processes",
                    getattr(topo, "n_ranks", None),
                )
                if self._mpi_world_size is None:
                    raise RuntimeError(
                        f"Active topology {type(topo).__name__} exposes "
                        "neither ``.n_processes`` nor ``.n_ranks``; "
                        "cannot determine MPI world size.  Extend the "
                        "early-detect hook in _bootstrap_runtime."
                    )

        logger.info(
            f"  Runtime: backend={rc.backend}, precision={self.config.precision}, "
            f"x64={rc.x64}, distributed={rc.distributed}"
        )

    def _setup_parallel(self) -> None:
        """Shard or scatter state after grid/state creation.

        The device mesh and MPI topology were already set up by
        ``_bootstrap_runtime()``.  This method handles the data-level
        work that requires knowing the grid shape: scatter for MPI,
        or shard for multi-device SPMD.

        **MPI strategy (replicated dynamics):**
        State and tracers are kept at full ``(6, n, n, ...)`` shape on
        every rank so that ``pad_halo_mpi`` (which expects the 6-face
        layout) works unchanged.  Only physics-related arrays (lat, lon,
        SST/SIC, ozone, aerosol) are scattered to rank-local for the
        column-parallel physics.  Conservation fixers use an
        ``owned_mask`` to sum only owned faces, then ``global_sum_mpi``
        to combine across ranks.
        """
        # Device config was set by _bootstrap_runtime().  If None or
        # single-device without distribution, nothing to do.
        if self._device_config is None:
            return
        if (not self._device_config.is_distributed
                and self._device_config.n_devices <= 1):
            return

        # MPAS / Voronoi cell-partition MPI.  Like the lat-lon band, each
        # rank already owns its local (owned+halo) mesh state directly:
        # ``_create_grid`` sliced ``self.grid`` to ``vlayout.local_mesh`` and
        # ``_init_state`` built the state on it, and ``_create_forcing`` built
        # the SST function on the local cells — so there is no scatter from
        # global and no forcing re-slice wrapper (contrast the lat-lon band,
        # whose forcing is global + sliced).  Physics runs column-local on
        # every local cell; halo columns are recomputed but harmless (their
        # prognostic values are overwritten by the next step's halo exchange).
        # The owned-cell mask gates conservation reductions and the
        # diagnostics gather.  Set the markers and return BEFORE the
        # cubed-sphere face path below (which assumes a leading dim of 6 and
        # would mis-handle the 1-D ``(nCells,)`` cell layout).
        if (self._device_config.is_distributed
                and self._voronoi_layout is not None):
            vlayout = self._voronoi_layout
            self._layout = vlayout
            self._physics_lat = self.grid.grid_lat
            self._physics_lon = self.grid.grid_lon
            self._mpi_rank = vlayout.rank
            self._mpi_world_size = vlayout.n_ranks
            logger.info(
                "  Parallel: MPAS cell-partition MPI — rank %d/%d, "
                "%d owned cells (+%d halo)",
                vlayout.rank, vlayout.n_ranks,
                vlayout.partition.n_owned_cells,
                vlayout.partition.n_local_cells
                - vlayout.partition.n_owned_cells,
            )
            return

        # Stage 3-C: lat-lon band MPI follows a different parallel
        # protocol than cubed-sphere replicated dynamics — each rank
        # owns its band's state directly (no 6-face replication), so
        # no scatter-from-global is needed and no ColumnAdapter
        # rebuild applies.  But SST/SIC + lat/lon physics arrays DO
        # need rank-local slicing for the column-local physics.
        if (self._device_config.is_distributed
                and self.config.grid.grid_type == "latlon"):
            from legoesm.grids.halo import get_mpi_topology
            from legoesm.parallel.latlon_mpi import LatLonBandLayout
            _layout = get_mpi_topology()
            if isinstance(_layout, LatLonBandLayout):
                self._layout = _layout
                self._mpi_rank = _layout.rank
                self._mpi_world_size = _layout.n_ranks

                s, e = _layout.lat_start, _layout.lat_end

                # Slice lat/lon arrays for physics (column-local).
                # The grid in self.grid is already rank-local
                # (Stage 3-B), so its lat2d/lon2d are already
                # band-sliced — reuse them.
                self._physics_lat = self.grid.lat2d
                self._physics_lon = self.grid.lon2d

                # Wrap SST/SIC to return only this rank's lat band.
                # The global ``get_sst_sic(day)`` returns shape
                # ``(n_lat_global, n_lon)``; slice along axis 0.
                if self.get_sst_sic is not None:
                    _global_fn = self.get_sst_sic

                    def _band_get_sst_sic(day, _s=s, _e=e, _fn=_global_fn):
                        sst, sic = _fn(day)
                        sst = jnp.asarray(sst)
                        sic = jnp.asarray(sic)
                        # SST/SIC shapes: typically (n_lat, n_lon),
                        # sometimes (n_lat, n_lon, 1) for ensemble.
                        # Slice axis 0 unconditionally.
                        return sst[_s:_e], sic[_s:_e]

                    self.get_sst_sic = _band_get_sst_sic

                logger.info(
                    "  Parallel: lat-lon band MPI — rank %d/%d, "
                    "rows [%d:%d) of %d (n_lat_local=%d)",
                    _layout.rank, _layout.n_ranks, s, e,
                    _layout.n_lat_global, _layout.n_lat_local,
                )
                return  # Skip the cubed-sphere replicated-dynamics path below.

        if self._device_config.is_distributed:
            from legoesm.parallel.distributed import (
                get_active_layout, set_active_layout,
                get_active_topology,
            )
            from legoesm.parallel.layout import scatter

            topo = get_active_topology()
            layout = get_active_layout()
            if layout is None and topo is not None:
                # Deferred layout: grid_n wasn't known at init time
                from legoesm.parallel.layout import make_layout
                n = self.state.T.data.shape[1]  # per-face resolution
                layout = make_layout(topo.rank, topo.n_processes, n)
                set_active_layout(layout)

            from legoesm.parallel.layout import DistributedLayout
            if isinstance(layout, DistributedLayout):
                # Store MPI metadata for later phases
                self._layout = layout
                self._mpi_rank = topo.rank
                self._mpi_world_size = topo.n_processes
                self._owned_face_ids = jnp.asarray(
                    list(layout.ownership.face_ids)
                )

                # NOTE: State and tracers are NOT scattered — dynamics
                # needs full (6, n, n) for pad_halo_mpi.  Non-owned faces
                # will diverge from truth but owned faces stay correct via
                # MPI halo exchange.

                # Scatter lat/lon for rank-local physics
                self._physics_lat = scatter(self._grid_lat, layout)
                self._physics_lon = scatter(self._grid_lon, layout)

                # Rebuild physics adapter for rank-local column count
                from legoesm.core.grid_adapters import ColumnAdapter
                local_shape_2d = tuple(int(s) for s in self._physics_lat.shape)
                local_ncol = 1
                for s in local_shape_2d:
                    local_ncol *= s
                local_adapter = ColumnAdapter(ncol=local_ncol, shape_2d=local_shape_2d)
                if self.physics is not None:
                    self.physics.adapter = local_adapter
                    # Scatter the slab-land surface fields to owned faces
                    # so the rank-local physics columns match f_land /
                    # albedo_land (the MPI ``owned_face_ids`` path).
                    if self.physics.f_land is not None:
                        self.physics.f_land = scatter(
                            self.physics.f_land, layout)
                        self.physics.albedo_land = scatter(
                            self.physics.albedo_land, layout)

                # Wrap SST/SIC forcing to return rank-local arrays
                _global_get_sst_sic = self.get_sst_sic
                def _local_get_sst_sic(day, _layout=layout, _fn=_global_get_sst_sic):
                    sst, sic = _fn(day)
                    sst = jnp.asarray(sst)
                    sic = jnp.asarray(sic)
                    if sst.ndim >= 3 and sst.shape[0] == 6:
                        sst = scatter(sst, _layout)
                        sic = scatter(sic, _layout)
                    return sst, sic
                self.get_sst_sic = _local_get_sst_sic

                logger.info(
                    f"  Parallel: MPI distributed — rank {topo.rank}/{topo.n_processes}, "
                    f"owned faces {list(layout.ownership.face_ids)}, "
                    f"physics shape {local_shape_2d}"
                )
        else:
            # SPMD sharding: multi-GPU single-node, AND multi-controller
            # jax.distributed (distributed_mode='spmd' — the DeviceConfig
            # mesh spans the GLOBAL device set, so the same shard +
            # halo-backend activation gives true cubed-sphere domain
            # decomposition across processes; bench --cs-spmd receipts
            # jobs 8462928/8465445).
            from legoesm.parallel.sharded_dynamics import shard_state
            # Pass grid_type so a lat-lon state shards on the LATITUDE
            # axis, not the cubed-sphere 6-face rules (shard_state
            # defaults to "cubed_sphere").  tracers already go through
            # the grid-aware shard_pytree below; the main state must
            # match or a single-node multi-GPU lat-lon run shards the
            # state under the wrong layout (codex P1, 2026-06-13).
            self.state = shard_state(
                self.state, self._device_config,
                grid_type=self.config.grid.grid_type)

            from legoesm.parallel.mesh import shard_pytree
            self.tracers = shard_pytree(self.tracers, self._device_config)

            # Issue #275 fix A: activate explicit SPMD halo backend so
            # ``pad_halo_4d`` lowers to ``ppermute`` / ``all_gather``
            # collectives instead of implicit cross-shard
            # ``dynamic_slice`` reads.  Without this call, multi-GPU
            # cubed-sphere AMIP pays an implicit-collective tax that
            # XLA's latency-hiding scheduler cannot pipeline.
            #
            # Activation predicate mirrors ``make_sharded_step``'s
            # supported set in ``parallel/sharded_dynamics.py``: face-
            # only sharding (no sub-face tiling), a ``face`` mesh axis,
            # cubed-sphere grid type, and ``n_devices in (1, 2, 3, 6)``
            # because ppermute / all_gather kernels assume divisors of
            # 6 faces.  Non-cubed-sphere grids and unsupported device
            # counts keep the local backend silently (they cannot
            # benefit); sub-face tiling (>6 devices) keeps it with a
            # LOUD warning (the tiled dycore step is unwired — P4).
            self._maybe_activate_spmd_halo_backend()

        logger.info(
            f"  Parallel: {self._device_config.n_devices} devices, "
            f"tiling={self._device_config.tiling}"
        )

    def _maybe_activate_spmd_halo_backend(self) -> None:
        """Activate the explicit SPMD halo backend when supported.

        Unsupported configurations are not activated.  Non-cubed-sphere
        grids, unsupported device counts, and no-mesh are skipped
        silently (they cannot benefit from the SPMD halo collectives).
        Sub-face tiling (>6 devices) never activates THIS backend (the
        tiled dycore stage carries its own in-stage shard_map halos) but
        is surfaced loudly either way: an INFO receipt when
        ``enable_tiled_dycore`` routes dynamics through the tiled stage
        (P4 increment 1b, experimental), or a WARNING that the run
        stays on the GSPMD-auto sliced step with local halos and will
        not strong-scale past 6 devices — never a silent degrade.

        For supported configurations, activation must either succeed
        or fail loudly.  Both import failures and activation failures
        raise ``RuntimeError`` unless ``LEGOESM_ALLOW_LOCAL_HALO_FALLBACK=1``
        is set in the environment, which is the explicit escape hatch
        for users who want to keep running on the slow local backend
        (e.g. while debugging a JAX or mpi4jax upgrade).

        Nested activation by a second driver in the same process is
        rejected when the existing SPMD mesh does not match this
        driver's mesh — reusing a mismatched mesh would route halo
        exchange through the wrong device topology.  When the meshes
        match identically, the second driver is a non-owner of the
        activation and ``_restore_halo_backend`` is a no-op for it.

        The supported predicate matches the SPMD halo kernels:

        * grid type is ``"cubed_sphere"`` (only grid with SPMD
          connectivity tables);
        * a JAX device mesh exists with a ``"face"`` axis;
        * face-only sharding (``tiling == (1, 1)``) — the ppermute
          kernel assumes one face per shard;
        * ``n_devices`` in ``(1, 2, 3, 6)`` — divisors of 6 faces.
        """
        cfg = self.config
        dc = self._device_config
        if dc is None or dc.mesh is None:
            return
        if getattr(cfg.grid, "grid_type", None) != "cubed_sphere":
            return
        if "face" not in dc.mesh.axis_names:
            return
        if getattr(dc, "tiling", (1, 1)) != (1, 1):
            # Sub-face tiling (>6 devices — the production GPU strong-
            # scaling regime).  The tiled ppermute EXCHANGE layer is
            # parity-proven (cubesphere_exchange, 24-proc), but this
            # SPMD halo backend is NOT the tiled paths' exchange layer
            # (both tiled lanes carry their own in-stage / mesh-bound
            # halos) — so it stays off either way.  What changes is the
            # DYNAMICS routing: with enable_tiled_dycore the compiled
            # segment runs the tiled D-grid core (P4 increment 1b,
            # experimental); without it run() dispatches these device
            # counts to the BLOCKED tiled cube loop
            # (_run_tiled_cube_spmd).  Inform loudly instead of
            # returning silently (codex P1, 2026-06-13; message split
            # when the flag landed).
            if getattr(self.config, "enable_tiled_dycore", False):
                # The build-time env gate lives in _maybe_build_tiled_step
                # (which runs later) — don't log an ACTIVE receipt for a
                # run that gate will refuse (codex round-16 Low).
                import os as _os
                if (_os.environ.get("LEGOESM_TILED_DYCORE_EXPERIMENTAL")
                        == "1"):
                    logger.info(
                        "Sub-face tiling %s (%d devices): tiled dycore "
                        "step ACTIVE (enable_tiled_dycore, P4 increment "
                        "1b — experimental); the tiled stage uses its "
                        "own in-stage halos (the ppermute SPMD backend "
                        "stays off).",
                        dc.tiling, dc.n_devices,
                    )
                else:
                    logger.warning(
                        "Sub-face tiling %s (%d devices): "
                        "enable_tiled_dycore is set but "
                        "LEGOESM_TILED_DYCORE_EXPERIMENTAL=1 is not — "
                        "the segment build will refuse loudly.",
                        dc.tiling, dc.n_devices,
                    )
                return
            logger.info(
                "SPMD halo backend not armed for sub-face tiling %s "
                "(%d devices): run() dispatches these device counts to "
                "the BLOCKED tiled cube loop (_run_tiled_cube_spmd), "
                "whose in-stage ppermute pads are mesh-bound and do not "
                "use the global halo backend.  Out-of-envelope configs "
                "(full unified physics, diagnostics writers) are refused "
                "loudly there — nothing silently degrades to local halos.",
                dc.tiling, dc.n_devices,
            )
            return
        if dc.n_devices not in (1, 2, 3, 6):
            return

        # Codex review (issue #275): both import and activation must
        # fail loudly for supported configurations.  Silent fallback
        # would mask the exact regression this fix is meant to
        # prevent.  ``LEGOESM_ALLOW_LOCAL_HALO_FALLBACK=1`` opts back
        # into the legacy warn-and-skip behaviour.
        allow_fallback = os.environ.get(
            "LEGOESM_ALLOW_LOCAL_HALO_FALLBACK", "",
        ).strip().lower() in {"1", "true", "yes", "on"}

        try:
            from legoesm.grids.halo import get_halo_backend
            from legoesm.parallel.cubesphere_exchange import (
                activate_spmd_halo_backend,
                get_spmd_mesh,
            )
        except ImportError as exc:
            if allow_fallback:
                logger.warning(
                    "SPMD halo backend unavailable (import failed: %s); "
                    "LEGOESM_ALLOW_LOCAL_HALO_FALLBACK=1 set, so "
                    "falling back to local halo backend.  Multi-GPU "
                    "performance will be degraded.",
                    exc,
                )
                return
            raise RuntimeError(
                f"SPMD halo backend imports failed for a supported "
                f"config (grid=cubed_sphere, n_devices={dc.n_devices}, "
                f"tiling={dc.tiling}). Install the optional "
                f"``cubesphere_exchange`` deps or set "
                f"LEGOESM_ALLOW_LOCAL_HALO_FALLBACK=1 to degrade "
                f"silently.  Underlying error: {exc}"
            ) from exc

        # Refuse to overwrite an already-active SPMD activation: a
        # second driver in the same process must either share the
        # exact same mesh (non-owner reuse) or wait for the prior
        # driver to tear down.  See ``_meshes_compatible`` for the
        # comparison semantics — object identity is the fast path,
        # axis-names + device-ids is the fallback so two
        # semantically-equivalent ``Mesh`` objects (e.g. independently
        # constructed in coupled-model code) still compare equal.
        #
        # Known limitation (tracked as a follow-up): when the prior
        # driver tears down (refcount goes to zero) the global
        # backend reverts to ``"local"`` while the non-owner second
        # driver is mid-run, breaking its halo exchange.  Production
        # use is single-driver-per-process, so this is acceptable
        # short-term; the proper fix is a refcount inside
        # ``cubesphere_exchange.py``.
        previous_backend = get_halo_backend()
        if previous_backend == "spmd":
            active_mesh = get_spmd_mesh()
            if not _meshes_compatible(active_mesh, dc.mesh):
                raise RuntimeError(
                    "SPMD halo backend is already active in this "
                    "process with a different mesh.  Reusing it for "
                    "the current driver would route halo exchange "
                    "through the wrong device topology.  Tear down "
                    "the prior driver (or call "
                    "``deactivate_spmd_halo_backend()``) before "
                    "creating a new driver."
                )
            logger.info(
                "  Parallel: reusing existing SPMD halo backend "
                "(matching mesh, non-owner driver)"
            )
            return

        # Resolution and vertical level count are logging-only inputs
        # to ``activate_spmd_halo_backend`` — the old volume-based
        # ppermute/all_gather auto-selection is RETIRED (the all_gather
        # variant replicates ALL compute, HLO probe job 8456476;
        # ppermute is the only auto-selectable exchange, all_gather is
        # an explicit ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic).
        # ``self.state.T.data`` has shape ``(6, n, n, nlev)`` for
        # cubed-sphere hydrostatic states.
        n_face = int(self.state.T.data.shape[1])
        nlev = (
            int(self.state.T.data.shape[-1])
            if self.state.T.data.ndim >= 4
            else 1
        )

        try:
            activate_spmd_halo_backend(
                dc.mesh, n=n_face, nlev=nlev,
            )
        except Exception as exc:  # noqa: BLE001
            if allow_fallback:
                logger.warning(
                    "SPMD halo activation failed (%s); "
                    "LEGOESM_ALLOW_LOCAL_HALO_FALLBACK=1 set, so "
                    "falling back to local halo backend.  Multi-GPU "
                    "performance will be degraded.",
                    exc,
                )
                return
            raise RuntimeError(
                f"SPMD halo activation failed for a supported config "
                f"(grid=cubed_sphere, n_devices={dc.n_devices}, "
                f"tiling={dc.tiling}, n={n_face}, nlev={nlev}). "
                f"Set LEGOESM_ALLOW_LOCAL_HALO_FALLBACK=1 to degrade "
                f"silently to the local halo backend.  Underlying "
                f"error: {exc}"
            ) from exc

        self._spmd_halo_activated = True
        self._previous_halo_backend = previous_backend
        logger.info(
            "  Parallel: SPMD halo backend activated "
            "(n=%d, nlev=%d, devices=%d)",
            n_face, nlev, dc.n_devices,
        )

    def _restore_halo_backend(self) -> None:
        """Restore the halo backend captured before SPMD activation.

        Always safe to call — no-op when SPMD activation never
        happened.  Idempotent.  Used by :meth:`_finalize_run`,
        :meth:`__del__`, and tests that need to reset the
        process-global halo dispatch between runs.
        """
        if not self._spmd_halo_activated:
            return
        try:
            from legoesm.parallel.cubesphere_exchange import (
                deactivate_spmd_halo_backend,
            )
            deactivate_spmd_halo_backend()
            if self._previous_halo_backend == "mpi":
                # An MPI driver in the same process would have
                # already set the MPI backend; ``deactivate`` reverts
                # to ``"local"`` so reinstate MPI when that was the
                # prior dispatch.  This is defensive — the typical
                # case is ``"local"`` → ``"spmd"`` → ``"local"``.
                from legoesm.grids.halo import (
                    set_halo_backend, get_mpi_topology,
                )
                _topology = get_mpi_topology()
                if _topology is not None:
                    set_halo_backend("mpi", _topology)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SPMD halo deactivation failed (%s); halo backend "
                "may be left in 'spmd' state for subsequent code in "
                "this process.",
                exc,
            )
        finally:
            self._spmd_halo_activated = False
            self._previous_halo_backend = None

    def __del__(self):
        # Defensive: never let a leaked driver leave the halo backend
        # in ``"spmd"`` state.  ``__del__`` is best-effort and may run
        # during interpreter shutdown when imports fail, so swallow
        # everything.
        try:
            self._restore_halo_backend()
        except Exception:  # noqa: BLE001
            pass

    def _is_latlon_mpi(self) -> bool:
        """True iff this driver is running under lat-lon band MPI.

        Cubed-sphere uses ``_owned_face_ids``; lat-lon uses
        ``_layout`` but never ``_owned_face_ids``.  Distinguishing them
        keeps the save/load dispatch unambiguous so a future grid that
        sets a ``LatLonBandLayout``-shaped ``_layout`` won't silently
        steal the cubed-sphere branch.
        """
        from legoesm.parallel.latlon_mpi import LatLonBandLayout
        return (
            self._device_config is not None
            and self._device_config.is_distributed
            and self._layout is not None
            and isinstance(self._layout, LatLonBandLayout)
            and self._owned_face_ids is None
        )

    def _gather_state_for_global_checkpoint(self):
        """Collective gather of rank-local bands → rank-0 global state.

        Every rank in the lat-lon MPI communicator MUST call this; it
        runs ``mpi4py.comm.gather`` per state field.  Used by
        ``save_checkpoint`` so the resulting ``.npz`` is a single
        global-shape file (the format ``run_amip_1deg_latlon_mpi.sbatch``
        and ``--restart-from`` expect).

        Returns
        -------
        (state_global, tracers_global) on rank 0, where ``state_global``
        is a fresh HydrostaticState whose Field ``.data`` arrays span
        the full ``n_lat_global`` lat axis.  All other ranks get
        ``(None, None)``.
        """
        from legoesm.parallel.latlon_mpi import gather_field_latlon

        s = self.state
        T_g = gather_field_latlon(s.T.data, self._layout)
        u_g = gather_field_latlon(s.u.data, self._layout)
        v_g = gather_field_latlon(s.v.data, self._layout, is_v_face=True)
        p_g = gather_field_latlon(s.p_s.data, self._layout)
        phi_g = gather_field_latlon(s.phis.data, self._layout)

        # Collective tracer-key safety (Codex review MEDIUM #1).  If
        # ranks disagree on which tracers are active, the per-tracer
        # ``comm.gather`` below would either deadlock or fall out of
        # sync silently.  Allgather sorted-key tuples and assert
        # identical before iterating — fail loudly if not.
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        my_keys = tuple(sorted(
            k for k, v in self.tracers.items() if v is not None
        ))
        all_keys = comm.allgather(my_keys)
        if any(k != all_keys[0] for k in all_keys):
            raise RuntimeError(
                "Tracer key sets diverge across ranks under lat-lon "
                "MPI; refusing to checkpoint to avoid silent "
                "corruption.  Rank-by-rank keys: "
                f"{all_keys!r}.  Every rank's TracerRegistry must "
                "produce the same set of active tracer names."
            )

        # v-face boundary-row consistency (Codex review MEDIUM #2).
        # Each rank K (except the northernmost) shares v[lat_end_K] with
        # rank K+1's v[lat_start_{K+1}].  After every dycore step these
        # are written via different code paths on the two ranks; the
        # invariant 'duplicated row is bit-identical' must hold or the
        # gather below will silently pick rank K's value over K+1's.
        #
        # Implementation note: use ``sendrecv`` to ship each rank's
        # v[-1] north + receive the southern neighbour's v[-1], then
        # ``allreduce(MAX)`` over every rank's local diff so EVERY rank
        # raises (or none does) — keeps the assertion collective-safe.
        # A previous per-rank ``raise`` design would deadlock the
        # northernmost rank (no south to compare → doesn't raise →
        # walks into the gather collective while raised ranks have
        # left it).
        if self._layout.north_rank is not None:
            my_last_v = np.asarray(self.state.v.data[-1])
        else:
            my_last_v = None
        neighbour_south_last_v = comm.sendrecv(
            sendobj=my_last_v,
            dest=(self._layout.north_rank if self._layout.north_rank is not None
                  else MPI.PROC_NULL),
            sendtag=0,
            source=(self._layout.south_rank if self._layout.south_rank is not None
                    else MPI.PROC_NULL),
            recvtag=0,
        )
        if self._layout.south_rank is not None:
            my_first_v = np.asarray(self.state.v.data[0])
            local_diff = float(np.abs(my_first_v - neighbour_south_last_v).max())
        else:
            local_diff = 0.0
        global_max_diff = comm.allreduce(local_diff, op=MPI.MAX)
        if global_max_diff > 0.0:
            raise RuntimeError(
                "v-face boundary row diverges somewhere in the rank "
                f"chain; max |diff| across all rank pairs = "
                f"{global_max_diff:.3e}.  Halo-exchange invariant "
                "violated; refusing to checkpoint to avoid writing a "
                "non-deterministic global v field."
            )

        tracers_g: dict | None = {}
        for name in all_keys[0]:
            arr = self.tracers[name]
            tracers_g[name] = gather_field_latlon(arr, self._layout)

        if self._mpi_rank != 0:
            return None, None

        state_g = s._replace(
            T=s.T.replace(data=T_g),
            u=s.u.replace(data=u_g),
            v=s.v.replace(data=v_g),
            p_s=s.p_s.replace(data=p_g),
            phis=s.phis.replace(data=phi_g),
        )
        return state_g, tracers_g

    def _scatter_global_state_to_bands(self, state_global, tracers_global):
        """Mirror of :meth:`_gather_state_for_global_checkpoint` for restart.

        Rank 0 holds the global state just loaded from the checkpoint;
        all other ranks pass ``None``.  Every rank participates in the
        collective bcast inside :func:`scatter_field_latlon` and gets
        back its own lat band.

        Failure semantics (Codex review round 3, MEDIUM #2): the
        tracer-key collective-safety check runs BEFORE any state
        scatter so a divergent TracerRegistry leaves ``self.state``
        untouched.  A failed load can therefore be retried (or the
        driver torn down cleanly) without first having to undo a
        half-loaded band-shaped state.
        """
        from legoesm.parallel.latlon_mpi import scatter_field_latlon
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        # ----------------------------------------------------------
        # Stage A — collective sanity check on tracer-registry parity.
        # Done BEFORE any state mutation so a divergent registry
        # surfaces an error without side effects.
        # ----------------------------------------------------------
        if self._mpi_rank == 0:
            tracer_keys = sorted(
                k for k, v in tracers_global.items() if v is not None
            )
        else:
            tracer_keys = None
        tracer_keys = comm.bcast(tracer_keys, root=0)

        # Allgather every rank's missing-keys tuple so every rank sees
        # the same divergence flag and raises in lockstep (or none
        # does).  A per-rank raise pattern would leave non-raised
        # ranks blocked inside ``scatter_field_latlon``'s inner bcast.
        my_existing_keys = set(self.tracers.keys())
        my_missing = tuple(sorted(
            k for k in tracer_keys if k not in my_existing_keys
        ))
        all_missing = comm.allgather(my_missing)
        if any(m for m in all_missing):
            raise RuntimeError(
                "TracerRegistry key sets diverge across ranks at load "
                "time; every rank must already have the keys the "
                "checkpoint contains before load_checkpoint runs.  "
                "Rank-by-rank missing keys: "
                f"{all_missing!r}."
            )

        # ----------------------------------------------------------
        # Stage B — scatter state fields.  Past this point self.state
        # is mutated; failures from now on are non-atomic but every
        # error path that follows is a JAX shape/dtype regression of
        # the helper itself (a code bug, not a data-dependent failure).
        # ----------------------------------------------------------
        if self._mpi_rank == 0:
            s = state_global
            T_g = s.T.data
            u_g = s.u.data
            v_g = s.v.data
            p_g = s.p_s.data
            phi_g = s.phis.data
        else:
            s = self.state  # use rank-local Field templates for .replace()
            T_g = u_g = v_g = p_g = phi_g = None

        T_loc = scatter_field_latlon(T_g, self._layout)
        u_loc = scatter_field_latlon(u_g, self._layout)
        v_loc = scatter_field_latlon(v_g, self._layout, is_v_face=True)
        p_loc = scatter_field_latlon(p_g, self._layout)
        phi_loc = scatter_field_latlon(phi_g, self._layout)

        self.state = s._replace(
            T=s.T.replace(data=T_loc),
            u=s.u.replace(data=u_loc),
            v=s.v.replace(data=v_loc),
            p_s=s.p_s.replace(data=p_loc),
            phis=s.phis.replace(data=phi_loc),
        )

        # ----------------------------------------------------------
        # Stage C — scatter tracers.  Key set already validated in
        # Stage A so this loop is just per-tracer slicing.
        # ----------------------------------------------------------
        for name in tracer_keys:
            global_arr = tracers_global[name] if self._mpi_rank == 0 else None
            self.tracers[name] = scatter_field_latlon(
                global_arr, self._layout,
            )

    def _write_blowup_state(self, step: int, day: float) -> None:
        """Persist the FAILING state as ``blowup_state_day_XXXX.npz`` for autopsy.

        Blowups used to discard the non-finite state ("caught at checkpoint;
        not written"), leaving nothing to inspect — the #871 hunt had only the
        last *healthy* checkpoint.  Reuse the canonical writer, then RENAME the
        produced ``checkpoint_day_XXXX.npz`` (+ ``.meta.json``) so (a) every
        backend branch of ``save_checkpoint`` is covered without touching its
        naming, and (b) the restart-chain glob (``checkpoint_day_*.npz``) can
        NEVER auto-resume from the poisoned state.  Fail-open: a dump failure
        must not mask the BLOWUP status itself.
        """
        # ``save_checkpoint`` branches use TWO filename conventions (codex):
        # the MPAS branch writes the ABSOLUTE rounded day, the generic branch
        # writes the ELAPSED truncated day (day - config.start_day).  Don't
        # guess which fires — protect BOTH candidates, then detect which one
        # the writer actually produced.
        _start = float(getattr(self.config, "start_day", 0.0) or 0.0)
        _days = {int(round(day)), int(day - _start)}
        # Both filename CONVENTIONS x both FORMS: single-file ``.npz`` and the
        # distributed per-rank checkpoint DIRECTORY (``checkpoint_day_NNNN/``)
        # — Path.rename moves a directory just like a file.
        candidates = [self._output_dir / f"checkpoint_day_{d:04d}{suf}"
                      for d in sorted(_days) for suf in (".npz", "")]
        # Filesystem ops are ROOT-ONLY and BEST-EFFORT (each in its own try):
        # under MPI every rank calls this helper (post-bcast) and the writer
        # is collective, so a rename raced/failed on one rank must NEVER make
        # that rank skip ``save_checkpoint`` while the others enter it — that
        # hangs the collective (codex critical).  All ranks always reach the
        # save call; only root touches files.
        _is_root = getattr(self, "_mpi_rank", None) in (None, 0)
        backups: list[tuple] = []
        if _is_root:
            # A HEALTHY checkpoint can already exist under a candidate name
            # (daily-print blowup at day N.x after the periodic write at N.0).
            # Move every existing candidate (+meta) aside so the sick-state
            # write can't clobber it; restored below.
            try:
                for c in candidates:
                    for p in (c, c.with_name(c.stem + ".meta.json")):
                        if p.exists():
                            b = p.with_name(p.name + ".pre_blowup")
                            p.rename(b)
                            backups.append((b, p))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Blow-up dump: backup step failed: {exc!r}")
        try:
            self.save_checkpoint(step, day)
        except Exception as exc:  # noqa: BLE001 — forensics must never mask the blowup
            logger.warning(f"Blow-up dump: state write failed (non-fatal): {exc!r}")
        if _is_root:
            try:
                src = next((c for c in candidates if c.exists()), None)
                if src is not None:
                    dst = src.with_name(src.name.replace(
                        "checkpoint_day_", "blowup_state_day_"))
                    src.rename(dst)
                    src_meta = src.with_name(src.stem + ".meta.json")
                    if src_meta.exists():
                        src_meta.rename(dst.with_name(dst.stem + ".meta.json"))
                    logger.error(f"Blow-up state written for autopsy: {dst}")
                else:
                    logger.warning(
                        "Blow-up dump: writer produced none of "
                        f"{[c.name for c in candidates]} (distributed/custom "
                        "layout?); state not saved.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Blow-up dump failed (non-fatal): {exc!r}")
            for b, p in backups:   # ALWAYS restore the healthy checkpoints
                try:
                    b.rename(p)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Blow-up dump: could not restore {p.name} from "
                        f"{b.name}: {exc!r}")

    def save_checkpoint(self, step: int, day: float) -> None:
        """Save checkpoint to output directory using unified restart API.

        When running under MPI with partitioned state, uses per-rank
        distributed checkpoint so every rank writes its own partition
        concurrently (no barrier).  For lat-lon band MPI, gathers all
        ranks' bands onto rank 0 and writes a single global ``.npz`` so
        ``--restart-from`` and the yearly-map plotter see a single
        canonical file.  For replicated state, only rank 0 writes and
        signals completion via a lightweight barrier.
        """
        self._last_checkpoint_step = step
        elapsed_day = day - self.config.start_day

        # MPAS path: the Voronoi hydrostatic state is ``u`` (edge-normal,
        # shape (nEdges, nlev)) + T/p_s/phis on cells, with ``v=None`` and
        # ``tracers=None``.  The shared ``save_restart`` assumes the
        # cubed-sphere/lat-lon layout (it reads ``state.v.data`` and infers
        # resolution from a (6, n, n, nlev) T-shape), so it cannot
        # serialise an MPAS state.  Write the four prognostic arrays
        # directly, keeping the ``checkpoint_day_NNNN.npz`` filename the
        # restart-chain sbatch globs.  Single-process only (the _run_mpas
        # path is not MPI-sharded).
        #
        # Filename uses the ABSOLUTE simulated ``day`` (not
        # ``day - config.start_day``): across a chained 100-yr run each
        # link resumes at the prior link's absolute day, so absolute-day
        # filenames stay strictly monotonic and never collide even if a
        # job is (mis)configured with a non-zero ``--start-day``.  The
        # restart-chain launcher's ``TARGET_DAYS`` / latest-checkpoint glob
        # are therefore in absolute simulated days.
        if self.config.grid.grid_type == "mpas":
            ckpt_path = self._output_dir / f"checkpoint_day_{int(round(day)):04d}.npz"
            s = self.state
            _ps_carry = getattr(self, "_mpas_phys_state", None)
            # Codex adversarial (#413): never LAUNDER a bad restart.
            # ``load_checkpoint`` stages physstate_* into _carry_aux but
            # does NOT validate/adopt it into the save channel — only a run
            # (``_run_mpas``, which seeds + shape/dtype/scheme/completeness-
            # validates the overlay) sets ``_mpas_phys_state``.  So if
            # staged carry is present but the channel is still None, no run
            # has validated it; saving now would emit either a
            # physstate-free file (read as a fresh seed next restart) or an
            # unvalidated/relabelled carry — silently branching the
            # trajectory.  Refuse instead: run the model before
            # checkpointing, or strip ALL physstate_* to opt into a fresh
            # seed.
            if (_ps_carry is None
                    and isinstance(self._carry_aux, dict)
                    and any(k.startswith("physstate_")
                            for k in self._carry_aux)):
                raise ValueError(
                    "save_checkpoint: the loaded MPAS checkpoint staged "
                    "physstate_* carry that was NOT adopted into the save "
                    "channel (incomplete, meta-only, or a convection-scheme "
                    "mismatch with the configured run).  Saving now would "
                    "launder a corrupted / mismatched restart into a "
                    "plausible checkpoint and silently branch the trajectory "
                    "(issue #405/#413).  Run the model before checkpointing, "
                    "or strip ALL physstate_* entries to opt into a fresh "
                    "seed."
                )
            # Under MPAS cell-partition MPI each rank holds only its owned+halo
            # band; gather the owned cells/edges into the GLOBAL field on rank 0
            # so the restart chain reads a single canonical global checkpoint
            # (mirrors the lat-lon band gather).  All ranks must participate in
            # each gather (collective); non-root ranks then bail before I/O.
            if self._voronoi_layout is not None:
                from legoesm.parallel.voronoi_mpi import gather_voronoi_field
                part = self._voronoi_layout.partition
                u_d = gather_voronoi_field(s.u.data, part, "edge")
                T_d = gather_voronoi_field(s.T.data, part, "cell")
                ps_d = gather_voronoi_field(s.p_s.data, part, "cell")
                phis_d = gather_voronoi_field(s.phis.data, part, "cell")
                trc_d = (None if s.tracers is None else {
                    _k: gather_voronoi_field(s.tracers[_k].data, part, "cell")
                    for _k in s.tracers})
                # Stateful-physics carry (#413): every PhysicsState field
                # except the (replicated) PRNG key is cell-dimensioned —
                # gather owned cells like the state.  The 3-D GWD
                # spectrum gathers through a (nCells, az*wn) reshape.
                ps_d_carry = None
                if _ps_carry is not None:
                    ps_d_carry = {}
                    for _name in _ps_carry._fields:
                        _val = getattr(_ps_carry, _name)
                        if _name == "prng_key":
                            ps_d_carry[_name] = _val
                        elif _val.ndim == 3:
                            _az_wn = _val.shape[1:]
                            _flat = gather_voronoi_field(
                                _val.reshape(_val.shape[0], -1),
                                part, "cell",
                            )
                            ps_d_carry[_name] = _flat.reshape(
                                (-1,) + _az_wn)
                        else:
                            ps_d_carry[_name] = gather_voronoi_field(
                                _val, part, "cell")
                if self._mpi_rank != 0:
                    return
            else:
                u_d, T_d, ps_d, phis_d = (
                    s.u.data, s.T.data, s.p_s.data, s.phis.data)
                trc_d = (None if s.tracers is None
                         else {_k: s.tracers[_k].data for _k in s.tracers})
                ps_d_carry = (None if _ps_carry is None
                              else _ps_carry._asdict())
            _save = dict(
                u=np.asarray(u_d), T=np.asarray(T_d),
                p_s=np.asarray(ps_d), phis=np.asarray(phis_d),
                step=np.asarray(int(step)), day=np.asarray(float(day)),
            )
            # Stateful-physics carry (#413): persisted under
            # ``physstate_<field>`` so a chained restart resumes the
            # prognostic physics memory instead of silently reseeding.
            # The convection scheme tag travels with it (codex round 8:
            # the profile-prognostic schemes share the carry shape, so
            # shape checks alone cannot catch a cross-scheme restore).
            if ps_d_carry is not None:
                for _name, _val in ps_d_carry.items():
                    _save[f"physstate_{_name}"] = np.asarray(_val)
                _save["physstate_meta_conv_scheme"] = np.asarray(
                    str(getattr(self.config, "convection", "none")))
            # Persist moisture tracers too (moist MPAS runs), so a chained
            # restart does not silently drop water.  The ``trc_`` prefix avoids
            # colliding with u/T/p_s/phis; ``tracer_names`` lets load rebuild
            # the dict.  Dry runs (tracers=None) write neither and are
            # byte-identical to before.
            if trc_d is not None:
                _save["tracer_names"] = np.asarray(sorted(trc_d.keys()))
                for _k in trc_d:
                    _save[f"trc_{_k}"] = np.asarray(trc_d[_k])
            np.savez(ckpt_path, **_save)
            logger.info(f"  Checkpoint: {ckpt_path.name} (mpas)")
            self._save_cmor_accumulator_sidecar(day)
            return

        # Spectral path (FIX_RESTART_TIME iteration 4): the spectral PE
        # state is five complex coefficient Fields (vor/div/T/lnps/phis
        # ``_hat``) + an optional grid-space tracers dict — the shared
        # ``save_restart`` assumes the cube/lat-lon layout (reads
        # ``state.v.data``, infers resolution from the T-shape) and
        # cannot serialise it.  Write the coefficient arrays directly
        # (npz handles complex128 natively) under the absolute-day
        # filename the restart chain globs (MPAS convention).  The
        # spectral loop refuses stateful physics and holds no
        # held-radiation carry, so this payload is complete for
        # bit-exact continuation.  Single-process only (spectral
        # transforms are global; the path is never MPI-sharded).
        if self.config.dycore.discretization == "spectral":
            ckpt_path = (self._output_dir
                         / f"checkpoint_day_{int(round(day)):04d}.npz")
            s = self.state
            _save = dict(
                vor_hat=np.asarray(s.vor_hat.data),
                div_hat=np.asarray(s.div_hat.data),
                T_hat=np.asarray(s.T_hat.data),
                lnps_hat=np.asarray(s.lnps_hat.data),
                phis_hat=np.asarray(s.phis_hat.data),
                step=np.asarray(int(step)), day=np.asarray(float(day)),
                spectral_layout=np.asarray(1),
            )
            if s.tracers is not None:
                _save["tracer_names"] = np.asarray(sorted(s.tracers.keys()))
                for _k in s.tracers:
                    _save[f"trc_{_k}"] = np.asarray(s.tracers[_k].data)
            np.savez(ckpt_path, **_save)
            logger.info(f"  Checkpoint: {ckpt_path.name} (spectral)")
            self._save_cmor_accumulator_sidecar(day)
            return

        # Distributed path
        if (self._device_config is not None
                and self._device_config.is_distributed):

            # Partitioned state: each rank writes its own partition
            if self._layout is not None and self._owned_face_ids is not None:
                ckpt_dir = self._output_dir / f"checkpoint_day_{int(elapsed_day):04d}"
                from legoesm.driver.distributed_checkpoint import save_checkpoint_distributed
                save_checkpoint_distributed(
                    path=ckpt_dir,
                    state=self.state,
                    rank=self._mpi_rank,
                    n_ranks=self._mpi_world_size,
                    step=step,
                    day=day,
                    config=self.config,
                    q_v=self.q_v,
                    q_c=self.q_c,
                    q_r=self.q_r,
                    # Per-rank distributed: carry_aux is rank-local, so the
                    # evolved double-moment tracers ride it safely (no gather).
                    diag_accumulators=self._checkpoint_carry_aux(),
                )
                # Lightweight barrier: only needed so rank 0's metadata.json
                # is flushed before any rank tries to load the checkpoint.
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
                if self._mpi_rank == 0:
                    logger.info(f"  Checkpoint: {ckpt_dir.name} (distributed, {self._mpi_world_size} ranks)")
                    # Rank 0 writes the single CMOR sidecar next to the
                    # per-rank checkpoint dir (an SPMD restart would otherwise
                    # find no sidecar and lose the in-progress month).
                    self._save_cmor_accumulator_sidecar(day)
                return

            # Lat-lon band MPI: gather rank-local bands → rank 0 writes
            # a single global-shape ``.npz``.  This matches the format
            # expected by ``--restart-from`` in run_amip.py and the
            # checkpoint-glob in ``run_amip_1deg_latlon_mpi.sbatch``.
            if self._is_latlon_mpi():
                # The lat-lon MPI gather path writes a single global-shape
                # checkpoint from rank 0's carry_aux, which holds the
                # prognostic slab-land T_land as a rank-LOCAL latitude
                # band.  Broadcasting that on restart would give every
                # rank rank-0's band (wrong shape/values) and corrupt the
                # land surface.  Banded carry-aux gather/scatter is not
                # implemented yet, so fail fast rather than write an
                # invalid restart (#325). The per-rank distributed format
                # (each rank saves/loads its own band) and single-process
                # npz are restart-exact for slab-land.
                if (self.physics is not None
                        and getattr(self.physics, "f_land", None) is not None):
                    raise ValueError(
                        "Lat-lon MPI checkpointing does not yet gather the "
                        "banded slab-land temperature (T_land) into the "
                        "global checkpoint — a restart would corrupt land "
                        "surface state (#325). Use the per-rank distributed "
                        "checkpoint format or run single-process for "
                        "slab-land lat-lon MPI runs."
                    )
                # Same banded-carry_aux limitation for double-moment tracers:
                # q_i/q_s/q_g/N_c/N_r/N_i ride carry_aux as rank-local bands and
                # are not gathered into the global checkpoint, so a restart would
                # corrupt them. Fail fast rather than write an invalid restart.
                if (isinstance(self.tracers, dict)
                        and self.tracer_registry.has("q_i")):
                    raise ValueError(
                        "Lat-lon MPI checkpointing does not yet gather the "
                        "banded double-moment tracers (q_i/q_s/q_g/N_c/N_r/N_i) "
                        "into the global checkpoint — a restart would corrupt "
                        "them. Use the per-rank distributed checkpoint format "
                        "or run single-process for double-moment lat-lon MPI "
                        "runs."
                    )
                # Same limitation for the stateful-physics carries (#413):
                # tke/qke/gwd_spectrum — and conv_prog when the configured
                # convection is a real carry (scalar-/profile-prognostic
                # or stochastic; codex round 7) — ride carry_aux
                # rank-locally and are not gathered; a restart would
                # corrupt the physics memory.
                from legoesm.atmosphere.physics.convection.integration \
                    import convection_scheme_traits as _conv_traits
                _ct = _conv_traits(getattr(self.config, "convection", "none"))
                _conv_is_carry = (_ct.is_scalar_prognostic
                                  or _ct.is_profile_prognostic
                                  or _ct.is_stochastic)
                if (isinstance(self._carry_aux, dict)
                        and (any(k in self._carry_aux
                                 for k in ("tke", "qke", "gwd_spectrum"))
                             or (_conv_is_carry
                                 and "conv_prog" in self._carry_aux))):
                    raise ValueError(
                        "Lat-lon MPI checkpointing does not yet gather the "
                        "rank-local stateful-physics carries "
                        "(tke/qke/gwd_spectrum/conv_prog) into the global "
                        "checkpoint — a restart would corrupt the "
                        "prognostic physics memory (issue #405/#413). Use "
                        "the per-rank distributed checkpoint format or run "
                        "single-process for stateful-physics lat-lon MPI "
                        "runs."
                    )
                state_g, tracers_g = self._gather_state_for_global_checkpoint()
                if self._mpi_rank == 0:
                    ckpt_path = (
                        self._output_dir
                        / f"checkpoint_day_{int(elapsed_day):04d}.npz"
                    )
                    save_restart(
                        path=ckpt_path,
                        state=state_g,
                        q_v=tracers_g.get("q_v"),
                        step=step,
                        day=day,
                        config=self.config,
                        q_c=tracers_g.get("q_c"),
                        q_r=tracers_g.get("q_r"),
                        carry_aux=self._carry_aux,
                    )
                    logger.info(
                        f"  Checkpoint: {ckpt_path.name} "
                        f"(lat-lon MPI gathered, "
                        f"{self._mpi_world_size} ranks)"
                    )
                    self._save_cmor_accumulator_sidecar(day)
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
                return

            # Replicated state: only rank 0 writes
            if self._mpi_rank == 0:
                ckpt_path = self._output_dir / f"checkpoint_day_{int(elapsed_day):04d}.npz"
                save_restart(
                    path=ckpt_path,
                    state=self.state,
                    q_v=self.q_v,
                    step=step,
                    day=day,
                    config=self.config,
                    q_c=self.q_c,
                    q_r=self.q_r,
                    # Replicated state: rank-0's full fields → DM tracers ride
                    # carry_aux (same full arrays on every rank).
                    carry_aux=self._checkpoint_carry_aux(),
                )
                logger.info(f"  Checkpoint: {ckpt_path.name} (rank 0)")
                self._save_cmor_accumulator_sidecar(day)
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()
            return

        # Single-process path (also the multi-controller SPMD write tail:
        # the state is gathered to a host replica first, process 0 writes).
        _state, _q_v, _q_c, _q_r = self.state, self.q_v, self.q_c, self.q_r
        _carry_aux = self._checkpoint_carry_aux()
        if self._is_spmd_multiprocess():
            # Multi-controller SPMD (cs_spmd step 5a): every process holds
            # only its shard of the face-sharded global arrays, and every
            # process runs this method in lockstep.  Gather each leaf to a
            # host-replicated array on ALL processes (process_allgather is
            # a COLLECTIVE — a rank-0-only gather would desync the
            # program), then only process 0 falls through to the standard
            # single-file write below.  Parity receipt for the gather
            # pattern: validate_driver_cs_spmd_parity.py, job 8686550.
            _gather = self._gather_spmd_tree_to_host
            _state = _gather(_state)
            _q_v, _q_c, _q_r = _gather(_q_v), _gather(_q_c), _gather(_q_r)
            _carry_aux = _gather(_carry_aux)
            # NOTE: do NOT return on non-root here.  The zarr carry guards
            # below are pure config/pytree checks that must raise
            # IDENTICALLY on every process (codex HIGH: a rank-0-only
            # raise after rank 1 already returned leaves rank 1 running
            # toward the next collective — a hang).  Non-root returns just
            # before the actual write instead.
            _spmd_nonroot = jax.process_index() != 0
        else:
            _spmd_nonroot = False

        ckpt_path = self._output_dir / f"checkpoint_day_{int(elapsed_day):04d}.npz"
        backend = self.config.output.checkpoint_format if hasattr(self.config.output, 'checkpoint_format') else "npz"

        # The zarr backend does not yet round-trip ``carry_aux`` (it is
        # dropped on save and padded empty on load), so it cannot persist
        # the prognostic slab-land skin temperature.  Restarting a
        # slab-land run from a zarr checkpoint would silently reinitialize
        # T_land from the lowest-level air temperature and branch the
        # trajectory.  Fail fast rather than corrupt restart (#325); npz
        # persists carry_aux (incl. T_land) and is restart-exact.
        if (
            backend == "zarr"
            and self.physics is not None
            and getattr(self.physics, "f_land", None) is not None
        ):
            raise ValueError(
                "checkpoint_format='zarr' cannot persist the prognostic "
                "slab-land temperature (T_land) — a restart would silently "
                "reinitialize it (#325). Use checkpoint_format='npz' for "
                "slab-land runs (or add carry_aux support to the zarr "
                "backend)."
            )
        # Same zarr carry_aux limitation for the evolved double-moment tracers:
        # they ride carry_aux (dmtr_*), which zarr does not round-trip, so a
        # zarr restart would silently reset q_i/q_s/q_g/N_c/N_r/N_i to zeros.
        if (
            backend == "zarr"
            and isinstance(self.tracers, dict)
            and self.tracer_registry.has("q_i")
        ):
            raise ValueError(
                "checkpoint_format='zarr' cannot persist the prognostic "
                "double-moment tracers (q_i/q_s/q_g/N_c/N_r/N_i) — they ride "
                "carry_aux, which the zarr backend does not round-trip, so a "
                "restart would silently reinitialize them. Use "
                "checkpoint_format='npz' for double-moment runs."
            )
        # Same zarr carry_aux limitation for the stateful-physics carries
        # (issue #413): tke/qke/gwd_spectrum — and conv_prog when the
        # configured convection is a real carry (codex round 7) — ride
        # carry_aux, so a zarr restart would silently reseed the
        # prognostic physics memory.
        from legoesm.atmosphere.physics.convection.integration import (
            convection_scheme_traits as _conv_traits,
        )
        _ct = _conv_traits(getattr(self.config, "convection", "none"))
        _conv_is_carry = (_ct.is_scalar_prognostic
                          or _ct.is_profile_prognostic
                          or _ct.is_stochastic)
        if (
            backend == "zarr"
            and isinstance(self._carry_aux, dict)
            and (any(k in self._carry_aux
                     for k in ("tke", "qke", "gwd_spectrum"))
                 or (_conv_is_carry and "conv_prog" in self._carry_aux))
        ):
            raise ValueError(
                "checkpoint_format='zarr' cannot persist the stateful-"
                "physics carries (tke/qke/gwd_spectrum/conv_prog) — they "
                "ride carry_aux, which the zarr backend does not "
                "round-trip, so a restart would silently reseed the "
                "prognostic physics state (issue #405/#413). Use "
                "checkpoint_format='npz' for stateful-physics runs."
            )

        # Multi-controller SPMD: every process ran the collective gather
        # and the (identical) guards above; only process 0 writes.  The
        # write itself is wrapped so a root-only I/O failure reaches every
        # process via the rendezvous below (codex HIGH: otherwise non-root
        # returned here and hung on the run loop's next collective while
        # root died).
        _write_err: Exception | None = None
        if not _spmd_nonroot:
            try:
                save_restart(
                    path=ckpt_path,
                    state=_state,
                    q_v=_q_v,
                    step=step,
                    day=day,
                    config=self.config,
                    q_c=_q_c,
                    q_r=_q_r,
                    carry_aux=_carry_aux,
                    backend=backend,
                )
                logger.info(f"  Checkpoint: {ckpt_path.name}")
                self._save_cmor_accumulator_sidecar(day)
            except Exception as e:
                _write_err = e
        self._spmd_barrier_on_root_error(_write_err)

    def _save_cmor_accumulator_sidecar(self, day: float) -> None:
        """Persist the CMOR monthly/daily accumulator state to a sidecar next
        to the checkpoint just written.

        A restart-chain link is only ~10 days but a calendar month is ~30, so
        the monthly accumulator — recreated empty on every restart — never
        completes a month and the CMOR ``Amon`` means are lost.  This additive
        sidecar (``cmor_accum_day_<day>.npz``) lets the next link resume the
        in-progress month.  It never touches the checkpoint ``.npz`` schema and
        is fully guarded so a sidecar-write failure can never abort
        checkpointing (a no-op when CMIP output is off).

        Suppressed via ``self._suppress_cmor_sidecar`` for the TERMINAL final
        checkpoint of a COMPLETED run: ``diagnostics.save`` has already written
        every bucket to NetCDF (without popping), so a sidecar there would make
        a run-extending restart re-append the already-written months."""
        if getattr(self, "_suppress_cmor_sidecar", False):
            return
        diag = getattr(self, "diagnostics", None)
        if diag is None:
            return
        try:
            sidecar = (self._output_dir
                       / f"cmor_accum_day_{int(round(day)):04d}.npz")
            diag.save_cmor_accumulators(sidecar)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            # LOUD (the checkpoint is already committed; a missing/stale sidecar
            # can lose or duplicate the in-progress month on restart) but never
            # fatal — a sidecar failure must not abort the run.
            logger.error(
                f"  CMOR accumulator sidecar FAILED for day {day:.2f} "
                f"(checkpoint committed; restart may lose/duplicate the "
                f"in-progress month): {exc}")

    def _moisture_advection_active(self) -> bool:
        """True iff resolved-wind moisture advection is on (issue #771).

        Config-gated to the tracer-capable dycore: cubed_sphere + cdgrid
        (the only PE step whose RHS advects ``state.tracers``).  Other
        grid/discretization combos keep the legacy column-locked moisture
        and log a notice once so the gap is visible, not silent.
        """
        cfg = self.config
        if not getattr(cfg, "moisture_advection", False):
            return False
        # ``centered``/``finite_volume`` resolve to the cdgrid PE dycore (see
        # atmosphere.dynamics DISPATCH), so they are tracer-capable too — the
        # gate must accept them or an opt-in run on those aliases would drop to
        # the legacy column-locked path despite running a cdgrid step (#771).
        supported = (cfg.grid.grid_type == "cubed_sphere"
                     and cfg.dycore.discretization
                     in ("cdgrid", "centered", "finite_volume"))
        if not supported and not getattr(self, "_warned_no_advection", False):
            self._warned_no_advection = True
            logger.info(
                "  moisture_advection: not available on %s/%s (tracer "
                "advection is wired for cubed_sphere+cdgrid only, #771) — "
                "running legacy column-locked moisture.",
                cfg.grid.grid_type, cfg.dycore.discretization)
        return supported

    def _is_spmd_multiprocess(self) -> bool:
        """True iff this run is multi-controller SPMD across >1 process
        (distributed_mode='spmd' under a real multi-process launch) — the
        regime where output writes are root-gated and every gather is a
        collective."""
        return (self.config.distributed
                and getattr(self.config, "distributed_mode", "mpi") == "spmd"
                and jax.process_count() > 1)

    def _spmd_barrier_on_root_error(self, err: Exception | None) -> None:
        """Rendezvous all SPMD processes on the success of a ROOT-ONLY write.

        Root-gated I/O (diagnostics.save / save_results / the checkpoint
        save_restart tail) runs on process 0 only.  Without a rendezvous, a
        root-only exception kills process 0 while the other processes sail
        into the NEXT collective (segment scan, process_allgather) and hang
        until walltime (codex 2026-07-03 HIGH).  Every process calls this
        with its local error (non-root: None); the root flag is broadcast
        and EVERY process raises when root failed.  No-op outside
        multi-process SPMD (single process / mpi4jax keep their native
        exception flow)."""
        if not self._is_spmd_multiprocess():
            if err is not None:
                raise err
            return
        from jax.experimental import multihost_utils as _mhu
        # allgather+max (not broadcast_one_to_all): symmetric — ANY
        # process's failure surfaces on every process, not just root's.
        _flags = _mhu.process_allgather(
            jnp.asarray(0.0 if err is None else 1.0))
        if err is not None:
            raise err
        if float(jnp.max(_flags)) != 0.0:
            raise RuntimeError(
                "multi-controller SPMD: another process failed during a "
                "root-gated output write (see its traceback); aborting "
                "this process in lockstep instead of hanging on the next "
                "collective."
            )

    def _gather_spmd_tree_to_host(self, tree):
        """Gather every non-fully-addressable jax.Array leaf of *tree* to a
        process-local REPLICATED jax array (multi-controller SPMD;
        collective — EVERY process must call this with the same tree).
        Fully-addressable leaves and non-array leaves pass through
        unchanged; pytree structure (Fields, dicts, NamedTuples) is
        preserved.

        The gathered leaf is re-wrapped ``jnp.asarray`` (NOT left as
        numpy): downstream consumers include jnp/``lax.scan`` code (the
        energy tracker inside the full diagnostics ``collect()`` indexes
        with traced integers — a numpy leaf there raises
        ``TracerArrayConversionError``, smoke job 8687797) as well as
        plain ``np.asarray`` writers, and a single-device jax array
        serves both."""
        if tree is None:
            return None
        import numpy as _np
        from jax.experimental import multihost_utils as _mhu

        def _leaf(x):
            if isinstance(x, jax.Array) and not x.is_fully_addressable:
                return jnp.asarray(
                    _np.asarray(_mhu.process_allgather(x, tiled=True)))
            return x

        return jax.tree_util.tree_map(_leaf, tree)

    def load_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Load state from a checkpoint using unified restart API.

        Returns (step, day).  Detects distributed checkpoint directories
        and loads per-rank data when running under MPI.
        """
        path = Path(path)

        # MPAS path: mirror of the dedicated MPAS branch in
        # ``save_checkpoint``.  Reconstruct the Voronoi ``HydrostaticState``
        # (u edge-normal, T/p_s/phis on cells; v=None, tracers=None) from
        # the four-array npz and return (step, day) so the chained job
        # continues from the saved absolute day.
        if self.config.grid.grid_type == "mpas":
            # Fail loud on a missing/dir path rather than silently falling
            # through to the cube/lat-lon ``load_restart`` (which would
            # raise a confusing non-MPAS error).
            if not path.is_file():
                raise FileNotFoundError(
                    f"MPAS checkpoint not found (or is a directory): {path}"
                )
            import jax.numpy as jnp
            from legoesm.core.state import HydrostaticState
            from legoesm.core.field import Field
            d = np.load(path)
            # Under MPAS cell-partition MPI the checkpoint is GLOBAL but
            # ``self.state`` is this rank's local (owned+halo) band, so scatter
            # the global arrays to local cells/edges (mirror of the save-side
            # gather).  Serial runs use the global arrays directly.  The shape
            # guard compares against the GLOBAL mesh size under MPI, the local
            # state otherwise.
            _mpi = self._voronoi_layout is not None
            if _mpi:
                from legoesm.parallel.voronoi_partition import scatter_to_local
                part = self._voronoi_layout.partition
                _guard = (("u", part.nEdges_global, self.state.u.data.shape[1:]),
                          ("T", part.nCells_global, self.state.T.data.shape[1:]),
                          ("p_s", part.nCells_global, self.state.p_s.data.shape[1:]),
                          ("phis", part.nCells_global,
                           self.state.phis.data.shape[1:]))
                for _name, _n_global, _trail in _guard:
                    if (d[_name].shape[0] != _n_global
                            or tuple(d[_name].shape[1:]) != tuple(_trail)):
                        raise ValueError(
                            f"MPAS checkpoint {path.name} {_name}-shape "
                            f"{tuple(d[_name].shape)} != global mesh "
                            f"({_n_global}, {tuple(_trail)}); rebuild with the "
                            f"same --resolution/--nlev.")

                def _scatter(name, entity):
                    return scatter_to_local(jnp.asarray(d[name]), part, entity)
                _u, _T, _ps, _phis = (_scatter("u", "edge"),
                                      _scatter("T", "cell"),
                                      _scatter("p_s", "cell"),
                                      _scatter("phis", "cell"))
            else:
                # Shape guard on EVERY prognostic field: the mesh built from
                # --resolution/--nlev must match the checkpoint, else the TRiSK
                # gathers index out of range (u) or broadcast wrong (T/p_s/phis)
                # — both silent.  Reject a corrupt/mismatched file up front.
                for _name, _ck in (("u", self.state.u.data),
                                   ("T", self.state.T.data),
                                   ("p_s", self.state.p_s.data),
                                   ("phis", self.state.phis.data)):
                    if tuple(d[_name].shape) != tuple(_ck.shape):
                        raise ValueError(
                            f"MPAS checkpoint {path.name} {_name}-shape "
                            f"{tuple(d[_name].shape)} != current mesh "
                            f"{_name}-shape {tuple(_ck.shape)}; rebuild with the "
                            f"same --resolution/--nlev."
                        )
                _u, _T, _ps, _phis = (jnp.asarray(d["u"]), jnp.asarray(d["T"]),
                                      jnp.asarray(d["p_s"]), jnp.asarray(d["phis"]))
            self.state = HydrostaticState(
                u=Field(data=_u, name="u",
                        dims=("nEdges", "nlev"), units="m/s"),
                T=Field(data=_T, name="T",
                        dims=("nCells", "nlev"), units="K"),
                p_s=Field(data=_ps, name="p_s",
                          dims=("nCells",), units="Pa"),
                phis=Field(data=_phis, name="phis",
                           dims=("nCells",), units="m2/s2"),
            )
            # Restore moisture tracers (moist MPAS runs); absent ⇒ dry restart.
            if "tracer_names" in d:
                _names = [str(n) for n in d["tracer_names"]]
                self.state = self.state._replace(tracers={
                    _k: Field(
                        data=(scatter_to_local(
                                  jnp.asarray(d[f"trc_{_k}"]), part, "cell")
                              if _mpi else jnp.asarray(d[f"trc_{_k}"])),
                        name=_k, dims=("nCells", "nlev"), units="kg/kg")
                    for _k in _names
                })
            # Restore the stateful-physics carry (#413): stash the
            # ``physstate_<field>`` arrays into carry_aux for the
            # _run_mpas seed overlay.  Cell-dimensioned fields scatter to
            # the rank-local band under MPI (the replicated PRNG key does
            # not); the 3-D GWD spectrum scatters via a (nCells, az*wn)
            # reshape, mirroring the save-side gather.
            #
            # Codex adversarial (#413 stale-persistence class): a driver
            # reused across loads must DROP any physstate_* left from a
            # prior checkpoint first, so a checkpoint with no (or only a
            # subset of) physstate fields freshly seeds in _run_mpas
            # instead of silently resuming carry from the WRONG file.
            if not isinstance(self._carry_aux, dict):
                self._carry_aux = {}
            for _stale in [k for k in self._carry_aux
                           if k.startswith("physstate_")]:
                del self._carry_aux[_stale]
            # ...and clear the SAVE channel (``_mpas_phys_state``, read by
            # save_checkpoint) so a stale carry from a PRIOR run on a
            # reused driver cannot leak.  It is left None until a run
            # validates + overlays this checkpoint's staged carry
            # (_run_mpas); a save before then is refused (see
            # save_checkpoint) rather than emitting an unvalidated carry.
            self._mpas_phys_state = None
            _ps_keys = [k for k in d.files if k.startswith("physstate_")]
            if _ps_keys:
                for _k in _ps_keys:
                    _name = _k[len("physstate_"):]
                    if _name.startswith("meta_"):
                        # Plain-string metadata (scheme tag) — no jnp,
                        # no scatter.
                        self._carry_aux[_k] = str(d[_k])
                        continue
                    _val = jnp.asarray(d[_k])
                    if _mpi and _name != "prng_key":
                        if _val.ndim == 3:
                            _az_wn = _val.shape[1:]
                            _val = scatter_to_local(
                                _val.reshape(_val.shape[0], -1),
                                part, "cell",
                            ).reshape((-1,) + _az_wn)
                        else:
                            _val = scatter_to_local(_val, part, "cell")
                    self._carry_aux[_k] = _val
                logger.info(
                    "  Restored physics-state carry fields: %s",
                    sorted(k[len("physstate_"):] for k in _ps_keys),
                )
            # Version-skew guard: a checkpoint carrying a non-meta
            # physstate_* field this build does not know would be SILENTLY
            # dropped by _run_mpas's schema-filtered overlay — an older
            # binary could thus rewrite a newer checkpoint, losing
            # prognostic carry a future reader then fresh-seeds (a silent
            # trajectory branch).  Refuse loudly at the load boundary so
            # BOTH the run and save consumers are protected at one
            # chokepoint.
            from legoesm.atmosphere.physics.physics_state import PhysicsState
            _unknown = sorted(
                _k[len("physstate_"):] for _k in self._carry_aux
                if _k.startswith("physstate_")
                and not _k[len("physstate_"):].startswith("meta_")
                and _k[len("physstate_"):] not in PhysicsState._fields
            )
            if _unknown:
                raise ValueError(
                    f"MPAS restart carries unknown physstate field(s) "
                    f"{_unknown} absent from this build's PhysicsState "
                    f"schema {list(PhysicsState._fields)} — this binary is "
                    "too old to represent them and would silently DROP them "
                    "on a rewrite, branching the trajectory for any newer "
                    "reader (issue #405/#413).  Use a build that understands "
                    "the checkpoint, or strip the unknown physstate_* "
                    "entries to accept the loss explicitly."
                )
            # The staged carry is deliberately NOT reconstructed into the
            # SAVE channel (``_mpas_phys_state``) here.  Validating it
            # faithfully needs the seeded reference — per-field shape +
            # dtype + the GWD-dtype rule + the convection-scheme tag —
            # which _run_mpas already builds; duplicating that at the load
            # boundary led to a string of partial/scheme/version/shape
            # bypasses where a load-then-save would launder a bad carry.
            # Instead the channel stays None after a load that carries
            # physstate_*, and ``save_checkpoint`` REFUSES to emit until a
            # run has validated + overlaid the carry (the authoritative
            # path).  A run-then-save and the fresh-seed opt-out (zero
            # physstate_*) are unaffected; this closes the whole
            # load-then-save laundering class at the save boundary.
            step = int(d["step"])
            day = float(d["day"])
            logger.info(f"  Loaded MPAS checkpoint: step={step}, day={day:.2f}"
                        + ("" if "tracer_names" not in d
                           else f", tracers={[str(n) for n in d['tracer_names']]}"))
            self._loaded_checkpoint_step_day = (step, day)
            return step, day

        # Spectral path (FIX_RESTART_TIME iteration 4): mirror of the
        # spectral branch in ``save_checkpoint``.  Reconstruct the
        # ``SpectralHydrostaticState`` (five complex coefficient Fields
        # + optional grid-space tracers) with Field metadata taken from
        # the current state; the generic ``load_restart`` reads
        # grid-layout fields and cannot deserialise it.
        if self.config.dycore.discretization == "spectral":
            if not path.is_file():
                raise FileNotFoundError(
                    f"spectral checkpoint not found (or is a directory): "
                    f"{path}"
                )
            from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
                reconstruct_spectral_state_from_npz,
            )
            # Shared reconstruction (template=self.state ⇒ reuse the configured Field
            # metadata + validate shapes + refuse to drop water); the standalone
            # ``load_restart`` uses the SAME helper with template=None (iter 92).
            with np.load(path) as d:
                self.state, step, day = reconstruct_spectral_state_from_npz(
                    d, template=self.state)
            logger.info(
                f"  Loaded spectral checkpoint: step={step}, day={day:.2f}")
            self._loaded_checkpoint_step_day = (step, day)
            return step, day

        # Distributed path: directory with per-rank .npz files
        if (path.is_dir()
                and self._device_config is not None
                and self._device_config.is_distributed):
            from legoesm.driver.distributed_checkpoint import (
                load_checkpoint_distributed,
            )
            from legoesm.parallel.distributed import get_active_topology
            topology = get_active_topology()
            if topology is not None:
                arrays, step, day, _, _diag_aux = load_checkpoint_distributed(
                    path, topology.rank, topology.n_processes,
                )
                # Restore carry auxiliaries (held radiation, conv_prog,
                # and the prognostic slab-land T_land) so a distributed
                # restart is trajectory-exact rather than silently
                # reinitializing them (#325 restart-safety).
                self._carry_aux = _diag_aux if _diag_aux else {}
                # Restore evolved double-moment tracers persisted via carry_aux
                # (per-rank distributed checkpoint is restart-exact for them).
                self._restore_dm_tracers_from_carry_aux()
                self._restore_land_ml_from_carry_aux()
                from legoesm.core.state import HydrostaticState
                from legoesm.core.field import Field
                import jax.numpy as jnp
                self.state = HydrostaticState(
                    T=Field(data=jnp.asarray(arrays["T"]),
                            name="T", dims=("face", "x", "y", "level"),
                            units="K"),
                    u=Field(data=jnp.asarray(arrays["u"]),
                            name="u", dims=("face", "x", "y", "level"),
                            units="m/s"),
                    v=Field(data=jnp.asarray(arrays["v"]),
                            name="v", dims=("face", "x", "y", "level"),
                            units="m/s"),
                    p_s=Field(data=jnp.asarray(arrays["p_s"]),
                              name="p_s", dims=("face", "x", "y"),
                              units="Pa"),
                    phis=Field(data=jnp.asarray(arrays["phis"]),
                               name="phis", dims=("face", "x", "y"),
                               units="m2/s2"),
                )
                if "q_v" in arrays:
                    self.q_v = jnp.asarray(arrays["q_v"])
                if "q_c" in arrays:
                    self.q_c = jnp.asarray(arrays["q_c"])
                if "q_r" in arrays:
                    self.q_r = jnp.asarray(arrays["q_r"])
                logger.info(
                    f"  Loaded distributed restart: step={step}, day={day}, "
                    f"rank={topology.rank}"
                )
                self._loaded_checkpoint_step_day = (step, day)
                return step, day

        # Lat-lon band MPI: rank 0 loads the global ``.npz`` against
        # the global grid, then scatters bands to all ranks.  Mirror of
        # the gather path in ``save_checkpoint`` so a chained job picks
        # up exactly where the prior one left off.
        if self._is_latlon_mpi() and path.is_file():
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            state_global = None
            tracers_global = None
            step = 0
            day = 0.0
            carry_aux: dict = {}

            # Codex review round 4 BLOCK fix: rank 0's ``load_restart``
            # can raise on a missing file, a corrupt npz, or a
            # ``.meta.json`` digest mismatch.  If we let that exception
            # propagate locally on rank 0, the non-root ranks would
            # walk into the next ``comm.bcast`` waiting for a payload
            # that never arrives → multi-rank MPI deadlock.  Catch on
            # rank 0, bcast an OK/error status integer FIRST, and have
            # every rank raise in lockstep when rank 0 failed.
            load_error: str | None = None
            if self._mpi_rank == 0:
                local_grid = self.grid
                self.grid = self._grid_global
                try:
                    result = load_restart(
                        path, self.grid, self.sigma, strict=True,
                    )
                    (state_global, q_v_g, step, day,
                     _, _, q_c_g, q_r_g, metadata, carry_aux) = result
                    tracers_global = {"q_v": q_v_g}
                    if q_c_g is not None:
                        tracers_global["q_c"] = q_c_g
                    if q_r_g is not None:
                        tracers_global["q_r"] = q_r_g
                    if metadata:
                        logger.info(
                            f"  Loaded restart: step={step}, day={day}, "
                            f"digest={metadata.state_digest[:16]}... "
                            f"(lat-lon MPI gathered file, will scatter to "
                            f"{self._mpi_world_size} ranks)"
                        )
                except Exception as exc:  # noqa: BLE001 — collective gate
                    load_error = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        f"Rank 0 load_restart failed: {load_error}.  "
                        "Will bcast error status so other ranks raise "
                        "in lockstep instead of deadlocking on the "
                        "next collective."
                    )
                finally:
                    self.grid = local_grid

            # Bcast the rank-0 error status BEFORE any other collective.
            # Every rank sees the same string (None on success).  If
            # non-empty, every rank raises identically.
            load_error = comm.bcast(load_error, root=0)
            if load_error is not None:
                raise RuntimeError(
                    f"Lat-lon MPI restart aborted because rank 0 "
                    f"load_restart failed: {load_error}.  No state "
                    "scatter happened; every rank is at the same "
                    "pre-load state."
                )

            # Broadcast scalar / dict metadata.  State + tracer arrays
            # are scattered band-wise inside the helper, so the heavy
            # arrays do NOT round-trip through rank 0's Python.
            step = comm.bcast(step, root=0)
            day = comm.bcast(day, root=0)
            carry_aux = comm.bcast(carry_aux, root=0)

            # Multilayer (Richards) land state (#769) cannot be band-scattered:
            # the gathered file's land_ml_* are the writer's GLOBAL-column
            # soil/snow/carbon fields, and this band branch never calls
            # ``_restore_land_ml_from_carry_aux`` — so they would sit unrestored
            # in every rank's _carry_aux (the soil silently cold-starts) and, if
            # adopted, hand every rank global-shape columns.  Multilayer land is
            # single-rank-only anyway (the downstream _run_compiled guard aborts
            # multilayer-under-distributed).  Check the just-bcast ``carry_aux``
            # and fail fast BEFORE ``_scatter_global_state_to_bands`` mutates
            # ``self.state`` — every rank has the same bcast dict, so the raise
            # is symmetric (no half-scattered state, no collective deadlock).
            if self._carry_has_unscatterable_land_ml(carry_aux):
                raise ValueError(
                    "Lat-lon MPI restart cannot band-scatter the multilayer "
                    "(Richards) land state (land_ml_*) from a global "
                    "checkpoint — multilayer land is single-rank-only, so the "
                    "gathered soil/snow/carbon columns cannot be partitioned "
                    "onto the bands (the soil would silently cold-start, "
                    "issue #769). Run single-process for multilayer-land runs."
                )

            self._scatter_global_state_to_bands(state_global, tracers_global)
            self._carry_aux = carry_aux if carry_aux else {}
            # Double-moment tracers cannot be band-scattered here: carry_aux is
            # broadcast whole to every rank, so any persisted dmtr_* would give
            # every rank GLOBAL-shape DM state instead of its band. Fail fast
            # (mirrors the save-side guard) rather than corrupt the restart.
            if isinstance(self._carry_aux, dict) and any(
                k.startswith("dmtr_") for k in self._carry_aux
            ):
                raise ValueError(
                    "Lat-lon MPI restart cannot band-scatter double-moment "
                    "tracers (q_i/q_s/q_g/N_c/N_r/N_i) from a global checkpoint "
                    "— use the per-rank distributed checkpoint format or run "
                    "single-process for double-moment lat-lon MPI runs."
                )
            # Same limitation for the stateful-physics carries (#413):
            # the broadcast hands every rank the writer's (global or
            # other-rank) flattened-column fields; restoring them would
            # corrupt or (via the fail-fast shape check in
            # _prepare_run_context) abort the run.  Fail fast HERE with
            # the actionable message.  conv_prog included when the
            # configured convection is a real carry (codex round 7).
            from legoesm.atmosphere.physics.convection.integration import (
                convection_scheme_traits as _conv_traits,
            )
            _ct = _conv_traits(getattr(self.config, "convection", "none"))
            _conv_is_carry = (_ct.is_scalar_prognostic
                              or _ct.is_profile_prognostic
                              or _ct.is_stochastic)
            if isinstance(self._carry_aux, dict) and (
                any(k in self._carry_aux
                    for k in ("tke", "qke", "gwd_spectrum"))
                or (_conv_is_carry and "conv_prog" in self._carry_aux)
            ):
                raise ValueError(
                    "Lat-lon MPI restart cannot band-scatter the "
                    "stateful-physics carries "
                    "(tke/qke/gwd_spectrum/conv_prog) from a global "
                    "checkpoint — the prognostic physics memory would be "
                    "silently reseeded (issue #405/#413). Use the "
                    "per-rank distributed checkpoint format or run "
                    "single-process for stateful-physics lat-lon MPI runs."
                )
            self._loaded_checkpoint_step_day = (step, day)
            return step, day

        # Single-process path
        result = load_restart(
            path, self.grid, self.sigma, strict=True,
        )
        state, q_v, step, day, _, _, q_c, q_r, metadata, carry_aux = result
        self.state = state
        self.q_v = q_v
        if q_c is not None:
            self.q_c = q_c
        if q_r is not None:
            self.q_r = q_r
        self._carry_aux = carry_aux if carry_aux else {}
        # Restore evolved double-moment tracers persisted via carry_aux
        # (serial npz is restart-exact for them).
        self._restore_dm_tracers_from_carry_aux()
        self._restore_land_ml_from_carry_aux()
        if metadata:
            logger.info(f"  Loaded restart: step={step}, day={day}, "
                       f"digest={metadata.state_digest[:16]}...")
        self._loaded_checkpoint_step_day = (step, day)
        return step, day

    def _maybe_wallclock_exit(self, ckpt_fn, step: int, day: float) -> None:
        """Checkpoint and ``exit(0)`` cleanly if the wallclock budget is nearly
        spent, so a SLURM dependency chain resumes from this state.

        Single-rank only: under MPI an independent per-rank ``sys.exit`` would
        desync ranks (others block on the next collective), so a collective
        decision (broadcast the flag) is required and is deferred — the run-start
        warning notes it is inactive under MPI.
        """
        if self._mpi_world_size not in (None, 1):
            return
        max_wall = self.config.output.max_wallclock_seconds
        if not _wallclock_exhausted(
                time.time() - self._run_wallclock_start, max_wall,
                self.config.output.restart_buffer_seconds):
            return
        logger.info(
            f"Wallclock budget {max_wall:.0f}s nearly reached at day {day:.2f}; "
            f"checkpointing and exiting cleanly for restart.")
        if step != self._last_checkpoint_step:
            ckpt_fn(step, day)
        self.diagnostics.flush_to_disk(self._output_dir)
        # Write the CMOR tables that the normal end-of-run ``save`` would emit
        # but this ``sys.exit(0)`` never reaches — flushing only COMPLETED
        # periods so a restart chain does not double-write a boundary period:
        #   * completed months (flush_cmip_monthly pops months < the current),
        #   * completed days   (finalize_cmip_daily pops days < the current),
        #   * the fx table     (areacella/sftlf/orog — static, referenced by
        #                       every variable's ``external_variables`` and
        #                       required for area-weighted / land-ocean-split
        #                       diagnostics).
        # Without this every wallclock-graceful AMIP run (a year rarely
        # finishes in one SLURM window) would drop fx entirely and lose the
        # segment's monthly/daily output.  The month/day STRADDLING the exit is
        # preserved across the restart by the CMOR accumulator sidecar (see the
        # re-persist below), so a chained link completes it instead of losing
        # it.  All three flushes are guarded on the CMIP writer.
        self.diagnostics.flush_cmip_monthly(day, write=True)
        self.diagnostics.finalize_cmip_daily(day)
        self.diagnostics.finalize_cmip_fixed()
        # The checkpoint sidecar written above by ``ckpt_fn`` captured the
        # accumulators PRE-flush (with the just-completed months/days still in
        # them); those are now on disk in the CMOR NetCDF, so re-persist the
        # sidecar to reflect the drained state.  The restart then resumes from
        # the IN-PROGRESS month/day ONLY — otherwise the boundary period would
        # be double-written (duplicate ``time`` coords) on the next link's
        # flush, since ``CFWriter.write_field`` appends blindly.
        self._save_cmor_accumulator_sidecar(day)
        sys.exit(0)

    def run(self, start_step: int = 0, start_day: float | None = None,
            compiled: bool = True, segment_callback=None,
            checkpoint_callback=None) -> str:
        """Run the time integration.

        Parameters
        ----------
        start_step : int
            Starting time step (for restart).
        start_day : float, optional
            Starting day (for restart). Defaults to config.start_day.
        compiled : bool
            If ``True`` (default), use compiled segment execution via
            ``jax.lax.scan``.  If ``False``, use the legacy per-step
            Python loop (useful for debugging or when the compiled path
            is not applicable).
        segment_callback : callable, optional
            Called at each diagnostic interval boundary with
            ``(driver, day, dt_segment)`` for coupled-model integration.

        Returns
        -------
        str
            Run status ("COMPLETED" or "BLOWUP at day ...").
        """
        # SW is not runnable via ModelDriver — reject at the public entry even
        # if a caller reached run() without setup() (codex M2 review).
        self._reject_shallow_water_unrunnable()
        self._segment_callback = segment_callback
        # Checkpoint hook (a coupled driver passes its own save_checkpoint so
        # the FULL coupled state — not just the atmosphere — is written on a
        # periodic or wallclock-budget checkpoint).  Run-start wallclock anchor
        # for the budget check.
        self._checkpoint_callback = checkpoint_callback
        self._run_wallclock_start = time.time()
        if (self.config.output.max_wallclock_seconds > 0
                and self._mpi_world_size not in (None, 1)):
            logger.warning(
                "max_wallclock_seconds is set but the run is MPI-sharded; "
                "wallclock checkpoint-and-exit is single-rank only (a collective "
                "exit is not yet implemented) and will NOT fire under MPI.")
        # Issue #275 fix A: ``try/finally`` here — not inside
        # ``_finalize_run`` — so that an exception thrown anywhere in
        # the time loop still triggers SPMD halo backend restoration.
        # Without this, a Blowup / NaN that propagates out of
        # ``_run_compiled`` would leave the process-global halo
        # backend stuck in ``"spmd"`` state and break subsequent
        # drivers or tests in the same Python process.
        try:
            # MPAS and spectral states use different pytree layouts;
            # use dedicated simple run loops.
            if self.config.grid.grid_type == "mpas":
                status = self._run_mpas(start_step, start_day)
            elif self.config.dycore.discretization == "spectral":
                status = self._run_spectral(start_step, start_day)
            elif (self.config.enable_latlon_spmd
                    and self.config.grid.grid_type == "latlon"):
                # Single-process multi-device lat-band SPMD (A1): a dedicated
                # segment loop over the validated run_atm_latlon_spmd, distinct
                # from the jitted compiled_segments scan (zero surgical risk to
                # the shared hot loop).
                status = self._run_compiled_latlon_spmd(start_step, start_day)
            elif (self.config.grid.grid_type == "cubed_sphere"
                    and self._device_config is not None
                    and self._device_config.mesh is not None
                    and tuple(getattr(self._device_config, "tiling",
                                      (1, 1))) != (1, 1)
                    and not getattr(self.config, "enable_tiled_dycore",
                                    False)):
                # Sub-face-tiled cube SPMD (6*kt^2 > 6 devices): the BLOCKED
                # persistent tiled loop (same dedicated-lane precedent as the
                # lat-lon branch above).  Out-of-envelope configs are refused
                # loudly inside — never a silent fall-through to the
                # non-tile-aware compiled path.  With enable_tiled_dycore the
                # compiled segment IS tile-aware (P4 increment 1b): the run
                # falls through to the compiled lane below, which routes
                # dynamics through make_tiled_cc_step
                # (_maybe_build_tiled_step) instead of this blocked loop.
                status = self._run_tiled_cube_spmd(start_step, start_day)
            elif compiled:
                status = self._run_compiled(start_step, start_day)
            else:
                status = self._run_per_step(start_step, start_day)
            # Record the final-state digest into the run manifest so
            # ``legoesm reproduce --check`` has a bit-repro reference — but ONLY
            # for a clean run: a BLOWUP/failed status must never become a valid
            # reproducibility reference.
            if status == "COMPLETED":
                self._record_final_state_digest()
            return status
        finally:
            # Idempotent — no-op if activation never happened or if
            # ``_finalize_run`` already restored the backend on the
            # success path.
            self._restore_halo_backend()

    # ==================================================================
    # MPAS execution path (uses unified physics pipeline)
    # ==================================================================

    def _mpas_global_diag(self, T_data, p_s_data, u_data, cwv_field):
        """Global owned-cell diagnostics for an MPAS cell-partition MPI run.

        Reduces over this rank's OWNED cells / edges (halo entities masked
        out) then allreduces across ranks, so the lightweight timeseries
        means and extremes are TRUE globals — not the rank-local,
        halo-double-counted values a plain ``jnp.mean(T_data)`` would give.
        Mirrors the owned-mask + allreduce convention of the mass fixer.

        Returns ``(mean_T, mean_ps, max_u, T_min, T_max, T_finite, cwv)`` as
        host floats / bool.  ``cwv`` is NaN when ``cwv_field`` is None.
        """
        from mpi4py import MPI as _MPI
        vl = self._voronoi_layout
        om_c = vl.owned_mask_cells          # (n_local_cells,) bool
        om_e = vl.owned_mask_edges          # (n_local_edges,) bool
        nlev = T_data.shape[-1]
        T_owned = jnp.where(om_c[:, None], T_data, 0.0)
        ps_owned = jnp.where(om_c, p_s_data, 0.0)
        absu_owned = jnp.where(om_e[:, None], jnp.abs(u_data), 0.0)
        T_min_l = jnp.min(jnp.where(om_c[:, None], T_data, jnp.inf))
        T_max_l = jnp.max(jnp.where(om_c[:, None], T_data, -jnp.inf))
        finite_l = jnp.all(jnp.isfinite(T_owned))
        cwv_sum_l = (jnp.sum(jnp.where(om_c, cwv_field, 0.0))
                     if cwv_field is not None else jnp.asarray(0.0))
        # One device→host transfer for all local reductions.
        _loc = np.asarray(jnp.stack([
            jnp.sum(T_owned), jnp.sum(ps_owned), jnp.max(absu_owned),
            T_min_l, T_max_l, finite_l.astype(T_data.dtype),
            cwv_sum_l.astype(T_data.dtype),
        ]))
        comm = _MPI.COMM_WORLD
        # THREE batched buffer allreduces instead of eight scalar pickle
        # rounds (each scalar ``comm.allreduce`` is its own latency-bound
        # collective; at multi-node rank counts the per-diag latency is
        # 8x a single round for no reason).  The finite flag (as a float)
        # rides the MIN batch: all-ranks-finite  <=>  min(finite) == 1.
        _sums = np.array([_loc[0], _loc[1], _loc[6]], dtype=np.float64)
        _maxs = np.array([_loc[2], _loc[4]], dtype=np.float64)
        _mins = np.array([_loc[3], _loc[5]], dtype=np.float64)
        comm.Allreduce(_MPI.IN_PLACE, _sums, op=_MPI.SUM)
        comm.Allreduce(_MPI.IN_PLACE, _maxs, op=_MPI.MAX)
        comm.Allreduce(_MPI.IN_PLACE, _mins, op=_MPI.MIN)
        g_sum_T, g_sum_ps, g_sum_cwv = (float(v) for v in _sums)
        g_max_u, g_T_max = (float(v) for v in _maxs)
        g_T_min, g_finite_min = (float(v) for v in _mins)
        g_finite = bool(g_finite_min > 0.5)
        # Owned-cell count is partition-static: allreduce ONCE and cache.
        if self._mpas_g_n_cells is None:
            self._mpas_g_n_cells = comm.allreduce(
                int(vl.partition.n_owned_cells), op=_MPI.SUM)
        g_n_cells = self._mpas_g_n_cells
        mean_T = g_sum_T / (g_n_cells * nlev)
        mean_ps = g_sum_ps / g_n_cells
        cwv = (g_sum_cwv / g_n_cells) if cwv_field is not None else float("nan")
        return mean_T, mean_ps, g_max_u, g_T_min, g_T_max, g_finite, cwv

    def _insolation_day(self, day):
        """Model ``day`` shifted by the static seasonal insolation offset
        (``_insolation_day_offset``, from ``config.insolation_start_doy``). The
        ONE place the offset is applied, consumed by BOTH insolation mechanisms:
        the ``day_to_calendar`` day_of_year (rrtmgp/forcing-dict path, via
        :meth:`_calendar_for_radiation`) AND the gray-radiation
        ``daily_mean_insolation`` solar declination. Decoupled from the
        relative-indexed SST forcing (which keeps the un-shifted ``day``).
        Offset 0.0 (``insolation_start_doy is None``) => identical to ``day``.
        """
        return day + self._insolation_day_offset

    def _calendar_for_radiation(self, day):
        """``day_to_calendar`` for the radiation insolation day_of_year/seconds,
        applying the seasonal offset (:meth:`_insolation_day`). The integer-day
        offset (the whole-day-of-year case) shifts ``day_of_year`` while leaving
        ``seconds_of_day`` (the diurnal phase) unchanged; offset 0.0 =>
        byte-identical to ``day_to_calendar(day)``.
        """
        return day_to_calendar(self._insolation_day(day))

    def _run_mpas(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run MPAS model with the unified physics pipeline.

        Uses the same physics pipeline as cubed-sphere/lat-lon, built
        via ``_create_physics()`` (includes RRTMGP, convection, etc.).
        Falls back to bare Held-Suarez forcing only when the config
        has radiation='none'.

        Restart contract (differs from the cube/lat-lon paths — read
        before writing a chain launcher):  this loop runs ``cfg.days``
        steps **from the (restart-loaded) state**, i.e. ``cfg.days`` is the
        number of days *this job* advances, NOT total days since the
        epoch.  ``start_day`` (the absolute day the checkpoint was written
        at) sets the time origin; ``start_step`` is used only for the
        absolute-step value stored in checkpoint metadata.  A chained
        launcher must therefore pass ``--days = TARGET - latest_checkpoint_day``
        (remaining days) for each link — see
        ``run_amip_mpas_100yr_gpu.sbatch``.  The cube/lat-lon paths use
        ``range(start_step, n_steps_total)`` and treat ``--days`` as total;
        do not copy their launcher convention here.
        """
        import time

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        n_steps_total = int(N_DAYS * 86400.0 / DT)
        DIAG_INTERVAL = int(cfg.output.diag_days * 86400.0 / DT) if cfg.output.diag_days > 0 else n_steps_total
        # ``checkpoint_days`` → step cadence.  Enables the 100-yr restart
        # chain (run_amip_mpas_100yr_gpu.sbatch): the driver writes
        # ``checkpoint_day_NNNN.npz`` every cadence and the launcher resumes
        # the next SLURM link from the latest one.  0 ⇒ no checkpointing.
        CHECKPOINT_INTERVAL = (
            int(cfg.output.checkpoint_days * 86400.0 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )
        START_DAY = start_day if start_day is not None else cfg.start_day

        # Build MPAS-compatible physics via make_physics (same code path as
        # cubed-sphere/lat-lon).  Includes RRTMGP + Held-Suarez forcing
        # when radiation is configured.
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig, apply_microphysics_experiment_flags,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
        from legoesm.atmosphere.physics.radiation.config import (
            RRTMGPConfig, OzoneProfileConfig,
        )

        # Phase D: wire the CMIP6 / experiment greenhouse-gas concentrations
        # (``cfg.co2_ppmv`` etc. — set at setup by ``ghg_at_year`` / the
        # experiment template, i.e. the transient AMIP value for the start
        # year) into the rrtmgp sub-config, so MPAS rrtmgp uses the prescribed
        # GHG forcing instead of the RRTMGPConfig defaults (415/1900/332).
        # The cube/lat-lon path does this; the MPAS path previously dropped it,
        # so a non-default-GHG (e.g. 1979 CO2=337) MPAS run was silently forced
        # at 415 ppmv.  Gray radiation ignores the rrtmgp sub-config, so this
        # is a no-op there.  (Time-varying-over-the-run GHG is a follow-on:
        # thread it through ``forcing`` like T_sfc; here it is fixed at the
        # start-year value.)
        # Phase D perf: run the RRTMGP optics tables + RTE solve in float32 even
        # under JAX x64 (the dycore stays fp64).  rrtmgp-on-MPAS was the
        # compute-bound limit that forced the moist-AMIP commit to fall back to
        # gray for long runs; the fp64 RTE solve dominates on fp64-limited GPUs
        # (e.g. RTX 8000, fp64 ~ 1/32 of fp32), so the fp32 path is ~2x faster
        # with heating identical to <0.01 K/day vs fp64 (benchmarked).  Honors
        # the ``RRTMGPConfig.compute_fp32`` contract ("the MPAS driver enables
        # it for the long-run rrtmgp path").  Enabled ONLY for rrtmgp — gray
        # radiation ignores the rrtmgp sub-config, so leave the default there.
        # compute_fp32 ENABLED for the rrtmgp MPAS path: run the optics tables
        # + RTE solve in float32 even under JAX x64 (the dycore stays fp64).
        # The earlier fp32 crash (PR #343) was a float64 leak in the OPTICS
        # interpolant reference grids (gas_optics / cloud_optics linspace +
        # vmr stack allocated strong-typed float64), which re-promoted the
        # optical depth and broke the RTE scan carry dtype.  Fixed on branch
        # fp32_mpas by keying those grids off the input dtype; GPU-validated
        # (job 8113954, L4 nCells=2562): fp32 heating is float32 + finite and
        # matches fp64 to rel-diff 1.5e-3 (REAL fp32, not a no-op).  Enabled
        # only for rrtmgp -- gray ignores the rrtmgp sub-config.
        # ``run_amip.py`` aliases the CLI ``rrtmgp`` → ``rrtmg`` for backward
        # compatibility, but the canonical ``RadiationConfig.scheme`` is
        # ``"rrtmgp"`` (the only non-gray value ``make_radiation_physics`` /
        # ``_call_radiation_backend`` recognise — line 687 pre-builds the
        # optics ONLY for ``scheme == "rrtmgp"``).  Passing the bare alias
        # ``"rrtmg"`` skipped the pre-build, so the RRTMGP optics tables were
        # (re)loaded from disk INSIDE the per-step JIT — a TracerArrayConversion
        # crash (``np.asarray`` on a traced lookup table) that broke MPAS rrtmgp
        # both serial and under MPI, and also left ``compute_fp32`` permanently
        # off.  Normalise the alias back to the canonical scheme here.
        _rad_scheme = (
            "rrtmgp" if cfg.radiation in ("rrtmg", "rrtmgp")
            else cfg.radiation
        )
        _rrtmgp_fp32 = (_rad_scheme == "rrtmgp")
        # Cloud-radiation coupling: thread the configured cloud-fraction
        # scheme into RRTMGP (cloud optics from microphysics condensate +
        # number).  ``include_clouds`` must be consistent with the cloud
        # scheme or ``_validate_cloud_gate`` raises (silently-clear-sky
        # guard).  Previously the MPAS path left cloud_scheme="none", so
        # an "AMIP" MPAS run radiated CLEAR-SKY regardless of --clouds.
        _cloud_scheme = (cfg.cloud_scheme
                         if _rad_scheme == "rrtmgp" else "none")
        # Aerosol-CCN specified-Nc coupling (Andreae 2009 AOD->CCN) + sub-grid
        # in-cloud autoconversion: thread both ExperimentConfig switches onto
        # the selected microphysics sub-config through the SAME shared helper
        # the coupled-path ``_resolve_microphysics`` uses, so the combined-
        # physics microphysics + radiation factories diagnose N_c from the
        # prescribed ``forcing["aerosol_od"]`` and the warm-rain closures match
        # the cube/lat-lon path exactly.  The helper fails loudly on a scheme
        # that lacks a requested switch (only Morrison implements them).  The
        # external-aerosol-forcing requirement is enforced upstream by run_amip
        # and by the factories' fail-fast (no aerosol_od => raise).
        _micro_cfg = MicrophysicsConfig(scheme=cfg.microphysics)
        _msub = getattr(_micro_cfg, cfg.microphysics, None)
        if _msub is not None:
            _micro_cfg = _micro_cfg._replace(**{
                cfg.microphysics: apply_microphysics_experiment_flags(
                    _msub, cfg.microphysics,
                    nc_from_aerosol=cfg.nc_from_aerosol,
                    subgrid_autoconversion=cfg.subgrid_autoconversion,
                )
            })
        from legoesm.atmosphere.physics.radiation.solar import earth_orbit
        _orbit_params = earth_orbit() if cfg.orbital_insolation else None
        phys_cfg = PhysicsConfig(
            radiation=RadiationConfig(
                scheme=_rad_scheme if _rad_scheme != "none" else "none",
                rrtmgp=RRTMGPConfig(
                    co2_ppmv=cfg.co2_ppmv,
                    ch4_ppbv=cfg.ch4_ppbv,
                    n2o_ppbv=cfg.n2o_ppbv,
                    compute_fp32=_rrtmgp_fp32,
                    include_clouds=(_cloud_scheme != "none"),
                    gpoint_batch_size=getattr(
                        cfg, "rrtmgp_gpoint_batch_size", 0),
                    gpoint_checkpoint=getattr(
                        cfg, "rrtmgp_gpoint_checkpoint", True),
                    column_chunk_size=getattr(
                        cfg, "rrtmgp_column_chunk_size", 0),
                ),
                cloud_scheme=_cloud_scheme,
                # Tuned cloud scalars (rh_crit / q_c_diagnostic / Xu-Randall)
                # reach the MPAS radiation clouds too (#870 Phase 1).
                cloud_config=_standalone_cloud_config(cfg, _cloud_scheme),
                diurnal_cycle=cfg.diurnal_cycle,
                orbit=_orbit_params,
                # Ozone source (default "standard" matches the bare default; a
                # non-standard --ozone-source now flows to MPAS rrtmgp).  The
                # external CMIP6 ozone FILE arrives per-step via the traced
                # ``forcing["o3_vmr"]`` (precedence over this source).
                ozone=OzoneProfileConfig(source=cfg.ozone_source),
            ),
            convection=ConvectionConfig(scheme=cfg.convection),
            turbulence=turbulence_config_for(cfg),
            microphysics=_micro_cfg,
            gravity_wave_drag=GravityWaveDragConfig(scheme=cfg.gravity_wave_drag),
        )
        # Phase D perf: shard the per-column RRTMGP workload across all local
        # devices (issue #273 ``column_mesh``).  rrtmgp is the dominant MPAS
        # cost — the per-column k-distribution × RTE solve already saturates a
        # single GPU, so sharding the ``nCells`` columns across N devices is
        # embarrassingly parallel and scales ~linearly (the dynamics stays on
        # the default device; only the radiation columns shard).  Enabled only
        # for rrtmgp with >1 device AND nCells divisible by the device count —
        # the radiation kernel requires an exact split, and padding the
        # UNSTRUCTURED column axis is unsafe (a phantom cell has no mesh
        # geometry), so we fall back to single-device otherwise.
        # SINGLE-PROCESS ONLY: under MPI (``is_distributed``) every rank sees
        # the full node device set, so a per-rank column mesh would shard each
        # rank's already-rank-local columns across ALL node GPUs and collide
        # with the MPI halo exchange (which works on rank-local unsharded
        # arrays).  The MPI path does its own device distribution; column
        # sharding is the single-process multi-GPU lever.  ``len(jax.devices())``
        # (not ``local_device_count``) matches ``create_column_mesh``'s own
        # ``jax.devices()`` so the divisibility check and the built mesh agree.
        if _rrtmgp_fp32:
            logger.info(
                "  RRTMGP compute_fp32: optics tables + RTE solve in float32 "
                "(dycore stays fp64)"
            )
        _column_mesh = None
        _single_process = (self._device_config is None
                           or not self._device_config.is_distributed)
        _n_dev = len(jax.devices())
        _ncell = int(self.state.T.data.shape[0])
        if (_rad_scheme == "rrtmgp" and _single_process
                and _n_dev > 1 and _ncell % _n_dev == 0):
            from legoesm.parallel.column_shard import create_column_mesh
            _column_mesh = create_column_mesh(_n_dev)
            logger.info(
                f"  RRTMGP column-sharding: {_ncell} cells / {_n_dev} devices "
                f"= {_ncell // _n_dev} cols/device"
            )
        elif _rad_scheme == "rrtmgp" and _single_process and _n_dev > 1:
            logger.warning(
                f"  RRTMGP column-sharding skipped: nCells={_ncell} not "
                f"divisible by n_devices={_n_dev}; running single-device "
                f"(throughput not scaled across GPUs)"
            )
        physics_fn = make_physics(phys_cfg, model_type="mpas", dt=DT,
                                  column_mesh=_column_mesh)

        # ---- Radiation sub-cycle (issue #316, MPAS port) ----
        # The MPAS physics_fn fuses radiation into ``model.step`` and ran the
        # full RRTMGP solve EVERY timestep (e.g. 60×/hour at dt=60), which is
        # the dominant cost and made the merged-dycore MPAS path unusably slow
        # (L4 ~0.44 steps/s).  Build a SECOND variant (``need_rad=False``) that
        # skips the RRTMGP/gray solve and re-uses the cached heating tendency
        # (``PhysicsState.rad_heating``, written by the full variant); the step
        # loop alternates the two by ``step % RAD_UPDATE_STEPS``.  At dt=60 /
        # RAD_UPDATE_STEPS=60 that is a 1-hour radiation cadence — the
        # CESM/E3SM standard the cubed-sphere path already validates — i.e.
        # ~RAD_UPDATE_STEPS× fewer RRTMGP solves with no new physics knob.
        # Only built when subcycling is active AND radiation is configured;
        # the held variant compiles a separate (RRTMGP-free, much cheaper)
        # ``model.step`` the first time it is used.
        RAD_UPDATE_STEPS = max(1, int(cfg.rad_update_steps))
        _subcycle_rad = RAD_UPDATE_STEPS > 1 and cfg.radiation != "none"
        physics_fn_norad = (
            make_physics(phys_cfg, model_type="mpas", dt=DT,
                         column_mesh=_column_mesh, need_rad=False)
            if _subcycle_rad else None
        )
        if _subcycle_rad:
            logger.info(
                "  Radiation sub-cycle: RRTMGP solved every "
                f"{RAD_UPDATE_STEPS} steps (cadence "
                f"{DT * RAD_UPDATE_STEPS / 3600.0:.2f} h); held heating reused "
                "in between"
            )

        # ---- AMIP surface boundary: anchor radiation to the prescribed SST -
        # Without this the MPAS hydrostatic radiation falls back to using the
        # lowest model-level temperature ``T[..., -1]`` as the surface
        # temperature (see integration.py ``_make_hydrostatic_radiation``), so
        # the column has NO external thermal anchor and cold-drifts toward a
        # dry gray-radiative equilibrium fully decoupled from the prescribed
        # SST (measured: <T_atm> 290 -> 257 K over 30 days, still falling).
        # The cubed-sphere/lat-lon and spectral paths apply the SST every
        # step; the MPAS path previously applied nothing — so an "AMIP" run on
        # MPAS was not actually SST-forced.
        #
        # We set a FIXED (annual-mean, sea-ice-blended) per-cell surface
        # temperature ONCE here, before the step loop.  ``model.step`` is
        # ``jax.jit`` with ``physics_fn`` marked *static* (primitive_eq_mpas.py
        # ``_step_jit`` static_argnums), so the override-closure cell is read
        # at trace time and BAKED into the first compile; a per-step update
        # would be silently ignored (stale value) unless every step paid a
        # retrace.  A fixed climatological anchor is therefore the correct
        # shape for the JIT'd MPAS path, is byte-identical across every
        # restart-chain link, and captures the first-order SST -> atmosphere
        # coupling (surface longwave).  The seasonal SST cycle and the
        # turbulent surface fluxes are intentionally NOT applied here: the
        # former needs traced forcing threaded through the MPAS step signature
        # and the latter needs the edge->cell wind interp (AMIP.md Known #3).
        # ---- AMIP surface boundary: TIME-VARYING prescribed SST ----
        # The prescribed SST/SIC is applied as the radiative surface
        # temperature via a per-step TRACED ``forcing={"T_sfc": (nCells,)}``
        # threaded through ``model.step`` -> combined physics -> radiation
        # (see primitive_eq_mpas.step + radiation integration ``_wants_forcing``).
        # Unlike the earlier fixed-anchor ``set_T_sfc_override`` (a JIT-static
        # closure that could only carry ONE baked value), ``forcing`` is a jit
        # argument, so the SST can vary in time (seasonal cycle) WITHOUT
        # retracing — matching the cube/spectral AMIP paths.  Without any
        # surface anchor the MPAS radiation falls back to ``T[..., -1]`` (the
        # lowest air level) and the column cold-drifts, decoupled from the SST.
        _sst_forcing = (cfg.radiation != "none" and self.get_sst_sic is not None)
        _compute_T_sfc = None
        if _sst_forcing:
            from legoesm.forcing.surface_utils import blend_surface_temperature
            _T_ice = cfg.T_ice
            _ncell = int(self.state.T.data.shape[0])

            def _compute_T_sfc(day):
                # Prescribed SST/SIC at the MPAS cell latitudes (get_sst_sic is
                # built on grid.grid_lat = mesh.latCell for analytical/AMIP
                # data), sea-ice-blended, as a (nCells,) surface temperature.
                _sst, _sic = self.get_sst_sic(day)
                return blend_surface_temperature(
                    jnp.asarray(_sst), jnp.asarray(_sic), _T_ice).reshape(-1)

            # Shape guard once, up front: a non-per-cell get_sst_sic would
            # otherwise surface as an opaque error deep inside the JIT trace.
            _ts0 = _compute_T_sfc(START_DAY)
            if _ts0.shape != (_ncell,):
                raise ValueError(
                    f"MPAS SST forcing shape {tuple(_ts0.shape)} != "
                    f"(nCells={_ncell},); get_sst_sic must return per-cell "
                    f"arrays on the MPAS mesh (grid.grid_lat = latCell)."
                )
            logger.info(
                "  AMIP SST surface forcing (time-varying): "
                f"day {START_DAY:.1f} T_sfc=[{float(jnp.min(_ts0)):.1f},"
                f"{float(jnp.max(_ts0)):.1f}] mean={float(jnp.mean(_ts0)):.1f} K"
            )

        # Wrap with Held-Suarez forcing when enabled.  Factored into a helper
        # so the SAME wrap applies to BOTH radiation sub-cycle variants (the
        # full ``physics_fn`` and the held ``physics_fn_norad``).
        if cfg.held_suarez_forcing:
            from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
            from legoesm.core.state import HydrostaticTendencies
            from legoesm.atmosphere.physics.combined import (
                physics_config_requires_phys_state,
            )

            def _wrap_hs(_rrtmgp_fn):
                def _hs_physics_fn(state, mesh, sigma_coord, phys_state=None, forcing=None):
                    rrtmgp_result = _rrtmgp_fn(
                        state, mesh, sigma_coord, phys_state=phys_state, forcing=forcing)
                    rrtmgp_tend = rrtmgp_result[0] if isinstance(rrtmgp_result, tuple) else rrtmgp_result
                    phys_state_out = rrtmgp_result[1] if isinstance(rrtmgp_result, tuple) else None
                    hs_tend = held_suarez_forcing_mpas(state, mesh, sigma_coord)
                    summed = HydrostaticTendencies(
                        du_dt=rrtmgp_tend.du_dt.replace(
                            data=rrtmgp_tend.du_dt.data + hs_tend.du_dt.data),
                        dT_dt=rrtmgp_tend.dT_dt.replace(
                            data=rrtmgp_tend.dT_dt.data + hs_tend.dT_dt.data),
                        dp_s_dt=rrtmgp_tend.dp_s_dt.replace(
                            data=rrtmgp_tend.dp_s_dt.data + hs_tend.dp_s_dt.data),
                        dphis_dt=rrtmgp_tend.dphis_dt.replace(
                            data=rrtmgp_tend.dphis_dt.data + hs_tend.dphis_dt.data),
                        tracer_tendencies=rrtmgp_tend.tracer_tendencies,
                    )
                    return summed, phys_state_out

                if hasattr(_rrtmgp_fn, 'set_time'):
                    _hs_physics_fn.set_time = _rrtmgp_fn.set_time
                if hasattr(_rrtmgp_fn, 'reset_state'):
                    _hs_physics_fn.reset_state = _rrtmgp_fn.reset_state
                # Forward the surface-T override too (symmetry with set_time /
                # reset_state).  The SST anchor is already baked into
                # _rrtmgp_fn's closure before wrapping, but forwarding keeps
                # the hook reachable on the wrapped fn so a later
                # set_T_sfc_override call still lands.
                if hasattr(_rrtmgp_fn, 'set_T_sfc_override'):
                    _hs_physics_fn.set_T_sfc_override = _rrtmgp_fn.set_T_sfc_override
                # Carry the forcing-aware marker so the dispatcher/step still
                # forwards the traced ``forcing`` (T_sfc) through the HS wrapper.
                if getattr(_rrtmgp_fn, '_wants_forcing', False):
                    _hs_physics_fn._wants_forcing = True
                # Propagate the stateful-carry tag (#413): this combine wrapper
                # is a plain closure, so refuse_unthreaded_stateful_physics
                # cannot unwrap it.  Forward the marker from the config (the
                # source of truth) so the carry contract and the sharded /
                # spectral refusals still see a stateful physics THROUGH this
                # wrapper, not a deceptively diagnostic-looking callable.
                if physics_config_requires_phys_state(phys_cfg):
                    _hs_physics_fn._requires_phys_state = True
                return _hs_physics_fn

            physics_fn = _wrap_hs(physics_fn)
            if physics_fn_norad is not None:
                physics_fn_norad = _wrap_hs(physics_fn_norad)

        run_status = "COMPLETED"
        logger.info(f"Starting MPAS: {n_steps_total} steps, {N_DAYS} days "
                    f"(from day {START_DAY:.1f})")

        # Light-weight time series — see _run_spectral for the rationale
        # (the MPAS path also bypasses the unified DiagnosticCollector).
        # ``CWV`` (column water vapor) is recorded on moist runs (NaN on dry);
        # ``_save_lightweight_timeseries`` already persists a ``CWV`` channel
        # and ``validate_amip_run.py`` checks its bounds.
        _ts: dict[str, list] = {
            "days": [], "T_atm": [], "T_min": [], "T_max": [],
            "max_wind": [], "dry_mass_ps": [], "T_finite": [], "CWV": [],
        }

        t_start = time.time()

        # Run ``cfg.days`` steps FROM the (possibly restart-loaded) state.
        # ``step`` is LOCAL (0-based) to this job: the loaded state +
        # ``START_DAY`` carry the continuation, and the launcher passes
        # ``--days`` = remaining days.  Using ``range(start_step, …)`` here
        # would be empty once ``start_step`` is the prior job's absolute
        # step count, and would also break the ``day = START_DAY +
        # (step+1)·dt`` math below (double-counting elapsed time).  The
        # absolute step (for checkpoint metadata only) is ``start_step +
        # step + 1``.
        # Per-step traced SST forcing.  Rebuilt only when the simulated day
        # changes (SST carries no sub-daily signal) so the host cost is one
        # ``get_sst_sic`` call per day, not per step; the dict structure is
        # constant so the jit'd step compiles once (the value is traced).
        _forcing = None
        _last_force_day = None
        # External CMIP6 forcing (ozone file / aerosol / transient GHG) on
        # the MPAS path: threaded through the same per-step TRACED
        # ``forcing`` dict as T_sfc, refreshed at daily cadence.  The
        # radiation factory (``_make_hydrostatic_radiation``) reads
        # ``forcing["o3_vmr"|"aerosol_od"|"ghg_vmr"]`` and forwards them
        # to RRTMGP — closing the "MPAS path bypasses external forcing"
        # gap flagged by the CMIP6 deck driver.  Gated on a spectral
        # radiation scheme: gray ignores all three channels.
        # ``ghg_vmr`` values are wrapped in jnp.asarray so a monthly
        # value change stays a traced-value change (no retrace); the
        # dict KEY STRUCTURE is decided once here and never changes
        # mid-run (pytree stability for the JIT'd step).
        _ext_forcing = (
            cfg.radiation in ("rrtmg", "rrtmgp")
            and (self._ozone_ext_active or self._aerosol_active
                 or self._ghg_active or bool(self._experiment))
        )
        # Aerosol-CCN specified-Nc fill needs ``forcing["aerosol_od"]`` in the
        # MICROPHYSICS step regardless of the radiation scheme (the second
        # indirect / KK2000 ``Nc^-1.79`` effect is independent of how
        # radiation is solved).  Without this an --aerosol-ccn run on gray
        # radiation would never thread aerosol_od and the microphysics fill
        # would raise its (then-misleading) fail-fast.  Requires a real
        # aerosol source.
        if cfg.nc_from_aerosol and self._aerosol_active:
            _ext_forcing = True
        # Operator-split physics carry (prognostic TKE / convection state).
        # ``model.step`` stashes the OUT state on ``self.model._phys_state``;
        # feed it back next step.  SEEDED here (issue #405): starting from
        # ``None`` is NOT lazy-init — ``update_physics_state(None, ...)``
        # returns ``None``, so the carry would stay ``None`` forever and
        # every stateful scheme (tke/mynn25/bechtold/prognostic GWD) would
        # silently reseed its prognostic fields each timestep.
        from legoesm.atmosphere.physics.physics_state import (
            init_physics_state,
        )
        from legoesm.core.precision import get_policy as _get_policy
        _ncol_phys = int(self.state.T.data.shape[0])
        _nlev_phys = int(self.state.T.data.shape[1])
        # Storage dtype for the persistent carry — EXCEPT when the
        # prognostic-spectral GWD is active: its wave-action spectrum
        # integration wants the default/compute dtype (codex review;
        # see the GWD integration note in physics_state.init docs).
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            gwd_carries_spectrum,
        )
        _seed_dtype = (
            None
            if gwd_carries_spectrum(phys_cfg.gravity_wave_drag.scheme)
            else _get_policy().storage
        )
        _phys_state = init_physics_state(
            _ncol_phys, _nlev_phys, phys_cfg, dtype=_seed_dtype,
        )
        # MPAS cell-partition MPI (codex round-9 HIGH): the seed above uses
        # rank-LOCAL ncol, so ``col_index`` would be ``arange(local)`` on
        # EVERY rank — duplicate global identities across ranks make the
        # stochastic per-column fold decomposition-VARIANT (the exact bug
        # class A1 increment 2 fixed for lat-band SPMD).  The partition's
        # ``local_cells`` are the (owned+halo) GLOBAL cell ids in local
        # order; halo columns get their true owner's id, so their draws
        # match the owning rank (halo values are overwritten by the
        # exchange regardless).
        if self._voronoi_layout is not None:
            _phys_state = _phys_state._replace(
                col_index=jnp.asarray(
                    self._voronoi_layout.partition.local_cells,
                    dtype=jnp.int32))
        # Checkpoint restore (#413): the MPAS load path stashes the
        # persisted PhysicsState fields in carry_aux under
        # ``physstate_<field>``.  Overlay them onto the fresh seed and
        # cast to the seed dtype so the carry honours the GWD dtype
        # rule above.  A PRESENT field with the wrong shape means a
        # corrupted / wrong-resolution / wrong-config restart — fail
        # fast rather than silently reseed the prognostic physics
        # memory (codex review: a warning is the #405 silent-wrong bug
        # class again; deliberately switching schemes across restart
        # should drop the stale physstate_* entries from the
        # checkpoint, not rely on a silent fallback).
        if isinstance(self._carry_aux, dict):
            # Cross-scheme convection restore check (codex round 8):
            # the profile-prognostic schemes share the carry shape, so
            # the per-field shape check below cannot catch a scheme
            # change.  Tag absent = legacy checkpoint (warn + accept).
            _saved_conv = self._carry_aux.get("physstate_meta_conv_scheme")
            _has_conv_carry = any(
                k in self._carry_aux
                for k in ("physstate_conv_prog_profile",
                          "physstate_conv_stoch_state")
            )
            if _has_conv_carry:
                if _saved_conv is None:
                    logger.warning(
                        "  Restored MPAS convection carry has no scheme "
                        "tag (legacy checkpoint) — assuming it matches "
                        "convection=%r", cfg.convection,
                    )
                elif str(_saved_conv) != str(cfg.convection):
                    raise ValueError(
                        f"Restored MPAS convection carry was written by "
                        f"scheme {str(_saved_conv)!r} but this run "
                        f"configures convection={cfg.convection!r} — "
                        "reusing it would feed one scheme's memory to "
                        "another (issue #405/#413).  Fix the config or "
                        "strip the physstate_* entries to opt into a "
                        "fresh seed."
                    )
            # Codex adversarial (#413): the MPAS save writes EVERY
            # PhysicsState field together (all are concrete arrays —
            # init_physics_state never leaves one None) AND the
            # ``physstate_meta_conv_scheme`` tag, all inside one
            # ``if carry is not None`` block.  So the PRESENCE of ANY
            # ``physstate_*`` key — including the meta tag alone — means
            # the checkpoint intended to carry physics state; the full
            # non-meta field set must then be present.  A subset (partial
            # write / hand-stripped / skewed writer), or a meta-only
            # remnant, would overlay what it has and silently leave the
            # rest at a FRESH seed — mixing restored and reseeded memory
            # and branching the trajectory.  Only ZERO physstate_* keys
            # (the documented "strip ALL physstate_* entries") opts into a
            # clean fresh seed.
            _any_physstate = any(
                k.startswith("physstate_") for k in self._carry_aux)
            _present_fields = {
                k[len("physstate_"):] for k in self._carry_aux
                if k.startswith("physstate_")
                and not k[len("physstate_"):].startswith("meta_")
            }
            # Schema-growth migration (grow-only allowlist): PhysicsState
            # fields ADDED after a checkpoint format was in production.  A
            # checkpoint written by an older build legitimately lacks these;
            # they seed from the fresh init (zeros) instead of tripping the
            # completeness gate below.  Only fields whose zero-seed is the
            # correct pre-feature state may be listed (aerosol_number: the
            # prognostic-aerosol tracer is opt-in and zero before the feature
            # existed).  Any OTHER missing field is still a partial/corrupted
            # carry and must fail loudly (issue #405/#413).
            _NEW_OPTIONAL_PS_FIELDS = frozenset({"aerosol_number"})
            if _any_physstate:
                # ``col_index`` is exempt from the completeness contract:
                # it is CONSTANT derivable identity data (arange(ncol),
                # never evolved), added 2026-07 — checkpoints written
                # before then legitimately lack it, and the fresh seed's
                # arange is byte-identical to what the save would have
                # stored.  Every EVOLVING field stays mandatory.
                _missing = [f for f in _phys_state._fields
                            if f not in _present_fields
                            and f != "col_index"]
                _new_missing = [f for f in _missing
                                if f in _NEW_OPTIONAL_PS_FIELDS]
                _missing = [f for f in _missing
                            if f not in _NEW_OPTIONAL_PS_FIELDS]
                if _new_missing:
                    logger.info(
                        "  MPAS restart checkpoint predates PhysicsState "
                        "field(s) %s — seeding them fresh (zeros); all other "
                        "physics memory is restored.", sorted(_new_missing),
                    )
                if _missing:
                    raise ValueError(
                        "MPAS restart physics-state carry is INCOMPLETE: "
                        f"present {sorted(_present_fields)}, missing "
                        f"{sorted(_missing)}.  The save writes every "
                        "PhysicsState field (and the scheme-tag) together, "
                        "so a subset or a meta-only remnant is a partial / "
                        "corrupted / hand-edited checkpoint; overlaying it "
                        "would mix restored and freshly-seeded memory and "
                        "silently branch the trajectory (issue #405/#413).  "
                        "Restore a complete checkpoint, or strip ALL "
                        "physstate_* entries (fields AND meta) to opt into "
                        "a fresh seed."
                    )
            _restored_ps = {}
            for _k, _v in self._carry_aux.items():
                if not _k.startswith("physstate_"):
                    continue
                _name = _k[len("physstate_"):]
                if _name not in _phys_state._fields:
                    continue
                _seed_field = getattr(_phys_state, _name)
                _val = jnp.asarray(_v)
                if tuple(_val.shape) != tuple(_seed_field.shape):
                    raise ValueError(
                        f"Restart physics-state field {_name!r} has "
                        f"shape {tuple(_val.shape)} but the configured "
                        f"run expects {tuple(_seed_field.shape)} — the "
                        "checkpoint does not match this configuration "
                        "(resolution / scheme config change or a "
                        "corrupted file).  Silently reseeding would "
                        "branch the trajectory (issue #405/#413); fix "
                        "the config or strip the physstate_* entries "
                        "from the checkpoint to opt into a fresh seed."
                    )
                _restored_ps[_name] = _val.astype(_seed_field.dtype)
            if _restored_ps:
                _phys_state = _phys_state._replace(**_restored_ps)
                logger.info(
                    "  Restored physics state from checkpoint: %s",
                    sorted(_restored_ps),
                )
        # Latest carry for save_checkpoint (#413) — updated every step.
        self._mpas_phys_state = _phys_state

        # MPAS cell-partition MPI: swap the serial ``model.step`` for the
        # halo-exchanging MPI step.  Same operator-split as the serial step
        # (dynamics RK incl. tracer advection → physics → floors → mass fix),
        # but each RK stage first exchanges cell/edge/tracer halos so every
        # owned boundary cell sees fresh neighbour values, and the mass fixer
        # sums only owned cells with a global allreduce (the model's internal
        # fixer would double-count halo cells).  The same column-local
        # ``physics_fn`` and the traced ``forcing`` / ``phys_state`` carry are
        # threaded through unchanged.  Built once outside the loop.
        _mpi_step = None
        _mpi_step_norad = None
        if self._voronoi_layout is not None:
            from legoesm.parallel.voronoi_mpi import make_voronoi_mpi_step
            _mpi_step = make_voronoi_mpi_step(
                self.model, self._voronoi_layout, self.model.sigma_coord,
                config=self.model.config, physics_fn=physics_fn,
                return_phys_state=True,
            )
            # Radiation sub-cycle: a matching held-radiation MPI step so the
            # cell-partition path also skips RRTMGP on the held steps.
            if physics_fn_norad is not None:
                _mpi_step_norad = make_voronoi_mpi_step(
                    self.model, self._voronoi_layout, self.model.sigma_coord,
                    config=self.model.config, physics_fn=physics_fn_norad,
                    return_phys_state=True,
                )
            logger.info(
                "  MPAS MPI step active (rank %d/%d)",
                self._voronoi_layout.rank, self._voronoi_layout.n_ranks,
            )

        _forcing_daily: dict = {}
        from legoesm.forcing.time_utils import daily_forcing_bucket
        for step in range(n_steps_total):
            if _sst_forcing or _ext_forcing:
                _force_day = START_DAY + step * DT / 86400.0
                # floor, not int() — see daily_forcing_bucket (negative
                # fractional days land in the wrong bucket under
                # truncation; day_to_calendar already handles negative
                # days via modulo).
                _fd_int = daily_forcing_bucket(_force_day)
                if _fd_int != _last_force_day:
                    # Sample the daily fields at the CANONICAL day boundary
                    # (``float(_fd_int)``), NOT at the first step that
                    # enters the day: a restart link's first step lands
                    # mid-day, so sampling at ``_force_day`` gave a chained
                    # run slightly different seasonal SST/ozone than the
                    # straight run and broke bit-exact restart continuation
                    # (FIX_RESTART_TIME; found by the restart validation
                    # harness — T diverged ~1e-8 over 2 steps).  Runs
                    # starting at integer days (all production configs) are
                    # bit-identical to before.
                    _force_day_canonical = float(_fd_int)
                    _forcing_daily = {}
                    if _sst_forcing:
                        _forcing_daily["T_sfc"] = _compute_T_sfc(
                            _force_day_canonical)
                    if _ext_forcing:
                        _ext_p_s, _ext_lat = self._owned_p_s_and_lat()
                        _o3, _aer, _ghg = self._precompute_external_forcing(
                            _force_day_canonical, _ext_p_s,
                            jnp.asarray(_ext_lat),
                        )
                        _forcing_daily["o3_vmr"] = _o3
                        _forcing_daily["aerosol_od"] = _aer
                        # Volcanic LONGWAVE aerosol (gap #9): set the key
                        # only when the LW source is active (else omit ⇒
                        # ``forcing.get("aerosol_lw_od")`` is None ⇒ RRTMGP
                        # no-op ⇒ byte-identical).
                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
                        if self._aerosol_lw_active and _aer_lw is not None:
                            _forcing_daily["aerosol_lw_od"] = _aer_lw
                        if _ghg is not None:
                            _forcing_daily["ghg_vmr"] = {
                                k: jnp.asarray(v) for k, v in _ghg.items()
                            }
                    _last_force_day = _fd_int
                # Per-STEP traced calendar time: the radiation factory
                # closure (``set_time``) is baked at trace time inside the
                # JIT'd MPAS step, so without these two traced scalars the
                # whole run would see the insolation of the initial day —
                # no seasonal or diurnal cycle.  Scalars only; the heavier
                # daily fields above are reused between updates.
                _doy, _sod = self._calendar_for_radiation(_force_day)
                _forcing = dict(_forcing_daily)
                _forcing["day_of_year"] = jnp.asarray(_doy)
                _forcing["seconds_of_day"] = jnp.asarray(_sod)
            # Radiation sub-cycle: solve RRTMGP on step 0 (cache warm-up,
            # always) and every RAD_UPDATE_STEPS-th step; reuse the held
            # heating (PhysicsState.rad_heating) in between.  ``step`` is
            # job-local, so the first step of every job/restart link re-solves
            # radiation and repopulates the cache before any held step reads
            # it.  When subcycling is off, every step is a full step.
            _use_rad = (not _subcycle_rad) or (step % RAD_UPDATE_STEPS == 0)
            if _mpi_step is not None:
                _mstep = _mpi_step if _use_rad else _mpi_step_norad
                self.state, _phys_state = _mstep(
                    self.state, DT, _forcing, _phys_state)
            else:
                _pfn = physics_fn if _use_rad else physics_fn_norad
                self.state = self.model.step(
                    self.state, DT, physics_fn=_pfn, forcing=_forcing,
                    phys_state=_phys_state)
                _phys_state = self.model._phys_state
            # Keep the persisted-carry handle fresh for save_checkpoint
            # (#413) — reference assignment, no device work.
            self._mpas_phys_state = _phys_state

            # Diagnostics at intervals
            if DIAG_INTERVAL > 0 and (step + 1) % DIAG_INTERVAL == 0:
                elapsed_day = (step + 1) * DT / 86400.0
                T_data = self.state.T.data
                p_s_data = self.state.p_s.data
                u_data = self.state.u.data

                # Column water vapor field on moist runs (reuse the shared
                # integral); ``None`` on dry runs keeps the series aligned.
                _cwv_field = None
                if (self.state.tracers is not None
                        and "q_v" in self.state.tracers):
                    from legoesm.diagnostics.column_integrals import (
                        column_water_vapor,
                    )
                    _cwv_field = column_water_vapor(
                        self.state.tracers["q_v"].data, p_s_data,
                        self.sigma.dsigma)

                if self._voronoi_layout is not None:
                    # MPAS cell-partition MPI: the state spans owned+halo
                    # cells, and each rank holds only its band — so a plain
                    # ``jnp.mean`` over ``T_data`` would double-count halo
                    # cells AND be rank-local.  Reduce over OWNED cells only
                    # and allreduce to a true global diagnostic (mirrors the
                    # owned-mask + allreduce mass fixer).
                    mean_T, mean_ps, max_u, T_min, T_max, T_finite, _cwv = \
                        self._mpas_global_diag(
                            T_data, p_s_data, u_data, _cwv_field)
                else:
                    # Serial / single-rank: fuse the reductions into one
                    # device→host transfer (each ``float()`` is a GPU stall).
                    _stats = jnp.stack([
                        jnp.mean(T_data),
                        jnp.mean(p_s_data),
                        jnp.max(jnp.abs(u_data)),
                        jnp.min(T_data),
                        jnp.max(T_data),
                        jnp.all(jnp.isfinite(T_data)).astype(T_data.dtype),
                    ])
                    _stats_host = np.asarray(_stats)
                    mean_T = float(_stats_host[0])
                    mean_ps = float(_stats_host[1])
                    max_u = float(_stats_host[2])
                    T_min = float(_stats_host[3])
                    T_max = float(_stats_host[4])
                    T_finite = bool(_stats_host[5] > 0.5)
                    _cwv = (float(jnp.mean(_cwv_field))
                            if _cwv_field is not None else float("nan"))

                _ts["days"].append(elapsed_day)
                _ts["T_atm"].append(mean_T)
                _ts["T_min"].append(T_min)
                _ts["T_max"].append(T_max)
                _ts["max_wind"].append(max_u)
                _ts["dry_mass_ps"].append(mean_ps)
                _ts["T_finite"].append(T_finite)
                _ts["CWV"].append(_cwv)

                elapsed = time.time() - t_start
                rate = elapsed_day / (elapsed + 1e-10)
                logger.info(
                    f"  Day {elapsed_day:6.1f}: T=[{T_min:.1f},{T_max:.1f}]K "
                    f"mean={mean_T:.1f}K  p_s={mean_ps/100:.1f}hPa  "
                    f"|u|_max={max_u:.1f}m/s"
                    + ("" if _cwv != _cwv else f"  CWV={_cwv:.1f}kg/m2")
                    + f"  ({rate:.1f} sim-days/s)"
                )

                # Blowup detection: finiteness AND physical bounds (#871 — a
                # runaway to 8e8 K was finite for 1138 steps under a
                # finiteness-only guard; the T_min floor masked its low side).
                from legoesm.driver.diagnostics import (
                    physical_state_blowup_reason,
                    t_min_floor_blowup_reason,
                )
                _bounds_reason = physical_state_blowup_reason(
                    elapsed_day, T_min, T_max)
                # LOUD T_min-floor guard (#930): a column pinned at the dycore
                # floor is a masked runaway.  Label it specifically and PREFER
                # it over the generic bounds message.  MPAS-only, eager path —
                # no SegmentCarry / _step_jit signature change.
                _floor_reason = t_min_floor_blowup_reason(
                    elapsed_day, T_min, float(self.model.config.T_min))
                _reason = _floor_reason or _bounds_reason
                if (not T_finite) or _reason is not None:
                    run_status = (_reason
                                  or f"BLOWUP at day {elapsed_day:.1f}")
                    logger.error(run_status)
                    self._write_blowup_state(
                        start_step + step + 1,
                        START_DAY + (step + 1) * DT / 86400.0)
                    break

            # Periodic checkpoint for the 100-yr restart chain — cadence is
            # independent of the diagnostic interval.  ``day`` is the
            # absolute simulated day (START_DAY + local elapsed); the
            # filename uses that absolute day so links across SLURM jobs
            # stay monotonic.  Absolute step (metadata only) =
            # start_step+step+1.  Guard finiteness FIRST: the checkpoint
            # cadence need not align with the diagnostic cadence, so a NaN
            # could occur between two diagnostic steps — never persist a
            # blown-up state (it would poison every subsequent chain link).
            if CHECKPOINT_INTERVAL > 0 and (step + 1) % CHECKPOINT_INTERVAL == 0:
                # Check EVERY prognostic, not just u: a radiation-driven NaN
                # surfaces in T (and propagates to p_s) and can occur between
                # diagnostic steps, so a u-only guard could persist a state
                # with finite u but NaN T — poisoning every subsequent chain
                # link.
                _s = self.state
                _finite = (jnp.all(jnp.isfinite(_s.u.data))
                           & jnp.all(jnp.isfinite(_s.T.data))
                           & jnp.all(jnp.isfinite(_s.p_s.data))
                           & jnp.all(jnp.isfinite(_s.phis.data)))
                if not bool(_finite):
                    _bad_day = START_DAY + (step + 1) * DT / 86400.0
                    run_status = f"BLOWUP at day {_bad_day - START_DAY:.1f}"
                    logger.error(
                        f"{run_status} (not a resumable checkpoint; writing "
                        "blowup_state for autopsy)")
                    self._write_blowup_state(start_step + step + 1, _bad_day)
                    break
                _ckpt_day = START_DAY + (step + 1) * DT / 86400.0
                _ckpt = getattr(self, "_checkpoint_callback", None) or self.save_checkpoint
                _ckpt(start_step + step + 1, _ckpt_day)
                # Wallclock-aware clean exit for long HPC dependency chains.
                self._maybe_wallclock_exit(_ckpt, start_step + step + 1, _ckpt_day)

        # Final checkpoint so the next chain link resumes from the exact end
        # state.  Skipped (a) on blow-up — state is non-finite — and (b) when
        # the last loop step already hit the periodic cadence, which would
        # re-write the identical file (wasted device→host transfer + I/O
        # every whole-multiple job boundary).
        if (CHECKPOINT_INTERVAL > 0 and run_status == "COMPLETED"
                and n_steps_total % CHECKPOINT_INTERVAL != 0):
            _final_day = START_DAY + n_steps_total * DT / 86400.0
            self.save_checkpoint(start_step + n_steps_total, _final_day)

        elapsed = time.time() - t_start
        logger.info(f"MPAS run {run_status} in {elapsed:.1f}s")
        self._save_lightweight_timeseries(_ts, run_status, t_start)
        return run_status

    # ==================================================================
    # Spectral execution path (Held-Suarez + radiation on Gaussian grid)
    # ==================================================================

    @staticmethod
    def _refuse_stateful_physics_unthreaded(cfg) -> None:
        """Refuse prognostic-carry physics on loops that drop the carry.

        Issue #405: ``update_physics_state(None, ...)`` returns ``None``,
        so a loop that never seeds/threads ``PhysicsState`` silently
        reseeds every stateful scheme each timestep — the run "succeeds"
        with physics that has no memory.  Loud refusal beats silent
        wrong numbers.  Lifted (#413) from ``_run_mpas`` (seeds/threads
        ``PhysicsState``), ``_run_compiled`` (tke/qke/gwd_spectrum ride
        the ``SegmentCarry``), and ``_run_per_step`` (carries threaded
        through ``step_unified``) — each with a stateful-memory test.
        Remaining caller: ``_run_spectral``, whose physics closure does
        not thread a carry yet; remove that call only together with
        real carry plumbing (and a stateful-memory test).

        The stateful-turbulence check comes from the shared
        ``turbulence_scheme_traits`` so this guard cannot drift from
        the seeding/dispatch sites (the hand-written set omitted
        ``clubb_lite``, which also carries TKE).
        """
        from legoesm.atmosphere.physics.turbulence.integration import (
            turbulence_scheme_traits,
        )
        from legoesm.atmosphere.physics.convection.integration import (
            convection_scheme_traits,
        )
        _turb = getattr(cfg, "turbulence", "none")
        _conv = getattr(cfg, "convection", "none")
        # Accept both driver CLI configs (plain scheme strings) and
        # PhysicsConfig-style objects (sub-config with .scheme).
        _turb = getattr(_turb, "scheme", _turb)
        _conv = getattr(_conv, "scheme", _conv)
        _gwd = getattr(cfg, "gravity_wave_drag", "none")
        _gwd = getattr(_gwd, "scheme", _gwd)
        # Convection: scalar-prognostic (mass_flux/edmf M_c / a_u),
        # profile-prognostic (ZM/KF/Emanuel/Tiedtke/Bechtold read and
        # relax conv_prog_profile — Tiedtke concretely carries updraft
        # mass-flux memory; codex round 5 widened this from the
        # hand-written bechtold/mass_flux/edmf set), and stochastic
        # (Bechtold AR1 state).
        _ct = convection_scheme_traits(_conv)
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            gwd_carries_spectrum,
        )
        if (turbulence_scheme_traits(_turb).carries_energy
                or _ct.is_scalar_prognostic
                or _ct.is_profile_prognostic
                or _ct.is_stochastic
                or gwd_carries_spectrum(_gwd)):
            raise NotImplementedError(
                f"turbulence={_turb!r} / convection={_conv!r} / "
                f"gwd={_gwd!r} carry prognostic physics state, which "
                "this run loop does not thread between steps yet "
                "(issue #405) — the carry would silently reseed every "
                "timestep.  Use a diagnostic scheme (louis / "
                "holtslag_boville; sbm / kuo / dca; linear GWD), or a "
                "driver path that threads PhysicsState (the MPAS, "
                "compiled, and per-step loops)."
            )

    def _run_spectral(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run spectral PE model with physics coupling.

        Physics tendencies are computed on the Gaussian grid and converted
        back to spectral space via SH analysis.
        """
        import time
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            spectral_pe_to_grid,
            SpectralHydrostaticState,
        )
        from legoesm.grids.gaussian import (
            sh_analysis_3d,
            sh_analysis_oc2_3d,
            sh_analysis_dmu_3d,
        )
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation, earth_orbit,
        )
        # Realistic orbit (Berger 1978) for this legacy spectral dry-gray path;
        # None ⇒ circular orbit (idealized runs unchanged).  gray daily-mean
        # folds the (a/r)^2 factor into the returned insolation directly.
        _orbit_params = (earth_orbit()
                         if getattr(self.config, "orbital_insolation", False)
                         else None)
        from legoesm.forcing.surface_utils import blend_surface_temperature

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        n_steps_total = int(N_DAYS * 86400.0 / DT)
        DIAG_INTERVAL = int(cfg.output.diag_days * 86400.0 / DT) if cfg.output.diag_days > 0 else n_steps_total
        # ``checkpoint_days`` → step cadence (FIX_RESTART_TIME iteration
        # 4: the spectral loop historically wrote NO checkpoints, so a
        # --spectral AMIP run silently ignored --checkpoint-days and
        # --restart-from was impossible).  The cadence is on ABSOLUTE
        # steps — this loop runs ``range(start_step, n_steps_total)`` —
        # so straight and resumed runs checkpoint at identical steps by
        # construction.  0 ⇒ no checkpointing (historical behavior).
        CHECKPOINT_INTERVAL = (
            int(cfg.output.checkpoint_days * 86400.0 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )
        START_DAY = start_day if start_day is not None else cfg.start_day
        # Restart-time normalization (FIX_RESTART_TIME, ported from
        # fix/persist-physics where it is production-validated): these
        # loops index time as ``START_DAY + ABSOLUTE_step * DT/86400``,
        # so START_DAY must be the EPOCH day (day at step 0).  Every
        # production caller (run_amip --restart-from, the coupled
        # drivers) passes the CHECKPOINT day from load_checkpoint — which
        # double-counted the already-elapsed time and ran every restarted
        # link with forcing shifted forward by the checkpoint day
        # (seasonally wrong SST / calendar; the restarted segment saw
        # day_of_year 4 instead of 3 in the validation harness).
        # Normalize ONLY when the caller passes back EXACTLY what this
        # driver's load_checkpoint returned (the recorded hint) — a
        # caller passing its own (e.g. epoch) start_day keeps the legacy
        # epoch semantics verbatim.  The MPAS loop keeps its own
        # local-step + checkpoint-day contract.
        if (start_day is not None and start_step > 0
                and self._loaded_checkpoint_step_day
                == (start_step, start_day)):
            START_DAY = start_day - start_step * DT / 86400.0
        a = self.grid.radius

        # Rayleigh friction profile
        sigma_full = self.sigma.sigma_full
        K_F = 1.0 / 86400.0
        SIGMA_B = 0.7
        k_f = K_F * jnp.maximum(0.0, (sigma_full - SIGMA_B) / (1.0 - SIGMA_B))

        gray_config = GrayRadiationConfig()
        shape_2d = (self.grid.n_lat, self.grid.n_lon)
        shape_3d = (*shape_2d, cfg.grid.nlev)
        S_0 = constants.S_0
        T_ice = cfg.T_ice

        # Precompute spectral transform constants
        _im_over_a = 1j * self.grid.ms.astype(jnp.float64) / a
        _one_over_a = 1.0 / a
        cos_lat_3d = self.grid.cos_lat[:, None, None]

        def _spectral_physics_fn(state, grid, sigma_coord, forcing_data=None):
            """Compute physics tendencies and return spectral tendencies.

            Iter-97 migration: when ``forcing_data`` is supplied (the
            new iter-92/95 API), reads ``day``, ``sst``, ``sic``,
            ``insol`` from the TRACED pytree.  Closes the iter-74
            ``_DayRef`` JIT-cache stale-day pathology for production
            spectral runs with diurnal/seasonal forcing.  Falls back
            to closure-captured ``self._current_day`` for backward
            compat with the legacy 3-arg call.
            """
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
            T_g = fields['T']
            u_g = fields['u']
            v_g = fields['v']
            p_s_g = fields['p_s']

            # SST/SIC: prefer TRACED forcing_data (iter-97 fix); fall
            # back to closure-captured day for legacy callers.
            if forcing_data is not None and "sst" in forcing_data:
                sst = forcing_data["sst"]
                sic = forcing_data["sic"]
                current_day = forcing_data["day"]
            else:
                sst, sic = self.get_sst_sic(self._current_day)
                current_day = self._current_day
            # Broadcast from (n_lat,) to (n_lat, n_lon) if needed
            if sst.ndim == 1 and len(shape_2d) == 2:
                sst = jnp.broadcast_to(sst[:, None], shape_2d)
                sic = jnp.broadcast_to(sic[:, None], shape_2d)
            T_sfc = blend_surface_temperature(sst, sic, T_ice)

            p_full = p_s_g[..., None] * sigma_full
            p_half = p_s_g[..., None] * self.sigma.sigma_half
            T_col = T_g.reshape(-1, cfg.grid.nlev)
            p_full_col = p_full.reshape(-1, cfg.grid.nlev)
            p_half_col = p_half.reshape(-1, cfg.grid.nlev + 1)
            q_v_col = jnp.zeros_like(T_col)  # dry physics
            T_sfc_col = T_sfc.reshape(-1)
            # Ensure lat is 2D (n_lat, n_lon) — may already be for Gaussian grids
            if self._grid_lat.ndim == 1:
                lat_2d = jnp.broadcast_to(self._grid_lat[:, None], shape_2d)
            else:
                lat_2d = self._grid_lat
            lat_col = lat_2d.reshape(-1)

            # Insolation: TRACED from forcing_data when supplied
            if forcing_data is not None and "insol" in forcing_data:
                insol = forcing_data["insol"]
            else:
                insol = daily_mean_insolation(
                    lat_col, self._insolation_day(current_day), S_0,
                    orbit=_orbit_params)
            rad_out = gray_radiation(
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, lat=lat_col,
                q_v=q_v_col, insolation=insol, config=gray_config,
            )
            dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

            # Held-Suarez Newtonian temperature relaxation
            if self._hs_newtonian_relax is not None:
                dT_dt_rad = dT_dt_rad + self._hs_newtonian_relax(
                    T_g, p_s_g, self._grid_lat)

            # Rayleigh friction
            du_dt = -k_f * u_g
            dv_dt = -k_f * v_g

            # Convert (du, dv) to spectral (dvor, ddiv)
            du_cos = du_dt * cos_lat_3d
            dv_cos = dv_dt * cos_lat_3d
            dvor_hat = (
                _im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
                + _one_over_a * sh_analysis_dmu_3d(grid, du_cos)
            )
            ddiv_hat = (
                _im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
                - _one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
            )
            dT_hat = sh_analysis_3d(grid, dT_dt_rad)
            dlnps_hat = jnp.zeros_like(state.lnps_hat.data)

            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=dvor_hat),
                div_hat=state.div_hat.replace(data=ddiv_hat),
                T_hat=state.T_hat.replace(data=dT_hat),
                lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            )

        # ---- Full-physics spectral AMIP (CMIP6 deck) ----
        # The legacy ``_spectral_physics_fn`` above is the dry gray
        # radiation + Rayleigh-friction stack used by Held-Suarez-like
        # idealized runs.  For AMIP CMIP6 (rrtmg radiation and/or moist
        # physics) we build the SAME unified physics pipeline as the
        # MPAS path (``make_physics(model_type='spectral_pe')``):
        # RRTMGP + convection + turbulence + microphysics + GWD with
        # external forcing (ozone file / aerosol / transient GHG /
        # SST anchor / traced calendar time) threaded through the
        # per-step TRACED ``forcing_data`` dict.  Previously this path
        # hard-coded dry gray radiation + constant solar, so a CMIP6
        # AMIP run on gaussian/spectral silently ignored every forcing
        # channel (the deck-driver "spectral path bypasses external
        # forcing" warning).
        _full_physics = (
            cfg.radiation in ("rrtmg", "rrtmgp")
            or cfg.microphysics != "none" or cfg.convection != "none"
            or cfg.turbulence != "none" or cfg.gravity_wave_drag != "none"
        )
        _phys_fn_loop = _spectral_physics_fn
        _ext_forcing = False
        if _full_physics:
            from legoesm.atmosphere.physics.combined import (
                PhysicsConfig, make_physics,
            )
            from legoesm.atmosphere.physics.radiation.config import (
                RadiationConfig, RRTMGPConfig, OzoneProfileConfig,
            )
            from legoesm.atmosphere.physics.convection.config import (
                ConvectionConfig,
            )
            from legoesm.atmosphere.physics.microphysics.config import (
                MicrophysicsConfig,
            )
            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
                GravityWaveDragConfig,
            )

            # Normalize the CLI radiation alias ("rrtmg") to the
            # physics-layer scheme name — see the _run_mpas rationale.
            _rad_scheme = ("rrtmgp" if cfg.radiation in ("rrtmg", "rrtmgp")
                           else cfg.radiation)
            _cloud_scheme = (cfg.cloud_scheme
                             if _rad_scheme == "rrtmgp" else "none")
            # phys_state is NOT threaded through the spectral step
            # (the SI/leapfrog JIT treats physics_fn as static and only
            # returns the state).  Prognostic-carry schemes would
            # silently re-initialize their carry every step — refuse
            # loudly instead of degrading.
            self._refuse_stateful_physics_unthreaded(cfg)
            from legoesm.atmosphere.physics.radiation.solar import earth_orbit
            _orbit_params = earth_orbit() if cfg.orbital_insolation else None
            phys_cfg = PhysicsConfig(
                radiation=RadiationConfig(
                    scheme=_rad_scheme,
                    rrtmgp=RRTMGPConfig(
                        co2_ppmv=cfg.co2_ppmv,
                        ch4_ppbv=cfg.ch4_ppbv,
                        n2o_ppbv=cfg.n2o_ppbv,
                        include_clouds=(_cloud_scheme != "none"),
                        gpoint_batch_size=getattr(
                            cfg, "rrtmgp_gpoint_batch_size", 0),
                        gpoint_checkpoint=getattr(
                            cfg, "rrtmgp_gpoint_checkpoint", True),
                        column_chunk_size=getattr(
                            cfg, "rrtmgp_column_chunk_size", 0),
                    ),
                    cloud_scheme=_cloud_scheme,
                    # Tuned cloud scalars for the spectral standalone
                    # radiation path (#870 Phase 1).
                    cloud_config=_standalone_cloud_config(cfg, _cloud_scheme),
                    diurnal_cycle=cfg.diurnal_cycle,
                    orbit=_orbit_params,
                    ozone=OzoneProfileConfig(source=cfg.ozone_source),
                ),
                convection=ConvectionConfig(scheme=cfg.convection),
                turbulence=turbulence_config_for(cfg),
                microphysics=MicrophysicsConfig(scheme=cfg.microphysics),
                gravity_wave_drag=GravityWaveDragConfig(
                    scheme=cfg.gravity_wave_drag),
            )
            _combined_fn = make_physics(
                phys_cfg, model_type="spectral_pe", dt=DT,
            )

            def _amip_physics_fn(state, grid, sigma_coord, forcing_data=None):
                # The spectral step extracts ``result[0]`` from tuple
                # returns, so the (tendency, phys_state_out) pair from
                # the combined pipeline is handled; phys_state_out is
                # dropped (stateless schemes only — guarded above).
                return _combined_fn(state, grid, sigma_coord,
                                    phys_state=None, forcing=forcing_data)

            _phys_fn_loop = _amip_physics_fn
            _ext_forcing = (
                _rad_scheme == "rrtmgp"
                and (self._ozone_ext_active or self._aerosol_active
                     or self._ghg_active or bool(self._experiment))
            )
            logger.info(
                "  Spectral full-physics AMIP pipeline: "
                f"radiation={_rad_scheme} clouds={_cloud_scheme} "
                f"convection={cfg.convection} turbulence={cfg.turbulence} "
                f"microphysics={cfg.microphysics} "
                f"external_forcing={'ON' if _ext_forcing else 'off'}"
            )

        self._current_day = START_DAY
        run_status = "COMPLETED"
        logger.info(f"Starting spectral: {n_steps_total - start_step} steps, {N_DAYS} days")

        # Iter-97 migration: build TRACED forcing_data each step
        # (day, sst, sic, insol as JAX arrays) and pass through the
        # iter-92/95 ``model.step(forcing_data=...)`` API.  Single
        # JIT compile (physics_fn identity stable) + dynamic forcing
        # values (no retrace, no cache leak).  Closes the iter-74
        # ``_DayRef`` JIT-cache stale-day pathology for production
        # spectral runs with diurnal/seasonal cycle.

        # Pre-compute lat_col_local for insolation (constant across steps)
        if self._grid_lat.ndim == 1:
            _lat_2d_loop = jnp.broadcast_to(
                self._grid_lat[:, None], shape_2d,
            )
        else:
            _lat_2d_loop = self._grid_lat
        _lat_col_loop = _lat_2d_loop.reshape(-1)

        # Light-weight time series for AMIP / validation.  The spectral
        # path is otherwise diagnostic-free; without these arrays the
        # `validate_amip_run.py` post-run check rejects the run for
        # missing ``timeseries.npz`` (caught when extending the AMIP
        # CMIP6 deck to all four grid types).
        _ts: dict[str, list] = {
            "days": [], "T_atm": [], "T_min": [], "T_max": [],
            "max_wind": [], "dry_mass_ps": [], "T_finite": [],
        }

        t_start = time.time()
        _ext_daily: dict = {}
        _last_ext_day = None
        for step in range(start_step, n_steps_total):
            self._current_day = START_DAY + (step + 1) * DT / 86400.0

            if _full_physics:
                # Traced forcing for the unified pipeline: SST/SIC-blend
                # surface anchor + per-step calendar time + (daily) the
                # external CMIP6 ozone/aerosol/GHG fields.
                sst_step, sic_step = self.get_sst_sic(self._current_day)
                if sst_step.ndim == 1:
                    sst_step = jnp.broadcast_to(sst_step[:, None], shape_2d)
                    sic_step = jnp.broadcast_to(sic_step[:, None], shape_2d)
                _T_sfc_step = blend_surface_temperature(
                    sst_step, sic_step, T_ice).reshape(-1)
                _fd_int = daily_forcing_bucket(self._current_day)
                if _ext_forcing and _fd_int != _last_ext_day:
                    _f_now = spectral_pe_to_grid(
                        self.state, self.grid, self.sigma)
                    # Sample at the CANONICAL day boundary, not the first
                    # step entering the day — a restart link's first step
                    # lands mid-day (same bug class as the MPAS loop; see
                    # daily_forcing_bucket / FIX_RESTART_TIME).
                    _o3, _aer, _ghg = self._precompute_external_forcing(
                        float(_fd_int), _f_now['p_s'], _lat_2d_loop,
                    )
                    _ext_daily = {"o3_vmr": _o3, "aerosol_od": _aer}
                    # Volcanic LONGWAVE aerosol (gap #9): only when active
                    # (omitted ⇒ None ⇒ RRTMGP no-op ⇒ byte-identical).
                    _aer_lw = getattr(self, "_aerosol_lw_od", None)
                    if self._aerosol_lw_active and _aer_lw is not None:
                        _ext_daily["aerosol_lw_od"] = _aer_lw
                    if _ghg is not None:
                        _ext_daily["ghg_vmr"] = {
                            k: jnp.asarray(v) for k, v in _ghg.items()
                        }
                    _last_ext_day = _fd_int
                _doy, _sod = self._calendar_for_radiation(self._current_day)
                forcing_data = {
                    "T_sfc": _T_sfc_step,
                    "day_of_year": jnp.asarray(_doy),
                    "seconds_of_day": jnp.asarray(_sod),
                    **_ext_daily,
                }
            else:
                # Legacy dry gray path: traced SST/SIC + daily-mean insol
                sst_step, sic_step = self.get_sst_sic(self._current_day)
                insol_step = daily_mean_insolation(
                    _lat_col_loop, self._insolation_day(self._current_day), S_0,
                    orbit=_orbit_params)
                forcing_data = {
                    "day": jnp.asarray(self._current_day),
                    "sst": sst_step,
                    "sic": sic_step,
                    "insol": insol_step,
                }
            self.state = self.model.step(
                self.state, DT,
                physics_fn=_phys_fn_loop,
                forcing_data=forcing_data,
            )

            if DIAG_INTERVAL > 0 and (step + 1) % DIAG_INTERVAL == 0:
                elapsed_day = (step + 1) * DT / 86400.0
                fields = spectral_pe_to_grid(self.state, self.grid, self.sigma)
                T_g = fields['T']
                p_s_g = fields['p_s']
                u_g, v_g = fields['u'], fields['v']

                # Fuse the diagnostic reductions into one stack so we
                # device→host-transfer once instead of five times.  At
                # diagnostic cadence this saves O(DIAG_INTERVAL) GPU
                # stalls per simulated period.
                _stats = jnp.stack([
                    jnp.mean(T_g),
                    jnp.min(T_g),
                    jnp.max(T_g),
                    jnp.mean(p_s_g),
                    jnp.max(jnp.sqrt(u_g ** 2 + v_g ** 2)),
                    jnp.all(jnp.isfinite(T_g)).astype(T_g.dtype),
                ])
                _stats_host = np.asarray(_stats)
                mean_T = float(_stats_host[0])
                T_min = float(_stats_host[1])
                T_max = float(_stats_host[2])
                mean_ps = float(_stats_host[3])
                max_wind = float(_stats_host[4])
                T_finite = bool(_stats_host[5] > 0.5)

                _ts["days"].append(elapsed_day)
                _ts["T_atm"].append(mean_T)
                _ts["T_min"].append(T_min)
                _ts["T_max"].append(T_max)
                _ts["max_wind"].append(max_wind)
                _ts["dry_mass_ps"].append(mean_ps)
                _ts["T_finite"].append(T_finite)

                elapsed = time.time() - t_start
                rate = elapsed_day / (elapsed + 1e-10)
                logger.info(
                    f"  Day {elapsed_day:6.1f}: T=[{T_min:.1f},{T_max:.1f}]K "
                    f"mean={mean_T:.1f}K  p_s={mean_ps/100:.1f}hPa  "
                    f"|v|_max={max_wind:.1f}m/s  ({rate:.1f} sim-days/s)"
                )

                from legoesm.driver.diagnostics import (
                    physical_state_blowup_reason,
                )
                _bounds_reason = physical_state_blowup_reason(
                    elapsed_day, T_min, T_max)
                if (not T_finite) or _bounds_reason is not None:
                    run_status = (_bounds_reason
                                  or f"BLOWUP at day {elapsed_day:.1f}")
                    logger.error(run_status)
                    self._write_blowup_state(step + 1, self._current_day)
                    break

            # Periodic checkpoint (FIX_RESTART_TIME iteration 4) —
            # cadence on ABSOLUTE steps, independent of the diagnostic
            # interval.  Guard finiteness FIRST (MPAS pattern): the
            # checkpoint cadence need not align with the diagnostic
            # cadence, so a NaN can occur between diagnostic steps —
            # never persist a blown-up state (it would poison every
            # subsequent chain link).
            if (CHECKPOINT_INTERVAL > 0
                    and (step + 1) % CHECKPOINT_INTERVAL == 0):
                _s = self.state

                def _all_finite_c(arr):
                    # Complex coefficients: finite iff BOTH parts are.
                    return (jnp.all(jnp.isfinite(arr.real))
                            & jnp.all(jnp.isfinite(arr.imag)))

                _finite = (_all_finite_c(_s.vor_hat.data)
                           & _all_finite_c(_s.div_hat.data)
                           & _all_finite_c(_s.T_hat.data)
                           & _all_finite_c(_s.lnps_hat.data))
                if not bool(_finite):
                    run_status = (
                        f"BLOWUP at day {self._current_day - START_DAY:.1f}")
                    logger.error(
                        f"{run_status} (not a resumable checkpoint; writing "
                        "blowup_state for autopsy)")
                    self._write_blowup_state(step + 1, self._current_day)
                    break
                self.save_checkpoint(step + 1, self._current_day)

        # Final checkpoint so the next chain link resumes from the exact
        # end state — skipped on blow-up and when the last step already
        # hit the periodic cadence (identical file).
        if (CHECKPOINT_INTERVAL > 0 and run_status == "COMPLETED"
                and n_steps_total % CHECKPOINT_INTERVAL != 0):
            _final_day = START_DAY + n_steps_total * DT / 86400.0
            self.save_checkpoint(n_steps_total, _final_day)

        elapsed = time.time() - t_start
        logger.info(f"Spectral run {run_status} in {elapsed:.1f}s")
        # Persist the lightweight time series so AMIP validators can
        # consume the run.  Mirrors the layout written by
        # `DiagnosticCollector.save` for the keys we actually populate;
        # other keys are filled with NaN of the same length so the
        # downstream consumer can iterate without ``KeyError``.
        self._save_lightweight_timeseries(_ts, run_status, t_start)
        return run_status

    def _save_lightweight_timeseries(
        self,
        ts: dict,
        run_status: str,
        t_start: float,
    ) -> None:
        """Write a minimal ``timeseries.npz`` for spectral / MPAS runs.

        These two run paths short-circuit the unified
        ``DiagnosticCollector`` pipeline (the collector expects
        gridpoint-space ``HydrostaticState`` fields, but the spectral
        path keeps state in spectral coefficients and the MPAS path
        uses voronoi cell-state with a different layout).  Without
        this fallback the run silently produces no ``timeseries.npz``
        and the AMIP validation harness rejects it for missing data.
        """
        from pathlib import Path
        import numpy as np

        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(ts.get("days", []))
        if n == 0:
            # Run too short for any diagnostic interval to fire (e.g.
            # ``--days 1 --diag-days 5``).  Synthesise a single
            # end-of-run sample from the final state so the validation
            # harness still has a ``timeseries.npz`` + ``results.txt``
            # to consume.  This avoids the silent-no-output behaviour
            # that the iter-4 codex review flagged: previously a
            # 1-day spectral / MPAS smoke run produced no validator
            # artefacts and the smoke test fell through.
            #
            # Spectral states (``SpectralHydrostaticState``) keep
            # ``T_hat / vor_hat / div_hat / lnps_hat`` in spectral
            # coefficients and **do not** expose ``state.T``.  Convert
            # via ``spectral_pe_to_grid`` first so ``mean(T)`` and
            # ``max(|u|)`` are physical (Codex iter-5 review).
            try:
                # Late import to avoid hard dependency at module
                # import time when spectral support is unavailable.
                from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
                    SpectralHydrostaticState, spectral_pe_to_grid,
                )
                if isinstance(self.state, SpectralHydrostaticState):
                    fields = spectral_pe_to_grid(
                        self.state, self.grid, self.sigma,
                    )
                    T_arr = fields["T"]
                    u_arr = fields["u"]
                    v_arr = fields.get("v")
                    ps_arr = fields["p_s"]
                else:
                    T_arr = self.state.T.data
                    u_arr = self.state.u.data
                    v_arr = (self.state.v.data
                             if getattr(self.state, "v", None) is not None
                             else None)
                    ps_arr = self.state.p_s.data
                final_day = float(self.config.days)
                T_final = float(jnp.mean(T_arr))
                # max_wind must include v: a meridional spike or NaN in
                # ``v`` would otherwise be invisible to the validator
                # and let a divergent run pass (Codex iter-6 review).
                if v_arr is not None:
                    wind_speed = jnp.sqrt(u_arr ** 2 + v_arr ** 2)
                    u_final = float(jnp.max(wind_speed))
                    wind_finite = bool(jnp.all(jnp.isfinite(v_arr)))
                else:
                    u_final = float(jnp.max(jnp.abs(u_arr)))
                    wind_finite = True
                ps_final = float(jnp.mean(ps_arr))
                T_finite = (bool(jnp.all(jnp.isfinite(T_arr)))
                            and wind_finite)
            except Exception as exc:
                logger.warning(
                    f"_save_lightweight_timeseries: end-of-run "
                    f"summary failed ({exc!r}); writing empty "
                    f"timeseries.npz anyway."
                )
                final_day = 0.0
                T_final = float("nan")
                u_final = float("nan")
                ps_final = float("nan")
                T_finite = False
            ts = {
                "days": [final_day],
                "T_atm": [T_final],
                "max_wind": [u_final],
                "dry_mass_ps": [ps_final],
                "T_finite": [T_finite],
            }
            n = 1
        nan = np.full(n, np.nan, dtype=np.float64)
        days = np.array(ts["days"], dtype=np.float64)

        def _arr(key: str) -> np.ndarray:
            v = ts.get(key)
            if v is None or len(v) == 0:
                return nan
            return np.array(v, dtype=np.float64)

        np.savez(
            out_dir / "timeseries.npz",
            days=days,
            T_atm=_arr("T_atm"),
            T_low=_arr("T_low") if "T_low" in ts else nan,
            max_wind=_arr("max_wind"),
            dry_mass_ps=_arr("dry_mass_ps"),
            sst=_arr("sst") if "sst" in ts else nan,
            sic=_arr("sic") if "sic" in ts else nan,
            precip=_arr("precip") if "precip" in ts else nan,
            CWV=_arr("CWV") if "CWV" in ts else nan,
            sw_up_toa=_arr("sw_up_toa") if "sw_up_toa" in ts else nan,
            lw_up_toa=_arr("lw_up_toa") if "lw_up_toa" in ts else nan,
            sw_net_sfc=_arr("sw_net_sfc") if "sw_net_sfc" in ts else nan,
            lw_net_sfc=_arr("lw_net_sfc") if "lw_net_sfc" in ts else nan,
            energy_residual=_arr("energy_residual") if "energy_residual" in ts else nan,
            moisture_residual=_arr("moisture_residual") if "moisture_residual" in ts else nan,
        )
        # Persist the run summary in the same place run_amip's main path
        # writes it, so `validate_amip_run.py` can read the status line.
        import time as _time
        wall = _time.time() - t_start
        cfg = self.config
        with open(out_dir / "results.txt", "w") as f:
            f.write("legoESM AMIP run\n")
            f.write(
                f"Grid: {cfg.grid.grid_type} {cfg.grid.resolution} / "
                f"L{cfg.grid.nlev}, dt={cfg.dycore.dt}s, {cfg.days} days\n"
            )
            f.write(f"Radiation: {cfg.radiation}\n")
            f.write(f"Status: {run_status}\n\n")
            f.write(f"Wall time: {wall:.1f}s\n\n")
            if n > 0 and len(ts.get("T_atm", [])) > 0:
                f.write(f"Final <T_atm>: {ts['T_atm'][-1]:.3f} K\n")
                f.write(f"Final max_wind: {ts['max_wind'][-1]:.2f} m/s\n")

    # ==================================================================
    # Shared run helpers (used by both compiled and per-step paths)
    # ==================================================================

    def _latlon_spmd_mesh(self):
        """Build the 1-D ``("lat",)`` device mesh for the multi-device lat-band
        SPMD run from ``config.n_devices`` (``"auto"`` = all visible devices).
        Returns ``None`` for a single device (the run_atm_latlon_spmd mesh=None
        single-device fallback).

        Under route-B multicontroller (``jax.process_count() > 1``, after
        ``init_multicontroller_distributed``) ``jax.devices()`` is the GLOBAL
        device list; the mesh must span ALL of them (one band per device across
        every process) — a strict subset would leave some processes' devices out
        of the program (non-addressable participation hazard)."""
        import numpy as _np
        nd_cfg = self.config.n_devices
        devs = jax.devices()
        nd = len(devs) if nd_cfg == "auto" else int(nd_cfg)
        nd = max(1, min(nd, len(devs)))
        if jax.process_count() > 1 and nd != len(devs):
            raise ValueError(
                f"--multicontroller (route-B) uses ALL {len(devs)} global "
                f"devices across {jax.process_count()} processes; "
                f"n_devices={nd_cfg!r} selects {nd} — set n_devices='auto' "
                f"or {len(devs)}.")
        if nd <= 1:
            return None
        n_lat = int(self.grid.n_lat)
        if n_lat % nd != 0:
            raise ValueError(
                f"enable_latlon_spmd: n_lat ({n_lat}) not divisible by "
                f"n_devices ({nd}) — pick n_devices among the divisors of "
                f"{n_lat} so every lat band is uniform.")
        return jax.sharding.Mesh(_np.array(devs[:nd]), axis_names=("lat",))

    def _latlon_spmd_physics_fn(self):
        """Select the STATELESS physics closure for the lat-band SPMD run, or
        raise if the configured physics is the stateful unified pipeline (not
        yet SPMD-routed). Supports Held-Suarez forcing and dynamics-only."""
        cfg = self.config
        if cfg.held_suarez_forcing:
            from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
            return held_suarez_forcing_latlon
        active = {
            name: val for name, val in (
                ("radiation", cfg.radiation),
                ("convection", cfg.convection),
                ("turbulence", cfg.turbulence),
                ("microphysics", cfg.microphysics),
                ("gravity_wave_drag", cfg.gravity_wave_drag),
                ("cloud_scheme", cfg.cloud_scheme),
            ) if val not in (None, "none")
        }
        if not active:
            return None                     # dynamics-only (dry)
        raise NotImplementedError(
            "enable_latlon_spmd supports dynamics-only (all parameterizations "
            f"'none') or held_suarez_forcing=True; the stateful unified physics "
            f"{active} is not yet SPMD-routed — its PhysicsState carry needs the "
            "lat-major reshape-aware shard (see run_atm_latlon_spmd_segment). "
            "Set those schemes to 'none' or use held_suarez_forcing=True.")

    def _run_compiled_latlon_spmd(self, start_step: int = 0,
                                  start_day: float | None = None) -> str:
        """Single-process multi-device lat-band SPMD run for the lat-lon C-grid
        hydrostatic atm (``config.enable_latlon_spmd``).

        Integrates via ``run_atm_latlon_spmd`` — the Stage-5/7-validated lat-band
        step + the HydrostaticState<->C-grid bridge — in segments, gathering a
        cell-centered ``HydrostaticState`` per segment for the coupler callback +
        a host-side NaN-blowup guard. A dedicated path, NOT the jitted
        ``compiled_segments`` scan (zero surgical risk to the shared hot loop).

        Supports DYNAMICS-ONLY or stateless Held-Suarez. Stateful physics and the
        diagnostics/checkpoint writers are follow-ups (rejected loudly); a coupled
        driver consumes the per-segment state via ``segment_callback``.
        """
        import time
        from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
            run_atm_latlon_spmd)

        cfg = self.config
        if cfg.output.checkpoint_days > 0 or cfg.output.diag_days > 0:
            raise NotImplementedError(
                "enable_latlon_spmd does not yet support the diagnostics / "
                "checkpoint writers (set diag_days=0, checkpoint_days=0); the "
                "gathered root-only writers are a follow-up — use the "
                "segment_callback hook for I/O.")
        mesh = self._latlon_spmd_mesh()
        # Operator-split lane: the general run_amip COLUMN-LOCAL unified physics
        # (radiation / turbulence / convection / microphysics / GWD / cloud) is
        # active — route to the sharded operator-split integration (dynamics ->
        # mass fixer -> step_unified physics -> tail), a faithful multi-device
        # twin of _run_compiled.  Held-Suarez and dynamics-only fall through to
        # the stateless run_atm_latlon_spmd lane below.
        if self._operator_split_spmd_active():
            if getattr(cfg, "latlon_spmd_compiled_segments", False):
                # Never a silent no-op: the operator-split unified-physics
                # SPMD lane steps per-step (no compiled-scan segments yet),
                # so a set flag would silently change nothing there.
                raise NotImplementedError(
                    "latlon_spmd_compiled_segments=True applies to the "
                    "STATELESS lat-lon SPMD lane (dynamics-only / "
                    "Held-Suarez via run_atm_latlon_spmd); the operator-"
                    "split unified-physics SPMD lane does not run compiled "
                    "scan segments yet. Unset the flag, or set the "
                    "parameterizations to 'none' / use held_suarez_forcing.")
            return self._run_operator_split_spmd(start_step, start_day, mesh)
        physics_fn = self._latlon_spmd_physics_fn()      # None / HS / raise
        DT = cfg.dycore.dt
        n_steps_total = int(cfg.days * 86400.0 / DT)
        START_DAY = start_day if start_day is not None else cfg.start_day
        # Restart-time normalization (mirrors _run_spectral / _prepare_run_context):
        # these loops index time as START_DAY + ABSOLUTE_step * DT/86400, so
        # START_DAY must be the EPOCH day.  Normalize ONLY when the caller passes
        # back EXACTLY what this driver's load_checkpoint returned (the recorded
        # hint), so a caller that already passes an epoch start_day is unchanged.
        if (start_day is not None and start_step > 0
                and getattr(self, "_loaded_checkpoint_step_day", None)
                == (start_step, start_day)):
            START_DAY = start_day - start_step * DT / 86400.0
        # One-day output/coupling cadence (no diagnostics writer here yet).
        seg_len = max(1, int(86400.0 / DT))
        n_run = n_steps_total - start_step
        if n_run < 1:
            return "COMPLETED"

        logger.info(
            "lat-lon SPMD run: %d steps, %s, physics=%s, mesh=%s",
            n_run, f"{seg_len}-step segments",
            ("held_suarez" if cfg.held_suarez_forcing
             else ("dynamics-only" if physics_fn is None else "custom")),
            (None if mesh is None else mesh.devices.size))

        # ``step_done`` is the cumulative step count WITHIN this run (1-based from
        # the run start); ``_prev`` tracks the previous boundary so the coupler
        # gets the ACTUAL (possibly short final) segment length, and the day uses
        # the ABSOLUTE step index ``start_step + step_done``.
        _prev = [0]

        def _on_segment(hs_global, step_done):
            self.state = hs_global
            seg_n = step_done - _prev[0]
            _prev[0] = step_done
            day = START_DAY + (start_step + step_done) * DT / 86400.0
            self._current_day = day
            if self._segment_callback is not None:
                self._segment_callback(self, day, DT * seg_n)

        t0 = time.time()
        hs_final, status = run_atm_latlon_spmd(
            self.model, mesh, self.state, DT, n_run,
            segment_steps=seg_len, physics_fn=physics_fn,
            on_segment=_on_segment,
            # M2b opt-in (--latlon-spmd-compiled-segments): one compiled
            # lax.scan per segment; default False = per-step path.
            compiled_segments=cfg.latlon_spmd_compiled_segments)
        self.state = hs_final
        logger.info("lat-lon SPMD run: %s (%.1fs)", status, time.time() - t0)
        return status

    def _tiled_cube_column_physics_fn(self):
        """Select the per-tile COLUMN physics for the tiled cube SPMD run,
        or raise if the configured physics is outside the tiled envelope.

        Supported: dynamics-only (all parameterizations ``'none'``) ->
        ``None``; Kessler warm-rain microphysics ALONE -> the shared
        column bridge (``make_kessler_column_physics_fn`` — the SAME
        ``kessler_column_tendencies`` core the face-sharded lanes run).
        Anything else (the stateful unified pipeline, Held-Suarez on the
        cube) is not tiled-routed — refuse loudly, never silently drop a
        parameterization.
        """
        cfg = self.config
        active = {
            name: val for name, val in (
                ("radiation", cfg.radiation),
                ("convection", cfg.convection),
                ("turbulence", cfg.turbulence),
                ("microphysics", cfg.microphysics),
                ("gravity_wave_drag", cfg.gravity_wave_drag),
                ("cloud_scheme", cfg.cloud_scheme),
            ) if val not in (None, "none")
        }
        if getattr(cfg, "held_suarez_forcing", False):
            raise NotImplementedError(
                "tiled cube SPMD: held_suarez_forcing is not tiled-routed "
                "(the HS closure is a full-state physics_fn, not a column "
                "fn); run dynamics-only or <=6 devices.")
        if not active:
            return None                     # dynamics-only (dry)
        if active == {"microphysics": "kessler"}:
            from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
                make_kessler_column_physics_fn,
            )
            return make_kessler_column_physics_fn(
                self.sigma, float(cfg.dycore.dt))
        raise NotImplementedError(
            "tiled cube SPMD (6*kt^2 devices): this SIMPLE lane supports "
            "dynamics-only (all parameterizations 'none') or Kessler "
            f"microphysics alone; the configured physics {active} should "
            "have dispatched to the operator-split tiled lane "
            "(_run_operator_split_tiled_cube) — reaching this raise means "
            "the dispatch predicate and this selector disagree (a bug).")

    def _tiled_cube_unified_active(self) -> bool:
        """True when a tiled cube run must use the OPERATOR-SPLIT lane —
        the general run_amip column-local unified physics (or Held-Suarez,
        which rides the operator-split statics' hs_newtonian_relax) is
        active.  False for dynamics-only and for Kessler-ALONE, which the
        simple blocked-loop lane handles (its shipped parity gates)."""
        cfg = self.config
        if getattr(cfg, "held_suarez_forcing", False):
            return True
        active = {
            name: val for name, val in (
                ("radiation", cfg.radiation),
                ("convection", cfg.convection),
                ("turbulence", cfg.turbulence),
                ("microphysics", cfg.microphysics),
                ("gravity_wave_drag", cfg.gravity_wave_drag),
                ("cloud_scheme", cfg.cloud_scheme),
            ) if val not in (None, "none")
        }
        return bool(active) and active != {"microphysics": "kessler"}

    def _run_operator_split_tiled_cube(self, start_step, start_day,
                                       mesh, kt: int) -> str:
        """Sub-face-TILED operator-split run: the cube twin of
        :meth:`_run_operator_split_spmd` (the faithful multi-device
        ``_run_compiled`` twin) over
        ``driver/tiled_operator_split_step.make_tiled_operator_split_step``
        — the REAL unified PhysicsPipeline built at TILE ncol
        (``build_tile_step_unified``), the ``SegmentCarry`` tile-sharded
        and THREADED across segments, per-segment external forcing as a
        traced argument, per-segment gather for the callback + blowup
        guard only.

        Refused loudly: ensembles, multilayer land, the diagnostics /
        checkpoint writers (this lane; the simple tiled lane has them),
        and — inside ``build_tile_step_unified`` — land-active pipelines
        and horizontal-operator convection schemes.
        """
        import time
        from legoesm.forcing.external import get_solar_forcing_at_time
        from legoesm.driver.compiled_segments import (
            pack_carry, pack_forcing, unpack_carry,
            build_operator_split_statics, GHG_SPECIES_ORDER,
        )
        from legoesm.driver.tiled_operator_split_step import (
            build_tile_step_unified, make_tiled_operator_split_step,
            shard_tiled_split_carry, unpack_carry_tiled,
        )
        from legoesm.core.conservation import (
            compute_global_moisture, global_area_sum,
        )

        cfg = self.config
        n = int(self.grid.n)
        if jax.process_count() > 1:
            raise NotImplementedError(
                "operator-split tiled cube: multicontroller (route-B "
                "cross-process) is a follow-up; run single-process "
                "multi-device.")
        if self._ensemble_size > 1:
            raise NotImplementedError(
                "operator-split tiled cube does not support ensembles "
                "(the vmap'd carry's leading axis is the ensemble, not a "
                "tile).")
        if self._land_ml_state is not None:
            raise NotImplementedError(
                "operator-split tiled cube does not support multilayer "
                "(Richards) land (land_ml has no tile packing).")
        if self._moisture_advection_active():
            raise NotImplementedError(
                "operator-split tiled cube: resolved-wind moisture "
                "advection (moisture_advection=True) is not tiled — the "
                "tiled dynamics advects no tracers, so the serial and "
                "tiled trajectories would diverge O(1). Run column-locked "
                "moisture or <=6 devices.")
        if cfg.output.checkpoint_days > 0 or cfg.output.diag_days > 0:
            raise NotImplementedError(
                "operator-split tiled cube does not yet run the "
                "diagnostics / checkpoint writers (set diag_days=0, "
                "checkpoint_days=0; use the segment_callback for I/O).")

        ctx = self._prepare_run_context(start_step, start_day,
                                        restore_carry=True)
        DT = ctx["DT"]
        START_DAY = ctx["START_DAY"]
        n_steps_total = ctx["n_steps_total"]
        dsigma = ctx["dsigma"]
        shape_2d = ctx["shape_2d"]
        _sd = ctx["_sd"]

        # REAL unified physics at TILE ncol (refuses land / horizontal-
        # operator convection inside).
        tile_su, _tile_pipeline = build_tile_step_unified(
            self.grid, self.sigma, cfg, kt)

        ghg_vmr = ctx["ghg_vmr"]
        ghg_keys: tuple = ()
        if isinstance(ghg_vmr, dict):
            ghg_keys = tuple(k for k in GHG_SPECIES_ORDER if k in ghg_vmr)
        ghg_keys = ghg_keys or None

        _carry_aux = self._carry_aux
        _target_moisture = _carry_aux.get("target_moisture",
                                          jnp.asarray(0.0))
        if cfg.fix_moisture and float(_target_moisture) == 0.0:
            _target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid)
        _target_mass = _carry_aux.get("target_mass", jnp.asarray(0.0))
        if cfg.dycore.fix_mass and float(_target_mass) == 0.0:
            _target_mass = global_area_sum(self.state.p_s.data, self.grid)

        statics = build_operator_split_statics(
            step_unified=tile_su, forcing=None,
            lat=self.grid.grid_lat, lon=self.grid.grid_lon, dt=DT,
            tau_equator=cfg.tau_equator, tau_pole=cfg.tau_pole,
            sbm_tau_c=cfg.sbm_tau_c, sbm_RH_ref=cfg.sbm_RH_ref,
            C_H=cfg.C_H, C_E=cfg.C_E,
            albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
            ghg_vmr_override=None,
            hs_newtonian_relax=self._hs_newtonian_relax,
            energy_consistent_moisture_clip=(
                cfg.energy_consistent_moisture_clip),
            do_sat_adjust=(cfg.microphysics == "none"),
            fix_moisture=cfg.fix_moisture,
            sigma_full=ctx["sigma_full"], dsigma=dsigma, grid=self.grid,
            owned_mask=None, qv_smooth_coeff=self._qv_smooth_coeff,
            fric_decay=self._fric_decay,
            hyperdiffusion_3d=self._hyperdiffusion_3d_fn,
        )
        tiled_step = make_tiled_operator_split_step(
            self.model, mesh, statics,
            fix_mass=cfg.dycore.fix_mass,
            rad_update_steps=ctx["RAD_UPDATE_STEPS"],
            start_day=START_DAY, kt=kt, ghg_keys=ghg_keys)

        _dm = ({k: self.tracers.get(k)
                for k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}
               if isinstance(self.tracers, dict) else {})
        carry = pack_carry(
            self.state, self.q_v, self.q_c, self.q_r,
            conv_prog=ctx["conv_prog"],
            held_dT_rad=ctx["held_dT_rad"],
            held_sw_net_sfc=ctx["held_sw_net_sfc"],
            held_lw_net_sfc=ctx["held_lw_net_sfc"],
            held_sw_up_toa=ctx["held_sw_up_toa"],
            held_lw_up_toa=ctx["held_lw_up_toa"],
            held_sw_down_toa=ctx["held_sw_down_toa"],
            step_index=start_step,
            conv_precip_prev=getattr(self, "_conv_precip_prev", None),
            target_moisture=_target_moisture, target_mass=_target_mass,
            precip_accum=jnp.zeros(shape_2d, dtype=_sd),
            T_land=ctx["T_land"], w_land=ctx["w_land"],
            snow=ctx.get("snow"),
            tke=ctx["tke"], qke=ctx["qke"],
            gwd_spectrum=ctx["gwd_spectrum"],
            **_dm,
        )
        carry_template = carry
        carry = shard_tiled_split_carry(carry, mesh, n)

        logger.info(
            "operator-split TILED cube: (6, %d, %d) tiles over %d devices "
            "(%s).", kt, kt, mesh.devices.size, jax.default_backend())

        current_s_0 = ctx["current_s_0"]
        solar_weights = ctx["solar_weights"]
        o3_vmr, aerosol_od = ctx["o3_vmr"], ctx["aerosol_od"]

        seg_len = max(1, int(86400.0 / DT))
        current_step = start_step
        status = "COMPLETED"
        seg_idx = -1
        # #921: prime every NCCL clique this step uses (the halo
        # collective-permutes + the target-mass / moisture-fixer psums) in a
        # fixed, rank-independent order before the first real step, so
        # multi-process (route-B) comm-init cannot deadlock.  No-op
        # single-process (CPU-virtual / single-GPU) — those lanes are unchanged.
        from legoesm.parallel.tiled_production_cdgrid import (
            warmup_tiled_cube_comms,
        )
        warmup_tiled_cube_comms(mesh, kt)
        t0 = time.time()
        while current_step < n_steps_total:
            seg_idx += 1
            seg_steps = min(seg_len, n_steps_total - current_step)
            seg_end_step = current_step + seg_steps
            day = START_DAY + seg_end_step * DT / 86400.0
            doy, sod = self._calendar_for_radiation(day)
            sst, sic = self.get_sst_sic(day)
            if seg_idx > 0 or start_step > 0:
                _solar_now = get_solar_forcing_at_time(
                    self._solar_config, day)
                current_s_0 = float(_solar_now["tsi"])
                if self._use_solar_spectral:
                    solar_weights = jnp.asarray(
                        _solar_now["solar_fraction_by_gpt"])
                _phys_p_s, _phys_lat = self._owned_p_s_and_lat()
                o3_vmr, aerosol_od, ghg_vmr = (
                    self._precompute_external_forcing(
                        day, _phys_p_s, _phys_lat))
            _alb, _T, _emis = (None, None, None)
            if self.get_sfc_override is not None:
                _alb, _T, _emis = self.get_sfc_override(day)
            _shflx, _lhflx = (None, None)
            if self.get_sfc_flux_override is not None:
                _shflx, _lhflx = self.get_sfc_flux_override(day)
            forcing = pack_forcing(
                sst=jnp.asarray(sst), sic=jnp.asarray(sic),
                day_of_year=doy, seconds_of_day=sod,
                solar_weights=solar_weights, s_0=current_s_0,
                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
                ghg_vmr=ghg_vmr,
                sfc_albedo_override=_alb, sfc_T_override=_T,
                sfc_emissivity_override=_emis,
                sfc_shflx_override=_shflx, sfc_lhflx_override=_lhflx,
            )

            if seg_idx == 0:
                _t_jit = time.time()
            for _ in range(seg_steps):
                carry = tiled_step(carry, forcing)
            if seg_idx == 0:
                jax.block_until_ready(carry.u)
                logger.info("  operator-split tiled segment 0 (incl. JIT) "
                            "in %.1fs", time.time() - _t_jit)
            current_step += seg_steps

            # Gather: tile layout -> serial layout -> cc state for the
            # callback + blowup guard (the threaded carry never gathers).
            carry_serial = unpack_carry_tiled(carry, n, carry_template)
            state, self.q_v, self.q_c, self.q_r = unpack_carry(
                carry_serial, self.state)[:4]
            self.state = state
            self._conv_precip_prev = carry_serial.conv_precip_prev
            if isinstance(self.tracers, dict):
                for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                    _v = getattr(carry_serial, _nm)
                    if _v is not None:
                        self.tracers[_nm] = _v
            finite = bool(jnp.isfinite(state.p_s.data).all()
                          & jnp.isfinite(state.T.data).all())
            if not finite:
                status = f"BLOWUP at step {current_step}"
                logger.info("operator-split tiled cube: %s (%.1fs)",
                            status, time.time() - t0)
                self._write_blowup_state(current_step, day)
                return status
            self._current_day = day
            if self._segment_callback is not None:
                self._segment_callback(self, day, DT * seg_steps)
                # Fold callback mutations back into the threaded tiled
                # carry (the lat-band lane's contract — _run_compiled
                # re-packs from self.* every segment; codex).  The
                # driver-visible fields are grid-shaped carry leaves
                # (exact tile partition), so a per-leaf device_put onto
                # the carry's existing sharding suffices; a no-op
                # resharding when the callback does not mutate.  The
                # threaded physics carry (held radiation / tke /
                # conv_prog) keeps its sharded values.
                _fold = dict(
                    u=jax.device_put(self.state.u.data, carry.u.sharding),
                    v=jax.device_put(self.state.v.data, carry.v.sharding),
                    T=jax.device_put(self.state.T.data, carry.T.sharding),
                    p_s=jax.device_put(self.state.p_s.data,
                                       carry.p_s.sharding),
                    q_v=jax.device_put(self.q_v, carry.q_v.sharding),
                    q_c=jax.device_put(self.q_c, carry.q_c.sharding),
                    q_r=jax.device_put(self.q_r, carry.q_r.sharding))
                if isinstance(self.tracers, dict):
                    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                        _cv = getattr(carry, _nm)
                        _sv = self.tracers.get(_nm)
                        if _cv is not None and _sv is not None:
                            _fold[_nm] = jax.device_put(_sv, _cv.sharding)
                carry = carry._replace(**_fold)
        logger.info("operator-split tiled cube: %s (%.1fs)", status,
                    time.time() - t0)
        return status

    def _run_tiled_cube_spmd(self, start_step: int = 0,
                             start_day: float | None = None) -> str:
        """Sub-face-tiled cube SPMD run (``n_devices = 6*kt^2 > 6``).

        Integrates via the BLOCKED persistent tiled loop
        (:func:`legoesm.atmosphere.dynamics.gcm.tiled_step_adapter.make_tiled_cc_loop`)
        scanned into per-SEGMENT executables
        (:func:`legoesm.atmosphere.dynamics.gcm.tiled_step_adapter.scan_tiled_cc_steps`,
        M3b increment 1): state stays TILE-SHARDED across steps AND each
        segment is ONE ``lax.scan`` dispatch (no per-step host dispatch,
        no full-face all-gather inside the scan — HLO-gated by
        ``tests/parallel/test_cube_tile_native_segment.py``); a
        cell-centred ``HydrostaticState`` is gathered once per SEGMENT for
        the coupler callback + a host-side NaN-blowup guard.  A dedicated
        path, NOT the jitted ``compiled_segments`` scan — the exact
        precedent of ``_run_compiled_latlon_spmd`` (zero surgical risk to
        the shared hot loop).

        Supports DYNAMICS-ONLY or Kessler-microphysics-only configs (the
        tiled envelope; anything else is refused loudly by
        ``_tiled_cube_column_physics_fn`` / the adapter's envelope
        validation).  Writers: the lightweight ``timeseries.npz``
        diagnostics (``diag_days``) and the checkpoint writer
        (``checkpoint_days``, via the ``_checkpoint_callback``-or-
        ``save_checkpoint`` hook contract) fire on segment boundaries —
        the segment length is the gcd of the active cadences with the
        1-day coupling cadence, so every writer sees the gathered cc
        state, never the blocked in-loop state.
        """
        import time as _time

        import jax
        import numpy as _np

        from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
            make_tiled_cc_loop, scan_tiled_cc_steps,
        )

        cfg = self.config
        dc = self._device_config
        tiling = tuple(getattr(dc, "tiling", (1, 1)))
        kt = int(tiling[0])
        if (tiling[0] != tiling[1] or kt < 2
                or dc.n_devices != 6 * kt * kt):
            raise ValueError(
                f"tiled cube SPMD: device config tiling={tiling}, "
                f"n_devices={dc.n_devices} is not a (kt, kt) sub-face "
                f"tiling with 6*kt^2 devices.")
        mesh = dc.mesh
        if tuple(getattr(mesh, "axis_names", ())) != (
                "face", "tile_i", "tile_j"):
            raise ValueError(
                f"tiled cube SPMD: mesh axes {getattr(mesh, 'axis_names', None)} "
                f"!= ('face', 'tile_i', 'tile_j') — build the device mesh "
                f"via create_device_mesh(n_devices=6*kt^2).")

        # Unified-pipeline physics (anything beyond dynamics-only /
        # Kessler-alone / HS) routes to the OPERATOR-SPLIT tiled lane —
        # the faithful _run_compiled twin (tile-sharded SegmentCarry,
        # real step_unified built at tile ncol).  Kessler-alone stays on
        # this simple blocked-loop lane (its shipped parity gates).
        if self._tiled_cube_unified_active():
            return self._run_operator_split_tiled_cube(
                start_step, start_day, mesh, kt)

        column_physics_fn = self._tiled_cube_column_physics_fn()
        DT = cfg.dycore.dt
        n_steps_total = int(cfg.days * 86400.0 / DT)
        START_DAY = start_day if start_day is not None else cfg.start_day
        # Restart-time normalization (mirrors _run_compiled_latlon_spmd).
        if (start_day is not None and start_step > 0
                and getattr(self, "_loaded_checkpoint_step_day", None)
                == (start_step, start_day)):
            START_DAY = start_day - start_step * DT / 86400.0
        # Writer cadences (steps).  0 = off.  The segment length is the gcd
        # of the ACTIVE cadences with the 1-day coupling cadence, so every
        # writer fires exactly on a segment boundary (where the gathered cc
        # state exists) — never mid-loop on the blocked state.
        import math as _math

        day_steps = max(1, int(86400.0 / DT))
        diag_steps = (int(cfg.output.diag_days * 86400.0 / DT)
                      if cfg.output.diag_days > 0 else 0)
        ckpt_steps = (int(cfg.output.checkpoint_days * 86400.0 / DT)
                      if cfg.output.checkpoint_days > 0 else 0)
        for _nm, _cad in (("diag_days", diag_steps),
                          ("checkpoint_days", ckpt_steps)):
            if _cad < 0 or (getattr(cfg.output, _nm) > 0 and _cad == 0):
                raise ValueError(
                    f"tiled cube SPMD: {_nm}={getattr(cfg.output, _nm)} is "
                    f"shorter than one step (dt={DT}s).")
        seg_len = day_steps
        for _cad in (diag_steps, ckpt_steps):
            if _cad > 0:
                seg_len = _math.gcd(seg_len, _cad)
        seg_len = max(1, seg_len)
        n_run = n_steps_total - start_step
        if n_run < 1:
            return "COMPLETED"
        # Lightweight timeseries (the spectral/MPAS fallback writer's schema
        # — the AMIP validation harness's minimum contract).
        _ts: dict[str, list] = {
            "days": [], "T_atm": [], "T_min": [], "T_max": [],
            "max_wind": [], "dry_mass_ps": [], "T_finite": [],
        }

        logger.info(
            "tiled cube SPMD run: %d steps, %d-step segments, kt=%d "
            "(%d devices), physics=%s",
            n_run, seg_len, kt, dc.n_devices,
            "kessler" if column_physics_fn is not None else "dynamics-only")

        enter, tiled_step, tiled_exit = make_tiled_cc_loop(
            self.model, mesh, kt=kt, dt=float(DT),
            column_physics_fn=column_physics_fn)
        # M3b increment 1: each segment is ONE compiled lax.scan of the
        # blocked step — one host dispatch per SEGMENT instead of per step,
        # carry persistently tile-sharded, donated between segments.  At
        # most TWO distinct lengths compile (the regular segment + the
        # final remainder); the FIRST segment additionally compiles its own
        # signature (the enter carry is f32-compute until the fixer's f64
        # p_s promotion — scan_tiled_cc_steps' dtype fixed-point unroll),
        # exactly as the prior per-step lane compiled two step signatures.
        _segments: dict[int, object] = {}

        def _scanned(k: int):
            fn = _segments.get(k)
            if fn is None:
                fn = _segments[k] = scan_tiled_cc_steps(tiled_step, k)
            return fn
        # Moist: the driver keeps tracers in ``self.tracers`` (raw arrays,
        # the tracer-property store) — ``self.state.tracers`` is None after
        # cube setup.  Attach exact {q_v,q_c,q_r} Fields for the loop's
        # tracer contract (codex BLOCKER: without this, a REAL driver init
        # can never enter the advertised Kessler lane).
        if (column_physics_fn is not None
                and getattr(self.state, "tracers", None) is None):
            from legoesm.core.field import Field

            missing = [nm for nm in ("q_v", "q_c", "q_r")
                       if self.tracers.get(nm) is None]
            if missing:
                raise NotImplementedError(
                    f"tiled cube SPMD (kessler): driver tracers missing "
                    f"{missing} — initialize a moist cube state (moist IC) "
                    f"or run dynamics-only.")
            self.state = self.state._replace(tracers={
                nm: Field(data=self.tracers[nm], name=nm,
                          dims=self.state.T.dims, units="kg/kg")
                for nm in ("q_v", "q_c", "q_r")
            })
        template = self.state
        blocked = enter(self.state)

        # #921: prime every NCCL clique the blocked step uses (the halo
        # collective-permutes + the in-stage mass-fixer psum) in a fixed,
        # rank-independent order before the first real step, so multi-process
        # (route-B one-process-per-GPU) comm-init cannot deadlock.  No-op
        # single-process (CPU-virtual / single-GPU) — those lanes are unchanged.
        from legoesm.parallel.tiled_production_cdgrid import (
            warmup_tiled_cube_comms,
        )
        warmup_tiled_cube_comms(mesh, kt)

        t0 = _time.time()
        step_done = 0
        while step_done < n_run:
            seg_n = min(seg_len, n_run - step_done)
            blocked = _scanned(seg_n)(blocked)
            step_done += seg_n
            # Per-SEGMENT gather: coupler callback + host blowup guard
            # (the in-loop state never gathers).  Callbacks CONSUME the
            # gathered state (the lat-lon SPMD lane's contract); mutations
            # are NOT folded back into the blocked loop state.
            self.state = tiled_exit(blocked, template)
            if self.state.tracers is not None:
                # Keep the canonical driver tracer-property store
                # (self.tracers -> q_v/q_c/q_r properties, used by
                # callbacks/diagnostics) in sync with the advanced
                # moisture (codex MAJOR: it otherwise holds the INITIAL
                # fields for the whole run).
                for _nm, _f in self.state.tracers.items():
                    self.tracers[_nm] = _f.data
            day = START_DAY + (start_step + step_done) * DT / 86400.0
            self._current_day = day
            if not bool(_np.all(_np.isfinite(
                    _np.asarray(self.state.T.data)))):
                logger.error("tiled cube SPMD: non-finite T at day %.3f",
                             day)
                if diag_steps > 0:
                    self._save_lightweight_timeseries(
                        _ts, f"BLOWUP at day {day:.3f}", t0)
                # State is gathered at this point (tiled_exit above), so the
                # forensic dump sees the full failing state.
                self._write_blowup_state(start_step + step_done, day)
                return f"BLOWUP at day {day:.3f}"
            abs_step = start_step + step_done
            if diag_steps > 0 and abs_step % diag_steps == 0:
                s = self.state
                _stats = _np.asarray(jnp.stack([
                    jnp.mean(s.T.data), jnp.min(s.T.data),
                    jnp.max(s.T.data), jnp.mean(s.p_s.data),
                    jnp.max(jnp.sqrt(s.u.data ** 2 + s.v.data ** 2)),
                ]))
                _ts["days"].append(day - self.config.start_day)
                _ts["T_atm"].append(float(_stats[0]))
                _ts["T_min"].append(float(_stats[1]))
                _ts["T_max"].append(float(_stats[2]))
                _ts["dry_mass_ps"].append(float(_stats[3]))
                _ts["max_wind"].append(float(_stats[4]))
                _ts["T_finite"].append(True)   # guarded above
            if ckpt_steps > 0 and abs_step % ckpt_steps == 0:
                # run()'s established hook contract (the _run_compiled
                # pattern): a coupled driver's _checkpoint_callback owns the
                # FULL coupled state; else the driver's own writer on the
                # gathered cc state.
                _ckpt = (getattr(self, "_checkpoint_callback", None)
                         or self.save_checkpoint)
                _ckpt(abs_step, day)
            if self._segment_callback is not None:
                self._segment_callback(self, day, DT * seg_n)
        if diag_steps > 0:
            self._save_lightweight_timeseries(_ts, "COMPLETED", t0)
        logger.info("tiled cube SPMD run: COMPLETED (%.1fs)",
                    _time.time() - t0)
        return "COMPLETED"

    def _operator_split_spmd_active(self) -> bool:
        """True when the lat-band SPMD run must use the OPERATOR-SPLIT lane — the
        general run_amip COLUMN-LOCAL unified physics is active.

        False for ``held_suarez_forcing`` and dynamics-only (all schemes
        ``'none'``), which the stateless :func:`run_atm_latlon_spmd` lane handles.
        The only decomposition-VARIANT unified physics — stochastic Bechtold
        (``enable_stochastic=True``, a per-column PRNG keyed by global column) —
        is refused UPSTREAM at pipeline build (``physics_pipeline`` raises before
        this lane runs), and every radiation scheme (gray / rrtmgp / rrtmg) is
        1-D column, so no extra column-locality guard is needed here."""
        cfg = self.config
        if cfg.held_suarez_forcing:
            return False
        return any(v not in (None, "none") for v in (
            cfg.radiation, cfg.convection, cfg.turbulence,
            cfg.microphysics, cfg.gravity_wave_drag, cfg.cloud_scheme))

    def _run_operator_split_spmd(self, start_step, start_day, mesh) -> str:
        """Lat-band-SPMD OPERATOR-SPLIT run: the faithful multi-device twin of
        :meth:`_run_compiled` for the general run_amip column-local unified
        physics.

        Integrates dynamics -> dry-mass fixer -> ``step_unified`` physics -> Euler
        write-back -> moisture/saturation/Rayleigh tail with the ``SegmentCarry``
        sharded by LATITUDE BAND and THREADED across segments (held radiation /
        tke / qke / conv_prog / double-moment tracers all ride the carry, so no
        per-segment unpack/repack).  Host Python runs only between segments to
        re-sample the external forcing (SST/SIC/solar/GHG/ozone/aerosol) and
        gather a cell-centered ``HydrostaticState`` COPY for the segment callback
        + a NaN/Inf blow-up guard — the sharded carry is never gathered back into
        the integration.  Column-local physics is decomposition-INVARIANT, so
        this is bit-faithful to ``_run_compiled`` up to the limited-FV-PPM
        cut-boundary residual the sharded dynamics already carries.

        Refused LOUDLY (carry has no lat-band partition spec): ensembles (the
        vmap'd carry's leading axis is the ensemble, not the band) and multilayer
        (Richards) land (single-rank-validated).  The diagnostics / checkpoint
        writers are refused by :meth:`_run_compiled_latlon_spmd`."""
        import time
        from jax.sharding import NamedSharding, PartitionSpec as P
        from legoesm.forcing.external import get_solar_forcing_at_time
        from legoesm.driver.compiled_segments import (
            pack_carry, pack_forcing, unpack_carry,
            build_operator_split_statics, GHG_SPECIES_ORDER,
        )
        from legoesm.driver.sharded_operator_split_step import (
            make_sharded_operator_split_step,
            shard_operator_split_carry, shard_operator_split_forcing,
        )
        from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
            build_band_grids_atm,
        )
        from legoesm.core.conservation import (
            compute_global_moisture, global_area_sum,
        )
        from legoesm.parallel.latlon_spmd import replicate_leaf

        cfg = self.config
        n_dev = mesh.devices.size
        # Route-B multicontroller (jax.distributed cross-process NCCL): the mesh
        # spans devices across processes. ``_mp`` gates the cross-process gather
        # (replicate_leaf's jit-identity all-gather vs the single-process
        # device_put) and the rank-0 log gating; both are no-ops when
        # process_count()==1 (single-controller / serial), so never-regress.
        _mp = jax.process_count() > 1
        _io_rank = jax.process_index() == 0

        # --- Refusals: the carry has no lat-band partition spec for these ---
        if self._ensemble_size > 1:
            raise NotImplementedError(
                "operator-split lat-band SPMD does not support ensembles (the "
                "vmap'd carry's leading axis is the ensemble, not the lat band)"
                " — run ensemble members as separate single-member SPMD jobs.")
        if self._land_ml_state is not None:
            raise NotImplementedError(
                "operator-split lat-band SPMD does not yet support multilayer "
                "(Richards) land (use_multilayer_land): the prognostic soil "
                "column is single-rank-validated (rides carry.land_ml with no "
                "band spec).  Use the slab land tile for the SPMD lane.")

        ctx = self._prepare_run_context(start_step, start_day, restore_carry=True)
        DT = ctx["DT"]
        START_DAY = ctx["START_DAY"]
        N_DAYS = ctx["N_DAYS"]
        n_steps_total = ctx["n_steps_total"]
        dsigma = ctx["dsigma"]
        shape_2d = ctx["shape_2d"]
        _sd = ctx["_sd"]
        _seg_lat = (self._physics_lat if self._physics_lat is not None
                    else self._grid_lat)
        _seg_lon = (self._physics_lon if self._physics_lon is not None
                    else self._grid_lon)
        # The sharded step takes each band's physics lat/lon from
        # band_geom.grid_lat/grid_lon (the band-sliced 2D cell fields).  For this
        # SINGLE-PROCESS lane (enable_latlon_spmd enforces distributed=False, so
        # _physics_lat is never the MPI band-sliced lat2d/scatter variant),
        # _seg_lat/_seg_lon resolve to grid.grid_lat/grid_lon — EXACTLY the
        # global field those per-band slices reconstruct — so band_geom.grid_lat
        # is bit-faithful to the serial physics lat.  Assert it so a future
        # rank-local _physics_lat can never silently ride this lane with a
        # mismatched column coordinate (codex review).
        assert (np.array_equal(np.asarray(_seg_lat),
                               np.asarray(self.grid.grid_lat))
                and np.array_equal(np.asarray(_seg_lon),
                                   np.asarray(self.grid.grid_lon))), (
            "operator-split SPMD: physics lat/lon differ from grid.grid_lat/lon "
            "(unexpected single-process); band_geom.grid_lat would diverge from "
            "the serial physics column coordinate.")

        # --- BAND step_unified: build the physics pipeline on ONE band grid so
        # its ColumnAdapter bakes the band ncol (bands are uniform, so one build
        # serves every band — the make_sharded_operator_split_step CONTRACT). ---
        band_grid = build_band_grids_atm(self.grid, n_dev)[0]
        band_physics = build_physics_pipeline(band_grid, self.sigma, cfg)
        band_su = band_physics.build_step_unified(static_need_rad=True)

        # --- GHG species order (mirrors build_segment_fn): the sharded step
        # rebuilds ghg_vmr_override from the per-segment forcing.ghg_vmr; None
        # keys (gray / no GHG) leaves the (None) override untouched. ---
        ghg_vmr = ctx["ghg_vmr"]
        ghg_keys: tuple = ()
        if isinstance(ghg_vmr, dict):
            ghg_keys = tuple(k for k in GHG_SPECIES_ORDER if k in ghg_vmr)
        ghg_keys = ghg_keys or None

        # --- Conservation targets (from carry_aux restore or the IC; the full
        # grid is un-sharded here, so global_area_sum is a plain global sum). ---
        _carry_aux = self._carry_aux
        _target_moisture = _carry_aux.get("target_moisture", jnp.asarray(0.0))
        if cfg.fix_moisture and float(_target_moisture) == 0.0:
            _target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid)
        _target_mass = _carry_aux.get("target_mass", jnp.asarray(0.0))
        if cfg.dycore.fix_mass and float(_target_mass) == 0.0:
            _target_mass = global_area_sum(self.state.p_s.data, self.grid)

        # --- Global operator-split statics (band step_unified; grid/lat/lon/
        # forcing are swapped per band inside the sharded step; ghg_vmr_override
        # is rebuilt per segment from the forcing when ghg_keys is not None). ---
        statics = build_operator_split_statics(
            step_unified=band_su, forcing=None,
            lat=_seg_lat, lon=_seg_lon, dt=DT,
            tau_equator=cfg.tau_equator, tau_pole=cfg.tau_pole,
            sbm_tau_c=cfg.sbm_tau_c, sbm_RH_ref=cfg.sbm_RH_ref,
            C_H=cfg.C_H, C_E=cfg.C_E,
            albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
            ghg_vmr_override=None,
            hs_newtonian_relax=self._hs_newtonian_relax,
            energy_consistent_moisture_clip=cfg.energy_consistent_moisture_clip,
            do_sat_adjust=(cfg.microphysics == "none"),
            fix_moisture=cfg.fix_moisture,
            sigma_full=ctx["sigma_full"], dsigma=dsigma, grid=self.grid,
            owned_mask=None, qv_smooth_coeff=self._qv_smooth_coeff,
            fric_decay=self._fric_decay,
            hyperdiffusion_3d=self._hyperdiffusion_3d_fn,
        )
        sharded_step = make_sharded_operator_split_step(
            self.model, mesh, statics,
            fix_mass=cfg.dycore.fix_mass,
            rad_update_steps=ctx["RAD_UPDATE_STEPS"],
            start_day=START_DAY, ghg_keys=ghg_keys)

        # --- Seed the carry ONCE (cell-centered, full grid) and shard it; it
        # threads across every segment (held radiation / tke / conv_prog ride
        # it), so unlike _run_compiled there is no per-segment pack/unpack. ---
        _dm = ({k: self.tracers.get(k)
                for k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}
               if isinstance(self.tracers, dict) else {})
        carry = pack_carry(
            self.state, self.q_v, self.q_c, self.q_r,
            conv_prog=ctx["conv_prog"],
            held_dT_rad=ctx["held_dT_rad"],
            held_sw_net_sfc=ctx["held_sw_net_sfc"],
            held_lw_net_sfc=ctx["held_lw_net_sfc"],
            held_sw_up_toa=ctx["held_sw_up_toa"],
            held_lw_up_toa=ctx["held_lw_up_toa"],
            held_sw_down_toa=ctx["held_sw_down_toa"],
            step_index=start_step,
            # Seed the lagged convective-cloud precip (radiation runs before
            # convection). _run_compiled seeds this too; without it a restart
            # with active convection would run the first SPMD step on the
            # pack_carry default (zeros) while serial uses the restored lag
            # (codex #789-F2).  Ensembles are refused above, so the single-member
            # path is unconditional (matches _run_compiled's ensemble_size==1 arm).
            conv_precip_prev=getattr(self, "_conv_precip_prev", None),
            target_moisture=_target_moisture, target_mass=_target_mass,
            precip_accum=jnp.zeros(shape_2d, dtype=_sd),
            T_land=ctx["T_land"], w_land=ctx["w_land"], snow=ctx.get("snow"),
            tke=ctx["tke"], qke=ctx["qke"], gwd_spectrum=ctx["gwd_spectrum"],
            **_dm,
        )
        carry = shard_operator_split_carry(carry, mesh)

        if _io_rank:
            _lane = "route-B multicontroller" if _mp else "single-controller"
            logger.info(
                "operator-split SPMD (%s): lat-band sharded over %d device(s) "
                "across %d process(es) (%s).",
                _lane, n_dev, jax.process_count(), jax.default_backend())

        # Per-segment external forcing: seg 0 of a FRESH run uses the START_DAY
        # precompute in ctx (byte-faithful to _run_compiled); later segments (or
        # a resumed run) re-sample at the segment-end day.
        current_s_0 = ctx["current_s_0"]
        solar_weights = ctx["solar_weights"]
        o3_vmr, aerosol_od = ctx["o3_vmr"], ctx["aerosol_od"]

        seg_len = max(1, int(86400.0 / DT))
        rep = NamedSharding(mesh, P())
        current_step = start_step
        status = "COMPLETED"
        seg_idx = -1
        t0 = time.time()
        while current_step < n_steps_total:
            seg_idx += 1
            seg_steps = min(seg_len, n_steps_total - current_step)
            seg_end_step = current_step + seg_steps
            day = START_DAY + seg_end_step * DT / 86400.0
            doy, sod = self._calendar_for_radiation(day)
            sst, sic = self.get_sst_sic(day)
            if seg_idx > 0 or start_step > 0:
                _solar_now = get_solar_forcing_at_time(self._solar_config, day)
                current_s_0 = float(_solar_now["tsi"])
                if self._use_solar_spectral:
                    solar_weights = jnp.asarray(
                        _solar_now["solar_fraction_by_gpt"])
                _phys_p_s, _phys_lat = self._owned_p_s_and_lat()
                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
                    day, _phys_p_s, _phys_lat)
            _alb, _T, _emis = (None, None, None)
            if self.get_sfc_override is not None:
                _alb, _T, _emis = self.get_sfc_override(day)
            _shflx, _lhflx = (None, None)
            if self.get_sfc_flux_override is not None:
                _shflx, _lhflx = self.get_sfc_flux_override(day)
            forcing = pack_forcing(
                sst=jnp.asarray(sst), sic=jnp.asarray(sic),
                day_of_year=doy, seconds_of_day=sod,
                solar_weights=solar_weights, s_0=current_s_0,
                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
                ghg_vmr=ghg_vmr,
                sfc_albedo_override=_alb, sfc_T_override=_T,
                sfc_emissivity_override=_emis,
                sfc_shflx_override=_shflx, sfc_lhflx_override=_lhflx,
            )
            forcing = shard_operator_split_forcing(forcing, mesh)

            if seg_idx == 0:
                _t_jit = time.time()
            for _ in range(seg_steps):
                carry = sharded_step(carry, forcing)
            if seg_idx == 0:
                jax.block_until_ready(carry.u)
                if _io_rank:
                    logger.info("  operator-split SPMD segment 0 (incl. JIT) in "
                                "%.1fs", time.time() - _t_jit)
            current_step += seg_steps

            # Gather the sharded carry to a replicated cell-centered copy for
            # the callback + NaN guard.  ``replicate_leaf`` is the shared gather
            # primitive: single-process -> jax.device_put (byte-identical);
            # route-B (``_mp``) -> a jit-identity with replicated out_shardings
            # (the supported cross-process all-gather — a top-level device_put
            # cannot reshard shards living on other processes' devices).
            carry_full = jax.tree.map(
                lambda x: replicate_leaf(x, rep, multiprocess=_mp), carry)
            state, self.q_v, self.q_c, self.q_r = unpack_carry(
                carry_full, self.state)[:4]
            self.state = state
            # Refresh the driver-visible carry attributes _run_compiled also
            # writes back (so a segment callback observes the same self.* as the
            # serial path): the lagged convective precip + evolved double-moment
            # tracers.  Held radiation / conv_prog / tke are LOCALS in the serial
            # loop (not self.*), so they stay on the threaded carry — a callback
            # cannot read them from self in either path.
            self._conv_precip_prev = carry_full.conv_precip_prev
            if isinstance(self.tracers, dict):
                for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                    _v = getattr(carry_full, _nm)
                    if _v is not None:
                        self.tracers[_nm] = _v
            finite = bool(jnp.isfinite(state.p_s.data).all()
                          & jnp.isfinite(state.T.data).all())
            if not finite:
                # ``finite`` reads the REPLICATED gathered state, so every rank
                # sees the identical verdict and returns in lockstep (no
                # process_allgather needed — unlike a per-rank wallclock timer).
                status = f"BLOWUP at step {current_step}"
                if _io_rank:
                    logger.info("operator-split SPMD run: %s (%.1fs)",
                                status, time.time() - t0)
                # Replicated verdict -> all ranks call in lockstep; the
                # helper's filesystem ops are root-gated internally.
                self._write_blowup_state(current_step, day)
                return status
            # current_step == seg_end_step now, so ``day`` is the segment-end day.
            self._current_day = day
            if self._segment_callback is not None:
                self._segment_callback(self, day, DT * seg_steps)
                # Fold any callback-applied change to the atm PROGNOSTIC state
                # back into the threaded sharded carry (a coupled/DA callback may
                # nudge state/moisture; _run_compiled picks this up by re-packing
                # from self.* each segment).  Re-shard the (possibly mutated)
                # replicated fields onto the carry's existing per-leaf sharding;
                # a no-op resharding of identical values when the callback does
                # not mutate.  The non-driver carry fields (held radiation, tke,
                # conv_prog) keep their threaded sharded values.
                _fold = dict(
                    u=jax.device_put(self.state.u.data, carry.u.sharding),
                    v=jax.device_put(self.state.v.data, carry.v.sharding),
                    T=jax.device_put(self.state.T.data, carry.T.sharding),
                    p_s=jax.device_put(self.state.p_s.data, carry.p_s.sharding),
                    q_v=jax.device_put(self.q_v, carry.q_v.sharding),
                    q_c=jax.device_put(self.q_c, carry.q_c.sharding),
                    q_r=jax.device_put(self.q_r, carry.q_r.sharding))
                # Double-moment tracers: a DA/coupling callback may nudge
                # self.tracers[q_i…N_i]; fold those back too (same repack-from-self
                # _run_compiled does), else a double-moment run diverges after a
                # tracer-mutating callback (codex #789-F4).  Only carry-threaded
                # (non-None) fields are re-shardable.
                if isinstance(self.tracers, dict):
                    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                        _cv, _sv = getattr(carry, _nm), self.tracers.get(_nm)
                        if _cv is not None and _sv is not None:
                            _fold[_nm] = jax.device_put(_sv, _cv.sharding)
                carry = carry._replace(**_fold)

        if _io_rank:
            # Rank-0 completion summary. self.state is the last segment's
            # REPLICATED gathered copy, so the max reductions are valid on any
            # rank; the selfspawn route-B gate greps this line + the finite
            # magnitudes to confirm the federation integrated without a hang.
            _maxT = float(jnp.abs(self.state.T.data).max())
            _maxU = float(jnp.abs(self.state.u.data).max())
            logger.info(
                "operator-split SPMD run: %s, %d steps, %d device(s) / %d "
                "process(es), max|T|=%.3f max|u|=%.3f (%.1fs)",
                status, n_steps_total - start_step, n_dev,
                jax.process_count(), _maxT, _maxU, time.time() - t0)
        return status

    def _prepare_run_context(self, start_step, start_day, restore_carry=False):
        """Prepare shared state for a run loop.

        Returns a dict with all derived quantities both run paths need:
        intervals, shapes, solar forcing, physics step, external forcing,
        held radiation arrays, and conservation targets.
        """
        from legoesm.forcing.external import get_solar_forcing_at_time

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        START_DAY = start_day if start_day is not None else cfg.start_day
        # Restart-time normalization (FIX_RESTART_TIME, ported from
        # fix/persist-physics where it is production-validated): these
        # loops index time as ``START_DAY + ABSOLUTE_step * DT/86400``,
        # so START_DAY must be the EPOCH day (day at step 0).  Every
        # production caller (run_amip --restart-from, the coupled
        # drivers) passes the CHECKPOINT day from load_checkpoint — which
        # double-counted the already-elapsed time and ran every restarted
        # link with forcing shifted forward by the checkpoint day
        # (seasonally wrong SST / calendar; the restarted segment saw
        # day_of_year 4 instead of 3 in the validation harness).
        # Normalize ONLY when the caller passes back EXACTLY what this
        # driver's load_checkpoint returned (the recorded hint) — a
        # caller passing its own (e.g. epoch) start_day keeps the legacy
        # epoch semantics verbatim.  The MPAS loop keeps its own
        # local-step + checkpoint-day contract.
        if (start_day is not None and start_step > 0
                and self._loaded_checkpoint_step_day
                == (start_step, start_day)):
            START_DAY = start_day - start_step * DT / 86400.0
        RAD_UPDATE_STEPS = cfg.rad_update_steps

        n_steps_total = int(N_DAYS * 86400 / DT)
        diag_interval = int(cfg.output.diag_days * 86400 / DT)
        checkpoint_interval = (
            int(cfg.output.checkpoint_days * 86400 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )

        sigma_full = self.sigma.sigma_full
        dsigma = self.sigma.dsigma

        # Use actual state shape (rank-local after MPI scatter, global otherwise)
        shape_2d = self.state.p_s.data.shape
        if self._ensemble_size > 1:
            shape_2d = shape_2d[1:]
        shape_3d = (*shape_2d, cfg.grid.nlev)
        conv_ncol = int(self.physics.adapter.ncol) if self.physics is not None else int(np.prod(shape_2d))

        # Solar forcing
        solar_init = get_solar_forcing_at_time(self._solar_config, START_DAY)
        current_s_0 = float(solar_init["tsi"])
        solar_weights = (
            jnp.asarray(solar_init["solar_fraction_by_gpt"])
            if self._use_solar_spectral
            else self._solar_weights_template
        )

        # Issue #316: build two ``step_unified`` variants — one that
        # always calls radiation (``static_need_rad=True``) and one
        # that always uses held radiation (``static_need_rad=False``).
        # Both elide the inner ``lax.cond``, which lets
        # ``build_segment_fn`` run a cond-free subcycled scan and
        # keeps XLA JIT time bounded at long ``segment_length``.
        # When ``rad_update_steps <= 1`` the no-rad variant is unused
        # but the rad-only variant still helps by collapsing the
        # legacy ``lax.cond(True, _, _)`` to a single branch.
        step_unified = self.physics.build_step_unified(static_need_rad=True)
        step_unified_no_rad = (
            self.physics.build_step_unified(static_need_rad=False)
            if RAD_UPDATE_STEPS > 1 else None
        )

        # Held radiation arrays — restore from carry or zero-init
        _ens = self._ensemble_size
        _ens_3d = (_ens, *shape_3d) if _ens > 1 else shape_3d
        _ens_2d = (_ens, *shape_2d) if _ens > 1 else shape_2d
        _aux = self._carry_aux if restore_carry else {}
        _sd = self.state.T.data.dtype  # inherit storage dtype from state
        held_dT_rad = _aux.get("held_dT_rad", jnp.zeros(_ens_3d, dtype=_sd))
        held_sw_net_sfc = _aux.get("held_sw_net_sfc", jnp.zeros(_ens_2d, dtype=_sd))
        held_lw_net_sfc = _aux.get("held_lw_net_sfc", jnp.zeros(_ens_2d, dtype=_sd))
        held_sw_up_toa = _aux.get("held_sw_up_toa", jnp.zeros(_ens_2d, dtype=_sd))
        held_lw_up_toa = _aux.get("held_lw_up_toa", jnp.zeros(_ens_2d, dtype=_sd))
        held_sw_down_toa = _aux.get("held_sw_down_toa", jnp.zeros(_ens_2d, dtype=_sd))
        # Clear-sky held TOA up-fluxes (#843): restored from a checkpoint's
        # diag_accumulators when present, else zeros (cold start / pre-#843).
        held_sw_up_toa_clr = _aux.get(
            "held_sw_up_toa_clr", jnp.zeros(_ens_2d, dtype=_sd))
        held_lw_up_toa_clr = _aux.get(
            "held_lw_up_toa_clr", jnp.zeros(_ens_2d, dtype=_sd))
        # Convective carry: scalar (ncol,) for mass_flux/edmf, full
        # (ncol, nlev) conv_prog_profile for the profile-prognostic
        # schemes (ZM/KF/Emanuel/Tiedtke/Bechtold).  The shape must be
        # seeded correctly HERE — inside the compiled segment scan the
        # pipeline cannot re-shape the carry.
        from legoesm.atmosphere.physics.convection.integration import (
            convection_scheme_traits,
        )
        _nlev = int(self.sigma.sigma_full.shape[0])
        if convection_scheme_traits(cfg.convection).is_profile_prognostic:
            conv_cell_shape = (conv_ncol, _nlev)
        else:
            conv_cell_shape = (conv_ncol,)
        conv_shape = (_ens, *conv_cell_shape) if _ens > 1 else conv_cell_shape
        if cfg.convection in ("mass_flux", "edmf"):
            from legoesm.atmosphere.physics.convection.config import ConvectionConfig
            _conv_cfg = ConvectionConfig(scheme=cfg.convection)
        else:
            _conv_cfg = None

        if cfg.convection == "mass_flux":
            conv_prog_default = jnp.full(conv_shape, _conv_cfg.mass_flux.M_c_init, dtype=_sd)
        elif cfg.convection == "edmf":
            conv_prog_default = jnp.full(conv_shape, _conv_cfg.edmf.a_u_init, dtype=_sd)
        else:
            conv_prog_default = jnp.zeros(conv_shape, dtype=_sd)
        conv_prog = _aux.get("conv_prog", conv_prog_default)
        # Scheme-tag check (codex round 8): a restored carry whose
        # SCHEME differs from the configured one must not be reused
        # even when the shape matches (mass_flux M_c fed as edmf a_u).
        # Tag absent = legacy checkpoint — accept with a loud warning.
        _restored_conv_scheme = _aux.get("conv_prog_scheme")
        if "conv_prog" in _aux:
            if _restored_conv_scheme is None:
                if cfg.convection not in ("none",):
                    logger.warning(
                        "  Restored conv_prog carries no scheme tag "
                        "(legacy checkpoint) — assuming it matches "
                        "convection=%r", cfg.convection,
                    )
            elif str(_restored_conv_scheme) != str(cfg.convection):
                raise ValueError(
                    f"Restored convection carry was written by scheme "
                    f"{str(_restored_conv_scheme)!r} but this run "
                    f"configures convection={cfg.convection!r} — reusing "
                    "it would feed one scheme's memory to another "
                    "(issue #405/#413).  Fix the config or remove the "
                    "conv_prog entries from the checkpoint to opt into "
                    "a fresh seed."
                )
        if tuple(conv_prog.shape) != conv_shape:
            # A RESTORED conv_prog with the wrong shape is a checkpoint/
            # config mismatch — profile-prognostic convection is a real
            # carry (Tiedtke relaxes the previous updraft profile), so
            # silently reseeding here is the #405 bug class (codex
            # round 6; mirrors the tke/qke/gwd_spectrum rule above).
            # Legacy checkpoints written before the convection scheme
            # changed must drop the stale carry_conv_prog entry to opt
            # into a fresh seed.
            raise ValueError(
                f"Restored convection carry 'conv_prog' has shape "
                f"{tuple(conv_prog.shape)} but convection="
                f"{cfg.convection!r} expects {tuple(conv_shape)} — "
                "checkpoint and configuration do not match (scheme or "
                "resolution change, or corruption).  Fix the config or "
                "remove the carry from the checkpoint to opt into a "
                "fresh seed (issue #405/#413)."
            )

        # Land skin temperature — restored from the checkpoint when available,
        # else seeded from the lowest model-level air temperature (the thin
        # slab / skin equilibrates within ~1 day).  ``None`` when the land tile
        # is inactive (ocean-only run).
        if self.physics is not None and self.physics.f_land is not None:
            if "T_land" in _aux:
                T_land = _aux["T_land"]
            elif (getattr(self.config, "land_ic_path", "")
                  and self._land_ml_state is not None):
                # Spun-up land IC (#746): seed the skin from the equilibrated
                # TOP-SOIL temperature so the FIRST physics step's land
                # turbulent fluxes (tiled BL sensible/latent, land q_sfc) are
                # consistent with the spun-up column — else the cold-start air
                # temp would leak the day-0 shock into the BL at the closure
                # seam, undoing part of the spin-up.  T_soil[:, 0] is (ncol,);
                # reshape to the gridded T_land layout (row-major, the inverse
                # of the flatten the land init used).
                T_land = jnp.asarray(
                    self._land_ml_state.T_soil[:, 0]
                ).reshape(self.state.T.data[..., -1].shape).astype(_sd)
            else:
                T_land = self.state.T.data[..., -1].astype(_sd)
        else:
            T_land = None

        # Prognostic soil-water bucket [kg/m^2] — restored from the
        # checkpoint when available, otherwise seeded at a fraction of the
        # bucket capacity.  ``None`` (no carry, byte-identical legacy path)
        # unless the bucket is active.
        if (self.physics is not None and self.physics.f_land is not None
                and getattr(self.physics, "land_soil_bucket", False)):
            _w_init = (self.physics.land_bucket_w_max
                       * self.physics.land_bucket_w_init_frac)
            w_land = _aux.get(
                "w_land",
                jnp.full_like(self.state.p_s.data.astype(_sd), _w_init),
            )
        else:
            w_land = None

        # Prognostic snow water equivalent [kg/m^2] (snow-albedo feedback) —
        # restored from the checkpoint when available, else a zero cold start.
        # ``None`` (byte-identical legacy path) unless the feedback is active.
        if (self.physics is not None and self.physics.f_land is not None
                and getattr(self.physics, "snow_albedo_feedback", False)):
            snow = _aux.get(
                "snow",
                jnp.zeros_like(self.state.p_s.data.astype(_sd)),
            )
        else:
            snow = None

        # Stateful-physics carries (issue #413): prognostic turbulent
        # energy (tke / qke) and the prognostic-spectral GWD wave-action
        # spectrum, seeded via the canonical ``init_physics_state`` and
        # restored from the checkpoint's carry_aux when present.  All
        # ``None`` (zero overhead, byte-identical carry) when every
        # scheme is diagnostic.  Flattened-column rank-local layout like
        # conv_prog; like conv_prog, a wrong-shape restore (scheme
        # switch across restart) re-seeds the default rather than
        # crashing the scan trace.
        from legoesm.atmosphere.physics.turbulence.integration import (
            turbulence_scheme_traits,
        )
        _turb_traits = turbulence_scheme_traits(cfg.turbulence)
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            gwd_carries_spectrum,
        )
        _gwd_prognostic = gwd_carries_spectrum(cfg.gravity_wave_drag)
        tke = qke = gwd_spectrum = None
        if _turb_traits.carries_energy or _gwd_prognostic:
            from legoesm.atmosphere.physics.combined import PhysicsConfig
            from legoesm.atmosphere.physics.turbulence import (
                TurbulenceConfig,
            )
            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
                GravityWaveDragConfig,
            )
            from legoesm.atmosphere.physics.physics_state import (
                init_physics_state,
            )
            # Seed dtype rule mirrors the MPAS loop (codex fix riding
            # 128a037e): the prognostic-spectral GWD kernel's internal
            # level scan promotes to the default float dtype (f64 under
            # x64) via its config-derived wavelength grid, so an f32
            # spectrum carry would change dtype across the scan.  Seed
            # the default dtype when that scheme is active; storage
            # dtype otherwise.
            _seed_dtype = None if _gwd_prognostic else _sd
            _seed_ps = init_physics_state(
                conv_ncol, _nlev,
                PhysicsConfig(
                    turbulence=TurbulenceConfig(scheme=cfg.turbulence),
                    gravity_wave_drag=GravityWaveDragConfig(
                        scheme=cfg.gravity_wave_drag,
                    ),
                ),
                dtype=_seed_dtype,
            )

            def _seed_carry(name, default):
                if _ens > 1:
                    default = jnp.tile(
                        default[None], (_ens,) + (1,) * default.ndim,
                    )
                if name not in _aux:
                    return default
                restored = _aux[name]
                if tuple(restored.shape) != tuple(default.shape):
                    # A PRESENT carry with the wrong shape = checkpoint /
                    # config mismatch.  Fail fast — silently reseeding
                    # is the #405 bug class (codex review).
                    raise ValueError(
                        f"Restored stateful-physics carry {name!r} has "
                        f"shape {tuple(restored.shape)} but this run "
                        f"expects {tuple(default.shape)} — checkpoint "
                        "and configuration do not match (resolution / "
                        "scheme-config / ensemble change or corruption)."
                        "  Fix the config or remove the carry from the "
                        "checkpoint to opt into a fresh seed "
                        "(issue #405/#413)."
                    )
                # Pin the restored carry to the seed dtype (the GWD
                # dtype rule above survives older f32 checkpoints).
                if restored.dtype != default.dtype:
                    restored = restored.astype(default.dtype)
                return restored

            if _turb_traits.energy_field == "tke":
                tke = _seed_carry("tke", _seed_ps.tke)
            elif _turb_traits.energy_field == "qke":
                qke = _seed_carry("qke", _seed_ps.qke)
            if _gwd_prognostic:
                gwd_spectrum = _seed_carry(
                    "gwd_spectrum", _seed_ps.gwd_spectrum,
                )

        # External forcing (rank-local p_s and lat for MPI)
        _phys_p_s, _phys_lat = self._owned_p_s_and_lat()
        o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
            START_DAY, _phys_p_s, _phys_lat,
        )

        lat_deg_grid = np.degrees(np.asarray(self._grid_lat))

        return {
            "cfg": cfg, "DT": DT, "N_DAYS": N_DAYS, "START_DAY": START_DAY,
            "RAD_UPDATE_STEPS": RAD_UPDATE_STEPS,
            "n_steps_total": n_steps_total,
            "diag_interval": diag_interval,
            "checkpoint_interval": checkpoint_interval,
            "sigma_full": sigma_full, "dsigma": dsigma,
            "shape_2d": shape_2d, "shape_3d": shape_3d,
            "current_s_0": current_s_0, "solar_weights": solar_weights,
            "step_unified": step_unified,
            "step_unified_no_rad": step_unified_no_rad,
            "held_dT_rad": held_dT_rad,
            "held_sw_net_sfc": held_sw_net_sfc,
            "held_lw_net_sfc": held_lw_net_sfc,
            "held_sw_up_toa": held_sw_up_toa,
            "held_lw_up_toa": held_lw_up_toa,
            "held_sw_down_toa": held_sw_down_toa,
            "held_sw_up_toa_clr": held_sw_up_toa_clr,
            "held_lw_up_toa_clr": held_lw_up_toa_clr,
            "conv_prog": conv_prog,
            "T_land": T_land,
            "w_land": w_land,
            "snow": snow,
            "tke": tke,
            "qke": qke,
            "gwd_spectrum": gwd_spectrum,
            "o3_vmr": o3_vmr, "aerosol_od": aerosol_od, "ghg_vmr": ghg_vmr,
            "lat_deg_grid": lat_deg_grid,
            "_sd": _sd,
        }

    def _finalize_run(self, run_status, t_jit, t_start, n_steps_total,
                      START_DAY, N_DAYS, checkpoint_interval):
        """Shared finalization: save diagnostics, results, final checkpoint."""
        jax.block_until_ready(self.state.u.data)
        total_wall = time.time() - t_start
        logger.info(f"Done: {total_wall:.1f}s wall time, status={run_status}")

        _is_root = (self._mpi_rank is None or self._mpi_rank == 0)
        # Root-only writes are wrapped so a root failure surfaces on EVERY
        # SPMD process BEFORE the final save_checkpoint (whose spmd tail
        # opens with a collective gather — a dead root there is a hang,
        # codex HIGH).  No-op (native raise) outside multi-process SPMD.
        _finalize_err: Exception | None = None
        if _is_root:
            try:
                self.diagnostics.save(self._output_dir)
                self.save_results(run_status, t_jit, total_wall)
                logger.info(self.diagnostics.print_summary())
            except Exception as e:
                _finalize_err = e
        self._spmd_barrier_on_root_error(_finalize_err)

        # Only a CLEAN run yields a restartable final checkpoint.  A BLOWUP
        # leaves the state finite-but-unphysical, and labelling that garbage
        # with the TARGET day (``START_DAY + N_DAYS``) let a SLURM ``afterok``
        # chain restart from it and skip straight to "done" — the 3-yr-chain
        # false-completion (blew up at day 515, wrote ``checkpoint_day_1095``).
        # Mirror the spectral / MPAS paths, which already gate their final
        # checkpoint on ``run_status == "COMPLETED"``.  The last PERIODIC
        # checkpoint (written at the actual elapsed day) remains the restart
        # point for a blown-up run.
        if checkpoint_interval > 0 and run_status == "COMPLETED":
            # Route through the coupled checkpoint callback when one is set
            # (mirrors the periodic path): a coupled run must persist the FULL
            # coupled state (atm + ocean + surface + CO2) at the final day too,
            # or ``run_coupled --resume`` finds the atmosphere checkpoint but
            # no ``coupled_day_*.npz`` and silently resumes with a stale ocean.
            # Single-rank only: the coupled tail (CoupledESMDriver.
            # save_checkpoint) writes rank-LOCAL ocean/surface pytrees with no
            # gather, so under mpi4jax every rank would race the same npz with
            # its own slice.  Under MPI keep the atmosphere save, which IS
            # collective-safe (gathers internally, root writes) — the
            # coupled-MPI full-state final checkpoint is a pre-existing gap
            # shared with the periodic path.
            _ckpt = ((getattr(self, "_checkpoint_callback", None)
                      if self._mpi_rank is None else None)
                     or self.save_checkpoint)
            # A COMPLETED run has NO in-progress month to resume, and
            # ``diagnostics.save`` above already finalized every bucket to
            # NetCDF WITHOUT popping — so the terminal checkpoint must carry no
            # CMOR sidecar, or a run-EXTENDING restart would re-append every
            # already-written month (duplicate time coords).  The flag rides
            # through whichever ``_ckpt`` callback writes the sidecar.
            self._suppress_cmor_sidecar = True
            try:
                _ckpt(n_steps_total, START_DAY + N_DAYS)
            finally:
                self._suppress_cmor_sidecar = False

        # Issue #275 fix A lifecycle: restore the halo backend captured
        # at activation time so subsequent drivers / tests in the same
        # process see a clean ``"local"`` (or pre-existing MPI) state.
        self._restore_halo_backend()

        return run_status

    # ==================================================================
    # Differentiable training entry (parameter calibration)
    # ==================================================================

    def build_training_segment(self, n_steps: int, day: float = 0.0):
        """Build a DIFFERENTIABLE single-segment forward for parameter calibration.

        Returns ``(run_segment_raw, carry0, forcing)``:
          * ``run_segment_raw`` — the NON-JIT, non-donating ``.raw`` segment
            (buffer donation conflicts with reverse-mode AD), composable inside
            ``eqx.filter_value_and_grad`` / ``jax.grad``.
          * ``carry0`` — the initial ``SegmentCarry`` (multilayer land seeded in
            ``setup`` when ``use_multilayer_land``; rides ``carry0.land_ml``).
          * ``forcing`` — the ``SegmentForcing`` for ``day`` (prescribed SST/SIC,
            solar, ozone, aerosol).

        The caller runs ``final = run_segment_raw(carry0, n_steps, forcing)``.
        Trainable LAND params reach the tile by assigning a *traced*
        ``LandSurfaceParams`` to ``self.physics.land_ml_params`` BEFORE the call
        inside the loss closure — the segment reads that attribute at call time,
        so gradients flow back to the params (no hot-loop refactor needed; see
        ``scripts/run/train_coupled_land_era5.py``).  ``gradient_checkpoint`` is
        forced on to bound the reverse-mode memory of the unrolled scan.

        Reuses ``_prepare_run_context`` (the same step_unified / forcing / carry
        seeds as ``_run_compiled``) so the training forward is byte-faithful to a
        production segment.  Single-rank / single-device only (the carry has no
        SPMD/MPI partition spec)."""
        from legoesm.driver.compiled_segments import (
            pack_carry, build_segment_fn, pack_forcing,
        )
        if (self._device_config is not None
                and getattr(self._device_config, "is_distributed", False)):
            raise NotImplementedError(
                "build_training_segment is single-rank / single-device only "
                "(the SegmentCarry has no partition spec for SPMD/MPI sharding)."
            )

        ctx = self._prepare_run_context(0, day, restore_carry=False)
        cfg = self.config
        DT = ctx["DT"]
        _sd = ctx["_sd"]
        _seg_lat = (self._physics_lat if self._physics_lat is not None
                    else self._grid_lat)
        _seg_lon = (self._physics_lon if self._physics_lon is not None
                    else self._grid_lon)

        # Un-jitted step (jit=False): a calibrator feeds a TRACED value into the
        # pipeline via an attribute the step reads (e.g. land_ml_params); a jitted
        # step would capture that tracer as a closure constant and leak it across
        # value_and_grad calls.  The step is inlined into run_segment's lax.scan.
        step_unified = self.physics.build_step_unified(
            static_need_rad=True, jit=False)

        run_segment = build_segment_fn(
            model=self.model,
            tiled_step_fn=self._maybe_build_tiled_step(DT),
            step_unified=step_unified,
            step_unified_no_rad=None,
            grid=self.grid,
            sigma_full=ctx["sigma_full"], dsigma=ctx["dsigma"], dt=DT,
            rad_update_steps=ctx["RAD_UPDATE_STEPS"],
            microphysics=cfg.microphysics,
            fix_moisture=cfg.fix_moisture, fix_mass=cfg.dycore.fix_mass,
            fric_decay=self._fric_decay, qv_smooth_coeff=self._qv_smooth_coeff,
            lat=_seg_lat, lon=_seg_lon, start_day=ctx["START_DAY"],
            gradient_checkpoint=True,   # remat the scan -> bounded backward memory
            hyperdiffusion_3d_fn=self._hyperdiffusion_3d_fn,
            tau_equator=cfg.tau_equator, tau_pole=cfg.tau_pole,
            sbm_tau_c=cfg.sbm_tau_c, sbm_RH_ref=cfg.sbm_RH_ref,
            C_H=cfg.C_H, C_E=cfg.C_E,
            albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
            ghg_vmr_override=ctx["ghg_vmr"],
            hs_newtonian_relax=self._hs_newtonian_relax,
            energy_consistent_moisture_clip=cfg.energy_consistent_moisture_clip,
            advect_moisture=self._moisture_advection_active(),
            pipeline=self.physics,
        )

        # Conservation-fixer targets MUST come from the IC, not zero: fix_mass is on
        # by default, so a zero dry-mass target would drain the atmosphere to NaN
        # (mirrors the targets _run_compiled computes before the segment loop).
        from legoesm.core.conservation import (
            compute_global_moisture, global_area_sum,
        )
        target_mass = (global_area_sum(self.state.p_s.data, self.grid)
                       if cfg.dycore.fix_mass else jnp.asarray(0.0))
        target_moisture = (
            compute_global_moisture(self.q_v, self.state.p_s.data,
                                    ctx["dsigma"], self.grid)
            if cfg.fix_moisture else jnp.asarray(0.0))

        day_of_year, seconds_of_day = day_to_calendar(ctx["START_DAY"])
        carry0 = pack_carry(
            self.state, self.q_v, self.q_c, self.q_r,
            conv_prog=ctx["conv_prog"],
            held_dT_rad=ctx["held_dT_rad"],
            held_sw_net_sfc=ctx["held_sw_net_sfc"],
            held_lw_net_sfc=ctx["held_lw_net_sfc"],
            held_sw_up_toa=ctx["held_sw_up_toa"],
            held_lw_up_toa=ctx["held_lw_up_toa"],
            held_sw_down_toa=ctx["held_sw_down_toa"],
            step_index=0,
            target_moisture=target_moisture, target_mass=target_mass,
            precip_accum=jnp.zeros(ctx["shape_2d"], dtype=_sd),
            T_land=ctx["T_land"], w_land=ctx["w_land"],
            land_ml=self._land_ml_state,
            tke=ctx["tke"], qke=ctx["qke"], gwd_spectrum=ctx["gwd_spectrum"],
            **(
                {k: self.tracers.get(k)
                 for k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}
                if isinstance(self.tracers, dict) else {}
            ),
        )

        if self.get_sst_sic is not None:
            sst, sic = self.get_sst_sic(day)
        else:
            sst = jnp.full(ctx["shape_2d"], 290.0, dtype=_sd)
            sic = jnp.zeros(ctx["shape_2d"], dtype=_sd)
        forcing = pack_forcing(
            sst=jnp.asarray(sst), sic=jnp.asarray(sic),
            day_of_year=day_of_year, seconds_of_day=seconds_of_day,
            solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
            ghg_vmr=ctx["ghg_vmr"],
            # NOTE: transient cover is deliberately NOT injected here.  The training
            # segment returns run_segment.raw (un-jitted), so the land calibration
            # differentiates w.r.t. the ``pipe.land_ml_params`` ATTRIBUTE (read fresh
            # each trace — see test_build_training_segment_land_gradient); a non-None
            # land_ml_params forcing leaf would SHADOW that attribute and break the
            # gradient.  The un-jitted path never had the closure-bake problem the
            # traced arg fixes (that is a production-only, jitted _run_compiled fix).
        )
        # SPMD: commit grid-shaped forcing leaves to the state's sharding
        # (no-op single-device / mpi4jax-distributed) — see shard_forcing.
        from legoesm.driver.compiled_segments import shard_forcing
        forcing = shard_forcing(forcing, self._device_config)
        return run_segment.raw, carry0, forcing

    # ==================================================================
    # Compiled segment execution path
    # ==================================================================

    def _run_compiled(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run using compiled segments (jax.lax.scan over N steps).

        The hot integration loop is compiled into segments of
        ``segment_length`` steps.  Host Python only runs between
        segments for diagnostics, checkpoints, and forcing updates.
        """
        from legoesm.forcing.external import get_solar_forcing_at_time
        from legoesm.driver.compiled_segments import (
            pack_carry, unpack_carry,
            compute_segment_length, build_segment_fn, pack_forcing,
            shard_forcing,
        )

        ctx = self._prepare_run_context(start_step, start_day, restore_carry=True)
        cfg = ctx["cfg"]
        # Issue #405/#413: this loop now seeds and threads the
        # stateful-physics carries (tke / qke / gwd_spectrum ride the
        # SegmentCarry alongside conv_prog and persist via carry_aux),
        # so the former stateful-physics refusal is lifted here.
        # Stochastic Bechtold remains refused at pipeline build time
        # (its PRNG-key carry is not threaded).
        DT = ctx["DT"]
        N_DAYS = ctx["N_DAYS"]
        START_DAY = ctx["START_DAY"]
        RAD_UPDATE_STEPS = ctx["RAD_UPDATE_STEPS"]
        n_steps_total = ctx["n_steps_total"]
        diag_interval = ctx["diag_interval"]
        checkpoint_interval = ctx["checkpoint_interval"]
        sigma_full = ctx["sigma_full"]
        dsigma = ctx["dsigma"]
        shape_2d = ctx["shape_2d"]
        current_s_0 = ctx["current_s_0"]
        solar_weights = ctx["solar_weights"]
        step_unified = ctx["step_unified"]
        step_unified_no_rad = ctx.get("step_unified_no_rad")
        held_dT_rad = ctx["held_dT_rad"]
        held_sw_net_sfc = ctx["held_sw_net_sfc"]
        held_lw_net_sfc = ctx["held_lw_net_sfc"]
        held_sw_up_toa = ctx["held_sw_up_toa"]
        held_lw_up_toa = ctx["held_lw_up_toa"]
        held_sw_down_toa = ctx["held_sw_down_toa"]
        # Clear-sky held TOA up-fluxes (#843): persisted across segments +
        # checkpoints like the all-sky held fields; zeros on a cold start (or a
        # pre-#843 checkpoint) — refreshed at the first radiation step.
        held_sw_up_toa_clr = ctx.get(
            "held_sw_up_toa_clr", jnp.zeros_like(held_sw_up_toa))
        held_lw_up_toa_clr = ctx.get(
            "held_lw_up_toa_clr", jnp.zeros_like(held_lw_up_toa))
        conv_prog = ctx["conv_prog"]
        T_land = ctx["T_land"]
        w_land = ctx["w_land"]
        snow = ctx.get("snow")
        phys_tke = ctx["tke"]
        phys_qke = ctx["qke"]
        phys_gwd_spectrum = ctx["gwd_spectrum"]
        o3_vmr = ctx["o3_vmr"]
        aerosol_od = ctx["aerosol_od"]
        ghg_vmr = ctx["ghg_vmr"]
        lat_deg_grid = ctx["lat_deg_grid"]
        _sd = ctx["_sd"]

        # Segment computation.  fallback: with diagnostics AND
        # checkpoints disabled the GCD is empty — chunk at
        # ``forcing_update_days`` (default one model day) so host
        # boundaries (stability check, forcing update, multi-controller
        # rendezvous) fire at that cadence, not every step.  Forcing is
        # re-sampled ONLY at segment boundaries, so this knob is the
        # forcing cadence for cadence-less runs — warn when the dataset
        # is time-varying so the throttle is an explicit choice.
        _fb_days = cfg.forcing_update_days
        segment_length = compute_segment_length(
            diag_interval, checkpoint_interval, RAD_UPDATE_STEPS,
            fallback_interval=int(_fb_days * 86400 / DT),
        )
        if (diag_interval <= 0 and checkpoint_interval <= 0
                and cfg.dataset != "analytical"):
            logger.warning(
                "No diagnostics/checkpoint cadence: time-varying forcing "
                "(dataset=%r) is re-sampled only every "
                "forcing_update_days=%.3g d (segment fallback).  Set "
                "forcing_update_days or enable diagnostics for a finer "
                "cadence.", cfg.dataset, _fb_days,
            )
        if cfg.output.cmip_output and cfg.output.diag_days > 1:
            # Amon monthly means are safe at any cadence (fluxes/T_low are
            # time-integrated inside the segment scan), but the CMIP ``day``
            # table is fed one sample per diagnostic interval: at
            # diag_days=N>1 each written "day" is really an N-day-apart
            # sample and tasmin/tasmax degenerate to that sample.
            logger.warning(
                "CMIP day-table output with diag_days=%.3g > 1: daily "
                "fields are sampled every %.3g days and tasmin/tasmax are "
                "NOT true daily extremes.  Amon monthly means are "
                "unaffected (segment-accumulated).  Set diag_days=1 for "
                "meaningful day-table output.",
                cfg.output.diag_days, cfg.output.diag_days,
            )
        n_steps_remaining = n_steps_total - start_step
        n_segments = (n_steps_remaining + segment_length - 1) // segment_length

        _ens = self._ensemble_size
        _ens_2d = (_ens, *shape_2d) if _ens > 1 else shape_2d

        # Compute fixed moisture target for conservation fixer —
        # restore from checkpoint if available, else compute from IC.
        # At initialization (step 0) all ranks have identical state, so
        # owned_mask is not strictly needed, but we include it for consistency.
        from legoesm.core.conservation import compute_global_moisture, global_area_sum
        _carry_aux = self._carry_aux
        _owned_mask = None
        if self._owned_face_ids is not None:
            _owned_mask = jnp.zeros(6, dtype=jnp.float32)
            _owned_mask = _owned_mask.at[self._owned_face_ids].set(1.0)

        _target_moisture = _carry_aux.get("target_moisture", jnp.asarray(0.0))
        if cfg.fix_moisture and float(_target_moisture) == 0.0:
            _target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid,
                owned_mask=_owned_mask,
            )
            logger.info(f"  Moisture target: {float(_target_moisture):.6e} kg")

        # Compute fixed dry mass target for target-anchored conservation
        _target_mass = _carry_aux.get("target_mass", jnp.asarray(0.0))
        if cfg.dycore.fix_mass and float(_target_mass) == 0.0:
            _target_mass = global_area_sum(
                self.state.p_s.data, self.grid, owned_mask=_owned_mask,
            )
            logger.info(f"  Mass target: {float(_target_mass):.6e} Pa·m²")

        run_status = "COMPLETED"

        # Build the compiled segment function ONCE (outside the loop).
        # Per-segment forcing (SST, SIC, solar, ozone, aerosol) is now
        # passed as an explicit SegmentForcing argument to run_segment,
        # so changing forcing values does NOT trigger JIT recompilation.
        # lat/lon for physics: rank-local when MPI, global otherwise
        _seg_lat = self._physics_lat if self._physics_lat is not None else self._grid_lat
        _seg_lon = self._physics_lon if self._physics_lon is not None else self._grid_lon

        # Single-process multi-GPU SPMD (third replication site — the
        # moist/AMIP segment path): hand the device config to
        # ``build_segment_fn`` so the compiled segment pins explicit
        # carry/forcing in/out shardings instead of silently compiling
        # replicated carry compute on every device.  Gated to exactly
        # the ``_setup_parallel`` SPMD branch that face-sharded the
        # state via ``shard_state``: single-node (not MPI-distributed),
        # more than one device, a live mesh with a ``"face"`` axis
        # (the shared face-sharding leaf policy), and no
        # replicated-dynamics owned-face decomposition.  Everything
        # else (single device, MPI ranks, lat-lon band, voronoi,
        # spectral level mesh) passes ``None`` — byte-identical legacy
        # behaviour, including the non-JIT ``.raw`` training contract.
        # Sub-face tiling is excluded for now (codex r1): the tiled
        # SPMD halo path is unvalidated — ``_maybe_activate_spmd_halo_
        # backend`` and the scaling bench both restrict to face-only
        # ``tiling == (1, 1)``; widen all three together once tiled
        # halo exchange has HLO/tripwire coverage.
        #
        # Ensembles (``ensemble_size > 1``) are excluded (codex r4): the
        # segment is then dispatched under ``jax.vmap`` (see the ensemble
        # branch below), so the carry leaves are ``BatchTracer``s and the
        # segment's tracer guard (correctly) refuses to pin shardings from
        # a tracer's abstract layout — leaving pinning OFF, which is the
        # silent-replication failure this gate exists to prevent.  An
        # ensemble-aware face-sharded-under-vmap path is a separate effort
        # (the ensemble axis is leading, not the face axis); until it has
        # HLO/tripwire coverage, ensemble SPMD keeps the legacy donating
        # kernels (no behaviour change vs before this fix).
        _seg_device_config = None
        if (
            self._device_config is not None
            and not self._device_config.is_distributed
            and self._device_config.n_devices > 1
            and self._device_config.mesh is not None
            and "face" in getattr(self._device_config.mesh, "axis_names", ())
            and getattr(self._device_config, "tiling", (1, 1)) == (1, 1)
            and self._owned_face_ids is None
            and self._ensemble_size <= 1
        ):
            _seg_device_config = self._device_config

        run_segment = build_segment_fn(
            model=self.model,
            tiled_step_fn=self._maybe_build_tiled_step(DT),
            step_unified=step_unified,
            step_unified_no_rad=step_unified_no_rad,
            grid=self.grid,
            sigma_full=sigma_full,
            dsigma=dsigma,
            dt=DT,
            rad_update_steps=RAD_UPDATE_STEPS,
            microphysics=cfg.microphysics,
            fix_moisture=cfg.fix_moisture,
            fix_mass=cfg.dycore.fix_mass,
            fric_decay=self._fric_decay,
            qv_smooth_coeff=self._qv_smooth_coeff,
            lat=_seg_lat,
            lon=_seg_lon,
            start_day=START_DAY,
            gradient_checkpoint=(
                cfg.gradient_checkpoint
                if cfg.gradient_checkpoint
                else segment_length > 50
            ),
            hyperdiffusion_3d_fn=self._hyperdiffusion_3d_fn,
            tau_equator=cfg.tau_equator,
            tau_pole=cfg.tau_pole,
            sbm_tau_c=cfg.sbm_tau_c,
            sbm_RH_ref=cfg.sbm_RH_ref,
            C_H=cfg.C_H,
            C_E=cfg.C_E,
            albedo_ice=cfg.albedo_ice,
            albedo_ocean=cfg.albedo_ocean,
            ghg_vmr_override=ghg_vmr,
            owned_face_ids=self._owned_face_ids,
            hs_newtonian_relax=self._hs_newtonian_relax,
            device_config=_seg_device_config,
            energy_consistent_moisture_clip=cfg.energy_consistent_moisture_clip,
            advect_moisture=self._moisture_advection_active(),
            # Un-fused-radiation host path (ExperimentConfig.unfused_radiation,
            # default OFF): the pipeline lets build_segment_fn expose
            # run_norad_scan / run_rad so rrtmgp and the no-rad scan compile
            # as two separate executables.  Passing it is harmless when the
            # flag is off (the attributes are simply never invoked).
            pipeline=self.physics,
        )

        logger.info(
            f"Starting compiled run: {n_steps_remaining} steps, "
            f"{n_segments} segments of {segment_length} steps"
        )

        current_step = start_step
        t_jit = 0.0
        t_start = time.time()

        # Multilayer (Richards) land is validated single-device / single-rank only:
        # the prognostic land state rides the carry with no partition spec, so a
        # device-mesh shard_pytree (SPMD) or multi-rank MPI scatter would mis-handle
        # it.  Fail LOUDLY rather than silently degrade to the slab or shard a state
        # that has no sharding contract (CLAUDE.md: no silent degrade under SPMD/MPI;
        # mirrors the increment-1 single-rank scope of SegmentCarry.land_ml).
        if (self._land_ml_state is not None
                and self._device_config is not None
                and getattr(self._device_config, "is_distributed", False)):
            raise NotImplementedError(
                "use_multilayer_land is not yet supported under distributed "
                "execution (SPMD device mesh or multi-rank MPI): the multilayer "
                "land state has no partition spec and is validated single-rank "
                "only.  Run on a single device / single MPI rank, or use slab "
                "land (use_multilayer_land=False) for distributed runs."
            )

        # while (not ``range(n_segments)``): the adaptive-dt path halves
        # DT and recomputes ``n_steps_total``/``segment_length`` mid-run
        # — a fixed segment count would TRUNCATE the run after a CFL
        # halving (pre-existing; codex review 2026-06-12).
        seg_idx = -1
        while current_step < n_steps_total:
            seg_idx += 1
            seg_steps = min(segment_length, n_steps_total - current_step)
            seg_end_step = current_step + seg_steps
            day = START_DAY + seg_end_step * DT / 86400.0
            day_of_year, seconds_of_day = self._calendar_for_radiation(day)
            sst, sic = self.get_sst_sic(day)

            # Re-sample time-varying external forcing at every segment
            # boundary so transient CMIP6 runs (historical / SSP: GHG, ozone,
            # aerosol, solar all vary year-to-year — and ozone/aerosol vary
            # seasonally within a year) track the calendar.
            #
            # Previously gated behind ``RAD_UPDATE_STEPS > 1`` — but that knob
            # is the intra-run radiation sub-step CADENCE, orthogonal to forcing
            # transience.  With the default ``rad_update_steps=1`` the branch was
            # dead, so the compiled path froze ALL external forcing at the
            # START_DAY precompute (~line 4662): a 1850-2014 historical run saw
            # 1850 CO2 (and START_DAY's ozone season) for all 165 years.
            # ``_precompute_external_forcing`` is host-side numpy interpolation
            # (cheap, not in the JIT); the SegmentForcing leaves
            # (s_0/ghg_vmr/o3_vmr/aerosol_od/solar_weights) are traced arrays of
            # FIXED shape, so re-sampling changes only leaf *values* — no retrace.
            # Byte-identical for truly-constant forcing (gray radiation, or
            # constant GHG with no climatological ozone/aerosol); for runs with
            # transient or climatological forcing it (correctly) now follows the
            # calendar.  Matches the per-step path (_run_mpas / _run_spectral,
            # ~line 5447) which already re-samples every radiation step.
            #
            # ``start_step > 0`` ALSO refreshes segment 0 of a RESUMED run
            # (FIX_RESTART_TIME codex finding): the prepare-context values were
            # sampled at the epoch START_DAY, but the straight run refreshed this
            # (absolute) segment at its end day — without the refresh a restarted
            # AMIP/CMIP run's first segment uses epoch-day ozone/aerosol/GHG/solar
            # and diverges from the uninterrupted run.  Fresh runs (start_step ==
            # 0) keep the START_DAY precompute for segment 0 unchanged.
            if seg_idx > 0 or start_step > 0:
                solar_now = get_solar_forcing_at_time(self._solar_config, day)
                current_s_0 = float(solar_now["tsi"])
                if self._use_solar_spectral:
                    solar_weights = jnp.asarray(solar_now["solar_fraction_by_gpt"])
                _phys_p_s, _phys_lat = self._owned_p_s_and_lat()
                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
                    day, _phys_p_s, _phys_lat,
                )

            # Coupler-provided dynamic surface albedo / skin temperature for
            # this segment (None unless a coupled driver wired the feedback).
            _sfc_albedo_ovr, _sfc_T_ovr, _sfc_emis_ovr = (None, None, None)
            if self.get_sfc_override is not None:
                _sfc_albedo_ovr, _sfc_T_ovr, _sfc_emis_ovr = \
                    self.get_sfc_override(day)

            # Coupler-provided SHARED surface SH/LH fluxes for this segment
            # (None unless a coupled driver wired the shared-flux feedback).
            # When present the atmosphere consumes these instead of its own
            # bulk fluxes so the air-sea heat+water budget closes.
            _sfc_shflx_ovr, _sfc_lhflx_ovr = (None, None)
            if self.get_sfc_flux_override is not None:
                _sfc_shflx_ovr, _sfc_lhflx_ovr = self.get_sfc_flux_override(day)

            # Pack per-segment forcing into a SegmentForcing pytree.
            forcing = pack_forcing(
                sst=sst, sic=sic,
                day_of_year=day_of_year, seconds_of_day=seconds_of_day,
                solar_weights=solar_weights, s_0=current_s_0,
                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
                ghg_vmr=ghg_vmr,
                sfc_albedo_override=_sfc_albedo_ovr,
                sfc_T_override=_sfc_T_ovr,
                sfc_emissivity_override=_sfc_emis_ovr,
                sfc_shflx_override=_sfc_shflx_ovr,
                sfc_lhflx_override=_sfc_lhflx_ovr,
                # Transient land-use cover: this segment's re-weighted multilayer
                # land params (None unless transient_land_cover is active), fed as a
                # traced arg so the jitted step follows the cover — the 5th-issue fix.
                land_ml_params=self._transient_land_ml_params(day),
            )
            # SPMD: commit grid-shaped forcing leaves to the state's
            # sharding (no-op single-device / mpi4jax-distributed).
            forcing = shard_forcing(forcing, self._device_config)

            # Pack state into carry
            carry = pack_carry(
                self.state, self.q_v, self.q_c, self.q_r,
                conv_prog=conv_prog,
                held_dT_rad=held_dT_rad,
                held_sw_net_sfc=held_sw_net_sfc,
                held_lw_net_sfc=held_lw_net_sfc,
                held_sw_up_toa=held_sw_up_toa,
                held_lw_up_toa=held_lw_up_toa,
                held_sw_down_toa=held_sw_down_toa,
                # Clear-sky held TOA up-fluxes (#843): persisted across
                # segments (zeros when the diagnostic is off).
                held_sw_up_toa_clr=held_sw_up_toa_clr,
                held_lw_up_toa_clr=held_lw_up_toa_clr,
                step_index=current_step,
                target_moisture=_target_moisture,
                target_mass=_target_mass,
                precip_accum=jnp.zeros(_ens_2d, dtype=_sd),
                # Segment-mean flux / T_low accumulators (CMOR diurnal-alias
                # fix): reset to zero at every segment start like precip.
                sw_up_toa_accum=jnp.zeros(_ens_2d, dtype=_sd),
                lw_up_toa_accum=jnp.zeros(_ens_2d, dtype=_sd),
                # Clear-sky TOA up-flux accumulators (#843): reset each segment.
                sw_up_toa_clr_accum=jnp.zeros(_ens_2d, dtype=_sd),
                lw_up_toa_clr_accum=jnp.zeros(_ens_2d, dtype=_sd),
                sw_down_toa_accum=jnp.zeros(_ens_2d, dtype=_sd),
                sw_net_sfc_accum=jnp.zeros(_ens_2d, dtype=_sd),
                lw_net_sfc_accum=jnp.zeros(_ens_2d, dtype=_sd),
                t_low_accum=jnp.zeros(_ens_2d, dtype=_sd),
                # Persist the lagged convective-cloud precip ACROSS segment
                # boundaries (radiation runs before convection; without this the
                # lag would reset to zeros at step 0 of every segment).  Only the
                # single-member path is threaded (ensemble carries an extra axis
                # pack_carry doesn't expect); ensemble runs are not the realism
                # target, so they reset per segment.
                conv_precip_prev=(getattr(self, "_conv_precip_prev", None)
                                  if self._ensemble_size == 1 else None),
                T_land=T_land,
                # Prognostic multilayer land state (None ⇒ slab path, byte-identical
                # carry).  Seeded in _setup_multilayer_land; the segment advances it
                # and the readback below persists it across segment boundaries.
                land_ml=(self._land_ml_state
                         if self._ensemble_size == 1 else None),
                w_land=w_land,
                snow=snow,
                tke=phys_tke,
                qke=phys_qke,
                gwd_spectrum=phys_gwd_spectrum,
                # Double-moment hydrometeors (None unless the moisture registry +
                # microphysics carry them) so coupled/training radiation gets
                # droplet-number-aware r_eff AND a double-moment scheme evolves
                # its full state. None ⇒ legacy warm-rain carry.
                **(
                    {k: self.tracers.get(k)
                     for k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}
                    if isinstance(self.tracers, dict) else {}
                ),
            )

            # Shard carry across devices for SPMD execution
            if self._device_config is not None and self._device_config.mesh is not None:
                from legoesm.parallel.mesh import shard_pytree
                carry = shard_pytree(carry, self._device_config)

            # Time first segment for JIT measurement
            if seg_idx == 0:
                t_jit_start = time.time()

            # Execute compiled segment (vmap over ensemble if needed)
            if self._ensemble_size > 1:
                carry = jax.vmap(run_segment, in_axes=(0, None, None))(carry, seg_steps, forcing)
            elif (cfg.unfused_radiation
                    and RAD_UPDATE_STEPS > 1
                    and seg_steps % RAD_UPDATE_STEPS == 0
                    and self._ensemble_size == 1
                    and getattr(run_segment, "run_norad_scan", None) is not None
                    and getattr(run_segment, "run_rad", None) is not None):
                # Un-fused radiation (issue: ~3h XLA compile).  Lift the
                # radiation-cycle loop from XLA to the HOST so rrtmgp and the
                # no-rad dynamics+physics scan are TWO SEPARATE executables.
                # Cadence: each outer cycle advances RAD_UPDATE_STEPS steps
                # with the CURRENT held_*, THEN recomputes fresh held_*/T_land
                # from the post-cycle state for the NEXT cycle.  Same absolute
                # radiation update boundary as the fused subcycle (fresh
                # radiation refreshed once per RAD_UPDATE_STEPS), but NOT
                # step-for-step identical: the fused ``_run_subcycled`` applies
                # the fresh radiation IN the last step of each cycle, whereas
                # here every step of the cycle uses held radiation and the
                # refresh lands at the cycle boundary.  This is the intended
                # one-step phase shift (numerically equivalent to ~1e-6; same
                # kernels, different compile boundary).
                n_outer = seg_steps // RAD_UPDATE_STEPS
                for _ic in range(n_outer):
                    # advance RAD_UPDATE_STEPS no-rad steps w/ current held_*
                    carry = run_segment.run_norad_scan(carry, forcing)
                    # recompute fresh held_*/T_land from post-cycle state
                    carry = run_segment.run_rad(carry, forcing)
            else:
                carry = run_segment(carry, seg_steps, forcing)           # legacy fused path (byte-identical)

            # Carry the lagged convective-cloud precip into the next segment
            # (single-member only — see pack_carry above).
            if self._ensemble_size == 1:
                self._conv_precip_prev = carry.conv_precip_prev
                # Persist the evolved multilayer land state across segments (the
                # prognostic soil column — no-op when slab/None).
                if self._land_ml_state is not None:
                    self._land_ml_state = carry.land_ml

            if seg_idx == 0:
                jax.block_until_ready(carry.u)
                t_jit = time.time() - t_jit_start
                logger.info(f"  Segment 0 (incl. JIT) in {t_jit:.1f}s")

            # Unpack carry back to driver state.
            # For ensemble runs, unpack the ensemble-mean for diagnostics;
            # keep full ensemble in carry for the next segment.
            _target_moisture = carry.target_moisture
            _target_mass = carry.target_mass
            if self._ensemble_size > 1:
                from legoesm.parallel.ensemble import ensemble_mean
                mean_carry = ensemble_mean(carry)
                (self.state, self.q_v, self.q_c, self.q_r, conv_prog,
                 held_tuple, _, seg_precip,
                 seg_shflx, seg_lhflx) = unpack_carry(mean_carry, self._state_template)
                _dm_carry = mean_carry
            else:
                (self.state, self.q_v, self.q_c, self.q_r, conv_prog,
                 held_tuple, _, seg_precip,
                 seg_shflx, seg_lhflx) = unpack_carry(carry, self.state)
                _dm_carry = carry
            # Write evolved double-moment hydrometeors back into the registry
            # dict (unpack_carry only returns q_v/q_c/q_r; q_i/q_s/q_g/N_c/N_r/N_i
            # ride the carry directly). No-op for warm-rain (carry fields None).
            if isinstance(self.tracers, dict):
                for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                    _val = getattr(_dm_carry, _nm)
                    if _val is not None:
                        self.tracers[_nm] = _val
            (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
             held_sw_up_toa, held_lw_up_toa, held_sw_down_toa) = held_tuple
            # Clear-sky held TOA up-fluxes (#843) are NOT in the 6-tuple
            # unpack_carry returns — read them straight off the carry (like
            # q_i / tke) so the next segment persists them.
            held_sw_up_toa_clr = _dm_carry.held_sw_up_toa_clr
            held_lw_up_toa_clr = _dm_carry.held_lw_up_toa_clr

            if os.environ.get("LEGOESM_DEBUG_HELD"):
                import numpy as _np
                _h = lambda a: (round(float(_np.mean(_np.asarray(a))), 1),
                                round(float(_np.min(_np.asarray(a))), 1),
                                round(float(_np.max(_np.asarray(a))), 1))
                logger.info(
                    "DEBUG held@day%.1f (mean,min,max): sw_down_toa=%s "
                    "lw_up_toa=%s", day, _h(held_sw_down_toa),
                    _h(held_lw_up_toa))

            # BUG-B state dump: when the held TOA SW goes super-physical, save the
            # full radiation-input state so it can be replayed offline through
            # compute_radiation_core (which produced this garbage) and bisected.
            _dump_path = os.environ.get("LEGOESM_DUMP_RAD")
            if _dump_path and not getattr(self, "_rad_dumped", False):
                import numpy as _np
                if float(_np.max(_np.asarray(held_sw_down_toa))) > 600.0:
                    self._rad_dumped = True
                    _tr = self.tracers if isinstance(self.tracers, dict) else {}
                    _g = lambda x: (None if x is None else _np.asarray(x))
                    _np.savez_compressed(
                        _dump_path,
                        T=_g(self.state.T.data), p_s=_g(self.state.p_s.data),
                        u=_g(self.state.u.data), v=_g(self.state.v.data),
                        q_v=_g(self.q_v), q_c=_g(self.q_c), q_r=_g(self.q_r),
                        q_i=_g(_tr.get("q_i")), N_c=_g(_tr.get("N_c")),
                        N_i=_g(_tr.get("N_i")),
                        sst=_g(sst), sic=_g(sic),
                        lat=_g(_seg_lat), lon=_g(_seg_lon),
                        day_of_year=_np.asarray(forcing.day_of_year),
                        seconds_of_day=_np.asarray(forcing.seconds_of_day),
                        s_0=_np.asarray(forcing.s_0),
                        solar_weights=_g(forcing.solar_weights),
                        o3_vmr=_g(forcing.o3_vmr),
                        aerosol_od=_g(forcing.aerosol_od),
                        T_land=_g(getattr(carry, "T_land", None)),
                        held_sw_down_toa=_g(held_sw_down_toa),
                        held_sw_up_toa=_g(held_sw_up_toa),
                        held_lw_up_toa=_g(held_lw_up_toa),
                        day=_np.asarray(day),
                    )
                    logger.info("BUG-B: dumped corrupt-radiation state to %s "
                                "at day %.1f (sw_down_toa max=%.1f)",
                                _dump_path, day,
                                float(_np.max(_np.asarray(held_sw_down_toa))))

            # Keep carry auxiliary fields for checkpoint persistence and coupling
            self._carry_aux = {
                "held_dT_rad": held_dT_rad,
                "held_sw_net_sfc": held_sw_net_sfc,
                "held_lw_net_sfc": held_lw_net_sfc,
                "held_sw_up_toa": held_sw_up_toa,
                "held_lw_up_toa": held_lw_up_toa,
                "held_sw_down_toa": held_sw_down_toa,
                # Clear-sky held fields (#843) persisted for checkpoint restart.
                "held_sw_up_toa_clr": held_sw_up_toa_clr,
                "held_lw_up_toa_clr": held_lw_up_toa_clr,
                "conv_prog": conv_prog,
                "target_moisture": _target_moisture,
                "target_mass": _target_mass,
                "seg_precip": seg_precip,
                "seg_shflx": seg_shflx,
                "seg_lhflx": seg_lhflx,
            }
            # Carry the slab-land temperature to the next segment and
            # into the checkpoint (mirrors the held-radiation fields).
            if carry.T_land is not None:
                T_land = carry.T_land
                self._carry_aux["T_land"] = T_land
            # Soil-water bucket: carry to the next segment + checkpoint.
            if carry.w_land is not None:
                w_land = carry.w_land
                self._carry_aux["w_land"] = w_land
            if carry.snow is not None:
                snow = carry.snow
                self._carry_aux["snow"] = snow
            # Stateful-physics carries (issue #413): thread the FULL
            # (per-member under ensembles) fields to the next segment
            # and persist them via carry_aux (mirrors T_land).
            if carry.tke is not None:
                phys_tke = carry.tke
                self._carry_aux["tke"] = phys_tke
            if carry.qke is not None:
                phys_qke = carry.qke
                self._carry_aux["qke"] = phys_qke
            if carry.gwd_spectrum is not None:
                phys_gwd_spectrum = carry.gwd_spectrum
                self._carry_aux["gwd_spectrum"] = phys_gwd_spectrum

            current_step = seg_end_step

            # --- Host-side actions at segment boundaries ---
            elapsed_day = day - START_DAY

            # Stability check at EVERY segment boundary, BEFORE diagnostics
            # / segment-callback / adaptive-dt (codex round-2/3 MAJORs:
            # spmd forces diag_days=0 which previously skipped this check
            # entirely, and a coupled segment_callback must never observe
            # an unstable state).  Read-only on state, cheap vs a segment.
            # All ranks must agree on the verdict to avoid divergent loop
            # exits:
            #   - multi-controller SPMD (distributed_mode='spmd'): EVERY
            #     process runs the check itself — the state is globally
            #     sharded and jit-level reductions are SPMD-global, so the
            #     verdict is process-identical by construction; mpi4py here
            #     would be a second control plane beside jax.distributed
            #     (and under a non-MPI launcher COMM_WORLD is size-1 per
            #     process, leaving nonzero ranks with error=None).
            #   - mpi4jax topologies: root checks, mpi4py bcasts.
            _is_root_seg = (self._mpi_rank is None or self._mpi_rank == 0)
            _spmd_mc = (self._mpi_rank is not None
                        and self._device_config is not None
                        and not self._device_config.is_distributed)
            if _is_root_seg or _spmd_mc:
                error = self.diagnostics.check_stability(self.state, elapsed_day)
            else:
                error = None
            if self._mpi_rank is not None and not _spmd_mc:
                from mpi4py import MPI
                error = MPI.COMM_WORLD.bcast(error, root=0)
            if error:
                if _is_root_seg:
                    logger.warning(f"  {error}")
                run_status = error
                # All ranks reach here post-bcast, so the (possibly
                # collective) checkpoint write inside the dump is consistent.
                self._write_blowup_state(current_step, day)
                break

            # Diagnostics
            if diag_interval > 0 and current_step % diag_interval == 0:
                # Convert accumulated quantities to rates over segment duration.
                _seg_dur = seg_steps * DT
                seg_precip_rate = seg_precip / _seg_dur
                seg_shflx_rate = seg_shflx / _seg_dur  # W/m²
                seg_lhflx_rate = seg_lhflx / _seg_dur  # W/m²
                # Segment-MEAN radiative fluxes / T_low (time integrals from
                # the carry / segment duration) instead of the segment-end
                # instantaneous held_* values: the held snapshots put a full
                # day/night diurnal alias into every CMOR Amon "monthly mean"
                # (fixed-UTC sampling) and mixed instantaneous fluxes into
                # the energy budget. _dm_carry = ensemble mean under
                # ensembles, the plain carry otherwise (accums averaged
                # member-wise, consistent with seg_precip).
                seg_sw_up_toa = _dm_carry.sw_up_toa_accum / _seg_dur
                seg_lw_up_toa = _dm_carry.lw_up_toa_accum / _seg_dur
                # Clear-sky TOA up-fluxes for CMOR rsutcs/rlutcs (#843).  None
                # when the diagnostic is off -> the collector skips the field
                # (byte-identical to the pre-#843 output); a real segment-mean
                # (accum / duration) when on.
                seg_sw_up_toa_clr = (
                    _dm_carry.sw_up_toa_clr_accum / _seg_dur
                    if self.config.output.clear_sky_diag else None)
                seg_lw_up_toa_clr = (
                    _dm_carry.lw_up_toa_clr_accum / _seg_dur
                    if self.config.output.clear_sky_diag else None)
                seg_sw_down_toa = _dm_carry.sw_down_toa_accum / _seg_dur
                seg_sw_net_sfc = _dm_carry.sw_net_sfc_accum / _seg_dur
                seg_lw_net_sfc = _dm_carry.lw_net_sfc_accum / _seg_dur
                seg_t_low_mean = _dm_carry.t_low_accum / _seg_dur

                diag_info = self._sync_and_collect_diagnostics(
                    elapsed_day=elapsed_day,
                    day=day,
                    state=self.state,
                    q_v=self.q_v,
                    q_c=self.q_c,
                    q_r=self.q_r,
                    q_i=self.q_i,
                    sst=sst,
                    sic=sic,
                    precip_total=seg_precip_rate,
                    sw_up_toa=seg_sw_up_toa,
                    lw_up_toa=seg_lw_up_toa,
                    sw_up_toa_clr=seg_sw_up_toa_clr,
                    lw_up_toa_clr=seg_lw_up_toa_clr,
                    sw_net_sfc=seg_sw_net_sfc,
                    lw_net_sfc=seg_lw_net_sfc,
                    sw_down_toa=seg_sw_down_toa,
                    T_ice=cfg.T_ice,
                    lat_deg_grid=lat_deg_grid,
                    shflx=seg_shflx_rate,
                    lhflx=seg_lhflx_rate,
                    t_low_mean=seg_t_low_mean,
                    q_s=self.tracers.get("q_s") if isinstance(self.tracers, dict) else None,
                    q_g=self.tracers.get("q_g") if isinstance(self.tracers, dict) else None,
                )

                # CFL computed host-side from final segment state (not in hot loop)
                from legoesm.core.cfl import cfl_number_from_state, estimate_min_dx_cubed_sphere
                _dx_min = estimate_min_dx_cubed_sphere(cfg.grid.resolution) if hasattr(self.grid, 'n') else 1e6

                # Under MPI, CFL on owned faces only, then global max
                if self._owned_face_ids is not None:
                    _ofi = self._owned_face_ids
                    _seg_max_cfl = float(cfl_number_from_state(
                        self.state.u.data[_ofi], self.state.v.data[_ofi], _dx_min, DT,
                    ))
                    from mpi4py import MPI
                    _seg_max_cfl = MPI.COMM_WORLD.allreduce(_seg_max_cfl, op=MPI.MAX)
                else:
                    _seg_max_cfl = float(cfl_number_from_state(
                        self.state.u.data, self.state.v.data, _dx_min, DT,
                    ))

                # Logging: rank 0 only under MPI
                _is_root = (self._mpi_rank is None or self._mpi_rank == 0)
                if _is_root:
                    elapsed_wall = time.time() - t_start
                    days_done = elapsed_day
                    eta_str = ""
                    if days_done > 0:
                        rate = elapsed_wall / days_done
                        remaining = (N_DAYS - days_done) * rate
                        eta_str = f", ETA {remaining/3600:.1f}h"
                    logger.info(
                        f"  Day {elapsed_day:6.0f}: T={diag_info.get('mean_T', 0):.1f}K, "
                        f"max_v={diag_info.get('max_v', 0):.1f}m/s"
                        f"{eta_str}"
                    )
                    if _seg_max_cfl > 0:
                        logger.info(f"    CFL max: {_seg_max_cfl:.2f}")

                # Segment callback for coupled integration (e.g., coupler step)
                if self._segment_callback is not None:
                    dt_seg = float(seg_steps * DT)
                    self._segment_callback(self, day, dt_seg)

                # Adaptive dt: if CFL exceeds threshold, halve dt and rebuild
                if _seg_max_cfl > 1.0:
                    DT = DT / 2.0
                    logger.warning(
                        f"  CFL={_seg_max_cfl:.2f} > 1.0 at day {elapsed_day:.0f}. "
                        f"Halving dt to {DT:.0f}s."
                    )
                    n_steps_total = int(cfg.days * 86400 / DT)
                    diag_interval = int(cfg.output.diag_days * 86400 / DT)
                    checkpoint_interval = (
                        int(cfg.output.checkpoint_days * 86400 / DT)
                        if cfg.output.checkpoint_days > 0 else 0
                    )
                    segment_length = compute_segment_length(
                        diag_interval, checkpoint_interval, RAD_UPDATE_STEPS,
                        fallback_interval=int(
                            cfg.forcing_update_days
                            * 86400 / DT),
                    )
                    run_segment = build_segment_fn(
                        model=self.model, step_unified=step_unified,
                        tiled_step_fn=self._maybe_build_tiled_step(DT),
                        step_unified_no_rad=step_unified_no_rad,
                        grid=self.grid, sigma_full=sigma_full, dsigma=dsigma,
                        dt=DT, rad_update_steps=RAD_UPDATE_STEPS,
                        microphysics=cfg.microphysics, fix_moisture=cfg.fix_moisture,
                        fix_mass=cfg.dycore.fix_mass,
                        fric_decay=self._fric_decay, qv_smooth_coeff=self._qv_smooth_coeff,
                        lat=_seg_lat, lon=_seg_lon, start_day=START_DAY,
                        gradient_checkpoint=cfg.gradient_checkpoint or segment_length > 50,
                        hyperdiffusion_3d_fn=self._hyperdiffusion_3d_fn,
                        tau_equator=cfg.tau_equator, tau_pole=cfg.tau_pole,
                        sbm_tau_c=cfg.sbm_tau_c, sbm_RH_ref=cfg.sbm_RH_ref,
                        C_H=cfg.C_H, C_E=cfg.C_E,
                        albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
                        ghg_vmr_override=ghg_vmr,
                        owned_face_ids=self._owned_face_ids,
                        hs_newtonian_relax=self._hs_newtonian_relax,
                        device_config=_seg_device_config,
                        energy_consistent_moisture_clip=cfg.energy_consistent_moisture_clip,
                        advect_moisture=self._moisture_advection_active(),
                        pipeline=self.physics,
                    )

            # Checkpoint (a coupled run routes this through its own
            # save_checkpoint so the coupled state is written too).
            _ckpt = getattr(self, "_checkpoint_callback", None) or self.save_checkpoint
            _checkpoint_written = (checkpoint_interval > 0
                                   and current_step % checkpoint_interval == 0)
            if _checkpoint_written:
                _ckpt(current_step, day)

            # Wallclock-aware clean exit for long HPC dependency chains.
            self._maybe_wallclock_exit(_ckpt, current_step, day)

            # Periodic diagnostic flush (every ~365 days) to cap memory — rank 0 only
            if (elapsed_day > 0 and int(elapsed_day) % 365 == 0
                    and diag_interval > 0 and current_step % diag_interval == 0
                    and (self._mpi_rank is None or self._mpi_rank == 0)):
                self.diagnostics.flush_to_disk(self._output_dir)

            # Incremental CMIP monthly flush — write completed months and
            # free their memory so long runs don't accumulate all months.
            # EVERY process pops (under multi-controller SPMD the non-root
            # accumulators fill identically and would otherwise grow
            # unbounded, codex round-10); only root writes.
            if diag_interval > 0 and current_step % diag_interval == 0:
                _is_root = self._mpi_rank is None or self._mpi_rank == 0
                self.diagnostics.flush_cmip_monthly(day, write=_is_root)
                # If a checkpoint was written THIS step (above), its CMOR
                # sidecar captured the accumulators PRE-flush; the flush just
                # drained + wrote the completed months to NetCDF.  Re-persist
                # the (now drained) sidecar so a restart from that checkpoint
                # resumes ONLY the in-progress month — else the flushed months
                # would be re-appended (duplicate time coords) on resume.  Root
                # only (mirrors the sidecar write inside save_checkpoint) and
                # bounded to the checkpoint cadence — never on a bare flush.
                if _checkpoint_written and _is_root:
                    self._save_cmor_accumulator_sidecar(day)

        return self._finalize_run(
            run_status, t_jit, t_start,
            n_steps_total, START_DAY, N_DAYS, checkpoint_interval,
        )

    # ==================================================================
    # Legacy per-step execution path
    # ==================================================================

    def _run_per_step(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run using per-step Python orchestration (legacy path).

        Uses a JIT-compiled unified physics step with ``jax.lax.cond``
        for radiation sub-cycling (held tendencies reused between
        radiation update steps).

        This path is retained for debugging and as a reference
        implementation.  For production use, prefer ``run(compiled=True)``.
        """
        hyperdiffusion_3d = self._hyperdiffusion_3d_fn
        from legoesm.forcing.external import get_solar_forcing_at_time

        # restore_carry=True matches the compiled path: when continuing
        # from a checkpoint it restores held radiation, conv_prog, and the
        # prognostic slab-land T_land from self._carry_aux; on a fresh
        # start self._carry_aux is empty so this is a no-op (#325
        # restart-safety for the non-compiled reference path).
        ctx = self._prepare_run_context(start_step, start_day, restore_carry=True)
        cfg = ctx["cfg"]
        # Issue #405/#413: this loop now seeds and threads the
        # stateful-physics carries (tke / qke / gwd_spectrum, alongside
        # conv_prog) through the unified physics step and persists them
        # via carry_aux, so the former stateful-physics refusal is
        # lifted here.  Stochastic Bechtold remains refused at pipeline
        # build time (its PRNG-key carry is not threaded).
        DT = ctx["DT"]
        N_DAYS = ctx["N_DAYS"]
        START_DAY = ctx["START_DAY"]
        MICROPHYSICS = cfg.microphysics
        RAD_UPDATE_STEPS = ctx["RAD_UPDATE_STEPS"]
        n_steps_total = ctx["n_steps_total"]
        diag_interval = ctx["diag_interval"]
        checkpoint_interval = ctx["checkpoint_interval"]
        sigma_full = ctx["sigma_full"]
        dsigma = ctx["dsigma"]
        shape_2d = ctx["shape_2d"]
        current_s_0 = ctx["current_s_0"]
        solar_weights = ctx["solar_weights"]
        step_unified = ctx["step_unified"]
        # Issue #316: step_unified is the static_need_rad=True variant.
        # The held-only variant is built lazily here when RAD_UPDATE_STEPS
        # > 1 (otherwise need_rad_py is always True and the second
        # variant is unused).  Python-side dispatch on need_rad_py keeps
        # each call's HLO graph minimal and matches the cond-elided
        # design used by _run_compiled / build_segment_fn.
        step_unified_no_rad = ctx.get("step_unified_no_rad")
        if RAD_UPDATE_STEPS > 1 and step_unified_no_rad is None:
            step_unified_no_rad = self.physics.build_step_unified(
                static_need_rad=False,
            )
        held_dT_rad = ctx["held_dT_rad"]
        held_sw_net_sfc = ctx["held_sw_net_sfc"]
        held_lw_net_sfc = ctx["held_lw_net_sfc"]
        held_sw_up_toa = ctx["held_sw_up_toa"]
        held_lw_up_toa = ctx["held_lw_up_toa"]
        held_sw_down_toa = ctx["held_sw_down_toa"]
        conv_prog = ctx["conv_prog"]
        # Slab-land skin temperature (#325): threaded through the
        # non-compiled per-step path so the reference run evolves land
        # T_sfc consistently with the compiled-segment path.  ``None``
        # when the land tile is inactive.
        T_land = ctx["T_land"]
        # Soil-water bucket (None unless active): threaded like T_land.
        w_land = ctx["w_land"]
        snow = ctx.get("snow")
        # Stateful-physics carries (issue #413), mirroring the compiled
        # path: None for diagnostic schemes (zero overhead).
        phys_tke = ctx["tke"]
        phys_qke = ctx["qke"]
        phys_gwd_spectrum = ctx["gwd_spectrum"]

        def _phys_carry_step_inputs():
            """Keyword inputs for the active stateful-physics carries."""
            kw = {}
            if phys_tke is not None:
                kw["tke"] = phys_tke
            if phys_qke is not None:
                kw["qke"] = phys_qke
            if phys_gwd_spectrum is not None:
                kw["gwd_spectrum"] = phys_gwd_spectrum
            return kw
        o3_vmr = ctx["o3_vmr"]
        aerosol_od = ctx["aerosol_od"]
        ghg_vmr = ctx["ghg_vmr"]
        lat_deg_grid = ctx["lat_deg_grid"]

        # Moisture conservation fixer
        FIX_MOISTURE = cfg.fix_moisture
        if FIX_MOISTURE:
            target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid,
            )

        run_status = "COMPLETED"

        logger.info(f"Starting: {n_steps_total - start_step} steps, {N_DAYS} days")

        # --- JIT warmup ---
        t_jit_start = time.time()
        day = START_DAY + (start_step + 1) * DT / 86400.0
        day_of_year, seconds_of_day = self._calendar_for_radiation(day)
        sst, sic = self.get_sst_sic(day)

        # Warmup radiation cadence (FIX_RESTART_TIME iteration-2 codex
        # finding, high): the warmup executes step ``start_step``, so a
        # RESUMED run must use the SAME need_rad predicate as the main
        # loop — the straight run's step ``start_step`` was held-only
        # unless (start_step+1) hit the radiation cadence, and an
        # unconditional radiation solve here both overwrites the
        # checkpoint-restored held tendencies and samples forcing at the
        # wrong time.  Fresh starts (start_step == 0) and resumes
        # without restored held tendencies keep the historical
        # always-radiate warmup (zero-initialized held fields would be
        # worse than a recompute).
        warmup_need_rad = (
            start_step == 0
            or RAD_UPDATE_STEPS <= 1
            or (start_step + 1) % RAD_UPDATE_STEPS == 0
            or "held_dT_rad" not in self._carry_aux
        )
        if warmup_need_rad and start_step > 0:
            # Mirror the straight run's refresh at the top of this
            # step's body: solar + external forcing sampled at
            # day(start_step+1), not the prepare-context epoch values.
            solar_now = get_solar_forcing_at_time(self._solar_config, day)
            current_s_0 = float(solar_now["tsi"])
            if self._use_solar_spectral:
                solar_weights = jnp.asarray(
                    solar_now["solar_fraction_by_gpt"])
            _phys_p_s, _phys_lat = self._owned_p_s_and_lat()
            o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
                day, _phys_p_s, _phys_lat,
            )
        _warmup_step_fn = (
            step_unified
            if warmup_need_rad or step_unified_no_rad is None
            else step_unified_no_rad
        )

        self.state = self.model.step_with_physics(self.state, DT)

        # Double-moment hydrometeor inputs (None unless the registry carries
        # them) so coupled radiation/microphysics see ice + droplet number.
        _dm_step_in = self._double_moment_step_inputs()
        # step_unified returns a 4-tuple (issue: PR #650 added the 4th
        # ``land_ml`` multilayer-land state). Thread it exactly as the
        # compiled-segment path does (single-member only; ensemble carries
        # an extra axis the tile doesn't expect) so the per-step driver
        # advances the prognostic soil column instead of crashing on the
        # arity mismatch.
        phys_out, (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa), \
            T_land, land_ml = \
            _warmup_step_fn(
                jnp.bool_(warmup_need_rad),
                self.state.T.data, self.state.p_s.data,
                self.q_v, self.q_c, self.q_r, conv_prog,
                self.state.u.data, self.state.v.data,
                sst, sic, self._grid_lat, self._grid_lon,
                day_of_year, seconds_of_day, DT,
                solar_weights, current_s_0,
                o3_vmr, aerosol_od,
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                ghg_vmr_override=ghg_vmr,
                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
                land_ml=(self._land_ml_state
                         if self._ensemble_size == 1 else None),
                T_land=T_land, w_land=w_land, snow=snow, **_dm_step_in,
                **_phys_carry_step_inputs(),
            )
        conv_prog = phys_out.conv_prog
        if phys_out.w_land is not None:
            w_land = phys_out.w_land
        if phys_out.snow is not None:
            snow = phys_out.snow
        # Stash the FULL restart-relevant carry set at the warmup step
        # (codex rounds 4/6/8): a one-step run never enters the main
        # loop, and _finalize_run would otherwise checkpoint stale or
        # missing carries (held radiation and T_land included — they
        # were historically only written at diagnostic boundaries).
        self._carry_aux.update({
            "held_dT_rad": held_dT_rad,
            "held_sw_net_sfc": held_sw_net_sfc,
            "held_lw_net_sfc": held_lw_net_sfc,
            "held_sw_up_toa": held_sw_up_toa,
            "held_lw_up_toa": held_lw_up_toa,
            "held_sw_down_toa": held_sw_down_toa,
            "conv_prog": conv_prog,
        })
        if T_land is not None:
            self._carry_aux["T_land"] = T_land
        if w_land is not None:
            self._carry_aux["w_land"] = w_land
        if snow is not None:
            self._carry_aux["snow"] = snow
        # Persist the evolved multilayer land state across the run (matches
        # the compiled-segment writeback; no-op for slab/None or ensemble).
        if self._ensemble_size == 1 and self._land_ml_state is not None:
            self._land_ml_state = land_ml
        if phys_out.tke is not None:
            phys_tke = phys_out.tke
            self._carry_aux["tke"] = phys_tke
        if phys_out.qke is not None:
            phys_qke = phys_out.qke
            self._carry_aux["qke"] = phys_qke
        if phys_out.gwd_spectrum is not None:
            phys_gwd_spectrum = phys_out.gwd_spectrum
            self._carry_aux["gwd_spectrum"] = phys_gwd_spectrum

        # Apply warmup tendencies
        new_T = self.state.T.data + DT * phys_out.dT_dt
        if self._hs_newtonian_relax is not None:
            new_T = new_T + DT * self._hs_newtonian_relax(
                self.state.T.data, self.state.p_s.data, self._grid_lat)
        # Issue #323: keep the q_v floor moist-static-energy neutral when the
        # opt-in flag is set (mirror the compiled-segment path so the flag is
        # not a silent no-op in the per-step driver).
        _qv_raw = self.q_v + DT * phys_out.dq_v_dt
        if self.config.energy_consistent_moisture_clip:
            self.q_v, new_T = energy_consistent_moisture_floor(_qv_raw, new_T)
        else:
            self.q_v = jnp.maximum(_qv_raw, 0.0)
        self.q_c = jnp.maximum(self.q_c + DT * phys_out.dq_c_dt, 0.0)
        self.q_r = jnp.maximum(self.q_r + DT * phys_out.dq_r_dt, 0.0)
        self._apply_double_moment_tendencies(phys_out, DT)

        if MICROPHYSICS == "none":
            p_full = self.state.p_s.data[..., None] * sigma_full
            q_sat = saturation_mixing_ratio(new_T, p_full)
            excess = jnp.maximum(self.q_v - q_sat, 0.0)
            self.q_v = self.q_v - excess
            new_T = new_T + constants.L_v * excess / constants.c_pd

        self.state = self.state._replace(T=self.state.T.replace(data=new_T))
        if hasattr(phys_out, 'du_dt') and phys_out.du_dt is not None:
            self.state = self.state._replace(
                u=self.state.u.replace(data=self.state.u.data + DT * phys_out.du_dt),
                v=self.state.v.replace(data=self.state.v.data + DT * phys_out.dv_dt),
            )
        self.q_v = jnp.maximum(
            self.q_v + DT * hyperdiffusion_3d(self.q_v, self.grid, self._qv_smooth_coeff), 0.0
        )
        self.state = self.state._replace(
            u=self.state.u.replace(data=self.state.u.data * self._fric_decay),
            v=self.state.v.replace(data=self.state.v.data * self._fric_decay),
        )

        jax.block_until_ready(self.state.u.data)
        t_jit = time.time() - t_jit_start
        logger.info(f"  JIT compiled in {t_jit:.1f}s")

        # --- Main time loop ---
        t_start = time.time()

        for step in range(start_step + 1, n_steps_total):
            day = START_DAY + (step + 1) * DT / 86400.0
            day_of_year, seconds_of_day = self._calendar_for_radiation(day)

            sst, sic = self.get_sst_sic(day)

            # (a) Dynamics
            self.state = self.model.step_with_physics(self.state, DT)

            # (b) Physics with radiation sub-cycling
            need_rad_py = (RAD_UPDATE_STEPS <= 1) or ((step + 1) % RAD_UPDATE_STEPS == 0)
            need_rad_jax = jnp.bool_(need_rad_py)

            # Update external forcing on radiation steps
            if need_rad_py:
                # Solar
                solar_now = get_solar_forcing_at_time(self._solar_config, day)
                current_s_0 = float(solar_now["tsi"])
                if self._use_solar_spectral:
                    solar_weights = jnp.asarray(solar_now["solar_fraction_by_gpt"])

                # Ozone + aerosol + GHG (rank-local for MPI)
                _phys_p_s, _phys_lat = self._owned_p_s_and_lat()
                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
                    day, _phys_p_s, _phys_lat,
                )

                # CMIP GHG trajectory — already handled inside
                # _precompute_external_forcing for transient experiments.

            # Issue #316: Python-side dispatch picks rad-only or
            # no-rad-only variant — both are cond-free; XLA only sees
            # the active branch's HLO.
            _step_fn = (
                step_unified if need_rad_py or step_unified_no_rad is None
                else step_unified_no_rad
            )
            _dm_step_in = self._double_moment_step_inputs()
            # 4-tuple return (PR #650 land_ml); thread the multilayer-land
            # state like the compiled-segment path (single-member only).
            phys_out, (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa), \
                T_land, land_ml = \
                _step_fn(
                    need_rad_jax,
                    self.state.T.data, self.state.p_s.data,
                    self.q_v, self.q_c, self.q_r, conv_prog,
                    self.state.u.data, self.state.v.data,
                    sst, sic, self._grid_lat, self._grid_lon,
                    day_of_year, seconds_of_day, DT,
                    solar_weights, current_s_0,
                    o3_vmr, aerosol_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    ghg_vmr_override=ghg_vmr,
                    aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
                    land_ml=(self._land_ml_state
                             if self._ensemble_size == 1 else None),
                    T_land=T_land, w_land=w_land, snow=snow, **_dm_step_in,
                    **_phys_carry_step_inputs(),
                )
            conv_prog = phys_out.conv_prog
            if phys_out.w_land is not None:
                w_land = phys_out.w_land
            if phys_out.snow is not None:
                snow = phys_out.snow
            # Persist the full restart-relevant set per step (codex
            # rounds 6/8): checkpoints can fire on any step, so the
            # held-radiation fields and every carry must be current —
            # not just whatever the last diagnostic boundary wrote.
            self._carry_aux.update({
                "held_dT_rad": held_dT_rad,
                "held_sw_net_sfc": held_sw_net_sfc,
                "held_lw_net_sfc": held_lw_net_sfc,
                "held_sw_up_toa": held_sw_up_toa,
                "held_lw_up_toa": held_lw_up_toa,
                "held_sw_down_toa": held_sw_down_toa,
                "conv_prog": conv_prog,
            })
            if T_land is not None:
                self._carry_aux["T_land"] = T_land
            if w_land is not None:
                self._carry_aux["w_land"] = w_land
            if snow is not None:
                self._carry_aux["snow"] = snow
            # Advance the prognostic multilayer land state per step (matches
            # the compiled-segment writeback; no-op for slab/None or ensemble).
            if self._ensemble_size == 1 and self._land_ml_state is not None:
                self._land_ml_state = land_ml
            # Stateful-physics carries (issue #413): feed the updated
            # values back next step + persist for checkpoints.
            if phys_out.tke is not None:
                phys_tke = phys_out.tke
                self._carry_aux["tke"] = phys_tke
            if phys_out.qke is not None:
                phys_qke = phys_out.qke
                self._carry_aux["qke"] = phys_qke
            if phys_out.gwd_spectrum is not None:
                phys_gwd_spectrum = phys_out.gwd_spectrum
                self._carry_aux["gwd_spectrum"] = phys_gwd_spectrum

            # (c) Update state
            new_T = self.state.T.data + DT * phys_out.dT_dt

            # Held-Suarez Newtonian temperature relaxation
            if self._hs_newtonian_relax is not None:
                new_T = new_T + DT * self._hs_newtonian_relax(
                    self.state.T.data, self.state.p_s.data, self._grid_lat)

            # Issue #323: energy-consistent q_v floor (see warmup path).
            _qv_raw = self.q_v + DT * phys_out.dq_v_dt
            if self.config.energy_consistent_moisture_clip:
                self.q_v, new_T = energy_consistent_moisture_floor(_qv_raw, new_T)
            else:
                self.q_v = jnp.maximum(_qv_raw, 0.0)
            self.q_c = jnp.maximum(self.q_c + DT * phys_out.dq_c_dt, 0.0)
            self.q_r = jnp.maximum(self.q_r + DT * phys_out.dq_r_dt, 0.0)

            # Apply ice/number tracer tendencies when full registry is active
            self._apply_double_moment_tendencies(phys_out, DT)

            # Saturation adjustment
            if MICROPHYSICS == "none":
                p_full = self.state.p_s.data[..., None] * sigma_full
                q_sat = saturation_mixing_ratio(new_T, p_full)
                excess = jnp.maximum(self.q_v - q_sat, 0.0)
                self.q_v = self.q_v - excess
                new_T = new_T + constants.L_v * excess / constants.c_pd
                precip_ls = jnp.sum(
                    excess * self.state.p_s.data[..., None] * dsigma, axis=-1
                ) / (constants.g * DT)
            else:
                precip_ls = jnp.zeros(shape_2d, dtype=new_T.dtype)

            self.state = self.state._replace(
                T=self.state.T.replace(data=new_T)
            )

            # Apply momentum tendencies from turbulence/GWD
            if hasattr(phys_out, 'du_dt') and phys_out.du_dt is not None:
                new_u = self.state.u.data + DT * phys_out.du_dt
                new_v = self.state.v.data + DT * phys_out.dv_dt
                self.state = self.state._replace(
                    u=self.state.u.replace(data=new_u),
                    v=self.state.v.replace(data=new_v),
                )

            # Moisture conservation fixer
            if FIX_MOISTURE:
                self.q_v = fix_moisture_hydrostatic(
                    self.q_v, target_moisture,
                    self.state.p_s.data, dsigma, self.grid,
                )

            # Moisture smoothing
            self.q_v = jnp.maximum(
                self.q_v + DT * hyperdiffusion_3d(self.q_v, self.grid, self._qv_smooth_coeff),
                0.0,
            )

            # Rayleigh friction
            self.state = self.state._replace(
                u=self.state.u.replace(data=self.state.u.data * self._fric_decay),
                v=self.state.v.replace(data=self.state.v.data * self._fric_decay),
            )

            # Diagnostics (diag_interval == 0 = writer disabled — guard the
            # modulo; spmd configs FORCE diag_days=0, codex round-2 MAJOR)
            elapsed_day = day - START_DAY
            if diag_interval > 0 and (step + 1) % diag_interval == 0:
                diag_info = self._sync_and_collect_diagnostics(
                    elapsed_day=elapsed_day,
                    day=day,
                    state=self.state,
                    q_v=self.q_v,
                    q_c=self.q_c,
                    q_r=self.q_r,
                    q_i=self.q_i,
                    sst=sst,
                    sic=sic,
                    precip_total=phys_out.precip + precip_ls,
                    sw_up_toa=phys_out.sw_up_toa,
                    lw_up_toa=phys_out.lw_up_toa,
                    sw_net_sfc=phys_out.sw_net_sfc,
                    lw_net_sfc=phys_out.lw_net_sfc,
                    sw_down_toa=phys_out.sw_down_toa,
                    T_ice=cfg.T_ice,
                    lat_deg_grid=lat_deg_grid,
                    q_s=self.tracers.get("q_s") if isinstance(self.tracers, dict) else None,
                    q_g=self.tracers.get("q_g") if isinstance(self.tracers, dict) else None,
                )

                logger.info(f"  Day {elapsed_day:6.0f}: T={diag_info['mean_T']:.1f}K, "
                      f"precip={diag_info['mean_precip']:.1f}mm/d, "
                      f"max_v={diag_info['max_v']:.1f}m/s")

                # Stability check
                error = self.diagnostics.check_stability(self.state, elapsed_day)
                if error:
                    logger.warning(f"  {error}")
                    run_status = error
                    self._write_blowup_state(step + 1, day)
                    break

                # Refresh coupling-facing carry_aux entries.  UPDATE —
                # never rebuild the dict (codex rounds 3/8): a rebuild
                # dropped held_dT_rad / held_*_toa / T_land / the
                # stateful-physics carries right before any checkpoint
                # written on a diagnostic step, silently reseeding them
                # on restart.  The per-step stash above keeps the full
                # restart set current; this adds the diagnostics-only
                # extras.
                self._carry_aux.update({
                    "held_sw_net_sfc": phys_out.sw_net_sfc,
                    "held_lw_net_sfc": phys_out.lw_net_sfc,
                    "conv_prog": conv_prog,
                    "seg_precip": phys_out.precip,
                })

                # Segment callback for coupled integration
                if self._segment_callback is not None:
                    self._segment_callback(self, day, DT)

            # Checkpoint
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                self.save_checkpoint(step + 1, day)

        return self._finalize_run(
            run_status, t_jit, t_start,
            n_steps_total, START_DAY, N_DAYS, checkpoint_interval,
        )

    def save_results(self, run_status: str, jit_time: float, wall_time: float) -> None:
        """Write results.txt summary file."""
        cfg = self.config
        d = self.diagnostics
        with open(self._output_dir / "results.txt", "w") as f:
            f.write(f"legoESM AMIP run\n")
            f.write(f"Grid: {cfg.grid.grid_type} {cfg.grid.resolution} / "
                    f"L{cfg.grid.nlev}, dt={cfg.dycore.dt}s, {cfg.days} days\n")
            f.write(f"Radiation: {cfg.radiation}\n")
            f.write(f"Status: {run_status}\n\n")
            f.write(f"JIT compilation: {jit_time:.1f}s\n")
            f.write(f"Wall time: {wall_time:.1f}s\n\n")
            if d.times:
                f.write(f"Final <T_atm>: {d.T_atm[-1]:.3f} K\n")
                f.write(f"Final <Precip>: {d.precip[-1]:.2f} mm/day\n")
                f.write(f"Final <CWV>: {d.CWV[-1]:.1f} kg/m2\n")
            f.write(f"\n{d.energy_tracker.summary()}\n")
        logger.info(f"  Results saved to {self._output_dir}")
