"""Regression ratchet: an existing scheme/factory unknown-value guard must not be
silently deleted.

CLAUDE.md: *"Every ``scheme="..."`` factory MUST ``raise ValueError`` on unknown.
Silent ``else: <default>`` masks typos + dead branches."* The historical bugs are
concrete and invisible — a ``cloud_fraction`` typo silently ran sundqvist; a
``carbon_cycle`` typo silently zeroed CO2; MPAS PV typos silently fell back to
enstrophy. A misspelled config runs a *different physics scheme* and nobody
notices.

**Scope (deliberately narrow).** This is a *regression* guard: it prevents an
already-hardened dispatcher from losing its guard. It does **not** prove a brand
new dispatcher is hardened — that gap is covered by ``test_validate_strict_coverage``
(config-level membership) and the ``check_banned_literals``/edit-time hook plus
code review. A tree-wide "silent ``else``" AST scan was rejected as too noisy
(most ``else:`` are boolean feature-gates), so this is registry/discovery-targeted.

Mechanism:
  * ``discover_hardened_dispatchers()`` finds every production function that
    ``raise``\\s ``ValueError``/``KeyError`` with a message naming an *unknown /
    unsupported / unrecognized / not-registered / invalid* **dispatch** quantity
    (scheme, model_type, grid_type, integrator, eos, …). It scans only each
    function's **own scope** — a ``raise`` inside a *nested* function/lambda/class
    does not count toward the enclosing function (else a silent dispatcher with a
    raising helper would look hardened).
  * ``BASELINE_DISPATCHERS`` pins the current set; the ratchet is **grow-only**
    (baseline ⊆ discovered). A vanished entry → red, naming the lost guard.
  * **Behavioural teeth** — the cheap pure-string factories are actually called
    with a bogus value and must raise, proving guards *work*, not merely exist.

Known limitation: a guard whose message is built in a helper
(``raise ValueError(_msg(name))``) carries no inline string and would drop from
discovery — so keep unknown-value messages inline (good practice anyway). A
tripwire, not a proof. Self-tests prove non-vacuity.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests import _ratchet_audit as ra

_MSG_RE = re.compile(
    r"(unknown|unsupported|unrecognized|expected one of|must be one of|not in"
    r"|not registered|no registered|invalid|available)",
    re.I,
)
_DISP_RE = re.compile(
    r"(scheme|model_type|discretization|integrator|grid_type|eos|advection|limiter"
    r"|drag|micro|cloud|radiation|convection|turbulence|method|formula|variant|pgf"
    r"|entity|partition|parameterization|dycore|kernel|registry|backend|solver|preset"
    r"|normali[sz]ation)",
    re.I,
)
# NB: deliberately NOT including very common words (mode/source/profile/transform)
# — they pull ~25 non-dispatch ``ValueError``s into the set. This ratchet protects
# the *discovered* inline unknown-dispatch guards; it is not an exhaustive census
# of every guarded dispatcher (a guard with an idiosyncratic message may be
# uncovered). New/uncovered factories are the job of ``test_validate_strict_coverage``
# (config membership) + the ``check_banned_literals`` edit hook + code review.


def _own_scope_nodes(fn: ast.AST) -> list[ast.AST]:
    """Nodes lexically inside ``fn`` but NOT inside a nested function/lambda/class
    scope — so a ``raise`` in a closure/helper doesn't credit the parent."""
    out: list[ast.AST] = []

    def add(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            out.append(child)
            add(child)

    add(fn)
    return out


def _raise_is_unknown_dispatch(node: ast.Raise) -> bool:
    if node.exc is None:
        return False
    exc = node.exc
    call = exc if isinstance(exc, ast.Call) else None
    name = (
        call.func.id
        if call is not None and isinstance(call.func, ast.Name)
        else (exc.id if isinstance(exc, ast.Name) else None)
    )
    if name not in ("ValueError", "KeyError"):
        return False
    text = " ".join(
        d.value for d in ast.walk(node) if isinstance(d, ast.Constant) and isinstance(d.value, str)
    )
    return bool(_MSG_RE.search(text) and _DISP_RE.search(text))


def function_is_hardened_dispatcher(fn: ast.AST) -> bool:
    """True if ``fn``'s own scope contains an unknown-dispatch-value ``raise``."""
    return any(
        isinstance(n, ast.Raise) and _raise_is_unknown_dispatch(n)
        for n in _own_scope_nodes(fn)
    )


def _production_files() -> list:
    return [
        f
        for f in ra.discover_py_files()
        if not ra.rel(f).startswith(("tests/", "scripts/"))
    ]


def discover_hardened_dispatchers() -> tuple[set[tuple[str, str]], list[str]]:
    """((relpath, function_name) set, list of unparseable files)."""
    found: set[tuple[str, str]] = set()
    parse_failures: list[str] = []
    for f in _production_files():
        rel = ra.rel(f)
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            parse_failures.append(rel)
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and function_is_hardened_dispatcher(n):
                found.add((rel, n.name))
    return found, parse_failures


# Pinned baseline (iter 2026-06-09, nested-scope-aware detector, +solver/preset
# nouns). GROW-ONLY: removing a guard => the pair disappears => red. Update (with
# justification) only when a dispatcher is intentionally renamed/removed.
BASELINE_DISPATCHERS: frozenset[tuple[str, str]] = frozenset(
    {
        # Distributed PCG preconditioner selection ("jacobi" | "poly"):
        # an unknown name must raise, never fall back to Jacobi.
        ("packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py", "barotropic_implicit_mpas"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py", "create_model"),
        # kt whitelist: an unvalidated 6*kt^2 tile count must raise, not
        # silently replicate the global state per device (#1360).
        ("packages/core/legoesm/parallel/tiled_production_cdgrid.py", "_validate_tiled_step_factory_args"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py", "get_solver_class"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py", "resolve_solver_name"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py", "plane_compressible_euler_slow_tendencies"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py", "validate_plane_config"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane_halo.py", "plane_compressible_euler_slow_tendencies_halo"),
        # MPAS atm dycore pv_scheme guards (hardened 2026-06-22; previously a
        # bare ``else`` silently fell back to the energy-conserving PV flux).
        ("packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_mpas.py", "mpas_compressible_euler_slow_tendencies"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py", "mpas_hydrostatic_tendencies"),
        ("packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_mpas.py", "mpas_shallow_water_tendencies"),
        ("packages/atmosphere/legoesm/atmosphere/physics/clouds/cloud_fraction.py", "compute_cloud_properties"),
        ("packages/atmosphere/legoesm/atmosphere/physics/combined.py", "make_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/dca.py", "dca_convection"),
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py", "_get_convection_fn"),
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py", "make_convection_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/kuo.py", "kuo_convection"),
        # Convective precip-split selector (precip_split_scheme: constant vs the
        # physical Sundqvist-1978 autoconversion on the plume q_c_u); a typo must
        # raise, not silently run the wrong precip physics.
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/bechtold.py", "bechtold_convection"),
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/tiedtke.py", "tiedtke_convection"),
        # ZM land-fraction policy ("required" | "none"): a typo must raise, not
        # silently run every column with ocean coefficients.
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py", "zhang_mcfarlane_convection"),
        # Renamed _get_gwd_fn -> get_gwd_fn (private-import promotion,
        # 2026-06-10); the unknown-scheme raise itself is unchanged.
        ("packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py", "get_gwd_fn"),
        ("packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py", "make_gwd_physics"),
        # Aerosol -> cloud-droplet activation scheme selector (proxy vs ARG2000);
        # a typo'd scheme must raise, not silently run different activation.
        ("packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py", "activated_nc_field"),
        ("packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py", "_get_microphysics_fn"),
        ("packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py", "make_microphysics_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py", "morrison_microphysics"),
        # thompson snow_scheme guard (hardened 2026-06-22; previously a bare
        # ``else`` silently used the bulk power-law fall speed on a typo).
        ("packages/atmosphere/legoesm/atmosphere/physics/microphysics/thompson.py", "thompson_microphysics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py", "_get_radiation_fn"),
        ("packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py", "make_radiation_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics.py", "optics_factory"),
        ("packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py", "get_turbulence_fn"),
        ("packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py", "make_turbulence_physics"),
        # CLUBB cloud_source selector (native ADG1-PDF vs shared grid-scale saturation;
        # the D9 forced-shared control). A typo must raise, not silently run the PDF cloud.
        ("packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py", "diagnose_cloud_and_buoyancy"),
        ("packages/atmosphere/legoesm/atmosphere/forcing/scm/scm.py", "__init__"),
        ("packages/core/legoesm/core/bulk_flux.py", "validate_bulk_scheme"),
        # Stable-regime MOST stability-function dispatch (stability_scheme):
        # the validator + the shared stable-branch dispatch twins (grow-only
        # lock so a silent-Dyer fallback can't be reintroduced).  2026-08-02:
        # the else-raise moved from psi_m/psi_h into the factored
        # _stable_psi_m/_stable_psi_h helpers (now ALSO consumed by
        # psi_m_coare/psi_h_coare for the selectable coare3 stable branch —
        # the inert surface_stability_scheme fix); psi_m/psi_h keep their
        # entry-time validate_stability_scheme guard, behaviourally locked by
        # tests/unit/test_stable_stability_functions.py unknown-scheme raises.
        ("packages/core/legoesm/core/bulk_flux.py", "validate_stability_scheme"),
        ("packages/core/legoesm/core/bulk_flux.py", "_stable_psi_m"),
        ("packages/core/legoesm/core/bulk_flux.py", "_stable_psi_h"),
        ("packages/core/legoesm/core/tracers.py", "index"),
        ("packages/core/legoesm/grids/capability.py", "instantiate"),
        ("packages/core/legoesm/grids/capability.py", "validate_runtime"),
        # normalization guard: a typo must not silently select 'dstarea', which
        # returns coverage x field on a partly covered polar row instead of the
        # area-weighted mean ('fracarea').
        ("packages/core/legoesm/grids/conservative_regrid.py", "compute_overlap_weights"),
        ("packages/core/legoesm/grids/cubed_sphere.py", "gnomonic_grids"),
        ("packages/core/legoesm/grids/factory.py", "create_grid"),
        ("packages/core/legoesm/grids/factory.py", "create_regional_grid"),
        ("packages/core/legoesm/grids/halo.py", "set_halo_backend"),
        ("packages/core/legoesm/parallel/device_config.py", "get_optimal_mesh"),
        ("packages/core/legoesm/parallel/halo_exchange_voronoi.py", "exchange_local_simulated"),
        ("packages/core/legoesm/parallel/runtime.py", "halo_exchange"),
        ("packages/core/legoesm/parallel/voronoi_mpi.py", "gather_voronoi_field"),
        ("packages/core/legoesm/parallel/voronoi_mpi.py", "initialize_voronoi_mpi"),
        ("packages/core/legoesm/parallel/voronoi_partition.py", "partition_voronoi_mesh"),
        ("packages/core/legoesm/parallel/voronoi_partition.py", "reorder_voronoi_for_sharding"),
        ("packages/core/legoesm/timestepping/dispatch.py", "dispatch_integrator"),
        ("packages/core/legoesm/timestepping/split_explicit.py", "split_explicit_step"),
        ("packages/coupler/legoesm/coupler/coupler.py", "ocean_tile_response"),
        ("packages/coupler/legoesm/coupler/lake/two_layer_lake.py", "step_lake"),
        # auto_dt_rce grid_type guard (hardened 2026-06-25; previously only
        # 'voronoi' was special-cased and mpas/icosahedral fell through to the
        # cubed_sphere ladder returning an unsafe dt, and an unknown grid_type
        # silently used the ladder).
        ("packages/coupler/legoesm/driver/rce_dt.py", "auto_dt_rce"),
        ("packages/coupler/legoesm/driver/kernel_registry.py", "resolve_kernel"),
        ("packages/coupler/legoesm/driver/physics_pipeline.py", "_resolve_physics_parameterization"),
        ("packages/coupler/legoesm/driver/physics_pipeline.py", "build_physics_pipeline"),
        ("packages/ice/legoesm/ice/sea_ice.py", "_bulk_flux_dispatch"),
        ("packages/ice/legoesm/ice/sea_ice.py", "step_sea_ice"),
        ("packages/ice/legoesm/ice/sea_ice.py", "_closing_rate_from_velocity"),
        ("packages/ice/legoesm/ice/shortwave.py", "compute_ice_sw"),
        ("packages/land/legoesm/land/carbon/carbon_cycle.py", "step_carbon"),
        # The multilayer-land bulk dispatch moved into the pluggable surface
        # scheme (surface_scheme/simple_seb.py) in the land/stable refactor
        # and LOST its unknown-scheme raise on the way; the guard is restored
        # there (fn-entry validate_bulk_scheme) and the baseline entry follows
        # the code.
        ("packages/land/legoesm/land/surface_scheme/simple_seb.py",
         "compute_simple_seb_fluxes"),
        ("packages/land/legoesm/land/slab_land.py", "step_land"),
        # Multilayer-land snowpack dispatch (bulk|layered), 2026-09-26.
        ("packages/land/legoesm/land/multilayer_land.py", "_step_multilayer_land_impl"),
        # Two-leaf canopy stomatal-model dispatch (ball_berry|medlyn): hardened
        # 2026-07-08 during the stomata consolidation — a bare ``else`` used to
        # silently run Ball-Berry on any typo. Guarded at BOTH ends: a fail-early
        # CanopyConfig.validate() at the non-jitted setup entry, and the leaf
        # kernel _compute_gs_and_ci as a trace-time backstop.
        ("packages/land/legoesm/land/canopy/config.py", "validate"),
        ("packages/land/legoesm/land/canopy/energy_balance.py", "_compute_gs_and_ci"),
        # LE_module leaf-energy dispatch (BT|PM): the internal residual uses a
        # bare ``else: # PM``, so a typo silently runs Penman-Monteith. Guarded
        # at the config validator AND at the solver entry (direct-call path).
        ("packages/land/legoesm/land/canopy/solver.py", "solve_canopy_closure"),
        # CLM-ML canopy-airspace turbulence dispatch (rsl_bonan|most): selects
        # whether the Harman & Finnigan roughness-sublayer correction is active.
        # An unknown value must raise, not fall through to the RSL default —
        # the two schemes give different surface exchange, so a silent default
        # would report MOST while running RSL. Guarded at BOTH ends:
        # CLMMLCanopyConfig.validate() at the flux entry (the ``validate`` pair
        # above covers both canopy configs) and the applier itself.
        ("packages/land/legoesm/land/canopy/clm_ml_interface.py",
         "_apply_turbulence_scheme"),
        ("packages/ml/legoesm/ml/physics/model.py", "_validate_microphysics_scheme"),
        ("packages/ml/legoesm/ml/training.py", "create_optimizer"),
        ("packages/ml/legoesm/training/aimip_params.py", "make_aimip_classical_spectral_physics"),
        # Unified campaign driver (D1): training-core dispatch (reserved cores
        # raise NotImplementedError, unknown raises ValueError) and the
        # campaign-wide rrtmgp radiation pin (only --smoke escapes
        # are explicit args, never silent).
        ("packages/ml/legoesm/training/campaign_driver.py", "validate_training_core"),
        ("packages/ml/legoesm/training/campaign_driver.py", "validate_campaign_radiation"),
        ("packages/ml/legoesm/training/loss_presets.py", "load_loss_preset"),
        ("packages/ocean/legoesm/ocean/biogeochemistry/config.py", "init_biogeo_state"),
        ("packages/ocean/legoesm/ocean/config.py", "to_ocean_config"),
        # MED-1 follow-up: under-ice relaxation validates freeze_scheme at fn
        # entry against eos.VALID_FREEZE_SCHEMES (per-cell liquidus target).
        ("packages/ocean/legoesm/ocean/coupler/omip2_applicator.py", "under_ice_freeze_relax"),
        ("packages/ocean/legoesm/ocean/dynamics/_flux_limiters.py", "resolve_tvd_limiter"),
        # In-substep barotropic Coriolis scheme (node 16; "avg" 4-pt avg vs
        # "een" NEMO enstrophy-conserving). A typo must raise, not silently
        # run the legacy null-mode 4-pt average.
        ("packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py", "barotropic_substeps_latlon_cgrid"),
        # Shared by BOTH barotropic entry points: the reconciliation-target
        # guard moved here so the standard-halo path and its wide-halo twin
        # cannot diverge on it again.
        ("packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py", "_reconcile_targets"),
        # NEMO mlf_baro_corr's AFTER-level reconciliation (barotropic_after_
        # reconcile). Called at fn entry on the static config value from BOTH
        # outer-step paths (_leapfrog_step and _nemo_mlf_step), so a typo stops
        # the step instead of silently selecting "off" -- i.e. silently NOT
        # running a reconciliation the card asked for.
        ("packages/ocean/legoesm/ocean/dynamics/barotropic_common.py", "validate_after_reconcile"),
        # n2_mode + n2_eos_form guards on the EVD convective trigger.
        ("packages/ocean/legoesm/ocean/physics/convection/enhanced_diffusion.py", "convective_K_A_flag"),
        # In-substep C-grid face-depth scheme (barotropic_face_depth:
        # min_rule | nemo_ssh_avg, #1226 zero-deviation item 2). A typo must
        # raise, not silently run the wrong flux/drag face-thickness rule.
        ("packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py", "_run_substep_loop"),
        # Barotropic substep loop-ENTRY seed face-depth scheme
        # (barotropic_seed_face_depth: min_rule | nemo_ssh_avg, #1226 round 2
        # item 1). A typo must raise, not silently reweight the seeded
        # U_bar/V_bar by the wrong face-thickness convention (adversarial
        # review of 7da6d7989 found this option is NOT inert where
        # min_water_column_m binds asymmetrically — see the
        # BarotropicConfig.barotropic_seed_face_depth docstring).
        ("packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py", "_depth_average_to_faces"),
        # AL81/EEN q-boundary convention (q_boundary: neumann_fill | nemo_live).
        # A typo must raise, not silently pick the coast-vorticity behaviour
        # (nemo_live keeps wall shear vorticity live; neumann_fill erases it).
        ("packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py", "pv_flux_al81_partial_cell"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_model.py", "__init__"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py", "_compute_advection_flux_div"),
        # bbl_adv_option=2 (in-stage Campin-Goosse BBL) x tracer_time_integrator
        # lane guard (S-42, docs/ocean/fidelity/nemo_branch_isomorphism_map.md):
        # the in-stage BBL hook is built ONLY on the rk3_ws tracer lane, so a
        # typo/mismatch on euler/ab2/rk3 used to resolve bbl_adv_option=2 and
        # silently run NO boundary layer. Lock the guard so it can't be
        # silently deleted; test_config_footguns.py exercises it directly.
        # Also guards momentum_flux_scheme="nemo_up3" x momentum_time_integrator
        # pairing (S-46/S-47, same map doc): NEMO's e3u(Kmm) face thickness for
        # dyn_adv_up3 is wired only inside the rk3_ws stage program, so nemo_up3
        # on any other integrator would silently run the legacy (measured
        # first-order-wrong) face-thickness rule instead -- a pairing NEMO
        # itself never runs. test_config_footguns.py exercises it directly.
        ("packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py", "_validate_config"),
        # outer_integrator dispatch (nemo_mlf P2, docs/ocean/fidelity/
        # nemo_mlf_step_transcription_spec.md §4/§7): a typo here would
        # silently fall through to _step_impl (the else branch) instead of
        # raising -- lock the guard so it can't be silently deleted.
        ("packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py", "_step_jitted"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py", "__init__"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py", "ocean_baroclinic_tendencies_cdgrid"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py", "_bc_ke_and_pressure_gradients"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py", "_bc_pv_flux"),
        # #1455: lateral_viscosity_e3_weighting dispatch (raises on an unknown
        # e3-weighting variant, and when paired with any operator other than
        # nemo_div_curl).
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py", "_bc_horizontal_viscosity"),
        # shortwave_scheme guard of the external (JRA55 bulk) surface forcing
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py", "_bc_external_surface_forcing"),
        # VORTEX round 5: after_ssh_form dispatch of the FIRST wzv call.  The
        # RK3 and modified-leapfrog programs leave different things in NEMO's
        # after-SSH slot when wzv reads it (stprk3.F90:225 against ssh_nxt),
        # so an unknown form must raise rather than silently run one of them.
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py", "nemo_qco_wzv_operands"),
        ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py", "mpas_ocean_baroclinic_tendencies"),
        # (nemo_drag_r_from_speed_sq's internal legacy-rejection raise is not
        # scanner-shaped; the canonical unknown-scheme guard is the validator.)
        ("packages/ocean/legoesm/ocean/dynamics/ocean_tendency_common.py", "validate_bottom_drag_scheme"),
        ("packages/ocean/legoesm/ocean/eos.py", "make_eos_fn"),
        ("packages/ocean/legoesm/ocean/eos.py", "freezing_point"),
        ("packages/ocean/legoesm/ocean/experiments/dino.py", "create_forcings"),
        ("packages/ocean/legoesm/ocean/experiments/dino.py", "create_initial_conditions"),
        # DINO model-config builder guards its scheme fields (gm_kappa_scheme,
        # lateral_tracer_mixing, gm_redi_mld_criterion) with fn-entry raises;
        # lock so the gm_redi_mld_criterion N2-integral guard can't be dropped.
        ("packages/ocean/legoesm/ocean/experiments/dino.py", "dino_lat_lon_model_config"),
        # MPAS DINO builder rejects the lat-lon-C-grid-only scheme fields
        # (gm_kappa_scheme=treguier, convection_evd_n2_time_level=
        # nemo_now_before). The MPAS vmix bridge threads neither the Nnn/Nbb
        # tracers nor the Nnn eta the zdfevd trigger arms need, so honouring
        # the field there would silently run the solver-state trigger (#1317).
        ("packages/ocean/legoesm/ocean/experiments/dino.py", "dino_mpas_model_config"),
        # make_bottom_drag_physics entry removed 2026-07-19: the dead
        # physics-level bottom-drag factory was deleted on main (c5f88d325);
        # the canonical guard is validate_bottom_drag_scheme above.
        ("packages/ocean/legoesm/ocean/physics/convection/integration.py", "make_convection_physics"),
        ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "gm_redi_tracer_tendency_latlon"),
        # gm_bolus_advection dispatch (centred | through_fct; hardened 2026-07-18
        # with the NEMO ldf_eiv_trp bolus-through-FCT option — a typo would
        # silently drop the GM bolus from BOTH paths).
        ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "nemo_iso_lap_tracer_tendency_latlon_cgrid"),
        ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py", "gm_redi_tracer_tendency_mpas"),
        ("packages/ocean/legoesm/ocean/physics/lateral_mixing/integration.py", "make_lateral_mixing_physics"),
        # MPAS ocean vmix/surface/convection factory: catke rejected, richardson
        # rejected (K-profile would be silently dropped), tke gated implicit-only
        # (hardened 2026-07 with the MPAS-TKE bridge). Grow-only lock so these
        # unknown-scheme raises can't be silently deleted.
        ("packages/ocean/legoesm/ocean/physics/mpas_physics.py", "make_mpas_ocean_physics"),
        ("packages/ocean/legoesm/ocean/physics/surface_forcing/bulk_formulas.py", "bulk_formula_surface_forcing"),
        ("packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py", "make_surface_forcing_physics"),
        ("packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py", "make_vertical_mixing_physics"),
        # K-profile composition + EVD trigger time-level dispatch
        # (vmix_background_mode: additive | nemo_max_floor;
        # EnhancedDiffusionConfig.evd_n2_time_level: solver_state |
        # nemo_now_before, #1317). Both are nested scheme-Config fields that
        # validate_strict never sees, so this fn-entry raise is the only
        # defence — a typo would silently pick a different composition or run
        # the trigger on the post-explicit Kaa state.
        ("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py", "compute_vertical_K_profiles"),
        # TKE surface-BC dispatch (surface_bc: veros_flux | nemo_dirichlet;
        # hardened 2026-07-16 with the NEMO nn_bc_surf=1 Dirichlet option — a
        # typo would silently run the Veros flux BC, a ~60x different surface
        # TKE under wind).
        ("packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py", "_surface_tke_dirichlet"),
        # TKE shear-production discretization dispatch (tke_shear_production:
        # squared_centered | nemo_burchard | nemo_face_native; #1226
        # sh2_walk.py Candidate E/F face-native zdfsh2.F90 transcription
        # added 2026-07-30) — a typo would silently keep the T-collapsed
        # squared form instead of the face-native shear NEMO actually
        # computes.
        ("packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py", "tke_vertical_mixing"),
        ("packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py", "_validate_post_mixing_cfg"),
        ("packages/ocean/legoesm/ocean/scm.py", "__init__"),
        ("packages/tools/legoesm/forcing/amip.py", "get_amip_preset"),
        ("packages/tools/legoesm/forcing/experiments.py", "create_experiment_config"),
    }
)

# Config-driven physics/dispatch factories that matter most (subset of baseline).
CRITICAL_FACTORIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py", "make_convection_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py", "make_turbulence_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py", "make_gwd_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py", "make_radiation_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py", "make_microphysics_physics"),
        ("packages/atmosphere/legoesm/atmosphere/physics/clouds/cloud_fraction.py", "compute_cloud_properties"),
        ("packages/land/legoesm/land/carbon/carbon_cycle.py", "step_carbon"),
        ("packages/coupler/legoesm/coupler/coupler.py", "ocean_tile_response"),
        ("packages/ice/legoesm/ice/sea_ice.py", "_bulk_flux_dispatch"),
        ("packages/ocean/legoesm/ocean/eos.py", "make_eos_fn"),
        ("packages/core/legoesm/timestepping/dispatch.py", "dispatch_integrator"),
        ("packages/core/legoesm/grids/factory.py", "create_grid"),
    }
)

_DISCOVERED, _PARSE_FAILURES = discover_hardened_dispatchers()


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(ra.discover_py_files())
    assert len(_DISCOVERED) >= 60, (
        f"only {len(_DISCOVERED)} hardened dispatchers discovered — detector or "
        f"discovery is broken (expected ~76)."
    )


def test_no_production_file_fails_to_parse() -> None:
    """A production source file that cannot be parsed would be silently dropped
    from discovery — surface it as a failure (round-1 #9 philosophy)."""
    assert not _PARSE_FAILURES, f"production files failed to parse: {_PARSE_FAILURES}"


def test_no_dispatch_guard_removed() -> None:
    """Grow-only ratchet: every baseline dispatcher must still raise on an unknown
    value. A missing one means its guard was deleted (silent-default regression)
    or it was renamed/removed without updating the baseline."""
    missing = sorted(BASELINE_DISPATCHERS - _DISCOVERED)
    assert not missing, (
        "Unknown-scheme guard(s) lost since baseline — re-add the inline "
        "``raise ValueError(f\"Unknown ...\")`` (CLAUDE.md dispatch rule), or, if "
        "the function was intentionally renamed/removed, update "
        "BASELINE_DISPATCHERS with a justifying note:\n"
        + "\n".join(f"  {rel}::{fn}" for rel, fn in missing)
    )


def test_baseline_entries_reference_existing_files() -> None:
    missing = sorted({rel for rel, _ in BASELINE_DISPATCHERS if not (ra.repo_root() / rel).is_file()})
    assert not missing, f"BASELINE_DISPATCHERS names non-existent files: {missing}"


def test_critical_factories_are_a_baseline_subset() -> None:
    extra = sorted(CRITICAL_FACTORIES - BASELINE_DISPATCHERS)
    assert not extra, f"CRITICAL_FACTORIES entries not in baseline: {extra}"


@pytest.mark.parametrize("entry", sorted(CRITICAL_FACTORIES), ids=lambda e: f"{e[0].split('/')[-1]}::{e[1]}")
def test_critical_factory_hardened(entry: tuple[str, str]) -> None:
    assert entry in _DISCOVERED, (
        f"critical scheme factory {entry[1]} in {entry[0]} no longer raises on an "
        f"unknown value — restore the inline unknown-scheme guard."
    )


# ---------------------------------------------------------------------------
# Behavioural teeth — invoke cheap pure-string factories with a bogus value and
# require ValueError. The remaining baseline entries are AST-only because they
# need a built state / config / JIT context too expensive to construct here.
# ---------------------------------------------------------------------------
def test_eos_factory_rejects_unknown_scheme() -> None:
    from legoesm.ocean.eos import make_eos_fn

    with pytest.raises(ValueError, match="(?i)unknown|unsupported"):
        make_eos_fn("__nonexistent_eos__")


def test_tvd_limiter_resolver_rejects_unknown() -> None:
    from legoesm.ocean.dynamics._flux_limiters import resolve_tvd_limiter

    with pytest.raises(ValueError, match="(?i)unknown|unsupported|limiter"):
        resolve_tvd_limiter("__nonexistent_limiter__")


def test_bulk_scheme_validator_rejects_unknown() -> None:
    from legoesm.core.bulk_flux import validate_bulk_scheme

    with pytest.raises(ValueError, match="(?i)unknown|unsupported|bulk"):
        validate_bulk_scheme("__nonexistent_bulk__")


# ---------------------------------------------------------------------------
# Non-vacuity self-tests
# ---------------------------------------------------------------------------
def test_detector_flags_guarded_function() -> None:
    src = (
        "def make(scheme):\n"
        "    if scheme == 'a':\n        return 1\n"
        "    raise ValueError(f'Unknown scheme {scheme!r}; expected one of a')\n"
    )
    assert function_is_hardened_dispatcher(ast.parse(src).body[0])


def test_detector_ignores_silent_default_function() -> None:
    src = (
        "def make(scheme):\n"
        "    if scheme == 'a':\n        return 1\n"
        "    return 0  # silent default — no raise\n"
    )
    assert not function_is_hardened_dispatcher(ast.parse(src).body[0])


def test_detector_ignores_unrelated_valueerror() -> None:
    src = "def f(x):\n    if x < 0:\n        raise ValueError('x must be positive')\n"
    assert not function_is_hardened_dispatcher(ast.parse(src).body[0])


def test_detector_ignores_raise_in_nested_scope() -> None:
    """A dispatcher whose only unknown-scheme raise lives in a NESTED function is
    NOT credited as hardened — the parent itself falls through silently."""
    src = (
        "def make(scheme):\n"
        "    def _inner(s):\n"
        "        raise ValueError(f'Unknown scheme {s!r}')\n"
        "    return 0  # parent silently defaults\n"
    )
    assert not function_is_hardened_dispatcher(ast.parse(src).body[0])
