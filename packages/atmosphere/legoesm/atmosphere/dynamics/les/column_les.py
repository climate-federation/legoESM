"""Spin off a forced column LES for each worst-performing column + diagnose K/w_e.

Stage 4–6 library of ``docs/COMPARE_REANALYSIS.md``: the orchestration that ties
the LES-side components into a runnable pipeline for one flagged column —

  manifest record + GCM column
    → regime + LES resolution      (les_regime, iter 12)
    → plane grid + stretched height coord with the column θ as θ_ref
    → SCM/large-scale forcing       (column_forcing iter 5 → plane physics_fn)
    → Newtonian top relaxation      (les_vertical_mapping, iter 13)
    → run the plane NH LES at LES resolution   (compressible_euler_plane)
    → diagnose the closure coefficient   (column_les_diagnosis, iter 14).

The data-agnostic orchestration (:func:`build_column_les_setup`,
:func:`run_column_les_pipeline`) is importable + unit-tested with the heavy LES
run injected as ``run_les_fn``; :func:`run_forced_les` is the real plane-dycore
time loop (reusing the ``run_les_plane`` machinery).  The CLI that loops a
manifest lives in ``scripts/run/run_column_les.py`` (imports this module).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.atmosphere.column_forcing import (
    ColumnLargeScaleState,
    build_column_scm_forcing,
    coriolis_f_c,
)
from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
    column_les_realism,
    diagnose_column_coefficient,
    gate_diagnosis_realism,
    mask_diagnosis_above,
)
from legoesm.atmosphere.dynamics.les.les_regime import (
    LESRegimeConfig,
    les_resolution_for_column,
)
from legoesm.atmosphere.dynamics.les.les_vertical_mapping import (
    build_top_relaxation,
    interpolate_column_to_les,
)

# --- LES dynamics stability (split-explicit acoustic CFL) ---
# Hard upper bound on the HORIZONTAL acoustic Courant for the plane-LES dycore
# (Skamarock-Klemp split-explicit forward-backward + RK3, the run_forced_les config).
# Set to the CLASSICAL acoustic CFL limit 1.0 — the codebase's own documented-safe
# region (the real-LES test runs at C_a ≈ 0.57 and notes "< 1"); there is no evidence
# the FB scheme is stable above it, so 1.0 (not a looser guess) is the defensible bound.
# This is the standard acoustic CFL — NOT the subgrid/Smagorinsky re-tuning §7 flags.
# c_sound is the DRY sound speed, which is EXACT for this dycore (its EOS/pressure is
# dry — R_d, c_vd — so its acoustic mode propagates at the dry speed, not moist).
_LES_ACOUSTIC_CFL_MAX = 1.0
# The pre-flight uses the REST-state c_sound, but the column WARMS convectively during
# the run (c_s ∝ sqrt(θ); +30 K on ~290 K ≈ +5 %), raising the run-time acoustic
# Courant above the rest estimate. Inflate the rest Courant by this margin so a config
# that is marginal at rest but unstable once warm is rejected (Codex iter 103).
_LES_ACOUSTIC_WARMING_MARGIN = 1.05


class ColumnLESConfig(NamedTuple):
    """Campaign config for the column-LES spin-off."""

    regime: LESRegimeConfig = LESRegimeConfig()
    relax_width_frac: float = 0.25   # relaxation layer = top fraction of the domain
    relax_tau_s: float = 600.0       # Newtonian relaxation timescale [s]
    p_sfc_Pa: float = 1.0e5
    diagnosis_method: str = "eddy_diffusivity"
    # The GCM CLUBBLiteConfig.l_mix_max, REQUIRED for the dimensionless
    # ``"clubb_coefficient"`` diagnosis (C_K = K_m/(ℓ·√wp2)) so the diagnosed
    # mixing length matches the GCM's; unused by the other methods.
    clubb_l_mix_max: float | None = None
    # Multi-coefficient: when set, the spin-off LES is diagnosed for EVERY method
    # in one run (e.g. ("clubb_coefficient", "prandtl_number")) and
    # ``process_column`` returns ``{method: diagnosis}``; overrides
    # ``diagnosis_method``.  ``None`` ⇒ the single-coefficient ``diagnosis_method``.
    diagnosis_methods: tuple[str, ...] | None = None
    # LES-realism TRUST gate (§9): a dead/blown-up LES has its diagnosis
    # invalidated (the column keeps its background) instead of injecting a
    # meaningless coefficient.  ``les_realism_wp2_floor=None`` ⇒ the diagnosis
    # module's default "turbulence developed" threshold.
    gate_les_realism: bool = True
    les_realism_wp2_floor: float | None = None
    # Thermodynamic-drift threshold [K] (iter 66): reject a turbulent + finite but
    # DRIFTED LES (mean θ wandered off the GCM column). None ⇒ the module default.
    les_realism_theta_drift_K: float | None = None
    # Moisture physical-sanity cap [kg/kg] (iter 67): reject a finite-but-runaway
    # water-vapor blow-up (max q_v above this). None ⇒ the module default.
    les_realism_q_v_max: float | None = None
    # OPT-IN supersaturation cap [-] (iter 68): reject a turbulent + finite LES whose
    # max RH = q_v/q_sat exceeds this (cold-cloud runaway). None ⇒ OFF (the default;
    # the test mocks use an unphysical uniform q). Recommended ~1.5 when enabled.
    les_realism_rh_max: float | None = None
    # OPT-IN prescribed surface-flux BC (iter 364): when True, the spin-off LES gets a
    # ``prescribe="fluxes"`` surface BC = the GCM bulk surface sensible/latent fluxes
    # (REUSED, no new tunables) converted to kinematic θ/q_v fluxes — the surface-driven
    # turbulence driver a surface-flux-free LES omits.  Requires the column SST to be
    # threaded to ``process_column`` (``make_les_diagnose_fn`` supplies ``model_ctx.sst_K``);
    # ``False`` (default) ⇒ the iter-148 surface-flux-free LES, byte-unchanged.
    surface_flux: bool = False


def validate_column_les_config(config: ColumnLESConfig) -> None:
    """Reject non-physical campaign settings (relaxation, surface pressure)."""
    if not (0.0 < config.relax_width_frac <= 1.0):
        raise ValueError(
            f"relax_width_frac must be in (0, 1], got {config.relax_width_frac}."
        )
    if config.relax_tau_s <= 0.0:
        raise ValueError(f"relax_tau_s must be > 0, got {config.relax_tau_s}.")
    if config.p_sfc_Pa <= 0.0:
        raise ValueError(f"p_sfc_Pa must be > 0, got {config.p_sfc_Pa}.")
    # Realism-gate thresholds: None ⇒ the diagnosis module default; an explicit
    # value must be finite + positive (0 / negative / NaN / inf would silently
    # reject every column — a confusing footgun, Codex iter-66).
    for _name, _v in (("les_realism_wp2_floor", config.les_realism_wp2_floor),
                      ("les_realism_theta_drift_K", config.les_realism_theta_drift_K),
                      ("les_realism_q_v_max", config.les_realism_q_v_max),
                      ("les_realism_rh_max", config.les_realism_rh_max)):
        if _v is not None and not (0.0 < _v < float("inf")):
            raise ValueError(
                f"{_name} must be None or a finite positive value, got {_v}.")
    # clubb_coefficient (single OR one of the multi methods) needs l_mix_max.
    active_methods = (
        config.diagnosis_methods
        if config.diagnosis_methods is not None
        else (config.diagnosis_method,)
    )
    if config.diagnosis_methods is not None and len(config.diagnosis_methods) == 0:
        raise ValueError("diagnosis_methods must be a non-empty tuple (or None).")
    # clubb_coefficient (C_K) and c_eps both evaluate the GCM mixing length.
    needs_lmix = {"clubb_coefficient", "c_eps"}
    if needs_lmix.intersection(active_methods) and (
        config.clubb_l_mix_max is None or config.clubb_l_mix_max <= 0.0
    ):
        raise ValueError(
            f"the {sorted(needs_lmix.intersection(active_methods))} diagnosis "
            "requires clubb_l_mix_max > 0 (the GCM CLUBBLiteConfig.l_mix_max)."
        )


def coefficient_value(diagnosis: Any, method: str):
    """Reduce a diagnosis object to the representative coefficient for ``method``.

    ``"eddy_diffusivity"`` → the ``K`` profile; ``"entrainment"`` → the scalar
    ``w_entrainment``.  Raises on an unexpected diagnosis/method pair rather than
    silently emitting ``NaN``.
    """
    if method == "eddy_diffusivity" and hasattr(diagnosis, "K"):
        return diagnosis.K
    if method == "clubb_coefficient" and hasattr(diagnosis, "C_K"):
        return diagnosis.C_K
    if method == "prandtl_number" and hasattr(diagnosis, "Pr_t"):
        return diagnosis.Pr_t
    if method == "c_eps" and hasattr(diagnosis, "C_eps"):
        return diagnosis.C_eps
    if method == "entrainment" and hasattr(diagnosis, "w_entrainment"):
        return diagnosis.w_entrainment
    raise ValueError(
        f"diagnosis {type(diagnosis).__name__} has no coefficient for method "
        f"{method!r}."
    )


class ColumnLESSetup(NamedTuple):
    """Everything needed to run + diagnose one column's LES."""

    grid: Any
    height_coord: Any
    forcing_physics: Callable        # plane physics_fn (large-scale forcing)
    relax_theta_target: jax.Array    # (nlev,) GCM θ on the LES grid
    relax_rate: jax.Array            # (nlev,) Newtonian rate [1/s]
    q_v_init: jax.Array              # (nlev,) GCM q_v on the LES grid (moist IC)
    regime: str
    resolution: Any
    f_c: float


def build_column_les_setup(
    *,
    cape_J_kg: float,
    lat_rad: float,
    gcm_z: jax.Array,
    gcm_theta: jax.Array,
    ls_state: ColumnLargeScaleState,
    config: ColumnLESConfig = ColumnLESConfig(),
) -> ColumnLESSetup:
    """Build the plane grid + height coord + forcing + relaxation for one column.

    ``cape_J_kg`` (from the manifest env tag) picks the regime/resolution;
    ``lat_rad`` sets the f-plane Coriolis; ``gcm_z`` / ``gcm_theta`` are the GCM
    column heights + potential temperature (ascending or top-down — interpolation
    sorts internally); ``ls_state`` (from the grid-side extractor, iter 10) is
    the large-scale forcing.  The LES top must lie within the GCM column
    (:func:`build_top_relaxation` raises otherwise).
    """
    from legoesm.atmosphere.dynamics.plane_large_scale_forcing import (
        make_plane_ls_forcing_physics,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_stretched_height_coordinate

    validate_column_les_config(config)
    # The forced-LES path applies the LARGE-SCALE forcing channels (subsidence,
    # horizontal advection, geostrophic wind) + the top relaxation toward the GCM
    # column θ, AND now (iter 151) a PRESCRIBED-FLUX surface BC (prescribe="fluxes":
    # the surface kinematic θ/q_v fluxes injected on the surface cell by
    # make_plane_ls_forcing_physics, the separate surface-scheme source the no-flux
    # interior SGS leaves room for). It does NOT yet apply a prescribed surface-
    # TEMPERATURE BC (prescribe="T_s" — that needs a bulk-flux closure C_H·|U|·
    # (θ_sfc−θ_1); a separate future channel). Silently dropping a requested T_s
    # would force a convective column's LES with NO surface buoyancy flux — its
    # PRIMARY turbulence driver — and yield a quietly-wrong closure diagnosis, so
    # fail LOUD (dispatch-hardening). The compare-reanalysis extractors leave
    # prescribe="none", so this never fires today. (An invalid prescribe value still
    # gets its own error from build_column_scm_forcing's validate_forcing;
    # prescribe="none" with surface fields nonetheless set is caught downstream by its
    # _check_surface_exclusivity — so every misconfiguration is LOUD, none dropped.)
    if ls_state.prescribe == "T_s":
        raise ValueError(
            "build_column_les_setup: ls_state.prescribe='T_s' requests a prescribed "
            "surface-TEMPERATURE BC, but the forced-LES path does NOT yet apply one — "
            "it needs a bulk-flux closure C_H·|U|·(θ_sfc−θ_1) to convert T_s to a "
            "surface kinematic flux. Use prescribe='fluxes' (supply the surface "
            "kinematic θ/q_v fluxes directly — now supported), or prescribe='none' "
            "(surface-flux-free), or implement the bulk-flux T_s path before using it."
        )
    regime, res = les_resolution_for_column(cape_J_kg, config.regime)
    f_c = coriolis_f_c(lat_rad)
    grid = create_plane_grid(
        nx=res.nx, ny=res.ny, nlev=res.nlev, dx=res.dx_m, dy=res.dx_m,
        coriolis_mode="f_plane", f0=f_c,
    )
    # The GCM column θ is the LES reference θ(z): interpolate to whatever heights
    # the stretched coordinate builds.
    theta_ref_fn = lambda z: interpolate_column_to_les(gcm_z, gcm_theta, z)  # noqa: E731
    hc = create_stretched_height_coordinate(
        res.nlev, H=res.domain_top_m, dz_sfc=res.dz_sfc_m,
        theta_ref_fn=theta_ref_fn, p_sfc=config.p_sfc_Pa,
    )

    # Steady large-scale forcing profiles from the validated SCMForcing assembly.
    # The forcing lives on the GCM column levels (= gcm_z); interpolate each
    # channel onto the LES height grid (hc.z_full) before handing it to the
    # plane forcing physics, which expects (nlev_LES,) profiles.
    forcing = build_column_scm_forcing(ls_state)

    gcm_z = jnp.asarray(gcm_z)

    def _to_les(profile_fn):
        if profile_fn is None:
            return None
        profile = jnp.asarray(profile_fn(0.0))
        if profile.shape != gcm_z.shape:
            raise ValueError(
                f"forcing profile shape {profile.shape} != gcm_z shape "
                f"{gcm_z.shape}; the forcing must be co-located on the GCM column."
            )
        return interpolate_column_to_les(gcm_z, profile, hc.z_full)

    # Prescribed-FLUX surface BC (prescribe="fluxes"): the scalar surface kinematic
    # θ/q_v fluxes (steady scalar callables) injected on the LES surface cell. None
    # for prescribe="none"/"T_s" (T_s is rejected above); each flux is independently
    # optional (validate_forcing requires ≥1 when prescribe="fluxes").
    w_theta_sfc = w_qv_sfc = None
    if forcing.prescribe == "fluxes":
        w_theta_sfc = None if forcing.w_th_s is None else float(forcing.w_th_s(0.0))
        w_qv_sfc = None if forcing.w_qv_s is None else float(forcing.w_qv_s(0.0))
        # Fail LOUD on a non-finite surface flux: validate_forcing only checks the
        # flux callables are PRESENT (state-blind), not their VALUES. A NaN/inf flux
        # (e.g. from a future SST→bulk-flux extractor div-by-zero) would silently
        # poison the LES surface cell → a blown-up run + invalid diagnosis, caught
        # only late by the realism gate. Catch it here at setup.
        for _name, _val in (("w_th_s", w_theta_sfc), ("w_qv_s", w_qv_sfc)):
            if _val is not None and not math.isfinite(_val):
                raise ValueError(
                    f"build_column_les_setup: prescribed surface flux {_name}={_val} "
                    "is non-finite — a NaN/inf surface flux would poison the LES "
                    "surface cell (blow up the run, invalidate the diagnosis). Check "
                    "the surface-flux source."
                )

    forcing_physics = make_plane_ls_forcing_physics(
        hc,
        w_ls=_to_les(forcing.subsidence_w),
        theta_adv=_to_les(forcing.theta_adv),
        qv_adv=_to_les(forcing.qv_adv),
        w_theta_sfc=w_theta_sfc,
        w_qv_sfc=w_qv_sfc,
    )

    # Wire the GCM column's geostrophic wind into the plane Coriolis as its
    # reference wind: the f-plane Coriolis term reads u_geo0/v_geo0 from the
    # height coordinate and applies f×(V − V_geo).  None (cubed-sphere /
    # equatorial columns, where the extractor leaves u_geo=None) keeps u_geo0=None
    # → the Coriolis falls back to f×V (no geostrophic target), unchanged.
    u_geo0 = _to_les(forcing.u_geo)
    v_geo0 = _to_les(forcing.v_geo)
    if u_geo0 is not None:
        hc = hc._replace(u_geo0=u_geo0, v_geo0=v_geo0)

    relax_width_m = config.relax_width_frac * res.domain_top_m
    target, rate = build_top_relaxation(
        hc.z_full, res.domain_top_m, gcm_z, gcm_theta,
        relax_width_m=relax_width_m, inv_tau=1.0 / config.relax_tau_s,
    )
    # Moist initial condition: the GCM column q_v interpolated onto the LES grid
    # (the LES must carry moisture so the resolved w'q_v' flux + the moist
    # diagnosis exist).
    q_v_init = interpolate_column_to_les(gcm_z, ls_state.q_v, hc.z_full)
    return ColumnLESSetup(
        grid=grid, height_coord=hc, forcing_physics=forcing_physics,
        relax_theta_target=target, relax_rate=rate, q_v_init=q_v_init,
        regime=regime, resolution=res, f_c=f_c,
    )


def run_column_les_pipeline(
    setup: ColumnLESSetup,
    run_les_fn: Callable[[ColumnLESSetup], Any],
    *,
    method: str = "eddy_diffusivity",
    methods: list[str] | tuple[str, ...] | None = None,
    qv_slot: int = 0,
    l_mix_max: float | None = None,
    gate_realism: bool = True,
    realism_wp2_floor: float | None = None,
    realism_theta_drift_K: float | None = None,
    realism_q_v_max: float | None = None,
    realism_rh_max: float | None = None,
    relax_width_frac: float | None = None,
):
    """Run the LES (via ``run_les_fn``) and diagnose its closure coefficient(s).

    ``run_les_fn(setup)`` returns the finished plane LES state; the closure
    coefficient is then diagnosed with :func:`diagnose_column_coefficient`
    (``method`` raises on unknown).  ``run_les_fn`` is injected so the heavy run
    can be mocked in tests and swapped for :func:`run_forced_les` in production.
    ``l_mix_max`` (the GCM mixing length) is required for ``"clubb_coefficient"``
    and ignored otherwise.

    **Diagnose-many** (``methods``): the LES run is the loop's dominant cost, so
    when ``methods`` (a list) is given the LES runs ONCE and every method is
    diagnosed from the SAME final state — returning ``{method: diagnosis}`` (vs a
    single diagnosis for the scalar ``method`` path).  This lets a multi-coefficient
    correction (e.g. ``C_K`` AND ``Pr_t``) share one spin-off LES per column.

    **Realism trust gate** (``gate_realism``, default on, §9): the LES is checked
    ONCE with :func:`~legoesm.atmosphere.dynamics.les.column_les_diagnosis.column_les_realism`
    (developed turbulence + finite); a dead / blown-up LES has its diagnosis
    INVALIDATED (the loop then keeps the column's background coefficient) rather
    than injecting a finite-but-meaningless value.  The same realism flag gates
    every method in the diagnose-many path.

    **Sponge exclusion** (``relax_width_frac``): when given, diagnosis levels inside the LES
    top relaxation layer (height ``≥ domain_top·(1 − relax_width_frac)``) are invalidated
    (:func:`mask_diagnosis_above`) — the relaxation nudges θ toward the GCM column there, so a
    coefficient diagnosed in the sponge is contaminated and must not be averaged in ("keep the
    diagnosis below the sponge", §Risks).  ``None`` (default) keeps every interior level
    (byte-identical); pass ``config.relax_width_frac`` to enable it.
    """
    final_state = run_les_fn(setup)
    # The sponge base height: levels at/above this are inside the top relaxation layer.
    z_max = None
    if relax_width_frac is not None:
        domain_top = jnp.max(jnp.asarray(setup.height_coord.z_full))
        z_max = domain_top * (1.0 - relax_width_frac)
    # None ⇒ column_les_realism uses its own (module-default) threshold for each.
    realism_kw = {"qv_slot": qv_slot}      # the moisture cap reads tracer slot qv_slot
    if realism_wp2_floor is not None:
        realism_kw["wp2_floor"] = realism_wp2_floor
    if realism_q_v_max is not None:
        realism_kw["q_v_max"] = realism_q_v_max
    if realism_rh_max is not None:
        realism_kw["rh_max"] = realism_rh_max
    if realism_theta_drift_K is not None:
        realism_kw["theta_drift_rms_max_K"] = realism_theta_drift_K
    realistic = (
        column_les_realism(final_state, setup.height_coord, **realism_kw)
        if gate_realism else jnp.asarray(True)
    )
    if methods is not None:
        if len(methods) == 0:
            raise ValueError("methods must be a non-empty list (or None).")
        return {
            m: mask_diagnosis_above(
                gate_diagnosis_realism(
                    diagnose_column_coefficient(
                        final_state, setup.height_coord, method=m, qv_slot=qv_slot,
                        l_mix_max=l_mix_max),
                    realistic),
                z_max,
            )
            for m in methods
        }
    return mask_diagnosis_above(
        gate_diagnosis_realism(
            diagnose_column_coefficient(
                final_state, setup.height_coord, method=method, qv_slot=qv_slot,
                l_mix_max=l_mix_max),
            realistic),
        z_max,
    )


def run_forced_les(
    setup: ColumnLESSetup, *, dt_s: float, n_steps: int, theta_prime_seed: float = 0.1
):
    """Real plane-NH LES time loop with the large-scale forcing + top relaxation.

    Reuses the ``compressible_euler_plane`` dycore (``PlaneCompressibleEulerModel``)
    with the setup's ``forcing_physics`` applied each step and the Newtonian top
    relaxation nudging θ toward the GCM column near the LES top.  Returns the
    final plane state.  Heavy (LES resolution) — used by :func:`main`, not the
    unit tests.
    """
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel,
        make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.atmosphere.dynamics.les.les_vertical_mapping import relaxation_tendency

    # Top relaxation: the target equals ``hc.theta_ref`` by construction (both are
    # the GCM column θ interpolated onto ``hc.z_full``), so this relaxes the LES
    # total θ back to the GCM reference near the lid — i.e. damps the resolved θ'
    # perturbation in the relaxation layer (the standard top sponge toward the
    # GCM column), leaving the diagnosis region below it untouched.
    grid, hc = setup.grid, setup.height_coord
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        use_coriolis=True, semi_implicit_acoustic=True,
        substep_horizontal_acoustic=True, smagorinsky_cs=0.17,
        sgs_vertical_diffusion=True, fix_mass=True, anchor_mass_to_initial=True,
    )
    # FAIL FAST on an acoustically-unstable timestep BEFORE the (multi-day) run: this
    # config solves the vertical acoustic mode semi-implicitly + substeps the
    # horizontal, so the binding EXPLICIT acoustic CFL is HORIZONTAL (dx). A grossly-
    # oversized dt_s blows the LES up → an invalid diagnosis + wasted HPC compute. This
    # checks the ACOUSTIC mode ONLY (the most restrictive explicit mode for a
    # compressible dycore — c_sound ≫ wind); it is not a full advective/diffusion/
    # buoyancy stability proof. Host-side scalar check (one sync; run_forced_les is
    # eager), rest-state c_s inflated by the convective-warming margin.
    from legoesm.atmosphere.dynamics.shared.cfl_diagnostic import acoustic_courant_horizontal
    c_acoustic = float(acoustic_courant_horizontal(
        hc, grid, dt_s, n_acoustic_substeps=cfg.n_acoustic_substeps))
    c_effective = c_acoustic * _LES_ACOUSTIC_WARMING_MARGIN
    if c_effective > _LES_ACOUSTIC_CFL_MAX:
        # Recommend a dt safely BELOW the limit (0.95×), not exactly on it, so a user
        # copying it verbatim is not left sitting at the marginal boundary.
        dt_safe = 0.95 * _LES_ACOUSTIC_CFL_MAX * dt_s / c_effective
        raise ValueError(
            f"run_forced_les: horizontal ACOUSTIC Courant {c_acoustic:.3g} "
            f"(×{_LES_ACOUSTIC_WARMING_MARGIN} convective-warming margin = "
            f"{c_effective:.3g}) exceeds the stable limit {_LES_ACOUSTIC_CFL_MAX} "
            f"(dx={float(grid.dx):.3g} m, dt_s={dt_s} s, "
            f"n_acoustic_substeps={cfg.n_acoustic_substeps}) — the LES would blow up. "
            f"Reduce dt_s to <= {dt_safe:.3g} s. NOTE: this is the ACOUSTIC CFL only, "
            "not a full advective/diffusion/buoyancy stability check.")
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    dtype = grid.area_T.dtype
    state = make_rest_state(grid, hc, dtype=dtype)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    # Small θ' seed so resolved eddies spin up.
    key = jax.random.PRNGKey(0)
    seed = theta_prime_seed * jax.random.normal(key, (ny, nx, nlev), dtype=dtype)
    # Moist IC: broadcast the GCM column q_v (on the LES grid) into tracer slot 0
    # so the LES carries moisture (a dry rest state has no tracers, which would
    # break the resolved w'q_v' flux + the moist closure diagnosis).
    q_v_init = jnp.asarray(setup.q_v_init, dtype=dtype)
    tracers = jnp.zeros((ny, nx, nlev, 1), dtype=dtype).at[..., 0].set(
        jnp.broadcast_to(q_v_init, (ny, nx, nlev))
    )
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=seed),
        tracers=state.tracers.replace(data=tracers),
    )

    target = jnp.asarray(setup.relax_theta_target, dtype=dtype)
    rate = jnp.asarray(setup.relax_rate, dtype=dtype)

    def _relax(st):
        theta_total = hc.theta_ref + st.theta_prime.data
        tend = relaxation_tendency(theta_total, target, rate)
        return st._replace(
            theta_prime=st.theta_prime.replace(data=st.theta_prime.data + dt_s * tend)
        )

    for _ in range(int(n_steps)):
        state = model.step(state, dt=dt_s, physics_fn=setup.forcing_physics)
        state = _relax(state)
    return state


def column_surface_kinematic_fluxes(
    *,
    T_col: jax.Array,
    q_v_col: jax.Array,
    u_col: jax.Array,
    v_col: jax.Array,
    p_full_col: jax.Array,
    sst_K: jax.Array,
    p_s: jax.Array,
    surface_config: Any = None,
) -> tuple[jax.Array, jax.Array]:
    """Surface kinematic θ/q_v fluxes for the LES ``prescribe="fluxes"`` BC.

    Computes the GCM column's surface sensible/latent heat fluxes by REUSING the GCM
    bulk surface scheme (:func:`...surface_layer.compute_surface_fluxes` — so NO new
    tunables; the same `Cd_neutral`/`Ch_neutral` the GCM used) from the SURFACE cell
    (the highest-pressure level — ordering-robust) + the column SST, then converts the
    ``W/m²`` fluxes to the kinematic fluxes the iter-151 surface BC consumes
    (``plane_large_scale_forcing``, ``w'θ'_s`` [K·m/s], ``w'q'_s`` [kg/kg·m/s]):

        ``w'θ'_s = shflx / (ρ₁·c_pd) · (p_ref/p_s)^κ``   (T→θ exner scaling at the surface)
        ``w'q'_s = lhflx / (ρ₁·L_v)``

    A warm SST (``SST > T₁``) gives ``shflx>0 ⇒ w'θ'_s>0`` (the surface WARMS the air —
    the convective-column turbulence driver the surface-flux-free LES omits).
    ``surface_config`` defaults to the standard :class:`SurfaceLayerConfig`.
    Returns ``(w_th_s, w_qv_s)`` scalars.  Pure-JAX, differentiable.

    SCOPE / assumptions (opt-in, default-OFF; the monotonic + LES-realism gates bound any
    residual error to "no improvement", never a worse bias):

    * **Ocean-oriented ``q_sfc``.**  ``q_sfc = q_sat(SST)`` assumes a SATURATED surface.
      The SENSIBLE (buoyancy) flux — the PRIMARY turbulence driver — is exact (it uses
      ``T_sfc = SST``, not ``q_sfc``); only the LATENT flux over-estimates a sub-saturated
      LAND surface.  Most convective worst columns are oceanic, and the latent flux is
      secondary for the momentum closure.
    * **Default bulk coefficients are acceptable.**  The diagnosed
      ``C_K = K_m/(ℓ·√wp2)`` is a CLOSURE constant — ~invariant to the surface-flux
      MAGNITUDE (self-similar turbulence).  The flux mainly governs WHETHER turbulence
      develops (the diagnosis-VALIDITY rate), not the C_K VALUE, so using the standard
      ``SurfaceLayerConfig`` rather than threading the GCM's exact coefficients is
      second-order.
    * **Constant flux.**  Computed ONCE from the time-mean column (the ``prescribe="fluxes"``
      design, iter 151) — appropriate for the short worst-column spin-off; an over-warming
      run is caught by the realism gate (``theta_drift``).
    """
    from legoesm.atmosphere.physics._shared import exner_function, virtual_temperature
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        SurfaceLayerConfig,
        compute_surface_fluxes,
    )
    from legoesm.thermo import saturation_mixing_ratio

    from legoesm import constants

    cfg = SurfaceLayerConfig() if surface_config is None else surface_config
    p_full_col = jnp.asarray(p_full_col)
    s = jnp.argmax(p_full_col)                          # surface cell = highest pressure
    T_1 = jnp.take(jnp.asarray(T_col), s)
    q_1 = jnp.take(jnp.asarray(q_v_col), s)
    u_1 = jnp.take(jnp.asarray(u_col), s)
    v_1 = jnp.take(jnp.asarray(v_col), s)
    p_1 = jnp.take(p_full_col, s)
    sst_K = jnp.asarray(sst_K)
    p_s = jnp.asarray(p_s)
    rho_1 = p_1 / (constants.R_d * virtual_temperature(T_1, q_1))
    q_sfc = saturation_mixing_ratio(sst_K, p_s)
    a1 = jnp.atleast_1d
    _, _, shflx, lhflx, _ = compute_surface_fluxes(
        a1(u_1), a1(v_1), a1(T_1), a1(q_1), a1(sst_K), a1(q_sfc), a1(rho_1), cfg)
    # θ-flux = (sensible heat flux)/(ρ·c_p) · 1/Π, with the canonical Exner helper
    # (1/Π = (p_ref/p)^κ) — no re-derived Poisson power (CLAUDE.md "never re-derive").
    exner_inv = 1.0 / exner_function(p_s)
    w_th_s = shflx[0] / (rho_1 * constants.c_pd) * exner_inv
    w_qv_s = lhflx[0] / (rho_1 * constants.L_v)
    return w_th_s, w_qv_s


def extract_gcm_column(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma: Any,
    col_index: tuple[int, ...],
    lat_rad: float,
    phis: jax.Array | None = None,
    sst_K: jax.Array | None = None,
    surface_config: Any = None,
) -> tuple[jax.Array, jax.Array, ColumnLargeScaleState]:
    """Extract one column's heights, θ profile, and large-scale forcing state.

    Reuses the grid-side extractor (iter 10/27) for the ``ColumnLargeScaleState``
    and the hydrostatic height integral + Exner for the LES reference profiles.
    Returns ``(gcm_z, gcm_theta, ls_state)`` — ``gcm_z`` / ``gcm_theta`` are the
    column's full-level heights [m] and potential temperature [K].

    Grid-agnostic: ``col_index`` is ``(i_lat, i_lon)`` on a lat-lon grid or
    ``(face, i, j)`` on a cubed-sphere grid — the spatial gather
    ``arr[tuple(col_index)]`` keeps the trailing level axis either way, and the
    forcing extraction routes through the grid dispatcher.

    ``sst_K`` (OPT-IN; default ``None`` ⇒ the iter-148 surface-flux-free LES, unchanged):
    when the column's surface field is supplied, a ``prescribe="fluxes"`` surface BC is
    added — the GCM bulk surface sensible/latent fluxes (REUSED via
    :func:`column_surface_kinematic_fluxes`, no new tunables) converted to the kinematic
    θ/q_v fluxes — so a surface-driven (convective) column's LES gets its PRIMARY
    turbulence driver.  ``surface_config`` (a :class:`SurfaceLayerConfig`) overrides the
    default bulk coefficients.
    """
    from legoesm.atmosphere.dynamics.column_large_scale_extract import (
        extract_column_forcing,
    )
    from legoesm.atmosphere.physics._shared import (
        compute_heights_from_sigma,
        exner_function,
    )

    ls_state = extract_column_forcing(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=lat_rad, col_index=col_index, phis=phis,
    )
    idx = tuple(int(c) for c in col_index)
    p_s_col = jnp.asarray(p_s)[idx]
    # The column's TRUE full/half-level pressures from the model's coordinate (hybrid-correct;
    # iter 341): the LES reference θ + heights must be built on the model's ACTUAL levels
    # (``p = A·p_ref + B·p_s`` for a hybrid coordinate, the default), not pure-sigma ``σ·p_s``.
    # For a SigmaCoordinate ``pressure_at_full == σ·p_s`` (byte-identical).
    p_full_col = jnp.asarray(sigma.pressure_at_full(p_s_col))
    p_half_col = jnp.asarray(sigma.pressure_at_half(p_s_col))
    T_col = jnp.asarray(T)[idx]
    q_col = jnp.asarray(q_v)[idx]
    z_full, _ = compute_heights_from_sigma(
        T_col[None, :], p_half_col[None, :], q_col[None, :]
    )
    gcm_z = z_full[0]
    gcm_theta = T_col / exner_function(p_full_col)

    # OPT-IN prescribed surface-flux BC (default off: sst_K=None ⇒ surface-flux-free,
    # iter-148 behaviour byte-unchanged).  The surface flux is the GCM bulk flux REUSED +
    # converted to kinematic — its iter-148 lock relaxed now that the LES applies it (151).
    if sst_K is not None:
        # The bulk flux needs the CELL-CENTRED surface wind |U₁| = √(u₁²+v₁²).  The MPAS
        # path passes ``u=u_edge, v=None`` (edge-normal velocity), so RECONSTRUCT the
        # cell-centred geographic wind from the edge normals via the canonical Perot
        # reconstruction (``reconstruct_cell_velocity`` — REUSE, the same the MPAS column
        # physics use; differentiable) before gathering the column.  Cell grids already
        # carry (u, v).
        if v is None:
            from legoesm.grids.voronoi import reconstruct_cell_velocity
            u_cell, v_cell = reconstruct_cell_velocity(jnp.asarray(u), grid)
            u_col = u_cell[idx]
            v_col = v_cell[idx]
        else:
            u_col = jnp.asarray(u)[idx]
            v_col = jnp.asarray(v)[idx]
        sst_col = jnp.asarray(sst_K)[idx]
        w_th_s, w_qv_s = column_surface_kinematic_fluxes(
            T_col=T_col, q_v_col=q_col, u_col=u_col, v_col=v_col,
            p_full_col=p_full_col, sst_K=sst_col, p_s=p_s_col,
            surface_config=surface_config,
        )
        ls_state = ls_state._replace(prescribe="fluxes", w_th_s=w_th_s, w_qv_s=w_qv_s)
    return gcm_z, gcm_theta, ls_state


def process_column(
    record: Any,
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma: Any,
    config: ColumnLESConfig,
    run_les_fn: Callable[[ColumnLESSetup], Any],
    phis: jax.Array | None = None,
    sst_K: jax.Array | None = None,
):
    """Full per-column pipeline: extract → setup → run (injected) → diagnose.

    ``record`` is a manifest :class:`~legoesm.training.column_manifest.ColumnRecord`
    (provides ``grid_index``, ``lat_deg``, ``environment.cape_J_kg``).  Returns the
    diagnosed closure-coefficient object.

    ``phis`` (optional, the model's STATIC surface geopotential ``g·z_s`` on the
    full grid, co-located with ``p_s``) activates the orographic geostrophic-forcing
    term for TERRAIN columns (iter 117); ``None`` (default) keeps the flat/ocean
    behaviour (geostrophic wind from the above-surface geopotential only).

    ``sst_K`` (the surface field, co-located with ``p_s``) is REQUIRED when
    ``config.surface_flux`` (iter 364) — it activates the ``prescribe="fluxes"`` surface
    BC for the spin-off LES; the column SST is gathered at ``record.grid_index``.  Fail
    LOUD if ``config.surface_flux`` is set but no ``sst_K`` is supplied (dispatch-hardening:
    a silently surface-flux-free LES would defeat the purpose).
    """
    if config.surface_flux and sst_K is None:
        raise ValueError(
            "process_column: config.surface_flux=True but no sst_K supplied — the "
            "prescribed-flux surface BC needs the column SST (make_les_diagnose_fn "
            "threads model_ctx.sst_K; a coupled/AMIP state must carry it).")
    lat_rad = float(jnp.deg2rad(record.lat_deg))
    gcm_z, gcm_theta, ls_state = extract_gcm_column(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=tuple(record.grid_index), lat_rad=lat_rad, phis=phis,
        sst_K=(sst_K if config.surface_flux else None),
    )
    setup = build_column_les_setup(
        cape_J_kg=record.environment.cape_J_kg, lat_rad=lat_rad,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls_state, config=config,
    )
    return run_column_les_pipeline(
        setup, run_les_fn, method=config.diagnosis_method,
        methods=config.diagnosis_methods, l_mix_max=config.clubb_l_mix_max,
        gate_realism=config.gate_les_realism,
        realism_wp2_floor=config.les_realism_wp2_floor,
        realism_theta_drift_K=config.les_realism_theta_drift_K,
        realism_q_v_max=config.les_realism_q_v_max,
        realism_rh_max=config.les_realism_rh_max,
        relax_width_frac=config.relax_width_frac,   # exclude the top sponge from the diagnosis
    )

