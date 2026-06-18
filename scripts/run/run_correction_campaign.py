"""Offline LES-informed correction CAMPAIGN driver — the HPC entry point.

Composes the full loop (``docs/COMPARE_REANALYSIS.md``) into a runnable campaign:

    run AMIP/CMIP → time-mean → compare to ERA5 → rank worst columns
      → (cluster) → LES-diagnose → correct a ``clubb_lite`` coefficient
      (``C_K`` or ``Pr_t``, by ``--diagnosis-method``) → RE-RUN → repeat.

Everything below the comparison is the already-tested training package; this
module supplies the two composition pieces that turn it into a runnable campaign:

* :func:`make_clubb_build_driver` — wraps a mode-specific ``build_base_driver``
  (AMIP ``ModelDriver`` / CMIP ``CoupledESMDriver``) so each run injects the
  corrected ``clubb_lite`` config via ``ExperimentConfig.turbulence_override``
  (iter 35) — the parameter-update mechanism.
* :func:`make_les_diagnose_fn` — turns the model column state (the loop's
  ``model_ctx``) into the LES closure coefficient via
  ``run_column_les.process_column`` (extract the GCM column forcing → spin off a
  real plane LES → diagnose ``K`` / ``w_e``).

:func:`build_correction_campaign` assembles ``make_run_fn`` ∘ ``make_compare_fn``
∘ the LES ``diagnose_fn`` and runs ``run_correction_campaign``; the ``main`` CLI
loads a base config + an ERA5 reference (reusing
``scripts/validate/compare_amip_era5.py``) and drives it on real data at scale.
The bias *sign* is an empirical result of the real run — this module only wires
the campaign.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.training.correction_loop import make_compare_fn, run_correction_campaign
from legoesm.training.run_to_column_mean import make_run_fn


def make_clubb_build_driver(
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """``build_driver(clubb_cfg)`` that injects ``clubb_cfg`` via ``turbulence_override``.

    ``base_atm_config`` is the base :class:`ExperimentConfig` (its ``turbulence``
    MUST already be ``"clubb_lite"`` — the campaign corrects ``clubb_lite.C_K``,
    it does NOT switch schemes; a non-clubb base config raises rather than being
    silently rewritten).  Each call sets only ``turbulence_override`` to the
    corrected :class:`CLUBBLiteConfig` and hands the result to
    ``build_base_driver`` (which constructs + ``setup()``-s the AMIP/CMIP driver —
    the run mode stays injected, so this is grid/mode-agnostic).
    """
    if getattr(base_atm_config, "turbulence", None) != "clubb_lite":
        raise ValueError(
            "make_clubb_build_driver: base_atm_config.turbulence must be "
            f"'clubb_lite' (the campaign corrects clubb_lite.C_K, it does not "
            f"switch schemes); got {getattr(base_atm_config, 'turbulence', None)!r}."
        )

    def build_driver(clubb_cfg: Any) -> Any:
        cfg = base_atm_config._replace(
            turbulence_override=TurbulenceConfig(
                scheme="clubb_lite", clubb_lite=clubb_cfg),
        )
        return build_base_driver(cfg)

    return build_driver


def make_les_diagnose_fn(
    grid: Any,
    sigma: Any,
    *,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
) -> Callable[[Any, Any], Any]:
    """``diagnose_fn(record, model_ctx)`` that spins off + diagnoses a column LES.

    ``model_ctx`` is the model :class:`ColumnState` the loop passes through (from
    :func:`make_compare_fn`); ``run_column_les.process_column`` extracts that
    column's GCM large-scale forcing, runs the plane LES (``run_les_fn``), and
    diagnoses the closure coefficient.  ``grid`` / ``sigma`` are the model grid +
    vertical coordinate the forcing extractor needs.
    """
    from legoesm.atmosphere.dynamics.column_les import process_column

    def diagnose_fn(record: Any, model_ctx: Any) -> Any:
        if getattr(model_ctx, "u", None) is None or getattr(model_ctx, "v", None) is None:
            raise ValueError(
                "make_les_diagnose_fn: model_ctx needs cell-centred u/v for the "
                "column-LES forcing extraction (an MPAS edge-velocity state is "
                "not supported)."
            )
        return process_column(
            record,
            T=model_ctx.T, q_v=model_ctx.q_v, u=model_ctx.u,
            v=model_ctx.v, p_s=model_ctx.p_s,
            grid=grid, sigma=sigma, config=les_config, run_les_fn=run_les_fn,
        )

    return diagnose_fn


# Coefficient name → (promotion_key, LES diagnosis method) for the multi campaign.
COEFFICIENT_SPEC_MAP = {
    "C_K": ("clubb_lite_C_K", "clubb_coefficient"),
    "Pr_t": ("clubb_lite_Pr_t", "prandtl_number"),
    "C_eps": ("clubb_lite_C_eps", "c_eps"),
}

# Single-coefficient LES diagnosis method → (promotion_key, CLUBBLiteConfig field).
# eddy_diffusivity (legacy, dimensional) targets C_K like clubb_coefficient.
METHOD_PROMOTION = {
    "clubb_coefficient": ("clubb_lite_C_K", "C_K"),
    "eddy_diffusivity": ("clubb_lite_C_K", "C_K"),
    "prandtl_number": ("clubb_lite_Pr_t", "Pr_t"),
    "c_eps": ("clubb_lite_C_eps", "C_eps"),
}
# Diagnosis methods that evaluate the GCM mixing length (need clubb_l_mix_max).
METHODS_NEED_LMIX = frozenset({"clubb_coefficient", "c_eps"})


def grid_latlon_deg(grid, lat_deg, lon_deg):
    """Default ``lat_deg``/``lon_deg`` to the grid's centre lat/lon in degrees."""
    import jax.numpy as jnp
    import numpy as np

    rad2deg = 180.0 / np.pi
    if lat_deg is None:
        lat_deg = jnp.asarray(np.asarray(grid.grid_lat) * rad2deg)
    if lon_deg is None:
        lon_deg = jnp.asarray(np.asarray(grid.grid_lon) * rad2deg)
    return lat_deg, lon_deg


def compose_compare_fn(*, base_atm_config, build_base_driver, extract_column_state,
                        reference, sigma, area_weights, n_worst,
                        lat_deg, lon_deg):
    """``compare_fn(config)`` = build clubb driver → run AMIP/CMIP → time-mean →
    compare to ``reference`` (shared by the single + multi build functions)."""
    import jax.numpy as jnp

    build_driver = make_clubb_build_driver(base_atm_config, build_base_driver)
    run_fn = make_run_fn(build_driver, extract_column_state)
    return make_compare_fn(
        reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=lat_deg, lon_deg=lon_deg, area_weights=area_weights,
        n_worst=n_worst, run_amip_fn=run_fn,
    )


def maybe_env_grid_fn(feedback_strategy, sigma):
    """The env-generalization ``env_grid_fn(model_ctx)`` for ``feedback_strategy=
    'environment'`` (``None`` for the static scatter)."""
    if feedback_strategy != "environment":
        return None
    from functools import partial as _partial

    from legoesm.training.feedback_assembly import column_environment_grid
    return _partial(column_environment_grid, sigma=sigma)


def build_correction_campaign(
    *,
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any],
    reference: Any,
    sigma: Any,
    grid: Any,
    area_weights: Any,
    n_iterations: int,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    n_worst: int,
    les_budget: int | None = None,
    env_scales: Any | None = None,
    initial_clubb: Any | None = None,
    lat_deg: Any | None = None,
    lon_deg: Any | None = None,
    initial_field: Any | None = None,
    start_round: int = 0,
    checkpoint_callback: Any | None = None,
    feedback_strategy: str = "static",
    accept_only_if_improved: bool = True,
    step_fractions: Any | None = None,
    clip_to_bounds: bool = True,
    bias_tol: float | None = None,
    patience: int = 2,
):
    """Assemble + run the LES-informed ``clubb_lite.C_K`` correction campaign.

    Composes :func:`make_clubb_build_driver` → :func:`make_run_fn` →
    :func:`make_compare_fn` (vs the regridded ERA5 ``reference``) → the LES
    :func:`make_les_diagnose_fn`, and drives
    :func:`~legoesm.training.correction_loop.run_correction_campaign` for
    ``n_iterations`` rounds (``les_budget`` caps the LES count by environment
    clustering).  ``grid_shape`` is taken from ``reference.T`` (the model grid).

    Restart (§1): pass ``initial_clubb`` + ``initial_field`` + ``start_round`` to
    RESUME a checkpointed campaign, and ``checkpoint_callback(round, result,
    field)`` to persist each round.  Returns the :class:`CampaignResult`.

    ``accept_only_if_improved`` (default true here — the bias-reduction campaign
    SHOULD be monotonic) keeps a round only if it lowered the global bias, so the
    accumulated ``clubb_lite.C_K`` field never regresses (the done-criterion).
    ``step_fractions`` (e.g. ``[1.0, 0.5, 0.25]``) enables the per-round line
    search over the correction magnitude toward the LES diagnosis (robust to the
    LES↔GCM overshoot); ``None`` ⇒ the full single step.  ``clip_to_bounds``
    (default true) clamps the diagnosed coefficient to its registered physical
    bounds so a degenerate LES cannot inject an unphysical value.  The promotable
    coefficient is selected by ``les_config.diagnosis_method``: ``prandtl_number``
    corrects ``clubb_lite_Pr_t`` (the dimensionless ``Pr_t = K_m/K_h``), otherwise
    ``clubb_lite_C_K``.
    """
    grid_shape = tuple(int(d) for d in reference.T.shape[:-1])
    lat_deg, lon_deg = grid_latlon_deg(grid, lat_deg, lon_deg)

    # Keep the loop's reduction in lock-step with the LES diagnosis, and select
    # the promotable coefficient + its production default by method:
    #   prandtl_number              → clubb_lite_Pr_t (K_m/K_h)
    #   clubb_coefficient / eddy_*  → clubb_lite_C_K
    clubb_cfg = initial_clubb if initial_clubb is not None else CLUBBLiteConfig()
    if getattr(les_config, "diagnosis_methods", None) is not None:
        # process_column would return a {method: diagnosis} dict, which the
        # single-coefficient loop cannot reduce. The simultaneous multi-coefficient
        # campaign (run_multi_correction_*) is a separate, not-yet-wired path.
        raise ValueError(
            "build_correction_campaign is single-coefficient: set "
            "les_config.diagnosis_method, not diagnosis_methods. For SIMULTANEOUS "
            "multi-coefficient correction use build_multi_correction_campaign.")
    diagnosis_method = les_config.diagnosis_method
    if diagnosis_method not in METHOD_PROMOTION:
        raise ValueError(
            f"unknown diagnosis_method {diagnosis_method!r}; choose from "
            f"{tuple(METHOD_PROMOTION)}.")
    promotion_key, field_name = METHOD_PROMOTION[diagnosis_method]
    background = float(getattr(CLUBBLiteConfig(), field_name))
    # clubb_coefficient (C_K) and c_eps evaluate the GCM mixing length; auto-
    # populate clubb_l_mix_max from the CLUBB config so the diagnosis matches it.
    if diagnosis_method in METHODS_NEED_LMIX and \
            getattr(les_config, "clubb_l_mix_max", None) is None:
        les_config = les_config._replace(
            clubb_l_mix_max=float(clubb_cfg.l_mix_max))

    compare_fn = compose_compare_fn(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, reference=reference, sigma=sigma,
        area_weights=area_weights, n_worst=n_worst,
        lat_deg=lat_deg, lon_deg=lon_deg,
    )
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn)
    env_grid_fn = maybe_env_grid_fn(feedback_strategy, sigma)

    return run_correction_campaign(
        initial_clubb if initial_clubb is not None else CLUBBLiteConfig(),
        int(n_iterations),
        compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key=promotion_key, grid_shape=grid_shape,
        diagnosis_method=diagnosis_method,
        background=background, les_budget=les_budget, env_scales=env_scales,
        initial_field=initial_field, start_round=start_round,
        checkpoint_callback=checkpoint_callback,
        feedback_strategy=feedback_strategy, env_grid_fn=env_grid_fn,
        accept_only_if_improved=accept_only_if_improved,
        step_fractions=step_fractions, clip_to_bounds=clip_to_bounds,
        bias_tol=bias_tol, patience=patience,
    )


def build_multi_correction_campaign(
    *,
    base_atm_config: Any,
    build_base_driver: Callable[[Any], Any],
    extract_column_state: Callable[..., Any],
    reference: Any,
    sigma: Any,
    grid: Any,
    area_weights: Any,
    n_iterations: int,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    n_worst: int,
    coefficients: tuple[str, ...] = ("C_K", "Pr_t"),
    les_budget: int | None = None,
    env_scales: Any | None = None,
    initial_clubb: Any | None = None,
    lat_deg: Any | None = None,
    lon_deg: Any | None = None,
    initial_fields: dict | None = None,
    start_round: int = 0,
    checkpoint_callback: Any | None = None,
    feedback_strategy: str = "static",
    accept_only_if_improved: bool = True,
    step_fractions: Any | None = None,
    clip_to_bounds: bool = True,
    sequential: bool = False,
    bias_tol: float | None = None,
    patience: int = 2,
):
    """Assemble + run the SIMULTANEOUS multi-coefficient correction campaign.

    Like :func:`build_correction_campaign` but corrects EVERY coefficient in
    ``coefficients`` (e.g. ``("C_K", "Pr_t")``) together from ONE LES run per
    column: it builds a :class:`~legoesm.training.correction_loop.CorrectionSpec`
    per coefficient, sets ``les_config.diagnosis_methods`` to their union so the
    spin-off LES is diagnosed for all of them at once, and drives
    :func:`~legoesm.training.correction_loop.run_multi_correction_campaign`.
    Auto-populates ``clubb_l_mix_max`` when ``"C_K"`` (the ``clubb_coefficient``
    diagnosis) is requested.  ``initial_fields`` (``{promotion_key: field}``)
    resumes the accumulated per-coefficient state.  Returns the
    :class:`~legoesm.training.correction_loop.MultiCampaignResult`.
    """
    from legoesm.training.correction_loop import (
        CorrectionSpec,
        run_multi_correction_campaign,
    )

    if not coefficients:
        raise ValueError("coefficients must be a non-empty tuple.")
    grid_shape = tuple(int(d) for d in reference.T.shape[:-1])
    lat_deg, lon_deg = grid_latlon_deg(grid, lat_deg, lon_deg)
    clubb_cfg = initial_clubb if initial_clubb is not None else CLUBBLiteConfig()

    specs = []
    for name in coefficients:
        if name not in COEFFICIENT_SPEC_MAP:
            raise ValueError(
                f"unknown coefficient {name!r}; choose from "
                f"{tuple(COEFFICIENT_SPEC_MAP)}.")
        key, method = COEFFICIENT_SPEC_MAP[name]
        specs.append(CorrectionSpec(key, method, float(getattr(CLUBBLiteConfig(), name))))

    # Diagnose every coefficient's method from ONE LES run (dedup, keep order).
    methods = tuple(dict.fromkeys(s.diagnosis_method for s in specs))
    les_config = les_config._replace(diagnosis_methods=methods)
    # clubb_coefficient (C_K) and c_eps both need the GCM mixing length.
    if {"clubb_coefficient", "c_eps"}.intersection(methods) and \
            getattr(les_config, "clubb_l_mix_max", None) is None:
        les_config = les_config._replace(clubb_l_mix_max=float(clubb_cfg.l_mix_max))

    compare_fn = compose_compare_fn(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, reference=reference, sigma=sigma,
        area_weights=area_weights, n_worst=n_worst,
        lat_deg=lat_deg, lon_deg=lon_deg,
    )
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn)
    env_grid_fn = maybe_env_grid_fn(feedback_strategy, sigma)

    return run_multi_correction_campaign(
        clubb_cfg, int(n_iterations), specs,
        compare_fn=compare_fn, diagnose_fn=diagnose_fn, grid_shape=grid_shape,
        les_budget=les_budget, env_scales=env_scales,
        feedback_strategy=feedback_strategy, env_grid_fn=env_grid_fn,
        accept_only_if_improved=accept_only_if_improved,
        step_fractions=step_fractions, clip_to_bounds=clip_to_bounds,
        sequential=sequential, bias_tol=bias_tol, patience=patience,
        initial_fields=initial_fields, start_round=start_round,
        checkpoint_callback=checkpoint_callback,
    )


def make_base_driver_builder(
    mode: str, *, coupled_preset: Any = None, ocean_grid: Any = None
):
    """Return ``(build_base_driver, extract_column_state)`` for the run ``mode``.

    * ``"amip"`` → a :class:`ModelDriver` (prescribed SST) + ``amip_column_state``.
    * ``"cmip"`` → a :class:`CoupledESMDriver` (interactive ocean; ``coupled_preset``
      required) + ``cmip_column_state``.  ``ocean_grid=None`` (the default) makes
      the coupled driver use its OWN atmosphere grid for the ocean — same-grid
      coupling with an identity remap (and the ocean SST lands on the atm column
      shape).  Pass an explicit ``ocean_grid`` only for a genuinely different
      ocean grid.

    Each ``build_base_driver(cfg)`` constructs + ``setup()``-s the driver.  Raises
    on an unknown mode (dispatch hardening) or a ``"cmip"`` call missing
    ``coupled_preset`` — so a typo selects nothing silently.
    """
    from legoesm.training.run_to_column_mean import (
        amip_column_state,
        cmip_column_state,
    )

    if mode == "amip":
        from legoesm.driver.model_driver import ModelDriver

        def build_amip(cfg: Any) -> Any:
            driver = ModelDriver(cfg)
            driver.setup()
            return driver

        return build_amip, amip_column_state

    if mode == "cmip":
        if coupled_preset is None:
            raise ValueError(
                "make_base_driver_builder: mode='cmip' requires coupled_preset."
            )
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        def build_cmip(cfg: Any) -> Any:
            # ocean_grid=None ⇒ the coupled driver uses its own atm grid (same
            # object ⇒ identity remap; ocean SST on the atm column shape).
            driver = CoupledESMDriver(cfg, coupled_preset, ocean_grid=ocean_grid)
            driver.setup()
            return driver

        return build_cmip, cmip_column_state

    raise ValueError(
        f"make_base_driver_builder: unknown mode {mode!r}; choose 'amip' or 'cmip'."
    )


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="base ExperimentConfig JSON")
    p.add_argument("--mode", choices=("amip", "cmip"), default="amip",
                   help="AMIP (prescribed SST) or CMIP (coupled ocean)")
    p.add_argument("--coupled-preset", default="aquaplanet",
                   help="coupled_config preset name (CMIP mode)")
    p.add_argument("--era5-zarr", required=True, help="ERA5 zarr (reference)")
    p.add_argument("--era5-cache", default=None, help="ERA5 local cache dir")
    p.add_argument("--era5-time-idx", type=int, default=0)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--bias-tol", type=float, default=None,
                   help="convergence tolerance: stop early once --patience "
                        "consecutive rounds improve the global bias by less than "
                        "this (or are rejected). Default: run all --iterations.")
    p.add_argument("--patience", type=int, default=2,
                   help="rounds of no-progress before --bias-tol early-stops")
    p.add_argument("--n-worst", type=int, default=20)
    p.add_argument("--les-budget", type=int, default=None,
                   help="cap LES to K env-cluster representatives (default: all)")
    p.add_argument("--feedback-strategy", choices=("static", "environment"),
                   default="static",
                   help="static: correct only the worst columns; environment: "
                        "generalize the diagnoses to all env-similar columns")
    p.add_argument("--keep-worsening-rounds", action="store_true",
                   help="accumulate every round unconditionally (default: keep a "
                        "round only if it lowered the global bias — monotonic)")
    p.add_argument("--step-fractions", default=None,
                   help="comma-separated line-search step fractions in (0,1] "
                        "(e.g. '1.0,0.5,0.25'): backtrack the correction magnitude "
                        "toward the LES diagnosis, keeping the largest improving "
                        "step (default: full single step)")
    p.add_argument("--coefficients", default=None,
                   help="comma-separated coefficients to correct from one LES run "
                        "(e.g. 'C_eps,C_K,Pr_t'): routes to the multi-coefficient "
                        "campaign. Omit for the single-coefficient "
                        "--diagnosis-method path.")
    p.add_argument("--staged", action="store_true",
                   help="multi-coefficient: correct the coefficients IN ORDER, each "
                        "with its own line search + gate (block coordinate descent), "
                        "instead of one shared step. Recommended order C_eps,C_K,Pr_t "
                        "(let wp2 converge before C_K). Default: one combined step.")
    p.add_argument("--diagnosis-method",
                   choices=("clubb_coefficient", "prandtl_number", "c_eps",
                            "eddy_diffusivity"),
                   default="clubb_coefficient",
                   help="single-coefficient LES diagnosis: clubb_coefficient "
                        "(default) → C_K = K_m/(l*sqrt(wp2)); prandtl_number → "
                        "Pr_t = K_m/K_h; c_eps → C_eps = P*l/wp2^1.5 (the wp2-"
                        "dissipation coefficient); eddy_diffusivity → dimensional "
                        "heat K [m^2/s] (legacy). For SEVERAL at once use "
                        "--coefficients.")
    p.add_argument("--allow-unphysical-coeff", action="store_true",
                   help="do NOT clamp the diagnosed C_K to its registered physical "
                        "bounds (default: clamp, so a degenerate LES cannot inject "
                        "an out-of-range / destabilizing coefficient)")
    p.add_argument("--les-dt", type=float, default=1.0, help="LES timestep [s]")
    p.add_argument("--les-hours", type=float, default=2.0, help="LES duration [h]")
    p.add_argument("--out", default="corrected_clubb_config.json")
    p.add_argument("--checkpoint", default=None,
                   help="write a per-round checkpoint JSON (restartable campaign)")
    p.add_argument("--resume", default=None,
                   help="resume from a --checkpoint JSON (continues the accumulation)")
    return p


def _summary_to_json(summary):  # pragma: no cover - HPC path
    """JSON-serializable form of a :class:`CampaignSummary` for the output file."""
    return {
        "n_rounds": summary.n_rounds, "n_accepted": summary.n_accepted,
        "acceptance_rate": summary.acceptance_rate, "stop_reason": summary.stop_reason,
        "initial_bias": summary.initial_bias, "final_bias": summary.final_bias,
        "absolute_reduction": summary.absolute_reduction,
        "fractional_reduction": summary.fractional_reduction,
        "coefficients": [
            {"promotion_key": c.promotion_key, "n_columns": c.n_columns,
             "field_min": c.field_min, "field_max": c.field_max,
             "field_mean": c.field_mean, "field_std": c.field_std,
             "n_at_lower_bound": c.n_at_lower_bound,
             "n_at_upper_bound": c.n_at_upper_bound,
             "bounds": list(c.bounds) if c.bounds is not None else None}
            for c in summary.coefficients],
    }


def _area_weights(grid):  # pragma: no cover - HPC path
    """Per-column quadrature weights for the bias aggregation.

    Prefers the grid's true cell areas (``grid_area`` — incl. Gaussian quadrature
    weights — or ``area``); falls back to cos-latitude (a lat-lon proxy) with a
    warning if the grid exposes neither.
    """
    import warnings

    import jax.numpy as jnp
    import numpy as np

    for attr in ("grid_area", "area"):
        a = getattr(grid, attr, None)
        if a is not None:
            return jnp.asarray(a)
    warnings.warn(
        "run_correction_campaign: grid exposes no cell-area weights; using "
        "cos-latitude (a lat-lon proxy) for the bias aggregation.", stacklevel=2)
    return jnp.cos(jnp.deg2rad(jnp.asarray(np.asarray(grid.grid_lat))))


def _run_multi_main(args, base_cfg, grid, sigma, reference, build_base_driver,
                    extract_fn, run_les):  # pragma: no cover - heavy I/O
    """Multi-coefficient campaign entry (``--coefficients``): a dict checkpoint /
    resume / output for the per-coefficient accumulated fields."""
    import json

    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    coefficients = tuple(c.strip() for c in args.coefficients.split(","))
    # promotion_key -> config field name (the coefficient name IS the field name).
    promo_to_field = {
        COEFFICIENT_SPEC_MAP[c][0]: c
        for c in coefficients if c in COEFFICIENT_SPEC_MAP
    }
    gshape = grid.grid_shape_2d
    initial_clubb, initial_fields, start_round = None, None, 0
    if args.resume:
        with open(args.resume) as f:
            ckpt = json.load(f)
        if sorted(ckpt.get("coefficients", [])) != sorted(coefficients):
            raise SystemExit(
                f"checkpoint coefficients {ckpt.get('coefficients')} != requested "
                f"{list(coefficients)}; resume with the matching set.")
        initial_fields = {
            k: jnp.asarray(v).reshape(gshape) for k, v in ckpt["fields"].items()}
        # Rebuild the config FROM the accumulated fields (single source of truth).
        overrides = {
            promo_to_field[k]: jnp.asarray(v).reshape(-1)
            for k, v in ckpt["fields"].items()}
        initial_clubb = CLUBBLiteConfig(**overrides)
        start_round = int(ckpt["round"]) + 1
        print(f"[campaign] resuming multi from {args.resume} at round {start_round}")

    checkpoint_callback = None
    if args.checkpoint:
        def checkpoint_callback(round_idx, res, fields):  # noqa: ARG001
            with open(args.checkpoint, "w") as f:
                json.dump({"round": int(round_idx),
                           "coefficients": list(coefficients),
                           "fields": {k: np.asarray(v).reshape(-1).tolist()
                                      for k, v in fields.items()}}, f)

    result = build_multi_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma, grid=grid,
        area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(), n_worst=args.n_worst, coefficients=coefficients,
        les_budget=args.les_budget, run_les_fn=run_les,
        initial_clubb=initial_clubb, initial_fields=initial_fields,
        start_round=start_round, checkpoint_callback=checkpoint_callback,
        feedback_strategy=args.feedback_strategy,
        accept_only_if_improved=not args.keep_worsening_rounds,
        step_fractions=([float(s) for s in args.step_fractions.split(",")]
                        if args.step_fractions else None),
        clip_to_bounds=not args.allow_unphysical_coeff, sequential=args.staged,
        bias_tol=args.bias_tol, patience=args.patience)

    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    for i, (b0, b1, imp) in enumerate(biases):
        kept = accepted[i] if i < len(accepted) else True
        print(f"[campaign] round {start_round + i}: bias {b0:.5g} -> {b1:.5g} "
              f"(step {steps[i]:.3g}; {'IMPROVED' if imp else 'no improvement'}; "
              f"{'kept' if kept else 'REJECTED'})")
    from legoesm.training.campaign_summary import (
        campaign_health,
        summarize_campaign,
    )
    summary = summarize_campaign(result)
    health = campaign_health(summary)
    with open(args.out, "w") as f:
        json.dump({"coefficients": list(coefficients),
                   "fields": {k: np.asarray(v).reshape(-1).tolist()
                              for k, v in result.final_fields.items()},
                   "grid": _grid_provenance(base_cfg, grid),
                   "biases": biases, "accepted": accepted,
                   "step_fractions": steps,
                   "summary": _summary_to_json(summary),
                   "health": {"status": health.status, "message": health.message}},
                  f, indent=2)
    print(f"[campaign] wrote corrected multi-coefficient config to {args.out}")
    print(summary.report())
    print(f"[campaign] {health.status.upper()}: {health.message}")
    _print_deploy_hint(args.out, grid)
    return 0


def _grid_provenance(base_cfg, grid) -> dict:
    """Adapter-order grid fingerprint + human-readable grid identity for the output.

    Recorded so a deploy onto a DIFFERENT grid (even one with the same column
    count) is caught by :func:`deploy_correction.assert_deploy_compatible` instead
    of silently landing the per-column coefficients on the wrong cells.
    """
    from legoesm.training.deploy_correction import grid_fingerprint

    prov = grid_fingerprint(grid)
    gc = base_cfg.grid
    prov.update(grid_type=gc.grid_type, resolution=gc.resolution, nlev=gc.nlev)
    return prov


def _print_deploy_hint(out_path: str, grid) -> None:
    """Verify the just-written output deploys onto its OWN grid + print the hint.

    Round-trips the campaign JSON through the production deploy loader WITH the
    grid the output was made for, so a non-deployable or grid-inconsistent output
    fails LOUDLY here (at write time) rather than silently in a downstream run.
    """
    from legoesm.training.deploy_correction import corrected_turbulence_override

    corrected_turbulence_override(out_path, grid=grid)  # raises if not deployable
    print(
        "[campaign] deploy into a production run with: "
        "ExperimentConfig(..., turbulence='clubb_lite', "
        f"turbulence_override=corrected_turbulence_override({out_path!r}, grid=grid))"
    )


def load_base_config_and_grid(config_path: str):
    """Load the base ``ExperimentConfig`` + build its grid / vertical coordinate.

    The grid + vertical coordinate are built the EXACT way the run builds them (so
    the ERA5 regrid + LES forcing extraction match the run's grid + vertical coord,
    hybrid vs sigma), but WITHOUT the heavy ``setup()`` side effects — the campaign
    is single-rank offline orchestration, so the driver's own grid constructor
    suffices.  ``_bootstrap_runtime`` runs first so the precision policy is applied
    (else a non-default-precision sigma would get the wrong dtype).  Shared by the
    campaign CLI and the perfect-model OSSE CLI.
    """
    import json

    from legoesm.driver.config import experiment_config_from_dict
    from legoesm.driver.model_driver import ModelDriver

    with open(config_path) as f:
        base_cfg = experiment_config_from_dict(json.load(f))
    probe = ModelDriver(base_cfg)
    probe._bootstrap_runtime()
    probe._create_grid()
    return base_cfg, probe.grid, probe.sigma


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy I/O
    """CLI: load a base config + an ERA5 reference, run the campaign, write the
    corrected ``clubb_lite`` config + a per-round bias report.

    ``--mode amip`` (prescribed SST) or ``--mode cmip`` (coupled ocean on the same
    grid, ``--coupled-preset``) — the mode selects the driver + SST source via
    :func:`make_base_driver_builder`.  Heavy I/O (ERA5, the driver, the LES) is
    imported lazily here so the composition helpers above stay importable +
    unit-testable without it.  Reuses the ERA5→model-grid regrid of
    ``scripts/validate/compare_amip_era5.py``.

    Advanced :func:`build_correction_campaign` knobs (``env_scales`` for the
    clustering metric, ``initial_clubb`` to warm-start from a saved corrected
    config) are exposed only via that API, not this CLI; ``--les-budget`` is the
    one clustering knob surfaced here.
    """
    import json

    import numpy as np
    from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig, run_forced_les
    from legoesm.training.compare_reanalysis import column_state_from_carry
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice

    from scripts.validate.compare_amip_era5 import (
        canonical_grid_type,
        select_era5_regrid,
    )

    args = _build_arg_parser().parse_args(argv)
    base_cfg, grid, sigma = load_base_config_and_grid(args.config)

    # Run mode: AMIP (prescribed SST) or CMIP (coupled ocean on the SAME grid as
    # the atmosphere — ocean_grid=None lets the coupled driver use its own atm
    # grid, an identity remap, so the ocean SST lands on the atm column shape).
    coupled_preset = None
    if args.mode == "cmip":
        from legoesm.driver.coupled_config import PRESETS
        if args.coupled_preset not in PRESETS:
            raise SystemExit(
                f"unknown --coupled-preset {args.coupled_preset!r}; "
                f"choose from {sorted(PRESETS)}")
        coupled_preset = PRESETS[args.coupled_preset]()
    build_base_driver, extract_fn = make_base_driver_builder(
        args.mode, coupled_preset=coupled_preset, ocean_grid=None)

    # ERA5 reference regridded to the model grid + sigma (same regrid as the
    # one-shot compare driver).
    canon = canonical_grid_type(base_cfg.grid.grid_type)
    era5_slice = load_era5_slice(
        TrainingERA5Config(zarr_store=args.era5_zarr, local_cache_dir=args.era5_cache),
        args.era5_time_idx)
    reference = column_state_from_carry(select_era5_regrid(canon)(era5_slice, grid, sigma))

    import jax.numpy as jnp
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    n_steps = int(args.les_hours * 3600.0 / args.les_dt)
    run_les = partial(run_forced_les, dt_s=args.les_dt, n_steps=n_steps)
    if args.coefficients is not None:
        return _run_multi_main(
            args, base_cfg, grid, sigma, reference, build_base_driver, extract_fn,
            run_les)

    # Restart: resume from a checkpoint (corrected config + accumulated field +
    # round), so a multi-day campaign survives a job timeout (§1 restartable).
    # The CLUBB field the diagnosis corrects (C_K / Pr_t / C_eps) — so the
    # checkpoint/output persist the right field.
    corrected_field = METHOD_PROMOTION[args.diagnosis_method][1]
    initial_clubb, initial_field, start_round = None, None, 0
    if args.resume:
        with open(args.resume) as f:
            ckpt = json.load(f)
        # Guard against resuming a checkpoint for a DIFFERENT corrected coefficient
        # (e.g. a Pr_t checkpoint with --diagnosis-method=clubb_coefficient) — the
        # field would be silently loaded into the wrong config slot.
        ckpt_field = ckpt.get("corrected_field")
        if ckpt_field is not None and ckpt_field != corrected_field:
            raise SystemExit(
                f"checkpoint corrects {ckpt_field!r} but --diagnosis-method "
                f"requests {corrected_field!r}; resume with the matching method.")
        initial_field = jnp.asarray(ckpt["field"]).reshape(grid.grid_shape_2d)
        # The accumulated FIELD is the single source of truth for the accepted
        # state; the per-column coefficient is exactly its flattened form (the
        # column-ordering contract). Rebuilding the config FROM the field cannot
        # desync from initial_field, even after a rejected-round checkpoint.
        initial_clubb = CLUBBLiteConfig(
            **{corrected_field: initial_field.reshape(-1)})
        start_round = int(ckpt["round"]) + 1
        print(f"[campaign] resuming from {args.resume} at round {start_round}")

    checkpoint_callback = None
    if args.checkpoint:
        def checkpoint_callback(round_idx, res, field):  # noqa: ARG001
            # Persist the ACCEPTED accumulated `field` (post-gate base), NOT
            # res.updated_config — under the monotonic gate a rejected round's
            # updated_config is discarded while `field` stays the accepted state.
            # The coefficient is stored under its real name (C_K or Pr_t) for
            # inspection + the resume-method guard; resume reconstructs from `field`.
            flat = np.asarray(field).reshape(-1)
            with open(args.checkpoint, "w") as f:
                json.dump({"round": int(round_idx),
                           "corrected_field": corrected_field,
                           corrected_field: flat.tolist(),
                           "field": np.asarray(field).tolist()}, f)

    result = build_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma,
        grid=grid, area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(diagnosis_method=args.diagnosis_method),
        n_worst=args.n_worst,
        les_budget=args.les_budget,
        run_les_fn=run_les,
        initial_clubb=initial_clubb, initial_field=initial_field,
        start_round=start_round, checkpoint_callback=checkpoint_callback,
        feedback_strategy=args.feedback_strategy,
        accept_only_if_improved=not args.keep_worsening_rounds,
        step_fractions=(
            [float(s) for s in args.step_fractions.split(",")]
            if args.step_fractions else None),
        clip_to_bounds=not args.allow_unphysical_coeff,
        bias_tol=args.bias_tol, patience=args.patience,
    )
    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    for i, (b0, b1, imp) in enumerate(biases):
        kept = accepted[i] if i < len(accepted) else True
        print(f"[campaign] round {start_round + i}: bias {b0:.5g} -> {b1:.5g} "
              f"(step {steps[i]:.3g}; {'IMPROVED' if imp else 'no improvement'}; "
              f"{'kept' if kept else 'REJECTED'})")
    from legoesm.training.campaign_summary import (
        campaign_health,
        summarize_campaign,
    )
    promotion_key = METHOD_PROMOTION[args.diagnosis_method][0]
    summary = summarize_campaign(result, promotion_key=promotion_key)
    health = campaign_health(summary)
    with open(args.out, "w") as f:
        json.dump({corrected_field: np.asarray(
                       getattr(result.final_config, corrected_field)).tolist(),
                   "grid": _grid_provenance(base_cfg, grid),
                   "biases": biases, "accepted": accepted,
                   "step_fractions": steps,
                   "summary": _summary_to_json(summary),
                   "health": {"status": health.status, "message": health.message}},
                  f, indent=2)
    print(f"[campaign] wrote corrected clubb config to {args.out}")
    print(summary.report())
    print(f"[campaign] {health.status.upper()}: {health.message}")
    _print_deploy_hint(args.out, grid)
    _maybe_write_env_kernel(args, result)
    return 0


def _maybe_write_env_kernel(args, result):  # pragma: no cover - HPC path
    """Export the RAW environment kernel (the grid-AGNOSTIC cross-resolution deploy
    artifact) when the campaign used ``--feedback-strategy environment``.

    Writes ``<out>.env_kernel.json`` ONLY when a kernel is present (a CLUBB
    environment-strategy round produced one); static / non-CLUBB campaigns skip it
    silently — those have no env→coefficient regression to transfer.  Exports the
    last ACCEPTED round's kernel (under the monotonic gate a REJECTED final round's
    kernel is inconsistent with the accepted per-column ``--out`` it would sit
    beside, so it must NOT be the exported artifact).  This is the PRODUCER half of
    the iter-69 deploy library: the saved JSON deploys on a DIFFERENT-resolution
    grid via :func:`deploy_correction.apply_env_kernel_override`.
    """
    import json

    from legoesm.training.correction_loop import last_accepted_env_kernel
    from legoesm.training.deploy_correction import env_kernel_to_dict

    # The last ACCEPTED round's kernel (run_correction_campaign fills `accepted`
    # every round; a rejected final round's kernel must not sit beside the
    # accepted per-column --out). None ⇒ static / non-CLUBB / no-op campaign.
    kernel = last_accepted_env_kernel(result)
    if kernel is None:
        return
    out = f"{args.out}.env_kernel.json"
    with open(out, "w") as f:
        json.dump(env_kernel_to_dict(kernel), f, indent=2)
    print(f"[campaign] wrote RAW environment kernel (cross-resolution deploy) to {out}")
    print(
        "[campaign] deploy on ANY grid with: apply_env_kernel_override("
        "env_kernel_from_dict(json.load(open(...))), "
        "column_environment_grid(model, sigma, env_config=..., p_full=..., "
        "p_half=...)[0])"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
