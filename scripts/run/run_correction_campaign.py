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


def resolve_orographic_phis(orographic_mode: str, phis_provider: Callable[[], Any]) -> Any:
    """Resolve the orographic LES-forcing topography ``phis`` from the
    ``--orographic-forcing`` mode + a deferred ``phis_provider``.

    * ``"off"`` → ``None`` (flat; legacy behaviour).  ``phis_provider`` is NOT called.
    * ``"auto"`` → the model's OWN static ``phis`` (via the provider), mapped to
      ``None`` when the model is flat (:func:`...run_to_column_mean.phis_or_none_if_flat`)
      — so a flat/aquaplanet model stays flat and passing ``auto`` is always safe.
    * ``"on"`` → as ``auto`` but FAILS LOUD (``SystemExit``) if the model is flat —
      the user explicitly requested terrain forcing, so a silent flat run (which
      would hide the missing orography) is refused.

    ``phis_provider`` is a zero-arg callable that BUILDS the model's static
    topography (e.g. ``ModelDriver(base_cfg).static_topography_phis()`` — no
    filesystem writes); deferring it means ``"off"`` constructs nothing.  Unknown
    mode raises ``ValueError`` (dispatch hardening; ``argparse choices=`` already
    constrains the CLI, so this is defense-in-depth).
    """
    if orographic_mode == "off":
        return None
    if orographic_mode not in ("auto", "on"):
        raise ValueError(
            f"unknown orographic_forcing mode {orographic_mode!r}; choose "
            "auto/on/off.")
    from legoesm.training.run_to_column_mean import phis_or_none_if_flat
    phis = phis_or_none_if_flat(phis_provider())
    if orographic_mode == "on" and phis is None:
        raise SystemExit(
            "--orographic-forcing=on requires model topography, but the model "
            "exposes none (flat/aquaplanet: phis is identically zero). Use "
            "--orographic-forcing=auto for a flat model, or supply a base config "
            "with real topography.")
    return phis


def make_les_diagnose_fn(
    grid: Any,
    sigma: Any,
    *,
    les_config: Any,
    run_les_fn: Callable[[Any], Any],
    phis: Any | None = None,
) -> Callable[[Any, Any], Any]:
    """``diagnose_fn(record, model_ctx)`` that spins off + diagnoses a column LES.

    ``model_ctx`` is the model :class:`ColumnState` the loop passes through (from
    :func:`make_compare_fn`); ``run_column_les.process_column`` extracts that
    column's GCM large-scale forcing, runs the plane LES (``run_les_fn``), and
    diagnoses the closure coefficient.  ``grid`` / ``sigma`` are the model grid +
    vertical coordinate the forcing extractor needs.

    ``phis`` (optional, the model's STATIC surface geopotential ``g·z_s`` on the
    full grid, co-located with the state's ``p_s``) activates the orographic
    geostrophic-forcing term for TERRAIN worst-columns (iter 117); ``None``
    (default) keeps the flat/ocean behaviour.  It is a static topography field
    (closed over here), NOT carried on the per-step ``model_ctx``.  For a TERRAIN
    campaign, derive the CONSISTENT field from the model with
    :func:`legoesm.training.run_to_column_mean.model_phis_from_driver` (it returns
    ``None`` for a flat model, so passing it is always safe) rather than a separate
    file that could mismatch the run's topography.
    """
    from legoesm.atmosphere.dynamics.column_les import process_column

    def diagnose_fn(record: Any, model_ctx: Any) -> Any:
        u_edge = getattr(model_ctx, "u_edge", None)
        if u_edge is not None:
            # MPAS: the comparison state carries the NATIVE edge-normal velocity;
            # the Voronoi forcing extractor takes u=u_edge, v=None (iter 73).  The
            # model grid MUST be the VoronoiMesh — guard against an accidental
            # u_edge on a non-MPAS model_ctx (Codex) before the dispatch.
            from legoesm.grids.voronoi import VoronoiMesh
            if not isinstance(grid, VoronoiMesh):
                raise ValueError(
                    "make_les_diagnose_fn: model_ctx carries u_edge (an MPAS edge "
                    f"velocity) but grid is {type(grid).__name__}, not a VoronoiMesh "
                    "— grid/state mismatch."
                )
            return process_column(
                record,
                T=model_ctx.T, q_v=model_ctx.q_v, u=u_edge, v=None,
                p_s=model_ctx.p_s,
                grid=grid, sigma=sigma, config=les_config, run_les_fn=run_les_fn,
                phis=phis,
            )
        if getattr(model_ctx, "u", None) is None or getattr(model_ctx, "v", None) is None:
            raise ValueError(
                "make_les_diagnose_fn: model_ctx needs cell-centred u/v for the "
                "column-LES forcing extraction (a cell-wind grid with missing "
                "winds; an MPAS state must carry u_edge)."
            )
        return process_column(
            record,
            T=model_ctx.T, q_v=model_ctx.q_v, u=model_ctx.u,
            v=model_ctx.v, p_s=model_ctx.p_s,
            grid=grid, sigma=sigma, config=les_config, run_les_fn=run_les_fn,
            phis=phis,
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
                        lat_deg, lon_deg, valid_mask=None, manifest_reducer=None):
    """``compare_fn(config)`` = build clubb driver → run AMIP/CMIP → time-mean →
    compare to ``reference`` (shared by the single + multi build functions).

    ``valid_mask`` (optional, column-shaped) restricts the worst-column ranking —
    e.g. an ocean/land mask, or (DISTRIBUTED MPAS) the rank's owned-cell mask from
    :func:`legoesm.training.compare_reanalysis.owned_cell_valid_mask` so a halo cell
    is not ranked + corrected on a rank that does not own it.  ``manifest_reducer``
    (optional) post-processes the worst-column manifest — under distributed MPAS it
    is the cross-rank global top-k
    (:func:`legoesm.training.distributed_manifest.gather_global_worst_columns`) so
    the GLOBAL ``n_worst`` worst cells are diagnosed, not ``R × n_worst``."""
    import jax.numpy as jnp

    build_driver = make_clubb_build_driver(base_atm_config, build_base_driver)
    run_fn = make_run_fn(build_driver, extract_column_state)
    return make_compare_fn(
        reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=lat_deg, lon_deg=lon_deg, area_weights=area_weights,
        n_worst=n_worst, run_amip_fn=run_fn, valid_mask=valid_mask,
        manifest_reducer=manifest_reducer,
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
    phis: Any | None = None,
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
    stop_on_no_valid_diagnoses: bool = True,
    valid_mask: Any | None = None,
    manifest_reducer: Any | None = None,
    global_reduce: Any | None = None,
    validate_reference: bool = True,
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
    if validate_reference:                           # fail-fast on a units/sign error
        from legoesm.training.compare_reanalysis import validate_reference_physical
        validate_reference_physical(reference, name="reference")
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
        lat_deg=lat_deg, lon_deg=lon_deg, valid_mask=valid_mask,
        manifest_reducer=manifest_reducer,
    )
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn, phis=phis)
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
        stop_on_no_valid_diagnoses=stop_on_no_valid_diagnoses,
        global_reduce=global_reduce,
    )


def _distributed_campaign_kwargs(layout, reference, area_weights, n_worst, base_valid_mask):
    """The distributed-MPAS hook + rank-local-slice kwargs shared by the single AND
    multi distributed wrappers — the ONE place the three hooks (owned ``valid_mask``,
    global top-k ``manifest_reducer``, collective ``global_reduce``) are composed
    from the partition ``layout`` (so a caller cannot set them inconsistently) and
    the GLOBAL ``reference`` + ``area_weights`` are sliced to the rank's local cells
    (``layout.partition.local_cells``).  Returns the kwargs both ``build_*`` campaign
    builders accept identically.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.training.compare_reanalysis import validate_reference_physical
    from legoesm.training.distributed_campaign import (
        distributed_campaign_hooks,
        slice_reference_to_local,
    )

    valid_mask, manifest_reducer, global_reduce = distributed_campaign_hooks(
        layout, n_worst, base_valid_mask=base_valid_mask)
    local_cells = np.asarray(layout.partition.local_cells)
    n_global = int(layout.partition.nCells_global)   # authoritative GLOBAL cell count
    area_np = np.asarray(area_weights)
    # FAIL FAST on a GLOBAL/mesh cell-count mismatch (same silent-clamp hazard as the
    # reference slice): a wrong-length GLOBAL area_weights would make the JAX gather
    # ``area_weights[local_cells]`` clamp overflowing ids to the last weight, silently
    # mis-weighting the rank's columns. EXACT-equality vs nCells_global also rejects a
    # too-LONG array (a too-short array alone is caught by a bounds check). 1-D per-cell.
    if area_np.ndim != 1:
        raise ValueError(
            "_distributed_campaign_kwargs: area_weights must be 1-D per-cell "
            f"(got shape {area_np.shape}).")
    if area_np.shape[0] != n_global:
        raise ValueError(
            f"_distributed_campaign_kwargs: area_weights has {area_np.shape[0]} cells "
            f"but the partitioned global mesh has {n_global} — the GLOBAL area_weights "
            "must be defined on the SAME mesh as the model run.")
    if local_cells.size:
        lo, hi = int(local_cells.min()), int(local_cells.max())
        if lo < 0 or hi >= n_global:
            raise ValueError(
                "_distributed_campaign_kwargs: local_cells index out of range "
                f"[{lo}, {hi}] for a global mesh of {n_global} cells.")
    # COLLECTIVE-SAFE reference validation, in TWO ordered steps:
    #  (1) a DETERMINISTIC shape check (reference cell count vs the global mesh) — this
    #      runs FIRST so a mis-passed rank-LOCAL reference fails identically on every
    #      rank (all raise) before the value-dependent check below;
    #  (2) the physical (units/sign) check on the GLOBAL reference — identical on every
    #      rank (same global reference) ⇒ all ranks raise or none, so no rank can pass
    #      through while another raises and deadlock the campaign loop's collectives.
    # The inner per-SLICE re-validation is disabled (``validate_reference=False`` below)
    # so a partial defect cannot raise on only the owning ranks (iter 98 lesson).
    ref_n = int(np.asarray(reference.T).shape[0])    # T is a required cell field
    if ref_n != n_global:
        raise ValueError(
            f"_distributed_campaign_kwargs: the GLOBAL reference has {ref_n} cells but "
            f"the partitioned mesh has {n_global} — pass the GLOBAL reference (not a "
            "rank-local slice); it must be defined on the SAME mesh as the model run.")
    validate_reference_physical(reference, name="global reference")
    return dict(
        reference=slice_reference_to_local(
            reference, local_cells, expected_n_cells=n_global),
        area_weights=jnp.asarray(area_np)[jnp.asarray(local_cells)],
        n_worst=n_worst, valid_mask=valid_mask,
        manifest_reducer=manifest_reducer, global_reduce=global_reduce,
        validate_reference=False,                    # global ref already validated above
    )


def build_distributed_correction_campaign(
    *, layout: Any, reference: Any, area_weights: Any, n_worst: int,
    base_valid_mask: Any = None, **campaign_kwargs: Any,
):
    """Run :func:`build_correction_campaign` on a DISTRIBUTED-MPAS partition.

    The distributed-MPAS entry point that COMPOSES iters 86–88 into one call: from
    the rank's partition ``layout`` it builds the three hooks (owned ``valid_mask``,
    global top-k ``manifest_reducer``, collective ``global_reduce``) via
    :func:`legoesm.training.distributed_campaign.distributed_campaign_hooks`, and
    slices BOTH the GLOBAL ERA5 ``reference`` (a :class:`ColumnState`) AND the
    GLOBAL ``area_weights`` down to the rank's local cells
    (``layout.partition.local_cells``) so they align with the rank-local model
    state, then forwards everything to :func:`build_correction_campaign`.

    ``campaign_kwargs`` are the usual campaign args (``base_atm_config``,
    ``build_base_driver``, ``extract_column_state``, ``sigma``, ``grid`` = the
    rank-local mesh, ``n_iterations``, ``les_config``, ``run_les_fn``, …).  They MUST
    NOT include ``valid_mask`` / ``manifest_reducer`` / ``global_reduce`` /
    ``reference`` / ``area_weights`` / ``n_worst`` — this function supplies them; a
    duplicate is a Python ``TypeError`` (it is the ONE place those distributed hooks
    are wired, so they cannot be set inconsistently).  Returns the ``CampaignResult``.

    MUST be called on EVERY rank (the hooks are collective); ``grid`` is the rank's
    local ``VoronoiMesh`` (so the manifest lat/lon + grid_shape are rank-local too).
    """
    return build_correction_campaign(
        **_distributed_campaign_kwargs(
            layout, reference, area_weights, n_worst, base_valid_mask),
        **campaign_kwargs,
    )


def build_distributed_multi_correction_campaign(
    *, layout: Any, reference: Any, area_weights: Any, n_worst: int,
    base_valid_mask: Any = None, **campaign_kwargs: Any,
):
    """Run :func:`build_multi_correction_campaign` (SIMULTANEOUS multi-coefficient)
    on a DISTRIBUTED-MPAS partition — the multi-coefficient sibling of
    :func:`build_distributed_correction_campaign` (iter 95).

    Identical distributed composition (the three hooks + the rank-local
    ``reference``/``area_weights`` slice via :func:`_distributed_campaign_kwargs`),
    but corrects EVERY coefficient in ``coefficients`` (e.g. ``("C_K", "Pr_t",
    "C_eps")``, in ``campaign_kwargs``) together from one LES per worst cell.  Same
    contract: call on EVERY rank; ``grid`` is the rank-local ``VoronoiMesh``;
    ``campaign_kwargs`` MUST NOT include ``valid_mask`` / ``manifest_reducer`` /
    ``global_reduce`` / ``reference`` / ``area_weights`` / ``n_worst`` (supplied
    here — a duplicate is a ``TypeError``).  Returns the ``MultiCampaignResult``.
    """
    return build_multi_correction_campaign(
        **_distributed_campaign_kwargs(
            layout, reference, area_weights, n_worst, base_valid_mask),
        **campaign_kwargs,
    )


def build_distributed_mpas_campaign(
    *, global_mesh: Any, reference: Any, area_weights: Any, n_worst: int,
    build_local_driver: Callable[[Any, Any], Any], multi: bool = False,
    base_valid_mask: Any = None, rank: int | None = None, n_ranks: int | None = None,
    validate_partition: bool = True, **campaign_kwargs: Any,
):
    """One-call RUNNABLE distributed-MPAS campaign entry point (iter 96): partition the
    GLOBAL mesh, then run the (single- or multi-coefficient) distributed campaign on
    the rank-local mesh.

    Adds the MPI-aware SETUP on top of the iter-89/95 wrappers (which take a ready
    ``layout``) so an HPC user supplies only the GLOBAL mesh + GLOBAL ERA5
    ``reference``/``area_weights`` and a ``build_local_driver(config, local_mesh)``:

    * ``make_voronoi_partition_layout(global_mesh, rank, n_ranks)`` → the rank's
      partition + ``local_mesh`` (owned + halo cells);
    * the ``local_mesh`` is wired as the campaign ``grid`` (so the manifest lat/lon +
      ``grid_shape`` are rank-local) and ``build_local_driver(cfg, local_mesh)`` builds
      the rank-local distributed model driver — this prevents the common mis-setup of
      passing the GLOBAL mesh as the grid;
    * the three distributed hooks + the rank-local reference/area slice come from
      :func:`build_distributed_correction_campaign` (``multi=False``) /
      :func:`build_distributed_multi_correction_campaign` (``multi=True``).

    Call on EVERY rank with the SAME ``global_mesh`` + GLOBAL ``reference`` /
    ``area_weights`` (each rank diagnoses + corrects only its owned cells; the loop
    is collective).  ``rank``/``n_ranks`` default to ``MPI.COMM_WORLD`` (pass them
    explicitly for testing).  ``campaign_kwargs`` are the usual campaign args MINUS
    ``grid`` / ``build_base_driver`` / the distributed hooks (all supplied here).

    ``validate_partition`` (default ``True``) runs a ONE-TIME collective pre-flight
    (:func:`legoesm.training.distributed_campaign.assert_partition_covers_global`)
    asserting the owned sets across ranks cover the global mesh EXACTLY once — a gap
    is silently never corrected, an overlap is double-counted in the global top-k.
    Set ``False`` only to skip the (collective) check, e.g. a non-MPI unit test.
    """
    if rank is None or n_ranks is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank() if rank is None else rank
        n_ranks = comm.Get_size() if n_ranks is None else n_ranks

    from legoesm.parallel.voronoi_mpi import make_voronoi_partition_layout

    layout = make_voronoi_partition_layout(global_mesh, rank, n_ranks)
    if validate_partition:                           # collective: every rank, before the run
        from legoesm.training.distributed_campaign import (
            assert_partition_covers_global,
        )
        assert_partition_covers_global(layout)
    build = (build_distributed_multi_correction_campaign if multi
             else build_distributed_correction_campaign)
    return build(
        layout=layout, reference=reference, area_weights=area_weights, n_worst=n_worst,
        base_valid_mask=base_valid_mask, grid=layout.local_mesh,
        build_base_driver=lambda cfg: build_local_driver(cfg, layout.local_mesh),
        **campaign_kwargs,
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
    phis: Any | None = None,
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
    stop_on_no_valid_diagnoses: bool = True,
    valid_mask: Any | None = None,
    manifest_reducer: Any | None = None,
    global_reduce: Any | None = None,
    validate_reference: bool = True,
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
    if validate_reference:                           # fail-fast on a units/sign error
        from legoesm.training.compare_reanalysis import validate_reference_physical
        validate_reference_physical(reference, name="reference")
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
        lat_deg=lat_deg, lon_deg=lon_deg, valid_mask=valid_mask,
        manifest_reducer=manifest_reducer,
    )
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn, phis=phis)
    env_grid_fn = maybe_env_grid_fn(feedback_strategy, sigma)

    return run_multi_correction_campaign(
        clubb_cfg, int(n_iterations), specs,
        compare_fn=compare_fn, diagnose_fn=diagnose_fn, grid_shape=grid_shape,
        les_budget=les_budget, env_scales=env_scales,
        feedback_strategy=feedback_strategy, env_grid_fn=env_grid_fn,
        accept_only_if_improved=accept_only_if_improved,
        step_fractions=step_fractions, clip_to_bounds=clip_to_bounds,
        sequential=sequential, bias_tol=bias_tol, patience=patience,
        stop_on_no_valid_diagnoses=stop_on_no_valid_diagnoses,
        initial_fields=initial_fields, start_round=start_round,
        checkpoint_callback=checkpoint_callback, global_reduce=global_reduce,
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
                        "this (or are rejected). Default: run all --iterations "
                        "(but see --keep-dry-rounds: a DRY-LES streak still aborts).")
    p.add_argument("--patience", type=int, default=2,
                   help="rounds of no-progress (or dry-LES rounds) before an "
                        "early-stop / abort fires")
    p.add_argument("--keep-dry-rounds", action="store_true",
                   help="do NOT abort when --patience consecutive rounds produce zero "
                        "VALID LES diagnoses (default: abort with "
                        "stop_reason='no_valid_diagnoses' to save compute on a "
                        "spin-off LES that develops no turbulence)")
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
    p.add_argument("--les-dt", type=float, default=0.5,
                   help="LES timestep [s]. Default 0.5 keeps the acoustic Courant < 1 "
                        "at the shallow-regime dx=50 m (n_acoustic=6); a larger dt is "
                        "rejected by run_forced_les' acoustic-CFL pre-flight.")
    p.add_argument("--les-hours", type=float, default=2.0, help="LES duration [h]")
    p.add_argument("--orographic-forcing", choices=("auto", "on", "off"),
                   default="auto",
                   help="orographic geostrophic LES-forcing term over terrain. "
                        "'auto' (default): use the model's OWN static topography if "
                        "it has any (a flat/aquaplanet model stays flat — always "
                        "safe). 'on': REQUIRE terrain (error if the model is flat). "
                        "'off': force flat (legacy; no orographic term).")
    p.add_argument("--out", default="corrected_clubb_config.json")
    p.add_argument("--checkpoint", default=None,
                   help="write a per-round checkpoint JSON (restartable campaign)")
    p.add_argument("--resume", default=None,
                   help="resume from a --checkpoint JSON (continues the accumulation)")
    return p


def _campaign_knobs_from_args(args) -> dict:
    """The arg-derived campaign knobs SHARED by the single- and multi-coefficient CLI
    build calls — the ONE place the CLI flags map to ``build_correction_campaign`` /
    ``build_multi_correction_campaign`` kwargs.

    Centralizing the mapping (especially the boolean NEGATIONS — ``--keep-dry-rounds``
    → ``stop_on_no_valid_diagnoses=False``, ``--allow-unphysical-coeff`` →
    ``clip_to_bounds=False``, ``--keep-worsening-rounds`` → ``accept_only_if_improved=
    False`` — and the ``--step-fractions`` CSV parse) means the two call sites cannot
    DRIFT: a forgotten / inverted flag is a silent bug that would only surface on a
    multi-day HPC launch.  Pure (no I/O), so it is unit-tested directly.
    """
    return dict(
        n_worst=args.n_worst,
        les_budget=args.les_budget,
        feedback_strategy=args.feedback_strategy,
        accept_only_if_improved=not args.keep_worsening_rounds,
        step_fractions=([float(s) for s in args.step_fractions.split(",")]
                        if args.step_fractions else None),
        clip_to_bounds=not args.allow_unphysical_coeff,
        bias_tol=args.bias_tol,
        patience=args.patience,
        stop_on_no_valid_diagnoses=not args.keep_dry_rounds,
    )


def _per_variable_bias_dict(pvb):
    """JSON form of a raw :class:`PerVariableBias` (the 4 global RMSEs) — a NaN precip
    (precip not compared) serializes as ``null``, never a misleading ``0``/``NaN``."""
    if pvb is None:
        return None
    import math

    def _f(x):
        v = float(x)
        return None if math.isnan(v) else v

    return {"T_rmse_K": _f(pvb.global_T_rmse_K),
            "qv_rmse_kg_kg": _f(pvb.global_qv_rmse_kg_kg),
            "wind_rmse_m_s": _f(pvb.global_wind_rmse_m_s),
            "precip_err_mm_day": _f(pvb.global_precip_err_mm_day)}


def _per_variable_bias_from_dict(d):
    """Reconstruct a :class:`PerVariableBias` from :func:`_per_variable_bias_dict` —
    ``null`` precip → ``NaN`` (preserving the precip-not-compared semantics).  ``None``
    passes through.  Used to restore the ORIGINAL round-0 baseline on a campaign resume
    so the summary's per-variable trajectory is CUMULATIVE."""
    if d is None:
        return None
    import jax.numpy as jnp
    from legoesm.training.bias_metrics import PerVariableBias

    def _v(x):
        return jnp.asarray(float("nan") if x is None else float(x))

    return PerVariableBias(_v(d["T_rmse_K"]), _v(d["qv_rmse_kg_kg"]),
                           _v(d["wind_rmse_m_s"]), _v(d["precip_err_mm_day"]))


def _capture_initial_record(box, res):
    """Per-round checkpoint bookkeeping for a CUMULATIVE-across-resumes summary.

    (1) Captures the campaign-START baseline (combined + per-variable) into ``box`` on
    the FIRST round of a FRESH run — later rounds (and all resumed rounds, where ``box``
    was pre-seeded from the checkpoint) PRESERVE the original.  A non-finite baseline is
    NOT stored (it would poison every later resume's reported trajectory — Codex); the
    next round retries.  (2) Accumulates THIS segment's running LES-diagnosis-count sums
    (``n_diag_seg`` / ``n_valid_seg``) — added to the prior-segments' totals when the
    checkpoint persists the cumulative counts.  Called EVERY round (accepted or rejected,
    matching the field persistence)."""
    import math

    if box.get("initial_bias") is None:
        b = float(res.bias.baseline_bias)
        if math.isfinite(b):
            box["initial_bias"] = b
            pv = getattr(res, "per_variable_bias", None)
            box["initial_per_variable"] = (
                _per_variable_bias_dict(pv.baseline) if pv is not None else None)
    box["n_diag_seg"] = box.get("n_diag_seg", 0) + int(getattr(res, "n_diagnosed", 0))
    box["n_valid_seg"] = (
        box.get("n_valid_seg", 0) + int(getattr(res, "n_diagnoses_valid", 0)))


def _per_variable_to_json(pv):
    """JSON form of a CampaignSummary.per_variable (PerVariableBiasImprovement) — the
    round-0→final per-variable global bias.  ``None`` when the campaign carried no
    error_fields; a NaN precip (precip not compared) serializes as ``null`` (valid
    JSON), never a misleading ``0`` or ``NaN`` token."""
    if pv is None:
        return None
    return {
        "baseline": _per_variable_bias_dict(pv.baseline),
        "final": _per_variable_bias_dict(pv.updated),
        "improved": {"T": bool(pv.T_improved), "qv": bool(pv.qv_improved),
                     "wind": bool(pv.wind_improved), "precip": bool(pv.precip_improved)},
    }


def _summary_to_json(summary):  # pragma: no cover - HPC path
    """JSON-serializable form of a :class:`CampaignSummary` for the output file."""
    return {
        "n_rounds": summary.n_rounds, "n_accepted": summary.n_accepted,
        "acceptance_rate": summary.acceptance_rate, "stop_reason": summary.stop_reason,
        "initial_bias": summary.initial_bias, "final_bias": summary.final_bias,
        "absolute_reduction": summary.absolute_reduction,
        "fractional_reduction": summary.fractional_reduction,
        "n_diagnosed_total": summary.n_diagnosed_total,
        "n_diagnoses_valid_total": summary.n_diagnoses_valid_total,
        "per_variable_bias": _per_variable_to_json(summary.per_variable),
        "coefficients": [
            {"promotion_key": c.promotion_key, "n_columns": c.n_columns,
             "field_min": c.field_min, "field_max": c.field_max,
             "field_mean": c.field_mean, "field_std": c.field_std,
             "n_at_lower_bound": c.n_at_lower_bound,
             "n_at_upper_bound": c.n_at_upper_bound,
             "bounds": list(c.bounds) if c.bounds is not None else None}
            for c in summary.coefficients],
    }


def build_campaign_output_dict(result, *, grid_provenance, summary, health,
                               corrected_field=None, coefficients=None):
    """Assemble the JSON-serializable campaign-output dict — the on-disk artifact the
    DEPLOY path (:func:`legoesm.training.deploy_correction.corrected_clubb_config`)
    reads to update a production AMIP/CMIP run (the literal "update the parameters"
    plumbing).  SHARED by the single- and multi-coefficient CLI write sites so the
    written format CANNOT DRIFT from what the deploy loader expects (round-trip
    tested).  Pure (no I/O).

    Pass EXACTLY one of ``corrected_field`` (single — a top-level ``C_K``/``Pr_t``/
    ``C_eps`` per-column array, read from ``result.final_config``) or ``coefficients``
    (multi — a ``"fields"`` dict keyed by promotion_key, read from
    ``result.final_fields``).  The two shapes mirror :func:`corrected_clubb_config`.
    """
    import numpy as np

    if (corrected_field is None) == (coefficients is None):
        raise ValueError(
            "build_campaign_output_dict: pass EXACTLY one of corrected_field "
            "(single-coefficient) or coefficients (multi-coefficient).")
    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    if corrected_field is not None:
        # NO reshape (behavior-preserving): the single final_config field is the
        # per-column 1-D array; ``.tolist()`` keeps a 0-D scalar (a no-op / zero-round
        # campaign that corrected NOTHING) as a bare float so the deploy loader's
        # 1-D assertion REJECTS it LOUDLY rather than silently promoting it to a
        # length-1 single-column array (Codex iter 105).
        payload = {corrected_field: np.asarray(
            getattr(result.final_config, corrected_field)).tolist()}
    else:
        payload = {
            "coefficients": list(coefficients),
            "fields": {k: np.asarray(v).reshape(-1).tolist()
                       for k, v in result.final_fields.items()},
        }
    return {**payload,
            "grid": grid_provenance,
            "biases": biases, "accepted": accepted, "step_fractions": steps,
            "summary": _summary_to_json(summary),
            "health": {"status": health.status, "message": health.message}}


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


def _format_per_variable_bias(pvb) -> str:
    """One-line per-VARIABLE RMSE baseline→updated + which variables improved, for a
    round's ``per_variable_bias`` (a ``PerVariableBiasImprovement``), or ``""`` when the
    round carried none.  Surfaces a round that lowered the COMBINED score by trading
    variables off (e.g. better T, worse wind).  precip is omitted (campaign compares
    have no ERA5 precip ⇒ NaN)."""
    if pvb is None:
        return ""
    b, u = pvb.baseline, pvb.updated
    vals = (f"T {float(b.global_T_rmse_K):.4g}->{float(u.global_T_rmse_K):.4g}K "
            f"qv {float(b.global_qv_rmse_kg_kg):.4g}->{float(u.global_qv_rmse_kg_kg):.4g} "
            f"wind {float(b.global_wind_rmse_m_s):.4g}->{float(u.global_wind_rmse_m_s):.4g}m/s")
    imp = [n for n, f in (("T", pvb.T_improved), ("qv", pvb.qv_improved),
                          ("wind", pvb.wind_improved)) if bool(f)]
    return f"    per-var RMSE {vals} | improved: {','.join(imp) if imp else 'none'}"


def _run_multi_main(args, base_cfg, grid, sigma, reference, build_base_driver,
                    extract_fn, run_les, *, phis=None):  # pragma: no cover - heavy I/O
    """Multi-coefficient campaign entry (``--coefficients``): a dict checkpoint /
    resume / output for the per-coefficient accumulated fields.  ``phis`` (the
    model's static orographic topography, resolved once in :func:`main`) is
    forwarded to the multi-coefficient campaign builder so the orographic
    LES-forcing term activates identically to the single-coefficient path."""
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
    # Cumulative-trajectory record across resumes (see the single-coefficient main()).
    init_box = {"initial_bias": None, "initial_per_variable": None,
                "n_diagnosed_prior": 0, "n_diagnoses_valid_prior": 0}
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
        init_box["initial_bias"] = ckpt.get("initial_bias")
        init_box["initial_per_variable"] = ckpt.get("initial_per_variable")
        init_box["n_diagnosed_prior"] = int(ckpt.get("n_diagnosed_total", 0))
        init_box["n_diagnoses_valid_prior"] = int(ckpt.get("n_diagnoses_valid_total", 0))
        print(f"[campaign] resuming multi from {args.resume} at round {start_round}")

    checkpoint_callback = None
    if args.checkpoint:
        def checkpoint_callback(round_idx, res, fields):
            _capture_initial_record(init_box, res)
            with open(args.checkpoint, "w") as f:
                json.dump({"round": int(round_idx),
                           "coefficients": list(coefficients),
                           "fields": {k: np.asarray(v).reshape(-1).tolist()
                                      for k, v in fields.items()},
                           "initial_bias": init_box["initial_bias"],
                           "initial_per_variable": init_box["initial_per_variable"],
                           "n_diagnosed_total": (init_box["n_diagnosed_prior"]
                                                 + init_box.get("n_diag_seg", 0)),
                           "n_diagnoses_valid_total": (init_box["n_diagnoses_valid_prior"]
                                                       + init_box.get("n_valid_seg", 0))}, f)

    result = build_multi_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma, grid=grid,
        area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(), coefficients=coefficients,
        run_les_fn=run_les, phis=phis,
        initial_clubb=initial_clubb, initial_fields=initial_fields,
        start_round=start_round, checkpoint_callback=checkpoint_callback,
        sequential=args.staged,
        **_campaign_knobs_from_args(args))

    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    for i, (b0, b1, imp) in enumerate(biases):
        kept = accepted[i] if i < len(accepted) else True
        print(f"[campaign] round {start_round + i}: bias {b0:.5g} -> {b1:.5g} "
              f"(step {steps[i]:.3g}; {'IMPROVED' if imp else 'no improvement'}; "
              f"{'kept' if kept else 'REJECTED'})")
        pv_line = _format_per_variable_bias(result.iterations[i].per_variable_bias)
        if pv_line:
            print(pv_line)
    from legoesm.training.campaign_summary import (
        campaign_health,
        summarize_campaign,
    )
    summary = summarize_campaign(
        result,
        initial_bias_override=init_box["initial_bias"],
        initial_per_variable_override=_per_variable_bias_from_dict(
            init_box["initial_per_variable"]),
        n_diagnosed_prior=init_box["n_diagnosed_prior"],
        n_diagnoses_valid_prior=init_box["n_diagnoses_valid_prior"],
    )
    health = campaign_health(summary)
    with open(args.out, "w") as f:
        json.dump(build_campaign_output_dict(
            result, grid_provenance=_grid_provenance(base_cfg, grid),
            summary=summary, health=health, coefficients=coefficients),
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


def refuse_unsupported_multirank(comm_size: int | None = None) -> None:
    """Refuse a multi-rank ``mpirun`` invocation of this SINGLE-PROCESS CLI.

    The CLI wires NONE of the distributed hooks — the owned-cell ``valid_mask``
    (iter 86), the global top-k ``manifest_reducer`` (iter 87), or the collective
    ``global_reduce`` (iter 88) — and performs no MPI scatter.  Under ``mpirun -np
    N`` it would therefore EITHER deadlock (a distributed model makes ``compare_fn``
    collective, but the loop's gate/line-search stay rank-local without
    ``global_reduce``) OR run N REDUNDANT full-grid campaigns.  Refuse LOUDLY rather
    than silently burning N× the compute or hanging; the distributed-MPAS campaign
    driver that composes those three hooks + scatter is a separate, not-yet-built
    entry point.  ``comm_size`` is injectable for testing; otherwise it is probed
    from ``MPI.COMM_WORLD`` (absent mpi4py ⇒ single process ⇒ allowed).
    """
    if comm_size is None:
        try:
            from mpi4py import MPI
            comm_size = int(MPI.COMM_WORLD.Get_size())
        except Exception:
            comm_size = 1
    if comm_size > 1:
        raise SystemExit(
            f"run_correction_campaign is a SINGLE-PROCESS CLI but was launched on "
            f"{comm_size} MPI ranks.  It wires none of the distributed hooks "
            f"(owned-cell valid_mask, global top-k manifest_reducer, global_reduce) "
            f"and does no scatter, so a multi-rank run would deadlock or run "
            f"{comm_size} redundant full campaigns.  Run it on ONE rank; a "
            f"distributed-MPAS campaign needs the dedicated distributed driver "
            f"(docs/COMPARE_REANALYSIS.md)."
        )


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

    refuse_unsupported_multirank()   # single-process CLI: refuse mpirun -np >1
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

    # Orographic LES-forcing topography: the model's OWN static phis (so it is
    # CONSISTENT with the AMIP/CMIP run that produces the comparison state), built
    # via a SIDE-EFFECT-FREE probe (ModelDriver.static_topography_phis — no manifest
    # / output-dir writes). Resolved ONCE here and threaded into BOTH the single-
    # and multi-coefficient paths. base_cfg is the atm config for both modes, so the
    # probe's topography equals the CMIP coupled driver's atm topography too.
    from legoesm.driver.model_driver import ModelDriver
    phis = resolve_orographic_phis(
        args.orographic_forcing,
        lambda: ModelDriver(base_cfg).static_topography_phis())

    if args.coefficients is not None:
        return _run_multi_main(
            args, base_cfg, grid, sigma, reference, build_base_driver, extract_fn,
            run_les, phis=phis)

    # Restart: resume from a checkpoint (corrected config + accumulated field +
    # round), so a multi-day campaign survives a job timeout (§1 restartable).
    # The CLUBB field the diagnosis corrects (C_K / Pr_t / C_eps) — so the
    # checkpoint/output persist the right field.
    corrected_field = METHOD_PROMOTION[args.diagnosis_method][1]
    initial_clubb, initial_field, start_round = None, None, 0
    # Cumulative-trajectory record across job-timeout resumes (the campaign-START
    # baseline + the prior-segments' diagnosis counts): seeded from the checkpoint on
    # resume so the final summary reports the TRUE start→final reduction + cumulative
    # counts, not just the last resumed segment.
    init_box = {"initial_bias": None, "initial_per_variable": None,
                "n_diagnosed_prior": 0, "n_diagnoses_valid_prior": 0}
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
        # Restore the ORIGINAL campaign-start baseline + prior cumulative counts (absent
        # in pre-iter-137/138 checkpoints ⇒ None/0 ⇒ falls back to segment-only, the old
        # behaviour).
        init_box["initial_bias"] = ckpt.get("initial_bias")
        init_box["initial_per_variable"] = ckpt.get("initial_per_variable")
        init_box["n_diagnosed_prior"] = int(ckpt.get("n_diagnosed_total", 0))
        init_box["n_diagnoses_valid_prior"] = int(ckpt.get("n_diagnoses_valid_total", 0))
        print(f"[campaign] resuming from {args.resume} at round {start_round}")

    checkpoint_callback = None
    if args.checkpoint:
        def checkpoint_callback(round_idx, res, field):
            # Persist the ACCEPTED accumulated `field` (post-gate base), NOT
            # res.updated_config — under the monotonic gate a rejected round's
            # updated_config is discarded while `field` stays the accepted state.
            # The coefficient is stored under its real name (C_K or Pr_t) for
            # inspection + the resume-method guard; resume reconstructs from `field`.
            _capture_initial_record(init_box, res)
            flat = np.asarray(field).reshape(-1)
            with open(args.checkpoint, "w") as f:
                json.dump({"round": int(round_idx),
                           "corrected_field": corrected_field,
                           corrected_field: flat.tolist(),
                           "field": np.asarray(field).tolist(),
                           "initial_bias": init_box["initial_bias"],
                           "initial_per_variable": init_box["initial_per_variable"],
                           "n_diagnosed_total": (init_box["n_diagnosed_prior"]
                                                 + init_box.get("n_diag_seg", 0)),
                           "n_diagnoses_valid_total": (init_box["n_diagnoses_valid_prior"]
                                                       + init_box.get("n_valid_seg", 0))}, f)

    result = build_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma,
        grid=grid, area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(diagnosis_method=args.diagnosis_method),
        run_les_fn=run_les, phis=phis,
        initial_clubb=initial_clubb, initial_field=initial_field,
        start_round=start_round, checkpoint_callback=checkpoint_callback,
        **_campaign_knobs_from_args(args),
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
        pv_line = _format_per_variable_bias(result.iterations[i].per_variable_bias)
        if pv_line:
            print(pv_line)
    from legoesm.training.campaign_summary import (
        campaign_health,
        summarize_campaign,
    )
    promotion_key = METHOD_PROMOTION[args.diagnosis_method][0]
    summary = summarize_campaign(
        result, promotion_key=promotion_key,
        initial_bias_override=init_box["initial_bias"],
        initial_per_variable_override=_per_variable_bias_from_dict(
            init_box["initial_per_variable"]),
        n_diagnosed_prior=init_box["n_diagnosed_prior"],
        n_diagnoses_valid_prior=init_box["n_diagnoses_valid_prior"],
    )
    health = campaign_health(summary)
    with open(args.out, "w") as f:
        json.dump(build_campaign_output_dict(
            result, grid_provenance=_grid_provenance(base_cfg, grid),
            summary=summary, health=health, corrected_field=corrected_field),
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
