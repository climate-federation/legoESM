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

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.training.correction_loop import make_compare_fn, run_correction_campaign
from legoesm.training.run_to_column_mean import make_run_fn

# Run-as-a-script bootstrap (iter 321): a direct ``python scripts/run/run_correction_campaign.py``
# invocation — the sbatch + runbook way — puts the script's OWN directory on ``sys.path``, NOT the
# repo root, so the ``from scripts.validate.compare_amip_era5 import`` inside ``main`` would raise
# ``ModuleNotFoundError: No module named 'scripts'``. Add the repo root so the documented
# script-path invocation works (it already worked via ``-m`` / pytest, which put the CWD on path).
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
    from legoesm.atmosphere.dynamics.les.column_les import process_column

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
                sst_K=getattr(model_ctx, "sst_K", None),
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
            sst_K=getattr(model_ctx, "sst_K", None),
        )

    return diagnose_fn


def fast_validation_les_regime() -> Any:
    """A TINY (8x8x8, dx=50 m, 2 km top) LES regime for WIRING / COMPOSITION validation ONLY.

    Both the shallow AND the deep regime are this same small box, so a spin-off LES finishes
    in SECONDS — turning the multi-minute production LES (shallow 128x128x60 / deep
    256x256x80) into a fast pre-flight that still exercises the WHOLE loop end-to-end (column
    forcing extract -> plane LES -> closure diagnosis -> monotonic gate).

    It is DELIBERATELY under-resolved: the diagnosed closure coefficient is GARBAGE (the
    realism gate rejects it), so this is NOT a science regime and must NEVER drive an actual
    parameter diagnosis or a go/no-go verdict — it answers ONLY "does the harness COMPOSE on
    this config?".  The production diagnosis / go/no-go uses the default ``ColumnLESConfig``
    regime.  Shared by the fast complete-loop validators (the real-ERA5 full-loop check and
    the OSSE ``--quick`` wiring smoke) so the tiny-regime numerics live in ONE place — no
    duplicated LES dimensions across scripts (iter 504).  Self-validating: the tiny dims are
    run through ``validate_regime_config`` so a future edit that violates the regime
    invariants (e.g. ``dz_sfc_m * nlev >= domain_top_m``) fails LOUD here, not deep in a run.
    """
    from legoesm.atmosphere.dynamics.les.les_regime import (
        LESRegimeConfig,
        LESResolutionConfig,
        validate_regime_config,
    )

    tiny = LESResolutionConfig(dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0,
                               dz_sfc_m=50.0)
    regime = LESRegimeConfig(shallow=tiny, deep=tiny)
    validate_regime_config(regime)
    return regime


class _RealismCapture:
    """Wrap a ``run_les_fn`` so every spin-off LES's per-criterion realism breakdown is
    captured (campaign-aggregate observability, iter 512).  A drop-in CALLABLE: the diagnosis
    pipeline calls ``run_les_fn(setup)`` host-side (one per worst column), so this computes the
    breakdown on the FINISHED state and stores ONLY the small ``LESRealismBreakdown`` (six
    bools), never the heavy state — so a multi-day campaign accumulates a tiny list, not a heap
    of plane LESs.  The campaign uses DEFAULT realism thresholds (no CLI flag exposes them), so
    this breakdown matches the gate the pipeline applied.  ``run_les_fn`` is the only contract;
    the diagnosis result is unchanged (this is pure observation)."""

    def __init__(self, run_les_fn: Callable[[Any], Any]) -> None:
        self._run_les_fn = run_les_fn
        self.breakdowns: list[Any] = []

    def __call__(self, setup: Any) -> Any:
        from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
            column_les_realism_breakdown,
        )
        final_state = self._run_les_fn(setup)
        self.breakdowns.append(
            column_les_realism_breakdown(final_state, setup.height_coord))
        return final_state


def realism_summary(breakdowns: Any):
    """Stack a LIST of captured per-column ``LESRealismBreakdown``\\ s into a single
    ``RealismRejectionSummary`` (iter 512/528), or ``None`` when nothing was captured (a
    dry-run / no LES ran).  Shared by the printed line + the output-JSON dict (so the two
    cannot drift) AND the public entry the DISTRIBUTED operator calls on its rank's
    ``_RealismCapture.breakdowns`` before :func:`reduce_realism_summary_mpi` (runbook §7)."""
    if not breakdowns:
        return None
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
        LESRealismBreakdown,
        summarize_realism_breakdowns,
    )
    stacked = LESRealismBreakdown(*(
        jnp.stack([getattr(b, f) for b in breakdowns])
        for f in LESRealismBreakdown._fields))
    return summarize_realism_breakdowns(stacked)


def _realism_campaign_summary_line(breakdowns: Any, *, prefix: str = "[campaign]") -> str | None:
    """One-line aggregate LES-realism report from the captured per-column
    ``LESRealismBreakdown``\\ s (iter 512): how many spin-offs were realistic, and the
    dominant rejection modes — so an operator reading a finished run sees WHETHER the LESs were
    trustworthy and WHY any were not.  ``prefix`` tags the line (``[campaign]`` / ``[osse]``).
    ``None`` when nothing was captured (a dry-run, or no LES ran), so the caller prints
    nothing."""
    s = realism_summary(breakdowns)
    if s is None:
        return None
    if s.n_rejected == 0:
        return f"{prefix} LES realism: all {s.n_total} spin-offs realistic."
    modes = [f"{n}x {lbl}" for lbl, n in (
        ("laminar", s.n_not_turbulent), ("blow-up", s.n_not_finite),
        ("theta-drift", s.n_thermo_drift), ("moisture-runaway", s.n_moisture_runaway),
        ("supersaturated", s.n_supersaturated)) if n]
    return (f"{prefix} LES realism: {s.n_realistic}/{s.n_total} realistic, "
            f"{s.n_rejected} rejected ({', '.join(modes)}) — rejected spin-offs leave their "
            "columns uncorrected; the run verdict flags a starved correction.")


def print_realism_summary(run_les: Any, *, prefix: str = "[campaign]") -> None:
    """Print the aggregate LES-realism line for a finished run, reading the breakdowns a
    :class:`_RealismCapture`-wrapped ``run_les`` accumulated (iter 520).  Shared by the
    campaign + OSSE report sections; a no-op (prints nothing) when nothing was captured."""
    line = _realism_campaign_summary_line(getattr(run_les, "breakdowns", None), prefix=prefix)
    if line:
        print(line)


def _realism_summary_dict(breakdowns: Any) -> dict | None:
    """The campaign-aggregate realism counts as a plain ``dict`` for the output JSON
    (machine-readable post-run analysis across many campaigns); ``None`` when no LES ran, so
    the key is omitted (like ``averaging`` / ``les_config``)."""
    s = realism_summary(breakdowns)
    return None if s is None else dict(s._asdict())


def reduce_realism_summary_mpi(summary: Any, global_reduce: Callable[[Any], Any]) -> Any:
    """Collective-sum a PER-RANK ``RealismRejectionSummary`` into the GLOBAL one — the
    DISTRIBUTED-MPAS campaign's realism aggregate (iter 528).  Each rank summarises ITS owned
    worst columns' captured breakdowns (``summarize_realism_breakdowns`` on a
    ``_RealismCapture``-wrapped ``run_les``), then passes the per-rank summary here on EVERY
    rank; ``global_reduce`` (an MPI allreduce-SUM, e.g. ``global_sum_mpi`` from the distributed
    setup) sums the eight counts so every rank gets the SAME global summary (collective, so a
    rank-0-only print does not deadlock).  Pure host-side (eight ints packed → reduced →
    unpacked); inject ``global_reduce`` so it is unit-testable without ``mpirun``."""
    import jax.numpy as jnp

    packed = jnp.asarray([getattr(summary, f) for f in summary._fields])
    reduced = global_reduce(packed)
    return type(summary)(*(int(x) for x in reduced))


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


def assert_per_column_fields_match_grid(grid_shape, *, area_weights, valid_mask=None,
                                        lat_deg=None, lon_deg=None):
    """Pre-flight (caught by ``--dry-run``): operator-supplied per-column fields MUST be
    placed on the model grid ``grid_shape`` (``= reference.T.shape[:-1]``).

    Two failure modes, both otherwise surfacing only at the FIRST compare (after a short
    but non-free model run); this raises at CONSTRUCTION so the launch dry-run catches them:

    * ``area_weights`` / ``valid_mask`` are BROADCAST onto the per-column score
      (``bias_metrics._area_weighted_mean`` → ``jnp.broadcast_to(.., score.shape)``), so a
      mismatched-resolution field must be broadcastable to ``grid_shape``.  Checked with
      ``np.broadcast_shapes`` — the SAME rule as the runtime ``broadcast_to`` (no false
      positives AND no false negatives).
    * ``lat_deg`` / ``lon_deg`` are GATHERED per-column in the manifest
      (``lat_flat[flat_index]``), so a mismatch would mis-index; validated via the manifest's
      own :func:`~legoesm.training.column_manifest.assert_coords_match_grid` (grid-shaped OR
      rectilinear 1-D ``lat[n_lat]``+``lon[n_lon]``)."""
    import numpy as np

    gshape = tuple(int(d) for d in grid_shape)
    for name, arr in (("area_weights", area_weights), ("valid_mask", valid_mask)):
        if arr is None:
            continue
        shape = tuple(np.shape(arr))
        # broadcast_to(arr, gshape) succeeds IFF the mutual broadcast equals gshape
        # exactly (a larger result ⇒ arr does not fit INTO gshape).  This is the precise
        # runtime condition — no false positives AND no false negatives.
        ok = False
        try:
            ok = np.broadcast_shapes(shape, gshape) == gshape
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(
                f"{name} shape {shape} is not broadcastable to the model grid {gshape} "
                f"(= reference.T.shape[:-1]); the bias aggregation broadcasts {name} onto "
                "the per-column score, so this would crash at the first compare. Pass a "
                "per-column field on the SAME grid as the reference."
            )
    if lat_deg is not None and lon_deg is not None:
        from legoesm.training.column_manifest import assert_coords_match_grid
        assert_coords_match_grid(lat_deg, lon_deg, gshape)


def compose_compare_fn(*, base_atm_config, build_base_driver, extract_column_state,
                        reference, sigma, area_weights, n_worst,
                        lat_deg, lon_deg, valid_mask=None, manifest_reducer=None,
                        spinup_days=0.0):
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
    # spinup_days DISCARDS the un-equilibrated transient from the climatology time-mean
    # (iter 446) so the loop targets the equilibrated bias, not a spin-up-contaminated one.
    run_fn = make_run_fn(build_driver, extract_column_state, spinup_days=spinup_days)
    return make_compare_fn(
        reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=lat_deg, lon_deg=lon_deg, area_weights=area_weights,
        n_worst=n_worst, run_amip_fn=run_fn, valid_mask=valid_mask,
        manifest_reducer=manifest_reducer,
        # Pass the COORDINATE so the per-run layer pressures (hence the bias mass weights)
        # are correct for a HYBRID coordinate (the default) — not pure-sigma (iter 337/338).
        coordinate=sigma,
    )


def maybe_env_grid_fn(feedback_strategy, sigma):
    """The env-generalization ``env_grid_fn(model_ctx)`` for ``feedback_strategy=
    'environment'`` (``None`` for the static scatter)."""
    if feedback_strategy != "environment":
        return None
    from functools import partial as _partial

    from legoesm.training.feedback_assembly import column_environment_grid
    return _partial(column_environment_grid, sigma=sigma)


class CampaignDryRun(NamedTuple):
    """Result of ``build_*_correction_campaign(dry_run=True)``: the campaign
    CONSTRUCTED — the reference passed the physical-plausibility check, the
    ``clubb_lite`` build-driver (``turbulence='clubb_lite'``) + the compare/diagnose
    functions assembled, and the diagnosis-method/coefficient dispatch validated —
    but it was NOT run.  The cheap launch PRE-FLIGHT for a multi-day HPC campaign:
    every construction-time failure (bad config / units / grid / method / scheme)
    surfaces in milliseconds instead of after burning the run.
    """

    grid_shape: tuple[int, ...]
    n_worst: int
    feedback_strategy: str
    coefficients: tuple[str, ...]   # the corrected clubb_lite field name(s)
    n_iterations: int               # rounds the real run would execute
    les_per_round: int              # LES spun off per round (les_budget, else n_worst)
    surface_flux: bool = False      # spin-off LES surface-flux BC ON (iter 364)? (operator-visible)


def _build_campaign_harness(
    *, base_atm_config, build_base_driver, extract_column_state, reference, sigma, grid,
    area_weights, n_worst, lat_deg, lon_deg, valid_mask, manifest_reducer, les_config,
    run_les_fn, phis, feedback_strategy, spinup_days=0.0,
):
    """The compare/diagnose/env-grid harness SHARED by the single- + multi-coefficient
    campaign builders (parallel to ``_build_osse_harness``): the ERA5 ``compare_fn`` (via
    :func:`compose_compare_fn`, which itself factors the run→time-mean ``run_fn``), the LES
    ``diagnose_fn``, and the env-grid fn for ``feedback_strategy='environment'``.  The
    single-vs-multi difference is ONLY the ``les_config`` (``diagnosis_method`` vs
    ``diagnosis_methods``), resolved by the caller — so the wiring lives in ONE place, not
    copy-pasted (CLAUDE.md: no duplicate harness wiring).  Returns
    ``(compare_fn, diagnose_fn, env_grid_fn)``.
    """
    compare_fn = compose_compare_fn(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, reference=reference, sigma=sigma,
        area_weights=area_weights, n_worst=n_worst,
        lat_deg=lat_deg, lon_deg=lon_deg, valid_mask=valid_mask,
        manifest_reducer=manifest_reducer, spinup_days=spinup_days,
    )
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_config, run_les_fn=run_les_fn, phis=phis)
    env_grid_fn = maybe_env_grid_fn(feedback_strategy, sigma)
    return compare_fn, diagnose_fn, env_grid_fn


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
    dry_run: bool = False,
    spinup_days: float = 0.0,
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
    assert_per_column_fields_match_grid(
        grid_shape, area_weights=area_weights, valid_mask=valid_mask,
        lat_deg=lat_deg, lon_deg=lon_deg)

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

    compare_fn, diagnose_fn, env_grid_fn = _build_campaign_harness(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, reference=reference, sigma=sigma,
        grid=grid, area_weights=area_weights, n_worst=n_worst, lat_deg=lat_deg,
        lon_deg=lon_deg, valid_mask=valid_mask, manifest_reducer=manifest_reducer,
        les_config=les_config, run_les_fn=run_les_fn, phis=phis,
        feedback_strategy=feedback_strategy, spinup_days=spinup_days)

    if dry_run:
        # Everything CONSTRUCTED (reference validated, clubb build-driver +
        # compare/diagnose fns assembled, method/coefficient checks passed) — return
        # WITHOUT the (multi-day) run. The launch pre-flight.
        return CampaignDryRun(
            grid_shape=grid_shape, n_worst=int(n_worst),
            feedback_strategy=feedback_strategy, coefficients=(field_name,),
            n_iterations=int(n_iterations),
            les_per_round=_les_per_round_estimate(les_budget, n_worst),
            surface_flux=bool(getattr(les_config, "surface_flux", False)))

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
    validate_partition: bool = True, return_layout: bool = False,
    **campaign_kwargs: Any,
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

    ``return_layout`` (default ``False``) returns ``(result, layout)`` instead of just
    ``result``: the rank's partition ``layout`` is built INTERNALLY here, but the
    rank-local → GLOBAL persist step
    (:func:`legoesm.training.distributed_campaign.assemble_global_campaign_result`)
    NEEDS it, so a turnkey driver passes ``return_layout=True`` and feeds the ``layout``
    straight into the persist — without rebuilding the partition itself.
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
    result = build(
        layout=layout, reference=reference, area_weights=area_weights, n_worst=n_worst,
        base_valid_mask=base_valid_mask, grid=layout.local_mesh,
        build_base_driver=lambda cfg: build_local_driver(cfg, layout.local_mesh),
        **campaign_kwargs,
    )
    return (result, layout) if return_layout else result


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
    dry_run: bool = False,
    spinup_days: float = 0.0,
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
    if len(set(coefficients)) != len(coefficients):
        # A DUPLICATE (e.g. ``--coefficients C_K,C_K``, a typo) silently doubles the LES
        # cost single-rank and is a confusing ``TypeError`` in the distributed hooks
        # (the spec→key map collides); fail loud at construction (caught by --dry-run).
        dups = sorted({c for c in coefficients if list(coefficients).count(c) > 1})
        raise ValueError(
            f"coefficients has duplicate(s) {dups}: each coefficient is corrected ONCE — "
            "a duplicate doubles the (expensive) LES cost and breaks the distributed path. "
            "Pass each of C_K/Pr_t/C_eps at most once.")
    if validate_reference:                           # fail-fast on a units/sign error
        from legoesm.training.compare_reanalysis import validate_reference_physical
        validate_reference_physical(reference, name="reference")
    grid_shape = tuple(int(d) for d in reference.T.shape[:-1])
    lat_deg, lon_deg = grid_latlon_deg(grid, lat_deg, lon_deg)
    assert_per_column_fields_match_grid(
        grid_shape, area_weights=area_weights, valid_mask=valid_mask,
        lat_deg=lat_deg, lon_deg=lon_deg)
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

    compare_fn, diagnose_fn, env_grid_fn = _build_campaign_harness(
        base_atm_config=base_atm_config, build_base_driver=build_base_driver,
        extract_column_state=extract_column_state, reference=reference, sigma=sigma,
        grid=grid, area_weights=area_weights, n_worst=n_worst, lat_deg=lat_deg,
        lon_deg=lon_deg, valid_mask=valid_mask, manifest_reducer=manifest_reducer,
        les_config=les_config, run_les_fn=run_les_fn, phis=phis,
        feedback_strategy=feedback_strategy, spinup_days=spinup_days)

    if dry_run:
        # Constructed every per-coefficient spec + the compare/diagnose fns (the
        # method-union LES, the C_K/c_eps l_mix_max auto-populate) WITHOUT running.
        return CampaignDryRun(
            grid_shape=grid_shape, n_worst=int(n_worst),
            feedback_strategy=feedback_strategy, coefficients=tuple(coefficients),
            n_iterations=int(n_iterations),
            les_per_round=_les_per_round_estimate(les_budget, n_worst),
            surface_flux=bool(getattr(les_config, "surface_flux", False)))

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


def _resolve_coupled_preset(args):
    """The RESOLVED coupled preset object for ``--mode cmip`` (``None`` for AMIP).

    ``make_base_driver_builder`` needs a RESOLVED ``CoupledConfig`` (it reads
    ``preset.ocean_mode`` …), NOT the ``--coupled-preset`` NAME string — passing the raw
    name is the iter-240 ``'str' has no attribute ocean_mode`` crash.  Validated against
    ``PRESETS`` with a fail-loud ``SystemExit`` on an unknown name.  Shared by the campaign
    + OSSE CLIs so the OSSE CMIP go/no-go uses the SAME resolution the campaign does (it
    previously passed the raw name — a latent bug in the untested ``pragma:no-cover`` main).
    """
    if args.mode != "cmip":
        return None
    from legoesm.driver.coupled_config import PRESETS

    if args.coupled_preset not in PRESETS:
        raise SystemExit(
            f"unknown --coupled-preset {args.coupled_preset!r}; "
            f"choose from {sorted(PRESETS)}")
    return PRESETS[args.coupled_preset]()


# A climatology window (in CONFIG days) at/above which a missing spin-up exclusion is worth
# warning about — below this it is a quick test, not a climatology (campaign-control judgment).
_SPINUP_WARN_DAYS = 5


def _spinup_warning_line(spinup_days, days) -> str | None:
    """The spin-up reminder for a multi-day climatology run with NO spin-up exclusion (iter
    447): the time-mean would include the un-equilibrated transient (iter 445/446).  Returns
    the note string, or ``None`` when --spinup-days is set OR the run is a short test."""
    if float(spinup_days or 0.0) <= 0.0 and int(days or 0) >= _SPINUP_WARN_DAYS:
        return (f"[campaign] NOTE: --spinup-days 0 — the {int(days)}-day climatology time-mean "
                "INCLUDES the un-equilibrated model spin-up (iter 445/446 saw a 91 K aloft bias "
                "in a 1-day run). Set --spinup-days (e.g. a few tens of days, << --days) so the "
                "loop targets the EQUILIBRATED bias.")
    return None


# build_era5_amip_forcing builds a SINGLE-MONTH SST forcing (one date -> one monthly chunk); a
# run longer than this cyclically REPEATS the month (get_forcing_at_time wraps over the forcing
# period — intended for a FULL annual cycle, not a sub-annual one). ~31 d = the longest month.
_OFFLINE_FORCING_SPAN_DAYS = 31


def _days_below_cadence_warning(days, diag_days) -> str | None:
    """Launch warning when the run length is <= the diagnostic cadence (``output.diag_days``):
    the model climatology TIME-MEAN samples at each segment boundary, so a run shorter than ONE
    cadence fires NO boundary and a REAL run FAILS LOUD ('no segment boundary fired').  The
    ``--dry-run`` does NOT run the time-mean, so it would otherwise PASS a config that cannot
    actually run (the iter-484/499 dry-run-false-confidence class) — surface it at LAUNCH.
    Returns the note, or ``None`` (runnable, or a getattr-missing stub)."""
    if not days or not diag_days or int(days) > int(diag_days):
        return None
    return (f"[campaign] WARNING: days={int(days)} <= the {int(diag_days)}-day diagnostic cadence "
            "(output.diag_days): the model climatology time-mean fires NO segment boundary, so a "
            f"REAL run FAILS LOUD ('no segment boundary fired'). Use days > {int(diag_days)} "
            "(regenerate with write_amip_clubb_lite_config.py --days N).")


_REALISTIC_RADIATION = ("rrtmgp", "rrtmg")   # spectral schemes where BL mixing (C_K) matters


def _idealized_radiation_low_leverage_warning(radiation) -> str | None:
    """Launch warning that an IDEALIZED radiation scheme (gray / none — anything but the
    spectral rrtmgp/rrtmg) gives the bias TINY C_K leverage, so the LES-informed correction
    cannot meaningfully reduce it.  iters 412/514 measured this: a 40% C_K change moved the
    OSSE-twin combined-score bias by only ~0.0025 (vs the ~11 real-ERA5 idealization bias) —
    the bias is C_K-INSENSITIVE under idealized radiation, so the run will likely report
    'stalled'/'no_change' NOT because the loop is broken but because there is nothing C_K can
    fix.  Surfaced at LAUNCH (campaign + OSSE share ``_build_run_setup``) so the operator does
    not misread the verdict or burn HPC time.  ``None`` (no warning) for a realistic scheme."""
    if radiation is None or str(radiation).lower() in _REALISTIC_RADIATION:
        return None
    return (f"[campaign] WARNING: radiation={radiation!r} is IDEALIZED (not rrtmgp/rrtmg) — the "
            "bias has TINY C_K leverage under it, so the LES-informed correction cannot "
            "meaningfully reduce a C_K-INSENSITIVE bias (iters 412/514: a 40% C_K change moved "
            "the OSSE-twin bias only ~0.0025) and the run will likely report 'stalled'/"
            "'no_change'. Use radiation='rrtmgp' for a meaningful correction / go/no-go.")


def _offline_forcing_window_warning(days, *, span_days=_OFFLINE_FORCING_SPAN_DAYS) -> str | None:
    """Warn when an OFFLINE single-month AMIP forcing (``--amip-forcing-from-local-era5``) drives a
    run LONGER than its ~1-month coverage (iter 458): ``get_forcing_at_time`` then CYCLICALLY WRAPS
    the month, so the SST REPEATS the month while the insolation advances seasonally (iter 449) —
    they DESYNC over the run, biasing the comparison. The AMIP default ``days=200`` trips this.
    Returns ``None`` when the run fits within the forcing span (no wrap)."""
    if days is None or int(days) <= span_days:
        return None
    return (f"[campaign] NOTE: --amip-forcing-from-local-era5 builds a SINGLE-MONTH "
            f"(~{span_days} d) SST forcing, but days={int(days)} exceeds it; get_forcing_at_time "
            "CYCLICALLY REPEATS the month while the insolation advances seasonally, so the SST "
            f"and insolation DESYNC over the run. Use a climatology window <= ~{span_days} days "
            "(keeps SST + insolation aligned), or build a multi-month forcing.")


def _offline_reference_window_warning(days, era5_n_times) -> str | None:
    """Warn when the OFFLINE ERA5 reference window is much SHORTER than the model climatology
    window (iter 463) — the un-automated half of runbook §3's "match the windows". The offline
    times are hourly, so the reference spans ``era5_n_times/24`` days; comparing a short
    reference (the default ``--era5-n-times 1`` = one hour) to a multi-day model time-mean
    compares WEATHER to CLIMATE — a spurious bias the loop would then "correct". Silent for a
    short test (``days < _SPINUP_WARN_DAYS``) or when the reference covers >= half the model
    window (close enough)."""
    if not days or int(days) < _SPINUP_WARN_DAYS:
        return None
    ref_days = float(era5_n_times) / 24.0
    if ref_days >= 0.5 * float(days):
        return None
    return (f"[campaign] NOTE: the offline ERA5 reference averages {int(era5_n_times)} hourly "
            f"time(s) (~{ref_days:.1f} day(s)) but the model climatology window is {int(days)} "
            "days — a short reference vs a multi-day model mean compares WEATHER to CLIMATE "
            f"(runbook §3). Raise --era5-n-times toward ~{24 * int(days)} (with --era5-n-days "
            f">= {int(days)}) so the reference spans the model window.")


def _surface_flux_land_warning(surface_flux, ocean_only, has_land_mask) -> str | None:
    """Warn when ``--surface-flux`` is set on a config WITH a land mask but WITHOUT
    ``--ocean-only`` (iter 465). The surface-flux LES BC derives the flux from the column SST
    (via the GCM bulk scheme, valid over OCEAN); a LAND worst-column would get a flux from a
    NON-ocean SST (a fill/extrapolated value) — a wrong/invalid diagnosis that WASTES the LES
    budget. Pair ``--surface-flux`` with ``--ocean-only`` so only ocean columns (valid SST) are
    ranked. Silent without a land mask (aquaplanet: all ocean) or when ``--ocean-only`` is set."""
    if not surface_flux or ocean_only or not has_land_mask:
        return None
    return ("[campaign] NOTE: --surface-flux derives the LES surface BC from the column SST "
            "(valid over OCEAN), but this config has a land mask and --ocean-only is OFF — a LAND "
            "worst-column would get a surface flux from a non-ocean SST (wrong/invalid diagnosis, "
            "wasting the LES budget). Add --ocean-only so only ocean columns are ranked.")


# Months seasonally FAR from the model's JANUARY-based insolation (cfg.start_day defaults to 0,
# so day_to_calendar(0)=Jan 1) — the solar declination differs most across Apr–Sep, so a
# non-January offline ERA5 window mismatches the insolation season (iter 447, the deferred gap).
_INSOLATION_OFF_SEASON_MONTHS = frozenset({4, 5, 6, 7, 8, 9})


def _insolation_season_note(local_era5_date) -> str | None:
    """The model-vs-ERA5 INSOLATION season-mismatch note for an OFFLINE ERA5 date in the off-season
    half (Apr–Sep).  By default the model's solar calendar is JANUARY-based (model day 0 -> Jan 1),
    so a spring/summer/autumn comparison runs ~off-season insolation while the AMIP SST IS aligned.
    Two fixes, surfaced here: (1, iter 449) set ``insolation_start_doy=<doy>`` in the config to map
    model day 0 to the ERA5 date's day-of-year for the radiation insolation ONLY (CODEX PENDING for
    the radiation path), or (2, fully validated) use a January window so day 0 -> Jan 1 aligns BOTH
    the relative SST forcing and the insolation.  Returns ``None`` for no offline date / a
    near-January window / a malformed date."""
    s = str(local_era5_date or "")
    if len(s) < 8 or not s[:8].isdigit():
        return None
    if int(s[4:6]) not in _INSOLATION_OFF_SEASON_MONTHS:
        return None
    from legoesm.forcing.time_utils import noleap_day_of_year
    try:
        doy = noleap_day_of_year(int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None
    return (f"[campaign] NOTE: the ERA5 date {s} (noleap day-of-year {doy}) is in the off-season "
            f"half (month {int(s[4:6])}); the model's insolation defaults to JANUARY-based (model "
            "day 0 -> Jan 1), so the solar season MISMATCHES the SST-aligned comparison. Align it "
            f"with EITHER insolation_start_doy={doy} in the config (radiation-only, decoupled from "
            "the relative SST; CODEX PENDING) OR a January ERA5 window (validated; day 0 -> Jan 1 "
            "aligns BOTH the SST forcing AND the insolation).")


def _build_run_setup(args):
    """The run-setup preamble SHARED by the campaign + OSSE CLIs (CLAUDE.md: no duplicate
    wiring): load the base config/grid/sigma, build the mode-specific driver builder +
    column extractor (with the RESOLVED coupled preset — :func:`_resolve_coupled_preset`),
    the CFL-checked forced-LES runner, and the orographic ``phis`` (the model's OWN static
    topography via the side-effect-free probe, so the OSSE go/no-go uses the IDENTICAL
    forcing the real campaign will — iter 126/127).  Returns ``(base_cfg, grid, sigma,
    build_base_driver, extract_fn, run_les, phis)``; the campaign adds the ERA5 reference,
    the OSSE adds the pseudo-truth.
    """

    from legoesm.atmosphere.dynamics.les.column_les import run_forced_les
    from legoesm.driver.model_driver import ModelDriver

    base_cfg, grid, sigma = load_base_config_and_grid(args.config)
    # The off-season insolation NOTE is suppressed when --align-insolation will set the season
    # itself (its own confirmation line prints in _maybe_align_insolation instead).
    _insol_date = (None if getattr(args, "align_insolation", False)
                   else getattr(args, "local_era5_date", None))
    # The offline-forcing wrap warning only applies when building the offline forcing; the span
    # scales with --amip-forcing-n-months (iter 459) — N months => ~N*31 days of coverage.
    _n_months = max(1, int(getattr(args, "amip_forcing_n_months", 1) or 1))
    _offline_forcing_note = (
        _offline_forcing_window_warning(getattr(base_cfg, "days", 0),
                                        span_days=_n_months * _OFFLINE_FORCING_SPAN_DAYS)
        if getattr(args, "amip_forcing_from_local_era5", False) else None)
    # The reference-window check only applies to the OFFLINE (hourly) reference path.
    _ref_window_note = (
        _offline_reference_window_warning(getattr(base_cfg, "days", 0),
                                          int(getattr(args, "era5_n_times", 1) or 1))
        if getattr(args, "local_era5_dir", None) else None)
    _surface_flux_note = _surface_flux_land_warning(
        getattr(args, "surface_flux", False), getattr(args, "ocean_only", False),
        bool(getattr(base_cfg, "land_mask_path", "")))
    for _note in (_spinup_warning_line(getattr(args, "spinup_days", 0.0),
                                       getattr(base_cfg, "days", 0)),
                  _days_below_cadence_warning(
                      getattr(base_cfg, "days", 0),
                      getattr(getattr(base_cfg, "output", None), "diag_days", 0)),
                  _idealized_radiation_low_leverage_warning(
                      getattr(base_cfg, "radiation", None)),
                  _insolation_season_note(_insol_date),
                  _offline_forcing_note,
                  _ref_window_note,
                  _surface_flux_note):
        if _note:
            print(_note, flush=True)
    build_base_driver, extract_fn = make_base_driver_builder(
        args.mode, coupled_preset=_resolve_coupled_preset(args), ocean_grid=None)
    n_steps = _les_n_steps(args.les_hours, args.les_dt)
    # Wrap the LES runner ONCE here (shared by the campaign + OSSE CLIs) so EVERY spin-off
    # LES's realism breakdown is captured for the end-of-run aggregate report (iter 512/520);
    # pure observation, the diagnosis is unchanged.
    run_les = _RealismCapture(partial(run_forced_les, dt_s=args.les_dt, n_steps=n_steps))
    phis = resolve_orographic_phis(
        args.orographic_forcing,
        lambda: ModelDriver(base_cfg).static_topography_phis())
    return base_cfg, grid, sigma, build_base_driver, extract_fn, run_les, phis


def _maybe_apply_local_era5_forcing(args, base_cfg):
    """If ``--amip-forcing-from-local-era5``, build the AMIP forcing from the local ERA5
    archive and inject it into ``base_cfg``; else return ``base_cfg`` unchanged.

    Injecting into ``base_cfg`` (the campaign template) means every corrected round's config
    — derived by ``_replace``-ing only the closure coefficient — carries the same prescribed
    SST forcing.  ``phis`` (topography) is unaffected by the SST dataset, so it stays correct
    even though ``_build_run_setup`` computed it from the pre-injection config.
    """
    if not getattr(args, "amip_forcing_from_local_era5", False):
        return base_cfg
    from scripts.data.load_local_era5 import (
        apply_amip_forcing_to_config,
        build_era5_amip_forcing,
    )

    fcfg = build_era5_amip_forcing(
        args.local_era5_dir, args.local_era5_date, args.amip_forcing_out,
        hour_stride=args.amip_forcing_hour_stride,
        n_months=getattr(args, "amip_forcing_n_months", 1))
    return apply_amip_forcing_to_config(base_cfg, fcfg)


def _maybe_align_insolation(args, base_cfg):
    """If ``--align-insolation`` (opt-in), set ``base_cfg.insolation_start_doy`` to the noleap
    day-of-year of the offline ``--local-era5-date`` so the radiation insolation runs the SAME
    solar season as the ERA5 comparison (iter 449/450) — decoupled from the relative-indexed SST
    forcing (the iter-449 driver seam). Injecting into ``base_cfg`` (the campaign template) carries
    the alignment into every corrected round. Flag off => unchanged (production default).

    Fails LOUDLY (``SystemExit``) when the flag is set without an offline date (the season is taken
    from that date) or with a malformed date. CODEX PENDING: the radiation-path behaviour change is
    pending the mandatory adversarial review; the flag is OFF by default, so it changes nothing
    until an operator opts in."""
    if not getattr(args, "align_insolation", False):
        return base_cfg
    s = str(getattr(args, "local_era5_date", None) or "")
    if len(s) < 8 or not s[:8].isdigit():
        raise SystemExit(
            "--align-insolation needs an offline --local-era5-date YYYYMMDD (the insolation "
            f"season is taken from that date); got {s!r}. For a Zarr ERA5 reference, set "
            "insolation_start_doy in the config directly instead.")
    from legoesm.forcing.time_utils import noleap_day_of_year
    try:
        doy = noleap_day_of_year(int(s[4:6]), int(s[6:8]))
    except ValueError as e:
        raise SystemExit(f"--align-insolation: invalid --local-era5-date {s}: {e}")
    print(f"[campaign] --align-insolation: insolation_start_doy={doy} (noleap day-of-year of the "
          f"ERA5 date {s}) so the model's solar season matches the SST-aligned comparison "
          "(iter 449 seam; radiation-only, CODEX PENDING).", flush=True)
    return base_cfg._replace(insolation_start_doy=float(doy))


def _effective_config_path(out_path: str) -> str:
    """The sidecar path for the EFFECTIVE config the campaign ran (next to ``--out``)."""
    import os
    return os.path.splitext(out_path)[0] + ".effective_config.json"


def _write_effective_config(base_cfg, out_path: str) -> str:
    """Persist the EFFECTIVE ``ExperimentConfig`` the campaign actually ran — the base config
    PLUS the RUNTIME injections (``--amip-forcing-from-local-era5`` => dataset/forcing_path;
    ``--align-insolation`` => insolation_start_doy) — to a sidecar next to ``--out`` (iter 464).

    Without this the runtime-flag injections leave NO on-disk record: the campaign calibrates C_K
    on (base + injections) but the base config JSON does not capture them, so the run is not
    reproducible and a SAME-WINDOW deploy (``build_deployed_config``) on the base JSON would run a
    DIFFERENT SST boundary + insolation season than the C_K was tuned for. This sidecar is the
    authoritative calibration config — use it as the deploy/re-run base. (A HELD-OUT-window verify
    still re-derives the window-specific forcing + ``insolation_start_doy`` for that window; the
    sidecar carries the non-window settings + documents what was calibrated.) Returns the path."""
    import json
    import os

    from legoesm.driver.config import config_to_dict
    path = _effective_config_path(out_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_to_dict(base_cfg), f, indent=2)
    return path


def _maybe_ocean_mask(args, base_cfg):
    """If ``--ocean-only`` (opt-in), build an OCEAN-only worst-column ranking ``valid_mask``
    from the model's STATIC land fraction so the scarce LES budget targets columns where the
    atmospheric closure is the right lever — over ocean the surface is PRESCRIBED (the AMIP SST
    pins it), so a column bias is attributable to the atmospheric column (incl. the turbulence
    closure); over land the model's own land-surface biases dominate and the closure cannot fix
    them. Flag OFF => ``None`` (rank ALL columns, the default).

    Sources the land fraction via the side-effect-free ``ModelDriver.static_land_fraction()``
    probe (the SAME minimal chain ``_build_run_setup`` already runs for the orographic ``phis``)
    and thresholds it through :func:`legoesm.training.compare_reanalysis.ocean_valid_mask`.
    Fails LOUD (``SystemExit``) if the mask excludes EVERY column (no ocean to rank — the
    'no silent no-op' convention)."""
    if not getattr(args, "ocean_only", False):
        return None
    import jax.numpy as jnp
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.training.compare_reanalysis import ocean_valid_mask
    land_fraction = ModelDriver(base_cfg).static_land_fraction()
    # ocean_valid_mask flattens ROW-MAJOR to (n_columns,); reshape it BACK to the model grid
    # shape so the mask is broadcastable to the 2D per-column score in the bias aggregation
    # (like area_weights) AND still flattens cleanly inside rank_worst_columns. A FLAT mask is
    # broadcastable only on a flat (MPAS nCells) grid — on a STRUCTURED grid (nlat,nlon) a
    # flat (n_columns,) mask is REJECTED by assert_per_column_fields_match_grid (n_columns does
    # not broadcast to (nlat,nlon)), which silently broke --ocean-only on lat-lon/cubed-sphere/
    # Gaussian grids until the dry-run harness check surfaced it.
    grid_shape = jnp.asarray(land_fraction).shape
    mask = ocean_valid_mask(
        land_fraction, max_land_fraction=args.max_land_fraction).reshape(grid_shape)
    n_ocean = int(jnp.sum(mask))
    if n_ocean == 0:
        raise SystemExit(
            f"--ocean-only masks out EVERY column (no column has land_fraction <= "
            f"{args.max_land_fraction}); nothing to rank. Raise --max-land-fraction, or drop "
            "--ocean-only (e.g. an all-land regional grid has no ocean columns).")
    print(f"[campaign] --ocean-only: ranking the {n_ocean} ocean columns "
          f"(land_fraction <= {args.max_land_fraction}) of {int(mask.size)} total.", flush=True)
    if getattr(args, "mode", "amip") == "cmip":
        print("[campaign] NOTE: --ocean-only is most meaningful for AMIP (the PRESCRIBED SST "
              "pins the ocean surface, so a bias there is attributable to the atmospheric "
              "closure). Under --mode cmip the ocean is INTERACTIVE, so an ocean-column bias "
              "also reflects the coupled-ocean SST bias — it still excludes the land-model "
              "biases, but does not cleanly isolate the closure.", flush=True)
    return mask


def _amip_forcing_provenance(args) -> dict | None:
    """The ``--amip-forcing-from-local-era5`` build provenance for the dry-run report, or
    ``None`` when the flag is off — derived from the CLI args (the source archive, date,
    output path, and hour stride that :func:`_maybe_apply_local_era5_forcing` used)."""
    if not getattr(args, "amip_forcing_from_local_era5", False):
        return None
    return {
        "source": args.local_era5_dir,
        "date": args.local_era5_date,
        "out": args.amip_forcing_out,
        "hour_stride": args.amip_forcing_hour_stride,
        "n_months": getattr(args, "amip_forcing_n_months", 1),
    }


def _resolve_era5_n_times(n_times: int) -> int:
    """Validate the ``--era5-n-times`` averaging window LOUDLY, returning the int.

    ``n_times`` is the number of consecutive ERA5 times averaged into the reference
    climatology (see :func:`legoesm.training.era5_to_state.load_era5_time_mean`); it
    must be ``>= 1`` (you cannot average fewer than one time).  A value ``< 1`` (e.g.
    a fat-fingered ``--era5-n-times 0`` or a negative) is a USER ERROR: the old
    ``max(1, int(...))`` silently clamped it to 1, MASKING the typo (CLAUDE.md
    fail-loud / no-silent-coerce doctrine).  Raise a clear, actionable
    ``SystemExit`` (the campaign's established CLI-error pattern) instead.
    """
    # argparse passes an int (type=int), but a direct Python caller could pass a
    # non-integral float; truncating it (2.9 -> 2) would be exactly the silent
    # coerce this validator exists to forbid, so reject it too.
    if isinstance(n_times, float) and not n_times.is_integer():
        raise SystemExit(
            f"--era5-n-times must be a whole number (got {n_times}): it counts ERA5 "
            "times to average — a fractional window is meaningless.")
    n = int(n_times)
    if n < 1:
        raise SystemExit(
            f"--era5-n-times must be >= 1 (got {n}): it is the number of consecutive "
            "ERA5 times (starting at --era5-time-idx) averaged into the reference "
            "climatology. Use 1 for a single snapshot (the default).")
    return n


def _build_arg_parser():
    import argparse
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="base ExperimentConfig JSON")
    p.add_argument("--compilation-cache-dir",
                   default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
                   help="dir for JAX's PERSISTENT compilation cache so the expensive rrtmgp "
                        "JIT (>16 min) is reused across the self-requeue + repeated launches. "
                        "Defaults to $JAX_COMPILATION_CACHE_DIR; empty = disabled.")
    p.add_argument("--cache-min-compile-secs", type=float, default=30.0,
                   help="(--compilation-cache-dir) cache only compiles slower than this [s] "
                        "(default 30 = the radiation graph, not trivial compiles).")
    p.add_argument("--mode", choices=("amip", "cmip"), default="amip",
                   help="AMIP (prescribed SST) or CMIP (coupled ocean)")
    p.add_argument("--coupled-preset", default="aquaplanet",
                   help="coupled_config preset name (CMIP mode)")
    p.add_argument("--era5-zarr", default=None,
                   help="ERA5 zarr/store (reference); OR use --local-era5-dir for a "
                        "local NCAR-RDA NetCDF archive (offline, no network).")
    p.add_argument("--era5-cache", default=None, help="ERA5 local cache dir")
    p.add_argument("--local-era5-dir", default=None,
                   help="Directory of LOCAL NCAR-RDA ERA5 NetCDF (e5.oper.an.{pl,sfc}."
                        "*.ll025*.nc) — reads REAL ERA5 OFFLINE via "
                        "scripts.data.load_local_era5 (no Zarr/network). Mutually "
                        "exclusive with --era5-zarr. Requires --local-era5-date.")
    p.add_argument("--local-era5-date", default=None,
                   help="Date (YYYYMMDD) selecting the --local-era5-dir day; "
                        "--era5-time-idx/--era5-n-times index that day's 24 hourly times.")
    p.add_argument("--align-insolation", action="store_true",
                   help="Set insolation_start_doy from --local-era5-date so the radiation "
                        "insolation runs the SAME solar season as the comparison (iter 449 "
                        "seam; radiation-only, decoupled from the SST). Off by default; "
                        "fails loud without an offline date. CODEX PENDING.")
    p.add_argument("--amip-forcing-from-local-era5", action="store_true",
                   help="(--mode amip) ALSO build the AMIP SST/sea-ice forcing from the "
                        "SAME --local-era5-dir archive (build_era5_amip_forcing) and inject "
                        "it into the config — a fully-OFFLINE realistic AMIP campaign (real "
                        "SST forcing AND real ERA5 compare) in one command, no hand-edited "
                        "dataset=custom config. Overrides any forcing in --config.")
    p.add_argument("--amip-forcing-out", default="era5_amip_forcing.nc",
                   help="Output NetCDF for --amip-forcing-from-local-era5 (a runtime "
                        "artifact; written at launch, gitignored). OVERWRITTEN if it "
                        "exists — give concurrent campaigns from the same directory "
                        "DISTINCT paths so they do not clobber each other's forcing.")
    p.add_argument("--amip-forcing-hour-stride", type=int, default=24,
                   help="Subsample the monthly-hourly ERA5 boundary forcing to every Nth "
                        "step (default 24 = daily) — the monthly-hourly file OOMs the "
                        "loader (iter 419).")
    p.add_argument("--amip-forcing-n-months", type=int, default=1,
                   help="Concatenate this many CONSECUTIVE monthly SST/SIC chunks (from "
                        "--local-era5-date's month) into the AMIP forcing (default 1) — use "
                        ">1 so a multi-month climatology window stays within the forcing "
                        "coverage instead of cyclically repeating one month (iter 458/459).")
    p.add_argument("--era5-time-idx", type=int, default=0)
    p.add_argument("--era5-n-times", type=int, default=1,
                   help="Average this many consecutive ERA5 times (starting at "
                        "--era5-time-idx) into a time-MEAN reference CLIMATOLOGY — so "
                        "the time-mean model is compared to a time-mean ERA5, not a "
                        "single synoptic snapshot (which injects weather noise into the "
                        "bias). Default 1 = a single time (the old behaviour).")
    p.add_argument("--era5-n-days", type=int, default=1,
                   help="(OFFLINE --local-era5-dir) load this many CONSECUTIVE days of ERA5 "
                        "(from --local-era5-date) so the offline reference can span a "
                        "multi-day CLIMATOLOGY window matching the model time-mean (default 1 "
                        "= a single day, ~24 hourly times). Pair with --era5-n-times up to "
                        "24*n_days. Zarr (--era5-zarr) ignores this (its store spans times).")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--spinup-days", type=float, default=0.0,
                   help="discard the first N days of model time from the climatology "
                        "time-mean (the un-equilibrated SPIN-UP transient) before comparing "
                        "to ERA5, so the loop targets the equilibrated bias not a spin-up-"
                        "contaminated one (iter 446; default 0 = keep everything). Keep "
                        "--days enough longer than --spinup-days for a stable post-spin-up mean.")
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
    p.add_argument("--ocean-only", action="store_true",
                   help="rank ONLY ocean columns (where the AMIP SST pins the surface so a "
                        "bias is attributable to the atmospheric/turbulence closure). Off by "
                        "default (rank all columns). Fails loud if no ocean columns exist.")
    p.add_argument("--max-land-fraction", type=float, default=0.5,
                   help="(--ocean-only) a column is ocean where land_fraction <= this "
                        "[0,1]; 0.0 = pure ocean, 0.5 = majority ocean (default).")
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
    p.add_argument("--surface-flux", action="store_true",
                   help="OPT-IN (iter 364/365): give each worst-column spin-off LES a "
                        "prescribed surface-flux BC — the GCM bulk surface sensible/latent "
                        "fluxes (reused, no new tunables) converted to kinematic θ/q_v "
                        "fluxes — so surface-driven (convective) columns get their primary "
                        "turbulence driver. Default OFF (surface-flux-free LES). Requires "
                        "the run's state to carry an SST (AMIP prescribed / CMIP coupled).")
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
    p.add_argument("--dry-run", action="store_true",
                   help="launch PRE-FLIGHT: load the config + ERA5 reference, build the "
                        "driver + the campaign (validating units/grid/scheme/method), "
                        "report what WOULD run, then exit 0 WITHOUT the (multi-day) run.")
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
    # FAIL LOUD on degenerate counts (caught by the launch dry-run, not after hours):
    # a non-positive --n-worst ranks NOTHING and a non-positive --les-budget runs NO
    # LES, so the whole multi-day campaign would silently correct nothing (the same
    # class as the iter-201 zero-LES-steps guard / the iter-388 "no silent no-op" rule).
    if int(args.n_worst) < 1:
        raise SystemExit(
            f"--n-worst {args.n_worst} must be >= 1: it is the number of worst columns "
            "ranked + LES-diagnosed each round; < 1 ranks nothing, so the campaign would "
            "correct NOTHING (a silent multi-day no-op).")
    if args.les_budget is not None and int(args.les_budget) < 1:
        raise SystemExit(
            f"--les-budget {args.les_budget} must be >= 1 (or unset for no cap): it caps "
            "the LES runs per round; < 1 runs no LES, so nothing is diagnosed or corrected.")
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
        spinup_days=args.spinup_days,
    )


def _per_variable_bias_dict(pvb):
    """JSON form of a raw :class:`PerVariableBias` (the 4 global RMSEs) — a non-finite
    metric (NaN precip 'not compared', or a ±inf RMSE from a blown-up/overflowed model)
    serializes as ``null``, never a misleading ``0`` or a non-standard ``NaN``/``Infinity``
    token.  Uses the shared :func:`_json_finite` so ALL four fields are sanitized for both
    NaN AND ±inf (a NaN-only guard would leak an ``Infinity`` from an overflow — iter 246)."""
    if pvb is None:
        return None
    return {"T_rmse_K": _json_finite(pvb.global_T_rmse_K),
            "qv_rmse_kg_kg": _json_finite(pvb.global_qv_rmse_kg_kg),
            "wind_rmse_m_s": _json_finite(pvb.global_wind_rmse_m_s),
            "precip_err_mm_day": _json_finite(pvb.global_precip_err_mm_day)}


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


def _json_finite(x):
    """A float-or-``None`` for JSON: a non-finite metric (NaN/±inf) serializes as
    ``null`` (valid JSON), never a non-standard ``NaN``/``Infinity`` token.

    Matches the per-variable convention (:func:`_per_variable_bias_dict`) so a
    DIVERGED run's biases (the iter-161 ``non_finite_bias`` health case — NaN
    ``final_bias``, or a ±inf ``fractional_reduction`` when ``initial_bias==0``)
    write consistently as ``null`` across the whole output dict.  ``None`` passes
    through unchanged."""
    import math

    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _summary_to_json(summary):
    """JSON-serializable form of a :class:`CampaignSummary` for the output file."""
    return {
        "n_rounds": summary.n_rounds, "n_accepted": summary.n_accepted,
        "acceptance_rate": summary.acceptance_rate, "stop_reason": summary.stop_reason,
        "initial_bias": _json_finite(summary.initial_bias),
        "final_bias": _json_finite(summary.final_bias),
        "absolute_reduction": _json_finite(summary.absolute_reduction),
        "fractional_reduction": _json_finite(summary.fractional_reduction),
        "n_diagnosed_total": summary.n_diagnosed_total,
        "n_diagnoses_valid_total": summary.n_diagnoses_valid_total,
        # NB: the per-round trajectory is NOT re-serialized here — the campaign output
        # already carries it at the TOP LEVEL as ``biases`` (per-round [baseline,
        # updated, …], NaN-sanitized) + ``accepted``, which the plotter reads. The
        # CampaignSummary.round_trace exists only for the human-readable .report() recap.
        "per_variable_bias": _per_variable_to_json(summary.per_variable),
        "coefficients": [
            {"promotion_key": c.promotion_key, "n_columns": c.n_columns,
             "field_min": _json_finite(c.field_min), "field_max": _json_finite(c.field_max),
             "field_mean": _json_finite(c.field_mean), "field_std": _json_finite(c.field_std),
             "n_at_lower_bound": c.n_at_lower_bound,
             "n_at_upper_bound": c.n_at_upper_bound,
             "bounds": list(c.bounds) if c.bounds is not None else None}
            for c in summary.coefficients],
    }


def _assert_corrected_field_finite(name, arr):
    """Return ``arr`` after asserting EVERY corrected per-column coefficient is finite.

    A corrected field must be finite to be a valid deploy artifact: a NaN/inf coefficient
    (an un-clipped ill-posed diagnosis that slipped the accept gate) would BOTH (a)
    serialize as a non-standard ``NaN``/``Infinity`` JSON token, making the WHOLE output
    file unparseable by the plotters / deploy reader, AND (b) deploy a non-finite
    coefficient that blows up the production run.  Fail LOUD here — NOT sanitize to null:
    a NaN coefficient is a BUG to surface (unlike the bias trajectory, which legitimately
    RECORDS a rejected diverged round's NaN, iter 245).  The accept gate keeps the final
    field finite by construction; this ENFORCES that invariant at the write boundary so a
    gate/clip regression fails at the campaign write, not silently downstream (iter 270).
    """
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if not bool(np.all(np.isfinite(a))):
        n_bad = int(np.count_nonzero(~np.isfinite(a)))
        raise ValueError(
            f"build_campaign_output_dict: corrected field {name!r} has {n_bad} "
            "non-finite (NaN/inf) value(s) — an un-clipped/ill-posed diagnosis reached "
            "the final field. The accept gate should keep it finite; a non-finite "
            "coefficient cannot be serialized (unparseable JSON) nor deployed.")
    return arr


def _averaging_provenance(args, base_cfg=None) -> dict:
    """The comparison's averaging windows — recorded in the output so the empirical
    result is SELF-DESCRIBING + reproducible.  A bias computed against a single ERA5
    SNAPSHOT (``era5_n_times=1``) is a different scientific quantity than one against an
    N-time CLIMATOLOGY (iter 140); the time indices + count make 'what was this bias
    measured against' explicit for the analyst, not implicit in the launch command.

    When ``base_cfg`` is given (the campaign output + dry-run; iter 313), ALSO record the
    MODEL-side averaging window — the model time-mean is over the run length ``days``,
    sampled every ``diag_days`` — so a deployer can confirm the model climatology window
    is comparable to the ERA5 one (BOTH time-means, the iter-267 window-alignment is
    TWO-sided), and a suspiciously short model window is visible.  ``base_cfg=None`` (the
    cross-grid env-kernel write) omits it, since that kernel deploys on a different grid
    whose own model window differs.
    """
    block = {
        "era5_time_idx": int(args.era5_time_idx),
        "era5_n_times": int(_resolve_era5_n_times(args.era5_n_times)),
    }
    # OFFLINE-only: the number of ERA5 DAYS loaded (iter 460/470) — the reference's coverage
    # (~24*era5_n_days hourly times available). Makes the offline multi-day reference visible in
    # the dry-run + output (the Zarr path spans times itself, so era5_n_days is N/A there).
    if getattr(args, "local_era5_dir", None):
        block["era5_n_days"] = int(getattr(args, "era5_n_days", 1) or 1)
    if base_cfg is not None:
        diag = int(getattr(base_cfg.output, "diag_days", 0))
        days = int(base_cfg.days)
        block["model_days"] = days
        block["model_diag_days"] = diag
        block["model_n_samples"] = (days // diag) if diag > 0 else None
    return block


def _les_provenance(args) -> dict:
    """The spin-off LES configuration — recorded in the output so the correction is
    SELF-DESCRIBING + reproducible (symmetric with :func:`_averaging_provenance`).

    Whether the LES used a prescribed SURFACE-FLUX BC (iter 364/365 — surface-driven
    turbulence for convective columns) changes HOW the closure was diagnosed (it raises
    the diagnosis-validity rate for surface-driven columns), so a later analysis can
    distinguish surface-flux runs from the on-disk artifact rather than the launch command;
    the cost/run knobs complete the LES-setup record."""
    return {
        "surface_flux": bool(args.surface_flux),
        "les_budget": (None if args.les_budget is None else int(args.les_budget)),
        "les_dt_s": float(args.les_dt),
        "les_hours": float(args.les_hours),
    }


def build_campaign_output_dict(result, *, grid_provenance, summary, health,
                               corrected_field=None, coefficients=None,
                               averaging=None, les_provenance=None, les_realism=None):
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
    # Sanitize the per-round bias trajectory: a DIVERGED round (NaN/±inf bias) would
    # otherwise emit a non-standard ``NaN``/``Infinity`` token here, making the whole
    # output FILE unparseable by the plotters / deploy reader (``json.load``) — the
    # same hazard ``_json_finite`` already guards in ``_summary_to_json`` (iter 245).
    biases = [(_json_finite(it.bias.baseline_bias), _json_finite(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    if corrected_field is not None:
        # NO reshape (behavior-preserving): the single final_config field is the
        # per-column 1-D array; ``.tolist()`` keeps a 0-D scalar (a no-op / zero-round
        # campaign that corrected NOTHING) as a bare float so the deploy loader's
        # 1-D assertion REJECTS it LOUDLY rather than silently promoting it to a
        # length-1 single-column array (Codex iter 105).
        arr = np.asarray(getattr(result.final_config, corrected_field))
        _assert_corrected_field_finite(corrected_field, arr)
        payload = {corrected_field: arr.tolist()}
    else:
        payload = {
            "coefficients": list(coefficients),
            "fields": {k: _assert_corrected_field_finite(k, np.asarray(v)).reshape(-1).tolist()
                       for k, v in result.final_fields.items()},
        }
    out = {**payload,
           "grid": grid_provenance,
           "biases": biases, "accepted": accepted, "step_fractions": steps,
           "summary": _summary_to_json(summary),
           "health": {"status": health.status, "message": health.message}}
    if averaging is not None:                        # comparison averaging-window provenance
        out["averaging"] = averaging
    if les_provenance is not None:                   # spin-off LES config (incl. surface_flux)
        out["les_config"] = les_provenance
    if les_realism is not None:                       # per-mode LES-realism rejection counts
        out["les_realism"] = les_realism
    return out


def _warn_if_ignored_diagnosis_method(coefficients, diagnosis_method: str) -> None:
    """Warn when ``--coefficients`` is given ALONGSIDE a NON-default ``--diagnosis-method``.

    The ``--coefficients`` (multi-coefficient) path corrects EXACTLY those coefficients
    and IGNORES ``--diagnosis-method`` — so a user who set a specific method expecting it
    to apply would otherwise be silently surprised (e.g. ``--coefficients C_K,Pr_t
    --diagnosis-method c_eps`` corrects C_K + Pr_t, NOT c_eps).  ``clubb_coefficient`` is
    the default, so it cannot be told apart from 'unset' — only a non-default method is
    flagged (avoids a spurious warning on every multi run).
    """
    import warnings

    if coefficients is not None and diagnosis_method != "clubb_coefficient":
        warnings.warn(
            f"--diagnosis-method={diagnosis_method!r} is IGNORED because --coefficients "
            f"was given; the multi-coefficient campaign corrects exactly "
            f"{coefficients!r}. Drop one of the two to remove this ambiguity.",
            stacklevel=2)


def _les_n_steps(les_hours: float, les_dt: float) -> int:
    """LES step count from ``--les-hours`` / ``--les-dt``, with a fail-loud guard.

    ``n = int(les_hours·3600 / les_dt)``.  A too-short ``--les-hours`` (or too-large
    ``--les-dt``) yields ``n < 1`` ⇒ ``run_forced_les``'s ``for _ in range(n)`` runs
    ZERO steps, the column LES develops NO turbulence, every diagnosis is invalid, and
    the multi-day campaign silently corrects NOTHING (only a ``no_valid_diagnoses``
    health verdict at the end).  Catch the misconfiguration at LAUNCH instead.  Shared
    by the campaign + perfect-model OSSE CLIs.
    """
    import math

    # A non-positive (or NaN) dt/duration is a clean fail-loud, not a bare
    # ZeroDivisionError (les_dt==0) or a confusing negative-step message (les_dt<0): both
    # are misconfigurations the launch must reject (``not (x > 0)`` also catches NaN).
    if not (les_dt > 0.0) or math.isnan(les_dt):
        raise SystemExit(f"--les-dt {les_dt} must be > 0 (the LES timestep [s]).")
    if not (les_hours > 0.0) or math.isnan(les_hours):
        raise SystemExit(f"--les-hours {les_hours} must be > 0 (the LES duration [h]).")
    n = int(les_hours * 3600.0 / les_dt)
    if n < 1:
        raise SystemExit(
            f"--les-hours {les_hours} / --les-dt {les_dt} ⇒ {n} LES steps; the forced "
            "LES needs ≥ 1 step (it would otherwise develop no turbulence and diagnose "
            "nothing). Increase --les-hours or decrease --les-dt.")
    return n


def _enable_line_buffered_stdout() -> None:
    """Line-buffer stdout so a multi-day SLURM run's per-round progress appears in the
    job log in REAL TIME.

    When stdout is redirected to a file (the SLURM log), Python BLOCK-buffers it, so the
    per-round bias prints would otherwise stay invisible until the buffer fills (many
    rounds) or the job exits — a user could not tell a 3-day run apart from a hung one.
    No-op when stdout lacks ``reconfigure`` (e.g. a captured / replaced stream).
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)


def _configure_jax_compilation_cache(cache_dir, min_compile_secs: float = 30.0):
    """Enable JAX's PERSISTENT on-disk compilation cache so the EXPENSIVE radiation JIT
    (rrtmgp: >16 min, iter 443) is written ONCE and REUSED.

    Chiefly de-risks the sbatch's SELF-REQUEUE: a requeued multi-day job re-runs from the
    last checkpoint but otherwise recompiles every segment from scratch — with the cache it
    reuses the compiles the killed job already paid for. Also helps repeated launches with
    the same config and any C_K-INDEPENDENT sub-compiles. (The deeper "compile ONCE via a
    traced C_K input so every round shares one graph" is a separate architecture change —
    the campaign rebuilds the driver per config, so each distinct C_K array currently bakes
    a new constant and recompiles; tracked as CODEX PENDING in docs/COMPARE_REANALYSIS.md.)

    ``min_compile_secs`` caches only compiles SLOWER than this (default 30 s targets the
    radiation graph, not trivial compiles). Empty ``cache_dir`` => no-op (disabled). The
    cache key includes the HLO + jaxlib version + backend/platform, so a code change or a
    different node type MISSES (recompiles) rather than serving a stale / wrong-arch binary.
    MUST run before the first JAX compilation — the campaign calls it at the top of main(),
    before any driver build. Returns the configured dir (or ``None`` when disabled)."""
    if not cache_dir:
        return None
    import jax

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", float(min_compile_secs))
    print(f"[campaign] JAX persistent compilation cache: {cache_dir} (caching compiles > "
          f"{min_compile_secs:g}s) — amortizes the rrtmgp JIT across the self-requeue + "
          "repeated launches.", flush=True)
    return str(cache_dir)


def _print_round_progress(round_idx: int, res: Any, total_rounds: int) -> None:
    """Print a per-round progress line to (line-buffered) stdout DURING the run.

    The campaign loop is otherwise SILENT until it returns (the per-round bias table prints
    only at the end), so a multi-day SLURM run's ``.out`` log showed NO progress —
    :func:`_enable_line_buffered_stdout`'s real-time-progress promise needs something printed
    PER ROUND (iter 318).  Lets the operator tell a running campaign from a hung one and watch
    the bias fall live.  Works for the single + multi result (both carry ``bias`` /
    ``n_corrected`` / ``n_diagnoses_valid`` / ``step_fraction``).

    NOTE (round numbering, iter 326): the live line is DELIBERATELY 1-based ("round {idx+1}/N",
    natural progress) while the final per-round table (:func:`_format_round_line`) and the
    checkpoint/resume arithmetic use the 0-based canonical ``round_idx`` (``start_round + i``).
    So live "round 3/10" is the SAME round the summary calls "round 2" and the checkpoint stores
    as ``round=2`` — an intentional display offset, not a desync; keep them in sync if changed."""
    b = res.bias
    step = float(getattr(res, "step_fraction", 1.0))
    if not bool(b.improved):
        status = "rejected"
    elif step < 1.0 - 1e-9:
        # The line search BACKTRACKED the full LES step — the raw diagnosis was too
        # aggressive (the full step did not improve, a partial one did): a signal about
        # the LES↔GCM closure transfer the operator wants live.
        status = f"kept, step {step:.2g}"
    else:
        status = "kept"
    print(f"[campaign] round {int(round_idx) + 1}/{int(total_rounds)}: bias "
          f"{float(b.baseline_bias):.5g} -> {float(b.updated_bias):.5g} ({status}); "
          f"{int(res.n_corrected)} cols corrected, "
          f"{int(res.n_diagnoses_valid)} valid LES diagnoses")


def _assert_output_path_writable(path: str, *, flag: str) -> None:
    """Fail LOUD up front if an output ``path``'s parent directory is missing or not
    writable.

    The campaign opens ``--out`` only at the very END of the (multi-day) run, so a
    typo'd or unwritable directory would otherwise crash ``json.dump`` AFTER burning
    the whole run.  This millisecond pre-flight catches it at launch instead.  The
    parent dir is NOT auto-created (a missing dir is far more likely a typo than the
    user's intent — surfacing it is safer than silently scattering output).
    """
    import os

    parent = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(parent):
        raise SystemExit(
            f"--{flag} {path!r}: parent directory {parent!r} does not exist — create "
            f"it or fix the path before launching (the campaign writes --{flag} only "
            f"at the END of the multi-day run).")
    if not os.access(parent, os.W_OK):
        raise SystemExit(
            f"--{flag} {path!r}: parent directory {parent!r} is not writable — fix "
            f"permissions or choose a writable path before launching.")


def _assert_era5_zarr_readable(path: str) -> None:
    """Fail LOUD up front if a LOCAL ``--era5-zarr`` reference store does not exist.

    The ERA5 reference is opened only AFTER the model/grid build + driver setup
    (``load_era5_time_mean``), so a typo'd LOCAL path would otherwise crash with a
    cryptic zarr/fsspec error AFTER burning that setup — wasting cluster time at the
    start of a multi-day run.  This millisecond pre-flight catches it at launch,
    mirroring the ``--out`` / ``--checkpoint`` guards.

    A REMOTE URI (``gs://`` / ``s3://`` / ``http://`` / …, detected by ``"://"``) is
    NOT existence-checked here — the fsspec/zarr backend resolves it — so a valid
    CLOUD store is NEVER falsely rejected (a false positive would be strictly worse
    than the late error it replaces).
    """
    import os

    if "://" in path:               # remote URI: leave resolution to the storage backend
        return
    if not os.path.exists(path):    # a Zarr is a directory (or a .zip) on the local FS
        raise SystemExit(
            f"--era5-zarr {path!r}: no ERA5 reference store at that path (a Zarr is a "
            "directory or .zip). Fix the path before launching, or use a gs://, s3://, "
            "or http:// URI for a remote store.")


def _warn_if_grid_exceeds_era5_lat_coverage(era5_lat_rad, model_lat_rad) -> None:
    """WARN (do NOT block) when the model grid's LATITUDE extent reaches MATERIALLY
    beyond the ERA5 reference's coverage — the tell-tale of a REGIONAL ``--era5-zarr``.

    Unlike a typo'd path (which crashes), a regional store loads fine and the IDW
    regrid SILENTLY EXTRAPOLATES a GARBAGE reference outside its domain — the worst
    failure mode: a whole multi-day run wasted on wrong data with no error.  This
    surfaces it BEFORE the run.  Tolerance = 2× the ERA5 latitude spacing (the IDW
    handles a cell or two of edge extrapolation), so a near-GLOBAL ERA5 whose pole
    rows sit just inside the model's is NEVER flagged.  A WARNING, not an error: an
    operator deliberately running a regional domain may proceed; longitude is not
    checked here (periodic; a regional lon box is far rarer than a lat band).  Both
    inputs are RADIANS (``ERA5Slice.lat`` and ``grid.grid_lat``)."""
    import warnings

    import numpy as np

    e = np.rad2deg(np.asarray(era5_lat_rad, dtype=float)).ravel()
    g = np.rad2deg(np.asarray(model_lat_rad, dtype=float)).ravel()
    if e.size < 2 or g.size == 0:
        return
    uniq = np.unique(e)
    spacing = float(np.median(np.abs(np.diff(uniq)))) if uniq.size > 1 else 0.0
    tol = 2.0 * spacing
    e_lo, e_hi = float(e.min()), float(e.max())
    over = max((e_lo - tol) - float(g.min()), float(g.max()) - (e_hi + tol), 0.0)
    if over > 0.0:
        warnings.warn(
            f"--era5-zarr latitude coverage [{e_lo:.1f}, {e_hi:.1f}]deg does NOT span "
            f"the model grid (which reaches ~{over:.0f}deg beyond it): the reference is "
            "EXTRAPOLATED there. A REGIONAL ERA5 store yields a GARBAGE reference "
            "outside its domain — verify the store is GLOBAL before a multi-day run.",
            stacklevel=2)


def _fsync_dir(parent: str) -> None:
    """Best-effort ``fsync`` of a directory so a rename within it is crash-durable.

    Some filesystems (or platforms) do not support a directory ``fsync``; swallow that
    ``OSError`` rather than fail an otherwise-successful atomic write.  Used by
    :func:`_atomic_write_json` after ``os.replace`` to persist the directory ENTRY.
    """
    import os

    try:
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _atomic_write_json(path: str, obj: Any, *, indent: int | None = None) -> None:
    """Write ``obj`` as JSON to ``path`` ATOMICALLY + DURABLY: serialize to a temp file
    in the SAME directory, flush + ``fsync``, ``os.replace`` it onto ``path``, then
    ``fsync`` the parent directory so the rename survives a crash.

    A crash / SLURM-kill mid-write therefore leaves the PREVIOUS file intact — a
    partial write can never corrupt the checkpoint a multi-day resume depends on
    (``json.load`` on a half-written file would otherwise abort the restart). The
    temp file shares ``path``'s directory so ``os.replace`` stays on ONE filesystem
    (where it is atomic); a failed write unlinks the temp rather than leaking it.
    """
    import json
    import os
    import stat
    import tempfile

    parent = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp_campaign_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates 0o600; ``os.replace`` swaps the inode, so without this the
        # output/checkpoint would silently become owner-only (vs ``open(path,"w")``'s
        # umask default — a shared-HPC reader would be locked out). Preserve an existing
        # file's mode (the per-round re-checkpoint), else the umask default for a new one.
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            cur_umask = os.umask(0)
            os.umask(cur_umask)
            mode = 0o666 & ~cur_umask
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        # Durably persist the RENAME, not just the file's data: ``os.replace`` is
        # atomic (the path never points at a half-written inode), but the directory
        # ENTRY change is not crash-durable until the PARENT directory is fsynced.
        # Without this a crash/SLURM-kill right after the replace can revert to the
        # PREVIOUS checkpoint, losing the last completed (expensive) round.
        _fsync_dir(parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _les_per_round_estimate(les_budget: int | None, n_worst: int) -> int:
    """Upper-bound LES/round for the ``--dry-run`` compute estimate.

    Clustering picks ``les_budget`` REPRESENTATIVES but is CAPPED at the number of
    worst columns (``cluster_columns_by_environment`` returns ``min(les_budget,
    len(records))`` reps — never duplicates), and there are at most ``n_worst`` worst
    columns.  So the true upper bound is ``min(les_budget, n_worst)``, not
    ``les_budget``: a ``--les-budget 50 --n-worst 20`` run spins off ≤ 20 LES/round,
    and the pre-flight must not over-state the multi-day compute by 2.5×.  With no
    budget (no clustering) it is one LES per worst column, ``n_worst``.
    """
    if les_budget is None:
        return int(n_worst)
    return min(int(les_budget), int(n_worst))


def _dry_run_era5_line(era5: dict | None) -> str:
    """The ``--dry-run`` ERA5-reference line, or ``""`` when no averaging is given.

    Surfaces the comparison window at PRE-FLIGHT — the earliest point an operator can
    fix a snapshot-vs-climatology misconfiguration (iter 275, completing the iter-267
    provenance + §3 window-alignment note).  ``era5_n_times == 1`` is a single SNAPSHOT;
    compared to a multi-day model time-mean that is weather-vs-climate (a spurious bias
    the loop would 'correct'), so it is flagged with a WARNING here, not discovered after
    a multi-day run."""
    if not era5:
        return ""
    n = int(era5.get("era5_n_times", 1))
    idx = era5.get("era5_time_idx")
    at = "" if idx is None else f" @ idx {int(idx)}"
    nd = era5.get("era5_n_days")                          # OFFLINE day coverage (iter 460/470)
    over = "" if nd is None else f" over {int(nd)} day(s) of offline ERA5"
    md = era5.get("model_days")                          # the MODEL-side window (iter 313)
    model = "" if md is None else (
        f"; model mean over {int(md)} days ({era5.get('model_n_samples')} samples @ "
        f"{era5.get('model_diag_days')}-day cadence)")
    if n == 1:
        return (f"\n  ERA5 reference: SINGLE snapshot{at}{over}{model} — WARNING: a snapshot vs a "
                "multi-day model mean is weather-vs-climate; use --era5-n-times N "
                "(runbook §3).")
    return f"\n  ERA5 reference: {n}-time climatology{at}{over}{model}"


def _dry_run_amip_forcing_line(amip_forcing: dict | None) -> str:
    """The ``--dry-run`` line for an ``--amip-forcing-from-local-era5`` build, or ``""``.

    The forcing is BUILT during the pre-flight (``_maybe_apply_local_era5_forcing`` runs
    before the dry-run short-circuit), so reaching this line means the local archive was
    readable and the combined SST/sea-ice forcing was written — the operator's confirmation
    that the OFFLINE realistic-AMIP boundary condition is in place before the multi-day run
    (iter 424)."""
    if not amip_forcing:
        return ""
    return (
        f"\n  AMIP forcing: built from {amip_forcing['source']} ({amip_forcing['date']}) "
        f"→ {amip_forcing['out']} (SSTK/CI, every {int(amip_forcing['hour_stride'])}h)")


def _dry_run_report(dry: CampaignDryRun, *, mode: str, out: str,
                    era5: dict | None = None,
                    amip_forcing: dict | None = None) -> str:
    """Human-readable one-block summary of a successful ``--dry-run`` pre-flight: the
    campaign CONSTRUCTED (config/units/grid/scheme/method all validated), here is what
    a real launch WOULD run.  Pure (no I/O) so it is unit-testable."""
    return (
        "[campaign] DRY-RUN OK — construction validated, NOT run.\n"
        f"  mode={mode}  grid_shape={tuple(dry.grid_shape)}  n_worst={dry.n_worst}\n"
        f"  coefficients={tuple(dry.coefficients)}  "
        f"feedback_strategy={dry.feedback_strategy}  "
        f"surface_flux={'ON' if dry.surface_flux else 'off'}\n"
        f"  would run ≤ {dry.n_iterations} rounds × ≤ {dry.les_per_round} LES/round "
        f"(≤ {dry.n_iterations * dry.les_per_round} LES total)"
        + _dry_run_era5_line(era5)
        + _dry_run_amip_forcing_line(amip_forcing) + "\n"
        # The dominant runtime cost is the LES spin-off (a 3D plane LES per worst column), NOT
        # the AMIP/CMIP model runs — measured (iter 499/501): even a tiny res-4 gray campaign was
        # >10 min CPU, LES-dominated. Surface it so the operator budgets HPC on the LES count.
        "  COST: the LES spin-offs dominate the runtime (each is a 3D plane LES; the AMIP/CMIP "
        f"runs are a small fraction) — budget HPC on the ≤ "
        f"{dry.n_iterations * dry.les_per_round} LES total (iter 499/501).\n"
        f"  would write → {out}\n"
        "  (re-run without --dry-run to execute the multi-day campaign.)"
    )


def _area_weights(grid):
    """Per-column quadrature weights for the bias aggregation.

    Prefers the grid's true cell areas (``grid_area`` — incl. Gaussian quadrature
    weights — or ``area``); falls back to cos-latitude (a lat-lon proxy) with a
    warning if the grid exposes neither.  ``grid.grid_lat`` is stored in RADIANS
    across every grid family (lat-lon/cubed-sphere/Gaussian/Voronoi/plane), so the
    cos-latitude weight reads it directly — NO ``deg2rad`` (applying it would
    shrink the angle ~57x and collapse the weights to a near-uniform ≈1).
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
    return jnp.cos(jnp.asarray(np.asarray(grid.grid_lat)))  # grid_lat is [rad]


def _format_resume_line(resume_path: str, start_round: int, seed: dict,
                        *, multi: bool = False) -> str:
    """Resume console line surfacing the CHECKPOINTED PROGRESS, so an operator restarting
    a multi-day run after a SLURM timeout sees the invested work (the cumulative LES count
    + the campaign-start bias) — confirming a valid resume picking up real progress, not a
    cold start at round N.  Both metrics are already in the checkpoint seed (iters 137/138)
    but were discarded at the resume print (iter 274)."""
    ib = seed.get("initial_bias")
    nd = int(seed.get("n_diagnosed_prior", 0))
    bias = f"start bias {float(ib):.4g}" if ib is not None else "start bias unknown"
    mode = "multi " if multi else ""
    return (f"[campaign] resuming {mode}from {resume_path} at round {start_round} "
            f"({bias}; {nd} LES diagnoses done so far)")


def _format_round_line(round_idx: int, b0: float, b1: float, imp: bool, kept: bool,
                       step: float, n_valid: int, n_diagnosed: int) -> str:
    """One-line per-round campaign progress for the operator's multi-day-run console.

    Surfaces the LES-diagnosis VALIDITY (``n_valid/n_diagnosed``) alongside the bias
    move so a round that made NO correction because the spin-off LES developed no
    turbulence (``0/N`` valid — a forcing/setup issue) is distinguishable from one where
    the correction simply did not lower the bias (``N/N`` valid but ``no improvement`` —
    a science result).  Without it both read identically as 'no improvement; REJECTED'.
    """
    return (f"[campaign] round {round_idx}: bias {b0:.5g} -> {b1:.5g} "
            f"(step {step:.3g}; {'IMPROVED' if imp else 'no improvement'}; "
            f"{'kept' if kept else 'REJECTED'}; {n_valid}/{n_diagnosed} LES valid)")


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
                    extract_fn, run_les, *, phis=None,
                    valid_mask=None):  # pragma: no cover - heavy I/O
    """Multi-coefficient campaign entry (``--coefficients``): a dict checkpoint /
    resume / output for the per-coefficient accumulated fields.  ``phis`` (the
    model's static orographic topography, resolved once in :func:`main`) is
    forwarded to the multi-coefficient campaign builder so the orographic
    LES-forcing term activates identically to the single-coefficient path."""
    import json

    import numpy as np
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig

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
        initial_clubb, initial_fields, start_round, _seed = _load_multi_resume(
            ckpt, grid, coefficients, gshape, promo_to_field)
        init_box.update(_seed)
        print(_format_resume_line(args.resume, start_round, _seed, multi=True))

    _ckpt_write = None
    if args.checkpoint:
        def _ckpt_write(round_idx, res, fields):
            _capture_initial_record(init_box, res)
            # field finiteness asserted (NaN ⇒ unparseable checkpoint / corrupt resume) —
            # the checkpoint analog of the iter-270/245 output guards; the round/grid/
            # bias/count keys are the shared _checkpoint_common (iter 293).
            _atomic_write_json(args.checkpoint, {
                **_checkpoint_common(round_idx, base_cfg, grid, init_box),
                "coefficients": list(coefficients),
                "fields": {k: _assert_corrected_field_finite(k, np.asarray(v)).reshape(-1).tolist()
                           for k, v in fields.items()},
            })

    # Per-round hook: ALWAYS print real-time progress (iter 318); checkpoint only if requested.
    _total_rounds = start_round + int(args.iterations)

    def checkpoint_callback(round_idx, res, fields):
        _print_round_progress(round_idx, res, _total_rounds)
        if _ckpt_write is not None:
            _ckpt_write(round_idx, res, fields)

    result = build_multi_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma, grid=grid,
        area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(surface_flux=args.surface_flux),
        coefficients=coefficients,
        run_les_fn=run_les, phis=phis, valid_mask=valid_mask,
        initial_clubb=initial_clubb, initial_fields=initial_fields,
        start_round=start_round, checkpoint_callback=checkpoint_callback,
        sequential=args.staged, dry_run=args.dry_run,
        **_campaign_knobs_from_args(args))
    if args.dry_run:
        print(_dry_run_report(result, mode=args.mode, out=args.out,
                              era5=_averaging_provenance(args, base_cfg),
                              amip_forcing=_amip_forcing_provenance(args)))
        return 0

    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    for i, (b0, b1, imp) in enumerate(biases):
        kept = accepted[i] if i < len(accepted) else True
        it_i = result.iterations[i]
        print(_format_round_line(
            start_round + i, b0, b1, imp, kept, steps[i],
            getattr(it_i, "n_diagnoses_valid", 0), getattr(it_i, "n_diagnosed", 0)))
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
    _atomic_write_json(args.out, build_campaign_output_dict(
        result, grid_provenance=_grid_provenance(base_cfg, grid),
        summary=summary, health=health, coefficients=coefficients,
        averaging=_averaging_provenance(args, base_cfg),
        les_provenance=_les_provenance(args),
        les_realism=_realism_summary_dict(getattr(run_les, "breakdowns", None))), indent=2)
    print(f"[campaign] wrote corrected multi-coefficient config to {args.out}")
    print(summary.report())
    print(f"[campaign] {health.status.upper()}: {health.message}")
    print_realism_summary(run_les)
    _print_deploy_hint(args.out, grid)
    # Exit code = the health verdict (iter 287): 0 only when the run IMPROVED, non-zero
    # otherwise, so an HPC workflow gating on `run_campaign && deploy` does NOT deploy a
    # no-op/diverged correction. The deployable JSON is written either way.
    return _campaign_exit_code(health)


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


def _checkpoint_common(round_idx, base_cfg, grid, init_box) -> dict:
    """The round / grid / initial-bias / diagnosis-count keys SHARED by the single- and
    multi-coefficient checkpoint callbacks (iter 293) — the field part (``corrected_field``
    /``field`` vs ``coefficients``/``fields``) is per-callback.  ``initial_bias`` is
    ``_json_finite``-sanitised (a diverged initial is a recorded ``null``, not a bug — iter
    245/271); the cumulative diagnosis counts thread the resume seed (iter 137/138).  Keeps
    the checkpoint-key wiring in ONE place (CLAUDE.md: no duplicate wiring)."""
    return {
        "round": int(round_idx),
        "grid": _grid_provenance(base_cfg, grid),
        "initial_bias": _json_finite(init_box["initial_bias"]),
        "initial_per_variable": init_box["initial_per_variable"],
        "n_diagnosed_total": (init_box["n_diagnosed_prior"]
                              + init_box.get("n_diag_seg", 0)),
        "n_diagnoses_valid_total": (init_box["n_diagnoses_valid_prior"]
                                    + init_box.get("n_valid_seg", 0)),
    }


def _assert_resume_grid_matches(ckpt: dict, grid: Any) -> None:
    """Fail LOUD on a RESUME whose checkpoint was written on a DIFFERENT grid than the
    current run.

    The accumulated per-column field is reshaped onto ``grid.grid_shape_2d``; a
    different-SHAPE-but-same-SIZE grid (e.g. ``(6,8)`` vs ``(8,6)``) would reshape
    SUCCESSFULLY yet scramble the field onto the WRONG cells — the exact silent-
    wrong-cell hazard the deploy guard catches with a coordinate fingerprint.  Compares
    the checkpoint's recorded grid fingerprint (``ncol``/``shape_2d``/``coord_sha256``)
    to the current grid's.  A checkpoint predating provenance (no ``"grid"`` block) only
    WARNS — the field-length reshape stays the fallback size check.
    """
    import warnings

    from legoesm.training.deploy_correction import grid_fingerprint

    rec = ckpt.get("grid")
    if not isinstance(rec, dict) or not rec.get("coord_sha256"):
        warnings.warn(
            "resume: checkpoint has no grid fingerprint (an older checkpoint) — the "
            "grid identity cannot be verified, relying on the per-column field length "
            "alone. Resume on the SAME grid the campaign used.", stacklevel=2)
        return
    cur = grid_fingerprint(grid)
    for key in ("ncol", "shape_2d", "coord_sha256"):
        rv = rec.get(key)
        cv = cur[key]
        if key == "shape_2d" and rv is not None:
            rv = list(rv)
        if rv is not None and rv != cv:
            raise SystemExit(
                f"resume grid mismatch on {key!r}: checkpoint recorded {rv!r} but the "
                f"current grid has {cv!r}. The accumulated per-column field was built on "
                "a DIFFERENT grid and would resume onto the WRONG cells; resume on the "
                "SAME grid the campaign used.")


def _resume_seed(ckpt: dict) -> dict:
    """The ``init_box`` seed read from a checkpoint on resume — the READ counterpart of
    :func:`_checkpoint_common` (iter 294): the cumulative diagnosis counts (the write's
    ``n_*_total`` → the read's ``n_*_prior``) plus the campaign-start bias / per-variable,
    so the trajectory + LES-validity counts stay CUMULATIVE across a SLURM-timeout resume
    (iter 137/138).  Shared by both resume loaders (CLAUDE.md: no duplicate wiring)."""
    return {
        "initial_bias": ckpt.get("initial_bias"),
        "initial_per_variable": ckpt.get("initial_per_variable"),
        "n_diagnosed_prior": int(ckpt.get("n_diagnosed_total", 0)),
        "n_diagnoses_valid_prior": int(ckpt.get("n_diagnoses_valid_total", 0)),
    }


def _load_single_resume(ckpt: dict, grid: Any, corrected_field: str):
    """Reconstruct the SINGLE-coefficient resume state from a checkpoint dict.

    Validates the grid fingerprint (wrong-cell guard) + the corrected-coefficient
    identity (wrong-slot guard), rebuilds the accumulated per-column FIELD (the single
    source of truth — the CLUBB config is exactly its flattened form, so they cannot
    desync even after a rejected-round checkpoint) and the resume round, and returns
    ``(initial_field, initial_clubb, start_round, init_seed)`` where ``init_seed``
    carries the restored campaign-START baseline + prior cumulative diagnosis counts
    (absent in pre-iter-137 checkpoints ⇒ ``None``/0 fallback to segment-only).  Raises
    ``SystemExit`` on a grid / corrected-field mismatch.
    """
    import jax.numpy as jnp

    _assert_resume_grid_matches(ckpt, grid)            # wrong-cell guard (same as deploy)
    if "field" not in ckpt:
        # A MULTI-coefficient checkpoint carries a ``"fields"`` dict, not a single
        # ``"field"`` — resuming it as single (forgot --coefficients) would otherwise be a
        # bare ``KeyError``; fail loud with the fix, symmetric with the multi path's
        # coefficient-set guard (iter 252).
        hint = (" (it has a 'fields' dict — resume WITH the SAME --coefficients)"
                if "fields" in ckpt else "")
        raise SystemExit(
            f"single-coefficient resume: checkpoint has no 'field'{hint}; resume the run "
            "in the mode it was written (single-coefficient: no --coefficients).")
    ckpt_field = ckpt.get("corrected_field")
    if ckpt_field is not None and ckpt_field != corrected_field:
        raise SystemExit(
            f"checkpoint corrects {ckpt_field!r} but --diagnosis-method "
            f"requests {corrected_field!r}; resume with the matching method.")
    initial_field = jnp.asarray(ckpt["field"]).reshape(grid.grid_shape_2d)
    initial_clubb = CLUBBLiteConfig(**{corrected_field: initial_field.reshape(-1)})
    start_round = int(ckpt["round"]) + 1
    init_seed = _resume_seed(ckpt)
    return initial_field, initial_clubb, start_round, init_seed


def _load_multi_resume(ckpt: dict, grid: Any, coefficients, gshape, promo_to_field: dict):
    """Reconstruct the MULTI-coefficient resume state — the symmetric analog of
    :func:`_load_single_resume`.

    Validates the grid fingerprint (wrong-cell guard) + the coefficient SET (a mismatched
    set would load the wrong per-coefficient fields into the config), rebuilds the
    accumulated per-coefficient ``initial_fields`` (``{promotion_key: 2-D field}``) + the
    CLUBB config FROM those fields (single source of truth), and returns
    ``(initial_clubb, initial_fields, start_round, init_seed)`` with the restored
    campaign-START baseline + cumulative diagnosis counts.  Raises ``SystemExit`` on a
    grid / coefficient-set mismatch.
    """
    import jax.numpy as jnp

    _assert_resume_grid_matches(ckpt, grid)            # wrong-cell guard (same as deploy)
    if sorted(ckpt.get("coefficients", [])) != sorted(coefficients):
        raise SystemExit(
            f"checkpoint coefficients {ckpt.get('coefficients')} != requested "
            f"{list(coefficients)}; resume with the matching set.")
    initial_fields = {
        k: jnp.asarray(v).reshape(gshape) for k, v in ckpt["fields"].items()}
    overrides = {
        promo_to_field[k]: jnp.asarray(v).reshape(-1)
        for k, v in ckpt["fields"].items()}
    initial_clubb = CLUBBLiteConfig(**overrides)
    start_round = int(ckpt["round"]) + 1
    init_seed = _resume_seed(ckpt)
    return initial_clubb, initial_fields, start_round, init_seed


def _campaign_exit_code(health) -> int:
    """CLI exit status from the campaign health verdict (iter 287): 0 when the run
    IMPROVED (``health.ok``), 1 otherwise (stalled / no_change / no_valid_diagnoses /
    non_finite_bias).  Mirrors the OSSE go/no-go so an HPC workflow can gate ``run_campaign
    && deploy`` on a real improvement instead of deploying a no-op or diverged run."""
    return 0 if health.ok else 1


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

    with open(config_path) as f:
        base_cfg = experiment_config_from_dict(json.load(f))
    grid, sigma = _build_grid_for_config(base_cfg)
    return base_cfg, grid, sigma


def _build_grid_for_config(cfg):
    """Build the grid + vertical coordinate for ``cfg`` the EXACT way the run does.

    Factored from :func:`load_base_config_and_grid` so the cross-resolution OSSE CLI
    can build a FINE grid from the same base config at a different resolution
    (``cfg.grid._replace(resolution=...)``) the identical way — the driver's own grid
    constructor + the precision policy (``_bootstrap_runtime`` first so a
    non-default-precision sigma gets the right dtype), without the heavy ``setup()``.
    Returns ``(grid, sigma)``.
    """
    from legoesm.driver.model_driver import ModelDriver

    probe = ModelDriver(cfg)
    probe._bootstrap_runtime()
    probe._create_grid()
    return probe.grid, probe.sigma


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
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig
    from legoesm.training.compare_reanalysis import column_state_from_carry
    from legoesm.training.era5_to_state import (
        TrainingERA5Config,
        load_era5_time_mean,
    )

    from scripts.validate.compare_amip_era5 import (
        canonical_grid_type,
        select_era5_regrid,
    )

    _enable_line_buffered_stdout()   # real-time per-round progress in the SLURM log
    refuse_unsupported_multirank()   # single-process CLI: refuse mpirun -np >1
    args = _build_arg_parser().parse_args(argv)
    # Enable the persistent compilation cache FIRST — before any driver build / JIT — so the
    # expensive rrtmgp compile is reused across the self-requeue + rounds (iter 453).
    _configure_jax_compilation_cache(
        args.compilation_cache_dir, args.cache_min_compile_secs)
    _warn_if_ignored_diagnosis_method(args.coefficients, args.diagnosis_method)
    # ERA5 reference source: EXACTLY one of --era5-zarr (store/network) or
    # --local-era5-dir (offline NCAR-RDA NetCDF, iter 410) — fail loud at launch on a
    # missing/ambiguous source rather than a cryptic load error after the model build.
    if bool(args.era5_zarr) == bool(args.local_era5_dir):
        raise SystemExit(
            "run_correction_campaign: pass EXACTLY one ERA5 reference source — "
            "--era5-zarr <store> OR --local-era5-dir <dir> --local-era5-date YYYYMMDD "
            f"(got era5_zarr={args.era5_zarr!r}, local_era5_dir={args.local_era5_dir!r}).")
    if args.local_era5_dir and not args.local_era5_date:
        raise SystemExit(
            "run_correction_campaign: --local-era5-dir requires --local-era5-date "
            "YYYYMMDD (which day's 24 hourly ERA5 times to use).")
    # --amip-forcing-from-local-era5: only meaningful for a prescribed-SST AMIP run from
    # the local archive — fail loud at launch on a CMIP (interactive ocean) request or a
    # missing local source, not after the model build.
    if args.amip_forcing_from_local_era5:
        if args.mode != "amip":
            raise SystemExit(
                "run_correction_campaign: --amip-forcing-from-local-era5 requires --mode "
                "amip (CMIP uses an interactive ocean, not a prescribed SST forcing).")
        if not args.local_era5_dir:
            raise SystemExit(
                "run_correction_campaign: --amip-forcing-from-local-era5 requires "
                "--local-era5-dir (+ --local-era5-date) — the SAME local archive supplies "
                "both the SST forcing and the compare reference.")
        _assert_output_path_writable(args.amip_forcing_out, flag="amip-forcing-out")
    # Pre-flight: fail in milliseconds (not after a multi-day run) on an unwritable
    # output path — --out is opened only at the very end, --checkpoint each round.
    _assert_output_path_writable(args.out, flag="out")
    if args.checkpoint:
        _assert_output_path_writable(args.checkpoint, flag="checkpoint")
    # Pre-flight: fail fast on a typo'd LOCAL --era5-zarr (opened only AFTER the
    # model/grid build), instead of a cryptic zarr error wasting that setup.  (The local
    # archive path validates per-chunk in open_local_era5_dataset, so skip the zarr check.)
    if args.era5_zarr:
        _assert_era5_zarr_readable(args.era5_zarr)
    # Run-setup preamble (config/grid/sigma + driver builder with the RESOLVED coupled
    # preset + the CFL-checked LES runner + the orographic phis) — shared with the OSSE
    # CLI via _build_run_setup (iter 295).
    base_cfg, grid, sigma, build_base_driver, extract_fn, run_les, phis = _build_run_setup(
        args)
    # ``run_les`` is already a ``_RealismCapture`` (wrapped in ``_build_run_setup``), so the
    # report sections below read ``run_les.breakdowns`` (iter 520).
    # Turnkey OFFLINE realistic AMIP: build the SST/sea-ice forcing from the same local
    # ERA5 archive and inject it into base_cfg (so every corrected round carries it).
    base_cfg = _maybe_apply_local_era5_forcing(args, base_cfg)
    # Opt-in: align the radiation insolation season to the offline ERA5 date (iter 449/450).
    base_cfg = _maybe_align_insolation(args, base_cfg)
    # Persist the EFFECTIVE config (base + runtime injections) as the authoritative calibration
    # config for reproducibility + the deploy/re-run base (iter 464).
    _eff_cfg_path = _write_effective_config(base_cfg, args.out)
    print(f"[campaign] effective calibration config (base + runtime injections) -> "
          f"{_eff_cfg_path}; use it as the deploy/re-run --base-config so the production run "
          "reproduces the SST boundary + insolation the C_K was calibrated on.", flush=True)

    # ERA5 reference regridded to the model grid + sigma (same regrid as the
    # one-shot compare driver).
    canon = canonical_grid_type(base_cfg.grid.grid_type)
    # Time-MEAN ERA5 reference over --era5-n-times consecutive times (N=1 ⇒ a single
    # slice, the old behaviour): compares the time-mean model to a time-mean ERA5
    # climatology, not a single synoptic snapshot.
    n_times = _resolve_era5_n_times(args.era5_n_times)
    _era5_window = range(args.era5_time_idx, args.era5_time_idx + n_times)
    if args.local_era5_dir:
        # REAL ERA5 from the local NCAR-RDA archive (offline) — open --era5-n-days consecutive
        # days (iter 460), average the requested time window, through the SAME regrid chain.
        from scripts.data.load_local_era5 import open_local_era5_dataset_multiday
        _n_days = int(getattr(args, "era5_n_days", 1) or 1)
        if _n_days < 1:
            raise SystemExit(f"--era5-n-days must be >= 1, got {_n_days}.")
        _local_ds = open_local_era5_dataset_multiday(
            args.local_era5_dir, args.local_era5_date, _n_days)
        _n_avail = int(_local_ds.sizes["time"])
        if args.era5_time_idx + n_times > _n_avail:
            raise SystemExit(
                f"--era5-time-idx {args.era5_time_idx} + --era5-n-times {n_times} exceeds the "
                f"{_n_avail} ERA5 times available over --era5-n-days {_n_days} (~24/day). "
                "Lower --era5-n-times or raise --era5-n-days.")
        era5_slice = load_era5_time_mean(
            TrainingERA5Config(
                surface_variables=("surface_pressure", "skin_temperature")),
            _era5_window, ds=_local_ds)
    else:
        era5_slice = load_era5_time_mean(
            TrainingERA5Config(zarr_store=args.era5_zarr, local_cache_dir=args.era5_cache),
            _era5_window)
    # Surface a REGIONAL --era5-zarr before the run (it would silently extrapolate a
    # garbage reference where the model grid extends beyond the ERA5 coverage).
    _warn_if_grid_exceeds_era5_lat_coverage(era5_slice.lat, grid.grid_lat)
    reference = column_state_from_carry(select_era5_regrid(canon)(era5_slice, grid, sigma))

    # OCEAN-only ranking mask (iter 451) — computed ONCE here and passed to BOTH the single- and
    # multi-coefficient paths (the multi path previously dropped it silently — iter 466 fix).
    _ocean_mask = _maybe_ocean_mask(args, base_cfg)

    if args.coefficients is not None:
        return _run_multi_main(
            args, base_cfg, grid, sigma, reference, build_base_driver, extract_fn,
            run_les, phis=phis, valid_mask=_ocean_mask)

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
        initial_field, initial_clubb, start_round, _seed = _load_single_resume(
            ckpt, grid, corrected_field)
        init_box.update(_seed)
        print(_format_resume_line(args.resume, start_round, _seed))

    _ckpt_write = None
    if args.checkpoint:
        def _ckpt_write(round_idx, res, field):
            # Persist the ACCEPTED accumulated `field` (post-gate base), NOT
            # res.updated_config — under the monotonic gate a rejected round's
            # updated_config is discarded while `field` stays the accepted state.
            # The coefficient is stored under its real name (C_K or Pr_t) for
            # inspection + the resume-method guard; resume reconstructs from `field`.
            _capture_initial_record(init_box, res)
            # field finiteness asserted (NaN ⇒ unparseable checkpoint / corrupt resume) +
            # initial_bias sanitized to null (a diverged initial is a recorded metric, not
            # a bug) — the checkpoint analog of the iter-270 / iter-245 output guards.
            arr = _assert_corrected_field_finite(corrected_field, np.asarray(field))
            flat = arr.reshape(-1)
            _atomic_write_json(args.checkpoint, {
                **_checkpoint_common(round_idx, base_cfg, grid, init_box),
                "corrected_field": corrected_field,
                corrected_field: flat.tolist(),
                "field": arr.tolist(),
            })

    # Per-round hook: ALWAYS print real-time progress (iter 318); checkpoint only if requested.
    _total_rounds = start_round + int(args.iterations)

    def checkpoint_callback(round_idx, res, field):
        _print_round_progress(round_idx, res, _total_rounds)
        if _ckpt_write is not None:
            _ckpt_write(round_idx, res, field)

    result = build_correction_campaign(
        base_atm_config=base_cfg, build_base_driver=build_base_driver,
        extract_column_state=extract_fn, reference=reference, sigma=sigma,
        grid=grid, area_weights=_area_weights(grid), n_iterations=args.iterations,
        les_config=ColumnLESConfig(diagnosis_method=args.diagnosis_method,
                                   surface_flux=args.surface_flux),
        run_les_fn=run_les, phis=phis, valid_mask=_ocean_mask,
        initial_clubb=initial_clubb, initial_field=initial_field,
        start_round=start_round, checkpoint_callback=checkpoint_callback,
        dry_run=args.dry_run,
        **_campaign_knobs_from_args(args),
    )
    if args.dry_run:
        print(_dry_run_report(result, mode=args.mode, out=args.out,
                              era5=_averaging_provenance(args, base_cfg),
                              amip_forcing=_amip_forcing_provenance(args)))
        return 0
    biases = [(float(it.bias.baseline_bias), float(it.bias.updated_bias),
               bool(it.bias.improved)) for it in result.iterations]
    accepted = list(result.accepted)
    steps = [float(it.step_fraction) for it in result.iterations]
    for i, (b0, b1, imp) in enumerate(biases):
        kept = accepted[i] if i < len(accepted) else True
        it_i = result.iterations[i]
        print(_format_round_line(
            start_round + i, b0, b1, imp, kept, steps[i],
            getattr(it_i, "n_diagnoses_valid", 0), getattr(it_i, "n_diagnosed", 0)))
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
    _atomic_write_json(args.out, build_campaign_output_dict(
        result, grid_provenance=_grid_provenance(base_cfg, grid),
        summary=summary, health=health, corrected_field=corrected_field,
        averaging=_averaging_provenance(args, base_cfg),
        les_provenance=_les_provenance(args),
        les_realism=_realism_summary_dict(getattr(run_les, "breakdowns", None))), indent=2)
    print(f"[campaign] wrote corrected clubb config to {args.out}")
    print(summary.report())
    print(f"[campaign] {health.status.upper()}: {health.message}")
    print_realism_summary(run_les)
    _print_deploy_hint(args.out, grid)
    _maybe_write_env_kernel(args, result)
    # Exit code = the health verdict (iter 287, mirrors the OSSE go/no-go + the multi
    # main): 0 only when the run IMPROVED, non-zero otherwise, so a launch workflow does
    # not deploy a no-op/diverged correction. The deployable JSON is written either way.
    return _campaign_exit_code(health)


def _env_kernel_export_note(feedback_strategy, has_kernel):
    """The user-facing note for the env-kernel export decision, or ``None``.

    ``None`` when the kernel IS exported (``has_kernel``) OR a ``static`` / non-CLUBB
    campaign correctly has none to transfer.  Returns a WARNING string ONLY when an
    ``environment``-strategy campaign produced NO transferable kernel (every round was
    no-op or rejected ⇒ no env→coefficient regression) — so a user who ran
    ``--feedback-strategy environment`` expecting ``<out>.env_kernel.json`` learns WHY
    it is absent (rather than silently finding no cross-resolution artifact)."""
    if has_kernel or feedback_strategy != "environment":
        return None
    return ("--feedback-strategy environment requested but NO transferable "
            "env→coefficient kernel was produced (every round was no-op or rejected, so "
            "there is no regression to export) — no <out>.env_kernel.json written; "
            "check the LES-diagnosis validity (the campaign health verdict).")


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
    from legoesm.training.correction_loop import last_accepted_env_kernel
    from legoesm.training.deploy_correction import env_kernel_to_dict

    # The last ACCEPTED round's kernel (run_correction_campaign fills `accepted`
    # every round; a rejected final round's kernel must not sit beside the
    # accepted per-column --out). None ⇒ static / non-CLUBB / no-op campaign.
    kernel = last_accepted_env_kernel(result)
    if kernel is None:
        note = _env_kernel_export_note(
            getattr(args, "feedback_strategy", "static"), has_kernel=False)
        if note is not None:
            print(f"[campaign] WARNING: {note}")
        return
    out = f"{args.out}.env_kernel.json"
    # Stamp the SAME averaging-window provenance as the per-column --out (iter 269): a
    # cross-resolution deploy lands this kernel on a DIFFERENT grid, where knowing the
    # source climate it was trained against (a snapshot vs an N-time climatology) matters
    # MORE than the same-grid case. env_kernel_from_dict reads only the kernel keys, so
    # the extra block is inert on reload (round-trip locked).
    kernel_dict = env_kernel_to_dict(kernel)
    kernel_dict["averaging"] = _averaging_provenance(args)
    _atomic_write_json(out, kernel_dict, indent=2)
    print(f"[campaign] wrote RAW environment kernel (cross-resolution deploy) to {out}")
    print(
        "[campaign] deploy on ANY grid with: apply_env_kernel_override("
        "env_kernel_from_dict(json.load(open(...))), "
        "column_environment_grid(model, sigma, env_config=..., p_full=..., "
        "p_half=...)[0])"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
