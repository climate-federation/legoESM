"""Run an AMIP/CMIP model and time-mean its column state for ERA5 comparison.

The missing wiring between a *run* and the climatological compare
(``docs/COMPARE_REANALYSIS.md`` §1 step 1→2): run the model for a window and
sample the atmospheric column state at the diagnostic cadence
(``OutputConfig.diag_days``), accumulating it (iter-30 accumulator) into the time
mean that :func:`compare_state_to_reference` consumes as ``model``.  This is the
real ``run_amip_fn`` / ``run_cmip_fn`` that ``correction_loop.make_compare_fn``
(iter 21) expects — previously the loop injected a mock.

Sampling rides the driver's ``segment_callback`` hook (the same per-segment hook
the coupler uses for surface coupling): ``ModelDriver.run`` (AMIP) exposes it
natively, and ``CoupledESMDriver.run`` (CMIP) now composes an extra hook after
its coupling step.  The mode-specific :class:`ColumnState` extractor
(``*_column_state``) closes over the OUTER driver — prescribed SST for AMIP, the
coupled-ocean SST for CMIP — and ignores the callback's per-segment args.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from legoesm.training.column_state_accumulator import (
    accumulate_column_state,
    init_column_accumulator,
    mean_column_state,
)
from legoesm.training.compare_reanalysis import (
    ColumnState,
    column_state_from_hydrostatic,
    grid_winds_from_spectral,
)


def run_to_column_mean(
    driver: Any,
    extract_column_state: Callable[[Any, float, float], ColumnState],
    *,
    run_kwargs: dict | None = None,
) -> ColumnState:
    """Run a configured + ``setup()``-d ``driver`` and return its time-mean ColumnState.

    Samples ``extract_column_state(driver, day, dt_segment)`` at each segment
    boundary (the ``diag_days`` diagnostic cadence) via the driver's
    ``segment_callback`` and folds the samples with the iter-30 accumulator.
    ``extract_column_state`` maps the OUTER ``driver`` (closed over, NOT the
    callback's ``_driver`` arg) plus the segment ``day`` to a :class:`ColumnState`
    — the ``day`` MUST be threaded through because AMIP SST is time-dependent
    (``get_sst_sic(day)``), so a multi-day climatology mean would otherwise sample
    day-0 SST every segment.  CMIP's coupled SST is prognostic and ignores
    ``day``.  ``run_kwargs`` forwards extra args (e.g. ``start_day``) to
    ``driver.run``.

    Raises if no segment boundary fired (the run is shorter than one diagnostic
    interval) — a silent zero-sample mean would be a spurious all-zero state.
    """
    box: dict[str, Any] = {"acc": None}

    def _sample(_driver: Any, day: float, dt_segment: float) -> None:
        cs = extract_column_state(driver, day, dt_segment)
        if box["acc"] is None:
            box["acc"] = init_column_accumulator(cs)
        box["acc"] = accumulate_column_state(box["acc"], cs)

    driver.run(segment_callback=_sample, **(run_kwargs or {}))

    if box["acc"] is None:
        raise ValueError(
            "run_to_column_mean: no segment boundary fired — the run is shorter "
            "than one diagnostic interval (raise `days` or lower `diag_days`)."
        )
    return mean_column_state(box["acc"])


def make_run_fn(
    build_driver: Callable[[Any], Any],
    extract_column_state: Callable[[Any, float, float], ColumnState],
    *,
    run_kwargs: dict | None = None,
) -> Callable[[Any], ColumnState]:
    """Adapt a ``config → driver`` builder into the ``run_amip_fn``/``run_cmip_fn``
    that :func:`legoesm.training.correction_loop.make_compare_fn` (iter 21) expects.

    Returns ``run_fn(config) -> ColumnState`` (the climatological time mean): each
    call builds a fresh driver from ``config`` via ``build_driver`` (which MUST
    return a driver already ``setup()``-d), runs it, and time-means the column
    state (:func:`run_to_column_mean`).  This is the last real piece of the
    run→time-mean→compare chain — ``make_compare_fn(..., run_amip_fn=run_fn)`` then
    scores the mean against the (regridded) ERA5 ``reference`` and emits the
    worst-column manifest.

    ``build_driver`` carries the production-specific config→model wiring (apply
    the corrected scheme config, pick the grid/levels, prescribe SST for AMIP or
    attach the ocean for CMIP); keeping it injected leaves this adapter generic +
    unit-testable.  It MUST be deterministic except for ``config`` (same reference
    grid/masks across calls) so the loop's measured bias change reflects only the
    config update.
    """
    def run_fn(config: Any) -> ColumnState:
        driver = build_driver(config)
        return run_to_column_mean(driver, extract_column_state, run_kwargs=run_kwargs)

    return run_fn


def _sst_array(sst: Any) -> Any:
    """Unwrap a ``Field``-wrapped SST to its raw array (``.data``) if needed."""
    return sst.data if hasattr(sst, "data") else sst


def cmip_column_state(
    coupled_driver: Any, day: float = 0.0, dt_segment: float = 0.0
) -> ColumnState:
    """:class:`ColumnState` from a coupled (CMIP) driver: atm state + coupled SST.

    Uses the public ``state`` / ``q_v`` / ``get_sst_sic`` accessors.  ``sst_K`` is the
    coupled SST on the ATMOSPHERE grid — via ``coupled_driver.get_sst_sic(day)``, the same
    accessor AMIP uses, which the coupler remaps from the (possibly different) ocean grid
    onto the atm grid (``o2a``).  This keeps the SST env tag aligned with the atm columns
    for ANY ``ocean_grid`` (the raw ``ocean_state.T_sfc`` is on the OCEAN grid and would
    mis-align it — cf. the ``column_state_from_hydrostatic`` sst_K grid guard, iter 329);
    for a same-grid coupled run the o2a remapper is the identity, so this is the same SST
    as before.  ``dt_segment`` is unused.
    """
    # Spectral CMIP: synthesize the grid winds from the spectral state first
    # (no-op for grid / MPAS states); MPAS: column_state_from_hydrostatic
    # reconstructs the cell wind from the edge velocity via mesh=driver.grid.
    state = grid_winds_from_spectral(
        coupled_driver.state, getattr(coupled_driver, "grid", None),
        getattr(coupled_driver, "sigma", None))
    sst, _sic = coupled_driver.get_sst_sic(day)
    return column_state_from_hydrostatic(
        state,
        coupled_driver.q_v,
        sst_K=_sst_array(sst),
        mesh=getattr(coupled_driver, "grid", None),
    )


def amip_column_state(
    atm_driver: Any, day: float = 0.0, dt_segment: float = 0.0
) -> ColumnState:
    """:class:`ColumnState` from an atm-only (AMIP) driver: atm state + prescribed SST.

    ``sst_K`` is the AMIP prescribed SST AT ``day`` (``get_sst_sic(day)``) — the
    segment ``day`` MUST be threaded in (the prescribed SST is time-dependent), so
    each segment of a multi-day mean sees its own SST, not day-0.  ``dt_segment``
    is unused; the SIC is not needed for the column comparison.
    """
    sst, _sic = atm_driver.get_sst_sic(day)
    # Spectral AMIP: synthesize the grid winds from the spectral state first
    # (no-op for grid / MPAS states); MPAS reconstructs the cell wind in
    # column_state_from_hydrostatic via mesh=driver.grid.
    state = grid_winds_from_spectral(
        atm_driver.state, getattr(atm_driver, "grid", None),
        getattr(atm_driver, "sigma", None))
    return column_state_from_hydrostatic(
        state,
        atm_driver.q_v,
        sst_K=_sst_array(sst),
        mesh=getattr(atm_driver, "grid", None),
    )


def phis_or_none_if_flat(phis: Any) -> Any:
    """Map a static surface-geopotential field to ``None`` when it is absent OR
    identically zero (an effectively-flat / aquaplanet model), else return it.

    A flat ``ModelDriver`` initialises ``phis = zeros(...)`` rather than leaving it
    absent (``model_driver.py`` ``_create_topography``), so an all-zero field means
    "flat": the orographic LES-forcing term then correctly stays OFF — exact
    flat-path parity (an all-zero ``∇phis`` contributes nothing anyway), no wasted
    gradient work — and a fail-loud "terrain required but model is flat" check can
    rely on ``None``.  A coastline with ANY non-zero cell is REAL terrain (kept).

    Launch-time EAGER call on a concrete array (build ONE driver, reuse the result
    across campaign rounds), so the Python ``bool`` from ``jnp.all`` is fine — this
    is NOT a traced hot-loop path.
    """
    if phis is None:
        return None
    import jax.numpy as jnp
    return None if bool(jnp.all(jnp.asarray(phis) == 0)) else phis


def model_phis_from_driver(driver: Any) -> Any:
    """The model's STATIC surface geopotential ``phis = g·z_s`` on the full grid —
    the input to the orographic geostrophic LES-forcing term (iter 117).

    Pass the result as ``make_les_diagnose_fn(..., phis=…)`` / a campaign builder's
    ``phis=`` so the orographic correction uses the model's OWN topography (guaranteed
    CONSISTENT with the AMIP/CMIP run that produced the comparison state — no
    mismatched-field footgun).  Returns ``None`` for a flat/aquaplanet model — BOTH
    when the state carries no ``phis`` AND when ``phis`` is identically zero
    (:func:`phis_or_none_if_flat`).
    ``phis`` is STATIC (config-independent topography), so build ONE driver and reuse
    the result across campaign rounds.  Spectral states are synthesized to a grid
    state first (the same ``grid_winds_from_spectral`` the compare uses); grid/MPAS
    states pass through.  The ``Field`` wrapper is unwrapped to a raw array.
    """
    from legoesm.core.field import Field
    state = grid_winds_from_spectral(
        getattr(driver, "state", None), getattr(driver, "grid", None),
        getattr(driver, "sigma", None))
    phis = getattr(state, "phis", None)
    # Unwrap ONLY a Field (isinstance, not hasattr "data" — a raw NumPy array also
    # has a `.data` MEMORYVIEW that would be returned by mistake; Codex). A None phis
    # falls through unchanged (isinstance(None, Field) is False).
    arr = phis.data if isinstance(phis, Field) else phis
    return phis_or_none_if_flat(arr)
