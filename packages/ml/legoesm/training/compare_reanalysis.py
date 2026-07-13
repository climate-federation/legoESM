"""Compare an AMIP model state to ERA5 and emit a worst-column manifest.

Stage 2 orchestration of ``docs/COMPARE_REANALYSIS.md``: ties the per-column
metric (:mod:`legoesm.training.column_era5_metrics`) and the manifest +
environment tagging (:mod:`legoesm.training.column_manifest`) into one
comparison entry point operating on a model state and an ERA5 reference that are
**already on the same grid and vertical (sigma) levels**.

**Regrid-direction decision (closes the §6 open question):** ERA5 → model grid
+ model sigma levels.  Rationale: the diagnosis is per *model* column (we force
the LES like a model column), and the existing
:mod:`legoesm.training.era5_to_state` path already regrids ERA5 lat-lon →
model grid (spectral / cubed-sphere / lat-lon) and interpolates pressure → sigma
with humidity-convention handling.  Comparing on the model grid keeps the
area/mass weighting native to the model and avoids a second regrid of the model
state.  The driver therefore: (model state on model grid+sigma) vs
(``era5_to_*_carry`` → same grid+sigma).

This module is the data-agnostic core; a thin ``scripts/validate`` driver loads
the AMIP snapshot + ERA5 slice and calls :func:`compare_state_to_reference`.
The environment tags (CAPE/shear/SST) are computed from the **model** column —
that is the large-scale state the LES will be forced with — not from ERA5.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax  # noqa: F401  (used in string annotations under `from __future__`)
import jax.numpy as jnp
import numpy as np
from legoesm.training.column_era5_metrics import (
    ColumnErrorConfig,
    ColumnErrorFields,
    normalized_mass_weights,
    score_columns,
)
from legoesm.training.column_manifest import (
    ColumnEnvironmentFields,
    ColumnRecord,
    EnvironmentConfig,
    build_worst_column_manifest,
    compute_column_environment,
)

# Seconds per day, for accumulated-precip → rate conversion.
_SECONDS_PER_DAY = 86400.0


class ColumnState(NamedTuple):
    """Atmospheric column fields on a common grid + sigma levels.

    Profiles are ``[..., nlev]`` (surface-last); ``p_s`` is ``[...]`` [Pa].
    ``precip_mm_day`` is an optional ``[...]`` surface field; ``sst_K`` is the
    optional surface temperature used only for environment tagging — the
    prescribed SST under AMIP or the coupled-ocean SST under CMIP.

    ``u_edge`` is NOT a per-column comparison field: it is optional NATIVE MPAS
    metadata — the edge-normal velocity ``(nEdges, nlev)`` (a DIFFERENT cardinality
    from the cell fields) carried so the LES-forcing extractor can compute the
    exact native-edge divergence/ω for an MPAS worst column (the comparison itself
    uses the cell-reconstructed ``u``/``v``).  ``None`` for cell-wind grids
    (lat-lon / cubed-sphere) and for any ERA5 reference state.  The comparison
    (``_validate_aligned`` / RMSE) ignores it — it checks only the named cell
    fields — and the time-mean accumulator means it like any other present leaf
    (consistent: Perot reconstruction is linear, so mean(reconstruct(u_edge)) ==
    reconstruct(mean(u_edge))).
    """

    T: jax.Array
    q_v: jax.Array
    u: jax.Array
    v: jax.Array
    p_s: jax.Array
    precip_mm_day: jax.Array | None = None
    sst_K: jax.Array | None = None
    u_edge: jax.Array | None = None


class ColumnComparison(NamedTuple):
    """Result of comparing a model state to a reference state."""

    error_fields: ColumnErrorFields
    environment: ColumnEnvironmentFields
    manifest: list[ColumnRecord]
    have_precip: bool = False   # both states carried precip (precip entered the score)


def precip_mm_day_from_accum(
    precip_accum_kg_m2: jax.Array, segment_seconds: float
) -> jax.Array:
    """Convert accumulated precip [kg/m²] over a segment to a rate [mm/day].

    ``1 kg/m²`` of liquid water equals ``1 mm`` depth, so the rate is
    ``accum / segment_seconds * 86400``.
    """
    seg = max(float(segment_seconds), 1.0e-30)
    return jnp.asarray(precip_accum_kg_m2) / seg * _SECONDS_PER_DAY


def build_pressure_from_sigma(
    p_s: jax.Array, sigma_full: jax.Array, sigma_half: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Full- and half-level pressure [Pa] for a **pure-sigma** coordinate.

    ``p = sigma · p_s``.  Returns ``(p_full[..., nlev], p_half[..., nlev+1])``.
    For a hybrid-sigma-pressure coordinate the caller must instead pass the
    coordinate's own pressures explicitly to :func:`compare_state_to_reference`
    (``p_full`` / ``p_half`` overrides).
    """
    p_s = jnp.asarray(p_s)
    sigma_full = jnp.asarray(sigma_full, dtype=p_s.dtype)
    sigma_half = jnp.asarray(sigma_half, dtype=p_s.dtype)
    p_full = p_s[..., None] * sigma_full
    p_half = p_s[..., None] * sigma_half
    return p_full, p_half


def _validate_aligned(model: ColumnState, reference: ColumnState) -> None:
    """Validate model/reference are on the same grid + sigma and self-consistent.

    Checks every profile field matches between model and reference, and that all
    column-shaped fields (``p_s`` and the optional ``precip_mm_day`` / ``sst_K``)
    share the model's column shape ``T.shape[:-1]`` — so a transposed or
    mis-broadcast surface field fails loudly instead of mis-indexing.
    """
    col_shape = model.T.shape[:-1]
    for name in ("T", "q_v", "u", "v"):
        ms = getattr(model, name).shape
        rs = getattr(reference, name).shape
        if ms != rs:
            raise ValueError(
                f"compare_state_to_reference: model.{name} shape {ms} != "
                f"reference.{name} shape {rs}; both must be on the same "
                f"grid + sigma levels (regrid ERA5 → model first)."
            )
        if ms[:-1] != col_shape:
            raise ValueError(
                f"compare_state_to_reference: model.{name} column shape "
                f"{ms[:-1]} != model.T column shape {col_shape}."
            )
    for st_name, st in (("model", model), ("reference", reference)):
        for name in ("p_s", "precip_mm_day", "sst_K"):
            field = getattr(st, name)
            if field is not None and jnp.asarray(field).shape != col_shape:
                raise ValueError(
                    f"compare_state_to_reference: {st_name}.{name} shape "
                    f"{jnp.asarray(field).shape} != column shape {col_shape}."
                )


class ReferenceBoundsConfig(NamedTuple):
    """GENEROUS physical-plausibility bounds for a comparison/reference state.

    These are SANITY ranges, deliberately wide — their job is to catch a gross
    UNITS / sign error in a loaded ERA5 reference (the classic: T in °C not K, p_s
    in hPa not Pa, q_v in g/kg not kg/kg), NOT to tightly validate physics.  A
    units bug otherwise passes :func:`_validate_aligned` (which checks only shape)
    and produces a massive FAKE bias that the correction loop would then "improve"
    against garbage.  Defaults span the full Earth atmospheric range with margin.
    """

    T_min_K: float = 150.0           # < this ⇒ likely °C input (≈ −123 °C floor)
    T_max_K: float = 350.0
    q_v_min_kg_kg: float = -1.0e-6   # ~0: ERA5 q is non-negative (roundoff floor only;
                                     # NOT the model's advection-overshoot tolerance)
    q_v_max_kg_kg: float = 0.1       # > this ⇒ likely g/kg input (sat ≲ 0.04 kg/kg)
    p_s_min_Pa: float = 3.0e4        # < this ⇒ likely hPa input (≈ 300 hPa floor)
    p_s_max_Pa: float = 1.1e5
    wind_abs_max_ms: float = 200.0   # |u|,|v|: generous vs ≲100 m/s jets
    sst_min_K: float = 250.0
    sst_max_K: float = 320.0
    precip_min_mm_day: float = 0.0   # precip is non-negative
    precip_max_mm_day: float = 2000.0


def validate_reference_physical(
    state: ColumnState, *, bounds: ReferenceBoundsConfig | None = None,
    name: str = "reference",
) -> None:
    """Fail-fast pre-flight: every present field of ``state`` is finite AND within
    its GENEROUS physical range (:class:`ReferenceBoundsConfig`) — else raise
    ``ValueError`` naming the field, the observed min/max, and the likely units bug.

    The companion to :func:`_validate_aligned` (which checks only SHAPE): a
    units/sign error in a loaded ERA5 reference would otherwise silently produce a
    huge fake bias the correction loop "corrects" against.  Pure host-side check on
    a CONCRETE state (campaign-build time, never traced); call it ONCE on the
    reference the campaign actually uses.  ``u_edge`` (native MPAS edge velocity, a
    different cardinality) is intentionally NOT range-checked here.
    """
    if bounds is None:
        bounds = ReferenceBoundsConfig()
    checks = [
        ("T", state.T, bounds.T_min_K, bounds.T_max_K, "K (°C input?)"),
        ("q_v", state.q_v, bounds.q_v_min_kg_kg, bounds.q_v_max_kg_kg,
         "kg/kg (g/kg input?)"),
        ("u", state.u, -bounds.wind_abs_max_ms, bounds.wind_abs_max_ms, "m/s"),
        ("v", state.v, -bounds.wind_abs_max_ms, bounds.wind_abs_max_ms, "m/s"),
        ("p_s", state.p_s, bounds.p_s_min_Pa, bounds.p_s_max_Pa, "Pa (hPa input?)"),
    ]
    if state.sst_K is not None:
        checks.append(
            ("sst_K", state.sst_K, bounds.sst_min_K, bounds.sst_max_K, "K"))
    if state.precip_mm_day is not None:
        checks.append(
            ("precip_mm_day", state.precip_mm_day, bounds.precip_min_mm_day,
             bounds.precip_max_mm_day, "mm/day"))
    for field, value, lo, hi, unit_hint in checks:
        arr = jnp.asarray(value)
        if not bool(jnp.all(jnp.isfinite(arr))):
            raise ValueError(
                f"validate_reference_physical: {name}.{field} has non-finite "
                f"values — a comparison reference must be finite ({unit_hint}).")
        if not bool(jnp.all((arr >= lo) & (arr <= hi))):
            amin, amax = float(jnp.min(arr)), float(jnp.max(arr))
            raise ValueError(
                f"validate_reference_physical: {name}.{field} outside the plausible "
                f"range [{lo}, {hi}] {unit_hint}: observed [{amin:.4g}, {amax:.4g}] "
                "— check the reference UNITS/sign before running the campaign (a "
                "units error produces a large FAKE bias the loop would 'correct').")


def model_state_is_finite(state: Any) -> bool:
    """True iff every present prognostic field of a MODEL ``state`` is finite.

    The BOOL core of :func:`assert_model_state_finite` (which RAISES) AND the NON-fatal
    guard the loop's line search + the held-out verify need: a blown-up CANDIDATE must
    read as NOT-improved, not raise on one bad step (iters 392/393).  ``state is None``
    (a mock ``compare_fn`` carrying no state), or a state lacking the prognostic fields,
    ⇒ True (nothing to check) — so abstract-mock compare_fns are unaffected.

    Checks EVERY prognostic leaf, incl. ``u_edge`` (the NATIVE MPAS edge-normal velocity;
    it is not a per-column COMPARE field, but it IS prognostic — a diverged MPAS run can
    blow up the edge winds, and the Perot cell reconstruction is linear so a NaN there
    propagates, but guard it directly too rather than rely on that propagation).  Each is
    ``None``-defaulted, so a non-MPAS state simply skips ``u_edge`` (codex-review iter 401)."""
    if state is None:
        return True
    for fname in ("T", "q_v", "u", "v", "p_s", "sst_K", "precip_mm_day", "u_edge"):
        value = getattr(state, fname, None)
        if value is not None and not bool(jnp.all(jnp.isfinite(jnp.asarray(value)))):
            return False
    return True


def assert_model_state_finite(state: ColumnState, *, name: str = "model run") -> None:
    """Fail-fast: every present field of a MODEL run's time-mean ``state`` is finite.

    The model-side companion to :func:`validate_reference_physical` (which range-checks
    the loaded ERA5 REFERENCE for a units bug).  Here the concern is DIVERGENCE: a model
    run that blew up (CFL / instability) yields NaN/inf in T/q_v/u/v, which would
    otherwise flow silently into :func:`compare_state_to_reference`'s ``score_columns``
    (NaN-masked per level) and produce a GARBAGE bias the campaign cannot improve —
    wasting a multi-day HPC run while looking like the correction merely failing.

    Finiteness ONLY (no physical bounds: a model's units are correct by construction,
    unlike a loaded reanalysis).  Pure host-side check on a CONCRETE state; raises
    ``ValueError`` naming the field + the non-finite count + the divergence hint.  Use
    on the BASELINE (current-config) run, where a blown-up run is FATAL.  A diverged
    CANDIDATE/line-search run is instead made NOT-improved by
    :func:`model_state_is_finite` inside the loop's line search (iter 393) — NOT fatal,
    so one bad step REJECTS that candidate without crashing the multi-day campaign.  (An
    earlier design assumed the monotonic gate ALONE caught it; it did NOT — the
    per-column RMSE NaN-MASKS a blown-up state to a spurious ≈0 bias that reads as
    'improved', so the explicit state guard is REQUIRED.)
    """
    if model_state_is_finite(state):
        return
    fields = [("T", state.T), ("q_v", state.q_v), ("u", state.u), ("v", state.v),
              ("p_s", state.p_s)]
    if state.sst_K is not None:
        fields.append(("sst_K", state.sst_K))
    if state.precip_mm_day is not None:
        fields.append(("precip_mm_day", state.precip_mm_day))
    if getattr(state, "u_edge", None) is not None:
        # MPAS native edge velocity (iter 401): keep the detailed loop in sync with
        # model_state_is_finite's field set so a u_edge-only NaN raises here too rather
        # than fast-path-False → loop-finds-nothing → silent pass.
        fields.append(("u_edge", state.u_edge))
    for field, value in fields:
        arr = jnp.asarray(value)
        finite = jnp.isfinite(arr)
        if not bool(jnp.all(finite)):
            n_bad = int(jnp.sum(~finite))
            raise ValueError(
                f"assert_model_state_finite: {name}.{field} has {n_bad}/{arr.size} "
                "non-finite value(s) — the model run likely DIVERGED (CFL / instability). "
                "Fix the model stability (timestep / resolution / config) before "
                "correcting; the LES-informed correction cannot improve a blown-up run.")


def compare_state_to_reference(
    *,
    model: ColumnState,
    reference: ColumnState,
    sigma_full: jax.Array,
    sigma_half: jax.Array,
    lat_deg: jax.Array,
    lon_deg: jax.Array,
    time_index: int,
    n_worst: int,
    error_config: ColumnErrorConfig = ColumnErrorConfig(),
    env_config: EnvironmentConfig = EnvironmentConfig(),
    valid_mask: jax.Array | None = None,
    p_full: jax.Array | None = None,
    p_half: jax.Array | None = None,
) -> ColumnComparison:
    """Score a model state against a reference and emit the worst-column manifest.

    ``model`` / ``reference`` are :class:`ColumnState` on the SAME grid + sigma
    levels (ERA5 already regridded to the model grid).  ``sigma_full`` ``[nlev]``
    and ``sigma_half`` ``[nlev+1]`` define the mass weights and (pure-sigma)
    pressures; pass ``p_full`` / ``p_half`` to override for a hybrid coordinate.
    ``n_worst`` columns are selected; ``valid_mask`` (column-shaped) restricts
    selection (e.g. ocean-only).

    Precipitation enters the score only when **both** states carry it (ERA5
    precip is often unavailable); otherwise the precip term is dropped.  SST for
    the environment tag is taken from ``model.sst_K`` when present, else
    approximated by the model's lowest-level air temperature (the driver SHOULD
    supply the prescribed SST; the fallback WARNS once per session — iter 280 — so
    the operator learns the clustering / env-kernel tags are approximate).
    """
    _validate_aligned(model, reference)
    sigma_full = jnp.asarray(sigma_full)
    sigma_half = jnp.asarray(sigma_half)
    nlev = model.T.shape[-1]
    if sigma_full.shape[-1] != nlev or sigma_half.shape[-1] != nlev + 1:
        raise ValueError(
            f"sigma_full must have length nlev={nlev} and sigma_half "
            f"nlev+1={nlev + 1}; got {sigma_full.shape[-1]} and "
            f"{sigma_half.shape[-1]}."
        )

    # This routine builds a host-side manifest (not jit-traced), so sigma is
    # concrete: enforce the surface-last, increasing-sigma orientation here.
    # A reversed coordinate would make ``dsigma`` negative and silently corrupt
    # the mass weights (negative weights) and pure-sigma pressure orientation.
    dsigma_np = np.asarray(sigma_half)[1:] - np.asarray(sigma_half)[:-1]
    if not np.all(dsigma_np > 0):
        raise ValueError(
            "sigma_half must be strictly increasing top→surface (all dsigma>0); "
            "got a non-monotonic or reversed coordinate. Pass surface-last "
            "profiles with a top-to-surface sigma grid."
        )
    if (p_full is None) != (p_half is None):
        raise ValueError(
            "compare_state_to_reference: pass both p_full and p_half (hybrid "
            "coordinate override) or neither (got exactly one)."
        )

    # Build (or validate) the layer pressures FIRST, so the mass weights can derive from the
    # actual layer PRESSURE thickness (correct for a hybrid coordinate when the caller supplies
    # the hybrid p_half), not the sigma thickness (iter 338, fixing the iter-337 mis-weighting).
    if p_full is None:  # both-or-neither already enforced above
        p_full, p_half = build_pressure_from_sigma(
            model.p_s, sigma_full, sigma_half
        )
    else:
        if jnp.asarray(p_full).shape != model.T.shape:
            raise ValueError(
                f"p_full override shape {jnp.asarray(p_full).shape} != model.T "
                f"shape {model.T.shape}."
            )
        expected_half = model.T.shape[:-1] + (nlev + 1,)
        if jnp.asarray(p_half).shape != expected_half:
            raise ValueError(
                f"p_half override shape {jnp.asarray(p_half).shape} != "
                f"{expected_half}."
            )
        # Host-side: a reversed-but-correctly-shaped override would invert the
        # buoyancy integral in CAPE. Require pressure increasing top→surface.
        if not np.all(np.diff(np.asarray(p_half), axis=-1) > 0):
            raise ValueError(
                "p_half override must increase monotonically top→surface "
                "(surface-last); got a reversed or non-monotonic pressure."
            )

    # Per-column mass weights from the layer PRESSURE thickness dp = diff(p_half).  Pure-sigma
    # (p_half = sigma_half·p_s): the per-column p_s factor cancels in the axis=-1 normalization,
    # so this is byte-identical to the old diff(sigma_half) weights.  HYBRID (a supplied p_half):
    # dp = dA·p_ref + dB·p_s, so the weights are correct — vs the iter-337 bug where the sigma
    # thickness over-weighted upper-terrain levels by ~276% at p_s≠p_ref.
    dp = jnp.asarray(p_half)[..., 1:] - jnp.asarray(p_half)[..., :-1]
    weights = normalized_mass_weights(dp)

    have_both_precip = (
        model.precip_mm_day is not None and reference.precip_mm_day is not None
    )
    precip_model = model.precip_mm_day if have_both_precip else None
    precip_ref = reference.precip_mm_day if have_both_precip else None

    error_fields = score_columns(
        T_model=model.T, qv_model=model.q_v, u_model=model.u, v_model=model.v,
        T_ref=reference.T, qv_ref=reference.q_v,
        u_ref=reference.u, v_ref=reference.v,
        mass_weights=weights,
        precip_model_mm_day=precip_model,
        precip_ref_mm_day=precip_ref,
        config=error_config,
    )

    if model.sst_K is not None:
        sst = model.sst_K
    else:
        # Documented fallback: surface-air-temperature proxy when the driver did
        # not thread the prescribed SST.  CAPE/shear are unaffected; only the
        # SST environment tag is approximate in this branch.  WARN (once per session
        # — Python dedups identical warnings, so no per-round noise) so the empirical-
        # run operator learns the env tags are approximate: the docstring's "the driver
        # SHOULD supply SST" is a config issue worth surfacing, not silently degrading
        # the clustering / env-kernel by climate (iter 280).
        import warnings

        warnings.warn(
            "compare_state_to_reference: model.sst_K is None — using the lowest-level "
            "air temperature as the SST environment tag (the worst-column clustering + "
            "env-kernel tags are APPROXIMATE). Supply the prescribed/coupled SST via the "
            "driver for accurate environment tags.", stacklevel=2)
        sst = model.T[..., -1]

    environment = compute_column_environment(
        T=model.T, q_v=model.q_v, u=model.u, v=model.v,
        p_full=p_full, p_half=p_half, sst=sst,
        sigma_full=sigma_full, config=env_config,
    )

    manifest = build_worst_column_manifest(
        error_fields=error_fields,
        environment=environment,
        lat_deg=lat_deg, lon_deg=lon_deg,
        time_index=time_index, n=n_worst, valid_mask=valid_mask,
    )
    return ColumnComparison(
        error_fields=error_fields, environment=environment, manifest=manifest,
        have_precip=bool(have_both_precip),
    )


def column_state_from_carry(
    carry: Any,
    *,
    sst_K: jax.Array | None = None,
    precip_mm_day: jax.Array | None = None,
) -> ColumnState:
    """Build a :class:`ColumnState` from a ``SegmentCarry``-like object.

    Duck-typed on ``T``/``q_v``/``u``/``v``/``p_s`` so it works for the compiled
    ``SegmentCarry`` or any state carrying those fields (and so this module need
    not import the coupler).  Precip and SST are passed explicitly: precip
    derives from ``carry.precip_accum`` via :func:`precip_mm_day_from_accum` in
    the driver (it needs the segment duration), and SST is the prescribed AMIP
    forcing, neither of which is an atmospheric column field.
    """
    return ColumnState(
        T=carry.T, q_v=carry.q_v, u=carry.u, v=carry.v, p_s=carry.p_s,
        precip_mm_day=precip_mm_day, sst_K=sst_K,
    )


def grid_winds_from_spectral(state: Any, grid: Any, sigma: Any) -> Any:
    """Synthesize a spectral state to a GRID ``HydrostaticState`` for the compare.

    The spectral (Gaussian) dycore's prognostic state is a
    :class:`~legoesm.atmosphere.dynamics.gcm.spectral_pe.SpectralHydrostaticState`
    (``vor_hat``/``div_hat``/``T_hat``/``lnps_hat`` — complex SH coefficients), so
    it has no grid ``u``/``v`` for the worst-column ranking / wind RMSE or the LES
    forcing extractor.  This converts it to a grid ``HydrostaticState``
    (``u``/``v``/``T``/``p_s``/``phis``) via the dycore's OWN diagnostic synthesis
    (:func:`~legoesm.atmosphere.dynamics.gcm.spectral_pe.spectral_pe_to_grid` —
    ``uv_from_vordiv`` + ``sh_synthesis``, frame-consistent geographic east/north
    winds, matching the ERA5 reference on the Gaussian grid).  The spectral analog
    of the MPAS edge→cell reconstruction; like that, the COMPARE side produces grid
    winds while the model integrates in its native (here spectral) representation.

    ANY non-spectral state (a grid ``HydrostaticState`` for lat-lon/cubed, or an
    MPAS edge-velocity state) is returned UNCHANGED (``grid``/``sigma`` unused) so
    the caller can apply this unconditionally.  ``grid`` is the
    :class:`~legoesm.grids.gaussian.GaussianGrid` and ``sigma`` the vertical
    coordinate (both ``driver.grid``/``driver.sigma`` for a spectral run).
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState

    # isinstance (Codex) — structurally guaranteed, immune to a duck-typed object
    # that accidentally carries a vor_hat attribute.
    if not isinstance(state, SpectralHydrostaticState):
        return state
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    g = spectral_pe_to_grid(state, grid, sigma)
    return HydrostaticState(
        u=Field(g["u"]), v=Field(g["v"]), T=Field(g["T"]),
        p_s=Field(g["p_s"]), phis=Field(g["phis"]),
    )


def owned_cell_valid_mask(layout_or_mask: Any, *, base_mask: Any = None) -> jax.Array:
    """Ranking validity mask restricting a DISTRIBUTED MPAS rank to its OWNED cells.

    A rank-local :class:`~legoesm.grids.voronoi.VoronoiMesh` carries HALO (ghost)
    cells it does not own (the standard MPAS ``n_owned`` ≤ ``n_local`` split, owned
    cells first).  Without masking, the worst-column ranking
    (:func:`compare_state_to_reference` → :func:`rank_worst_columns`) could pick a
    halo cell on a rank that does NOT own it, so the SAME physical cell is spun off
    into an LES + corrected on multiple ranks (double-count), or a stale halo value
    drives the rank.  Pass the active partition's owned-cell mask as the comparison
    ``valid_mask`` so only owned cells are ranked.

    ``layout_or_mask`` is either a :class:`VoronoiPartitionLayout` (anything with an
    ``owned_mask_cells`` attribute — ``(n_local_cells,)`` bool, ``True`` for the
    first ``n_owned`` cells) or the raw bool mask itself.  ``base_mask`` (optional,
    same column shape) is a further validity restriction (e.g. an ocean/land mask
    when a metric is ocean-only); the result is the elementwise AND, so a cell is
    rankable only if it is BOTH owned AND base-valid.  Returns a flat
    ``(n_local_cells,)`` bool array suitable for ``valid_mask=`` on the compare.

    Serial / single-rank / gathered-global runs need NO owned mask (every cell is
    owned); this helper is for the rank-local distributed-MPAS compare only.

    **Prerequisite, not the whole story.** This restricts each rank to ranking its
    OWNED cells; a complete distributed ranking ALSO needs a CROSS-RANK global top-k
    of the per-rank worst cells (else ``R`` ranks each pick ``n_worst`` → ``R ×
    n_worst`` LES, not the global ``n_worst``).  That gather is a SEPARATE distributed
    step — :func:`legoesm.training.distributed_manifest.gather_global_worst_columns`
    (iter 87; ``mpirun``-validated), wired into the distributed correction loop — so
    the distributed-MPAS ranking IS complete (owned mask + cross-rank top-k); the
    single-process runner needs neither (every cell owned).
    """
    owned_attr = getattr(layout_or_mask, "owned_mask_cells", None)
    raw = layout_or_mask if owned_attr is None else owned_attr
    owned = jnp.asarray(raw, dtype=bool)
    # The MPAS owned mask is the rank-local cell axis: strictly 1-D (n_local_cells,).
    # Require 1-D + EXACT shape equality so a (n,1)-vs-(n,) or length-1 broadcast
    # mismatch fails LOUDLY instead of silently mis-ordering / broadcasting (Codex).
    if owned.ndim != 1:
        raise ValueError(
            f"owned_cell_valid_mask: the owned mask must be 1-D (n_local_cells,); "
            f"got shape {tuple(owned.shape)}."
        )
    if base_mask is None:
        return owned
    base = jnp.asarray(base_mask, dtype=bool)
    if base.shape != owned.shape:
        raise ValueError(
            f"owned_cell_valid_mask: base_mask shape {tuple(base.shape)} != owned "
            f"mask shape {tuple(owned.shape)} — they must align EXACTLY on the same "
            f"rank-local cell axis (both 1-D (n_local_cells,))."
        )
    return owned & base


def ocean_valid_mask(
    land_fraction: Any,
    *,
    max_land_fraction: float = 0.5,
    base_mask: Any = None,
) -> jax.Array:
    """Ranking validity mask restricting the worst-column selection to OCEAN columns.

    In an AMIP run the OCEAN surface is PRESCRIBED (the SST forcing pins it), so a
    model-vs-ERA5 column bias over ocean is attributable to the ATMOSPHERIC column —
    including the turbulence closure being tuned — the right lever for the LES-closure
    correction.  Over LAND the surface is the model's OWN land model, carrying its own
    biases (soil moisture, snow, skin temperature) the closure cannot fix, so ranking a
    land column as "worst" spends the scarce LES budget where the correction is the wrong
    lever.  Pass this as the comparison ``valid_mask`` (or as the ``base_mask`` of
    :func:`owned_cell_valid_mask` under distributed MPAS) to rank ocean columns only.

    ``land_fraction`` is the model's static land fraction on the model grid (e.g.
    :meth:`legoesm.driver.model_driver.ModelDriver.static_land_fraction`); ANY shape is
    flattened ROW-MAJOR to the column order the ranking uses (the iter-36 column-ordering
    contract — structured grids flatten ``(nlat, nlon)`` row-major; MPAS ``(nCells,)`` is
    already flat).  A column is OCEAN (valid) where ``land_fraction <= max_land_fraction``;
    an aquaplanet/flat model (all-zero land fraction) yields an all-``True`` mask, so
    ``ocean_valid_mask`` is a SAFE no-op there.  ``base_mask`` (optional, same flat column
    shape) is ANDed in (a cell is rankable only if BOTH ocean AND base-valid).  Returns a
    flat ``(n_columns,)`` bool array suitable for ``valid_mask=`` on the compare.
    """
    if not (0.0 <= float(max_land_fraction) <= 1.0):
        raise ValueError(
            "ocean_valid_mask: max_land_fraction must be a fraction in [0, 1], got "
            f"{max_land_fraction!r} (0.0 = pure ocean only; 0.5 = majority ocean; the "
            "land fraction itself is in [0, 1])."
        )
    ocean = jnp.asarray(land_fraction).reshape(-1) <= float(max_land_fraction)
    if base_mask is None:
        return ocean
    base = jnp.asarray(base_mask, dtype=bool).reshape(-1)
    if base.shape != ocean.shape:
        raise ValueError(
            f"ocean_valid_mask: base_mask shape {tuple(base.shape)} != ocean mask shape "
            f"{tuple(ocean.shape)} — both must flatten to the same column axis "
            f"(n_columns,)."
        )
    return ocean & base


def column_state_from_hydrostatic(
    atm_state: Any,
    q_v: jax.Array,
    *,
    sst_K: jax.Array | None = None,
    precip_mm_day: jax.Array | None = None,
    mesh: Any = None,
) -> ColumnState:
    """Build a :class:`ColumnState` from a driver atmosphere state + ``q_v``.

    The model driver's prognostic atmosphere state (``HydrostaticState``: ``u`` /
    ``v`` / ``T`` / ``p_s``) carries moisture *separately* (``q_v`` is the tracer
    returned alongside the state, e.g. by ``driver.restart.load_restart``), so —
    unlike :func:`column_state_from_carry` — ``q_v`` is passed explicitly.

    **Run-mode agnostic (AMIP *and* CMIP).**  The comparison to ERA5 acts on the
    atmosphere state regardless of how the surface was driven, so this single
    adapter feeds both modes — the only difference is the SST source: in **AMIP**
    ``sst_K`` is the prescribed forcing; in **CMIP** (coupled) it is the
    interactive SST from the ocean component's surface state.  SST enters only
    the environment tag, so it is optional (falls back to the surface-air
    temperature in :func:`compare_state_to_reference`).

    The driver state stores ``u``/``v``/``T``/``p_s`` as :class:`Field` wrappers,
    so they are unwrapped to raw arrays (raw arrays also pass through).

    **MPAS / Voronoi (``v is None``).**  An MPAS state stores the EDGE-NORMAL
    velocity in ``u`` ``(nEdges, nlev)`` and has no cell ``v`` (``v is None``); pass
    the run's ``mesh`` (the :class:`~legoesm.grids.voronoi.VoronoiMesh`, i.e.
    ``driver.grid``) and the cell-centered geographic ``(u_east, v_north)``
    ``(nCells, nlev)`` are reconstructed via the Perot
    :func:`~legoesm.grids.voronoi.reconstruct_cell_velocity` — the SAME diagnostic
    wind MPAS uses, and frame-consistent with an ERA5 reference regridded to the
    cells (geographic east/north).  ``mesh=None`` with ``v is None`` raises (the
    edge→cell reconstruction needs the mesh).  Lat-lon / cubed states (``v`` set)
    pass through unchanged and ignore ``mesh``.

    **Scope (single-rank / full mesh):** ``reconstruct_cell_velocity`` runs on the
    supplied mesh; for a serial / global mesh that is the whole grid (matching the
    column extractor + ERA5 regrid scope).  A DISTRIBUTED MPAS run (``driver.grid``
    = the rank-local mesh) reconstructs rank-local cell winds (owned + halo); pass
    :func:`owned_cell_valid_mask` (the active partition's ``owned_mask_cells``) as
    the comparison ``valid_mask`` so only OWNED cells are ranked (a halo cell would
    otherwise be spun off + corrected on multiple ranks).  The cross-rank global
    top-k of the per-rank worst cells is the separate
    :func:`legoesm.training.distributed_manifest.gather_global_worst_columns` step
    (iter 87, wired as the loop's ``manifest_reducer``).
    """
    def _arr(x):
        return x.data if hasattr(x, "data") else x

    T = jnp.asarray(_arr(atm_state.T))
    dt = T.dtype
    v_field = getattr(atm_state, "v", None)
    if v_field is None:
        # MPAS edge-velocity state: reconstruct the cell-centered geographic wind.
        if mesh is None:
            raise ValueError(
                "column_state_from_hydrostatic: atm_state.v is None (an MPAS "
                "edge-velocity state) — pass mesh= (the run's VoronoiMesh, e.g. "
                "driver.grid) so the cell-centered (u_east, v_north) can be "
                "reconstructed from the edge-normal velocity."
            )
        from legoesm.grids.voronoi import VoronoiMesh, reconstruct_cell_velocity
        if not isinstance(mesh, VoronoiMesh):
            raise ValueError(
                "column_state_from_hydrostatic: atm_state.v is None but mesh is "
                f"{type(mesh).__name__}, not a VoronoiMesh — cannot reconstruct the "
                "cell wind from an MPAS edge-velocity state."
            )
        u_edge = jnp.asarray(_arr(atm_state.u), dtype=dt)
        # Guard a mismatched-but-same-rank mesh: a wrong mesh would otherwise
        # reconstruct physically WRONG winds silently (Codex). nEdges/nCells are
        # static, so these host-side checks are jit-safe.
        n_edges = int(mesh.nEdges)
        if u_edge.shape[0] != n_edges:
            raise ValueError(
                f"column_state_from_hydrostatic: edge velocity leading dim "
                f"{u_edge.shape[0]} != mesh.nEdges {n_edges} — wrong mesh for this "
                "MPAS state."
            )
        n_cells = int(mesh.nCells)
        if T.shape[0] != n_cells:
            raise ValueError(
                f"column_state_from_hydrostatic: cell field leading dim "
                f"{T.shape[0]} != mesh.nCells {n_cells} — wrong mesh for this "
                "MPAS state."
            )
        u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)
        u_cell = jnp.asarray(u_cell, dtype=dt)
        v_cell = jnp.asarray(v_cell, dtype=dt)
        # Carry the NATIVE edge velocity for the LES-forcing extractor (the exact
        # divergence/ω needs the edge-normal velocity, which the cell wind cannot
        # recover) — the comparison still uses the reconstructed cell u/v above.
        u_edge_native = u_edge
    else:
        u_cell = jnp.asarray(_arr(atm_state.u), dtype=dt)
        v_cell = jnp.asarray(_arr(v_field), dtype=dt)
        u_edge_native = None
    # Guard each optional per-column SURFACE field (sst_K, precip_mm_day) the SAME way as
    # the MPAS edge→cell wind above: both enter the comparison indexed by ATMOSPHERE column
    # flat-index (column_manifest: field[flat_i] — the SST env tag, the precip score term).
    # A coupled (CMIP) run with an ocean grid DIFFERENT from the atmosphere
    # (make_base_driver_builder ocean_grid=...) yields ocean_state.T_sfc on the OCEAN grid; a
    # mismatched shape would silently misalign every column (or index out of bounds), so fail
    # LOUD rather than corrupt the tags/score.  Same-grid coupling (ocean_grid=None) + AMIP's
    # prescribed SST are already on the atmosphere grid and pass unchanged.  Also unwraps a
    # Field-wrapped surface array for consistency with the atm-state fields above.
    def _checked_surface(field, name):
        if field is None:
            return None
        field = jnp.asarray(_arr(field), dtype=dt)
        if field.shape != T.shape[:-1]:
            raise ValueError(
                f"column_state_from_hydrostatic: {name} shape {field.shape} != the "
                f"atmosphere column grid {T.shape[:-1]} — a per-column surface field on a "
                "DIFFERENT grid (e.g. a coupled CMIP ocean grid via ocean_grid=...) would "
                "misalign every column (the comparison indexes by atmosphere column "
                "flat-index); regrid it to the atmosphere grid first."
            )
        return field

    sst_K = _checked_surface(sst_K, "sst_K")
    precip_mm_day = _checked_surface(precip_mm_day, "precip_mm_day")
    return ColumnState(
        T=T,
        q_v=jnp.asarray(q_v, dtype=dt),
        u=u_cell,
        v=v_cell,
        p_s=jnp.asarray(_arr(atm_state.p_s), dtype=dt),
        precip_mm_day=precip_mm_day, sst_K=sst_K,
        u_edge=u_edge_native,
    )
