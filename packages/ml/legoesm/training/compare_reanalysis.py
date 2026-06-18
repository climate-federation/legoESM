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
    """

    T: jax.Array
    q_v: jax.Array
    u: jax.Array
    v: jax.Array
    p_s: jax.Array
    precip_mm_day: jax.Array | None = None
    sst_K: jax.Array | None = None


class ColumnComparison(NamedTuple):
    """Result of comparing a model state to a reference state."""

    error_fields: ColumnErrorFields
    environment: ColumnEnvironmentFields
    manifest: list[ColumnRecord]


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
    approximated by the model's lowest-level air temperature with a warning-free
    fallback (documented; the driver should supply the prescribed SST).
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

    dsigma = sigma_half[1:] - sigma_half[:-1]
    weights = normalized_mass_weights(dsigma)

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

    if model.sst_K is not None:
        sst = model.sst_K
    else:
        # Documented fallback: surface-air-temperature proxy when the driver did
        # not thread the prescribed SST.  CAPE/shear are unaffected; only the
        # SST environment tag is approximate in this branch.
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
        error_fields=error_fields, environment=environment, manifest=manifest
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
    = the rank-local mesh) would reconstruct rank-local cell winds and needs an
    owned-cell ``valid_mask`` (or gather-to-global) before ranking — a follow-up.
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
    else:
        u_cell = jnp.asarray(_arr(atm_state.u), dtype=dt)
        v_cell = jnp.asarray(_arr(v_field), dtype=dt)
    return ColumnState(
        T=T,
        q_v=jnp.asarray(q_v, dtype=dt),
        u=u_cell,
        v=v_cell,
        p_s=jnp.asarray(_arr(atm_state.p_s), dtype=dt),
        precip_mm_day=precip_mm_day, sst_K=sst_K,
    )
