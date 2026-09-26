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
    conservative_positive_clip, is_borrow_eligible_tracer,
)
from legoesm.core.tracers import (
    TracerRegistry,
    init_tracers,
    make_full_moisture_registry,
    make_moisture_registry,
)
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import (
    refuse_cap_floor_on_fv,
    convection_config_for,
    build_physics_pipeline,
    gwd_config_for,
    required_microphysics_tracer_slots,
    turbulence_config_for,
    validate_microphysics_tracer_slots,
)
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.driver.restart import save_restart, load_restart

logger = logging.getLogger("legoesm.driver")

# Keeps a seeded soil moisture strictly inside the van-Genuchten retention
# range: psi_from_theta is singular at saturation and at the residual.
_THETA_EDGE_GUARD = 1.0e-3



def _external_forcing_active(
    radiation_ok: bool,
    ozone_active: bool,
    aerosol_active: bool,
    aerosol_lw_active: bool,
    ghg_active: bool,
    experiment_active: bool,
) -> bool:
    """Whether the per-step external-forcing dict (o3/aerosol/ghg) must be built.

    Single source of truth for the ``_ext_forcing`` gate used on both the
    hydrostatic and spectral full-physics AMIP paths.  ``radiation_ok`` is the
    per-path scheme test (both paths need a gas-radiation scheme, but the
    spectral path is rrtmgp-only).  ``aerosol_lw_active`` MUST be included:
    volcanic stratospheric LW aerosol supplied alone (no ozone / SW aerosol /
    GHG / experiment) still has to reach RRTMGP's LW absorption slot.
    """
    return radiation_ok and (
        ozone_active or aerosol_active or aerosol_lw_active
        or ghg_active or experiment_active
    )


def _scatter_flat_columns(flat_arr, layout, n_tile):
    """Scatter a FLATTENED-column array ``(6*n*n, ...)`` to this rank's owned
    faces, returning ``(n_local*n*n, ...)`` in the identical face-major
    row-major column order the rank-local ``ColumnAdapter`` uses.

    The multilayer-land per-column state/params live in flattened column space
    (``ncol = 6*n*n``), while the cube face-scatter (:func:`layout.scatter`)
    operates on a leading FACE axis of size 6.  So reshape ``(6*n*n, ...) ->
    (6, n, n, ...)``, scatter the owned faces, then flatten back to
    ``(n_local*n*n, ...)``.  The column ordering matches the scattered
    ``_physics_lat`` / ``f_land`` exactly (same faces, same within-face raster),
    so soil columns land on their own faces — the invariant #769 depends on.
    """
    from legoesm.parallel.layout import scatter as _scatter
    trailing = flat_arr.shape[1:]
    faces = flat_arr.reshape((6, n_tile, n_tile) + trailing)
    local = _scatter(faces, layout)                       # (n_local, n, n, ...)
    n_local = local.shape[0]
    return local.reshape((n_local * n_tile * n_tile,) + trailing)


def _scatter_1based_columns(arr, layout, n_tile):
    """Scatter a CLM 1-based per-column array ``(ncol+1, ...)`` (index 0 unused,
    ``begp=1``) to this rank's owned columns, returning ``(n_local+1, ...)``.

    The CLM ``mlcanopy`` pytree stores every patch field 1-based, so strip index 0,
    scatter the ``(ncol, ...)`` body with the SAME face-major column ownership the
    soil / lat / params use (:func:`_scatter_flat_columns`), then re-prepend the
    original index-0 sentinel row.  The body path IS the (MPI-validated) 0-based
    column scatter, so this 1-based variant is correct by composition — only the
    strip/prepend wrapping is new.
    """
    body = arr[1:]                                            # (ncol, ...)
    local_body = _scatter_flat_columns(body, layout, n_tile)  # (n_local*n*n, ...)
    return jnp.concatenate([arr[:1], local_body], axis=0)     # (n_local+1, ...)


def _slice_grid_info_to_rank(grid_info, layout, n_tile, global_ncol):
    """Slice a GLOBAL per-column CLM-ML ``grid_info`` tuple to this rank's owned
    columns and renumber each entry's patch index ``.p`` to LOCAL 1-based.

    Scatters a global-index array to learn which global columns this rank owns, in
    the SAME face-major local order the scattered canopy fields use, picks those
    ``GridInfo`` entries, and sets ``.p = k+1`` for local position ``k`` — the
    interface realigns by ``.p`` and requires patches ``1..n_local``.
    """
    lidx = np.asarray(_scatter_flat_columns(
        jnp.arange(global_ncol, dtype=jnp.int32), layout, n_tile)).astype(int)
    return tuple(grid_info[int(g)]._replace(p=k + 1) for k, g in enumerate(lidx))


def _gather_flat_columns(local_arr, layout, n_tile, root_only=False):
    """Inverse of :func:`_scatter_flat_columns` — gather a rank-local
    flattened-column array ``(n_local*n*n, ...)`` back to the global
    ``(6*n*n, ...)`` (for a global checkpoint / diagnostic)."""
    from legoesm.parallel.layout import gather as _gather
    trailing = local_arr.shape[1:]
    n_local = local_arr.shape[0] // (n_tile * n_tile)
    faces = local_arr.reshape((n_local, n_tile, n_tile) + trailing)
    glob = _gather(faces, layout, root_only=root_only)    # (6, n, n, ...)
    return glob.reshape((6 * n_tile * n_tile,) + trailing)


def _scatter_voronoi_columns(arr, partition):
    """Scatter a global per-column array ``(nCells_global, ...)`` to this
    rank's LOCAL cells (owned + halo), in the same order the rank-local MPAS
    physics columns use (#1321).

    The Voronoi analogue of :func:`_scatter_flat_columns`.  Halo columns are
    included, not trimmed, so every per-column land array keeps the same
    leading length as the rank-local atmospheric state and no consumer needs a
    special case.  Integrating them is redundant but not wrong: the land step
    is column-local (no lateral soil coupling) and a halo column sees the same
    exchanged atmospheric forcing as its owner, so it tracks the owner exactly.
    ``gather_voronoi_field`` keeps only the owned prefix on the way out.
    """
    from legoesm.parallel.voronoi_partition import scatter_to_local
    return scatter_to_local(arr, partition, "cell")


def _scatter_1based_voronoi_columns(arr, partition):
    """CLM 1-based ``(nCells_global + 1, ...)`` -> ``(n_local_cells + 1, ...)``.

    Same strip / scatter / re-prepend composition as
    :func:`_scatter_1based_columns`, over the Voronoi cell partition.
    """
    body = _scatter_voronoi_columns(arr[1:], partition)
    return jnp.concatenate([arr[:1], body], axis=0)


def _land_columns_to_local(arr, partition):
    """Cut ONE global per-column land array down to this rank's cells.

    Land state written by anything global -- a gathered checkpoint, an offline
    spin-up restart -- arrives at ``nCells_global`` while everything the run
    holds is already rank-local.  Both per-column leaf shapes the land tile
    uses are handled here, including the CLM 1-based one whose row 0 is not a
    column, so no caller has to know which is which.  A no-op when there is no
    partition (serial) or the leaf is not per-column, so callers can map it
    over a whole pytree.
    """
    if partition is None or not hasattr(arr, "shape") or arr.ndim < 1:
        return arr
    n_global = int(partition.nCells_global)
    if int(arr.shape[0]) == n_global:
        return _scatter_voronoi_columns(arr, partition)
    if int(arr.shape[0]) == n_global + 1:          # CLM 1-based
        return _scatter_1based_voronoi_columns(arr, partition)
    return arr


def _map_flat_column_leaves(tree, n_tile, global_ncol, fn):
    """Apply ``fn(leaf, n_tile)`` to every array leaf of ``tree`` whose leading
    axis equals ``global_ncol`` (a per-column field); leave all other leaves
    (scalars, config, differently-shaped params) untouched.  Used to
    scatter/gather the multilayer-land params + carbon pytrees, which mix
    per-column arrays with scalar hyperparameters."""
    def _leaf(x):
        if (hasattr(x, "shape") and x.ndim >= 1
                and int(x.shape[0]) == global_ncol):
            return fn(x, n_tile)
        return x
    return jax.tree_util.tree_map(_leaf, tree)


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


def _standalone_cloud_config(cfg, cloud_scheme: str,
                             allow_convective_cloud: bool = False):
    """Tuned ``CloudConfig`` for the standalone (MPAS/spectral) radiation path.

    Mirrors the FV pipeline's ``build_cloud_config`` call (#689) so the tuned
    experiment-level cloud scalars (``cloud_rh_crit`` / ``cloud_q_c_diagnostic``
    / Xu-Randall knobs) reach the standalone backends too (#870 Phase 1) —
    previously these paths silently ran ``CloudConfig`` defaults.

    ``convective_cloud``: honoured on the MPAS path
    (``allow_convective_cloud=True`` — 2026-07-24: the hydrostatic radiation
    fn reads the LAGGED ``PhysicsState.conv_precip`` carry the convection
    module publishes, closing the old "does not thread conv_precip" gap);
    still FORCED OFF on the spectral path, whose lean loop refuses stateful
    physics and therefore has no ``conv_precip`` carry to read — enabling it
    there would trip ``compute_cloud_properties``' loud misconfiguration
    guard by design.  Returns ``None`` (=> scheme-default config) when the
    scheme is "none".
    """
    if cloud_scheme == "none":
        if getattr(cfg, "cloud_cap_floor_on", False):
            raise ValueError("cloud_cap_floor_on with cloud scheme 'none': the polar-cap "
                             "radiative floor would never be applied")
        return None
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config

    _conv_cloud = bool(getattr(cfg, "convective_cloud", False))
    if _conv_cloud and allow_convective_cloud:
        logger.info(
            "convective_cloud=True ACTIVE on the MPAS standalone path: "
            "Slingo-1987-inspired surrogate cumulus fraction driven by the one-step-lagged "
            "PhysicsState.conv_precip carry (the convection module's "
            "column-integrated in-updraft rain production). Schemes with "
            "no rain split publish zero — their cumulus fraction is zero."
        )
    elif _conv_cloud:
        # LOUD, not silent (repo doctrine): the spectral lean loop refuses
        # stateful physics, so there is no conv_precip carry to read — the
        # forced drop stays visible in the log.
        logger.warning(
            "convective_cloud=True is FORCED OFF on the spectral "
            "standalone radiation path: its lean loop carries no "
            "PhysicsState, so there is no conv_precip for the Slingo-1987-inspired surrogate "
            "fraction. The FV pipeline and the MPAS lane honour the "
            "setting."
        )

    return build_cloud_config(
        cloud_scheme,
        convective_cloud=(_conv_cloud and allow_convective_cloud),
        rh_crit=getattr(cfg, "cloud_rh_crit", None),
        q_c_diagnostic=getattr(cfg, "cloud_q_c_diagnostic", None),
        conv_cloud_max=getattr(cfg, "cloud_conv_cloud_max", None),
        conv_cloud_condensate=getattr(cfg, "cloud_conv_cloud_condensate", None),
        # Sub-grid cloud-optics inhomogeneity + the diagnostic-condensate
        # selectors.  These were MISSING here while the FV pipeline forwarded
        # them (physics_pipeline.py:2067-2077), so on the MPAS lane
        # --cloud-optics-inhomogeneity / --cloud-inhomogeneity-factor /
        # --cloud-fsd / --cloud-diagnostic-condensate-scheme /
        # --cloud-adiabatic-lwc-rate were accepted by the CLI and then
        # SILENTLY DROPPED — the run used the scheme defaults regardless of
        # the flag (codex review, 2026-07-30).  The docstring above claims
        # this function mirrors the FV call; it now actually does.
        cloud_inhomogeneity_factor=getattr(
            cfg, "cloud_inhomogeneity_factor", None),
        cloud_optics_inhomogeneity=getattr(
            cfg, "cloud_optics_inhomogeneity", None),
        cloud_fsd=getattr(cfg, "cloud_fsd", None),
        cloud_partial_coverage_optics=getattr(
            cfg, "cloud_partial_coverage_optics", None),
        cloud_vertical_overlap_optics=getattr(
            cfg, "cloud_vertical_overlap_optics", None),
        cloud_n_subcolumns=getattr(cfg, "cloud_n_subcolumns", None),
        diagnostic_condensate_scheme=getattr(
            cfg, "cloud_diagnostic_condensate_scheme", None),
        adiabatic_lwc_rate=getattr(cfg, "cloud_adiabatic_lwc_rate", None),
        p_xr=getattr(cfg, "cloud_p_xr", None),
        alpha_xr=getattr(cfg, "cloud_alpha_xr", None),
        clubb_cf_override_strength=getattr(
            cfg, "cloud_clubb_cf_override_strength", None),
        clubb_cf_override_floor=getattr(
            cfg, "cloud_clubb_cf_override_floor", None),
        saturation_scheme=getattr(cfg, "cloud_saturation_scheme", None),
        cover_condensate_q_ref=getattr(cfg, "cloud_cover_condensate_q_ref", None),
        cap_floor_on=getattr(cfg, "cloud_cap_floor_on", None),
        cap_floor_lat_deg=getattr(cfg, "cloud_cap_floor_lat_deg", None),
        cap_floor_p_max_pa=getattr(cfg, "cloud_cap_floor_p_max_pa", None),
        cap_floor_cf=getattr(cfg, "cloud_cap_floor_cf", None),
        cap_floor_q_c=getattr(cfg, "cloud_cap_floor_q_c", None),
    )


# Post-step hard-saturation-adjustment diagnostics cadence (loud counter): log
# roughly twice a day at dt~100 s.  A logging cadence + a count epsilon, not
# physics.
_HARD_SAT_LOG_CADENCE_STEPS = 432
# Cadence (steps) of the sedimentation sub-step report; same as the
# hard-saturation log so a run has ONE diagnostic rhythm.
_SED_SUBSTEP_LOG_CADENCE_STEPS = _HARD_SAT_LOG_CADENCE_STEPS


def effective_sed_substeps_cap(micro_cfg, scheme, cfg=None) -> int:
    """The sedimentation sub-step cap the lane will actually run.

    ``micro_cfg`` is the RESOLVED (already flat-threaded) microphysics
    config, so its scheme leaf is read first: it already carries whatever the
    flat ``ExperimentConfig`` field forwarded, and a deck that sets only the
    leaf is honoured.  A NON-DEFAULT flat field wins, because threading
    forwards it onto the leaf; the leaf survives only where the flat field
    sits at its default (GLM 2026-09-22: the old wording overstated this).
    Falls back to the flat field and then to the scheme default.
    """
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    if scheme != "morrison":
        # no other scheme sub-steps sedimentation; returning the Morrison
        # default here would name a cap nothing runs
        return 0
    _leaf = None
    if micro_cfg is not None:
        # a bare scheme leaf (what the finite-volume pipeline carries) or the
        # threaded container keyed by scheme name
        _leaf = (micro_cfg if hasattr(micro_cfg, "sed_cfl_substeps_max")
                 else getattr(micro_cfg, scheme, None))
    _cap = getattr(_leaf, "sed_cfl_substeps_max", None)
    if _cap is None and cfg is not None:
        _cap = getattr(cfg, "morrison_sed_cfl_substeps_max", None)
    return int(MorrisonConfig._field_defaults["sed_cfl_substeps_max"]
               if _cap is None else _cap)


def require_cpu_for_strict_sedimentation() -> None:
    """Refuse at startup when the strict sedimentation abort has no CPU.

    The abort is an equinox ``error_if``, i.e. a host callback: it needs a
    CPU device to place its inputs on, which the GPU-only lane
    (``JAX_PLATFORMS=cuda``) does not have.  Refusing here beats dying mid
    run (the first CAM6 60-day arm died at day 2).

    The declared platform list is preferred over querying devices: every
    rank reads the same string, so the refusal is collective by
    construction, and no backend is initialized to reach it (GLM
    2026-09-22).  Only when nothing is declared does this query JAX, after
    the multicontroller init (querying devices initializes the backend, and
    ``jax.distributed.initialize`` must run first -- codex 2026-09-22).
    """
    _plat = os.environ.get("JAX_PLATFORMS", "")
    _cpu_exc = None
    if _plat.strip():
        # jax accepts platform names case-insensitively (GLM 2026-09-22)
        _has_cpu = "cpu" in [t.strip().lower() for t in _plat.split(",")]
    else:
        import jax as _jax
        try:
            _has_cpu = bool(_jax.devices("cpu"))
        except Exception as _exc:   # RuntimeError today; jax promises nothing
            _cpu_exc = f"{type(_exc).__name__}: {_exc}"
            print(f"WARNING: no CPU device for the strict sedimentation "
                  f"abort: {_cpu_exc}", file=sys.stderr)
            _has_cpu = False
    if _has_cpu:
        return
    raise SystemExit(
        "morrison_sed_cfl_substeps_strict=True needs a CPU device for its "
        "host-callback abort, but none is available (JAX_PLATFORMS="
        f"{_plat or '<unset>'!r}).  Add 'cpu' to JAX_PLATFORMS or run with "
        "the flag off (the required sub-step count is still reported as a "
        "diagnostic)."
        + (f"  Backend error: {_cpu_exc}" if _cpu_exc else ""))


def _resolved_sed_flag(cfg, micro_cfg, leaf_field: str, flat_field: str):
    """A sedimentation switch as the RUN resolves it: the scheme leaf first
    (it already carries whatever the flat field forwarded), the flat field
    as the fallback -- the same precedence as the cap resolver."""
    _leaf = None
    if micro_cfg is not None:
        _leaf = (micro_cfg if hasattr(micro_cfg, leaf_field) else
                 getattr(micro_cfg, "morrison", None))
    _v = getattr(_leaf, leaf_field, None)
    return getattr(cfg, flat_field, False) if _v is None else _v


def strict_sed_abort_requested(cfg, micro_cfg=None) -> bool:
    """Whether THIS run arms the strict sedimentation abort.

    Read from the resolved leaf as well as the flat field: a deck that sets
    only the leaf still needs the CPU the host-callback abort lands on
    (codex 2026-09-22 -- the guard used to read the flat flag alone, so a
    leaf-only deck skipped it and would have died mid-run).
    """
    if getattr(cfg, "microphysics", None) != "morrison":
        return False
    return bool(_resolved_sed_flag(cfg, micro_cfg, "sed_cfl_substeps_strict",
                                   "morrison_sed_cfl_substeps_strict"))


def warn_sed_substeps_unreported(cfg, lane: str = "current",
                                 micro_cfg=None) -> bool:
    """Say out loud that THIS lane cannot report a clamped fall.

    The MPAS loop and the per-step loop publish the required sub-step count
    and log its window maximum; every other lane (compiled segments, the
    spectral loop, the lat-band and tiled-cube SPMD loops, fv3_duo) reduces
    inside ``lax.scan`` or discards the microphysics output, and carries no
    slot for the count, so a clamp there is invisible unless the strict
    abort is on.  Called from the lane DISPATCHER, so a new lane inherits
    the warning instead of inheriting silence (codex 2026-09-22).  A silent gap is what this round
    was opened to remove (GLM 2026-09-22), so the gap announces itself once
    at run start.  Returns True when the warning was emitted.
    """
    if getattr(cfg, "microphysics", None) != "morrison":
        return False            # no other scheme sub-steps sedimentation
    if not _resolved_sed_flag(cfg, micro_cfg, "sed_cfl_substeps",
                              "morrison_sed_cfl_substeps"):
        return False
    if strict_sed_abort_requested(cfg, micro_cfg):
        return False            # a clamp aborts, so nothing can hide
    logger.warning(
        "microphysics sedimentation: CFL sub-stepping is ON but the %s lane "
        "does not report the required sub-step count - a clamped fall would "
        "be silent here; the MPAS and per-step lanes do report it, and "
        "morrison_sed_cfl_substeps_strict=True makes a clamp fatal on every "
        "lane", lane)
    return True


def _sed_substeps_slot() -> int:
    """Index of ``sed_substeps_required`` in the MPAS ``_sfc_diag`` tuple.

    Derived from the SHARED slot contract (``MPAS_SFC_DIAG_EXTRA_KEYS``
    entries start at tuple slot 3), never hand-counted.
    """
    from legoesm.core.state import (
        MPAS_SFC_DIAG_BASE_KEYS,
        MPAS_SFC_DIAG_EXTRA_KEYS,
    )
    return (len(MPAS_SFC_DIAG_BASE_KEYS)
            + MPAS_SFC_DIAG_EXTRA_KEYS.index("sed_substeps_required"))


def _sed_published_count(sfc_diag, slot, counts):
    """The step's sub-step count from either publication route, or None.

    ``counts`` is the count a NON-MPAS lane returns directly (the
    finite-volume pipeline's ``PhysicsOutput.sed_substeps_required``);
    ``sfc_diag`` is the MPAS lean-loop export tuple.  Both routes exist
    because the two lanes publish diagnostics differently (codex 2026-09-22:
    reporting only inside the MPAS loop left the finite-volume lane silent).
    """
    if counts is not None:
        return jnp.max(getattr(counts, "data", counts))
    if not sfc_diag or len(sfc_diag) <= slot or sfc_diag[slot] is None:
        return None
    return jnp.max(getattr(sfc_diag[slot], "data", sfc_diag[slot]))


def sed_substeps_window_max(sfc_diag, running=None, slot=None, counts=None):
    """Running maximum of the published sub-step count, on device.

    Called EVERY step (no host sync): a single step's overflow between two
    reports would otherwise be lost, because the export slot holds only the
    latest step's count (codex 2026-09-22).  Returns the updated running
    maximum, or ``running`` unchanged when the lane published no count.
    """
    if slot is None:
        slot = _sed_substeps_slot()
    _req = _sed_published_count(sfc_diag, slot, counts)
    if _req is None:
        return running
    return _req if running is None else jnp.maximum(running, _req)


def report_sed_substep_overflow(sfc_diag, cap, step, slot=None, running=None,
                                counts=None):
    """Log the microphysics' required CFL sedimentation sub-step count.

    ``sfc_diag`` is the MPAS lean-loop export tuple (slot contract
    ``core.state.MPAS_SFC_DIAG_EXTRA_KEYS``); ``running`` is the window
    maximum accumulated by :func:`sed_substeps_window_max` since the last
    report.  Returns the reported maximum, or None when neither source
    carries a count (sub-stepping off, or a non-Morrison scheme).

    A count above the static cap means the loop CLAMPED: mass is still
    conserved, but that species fell slower than its terminal speed.  Without
    this line the clamp is invisible unless the run enables the strict abort.
    """
    if slot is None:
        slot = _sed_substeps_slot()
    _req = _sed_published_count(sfc_diag, slot, counts)
    if running is not None:
        _req = running if _req is None else jnp.maximum(running, _req)
    if _req is None:
        return None
    _n = int(_req)
    if _n > cap:
        logger.warning(
            "microphysics sedimentation: a column needed %d CFL sub-steps but "
            "the loop is capped at %d, so it CLAMPED (mass conserved, the "
            "species fell slower than its terminal speed) - raise "
            "morrison_sed_cfl_substeps_max or shorten the physics step; "
            "at step %d", _n, cap, step)
    else:
        logger.info(
            "microphysics sedimentation: max %d CFL sub-steps required of %d "
            "available, at step %d", _n, cap, step)
    return _n
_HARD_SAT_LOG_QV_EPS = 1.0e-9        # [kg/kg] count a point as "drained" above this


def _mpas_zenith_ocean_albedo(lat, day, orbit=None):
    """Briegleb (1992) open-ocean albedo at the daytime-effective daily-mean
    solar cosine, per cell.

    ``mu = Q_day / (S_0 * f_day)`` is the cosine a column sees AVERAGED OVER
    ITS SUNLIT HOURS, which is the right weighting for an albedo that is held
    fixed for the whole day.  The eccentricity factor is divided out because
    ``mu`` is a geometric cosine, not a flux.  ``S_0`` cancels from the ratio,
    so the value passed is immaterial and a fixed reference is used.

    Polar night gives ``f_day -> 0``; the floor keeps ``mu`` finite there and
    the albedo is irrelevant because there is no sunlight to reflect.
    """
    from legoesm.atmosphere.physics.radiation.solar import (
        daily_mean_insolation, daylight_fraction, earth_sun_distance_factor,
    )
    from legoesm.surface_albedo import OceanAlbedoConfig, ocean_albedo

    s_0 = constants.S_0
    eccf = (earth_sun_distance_factor(day, orbit) if orbit is not None else 1.0)
    q_day = daily_mean_insolation(lat, day, s_0, orbit=orbit) / eccf
    f_day = daylight_fraction(lat, day, orbit=orbit)
    mu = jnp.clip(q_day / (s_0 * jnp.maximum(f_day, 1.0e-6)), 0.0, 1.0)
    return ocean_albedo(mu, OceanAlbedoConfig(method="zenith"))



def _qv_level_conserving_floor(q_new, area, owned_mask=None):
    """Positivity floor that conserves the per-level area integral.

    The biharmonic branch of the smoother has no maximum principle, so a bare
    ``max(q, 0)`` CREATES water wherever it clips (measured: it seeded the
    cold-start ice explosion, 2026-09-15).  Clip the negatives, then rescale
    the positive cells of each level by

        factor_l = sum_c A_c q_l / sum_c A_c max(q_l, 0)   in [0, 1]

    so ``sum_c A_c q_out`` equals ``sum_c A_c q_new`` per level to roundoff;
    clipping only adds mass, so the factor removes exactly what it added.
    Under MPI the partial sums are OWNED-masked (halo cells are duplicates)
    and combined with a differentiable allreduce, so every rank scales by the
    same factor; halo cells are clipped and scaled pointwise like the rest.
    Pure jnp: no Python branch on a traced value, and the tiny-denominator
    guard divides a sanitised denominator so neither pass sees NaN.
    (Drafted by GLM-5.2, 2026-09-16.)
    """
    from legoesm.parallel.reductions import global_sum_if_distributed
    area = jnp.asarray(area)
    w = area if owned_mask is None else jnp.where(owned_mask, area, 0.0)
    q_clip = jnp.maximum(q_new, 0.0)
    num = global_sum_if_distributed(jnp.sum(w[:, None] * q_new, axis=0))
    den = global_sum_if_distributed(jnp.sum(w[:, None] * q_clip, axis=0))
    tiny = jnp.finfo(q_new.dtype).tiny
    safe_den = jnp.where(den > tiny, den, jnp.ones_like(den))
    factor = jnp.where(den > tiny, jnp.clip(num / safe_den, 0.0, 1.0), 0.0)
    return q_clip * factor[None, :]


def _mpas_qv_smooth_step(q_v, mesh, nu, dt, nu4=0.0, mid_refresh=None,
                         owned_mask=None):
    """MPAS post-step horizontal q_v smoothing (array-level, testable).

    UNWEIGHTED SCVT del2 (``scalar_del2_cell_3d``) + a q>=0 floor, mirroring
    the FV lane's module-scope ``_apply_qv_smoothing`` so the exact driver
    code is unit-tested directly (not just a replicated expression).  Plain
    (no dp weighting) is deliberate: the default hybrid coordinate's surface
    dp can be <= 0 over high terrain, which a mass-weighted form would divide
    by (Inf/NaN).  Monotone under the setup-time CFL guard
    ``nu*dt*scalar_del2_cell_cfl_factor(mesh) <= 1`` (the driver enforces
    <= 0.5), so the floor is then a no-op (to roundoff) and the per-level
    ``sum_c A_c q_c`` integral is conserved (in exact arithmetic; to roundoff —
    ~1e-7 relative in fp32).  The floor only sanitises finite negatives from a
    violated bound, NOT pre-existing NaN/Inf in ``q_v`` (max(NaN,0)=NaN).

    Parameters
    ----------
    q_v : jax.Array, shape (nCells, nlev) — vapour mixing ratio [kg/kg].
    mesh : VoronoiMesh
    nu : float — del2 diffusivity [m^2/s].
    dt : float — step [s].
    nu4 : float — del4 (biharmonic) diffusivity [m^4/s]; 0 disables the term.
        The biharmonic is SCALE-SELECTIVE: it separates two-cell from four-cell
        structure by a factor sixteen where the Laplacian separates them by
        four, so it can hold grid-scale noise down without flattening the
        resolved humidity gradients.  It has NO maximum principle, so with it
        on the positivity floor is load-bearing; it is the per-level
        conserving borrow ``_qv_level_conserving_floor`` (clip, then rescale
        the level's positives), so ``sum_c A_c q_c`` is conserved to roundoff
        and no water is created.  The driver's setup guard enforces
        ``nu4*dt*g_max^2 <= 0.5``.
    mid_refresh : Callable(array) -> array, optional — distributed-only halo
        refresh for the biharmonic's intermediate Laplacian.

    Returns
    -------
    jax.Array, shape (nCells, nlev) — smoothed, floored q_v (q_v dtype).
    """
    from legoesm.core.operators_voronoi import (
        scalar_del2_cell_3d,
        scalar_del4_cell_3d,
    )
    lap = scalar_del2_cell_3d(q_v, mesh)
    if nu4 <= 0.0:
        # BIT-IDENTICAL to the pre-biharmonic path, including the association
        # order of ``dt * nu * lap`` -- the existing bit-exactness test holds
        # the del2-only lane to equality, not to a tolerance.
        return jnp.maximum(q_v + dt * nu * lap.astype(q_v.dtype), 0.0)
    del4 = scalar_del4_cell_3d(q_v, mesh, mid_refresh=mid_refresh)
    tend = (nu * lap + nu4 * del4).astype(q_v.dtype)
    return _qv_level_conserving_floor(q_v + dt * tend, mesh.areaCell,
                                      owned_mask=owned_mask)


def clear_sky_pass_effective(
    *, clear_sky_diag: bool, radiation: str, spatial_feed_on: bool,
    feed_steps_reached: bool = True,
) -> tuple[bool, str | None]:
    """Should the MPAS lane actually run the clouds-off second radiation pass?

    ``--clear-sky-diag`` buys a second radiation solve per radiation step
    (~2x the radiation cost) for the CMOR rsutcs/rlutcs pair.  Two
    configurations consume that cost and can publish NOTHING, and both used
    to be indistinguishable from success (#843; the exact silent-drop class
    of #1385):

    * ``radiation="none"`` — ``_make_hydrostatic_combined`` builds no
      radiation module at all, so the tendency never carries
      ``sw_up_toa_clr``/``lw_up_toa_clr`` and ``make_radiation_physics``'s
      unsupported-model guard is never even reached.  The host-side cloud
      trio (clt/clwvi/clivi) is UNAFFECTED — it derives from the q_c/q_i
      tracers and the collector's own cloud scheme — so only the flux pair
      is lost here.
    * the SPATIAL CMOR feed is off.  All five new fields are written only
      into the spatial (``Amon``) accumulator — the zonal monthly one holds
      none of them — so ``monthly_means`` alone is NOT enough.  Without a live
      spatial feed nothing is published at all.  (A multi-rank Voronoi cell
      partition IS feedable since #1517, via the owned-cell gather, so it no
      longer disqualifies the pass.)
    * the run never REACHES a diagnostic boundary (``--days 1
      --diag-days 5``, or any ``diag_days`` whose interval exceeds the
      remaining steps): the feed loop simply never fires, so every CMOR
      field — not just these five — is absent, and the second pass would be
      paid on every radiation step for nothing.

    Returns ``(run_the_pass, reason_or_None)``.  A non-None reason is a
    human-readable clause for a LOUD warning; the caller warns rather than
    raising, because dry / Held-Suarez / throughput runs legitimately carry
    the flag from a launcher default and aborting them over a diagnostic
    would be worse than skipping it.  Slots 10/11 have exactly ONE consumer
    (``_feed_mpas_cmip_accumulators``), so skipping the pass when that
    consumer is absent loses nothing.
    """
    if not clear_sky_diag:
        return False, None
    if not spatial_feed_on:
        return False, ("the SPATIAL CMOR feed is off (needs --cmip-output; "
                       "--monthly-means alone feeds only the zonal "
                       "accumulator, which carries none of these fields), so "
                       "NONE of rsutcs/rlutcs/clt/clwvi/clivi can be "
                       "published")
    if not feed_steps_reached:
        return False, ("this run never reaches a diagnostic boundary "
                       "(diag_days exceeds the remaining run length), so the "
                       "CMOR feed never fires and no field at all is written")
    if radiation == "none":
        return False, ("radiation='none', so rsutcs/rlutcs cannot be "
                       "produced (the cloud trio clt/clwvi/clivi is "
                       "unaffected and still published)")
    return True, None


class _MPASSfcFluxAccum:
    """Per-step accumulator for the MPAS eager loop's ``_sfc_diag`` flux
    slots so the CMOR feed hands INTERVAL MEANS to the accumulators instead
    of the end-of-interval instantaneous snapshot (issue #1353: at
    ``diag_days=1`` every snapshot lands at the same model clock time, so
    the "monthly mean" of a day/night field like rsut kept the full
    instantaneous diurnal pattern while labeled ``time: mean``).

    Covers slots 0..7 of the ``_sfc_diag`` contract (0 sw_net_sfc and
    1 lw_net_sfc, read only by the energy tracker; 2 precip, 3 rlut,
    4 rsut, 5 rsdt, 6 hfss, 7 hfls) — the strongly diurnal flux fields —
    plus the clear-sky TOA pair (10 rsutcs, 11 rlutcs; #843 lean-lane
    port), which is only ever non-None when ``--clear-sky-diag`` is on
    (empty slots add nothing: dump/restore stay byte-identical when off).
    State-derived fields (tas/ta/ua/...) use the independent hourly feed.

    Sums stay on device (lazy ``jnp`` adds, no per-step host sync); the one
    device->host transfer happens in :meth:`mean` at diag cadence.  Upstream,
    each slot refreshes at its own cadence (radiation slots hold their last
    full-radiation value across held-radiation sub-steps), so every per-step
    sample is the physics' current flux estimate — the same "held" semantics
    the compiled cube path's ``held_*``/``*_accum`` mechanism averages.

    CHECKPOINTED (codex-2 finding 2): partial-interval sums/counts ride the
    MPAS checkpoint npz (``cmor_fluxsum_<slot>`` / ``cmor_fluxcnt_<slot>``,
    absent when empty) and are re-staged through ``_carry_aux`` on load, so
    a mid-interval wallclock exit + restart RESUMES the interval mean
    instead of dropping the pre-checkpoint samples.  On the aligned chain
    (checkpoint cadence a multiple of diag cadence) the diag block (feed +
    reset) runs before the checkpoint block at the same ``(step+1)``
    boundary, so the payload is empty — byte-identical checkpoints.

    Precision: sums are float64 on an x64 runtime.  On an fp32 runtime the
    float64 request silently degrades to float32; the worst case for a
    1-day interval at dt=75 s (n=1152 samples, |flux| ~ 500 W/m^2) is a
    relative error ~ n*eps/2 ~ 7e-5, i.e. ~0.03 W/m^2 — below CMOR
    reporting precision, documented rather than engineered around.
    """

    # Slots 0 and 1 (sw_net_sfc, lw_net_sfc, both +into surface) were added
    # 2026-09-05 for the #1354 energy budget.  They are NOT part of the CMOR
    # feed -- there is no surface-radiation table entry -- but they are the
    # LARGEST term in the column energy budget (~77 W/m^2 against a ~20 W/m^2
    # signal), and the energy tracker was reading them as fixed-clock-time
    # snapshots.  Measured on job 9632045: sampled sensible heat 8.0 W/m^2
    # against an accumulated 20.5, and an apparent leak of +34.7 W/m^2 where
    # accumulated channels gave ~11.  Accumulating costs one device-side add
    # per step per slot and is what makes the budget answerable at all.
    SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
    #: The slots the column energy budget reads (sw/lw net sfc, lw_up, sw_up,
    #: sw_dn, hfss, hfls).  After the first radiation call every one of them
    #: is non-None on EVERY step (the MPAS model holds the last radiation
    #: value across held-radiation sub-steps, primitive_eq_mpas.step), which
    #: is what lets ``window_ready`` demand equal per-slot counts.
    ENERGY_SLOTS = (0, 1, 3, 4, 5, 6, 7)

    def __init__(self, expected_steps: int = 0, window_start_day: float = 0.0,
                 dt_s: float = 0.0):
        self._sum: dict = {}   # slot -> device-side running sum
        self._n: dict = {}     # slot -> sample count
        # Model steps seen this interval (counted even when the physics
        # exported nothing) and the count a COMPLETE interval must have.
        # A window that is SHORT (first one after an off-cadence restart or
        # a feed-off link) or OVERLONG (a resumed window whose cadence or
        # calendar changed under it) must NOT be published as a full
        # interval mean (codex-6/7): the flux fields are withheld.
        self._steps: int = 0
        self.expected_steps: int = int(expected_steps)
        # Absolute simulated day at which the CURRENT window started — the
        # identity a resumed window is validated against (a cadence change
        # or a ``--restart-start-day`` time rebase makes the carried samples
        # incommensurable with the new calendar, codex-7).
        self.window_start_day: float = float(window_start_day)
        # Timestep the samples were taken at — part of the identity: a dt
        # change makes an equal STEP count a different DURATION, so
        # continuity cannot be judged with the new run's dt (codex-8).
        self.dt_s: float = float(dt_s)

    def add(self, sfc_diag) -> None:
        """Accumulate one step's ``_sfc_diag`` tuple (None-safe per slot)."""
        self._steps += 1
        if not sfc_diag:
            return
        for i in self.SLOTS:
            if len(sfc_diag) > i and sfc_diag[i] is not None:
                x = sfc_diag[i]
                # Accumulate in float64 (codex-1 minor 6): an fp32 running
                # sum over a long interval loses ~n*eps relative precision.
                # Under a non-x64 JAX config this politely degrades to
                # float32 — no worse than the source precision.
                x = jnp.asarray(
                    x.data if hasattr(x, "data") else x, dtype=jnp.float64)
                if i in self._sum:
                    self._sum[i] = self._sum[i] + x
                    self._n[i] += 1
                else:
                    self._sum[i] = x
                    self._n[i] = 1

    def mean(self, slot: int):
        """Interval-mean host array for *slot*, or None if never fed."""
        n = self._n.get(slot, 0)
        if n == 0:
            return None
        return np.asarray(self._sum[slot], dtype=np.float64) / float(n)

    def has_samples(self) -> bool:
        """True if any slot accumulated at least one sample this interval."""
        return bool(self._n)

    def window_ready(self, slots) -> bool:
        """True iff the window is complete AND every slot in *slots* was
        accumulated over the SAME number of samples.  A checkpoint written
        before a slot existed restores the others with a full count while
        the new slot only sees the post-restart remainder; ``is_complete()``
        passes (it counts steps, not per-slot samples) and the short slot
        would be published as a full interval mean (codex, #1354)."""
        counts = {self._n.get(i, 0) for i in slots}
        return self.is_complete() and len(counts) == 1 and counts != {0}

    def is_complete(self) -> bool:
        """True iff this interval saw EXACTLY the step count a complete
        window needs.  A SHORT window (first one after an off-cadence
        restart or a feed-off link) covers less time than its label claims;
        an OVERLONG one (a resumed window whose cadence changed under it)
        covers more — neither may be published (codex-6/7).
        ``expected_steps`` of 0 disables the check (unit-level use)."""
        return (self.expected_steps <= 0
                or self._steps == self.expected_steps)

    def dump(self) -> dict:
        """Checkpoint payload: ``cmor_fluxsum_<slot>``/``cmor_fluxcnt_<slot>``
        host arrays (empty dict when no samples)."""
        out: dict = {}
        for slot, s in self._sum.items():
            out[f"cmor_fluxsum_{slot}"] = np.asarray(s)
            out[f"cmor_fluxcnt_{slot}"] = np.int64(self._n[slot])
        if out:
            # Steps seen so far this interval — the resumed link continues
            # counting toward ``expected_steps`` so the completeness gate
            # sees the WHOLE window, not just the post-restart part — plus
            # the window's IDENTITY (its expected length and start day), so
            # a resume under a changed cadence or a rebased calendar is
            # detected and dropped instead of silently mixed (codex-7).
            out["cmor_fluxsteps"] = np.int64(self._steps)
            out["cmor_fluxexpected"] = np.int64(self.expected_steps)
            out["cmor_fluxday0"] = np.float64(self.window_start_day)
            out["cmor_fluxdt_s"] = np.float64(self.dt_s)
        return out

    @classmethod
    def checkpoint_keys(cls) -> tuple[str, ...]:
        """The exact key names this class reads/writes (whitelist)."""
        return tuple(
            f"cmor_flux{kind}_{slot}"
            for slot in cls.SLOTS for kind in ("sum", "cnt")
        ) + cls.IDENTITY_KEYS

    #: Identity keys a non-empty payload MUST carry (codex-8): expected
    #: window length, window start day, and the dt the samples were taken
    #: at — continuity cannot be judged with the NEW run's dt alone.
    IDENTITY_KEYS = ("cmor_fluxsteps", "cmor_fluxexpected",
                     "cmor_fluxday0", "cmor_fluxdt_s")

    def restore(self, staged: dict, *, resume_day: float | None = None,
                dt_s: float | None = None) -> int:
        """Resume a partial interval from checkpoint-staged keys (popping
        them ALWAYS).  Returns the number of slots restored — 0 when the
        payload is discarded as incommensurable.

        FAIL-LOUD on a malformed payload (half a sum/count pair, a
        non-positive count, missing step count): a silently dropped slot
        would publish a shortened interval mean as if it were complete
        (codex-3).

        DISCARD (return 0, keep this accumulator empty) when the payload's
        window IDENTITY does not match this run (codex-7): a different
        expected window length (cadence changed, incl. the per-link
        ``diag_days <= 0`` sentinel) or a window whose stored start day +
        elapsed steps does not land on this run's resume day (an explicit
        ``--restart-start-day`` calendar rebase).  Those samples cannot be
        combined with the new window without misdating the mean.
        """
        n_restored = 0
        _steps = staged.pop("cmor_fluxsteps", None)
        _expected = staged.pop("cmor_fluxexpected", None)
        _day0 = staged.pop("cmor_fluxday0", None)
        _dt_old = staged.pop("cmor_fluxdt_s", None)
        # PHASE 1 — structural validation of the sum/count pairs, BEFORE any
        # identity judgement: a half pair or a non-positive count is
        # CORRUPTION (raise), not an incommensurable-but-well-formed window
        # (discard).  Pops every key it inspects.
        _pairs: list[tuple[int, object, int]] = []
        for slot in self.SLOTS:
            s = staged.pop(f"cmor_fluxsum_{slot}", None)
            c = staged.pop(f"cmor_fluxcnt_{slot}", None)
            if s is None and c is None:
                continue
            if s is None or c is None:
                raise ValueError(
                    f"CMOR flux checkpoint payload for slot {slot} is "
                    f"incomplete (sum={'present' if s is not None else 'MISSING'}, "
                    f"count={'present' if c is not None else 'MISSING'}) — "
                    f"refusing to resume a corrupt partial interval.")
            c_int = int(c)
            if c_int <= 0:
                raise ValueError(
                    f"CMOR flux checkpoint slot {slot} has a non-positive "
                    f"sample count ({c_int}) — corrupt payload.")
            _pairs.append((slot, s, c_int))
        if _pairs and _steps is None:
            raise ValueError(
                "CMOR flux checkpoint payload restored sums but carries "
                "no 'cmor_fluxsteps' — cannot judge window completeness.")
        # PHASE 2 — window identity.
        _has_sums = bool(_pairs)
        _discard_reason = None
        if _has_sums and (_expected is None or _day0 is None
                          or _dt_old is None):
            # A payload missing any identity key cannot be validated at all
            # (codex-8) — refuse rather than resume blind.
            _discard_reason = "payload is missing window-identity keys"
        elif _expected is not None and self.expected_steps > 0 and (
                int(_expected) != self.expected_steps):
            _discard_reason = (
                f"expected window length changed "
                f"({int(_expected)} -> {self.expected_steps} steps)")
        elif (_dt_old is not None and dt_s
                and abs(float(_dt_old) - float(dt_s)) > 1e-9):
            _discard_reason = (
                f"timestep changed ({float(_dt_old):g} -> {float(dt_s):g} s): "
                f"the same step count is a different duration")
        elif (_day0 is not None and resume_day is not None
                and _dt_old is not None and _steps is not None):
            # Continuity to FLOATING-POINT tolerance only — a half-step slack
            # would accept a genuine rebase or a mixed-duration window
            # (codex-8).  Scale the tolerance with the day magnitude.
            _continued = float(_day0) + int(_steps) * float(_dt_old) / 86400.0
            _tol = 1e-9 * max(1.0, abs(float(resume_day)))
            if abs(_continued - float(resume_day)) > _tol:
                _discard_reason = (
                    f"calendar discontinuity (window would continue at day "
                    f"{_continued:.9f}, run resumes at "
                    f"{float(resume_day):.9f})")
        if _discard_reason is not None:
            logger.warning(
                "  CMOR flux accumulator: DISCARDED the checkpoint's partial "
                "interval — %s.  The first window of this run collects from "
                "scratch (its flux fields are withheld until a full window "
                "completes).", _discard_reason)
            if "discontinuity" in _discard_reason:
                # The CMOR monthly/daily/zonal SIDECAR is restored by the
                # driver before any calendar rebase and is NOT validated
                # here — its buckets may belong to the pre-rebase calendar
                # (pre-existing, codex-8).  Say so loudly; the safe action
                # is to delete the sidecar for a rebased run.
                logger.warning(
                    "  NOTE: the CMOR monthly/daily accumulator SIDECAR is "
                    "not calendar-validated — on a rebased restart delete "
                    "cmor_accum_day_*.npz, or its pre-rebase buckets will "
                    "mix into the new calendar.")
            return 0
        # PHASE 3 — commit (validated pairs only).
        for slot, s, c_int in _pairs:
            self._sum[slot] = jnp.asarray(s, dtype=jnp.float64)
            self._n[slot] = c_int
            n_restored += 1
        if n_restored:
            self._steps = int(_steps)
            if _day0 is not None:
                self.window_start_day = float(_day0)
        return n_restored

    def reset(self, window_start_day: float | None = None) -> None:
        """Start a new window.  *window_start_day* (the absolute day the new
        window begins at) keeps the identity current for the checkpoint."""
        self._sum.clear()
        self._n.clear()
        self._steps = 0
        if window_start_day is not None:
            self.window_start_day = float(window_start_day)


def _mpas_hard_saturation_poststep(T, q_v, q_c, p_full, dt,
                                   hard_threshold, hard_max_heating_K,
                                   ice_curve=False, q_i=None):
    """MPAS POST-STEP hard-saturation-adjustment drain (array-level, testable).

    Applies the reviewed hard-saturation-adjustment DRAIN
    (:func:`legoesm.atmosphere.physics.microphysics._warm_rain.hard_saturation_drain`
    -- bracketed-bisection on-curve solve + per-step latent-heating & vapour
    rate limit) on the FINAL post-dycore state, on the caller-supplied
    full-level pressure ``p_full`` (the driver passes
    ``self.sigma.pressure_at_full(p_s)``, correct on both the sigma and the
    hybrid lanes).  Conserves ``c_pd*T + L_v*q_v`` exactly: the SAME drain
    ``dq = rate*dt`` is removed from vapour, added to cloud water, and heats T by
    ``(L_v/c_pd)*dq``.  A pure drain (>= 0) -- no spurious evaporation.

    Parameters
    ----------
    T, q_v, p_full : array
        Temperature [K] (ncol, nlev), vapour [kg/kg] (ncol, nlev), full-level
        pressure [Pa] (ncol, nlev).
    q_c : array or None
        Cloud water [kg/kg] (ncol, nlev), the condensate RECIPIENT.  ``None`` if
        the run carries no q_c tracer -> the drain is a NO-OP (there is no
        reservoir to receive the condensate; draining vapour with nowhere to put
        it would LOSE total water).  On the moist warm-rain MPAS path q_c is
        always present (the driver hook additionally gates on it).
    dt : float
        Dycore step [s].
    hard_threshold, hard_max_heating_K : float
        RH trigger and per-step latent-heating cap [K] from the scheme config.

    ice_curve : bool, default False
        Mixed-phase drain (TTL dehydration fix): gate + land on the
        w(T)-blended liquid/ice saturation curve with the matching blended
        latent heat (frozen at the pre-adjustment T — the SAME array the
        drain's internal solve uses, so ``c_pd*T + L_eff(T0)*q_v`` is
        conserved exactly), and route each cell's condensate by phase:
        the liquid fraction w(T0) to ``q_c``, the ice fraction to ``q_i``
        when that tracer exists (else everything to ``q_c`` — Morrison's
        own freezing then handles the phase, documented approximation).
    q_i : array or None
        Cloud-ice tracer [kg/kg] (ncol, nlev) — the cold-cell condensate
        recipient under ``ice_curve``.  Ignored when ``ice_curve`` is off.

    Returns
    -------
    (T_new, q_v_new, q_c_new, q_i_new, dq) : tuple of arrays
        Updated fields and the per-step condensed increment ``dq`` [kg/kg]
        (>= 0), for diagnostics.  ``q_i_new`` is None when no q_i was given.
        All inputs are returned unchanged (dq = 0) when ``q_c is None`` --
        total water is conserved.
    """
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        hard_saturation_drain,
        mixed_phase_l_over_cp,
        mixed_phase_liquid_fraction,
    )
    if p_full.shape != T.shape:
        raise ValueError(
            f"p_full must be the full-level pressure {T.shape}, got "
            f"{p_full.shape} (a bare sigma array is no longer accepted).")
    if q_c is None:
        # No condensate reservoir -> cannot conserve total water by draining;
        # do nothing (the moist path always has q_c).
        return T, q_v, None, q_i, jnp.zeros_like(q_v)
    # The ice curve DEPOSITS to cloud ice; without a q_i reservoir it cannot do
    # so, and heating with the blended (L_s-weighted) latent heat while binning
    # the condensate as LIQUID q_c would inject (1-w)*L_f*dq of spurious energy.
    # So degrade FULLY to the energy-exact liquid drain when q_i is absent (the
    # rate AND the heating both liquid).  validate_strict forbids ice_curve
    # without Morrison (q_i present), so this is a defensive fallback only.
    _use_ice = bool(ice_curve) and (q_i is not None)
    rate = hard_saturation_drain(T, q_v, p_full, dt, hard_threshold,
                                 hard_max_heating_K, ice_curve=_use_ice)
    dq = rate * dt
    if _use_ice:
        # Heating with the SAME frozen-at-T0 blended latent heat the solve
        # used; condensate split by the same w(T0) -> cloud water / cloud ice.
        # (Landing q_c below the LIQUID curve at cold T would re-evaporate, so
        # the ice fraction MUST go to q_i for the drain to stick; the caller
        # seeds N_i for that ice mass.)
        _l_cp = mixed_phase_l_over_cp(T).astype(T.dtype)
        T_new = T + _l_cp * dq
        w_liq = mixed_phase_liquid_fraction(T).astype(q_v.dtype)
        q_c_new = q_c + w_liq * dq
        q_i_new = q_i + (1.0 - w_liq) * dq
    else:
        # Liquid path (ice_curve off, or requested without a cloud-ice
        # reservoir): condense to q_c with L_v -> c_pd*T + L_v*q_v exact.
        T_new = T + (constants.L_v / constants.c_pd) * dq
        q_c_new = q_c + dq
        q_i_new = q_i
    q_v_new = q_v - dq
    return T_new, q_v_new, q_c_new, q_i_new, dq


# Air-density floor for the Cooper ice-number/mass conversion, matching
# Morrison's ``_RHO_FLOOR`` (module_mp_graupel density floor in divisions) so the
# seed ceiling is Morrison-consistent at low density (without floors the p/(R_dT)
# ceiling loosens as 0.1/rho aloft — ~4x at 15 hPa, far more near the top).
_ICE_SEED_RHO_FLOOR = 0.1        # [kg/m^3]

# MultiLayerLandState fields that are numerical CACHES, not prognostic state:
# never written to a checkpoint and never required by one, so a checkpoint
# written before the field existed still restarts (the cache cold-starts).
_LAND_ML_CACHE_FIELDS = ("canopy_x",)


def _seed_nucleated_ice_number(N_i, dq_i, ice_nuc_mass, n_i_nuc_max, p_full, T):
    """Seed cloud-ice NUMBER [1/kg] for freshly deposited ice mass ``dq_i``.

    A two-moment scheme (Morrison) needs a matching NUMBER for the deposited ice
    or the new mass is ill-posed: M2005 diffusional growth self-gates as
    ``EPSI ~ N_i^(2/3)`` (orphan ice with ``N_i = 0`` can neither grow nor
    sublimate), and the fall-speed PSD slope ``lambda_i=(rho_ci pi N_i/q_i)^1/3``
    is degenerate at ``N_i = 0`` (ice cannot sediment out).  Give each new
    crystal the Morrison nucleation mass ``mi0 = 4/3 pi rho_ci r_nuc^3`` (mean
    size = the nucleation radius), i.e. ``dN_i = dq_i / mi0`` -- the SAME
    mass<->number closure Morrison's Cooper nucleation uses
    (``dq_i_nuc = dN_i_nuc * mi0``).

    The added number is CAPPED at Morrison's Cooper ice-number ceiling
    ``N_i_nuc_max`` [1/m^3], converted to per-mass via
    ``rho_air = p/(R_d T)`` floored at ``_ICE_SEED_RHO_FLOOR`` (0.1 kg/m^3) --
    exactly Morrison's own Cooper target conversion
    (``kc2 = min(...) / clip(rho, 0.1)``): a large (heating-cap-sized) deposit
    then GROWS the crystals (bigger mean size)
    rather than over-populating number -- an uncapped ``dq_i/mi0`` at the ~2 g/kg
    cap seeds ~1e8 m^-3, ~300x the 5e5 m^-3 (500/L) ceiling, which would perturb
    deposition and sedimentation.  Number is additive and carries no latent heat,
    so this does not affect the water/energy budgets the drain already conserves.
    ``rho_air`` uses dry ``p/(R_d T)``; Morrison's diagnostic uses moist/virtual-T
    density, so above the floor this cap is marginally TIGHTER (~0.6%, i.e.
    conservative -- fewer crystals) and at the low-density floor the two coincide.
    """
    rho_air = p_full / (constants.R_d * jnp.maximum(T, 1.0))
    n_i_max_perkg = n_i_nuc_max / jnp.maximum(rho_air, _ICE_SEED_RHO_FLOOR)
    d_n_raw = jnp.maximum(dq_i, 0.0) / jnp.maximum(ice_nuc_mass, 1.0e-30)
    headroom = jnp.maximum(n_i_max_perkg - N_i, 0.0)   # 0 if already at ceiling
    return N_i + jnp.minimum(d_n_raw, headroom)


def _is_mpas_cell_partitioned(drv) -> bool:
    """True iff *drv* is running a MULTI-rank MPAS/Voronoi cell partition —
    i.e. every per-cell array it holds is a rank-local ``(n_local_cells,)``
    slice of the global mesh, not the global field.

    ONE predicate for every consumer of that fact, so the branches cannot
    drift apart (a mismatch would silently mean local regrid weights fed with
    global arrays, or the reverse):

    * :meth:`ModelDriver._create_diagnostics` — give the collector the GLOBAL
      mesh + globally-gathered ``fx`` fields;
    * :meth:`ModelDriver._feed_mpas_cmip_accumulators` — route to the gather;
    * :meth:`ModelDriver.save_checkpoint` / :meth:`ModelDriver._run_mpas` — do
      NOT persist or resume the rank-local per-cell CMOR flux sums.

    A module function rather than a method so the diagnostic-feed test doubles
    (plain ``SimpleNamespace`` stand-ins) evaluate it identically to the real
    driver.
    """
    return (getattr(drv, "_voronoi_layout", None) is not None
            and (getattr(drv, "_mpi_world_size", 1) or 1) > 1)


def _validate_number_convention(payload, tracer_names) -> None:
    """Refuse a restart whose droplet number uses the old per-VOLUME units.

    Cloud and rain number are stored PER MASS [1/kg] since 2026-08-14 so the
    dycores' mass-mixing-ratio advection is the right operator for them. A file
    written before that holds [1/m^3]; reloading it as per-mass is wrong by the
    air density — roughly 1.2 near the surface and 2.5 in the upper
    troposphere — and nothing else in the file distinguishes the two. Fail
    loudly rather than continue with silently wrong droplet sizes.
    """
    if not any(n in ("N_c", "N_r") for n in tracer_names):
        return
    stamp = payload.get("number_convention") if hasattr(payload, "get") else None
    if stamp is None and "number_convention" in payload:
        stamp = payload["number_convention"]
    stamp = None if stamp is None else str(np.asarray(stamp).item())
    if stamp != "per_mass":
        raise ValueError(
            "This restart carries cloud/rain droplet number written under the "
            "old PER-VOLUME convention (no 'number_convention' stamp); the "
            "model now stores them PER MASS [1/kg]. Loading it as-is would "
            "scale droplet number by the air density. Re-run from the initial "
            "state, or divide the stored N_c/N_r by air density and add "
            "number_convention='per_mass' to the file."
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
        # #1028: set by _enter_persistent_dgrid() at run start; False means the
        # prognostic winds are cell-centred (every lane except the cube
        # hydrostatic one that opts in).
        self._persistent_dgrid = False
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
        # Set True once the per-column land state/params/carbon are scattered to
        # owned faces under cube-face MPI (_setup_parallel); satisfies the
        # distributed-multilayer guard in run().  Faces are embarrassingly
        # parallel so the gathered N-rank soil is bit-identical to serial.
        self._land_ml_scattered = False
        self._land_ml_n_tile = None
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
        # Gate for the MPAS CMOR spatial/zonal accumulator feed (set per run
        # in _run_mpas; False here so any other path is a safe no-op).
        self._mpas_cmip_feed_on = False
        # #1353 per-step flux accumulator (built in _run_mpas when the CMOR
        # feed is on; None keeps every other path a no-op).
        self._mpas_sfc_accum = None
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

    def _conserving_floor(self, field, dp=None):
        """Column-conserving non-negativity for a per-mass field
        ``(..., nlev)``: clip negatives, rescale the column's positives so
        the TRUE layer-mass (dp) weighted integral is unchanged (owner
        decision 2026-08-16: conserving form always — the plain
        ``max(q, 0)`` invents mass at every overdraw/undershoot, the
        MPAS-century +30 kg/m2/yr class). dp from ``pressure_at_half`` is
        hybrid-correct and reduces to ``dsigma * p_s`` on pure sigma (the
        per-column ``p_s`` cancels in the rescale); non-positive dp (broken
        hybrid layer over terrain) is zero-weighted rather than divided by —
        mirrors the MPAS floors stage. Pass a precomputed ``dp`` when
        flooring several fields against the same ``p_s`` (the double-moment
        loop) to skip recomputing it per field."""
        if dp is None:
            _ph = self.sigma.pressure_at_half(self.state.p_s.data)
            dp = jnp.maximum(_ph[..., 1:] - _ph[..., :-1], 0.0)
        fixed, _created = conservative_positive_clip(field, dp, axis=-1)
        return fixed

    def _apply_double_moment_tendencies(self, phys_out, dt) -> None:
        """Integrate the ice/snow/graupel + number tracers one step from the
        matching ``PhysicsOutput`` tendencies, floored non-negative by the
        column-conserving borrow (every one of these is per-mass and
        borrow-eligible; a plain clip invented number at x2.2/day compound on
        century3 until N_i overflowed). No-op unless the full-moisture
        registry is active (q_i present)."""
        if not (isinstance(self.tracers, dict)
                and self.tracer_registry.has("q_i")):
            return
        _upd = {
            "q_i": phys_out.dq_i_dt, "q_s": phys_out.dq_s_dt,
            "q_g": phys_out.dq_g_dt, "N_c": phys_out.dN_c_dt,
            "N_r": phys_out.dN_r_dt, "N_i": phys_out.dN_i_dt,
        }
        _ph = self.sigma.pressure_at_half(self.state.p_s.data)
        _dp = jnp.maximum(_ph[..., 1:] - _ph[..., :-1], 0.0)
        for k, tend in _upd.items():
            if self.tracers.get(k) is not None:
                assert is_borrow_eligible_tracer(k), k
                self.tracers[k] = self._conserving_floor(
                    self.tracers[k] + dt * tend, dp=_dp)

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
        if any(k in ("N_c", "N_r") for k in dm):
            # Same per-mass droplet-number stamp as the MPAS/spectral writers.
            base["dmtr_number_convention"] = np.asarray("per_mass")
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
            # The soil COLUMN this state belongs to.  The field shapes below
            # record the layer COUNT only, and two columns with the same count
            # can span different depths (8 layers over 3 m vs over 6.375 m), in
            # which case a resumed run reads every soil value at the wrong
            # depth.  Not a state field, so it gets its own namespace and the
            # restore's exact-field-set check ignores it.
            base["land_soil_dz"] = self._land_soil_dz()
            for _f, _v in self._land_ml_state._asdict().items():
                # Optional fields (TgC, surface_water) may be None — np.asarray
                # would pickle a 0-d object array into the npz and crash the
                # load-side jnp.asarray. Skip; restore only replaces saved keys.
                # Cache fields (the canopy warm start) are skipped too: they
                # carry no physics and an older reader would refuse the key.
                if _v is not None and _f not in _LAND_ML_CACHE_FIELDS:
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
        _stamp = self._carry_aux.pop("dmtr_number_convention", None)
        _keys = [k for k in self._carry_aux if k.startswith("dmtr_")]
        if any(k in ("dmtr_N_c", "dmtr_N_r") for k in _keys):
            _stamp = None if _stamp is None else str(np.asarray(_stamp).item())
            if _stamp != "per_mass":
                raise ValueError(
                    "checkpoint carries cloud/rain droplet number written "
                    "under the old PER-VOLUME convention (no "
                    "'dmtr_number_convention' stamp); the model now stores "
                    "them PER MASS [1/kg]. Re-run from the initial state, or "
                    "divide the stored N_c/N_r by air density and add the "
                    "stamp."
                )
        for key in _keys:
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

    def _check_land_soil_dz(self, ckpt_dz) -> None:
        """Refuse a checkpoint whose soil column is not this run's.

        The field-shape checks that follow see the layer COUNT only, so a state
        equilibrated over 6.375 m resumes into a 3 m run without a word — every
        soil temperature and moisture value read at the wrong depth.  A
        checkpoint written before the column was recorded carries nothing to
        check; that is only refused for a run whose column is not the historical
        default, so existing chains keep working.
        """
        want = self._land_soil_dz()
        if want is None:
            return
        calibrated = bool(getattr(self.config, "land_calibrated_physics", False))
        if ckpt_dz is None:
            if calibrated:
                raise ValueError(
                    "the checkpoint predates soil-column recording, so its layer "
                    f"depths cannot be checked against this run's column "
                    f"({want.tolist()} m, total {float(want.sum()):.4f} m). This "
                    "run is on the calibrated column, not the historical "
                    "default, so the checkpoint is most likely on the wrong one. "
                    "Start from a land initial condition on this column instead.")
            logger.warning(
                "Checkpoint predates soil-column recording; its layer depths "
                "CANNOT be checked against this run's column (%s m). If it was "
                "produced on a different column the soil profile is being read "
                "at the wrong depths.", want.tolist())
            return
        got = np.asarray(ckpt_dz, dtype=np.float64).reshape(-1)
        from legoesm.land.restart import soil_dz_matches
        if not soil_dz_matches(got, want):
            raise ValueError(
                f"checkpoint soil column does not match this run: checkpoint "
                f"layer thicknesses {got.tolist()} m (total "
                f"{float(got.sum()):.4f} m) vs current {want.tolist()} m (total "
                f"{float(want.sum()):.4f} m). The soil profile would be read at "
                "the wrong depths.")

    def _land_soil_dz(self):
        """This run's soil layer thicknesses [m], or ``None`` with no soil tile.

        The one place the column is turned into a checkpointable record, so the
        save and the check below cannot describe different things.
        """
        cfg = getattr(self.physics, "land_ml_cfg", None)
        if cfg is None:
            return None
        from legoesm.land.soil_grid import make_soil_grid
        return np.asarray(make_soil_grid(cfg.soil_grid).dz, dtype=np.float64)

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
        # The soil column the checkpoint's state belongs to (own namespace, so
        # it is not mistaken for a state field below).  Popped unconditionally
        # so it never leaks into the next re-save.
        _ckpt_dz = self._carry_aux.pop("land_soil_dz", None)
        keys = [k for k in self._carry_aux if k.startswith("land_ml_")]
        if not keys:
            return
        self._check_land_soil_dz(_ckpt_dz)
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
                    if getattr(template, f) is not None
                    and f not in _LAND_ML_CACHE_FIELDS}
        got = set(popped) - set(_LAND_ML_CACHE_FIELDS)
        if got != expected:
            raise ValueError(
                "land_ml checkpoint field set does not match the current "
                "MultiLayerLandState: "
                f"missing={sorted(expected - got)}, "
                f"unexpected={sorted(got - expected)}. Refusing to build a "
                "mixed restart state (the missing prognostic columns would "
                "silently stay at cold-start values).")
        # Under MPAS cell-partition MPI the checkpoint holds the GLOBAL
        # columns (save_checkpoint gathers them) while ``template`` is already
        # rank-local, so cut each restored field to this rank before the shape
        # check — otherwise every resumed distributed multilayer run fails the
        # comparison below (#1321).
        # ``getattr``: several tests drive this method with a SimpleNamespace
        # fake that carries only the carry_aux + land fields.
        _vl = getattr(self, "_voronoi_layout", None)
        _part = _vl.partition if _vl is not None else None

        def _to_local(arr):
            return _land_columns_to_local(arr, _part)

        fields = {}
        for name, val in popped.items():
            arr = _to_local(jnp.asarray(val))
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

    def _preflight_land_inputs(self) -> None:
        """Refuse a land initial condition that belongs to a different soil column.

        The layer COUNT is checked later, when the state is loaded, but two
        columns with the same count can span different depths (8 layers over 3 m
        against 8 over 6.375 m), and then every soil temperature and moisture
        value is read at the wrong depth with nothing anywhere complaining.

        Done HERE, from the config and the restart file's own record of its
        thicknesses, because both are the same on every rank — unlike the load
        itself, which happens only on ranks that own land.
        """
        cfg = self.config
        ic_path = getattr(cfg, "land_ic_path", "")
        if not (getattr(cfg, "use_multilayer_land", False) and ic_path):
            return
        if not os.path.exists(ic_path):
            # Symmetric refusal: the loader would otherwise report this only on
            # ranks that own land, leaving the rest waiting (codex round 11).
            raise FileNotFoundError(
                f"land_ic {ic_path!r} does not exist (or is a broken symlink). "
                "A multilayer run was asked to start from a spun-up land state "
                "and cannot.")
        from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
        from legoesm.land.config import biophysics_lmip_two_leaf_setup
        if getattr(cfg, "land_calibrated_physics", False):
            # The soil column a spun-up land state must match is the one the
            # calibration defines, so this reads the SAME setup the tile is
            # built from — a second copy of the geometry here would let a
            # restart pass the check and then run on a different column.
            want_grid = biophysics_lmip_two_leaf_setup()["soil_grid"]
        else:
            want_grid = SoilGridConfig(
                n_layers=cfg.multilayer_n_layers,
                total_depth=cfg.multilayer_soil_depth)
        want = np.asarray(make_soil_grid(want_grid).dz, dtype=np.float64)
        from legoesm.land.restart import (
            load_land_restart_soil_dz, soil_dz_matches,
        )
        got = load_land_restart_soil_dz(ic_path)
        if got is None:
            if getattr(cfg, "land_calibrated_physics", False):
                raise ValueError(
                    f"land IC {ic_path} predates soil-column recording, so its "
                    f"layer depths cannot be checked against this run's column "
                    f"({want.tolist()} m, total {float(want.sum()):.4f} m). This "
                    "run is on the calibrated column, not the historical "
                    "default, so the IC is most likely on the wrong one. Re-run "
                    "the land spin-up on this run's column.")
            return
        if not soil_dz_matches(got, want):
            raise ValueError(
                f"land IC {ic_path} belongs to a different soil column: its "
                f"layer thicknesses are {got.tolist()} m (total "
                f"{float(got.sum()):.4f} m) against this run's {want.tolist()} m "
                f"(total {float(want.sum()):.4f} m). The soil profile would be "
                "read at the wrong depths. Re-run the land spin-up on this "
                "run's column.")

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

        # Land inputs whose validity depends on FILE CONTENTS rather than the
        # config alone.  Placement is exact and both halves matter: AFTER the
        # bootstrap, because it builds the soil grid through the shared
        # (JAX-backed) helper rather than re-deriving the geometry, and nothing
        # may create a JAX array before the backend is chosen (codex round 11);
        # BEFORE ``_create_grid``, which is the first step where the ranks
        # diverge.  Everything it reads — the config and one array out of one
        # file — is identical on every rank, so the raise is symmetric, unlike
        # the equivalent check inside the land setup, which runs only on ranks
        # that own land and would leave the rest waiting (codex round 10).
        self._preflight_land_inputs()

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
                transition_exponent=getattr(gc, "transition_exponent", 3),
            )
        elif gc.vertical_coord == "cam_l32":
            from legoesm.grids.vertical import make_cam6_l32_levels
            self.sigma = make_cam6_l32_levels()
        elif gc.vertical_coord != "sigma":
            raise ValueError(
                f"unknown vertical_coord {gc.vertical_coord!r}; expected 'sigma', 'hybrid' or 'cam_l32'")
        else:
            from legoesm.grids.vertical import create_sigma_coordinate
            self.sigma = create_sigma_coordinate(
                gc.nlev, sigma_top=gc.sigma_top,
                tropopause_refine=getattr(gc, "tropopause_refine", 1.0),
                sigma_refine=getattr(gc, "sigma_refine", 0.12),
                refine_width=getattr(gc, "sigma_refine_width", 0.45),
                layout=gc.sigma_layout)

        _lid = (f", sigma_top={gc.sigma_top:g}, layout={gc.sigma_layout}"
                if gc.vertical_coord == "sigma" else "")
        logger.info(f"  Grid: {gc.grid_type} {gc.resolution}, "
              f"{gc.nlev} levels ({gc.vertical_coord}{_lid})")

        # Cache lat/lon accessors via GridProtocol for grid-agnostic use
        self._grid_lat = self.grid.grid_lat
        self._grid_lon = self.grid.grid_lon

    def _create_topography(self) -> None:
        """Load or generate topography and land-sea mask."""
        from legoesm.grids.topography import (
            TopographyConfig, load_real_topography,
            gaussian_mountain, phis_from_topography,
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
            # An idealized mountain is a DYNAMICAL forcing, not a statement
            # about the surface, so this case is all ocean unless a real mask
            # is named below.  Deriving f_land from the elevation here labelled
            # the WHOLE GLOBE land: the Gaussian bell has no cutoff, so z_s > 0
            # in every cell (measured at T31: minimum elevation 3.3e-44 m,
            # 100% of cells, land fraction 1.0), which is not a mountain
            # coastline by any reading.
            self._f_land = jnp.zeros(shape_2d, dtype=_sd)
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

        # Attach land_frac for convection and orographic GWD on grids carrying
        # the field (VoronoiMesh and GaussianGrid).  Where it comes from:
        # idealized topography (flat, gaussian) -> zeros, a real elevation file
        # -> fraction from the loaded elevation, land_mask_path -> the loaded
        # mask, overriding either.
        if (getattr(self.grid, "land_frac", "no-field") is None
                and self._f_land is not None):
            self.grid = self.grid._replace(
                land_frac=jnp.asarray(self._f_land, dtype=_sd).reshape(-1))
        # Under MPI the compiled step does NOT use ``self.grid`` -- it closes
        # over ``self._voronoi_layout.local_mesh``, which was built during grid
        # creation, i.e. BEFORE this attach.  Without this refresh the mask
        # reaches the serial lane and silently misses the distributed one, so a
        # run would get land physics or not depending on how it was launched.
        # That is the failure mode this whole land-mask work is about; keep the
        # two copies in step.
        if (self._voronoi_layout is not None
                and getattr(self.grid, "land_frac", None) is not None
                and getattr(self._voronoi_layout.local_mesh, "land_frac",
                            "no-field") is None):
            self._voronoi_layout = self._voronoi_layout._replace(
                local_mesh=self._voronoi_layout.local_mesh._replace(
                    land_frac=self.grid.land_frac))

        # Per-column subgrid orographic stddev for the orographic GWD launch
        # (tau_0 ∝ h_topo²). Attached to the grid pytree so the physics
        # integration's ``_extract_subgrid_topo_stddev`` finds it; without it
        # McFarlane/Lindzen fall back to the scalar ``config.h_topo`` — a
        # uniform 500 m mountain over ocean columns too. Only loaded when the
        # active GWD has an orographic member; otherwise the file is unused.
        # ``load_subgrid_orography``'s regrid target handles cube (rank-3),
        # structured lat-lon (rank-2) AND unstructured Voronoi (rank-1 cell
        # centres) grids — see ``_target_grid_degrees``.
        sso_path = getattr(self.config, "subgrid_orography_path", "")
        if sso_path:
            _oro_members = ("mcfarlane", "lindzen", "e3sm_cam")
            _gwd = str(getattr(self.config, "gravity_wave_drag", "none"))
            if not any(p in _oro_members for p in _gwd.split("+")):
                logger.warning(
                    "  subgrid_orography_path=%s set but gravity_wave_drag=%r "
                    "has no orographic member (%s) — file NOT loaded",
                    sso_path, _gwd, "/".join(_oro_members),
                )
            else:
                from legoesm.grids.topography import load_subgrid_orography
                sso = load_subgrid_orography(self.grid, sso_path).astype(_sd)
                try:
                    self.grid = self.grid._replace(subgrid_topo_stddev=sso)
                # NamedTuple._replace raises TypeError ("Got unexpected field
                # names"), NOT ValueError/AttributeError -- so this guard never
                # fired and a raw collections traceback escaped instead of the
                # message below. Found by a lat-lon run dying at setup.
                except (TypeError, ValueError, AttributeError) as e:
                    raise ValueError(
                        f"subgrid_orography_path is set but grid type "
                        f"{type(self.grid).__name__} has no subgrid_topo_stddev "
                        f"field (supported: CubedSphereGrid, GaussianGrid, "
                        f"VoronoiMesh)"
                    ) from e
                # The compiled MPI step closes over the layout's local mesh
                # (built before this attach), not ``self.grid`` -- same
                # hazard as the land_frac refresh above.  Without this the
                # distributed lane silently launched the scalar h_topo
                # fallback (a 500 m mountain over every ocean column) while
                # the serial lane read the file.
                if self._voronoi_layout is not None:
                    _lm = self._voronoi_layout.local_mesh
                    if sso.shape[0] != int(_lm.nCells):
                        raise ValueError(
                            f"subgrid orography has {sso.shape[0]} columns "
                            f"but the rank-local mesh has {int(_lm.nCells)}")
                    self._voronoi_layout = self._voronoi_layout._replace(
                        local_mesh=_lm._replace(subgrid_topo_stddev=sso))
                logger.info(
                    f"  Subgrid orography: {sso_path} "
                    f"(stddev max={float(jnp.max(sso)):.0f} m, "
                    f"mean={float(jnp.mean(sso)):.1f} m)"
                )
        else:
            # #1514 guard: an orographic GWD member with NO SSO file on a
            # run that has a real land/ocean distribution launches from the
            # scalar h_topo fallback — a fictional 500-m mountain over every
            # ocean column (measured: -0.29 Pa spurious Southern-Ocean drag;
            # it erased the eddy-driven westerlies and both storm tracks).
            # Loud, not fatal: idealized configs keep the documented legacy
            # fallback deliberately.
            from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
                orographic_scalar_fallback_warning,
            )
            _msg = orographic_scalar_fallback_warning(
                str(getattr(self.config, "gravity_wave_drag", "none")),
                sso_path,
                self._f_land is not None,
            )
            if _msg is not None:
                logger.warning("  %s", _msg)

        self._assert_vertical_coordinate_supports_this_orography()

    def _assert_vertical_coordinate_supports_this_orography(self) -> None:
        """Refuse a hybrid coordinate that inverts over this run's terrain.

        The check belongs here because this is the first point where BOTH the
        coordinate and the orography exist.  The surface pressure it tests is
        the hydrostatic reduction the driver itself uses to seed p_s over
        terrain (``p_s = p_ref * exp(-phis / (R_d * T_init))``, see the
        initial-state branches below) -- not an invented estimate, and not the
        seed field, which the MPAS lane deliberately leaves flat.

        Why it is fatal rather than a warning: see
        ``assert_hybrid_valid_for_surface_pressure``.  Short version, measured:
        the default L40 coordinate forbids orography above about 3450 m, which
        is 0.92% of the planet by area, and two cells inside that regime killed
        a 200-day idealized run in 200 steps (#1029).
        """
        gc = self.config.grid
        if getattr(gc, "vertical_coord", None) != "hybrid":
            return
        phis = getattr(self, "_phis_data", None)
        if phis is None:
            return
        phis_max = float(jnp.max(phis))
        if not (phis_max > 0.0):
            return          # flat: every column sits at p_ref

        from legoesm.grids.vertical import (
            assert_hybrid_valid_for_surface_pressure)
        from legoesm import constants

        T_init = float(getattr(self.config, "T_init", 288.0))
        p_s_min = float(constants.p_ref
                        * jnp.exp(-phis_max / (constants.R_d * T_init)))
        assert_hybrid_valid_for_surface_pressure(
            self.sigma, p_s_min,
            context=(f"{gc.grid_type} {gc.resolution}, {gc.nlev} hybrid levels"
                     f", topography={self.config.topography}"),
        )

    def _coeff_grid(self):
        """Grid that global scalars (diffusion coefficients, mean cell size)
        are derived from: the GLOBAL mesh under a partition, else the grid."""
        g = getattr(self, "_grid_global", None)
        return self.grid if g is None else g

    def _create_dycore(self) -> None:
        """Create the dynamical core model via the component factory.

        The factory resolves ``(model_type, discretization, grid_type)``
        from :attr:`config` and instantiates the correct solver with
        physically derived diffusion coefficients.
        """
        from legoesm.driver.component_factory import (
            create_atmosphere_dycore, compute_diffusion,
        )

        # Coefficients from the GLOBAL mesh under a cell partition: the local
        # mesh's min(dcEdge)/min(areaCell) differ per rank (measured +0.09 %
        # nu_del2 on rank 0 of a 2-rank res-3 split), so each rank would run
        # a different viscosity.
        coeff_grid = self._coeff_grid()
        self.model = create_atmosphere_dycore(
            self.config, self.grid, self.sigma, coeff_grid=coeff_grid)

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
        diff = compute_diffusion(coeff_grid, self.config.dycore)
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
                f"for {gc.grid_type} C{gc.resolution} — the GENERIC "
                f"advective+acoustic heuristic (assumed 50+340 m/s, "
                f"cfl_check_and_adjust), NOT a scheme-certified stability "
                f"envelope (in particular not the fv3_duo deck's "
                f"k_split/n_split envelope)"
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
                # Also forward the split-SIC file and variable-name overrides:
                # a preset names the dataset's canonical variables, but a user
                # staging e.g. HadISST SST alongside a separate SIC file (or a
                # renamed variable) still needs --sic-path/--*-var to reach the
                # loader — the bare ``_replace`` used to silently drop them and
                # read SIC from the SST file (audit 2026-07-17). Empty string
                # means "not set" (run_amip maps absent CLI flags to ""), which
                # keeps the preset's own value.
                preset = get_amip_preset(cfg.dataset)
                forcing_config = preset._replace(
                    path=cfg.forcing_path, T_ice=cfg.T_ice,
                    sst_offset=cfg.sst_offset, sic_scale=cfg.sic_scale,
                    sic_path=getattr(cfg, "sic_path", "") or preset.sic_path,
                    sst_var=cfg.sst_var or preset.sst_var,
                    sic_var=cfg.sic_var or preset.sic_var,
                    time_var=cfg.time_var or preset.time_var,
                    lat_var=cfg.lat_var or preset.lat_var,
                    lon_var=cfg.lon_var or preset.lon_var,
                )

            # Anchor the SST/SIC time axis to the run's start year so a model
            # day indexes the file by real calendar date (AMIP-II): a 1979 run
            # reads the 1979 records of a 1870-2022 input4MIPs file, not 1870.
            forcing = load_amip_forcing(
                forcing_config, forcing_grid,
                start_year=getattr(cfg, "start_year", None),
                run_days=getattr(cfg, "days", None),
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
            # REVERT of the 2026-07-17 audit change (#1179): passing phis into
            # held_suarez_init_mpas hydrostatically reduces the seed p_s over
            # topography (p_s = p_ref*exp(-phis/(R_d*T_init))) — and EVERY
            # moist MPAS AMIP run then NaNs within day 1 (ERA5 and uniform IC;
            # dt 100-240 s, L20/L40, sbm/bechtold, fp32/fp64; hunk-level
            # bisection conviction 2026-07-23, see
            # docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md).  A
            # control probe (phis field kept, p_s left FLAT) is stable, so the
            # reduction itself is the poison — the seed leaks into the run
            # through a channel the ERA5 overlay does not replace.  Until that
            # is root-caused, keep the empirically stable pre-audit semantics:
            # flat seed p_s + phis patched afterwards (a brief, bounded
            # startup pressure shock over orography for ic='default').
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
            # mismatched pressure grid -- the same ``pressure_at_full`` every
            # physics bridge and the driver now use (sigma-pressure ratchet:
            # tests/test_no_sigma_pressure_reimpl.py).
            p_full_init = self.sigma.pressure_at_full(self.state.p_s.data)
            q_sat_init = saturation_mixing_ratio(self.state.T.data, p_full_init)
            self.tracers["q_v"] = cfg.rh_init * q_sat_init * self.sigma.sigma_full ** 2
            self.tracers["q_v"] = jnp.minimum(self.tracers["q_v"], q_sat_init)

            # Fuse the two diagnostic means into one host transfer.
            _stats = jnp.stack([
                jnp.mean(self.tracers["q_v"]),
                jnp.mean(column_water_vapor(
                    self.tracers["q_v"], self.state.p_s.data,
                    self.sigma.dsigma,
                    dp=self.sigma.layer_thickness_dp(self.state.p_s.data),
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
                    smoothing_passes=cfg.topo_smoothing,
                    edge_blend_strength=cfg.topo_edge_blend,
                )
            elif cfg.dycore.discretization == "spectral":
                carry = era5_to_spectral_carry(
                    era5_slice, self.grid, self.sigma,
                    smoothing_passes=cfg.topo_smoothing,
                )
            elif cfg.grid.grid_type == "latlon":
                carry = era5_to_latlon_carry(
                    era5_slice, self.grid, self.sigma,
                    smoothing_passes=cfg.topo_smoothing,
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
                # ERA5 carry: (u, v) -> (vor_hat, div_hat) via the EXACT
                # left-inverse of the dycore's ``uv_from_vordiv_3d`` synthesis
                # (``vordiv_from_uv_exact_3d``, #976).  The plain Bourke
                # ``vordiv_from_uv_3d`` seeds a pole-row artifact at n=n_max
                # into the IC; the exact inverse keeps the winds clean so the
                # spectral IC matches the ERA5 winds, pole rows included.
                # T/ln(p_s)/phis via ``sh_analysis``.
                from legoesm.grids.gaussian import (
                    vordiv_from_uv_exact_3d, sh_analysis_3d, sh_analysis,
                )
                vor_hat, div_hat = vordiv_from_uv_exact_3d(
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
                # Re-attach the ERA5 q_v into the DYCORE-ADVECTED tracer
                # state, mirroring the spectral branch above.  Without this
                # the MPAS run advected the moist-init rest-state RH taper
                # (q_v ~ q_sat(seed T, seed p_s)·sigma²) while only the dead
                # driver-level dict got ERA5 moisture — the codex-confirmed
                # root cause (F1, 2026-07-23) of the "seed p_s leaks into
                # ERA5 runs" mystery: seed q_v scales with seed q_sat(p_s),
                # so the seed-p_s choice modulated the latent FUEL at the
                # moist-runaway cells.  See
                # docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md.
                if (self.state.tracers is not None
                        and "q_v" in self.state.tracers):
                    self.state = self.state._replace(tracers={
                        **self.state.tracers,
                        "q_v": self.state.tracers["q_v"].replace(
                            data=jnp.asarray(carry.q_v)),
                    })
                _stats_era5 = jnp.stack([
                    jnp.mean(self.tracers["q_v"]),
                    jnp.mean(column_water_vapor(
                        self.tracers["q_v"], self.state.p_s.data,
                        self.sigma.dsigma,
                        dp=self.sigma.layer_thickness_dp(self.state.p_s.data),
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
                        dp=self.sigma.layer_thickness_dp(self.state.p_s.data),
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
        # Thread the per-column subgrid orography into the compiled physics
        # pipeline. The grid attachment (``_create_topography``) feeds the
        # make_gwd_physics/spectral factories, which read the grid per call;
        # the pipeline's column hot path reads this attribute instead — both
        # are views of the same field and are re-scattered together under MPI.
        _sso = getattr(self.grid, "subgrid_topo_stddev", None)
        if _sso is not None:
            self.physics.subgrid_topo_stddev = _sso

        _has_land = (
            self._f_land is not None
            and bool(jnp.any(self._f_land > 0))
        )
        self._has_land_anywhere = _has_land
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
            elif getattr(self.config, "clm_surfdata_path", ""):
                # ERA5-TUNED per-column land albedo (the LMIP calibration):
                # PFT-weighted _TUNED_PFT_ALBEDO_MULTILAYER + soil-colour blend
                # + glacier override (0.7178), the exact product run_lmip.py and
                # the coupled driver already use.  Before this branch an AMIP
                # run passing only --clm-surfdata-path fell through to the
                # LATITUDE-vegetation fallback, so the calibrated map never
                # reached radiation on the MPAS lane.
                from legoesm.land.clm_surface_map import clm_surface_provider
                _lat_deg = np.degrees(np.asarray(self.grid.grid_lat)).reshape(-1)
                _lon_deg = np.degrees(np.asarray(self.grid.grid_lon)).reshape(-1)
                _lp = clm_surface_provider(
                    _lat_deg, _lon_deg,
                    surfdata_path=self.config.clm_surfdata_path,
                    variant="multilayer")()
                _alb = jnp.asarray(_lp.albedo_veg).reshape(
                    jnp.asarray(self.grid.grid_lat).shape)
                # Defensive only: the provider floors/normalises PFT cover, so
                # cells WITHOUT source land data come back as finite bare-soil
                # values, NOT NaN — this where() does not gate them (codex).
                # Ocean cells are irrelevant (radiation blends by f_land);
                # coastal model-land cells nearest to an ocean source cell get
                # the bare-soil template, an accepted nearest-neighbour limit.
                _alb = jnp.where(jnp.isfinite(_alb), _alb, lat_albedo)
                self.physics.albedo_land = _alb.astype(_sd)
                logger.info(
                    "  Land albedo: ERA5-tuned CLM multilayer map "
                    f"(mean={float(jnp.mean(_alb)):.3f}, "
                    f"max={float(jnp.max(_alb)):.3f})")
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
                if (_bucket or _stomatal) and not self.physics.surface_tiled:
                    # Non-tiled surface flux runs ONE bulk/turbulence scheme on
                    # the blended T_sfc with a WET q_sat (no beta), while the
                    # slab-land SEB depletes its bucket with the beta-limited
                    # flux: the moisture the atmosphere gains over land is NOT
                    # the water the bucket loses, so the land water budget does
                    # not close (audit 2026-07-17). The tiled path applies beta
                    # on both sides consistently.
                    logger.warning(
                        "  Land beta-limited evaporation "
                        "(--land-soil-bucket/--land-stomatal-beta) without "
                        "--surface-tiled: the atmosphere sees the WET blended-"
                        "surface latent flux, not the beta-limited land flux — "
                        "land water budget will not close. Pass "
                        "--surface-tiled (with --turbulence louis/clubb_lite/"
                        "clubb) for a consistent land-atm moisture budget."
                    )
                logger.info(
                    f"  Land tile: ACTIVE (slab land, C_land="
                    f"{self.physics.C_land:.1e} J/m2/K, "
                    f"f_land mean={float(jnp.mean(self._f_land)):.3f}, "
                    f"interface_flux={self.physics.land_interface_flux}, "
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

        # Runtime fail-fast (codex R3): land_interface_flux='unified' exists
        # to fix an energy-conservation defect — if the slab tile did NOT
        # actually activate (e.g. an all-zero land-mask file, or a topography
        # that derived no land, both of which pass validate_strict), the flag
        # would silently never apply.  Refuse instead of running a config the
        # user believes is conservative.  Under MPI the activation is
        # rank-local (an ocean-only rank legitimately has no land while a
        # neighbour does — codex R4), so the guard tests the GLOBAL
        # any-rank activation; every rank reaches this collective (the
        # guard is unconditional in _create_physics).
        if (getattr(self.config, "land_interface_flux",
                    "legacy_dual") == "unified"):
            _slab_on = bool(self.physics.slab_land_active)
            from legoesm.grids.halo import get_mpi_topology
            if get_mpi_topology() is not None:
                from legoesm.parallel.reductions import global_max_mpi
                _slab_on = bool(
                    float(global_max_mpi(
                        jnp.asarray(1.0 if _slab_on else 0.0))) > 0.0)
            if not _slab_on:
                raise ValueError(
                    "land_interface_flux='unified' was requested but the "
                    "slab land tile did not activate on any rank (no land "
                    "in the mask/topography, or no activation flag) — the "
                    "unified interface law would silently never apply. "
                    "Check --land-mask-file / --topography / "
                    "--slab-land-active."
                )

        # The multilayer tile's flux handoff to the atmosphere needs a soil
        # column to exist.  A land-mask FILE that is all ocean passes
        # validate_strict (which cannot read the file) and then builds nothing,
        # leaving the handoff — and any calibrated canopy conductance riding it —
        # silently inert.  The flat/no-mask spelling of the same trap IS caught
        # in validate_strict.
        #
        # DELIBERATELY RANK-LOCAL, AND ONLY FATAL WHEN SERIAL.  Under
        # cell-partition MPI an ocean-only rank legitimately has no land, so the
        # honest answer needs a cross-rank vote — and a collective here would sit
        # downstream of file loads and pipeline construction that can raise on
        # one rank and not another, leaving its peers blocked in the reduction
        # forever.  Five review rounds went into trying to place such a vote
        # safely before concluding it does not belong here at all: a guard is not
        # worth a deadlock.  Serial runs (where configs are written and tested)
        # get the hard error; distributed runs get a warning naming exactly what
        # could not be checked.
        if bool(getattr(self.config, "mpas_land_beta_soil", False)) \
                and not self._has_land_anywhere:
            _msg = (
                "mpas_land_beta_soil=True hands the multilayer land tile's "
                "solved humidity and fluxes to the atmosphere, but no soil "
                "column was built (no land in the mask/topography) — the "
                "handoff, and any calibrated canopy conductance riding it, "
                "would silently never apply. Check --land-mask-file / "
                "--topography / --use-multilayer-land."
            )
            # "Serial" means NO PEERS, not "no partition object": a one-rank
            # MPI launch still builds a layout, and that run has no ocean-only
            # neighbour to protect, so it should get the hard error too (codex).
            if int(getattr(self, "_mpi_world_size", 1) or 1) <= 1:
                raise ValueError(_msg)
            # Under MPI, SAY NOTHING.  A rank owning no land is the ordinary
            # case on a healthy run — warning here would fire on every
            # ocean-only rank of every correct run, and a warning that fires
            # when nothing is wrong trains everyone to ignore it (GLM review).
            # The config-level check catches the flat/no-mask spelling; counting
            # land points in the MASK FILE at config time would close the rest
            # without any collective, and is the follow-up.
            logger.debug(
                "rank owns no land; land-flux handoff inactive on this rank "
                "(normal for an ocean-only partition)")

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
        elif _scheme_name == "clm_ml":
            # CLM-ML-JAX multilayer canopy.  The ncol>1 traceable path (S2) makes
            # the coupled land tile steppable over all model columns inside the
            # jitted segment: the canopy is warm-started once (eager) at setup and
            # a concrete per-column ``grid_info`` is threaded into the jitted step
            # (see the warm-start block at the end of this method + the pipeline's
            # clm_ml_grid_info).  NOTE the two current limitations: (a) the land
            # tile's cos_zenith is the same 0.5 placeholder the two_leaf canopy
            # uses here (faithful diurnal zenith is a shared follow-up), and (b)
            # the per-column Python loop unrolls O(ncol) in the trace, so this is
            # for MODEST column counts, not full-AMIP resolution (the masked-vmap
            # rewrite, S3, removes the unroll).  MPI/SPMD is refused at the scatter
            # (the 1-based mlcanopy array does not match the per-column remap).
            from legoesm.land.canopy.config import CLMMLCanopyConfig
            _surface_scheme = CLMMLCanopyConfig(
                use_surfdata_pft=bool(getattr(
                    self.config, "clm_ml_use_surfdata_pft", False)))
        else:
            raise ValueError(
                f"Unknown land_surface_scheme {_scheme_name!r}; "
                "expected 'simple_seb', 'two_leaf', or 'clm_ml'."
            )

        # Unstructured (Voronoi/MPAS) meshes carry per-cell 1-D coordinates and
        # per-cell (nCells, ...) state — already the flattened (ncol,) column
        # order the land tile wants; the structured paths go through the
        # pipeline adapter.  ``_flat_cols`` is the one flattener both branches
        # share so every consumer below stays layout-agnostic.
        _unstructured = hasattr(self.grid, "latCell")
        if _unstructured:
            ad = None
            lat_rad = _np.asarray(self.grid.latCell).reshape(-1)
            lon_rad = _np.asarray(self.grid.lonCell).reshape(-1)

            def _flat_cols(x):
                return jnp.asarray(getattr(x, "data", x)).reshape(-1)
        else:
            ad = self.physics.adapter

            def _flat_cols(x):
                return jnp.asarray(
                    ad.flatten_2d(getattr(x, "data", x))).reshape(-1)
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
        # Deploy the tile in EXACTLY the model its parameters were calibrated
        # under: the biophysics LMIP two-leaf canopy
        # (``templates/land/biophysics/lmip_biophys_2deg.yaml``), which is the
        # offline configuration scored against observed land fluxes.
        #
        # AMIP no longer has a route to the legacy SimpleSEB-fitted bake. That
        # pairing — SimpleSEB tables under a coupled run — is what produced the
        # land surface-flux bias, and keeping a switch for it only preserved the
        # chance of selecting it again. ``calibrated_multilayer_setup`` remains
        # in the land package for reproducing old OFFLINE runs; nothing coupled
        # reaches it.
        if getattr(self.config, "land_calibrated_physics", False):
            from legoesm.land.config import apply_biophysics_lmip_two_leaf
            base = apply_biophysics_lmip_two_leaf(base)

        params, cfg = clm_multilayer_setup(surface_map, base_config=base)

        # RE-APPLY THE CALIBRATION'S ALBEDO SCALARS, AFTER the bake.
        # ``clm_multilayer_setup`` rebuilds ``land_albedo`` and overwrites all
        # five scalar fields with its own values, so applying them before it
        # loses them without a word: a run asking for a fresh-snow albedo of
        # 0.8077 was measured getting 0.8362. The bake's per-column
        # ``snow_cover_scale`` map is built here and must survive, so replace
        # the SCALARS on the object the bake just produced.
        if getattr(self.config, "land_calibrated_physics", False):
            from legoesm.land.clm_surface_map import biophysics_lmip_albedo_scalars
            cfg = cfg._replace(
                land_albedo=cfg.land_albedo._replace(
                    **biophysics_lmip_albedo_scalars()))

        # Temperature-dependent snow ageing (opt-in): applied LAST, so it
        # survives both the bake and the calibration re-apply above. The
        # calibration fitted only the calendar-clock scalars and never saw this
        # field, so it cannot be overwritten by it.
        _act = getattr(self.config, "snow_age_activation_K", None)
        if _act is not None:
            cfg = cfg._replace(
                land_albedo=cfg.land_albedo._replace(
                    snow_age_activation_K=float(_act)))
        # Snow-age e-folding time, same placement and for the same reason: the
        # calibration re-apply above sets tau_snow_decay, so an override has to
        # land after it or it is silently discarded.
        _tau_d = getattr(self.config, "land_snow_tau_days", None)
        if _tau_d is not None:
            cfg = cfg._replace(
                land_albedo=cfg.land_albedo._replace(
                    tau_snow_decay=float(_tau_d) * 86400.0))
            logger.info("  land snow-albedo age e-folding: %.3g days "
                        "(overrides the calibration)", float(_tau_d))
        # Soil freeze/thaw, same placement: the bake rebuilds ``thermal`` with
        # the switch at its library default, so it must be set after it.
        _ft = bool(self.config.land_soil_freeze_thaw)
        cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=_ft))
        logger.info("  land soil freeze/thaw: %s", "ON" if _ft else "off")
        if _ft and getattr(self.config, "land_calibrated_physics", False):
            logger.warning("  land soil freeze/thaw ON with the calibrated land "
                           "tables, which were fitted with it OFF")

        # A CANOPY SCHEME GETS CANOPY PARAMETERS.
        #
        # ``clm_multilayer_setup`` returns ``LandSurfaceParams`` — per-PFT
        # roughness, albedo and emissivity for a bulk surface.  A canopy scheme
        # reads different quantities (canopy height, clumping, band albedos,
        # roughness RATIO, per-PFT Vcmax25) and, given soil parameters, silently
        # falls back to generic constants, leaving the tuned values inert.
        #
        # Those canopy parameters already exist and are already per-PFT:
        # ``surface_data_to_land_params`` dispatches on the surface scheme and
        # builds ``CanopyLandParams`` from the harmonized surfdata — the same
        # route the offline LMIP simulations use, which is why THEY reproduce
        # observed latent heat and photosynthesis.  The coupled driver simply
        # never called it; it read the CLM provider, whose only variants are
        # "slab" and "multilayer".  This is that missing call, not new physics.
        #
        # Soil hydraulics / thermal / albedo stay with the CLM map: only the
        # SURFACE parameters come from the canopy builder.
        from legoesm.land.canopy import CanopyConfig
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        # Both canopy schemes take their per-PFT SURFACE params from
        # init_land_surface_data: a two-leaf CanopyConfig -> CanopyLandParams
        # (hc/rz0m); the CLM-ML canopy -> LandSurfaceParams with PRESCRIBED
        # LAI/SAI/htop (the accurate per-plant structure its interface requires,
        # vs the LAI-only clm_multilayer_setup fallback that left htop at the 5 m
        # scalar default).  Explicit positive dispatch on the two canopy types
        # (not `not SimpleSEBConfig`) so an unknown scheme cannot slip in.  #1624
        # read params.hc here unconditionally, which crashed CLM-ML (no hc); the
        # reads below dispatch on the actual struct.
        if isinstance(cfg.surface_scheme, (CanopyConfig, CLMMLCanopyConfig)):
            _sd_path = getattr(self.config, "surfdata_path", "")
            if not _sd_path:
                raise ValueError(
                    f"land_surface_scheme selects {type(cfg.surface_scheme).__name__}, "
                    "which needs per-PFT CANOPY parameters (canopy height, band "
                    "albedos, roughness ratio, Vcmax25). Those are built from the "
                    "harmonized surfdata, so --surfdata is required. Without it the "
                    "canopy would run on generic constants and every tuned per-PFT "
                    "value would be inert — refusing rather than doing that quietly.")
            from legoesm.land.boundary_data import init_land_surface_data
            _clm_params = params
            # THE CALIBRATION'S OWN TABLES, or none. Without these the builder
            # leaves the root fields unset and the code below fills them from
            # the CLM map — i.e. the LEGACY tables — so the calibration's root
            # depth, wilting point and field capacity never reached the model.
            # Same for the glacier albedo, which otherwise takes the 0.70/0.50
            # default instead of the calibrated pair.
            _cal_on = getattr(self.config, "land_calibrated_physics", False)
            _pft_root, _glacier_alb = None, None
            if _cal_on:
                from legoesm.land.clm_surface_map import (
                    biophysics_lmip_pft_root_params,
                    biophysics_lmip_glacier_albedo)
                _pft_root = biophysics_lmip_pft_root_params()
                _glacier_alb = biophysics_lmip_glacier_albedo()
            _, params, _ = init_land_surface_data(
                _sd_path, self.grid, cfg, float(self.config.start_day),
                year=(None if getattr(self.config, "start_year", None) is None
                      else float(self.config.start_year)),
                pft_root_params=_pft_root, glacier_alb=_glacier_alb)
            # KEEP THE SOIL-WATER THRESHOLDS THE COMMENT ABOVE PROMISES.
            # Replacing the parameter object wholesale also dropped the CLM
            # per-column ROOT DEPTH, WILTING POINT and FIELD CAPACITY, which the
            # canopy parameter object leaves unset — so the land step silently
            # fell back to one scalar value per field for the whole globe, and
            # every column's root-zone moisture stress changed. Those are soil
            # properties, not surface ones; carry them across (found by both
            # reviewers).
            _root_fields = {
                f: getattr(_clm_params, f)
                for f in ("root_depth", "theta_wp", "theta_fc")
                if getattr(params, f, None) is None
                and getattr(_clm_params, f, None) is not None}
            if _root_fields:
                params = params._replace(**_root_fields)
                logger.info(
                    "  Land tile: kept the CLM per-column %s with the canopy "
                    "parameters (they are soil properties, not surface ones).",
                    ", ".join(sorted(_root_fields)))
            # TWO PROVIDERS, ONE COLUMN INDEX.  The canopy parameters come from
            # one dataset and the soil beneath them from another, each regridded
            # independently.  A differing column COUNT or ORDER would attach every
            # canopy property to the wrong column and the run would still look
            # entirely healthy — so check the count structurally, at t=0, rather
            # than hope (GLM).
            # Struct differs by scheme: two-leaf -> CanopyLandParams (hc, rz0m);
            # CLM-ML -> LandSurfaceParams (htop, LAI, NO hc).  Read the column
            # count off whichever per-column height field the provider set, and
            # log the fields that exist for that struct.
            _hc = getattr(params, "hc", None)
            _height_field = _hc if _hc is not None else params.htop
            _n_canopy = int(jnp.asarray(_height_field).shape[0])
            _n_soil = int(jnp.asarray(lat_rad).size)
            if _n_canopy != _n_soil:
                raise ValueError(
                    f"canopy parameters have {_n_canopy} columns but the soil map "
                    f"has {_n_soil}: the two surface datasets did not regrid onto "
                    "the same columns, so every canopy property would sit on the "
                    "wrong one. Check --surfdata and --clm-surfdata-path cover "
                    "this grid.")
            if _hc is not None:
                logger.info(
                    "  Land tile: %s on per-PFT CANOPY parameters from %s "
                    "(canopy height %.2f-%.2f m, roughness ratio %.3f-%.3f). NOTE "
                    "emissivity is a single constant in this builder and the canopy "
                    "computes its own radiation from soil colour + leaf optics, so "
                    "the tuned per-PFT emissivity and vegetation albedo do NOT apply "
                    "to it — those are re-fit items, not wiring.",
                    type(cfg.surface_scheme).__name__, _sd_path,
                    float(jnp.min(_hc)), float(jnp.max(_hc)),
                    float(jnp.min(params.rz0m)), float(jnp.max(params.rz0m)))
            else:
                logger.info(
                    "  Land tile: %s on per-PFT CLM-ML surface parameters from %s "
                    "(prescribed canopy-top height %.2f-%.2f m, LAI %.2f-%.2f) — "
                    "the accurate per-plant structure the multilayer canopy needs.",
                    type(cfg.surface_scheme).__name__, _sd_path,
                    float(jnp.min(params.htop)), float(jnp.max(params.htop)),
                    float(jnp.min(params.LAI)), float(jnp.max(params.LAI)))

        # A CANOPY SCHEME MUST NOT SILENTLY RUN ON SOIL PARAMETERS.
        #
        # ``clm_multilayer_setup`` returns ``LandSurfaceParams`` — per-PFT
        # roughness, albedo and emissivity.  The two-leaf and CLM-ML canopies
        # read CANOPY properties (canopy height, band albedos, roughness ratio)
        # and, finding none, fall back to generic constants: the tuned per-PFT
        # values are then INERT, which a gradient test caught as exactly zero
        # sensitivity to the trained roughness.  Inert trained parameters are a
        # defect here, not a nuisance, and a comment in a document is not a
        # control (GLM).  Say it at startup, every run, so nobody reports a
        # canopy run as validating tuned land parameters it never used.
        from legoesm.land.canopy import CanopyConfig
        # Two-leaf only: it reads CanopyLandParams (hc/ALB/rz0m).  CLM-ML runs on
        # a LandSurfaceParams by design (its canopy structure is LAI/SAI/htop),
        # so this "running on SOIL parameters" check does not apply to it.
        if isinstance(cfg.surface_scheme, CanopyConfig):
            _canopy_fields = [f for f in ("hc", "ALB_VIS", "ALB_NIR", "rz0m")
                              if getattr(params, f, None) is not None]
            if not _canopy_fields:
                logger.warning(
                    "  Land tile: %s is running on SOIL parameters — it reads "
                    "canopy height, band albedos and roughness ratio, none of "
                    "which are present, so it is using GENERIC canopy constants "
                    "and the tuned per-PFT roughness / albedo / emissivity are "
                    "INERT. Do not report this run as validating tuned land "
                    "parameters. Use --land-surface-scheme simple_seb to run the "
                    "scheme those parameters belong to.",
                    type(cfg.surface_scheme).__name__)

        self.physics.land_ml_cfg = cfg
        self.physics.land_ml_params = params
        # The Farquhar branch needs a non-None carbon state at the call site as
        # well as the differland scheme.  Seed a PRESCRIBED one (fixed leaf
        # carbon -> fixed LAI = C_fol/LCMA); the pipeline discards the evolved
        # pools every step, which is exactly what the calibrator does, so the
        # baked Vc_max25/g1/LCMA act on the same LAI they were fitted with and no
        # multi-decade carbon spin-up is needed.  Left None otherwise, so a run
        # without the flag is byte-identical.
        # A canopy scheme carries its OWN photosynthesis (Farquhar GPP), so it
        # also needs the differland carbon state seeded even though the bulk
        # ``cfg.stomata`` throttle is off — otherwise the calibrated two-leaf
        # run, fitted WITH differland, would deploy under a "none" carbon cycle.
        from legoesm.land.surface_scheme import SimpleSEBConfig as _SimpleSEBConfig
        _is_canopy = not isinstance(cfg.surface_scheme, _SimpleSEBConfig)
        if (cfg.stomata.enabled or _is_canopy) and cfg.carbon.scheme == "differland":
            from legoesm.land.carbon.carbon_cycle import init_carbon_state
            self.physics.land_ml_carbon = init_carbon_state(
                (int(lat_rad.size),), cfg.carbon)
            logger.info(
                "  Land stomata: FARQUHAR (prescribed carbon state, "
                f"LAI = C_fol/LCMA from C_fol={cfg.carbon.C_fol_init:g} gC/m2) "
                "— the baked Vc_max25/g1/LCMA are ACTIVE")
        elif cfg.stomata.enabled:
            logger.info(
                "  Land stomata: JARVIS (no differland carbon state) — the baked "
                "Farquhar Vc_max25/g1/LCMA are NOT used")
        self.physics.land_ml_lat = jnp.asarray(lat_rad, dtype=storage_dtype)
        self.physics.land_ml_doy = 0.0
        # CONCRETE dynamics timestep [s].  The jitted segment passes ``dt`` as a
        # TRACER, but the CLM-ML canopy needs a concrete dt to resolve its static
        # ML sub-step count (num_ml_steps = ceil(dt/dtime_ml)).  The timestep is
        # fixed, so the concrete config dt is numerically identical to the traced
        # one; the pipeline uses it only on the clm_ml path (byte-identical for
        # simple_seb / two_leaf, which keep the traced dt).
        self.physics.land_ml_dt = float(self.config.dycore.dt)

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
        T_init = _flat_cols(self.state.T.data[..., -1]).astype(storage_dtype)
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
        _soil_init = getattr(self.config, "land_soil_init", "aridity")
        if _soil_init == "saturation_fraction":
            # Seed as a fraction of POROSITY, so the soil can start above field
            # capacity — the way to keep a run out of the dry-soil attractor
            # (low soil water -> weak evaporation -> dry boundary layer -> weaker
            # evaporation).  The aridity seed below cannot do this: it caps at
            # field capacity by construction.
            #
            # Held just below saturation because the van-Genuchten retention is
            # singular AT saturation — psi_from_theta needs theta < theta_sat —
            # and just above the residual for the same reason at the dry end.
            # Same guard the offline calibrator uses on its own seed.
            _th_sat = jnp.asarray(cfg.hydraulics.theta_sat)
            _th_res = jnp.asarray(cfg.hydraulics.theta_r)
            theta_init = jnp.clip(
                self.config.land_soil_moisture_init_frac * _th_sat,
                _th_res + _THETA_EDGE_GUARD, _th_sat - _THETA_EDGE_GUARD)
            logger.info(
                "  Land tile: soil seeded at %.2f x porosity (saturation_fraction) "
                "— NOT the aridity map; a wet start trades the desert runaway the "
                "aridity seed prevents for staying out of the dry-soil attractor.",
                self.config.land_soil_moisture_init_frac)
        elif _qv is not None:
            q_v_low = _flat_cols(
                getattr(_qv, "data", _qv)[..., -1]).astype(storage_dtype)
            p_s = _flat_cols(self.state.p_s).astype(storage_dtype)
            rh_low = q_v_low / jnp.maximum(
                saturation_mixing_ratio(T_init, p_s), 1e-12)
            # ``getattr(x, k, default)`` returns the ATTRIBUTE when it exists and
            # is None, which is not what is wanted here: the canopy parameter
            # struct declares these fields and leaves them unset unless per-PFT
            # root parameters were supplied, so the default has to cover None as
            # well as absent.
            def _or_cfg(name, fallback):
                v = getattr(params, name, None)
                return jnp.asarray(fallback if v is None else v)

            theta_wp = _or_cfg("theta_wp", cfg.theta_wp)
            theta_fc = _or_cfg("theta_fc", cfg.theta_fc)
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
            from legoesm.land.soil_grid import make_soil_grid
            # Layer COUNT alone does not identify a soil column: 8 layers over
            # 3 m at growth 1.5 and 8 over 6.375 m at growth 2 both pass the
            # shape check while placing every soil value at a different depth.
            # Pass the thicknesses so a spin-up on the wrong column is refused.
            # Under MPAS cell-partition MPI the spin-up file holds the GLOBAL
            # columns while ``ncol`` is already this rank's band, so the file is
            # checked against the GLOBAL count and then cut down -- the mirror
            # of what a gathered checkpoint restore does, through the same
            # helper.  Checking it against the local count instead made the
            # loader raise with a different number on every rank, which is what
            # forced any spun-up-land arm back to a single device.
            _vl_ic = getattr(self, "_voronoi_layout", None)
            _part_ic = _vl_ic.partition if _vl_ic is not None else None
            _expect_ncol = (int(_part_ic.nCells_global)
                            if _part_ic is not None else ncol)
            _ic_state, _ic_meta = load_land_restart(
                _land_ic_path, expected_land_mode="multilayer",
                expected_ncol=_expect_ncol,
                expected_n_layers=cfg.soil_grid.n_layers,
                # Same quantity this driver's PRE-LOAD check already carries
                # (``load_land_restart_soil_dz`` returns thicknesses), so the
                # column travels one form through both checks.
                expected_soil_dz=make_soil_grid(cfg.soil_grid).dz,
                # The calibrated column is not the historical default, so an
                # older restart carrying no stamp is almost certainly on the
                # wrong one: refuse it rather than warn.
                require_soil_dz=bool(getattr(
                    self.config, "land_calibrated_physics", False)))
            # Scatter BEFORE the graft: the template is rank-local, so a global
            # restart grafted onto it would carry the whole globe's columns into
            # a rank-local state.
            if _part_ic is not None:
                _ic_state = jax.tree_util.tree_map(
                    lambda a: _land_columns_to_local(a, _part_ic), _ic_state)
            # Graft the restart's prognostic columns onto the canonical template
            # (fixes the pytree structure), then cast the array leaves to the
            # run's storage precision (the restart deserialises float64).
            _merged = merge_land_restart_into_template(_ic_state, _template)
            # A REGRIDDED state's matric potential is not this run's.  The
            # Richards step evolves potential directly, but potential and water
            # content are tied through each column's own soil-texture retention
            # curve — and a regridded column carries the SOURCE column's
            # texture in its potential.  Water content is the conserved
            # quantity, so keep theta and re-derive psi on THIS run's
            # hydraulics, exactly as the cold-start does.  Scoped to states
            # whose metadata says they were regridded: a byte-exact same-grid
            # restart is left untouched.
            if _ic_meta.get("regridded_from"):
                from legoesm.land.soil_hydraulics import psi_from_theta
                _merged = _merged._replace(
                    psi_soil=psi_from_theta(_merged.theta_soil,
                                            cfg.hydraulics))
                logger.info(
                    "  Land tile: regridded IC (%s) — psi_soil re-derived "
                    "from theta_soil on this run's soil texture.",
                    _ic_meta.get("regridded_from"))
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

        # CLM-ML canopy: warm-start ONCE (eager) so the jitted run steps can thread
        # a concrete per-column ``grid_info`` (S2) and start from a warm
        # ``canopy_state``.  The jitted forward path is single-column unless the
        # caller supplies one GridInfo per column; the first (cold) canopy step
        # builds the vertical structure host-side and cannot run on the jax.jit
        # tape, so we run it HERE, eagerly, at setup.  The structure
        # (ncan/ntop/nbot) is a function of canopy height / PFT only (forcing-
        # INDEPENDENT), so a nominal concrete forcing suffices to build it; the
        # returned soil step is DISCARDED (only ``canopy_state`` is grafted onto
        # the cold-start template — the cold soil IC is preserved).  t_a10
        # cold-starts to T_lowest (converges in ~10 days).
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        if isinstance(cfg.surface_scheme, CLMMLCanopyConfig):
            # The eager cold-canopy warm-start (builds the per-column grid_info +
            # a warm canopy_state, host-side) is the ONE shared helper the offline
            # run_lmip_biophys driver also calls, so they cannot drift.  Per-column
            # PFT topology is opt-in via use_surfdata_pft (the surface map's
            # dominant PFT); None keeps the single pft_clm for all columns.
            from legoesm.land.canopy.clm_ml_interface import warm_start_clm_ml
            _pft_fracs = (surface_map["pft_fractions"]
                          if getattr(cfg.surface_scheme, "use_surfdata_pft", False)
                          else None)
            self._land_ml_state, _gi, _clm_ml_pft_per_col = warm_start_clm_ml(
                self._land_ml_state, cfg, params,
                lat=self.physics.land_ml_lat,
                u_min=self.physics.land_ml_u_min,
                dt=float(self.config.dycore.dt), ncol=ncol, T_init=T_init,
                carbon_state=self.physics.land_ml_carbon,
                pft_fractions=_pft_fracs, dtype=storage_dtype)
            self.physics.clm_ml_pft_per_col = _clm_ml_pft_per_col
            self.physics.clm_ml_grid_info = _gi
            logger.info(
                "  Land tile: CLM-ML canopy warm-started (%d columns; per-column "
                "grid_info threaded into the jitted step)",
                (len(_gi) if isinstance(_gi, tuple) else 1),
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
        _rebuilt = rebuild(fracs)()
        # The transient rebuild re-weights vegetation params by cover year via the
        # CLM (LAI-only) provider; it does NOT carry the CLM-ML canopy's prescribed
        # SAI / htop, which are STRUCTURAL (frozen, not cover-varying).  Carry them
        # from the setup-time params so a transient CLM-ML run does not silently
        # revert to the htop=5 / SAI=0.5 scalar defaults after step 0 (codex).
        _init = getattr(self.physics, "land_ml_params", None)
        if _init is not None:
            _carry = {f: getattr(_init, f)
                      for f in ("SAI", "htop")
                      if getattr(_rebuilt, f, None) is None
                      and getattr(_init, f, None) is not None}
            if _carry:
                _rebuilt = _rebuilt._replace(**_carry)
        return _rebuilt

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

        # PINNED to the bulk scheme, not the library default.  This adapter wants
        # ONE number per column — the blended soil/vegetation albedo the radiation
        # uses — and reads it as ``albedo_veg``.  The default is now the two-leaf
        # canopy, whose parameter struct has no such field (it carries BAND
        # albedos and does its own radiative transfer), so inheriting the default
        # crashed here.  A canopy run gets its albedo from the canopy itself; this
        # path is the static-map fallback and is bulk by construction.
        from legoesm.land.surface_scheme import SimpleSEBConfig
        land_cfg = LandConfig(surface_scheme=SimpleSEBConfig())
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
        from legoesm.forcing.surface_utils import (
            distribute_column_aod_to_layers,
            place_stratospheric_aod_profile_to_layers,
        )

        nlev = self.sigma.sigma_full.shape[0]
        shape_2d = p_s.shape
        ncol = int(np.prod(np.array(shape_2d)))

        p_full = self.sigma.pressure_at_full(p_s)
        p_half = self.sigma.pressure_at_half(p_s)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        lat_col = lat.reshape(ncol)

        if self._ozone_ext_active:
            o3_vmr = jnp.asarray(get_ozone_at_time(
                self._ozone_ext_config, day,
                lat_grid=lat_col, p_grid=p_full_col,
            ))
        else:
            # External ozone forcing inactive — build the inline profile
            # HERE so the radiation JIT signature stays stable (it always
            # sees a real array).  Previously ``o3_vmr`` was initialised to
            # zeros and threaded into the solver, which then clipped it to
            # 1e-10 and effectively disabled stratospheric ozone heating
            # (audit 2026-05-12 HIGH #3).
            #
            # Honor ``cfg.ozone_source`` (2026-07-21 audit): this precomputed
            # array reaches the compiled radiation as ``o3_vmr_override``,
            # which takes precedence over the backend's own
            # ``_compute_ozone_vmr`` — so building only the "standard"
            # profile here silently ignored ``--ozone-source analytical``/
            # ``none`` on the cd-grid/latlon pipeline while MPAS/spectral
            # honored them (grid-dependent physics from the same config).
            # Reuse the backend's dispatcher for exact parity.
            from legoesm.atmosphere.physics.radiation.integration import (
                compute_ozone_vmr as _compute_ozone_vmr,
            )
            from legoesm.atmosphere.physics.radiation.config import (
                OzoneProfileConfig,
            )
            _o3_inline = _compute_ozone_vmr(
                p_full_col, lat_col,
                OzoneProfileConfig(source=self.config.ozone_source),
            )
            if _o3_inline is None:
                # source="standard": the backend contract is o3_vmr=None →
                # its built-in US Std Atm 1976 fit; materialize the same
                # profile for the stable-signature override path.
                from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
                    standard_o3_profile,
                )
                _o3_inline = standard_o3_profile(p_full_col)
            o3_vmr = jnp.asarray(_o3_inline).astype(p_s.dtype)

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
        # od is a RRTMGP no-op).  ``get_aerosol_lw_at_time`` returns the
        # file's height-resolved ABSORPTION profile (ext*(1-omega), Planck-
        # weighted gray band collapse) plus its pressure edges; the profile
        # is placed at its true stratospheric pressure by a conservative
        # overlap remap -- NOT spread by full-column pressure mass, which
        # dumped ~90% of a stratospheric aerosol into the troposphere.
        # Stored as an instance attribute (NOT added to the 3-tuple return)
        # so the five existing unpack call sites keep their arity.
        aerosol_lw_od = jnp.zeros((ncol, nlev), dtype=p_s.dtype)
        if self._aerosol_lw_active:
            aerosol_lw_prof = get_aerosol_lw_at_time(
                self._aerosol_config, day, lat_grid=lat_col,
            )
            if aerosol_lw_prof is not None:
                prof_col, p_edges = aerosol_lw_prof
                aerosol_lw_od = place_stratospheric_aod_profile_to_layers(
                    jnp.asarray(prof_col), jnp.asarray(p_edges), p_half_col,
                ).astype(p_s.dtype)
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
                # p_xr/alpha_xr set the Xu-Randall cloud FRACTION, so the clt
                # diagnostic must thread them too or published clt drifts from
                # the radiation cloud fraction (codex review, pre-existing gap).
                p_xr=getattr(self.config, "cloud_p_xr", None),
                alpha_xr=getattr(self.config, "cloud_alpha_xr", None),
                diagnostic_condensate_scheme=getattr(
                    self.config, "cloud_diagnostic_condensate_scheme", None),
                adiabatic_lwc_rate=getattr(
                    self.config, "cloud_adiabatic_lwc_rate", None),
                # The RH saturation CURVE sets the cloud fraction itself, so
                # the clt diagnostic must thread it for the same reason as
                # p_xr/alpha_xr above: without it a mixed_phase run would
                # publish clt computed on the LIQUID curve while radiation
                # integrated the mixed-phase cloud — the published clt would
                # miss exactly the cold cirrus the switch adds (#1521).
                saturation_scheme=getattr(
                    self.config, "cloud_saturation_scheme", None),
                cover_condensate_q_ref=getattr(
                    self.config, "cloud_cover_condensate_q_ref", None),
            )
        self.diagnostics = DiagnosticCollector(
            nlev=self.config.grid.nlev,
            sigma_full=self.sigma.sigma_full,
            vcoord=self.sigma,
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
            # tas 2 m profile uses the SAME MOST bulk scheme and stable-branch
            # functions as the surface fluxes (defaults are byte-identical:
            # "constant" keeps the historical coare3 profile stand-in).
            surface_stability_scheme=getattr(
                self.config, "surface_stability_scheme", "dyer1974"),
            surface_bulk_scheme=getattr(
                self.config, "surface_bulk_scheme", "constant"),
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
            # Under a MULTI-rank Voronoi cell partition ``self.grid`` is this
            # rank's owned+halo submesh, so its IDW weights would bin
            # rank-local cells into the GLOBAL lat-lon boxes.  The CMOR feed
            # gathers owned cells to their GLOBAL slots
            # (:meth:`_feed_mpas_cmip_multirank`), so the collector must hold
            # the GLOBAL mesh's weights and the GLOBAL fixed fields.  Scoped
            # to the Voronoi partition: the lat-lon band layout also sets
            # ``_grid_global`` but reaches CMOR through ``collect()``, a path
            # this change deliberately leaves alone.
            _cmip_grid = self.grid
            _cmip_phis = self._phis_data
            _cmip_fland = self._f_land
            _vl = self._voronoi_layout
            if _is_mpas_cell_partitioned(self):
                from legoesm.parallel.voronoi_mpi import gather_voronoi_field
                if self._grid_global is None:
                    # Fail LOUD: silently keeping the rank-local weights here
                    # while the feed hands rank 0 global-length arrays is the
                    # exact defect this change removes, and it would surface
                    # only as an empty ``cmor/`` at the end of a long run.
                    raise RuntimeError(
                        "MPAS cell-partitioned run has no _grid_global — the "
                        "CMOR collector cannot be given global regrid weights."
                    )
                _cmip_grid = self._grid_global
                # Collective, but symmetric: every rank runs
                # ``_create_diagnostics`` in ``setup()``, and both gathers are
                # unconditional here.
                _cmip_phis = gather_voronoi_field(
                    self._phis_data, _vl.partition, "cell")
                _cmip_fland = gather_voronoi_field(
                    self._f_land, _vl.partition, "cell")
            self.diagnostics.set_cmip_grid_info(
                grid_type=self.config.grid.grid_type,
                grid=_cmip_grid,
                start_year=self.config.start_year,
            )
            # Register time-invariant fields for the CMIP6 ``fx`` file.
            # _phis_data is the ETOPO field; dynamics run with ERA5 phis but
            # CMOR orog reports the ETOPO field (the intended mountain mask).
            self.diagnostics.set_fixed_fields(
                phis=np.asarray(_cmip_phis),
                land_fraction=np.asarray(_cmip_fland),
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
                # #1354/#1515: forward the TOA-down + surface turbulent fluxes
                # so the energy-budget tracker runs on this (MPAS) path too.
                sw_down_toa=kwargs.get('sw_down_toa', None),
                shflx=kwargs.get('shflx', None),
                lhflx=kwargs.get('lhflx', None),
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
                p_full = self.sigma.pressure_at_full(p_s)
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
            config_hash_matches,
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
                # #1509: the --params values this run actually applied, so the
                # manifest does not depend on the referenced file surviving
                # unmodified. Absent (-> {}) when no --params were given.
                params_applied=getattr(self, "_params_applied", None),
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
        if not config_hash_matches(
                existing["config"]["config_hash"],
                existing["config"].get("resolved_config") or {},
                self._input_config):
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
        if self.config.dycore.discretization == "fv3_duo":
            # The duo lane owns its own SPMD (face-axis sharding via the
            # model knobs, PR #1656); the generic layouts below assume
            # the standard cubed-sphere/lat-lon state and would build a
            # face-partition MPI layout the duo bundle never consumes.
            return
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
                # Refuse multilayer land under a SUB-FACE TILED layout (>6
                # ranks, 6*k^2) BEFORE any scatter: a tiled rank owns a face
                # TILE while the compiled physics selects WHOLE owned faces
                # (_owned_face_ids), so the per-column soil scatter (face-axis
                # reshape) and whole-face physics are incompatible.  Raising
                # here — before any field is scattered or the adapter rebuilt —
                # keeps a rejected tiled setup from partially mutating the
                # driver.
                if (self._land_ml_state is not None
                        and getattr(layout, "is_tiled", False)):
                    raise NotImplementedError(
                        "use_multilayer_land under sub-face tiled MPI "
                        "(>6 ranks) is not supported: physics runs on whole "
                        "owned faces while the tiled layout owns face TILES. "
                        "Use <=6 ranks (whole-face cube MPI) for multilayer "
                        "land, or slab land for the tiled lane."
                    )
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

                # Scatter the per-column subgrid orography so the orographic
                # GWD launch reads this rank's owned-face columns (same
                # ownership as _physics_lat above). Without this, the GWD
                # integration's reshape(-1)[:ncol] would hand every rank the
                # first ncol GLOBAL columns — geographically wrong SSO.
                if getattr(self.grid, "subgrid_topo_stddev", None) is not None:
                    self.grid = self.grid._replace(
                        subgrid_topo_stddev=scatter(
                            self.grid.subgrid_topo_stddev, layout
                        )
                    )

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
                    # Rank-local SSO for the pipeline's column GWD path
                    # (same ownership as f_land / _physics_lat).
                    if getattr(self.physics, "subgrid_topo_stddev", None) \
                            is not None:
                        self.physics.subgrid_topo_stddev = scatter(
                            self.physics.subgrid_topo_stddev, layout)

                # Multilayer (Richards) land: scatter every per-column field
                # (state, lat, params, carbon) to owned faces so the rank-local
                # physics columns advance THIS rank's soil columns — the same
                # ownership as _physics_lat / f_land.  The columns are
                # embarrassingly parallel (no lateral soil coupling), so the
                # gathered N-rank state is bit-identical to the single-rank run
                # (gated by test_multilayer_land_scatter_mpi).  After this the
                # distributed-multilayer guard in ``run`` is satisfied.
                if self._land_ml_state is not None:
                    # (Sub-face tiled layouts were already refused above, before
                    # any scatter.)
                    n_tile = int(self.state.T.data.shape[1])
                    global_ncol = 6 * n_tile * n_tile
                    if self._land_cover_transient is not None:
                        # Transient LULC rebuilds params from GLOBAL cover each
                        # segment; scattering that per-segment rebuild is a
                        # follow-up.  Refuse rather than feed global params to
                        # rank-local columns (silent geographic mismatch).
                        raise NotImplementedError(
                            "transient_land_cover with multilayer land under "
                            "MPI is not yet supported (the per-segment param "
                            "rebuild is global); run single-rank, or use "
                            "static land cover for distributed multilayer runs."
                        )
                    # The 0-based per-column leaves (soil state, t_a10_arr) scatter
                    # normally.  The CLM-ML canopy_state's ``mlcanopy`` pytree is
                    # 1-based (ncol+1) so tree_map here SKIPS it (its leading axis is
                    # ncol+1, not global_ncol) — it is scattered separately below.
                    self._land_ml_state = _map_flat_column_leaves(
                        self._land_ml_state, n_tile, global_ncol,
                        lambda x, n: _scatter_flat_columns(x, layout, n))
                    if getattr(self.physics, "land_ml_lat", None) is not None:
                        self.physics.land_ml_lat = _scatter_flat_columns(
                            self.physics.land_ml_lat, layout, n_tile)
                    # Per-column CLM PFT (mixed-PFT columns) is a 0-based (global_ncol,)
                    # vector like land_ml_lat -> scatter it to the rank's local columns
                    # too, else the coupled local step would pass the GLOBAL-length
                    # array into the (ncol,)-shape-checked pft_per_col (codex).
                    if getattr(self.physics, "clm_ml_pft_per_col", None) is not None:
                        self.physics.clm_ml_pft_per_col = _scatter_flat_columns(
                            self.physics.clm_ml_pft_per_col, layout, n_tile)
                    if getattr(self.physics, "land_ml_params", None) is not None:
                        self.physics.land_ml_params = _map_flat_column_leaves(
                            self.physics.land_ml_params, n_tile, global_ncol,
                            lambda x, n: _scatter_flat_columns(x, layout, n))
                    if getattr(self.physics, "land_ml_carbon", None) is not None:
                        self.physics.land_ml_carbon = _map_flat_column_leaves(
                            self.physics.land_ml_carbon, n_tile, global_ncol,
                            lambda x, n: _scatter_flat_columns(x, layout, n))
                    # CLM-ML canopy: scatter the 1-based ``mlcanopy`` (ncol+1) leaves
                    # with the 1-based helper, then slice + renumber the per-column
                    # ``grid_info`` to this rank's owned columns.  Columns are
                    # embarrassingly parallel (no lateral canopy coupling), so the
                    # gathered N-rank canopy is bit-identical to the single-rank run
                    # (same invariant as the soil scatter).  This makes MPI a partial
                    # scale path even before S3: each rank compiles an O(ncol/ranks)
                    # canopy loop.
                    _cs = self._land_ml_state.canopy_state
                    if _cs is not None and getattr(_cs, "mlcanopy", None) is not None:
                        _cs = _map_flat_column_leaves(
                            _cs, n_tile, global_ncol + 1,
                            lambda x, n: _scatter_1based_columns(x, layout, n))
                        self._land_ml_state = self._land_ml_state._replace(
                            canopy_state=_cs)
                        # Per-rank grid_info: scatter a global-index array to learn
                        # which global columns this rank owns (in local order), slice
                        # the global grid_info tuple to them, and renumber ``.p`` to
                        # local 1-based (the interface realigns by ``.p`` and requires
                        # patches 1..n_local).
                        _gi = getattr(self.physics, "clm_ml_grid_info", None)
                        if isinstance(_gi, tuple):
                            self.physics.clm_ml_grid_info = (
                                _slice_grid_info_to_rank(
                                    _gi, layout, n_tile, global_ncol))
                    self._land_ml_scattered = True
                    self._land_ml_n_tile = n_tile

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

    # ------------------------------------------------------------------
    # #1028 — persistent D-grid winds on the cubed-sphere hydrostatic lane
    # ------------------------------------------------------------------
    # The cube dycore has always accepted either staggering: its ``step``
    # dispatches on the state type, and an ``FV3HydrostaticState`` skips the
    # cell-centre -> corner -> cell-centre projection that the cell-centre
    # entry performs on EVERY step.  That projection is a measured eddy
    # damper: one paired 200-day C36 Held-Suarez run with this as the only
    # change moved the equilibrated max wind 13.0 -> 38.6 m/s (sigma) and
    # 12.6 -> 40.9 (hybrid) (PR #1462, docs/atmosphere/
    # cube_structural_gaps_vs_fv3.md).  What was missing was a production
    # driver that CARRIES the D state; this is that wiring.
    #
    # Everything that reads winds as cell-centred (physics inputs, CFL,
    # diagnostics, output, checkpoints) goes through :meth:`_state_cc`, a
    # read-only view.  Nothing writes a converted state back over the native
    # winds, so the round trip never re-enters the prognostic path.

    def _persistent_dgrid_active(self) -> bool:
        """True iff the run is carrying D-staggered cube winds."""
        return bool(getattr(self, "_persistent_dgrid", False))

    def _enter_persistent_dgrid(self) -> None:
        """Convert the prognostic state to FV3 D staggering, once, at run start.

        Refuses LOUDLY on every lane this does not cover rather than silently
        running the damped cell-centre path (dispatch discipline): the whole
        point of the flag is the staggering the run actually carries, so a
        silent fall-back would be the defect it exists to remove.
        """
        self._persistent_dgrid = False
        if not getattr(self.config.dycore, "persistent_dgrid", False):
            return
        cfg = self.config
        _why = None
        if cfg.grid.grid_type != "cubed_sphere":
            _why = (f"grid_type={cfg.grid.grid_type!r}: D staggering is the "
                    "cubed-sphere FV3 wind layout")
        elif cfg.dycore.model_type != "hydrostatic":
            _why = (f"model_type={cfg.dycore.model_type!r}: only the "
                    "hydrostatic cube dycore carries FV3HydrostaticState")
        elif cfg.dycore.discretization not in ("cdgrid", "centered",
                                               "finite_volume"):
            _why = (f"discretization={cfg.dycore.discretization!r}: the "
                    "fv3_duo / spectral / sfno lanes carry their own state")
        elif getattr(self, "_ensemble_size", 1) not in (None, 1):
            _why = ("an ensemble run: the member axis would have to ride the "
                    "corner arrays and no gate covers that yet")
        elif getattr(self, "_owned_face_ids", None) is not None:
            _why = ("MPI face sharding: the corner arrays are (n+1) in both "
                    "horizontal directions and the owned-face slicing this "
                    "driver applies is written for (n)")
        elif self._is_spmd_multiprocess():
            _why = "multi-process SPMD (the corner sharding is unvalidated)"
        elif getattr(self, "_requires_surface_flux_export", False):
            _why = ("a COUPLED run: the coupled driver reads the atmosphere's "
                    "lowest-level winds straight off the state "
                    "(coupled_esm_driver) and has no cell-centre view yet")
        elif (self._device_config is not None
              and getattr(self._device_config, "mesh", None) is not None
              and tuple(getattr(self._device_config, "tiling", (1, 1)))
              != (1, 1)):
            _why = ("sub-face tiling: the tiled adapter converts to cell "
                    "centres by construction (tiled_step_adapter)")
        if _why is not None:
            raise SystemExit(
                "dycore.persistent_dgrid=True is not supported on this lane: "
                f"{_why}.  Persistent D-grid winds (#1028) are wired for the "
                "single-process cubed-sphere hydrostatic lane only.  Unset the "
                "flag to run the (eddy-damped) cell-centre path knowingly.")

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            FV3HydrostaticState, hydrostatic_to_fv3,
        )
        cdgrid = getattr(self.model, "cdgrid", None)
        if cdgrid is None:
            raise SystemExit(
                "dycore.persistent_dgrid=True but the dynamics model exposes "
                "no cdgrid; the cc->D lift has no geometry to use.")
        if isinstance(self.state, FV3HydrostaticState):
            # Already D (a restart that carried the native winds).
            self._persistent_dgrid = True
            return
        self.state = hydrostatic_to_fv3(self.state, cdgrid)
        self._persistent_dgrid = True
        logger.info(
            "#1028 persistent D-grid winds: the cube state stays in FV3 D "
            "staggering between steps (one cc->D lift at run start; the "
            "per-step cc round trip is gone).")

    def _state_cc(self, state=None):
        """Cell-centre VIEW of the prognostic state — read-only.

        Identity when the run is not carrying D winds, so every call site is
        byte-identical on the default path.  Never assign the result back to
        ``self.state``: that would reinstate the projection this removes.
        """
        st = self.state if state is None else state
        if not self._persistent_dgrid_active():
            return st
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            FV3HydrostaticState, fv3_to_hydrostatic,
        )
        if not isinstance(st, FV3HydrostaticState):
            return st
        return fv3_to_hydrostatic(st, self.model.cdgrid)

    def _add_cc_wind_increment(self, du_cc, dv_cc, dt):
        """Add an operator-split CELL-CENTRE wind tendency to ``self.state``.

        On the D lane the increment is lifted to the corners with the SAME
        rotation-aware interpolation the dycore applies to its own cell-centre
        physics tendencies (``center_to_dgrid_vector``; the dycore's
        ``_lift_uv_cc`` is a one-line wrapper over it).  The lift is linear, so
        lifting the increment and lifting the whole state give the same answer
        for this term -- what changes is that the CARRIED state is never
        projected back, which is where the eddy damping lived.
        """
        if not self._persistent_dgrid_active():
            self.state = self.state._replace(
                u=self.state.u.replace(data=self.state.u.data + dt * du_cc),
                v=self.state.v.replace(data=self.state.v.data + dt * dv_cc),
            )
            return
        from legoesm.core.operators_cdgrid import center_to_dgrid_vector
        du_d, dv_d = center_to_dgrid_vector(du_cc, dv_cc, self.model.cdgrid)
        self.state = self.state._replace(
            u_d=self.state.u_d.replace(data=self.state.u_d.data + dt * du_d),
            v_d=self.state.v_d.replace(data=self.state.v_d.data + dt * dv_d),
        )

    def _prognostic_wind_leaf(self):
        """The carried zonal-wind array, whatever its staggering.

        Used where the driver only needs SOMETHING to block on / inspect and
        must not care whether the run carries cell-centre or D-grid winds.
        """
        st = self.state
        return (st.u_d.data if hasattr(st, "u_d") else st.u.data)

    def _scale_winds(self, factor):
        """Multiply the prognostic winds by ``factor`` (Rayleigh decay).

        ``factor`` depends on the vertical coordinate only, so it commutes
        with the cc<->D interpolation: scaling the corner winds is the same
        operation as scaling the centre winds and lifting.  A horizontally
        varying drag would NOT commute and must not use this helper.
        """
        if not self._persistent_dgrid_active():
            self.state = self.state._replace(
                u=self.state.u.replace(data=self.state.u.data * factor),
                v=self.state.v.replace(data=self.state.v.data * factor),
            )
            return
        self.state = self.state._replace(
            u_d=self.state.u_d.replace(data=self.state.u_d.data * factor),
            v_d=self.state.v_d.replace(data=self.state.v_d.data * factor),
        )

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
        # restart-chain sbatch globs.  Under a Voronoi cell partition the
        # per-cell arrays ARE gathered to global below and only rank 0
        # writes — the "single-process only" note that used to sit here was
        # stale prose contradicted by the gather a few lines down.
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
            _land_ml_save = None      # set to the GLOBAL gather under MPI
            _skin_save = None         # ditto for the prognostic ice skin
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
                    from legoesm.atmosphere.physics.physics_state import (
                        PHYSSTATE_INPUT_FIELDS,
                    )
                    ps_d_carry = {}
                    for _name in _ps_carry._fields:
                        _val = getattr(_ps_carry, _name)
                        # Per-step INPUT fields (dyn_tendency_*) are never
                        # persisted: they are recomputed by the driver each
                        # step, so a checkpointed value would be stale, and a
                        # None value would emit an unloadable object array
                        # (allow_pickle=False).  Skip by NAME so BOTH None and
                        # a concrete-array carry are excluded (load re-seeds
                        # them None).
                        if _name in PHYSSTATE_INPUT_FIELDS:
                            continue
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
                # Multilayer-land columns are per-cell too (#1321).  Gathered
                # HERE, with the other collectives and BEFORE the rank-0 bail
                # below: the save site is rank-0-only, so gathering there would
                # hang every other rank.  The pytree structure is identical on
                # every rank (same config), so each rank issues the same
                # gathers in the same order.
                if self._land_ml_state is not None:
                    _nloc = int(part.n_local_cells)

                    def _g(x, _n=_nloc, _p=part):
                        if not hasattr(x, "shape") or getattr(x, "ndim", 0) < 1:
                            return x
                        if int(x.shape[0]) == _n:
                            return gather_voronoi_field(x, _p, "cell")
                        if int(x.shape[0]) == _n + 1:       # CLM 1-based
                            return jnp.concatenate(
                                [x[:1], gather_voronoi_field(x[1:], _p, "cell")],
                                axis=0)
                        return x

                    _land_ml_save = jax.tree_util.tree_map(
                        _g, self._land_ml_state)
                # Prognostic ice skin: a cell field, so it gathers like the
                # rest — and it gathers HERE, with the other collectives and
                # before the rank-0 bail, or every non-root rank hangs.  The
                # Every rank must agree on whether to issue this collective or
                # it deadlocks, and the selection below reads per-rank STATE,
                # not the config flag -- so the agreement is worth stating.
                # It holds because the live field is set for EVERY rank at the
                # top of ``_run_mpas`` whenever the feature is on (seeded, not
                # conditional on that rank owning ice), and the staged fallback
                # comes from a checkpoint every rank loads. A future change
                # that makes either one conditional on a rank's own cells would
                # reintroduce the hang.
                if getattr(self.config, "mpas_ice_skin_prognostic", False):
                    _skin_local = getattr(self, "_ice_T_skin", None)
                    if _skin_local is None and isinstance(self._carry_aux, dict):
                        _skin_local = self._carry_aux.get("ice_T_skin")
                    if _skin_local is not None:
                        _skin_save = gather_voronoi_field(
                            jnp.asarray(_skin_local).reshape(-1), part, "cell")
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
            # Vertical level POSITIONS travel with the state.  nlev alone no
            # longer identifies the grid: a uniform-σ L30 and a
            # tropopause-refined L30 (grid.tropopause_refine) have identical
            # SHAPES, so the shape guard on load cannot tell them apart and
            # would silently reinterpret every profile on the wrong levels.
            # Same reasoning as ``physstate_meta_conv_scheme`` below.
            # Stored as the (A, B) half-level pair, i.e. p_half = A·p_ref +
            # B·p_s, because that is what actually fixes the pressures.
            # ``p_ref`` is NOT stored: no driver path overrides it (the sole
            # constructor call below passes only nlev/p_top_Pa/stretching, and
            # ``make_hybrid_levels`` defaults to ``constants.p_ref``), so it is
            # a global invariant and equal (A, B) implies equal pressures.
            # If p_ref ever becomes configurable it MUST join this array
            # (codex round 3).
            # ``sigma_half`` ALONE is NOT sufficient for the hybrid: with
            # B = eta**n, A = eta - B + (p_top/p_ref)(1-eta), the sum
            # A + B = eta + (p_top/p_ref)(1-eta) is INDEPENDENT of the
            # transition exponent, so two physically different hybrid grids
            # share one sigma_half (codex round 2).  Pure σ is the A = 0,
            # B = σ member of the same family, so one array covers both.
            _vg = getattr(self.sigma, "A_half", None)
            _save["meta_vgrid"] = np.stack([
                (np.zeros_like(np.asarray(self.sigma.sigma_half,
                                          dtype=np.float64))
                 if _vg is None else np.asarray(_vg, dtype=np.float64)),
                np.asarray(getattr(self.sigma, "B_half", self.sigma.sigma_half),
                           dtype=np.float64),
            ])
            # Stateful-physics carry (#413): persisted under
            # ``physstate_<field>`` so a chained restart resumes the
            # prognostic physics memory instead of silently reseeding.
            # The convection scheme tag travels with it (codex round 8:
            # the profile-prognostic schemes share the carry shape, so
            # shape checks alone cannot catch a cross-scheme restore).
            if ps_d_carry is not None:
                from legoesm.atmosphere.physics.physics_state import (
                    PHYSSTATE_INPUT_FIELDS,
                )
                for _name, _val in ps_d_carry.items():
                    # Never persist per-step INPUT fields (the serial
                    # ``_asdict`` path includes them; the MPI-gather path
                    # already dropped them).  Skip by NAME so a concrete-array
                    # carry is excluded too, not only None (``np.asarray(None)``
                    # is an unloadable object array under allow_pickle=False).
                    if _name in PHYSSTATE_INPUT_FIELDS:
                        continue
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
                # Droplet number is stored PER MASS [1/kg] since 2026-08-14.
                # Stamp it: a file written under the old per-VOLUME convention
                # reloaded as per-mass is wrong by the air density, silently,
                # and there is no other way to tell the two apart.
                _save["number_convention"] = np.asarray("per_mass")
                for _k in trc_d:
                    _save[f"trc_{_k}"] = np.asarray(trc_d[_k])
            # Multilayer (Richards) land columns (MPAS port): same namespaced
            # ``land_ml_<field>`` payload as the coupled carry_aux channel
            # (_checkpoint_carry_aux), so the loader can reuse the shared
            # fail-loud restore (#730 contract).  None fields are skipped on
            # save; restore validates the field-set exactly.
            if self._land_ml_state is not None:
                # Under MPAS MPI this is the GLOBAL gather assembled above, so
                # the restart chain reads one canonical checkpoint rather than
                # a rank-local fragment (#1321).
                _lm_out = (_land_ml_save if _land_ml_save is not None
                           else self._land_ml_state)
                # The soil COLUMN these columns belong to: the field shapes
                # record the layer count only, and two columns with the same
                # count can span different depths.
                _lm_dz = self._land_soil_dz()
                if _lm_dz is not None:
                    _save["land_soil_dz"] = _lm_dz
                for _f, _v in _lm_out._asdict().items():
                    if _v is not None:
                        _save[f"land_ml_{_f}"] = np.asarray(_v)
            # Prognostic ice skin (mpas_ice_skin_prognostic): persist so a
            # 12h chain link resumes the equilibrated skin instead of
            # re-running the ~weeks-long spin-up from T_freeze_ocean every
            # restart.  Gated on THIS run's config so a driver reused
            # feature-on -> feature-off (without a load, which clears the
            # field) cannot launder a stale skin into an off-feature
            # checkpoint (codex-1 finding 6).  Absent on runs without the
            # feature (byte-identical restart).
            if getattr(self.config, "mpas_ice_skin_prognostic", False):
                _skin = getattr(self, "_ice_T_skin", None)
                if _skin is None and isinstance(self._carry_aux, dict):
                    # Loaded-but-not-yet-adopted: load_checkpoint stages the
                    # skin into _carry_aux and clears the live field until
                    # _run_mpas overlays it.  A save BEFORE that run (load ->
                    # save with no step) must persist the staged value, not
                    # strip it (codex-2 finding 3).
                    _skin = self._carry_aux.get("ice_T_skin")
                if self._voronoi_layout is not None:
                    # Under MPI the live field is this rank's owned+halo band;
                    # persist the GLOBAL gather taken above so the chain reads
                    # one canonical checkpoint at any rank count.
                    _skin = _skin_save
                if _skin is not None:
                    _save["ice_T_skin"] = np.asarray(_skin)
            # #1353 (codex-2 finding 2): within-interval CMOR flux sums —
            # so a mid-interval (wallclock / off-cadence) restart resumes
            # the interval mean instead of dropping the pre-checkpoint
            # samples.  Empty on the aligned chain (the diag feed + reset
            # ran before this at the same step boundary).  Loaded-but-not-
            # yet-adopted (load -> save with no run): forward the STAGED
            # payload from _carry_aux so a no-step re-save cannot strip the
            # partial interval (same rule as the ice skin above).
            #
            # NOT persisted under a MULTI-rank cell partition: the sums are
            # per-LOCAL-cell ``(n_local_cells,)`` and are NOT gathered, while
            # everything else in this checkpoint is global and only rank 0
            # writes it.  Persisting them would hand every rank of the next
            # link RANK 0's cells attributed to its own — and because RCB
            # partitions are near-equal, ``n_local_cells`` often MATCHES, so
            # the shape guard would pass and the fluxes would be silently
            # scrambled.  Dropping costs at most ONE partial diagnostic
            # interval of flux samples per restart, which the reset below
            # already treats as an accepted, bounded loss.
            _facc = getattr(self, "_mpas_sfc_accum", None)
            if _is_mpas_cell_partitioned(self):
                if _facc is not None and _facc.has_samples():
                    logger.warning(
                        "  CMOR flux accumulator: NOT persisted (%d-rank cell "
                        "partition — the per-cell sums are rank-local and "
                        "ungathered). The next link restarts this diagnostic "
                        "interval; at most one interval of flux samples is "
                        "lost.", getattr(self, "_mpi_world_size", 1))
            elif _facc is not None and _facc.has_samples():
                _save.update(_facc.dump())
            elif isinstance(self._carry_aux, dict):
                for _k, _v in self._carry_aux.items():
                    if _k.startswith("cmor_flux"):
                        _save[_k] = np.asarray(_v)
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
                # Same per-mass droplet-number stamp as the MPAS writer above.
                _save["number_convention"] = np.asarray("per_mass")
                for _k in s.tracers:
                    _save[f"trc_{_k}"] = np.asarray(s.tracers[_k].data)
            # #1310: persist the anchored mass-fixer target.  With
            # ``fix_mass + anchor_mass_to_initial`` the spectral model snapshots
            # ``_target_mass`` from the FIRST state it sees; on restart a fresh
            # model would re-anchor to the LOADED (mid-run) state — a target
            # ~1e-15 off the original epoch mass — so the per-step lnps rescale
            # pins the resumed trajectory to a different mass, breaking bitwise
            # continuation (uniform ~1e-8 drift by day 2).  Save the concrete
            # target and restore it via ``set_target_mass`` so the resumed run
            # anchors to the IDENTICAL mass.
            _tmass = getattr(self.model, "_target_mass", None)
            if _tmass is not None:
                _save["target_mass"] = np.asarray(_tmass, dtype=np.float64)
            # SCOPE (bit-exact restart): the state (*_hat + trc_*) and the
            # mass-fixer anchor (target_mass) are the full checkpoint for the
            # self-starting integrators (ssp_rk3/34/54 — the spectral default).
            # The opt-in ``leapfrog_si`` path additionally holds 3-time-level
            # history (``_state_prev``/``_prev_phys_tend``) that is NOT
            # persisted here, so a leapfrog_si resume re-bootstraps via the
            # forward-Euler startup branch and is NOT bitwise (a separate,
            # pre-existing gap; checkpointing that history is the follow-up).
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
                    state=self._state_cc(),      # #1028: cc layout on disk
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
        # #1028: checkpoints keep the CELL-CENTRE wind layout every reader and
        # analysis tool already expects, so a persistent-D run writes its
        # cell-centre view here and ``_enter_persistent_dgrid`` lifts it back
        # on the next start.  KNOWN COST, stated rather than hidden: that is
        # one cc<->D projection per RESTART, against one per STEP on the old
        # path -- at a 30-day restart cadence and a 600 s step, ~2e-4 of the
        # old rate -- and it means a restarted persistent-D run is not
        # bit-identical to an uninterrupted one.  Storing the corner winds
        # natively (new leaf names + a format stamp) is the follow-up that
        # removes even that.
        _state, _q_v, _q_c, _q_r = (
            self._state_cc(), self.q_v, self.q_c, self.q_r)
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
        # Same zarr carry_aux limitation for the HELD radiation fluxes: with a
        # radiation cadence (rad_update_steps > 1) the held sfc/TOA fluxes are
        # only recomputed every Nth step and ride carry_aux between updates, so
        # a zarr restart would drop them and reset the radiation phase — the
        # first post-restart segment would run with zero held fluxes until the
        # next update, branching the trajectory. With rad_update_steps == 1 the
        # held fields are recomputed every step, so dropping them is harmless
        # (audit 2026-07-17).
        if (
            backend == "zarr"
            and int(getattr(self.config, "rad_update_steps", 1)) > 1
            and isinstance(self._carry_aux, dict)
            and any(k.startswith("held_") for k in self._carry_aux)
        ):
            raise ValueError(
                "checkpoint_format='zarr' cannot persist the held radiation "
                "fluxes (held_dT_rad/held_*_sfc/held_*_toa) used by a "
                "radiation cadence (rad_update_steps="
                f"{int(getattr(self.config, 'rad_update_steps', 1))}) — they "
                "ride carry_aux, which the zarr backend does not round-trip, "
                "so a restart would reset the radiation phase (not bit-exact). "
                "Use checkpoint_format='npz' for rad_update_steps>1 runs."
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

        fv3_duo dispatches HERE, before any generic decode: the duo
        bundle has its own schema (``fv3duo_ckpt_v1``), and the loader
        is schema-gated so a foreign (cube/lat-lon/MPAS) checkpoint is
        refused loudly instead of dying on an unrelated shape error
        deeper in a decode (codex 2026-08-18 MAJOR, kept as a gate).
        """
        if self.config.dycore.discretization == "fv3_duo":
            return self._load_fv3_duo_checkpoint(Path(path))
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
            # Vertical LEVEL-POSITION guard.  The shape guards below only see
            # nlev, and nlev no longer identifies the σ grid: a uniform L30 and
            # a tropopause-refined L30 (grid.tropopause_refine) are the same
            # shape.  Restarting one on the other silently reinterprets every
            # T/q profile on the wrong pressures — a silent physics error, not
            # a crash.  Compares the (A, B) half-level pair (see the save side
            # for why A+B alone is blind to the hybrid transition exponent).
            # ``atol`` is a float32-STORAGE allowance, not a physics one: the
            # coordinate is regenerated deterministically from the config, so
            # an fp32-vs-fp64 run differs by at most one float32 ulp near
            # B = 1, i.e. 6e-8 — 5e-7 leaves ~8 ulps of margin while still
            # rejecting any real level move (>= 1e-3 in σ, 2000x larger).
            # Absent on pre-guard checkpoints (skip, stay backward-compatible).
            if "meta_vgrid" in getattr(d, "files", ()):
                _ck_vg = np.asarray(d["meta_vgrid"], dtype=np.float64)
                _A = getattr(self.sigma, "A_half", None)
                _cur_vg = np.stack([
                    (np.zeros(self.sigma.n_levels + 1) if _A is None
                     else np.asarray(_A, dtype=np.float64)),
                    np.asarray(getattr(self.sigma, "B_half",
                                       self.sigma.sigma_half),
                               dtype=np.float64),
                ])
                if (_ck_vg.shape != _cur_vg.shape
                        or not np.allclose(_ck_vg, _cur_vg, rtol=0.0,
                                           atol=5e-7)):
                    raise ValueError(
                        f"MPAS checkpoint {path.name} was written on a "
                        f"DIFFERENT vertical grid: checkpoint (A,B)_half[:, :3]"
                        f"={_ck_vg[:, :3]} vs current {_cur_vg[:, :3]} "
                        f"(nlev {_ck_vg.shape[-1] - 1} vs "
                        f"{_cur_vg.shape[-1] - 1}). Restarting would "
                        "reinterpret every profile on the wrong levels. "
                        "Rebuild with the same --nlev / --vertical-coord / "
                        "--p-top / --stretching / --tropopause-refine.")
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
                _validate_number_convention(d, _names)
                self.state = self.state._replace(tracers={
                    _k: Field(
                        data=(scatter_to_local(
                                  jnp.asarray(d[f"trc_{_k}"]), part, "cell")
                              if _mpi else jnp.asarray(d[f"trc_{_k}"])),
                        name=_k, dims=("nCells", "nlev"),
                        units=("1/kg" if _k in ("N_c", "N_r", "N_i")
                               else "kg/kg"))
                    for _k in _names
                })
            # Multilayer (Richards) land columns (MPAS port): stage the
            # ``land_ml_<field>`` arrays into carry_aux and reuse the shared
            # fail-loud restore (#730 exact-field-set contract).  Under MPI the
            # checkpoint holds the GLOBAL columns (``save_checkpoint`` gathers
            # them with the other collectives) and
            # ``_restore_land_ml_from_carry_aux`` cuts each field to this rank
            # before its shape check, so the distributed restart round-trips
            # (#1321).  It used to refuse here.
            _lml_keys = [k for k in d.files if k.startswith("land_ml_")]
            if _lml_keys:
                if not isinstance(self._carry_aux, dict):
                    self._carry_aux = {}
                for _k in _lml_keys:
                    self._carry_aux[_k] = jnp.asarray(d[_k])
                # Stage the soil column too (own namespace, not a state field)
                # so the shared restore refuses a state from another column.
                if "land_soil_dz" in d.files:
                    self._carry_aux["land_soil_dz"] = np.asarray(d["land_soil_dz"])
                self._restore_land_ml_from_carry_aux()
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
            # Ice skin: same stale-persistence rule — drop any prior staging,
            # then stage this checkpoint's skin (if present) for the
            # _run_mpas seed overlay.  The checkpoint is GLOBAL, so under
            # cell-partition MPI scatter it to this rank's band first (mirror
            # of the save-side gather); _run_mpas then shape-checks it against
            # the LOCAL cell count like every other staged field.
            self._carry_aux.pop("ice_T_skin", None)
            self._ice_T_skin = None
            if "ice_T_skin" in d.files:
                _skin_ck = np.asarray(d["ice_T_skin"]).reshape(-1)
                if _mpi:
                    if _skin_ck.shape[0] != part.nCells_global:
                        raise ValueError(
                            f"MPAS checkpoint {path.name} ice_T_skin length "
                            f"{_skin_ck.shape[0]} != global mesh "
                            f"({part.nCells_global},); rebuild with the same "
                            f"--resolution.")
                    _skin_ck = scatter_to_local(
                        jnp.asarray(_skin_ck), part, "cell")
                self._carry_aux["ice_T_skin"] = np.asarray(_skin_ck)
            # #1353 partial-interval CMOR flux sums: same stale-persistence
            # rule — drop prior staging, then stage this checkpoint's
            # payload for the _run_mpas accumulator restore.  WHITELISTED
            # key names (codex-3): an unexpected ``cmor_flux*`` key is a
            # corrupt/foreign payload, not something to forward blindly.
            _flux_keys = set(_MPASSfcFluxAccum.checkpoint_keys())
            for _stale in [k for k in self._carry_aux
                           if k.startswith("cmor_flux")]:
                del self._carry_aux[_stale]
            # ...and DROP the live accumulator from any prior run on this
            # reused driver: save_checkpoint prefers a live accumulator
            # over the staged payload, so a stale one would overwrite the
            # interval just loaded (codex-4, reproduced).  _run_mpas
            # rebuilds + restores it from the staging below.
            self._mpas_sfc_accum = None
            for _k in d.files:
                if _k in _flux_keys:
                    self._carry_aux[_k] = np.asarray(d[_k])
                elif _k.startswith("cmor_flux"):
                    raise ValueError(
                        f"checkpoint has unrecognised CMOR flux key {_k!r} "
                        f"(expected one of {sorted(_flux_keys)}) — refusing "
                        f"to resume from an unknown accumulator payload.")
            # ...and clear the SAVE channel (``_mpas_phys_state``, read by
            # save_checkpoint) so a stale carry from a PRIOR run on a
            # reused driver cannot leak.  It is left None until a run
            # validates + overlays this checkpoint's staged carry
            # (_run_mpas); a save before then is refused (see
            # save_checkpoint) rather than emitting an unvalidated carry.
            self._mpas_phys_state = None
            from legoesm.atmosphere.physics.physics_state import (
                PHYSSTATE_INPUT_FIELDS,
            )
            _ps_keys = [k for k in d.files if k.startswith("physstate_")]
            if _ps_keys:
                for _k in _ps_keys:
                    _name = _k[len("physstate_"):]
                    if _name.startswith("meta_"):
                        # Plain-string metadata (scheme tag) — no jnp,
                        # no scatter.
                        self._carry_aux[_k] = str(d[_k])
                        continue
                    # Per-step INPUT fields are never restored (recomputed each
                    # step).  Drop them at the LOAD boundary — BEFORE any
                    # ``d[_k]`` access / jnp.asarray / MPI scatter — so a legacy
                    # checkpoint that wrote one (e.g. a None-derived object
                    # array, or a wrong-shaped concrete input) cannot fail the
                    # load; the fresh seed's None is correct (codex r2).
                    if _name in PHYSSTATE_INPUT_FIELDS:
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
                # #1310: restore the anchored mass-fixer target so the resumed
                # run pins to the SAME mass as the straight run (see the save
                # branch).  set_target_mass pre-seeds it, so the model's
                # first-step _maybe_snapshot_target_mass (which only fires when
                # _target_mass is None) does NOT re-anchor to the loaded state.
                if "target_mass" in (d.files if hasattr(d, "files") else d) \
                        and hasattr(self.model, "set_target_mass"):
                    import jax.numpy as _jnp
                    self.model.set_target_mass(
                        _jnp.asarray(d["target_mass"], dtype=_jnp.float64))
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
            # The raise above means a lat-lon MPI restart never carries a
            # multilayer land state, so its soil-column stamp has nothing to
            # describe.  Drop it rather than let it ride into the next re-save
            # and label THAT state with a column it did not come from.
            if isinstance(self._carry_aux, dict):
                self._carry_aux.pop("land_soil_dz", None)
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
            Called at each SEGMENT boundary with ``(driver, day,
            dt_segment)`` for coupled-model integration, where
            ``dt_segment`` (= ``seg_steps * dt``) is the elapsed time since
            the previous call.  In the compiled lane the segment length is
            a divisor of the diagnostic interval, so this fires at least as
            often as diagnostics -- never less (issue F4).

        Returns
        -------
        str
            Run status ("COMPLETED" or "BLOWUP at day ...").
        """
        refuse_cap_floor_on_fv(self.config)
        # SW is not runnable via ModelDriver — reject at the public entry even
        # if a caller reached run() without setup() (codex M2 review).
        self._reject_shallow_water_unrunnable()
        # CLUBB cloud-fraction -> radiation carry is threaded through the per-step
        # rollout (``_run_per_step``) AND the single-device compiled segment
        # rollout (``_run_compiled`` -> the fused ``_make_single_step`` +
        # ``SegmentCarry.cloud_fraction``).  The MULTI-DEVICE / distinct-dycore
        # rollouts (MPAS, spectral, lat-lon-SPMD, tiled-cube — which route to
        # operator-split / sharded / tiled carry structures) do NOT yet thread it,
        # so an enabled feature there would silently no-op.  Those are all selected
        # by their own predicate BEFORE the ``compiled`` branch, so refuse LOUDLY
        # on them (dispatch-hardening) — but ``compiled`` alone is now fine (the
        # fused single-device path carries it).  The lat-lon-SPMD operator-split
        # lane is reached only under enable_latlon_spmd (covered below); the tiled
        # operator-split lane only under cube multi-device tiling (covered below).
        if getattr(self.config, "use_clubb_cloud_fraction", False):
            _grid = self.config.grid.grid_type
            _disc = self.config.dycore.discretization
            _latlon_spmd = getattr(self.config, "enable_latlon_spmd", False)
            _tiled_cube = (
                _grid == "cubed_sphere"
                and self._device_config is not None
                and getattr(self._device_config, "mesh", None) is not None
                and tuple(getattr(self._device_config, "tiling", (1, 1))) != (1, 1)
            )
            if _disc == "spectral" or _latlon_spmd or _tiled_cube:
                raise NotImplementedError(
                    "use_clubb_cloud_fraction is wired through the single-device "
                    "per-step AND fused-compiled rollouts and the MPAS lane, but "
                    "NOT the multi-device / spectral ones: it needs "
                    "discretization != "
                    "'spectral', enable_latlon_spmd=False, and no multi-device "
                    "cube tiling.  Got "
                    f"compiled={compiled}, grid={_grid!r}, discretization="
                    f"{_disc!r}, latlon_spmd={_latlon_spmd}, tiled_cube="
                    f"{_tiled_cube}.  Those rollouts do not yet thread the "
                    "cloud-fraction carry and would silently ignore the flag.  "
                    "Re-run single-device (per-step or compiled) on a latlon / "
                    "cubed-sphere hydrostatic config, or thread the cloud-fraction "
                    "carry through the operator-split / sharded / tiled steps first."
                )
        self._segment_callback = segment_callback
        # #1028: promote the cube's prognostic winds to FV3 D staggering ONCE,
        # before the time loop, so the run carries them between steps instead
        # of projecting cell-centre -> corner -> cell-centre every step.
        self._enter_persistent_dgrid()
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
            # fv3_duo steps its own six-face bundled pytree — keyed on the
            # DISCRETIZATION (its grid_type is the shared "cubed_sphere"),
            # so it must dispatch before every grid-keyed branch below.
            if strict_sed_abort_requested(
                    self.config, getattr(self.physics, "micro_config", None)):
                # every driver, not just run_amip: a deck that turns the
                # strict abort on must not reach day 2 of a GPU run before
                # discovering its host callback has nowhere to land
                # (GLM 2026-09-22)
                require_cpu_for_strict_sedimentation()
            if self.config.dycore.discretization == "fv3_duo":
                warn_sed_substeps_unreported(
                    self.config, "fv3_duo",
                    getattr(self.physics, "micro_config", None))
                status = self._run_fv3_duo(start_step, start_day)
            elif self.config.grid.grid_type == "mpas":
                status = self._run_mpas(start_step, start_day)
            elif self.config.dycore.discretization == "spectral":
                warn_sed_substeps_unreported(
                    self.config, "spectral",
                    getattr(self.physics, "micro_config", None))
                status = self._run_spectral(start_step, start_day)
            elif (self.config.enable_latlon_spmd
                    and self.config.grid.grid_type == "latlon"):
                # Single-process multi-device lat-band SPMD (A1): a dedicated
                # segment loop over the validated run_atm_latlon_spmd, distinct
                # from the jitted compiled_segments scan (zero surgical risk to
                # the shared hot loop).
                # NOTE: the coupled-lane refusal is NOT here.  This branch
                # cannot tell the two SPMD sub-lanes apart, and the
                # operator-split one DOES stash the held surface fields; the
                # refusal now lives in _run_compiled_latlon_spmd, on the
                # stateless sub-lane that genuinely produces none.
                warn_sed_substeps_unreported(
                    self.config, "lat-band SPMD",
                    getattr(self.physics, "micro_config", None))
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
                self._reject_coupled_lane(
                    "sub-face-tiled cube SPMD (6*kt^2 > 6 devices)",
                    "a blocked tiled / tiled operator-split envelope",
                    "Run the coupled case on <=6 devices (n_devices<=6, so "
                    "tiling==(1,1)) -- the compiled lane stashes the held "
                    "fields; this one does not.")
                warn_sed_substeps_unreported(
                    self.config, "tiled cube SPMD",
                    getattr(self.physics, "micro_config", None))
                status = self._run_tiled_cube_spmd(start_step, start_day)
            elif compiled:
                warn_sed_substeps_unreported(
                    self.config, "compiled segments",
                    getattr(self.physics, "micro_config", None))
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
        ``mean_T`` is PRESSURE-WEIGHTED (sum T*dp / sum dp; equal cell
        weight — areaCell weighting is a deferred refinement on the
        quasi-uniform SCVT).
        """
        from mpi4py import MPI as _MPI
        vl = self._voronoi_layout
        om_c = vl.owned_mask_cells          # (n_local_cells,) bool
        om_e = vl.owned_mask_edges          # (n_local_edges,) bool
        T_owned = jnp.where(om_c[:, None], T_data, 0.0)
        ps_owned = jnp.where(om_c, p_s_data, 0.0)
        absu_owned = jnp.where(om_e[:, None], jnp.abs(u_data), 0.0)
        T_min_l = jnp.min(jnp.where(om_c[:, None], T_data, jnp.inf))
        T_max_l = jnp.max(jnp.where(om_c[:, None], T_data, -jnp.inf))
        finite_l = jnp.all(jnp.isfinite(T_owned))
        cwv_sum_l = (jnp.sum(jnp.where(om_c, cwv_field, 0.0))
                     if cwv_field is not None else jnp.asarray(0.0))
        # Pressure-weighted mean T (sum T*dp / sum dp; equal cell weight —
        # the quasi-uniform SCVT makes areaCell weighting a negligible
        # refinement, and the pre-fix convention was equal-cell too), owned
        # cells only: an unweighted level mean is coordinate-dependent
        # (stretched hybrid grids overstate it by ~+9 K vs sigma on the same
        # state; quantified 2026-07-23), which made hybrid-vs-sigma
        # stability curves incomparable.  Mirrors the serial day-line
        # diagnostic.  The where() wraps the PRODUCT so a non-finite halo
        # p_s cannot leak NaN through 0*NaN (codex F-B6).
        _p_half_d = self.sigma.pressure_at_half(p_s_data)
        _dp_d = _p_half_d[..., 1:] - _p_half_d[..., :-1]
        Tdp_sum_l = jnp.sum(jnp.where(om_c[:, None], T_data * _dp_d, 0.0))
        dp_sum_l = jnp.sum(jnp.where(om_c[:, None], _dp_d, 0.0))
        # One device→host transfer for all local reductions.
        _loc = np.asarray(jnp.stack([
            Tdp_sum_l, jnp.sum(ps_owned), jnp.max(absu_owned),
            T_min_l, T_max_l, finite_l.astype(T_data.dtype),
            cwv_sum_l.astype(T_data.dtype), dp_sum_l,
        ]))
        comm = _MPI.COMM_WORLD
        # THREE batched buffer allreduces instead of eight scalar pickle
        # rounds (each scalar ``comm.allreduce`` is its own latency-bound
        # collective; at multi-node rank counts the per-diag latency is
        # 8x a single round for no reason).  The finite flag (as a float)
        # rides the MIN batch: all-ranks-finite  <=>  min(finite) == 1.
        _sums = np.array([_loc[0], _loc[1], _loc[6], _loc[7]],
                         dtype=np.float64)
        _maxs = np.array([_loc[2], _loc[4]], dtype=np.float64)
        _mins = np.array([_loc[3], _loc[5]], dtype=np.float64)
        comm.Allreduce(_MPI.IN_PLACE, _sums, op=_MPI.SUM)
        comm.Allreduce(_MPI.IN_PLACE, _maxs, op=_MPI.MAX)
        comm.Allreduce(_MPI.IN_PLACE, _mins, op=_MPI.MIN)
        g_sum_Tdp, g_sum_ps, g_sum_cwv, g_sum_dp = (float(v) for v in _sums)
        g_max_u, g_T_max = (float(v) for v in _maxs)
        g_T_min, g_finite_min = (float(v) for v in _mins)
        g_finite = bool(g_finite_min > 0.5)
        # Owned-cell count is partition-static: allreduce ONCE and cache.
        if self._mpas_g_n_cells is None:
            self._mpas_g_n_cells = comm.allreduce(
                int(vl.partition.n_owned_cells), op=_MPI.SUM)
        g_n_cells = self._mpas_g_n_cells
        mean_T = g_sum_Tdp / max(g_sum_dp, 1e-30)
        mean_ps = g_sum_ps / g_n_cells
        cwv = (g_sum_cwv / g_n_cells) if cwv_field is not None else float("nan")
        return mean_T, mean_ps, g_max_u, g_T_min, g_T_max, g_finite, cwv

    def _mpas_cmip_feed_enabled(self, diag) -> tuple[bool, bool]:
        """Decide whether the per-interval MPAS CMOR accumulator feed runs.

        Returns ``(feed_on, wants_cmip)``:

        * ``wants_cmip`` — the collector actually holds a CMOR spatial or a
          zonal monthly accumulator (``cmip_output`` / ``monthly_means`` on).
        * ``feed_on`` — ``wants_cmip`` AND the layout is one this driver knows
          how to feed:

          - SERIAL (``_voronoi_layout is None``) or a 1-rank Voronoi layout
            that owns the whole mesh — the collector's regrid weights ARE the
            global weights; fed directly (byte-identical path).
          - a MULTI-rank Voronoi CELL PARTITION — fed via the owned-cell ->
            global gather in :meth:`_feed_mpas_cmip_multirank`, with the
            collector holding GLOBAL regrid weights and GLOBAL ``fx`` fields
            (``_create_diagnostics`` under ``_is_mpas_cell_partitioned``).

        The cell-partition arm is what #1517 adds; it used to return
        ``feed_on=False``, because a per-rank feed would have binned rank-local
        owned+halo cells through LOCAL weights into the GLOBAL lat-lon boxes.

        ``feed_safe`` is written as a POSITIVE enumeration of the layouts that
        have a feed, not as "not the one bad case".  Spelled the other way it
        would be a tautology — ``layout is None or world <= 1 or (layout and
        world > 1)`` covers everything — and :meth:`_require_mpas_cmip_feed_supported`,
        the #1545 launch-time refusal, would be unreachable dead code rather
        than the tripwire it is meant to be.  As written, a MULTI-rank run with
        no Voronoi layout (some future parallel mode reaching ``_run_mpas``)
        is NOT feedable and still refuses at launch instead of silently
        writing empty CMOR files.  No such layout exists today, so the refusal
        does not fire in tree; that is the point of a tripwire.
        """
        _world = getattr(self, "_mpi_world_size", 1) or 1
        feed_safe = (
            # Serial: no layout and no peers.
            (self._voronoi_layout is None and _world <= 1)
            # A 1-rank Voronoi layout owns the whole mesh — its regrid weights
            # ARE the global weights.
            or (self._voronoi_layout is not None and _world <= 1)
            # Multi-rank Voronoi cell partition: the owned-cell gather (#1517).
            or _is_mpas_cell_partitioned(self)
        )
        wants_cmip = diag is not None and (
            getattr(diag, "_spatial_monthly", None) is not None
            or (getattr(diag, "monthly_means", False)
                and getattr(diag, "monthly_accum", None) is not None)
        )
        return (feed_safe and wants_cmip, wants_cmip)

    def _require_mpas_cmip_feed_supported(self, feed_on: bool,
                                          wants_cmip: bool) -> None:
        """Refuse a run that would write EMPTY CMOR output (#1545).

        A run whose layout has no CMOR feed (see
        :meth:`_mpas_cmip_feed_enabled`) but which asked for
        ``cmip_output``/``monthly_means`` completes normally and writes CMOR
        files containing nothing.  That used to be a rank-0 log warning — one
        line in a long log — so the cost was discovered only after the
        GPU-hours were spent.  A request for output the lane cannot produce is
        a launch error, not a note.

        WHAT STILL REACHES THIS, now that #1517 landed: the multi-rank Voronoi
        CELL PARTITION is fed (owned-cell gather), and serial / 1-rank layouts
        always were, so none of them arrive here any more.  What remains is a
        multi-rank run with NO Voronoi layout — a parallel mode that reaches
        ``_run_mpas`` without a cell partition.  None exists today; this is the
        tripwire for the next one, so that it fails at launch instead of
        silently reproducing the empty-CMOR defect.

        Raised on EVERY rank, deliberately NOT rank-0-gated: ``feed_on`` and
        ``wants_cmip`` are config/layout-derived and identical everywhere, so a
        rank-0-only raise would kill rank 0 and hang the rest at the next
        collective.

        ``LEGOESM_ALLOW_EMPTY_CMOR=1`` (exact value, matching the repo's other
        ``LEGOESM_ALLOW_*`` escape hatches) downgrades it to the old warning.
        The environment is the ONE input here that is genuinely per-process —
        an MPMD launcher can export it to some ranks and not others — so it is
        bcast from rank 0 before anyone acts on it (the repo's established
        status-bcast idiom).  Without that, a split environment sends some
        ranks onward and raises on the others: a hang, which is strictly worse
        than the empty output this replaces (pre-merge codex).

        The gather this was waiting on landed (#1517), so the cell-partition
        lane is no longer CMOR-less.
        """
        if not wants_cmip or feed_on:
            return
        world = getattr(self, "_mpi_world_size", 1)
        allow = os.environ.get("LEGOESM_ALLOW_EMPTY_CMOR") == "1"
        # ``COMM_WORLD``/``root=0`` matches the driver's established
        # status-bcast (the check_stability error bcast).  The gate is precise
        # rather than merely sufficient: reaching this line at all requires
        # ``feed_on`` False with ``wants_cmip`` True, which per
        # _mpas_cmip_feed_enabled means a MULTI-rank run with NO Voronoi
        # layout — the cell partition is fed via the gather since #1517.
        if getattr(self, "_mpi_rank", None) is not None and world > 1:
            from mpi4py import MPI
            allow = MPI.COMM_WORLD.bcast(allow, root=0)
        if allow:
            if getattr(self, "_mpi_rank", 0) == 0:
                logger.warning(
                    "  CMOR output requested on a %d-rank run whose layout "
                    "has no CMOR feed, and LEGOESM_ALLOW_EMPTY_CMOR=1 is set: "
                    "the monthly/daily CMOR accumulators WILL STAY EMPTY. The "
                    "run continues because you asked it to.", world)
            return
        raise NotImplementedError(
            f"CMOR output was requested (cmip_output / monthly_means on) on a "
            f"{world}-rank run whose layout has NO CMOR feed. The multi-rank "
            f"Voronoi CELL PARTITION is fed via the owned-cell gather "
            f"(#1517), and serial / 1-rank layouts feed directly — so this is "
            f"some other multi-rank layout, for which each rank holds only "
            f"its own cells and its own regrid weights and feeding it would "
            f"bin one rank's subdomain into the global lat-lon boxes. The run "
            f"would otherwise finish and write CMOR files containing NOTHING "
            f"(#1545).\n"
            f"  Options, in order of preference: (1) run single-rank, or on "
            f"the Voronoi cell partition, for CMOR spatial output; (2) turn "
            f"CMOR output off (cmip_output/monthly_means) if you only want "
            f"checkpoints and log diagnostics; (3) set "
            f"LEGOESM_ALLOW_EMPTY_CMOR=1 to proceed anyway and accept empty "
            f"CMOR files.")

    def _feed_mpas_cmip_accumulators(self, day: float, *, state_only=False,
                                     flux_only=False) -> None:
        """Feed the CMOR monthly/daily/zonal accumulators from the current
        MPAS (Voronoi) state at a diagnostic interval.

        The lean MPAS loop never called :meth:`DiagnosticCollector.collect`
        (the cube/lat-lon spatial-accumulation path), so the CMOR ``Amon`` /
        ``day`` accumulators the collector constructs stayed EMPTY
        (``call_counts: []`` / ``max_count_ever: 0``) even though the restart
        sidecar dutifully SAVED them each checkpoint.  This bridges that gap:
        it reconstructs geographic cell winds (edge-normal ``state.u`` -> cell
        ``(u_east, v_north)`` via the Perot reconstruction), reads the
        surface-precip export the physics stashes on the dynamics object, and
        hands the native cell arrays to
        :meth:`DiagnosticCollector.feed_cmip_accumulators_native`, which
        regrids to the CMOR lat-lon grid (the collector's Voronoi IDW weights)
        and bins the zonal means.

        SERIAL / 1-rank: the rank-local cell arrays ARE the global ones and the
        collector's regrid weights ARE the global weights, so
        :meth:`_mpas_cmip_native_kwargs` is fed straight through — this branch
        is byte-identical to the pre-multi-rank code.

        MULTI-rank cell partition: a per-rank feed would bin this rank's
        owned+halo cells through LOCAL regrid weights into the global lat-lon
        boxes (a rank-local, wrong "global" monthly mean), so the work is
        delegated to :meth:`_feed_mpas_cmip_multirank`, which GATHERS the
        OWNED cells to their global slots on rank 0 (root ``gather``, not an
        allgather — no peer needs the global field) and feeds there with
        global weights.  The lightweight timeseries is a true global either way via
        :meth:`_mpas_global_diag`.

        Fully guarded: a diagnostic-feed failure is LOUD but never aborts the
        run (the host-side accumulation cannot perturb the prognostic state).
        Under MPI the guarding is COLLECTIVE-SAFE — see
        :meth:`_feed_mpas_cmip_multirank`.
        """
        diag = getattr(self, "diagnostics", None)
        if diag is None:
            return
        try:
            if _is_mpas_cell_partitioned(self):
                self._feed_mpas_cmip_multirank(
                    day, diag, self._voronoi_layout, state_only=state_only,
                    flux_only=flux_only)
            else:
                if state_only:
                    _kw = self._mpas_cmip_native_kwargs(day, diag, state_only=True)
                    # Associate the right-end hourly sample with the hour it covers.
                    sample_bin = (np.floor(day * 24.0 + 1e-9) - 0.5) / 24.0
                    diag.feed_cmip_accumulators_native(sample_bin, **_kw)
                    diag.feed_daily_extremes_native(sample_bin, tas=_kw["tas"])
                else:
                    _kw = self._mpas_cmip_native_kwargs(day, diag)
                    diag.feed_cmip_accumulators_native(
                        day, include_state=not flux_only, **_kw)
                    self._feed_mpas_moisture_budget(day, diag, _kw)
        except Exception as exc:  # pragma: no cover - defensive diag guard
            logger.error(
                "  CMOR accumulator feed FAILED at day %.2f (run continues; "
                "the monthly/daily CMOR means for this interval are lost): %s",
                day, exc)
        finally:
            # Start the next interval's flux averaging fresh (also on a
            # failed feed — a stale sum would smear across intervals).  The
            # new window begins at THIS feed's day.
            _acc = getattr(self, "_mpas_sfc_accum", None)
            if _acc is not None and not state_only:
                _acc.reset(window_start_day=day)

    def _feed_mpas_moisture_budget(self, day: float, diag, kw: dict) -> None:
        """Record the atmospheric water-budget closure E - P - dW/dt.

        The model has always carried this tracker, and this lane has never fed
        it: the tracker is updated inside ``DiagnosticCollector.collect``, which
        the MPAS run loop does not call, so every run of this campaign published
        a blank moisture residual. A ~0.4 mm/day gap between the reported global
        evaporation and rainfall therefore sat unexamined for months. An
        instrument that is not wired is not a check.

        Fed from the SAME window-mean ``precip`` and ``hfls`` the CMOR output
        publishes, which is the tracker's own stated contract: the residual then
        closes against the numbers a reader can see in ``pr`` and ``hfls``,
        rather than against a second, privately-averaged pair that could differ
        for reasons nobody could trace.

        SERIAL ONLY, deliberately. The tracker takes a plain area-weighted mean,
        which under a cell partition would be rank-local and count halo cells
        twice -- a confidently wrong global number, which is worse than none.
        The multi-rank path needs the owned-mask-and-allreduce treatment
        ``_mpas_global_diag`` already does, and says so once rather than
        publishing rubbish.
        """
        if self._voronoi_layout is not None:
            if not getattr(self, "_logged_moisture_budget_mpi", False):
                logger.info(
                    "  moisture-budget closure NOT recorded under the cell "
                    "partition (the tracker's area mean is rank-local); serial "
                    "runs publish it.")
                self._logged_moisture_budget_mpi = True
            return
        # The CMOR slot getter falls back to an INSTANTANEOUS diagnostic when a
        # slot has no accumulated samples.  Mixing a mean rainfall with an
        # instantaneous evaporation (or the reverse) manufactures an imbalance
        # out of nothing, so this closure takes the window means or nothing.
        _acc = getattr(self, "_mpas_sfc_accum", None)
        if _acc is None or not _acc.has_samples() or not _acc.is_complete():
            return
        precip = kw.get("precip")
        hfls = kw.get("hfls")
        tracers = self.state.tracers
        if (precip is None or hfls is None or tracers is None
                or "q_v" not in tracers):
            return          # dry run, or a window whose fluxes were withheld
        area = getattr(self.grid, "areaCell", None)
        _p_s = self.state.p_s.data
        _p_half = self.sigma.pressure_at_half(_p_s)
        diag.moisture_tracker.update(
            tracers["q_v"].data, _p_s, self.sigma.dsigma,
            precip, hfls,
            elapsed_seconds=float(day) * 86400.0,
            area_weights=(None if area is None
                          else jnp.asarray(area).reshape(-1)),
            # Hybrid coordinates make ``p_s * dsigma`` wrong for a bottom-heavy
            # tracer over terrain; the half-level difference is right for either
            # coordinate.
            dp=_p_half[..., 1:] - _p_half[..., :-1],
        )

    def _mpas_surface_temperatures(self, day, diag, u_east, v_north):
        """Shared native whole-cell tas and skin ts for both sample cadences."""
        state = self.state
        q_v = (state.tracers["q_v"].data
               if state.tracers is not None and "q_v" in state.tracers else None)
        # 2 m ``tas`` via MOST similarity when prescribed sst/sic are on
        # this path (``get_sst_sic`` set for a radiation+SST run) — matches
        # the cube-path collect() ``tas`` instead of a bare lowest-level
        # proxy.  Uses the RECONSTRUCTED cell winds (``state.u`` is
        # edge-normal on MPAS, not cell-collocated).  A failure falls back
        # to the lowest model level (logged once) so a tas-only glitch never
        # drops the whole CMOR feed.
        tas = None
        ts = None
        _get_sst_sic = getattr(self, "get_sst_sic", None)
        if _get_sst_sic is not None:
            try:
                _sst, _sic = _get_sst_sic(day)
                _sst = jnp.asarray(_sst).reshape(-1)
                _sic = jnp.asarray(_sic).reshape(-1)
                # Report tas off the SAME ice surface the radiation +
                # turbulence saw: the per-cell prognostic skin when the
                # feature is on, else the constant T_ice.  Otherwise the
                # scorecard's 2 m extrapolation uses a 271.35 K ice surface
                # while the model cooled the skin (codex-1 finding 3).
                _tas_ice = getattr(self.config, "T_ice", None)
                if (getattr(self.config, "mpas_ice_skin_prognostic", False)
                        and getattr(self, "_ice_T_skin", None) is not None):
                    _tas_ice = self._ice_T_skin
                # Give the diagnostic the LAND surface too where the
                # interactive tile has produced one, else its profile is
                # anchored on the ocean/ice skin over land as well and the
                # published land tas is not the model's land at all.
                _tas_land_T = getattr(self, "_land_T_skin_last", None)
                _tas_land_q = getattr(self, "_land_qsfc_last", None)
                _tas_f_land = getattr(self, "_f_land", None)
                _tas_kw = {}
                if _tas_land_T is not None and _tas_f_land is not None:
                    _tas_kw = dict(
                        T_land=jnp.asarray(_tas_land_T).reshape(-1),
                        q_land=(None if _tas_land_q is None
                                else jnp.asarray(_tas_land_q).reshape(-1)),
                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1))
                from legoesm.forcing.surface_utils import blend_surface_temperature
                ts = blend_surface_temperature(_sst, _sic, _tas_ice)
                if _tas_land_T is not None and _tas_f_land is not None:
                    f_land = jnp.clip(jnp.asarray(_tas_f_land).reshape(-1), 0.0, 1.0)
                    ts = (f_land * jnp.asarray(_tas_land_T).reshape(-1)
                          + (1.0 - f_land) * ts)
                tas = diag._tas_2m(
                    state, q_v, _sst, _sic, _tas_ice,
                    u_low=u_east[..., -1], v_low=v_north[..., -1], **_tas_kw)
            except Exception as exc:
                if not getattr(self, "_logged_tas2m_fallback", False):
                    logger.warning(
                        "  CMOR tas: 2 m MOST calc failed (%s); using the "
                        "lowest model level as the tas proxy.", exc)
                    self._logged_tas2m_fallback = True
                tas = None
        return (state.T.data[..., -1] if tas is None else tas), ts

    def _mpas_cmip_native_kwargs(self, day: float, diag,
                                 u_override=None, *, state_only=False) -> dict:
        """Build the NATIVE (rank-local) cell-field kwargs for
        :meth:`DiagnosticCollector.feed_cmip_accumulators_native`.

        PURE and COLLECTIVE-FREE: reads ``self.state`` / the physics flux
        export and returns host/device arrays whose leading axis is the
        caller's cell axis (``n_local_cells`` under a Voronoi partition,
        ``nCells`` serial).  Every value may be ``None`` (the collector skips
        absent fields); ``flux_interval_days`` is the one scalar entry.

        Split out of :meth:`_feed_mpas_cmip_accumulators` so the multi-rank
        gather can reuse the SAME field construction — and so this work stays
        strictly outside any MPI collective (a rank that fails here must not
        leave its peers blocked; see :meth:`_feed_mpas_cmip_multirank`).

        ``u_override`` supplies an edge field whose HALO edges have already
        been refreshed.  It exists because this method must stay
        collective-free while the Perot reconstruction below genuinely needs
        valid halo edges: under a cell partition an OWNED cell on the cut is
        RINGED by halo edges, so a stale one corrupts that cell's ``ua``/``va``
        — silently, and only along the cut.

        The clear case is column-local physics: ``_phys_col_local``, set from
        ``physics_fn._column_local`` (today only the idealized Kessler MPAS
        forcing sets it), makes ``make_voronoi_mpi_step`` skip the pre-physics
        state exchange, and there is no post-physics one, so halo edges keep
        ``old_halo + dt*local_tendency``.  The full AMIP physics does NOT set
        that attribute and therefore takes the exchanging path.  The exchange
        is done unconditionally anyway: it costs one collective per DIAGNOSTIC
        INTERVAL and removes the reconstruction's dependence on which physics
        path ran, rather than leaving a correctness argument that has to be
        re-derived whenever that gate moves.

        ``None`` (serial / single-rank) uses ``state.u`` directly and is
        byte-identical.
        """
        from legoesm.grids.voronoi import reconstruct_cell_velocity
        state = self.state
        # Geographic cell-centre winds from the edge-normal velocity.
        u_edges = state.u.data if u_override is None else u_override
        u_east, v_north = reconstruct_cell_velocity(u_edges, self.grid)
        tas, ts = ModelDriver._mpas_surface_temperatures(
            self, day, diag, u_east, v_north)

        # Pressure vertical velocity, for the subsidence the scorecard could
        # previously only guess at.  Built from the SAME halo-refreshed edge
        # field, through the pair the column-forcing extractor already
        # composes on this mesh: cell divergence of the edge-normal wind, then
        # the coordinate-aware continuity integral.  Not the dycore's own
        # omega -- the dycore closes continuity in FLUX form div(u*dp) while
        # this rebuilds it from the advective div(v)*dp, and the two differ
        # wherever the surface-pressure gradient is large (see the comment in
        # primitive_eq_mpas beside the mass-flux branch).  It IS a closed
        # continuity solve, which the monthly-mean-wind estimate it replaces
        # was not: that one returned a global mean of -6 hPa/day where
        # continuity requires ~0, and amplitudes ~30x ERA5.
        wap = None
        _sigma = getattr(self, "sigma", None)
        if _sigma is None:
            # Absent only on a partially built driver. Say so: the enclosing
            # feed swallows exceptions, so a raise here would drop the WHOLE
            # CMOR stream silently rather than just this field.
            logger.warning(
                "no vertical coordinate on the driver: publishing no wap, so "
                "subsidence cannot be scored for this run")
        else:
            from legoesm.atmosphere.forcing.column_large_scale_extract import (
                omega_from_divergence)
            from legoesm.core.operators_voronoi import divergence_cell_3d
            _u_edge = state.u.data if u_override is None else u_override
            wap = omega_from_divergence(
                divergence_cell_3d(jnp.asarray(_u_edge), self.grid),
                state.p_s.data, _sigma)
        # Water vapour (moist runs only).
        q_v = None
        if (state.tracers is not None and "q_v" in state.tracers):
            q_v = state.tracers["q_v"].data
        # Cloud-diagnostic CMOR fields (clt/clwvi/clivi), gated by the SAME
        # --clear-sky-diag flag as the clear-sky TOA pair below (one knob turns
        # on the whole cloud/CRE CMOR set on this lane; default off =
        # byte-identical outputs for the running chains).  Pass the CLOUD
        # condensate tracers; the collector derives the RADIATIVE water paths +
        # max-random total cover.  ``q_i`` is absent on warm-rain microphysics,
        # which the collector skips rather than publishing a zero.
        q_c = None
        q_i = None
        _out_cfg = getattr(getattr(self, "config", None), "output", None)
        if (getattr(_out_cfg, "clear_sky_diag", False)
                and state.tracers is not None):
            if "q_c" in state.tracers:
                q_c = state.tracers["q_c"].data
            if "q_i" in state.tracers:
                q_i = state.tracers["q_i"].data
        # Flux fields (slots of the sfc_diag contract: 2 precip
        # [kg/m2/s], 3 lw_up_toa, 4 sw_up_toa, 5 sw_down_toa, 6 shflx,
        # 7 lhflx) — INTERVAL MEANS from the per-step accumulator when
        # it ran (#1353; makes the CMOR ``time: mean`` label true for
        # these diurnal fields at any diag cadence), else the last
        # step's instantaneous value (pre-#1353 fallback).  None on
        # runs without radiation/turbulence; the collector skips
        # absent fields.
        _sfc_diag = None if state_only else getattr(self.model, "_sfc_diag", None)
        _accum = None if state_only else getattr(self, "_mpas_sfc_accum", None)
        # A SHORT window (first interval after an off-cadence restart or
        # a feed-off link) covers less time than its label claims, so
        # WITHHOLD the flux fields entirely rather than publish a
        # partial-window mean — and do NOT fall back to the
        # instantaneous slots, which is the very defect #1353 fixes
        # (codex-6). The independent hourly state feed is unaffected.
        _accum_partial = (_accum is not None and _accum.has_samples()
                          and not _accum.is_complete())
        if _accum_partial:
            logger.warning(
                "  CMOR flux fields WITHHELD at day %.2f: this diagnostic "
                "window saw %d of %d steps (restart/feed-gap boundary) — "
                "publishing it would label a partial mean as a full "
                "interval.", day, _accum._steps, _accum.expected_steps)

        def _sfc_slot(i):
            if _accum is not None:
                if _accum_partial:
                    return None
                m = _accum.mean(i)
                if m is not None:
                    return m
            if (_sfc_diag is not None and len(_sfc_diag) > i
                    and _sfc_diag[i] is not None):
                return _sfc_diag[i].data
            return None
        precip = _sfc_slot(2)
        rlut = _sfc_slot(3)
        rsut = _sfc_slot(4)
        rsdt = _sfc_slot(5)
        hfss = _sfc_slot(6)
        hfls = _sfc_slot(7)
        # Clear-sky TOA pair (#843): slots populated only when
        # --clear-sky-diag is on, so these are None (fields absent from the
        # CMOR output, byte-identical) in the default configuration.
        rsutcs = _sfc_slot(10)
        rlutcs = _sfc_slot(11)
        lat_deg = np.degrees(np.asarray(self.grid.latCell))
        # Interval-mean flux fields carry their averaging window so the
        # feed can calendar-bin them at the interval MIDPOINT (#1353
        # codex-1 finding 1); None when the accumulator never ran
        # (instantaneous fallback -> endpoint binning, the legacy
        # snapshot semantics).
        # Midpoint calendar-binning is exact only for a window that
        # cannot straddle a calendar boundary: its TRUE length (the
        # integer step count times dt — not the requested ``diag_days``,
        # which the step arithmetic truncates) must divide the day
        # evenly.  Anything else — a multi-day cadence, or an arbitrary
        # sub-daily one like 0.3 d — falls back to endpoint binning, the
        # legacy snapshot semantics (codex-2/7).
        _flux_days = None
        if _accum is not None and _accum.has_samples():
            _win_days = (_accum.expected_steps * float(self.config.dycore.dt)
                         / 86400.0) if _accum.expected_steps > 0 else 0.0
            if 0.0 < _win_days <= 1.0:
                _per_day = 1.0 / _win_days
                # Duration must divide the day AND the window must SIT on
                # that 1/N-day grid: an off-grid phase (fractional
                # start_day / rebase) lets a window straddle midnight and
                # eventually a month boundary, which midpoint binning
                # cannot represent (codex-8).
                _phase = float(day) / _win_days
                # Tolerance in DAYS (absolute), not a relative one: a
                # relative slack grows with the simulation day and would
                # admit an off-grid window by ~an hour after a century
                # (codex-9).  1e-9 d ~ 0.1 ms, far below fp noise on a
                # float64 day counter.
                _phase_err_days = abs(_phase - round(_phase)) * _win_days
                if (abs(_per_day - round(_per_day)) < 1e-9
                        and _phase_err_days < 1e-9):
                    _flux_days = _win_days
        # Physics carry (CLUBB cloud fraction + deep-convection inputs) so
        # the CMOR cloud diagnostics diagnose cloud_scheme='cam6_clubb' the
        # same WAY radiation does (current-state sampling; see
        # DiagnosticCollector._cam6_cloud_kwargs for the lag caveat).
        # Cell-axis arrays (not the carry object) so the MPI gather can map
        # them; absent for every other cloud scheme (byte-identical feed).
        _cam6_carry = {}
        _ps_carry = getattr(self, "_mpas_phys_state", None)
        if (getattr(getattr(self, "config", None), "cloud_scheme", None)
                == "cam6_clubb" and _ps_carry is not None):
            _cam6_carry = dict(
                cloud_fraction=_ps_carry.cloud_fraction,
                conv_mass_flux_up=_ps_carry.conv_mass_flux_up,
                conv_icwmr=_ps_carry.conv_icwmr,
            )
        return dict(
            T=state.T.data,
            p_s=state.p_s.data,
            lat_deg=lat_deg,
            **_cam6_carry,
            q_v=q_v,
            q_c=q_c,
            q_i=q_i,
            u_east=u_east,
            v_north=v_north,
            precip=precip,
            phis=state.phis.data,
            tas=tas,
            ts=ts,
            rlut=rlut,
            rsut=rsut,
            rsdt=rsdt,
            hfss=hfss,
            hfls=hfls,
            rsutcs=rsutcs,
            rlutcs=rlutcs,
            wap=wap,
            flux_interval_days=_flux_days,
        )

    def _feed_mpas_cmip_multirank(self, day: float, diag, vlayout,
                                  *, state_only=False, flux_only=False) -> None:
        """Multi-rank CMOR feed: OWNED-cell -> global gather, commit on rank 0.

        Design
        ------
        GATHER-then-regrid, not a distributed regrid.  Every rank ships only
        its OWNED rows (``partition.local_cells[:n_owned_cells]`` — the
        partition builder concatenates ``sort(owned)`` then ``sort(halo)``, so
        halo rows are never sent, ``cell_owner`` is a total map and every
        global slot is written exactly once); rank 0 then runs the UNCHANGED
        serial regrid + plev19 + zonal pipeline on the assembled global field,
        using GLOBAL IDW weights (``_create_diagnostics`` hands the collector
        ``_grid_global`` under a Voronoi partition).  A distributed regrid
        would instead have to re-implement the collector's whole phase-1
        pipeline in partial-sum form and would change the serial summation
        order; gathering reuses the tested path verbatim, which is what makes
        serial equivalence provable (and measured: bitwise).

        Memory: the assembled global field lives ONLY on rank 0 and ONLY on
        the host (:func:`gather_owned_cells_to_root` gathers to root and
        returns NumPy — no ``jnp.array`` round-trip, so no rank puts a global
        copy on its device).  At 1 deg / res-6 (40 962 cells, nlev 32, fp32)
        that is ~5.2 MB per 3-D field and ~27 MB for the whole kwarg set on
        rank 0, roughly double transiently while ``gather`` holds the chunk
        list.  res-7 (163 842 cells) scales 4x, ~220 MB on rank 0.  Of the
        GATHERED arrays, non-root ranks hold only their own owned slice — that
        is a statement about this gather's payload, NOT about process peak
        memory, which also carries each rank's owned+halo state, the global
        diagnostic grid and ``fx`` fields, and the collector's own float64
        regrid temporaries on root.

        Collective safety (every rank reaches every collective on every path)
        ------------------------------------------------------------------
        PHASE 1 does ALL the failure-prone work with NO collectives: building
        the native fields, the shape check, AND the device->host copy of each
        owned slice.  Anything that can raise (OOM, a bad shape, a physics
        export gap) therefore raises OUTSIDE a collective and is caught
        locally.  PHASE 2 is a SINGLE ``allgather`` every rank executes
        unconditionally, carrying the local error status and the local
        field-name set; every rank derives the SAME decision from the SAME
        rank-ordered list, so an abort is unanimous.  PHASE 3 is a SINGLE
        collective carrying every agreed field: the assembly of the global
        arrays is root-only work and therefore happens strictly AFTER the last
        collective, never between two of them (a per-field gather loop would
        put a root-only ``n_global`` allocation between collectives, and an
        OOM there — root is the memory-tight rank — would strand every peer in
        the next field's gather; codex round 1).  PHASE 4 (the accumulator
        commit) is likewise rank-0-only and post-collective, so a commit
        failure cannot strand a peer either.

        Only rank 0's accumulators are ever written (``_finalize_mpas_cmip``
        and the CMOR sidecar are both rank-0-only), so non-root ranks skip the
        commit.  ``flux_interval_days`` is likewise taken from rank 0; it is
        derived from the step count and diag cadence, which are identical on
        every rank.
        """
        from mpi4py import MPI
        from legoesm.parallel.voronoi_mpi import gather_owned_cells_to_root
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        part = vlayout.partition

        # --- PHASE 0a: LOCAL preflight for the edge exchange, NO collectives.
        # WHY the exchange is needed at all: the Perot reconstruction in
        # _mpas_cmip_native_kwargs reads the edges AROUND each cell, and an
        # OWNED cell on the partition cut is ringed by HALO edges.  Nothing
        # guarantees those hold their owner's value here — the last thing to
        # touch ``u`` is a LOCAL update after the last exchange.  It is
        # demonstrably wrong under column-local physics: ``_phys_col_local``
        # (set via ``physics_fn._column_local``, which today only the
        # idealized Kessler MPAS forcing sets) makes ``make_voronoi_mpi_step``
        # skip the pre-physics state exchange, and there is no post-physics
        # one, so halo edges keep ``old_halo + dt*local_tendency``.  The full
        # AMIP physics does NOT set that attribute and so takes the exchanging
        # path — this is therefore insurance that makes the published
        # ``ua``/``va`` along the cut correct for EVERY physics path, not a
        # fix for one measured run.
        #
        # ``jnp.asarray``: the exchange is written in JAX ops (``.at[].set``),
        # so a host NumPy edge array dies inside it.  That conversion, and the
        # owned-id slice, are done HERE — outside any collective — because a
        # failure between two collectives is what strands peers.
        pre_err = None
        u_local = None
        owned_ids = None
        n_owned = 0
        try:
            # The gather helper uses COMM_WORLD; a partition built on a
            # different communicator would silently mis-assemble (or hang).
            # Checked HERE, inside the preflight, so a mismatch on one rank
            # rides the status agreement instead of raising ahead of it
            # (codex): raising early is itself the strand-the-peers bug.
            if comm.Get_size() != int(part.n_ranks):
                raise RuntimeError(
                    f"CMOR gather: partition n_ranks={part.n_ranks} != "
                    f"COMM_WORLD size {comm.Get_size()}")
            n_owned = int(part.n_owned_cells)
            owned_ids = np.asarray(part.local_cells[:n_owned])
            u_local = jnp.asarray(self.state.u.data)
        except Exception as exc:  # noqa: BLE001 - status is agreed below
            pre_err = f"{type(exc).__name__}: {exc}"
        except BaseException:  # noqa: BLE001 - see below; re-raised
            # NOT an ordinary failure: KeyboardInterrupt / SystemExit unwind
            # THIS rank out of the function entirely, so it never reaches the
            # 0b allgather while its peers do — a hang, not a crash (codex).
            # Only ``Exception`` is recoverable-by-agreement; anything else
            # must take the whole job down.
            comm.Abort(1)
            raise

        # --- PHASE 0b: ONE allgather, reached on every path, BEFORE anyone
        # enters the edge exchange.  Without it a rank that failed above would
        # be caught by the caller's diagnostic guard while its peers blocked
        # forever inside ``exchange_edge_field`` (codex).
        pre_status = comm.allgather(pre_err)
        pre_errors = [(i, e) for i, e in enumerate(pre_status) if e is not None]
        if pre_errors:
            if rank == 0:
                logger.error(
                    "  CMOR feed at day %.2f: SKIPPED — pre-exchange setup "
                    "failed on rank(s) %s: %s. Every rank skips in lockstep; "
                    "this interval's CMOR samples are lost, the run "
                    "continues.", day, [i for i, _ in pre_errors],
                    pre_errors[0][1])
            return

        # --- PHASE 0c: the edge halo exchange.  Unanimous arrival is now
        # guaranteed by 0b, so this collective cannot be entered by a subset.
        # A failure INSIDE it is genuinely unrecoverable: peers are already
        # blocked in the exchange and the communicator has no defined state to
        # return to.  Simply re-raising is NOT enough — the caller wraps this
        # method in a broad ``except Exception`` that logs and continues, so
        # this rank would walk on while its peers hang to walltime (codex).
        # ``Abort`` is the only response that ends the job rather than
        # deadlocking it.
        try:
            u_ex = vlayout.halo_exchange.exchange_edge_field(u_local)
        except BaseException as exc:  # noqa: BLE001 - re-raised after Abort
            logger.critical(
                "  CMOR feed at day %.2f: the edge halo exchange FAILED on "
                "rank %d (%s: %s). Peers are blocked inside that collective "
                "and cannot be released, so the job is aborted rather than "
                "left to hang.", day, rank, type(exc).__name__, exc)
            comm.Abort(1)
            raise

        # --- PHASE 1: everything that can fail, with NO collectives. --------
        owned: dict = {}
        flux_days = None
        err = None
        try:
            if state_only:
                kw = self._mpas_cmip_native_kwargs(
                    day, diag, u_override=u_ex, state_only=True)
            else:
                kw = self._mpas_cmip_native_kwargs(day, diag, u_override=u_ex)
            flux_days = kw.pop("flux_interval_days", None)
            for name, val in kw.items():
                if val is None:
                    continue
                n_rows = int(np.shape(val)[0]) if np.ndim(val) else -1
                if n_rows != part.n_local_cells:
                    # Not on the LOCAL cell axis -> cannot be mapped to global
                    # cells; refusing is the only safe act.
                    raise ValueError(
                        f"CMOR feed field {name!r} has leading axis {n_rows}, "
                        f"expected n_local_cells={part.n_local_cells}")
                # OWNED rows only, materialised on the HOST here (not inside
                # the collective).  ``np.asarray`` forces the device->host
                # copy now, so a per-rank OOM cannot strand the others.
                owned[name] = np.asarray(val[:n_owned])
        except Exception as exc:  # noqa: BLE001 - status is agreed below
            err = f"{type(exc).__name__}: {exc}"
        except BaseException:  # noqa: BLE001 - same lockstep argument as 0a
            # KeyboardInterrupt / SystemExit here would skip the phase-2
            # allgather on this rank alone and hang the peers in it (codex).
            comm.Abort(1)
            raise

        # --- PHASE 2: ONE collective, reached on every path. -----------------
        status = comm.allgather((err, sorted(owned), flux_days))
        errors = [(i, e) for i, (e, _, _) in enumerate(status)
                  if e is not None]
        # Intersect so a field present on only SOME ranks can never make the
        # per-rank gather counts diverge (that would deadlock).
        names = sorted(set.intersection(*[set(n) for _, n, _ in status]))
        dropped = sorted(
            set().union(*[set(n) for _, n, _ in status]) - set(names))
        if dropped and rank == 0:
            # Silent pruning would surface only as a low sample count in
            # post-run QA, so say it out loud.
            logger.warning(
                "  CMOR feed at day %.2f: field(s) %s are absent on at least "
                "one rank and are DROPPED for this interval (they stay out of "
                "the monthly sample count).", day, dropped)
        # The flux averaging WINDOW must be unanimous: it labels the CMOR
        # ``time: mean`` bin, and rank 0's value silently labelling everyone
        # else's samples would be a wrong number, not a crash (codex round 2).
        # On disagreement fall back to endpoint binning — the legacy snapshot
        # semantics, which claims no window.
        _fd = [f for _, _, f in status]
        if len(set(_fd)) > 1:
            if rank == 0:
                logger.warning(
                    "  CMOR feed at day %.2f: ranks disagree on the flux "
                    "averaging window (%s); falling back to endpoint binning "
                    "so no interval is labelled with a window it did not "
                    "cover.", day, _fd)
            flux_days = None
        else:
            flux_days = _fd[0]
        required = {"T", "p_s", "tas"} if state_only else {"T", "p_s"}
        if errors or not required.issubset(names):
            # Unanimous: every rank computed this from the same ``status``.
            if rank == 0:
                logger.error(
                    "  CMOR accumulator feed SKIPPED at day %.2f on all %d "
                    "ranks (run continues; this interval's CMOR means are "
                    "lost). rank-local failures=%s; fields common to all "
                    "ranks=%s", day, comm.Get_size(), errors or "none", names)
            return

        # --- PHASE 3: ONE collective, identical payload shape on every rank. -
        # The helper's root-only reconstruct runs AFTER it, so nothing that can
        # fail on rank 0 alone sits between two collectives.
        # ponytail: pickled ``gather``.  If the diagnostic cadence ever makes
        # this measurable, pack the agreed names into one (n_owned, -1) buffer
        # for a typed Gatherv.
        gathered = gather_owned_cells_to_root(
            {n: owned[n] for n in names}, owned_ids, part.nCells_global)

        # --- PHASE 4: commit on rank 0 only; no collectives beyond here. -----
        if rank == 0:
            if state_only:
                sample_bin = (np.floor(day * 24.0 + 1e-9) - 0.5) / 24.0
                diag.feed_cmip_accumulators_native(sample_bin, **gathered)
                diag.feed_daily_extremes_native(sample_bin, tas=gathered["tas"])
            else:
                diag.feed_cmip_accumulators_native(
                    day, flux_interval_days=flux_days,
                    include_state=not flux_only, **gathered)

    def _finalize_mpas_cmip(self, final_day: float | None = None) -> None:
        """Write the CMOR NetCDF (``Amon`` / ``day`` / ``fx``) from the fed
        accumulators at a CLEAN MPAS completion.

        ``_run_mpas`` bypasses the shared :meth:`_finalize_run` (whose
        ``diagnostics.save()`` ALSO writes the cube/lat-lon ``timeseries.npz`` +
        snapshots this lean path tracks SEPARATELY in ``_ts`` — calling it here
        would clobber the lightweight timeseries).  So mirror ONLY the
        CMIP-file finalize of ``save()``.  Wallclock-graceful exits already
        flush COMPLETED months incrementally and POP them
        (:meth:`_maybe_wallclock_exit`); this covers the final / standalone
        clean completion, writing whatever remains (in-progress + not-yet-
        flushed months, subject to the finalize partial-month guard) — which
        otherwise left the now-fed accumulators unwritten (empty ``cmor/``).

        The writers do NOT pop, so this is TERMINAL.  On success it (a) sets
        ``_suppress_cmor_sidecar`` so a still-to-be-written final checkpoint
        skips its CMOR sidecar, AND (b) RETIRES (unlinks) the terminal-day
        sidecar if one was ALREADY written this step — the exact-checkpoint-
        cadence case, where the in-loop periodic checkpoint wrote the sidecar
        BEFORE this finalizer runs and the after-loop "final checkpoint" is
        skipped, so suppression alone (which only blocks a FUTURE write) would
        leave a stale same-day sidecar that a run-EXTENDING restart would
        restore and re-append (duplicate ``time`` coords).  Mirrors the intent
        of :meth:`_finalize_run`.  Rank-0 only; fully guarded (a writer failure
        must not turn a COMPLETED run into a crash after the science is done —
        the accumulators/sidecar are left intact for inspection).

        NOTE: the underlying CMOR writer appends field-by-field and is not
        itself transactional, so a mid-write I/O failure can leave a partially
        written NetCDF (inherited from the shared ``save()`` path); on such a
        failure the sidecar is intentionally NOT retired, but a blind retry
        could still duplicate the already-appended fields — a durable
        per-write progress record is the follow-up for full crash-safety."""
        diag = getattr(self, "diagnostics", None)
        if diag is None or getattr(diag, "cf_writer", None) is None:
            return
        _is_root = (getattr(self, "_mpi_rank", None) is None
                    or self._mpi_rank == 0)
        if not _is_root:
            return
        try:
            diag._write_cmip_monthly_files()
            diag._write_cmip_daily_files()
            diag._write_cmip_fixed_files()
            diag.cf_writer.close()
            # Terminal: block any future sidecar write AND retire a same-day
            # sidecar already written by the in-loop periodic checkpoint.
            self._suppress_cmor_sidecar = True
            if final_day is not None:
                _out = getattr(self, "_output_dir", None)
                if _out is not None:
                    _sc = Path(_out) / (
                        f"cmor_accum_day_{int(round(final_day)):04d}.npz")
                    if _sc.exists():
                        _sc.unlink()
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.error(
                "  MPAS CMOR NetCDF finalize FAILED (accumulators/sidecar left "
                "intact for inspection; the NetCDF may be partially written — "
                "the shared CMOR writer is not transactional): %s", exc)

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

    # ------------------------------------------------------------------
    # FV3 six-face duo-cube lane (dry dynamics + optional certified
    # Held-Suarez forcing, fp64)
    # ------------------------------------------------------------------

    # Wind-speed blowup envelope for the duo lane's snapshot guard [m/s].
    # The DCMIP16 baroclinic-wave jet peaks near ~40 m/s; a state past this
    # bound is a diverged integration, not a strong storm.  Numerics guard,
    # not a tunable (same role as the acoustic lane's own envelope).
    _FV3_DUO_BLOWUP_UMAX_MS = 400.0
    _FV3_DUO_CKPT_SCHEMA = "fv3duo_ckpt_v1"
    _FV3_DUO_PRESS_KEYS = ("ps", "pe", "peln", "pk", "pkz")

    def _run_fv3_duo(self, start_step: int = 0,
                     start_day: float | None = None) -> str:
        """Dedicated lean lane for the certified FV3 duo-cube step
        (slice 1 dynamics + slice 2 restart + optional Held-Suarez).

        Dry DCMIP16 baroclinic wave ONLY (adiabatic, or with the
        certified Held-Suarez forcing when
        ``cfg.held_suarez_forcing``): the lane builds its
        own IC (the placeholder ``self.state`` from ``_init_state`` is a
        CD-grid scaffold this lane never reads; it is REPLACED by the
        final six-face bundle so the run-manifest digest reflects the
        actual final state).  Steps the bundled pytree
        ``{"state","press","q","omga","nh"}`` with the wrapper's jitted
        step and writes a minimal fp64 ``.npz`` snapshot at the
        ``diag_days`` cadence (own schema — ``fv3duo_snapshot_step_*.npz``
        with the face-stacked duo fields; NOT the cube/lat-lon
        checkpoint schema, which the duo layout does not fit).

        Slice-2 restart contract (READ THIS before writing a chain
        launcher — it deliberately CONTRASTS with ``_run_mpas``):
        ``cfg.days`` is TOTAL days since the epoch (the cube/lat-lon
        convention), NOT days-this-job.  A restarted job therefore
        passes the SAME ``--days`` as the straight run and this loop
        advances ``n_steps_total - loaded_step`` steps; ``_run_mpas``
        instead treats ``--days`` as the number of days *this job*
        advances and its chain launchers pass remaining days — do not
        copy that convention here.  Restart flows ONLY through
        WHY THE BUNDLE IS SUFFICIENT (stated, not assumed -- GLM
        2026-08-19 flagged that "restart is bitwise" proves only that the
        DYNAMICAL STATE round-trips, and is blind to whatever the driver
        loop carries): this lane carries NO loop state. There are no
        diagnostic accumulators (snapshots are written straight from the
        bundle, no time-means), the blowup guard is stateless (it
        re-evaluates each snapshot from scratch), the CFL clamp is a pure
        function of the config, and the step loop runs over GLOBAL step
        numbers (``range(loaded_step + 1, n_steps_total + 1)``), so the
        diagnostic and checkpoint cadences are pure functions of the
        restored step and their phase is restart-invariant by
        construction. Any FUTURE loop state -- an accumulator, a
        hysteretic clamp, a physics carry -- breaks that invariant and
        must join the bundle, not the loop.

        ``load_checkpoint`` on an ``fv3duo_ckpt_v1`` file, which stages
        the FULL persisted bundle (state/press/q/omga/nh, fp64) —
        nothing is rebuilt from delp, so the resumed trajectory is
        BITWISE the straight one (same jitted program, exact fp64 npz
        round-trip).  Checkpoints are written at the shared
        ``output.checkpoint_days`` cadence (absolute-step phase) plus
        the final step, as ``fv3duo_ckpt_step_*.npz`` (atomic:
        tmp + os.replace).

        Refusals that REMAIN (each loud, none silent): no MPI, no
        ensemble.  Single- AND multi-process face SPMD are both supported
        via --distributed-mode spmd, dry or with Held-Suarez (the
        multi-process HS step is the face-stacked JAX twin, see
        ``_fv3_duo_apply_held_suarez``); multi-process
        checkpoint/snapshot I/O goes through
        ``_fv3_duo_reshard_bundle_global`` / the gather-then-root-write
        helpers, not the single-process ``np.asarray`` path directly.
        All physics/forcing EXCEPT Held-Suarez are refused at model
        construction (component factory); ``cfg.held_suarez_forcing``
        applies the certified 3-pass HS step after each dynamics step.
        The HS step is a pure function of the bundle (no accumulators,
        no memory), so the restart invariant above — this lane carries
        NO loop state — survives it.  A restarted run overwrites the
        status marker (RUNNING-first, as always).
        """
        cfg = self.config
        loaded = self._loaded_checkpoint_step_day
        restart_bundle = getattr(self, "_fv3_duo_restart_bundle", None)
        if (start_step != 0 or loaded is not None) \
                and restart_bundle is None:
            raise ValueError(
                "fv3_duo restart flows ONLY through load_checkpoint on an "
                "fv3duo_ckpt_v1 checkpoint (which stages the bundle to "
                "resume from); a bare start_step has no bundle to start "
                "from. Run from step 0, or pass --restart-from an "
                "fv3duo_ckpt_step_*.npz.")
        loaded_step = 0
        if restart_bundle is not None:
            if loaded is not None and start_day is not None \
                    and start_day != loaded[1]:
                # codex MAJOR: only the STEP was bound, so
                # run(start_step=step, start_day=<anything>) was accepted
                # and wrote snapshots stamped with a shifted day. The dry
                # dynamics never reads day, so the bitwise state test
                # stayed green while the provenance drifted.
                raise ValueError(
                    f"fv3_duo restart: start_day={start_day} does not match "
                    f"the loaded checkpoint day {loaded[1]}; the pair is "
                    f"the checkpoint's, not the caller's.")
            if loaded is None or start_step != loaded[0]:
                raise ValueError(
                    f"fv3_duo restart: start_step={start_step} does not "
                    f"match the loaded checkpoint step "
                    f"{None if loaded is None else loaded[0]}; pass "
                    f"load_checkpoint's returned step through unchanged.")
            loaded_step = start_step
        if (self._mpi_world_size or 1) > 1 and not self._is_spmd_multiprocess():
            raise NotImplementedError(
                "fv3_duo driver runs refuse multi-process EXCEPT "
                "multi-controller SPMD (--distributed "
                "--distributed-mode spmd); MPI mode is not wired for "
                "this lane. Launch one process, or pass "
                "--distributed-mode spmd.")
        if cfg.distributed and cfg.distributed_mode != "spmd":
            raise NotImplementedError(
                "fv3_duo distributed runs are SPMD-only "
                "(--distributed-mode spmd); mpi mode is not wired.")
        if self._ensemble_size != 1:
            raise NotImplementedError(
                f"fv3_duo (slice 1) does not thread an ensemble axis; got "
                f"ensemble_size={self._ensemble_size}.")
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoDynamicsModel,
        )
        if not isinstance(self.model, FV3DuoDynamicsModel):
            raise TypeError(
                f"discretization='fv3_duo' but the constructed dycore is "
                f"{type(self.model).__name__} — the component factory and "
                f"this lane disagree; refusing to step the wrong model.")

        DT = cfg.dycore.dt
        n_steps_total = int(cfg.days * 86400.0 / DT)
        n_this_job = n_steps_total - loaded_step
        if n_this_job < 1:
            raise ValueError(
                f"fv3_duo: days={cfg.days} is TOTAL days ({n_steps_total} "
                f"steps at dt={DT}s) and the run "
                f"{'is already at/past that target (checkpoint step ' + str(loaded_step) + ')' if loaded_step else 'yields no steps'}"
                f"; nothing to run. days counts from the epoch, NOT from "
                f"the checkpoint (the MPAS days-this-job convention does "
                f"not apply to this lane).")
        # The SHARED cadence helper, not an inline copy: it is the single
        # definition of the diag/blow-up interval and it REFUSES a non-finite
        # diag_days (NaN/inf) instead of letting it fall through to the
        # "no cadence" sentinel — the inline form this replaced accepted NaN
        # and ran with a cadence nobody asked for.
        from legoesm.driver.diagnostics import diagnostic_interval_steps
        diag_interval = diagnostic_interval_steps(
            cfg.output.diag_days, DT, n_steps_total)
        ckpt_interval = (max(1, int(cfg.output.checkpoint_days * 86400.0
                                    / DT))
                         if cfg.output.checkpoint_days > 0 else 0)
        if start_day is None:
            start_day = cfg.start_day

        hs_on = bool(cfg.held_suarez_forcing)
        kessler_on = cfg.microphysics == "kessler"
        logger.info(
            "FV3 duo lane: C%d km=%d %s, Held-Suarez %s, Kessler %s, "
            "dt=%.1fs, %d "
            "steps (%.2f total days, %d this job%s), snapshots every %d "
            "steps, checkpoints every %s steps",
            cfg.grid.resolution, self.model.config.km,
            "hydrostatic" if self.model.config.hydrostatic
            else "nonhydrostatic",
            "ON" if hs_on else "off", "ON" if kessler_on else "off",
            DT, n_steps_total, cfg.days, n_this_job,
            f", RESTART from step {loaded_step}" if loaded_step else "",
            diag_interval, ckpt_interval or "never")

        # Overwrite any marker from a prior run in this directory FIRST:
        # setup permits a same-config retry, and a stale COMPLETED/BLOWUP
        # surviving an interrupted retry would misclassify it (codex
        # 2026-08-18 MAJOR). RUNNING is the nonterminal state.
        self._fv3_duo_write_status("RUNNING")
        if restart_bundle is not None:
            bundle = restart_bundle
        else:
            bundle = self._fv3_duo_fresh_ic()
        # Multi-process SPMD: BOTH a fresh IC and a restart-loaded bundle
        # come out of the above as fully-addressable arrays (dcmip16_
        # initial_state builds identically on every rank; the restart
        # loader's multi-process branch already broadcasts to an
        # identical-on-every-rank host bundle) -- neither is yet
        # sharded the way the compiled step's GSPMD mesh expects. This
        # reshards ONCE, uniformly, for either origin (codex: "one
        # model/helper method shared by initialization and restart").
        # No-op single-process.
        if self.model.window_layout is not None:
            # WINDOW layout (--fv3-duo-windows): the fresh IC already comes
            # out window-stacked on the window sharding (to_windows places
            # each process's own shards via make_array_from_callback); a
            # restart bundle is host FACES and takes the same road.  The
            # face reshard does not apply to window stacks.
            if restart_bundle is not None:
                bundle = self.model.to_windows(bundle)
        else:
            bundle = self._fv3_duo_reshard_bundle_global(bundle)
        t0 = time.time()
        for step in range(loaded_step + 1, n_steps_total + 1):
            bundle = self.model.step(bundle, DT)
            if hs_on:
                # fv_phys ordering: forcing applied to the POST-dynamics
                # state, once per outer (bdt) step, exactly as the
                # certified parity arm does.  Stateless — bundle in,
                # bundle out; the restart invariant is untouched.
                bundle = self._fv3_duo_apply_held_suarez(bundle, DT)
            if kessler_on:
                bundle = self._fv3_duo_apply_kessler(bundle, DT)
            # start_day is the ABSOLUTE day at loaded_step (the day the
            # checkpoint was written; cfg.start_day on a fresh run), so
            # `day` is absolute simulated time on both arms of a chain.
            day = start_day + (step - loaded_step) * DT / 86400.0
            if step % diag_interval == 0 or step == n_steps_total:
                blowup = self._fv3_duo_snapshot(bundle, step, day)
                if blowup is not None:
                    self._fv3_duo_write_status(blowup)
                    return blowup
            if ckpt_interval and (step % ckpt_interval == 0
                                  or step == n_steps_total):
                self._fv3_duo_save_checkpoint(bundle, step, day)
        # Final-state digest (run manifest) hashes self.state — hand it
        # the ACTUAL final bundle, not the unused CD-grid scaffold.
        self.state = bundle
        logger.info("FV3 duo lane COMPLETED: %d steps in %.1fs",
                    n_this_job, time.time() - t0)
        self._fv3_duo_write_status("COMPLETED")
        return "COMPLETED"

    def _fv3_duo_apply_held_suarez(self, bundle: dict, dt: float) -> dict:
        """One certified Held-Suarez physics step on the duo bundle.

        Multi-process SPMD dispatches to :meth:`_fv3_duo_apply_held_suarez_jax`
        (the face-stacked pure-JAX twin, pinned to this NumPy path at 1e-11
        in ``test_fv3_physics_coupling``): the per-face leaves of a
        multi-process bundle are not host-addressable, so the NumPy
        round-trip below cannot run there.  Single-process keeps this
        path byte-identical (user call 2026-09-21: one variable moves).

        Thin adapter only — ALL numerics live in
        ``apply_held_suarez_step`` (the 3-pass orchestration gated at
        1.7645e-8 vs the Fortran oracle).  That function wants the
        parity harness's per-face NumPy ``state``/``press`` dicts and
        mutates ``state[t]["u"/"v"/"pt"]``; the lane's bundle is
        face-stacked jax, so this unstacks (``state_3d_to_numpy``, the
        pinned exact inverse of ``state_3d_to_jax``), calls it, and
        restacks the three mutated fields.  ``press`` is read-only to
        the HS step (HS changes pt, not delp, so the hydrostatic
        pressures are unchanged) and the per-face slices of the
        ``p_var`` stacks are already the oracle's own per-face layouts
        (pe ``(n+2, km+1, n+2)``, peln ``(n, km+1, n)``, pkz
        ``(n, n, km)`` — ``p_var_hydrostatic``'s docstring); the HS
        step's internal transposes were written against exactly those.

        backend="numpy": the SPECIFICATION path that carries the
        certified score; the jitted dynamics step stays jax, and the
        per-step np round-trip is fine at the C12-C48 scales this
        lane runs.  strat=True: Fortran ``do_strat_HS_forcing`` defaults
        .true. (fv_arrays.F90) and the certified deck resolved .true. —
        deliberately NOT a config knob.

        Pure bundle -> bundle (inputs never mutated: u/v get writable
        copies after the unstack, BEFORE the step's in-place halo
        exchange), so the lane's "NO loop state" restart invariant
        survives.
        """
        if self._is_spmd_multiprocess():
            return self._fv3_duo_apply_held_suarez_jax(bundle, dt)
        if self.model.window_layout is not None:
            # single-process WINDOW layout: every window is host-addressable,
            # so the certified NumPy step runs on the six faces (owned cells
            # scattered back) and the result is re-windowed.  The NumPy
            # authority stays THE single-process path, faces or windows.
            faces = self._fv3_duo_apply_held_suarez_numpy(
                self.model.to_flat(bundle), dt)
            return self.model.to_windows(faces)
        return self._fv3_duo_apply_held_suarez_numpy(bundle, dt)

    def _fv3_duo_apply_held_suarez_numpy(self, bundle: dict, dt: float) -> dict:
        """The certified NumPy Held-Suarez step on a FACE-stacked bundle
        (the body of :meth:`_fv3_duo_apply_held_suarez`, unchanged)."""
        from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_numpy
        from legoesm.core.fv3_native_physics_coupling import (
            apply_held_suarez_step,
        )
        grid = self.model.grid
        n, ng, km = grid.n, grid.ng, self.model.config.km
        state_np = state_3d_to_numpy(bundle["state"])
        for face in state_np:
            # np.asarray of a jax leaf can be a READ-ONLY view; the HS
            # step's PASS-0 D-grid halo exchange writes the u/v halo
            # strips in place, so those two get writable copies.
            face["u"] = np.array(face["u"])
            face["v"] = np.array(face["v"])
        press_np = [
            {nm: np.asarray(bundle["press"][nm][t])
             for nm in self._FV3_DUO_PRESS_KEYS}
            for t in range(6)]
        apply_held_suarez_step(
            grid.ctx_np, state_np, press_np, dt=dt, n=n, ng=ng, km=km,
            strat=True, backend="numpy")
        new_state = dict(bundle["state"])
        for nm in ("u", "v", "pt"):  # the ONLY fields HS mutates
            new_state[nm] = jnp.asarray(
                np.stack([face[nm] for face in state_np]))
        return {**bundle, "state": new_state}

    def _fv3_duo_apply_held_suarez_jax(self, bundle: dict, dt: float) -> dict:
        """Face-stacked JAX Held-Suarez step for the multi-process lane.

        Jitted once per driver (metrics and halo tables are trace-time
        constants); the three moved leaves are pinned to the step's face
        sharding so the loop does not decay to replicated arrays.
        """
        from legoesm.core.fv3_native_physics_coupling import (
            apply_held_suarez_step_sixface_jax, stack_held_suarez_metrics,
        )
        fn = getattr(self, "_fv3_duo_hs_jax_fn", None)
        if fn is None:
            from legoesm.grids.fv3_duo_windows import (gather_windows,
                                                       scatter_owned)
            grid = self.model.grid
            n, ng, km = grid.n, grid.ng, self.model.config.km
            tab = self.model.sixface_halo_tables
            amat6, lat6, wv6 = stack_held_suarez_metrics(grid.ctx_np)
            sh = self.model.step_out_shardings
            lay = self.model.window_layout
            moved = ("u", "v", "pt")

            def _hs(state, press, dt):
                if lay is not None:
                    # window stacks -> six faces ON DEVICE (owned cells),
                    # the twin, then back to windows (pads rebuilt from the
                    # faces); untouched leaves keep their window arrays.
                    faces = {k: scatter_owned(lay, state[k], jnp)
                             for k in ("u", "v", "pt", "delp")}
                    pressf = {k: scatter_owned(lay, press[k], jnp)
                              for k in press}
                    out6 = apply_held_suarez_step_sixface_jax(
                        faces, pressf, tab, amat6, lat6, wv6, dt=dt, n=n,
                        ng=ng, km=km, strat=True)
                    out = dict(state)
                    for nm in moved:
                        out[nm] = gather_windows(lay, out6[nm], jnp)
                else:
                    out = apply_held_suarez_step_sixface_jax(
                        state, press, tab, amat6, lat6, wv6, dt=dt, n=n,
                        ng=ng, km=km, strat=True)
                if sh is not None:
                    for nm in moved:
                        out[nm] = jax.lax.with_sharding_constraint(out[nm],
                                                                   sh)
                return out
            fn = jax.jit(_hs)
            self._fv3_duo_hs_jax_fn = fn
        press = {nm: bundle["press"][nm] for nm in ("pe", "peln", "pkz")}
        new_state = fn(bundle["state"], press, float(dt))
        return {**bundle, "state": new_state}

    def _fv3_duo_fresh_ic(self) -> dict:
        """This deck's initial bundle -- the ONE builder for the fresh
        run AND the restart template, so the tracer count a checkpoint
        is validated against is the count the run actually carries.

        Kessler on: tracer slots ``KESSLER_TRACER_SLOTS`` -- the IC's own
        DCMIP16 specific humidity is slot 0; cloud and rain start at
        zero, as DCMIP 2016 test 161 does (NOT the lon-modulated
        passenger copies ``n_tracers > 1`` would build).
        """
        bundle = self.model.dcmip16_initial_state(do_pert=True)
        if self.config.microphysics == "kessler":
            q0 = bundle["q"][0]
            bundle = {**bundle,
                      "q": [q0, jnp.zeros_like(q0), jnp.zeros_like(q0)]}
        return bundle

    def _fv3_duo_apply_kessler(self, bundle: dict, dt: float) -> dict:
        """One operator-split Kessler step on the duo bundle, every lane.

        The bridge (``apply_kessler_step_sixface_jax``) is column-local
        and pure JAX, so ONE jitted function serves single-process faces,
        single-process windows and multi-process SPMD alike: under a
        window layout the owned block is scattered to faces and gathered
        back exactly as the Held-Suarez twin does.  pt, delp, the rebuilt
        pressures and the three Kessler tracers are pinned to the step's
        face sharding.  Stateless -- the restart invariant is untouched.
        """
        from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
            apply_kessler_step_sixface_jax,
        )
        fn = getattr(self, "_fv3_duo_kessler_jax_fn", None)
        if fn is None:
            from legoesm.grids.fv3_duo_windows import (gather_windows,
                                                       scatter_owned)
            from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
            grid = self.model.grid
            n, ng, km = grid.n, grid.ng, self.model.config.km
            ptop = self.model._ptop
            sh = self.model.step_out_shardings
            lay = self.model.window_layout
            kw = dict(n=n, ng=ng, km=km, ptop=ptop, akap=FV3_KAPPA)

            def _kessler(state, press, q, dt):
                if lay is not None:
                    faces = {k: scatter_owned(lay, state[k], jnp)
                             for k in ("pt", "delp")}
                    pressf = {k: scatter_owned(lay, press[k], jnp)
                              for k in press}
                    # EVERY tracer goes through the bridge: passengers
                    # beyond the Kessler slots are renormalised to the
                    # new layer mass there (codex 2026-09-24)
                    qf = [scatter_owned(lay, qi, jnp) for qi in q]
                    out6, press6, q6 = apply_kessler_step_sixface_jax(
                        faces, pressf, qf, dt=dt, **kw)
                    moved = {k: gather_windows(lay, out6[k], jnp)
                             for k in ("pt", "delp")}
                    press_new = {k: gather_windows(lay, press6[k], jnp)
                                 for k in press6}
                    q_new = [gather_windows(lay, qi, jnp) for qi in q6]
                else:
                    out6, press_new, q_new = apply_kessler_step_sixface_jax(
                        state, press, q, dt=dt, **kw)
                    moved = {k: out6[k] for k in ("pt", "delp")}
                    q_new = list(q_new)
                if sh is not None:
                    moved = {k: jax.lax.with_sharding_constraint(v, sh)
                             for k, v in moved.items()}
                    press_new = {k: jax.lax.with_sharding_constraint(v, sh)
                                 for k, v in press_new.items()}
                    q_new = [jax.lax.with_sharding_constraint(qi, sh)
                             for qi in q_new]
                return {**state, **moved}, press_new, q_new
            fn = jax.jit(_kessler)
            self._fv3_duo_kessler_jax_fn = fn
        new_state, new_press, new_q = fn(
            bundle["state"], dict(bundle["press"]), list(bundle["q"]),
            float(dt))
        return {**bundle, "state": new_state, "press": new_press,
                "q": new_q}

    def _fv3_duo_host_faces(self, bundle: dict) -> dict:
        """Host-side, FACE-stacked copy of the bundle for every write.

        Multi-process: a collective gather to identical host data on
        every rank first.  Window layout: each window's OWNED cells are
        scattered back into the six faces (``to_flat``).  Face layout:
        ``to_flat`` is the identity, so the certified face path is
        byte-identical.  ONE seam for snapshot, checkpoint and the
        restart template, so no writer can see a window stack.
        """
        if self._is_spmd_multiprocess():
            bundle = self._gather_spmd_tree_to_host(bundle)
        return self.model.to_flat(bundle)

    def _fv3_duo_reshard_bundle_global(self, bundle: dict) -> dict:
        """Reconstruct every bundle leaf as a properly GSPMD-sharded
        ``jax.Array`` spanning the multi-process device mesh.

        No-op outside multi-process SPMD: single-process (including
        single-process multi-GPU) already gets correct sharding from
        the model's own jitted step under GSPMD auto-partitioning --
        nothing to fix there.  Under multi-process SPMD, every leaf of
        *bundle* MUST already be identical host-visible data on every
        rank -- true for a freshly built ``dcmip16_initial_state()``
        bundle (built from the same deterministic deck on every rank)
        and true for a restart bundle (the checkpoint loader's
        multi-process branch, ``_broadcast_fv3_duo_bundle``, already
        broadcasts to every rank before returning).  Calls
        ``jax.make_array_from_callback`` identically on every rank,
        each materialising only its own addressable shards -- the SAME
        idiom ``spmd_multiprocess_parity.py``'s ``_global_arrays``
        proved correct, against the model's OWN ``step_out_shardings``
        (not an independently rebuilt one -- codex MAJOR, mp-driver-io
        design review 2026-08-27) so a fresh run and a restart both
        step under the identical certified sharding.  Used for BOTH a
        fresh IC and a restart bundle (codex: "one model/helper method
        shared by initialization and restart"), not just restart.
        """
        if not self._is_spmd_multiprocess():
            return bundle
        sharding = self.model.step_out_shardings
        if sharding is None:
            raise RuntimeError(
                "fv3_duo multi-process SPMD: the model was constructed "
                "without step_out_shardings -- the component factory "
                "and this reshard helper have gone out of sync.")

        def _put(x):
            h = np.asarray(x)
            return jax.make_array_from_callback(
                h.shape, sharding, lambda idx: h[idx])

        out = {"state": {k: _put(v) for k, v in bundle["state"].items()},
               "press": {k: _put(v) for k, v in bundle["press"].items()},
               "q": [_put(qv) for qv in bundle["q"]],
               "omga": _put(bundle["omga"])}
        out["nh"] = (None if bundle.get("nh") is None else
                     {k: _put(v) for k, v in bundle["nh"].items()})
        return out

    def _fv3_duo_write_status(self, status: str) -> None:
        """Persist the lane's terminal status as an EXPLICIT marker.

        ``fv3duo_status.txt`` next to the snapshots: RUNNING is written
        before the first step (overwriting any stale marker from a prior
        run in the same directory), then the terminal state — COMPLETED
        or BLOWUP with day/step. An interrupted run therefore reads
        RUNNING. SCOPE (codex 2026-08-18): the marker certifies the
        STEP LOOP's outcome only; the run manifest's ``state_digest`` is
        written afterwards by ``run()`` and can still fail
        independently — consult the manifest for provenance, the marker
        for loop outcome.

        Multi-process SPMD: root-gated (every process running an
        unconditional ``write_text`` on the SAME file is a data race)
        and rendezvous'd via ``_spmd_barrier_on_root_error`` (codex
        MAJOR, mp-driver-io design review 2026-08-27), so a root write
        failure aborts every process in lockstep instead of leaving the
        others to hang on the next collective.  Single-process path is
        the original one-liner, untouched.
        """
        if not self._is_spmd_multiprocess():
            (self._output_dir / "fv3duo_status.txt").write_text(
                status + "\n")
            return
        err = None
        if jax.process_index() == 0:
            try:
                (self._output_dir / "fv3duo_status.txt").write_text(
                    status + "\n")
            except Exception as exc:
                err = exc
        self._spmd_barrier_on_root_error(err)

    def _fv3_duo_snapshot_fields_and_blowup(self, bundle: dict):
        """Pure field-extraction + blowup-check for the MULTI-PROCESS
        ``_fv3_duo_snapshot`` arm only (the single-process arm keeps
        its ORIGINAL statement order -- savez before the blowup check
        -- verbatim, rather than sharing this helper, so that arm is
        not just equivalent but textually unchanged; codex MAJOR,
        mp-driver-io diff review round 2: an earlier version reordered
        the single-process arm to share this helper)."""
        fields = {nm: np.asarray(v) for nm, v in bundle["state"].items()}
        fields["ps"] = np.asarray(bundle["press"]["ps"])
        for iq, qt in enumerate(bundle["q"]):
            fields[f"q{iq}"] = np.asarray(qt)
        bad = sorted(nm for nm, a in fields.items()
                     if not np.isfinite(a).all())
        umax = max(float(np.abs(fields["u"]).max()),
                   float(np.abs(fields["v"]).max()))
        return fields, bad, umax

    def _fv3_duo_snapshot(self, bundle: dict, step: int, day: float):
        """Write one minimal duo snapshot; return a BLOWUP status or None.

        Own schema (face-stacked duo layout): the prognostics from
        ``bundle["state"]`` plus ``ps`` and the sphum passenger, with
        ``_step`` / ``_day`` stamps.  The finite + wind-envelope guard
        runs on the SAME host copies the write uses, so a diverged state
        is both persisted (for autopsy) and reported.

        Multi-process SPMD: *bundle* is gathered to identical
        host-visible data on EVERY rank first (a collective, so every
        rank must call this); the blowup/finiteness verdict is then a
        pure function of that identical data (no reduction needed, and
        no risk of ranks disagreeing), while the actual ``np.savez``
        write + its logging is root-gated and rendezvous'd end to end
        -- not just the write call, the path construction too -- so a
        root-only failure (path prep, logging, the write itself)
        aborts every process in lockstep (codex MAJOR, mp-driver-io
        design review 2026-08-27).  The single-process arm below is
        the ORIGINAL code, untouched (not even reordered).
        """
        if not self._is_spmd_multiprocess():
            bundle = self.model.to_flat(bundle)   # identity on faces
            fields = {nm: np.asarray(v) for nm, v in bundle["state"].items()}
            fields["ps"] = np.asarray(bundle["press"]["ps"])
            for iq, qt in enumerate(bundle["q"]):
                fields[f"q{iq}"] = np.asarray(qt)
            path = self._output_dir / f"fv3duo_snapshot_step_{step:06d}.npz"
            np.savez(path, _step=np.int64(step), _day=np.float64(day),
                     **fields)

            bad = sorted(nm for nm, a in fields.items()
                         if not np.isfinite(a).all())
            umax = max(float(np.abs(fields["u"]).max()),
                       float(np.abs(fields["v"]).max()))
            if bad or umax > self._FV3_DUO_BLOWUP_UMAX_MS:
                logger.error(
                    "FV3 duo BLOWUP at step %d (day %.3f): non-finite=%s, "
                    "max|wind|=%.3g m/s (envelope %.0f); state saved to %s",
                    step, day, bad or "none", umax,
                    self._FV3_DUO_BLOWUP_UMAX_MS, path)
                return f"BLOWUP at day {day:.3f} (step {step})"
            # ps stats over the COMPUTE window only — the padded halo rows are
            # zero by construction and would print as a fake 0 hPa minimum.
            _n, _ng = self.model.grid.n, self.model.grid.ng
            ps_win = fields["ps"][:, _ng:_ng + _n, _ng:_ng + _n]
            logger.info(
                "  fv3_duo step %d day %.3f: max|wind|=%.2f m/s, "
                "ps=[%.1f, %.1f] hPa -> %s",
                step, day, umax,
                float(ps_win.min()) / 100.0,
                float(ps_win.max()) / 100.0, path.name)
            return None

        # ---- multi-process SPMD ----
        bundle = self._fv3_duo_host_faces(bundle)
        fields, bad, umax = self._fv3_duo_snapshot_fields_and_blowup(bundle)
        blown = bool(bad) or umax > self._FV3_DUO_BLOWUP_UMAX_MS
        err = None
        if jax.process_index() == 0:
            try:
                path = (self._output_dir
                        / f"fv3duo_snapshot_step_{step:06d}.npz")
                np.savez(path, _step=np.int64(step), _day=np.float64(day),
                         **fields)
                if blown:
                    logger.error(
                        "FV3 duo BLOWUP at step %d (day %.3f): "
                        "non-finite=%s, max|wind|=%.3g m/s (envelope "
                        "%.0f); state saved to %s",
                        step, day, bad or "none", umax,
                        self._FV3_DUO_BLOWUP_UMAX_MS, path)
                else:
                    _n, _ng = self.model.grid.n, self.model.grid.ng
                    ps_win = fields["ps"][:, _ng:_ng + _n, _ng:_ng + _n]
                    logger.info(
                        "  fv3_duo step %d day %.3f: max|wind|=%.2f m/s, "
                        "ps=[%.1f, %.1f] hPa -> %s",
                        step, day, umax,
                        float(ps_win.min()) / 100.0,
                        float(ps_win.max()) / 100.0, path.name)
            except Exception as exc:
                err = exc
        self._spmd_barrier_on_root_error(err)
        return (f"BLOWUP at day {day:.3f} (step {step})") if blown else None

    def _fv3_duo_flatten_bundle(self, bundle: dict) -> dict:
        """Flatten the duo bundle to prefixed fp64 numpy arrays.

        ONE naming convention shared by the checkpoint writer, the
        loader's inverse, and the bitwise A/B test: ``state_<k>``,
        ``press_<k>`` (exactly the five p_var keys), ``q_<i>``,
        ``omga``, and ``nh_<k>`` (non-hydrostatic only).  Refuses a
        non-float64 leaf — the checkpoint contract is a bit-exact fp64
        round-trip, so a lossy leaf is a defect, not a cast site.
        """
        arrays: dict[str, np.ndarray] = {}
        for nm, v in bundle["state"].items():
            arrays[f"state_{nm}"] = np.asarray(v)
        if set(bundle["press"]) != set(self._FV3_DUO_PRESS_KEYS):
            raise RuntimeError(
                f"fv3_duo bundle press keys {sorted(bundle['press'])} != "
                f"{sorted(self._FV3_DUO_PRESS_KEYS)} — the p_var contract "
                f"changed under this writer; refusing a partial persist.")
        for nm in self._FV3_DUO_PRESS_KEYS:
            arrays[f"press_{nm}"] = np.asarray(bundle["press"][nm])
        for iq, qt in enumerate(bundle["q"]):
            arrays[f"q_{iq}"] = np.asarray(qt)
        arrays["omga"] = np.asarray(bundle["omga"])
        if bundle["nh"] is not None:
            for nm, v in bundle["nh"].items():
                arrays[f"nh_{nm}"] = np.asarray(v)
        bad = sorted(nm for nm, a in arrays.items()
                     if a.dtype != np.float64)
        if bad:
            raise RuntimeError(
                f"fv3_duo bundle leaves {bad} are not float64; the "
                f"checkpoint schema persists fp64 bit-exact only.")
        return arrays

    def _fv3_duo_checkpoint_write(self, arrays: dict, mcfg, step: int,
                                  day: float, nq: int, path: Path) -> None:
        """The ROOT-ONLY write body shared by both arms of
        ``_fv3_duo_save_checkpoint``: open the unique tmp file, write,
        fsync, atomically publish, fsync the directory, log.  Pulled
        out so the multi-process arm can wrap the WHOLE thing (not
        just ``np.savez``) in one try/except before the rendezvous
        (codex MAJOR, mp-driver-io design review 2026-08-27)."""
        from legoesm.io.git_provenance import git_provenance
        # UNIQUE tmp name (GLM 2026-08-19): a fixed "<name>.tmp" lets two
        # concurrent writers in one directory interleave their bytes, and
        # os.replace then atomically publishes garbage -- an atomic rename
        # of a non-atomically-produced file is not atomicity.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                _schema=self._FV3_DUO_CKPT_SCHEMA,
                _step=np.int64(step),
                _day=np.float64(day),
                _dt=np.float64(self.config.dycore.dt),
                _hydrostatic=np.bool_(mcfg.hydrostatic),
                # the thermodynamic mode is CONTRACT too (codex
                # 2026-09-24): a dry checkpoint resumed moist would gain
                # humidity feedback silently, and the tracer count alone
                # cannot tell the two apart
                _zvir=np.float64(self.model.zvir),
                _km=np.int64(mcfg.km),
                _resolution=np.int64(self.model.grid.n),
                # nq is CONTRACT, not decoration: the loader checks the
                # tracer leaves against it, because a contiguity check
                # alone accepts the empty set and resumes with tracers
                # silently dropped (codex BLOCKER 2026-08-19).
                _nq=np.int64(nq),
                _git_sha=git_provenance(Path(__file__)).commit,
                **arrays)
            # fsync BEFORE the rename, and the directory after it: page
            # cache survives SIGKILL but not node loss, and os.replace is
            # metadata-only. Without this the claim is "kill-atomic", not
            # crash-durable -- and crash durability is what a checkpoint
            # is for (GLM 2026-08-19).
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(_dfd)
        finally:
            os.close(_dfd)
        logger.info("  fv3_duo checkpoint: step %d day %.3f -> %s",
                    step, day, path.name)

    def _fv3_duo_save_checkpoint(self, bundle: dict, step: int,
                                 day: float) -> None:
        """Persist the FULL duo bundle as ONE fp64 ``fv3duo_ckpt_v1`` npz.

        The bundle IS the state: press/nh are persisted verbatim, never
        rebuilt from delp at load (a rebuild risks diverging from the
        certified in-step aliasing).  Atomic: written to a ``.tmp``
        sibling then ``os.replace``d, so a reader never sees a partial
        checkpoint and an interrupted write never poisons a restart.

        Multi-process SPMD: *bundle* is gathered to identical
        host-visible data on every rank first (a collective -- every
        rank must call this), then the ENTIRE write (not just
        ``np.savez``) runs root-gated and rendezvous'd via
        ``_spmd_barrier_on_root_error``, with a best-effort cleanup of
        the unique tmp file on a failed root write (codex MINOR).  The
        single-process arm below is the original code, untouched.
        """
        mcfg = self.model.config
        path = self._output_dir / f"fv3duo_ckpt_step_{step:09d}.npz"
        if not self._is_spmd_multiprocess():
            bundle = self.model.to_flat(bundle)   # identity on faces
            arrays = self._fv3_duo_flatten_bundle(bundle)
            self._fv3_duo_checkpoint_write(
                arrays, mcfg, step, day, len(bundle["q"]), path)
            return

        # ---- multi-process SPMD ----
        bundle = self._fv3_duo_host_faces(bundle)
        arrays = self._fv3_duo_flatten_bundle(bundle)
        nq = len(bundle["q"])
        err = None
        if jax.process_index() == 0:
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            try:
                self._fv3_duo_checkpoint_write(arrays, mcfg, step, day,
                                               nq, path)
            except Exception as exc:
                err = exc
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
        self._spmd_barrier_on_root_error(err)

    def _decode_fv3_duo_checkpoint_arrays(self, path: Path):
        """Read + fully validate ONE ``fv3duo_ckpt_v1`` npz; return
        ``(state, press, q, omga, nh, step, day)``.

        Schema-gated: refuses any npz that does not declare
        ``_schema='fv3duo_ckpt_v1'`` (a cube/lat-lon/MPAS checkpoint is
        a foreign schema this lane must never half-decode), then refuses
        km / resolution / hydrostatic / dt mismatches against the
        CONSTRUCTED model BEFORE touching any array.  Pulled out of
        ``_load_fv3_duo_checkpoint`` verbatim (single-process behavior
        unchanged) so the multi-process arm can wrap the WHOLE decode
        in one try/except before the rendezvous, and so
        ``_broadcast_fv3_duo_bundle`` has a template-shaped return to
        reuse (codex BLOCKER/MAJOR, mp-driver-io design review
        2026-08-27)."""
        mcfg = self.model.config
        schema = self._FV3_DUO_CKPT_SCHEMA
        with np.load(path, allow_pickle=False) as d:
            files = set(d.files)
            if "_schema" not in files or str(d["_schema"]) != schema:
                got = str(d["_schema"]) if "_schema" in files else "absent"
                raise ValueError(
                    f"{path.name} is not an {schema} checkpoint "
                    f"(_schema={got!r}); the fv3_duo lane decodes ONLY "
                    f"its own bundle schema — a cube/lat-lon/MPAS "
                    f"checkpoint cannot restart this lane.")
            for nm, have, want in (
                    ("km", int(d["_km"]), int(mcfg.km)),
                    ("resolution", int(d["_resolution"]),
                     int(self.model.grid.n)),
                    ("hydrostatic", bool(d["_hydrostatic"]),
                     bool(mcfg.hydrostatic))):
                if have != want:
                    raise ValueError(
                        f"fv3_duo checkpoint {path.name} {nm} mismatch: "
                        f"checkpoint {nm}={have!r} vs constructed model "
                        f"{nm}={want!r}; restarting would reinterpret "
                        f"the bundle on the wrong deck. Match the config "
                        f"to the checkpoint.")
            # _zvir: absent on checkpoints written before the moist arm
            # existed -- those were dry by construction (zvir = 0.0)
            zvir_ck = float(d["_zvir"]) if "_zvir" in files else 0.0
            if zvir_ck != float(self.model.zvir):
                raise ValueError(
                    f"fv3_duo checkpoint {path.name} thermodynamic-mode "
                    f"mismatch: checkpoint zvir={zvir_ck!r} vs constructed "
                    f"model zvir={self.model.zvir!r} (moist="
                    f"{mcfg.moist}); resuming would switch humidity "
                    f"feedback on or off mid-run. Match the config to "
                    f"the checkpoint.")
            dt_ck, dt_now = float(d["_dt"]), float(self.config.dycore.dt)
            if dt_ck != dt_now:
                raise ValueError(
                    f"fv3_duo checkpoint {path.name} dt mismatch: "
                    f"checkpoint dt={dt_ck}s vs configured (post-setup) "
                    f"dt={dt_now}s; the total-days step accounting and "
                    f"the bitwise restart contract both assume ONE dt "
                    f"across the chain.")

            def _leaf(nm):
                a = d[nm]
                if a.dtype != np.float64:
                    raise ValueError(
                        f"fv3_duo checkpoint {path.name}: array {nm!r} "
                        f"is {a.dtype}, not float64 — the schema is a "
                        f"bit-exact fp64 round-trip; refusing a lossy "
                        f"leaf.")
                return jnp.asarray(a)

            state = {nm[len("state_"):]: _leaf(nm) for nm in sorted(files)
                     if nm.startswith("state_")}
            press = {nm[len("press_"):]: _leaf(nm) for nm in sorted(files)
                     if nm.startswith("press_")}
            if (not state or "omga" not in files
                    or set(press) != set(self._FV3_DUO_PRESS_KEYS)):
                raise ValueError(
                    f"fv3_duo checkpoint {path.name} is truncated: state "
                    f"keys {sorted(state)}, press keys {sorted(press)} "
                    f"(need every press_* of "
                    f"{sorted(self._FV3_DUO_PRESS_KEYS)}), omga "
                    f"{'present' if 'omga' in files else 'MISSING'}.")
            qi = sorted(int(nm[len("q_"):]) for nm in files
                        if nm.startswith("q_"))
            # CONTIGUITY IS NOT ENOUGH (codex BLOCKER 2026-08-19): the
            # empty set is trivially contiguous, so a checkpoint whose
            # q_* leaves were all dropped loaded as q=[] and RESUMED
            # SUCCESSFULLY -- silent tracer loss, invisible to a bitwise
            # state test because zvir=0 keeps the tracer dynamically
            # passive. The count is part of the deck contract and is
            # stamped in the checkpoint, so check against it.
            # _nq is METADATA, so it is read raw and carries the "_"
            # prefix of the other stamps: routing it through _leaf made
            # the fp64 lossy-leaf guard refuse an int64 count (job
            # 9441043 -- the guard working, my naming wrong).
            n_expect = (int(np.asarray(d["_nq"])) if "_nq" in files
                        else None)
            if n_expect is None:
                raise ValueError(
                    f"fv3_duo checkpoint {path.name} predates the _nq stamp "
                    f"and cannot prove its tracer set is complete; rewrite "
                    f"it with the current writer.")
            if qi != list(range(n_expect)):
                raise ValueError(
                    f"fv3_duo checkpoint {path.name}: tracer leaves {qi} "
                    f"are not exactly q_0..q_{n_expect - 1} as the stamped "
                    f"nq={n_expect} requires (an empty or short set would "
                    f"otherwise resume with tracers silently dropped).")
            q = [_leaf(f"q_{i}") for i in qi]
            omga = _leaf("omga")
            nh = None
            if not mcfg.hydrostatic:
                nh = {nm[len("nh_"):]: _leaf(nm) for nm in sorted(files)
                      if nm.startswith("nh_")}
                if not nh:
                    raise ValueError(
                        f"fv3_duo checkpoint {path.name}: "
                        f"non-hydrostatic restart needs the persisted nh "
                        f"carry (nh_* arrays); found none.")
            step = int(d["_step"])
            day = float(d["_day"])
        return state, press, q, omga, nh, step, day

    def _validate_fv3_duo_broadcast_source(self, src_flat, template,
                                           nq: int, path: Path) -> None:
        """Rank-0-only: prove *src_flat* (the decoded checkpoint,
        flattened) matches EXACTLY the leaf set/shape/dtype
        ``_broadcast_fv3_duo_bundle`` will need, BEFORE any broadcast
        collective runs (codex BLOCKER, mp-driver-io diff review round
        2: an unvalidated mismatch would otherwise surface as a
        confusing ``broadcast_one_to_all`` collective failure on some
        ranks only -- not a clean, loud error on every rank).
        *template* is the model's own fresh IC, flattened the SAME way
        -- the source of truth for every non-``q_`` leaf's shape/dtype
        and for the whole non-``q_`` name set; ``q_`` leaves are sized
        against ``template["q_0"]`` (this deck's tracers share one
        field shape) over the *nq* range."""
        if "q_0" not in template:
            raise RuntimeError(
                "fv3_duo multi-process restart: the template IC has no "
                "q_0 tracer leaf to size the broadcast against.")
        q_shape, q_dtype = template["q_0"].shape, template["q_0"].dtype
        # GLM finding F5 (round 2b): nq was only checked for SELF-
        # consistency (it came from len(dq), so `expect` always agreed
        # with itself) -- never against this DECK's own tracer count.
        # A checkpoint with nq=0 (or any foreign count) would silently
        # pass. Cross-check against the template's own q_* count.
        template_nq = sum(1 for nm in template if nm.startswith("q_"))
        if nq != template_nq:
            raise ValueError(
                f"fv3_duo checkpoint {path.name}: checkpoint nq={nq} "
                f"!= this deck's own tracer count {template_nq} (from "
                f"the constructed model's template IC); refusing a "
                f"foreign/corrupt tracer count.")
        expect = ({nm for nm in template if not nm.startswith("q_")}
                  | {f"q_{i}" for i in range(nq)})
        got = set(src_flat)
        if got != expect:
            raise ValueError(
                f"fv3_duo checkpoint {path.name}: decoded leaf set "
                f"{sorted(got)} != expected {sorted(expect)} (model "
                f"template + stamped nq={nq}); refusing to broadcast "
                f"a mismatched bundle.")
        for nm, arr in src_flat.items():
            want_shape, want_dtype = self._fv3_duo_broadcast_leaf_spec(
                nm, template, q_shape, q_dtype)
            if arr.shape != want_shape or arr.dtype != want_dtype:
                raise ValueError(
                    f"fv3_duo checkpoint {path.name}: leaf {nm!r} is "
                    f"shape={arr.shape} dtype={arr.dtype}, expected "
                    f"{want_shape}/{want_dtype} from the constructed "
                    f"model's own template.")

    def _fv3_duo_broadcast_leaf_spec(self, nm: str, template, q_shape,
                                     q_dtype):
        """The (shape, dtype) a broadcast leaf named *nm* must have --
        ONE definition shared by ``_validate_fv3_duo_broadcast_source``
        and ``_broadcast_fv3_duo_bundle`` (GLM finding F6, round 2b: a
        duplicated ternary in both could drift and let a validated
        shape disagree with the shape actually broadcast)."""
        return ((q_shape, q_dtype) if nm.startswith("q_")
                else (template[nm].shape, template[nm].dtype))

    def _broadcast_fv3_duo_bundle(self, template, hdr, src_flat):
        """Multi-process SPMD: given a header array (step, day, nq) and
        rank 0's ALREADY-VALIDATED flat leaf dict (``src_flat``,
        ``None`` on every other rank -- see
        ``_validate_fv3_duo_broadcast_source``, which the caller runs
        BEFORE this), broadcast to the SAME 7-tuple, identical on
        every rank.

        Two-stage protocol (codex BLOCKER: ``broadcast_one_to_all``
        needs every rank to already know the exact shape/dtype it is
        receiving): first broadcast the header, then broadcast every
        array leaf by name, sized against *template* (built from a
        throwaway ``dcmip16_initial_state()`` -- identical on every
        rank by construction, so no further file access is needed).
        Assumes the CALLER already rendezvous'd on
        ``_spmd_barrier_on_root_error`` so every rank reaches this
        point only after rank 0's decode+validate is known to have
        SUCCEEDED."""
        from jax.experimental import multihost_utils as _mhu
        is_root = jax.process_index() == 0
        mcfg = self.model.config

        hdr = np.asarray(_mhu.broadcast_one_to_all(hdr, is_source=is_root))
        step, day, nq = int(round(hdr[0])), float(hdr[1]), int(round(hdr[2]))

        q_shape, q_dtype = template["q_0"].shape, template["q_0"].dtype
        # SORTED (GLM finding F4, round 2b): every rank must issue the
        # SAME SEQUENCE of broadcast_one_to_all calls in the SAME
        # order (each call is a separate collective) -- plain dict
        # iteration order is deterministic in CPython given identical
        # insertion order, which _fv3_duo_flatten_bundle already
        # guarantees, but an explicit sort removes any doubt rather
        # than relying on that guarantee holding forever.
        names = sorted(nm for nm in template if not nm.startswith("q_"))
        names += [f"q_{i}" for i in range(nq)]

        out: dict[str, np.ndarray] = {}
        for nm in names:
            shape, dtype = self._fv3_duo_broadcast_leaf_spec(
                nm, template, q_shape, q_dtype)
            src = src_flat[nm] if is_root else np.zeros(shape, dtype=dtype)
            out[nm] = np.asarray(
                _mhu.broadcast_one_to_all(src, is_source=is_root))

        state = {nm[len("state_"):]: jnp.asarray(v)
                 for nm, v in out.items() if nm.startswith("state_")}
        press = {nm[len("press_"):]: jnp.asarray(v)
                 for nm, v in out.items() if nm.startswith("press_")}
        q = [jnp.asarray(out[f"q_{i}"]) for i in range(nq)]
        omga = jnp.asarray(out["omga"])
        nh = None
        if not mcfg.hydrostatic:
            nh = {nm[len("nh_"):]: jnp.asarray(v)
                  for nm, v in out.items() if nm.startswith("nh_")}
        return state, press, q, omga, nh, step, day

    def _load_fv3_duo_checkpoint(self, path: Path) -> tuple[int, float]:
        """Load an ``fv3duo_ckpt_v1`` bundle and stage it for restart.

        The staged bundle is consumed by ``_run_fv3_duo`` (total-days
        convention: the resumed job advances ``total - loaded`` steps).

        Multi-process SPMD: ONLY rank 0 reads the file (avoids N-way
        NFS reads and any reliance on cross-node close-to-open
        visibility semantics) -- the file-existence check, the FULL
        single-process decode+validation
        (``_decode_fv3_duo_checkpoint_arrays``, unchanged), and a
        pre-broadcast shape/key validation
        (``_validate_fv3_duo_broadcast_source``) ALL run inside ONE
        try block that EVERY rank enters (not just root -- GLM finding
        F1, mp-driver-io review round 2b: the per-rank template
        construction every rank needs for its broadcast placeholders
        was previously OUTSIDE any try/barrier, so a failure there on
        ANY rank -- root or not -- would crash that rank before it
        ever reached the barrier, hanging every other rank in the
        allgather.  Every rank now feeds ITS OWN local error into the
        SAME ``_spmd_barrier_on_root_error`` rendezvous
        (``_spmd_barrier_on_root_error``'s contract supports a
        non-root local error too, not just root's -- it allgathers
        every rank's flag and re-raises each rank's own exception on
        that rank), so no rank can raise before every rank reaches the
        barrier, and no rank proceeds to a broadcast collective a
        failed rank will never join.  Single-process is the original
        code path, untouched.
        """
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoDynamicsModel,
        )
        if not isinstance(getattr(self, "model", None),
                          FV3DuoDynamicsModel):
            raise RuntimeError(
                "fv3_duo restart: load_checkpoint validates the bundle "
                "against the CONSTRUCTED duo model (km/resolution/"
                "hydrostatic) — call setup() before load_checkpoint.")

        mp = self._is_spmd_multiprocess()
        if not mp:
            if not path.is_file():
                raise FileNotFoundError(
                    f"fv3_duo checkpoint not found (or is a directory): "
                    f"{path}")
            state, press, q, omga, nh, step, day = (
                self._decode_fv3_duo_checkpoint_arrays(path))
            self._fv3_duo_restart_bundle = {
                "state": state, "press": press, "q": q, "omga": omga,
                "nh": nh}
            logger.info("  Loaded fv3_duo checkpoint: step=%d, day=%.3f",
                        step, day)
            self._loaded_checkpoint_step_day = (step, day)
            return step, day

        # ---- multi-process SPMD ----
        # Pre-try inits are raise-free `= None` bindings ONLY (codex
        # BLOCKER, round 3: even `np.zeros(3)` must be inside the try
        # -- a rank-local allocation failure there would exit that
        # rank before the barrier and hang the rest). EVERYTHING that
        # can raise is inside the single try every rank enters.
        template = None
        hdr = None
        src_flat = None
        err = None
        try:
            hdr = np.zeros(3, dtype=np.float64)
            # template: derivable from the CONSTRUCTED model alone
            # (grid + config, identical on every rank by construction)
            # -- built by EVERY rank (needed by all for broadcast
            # placeholders), inside this same try so a per-rank
            # failure here is caught too (GLM F1, round 2b).
            template = self._fv3_duo_flatten_bundle(self._fv3_duo_host_faces(
                self._fv3_duo_fresh_ic()))
            if jax.process_index() == 0:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"fv3_duo checkpoint not found (or is a "
                        f"directory): {path}")
                decoded = self._decode_fv3_duo_checkpoint_arrays(path)
                dstate, dpress, dq, domga, dnh, dstep, dday = decoded
                src_flat = self._fv3_duo_flatten_bundle(
                    {"state": dstate, "press": dpress, "q": dq,
                     "omga": domga, "nh": dnh})
                self._validate_fv3_duo_broadcast_source(
                    src_flat, template, len(dq), path)
                hdr[:] = (float(dstep), dday, float(len(dq)))
        except Exception as exc:
            err = exc
        self._spmd_barrier_on_root_error(err)
        state, press, q, omga, nh, step, day = (
            self._broadcast_fv3_duo_bundle(template, hdr, src_flat))
        self._fv3_duo_restart_bundle = {
            "state": state, "press": press, "q": q, "omga": omga,
            "nh": nh}
        logger.info("  Loaded fv3_duo checkpoint: step=%d, day=%.3f",
                    step, day)
        self._loaded_checkpoint_step_day = (step, day)
        return step, day

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
        at) sets the time origin; ``start_step`` supplies the absolute-step
        value stored in checkpoint metadata AND the phase of the periodic
        diagnostic cadence (so a chain keeps ONE global diagnostic phase —
        #1353).  A chained
        launcher must therefore pass ``--days = TARGET - latest_checkpoint_day``
        (remaining days) for each link — see
        ``run_amip_mpas_100yr_gpu.sbatch``.  The cube/lat-lon paths use
        ``range(start_step, n_steps_total)`` and treat ``--days`` as total;
        do not copy their launcher convention here.
        """
        # --budget-ledger (#1311, MPAS attribution port): the per-process
        # ledger on this lane is PER-COLUMN, (nCells, N_LEDGER, 2) — built for
        # the single-grid-point detonation question, where a global mean
        # washes the signal to ~1e-4 of itself.  Physics rows ride out of the
        # combined physics on HydrostaticTendencies.ledger_rows; the dycore
        # stage rows (dynamics/clips + the closure residual) are filled in
        # ``_step_jit``; the eager loop below accumulates the per-step ledger
        # and writes ``budget_ledger_columns.npz`` (per-column schema — a
        # deliberately DIFFERENT filename from the FV lane's global-row
        # ``budget_ledger.npz``) at the diagnostic cadence.  The
        # historical refusal (SILENTLY-inert flag -> loud refusal -> now a
        # working path) is retained below only for MPI, where the per-rank
        # ledger gather is not yet wired.
        # NOTE: _mpi_world_size EXISTS-but-is-None on the serial lane, so the
        # getattr default never applies — the `or 1` covers the None.
        if (bool(getattr(self.config.output, "budget_ledger", False))
                and ((getattr(self, "_mpi_world_size", 1) or 1) > 1
                     or getattr(self, "_voronoi_layout", None) is not None)):
            raise ValueError(
                "--budget-ledger on the MPAS lane is serial-only for now "
                "(#1311): the per-column ledger rides model.step's eager "
                "side-channel, which the cell-partition MPI step bypasses "
                "(even at -np 1 — the flag would be SILENTLY inert there, "
                "the exact #1311 failure mode this refusal exists to "
                "prevent).  Run the serial lane, or drop the flag."
            )
        import time

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        n_steps_total = int(N_DAYS * 86400.0 / DT)
        # A sub-daily cadence must never round DOWN to zero steps: 0 reads as
        # "diagnostics disabled" at every guard below, so a request for a very
        # fine cadence would silently turn the blow-up check OFF — the opposite
        # of what was asked.  Floor it at one step.
        from legoesm.driver.diagnostics import diagnostic_interval_steps
        DIAG_INTERVAL = diagnostic_interval_steps(
            cfg.output.diag_days, DT, n_steps_total)
        # Phase of the diagnostic cadence.  A real periodic cadence
        # (diag_days > 0) is phased on the ABSOLUTE step so a restart chain
        # keeps ONE global diagnostic clock (#1353: a restored partial flux
        # interval must complete at the boundary it belongs to, not at a
        # fresh job-local multiple).  With diag_days <= 0 the "interval" is
        # the sentinel ``n_steps_total``, which on THIS lane counts only
        # THIS job's steps (``cfg.days`` is per-link here, unlike the
        # cube/lat-lon paths) — phasing that absolutely would move the
        # once-at-the-end diagnostic off the end of the link, so keep it
        # job-local.
        DIAG_PHASE = start_step if cfg.output.diag_days > 0 else 0
        # ``checkpoint_days`` → step cadence.  Enables the 100-yr restart
        # chain (run_amip_mpas_100yr_gpu.sbatch): the driver writes
        # ``checkpoint_day_NNNN.npz`` every cadence and the launcher resumes
        # the next SLURM link from the latest one.  0 ⇒ no checkpointing.
        CHECKPOINT_INTERVAL = (
            int(cfg.output.checkpoint_days * 86400.0 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )
        START_DAY = start_day if start_day is not None else cfg.start_day

        # Whether to feed the CMOR spatial/zonal accumulators at diag cadence.
        # The lean MPAS loop historically fed nothing into them (empty CMOR
        # ``Amon``/``day`` output despite the sidecar SAVE running).  Multi-
        # rank cell partitions feed through the owned-cell -> global
        # GATHER-TO-ROOT in _feed_mpas_cmip_multirank (global regrid weights on
        # rank 0), so no layout in tree is excluded any more.  Computed once — the collector's
        # accumulator handles are stable for the run.
        _diag = getattr(self, "diagnostics", None)
        self._mpas_cmip_feed_on, _diag_wants_cmip = (
            self._mpas_cmip_feed_enabled(_diag))
        self._require_mpas_cmip_feed_supported(
            self._mpas_cmip_feed_on, _diag_wants_cmip)
        _ws = (getattr(self, "_mpi_world_size", 1) or 1)
        if (self._mpas_cmip_feed_on and _is_mpas_cell_partitioned(self)
                and getattr(self, "_mpi_rank", 0) == 0):
            logger.info(
                "  CMOR feed: %d-rank MPAS/Voronoi run — owned cells are "
                "gathered to their global slots each diagnostic interval and "
                "binned on rank 0 with GLOBAL regrid weights (#1517).", _ws)

        # --clear-sky-diag DEGRADES LOUDLY, NEVER SILENTLY (#843): skip the
        # second radiation pass in the configurations that cannot publish it,
        # and say so.  See ``clear_sky_pass_effective`` for the two cases.
        self._mpas_clear_sky_effective, _cs_why = clear_sky_pass_effective(
            clear_sky_diag=bool(getattr(cfg.output, "clear_sky_diag", False)),
            radiation=str(getattr(cfg, "radiation", "none")),
            # The SPATIAL accumulator is the only home of the five new
            # fields, and the feed must actually be allowed to run (every
            # layout in tree is, since #1517).
            spatial_feed_on=bool(
                self._mpas_cmip_feed_on
                and getattr(_diag, "_spatial_monthly", None) is not None),
            # First feed lands ``_rem`` steps in — the SAME arithmetic the
            # flux feed trigger uses.
            # If that is past the end of the run the loop never fires.
            feed_steps_reached=(
                DIAG_INTERVAL > 0
                and (DIAG_INTERVAL - (DIAG_PHASE % DIAG_INTERVAL))
                <= n_steps_total),
        )
        if _cs_why is not None and getattr(self, "_mpi_rank", 0) == 0:
            logger.warning(
                "  --clear-sky-diag is ON but %s.  The clouds-off second "
                "radiation pass is DISABLED rather than run and discarded.",
                _cs_why)

        # Per-step flux means retain the diagnostic interval. State means
        # and daily extremes use the independent hourly hook below.
        self._mpas_sfc_accum = (
            _MPASSfcFluxAccum(expected_steps=DIAG_INTERVAL,
                              window_start_day=START_DAY, dt_s=DT)
            if self._mpas_cmip_feed_on else None)
        if isinstance(self._carry_aux, dict):
            if (self._mpas_sfc_accum is not None
                    and not _is_mpas_cell_partitioned(self)):
                _n_res = self._mpas_sfc_accum.restore(
                    self._carry_aux, resume_day=START_DAY, dt_s=DT)
                if _n_res:
                    logger.info(
                        "  CMOR flux accumulator: resumed a partial diag "
                        "interval from the checkpoint (%d slots)", _n_res)
            else:
                # Either the feed is OFF for this run, or this is a MULTI-rank
                # cell partition — where a staged payload is a GLOBAL-length
                # (serial-written) or RANK-0-local array that this rank must
                # never adopt as its own cells.  Either way the checkpoint may
                # carry a partial interval from a feed-ON link: DROP it.
                # Forwarding it would let a later link resume samples that
                # skip this link's steps — a wrong interval mean (codex-5) —
                # or, multi-rank, attribute one rank's cells to another.
                # Loud: bounded (one diagnostic interval) but real data loss.
                _stale_flux = [k for k in self._carry_aux
                               if k.startswith("cmor_flux")]
                if _stale_flux:
                    for _k in _stale_flux:
                        del self._carry_aux[_k]
                    logger.warning(
                        "  CMOR flux accumulator: DROPPED a partial diag "
                        "interval staged in the checkpoint — the CMOR feed "
                        "is OFF on this run (cmip_output/monthly_means off) "
                        "or this is a multi-rank cell partition (the per-cell "
                        "sums are rank-local), so those samples cannot be "
                        "continued consistently across this link.")
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
                    homogeneous_ice_nucleation=getattr(
                        cfg, "homogeneous_ice_nucleation", False),
                    # The OTHER half of the CLUBB liquid partition: once the
                    # closure hands its diagnosed liquid to the host, the
                    # microphysics must stop manufacturing its own, exactly as
                    # CAM6 switches MG2's residual adjustment off under CLUBB
                    # (micro_mg_cam.F90:668-672).  One flag drives both halves
                    # so they can never be enabled apart -- the microphysics
                    # half alone would leave the model no liquid source at all.
                    liquid_from_closure=cfg._liquid_partition_resolved(),
                    # NOTE: hard_saturation_adjustment is INTENTIONALLY NOT
                    # threaded in-scheme on the MPAS path.  The integration
                    # trial showed the in-scheme placement cannot correct the
                    # MPAS dycore's per-step vertical vapour-transport spike
                    # within the dt window (blew up at day 24), while the same
                    # adjustment applied POST-STEP on the final state is the
                    # proven-stable intervention (40/40 + multi-month pilots).
                    # So on MPAS the flag is applied post-step below; the
                    # coupled path (physics_pipeline) still applies it in-scheme.
                    # The float trigger/heating-cap OVERRIDES are threaded here
                    # so the post-step reads below pick them up (the post-step
                    # drain reads the scheme sub-config, not ExperimentConfig).
                    hard_sat_adjust_threshold=getattr(
                        cfg, "hard_sat_adjust_threshold", None),
                    hard_sat_max_heating_K=getattr(
                        cfg, "hard_sat_max_heating_K", None),
                )
            })
            # Morrison ice-process flat scalars — same shared threading as
            # the FV lane (codex 2026-07-26: this lane previously ignored all
            # five, so a morrison_* override affected FV but not MPAS).
            from legoesm.driver.physics_pipeline import (
                thread_morrison_scalars,
            )
            _micro_cfg = _micro_cfg._replace(**{
                cfg.microphysics: thread_morrison_scalars(
                    cfg, cfg.microphysics,
                    getattr(_micro_cfg, cfg.microphysics)),
            })
        # The cap the lane ACTUALLY runs -- read from the THREADED scheme
        # config (_micro_cfg above), not from the flat field, which a deck's
        # own microphysics sub-config can override.
        _sed_cap_effective = effective_sed_substeps_cap(
            _micro_cfg, cfg.microphysics, cfg)
        # Post-step hard-saturation-adjustment guard (opt-in), MPAS: read its
        # threshold + per-step heating cap from the (unmodified) scheme config,
        # and fail LOUDLY here if a non-warm-rain scheme was requested with the
        # flag (matching apply_microphysics_experiment_flags' contract).
        _hard_sat_on = bool(getattr(cfg, "hard_saturation_adjustment", False))
        _hsub = getattr(_micro_cfg, cfg.microphysics, None)
        _hsub_has_field = (
            _hsub is not None
            and "hard_saturation_adjustment" in getattr(_hsub, "_fields", ()))
        if _hard_sat_on and not _hsub_has_field:
            raise ValueError(
                f"hard_saturation_adjustment=True is not supported by the "
                f"{cfg.microphysics!r} microphysics scheme on the MPAS path; "
                "use a scheme carrying the guard (microphysics/config."
                "HARD_SAT_GUARD_SCHEMES) or drop "
                "--hard-saturation-adjustment."
            )
        # _hsub already carries any --hard-sat-adjust-threshold /
        # --hard-sat-max-heating-k ExperimentConfig overrides: they are
        # threaded through apply_microphysics_experiment_flags above (the
        # day-137 drain-capacity lever; the --params route cannot reach this
        # lane's micro sub-config, which is built here rather than in the
        # flattened atm scalar map).
        _hard_sat_threshold = (
            _hsub.hard_sat_adjust_threshold if _hsub_has_field else None)
        _hard_sat_max_heating = (
            _hsub.hard_sat_max_heating_K if _hsub_has_field else None)
        # Mixed-phase (ice-curve) drain: TTL dehydration fix — gate + land on
        # the blended liquid/ice curve below freezing (validate_strict requires
        # hard_saturation_adjustment + grid_type='mpas' + microphysics='morrison'
        # when set).
        _hard_sat_ice_curve = bool(
            getattr(cfg, "hard_sat_ice_curve", False)) and _hard_sat_on
        # Morrison nucleation crystal mass mi0 = 4/3 pi rho_ci r_nuc^3 [kg] — the
        # mass<->number closure used to seed N_i for the ice the drain deposits
        # (else orphan q_i is deposition-inert and cannot sediment).  Read from
        # the (Morrison) micro sub-config; None disables number seeding.
        # ``_n_i_nuc_max`` [1/m^3] caps the seeded number at Morrison's Cooper
        # ceiling (per mass via rho_air) so a large deposit grows crystals.
        _ice_nuc_mass = None
        _n_i_nuc_max = None
        if _hard_sat_ice_curve and _hsub is not None:
            _rho_ci = getattr(_hsub, "rho_cloud_ice", None)
            _r_nuc = getattr(_hsub, "ice_nuc_radius", None)
            _n_i_nuc_max = getattr(_hsub, "N_i_nuc_max", None)
            if _rho_ci is not None and _r_nuc is not None:
                _ice_nuc_mass = (4.0 / 3.0) * jnp.pi * float(_rho_ci) \
                    * float(_r_nuc) ** 3
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
                # convective_cloud honoured here (2026-07-24): the lagged
                # PhysicsState.conv_precip carry exists on this path.
                cloud_config=_standalone_cloud_config(
                    cfg, _cloud_scheme, allow_convective_cloud=True),
                diurnal_cycle=cfg.diurnal_cycle,
                orbit=_orbit_params,
                # CLUBB sub-grid cloud fraction -> radiation (marine-Sc albedo
                # lever).  Wired even on the MPAS path so a request raises loudly
                # in make_radiation_physics (READ side is hydrostatic-only) rather
                # than being silently ignored; default False is byte-identical.
                use_clubb_cloud_fraction=getattr(
                    cfg, "use_clubb_cloud_fraction", False),
                # Ozone source (default "standard" matches the bare default; a
                # non-standard --ozone-source now flows to MPAS rrtmgp).  The
                # external CMIP6 ozone FILE arrives per-step via the traced
                # ``forcing["o3_vmr"]`` (precedence over this source).
                ozone=OzoneProfileConfig(source=cfg.ozone_source),
                # --clear-sky-diag on the MPAS lane (#843 lean-lane port):
                # clouds-off second radiation pass per radiation step ->
                # sfc_diag slots 10/11 -> CMOR rsutcs/rlutcs.  Previously the
                # flag was a SILENT NO-OP here (the accumulators lived only in
                # _run_compiled).  Default False = byte-identical build.
                # ``_mpas_clear_sky_effective`` is the flag AFTER the
                # publishability check above: it is False (with a loud
                # warning) when nothing could consume the second pass, so a
                # misconfigured run does not pay 2x radiation for a
                # discarded diagnostic.
                clear_sky_diag=bool(self._mpas_clear_sky_effective),
            ),
            # grid_dx_m: SCVT sqrt(mean cell area) [m] — auto-fills Bechtold's
            # IFS ZTAURES resolution factor (codex 2026-07-23 finding A;
            # areaCell is physical, sums to 4*pi*R^2).
            convection=convection_config_for(
                cfg,
                grid_dx_m=float(np.sqrt(np.mean(np.asarray(
                    self._coeff_grid().areaCell))))),
            turbulence=turbulence_config_for(cfg),
            microphysics=_micro_cfg,
            gravity_wave_drag=gwd_config_for(cfg),
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
        # --- MPAS land surface boundary knobs (validate_strict-bounded) -----
        # beta throttles the land-fraction surface humidity inside the MPAS
        # turbulence factory (closure const); the lapse adjusts the T_sfc
        # forcing anchor below.  Both default OFF => byte-identical builds.
        _land_beta = float(getattr(cfg, "mpas_land_beta", 1.0))
        # Phase 2b (#1312): traced per-cell beta_soil replaces the static
        # knob over land; the turbulence factory then needs f_land even when
        # the static knob is at its neutral 1.0.
        _land_beta_soil_on = bool(getattr(cfg, "mpas_land_beta_soil", False))
        _land_lapse_K_m = (
            float(getattr(cfg, "mpas_land_lapse_K_per_km", 0.0)) * 1.0e-3)
        _f_land_cells = None
        if self._f_land is not None:
            _f_land_cells = jnp.asarray(self._f_land).reshape(-1)
        if _land_beta != 1.0 or _land_lapse_K_m > 0.0:
            # Under MPI self._f_land is the RANK-LOCAL (owned+halo) field; an
            # ocean-only rank must not falsely abort a run whose GLOBAL mask
            # has land (codex F-C2).  The logical-OR allreduce is collective
            # and every rank computes the same verdict, so the raise (or
            # not) is deadlock-free.
            _has_land = (_f_land_cells is not None
                         and bool(jnp.any(_f_land_cells > 0.0)))
            if self._voronoi_layout is not None:
                from mpi4py import MPI as _MPI
                _has_land = bool(
                    _MPI.COMM_WORLD.allreduce(_has_land, op=_MPI.LOR))
            if not _has_land:
                raise ValueError(
                    "mpas_land_beta/mpas_land_lapse_K_per_km need a land "
                    "fraction, but none was loaded or it is all-zero "
                    "globally (flat / ocean-only topography) — the knobs "
                    "would be silently inert."
                )
        if _land_beta != 1.0 or _land_lapse_K_m > 0.0:
            logger.info(
                "  MPAS land boundary: lapse=%.2f K/km, beta=%.2f "
                "(f_land mean=%.3f)",
                _land_lapse_K_m * 1.0e3, _land_beta,
                float(jnp.mean(_f_land_cells)),
            )
        _budget_ledger_on = bool(getattr(cfg.output, "budget_ledger", False))
        # Optional vertical band for the ledger.  A column budget cannot see a
        # vertical-REDISTRIBUTION bias -- convection's column water row is
        # exactly zero by construction -- so a band is what lets the table say
        # which process supplies a LAYER.  The SAME weight goes to the physics
        # rows and to the dycore's snapshot-derived rows, or the table stops
        # summing to the column-store change.
        _ledger_band = getattr(cfg.output, "budget_ledger_sigma_band", None)
        _ledger_weight = None
        if _ledger_band is not None:
            if not _budget_ledger_on:
                raise ValueError(
                    "budget_ledger_sigma_band is set but budget_ledger is off; "
                    "the band would select levels of a ledger that is never "
                    "computed.")
            from legoesm.diagnostics.process_ledger import sigma_band_weight
            # HYBRID columns have no single sigma_half: the band would select
            # different pressures under different surface pressures, so the
            # weight would not mean one thing. Refused rather than silently
            # computed on an approximate coordinate.
            if not hasattr(self.sigma, "sigma_half"):
                raise ValueError(
                    "budget_ledger_sigma_band needs a sigma vertical "
                    f"coordinate; this run uses {cfg.grid.vertical_coord!r}, "
                    "whose layer pressures depend on the surface pressure so a "
                    "single sigma band does not select one pressure range.")
            _ledger_weight = sigma_band_weight(
                self.sigma.sigma_half, float(_ledger_band[0]),
                float(_ledger_band[1]))
            logger.info(
                "  budget ledger restricted to sigma band [%.3f, %.3f] "
                "(%.1f%% of the column mass by layer weight)",
                float(_ledger_band[0]), float(_ledger_band[1]),
                100.0 * float(jnp.sum(_ledger_weight * self.sigma.dsigma)
                              / jnp.sum(self.sigma.dsigma)))
        # ---- Physics cadence (CAM6 suite) ----
        # ``physics_update_steps > 1``: the full physics runs every
        # PHYS_UPDATE_STEPS-th step with physics timestep PHYS_UPDATE_STEPS*DT
        # and its whole increment is applied to the state on THAT step (the
        # returned state tendencies are scaled by N so the dycore's
        # ``state + DT * tend`` is the N*DT increment); the in-between steps
        # run a held variant that applies ZERO state tendency and re-publishes
        # the cached diagnostic rates (precip, surface / TOA fluxes, ledger
        # rows) for the per-step accumulators (one more compiled
        # ``model.step``, no per-step Python branch inside the trace).  This
        # is CAM-FV's sequence: physics_update applies ptend*ztodt once,
        # uv3s_update the wind increment once, and the dynamics substeps run
        # unforced (dp_coupling.F90; FV has no ftype -- se_ftype is SE-only).
        # The earlier held-RATE design (2026-09-21) re-applied a rate fixed on
        # the window's start state for N dynamics steps; Morrison's
        # condensation then kept heating a storm column after the dynamics
        # had removed its vapour (+65 K in one window, NaN by day 1 on the
        # L32 + ZM deck).  The cache is seeded from ``jax.eval_shape`` of the
        # full variant before the first step so the carry structure never
        # changes (no retrace).  Radiation cadence is unchanged:
        # ``rad_update_steps`` is validated as a multiple of
        # PHYS_UPDATE_STEPS, so every radiation step is a physics step.  Off
        # by default (cadence "off" = byte-identical).
        # Process ledger: the physics rows are the cached window-mean RATES
        # on every step while the state increment lands on one step, so the
        # per-step residual (ROW_OTHER) carries +(N-1)*R on the physics step
        # and -R on each held step; only sums over WHOLE windows attribute
        # correctly.  Diagnostic intervals are multiples of the cadence on
        # the CAM6 deck (diag_days*768 steps).
        # Post-increment order, as in CAM (qneg3 right after physics_update):
        # the dycore's dry-mass fix, conserving tracer clamp and hard-
        # saturation drain run on the incremented state in the same step.
        # Departures from CAM6 (documented, not emulated):
        #  * order: CAM applies the increment BEFORE its dynamics step
        #    (tphysbc -> coupler -> tphysac -> dynamics); here the step is
        #    dynamics -> physics increment, i.e. the dynamics on the physics
        #    step sees the pre-increment state (one DT of lag, not N).
        #  * coupling: CAM splits the physics around the surface coupler;
        #    here the land tile keeps its own cadence (land_update_seconds),
        #    forced with the cached atmospheric fluxes on every step.
        #  * held steps recompute the analytic Held-Suarez forcing (cheap);
        #    stochastic draws / carried memories advance on physics steps.
        # The physics TIMESTEP is the cadence (CAM: dtime = 1800 s is the
        # physics step): every scheme integrates its prognostic memory
        # (TKE, convective memory, ...) and forms its implicit updates over
        # PHYS_UPDATE_STEPS*DT, and the returned RATES times that span are
        # the one increment CAM applies per physics step.  With DT the
        # memories would advance 1/N of simulated time (codex round 1).
        PHYS_UPDATE_STEPS = cfg.physics_update_steps
        if (not isinstance(PHYS_UPDATE_STEPS, int)
                or isinstance(PHYS_UPDATE_STEPS, bool) or PHYS_UPDATE_STEPS < 1):
            raise ValueError(
                f"physics_update_steps must be an int >= 1, got {PHYS_UPDATE_STEPS!r}")
        _hold_phys = PHYS_UPDATE_STEPS > 1
        _phys_cadence_write = "write" if _hold_phys else "off"
        # The cache is per-job (step 0 of every job is a physics step, i.e.
        # a whole increment is applied at once), so a restart from inside a
        # window would apply a second increment where the uninterrupted run
        # had held steps.  Refuse it; the wallclock exit below only
        # checkpoints on window boundaries so the chain never produces one.
        if _hold_phys and start_step % PHYS_UPDATE_STEPS != 0:
            raise ValueError(
                f"physics_update_steps={PHYS_UPDATE_STEPS}: restart step "
                f"{start_step} is not on a physics-window boundary (the held "
                "cache is not persisted, so resuming mid-window would apply "
                "an extra physics increment); restart from a checkpoint "
                "whose absolute step is a multiple of the cadence")
        # ... and never WRITE one: every periodic checkpoint and the final
        # one must land on a window boundary, else the next link is refused.
        if _hold_phys and (n_steps_total % PHYS_UPDATE_STEPS != 0
                           or (CHECKPOINT_INTERVAL > 0
                               and CHECKPOINT_INTERVAL % PHYS_UPDATE_STEPS != 0)):
            raise ValueError(
                f"physics_update_steps={PHYS_UPDATE_STEPS}: the run length "
                f"({n_steps_total} steps) and the checkpoint interval "
                f"({CHECKPOINT_INTERVAL} steps) must both be multiples of the "
                "cadence so every checkpoint is a resumable window boundary")
        DT_PHYS = DT * PHYS_UPDATE_STEPS
        if cfg.cld_macmic_num_steps > 1:
            logger.info(
                f"  CAM macmic sub-cycle: turbulence + microphysics run "
                f"{cfg.cld_macmic_num_steps} x at "
                f"{DT_PHYS / cfg.cld_macmic_num_steps:.1f} s inside each "
                "physics step, after the convective increment")
        physics_fn = make_physics(phys_cfg, model_type="mpas", dt=DT_PHYS,
                                  budget_ledger_level_weight=_ledger_weight,
                                  column_mesh=_column_mesh,
                                  f_land=(_f_land_cells
                                          if (_land_beta != 1.0
                                              or _land_beta_soil_on)
                                          else None),
                                  land_beta=_land_beta,
                                  budget_ledger=_budget_ledger_on,
                                  physics_cadence=_phys_cadence_write,
                                  physics_cadence_steps=PHYS_UPDATE_STEPS,
                                  cld_macmic_num_steps=cfg.cld_macmic_num_steps)
        if _hold_phys:
            from legoesm.atmosphere.physics.combined import held_physics_variant
            physics_fn_held = held_physics_variant(physics_fn)
        else:
            physics_fn_held = None
        if _hold_phys:
            logger.info(
                "  Physics cadence: full physics every "
                f"{PHYS_UPDATE_STEPS} steps (physics timestep "
                f"{DT_PHYS:.1f} s = {DT_PHYS / 60.0:.1f} min); the whole "
                "physics increment (u, v, T, p_s, tracers) is applied on the "
                "physics step (CAM-FV physics_update), zero in between; "
                "cached precip / surface / TOA rates re-published every step"
            )

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
            make_physics(phys_cfg, model_type="mpas", dt=DT_PHYS,
                         column_mesh=_column_mesh, need_rad=False,
                         f_land=(_f_land_cells
                                 if (_land_beta != 1.0
                                     or _land_beta_soil_on)
                                 else None),
                         land_beta=_land_beta,
                         budget_ledger=_budget_ledger_on,
                         budget_ledger_level_weight=_ledger_weight,
                         physics_cadence=_phys_cadence_write,
                         physics_cadence_steps=PHYS_UPDATE_STEPS,
                         cld_macmic_num_steps=cfg.cld_macmic_num_steps)
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
        # Top-of-atmosphere Rayleigh sponge (#836) on the MPAS edge winds.
        # The hydrostatic MPAS PE has NO sponge of its own (unlike the NH MPAS
        # dycore), and the moist-AMIP slow-onset blowup (day ~10-14 across
        # L20/L30/L40, dt 50-100; 2026-07-23 forensics: top-level (k=0)
        # edge-wind 2Δσ noise doubling per 5 days at localized cells while
        # global norms stay flat) is the classic rigid-lid gravity-wave
        # accumulation signature.  Physics tendencies live on CELL columns and
        # cannot damp EDGE winds, so the sponge is applied as a post-step
        # decay on u — the same #836 sin² profile the lat-lon path folds into
        # fric_decay.  Config-gated: identity when --sponge is off.
        # sigma_full is monotone in both sigma and hybrid coordinates.
        _sponge_decay = None
        if getattr(cfg, "sponge_enabled", False):
            _k_sp_max = cfg.sponge_coeff_per_day / 86400.0
            _sig_top = max(cfg.sponge_sigma_top, 1e-6)  # coeff-ok: /~0 guard
            _sig_full = jnp.asarray(self.sigma.sigma_full)
            _frac = jnp.clip((_sig_top - _sig_full) / _sig_top, 0.0, 1.0)
            _k_sp = _k_sp_max * jnp.sin(0.5 * jnp.pi * _frac) ** 2
            _sponge_decay = jnp.exp(-_k_sp * DT).astype(
                self.state.u.data.dtype
            )
            logger.info(
                "  MPAS top sponge ON: coeff=%.2f/day above sigma=%.3f "
                "(top-level decay %.5f/step)",
                cfg.sponge_coeff_per_day, _sig_top,
                float(_sponge_decay[0]),
            )
        # Horizontal q_v smoothing (post-step, MPAS lane; see the config
        # field note for the full rationale).  The MPAS lane historically had
        # no horizontal moisture smoothing — the missing third suspect behind
        # the cell-scale CWV recharge/discharge speckle.  UNWEIGHTED SCVT del2
        # (∇²) + a q>=0 floor.  Unweighted (NOT mass-weighted) is REQUIRED:
        # the default MPAS vertical coordinate is hybrid, whose surface-layer
        # thickness dp = dA*p_ref + dB*p_s goes <= 0 for p_s below ~2/3 p_ref
        # (~660 hPa, reached over high terrain — Tibet, Andes, Antarctica)
        # because dA < 0 near the surface; a dp-weighted del2 would then
        # divide by a non-positive dp (Inf/NaN).  This conserves (to fp
        # roundoff) the per-level sum_c A_c q_c integral, NOT column water
        # vapour (an explicitly non-conservative filter).  The plain-del2
        # monotonicity factor is
        # geometry-only, so the CFL guard below is EXACT for the applied op.
        _qv_smooth_nu = float(getattr(cfg, "mpas_qv_smooth_del2_m2s", 0.0))
        _qv_smooth_nu4 = float(getattr(cfg, "mpas_qv_smooth_del4_m4s", 0.0))
        _qv_halo_refresh = None
        if _qv_smooth_nu > 0.0 or _qv_smooth_nu4 > 0.0:
            from legoesm.core.operators_voronoi import (
                scalar_del2_cell_cfl_factor,
            )
            if self._voronoi_layout is not None:
                # Distributed lane (#1321 item 2): the del2 stencil reads
                # 1-ring neighbour cells, so refresh the q_v CELL halo right
                # before each filter application — the last in-step exchange
                # ran before the final RK stage, and physics/floors updated
                # owned q_v after it.  With a fresh halo the OWNED-cell
                # stencil sees exactly the serial values (halo = owner
                # values), so owned results match the single-rank filter.
                # Halo cells come out with a partial-ring laplacian; the
                # only consumer before the next step's pre-stage exchange
                # overwrites them is the (optional) hard-saturation drain,
                # which is COLUMN-LOCAL — it touches halo rows but cannot
                # couple them into owned columns (codex-2 narrowing).
                _qv_halo_refresh = (
                    self._voronoi_layout.halo_exchange.exchange_cell_field)
            if self.state.tracers is None or "q_v" not in self.state.tracers:
                raise ValueError(
                    "mpas_qv_smooth_del2_m2s > 0 needs a q_v tracer; this "
                    "run has none (dry configuration) — the knob would be "
                    "silently inert."
                )
            # Explicit-del2 monotonicity/positivity bound: q + nu*dt*lap is a
            # convex combination of stencil values iff nu*dt*g_max <= 1, with
            # g_c = (1/A_c) sum_e dvEdge_e/dcEdge_e (shared helper — same
            # factor the operator obeys, so guard and operator cannot drift).
            # Enforce <= 0.5 (2x safety; keeps the q>=0 floor a no-op to
            # roundoff, so the per-level sum_c A_c q_c integral is conserved —
            # exact arithmetic, to fp roundoff ~1e-7 in fp32).
            # EXACT for the plain form actually applied — no dp-ratio guess.
            _g_max = scalar_del2_cell_cfl_factor(self.grid)
            if self._voronoi_layout is not None:
                # Rank-local mesh under MPI: the monotonicity bound must hold
                # on EVERY cell globally, so take the global max (setup-time
                # guard, diagnostic-only reduction — never in a loss).
                from legoesm.parallel.reductions import global_max_mpi
                _g_max = float(global_max_mpi(jnp.asarray(_g_max)))
            _cfl = _qv_smooth_nu * DT * _g_max
            if _cfl > 0.5:
                raise ValueError(
                    f"mpas_qv_smooth_del2_m2s={_qv_smooth_nu:g} violates the "
                    f"explicit-diffusion monotonicity bound: nu*dt*g_max = "
                    f"{_cfl:.3f} > 0.5 (dt={DT:g}s, mesh g_max={_g_max:.3e} "
                    f"1/m^2). Max stable coefficient here: "
                    f"{0.5 / (DT * _g_max):.3e} m^2/s."
                )
            if _qv_smooth_nu > 0.0:
                logger.info(
                    "  MPAS q_v del2 smoothing ON: nu=%.3g m^2/s "
                    "(nu*dt*g_max=%.4f of 0.5 monotone bound)",
                    _qv_smooth_nu, _cfl,
                )
            if _qv_smooth_nu4 > 0.0:
                # Gershgorin on the Laplacian gives |lambda| <= 2*g_max
                # (diagonal -g_c, off-diagonal row sum g_c), so the
                # biharmonic's spectral radius is at most 4*g_max^2 and
                # forward-Euler stability nu4*|lambda|*dt <= 2 reduces to
                # nu4*dt*g_max^2 <= 0.5.  Measured on the subdivision-6 mesh
                # the Laplacian's radius is 1.364*g_max, so this is
                # conservative by ~2x.  The biharmonic has NO maximum
                # principle, so unlike the del2 case the bound buys STABILITY
                # only; the q>=0 floor can still fire.
                _cfl4 = _qv_smooth_nu4 * DT * _g_max ** 2
                if _cfl4 > 0.5:
                    raise ValueError(
                        f"mpas_qv_smooth_del4_m4s={_qv_smooth_nu4:g} violates "
                        f"the explicit-biharmonic stability bound: "
                        f"nu4*dt*g_max^2 = {_cfl4:.3f} > 0.5 (dt={DT:g}s, mesh "
                        f"g_max={_g_max:.3e} 1/m^2). Max coefficient here: "
                        f"{0.5 / (DT * _g_max ** 2):.3e} m^4/s."
                    )
                logger.info(
                    "  MPAS q_v del4 smoothing ON: nu4=%.3g m^4/s "
                    "(nu4*dt*g_max^2=%.4f of 0.5 stability bound)",
                    _qv_smooth_nu4, _cfl4,
                )
            if _qv_smooth_nu > 0.0 and _qv_smooth_nu4 > 0.0:
                # The two guards above are each sufficient ALONE.  Applied in
                # one explicit update they add, and the two separate budgets
                # would admit a combined forward-Euler amplification of up to
                # 3 (1 from the del2 branch, 2 from the del4 branch) where 2
                # is the limit -- so the sum is bounded here as well (codex
                # review, 2026-09-11).
                _cfl_sum = DT * (2.0 * _qv_smooth_nu * _g_max
                                 + 4.0 * _qv_smooth_nu4 * _g_max ** 2)
                if _cfl_sum > 2.0:
                    raise ValueError(
                        f"the del2 and del4 q_v filters are individually "
                        f"stable but jointly are not: dt*(2*nu*g_max + "
                        f"4*nu4*g_max^2) = {_cfl_sum:.3f} > 2 "
                        f"(nu={_qv_smooth_nu:g} m^2/s, "
                        f"nu4={_qv_smooth_nu4:g} m^4/s, dt={DT:g}s, "
                        f"g_max={_g_max:.3e} 1/m^2)."
                    )
        _sst_forcing = (cfg.radiation != "none" and self.get_sst_sic is not None)
        _sic_day = None            # (nCells,) ice fraction of the last forcing day
        _ice_skin_on = bool(getattr(cfg, "mpas_ice_skin_prognostic", False))
        if _ice_skin_on and not _sst_forcing:
            raise ValueError(
                "mpas_ice_skin_prognostic needs the prescribed SST/SIC "
                "surface forcing (radiation != 'none' and an SST source) — "
                "there is no ice fraction to carry a skin on."
            )
        if _sst_forcing:
            from legoesm.forcing.surface_utils import (
                blend_surface_temperature,
                land_lapse_adjusted_surface_temperature,
            )
            _T_ice = cfg.T_ice
            _ncell = int(self.state.T.data.shape[0])
            # Prognostic ice skin (Semtner zero-layer + slab inertia): the
            # anchor's ice component becomes the per-cell skin array instead
            # of the constant cfg.T_ice.  Seeded at the seawater freezing
            # point; a checkpoint that carried a skin (staged by
            # load_checkpoint) resumes it so 12h chain links do not re-run
            # the multi-week conductive equilibration (C/g ~ 3 weeks at
            # h=2 m) every restart.  A FRESH run's first ~2 months are
            # therefore biased warm toward the old constant-T_ice behaviour
            # (30 d ~6 K, 60 d ~1.7 K, 90 d ~0.4 K residual under a steady
            # -25 W/m^2) — declare the spin-up in experiment metadata; the
            # scorecard's month-3-onward window clears it.
            if _ice_skin_on:
                from legoesm.forcing.surface_utils import (
                    prognostic_ice_skin_temperature,
                )
                _h_ice = float(cfg.mpas_ice_thickness_m)
                _staged_skin = None
                if isinstance(self._carry_aux, dict):
                    _staged_skin = self._carry_aux.get("ice_T_skin")
                if _staged_skin is not None:
                    _skin = jnp.asarray(_staged_skin).reshape(-1)
                    if _skin.shape != (_ncell,):
                        raise ValueError(
                            f"checkpoint ice_T_skin shape {_skin.shape} != "
                            f"(nCells={_ncell},) — mesh mismatch."
                        )
                    # Refuse a non-finite restored skin BEFORE it reaches the
                    # anchor blend: a NaN poisons even open-water cells there
                    # (sic*skin with sic=0 is 0*NaN = NaN), corrupting every
                    # T_sfc, not only ice cells (codex-3).
                    if not bool(jnp.all(jnp.isfinite(_skin))):
                        raise ValueError(
                            "checkpoint ice_T_skin has non-finite values — "
                            "refusing to resume from a corrupt skin.")
                    self._ice_T_skin = _skin
                    logger.info(
                        "  MPAS ice skin: resumed from checkpoint "
                        f"(min {float(jnp.min(_skin)):.1f} K)")
                else:
                    self._ice_T_skin = jnp.full(
                        _ncell, constants.T_freeze_ocean)
                    logger.info(
                        "  MPAS ice skin ON (Semtner zero-layer, h=%.2f m): "
                        "seeded at T_freeze_ocean; conductive relaxation "
                        "C/g ~ 3 weeks (fresh-run spin-up ~2 months)",
                        _h_ice)
            # Land anchor lapse correction: the AMIP loader fills land cells
            # with the NEAREST-OCEAN SST (sea-level temperature); anchoring
            # elevated land at that value overheats its surface by lapse*z.
            # Applied on the land fraction only; z clipped at 0 inside the
            # helper.  Static gate (validate_strict bounds the rate).
            _lapse_z = None
            if _land_lapse_K_m > 0.0:
                _lapse_z = (jnp.asarray(self.state.phis.data).reshape(-1)
                            / constants.g)

            def _blend_T_sfc(_sst, _sic):
                # Blend prescribed SST with the ice component (constant T_ice,
                # or the per-cell prognostic skin READ AT CALL TIME) and apply
                # the land-lapse correction.  Split out of the SST sampling so
                # the per-step loop can RE-ANCHOR from the cached daily SST/SIC
                # against the freshly advanced skin every model step — the
                # physics must consume the CURRENT skin, not the day-start
                # value, or the surface-flux feedback stays daily-lagged
                # (conditional-stability, codex-2 finding 2) and a mid-day
                # restart re-exposes the advanced skin early, branching the run
                # (codex-2 finding 1).
                # Flatten SST/SIC to (nCells,) FIRST so a (nCells,1)-shaped
                # source broadcasts ELEMENTWISE against the (nCells,) skin
                # rather than to (nCells,nCells) — the scalar-T_ice blend
                # tolerated (nCells,1) via a trailing reshape; the array skin
                # must not (codex-4).
                _sst = jnp.asarray(_sst).reshape(-1)
                _sic = jnp.asarray(_sic).reshape(-1)
                _ice_component = (
                    self._ice_T_skin if _ice_skin_on else _T_ice)
                _ts = blend_surface_temperature(
                    _sst, _sic, _ice_component).reshape(-1)
                if _lapse_z is not None:
                    # Cast the storage-dtype (possibly f32) statics to the
                    # anchor dtype so the correction is formed at anchor
                    # precision (mirrors the turbulence-path cast).  Where
                    # f_land and sic overlap (elevated icy coasts) the lapse
                    # cools the blended anchor's land fraction too —
                    # directionally harmless (see codex F5).
                    _ts = land_lapse_adjusted_surface_temperature(
                        _ts, _f_land_cells.astype(_ts.dtype),
                        _lapse_z.astype(_ts.dtype), _land_lapse_K_m)
                return _ts

            # (The former ``_compute_T_sfc(day)`` wrapper — a one-line
            # ``_blend_T_sfc(*get_sst_sic(day))`` — was inlined at its single
            # call site so the daily block can keep the sampled ``sic`` for the
            # surface-albedo blend.)

            # Shape guard once, up front: a non-per-cell get_sst_sic would
            # otherwise surface as an opaque error deep inside the JIT trace.
            # Validate the RAW SST and SIC shapes, NOT only the blended output:
            # with the prognostic skin an (nCells,) ice component, a SCALAR or
            # mis-counted SST/SIC would BROADCAST to (nCells,) and silently pass
            # an output-only check (giving every cell the same SST) — the check
            # the old scalar-T_ice blend used to catch (codex-3).
            _sst0, _sic0 = self.get_sst_sic(START_DAY)
            for _nm, _arr in (("SST", _sst0), ("SIC", _sic0)):
                if tuple(jnp.asarray(_arr).reshape(-1).shape) != (_ncell,):
                    raise ValueError(
                        f"MPAS {_nm} forcing shape "
                        f"{tuple(jnp.asarray(_arr).shape)} has "
                        f"{jnp.asarray(_arr).size} values != nCells={_ncell}; "
                        f"get_sst_sic must return per-cell arrays on the MPAS "
                        f"mesh (grid.grid_lat = latCell).")
            _ts0 = _blend_T_sfc(_sst0, _sic0)
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

        # ---- Surface shortwave albedo boundary condition -------------------
        # Radiation on this lane takes its surface albedo from the traced
        # ``forcing["sfc_albedo"]`` built below.  Before that channel existed
        # every column — land included — was solved at the scalar
        # ``RRTMGPConfig.sfc_albedo`` (0.06, OPEN OCEAN), so 35.6% of the globe
        # reflected shortwave like seawater.  Confirmed in three completed runs
        # whose published rsus/rsds implied an albedo of exactly 0.0600 at both
        # the global min and max.
        #
        # ``_create_physics`` already resolved the static land albedo in
        # precedence order (albedo_land_path -> surfdata -> latitude-vegetation
        # default) into ``self.physics.albedo_land``; the MPAS lane simply
        # never read it.  Reuse that field rather than re-deriving it.
        from legoesm.forcing.surface_utils import (
            blend_surface_property, blended_surface_albedo,
        )
        # dynamic_albedo is a REAL ExperimentConfig option that the FV lane
        # honours (physics_pipeline applies a zenith-angle-dependent ocean
        # albedo). The MPAS blend below is static, so selecting it here would
        # do nothing, silently — the "unknown/unimplemented selection quietly
        # does something else" failure the dispatch-hardening rule exists to
        # stop. Raise until the zenith curve is shared with this lane.
        # Zenith-angle-dependent open-ocean albedo (Briegleb 1992), the same
        # curve the FV lane applies.  A FIXED 0.06 is roughly right for an
        # overhead sun and badly wrong where the sun never rises far: measured
        # on this configuration the poles carry a -20.3 W/m^2 CLEAR-SKY
        # shortwave bias, i.e. the surface reflects too little, and a flat
        # ocean albedo is one of three candidate causes.
        #
        # Cadence: the MPAS surface albedo is assembled ONCE PER FORCING DAY
        # (with the SST/sea-ice sample), not per radiation call, so the cosine
        # used here is the DAYTIME-EFFECTIVE daily mean
        # ``mu = Q_day / (S_0 * f_day)`` -- exactly the quantity the FV lane
        # uses on its non-diurnal path, and consistent with the daily cadence
        # of the field it feeds.  Under a diurnal cycle this is an average over
        # the sunlit day rather than the instantaneous value; that is an
        # approximation of the ALBEDO, not of the insolation, and it is stated
        # rather than hidden.
        _zenith_ocean_alb = bool(getattr(cfg, "dynamic_albedo", False))
        _alb_lat = None
        if _zenith_ocean_alb:
            _, _alb_lat_np = self._owned_p_s_and_lat()
            _alb_lat = jnp.asarray(_alb_lat_np).reshape(-1)
        _albedo_ocean = float(cfg.albedo_ocean)
        _albedo_ice = float(cfg.albedo_ice)
        _albedo_land_static = None
        # RANK-LOCAL land test, computed here rather than reusing ``_has_land``
        # (which only exists inside the mpas_land_beta / lapse guard above and
        # would be undefined for a default config).  Rank-local is the RIGHT
        # scope for the albedo: each rank blends its own cells, and an
        # ocean-only rank correctly needs no land albedo.  No collective here,
        # so a per-rank verdict cannot deadlock.
        _alb_has_land = (_f_land_cells is not None
                         and bool(jnp.any(_f_land_cells > 0.0)))
        # Land fraction used by the albedo blend: None when there is no land,
        # so the helper's "land fraction without a land albedo" guard fires
        # only on a genuine misconfiguration.
        _alb_f_land = _f_land_cells if _alb_has_land else None
        _sea_albedo_day = None   # (nCells,) ocean/ice albedo of the last day
        _sfc_albedo_on = (cfg.radiation != "none")
        if _sfc_albedo_on and _alb_has_land:
            _alb_land = getattr(self.physics, "albedo_land", None)
            if _alb_land is None:
                # Fail loudly: silently reverting to the ocean albedo over land
                # is the defect this block exists to prevent.
                raise ValueError(
                    "MPAS run has a land fraction (f_land > 0) but no land "
                    "surface albedo was resolved. Refusing to apply the OCEAN "
                    f"albedo ({_albedo_ocean:g}) to every land column — that "
                    "under-reflects shortwave over 100% of the land surface. "
                    "Pass --albedo-land-file (a static land-albedo NetCDF), "
                    "or --surfdata, or ensure the latitude-vegetation default "
                    "(legoesm.surface_albedo.land_vegetation_albedo) is built "
                    "in ModelDriver._create_physics."
                )
            _albedo_land_static = jnp.asarray(_alb_land).reshape(-1)
            if _albedo_land_static.shape != (_ncell_alb := int(
                    self.state.T.data.shape[0]),):
                raise ValueError(
                    f"land albedo shape {tuple(_albedo_land_static.shape)} != "
                    f"(nCells={_ncell_alb},) — the albedo map was not "
                    f"regridded onto this MPAS mesh."
                )
            # A NaN or an out-of-range albedo would poison every sunlit column
            # silently (as a heating error, not a crash); refuse it here.
            if not bool(jnp.all(jnp.isfinite(_albedo_land_static))):
                raise ValueError(
                    "land surface albedo contains non-finite values — refusing "
                    "to hand a NaN surface boundary condition to radiation.")
            _alb_lo = float(jnp.min(_albedo_land_static))
            _alb_hi = float(jnp.max(_albedo_land_static))
            if not (0.0 <= _alb_lo and _alb_hi <= 1.0):
                raise ValueError(
                    f"land surface albedo out of physical range "
                    f"[{_alb_lo:.3f}, {_alb_hi:.3f}] — must lie in [0, 1].")
            logger.info(
                "  Surface albedo: ocean=%.3f ice=%.3f land=[%.3f,%.3f] "
                "mean=%.3f (f_land mean=%.3f)",
                _albedo_ocean, _albedo_ice, _alb_lo, _alb_hi,
                float(jnp.mean(_albedo_land_static)),
                float(jnp.mean(_f_land_cells)))
        elif _sfc_albedo_on:
            logger.info(
                "  Surface albedo: ocean=%.3f ice=%.3f (no land fraction)",
                _albedo_ocean, _albedo_ice)

        # ---- Interactive multilayer (Richards) land tile — MPAS port -------
        # Phase-1 coupling contract (tasks/mpas_land_port.md): the land is
        # stepped OUTSIDE the jitted atmosphere step, once per dt, forced by
        # the surface radiation/precip export the physics stashed on the
        # PREVIOUS step (``model._sfc_diag``: sw/lw down refresh on radiation
        # steps, precip every step) plus the current lowest-level state; the
        # land skin temperature feeds back through the existing traced
        # ``forcing["T_sfc"]`` channel (explicit flux coupling, one-step lag,
        # no retrace).  The static ``mpas_land_beta`` humidity throttle still
        # applies inside the turbulence factory; a traced per-cell beta is the
        # phase-2b follow-up (see the port plan).
        _land_ml_on = (bool(getattr(cfg, "use_multilayer_land", False))
                       and self._land_ml_state is not None)
        if _land_beta_soil_on and not bool(
                getattr(cfg, "use_multilayer_land", False)):
            # CONFIG-ONLY test, deliberately: it is the same on every rank, so
            # the raise is symmetric and needs no collective.
            #
            # This guard used to test whether THIS rank had built a soil column,
            # which is wrong under cell-partition MPI — an ocean-only rank
            # legitimately has none while its neighbours do, so it aborted alone.
            # Two attempts at a cross-rank vote made it worse: the first was
            # called only by the land-less ranks (deadlock), the second by every
            # rank but AFTER a rank-local land-albedo raise that can kill one
            # rank while the others block in the vote (codex rounds 5 and 6).
            # There is no collective to get wrong here: validate_strict already
            # guarantees mpas_land_beta_soil implies use_multilayer_land, and a
            # globally landless run is refused there too, so the only case the
            # old state test could still catch was the legitimate ocean-only
            # rank.  It now publishes nothing, which is what it should do.
            raise ValueError(
                "mpas_land_beta_soil=True requires the interactive multilayer "
                "land on the MPAS lane (use_multilayer_land); without it there "
                "is no soil moisture to derive beta_soil from — the flag would "
                "be silently inert."
            )
        _land_step_fn = None
        _land_T_skin = None            # (nCells,) land skin T of the last step
        _land_albedo_cells = None      # (nCells,) land albedo of the last step
        _land_beta_fn = None           # jitted land-state -> per-cell beta_soil
        _land_beta_cells = None        # (nCells,) traced beta of the last step
        _land_qsfc_cells = None        # (nCells,) land's solved q_sfc, last step
        _land_shflx_cells = None       # (nCells,) land's own sensible flux
        _land_lhflx_cells = None       # (nCells,) land's own latent flux
        _land_a2s_sum = None           # cadence: running forcing sum
        _land_a2s_n = 0                # cadence: steps accumulated
        if _land_ml_on:
            if not _sst_forcing:
                raise ValueError(
                    "use_multilayer_land on the MPAS lane requires the SST "
                    "surface-forcing channel (radiation != 'none' with SST "
                    "data): the land skin temperature feeds back through "
                    "forcing['T_sfc'], which only exists on that path.")
            if _f_land_cells is None:
                raise ValueError(
                    "use_multilayer_land on the MPAS lane requires a land "
                    "fraction (--topography / --land-mask-file); none was "
                    "loaded — the land tile would be silently inert.")
            # Distributed is supported (#1321).  No column scatter is needed:
            # ``_create_grid`` installs the rank-local mesh BEFORE
            # ``_create_physics`` runs, and ``_setup_multilayer_land`` sizes
            # its columns from ``self.grid.latCell``, so the soil columns are
            # built rank-local already.  (The guard that used to sit here said
            # the opposite — "the setup built them on the full mesh" — which
            # stopped being true when grid creation moved ahead of physics.)
            # What WAS missing is the downwelling-radiation slots the land
            # forcing reads; ``make_voronoi_mpi_step`` now publishes the full
            # 10-slot contract, and this asserts it rather than letting
            # ``_marshal_land_forcing`` return None and the soil silently
            # never advance.
            from legoesm.land.multilayer_land import step_multilayer_land
            from legoesm.core.coupling_fields import AtmToSurface
            from legoesm.grids.voronoi import reconstruct_cell_velocity
            _lml_cfg = self.physics.land_ml_cfg
            _lml_params = self.physics.land_ml_params
            _lml_lat = self.physics.land_ml_lat
            _lml_umin = float(getattr(self.physics, "land_ml_u_min", 1.0))
            # PRESCRIBED leaf carbon (fixed LAI).  Without it the coupled
            # stomatal dispatch falls back to Jarvis even when the config asks
            # for Farquhar, so the baked canopy conductance would be inert on
            # exactly the lane the AMIP campaign runs.  None on every other
            # configuration, which keeps those runs byte-identical.
            _lml_carbon = getattr(self.physics, "land_ml_carbon", None)

            # Land fraction as a closure constant of the compiled land step,
            # for the land-weighted held count below.
            _f_land_cols = jnp.asarray(self._f_land).reshape(-1)

            # --- Packed land columns (2026-08-25 step-cost profile: the tile
            # was ~58% of wall time while solving EVERY column, ocean
            # included, whose output every consumer blends away by f_land).
            # Gather the f_land > 0 columns — the mask is a compile-time
            # constant, so the gather is static — solve only those, scatter
            # back.  The land STATE stays full-grid between calls, so
            # checkpoints, restarts, MPI ownership and the beta_soil reader
            # are byte-identical.  f_land > 0, not > 0.5: fractional coastal
            # cells' outputs ARE consumed.  Scatter fills are 0.0, never NaN
            # (0 * NaN would contaminate the f_land blends).  clm_ml keeps
            # the full-grid path: its 1-based canopy arrays and GridInfo do
            # not repack.
            _land_pack_idx_np = np.nonzero(
                np.asarray(self._f_land).reshape(-1) > 0.0)[0]
            _land_pack_on = (
                str(getattr(cfg, "land_surface_scheme", "")) != "clm_ml"
                and 0 < _land_pack_idx_np.size < _f_land_cols.shape[0])
            _land_ncol_full = int(_f_land_cols.shape[0])
            _land_pack_idx = jnp.asarray(_land_pack_idx_np)

            from legoesm.land.multilayer_land import (
                gather_land_columns, scatter_land_columns, scatter_cells)

            def _land_pack(tree):
                return gather_land_columns(
                    tree, _land_pack_idx, _land_ncol_full)

            if _land_pack_on:
                # The CONFIG packs too: the coupled tile carries per-column
                # arrays inside it (spatial hydraulics theta_sat/theta_r are
                # (ncol, 1) — the accel_pack A/B crashed on exactly that
                # leaf when only params/lat were packed).
                _lml_cfg_p = _land_pack(_lml_cfg)
                _lml_params_p = _land_pack(_lml_params)
                _lml_lat_p = _land_pack(_lml_lat)
                _lml_carbon_p = (_land_pack(_lml_carbon)
                                 if _lml_carbon is not None else None)
                _f_land_cols_p = _f_land_cols[_land_pack_idx]
            else:
                _lml_cfg_p = _lml_cfg
                _lml_params_p, _lml_lat_p = _lml_params, _lml_lat
                _lml_carbon_p = _lml_carbon
                _f_land_cols_p = _f_land_cols

            # --- Land cadence: call the tile every _LAND_K host steps with
            # the forcing MEANED over the interval (a mean rate times the
            # tile's DT_LAND conserves the interval's precip mass and
            # radiant energy) and its fluxes/skin held in between — the
            # blend/consumer guards below are already None-tolerant, holding
            # is what they do before the first call today.  Time-based knob:
            # the interval, not the count, is what the canopy certifies, so
            # a dt change cannot silently stretch it.
            _LAND_K = (max(1, int(round(
                float(cfg.land_update_seconds) / DT)))
                if float(getattr(cfg, "land_update_seconds", 0.0)) > 0.0
                else 1)
            DT_LAND = _LAND_K * DT

            def _make_land_step(_dt_land):
              @jax.jit
              def _land_step(land_state, a2s, doy):
                from legoesm.land.multilayer_land import (
                    step_multilayer_land_with_diagnostics)
                _state_in = (_land_pack(land_state) if _land_pack_on
                             else land_state)
                _a2s_in = _land_pack(a2s) if _land_pack_on else a2s
                new_state, resp, _carbon, _sfc = (
                    step_multilayer_land_with_diagnostics(
                        _state_in, _a2s_in, _lml_cfg_p, _lml_umin, _dt_land,
                        lat=_lml_lat_p, doy=doy, land_params=_lml_params_p,
                        carbon_state=_lml_carbon_p))
                # resp.albedo is the END-OF-STEP land albedo, already
                # snow-brightened by the tile (band_albedo / snow_albedo) and
                # dry-soil-brightened.  It used to be discarded here, so the
                # land tile's snow-albedo feedback never reached radiation.
                # resp.q_surface is the scheme's SOLVED boundary humidity;
                # resp.shflx / resp.lhflx are the fluxes its OWN energy
                # balance closed with.  The FLUXES are the coupling now: the
                # humidity handoff was measured insufficient (the canopy's
                # boundary humidity sits close to the air by construction, so
                # the atmosphere re-applying its own exchange coefficient
                # delivered ~a tenth of the solved flux -- Amazon latent heat
                # 78 W/m2 offline vs 7 coupled, land 10 K cold in 30 days).
                # A column whose surface solve was rejected is HELD: its
                # state is reverted and it conserves neither energy nor water
                # over that step. The land step is inside a compiled region
                # where a print is not available on a GPU-only runtime, so the
                # count comes out here and the loop below reports it. Without
                # that, a run whose land is quietly frozen somewhere looks
                # exactly like a healthy one.
                _n_held = (_sfc.n_held if _sfc.n_held is not None
                           else jnp.zeros((), jnp.int32))
                # The raw count spans EVERY mesh column, but ocean columns run
                # the land solve on placeholder forcing and their output is
                # discarded by the land-fraction weighting — a held ocean
                # column costs nothing physically.  Count the LAND-weighted
                # holds separately, so the log can tell a frozen continent
                # from noise on discarded columns (codex, 2026-08-23: the
                # undivided counter read as half the mesh held when the
                # physically-meaningful share was unknown).
                _held_mask = getattr(_sfc, "held", None)
                _n_held_land = (
                    jnp.sum((jnp.asarray(_held_mask).reshape(-1)
                             & (jnp.asarray(_f_land_cols_p) > 0.5))
                            .astype(jnp.int32))
                    if _held_mask is not None
                    else jnp.zeros((), jnp.int32))
                if _land_pack_on:
                    # Scatter the advanced columns back into the full-grid
                    # state (ocean columns keep their frozen init values,
                    # exactly what the unpacked solve left them at after the
                    # f_land blend discarded its work) and the five consumed
                    # response fields into zero-filled cell arrays.
                    new_state = scatter_land_columns(
                        land_state, new_state,
                        _land_pack_idx, _land_ncol_full)
                    return ((new_state,)
                            + tuple(scatter_cells(
                                o, _land_pack_idx, _land_ncol_full)
                                for o in (
                                    resp.T_sfc, resp.albedo, resp.q_surface,
                                    resp.shflx, resp.lhflx))
                            + (_n_held, _n_held_land))
                return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,
                        resp.shflx, resp.lhflx, _n_held, _n_held_land)
              return _land_step

            _land_step_fn = _make_land_step(DT_LAND)
            # Bootstrap variant: one host step's forcing, tile advanced by
            # DT — used only for the first call after a (re)start so the
            # surface blend never runs land-free for a whole interval.
            _land_step_boot_fn = (
                _land_step_fn if _LAND_K == 1 else _make_land_step(DT))

            # Phase 2b (#1312): per-cell root-zone beta_soil -> the traced
            # ``forcing["beta_land"]`` the turbulence surface flux consumes.
            # Same helper (land_tile_beta_soil) and thresholds the coupled
            # pipeline's land tile applies — one formula, one place.
            _land_beta_fn = None
            if _land_beta_soil_on:
                from legoesm.land.multilayer_land import land_tile_beta_soil

                @jax.jit
                def _land_beta_fn(land_state):
                    return jnp.clip(
                        land_tile_beta_soil(
                            land_state.theta_soil, _lml_cfg, _lml_params),
                        0.0, 1.0)

                # The land's SOLVED surface humidity is handed to the
                # turbulence VERBATIM via forcing["q_sfc_land"] -- review
                # refuted the previous effective-beta round trip here (its
                # saturation anchors disagreed by 0.3-2.5 % and its clip could
                # only shrink the flux), so there is no inversion any more.

                if _land_beta != 1.0:
                    logger.info(
                        "  mpas_land_beta_soil: traced per-cell beta_soil "
                        "REPLACES the static mpas_land_beta=%.2f over land",
                        _land_beta)

            # Held-column accounting. The per-step count stays on the DEVICE
            # and is read only at the warning cadence, so the step loop pays no
            # synchronisation for it in between -- it is not free, it is as
            # frequent as the other post-step warnings.
            #
            # The device counter is RESET at every read and the running totals
            # are kept on the host. A 32-bit counter accumulating a whole run's
            # column-steps overflows: ten thousand columns at a seventy-five
            # second timestep reach two billion in about half a simulated year,
            # after which the warning this exists to guarantee would stop
            # firing precisely when holds had become systemic.
            _land_n_held_accum = jnp.zeros((), jnp.int32)
            _land_n_held_land_accum = jnp.zeros((), jnp.int32)
            _land_n_held_steps_accum = jnp.zeros((), jnp.int32)
            self._land_n_held_total = 0
            self._land_n_held_steps = 0

            from legoesm.land.forcing.solar import cos_solar_zenith as _csz
            _lml_lon = jnp.asarray(self.grid.lonCell).reshape(-1)

            @jax.jit
            def _cos_zen_fn(doy, hour_utc):
                return _csz(_lml_lat, _lml_lon, doy, hour_utc)

            def _marshal_land_forcing():
                """AtmToSurface from the last radiation export + current state.

                Returns None until the first radiation solve has populated the
                downwelling fluxes (step 0 pre-physics) — the land holds its
                initial state for that single step.  Mirrors the coupled
                pipeline's ``_step_multilayer_land_tile`` marshalling
                (physics_pipeline.py) field-for-field.
                """
                _sd = getattr(self.model, "_sfc_diag", None)
                # Slot contract (primitive_eq_mpas #1318 CMOR feed +
                # land port union): (sw_net, lw_net, precip, lw_up_toa,
                # sw_up_toa, sw_down_toa, shflx, lhflx,
                # sw_down_sfc, lw_down_sfc[, sw_up_toa_clr, lw_up_toa_clr])
                # — the DOWNWELLING surface fluxes the land needs are slots
                # 8/9 (NOT 3/4, which are TOA fields; 10/11 are the #843
                # clear-sky TOA pair, present only with --clear-sky-diag).
                if (_sd is None or len(_sd) < 10
                        or _sd[8] is None or _sd[9] is None):
                    return None
                sw_down = jnp.asarray(_sd[8].data).reshape(-1)
                lw_down = jnp.asarray(_sd[9].data).reshape(-1)
                precip = (jnp.asarray(_sd[2].data).reshape(-1)
                          if _sd[2] is not None else jnp.zeros_like(sw_down))
                T_air = self.state.T.data[:, -1]
                _qv_tr = (self.state.tracers or {}).get("q_v")
                q_air = (_qv_tr.data[:, -1] if _qv_tr is not None
                         else jnp.zeros_like(T_air))
                p_s = jnp.asarray(self.state.p_s.data).reshape(-1)
                u_c, v_c = reconstruct_cell_velocity(
                    self.state.u.data[:, -1], self.grid)
                # Same conventions as the coupled tile: p_lowest ~ 0.99 p_s,
                # ideal-gas rho at the lowest level, snow split at T_freeze.
                # The zenith is the REAL per-cell sun (same doy/seconds the
                # radiation uses this step): it was a fixed 0.5 when only the
                # bulk scheme ran here (which never reads it), but the two-leaf
                # canopy now runs on this lane and its radiation partitioning
                # is zenith-driven -- a fixed sun would give the canopy neither
                # a diurnal cycle nor night.
                from legoesm.core.coupling_fields import lowest_level_height
                z_lowest = lowest_level_height(
                    T_air, self.sigma.pressure_at_half(p_s),
                    self.sigma.pressure_at_full(p_s))
                return AtmToSurface(
                    z_lowest=z_lowest,
                    sw_down=sw_down, lw_down=lw_down,
                    precip_total=precip,
                    precip_snow=jnp.where(
                        T_air < constants.T_freeze, precip, 0.0),
                    T_lowest=T_air, q_lowest=q_air,
                    u_lowest=u_c, v_lowest=v_c,
                    p_lowest=0.99 * p_s, p_surface=p_s,
                    rho_lowest=p_s / (constants.R_d * T_air),
                    cos_zenith=_cos_zen_fn(
                        jnp.asarray(_doy, dtype=jnp.float64),
                        jnp.asarray(_sod, dtype=jnp.float64) / 3600.0),
                    co2_ppmv=jnp.full_like(T_air, float(
                        getattr(cfg, "co2_ppmv", 412.0))),
                    has_radiation=jnp.ones_like(T_air),
                    has_precipitation=jnp.ones_like(T_air),
                )
            logger.info(
                "  Interactive multilayer land (MPAS): %d columns, "
                "f_land mean=%.3f — skin T -> forcing['T_sfc'] blend, "
                "one-step-lag explicit coupling; packed=%s (%d land "
                "columns solved), cadence every %d step(s) (DT_land=%.1f s)",
                int(_lml_lat.shape[0]), float(jnp.mean(_f_land_cells)),
                _land_pack_on, int(_land_pack_idx_np.size),
                _LAND_K, DT_LAND,
            )

        # Wrap with Held-Suarez forcing when enabled.  Factored into a helper
        # so the SAME wrap applies to BOTH radiation sub-cycle variants (the
        # full ``physics_fn`` and the held ``physics_fn_norad``).
        if cfg.held_suarez_forcing:
            from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
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
                    # HS adds only the 4 DYNAMICS tendencies (du/dT/dp_s/dphis);
                    # ``_replace`` overrides just those and PRESERVES every
                    # diagnostic field on the radiation tendency — sw/lw net,
                    # precip, AND the CMOR TOA/turbulent-flux extras (sw_up_toa,
                    # lw_up_toa, sw_down_toa, shflx_sfc, lhflx_sfc). An explicit
                    # constructor that enumerated the forwarded fields silently
                    # dropped whichever were not listed (it lost sw/lw net + precip
                    # once already, codex); _replace makes the repack field-count
                    # agnostic so a future diagnostic cannot regress here.
                    summed = rrtmgp_tend._replace(
                        du_dt=rrtmgp_tend.du_dt.replace(
                            data=rrtmgp_tend.du_dt.data + hs_tend.du_dt.data),
                        dT_dt=rrtmgp_tend.dT_dt.replace(
                            data=rrtmgp_tend.dT_dt.data + hs_tend.dT_dt.data),
                        dp_s_dt=rrtmgp_tend.dp_s_dt.replace(
                            data=rrtmgp_tend.dp_s_dt.data + hs_tend.dp_s_dt.data),
                        dphis_dt=rrtmgp_tend.dphis_dt.replace(
                            data=rrtmgp_tend.dphis_dt.data + hs_tend.dphis_dt.data),
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
                if (physics_config_requires_phys_state(phys_cfg)
                        or getattr(_rrtmgp_fn, "_requires_phys_state", False)):
                    _hs_physics_fn._requires_phys_state = True
                return _hs_physics_fn

            physics_fn = _wrap_hs(physics_fn)
            if physics_fn_norad is not None:
                physics_fn_norad = _wrap_hs(physics_fn_norad)
            # Physics cadence: HS forcing is analytic and cheap, so it is
            # recomputed on held steps (the cache holds physics only).
            if physics_fn_held is not None:
                physics_fn_held = _wrap_hs(physics_fn_held)

        run_status = "COMPLETED"
        logger.info(f"Starting MPAS: {n_steps_total} steps, {N_DAYS} days "
                    f"(from day {START_DAY:.1f})")

        # Light-weight time series — see _run_spectral for the rationale
        # (the MPAS path also bypasses the unified DiagnosticCollector).
        # ``CWV`` (column water vapor) is recorded on moist runs (NaN on dry);
        # ``_save_lightweight_timeseries`` already persists a ``CWV`` channel
        # and ``validate_amip_run.py`` checks its bounds.
        # ``moisture_residual`` (E - P - dW/dt, mm/day) rides here because this
        # lane writes its OWN series and never calls the collector's saver --
        # which is why feeding the tracker was not enough on its own: it
        # updated in memory and was then discarded, leaving the published
        # residual blank exactly as before. Sampled from the tracker at each
        # daily write, so the series is as long as the others.
        _ts: dict[str, list] = {
            "days": [], "T_atm": [], "T_min": [], "T_max": [],
            "max_wind": [], "dry_mass_ps": [], "T_finite": [], "CWV": [],
            "moisture_residual": [],
            # Energy-budget series (#1354/#1515): the MPAS lane runs the
            # EnergyBudgetTracker per diag step from self.model._sfc_diag.
            "energy_toa_net": [], "energy_dE_dt": [], "energy_residual": [],
            "sw_net_sfc": [], "lw_net_sfc": [], "hfss": [], "hfls": [],
            # 1.0 = the seven energy channels above are diagnostic-INTERVAL
            # MEANS; 0.0 = end-of-interval snapshots, which alias the diurnal
            # cycle of the land-dominated turbulent fluxes (#1354/#1353).
            "energy_flux_interval_mean": [],
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
        # Does the convection scheme want the diurnal-cycle CAPE subtraction?
        # Resolved once: a static Python bool, so the per-step seeding below is
        # a trace-time branch and cannot retrace.
        _capdcycl_on = bool(getattr(
            getattr(self.config, "convection_config", None), "use_ifs_capdcycl",
            False)) or bool(getattr(self.config, "bechtold_use_ifs_capdcycl",
                                    False))
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
        _ext_forcing = _external_forcing_active(
            cfg.radiation in ("rrtmg", "rrtmgp"),
            self._ozone_ext_active, self._aerosol_active,
            self._aerosol_lw_active, self._ghg_active, bool(self._experiment),
        )
        # Transient solar file (CMIP6 TSI + optional 14-band spectral):
        # sampled DAILY like SST/ozone, threaded as traced
        # ``forcing["tsi"]``/["solar_spectral_fraction"] into the radiation
        # physics (the previously-inert channel of the MPAS deck — the
        # run_amip channel table used to print "solar file not threaded").
        # Spectral weights only reach the rrtmg/rrtmgp solver; gray consumes
        # the TSI scaling alone.
        _solar_ext = (cfg.radiation != "none"
                      and cfg.solar_source in ("file", "spectral_file"))
        _solar_spectral = (_solar_ext
                           and cfg.solar_source == "spectral_file"
                           and cfg.radiation in ("rrtmg", "rrtmgp"))
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
            _NEW_OPTIONAL_PS_FIELDS = frozenset({
                "aerosol_number",
                # conv_precip (2026-07-24): the Slingo-1987-inspired surrogate lag carry; zero-seed
                # is the correct pre-feature state (no convective cloud was
                # diagnosed before it existed).
                "conv_precip",
                # conv_heating (2026-09-21): the Beres convective-source lag
                # carry; zero-seed = no convective heating before the feature.
                "conv_heating",
                # conv_mass_flux_up / conv_icwmr (2026-09-21): the CAM6
                # deep-convective cloud-fraction lag carries; zero-seed is
                # the correct pre-feature state (deepcu exactly 0).
                "conv_mass_flux_up",
                "conv_icwmr",
            })
            if _any_physstate:
                from legoesm.atmosphere.physics.physics_state import (
                    PHYSSTATE_INPUT_FIELDS,
                )
                # ``col_index`` is exempt from the completeness contract:
                # it is CONSTANT derivable identity data (arange(ncol),
                # never evolved), added 2026-07 — checkpoints written
                # before then legitimately lack it, and the fresh seed's
                # arange is byte-identical to what the save would have
                # stored.  The ``PHYSSTATE_INPUT_FIELDS`` (the dyn_tendency_*
                # pair and the prescribed surface-flux overrides) are
                # likewise exempt: per-step driver INPUTS, never persisted
                # (the save skips their None), re-seeded fresh.  Every
                # EVOLVING field stays mandatory.
                _exempt_ps = {"col_index"} | PHYSSTATE_INPUT_FIELDS
                _missing = [f for f in _phys_state._fields
                            if f not in _present_fields
                            and f not in _exempt_ps]
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
            from legoesm.atmosphere.physics.physics_state import (
                PHYSSTATE_INPUT_FIELDS,
            )
            _restored_ps = {}
            for _k, _v in self._carry_aux.items():
                if not _k.startswith("physstate_"):
                    continue
                _name = _k[len("physstate_"):]
                if _name not in _phys_state._fields:
                    continue
                # Per-step INPUT fields are never restored: their fresh seed is
                # None (no ``.shape``/``.dtype``), and a checkpoint that
                # nonetheless carries one (written by an older build, or hand
                # edited) must be dropped BEFORE the shape/dtype validation
                # below — a driver recomputes them each step.
                if _name in PHYSSTATE_INPUT_FIELDS:
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
        _mpi_step_held = None
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
            _mpi_step_held = (
                make_voronoi_mpi_step(
                    self.model, self._voronoi_layout, self.model.sigma_coord,
                    config=self.model.config, physics_fn=physics_fn_held,
                    return_phys_state=True,
                )
                if physics_fn_held is not None else None
            )
            logger.info(
                "  MPAS MPI step active (rank %d/%d)",
                self._voronoi_layout.rank, self._voronoi_layout.n_ranks,
            )

        _forcing_daily: dict = {}
        # Phase 2b (#1312): seed the traced per-cell beta from the CURRENT
        # land state (cold-start or checkpoint-restored — beta is a pure
        # function of the restored soil moisture, so restarts are exact
        # without a new checkpoint field).  Seeding BEFORE the loop keeps
        # the ``beta_land`` forcing key structurally present from step 0 —
        # adding it mid-run would change the forcing pytree and retrace
        # ``model.step``.
        if _land_beta_fn is not None:
            _land_beta_cells = _land_beta_fn(self._land_ml_state)
            # Seed the SOLVED-humidity channel too (the forcing pytree must be
            # structurally stable from step 0 -- adding the key mid-run would
            # retrace model.step).  Before the first land step there is no
            # solved humidity, so reconstruct the bounded gradient form from
            # the seed beta at the model's own lowest level; the land's real
            # answer replaces it from step 1.
            from legoesm.thermo import saturation_mixing_ratio as _satmr0
            _qv_tr0 = (self.state.tracers or {}).get("q_v")
            if _qv_tr0 is not None:
                _q_air0 = jnp.asarray(_qv_tr0.data[:, -1]).reshape(-1)
                _ph0 = self.sigma.pressure_at_half(
                    jnp.asarray(self.state.p_s.data).reshape(-1))
                _p_low0 = 0.5 * (_ph0[..., -1] + _ph0[..., -2])
                _T_land0 = jnp.asarray(
                    self._land_ml_state.T_soil[:, 0]).reshape(-1)
                _qsat0 = _satmr0(_T_land0, _p_low0)
                _land_qsfc_cells = (
                    _q_air0 + _land_beta_cells * (_qsat0 - _q_air0))
                # Flux-channel seeds: zero exchange for the one step before
                # the land produces its first solved fluxes.
                _land_shflx_cells = jnp.zeros_like(_q_air0)
                _land_lhflx_cells = jnp.zeros_like(_q_air0)
        # Current forcing day's SST/SIC, cached at each daily boundary for the
        # per-step ice-skin advance AND per-step T_sfc re-anchor (None until
        # the first boundary / when the skin feature is off).
        _ice_sst_cur = None
        _ice_sic_cur = None
        # Budget-ledger accumulator (#1311): sum of the per-step per-column
        # ledgers over the current diagnostic interval; divided by the step
        # count at emission -> MEAN RATES, then reset.  Device arrays until
        # the (already host-syncing) diag block reads them.
        _led_accum = None
        _led_nsteps = 0
        # Window maximum of the microphysics' required CFL sedimentation
        # sub-step count (device array; reported and reset per window, and
        # flushed after the loop so a partial window is not lost).
        _sed_req_window = None
        from legoesm.forcing.time_utils import daily_forcing_bucket
        for step in range(n_steps_total):
            # Enter the daily-boundary block also when a coupler segment_callback
            # is present, so the ocean/land still steps even on a coupled run with
            # radiation=none (where _sst_forcing is False) — else coupling would
            # silently freeze. SST re-sampling below stays gated on _sst_forcing.
            if (_sst_forcing or _ext_forcing or _solar_ext
                    or self._segment_callback is not None):
                _force_day = START_DAY + step * DT / 86400.0
                # floor, not int() — see daily_forcing_bucket (negative
                # fractional days land in the wrong bucket under
                # truncation; day_to_calendar already handles negative
                # days via modulo).
                _fd_int = daily_forcing_bucket(_force_day)
                if _fd_int != _last_force_day:
                    # Coupled ocean/land: step the coupler's (grid-agnostic) slab
                    # ocean + land for the elapsed day BEFORE re-sampling SST, so
                    # the daily SST resample below reads the just-updated ocean
                    # SST (the coupled driver overrides get_sst_sic -> ocean SST).
                    # Daily coupling cadence, matching the SST-refresh cadence.
                    # step 0 has nothing to step yet (_last_force_day is None).
                    if (self._segment_callback is not None
                            and _last_force_day is not None):
                        # Export the surface net radiative fluxes the MPAS
                        # physics computed (sw/lw net [W/m^2, +into surface])
                        # to the coupler's forcing channel: _build_atm_forcing
                        # reads held_sw_net_sfc/held_lw_net_sfc from _carry_aux.
                        # Without this the lean MPAS loop stashed nothing, so
                        # the coupled ocean/land tiles were forced with zero
                        # shortwave (the #1202 coupled-voronoi gap). The compiled
                        # cube/latlon path stashes the equivalent from
                        # PhysicsOutput; this is the lean-path equivalent.
                        _sfc_diag = getattr(self.model, "_sfc_diag", None)
                        if _sfc_diag is not None:
                            if not isinstance(self._carry_aux, dict):
                                self._carry_aux = {}
                            # Each element is None on the step where its source
                            # is inactive (sw/lw on a held-radiation sub-step or
                            # radiation=none; precip on a dry run) — stash only
                            # the fresh ones, keeping the last value otherwise.
                            if _sfc_diag[0] is not None:
                                self._carry_aux["held_sw_net_sfc"] = _sfc_diag[0].data
                            if _sfc_diag[1] is not None:
                                self._carry_aux["held_lw_net_sfc"] = _sfc_diag[1].data
                            # Surface precip [kg/m^2/s] for the ocean P-E /
                            # land forcing (None on a dry MPAS run).
                            if len(_sfc_diag) > 2 and _sfc_diag[2] is not None:
                                self._carry_aux["seg_precip"] = _sfc_diag[2].data
                            if (not getattr(self, "_logged_sfc_export", False)
                                    and "held_sw_net_sfc" in self._carry_aux):
                                _sw = self._carry_aux["held_sw_net_sfc"]
                                logger.info(
                                    "  Coupled surface radiative forcing (MPAS "
                                    "export): sw_net_sfc mean=%.1f range=[%.1f,"
                                    "%.1f] W/m^2",
                                    float(jnp.mean(_sw)), float(jnp.min(_sw)),
                                    float(jnp.max(_sw)))
                                self._logged_sfc_export = True
                        self._current_day = _force_day
                        self._segment_callback(self, _force_day, 86400.0)
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
                        # Prognostic ice skin: cache THIS forcing day's SST+SIC
                        # for the PER-STEP re-anchor + advance in the step loop.
                        # The skin is advanced every model step (dt=DT) with the
                        # freshly exported fluxes, AND the anchor T_sfc is
                        # re-blended every step from these cached fields against
                        # the advanced skin.  Consequences:
                        #  - the physics consumes the CURRENT skin, so the
                        #    surface-flux feedback is per-step (r*lambda =
                        #    DT*lambda/C << 1), not daily-lagged (codex-2
                        #    finding 2 / codex-1 finding 7),
                        #  - a mid-day restart re-blends from the SAME restored
                        #    skin the straight run held, so it does not branch
                        #    the surface boundary (codex-2 finding 1),
                        #  - per-step flux use resolves the diurnal SW/turbulent
                        #    cycle a once-daily snapshot aliased (codex-1
                        #    finding 1),
                        #  - restart-exact: the checkpoint holds a fully
                        #    advanced skin, no pending daily advance to drop or
                        #    double-count (codex-1 finding 2).
                        # SST is daily piecewise-constant (prescribed); only the
                        # ice fraction of T_sfc evolves sub-daily with the skin.
                        # Sample SST/SIC ONCE and keep the ice fraction: the
                        # surface-albedo blend below needs the same ``sic`` the
                        # temperature blend used.  Identical to the previous
                        # ``_compute_T_sfc(day)`` (which is exactly
                        # ``_blend_T_sfc(*get_sst_sic(day))``) and to the
                        # ice-skin branch, so T_sfc is byte-identical.
                        _sst_now, _sic_now = self.get_sst_sic(
                            _force_day_canonical)
                        _sst_day = jnp.asarray(_sst_now).reshape(-1)
                        _sic_day = jnp.asarray(_sic_now).reshape(-1)
                        if _ice_skin_on:
                            _ice_sst_cur = _sst_day
                            _ice_sic_cur = _sic_day
                        _forcing_daily["T_sfc"] = _blend_T_sfc(
                            _sst_day, _sic_day)
                        # Tile-blended surface shortwave albedo.  ONE formula
                        # (forcing.surface_utils.blended_surface_albedo) shared
                        # with the FV lane's blend; ocean/ice first, then the
                        # land fraction.  Without this key radiation falls back
                        # to the scalar config albedo (0.06 = open ocean) for
                        # EVERY column, land included.
                        if _sfc_albedo_on:
                            _alb_ocean_day = _albedo_ocean
                            if _zenith_ocean_alb:
                                _alb_ocean_day = _mpas_zenith_ocean_albedo(
                                    _alb_lat, _force_day_canonical,
                                    getattr(self, "orbit", None))
                            _sea_albedo_day = blend_surface_property(
                                _sic_day, _albedo_ice, _alb_ocean_day)
                            _forcing_daily["sfc_albedo"] = (
                                blended_surface_albedo(
                                    _sic_day, _alb_f_land, _albedo_ice,
                                    _alb_ocean_day, _albedo_land_static))
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
                    if _solar_ext:
                        # Same canonical-day sampling as SST/ozone
                        # (FIX_RESTART_TIME: bit-exact restart continuation).
                        from legoesm.forcing.external import (
                            get_solar_forcing_at_time,
                        )
                        _sol = get_solar_forcing_at_time(
                            self._solar_config, _force_day_canonical)
                        _forcing_daily["tsi"] = jnp.asarray(
                            float(_sol["tsi"]))
                        if (_solar_spectral
                                and _sol.get("solar_fraction_by_gpt")
                                is not None):
                            _forcing_daily["solar_spectral_fraction"] = (
                                jnp.asarray(_sol["solar_fraction_by_gpt"]))
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
                # Surface heat fluxes for the convective diurnal-cycle CAPE
                # subtraction, which is what delays land storms from noon to
                # late afternoon.  They are the PREVIOUS step's: turbulence
                # produces them after convection inside the same step, so this
                # step's do not exist yet.  75 s of lag against a daily cycle.
                #
                # The keys are seeded UNCONDITIONALLY with zeros on the first
                # step, exactly as beta_land is, so the forcing pytree keeps a
                # stable structure and the compiled step does not retrace when
                # real values first arrive.  Zero flux on step one means the
                # subtraction is simply absent for that step, which is correct:
                # there has been no surface heating yet.
                if _capdcycl_on:
                    _sd_prev = getattr(self.model, "_sfc_diag", None)
                    _zero = jnp.zeros((self.grid.grid_shape_2d[0],),
                                      dtype=self.state.p_s.data.dtype)
                    for _slot, _key in ((6, "shflx_sfc"), (7, "lhflx_sfc")):
                        _v = None
                        if (_sd_prev is not None and len(_sd_prev) > _slot
                                and _sd_prev[_slot] is not None):
                            _v = jnp.asarray(_sd_prev[_slot].data).reshape(-1)
                        _forcing[_key] = _zero if _v is None else _v
                # Interactive land skin T (one-step lag): blend the multilayer
                # tile's last skin temperature into the surface anchor over the
                # land fraction.  Ocean/ice keep the prescribed SST/SIC blend;
                # before the first land step (_land_T_skin None) the anchor is
                # the unmodified SST field (incl. the lapse fallback) — the
                # cold-start value the land tile itself was initialized from.
                if (_land_ml_on and _land_T_skin is not None
                        and "T_sfc" in _forcing):
                    _forcing["T_sfc"] = (
                        (1.0 - _f_land_cells) * _forcing["T_sfc"]
                        + _f_land_cells * _land_T_skin)
                # Interactive land ALBEDO (same one-step lag as the skin T
                # above): the multilayer tile's end-of-step albedo already
                # carries the snow brightening and the dry-soil brightening, so
                # this is how the snow-albedo feedback reaches radiation on
                # this lane.  Re-blend against the day's ocean/ice albedo so
                # only the land fraction is replaced.
                if (_land_ml_on and _land_albedo_cells is not None
                        and _sea_albedo_day is not None
                        and "sfc_albedo" in _forcing):
                    _forcing["sfc_albedo"] = (
                        (1.0 - _f_land_cells) * _sea_albedo_day
                        + _f_land_cells * _land_albedo_cells)
                # Phase 2b (#1312): traced per-cell beta_soil (same one-step
                # lag as the skin T above; seeded pre-loop so the key is
                # structurally stable — no retrace).
                if _land_beta_cells is not None:
                    _forcing["beta_land"] = _land_beta_cells
                # The land's solved boundary humidity, used verbatim by the
                # turbulence over the land fraction (supersedes beta there).
                if _land_qsfc_cells is not None:
                    _forcing["q_sfc_land"] = _land_qsfc_cells
                # The land's OWN turbulent fluxes -- the actual coupling.
                if _land_shflx_cells is not None:
                    _forcing["shflx_land"] = _land_shflx_cells
                    _forcing["lhflx_land"] = _land_lhflx_cells
            # Radiation sub-cycle: solve RRTMGP on step 0 (cache warm-up,
            # always) and every RAD_UPDATE_STEPS-th step; reuse the held
            # heating (PhysicsState.rad_heating) in between.  ``step`` is
            # job-local, so the first step of every job/restart link re-solves
            # radiation and repopulates the cache before any held step reads
            # it.  When subcycling is off, every step is a full step.
            _use_rad = (not _subcycle_rad) or (step % RAD_UPDATE_STEPS == 0)
            # Physics cadence: step 0 of every job is a physics step (the
            # cache is per-job, never persisted), then every
            # PHYS_UPDATE_STEPS-th; the held variant re-applies the cache in
            # between.  Radiation steps are always physics steps (validated).
            _is_phys_step = (not _hold_phys) or (step % PHYS_UPDATE_STEPS == 0)
            # Seed ONLY on step 0 (a physics step, so the zero seed is never
            # applied); any other empty-cache step is refused by the held
            # variant instead of silently applying zeros.
            if _hold_phys and step == 0:
                assert _phys_state.held_physics is None
                from legoesm.atmosphere.physics.combined import physics_cache_seed
                from legoesm.core.precision import cast_pytree as _cast_pytree
                _phys_state = _phys_state._replace(
                    held_physics=physics_cache_seed(
                        physics_fn, _cast_pytree(self.state, None, "compute"),
                        (self._voronoi_layout.local_mesh
                         if self._voronoi_layout is not None else self.model.mesh),
                        self.model.sigma_coord, _phys_state, _forcing))
            if _mpi_step is not None:
                _mstep = ((_mpi_step if _use_rad else _mpi_step_norad)
                          if _is_phys_step else _mpi_step_held)
                self.state, _phys_state = _mstep(
                    self.state, DT, _forcing, _phys_state)
            else:
                _pfn = ((physics_fn if _use_rad else physics_fn_norad)
                        if _is_phys_step else physics_fn_held)
                self.state = self.model.step(
                    self.state, DT, physics_fn=_pfn, forcing=_forcing,
                    phys_state=_phys_state)
                _phys_state = self.model._phys_state
                # Budget-ledger accumulation (#1311): the per-step per-column
                # ledger rides the same eager side-channel as _phys_state.
                # None when the flag is off (structural, zero cost).
                if _budget_ledger_on:
                    _ls = getattr(self.model, "_step_ledger", None)
                    if _ls is not None:
                        _led_accum = (_ls if _led_accum is None
                                      else _led_accum + _ls)
                        _led_nsteps += 1
            # Prognostic ice skin: advance ONE model step (dt=DT) with the
            # freshly exported surface energy fluxes (sfc_diag slots
            # 0 sw_net, 1 lw_net [W/m^2, +into surface]; 6 shflx, 7 lhflx
            # [+upward]).  F_net_down = sw + lw - sh - lh (net downward gain
            # of the skin).  Requires slots 0,1 present — radiation != none
            # is enforced at setup, so they exist after the first (always
            # full-radiation) step of each run/link.  Eager, both lanes
            # (the cell-partition MPI step publishes the same ``_sfc_diag``
            # side channel post-jit, and the update is per-cell elementwise
# with no neighbour stencil, so it needs no halo exchange).
# See the daily-boundary note for why the
            # advance is per-step rather than a once-daily snapshot.
            if _ice_skin_on and _ice_sic_cur is not None:
                _sd = getattr(self.model, "_sfc_diag", None)
                _swn = (_sd[0].data if (_sd is not None and len(_sd) > 0
                                        and _sd[0] is not None) else None)
                _lwn = (_sd[1].data if (_sd is not None and len(_sd) > 1
                                        and _sd[1] is not None) else None)
                if _swn is not None and _lwn is not None:
                    _f_net = (jnp.asarray(_swn).reshape(-1)
                              + jnp.asarray(_lwn).reshape(-1))
                    if len(_sd) > 6 and _sd[6] is not None:
                        _f_net = _f_net - jnp.asarray(
                            _sd[6].data).reshape(-1)
                    if len(_sd) > 7 and _sd[7] is not None:
                        _f_net = _f_net - jnp.asarray(
                            _sd[7].data).reshape(-1)
                    self._ice_T_skin = prognostic_ice_skin_temperature(
                        self._ice_T_skin, _f_net, _ice_sic_cur,
                        dt_s=DT, h_ice_m=_h_ice)
                # Re-anchor the NEXT step's T_sfc from the cached daily
                # SST/SIC against the (advanced) skin, so the physics
                # consumes the CURRENT skin every step (per-step feedback,
                # codex-2 finding 2) and a mid-day restart reproduces the
                # straight run's surface boundary (codex-2 finding 1).  Runs
                # every step the feature is active (even one that skipped
                # the advance for missing fluxes) so T_sfc stays consistent
                # with self._ice_T_skin.
                _forcing_daily["T_sfc"] = _blend_T_sfc(
                    _ice_sst_cur, _ice_sic_cur)
            # Interactive multilayer land step (MPAS port): advance the soil/
            # snow columns with the surface fluxes this step just exported
            # (sw/lw down refresh on radiation steps; precip every step) and
            # the post-step lowest-level state.  Jitted closure, device-only —
            # no host sync; its skin T enters next step's forcing blend above.
            # Skipped (state held) only until the first radiation solve
            # populates the downwelling export.
            if _land_ml_on:
                _a2s = _marshal_land_forcing()
                if _a2s is not None:
                    # Cadence accumulator: sum the forcing on-device; the
                    # tile consumes the interval MEAN, so the interval's
                    # precip mass and radiant energy are conserved exactly
                    # (mean rate x DT_LAND = sum of per-step rate x DT).
                    # With _LAND_K == 1 the mean is the identity and the
                    # trajectory is unchanged.  cos_zenith is accumulated
                    # SW-WEIGHTED separately (weight sw_down + 1 W/m2, so it
                    # degrades to the plain mean at night): the canopy puts
                    # cos_zenith under the direct-beam extinction, and a
                    # plain mean over an interval that straddles the
                    # terminator hands it a small-positive sun with full
                    # daytime SW (GLM review, 2026-08-26).
                    _land_a2s_sum = (
                        _a2s if _land_a2s_sum is None
                        else jax.tree_util.tree_map(
                            jnp.add, _land_a2s_sum, _a2s))
                    _land_cz_w = _a2s.sw_down + 1.0
                    if _land_a2s_n == 0:
                        _land_cz_wsum = _a2s.cos_zenith * _land_cz_w
                        _land_w_sum = _land_cz_w
                    else:
                        _land_cz_wsum = (
                            _land_cz_wsum + _a2s.cos_zenith * _land_cz_w)
                        _land_w_sum = _land_w_sum + _land_cz_w
                    _land_a2s_n += 1
                # A full interval fires the cadence step; a SINGLE sample
                # fires the bootstrap step when no held outputs exist yet
                # (run start and every chain-link restart: the held fluxes
                # are loop-locals, so without this the first _LAND_K steps
                # of a resumed link would run with NO land in the surface
                # blend at all — codex P1, 2026-08-26).
                _land_call = (_land_a2s_n >= _LAND_K
                              or (_land_T_skin is None and _land_a2s_n >= 1))
                if _land_call:
                    _a2s_mean = (
                        _land_a2s_sum if _land_a2s_n == 1
                        else jax.tree_util.tree_map(
                            lambda s: s / _land_a2s_n, _land_a2s_sum))
                    _a2s_mean = _a2s_mean._replace(
                        cos_zenith=_land_cz_wsum / _land_w_sum)
                    _land_fn = (_land_step_fn if _land_a2s_n >= _LAND_K
                                else _land_step_boot_fn)
                    _land_a2s_sum = None
                    _land_a2s_n = 0
                    (self._land_ml_state, _land_T_skin,
                     _land_albedo_cells, _land_qsfc_step,
                     _land_shflx_step, _land_lhflx_step,
                     _land_n_held_step, _land_n_held_land_step) = _land_fn(
                        self._land_ml_state, _a2s_mean,
                        jnp.asarray(_doy, dtype=jnp.float64))
                    # Mirror the land tile's skin and surface humidity onto
                    # the driver, the same way the ice skin above is mirrored:
                    # the CMOR ``tas`` diagnostic runs in a different method and
                    # cannot see these closure locals, and without them it
                    # anchors its profile on the neighbouring OCEAN skin over
                    # land and publishes a land temperature the model never had.
                    self._land_T_skin_last = _land_T_skin
                    self._land_qsfc_last = _land_qsfc_step
                    # Accumulate ON DEVICE and read at the same cadence the
                    # other post-step warnings use: reading it every step
                    # would stall the accelerator once per step for a number
                    # that is almost always zero.
                    _land_n_held_accum = _land_n_held_accum + _land_n_held_step
                    _land_n_held_land_accum = (
                        _land_n_held_land_accum + _land_n_held_land_step)
                    _land_n_held_steps_accum = (
                        _land_n_held_steps_accum
                        + (_land_n_held_step > 0).astype(jnp.int32))
                    # Published only under the same switch that threads f_land
                    # into the turbulence factory: without the land fraction
                    # the consumer refuses the key, and adding it mid-run
                    # would change the forcing pytree and retrace.
                    if _land_beta_soil_on:
                        _land_qsfc_cells = _land_qsfc_step
                        _land_shflx_cells = _land_shflx_step
                        _land_lhflx_cells = _land_lhflx_step
                    if _land_beta_fn is not None and _land_qsfc_cells is None:
                        # Root-zone beta only until the humidity channel is
                        # live (or when the scheme solves none).
                        _land_beta_cells = _land_beta_fn(self._land_ml_state)
                if (step % _HARD_SAT_LOG_CADENCE_STEPS) == 0:
                    _window_cols = int(_land_n_held_accum)
                    _window_land = int(_land_n_held_land_accum)
                    _window_steps = int(_land_n_held_steps_accum)
                    _land_n_held_accum = jnp.zeros((), jnp.int32)
                    _land_n_held_land_accum = jnp.zeros((), jnp.int32)
                    _land_n_held_steps_accum = jnp.zeros((), jnp.int32)
                    if _window_cols:
                        self._land_n_held_total += _window_cols
                        self._land_n_held_steps += _window_steps
                        # Column-steps alone cannot separate one column
                        # failing every step from many columns failing
                        # once, and those are different problems: the
                        # first is a bad column, the second is a bad
                        # configuration. Report the number of STEPS that
                        # held as well, and the worst single step.
                        logger.warning(
                            "land: %d column-steps held in the last %d "
                            "steps (%d of them on LAND columns — the "
                            "physically meaningful share; the rest are "
                            "ocean columns whose land output is "
                            "discarded), on %d of those steps (a held "
                            "column's energy and water budgets do not "
                            "close); %d column-steps on %d steps since "
                            "the run began — at step %d",
                            _window_cols, _HARD_SAT_LOG_CADENCE_STEPS,
                            _window_land, _window_steps,
                            self._land_n_held_total,
                            self._land_n_held_steps, step)

            # Top sponge (#836): per-step Rayleigh decay of the edge winds
            # toward rest above sigma_top (see profile construction above).
            # Pure device elementwise multiply — no host sync, no retrace.
            if _sponge_decay is not None:
                self.state = self.state._replace(
                    u=self.state.u.replace(
                        data=self.state.u.data * _sponge_decay))
            # Post-step horizontal q_v smoothing (opt-in).  Placed BEFORE the
            # hard-saturation drain so the drain acts on the smoothed field
            # (smoothing spreads a grid-scale supersaturation spike across
            # neighbours; the drain then removes what remains).  Module-scope
            # _mpas_qv_smooth_step (the exact code the unit test exercises):
            # UNWEIGHTED SCVT del2 + q>=0 floor.  Conserves (to fp roundoff)
            # the per-level sum_c A_c q_c integral, NOT column water vapour —
            # an explicitly non-conservative filter (see the config field note).
            # Eager like the drain below (outside jit).
            if _qv_smooth_nu > 0.0 or _qv_smooth_nu4 > 0.0:
                _trc_sm = self.state.tracers
                _qv_sm_in = _trc_sm["q_v"].data
                if _qv_halo_refresh is not None:
                    # MPI lane: fresh cell halo so boundary-owned stencils
                    # read owner values (see the setup note, #1321).
                    _qv_sm_in = _qv_halo_refresh(_qv_sm_in)
                _qv_new_sm = _mpas_qv_smooth_step(
                    _qv_sm_in, self.grid, _qv_smooth_nu, DT,
                    nu4=_qv_smooth_nu4, mid_refresh=_qv_halo_refresh,
                    owned_mask=(None if self._voronoi_layout is None
                                else self._voronoi_layout.owned_mask_cells))
                _new_trc_sm = dict(_trc_sm)
                _new_trc_sm["q_v"] = _trc_sm["q_v"].replace(data=_qv_new_sm)
                self.state = self.state._replace(tracers=_new_trc_sm)
            # Post-step HARD SATURATION ADJUSTMENT (opt-in), applied on the FINAL
            # state AFTER the dycore's vertical vapour transport has acted this
            # step -- the load-bearing placement (in-scheme leaves the transport
            # spike uncorrected until the next physics call, and at dt~100 s the
            # transport+latent feedback detonates within that window).  Reuses
            # the reviewed _warm_rain hard-adjustment core EXACTLY (bracketed-
            # bisection on-curve solve + per-step latent-heating & vapour rate
            # limit + conservation bookkeeping) as a pure activation-gated drain
            # (no spurious evaporation).  Eager (this site is outside jit): the
            # jnp.sum counter is a rare host sync, at low cadence only.
            # Requires BOTH q_v and q_c tracers: q_c is the condensate recipient,
            # so draining without it would lose total water (moist warm-rain
            # runs always carry both).
            _trc = self.state.tracers
            if (_hard_sat_on and _trc is not None
                    and "q_v" in _trc and "q_c" in _trc):
                _qc_fld = _trc["q_c"]
                _qi_fld = _trc.get("q_i") if _hard_sat_ice_curve else None
                (_T_hs, _qv_hs, _qc_hs, _qi_hs,
                 _dq_hs) = _mpas_hard_saturation_poststep(
                    self.state.T.data, _trc["q_v"].data, _qc_fld.data,
                    self.sigma.pressure_at_full(self.state.p_s.data), DT,
                    _hard_sat_threshold, _hard_sat_max_heating,
                    ice_curve=_hard_sat_ice_curve,
                    q_i=None if _qi_fld is None else _qi_fld.data)
                if (step % _HARD_SAT_LOG_CADENCE_STEPS) == 0:
                    _n_hs = int(jnp.sum(_dq_hs > _HARD_SAT_LOG_QV_EPS))
                    if _n_hs > 0:
                        logger.warning(
                            "hard saturation adjustment (post-step%s): drained "
                            "%d points (max dq_v %.2f g/kg, <= %.1f K) at "
                            "step %d",
                            ", ice curve" if _hard_sat_ice_curve else "",
                            _n_hs, float(jnp.max(_dq_hs)) * 1e3,
                            float(_hard_sat_max_heating), step)
                _new_trc = dict(_trc)
                _new_trc["q_v"] = _trc["q_v"].replace(data=_qv_hs)
                _new_trc["q_c"] = _qc_fld.replace(data=_qc_hs)
                if _qi_fld is not None and _qi_hs is not None:
                    _new_trc["q_i"] = _qi_fld.replace(data=_qi_hs)
                    # Seed cloud-ice NUMBER for the deposited ice mass so the
                    # two-moment scheme sees a physical crystal count (orphan
                    # q_i with N_i=0 is deposition-inert and cannot sediment),
                    # capped at Morrison's Cooper ceiling N_i_nuc_max/rho_air.
                    _ni_fld = _trc.get("N_i")
                    if (_ni_fld is not None and _ice_nuc_mass is not None
                            and _n_i_nuc_max is not None):
                        _dq_i_dep = _qi_hs - _qi_fld.data
                        _pf = self.sigma.pressure_at_full(self.state.p_s.data)
                        _new_trc["N_i"] = _ni_fld.replace(
                            data=_seed_nucleated_ice_number(
                                _ni_fld.data, _dq_i_dep, _ice_nuc_mass,
                                _n_i_nuc_max, _pf, _T_hs))
                self.state = self.state._replace(
                    T=self.state.T.replace(data=_T_hs), tracers=_new_trc)
            # Keep the persisted-carry handle fresh for save_checkpoint
            # (#413) — reference assignment, no device work.
            self._mpas_phys_state = _phys_state

            # #1353: fold this step's surface/TOA fluxes into the CMOR
            # interval accumulator (lazy device adds — no host sync here).
            # Cost bound (codex-1 finding 5): 6 elementwise (cast+)adds on
            # (nCells,) surface fields per step — EXPECTED tens of µs of
            # async dispatch vs an L5 eager step of ~100+ ms, in a loop
            # that already dispatches eager per-step work (sponge, ice
            # skin, guards); the campaign's tracked sim-days/s throughput
            # is the standing regression check.  Fusing into ``_step_jit``
            # would couple the diag path into the compiled step and break
            # the "diag block is read-only / trajectory bit-identical"
            # property this lane documents — deliberately kept eager.
            if self._mpas_sfc_accum is not None:
                self._mpas_sfc_accum.add(
                    getattr(self.model, "_sfc_diag", None))

            # Sedimentation sub-steps: accumulate the window maximum on
            # device EVERY step (the export slot holds only the latest
            # step), and report it once per cadence window -- one host sync
            # per window, and only when the lane published a count.
            _sed_req_window = sed_substeps_window_max(
                getattr(self.model, "_sfc_diag", None), _sed_req_window)
            # ``step > 0``: at step 0 the window holds one step and the
            # report would fire before anything accumulated, which also let
            # a test mistake the opening report for the post-loop flush
            # (codex 2026-09-22).
            if step > 0 and (step % _SED_SUBSTEP_LOG_CADENCE_STEPS) == 0:
                report_sed_substep_overflow(
                    getattr(self.model, "_sfc_diag", None),
                    int(_sed_cap_effective), step, running=_sed_req_window)
                _sed_req_window = None

            # Hourly state means and extrema keep absolute time across restarts.
            # A non-divisor dt samples on the first step crossing each hour.
            # Flux means keep their independent per-step sums and diagnostic cadence.
            _sample_day = START_DAY + (step + 1) * DT / 86400.0
            _previous_day = START_DAY + step * DT / 86400.0
            if (self._mpas_cmip_feed_on
                    and int(np.floor(_sample_day * 24.0 + 1e-9))
                    > int(np.floor(_previous_day * 24.0 + 1e-9))):
                self._feed_mpas_cmip_accumulators(_sample_day, state_only=True)

            # Diagnostics at intervals — phased by DIAG_PHASE (= start_step
            # for a real periodic cadence, 0 for the once-at-the-end
            # sentinel; see its definition).  On the periodic path a restart
            # chain therefore keeps ONE global diagnostic phase: samples stay
            # evenly spaced in simulated time across links, and a partial
            # flux interval restored from the checkpoint (#1353) completes at
            # the boundary it belongs to instead of at a fresh job-local
            # multiple (codex-3 HIGH).  Off-cadence restarts emit their first
            # sample after the REMAINDER of the interval.
            if (DIAG_INTERVAL > 0
                    and (DIAG_PHASE + step + 1) % DIAG_INTERVAL == 0):
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
                        self.sigma.dsigma,
                        dp=self.sigma.layer_thickness_dp(p_s_data))

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
                    # mean T is PRESSURE-WEIGHTED (sum T*dp / sum dp; equal
                    # cell weight on the quasi-uniform SCVT): an unweighted
                    # level mean is coordinate-dependent — the stretched
                    # hybrid grid packs thin warm near-surface levels that
                    # each get one equal vote, overstating the global mean
                    # by +9.2 K vs sigma on the same state (quantified
                    # 2026-07-23, hybrid-L20 day-8 ckpt), which made
                    # hybrid-vs-sigma stability curves incomparable.
                    _p_half_diag = self.sigma.pressure_at_half(p_s_data)
                    _dp_diag = (_p_half_diag[..., 1:]
                                - _p_half_diag[..., :-1])
                    _stats = jnp.stack([
                        jnp.sum(T_data * _dp_diag) / jnp.sum(_dp_diag),
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
                # Energy-budget tracker on the MPAS lane (#1354/#1515).  The
                # per-step INSTANTANEOUS fluxes are in self.model._sfc_diag
                # (slot 0 sw_net_sfc, 1 lw_net_sfc [+into surface]; 3 rlut=
                # lw_up_toa, 4 rsut=sw_up_toa, 5 rsdt=sw_down_toa; 6 shflx,
                # 7 lhflx [+up]) — the same tuple the ice-skin advance reads.
                # toa_net = rsdt - rsut - rlut; the tracker's residual =
                # toa_net - dE/dt.  hfss/hfls are recorded for the closure
                # probe (LEAK = sfc_net_rad - hfss - hfls - residual).
                # FLUX TIMING: INTERVAL MEANS when the accumulator is running,
                # snapshots otherwise, and the series records WHICH.
                #
                # This comment used to argue the snapshot was adequate: a
                # fixed-time global sample spans all longitudes hence all local
                # times, so rsdt is S_0/4 and TOA carries only ~1-5 W/m^2 of
                # noise.  MEASURED 2026-09-04 (job 9632045) that argument holds
                # for solar geometry and FAILS for the turbulent fluxes: the
                # sampled sensible heat flux was 8.0 W/m^2 against the
                # accumulated 20.5 -- 2.5x -- because sensible heat is dominated
                # by LAND, which occupies limited longitudes with a sharply
                # asymmetric diurnal cycle that one local time per longitude
                # does not average.  The apparent leak came out +34.7 W/m^2
                # against a ~20 W/m^2 hypothesis; substituting accumulated
                # channels gave ~11.  A plausible wrong answer, the dangerous
                # kind.  So prefer self._mpas_sfc_accum (#1353's interval means,
                # widened to slots 0/1 above).  It is CMOR-gated, so with the
                # feed off the tracker falls back to snapshots and stamps
                # energy_flux_interval_mean = 0; the closure probe then REFUSES
                # to report a leak rather than quoting a contaminated one.
                # MPI-partitioned MPAS is skipped: the tracker uses local area
                # weights + local state with no owned-cell mask or allreduce
                # (halo double-count), exactly as the moisture tracker is
                # skipped on that lane (codex review).  Single-GPU / serial
                # only, which is the #1354 L5 lane.
                _ebd = getattr(self.diagnostics, "energy_tracker", None)
                _sd = getattr(self.model, "_sfc_diag", None)
                # GATE (codex review): `has_samples()` alone is not enough. A
                # window can be SHORT -- the first interval after a
                # feed-off->feed-on restart, or a checkpoint written before
                # slots 0/1 existed -- and its mean is then over the wrong
                # number of steps, or missing the surface-radiation pair
                # entirely. Either way it would be stamped "interval mean" and
                # sail past the probe, which is worse than the snapshot it
                # replaced because it looks trustworthy. Require a COMPLETE
                # window AND every slot the energy budget reads.
                _facc_e = getattr(self, "_mpas_sfc_accum", None)
                _use_accum = (_facc_e is not None
                              and _facc_e.window_ready(_facc_e.ENERGY_SLOTS))
                if (_facc_e is not None and _facc_e.is_complete()
                        and not _use_accum):
                    print("  energy tracker: complete window but energy "
                          "slots have unequal sample counts -- this sample "
                          "falls back to SNAPSHOT fluxes (stamped 0)")
                _qv_e = (self.state.tracers["q_v"].data
                         if (self.state.tracers is not None
                             and "q_v" in self.state.tracers) else None)
                # Frozen condensate (q_i+q_s+q_g) for the phase-complete energy
                # (#1354/#1515): without the -L_f*q_frozen term, deposition and
                # freezing read as a spurious source.  Sum whatever frozen
                # species this microphysics carries (None -> vapor-only MSE).
                _qfrz_e = None
                if self.state.tracers is not None:
                    for _fk in ("q_i", "q_s", "q_g"):
                        if _fk in self.state.tracers:
                            _fd = self.state.tracers[_fk].data
                            _qfrz_e = _fd if _qfrz_e is None else _qfrz_e + _fd

                def _slot(i):
                    # Interval mean first (the APPLIED quantity); the
                    # end-of-interval snapshot only when no accumulator ran.
                    # `mean()` returns a host array, so this round-trips
                    # device->host->device. That is 7 small transfers per
                    # DIAGNOSTIC step (12 in a 12-day run at --diag-days 1),
                    # not per model step, so it is not on the hot path
                    # (codex review, accepted rather than restructured --
                    # `mean()` is shared with the CMOR feed).
                    if _use_accum:   # window_ready() => every slot has a mean
                        return jnp.asarray(_facc_e.mean(i))
                    return (_sd[i].data if (_sd is not None and len(_sd) > i
                                            and _sd[i] is not None) else None)
                _sw_dn, _sw_up, _lw_up = _slot(5), _slot(4), _slot(3)
                _sw_ns, _lw_ns = _slot(0), _slot(1)
                _shf, _lhf = _slot(6), _slot(7)
                if (_ebd is not None and _qv_e is not None
                        and not _is_mpas_cell_partitioned(self)
                        and None not in (_sw_dn, _sw_up, _lw_up, _sw_ns, _lw_ns)):
                    from legoesm.diagnostics.energy_budget import (
                        area_weighted_mean as _awm,
                    )
                    from legoesm.grids.voronoi import reconstruct_cell_velocity
                    _awt = self.diagnostics._area_w
                    # MPAS u is EDGE-normal (nEdges, nlev); the column KE term
                    # needs cell-centred east/north winds (codex P0).  Perot
                    # reconstruction, the same the turbulence/coupler paths use.
                    _uc, _vc = reconstruct_cell_velocity(self.state.u.data,
                                                         self.grid)
                    _eb = _ebd.update(
                        self.state.T.data, _qv_e, _uc, _vc,
                        self.state.phis.data, p_s_data,
                        self.diagnostics.dsigma, self.diagnostics.sigma_full,
                        _sw_dn, _sw_up, _lw_up, _sw_ns, _lw_ns,
                        elapsed_seconds=elapsed_day * 86400.0,
                        area_weights=_awt,
                        dp=self.diagnostics._dp(p_s_data),
                        p_full=self.diagnostics._p_full(p_s_data),
                        q_frozen=_qfrz_e,
                    )
                    _ts["energy_toa_net"].append(float(_eb.toa_net))
                    _ts["energy_dE_dt"].append(float(_eb.dE_dt))
                    _ts["energy_residual"].append(float(_eb.residual))
                    _ts["sw_net_sfc"].append(float(_eb.sfc_sw_net))
                    _ts["lw_net_sfc"].append(float(_eb.sfc_lw_net))
                    # Which flux timing produced this sample.  The closure
                    # probe refuses to report a leak from snapshots, because a
                    # contaminated leak is plausible rather than obviously
                    # broken (#1354).
                    _ts["energy_flux_interval_mean"].append(
                        1.0 if _use_accum else 0.0)
                    _ts["hfss"].append(float(_awm(_shf, _awt))
                                       if _shf is not None else float("nan"))
                    _ts["hfls"].append(float(_awm(_lhf, _awt))
                                       if _lhf is not None else float("nan"))
                else:
                    for _ek in ("energy_toa_net", "energy_dE_dt",
                                "energy_residual", "sw_net_sfc", "lw_net_sfc",
                                "hfss", "hfls"):
                        _ts[_ek].append(float("nan"))
                # Latest closure the CMOR feed recorded, or NaN before the
                # first complete diagnostic window.  NaN, never 0: a zero here
                # reads as "the budget closes", which is the one answer this
                # series must never invent.
                _mt = getattr(getattr(self, "diagnostics", None),
                              "moisture_tracker", None)
                _ts["moisture_residual"].append(
                    float(_mt.residual[-1])
                    if _mt is not None and _mt.residual else float("nan"))

                # Ice-crystal number telemetry (2026-07-28, century3 day-803
                # NaN): N_i grew x2/day for 800 days with every CLIMATE
                # diagnostic nominal, because PSD clamps mask absurd N in all
                # rates — only the raw field overflowing was visible.  A
                # daily max makes any number runaway visible in the log
                # months before overflow.  Serial/single-rank only (the MPI
                # lane's fused global diag would need an allreduce-MAX
                # extension; rank-local would mislead).
                _ni_max = float("nan")
                if (self._voronoi_layout is None
                        and self.state.tracers is not None
                        and "N_i" in self.state.tracers):
                    _ni_max = float(jnp.max(
                        self.state.tracers["N_i"].data))
                elapsed = time.time() - t_start
                rate = elapsed_day / (elapsed + 1e-10)
                logger.info(
                    f"  Day {elapsed_day:6.1f}: T=[{T_min:.1f},{T_max:.1f}]K "
                    f"mean={mean_T:.1f}K  p_s={mean_ps/100:.1f}hPa  "
                    f"|u|_max={max_u:.1f}m/s"
                    + ("" if _cwv != _cwv else f"  CWV={_cwv:.1f}kg/m2")
                    + ("" if _ni_max != _ni_max
                       else f"  Ni^max={_ni_max:.1e}/kg")
                    + f"  ({rate:.4f} sim-days/s, {elapsed:.0f}s)"
                )

                # Budget-ledger emission (#1311): interval-mean per-column
                # rates -> budget_ledger_columns.npz (overwritten each
                # interval; the
                # per-interval history is in the log lines).  The log line
                # reports the column with the LARGEST dry-enthalpy total —
                # exactly the detonating-column question — attributed to its
                # largest row.  Host sync is fine here: this block already
                # pulled scalars off-device.
                if _budget_ledger_on and _led_accum is not None:
                    from legoesm.diagnostics.process_ledger import (
                        LEDGER_PROCESSES,
                    )
                    from legoesm.parallel.geometry_consistency import (
                        content_hash48 as _content_hash48,
                    )
                    _led_rates = np.asarray(_led_accum) / max(_led_nsteps, 1)
                    # Snapshot the SAME weights the energy tracker used for
                    # dE/dt earlier in this block, copied so a later mask or
                    # regrid cannot make the artifact disagree with the
                    # number it will be differenced against (GLM: aliasing).
                    _area_w_led = self.diagnostics._area_w
                    if _area_w_led is not None:
                        _area_w_led = np.asarray(
                            _area_w_led, dtype=np.float64).ravel().copy()
                    _tot_e = _led_rates.sum(axis=1)[:, 1]     # (ncol,) W/m^2
                    _hot = int(np.argmax(np.abs(_tot_e)))
                    _hot_rows = _led_rates[_hot, :, 1]
                    _hot_row = int(np.argmax(np.abs(_hot_rows)))
                    logger.info(
                        "  Ledger: max|column dE/dt|=%.3e W/m^2 at cell %d "
                        "(top row: %s %.3e; steps=%d)",
                        _tot_e[_hot], _hot, LEDGER_PROCESSES[_hot_row],
                        _hot_rows[_hot_row], _led_nsteps)
                    # PER-COLUMN schema, deliberately a DIFFERENT filename
                    # from the FV lane's global-row ``budget_ledger.npz``
                    # (days/rates history): a consumer reading one schema
                    # must never silently get the other.  Overwritten each
                    # interval — on a blow-up the surviving file is the last
                    # pre-detonation interval, which is the one that matters.
                    _led_area_kw = {}
                    if _area_w_led is not None:
                        # One line that makes a shape/rank/reorder mistake
                        # fail HERE instead of silently downstream (GLM's
                        # "missing invariant").
                        if _area_w_led.size != _led_rates.shape[0]:
                            raise ValueError(
                                "budget ledger has "
                                f"{_led_rates.shape[0]} columns but the area "
                                f"weights have {_area_w_led.size}; the "
                                "artifact would carry a reduction that does "
                                "not match its own rows.")
                        _led_area_kw = {
                            "area_cell": _area_w_led,
                            # Fingerprint so a reader supplying its own
                            # weights can prove they are THESE weights, in
                            # THIS order.  A length check cannot see a
                            # permutation, and a permuted weight vector is
                            # quietly wrong rather than loudly wrong.
                            # ``content_hash48`` is the repo's existing
                            # positional byte digest (parallel/
                            # geometry_consistency.py), written for exactly
                            # this "a permutation must not cancel" property
                            # and float64-exact, so it stores in the npz as a
                            # plain scalar.
                            "area_hash48": np.asarray(
                                _content_hash48(_area_w_led)),
                        }
                    if self.output_dir is not None:
                        np.savez(
                            str(self.output_dir
                                / "budget_ledger_columns.npz"),
                            ledger_rates=_led_rates,
                            processes=np.asarray(LEDGER_PROCESSES),
                            columns=np.asarray(
                                ("water_kg_m2_s", "energy_W_m2")),
                            n_steps=_led_nsteps,
                            day=elapsed_day + START_DAY,
                            # Rank locality is part of the reduction: under
                            # cell partitioning these rows would be ONE rank's
                            # columns, and no weighting makes that a global
                            # budget.  Today this is always False -- setup
                            # already REFUSES --budget-ledger whenever the
                            # world size exceeds one or a Voronoi layout
                            # exists, which is strictly broader than this
                            # predicate.  It is stamped anyway because that
                            # refusal is documented as "serial-only FOR NOW":
                            # when the ledger gather is wired the artifact
                            # becomes rank-local, and the reader should refuse
                            # at that moment rather than print a per-rank
                            # table as a global one.
                            cell_partitioned=bool(
                                _is_mpas_cell_partitioned(self)),
                            # The reduction the reader MUST use (#1354).  The
                            # rows are per-column, so a consumer picks the
                            # weighting -- and an unweighted mean is not a
                            # global mean on the SCVT mesh (areaCell max/min
                            # 1.471).  Worse, the energy-budget tracker this
                            # ledger gets differenced against is already
                            # area-weighted, so an unweighted row and its
                            # store tendency are different global operators
                            # and their difference means nothing.  Shipping
                            # the SAME weights the tracker used removes the
                            # reader's opportunity to get it wrong.  The key
                            # is OMITTED, not zero-filled, when there are no
                            # weights: an empty float array is a valid array
                            # that a third consumer can misread as a mesh,
                            # while a missing key raises (GLM).
                            **_led_area_kw,
                        )
                    _led_accum = None
                    _led_nsteps = 0

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

                # Feed the CMOR monthly/daily/zonal accumulators.  Placed
                # AFTER the finiteness/bounds guard so a blown-up state never
                # pollutes the accumulators (the guard ``break``s first).  The
                # day is ABSOLUTE (START_DAY + elapsed) so multi-link restart
                # chains keep monotonic calendar months (matches the cube
                # path). This interval feed adds flux means only; state
                # means were already sampled by the independent hourly hook.
                if self._mpas_cmip_feed_on:
                    self._feed_mpas_cmip_accumulators(
                        START_DAY + elapsed_day, flux_only=True)

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
                # Wallclock-aware clean exit for long HPC dependency chains --
                # only on a physics-window boundary (see the restart guard).
                if (start_step + step + 1) % PHYS_UPDATE_STEPS == 0:
                    self._maybe_wallclock_exit(
                        _ckpt, start_step + step + 1, _ckpt_day)

        # Write the CMOR NetCDF from the (now-fed) accumulators on a CLEAN
        # completion.  MUST run BEFORE the final checkpoint below so its
        # ``_suppress_cmor_sidecar`` (set on a successful write) reaches a
        # still-to-be-written final checkpoint; ``final_day`` also lets it
        # RETIRE a same-day sidecar already written by the last in-loop
        # periodic checkpoint (exact-checkpoint-cadence completion).  Gated on
        # the feed being active (serial / 1-rank with CMIP output); a no-op
        # otherwise.
        # Flush the partial sedimentation window: an overflow in the last
        # steps before the run ends must still be reported.
        if _sed_req_window is not None:
            report_sed_substep_overflow(
                None, int(_sed_cap_effective), n_steps_total,
                running=_sed_req_window)

        if run_status == "COMPLETED" and self._mpas_cmip_feed_on:
            self._finalize_mpas_cmip(
                START_DAY + n_steps_total * DT / 86400.0)

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
        # A sub-daily cadence must never round DOWN to zero steps: 0 reads as
        # "diagnostics disabled" at every guard below, so a request for a very
        # fine cadence would silently turn the blow-up check OFF — the opposite
        # of what was asked.  Floor it at one step.
        from legoesm.driver.diagnostics import diagnostic_interval_steps
        DIAG_INTERVAL = diagnostic_interval_steps(
            cfg.output.diag_days, DT, n_steps_total)
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
        # #405 prognostic-physics carry (leapfrog path only); stays False/None
        # for the diagnostic dry/gray path and the ssp_rk3 path.
        _spectral_prognostic = False
        _spectral_phys_state = None
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
            # Normalize the CLI radiation alias ("rrtmg") to the
            # physics-layer scheme name — see the _run_mpas rationale.
            _rad_scheme = ("rrtmgp" if cfg.radiation in ("rrtmg", "rrtmgp")
                           else cfg.radiation)
            _cloud_scheme = (cfg.cloud_scheme
                             if _rad_scheme == "rrtmgp" else "none")
            # #405: prognostic-carry physics is now threadable on the spectral
            # LEAPFROG path, which evaluates physics ONCE per step so the carry
            # is captured + advanced below (mirrors _run_mpas).  The per-RK-stage
            # ssp_rk3 path still evaluates physics multiple times per step, where
            # a single-step carry is ill-defined, so a prognostic scheme there
            # still refuses loudly rather than silently reseeding every step.
            _spectral_integrator = str(getattr(
                self.model.config, "time_integrator", "ssp_rk3")).lower()
            _spectral_leapfrog = _spectral_integrator in (
                "leapfrog", "leapfrog_si")
            if not _spectral_leapfrog:
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
                convection=convection_config_for(cfg),
                turbulence=turbulence_config_for(cfg),
                microphysics=MicrophysicsConfig(scheme=cfg.microphysics),
                gravity_wave_drag=gwd_config_for(cfg),
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

            # #405: thread the prognostic PhysicsState carry on the leapfrog
            # path — pass the COMBINED fn DIRECTLY (its (state, grid, sigma,
            # phys_state=, forcing=) signature is exactly what the spectral
            # step's stateful branch calls) and SEED the carry (mirrors
            # _run_mpas: init_physics_state(ncol, nlev, cfg)).  The stateless
            # ``_amip_physics_fn`` wrapper (which drops the carry + maps
            # forcing_data positionally) stays the default for the ssp_rk3 /
            # diagnostic path — byte-identical there.
            _spectral_prognostic = (
                _spectral_leapfrog
                and getattr(_combined_fn, "_requires_phys_state", False))
            if _spectral_prognostic:
                from legoesm.atmosphere.physics.physics_state import (
                    init_physics_state,
                )
                _ncol_sp = int(self.grid.n_lat) * int(self.grid.n_lon)
                _nlev_sp = int(self.sigma.n_levels)
                _spectral_phys_state = init_physics_state(
                    _ncol_sp, _nlev_sp, phys_cfg)
                # NOTE (restart, codex): a fresh seed each RUN is correct, but
                # the spectral checkpoint path does not yet persist/restore the
                # ``physstate_*`` carry (unlike _run_mpas #413), so a RESTARTED
                # prognostic-spectral run re-seeds and loses its physics memory.
                # Fresh runs are correct; carry persistence is a follow-up.
                _phys_fn_loop = _combined_fn
                logger.info(
                    "  #405: prognostic physics threaded on the spectral "
                    "leapfrog path (PhysicsState carry seeded + advanced "
                    "each step).")
            else:
                _phys_fn_loop = _amip_physics_fn
            _ext_forcing = _external_forcing_active(
                _rad_scheme == "rrtmgp",
                self._ozone_ext_active, self._aerosol_active,
                self._aerosol_lw_active, self._ghg_active,
                bool(self._experiment),
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
        # ---- Coupled ocean/land support (segment_callback + surface flux) ----
        # A coupled spectral run installs ``_segment_callback`` (the coupler's
        # daily ocean/land step). ``_build_atm_forcing`` reads the surface net
        # SW/LW from ``_carry_aux``; the spectral integrator is per-RK-stage so
        # we do NOT thread per-step fluxes — instead RECOMPUTE the surface net
        # radiation ONCE per day at the coupling boundary (one gray solve/day is
        # cheap) and stash it, mirroring how _run_mpas exports its sfc_diag.
        # Gated on a PRESENT callback → the standalone spectral path is
        # byte-identical (no callback ⇒ this whole block is dead).
        _has_segcb = getattr(self, "_segment_callback", None) is not None
        _last_coupling_day = None

        def _spectral_sfc_net_rad(state, day):
            """(sw_net_sfc, lw_net_sfc) [W/m^2, +into surface] from the current
            spectral state via the gray diagnostic radiation (the coupled-
            idealized path)."""
            _f = spectral_pe_to_grid(state, self.grid, self.sigma)
            _Tg, _psg = _f["T"], _f["p_s"]
            _pf = _psg[..., None] * sigma_full
            _ph = _psg[..., None] * self.sigma.sigma_half
            _Tc = _Tg.reshape(-1, cfg.grid.nlev)
            # Real grid-space moisture (gray LW is moist) so the recompute
            # matches the atmosphere's own gray radiation; zeros on a dry run.
            _trq = getattr(state, "tracers", None)
            _qvf = _trq.get("q_v") if _trq else None
            _qvd = (_qvf.data if hasattr(_qvf, "data") else _qvf)
            _qvc = (_qvd.reshape(-1, cfg.grid.nlev)
                    if _qvd is not None else jnp.zeros_like(_Tc))
            _sst, _sic = self.get_sst_sic(day)
            if _sst.ndim == 1 and len(shape_2d) == 2:
                _sst = jnp.broadcast_to(_sst[:, None], shape_2d)
                _sic = jnp.broadcast_to(_sic[:, None], shape_2d)
            _Tsfc = blend_surface_temperature(_sst, _sic, T_ice).reshape(-1)
            _lat2d = (jnp.broadcast_to(self._grid_lat[:, None], shape_2d)
                      if self._grid_lat.ndim == 1 else self._grid_lat)
            _insol = daily_mean_insolation(
                _lat2d.reshape(-1), self._insolation_day(day), S_0,
                orbit=_orbit_params)
            _rad = gray_radiation(
                T=_Tc, p_full=_pf.reshape(-1, cfg.grid.nlev),
                p_half=_ph.reshape(-1, cfg.grid.nlev + 1),
                sfc_temperature=_Tsfc, lat=_lat2d.reshape(-1),
                q_v=_qvc, insolation=_insol, config=gray_config)
            # Surface half-level is index -1 (gray module: F_*_sfc = *[:, -1]).
            _sw = (_rad.sw_flux_down[:, -1] - _rad.sw_flux_up[:, -1])
            _lw = (_rad.lw_flux_down[:, -1] - _rad.lw_flux_up[:, -1])
            return _sw.reshape(shape_2d), _lw.reshape(shape_2d)

        for step in range(start_step, n_steps_total):
            self._current_day = START_DAY + (step + 1) * DT / 86400.0

            # Coupled daily boundary (explicit coupling, mirrors _run_mpas): at
            # the first step of a new day, recompute + stash the surface fluxes
            # from the CURRENT (end-of-elapsed-day) state and step the coupler's
            # ocean/land for that day BEFORE this step advances the atmosphere.
            # The fluxes are recomputed from the SAME state the callback's
            # _build_atm_forcing reads, so forcing + state are self-consistent.
            # Step 0 has nothing to step yet (_last_coupling_day is None).
            if _has_segcb:
                _cd_int = daily_forcing_bucket(self._current_day)
                if (_last_coupling_day is not None
                        and _cd_int != _last_coupling_day):
                    _sw_net, _lw_net = _spectral_sfc_net_rad(
                        self.state, float(_cd_int))
                    if not isinstance(self._carry_aux, dict):
                        self._carry_aux = {}
                    self._carry_aux["held_sw_net_sfc"] = _sw_net
                    self._carry_aux["held_lw_net_sfc"] = _lw_net
                    # Surface precipitation export: the spectral lane recomputes
                    # only the surface RADIATION at the coupling boundary (above),
                    # not an accumulated precip rate.  Stash an EXPLICIT zero
                    # seg_precip (not a silent absence — the coupler's
                    # require_surface_radiation_aux guard forbids the SILENT
                    # zero-fill that would go unnoticed).  This is EXACT for the
                    # slab / two-layer ocean (no prognostic salinity, so P-E is
                    # unused); a 3-D dynamic ocean on the spectral grid is gated
                    # off (run_coupled), so no freshwater budget depends on it.
                    # Accumulating the true segment precip through the spectral
                    # step is the follow-up for a coupled spectral dynamic ocean.
                    self._carry_aux["seg_precip"] = jnp.zeros_like(_sw_net)
                    if not getattr(self, "_logged_spectral_precip", False):
                        logger.info(
                            "  Coupled spectral lane: surface precip export is "
                            "0 (exact for the slab ocean; the spectral 3-D ocean "
                            "is gated). Radiation is exported.")
                        self._logged_spectral_precip = True
                    self._segment_callback(self, self._current_day, 86400.0)
                _last_coupling_day = _cd_int

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
                phys_state=_spectral_phys_state,
            )
            # #405: capture the advanced prognostic carry for the next step
            # (the model publishes it on ``_phys_state``; None on the stateless
            # / ssp_rk3 path, where this is a byte-identical no-op).
            if _spectral_prognostic:
                _spectral_phys_state = self.model._phys_state

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
                    f"|v|_max={max_wind:.1f}m/s  ({rate:.4f} sim-days/s, {elapsed:.0f}s)"
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
                    # #1028: max_wind is a cell-centre diagnostic (the #1049
                    # dead-jet gate reads it), so take the cc view -- identity
                    # unless the run carries D-staggered cube winds.
                    _cc_ts = self._state_cc()
                    T_arr = _cc_ts.T.data
                    u_arr = _cc_ts.u.data
                    v_arr = (_cc_ts.v.data
                             if getattr(_cc_ts, "v", None) is not None
                             else None)
                    ps_arr = _cc_ts.p_s.data
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
            # #1354/#1515 energy-budget closure inputs (MPAS lane).
            energy_toa_net=_arr("energy_toa_net") if "energy_toa_net" in ts else nan,
            energy_dE_dt=_arr("energy_dE_dt") if "energy_dE_dt" in ts else nan,
            hfss=_arr("hfss") if "hfss" in ts else nan,
            hfls=_arr("hfls") if "hfls" in ts else nan,
            # Flux-timing provenance for the seven channels above. WITHOUT
            # this the closure probe refuses every real series as "timing
            # unknown" -- which is the correct refusal, and exactly what
            # happens when a collected channel is never persisted.
            energy_flux_interval_mean=(
                _arr("energy_flux_interval_mean")
                if "energy_flux_interval_mean" in ts else nan),
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

    def _reject_coupled_lane(self, lane: str, envelope: str,
                             remedy: str) -> None:
        """Refuse a COUPLED run on an atmosphere lane that never stashes the
        surface radiation / precipitation the coupler consumes.

        The coupled drivers read ``held_sw_net_sfc`` / ``held_lw_net_sfc`` /
        ``seg_precip`` out of ``self._carry_aux``.  The STATELESS lat-lon SPMD
        sub-lane (dynamics-only / Held-Suarez) and the sub-face-tiled cube
        lanes never write them, so the coupled
        ocean/land/ice tiles would be forced with ``sw_down=0`` and
        ``precip=0`` -- perpetual polar night plus an evaporation-only
        freshwater budget, and SILENTLY: no NaN, no exception, and the
        reconstructed ``lw_down`` collapses to a plausible ``sigma*T_sfc**4``
        (the emissivity cancels exactly), so it reads as a spin-up transient
        rather than a broken boundary condition.  Per dispatch-hardening
        doctrine this refuses rather than defaulting to zeros.

        Gated on ``_requires_surface_flux_export`` -- the marker a coupled
        driver sets on its atmosphere -- and NOT on ``_segment_callback``.
        ``run(segment_callback=...)`` is a GENERAL per-segment hook used by
        uncoupled diagnostic samplers (the operator-split fold-back parity
        tests, ``training/run_to_column_mean``, ``ml/physics/data``); refusing
        those would be a pure regression.  Only a consumer that actually reads
        ``_carry_aux`` sets the marker.
        """
        if not getattr(self, "_requires_surface_flux_export", False):
            return
        raise NotImplementedError(
            f"{lane} runs {envelope} and never stashes held_sw_net_sfc / "
            "held_lw_net_sfc / seg_precip into _carry_aux, so a coupled run "
            "would force the ocean/land/ice tiles with sw_down=0 and "
            f"precip=0 (silent: no NaN, plausible lw_down). {remedy}")

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
        # STATELESS sub-lane only (dynamics-only / Held-Suarez): no radiation,
        # no microphysics, so nothing ever writes held_sw_net_sfc /
        # held_lw_net_sfc / seg_precip into _carry_aux.  The operator-split
        # sub-lane above DOES stash them and returned already, so the refusal
        # sits here rather than at the run() dispatch (which cannot distinguish
        # the two).
        self._reject_coupled_lane(
            "enable_latlon_spmd (stateless lat-band SPMD)",
            "a dynamics-only / Held-Suarez envelope with no radiation or "
            "precipitation source",
            "Configure the general unified physics so the operator-split SPMD "
            "sub-lane is selected (it stashes the held fields), or run the "
            "coupled case single-device (enable_latlon_spmd=False).")
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
        # NOT routed through ``diagnostic_interval_steps``: this lane REFUSES a
        # sub-step cadence (below) instead of flooring it, because its segment
        # length is the gcd of the cadences — flooring to one step here would
        # silently run the whole lane one step per segment.  It still borrows
        # that helper's finiteness check, because ``NaN > 0`` is False and a
        # NaN cadence would otherwise slip past as "no cadence" (review).
        if not _math.isfinite(cfg.output.diag_days):
            raise ValueError(
                f"tiled cube SPMD: diag_days must be a finite number of days; "
                f"got {cfg.output.diag_days!r}.")
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
            pack_carry, pack_forcing, unpack_carry, segment_accum_to_rate,
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
        # Radiation cadence.  ``make_sharded_operator_split_step`` computes the
        # serial prologue's predicate per step (``_need_rad_and_time``,
        # sharded_operator_split_step.py:202) and passes it in as ``need_rad``,
        # but a ``step_unified`` built with ``static_need_rad=True`` DELETES
        # that predicate and always takes the radiation branch
        # (physics_pipeline.py:2556) -- which made ``rad_update_steps > 1`` a
        # silent no-op on this lane.  ``None`` above 1 keeps the pipeline's own
        # ``lax.cond``, which honours it.
        #
        # The two lanes' predicates agree with ZERO phase offset: serial
        # computes ``(step_idx + 1) % k == 0`` (compiled_segments.py:1934) and
        # its subcycle scan radiates on the LAST step of each cycle (:2254),
        # which is exactly what ``_need_rad_and_time`` selects.  Serial-vs-SPMD
        # held_lw_net_sfc at rad_update_steps=2 MEASURES 2.52e-5 (C8/nlev4,
        # gray+SBM, 151 steps = a 144-step segment plus a 7-step tail), the same
        # order as the 1.63e-5 at rad_update_steps=1 -- i.e. the ordinary
        # band-cut residual, with no cadence term left.  Gated by
        # tests/parallel/test_operator_split_spmd_carry_aux_export.py::
        # test_radiation_cadence_matches_serial.
        #
        # This was refused outright until the serial short-tail defect was
        # fixed: a segment whose length did not divide the cadence fell back to
        # a body that discarded the predicate, so serial radiated every step and
        # the same comparison read 1.97e-3 -- a SERIAL error, not this lane's.
        #
        # The issue-#316 rationale for eliding the cond -- bounding XLA compile
        # when step_unified is inlined into a LONG lax.scan -- does not apply
        # here: this lane dispatches ``sharded_step`` once per step from Python
        # (:10099), so there is no long scan to inline into.  The cond is NOT
        # free, though: both branches land in the jitted unified step, which is
        # itself invoked inside the jitted shard_map body.  UNMEASURED at
        # production resolution with rrtmgp -- the numbers quoted above are a
        # C8/nlev4 gray deck, so treat the compile-time and peak-memory cost of
        # the second branch as unknown rather than negligible (codex review).
        band_su = band_physics.build_step_unified(
            static_need_rad=(True if ctx["RAD_UPDATE_STEPS"] <= 1 else None))

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

            # --- Coupled-lane surface export (the reason this lane is no longer
            # refused to coupled drivers).  Both coupled drivers read
            # held_sw_net_sfc / held_lw_net_sfc / seg_precip off ``_carry_aux``
            # (``coupling_fields.require_surface_radiation_aux``); without them
            # the ocean/land/ice tiles are forced with sw_down=0 and precip=0,
            # SILENTLY.  Units/conventions are the PRODUCER-side contract
            # _run_compiled uses at :11282: held_* are the instantaneous net
            # surface fluxes [W/m2] radiation last held, seg_precip is a RATE
            # [kg/m2/s] positive-DOWN (into the surface).
            #
            # Read off ``carry_full`` (the REPLICATED gather), not ``carry``:
            # the coupler consumes them on the full global grid alongside
            # ``self.state``, and a band-sharded leaf would mis-shape against
            # ``jnp.zeros_like(p_s)`` in _build_atm_forcing.
            self._carry_aux["held_sw_net_sfc"] = carry_full.held_sw_net_sfc
            self._carry_aux["held_lw_net_sfc"] = carry_full.held_lw_net_sfc
            self._carry_aux["seg_precip"] = segment_accum_to_rate(
                carry_full.precip_accum, seg_steps, DT)
            # RESEED every segment accumulator.  ``segment_accum_to_rate``
            # divides by THIS segment's duration and its contract
            # (compiled_segments.py:531-534) is "from a zero reseed at every
            # segment start" — _run_compiled gets that free by re-packing the
            # carry each segment (:11138, which zeroes precip_accum and the
            # whole *_toa_accum / *_sfc_accum / t_low_accum family, and lets
            # pack_carry default shflx_accum/lhflx_accum to zeros).  This lane
            # THREADS one carry across all segments, so each accumulator would
            # otherwise be a RUN total: seg_precip alone is what the coupler
            # reads today, but resetting only that one leaves every sibling as a
            # silent trap for the SPMD diagnostics writers that are the declared
            # follow-up here (codex review).  Reset by SUFFIX so a future
            # accumulator cannot be forgotten, at each leaf's OWN dtype and
            # sharding (pack_carry promotes accumulators to the precision
            # policy's storage dtype, which need not equal state.T's ``_sd``).
            _reseed = {
                _nm: jax.device_put(
                    jnp.zeros(_leaf.shape, _leaf.dtype), _leaf.sharding)
                for _nm in carry._fields
                if _nm.endswith("_accum") or _nm == "max_cfl"
                # None leaves are the "diagnostic off" encoding (e.g.
                # budget_ledger_accum) — keep them None, not zeros, or the
                # carry's pytree structure changes mid-run and retraces.
                for _leaf in (getattr(carry, _nm),) if _leaf is not None
            }
            carry = carry._replace(**_reseed)

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
        from legoesm.driver.diagnostics import diagnostic_interval_steps
        diag_interval = diagnostic_interval_steps(
            cfg.output.diag_days, DT, 0)
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
        # CLUBB sub-grid cloud-fraction carry (marine-Sc albedo lever): diagnostic
        # CLUBB writes it out of the physics step; the NEXT radiation step reads
        # it.  None (default / feature-off) => step_unified gets None =>
        # byte-identical RH grid-scale cloud path.
        cloud_fraction = None
        # Build _seed_ps (and thus seed the cloud-fraction carry below) whenever a
        # stateful carry is active OR the CLUBB cf feature is on — do NOT rely on
        # ``carries_energy`` alone: a cf-producing closure that carried no energy
        # would otherwise skip the seed and the compiled feature would silently
        # no-op (carry stays None).  Diagnostic CLUBB carries energy today, so this
        # is defensive; the guard in run() already requires diagnostic CLUBB.
        if (_turb_traits.carries_energy or _gwd_prognostic
                or getattr(cfg, "use_clubb_cloud_fraction", False)):
            from legoesm.atmosphere.physics.combined import PhysicsConfig
            from legoesm.atmosphere.physics.turbulence import (
                TurbulenceConfig,
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
                    # MUST be the SAME resolved config the pipeline kernel gets
                    # (_resolve_gwd -> gwd_config_for): init_physics_state sizes
                    # and fills the gwd_spectrum carry from
                    # ``prognostic_spectral.n_azimuths/.n_wavenumbers/
                    # .launch_flux``, and a gravity_wave_drag_override may set
                    # all three.  A bare config here seeded (ncol,4,20) at the
                    # default launch flux while the kernel expected the
                    # override's shape/amplitude (codex round 1, finding 2).
                    gravity_wave_drag=gwd_config_for(cfg),
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
            # Seed the CLUBB cloud-fraction carry only when the feature is on
            # (diagnostic CLUBB is guaranteed by the build_physics_pipeline gate,
            # which raises at construction otherwise, so _seed_ps carries a real
            # zero-init cloud_fraction here).
            if getattr(cfg, "use_clubb_cloud_fraction", False):
                cloud_fraction = _seed_carry(
                    "cloud_fraction", _seed_ps.cloud_fraction,
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
            "cloud_fraction": cloud_fraction,
            "o3_vmr": o3_vmr, "aerosol_od": aerosol_od, "ghg_vmr": ghg_vmr,
            "lat_deg_grid": lat_deg_grid,
            "_sd": _sd,
        }

    def _finalize_run(self, run_status, t_jit, t_start, n_steps_total,
                      START_DAY, N_DAYS, checkpoint_interval):
        """Shared finalization: save diagnostics, results, final checkpoint."""
        jax.block_until_ready(self._prognostic_wind_leaf())
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
        # #1028: the persistent-D conversion happens in ``run()``, which a
        # training caller never enters -- so a driver built with the flag set
        # would hand back a CELL-CENTRE training segment while its config says
        # D-staggered.  Refuse rather than train on a different discretisation
        # than the production run the loss is meant to match (codex review).
        if getattr(self.config.dycore, "persistent_dgrid", False):
            raise NotImplementedError(
                "dycore.persistent_dgrid=True is not wired into "
                "build_training_segment: the D-staggering conversion happens in "
                "ModelDriver.run(), so this would silently build a cell-centre "
                "segment. Train with the flag off, or wire the conversion into "
                "the training carry first (#1028).  Note what training with "
                "the flag off costs once production turns it ON: the damped "
                "regime carries roughly a THIRD of the eddy amplitude, so a "
                "model fitted on it meets a distribution it never saw (GLM "
                "review) -- this refusal is fail-loud, not a resting place.")

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
            persistent_dgrid=self._persistent_dgrid_active(),   # #1028
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
            budget_ledger=cfg.output.budget_ledger,
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
        from legoesm.diagnostics.process_ledger import (
            N_LEDGER as _N_LEDGER_ROWS,
        )
        carry0 = pack_carry(
            self.state, self.q_v, self.q_c, self.q_r,
            conv_prog=ctx["conv_prog"],
            budget_ledger_accum=(
                jnp.zeros((_N_LEDGER_ROWS, 2))
                if cfg.output.budget_ledger else None),
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
            shard_forcing, segment_accum_to_rate,
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
        phys_cloud_fraction = ctx.get("cloud_fraction")
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
            persistent_dgrid=self._persistent_dgrid_active(),   # #1028
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
            budget_ledger=cfg.output.budget_ledger,
        )

        logger.info(
            f"Starting compiled run: {n_steps_remaining} steps, "
            f"{n_segments} segments of {segment_length} steps"
        )

        current_step = start_step
        t_jit = 0.0
        t_start = time.time()

        # Multilayer (Richards) land under distribution: the cube-face MPI path
        # scatters the per-column soil state/params/carbon to owned faces in
        # ``_setup_parallel`` (``_land_ml_scattered``), so each rank advances its
        # own columns and the gathered soil is bit-identical to serial (faces are
        # embarrassingly parallel — no lateral coupling).  The SPMD device-mesh
        # and lat-band MPI paths do NOT scatter it yet (no partition spec), so
        # they still fail LOUDLY rather than silently degrade to the slab or
        # shard a state with no sharding contract (CLAUDE.md: no silent degrade).
        if (self._land_ml_state is not None
                and self._device_config is not None
                and getattr(self._device_config, "is_distributed", False)
                and not self._land_ml_scattered):
            raise NotImplementedError(
                "use_multilayer_land is only supported under the cube-face MPI "
                "path (which scatters the soil columns to owned faces).  This "
                "run is distributed via an SPMD device mesh or lat-band MPI, "
                "which have no multilayer-land partition spec yet.  Run on a "
                "single device / single MPI rank, use cube-face MPI, or use slab "
                "land (use_multilayer_land=False) for these distributed modes."
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
            from legoesm.diagnostics.process_ledger import (
                N_LEDGER as _N_LEDGER_ROWS,
            )
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
                # Per-process budget-ledger accumulator: zeros seed (reset
                # each segment) when the diagnostic is on, None (byte-
                # identical carry) otherwise.
                budget_ledger_accum=(
                    jnp.zeros((_N_LEDGER_ROWS, 2))
                    if self.config.output.budget_ledger else None),
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
                cloud_fraction=phys_cloud_fraction,
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
                    # PHASE gate.  The host loop below hardcodes "advance
                    # exactly RAD_UPDATE_STEPS held steps, then refresh", which
                    # only lands on the configured boundary when the segment
                    # STARTS on one.  Without this a restart at a non-aligned
                    # step silently shifts the refresh by an arbitrary offset
                    # rather than the documented one step -- e.g. s=2, k=4
                    # refreshes after step 5 where the cadence asks for step 3
                    # (codex adversarial review).  Falls through to the fused
                    # path, which is phase-general (compiled_segments.py
                    # _run_subcycled) and therefore always correct; the only
                    # cost is this segment's XLA compile boundary.
                    and int(carry.step_index) % RAD_UPDATE_STEPS == 0
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
                        u=_g(self._state_cc().u.data),
                        v=_g(self._state_cc().v.data),
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
                # UNITS: the coupler contract (AtmToSurface.precip_total,
                # packages/core/legoesm/core/coupling_fields.py:18) is a RATE
                # [kg/m2/s], positive-downward (into the surface) -- the same
                # convention as the per-step PhysicsOutput.precip that fed the
                # accumulator.  ``seg_precip`` off the carry is the segment
                # ACCUMULATION [kg/m2] (compiled_segments.py:1197, docstring
                # :174-175), so it must be divided by the segment duration here,
                # at the producer.  Not at the consumer: the other two writers of
                # this key (``_sfc_diag[2]`` on the lean MPAS path, :5802, and
                # ``phys_out.precip`` on the per-step path, :9744) already store
                # rates, so a consumer-side divide would corrupt them.
                # ``seg_steps`` is the ACTUAL step count (the final segment is
                # short) and ``DT`` is still the dt this segment ran with -- the
                # adaptive-dt halving happens later, at :9173.  Under ensembles
                # ``seg_precip`` is the ensemble mean of the accumulators, which
                # is still an accumulation, so the divide is valid there too.
                "seg_precip": segment_accum_to_rate(seg_precip, seg_steps, DT),
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
            if carry.cloud_fraction is not None:
                phys_cloud_fraction = carry.cloud_fraction
                self._carry_aux["cloud_fraction"] = phys_cloud_fraction

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
                # #1028: the stability check reads cell-centre winds.
                error = self.diagnostics.check_stability(
                    self._state_cc(), elapsed_day)
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

            # Segment callback for coupled integration (e.g. the coupler
            # step).  Hoisted OUT of the diagnostics block (issue F4): it
            # must fire on EVERY segment so the coupler is handed the TRUE
            # elapsed time since the previous coupling call (seg_steps *
            # DT), not once per diagnostic interval.  When a checkpoint
            # cadence makes segment_length = GCD(diag, checkpoint) a PROPER
            # divisor of diag_interval the old diag-gated placement fired
            # only once per diag interval yet reported a single segment's
            # duration -- under-integrating the ocean/land/ice by
            # diag_interval / segment_length and silently dropping the
            # intervening segments' fluxes.  The coupler-facing carry_aux
            # (seg_precip RATE, seg_shflx, seg_lhflx, held_* fields) is
            # rebuilt every segment (see the dict above), so per-segment
            # firing reads fresh per-segment fluxes.  Stateless +
            # restart-safe: each segment self-reports its own duration;
            # coupling runs BEFORE the checkpoint write below, and
            # segment_length divides checkpoint_interval, so a checkpoint
            # boundary is always a segment boundary and the post-couple
            # coupled state is captured.  Placed AFTER the stability check so
            # the coupler never observes an unstable state.
            if self._segment_callback is not None:
                dt_seg = float(seg_steps * DT)
                self._segment_callback(self, day, dt_seg)

            # Diagnostics
            if diag_interval > 0 and current_step % diag_interval == 0:
                # Convert accumulated quantities to rates over segment duration.
                _seg_dur = seg_steps * DT
                # Same conversion the coupler-facing carry_aux entry uses, via
                # the shared helper (numerically identical to the previous
                # ``seg_precip / _seg_dur``: _seg_dur IS seg_steps * DT), so the
                # diagnostic precip rate and the coupled precip rate cannot
                # silently diverge.
                seg_precip_rate = segment_accum_to_rate(seg_precip, seg_steps, DT)
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

                # Per-process budget ledger: segment-mean rates
                # (accum/duration), appended host-side and rewritten to
                # budget_ledger.npz each diag step (small file).
                if (self.config.output.budget_ledger
                        and _dm_carry.budget_ledger_accum is not None):
                    import numpy as _np
                    from legoesm.diagnostics.process_ledger import (
                        LEDGER_COLUMNS, LEDGER_PROCESSES,
                    )
                    _led_rates = _np.asarray(
                        _dm_carry.budget_ledger_accum) / _seg_dur
                    if not hasattr(self, "_budget_ledger_days"):
                        self._budget_ledger_days = []
                        self._budget_ledger_rates = []
                    self._budget_ledger_days.append(float(day))
                    self._budget_ledger_rates.append(_led_rates)
                    if self.output_dir is not None:
                        _np.savez(
                            str(self.output_dir / "budget_ledger.npz"),
                            days=_np.asarray(self._budget_ledger_days),
                            rates=_np.asarray(self._budget_ledger_rates),
                            processes=_np.asarray(LEDGER_PROCESSES),
                            columns=_np.asarray(LEDGER_COLUMNS),
                        )

                diag_info = self._sync_and_collect_diagnostics(
                    elapsed_day=elapsed_day,
                    day=day,
                    state=self._state_cc(),      # #1028: diagnostics are cc
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
                # Same spacing the setup dt clamp judged (grid type, resolution,
                # polar filter): pole cell on lat-lon without the filter.
                from legoesm.core.cfl import cfl_number_from_state, estimate_min_dx
                _dx_min = estimate_min_dx(
                    cfg.grid.resolution, cfg.grid.grid_type,
                    getattr(self.grid, 'radius', constants.R_earth),
                    use_polar_filter=getattr(cfg.dycore, "use_polar_filter", False))

                # Under MPI, CFL on owned faces only, then global max
                # CFL is a CELL-CENTRE number (the dx estimate is the centre
                # spacing), so it reads the cell-centre view -- identity off
                # the persistent-D lane (#1028).
                _cc_cfl = self._state_cc()
                if self._owned_face_ids is not None:
                    _ofi = self._owned_face_ids
                    _seg_max_cfl = float(cfl_number_from_state(
                        _cc_cfl.u.data[_ofi], _cc_cfl.v.data[_ofi], _dx_min, DT,
                    ))
                    from mpi4py import MPI
                    _seg_max_cfl = MPI.COMM_WORLD.allreduce(_seg_max_cfl, op=MPI.MAX)
                else:
                    _seg_max_cfl = float(cfl_number_from_state(
                        _cc_cfl.u.data, _cc_cfl.v.data, _dx_min, DT,
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

                # Adaptive dt: if CFL exceeds threshold, halve dt and rebuild
                if _seg_max_cfl > 1.0:
                    DT = DT / 2.0
                    logger.warning(
                        f"  CFL={_seg_max_cfl:.2f} > 1.0 at day {elapsed_day:.0f}. "
                        f"Halving dt to {DT:.0f}s."
                    )
                    # Keep the CLM-ML canopy's CONCRETE dt in step with the halved
                    # DT: the canopy resolves its static ML sub-step count from
                    # this value, and the segment is rebuilt below (retraces), so a
                    # new count is fine.  A stale land_ml_dt would advance the
                    # canopy at the old timestep while the dynamics use the new one.
                    if getattr(self.physics, "land_ml_dt", None) is not None:
                        self.physics.land_ml_dt = DT
                    n_steps_total = int(cfg.days * 86400 / DT)
                    from legoesm.driver.diagnostics import diagnostic_interval_steps
                    diag_interval = diagnostic_interval_steps(
                        cfg.output.diag_days, DT, 0)
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
                        persistent_dgrid=self._persistent_dgrid_active(),  # #1028
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
                        budget_ledger=cfg.output.budget_ledger,
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
        # CLUBB sub-grid cloud-fraction carry (marine-Sc albedo lever): diagnostic
        # CLUBB writes it out; the next radiation step reads it.  None => feature
        # off => byte-identical (kw omits it => step_unified default None).
        phys_cloud_fraction = ctx.get("cloud_fraction")

        def _phys_carry_step_inputs():
            """Keyword inputs for the active stateful-physics carries."""
            kw = {}
            if phys_tke is not None:
                kw["tke"] = phys_tke
            if phys_qke is not None:
                kw["qke"] = phys_qke
            if phys_gwd_spectrum is not None:
                kw["gwd_spectrum"] = phys_gwd_spectrum
            if phys_cloud_fraction is not None:
                kw["cloud_fraction"] = phys_cloud_fraction
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

        # Sedimentation sub-step reporting: the window opens BEFORE the
        # warm-up step, which is a real physics step whose count would
        # otherwise be dropped -- a first-step-only overflow, or a one-step
        # run, would report nothing (codex 2026-09-22).
        _sed_req_window = None
        _sed_warned_silent = False
        _sed_cap_per_step = effective_sed_substeps_cap(
            getattr(self.physics, "micro_config", None),
            self.config.microphysics, self.config)

        self.state = self.model.step_with_physics(self.state, DT)
        # Physics reads winds at cell centres.  On the persistent-D lane this
        # is a READ-ONLY view of the corner winds (#1028); on every other lane
        # it is ``self.state`` itself, so the call below is byte-identical.
        _cc_in = self._state_cc()

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
                _cc_in.u.data, _cc_in.v.data,
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
        if phys_out.cloud_fraction is not None:
            phys_cloud_fraction = phys_out.cloud_fraction
            self._carry_aux["cloud_fraction"] = phys_cloud_fraction

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
            # Conserving form always (owner decision 2026-08-16): the plain
            # ``max(q, 0)`` invented water at every physics overdraw.
            self.q_v = self._conserving_floor(_qv_raw)
        self.q_c = self._conserving_floor(self.q_c + DT * phys_out.dq_c_dt)
        self.q_r = self._conserving_floor(self.q_r + DT * phys_out.dq_r_dt)
        self._apply_double_moment_tendencies(phys_out, DT)
        _sed_req_window = sed_substeps_window_max(
            None, _sed_req_window, counts=phys_out.sed_substeps_required)

        if MICROPHYSICS == "none":
            p_full = self.sigma.pressure_at_full(self.state.p_s.data)
            q_sat = saturation_mixing_ratio(new_T, p_full)
            excess = jnp.maximum(self.q_v - q_sat, 0.0)
            self.q_v = self.q_v - excess
            new_T = new_T + constants.L_v * excess / constants.c_pd

        self.state = self.state._replace(T=self.state.T.replace(data=new_T))
        if hasattr(phys_out, 'du_dt') and phys_out.du_dt is not None:
            # Cell-centre tendency; on the persistent-D lane it is lifted to
            # the corners before it is added (#1028).
            self._add_cc_wind_increment(phys_out.du_dt, phys_out.dv_dt, DT)
        # Conserving form always (owner decision 2026-08-16): the
        # hyperdiffusion tail is non-monotone, so its floor is the same
        # mass-creating clamp class as the physics floors above.
        self.q_v = self._conserving_floor(
            self.q_v + DT * hyperdiffusion_3d(self.q_v, self.grid, self._qv_smooth_coeff)
        )
        self._scale_winds(self._fric_decay)

        jax.block_until_ready(self._prognostic_wind_leaf())
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
            # Cell-centre view for the physics inputs (identity off the
            # persistent-D lane, #1028).
            _cc_step = self._state_cc()

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
                    _cc_step.u.data, _cc_step.v.data,
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
            # Sedimentation sub-steps: same window-maximum + cadence report as
            # the MPAS loop, from the count this lane returns directly
            # (codex 2026-09-22: reporting only inside _run_mpas left every
            # finite-volume run silent about a clamped fall).
            _sed_req_window = sed_substeps_window_max(
                None, _sed_req_window, counts=phys_out.sed_substeps_required)
            if step > 0 and (step % _SED_SUBSTEP_LOG_CADENCE_STEPS) == 0:
                if (_sed_req_window is None and not _sed_warned_silent):
                    # a reporting lane that published NOTHING (the multilayer
                    # microphysics branch returns no count) is as blind as a
                    # non-reporting one, so it says so once (GLM 2026-09-22)
                    _sed_warned_silent = warn_sed_substeps_unreported(
                        self.config, "per-step (no count published)",
                        getattr(self.physics, "micro_config", None))
                report_sed_substep_overflow(
                    None, int(_sed_cap_per_step), step,
                    running=_sed_req_window)
                _sed_req_window = None
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
            if phys_out.cloud_fraction is not None:
                phys_cloud_fraction = phys_out.cloud_fraction
                self._carry_aux["cloud_fraction"] = phys_cloud_fraction

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
                # Conserving form always (owner decision 2026-08-16).
                self.q_v = self._conserving_floor(_qv_raw)
            self.q_c = self._conserving_floor(self.q_c + DT * phys_out.dq_c_dt)
            self.q_r = self._conserving_floor(self.q_r + DT * phys_out.dq_r_dt)

            # Apply ice/number tracer tendencies when full registry is active
            self._apply_double_moment_tendencies(phys_out, DT)

            # Saturation adjustment
            if MICROPHYSICS == "none":
                p_full = self.sigma.pressure_at_full(self.state.p_s.data)
                q_sat = saturation_mixing_ratio(new_T, p_full)
                excess = jnp.maximum(self.q_v - q_sat, 0.0)
                self.q_v = self.q_v - excess
                new_T = new_T + constants.L_v * excess / constants.c_pd
                precip_ls = jnp.sum(
                    excess * self.sigma.layer_thickness_dp(self.state.p_s.data), axis=-1
                ) / (constants.g * DT)
            else:
                precip_ls = jnp.zeros(shape_2d, dtype=new_T.dtype)

            self.state = self.state._replace(
                T=self.state.T.replace(data=new_T)
            )

            # Apply momentum tendencies from turbulence/GWD
            if hasattr(phys_out, 'du_dt') and phys_out.du_dt is not None:
                # Cell-centre tendency; lifted to the corners first on the
                # persistent-D lane (#1028).
                self._add_cc_wind_increment(phys_out.du_dt, phys_out.dv_dt, DT)

            # Moisture conservation fixer
            if FIX_MOISTURE:
                self.q_v = fix_moisture_hydrostatic(
                    self.q_v, target_moisture,
                    self.state.p_s.data, dsigma, self.grid,
                )

            # Moisture smoothing — conserving floor (owner decision
            # 2026-08-16), same class as the warmup-lane site.
            self.q_v = self._conserving_floor(
                self.q_v + DT * hyperdiffusion_3d(self.q_v, self.grid, self._qv_smooth_coeff)
            )

            # Rayleigh friction
            self._scale_winds(self._fric_decay)

            # Diagnostics (diag_interval == 0 = writer disabled — guard the
            # modulo; spmd configs FORCE diag_days=0, codex round-2 MAJOR)
            elapsed_day = day - START_DAY
            if diag_interval > 0 and (step + 1) % diag_interval == 0:
                diag_info = self._sync_and_collect_diagnostics(
                    elapsed_day=elapsed_day,
                    day=day,
                    state=self._state_cc(),      # #1028: diagnostics are cc
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

                # Stability check (#1028: cell-centre winds)
                error = self.diagnostics.check_stability(
                    self._state_cc(), elapsed_day)
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

                # Segment callback for coupled integration.  This lane is a
                # PER-STEP loop (no segments); the callback fires exactly on
                # diagnostic boundaries, so the elapsed time since the
                # previous call is ``diag_interval * DT`` -- NOT a single
                # step ``DT`` (issue F4: passing DT under-reported the
                # elapsed coupling interval by the full diag_interval
                # factor).  This legacy/debug lane (run(compiled=False)) is
                # not a production coupled path -- CoupledESMDriver uses the
                # compiled lane -- but the elapsed dt is corrected here for
                # consistency with the hoisted compiled-lane callback.
                if self._segment_callback is not None:
                    self._segment_callback(self, day, float(diag_interval * DT))

            # Checkpoint
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                self.save_checkpoint(step + 1, day)

        report_sed_substep_overflow(
            None, int(_sed_cap_per_step), n_steps_total,
            running=_sed_req_window)
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
            # Per-process budget ledger: run-mean attribution table (also
            # printed to the log) when the diagnostic was on.
            if getattr(self, "_budget_ledger_rates", None):
                import numpy as _np
                from legoesm.diagnostics.process_ledger import (
                    format_ledger_table,
                )
                _tbl = format_ledger_table(
                    _np.mean(_np.asarray(self._budget_ledger_rates), axis=0),
                    header="Per-process column budget ledger (run mean)")
                f.write(f"\n{_tbl}\n")
                print(_tbl)
        logger.info(f"  Results saved to {self._output_dir}")
