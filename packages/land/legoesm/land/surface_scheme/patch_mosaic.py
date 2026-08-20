"""N-patch (tile) canopy mosaic over the two-leaf surface scheme.

A single grid column is subdivided into ``N`` independent land patches (tiles),
each running the *existing* two-leaf canopy with its own leaf-area index, C4
fraction, photosynthetic capacity and water-stress, weighted by its area
fraction.  The patch fluxes are aggregated to one grid-mean ``SurfaceFluxOutput``
by area weighting (``x_grid = sum_i frac_i * x_i``), which is exact for per-area
fluxes and the area-mean for intensive state.

Motivation — savanna / woody-savanna sites (US-Ton, AU-How, US-SRM) are a MOSAIC
of sparse C3 trees (deep-rooted, active through the dry season) and a C4/C3 grass
understory (shallow-rooted, active only in the wet season).  A single blended
canopy shares ONE LAI phenology and ONE water-stress, so it cannot represent the
wet-spring-grass → dry-summer-tree handoff that drives the GPP time series.  Two
patches (``N=2``) give each source its own parameters; ``N`` is selectable so the
same machinery serves a 3-patch (tree/shrub/grass) or single-patch column.

Implementation — the two-leaf canopy is already vectorised over its ``ncol`` axis,
so the ``N`` patches are run as ``ncol=N`` in ONE ``compute_two_leaf_canopy_fluxes``
call (no Python loop, no duplicated canopy numerics); the patch axis is then
area-weight-reduced.  A single patch (``frac=1``, unit scales) is bit-identical to
calling the canopy directly.

SCOPE — this is the diagnostic-mode mosaic: it gives each patch its own canopy
parameters and a per-patch water-stress SCALE.  A genuinely per-patch prognostic
water stress (deep-tree vs shallow-grass root-weighted beta from a shared soil
column) is the follow-up that lives in the multilayer soil step; it is the piece
that ultimately closes the savanna GPP-correlation gap and is NOT done here.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.land.surface_scheme.two_leaf_canopy import (
    compute_two_leaf_canopy_fluxes,
)

# The mosaic patch specs are STRUCTURAL sensitivity knobs (tile area fractions,
# per-tile LAI / Vcmax / water-stress scales, tile rooting depth) for an offline
# validation tool — not physics closure parameters injected through the trainable
# param collector.  They are therefore all classified ``excluded`` (fixed at
# run-config time), mirroring the grid-structure configs (e.g. SoilGridConfig).
# Only float fields WITH a default are spec-eligible; ``frac`` / ``root_depth_m``
# are required (no default) and ``ClmmlPatchSpec`` has no defaulted float field, so
# only ``PatchSpec``'s defaulted scale knobs are listed (all excluded).
__param_spec__ = {
    "PatchSpec": {
        "scheme_key": "land.patch_mosaic.patch",
        "excluded": {
            "lai_scale": "mosaic: per-tile LAI multiplier (structural)",
            "vcmax_c3_scale": "mosaic: per-tile C3 Vcmax scale (sensitivity knob)",
            "vcmax_c4_scale": "mosaic: per-tile C4 Vcmax scale (sensitivity knob)",
            "w_frac_rz_scale": "mosaic: per-tile water-stress scale (sensitivity knob)",
        },
        "params": {},
    },
}

# Default woody (tree) area fraction for the 2-tile savanna mosaic.
_DEFAULT_TREE_FRAC = 0.4

# Area fractions must sum to one within this tolerance (round-off on a handful of
# Python-float fractions); a larger drift is a mis-specified mosaic and raises.
_FRAC_SUM_TOL = 1.0e-6

# The CLM-ML MLpftcon table initialises PFT parameters for 1-based indices 1..16;
# any other index leaves them at the -999 sentinel, so a mosaic tile PFT must be a
# plain int in this range.
_CLM_PFT_MIN = 1
_CLM_PFT_MAX = 16


class PatchSpec(NamedTuple):
    """One land patch (tile) of a :class:`PatchMosaicConfig`.

    frac            : patch area fraction of the grid column [0-1]; the mosaic's
                      fractions must sum to 1.
    lai_scale       : patch LAI = ``lai_scale * driver_LAI`` [-].  A tree/grass
                      split conserves total leaf area when
                      ``sum_i frac_i * lai_scale_i == 1``; not enforced (a
                      sensitivity tool may deliberately break it).
    fc4             : patch C4 fraction [0-1]; ``None`` keeps the driver value
                      (e.g. 1.0 for a C4-grass patch, 0.0 for a C3-tree patch).
    vcmax_c3_scale  : scale the patch C3 (tree) Vcmax25 [-].
    vcmax_c4_scale  : scale the patch C4 (grass) Vcmax25 [-].
    w_frac_rz_scale : scale the patch root-zone water-stress beta [-].  A crude
                      diagnostic stand-in for per-patch rooting (deep tree vs
                      shallow grass); <1 => drier patch.  The clip keeps the
                      scaled beta in [0, 1].
    root_depth_m    : per-patch root-density e-folding depth [m] for the PROGNOSTIC
                      outer-loop two-leaf mosaic (deep tree vs shallow grass — the
                      real dry-season water-stress handoff).  ``None`` uses the
                      site root depth.  Ignored by the diagnostic inner-loop
                      (which prescribes a single ``w_frac_rz`` and has no soil
                      column), where ``w_frac_rz_scale`` is the rooting stand-in.
    """
    frac: float
    lai_scale: float = 1.0
    fc4: float | None = None
    vcmax_c3_scale: float = 1.0
    vcmax_c4_scale: float = 1.0
    w_frac_rz_scale: float = 1.0
    root_depth_m: float | None = None


class PatchMosaicConfig(NamedTuple):
    """A selectable-``N`` mosaic of :class:`PatchSpec` patches (``N = len``)."""
    patches: tuple[PatchSpec, ...]

    def validate(self) -> "PatchMosaicConfig":
        """Fail early on a mis-specified mosaic (dispatch hardening).

        Raises ``ValueError`` for: no patches, a non-positive / non-finite area
        fraction, fractions that do not sum to 1, or a non-finite / non-positive
        scale — so a typo aborts at setup, never silently runs a bad column.
        """
        n = len(self.patches)
        if n == 0:
            raise ValueError("PatchMosaicConfig needs >= 1 patch, got none")
        total = 0.0
        for i, p in enumerate(self.patches):
            if not (p.frac > 0.0) or not _finite(p.frac):
                raise ValueError(
                    f"patch {i}: frac must be finite and > 0, got {p.frac}")
            for name in ("lai_scale", "vcmax_c3_scale", "vcmax_c4_scale",
                         "w_frac_rz_scale"):
                v = getattr(p, name)
                if not (v >= 0.0) or not _finite(v):
                    raise ValueError(
                        f"patch {i}: {name} must be finite and >= 0, got {v}")
            if p.fc4 is not None and not (0.0 <= p.fc4 <= 1.0 and _finite(p.fc4)):
                raise ValueError(f"patch {i}: fc4 must be in [0, 1], got {p.fc4}")
            if p.root_depth_m is not None and (
                    not (p.root_depth_m > 0.0) or not _finite(p.root_depth_m)):
                raise ValueError(
                    f"patch {i}: root_depth_m must be finite and > 0, "
                    f"got {p.root_depth_m}")
            total += p.frac
        if abs(total - 1.0) > _FRAC_SUM_TOL:
            raise ValueError(
                f"patch area fractions must sum to 1, got {total} "
                f"(|delta| = {abs(total - 1.0):.3e} > {_FRAC_SUM_TOL:.0e})")
        return self


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def savanna_two_patch(tree_frac: float = _DEFAULT_TREE_FRAC, *,
                      grass_fc4: float = 1.0,
                      grass_vcmax_c4_scale: float = 1.0,
                      grass_vcmax_c3_scale: float = 1.0,
                      tree_vcmax_c3_scale: float = 1.0,
                      grass_w_frac_rz_scale: float = 1.0,
                      tree_w_frac_rz_scale: float = 1.0,
                      tree_root_m: float | None = None,
                      grass_root_m: float | None = None) -> PatchMosaicConfig:
    """A 2-patch woody-savanna mosaic: a C3 tree overstory + a grass understory.

    ``tree_frac`` is the woody (tree) area fraction; the grass patch is the
    residual ``1 - tree_frac``.  ``grass_fc4`` sets the grass photosynthetic
    pathway: 1.0 for a C4-grass understory (tropical savanna, e.g. AU-How) or 0.0
    for a C3-grass understory (Mediterranean savanna, e.g. US-Ton blue-oak /
    C3-annual-grass).  The patches keep the driver LAI by default (``lai_scale=1``);
    the pathway split and per-patch water-stress scales are the savanna tuning
    knobs.  Defaults are a neutral starting point (no data-tuned values baked in).
    """
    if not (0.0 < tree_frac < 1.0):
        raise ValueError(f"tree_frac must be in (0, 1), got {tree_frac}")
    tree = PatchSpec(frac=tree_frac, fc4=0.0,
                     vcmax_c3_scale=tree_vcmax_c3_scale,
                     w_frac_rz_scale=tree_w_frac_rz_scale,
                     root_depth_m=tree_root_m)
    grass = PatchSpec(frac=1.0 - tree_frac, fc4=grass_fc4,
                      vcmax_c3_scale=grass_vcmax_c3_scale,
                      vcmax_c4_scale=grass_vcmax_c4_scale,
                      w_frac_rz_scale=grass_w_frac_rz_scale,
                      root_depth_m=grass_root_m)
    return PatchMosaicConfig(patches=(tree, grass)).validate()


# --- CLM-ML (multilayer canopy) mosaic -------------------------------------
# The CLM-ML forward is jit-traced with a single scalar PFT (``grid.pft``);
# per-column PFT is deferred (S3), so tree + grass tiles cannot share one call.
# Instead each patch is an independent single-PFT CLM-ML column run OVER THE FULL
# TIME SERIES (its own prognostic soil), and the flux series are area-weighted.
# This gives each tile its own PFT, rooting depth and hence its own water-stress
# phenology (the deep-tree vs shallow-grass handoff) — at the cost of NO
# inter-patch soil-water competition (independent columns, not a shared column).


class ClmmlPatchSpec(NamedTuple):
    """One CLM-ML tile of a :class:`ClmmlMosaicConfig`.

    frac             : patch area fraction [0-1]; the mosaic's fractions sum to 1.
    pft_clm          : CLM PFT index (1-based) for this tile (e.g. a tree PFT for
                       the overstory, a C4-grass PFT for the understory).
    root_depth_m     : root-density e-folding depth [m] for this tile's soil column
                       (deep for trees, shallow for grass) — sets the water stress.
    vcmax25_override : optional Vcmax25 [umol m-2 s-1] override for the tile; ``None``
                       uses the CLM ``MLpftcon`` table value for ``pft_clm``.
    """
    frac: float
    pft_clm: int
    root_depth_m: float
    vcmax25_override: float | None = None


class ClmmlMosaicConfig(NamedTuple):
    """A selectable-``N`` mosaic of CLM-ML tiles run as independent columns."""
    patches: tuple[ClmmlPatchSpec, ...]

    def validate(self) -> "ClmmlMosaicConfig":
        """Fail early on a mis-specified CLM-ML mosaic (dispatch hardening)."""
        n = len(self.patches)
        if n == 0:
            raise ValueError("ClmmlMosaicConfig needs >= 1 patch, got none")
        total = 0.0
        for i, p in enumerate(self.patches):
            if not (p.frac > 0.0) or not _finite(p.frac):
                raise ValueError(
                    f"clmml patch {i}: frac must be finite and > 0, got {p.frac}")
            if not isinstance(p.pft_clm, int) or isinstance(p.pft_clm, bool):
                raise ValueError(
                    f"clmml patch {i}: pft_clm must be a plain int, got "
                    f"{type(p.pft_clm).__name__} {p.pft_clm!r}")
            if not (_CLM_PFT_MIN <= p.pft_clm <= _CLM_PFT_MAX):
                raise ValueError(
                    f"clmml patch {i}: pft_clm must be in "
                    f"[{_CLM_PFT_MIN}, {_CLM_PFT_MAX}], got {p.pft_clm}")
            if not (p.root_depth_m > 0.0) or not _finite(p.root_depth_m):
                raise ValueError(
                    f"clmml patch {i}: root_depth_m must be finite and > 0, "
                    f"got {p.root_depth_m}")
            if p.vcmax25_override is not None and (
                    not _finite(p.vcmax25_override) or p.vcmax25_override <= 0.0):
                raise ValueError(
                    f"clmml patch {i}: vcmax25_override must be finite and > 0, "
                    f"got {p.vcmax25_override}")
            total += p.frac
        if abs(total - 1.0) > _FRAC_SUM_TOL:
            raise ValueError(
                f"clmml patch area fractions must sum to 1, got {total}")
        return self


def savanna_clmml_two_patch(*, tree_frac: float, tree_pft: int, grass_pft: int,
                            tree_root_m: float, grass_root_m: float,
                            tree_vcmax25: float | None = None,
                            grass_vcmax25: float | None = None
                            ) -> ClmmlMosaicConfig:
    """A 2-tile woody-savanna CLM-ML mosaic: a deep-rooted tree overstory + a
    shallow-rooted grass understory, run as independent columns.

    ``tree_vcmax25`` / ``grass_vcmax25`` override the tile's MLpftcon table Vcmax25
    (e.g. with the driver's site value, for parity with two-leaf); ``None`` keeps
    the generic table value.
    """
    if not (0.0 < tree_frac < 1.0):
        raise ValueError(f"tree_frac must be in (0, 1), got {tree_frac}")
    tree = ClmmlPatchSpec(frac=tree_frac, pft_clm=tree_pft,
                          root_depth_m=tree_root_m, vcmax25_override=tree_vcmax25)
    grass = ClmmlPatchSpec(frac=1.0 - tree_frac, pft_clm=grass_pft,
                           root_depth_m=grass_root_m,
                           vcmax25_override=grass_vcmax25)
    return ClmmlMosaicConfig(patches=(tree, grass)).validate()


def area_weight_series(series, fracs) -> "jnp.ndarray":
    """Area-weighted sum of per-tile flux time series: ``sum_i frac_i * series_i``.

    ``series`` is a sequence of ``N`` equal-length 1-D arrays (one per tile);
    ``fracs`` the matching area fractions.  Used by the CLM-ML outer-loop mosaic to
    combine the independent-column flux series into one grid series.  All tile
    series must be 1-D and the SAME length, else a shorter/length-1 series would
    silently broadcast and be reused for every timestep.
    """
    import numpy as _np
    fr = _np.asarray(fracs)
    if fr.ndim != 1:
        raise ValueError(f"fracs must be 1-D, got shape {fr.shape}")
    if len(series) == 0 or len(series) != len(fr):
        raise ValueError(f"series/fracs length mismatch: {len(series)} vs {len(fr)}")
    arrs = [_np.asarray(s) for s in series]
    shape0 = arrs[0].shape
    if arrs[0].ndim != 1:
        raise ValueError(f"tile series must be 1-D, got shape {shape0}")
    for k, a in enumerate(arrs):
        if a.shape != shape0:
            raise ValueError(
                f"tile series {k} shape {a.shape} != tile 0 shape {shape0}")
    return sum(f * a for f, a in zip(fr, arrs))


def _bcast_patch(a: jnp.ndarray, n: int) -> jnp.ndarray:
    """Broadcast a single-column leaf ``(1, ...)`` to the patch axis ``(n, ...)``."""
    return jnp.broadcast_to(a, (n,) + a.shape[1:])


def compute_mosaic_canopy_fluxes(
    *,
    mosaic: PatchMosaicConfig,
    T_soil_top: jnp.ndarray,          # (1,) shared soil skin T [K]
    forcing,                          # AtmToSurface, single column (ncol=1)
    canopy_config,
    land_config,
    canopy_params,                    # CanopyLandParams, ncol=1
    w_frac_rz: jnp.ndarray,           # (1,) grid root-zone beta
    wind_speed: jnp.ndarray,          # (1,)
    wind_dir_x: jnp.ndarray,          # (1,)
    wind_dir_y: jnp.ndarray,          # (1,)
    soil_thermal_fn,
    dt: float,
    LAI_override: jnp.ndarray | None = None,   # (1,) grid driver LAI
    TgC_override: jnp.ndarray | None = None,   # (1,)
) -> SurfaceFluxOutput:
    """Run the ``N``-patch mosaic and return one area-weighted grid ``SurfaceFluxOutput``.

    Every input is a single-column (``ncol=1``) quantity for the grid cell — the
    same arguments the caller would pass to :func:`compute_two_leaf_canopy_fluxes`,
    with one added contract: ``soil_thermal_fn`` MUST be shape-polymorphic over the
    patch axis (it is called with a patch-shaped ``(N,)`` ground-heat-flux ``G`` and
    must return a skin ``T`` broadcastable to ``(N,)`` — a ``(1,)`` prescribed value
    is accepted and broadcast).  The diagnostic prescribed-``Ts``
    callback (``lambda G, dt: Ts``) satisfies this; a soil-model callback that
    closes over a single-column state does not and must be wrapped by the caller.

    The patches are expanded onto the ``ncol=N`` axis, run in one canopy call, and
    area-weight-reduced back to ``ncol=1``.
    """
    mosaic = mosaic.validate()             # fail early on a mis-specified mosaic
    patches = mosaic.patches
    n = len(patches)
    fracs = jnp.asarray([p.frac for p in patches])                     # (n,)
    lai_scale = jnp.asarray([p.lai_scale for p in patches])            # (n,)
    vc3_scale = jnp.asarray([p.vcmax_c3_scale for p in patches])
    vc4_scale = jnp.asarray([p.vcmax_c4_scale for p in patches])
    w_scale = jnp.asarray([p.w_frac_rz_scale for p in patches])

    # --- expand the shared single-column inputs onto the patch axis ---
    params_n = jax.tree_util.tree_map(lambda a: _bcast_patch(a, n), canopy_params)
    forcing_n = jax.tree_util.tree_map(lambda a: _bcast_patch(a, n), forcing)
    Tsoil_n = _bcast_patch(T_soil_top, n)
    wind_n = _bcast_patch(wind_speed, n)
    dirx_n = _bcast_patch(wind_dir_x, n)
    diry_n = _bcast_patch(wind_dir_y, n)

    # --- per-patch parameter overrides ---
    base_lai = params_n.LAI if LAI_override is None else _bcast_patch(LAI_override, n)
    lai_n = base_lai * lai_scale
    vc3_n = params_n.Vcmax25_C3_leaf * vc3_scale
    vc4_n = params_n.Vcmax25_C4_leaf * vc4_scale
    # fc4: keep the driver value where a patch left fc4=None, else the patch value.
    fc4_n = jnp.asarray([params_n.fC4[i] if p.fc4 is None else p.fc4
                         for i, p in enumerate(patches)])
    params_n = params_n._replace(LAI=lai_n, Vcmax25_C3_leaf=vc3_n,
                                 Vcmax25_C4_leaf=vc4_n, fC4=fc4_n)
    # Per-patch water stress: a crude diagnostic rooting stand-in (the prognostic
    # per-patch root-weighted beta is the follow-up).  Clipped to a valid beta.
    w_n = jnp.clip(_bcast_patch(w_frac_rz, n) * w_scale, 0.0, 1.0)
    TgC_n = None if TgC_override is None else _bcast_patch(TgC_override, n)

    # The soil-thermal callback must return the patch-shaped skin T (the diagnostic
    # caller ignores G and returns the prescribed Ts); broadcast to the flux shape.
    def _stf_n(G, dt_):
        return jnp.broadcast_to(soil_thermal_fn(G, dt_), jnp.shape(G))

    out = compute_two_leaf_canopy_fluxes(
        T_soil_top=Tsoil_n, forcing=forcing_n, canopy_config=canopy_config,
        land_config=land_config, canopy_params=params_n, w_frac_rz=w_n,
        wind_speed=wind_n, wind_dir_x=dirx_n, wind_dir_y=diry_n,
        soil_thermal_fn=_stf_n, dt=dt, LAI_override=lai_n, TgC_override=TgC_n,
    )
    return _area_weight(out, fracs)


def _area_weight(out: SurfaceFluxOutput, fracs: jnp.ndarray) -> SurfaceFluxOutput:
    """Area-weight-reduce a patch-axis ``SurfaceFluxOutput`` to one grid column.

    ``x_grid = sum_i frac_i * x_i`` over the leading (patch) axis, keeping the
    ``ncol=1`` dimension.  Exact for per-area fluxes; the area-mean for intensive
    state (albedo, ...).  Two fields are reduced non-linearly so the output stays
    self-consistent:

    * ``T_surface`` is the emission-equivalent (radiometric) temperature, so it is
      reconstructed to keep ``eps_grid * sigma * T_grid**4 == sum_i frac_i * eps_i *
      sigma * T_i**4`` (matching the linearly-summed ``lw_up``): with
      ``eps_grid = sum_i frac_i * eps_i``,
      ``T_grid = (sum_i frac_i * eps_i * T_i**4 / eps_grid)**0.25``.  A linear mean
      of ``T_surface`` would emit the wrong grid longwave.
    * ``n_iters`` is a solver iteration COUNT, not a flux — reduced by ``max`` (the
      worst-case patch), keeping it an integer count so a single unit patch is
      identical to the direct canopy call.

    ``None`` fields pass through.
    """
    def w(x):
        weights = fracs.reshape((-1,) + (1,) * (jnp.ndim(x) - 1))
        return jnp.sum(weights * x, axis=0, keepdims=True)
    agg = jax.tree_util.tree_map(w, out)
    if out.T_surface is not None and out.emissivity is not None:
        eps_grid = w(out.emissivity)
        t4 = w(out.emissivity * out.T_surface ** 4)
        agg = agg._replace(T_surface=(t4 / eps_grid) ** 0.25)
    if out.n_iters is not None:
        agg = agg._replace(n_iters=jnp.max(out.n_iters, axis=0, keepdims=True))
    return agg
