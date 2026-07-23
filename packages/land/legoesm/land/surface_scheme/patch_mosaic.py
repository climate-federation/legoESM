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

# Area fractions must sum to one within this tolerance (round-off on a handful of
# Python-float fractions); a larger drift is a mis-specified mosaic and raises.
_FRAC_SUM_TOL = 1.0e-6


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
    """
    frac: float
    lai_scale: float = 1.0
    fc4: float | None = None
    vcmax_c3_scale: float = 1.0
    vcmax_c4_scale: float = 1.0
    w_frac_rz_scale: float = 1.0


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
            total += p.frac
        if abs(total - 1.0) > _FRAC_SUM_TOL:
            raise ValueError(
                f"patch area fractions must sum to 1, got {total} "
                f"(|delta| = {abs(total - 1.0):.3e} > {_FRAC_SUM_TOL:.0e})")
        return self


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def savanna_two_patch(tree_frac: float = 0.4, *,
                      grass_vcmax_c4_scale: float = 1.0,
                      tree_vcmax_c3_scale: float = 1.0,
                      grass_w_frac_rz_scale: float = 1.0,
                      tree_w_frac_rz_scale: float = 1.0) -> PatchMosaicConfig:
    """A 2-patch woody-savanna mosaic: a C3 tree overstory + a C4 grass understory.

    ``tree_frac`` is the woody (tree) area fraction; the grass patch is the
    residual ``1 - tree_frac``.  The patches keep the driver LAI by default
    (``lai_scale=1``); the C4/C3 split and per-patch water-stress scales are the
    savanna tuning knobs.  Defaults are a neutral starting point (no data-tuned
    values are baked in — the real values await site validation).
    """
    if not (0.0 < tree_frac < 1.0):
        raise ValueError(f"tree_frac must be in (0, 1), got {tree_frac}")
    tree = PatchSpec(frac=tree_frac, fc4=0.0,
                     vcmax_c3_scale=tree_vcmax_c3_scale,
                     w_frac_rz_scale=tree_w_frac_rz_scale)
    grass = PatchSpec(frac=1.0 - tree_frac, fc4=1.0,
                      vcmax_c4_scale=grass_vcmax_c4_scale,
                      w_frac_rz_scale=grass_w_frac_rz_scale)
    return PatchMosaicConfig(patches=(tree, grass)).validate()


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
