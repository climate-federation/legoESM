"""
JAX translation of MLCanopyWaterMod Fortran module.

Canopy water interception, evaporation, and wetted fraction.
Provides three public routines:

- :func:`CanopyWettedFraction`: wetted and dry fractions of each layer.
- :func:`CanopyInterception`: interception and throughfall.
- :func:`CanopyEvaporation`: update intercepted water for evaporation/dew.

Original Fortran module: MLCanopyWaterMod
Fortran lines 1-215

Differentiability notes
-----------------------
* All ``np.asarray()`` calls removed — JAX arrays used directly.
* All ``np.`` operations replaced by ``jnp.``; ``math.tanh`` → ``jnp.tanh``.
* ``int(ncan[p])`` and ``slice(1, ncan+1)`` replaced by full static slices
  ``[p, 1:]`` over ``nlevmlcan`` layers; inactive layers are masked by
  ``dpai == 0``.
* Python ``if`` on traced values (``total_precip > 0``, ``n > 0``) replaced
  by ``jnp.where`` with safe denominators.
* ``int(np.sum(has_pai))`` replaced by ``jnp.sum(has_pai)`` with a JAX
  safe floor to avoid division by zero.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import (  # noqa: F401
    dewmx,
    fwet_exponent,
    interception_fraction,
    maximum_leaf_wetted_fraction,
    mmh2o,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import dtime_ml  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401

# ---------------------------------------------------------------------------
# Public: wetted fraction of canopy
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1))
def CanopyWettedFraction(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Calculate the wetted fraction and the green-dry fraction of each
    canopy layer.

    Mirrors Fortran subroutine ``CanopyWettedFraction`` (lines 31-80).

    For layers with ``dpai > 0`` (Fortran lines 61-73):

    .. code-block:: none

        h2ocanmx = dewmx * dpai(ic)
        fwet     = min(max(h2ocan/h2ocanmx, 0)^fwet_exponent,
                       maximum_leaf_wetted_fraction)
        fdry     = (1 - fwet) * dlai / dpai

    Layers with ``dpai == 0`` receive ``fwet = fdry = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        mlcanopy_inst: Canopy container; ``fwet_profile`` and
            ``fdry_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    fwet = mlcanopy_inst.fwet_profile
    fdry = mlcanopy_inst.fdry_profile

    # grid.p may be a TRACED lax.scan-over-columns index (S3); it is used only as a
    # gather/scatter index here (full 1: layer slices, dpai==0 masks inactive rows),
    # so no concrete ncan is needed and grid can be a plain dynamic arg.
    for fp in range(1 if grid is not None else num_filter):  # Fortran: do fp = 1, num_filter
        p = grid.p if grid is not None else filter_patch[fp]

        # Use full layer slices — dpai==0 on inactive layers acts as mask
        dpai_v = mlcanopy_inst.dpai_profile[p, 1:]  # (nlevmlcan,)
        h2ocan_v = mlcanopy_inst.h2ocan_profile[p, 1:]
        dlai_v = mlcanopy_inst.dlai_profile[p, 1:]

        has_pai = dpai_v > 0.0
        dpai_safe = jnp.where(has_pai, dpai_v, 1.0)  # avoid /0
        h2ocanmx = dewmx * dpai_safe

        # Fortran lines 63-68
        # Guard base of fractional power against zero to prevent inf
        # gradients.  0^0.67 → grad = 0.67*0^(-0.33) = inf.  Inside
        # jnp.where the inf gradient × 0 mask = NaN.  Clamp to tiny
        # positive floor so gradient stays finite.
        _eps_pow = jnp.asarray(1.0e-30)
        fwet_base = jnp.maximum(h2ocan_v / h2ocanmx, _eps_pow)
        fwet_v = jnp.where(
            has_pai,
            jnp.minimum(
                fwet_base**fwet_exponent,
                maximum_leaf_wetted_fraction,
            ),
            0.0,
        )
        fdry_v = jnp.where(has_pai, (1.0 - fwet_v) * dlai_v / dpai_safe, 0.0)

        fwet = fwet.at[p, 1:].set(fwet_v)
        fdry = fdry.at[p, 1:].set(fdry_v)

    return mlcanopy_inst._replace(
        fwet_profile=fwet,
        fdry_profile=fdry,
    )


# ---------------------------------------------------------------------------
# Public: interception and throughfall
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1))
def CanopyInterception(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Calculate canopy interception and throughfall.

    Mirrors Fortran subroutine ``CanopyInterception`` (lines 82-165).

    **Patch-level calculations** (Fortran lines 120-133):

    .. code-block:: none

        fracrain = qflx_rain / (qflx_snow + qflx_rain)  [or 0 if no precip]
        fracsnow = qflx_snow / (qflx_snow + qflx_rain)
        fpi      = interception_fraction * tanh(lai + sai)
        qflx_through_rain = qflx_rain * (1 - fpi)
        qflx_through_snow = qflx_snow * (1 - fpi)
        qflx_intr         = (qflx_snow + qflx_rain) * fpi

    Intercepted precipitation is distributed equally across all layers
    with ``dpai > 0`` (count = ``n``).  Per-layer water balance
    (Fortran lines 141-153):

    .. code-block:: none

        h2ocan(ic) = h2ocan_bef(ic) + qflx_intr * dtime / n
        xrun       = (h2ocan(ic) - h2ocanmx) / dtime
        if xrun > 0:
            qflx_candrip += xrun
            h2ocan(ic)    = h2ocanmx

    Total throughfall onto ground (Fortran lines 157-158):

    .. code-block:: none

        qflx_tflrain = qflx_through_rain + qflx_candrip * fracrain
        qflx_tflsnow = qflx_through_snow + qflx_candrip * fracsnow

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        mlcanopy_inst: Canopy container; ``h2ocan_profile``,
            ``qflx_intr_canopy``, ``qflx_tflrain_canopy``, and
            ``qflx_tflsnow_canopy`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dtime = dtime_ml  # Python float constant — no cast needed

    h2ocan = mlcanopy_inst.h2ocan_profile
    qflx_intr = mlcanopy_inst.qflx_intr_canopy
    qflx_tflrain = mlcanopy_inst.qflx_tflrain_canopy
    qflx_tflsnow = mlcanopy_inst.qflx_tflsnow_canopy

    for fp in range(1 if grid is not None else num_filter):  # Fortran: do fp = 1, num_filter
        p = grid.p if grid is not None else filter_patch[fp]  # grid.p may be TRACED (S3)

        rain_p = mlcanopy_inst.qflx_rain_forcing[p]
        snow_p = mlcanopy_inst.qflx_snow_forcing[p]

        # Rain/snow fractions — Fortran lines 120-125
        # jnp.where replaces Python if on JAX value
        total_precip = snow_p + rain_p
        # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
        total_safe = jnp.maximum(total_precip, 1.0e-30)
        fracrain = jnp.where(total_precip > 0.0, rain_p / total_safe, 0.0)
        fracsnow = jnp.where(total_precip > 0.0, snow_p / total_safe, 0.0)

        # Intercepted fraction (CLM5 form) — Fortran line 127
        lai_p = mlcanopy_inst.lai_canopy[p]
        sai_p = mlcanopy_inst.sai_canopy[p]
        fpi = interception_fraction * jnp.tanh(lai_p + sai_p)

        # Direct throughfall — Fortran lines 129-130
        qflx_through_rain = rain_p * (1.0 - fpi)
        qflx_through_snow = snow_p * (1.0 - fpi)

        # Intercepted precipitation — Fortran line 132
        qflx_intr_p = total_precip * fpi
        qflx_intr = qflx_intr.at[p].set(qflx_intr_p)

        # Per-layer water balance — Fortran lines 134-153
        dpai_v = mlcanopy_inst.dpai_profile[p, 1:]
        h2ocan_bef_v = mlcanopy_inst.h2ocan_bef_profile[p, 1:]

        has_pai = dpai_v > 0.0
        n_active = jnp.sum(has_pai.astype(jnp.float32))  # JAX scalar
        n_safe = jnp.maximum(n_active, 1.0)  # avoid /0

        # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
        dpai_safe = jnp.maximum(dpai_v, 1.0e-30)
        h2ocanmx_v = dewmx * dpai_safe
        # Distribute interception equally across active layers
        add_per_layer = jnp.where(n_active > 0.0, qflx_intr_p * dtime / n_safe, 0.0)
        h2ocan_v = jnp.where(has_pai, h2ocan_bef_v + add_per_layer, 0.0)
        xrun_v = jnp.where(has_pai, (h2ocan_v - h2ocanmx_v) / dtime, 0.0)
        drip_mask = xrun_v > 0.0

        # Collect canopy drip and cap h2ocan at maximum — Fortran lines 149-152
        qflx_candrip = jnp.sum(jnp.where(drip_mask, xrun_v, 0.0))
        h2ocan_v = jnp.where(drip_mask, h2ocanmx_v, h2ocan_v)

        h2ocan = h2ocan.at[p, 1:].set(h2ocan_v)

        # Total throughfall — Fortran lines 157-158
        qflx_tflrain = qflx_tflrain.at[p].set(qflx_through_rain + qflx_candrip * fracrain)
        qflx_tflsnow = qflx_tflsnow.at[p].set(qflx_through_snow + qflx_candrip * fracsnow)

    return mlcanopy_inst._replace(
        h2ocan_profile=h2ocan,
        qflx_intr_canopy=qflx_intr,
        qflx_tflrain_canopy=qflx_tflrain,
        qflx_tflsnow_canopy=qflx_tflsnow,
    )


# ---------------------------------------------------------------------------
# Public: update intercepted water for evaporation and dew
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1))
def CanopyEvaporation(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Update canopy intercepted water for evaporation and dew deposition.

    Mirrors Fortran subroutine ``CanopyEvaporation`` (lines 167-215).

    For each layer with ``dpai > 0``, dew deposition is applied first
    (negative evaporation/transpiration flux = condensation), then
    positive evaporation is removed.  Sunlit and shaded leaf populations
    are treated separately and weighted by ``fracsun`` and
    ``(1 - fracsun)`` (Fortran lines 195-210):

    .. code-block:: none

        # --- Dew (add negative fluxes) ---
        dew_sun = (evleaf_sun + trleaf_sun) * fracsun * dpai * mmh2o * dtime
        if dew_sun < 0:  h2ocan -= dew_sun

        dew_sha = (evleaf_sha + trleaf_sha) * (1-fracsun) * dpai * mmh2o * dtime
        if dew_sha < 0:  h2ocan -= dew_sha

        # --- Evaporation (remove positive fluxes) ---
        if evleaf_sun > 0:  h2ocan -= evleaf_sun * fracsun * dpai * mmh2o * dtime
        if evleaf_sha > 0:  h2ocan -= evleaf_sha * (1-fracsun) * dpai * mmh2o * dtime

    The commented-out floor ``h2ocan = max(0, h2ocan)`` is preserved
    as a comment.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        mlcanopy_inst: Canopy container; ``h2ocan_profile`` is updated
            in-place (input/output field).

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dtime = dtime_ml  # Python float constant
    h2ocan = mlcanopy_inst.h2ocan_profile

    for fp in range(1 if grid is not None else num_filter):  # Fortran: do fp = 1, num_filter
        p = grid.p if grid is not None else filter_patch[fp]  # grid.p may be TRACED (S3)

        dpai_v = mlcanopy_inst.dpai_profile[p, 1:]
        fracsun_v = mlcanopy_inst.fracsun_profile[p, 1:]
        h2o_v = h2ocan[p, 1:]

        has_pai = dpai_v > 0.0
        # factor = dpai * mmh2o * dtime; zero on empty layers (no division)
        factor_v = jnp.where(has_pai, dpai_v, 0.0) * mmh2o * dtime

        evleaf_sun_v = mlcanopy_inst.evleaf_leaf[p, 1:, isun]
        trleaf_sun_v = mlcanopy_inst.trleaf_leaf[p, 1:, isun]
        evleaf_sha_v = mlcanopy_inst.evleaf_leaf[p, 1:, isha]
        trleaf_sha_v = mlcanopy_inst.trleaf_leaf[p, 1:, isha]

        # Dew (negative flux adds to h2ocan) — Fortran lines 195-203
        dew_sun_v = (evleaf_sun_v + trleaf_sun_v) * fracsun_v * factor_v
        dew_sha_v = (evleaf_sha_v + trleaf_sha_v) * (1.0 - fracsun_v) * factor_v
        h2o_v = h2o_v - jnp.where(dew_sun_v < 0.0, dew_sun_v, 0.0)
        h2o_v = h2o_v - jnp.where(dew_sha_v < 0.0, dew_sha_v, 0.0)

        # Evaporation (positive flux removes from h2ocan) — Fortran lines 205-211
        h2o_v = h2o_v - jnp.where(evleaf_sun_v > 0.0, evleaf_sun_v * fracsun_v * factor_v, 0.0)
        h2o_v = h2o_v - jnp.where(
            evleaf_sha_v > 0.0, evleaf_sha_v * (1.0 - fracsun_v) * factor_v, 0.0
        )

        # Preserve unchanged values for inactive layers
        h2o_v = jnp.where(has_pai, h2o_v, h2ocan[p, 1:])
        h2ocan = h2ocan.at[p, 1:].set(h2o_v)

    return mlcanopy_inst._replace(h2ocan_profile=h2ocan)
