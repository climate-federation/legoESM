"""Convection output container.

ConvectionOutput is the common interface produced by every convection
backend (SBM, DCA, Kuo, mass-flux, EDMF). Convection no longer carries
its own surface-precipitation diagnostic; instead it emits a 3D
convective source for cloud water (``dq_c_conv_dt``). Downstream the
orchestrator routes that source into the cloud-water budget so that
microphysics processes the convective condensate through its
autoconversion / sedimentation / evaporation chain. Total surface
precipitation is the sole responsibility of microphysics
(``MicrophysicsOutput.precipitation``).

This routing was introduced because the previous scalar
``precipitation`` field forced every convection scheme to assume that
all detrained condensate falls instantly to the surface, bypassing
melting, evaporation in dry layers, and proper terminal-velocity
sedimentation — a restrictive simplification.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ConvectionOutput(NamedTuple):
    """Output from a convection scheme (backend-agnostic).

    All required fields are at full levels with shape ``(ncol, nlev)``
    except column-mean diagnostics which have shape ``(ncol,)``.

    Fields
    ------
    dT_dt : jax.Array
        Temperature tendency [K/s], shape (ncol, nlev).
    dq_v_dt : jax.Array
        Water vapor specific humidity tendency [kg/kg/s], shape
        (ncol, nlev).
    dq_c_conv_dt : jax.Array
        Convective source term for cloud water mixing ratio [kg/kg/s],
        shape (ncol, nlev). Replaces the legacy scalar surface
        ``precipitation``: convective condensate now joins the
        cloud-water bucket and is processed by microphysics, which
        owns the resulting surface precipitation diagnostic. By
        construction this field is non-negative.
    cape : jax.Array
        CAPE diagnostic [J/kg], shape (ncol,).
    convective_mask : jax.Array
        Smooth 0-1 convective indicator, shape (ncol,).
    du_dt_conv : jax.Array or None
        Optional convective momentum tendency for zonal wind [m/s²],
        shape (ncol, nlev).  ``None`` for schemes without convective
        momentum transport (the existing five schemes plus the
        Kain-Fritsch and Emanuel schemes added in PRs 2 and 3).
        Populated by Zhang-McFarlane (PR 1), Tiedtke (PR 4), and
        Bechtold/IFS (PR 5) via the Gregory et al. 1997 closure.  When
        ``None``, the integration bridge zero-fills the dynamical-core
        wind tendencies.
    dv_dt_conv : jax.Array or None
        Optional convective momentum tendency for meridional wind
        [m/s²], shape (ncol, nlev).  Same conventions as ``du_dt_conv``.
    """
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    dq_c_conv_dt: jax.Array
    cape: jax.Array
    convective_mask: jax.Array
    du_dt_conv: jax.Array | None = None
    dv_dt_conv: jax.Array | None = None
    dq_r_conv_dt: jax.Array | None = None
    """Optional convective source for RAIN mixing ratio [kg/kg/s], shape
    (ncol, nlev).  ``None`` (default) for schemes that detrain all
    condensate as cloud water.  Tiedtke's and Bechtold's in-updraft
    precipitation (via the shared ``split_convective_rain``) splits the
    plume condensate: the precipitated fraction is emitted here as rain --
    an already-fallen species the pipeline column-integrates DIRECTLY to
    surface precipitation (NOT routed through microphysics sedimentation,
    and invisible to radiation, which sees only ``q_c``/``q_i``) -- while
    the anvil remainder stays in ``dq_c_conv_dt``.  Without this split,
    100% of convective condensate loads the grid-scale cloud and the
    radiation, which microphysics cannot drain fast enough
    (source-buffered)."""


def split_convective_rain(dq_c_conv_dt, precip_efficiency):
    """Split a convective cloud-water source into rain + suspended cloud.

    Real convective updrafts convert a large fraction of their condensate
    to PRECIPITATION before detrainment.  Diverting that fraction to rain —
    an already-fallen species the pipeline column-integrates DIRECTLY to
    surface precipitation (``physics_pipeline`` ~L1152; NOT routed back
    through microphysics sedimentation, and invisible to radiation, which
    sees only ``q_c``/``q_i``) — instead of detraining 100% as suspended
    anvil cloud is what keeps the grid-scale cloud (and its albedo) from
    saturating faster than microphysics can drain it (the source-buffered
    over-bright-anvil / dry-column-runaway failure mode).

    Shared by every mass-flux-style scheme that detrains to cloud water
    (Tiedtke, Bechtold) so the rain split is defined once (no re-derived
    per-scheme copy).

    Parameters
    ----------
    dq_c_conv_dt : jax.Array
        Convective cloud-water source [kg/kg/s], shape (ncol, nlev). Clamped
        to its non-negative part here (a convective source is a source).
    precip_efficiency : float
        Fraction [0, 1] of the (positive) condensate converted to rain.
        ``0`` (default across schemes) ⇒ NO split, legacy behaviour: the
        full condensate stays as cloud water and ``dq_r_conv_dt`` is ``None``
        (byte-identical to the pre-split code path).

    Returns
    -------
    (dq_c_new, dq_r) : tuple[jax.Array, jax.Array | None]
        ``dq_c_new`` is the anvil cloud-water remainder ``(1-pe)`` of the
        positive condensate; ``dq_r`` is the rain source ``pe`` of it, or
        ``None`` when ``precip_efficiency <= 0``.  MASS is conserved:
        ``dq_c_new + dq_r == max(dq_c_conv_dt, 0)`` exactly (no extra
        source/sink introduced by the split).
    """
    dq_c_pos = jnp.maximum(dq_c_conv_dt, 0.0)
    if precip_efficiency > 0.0:
        pe = jnp.clip(precip_efficiency, 0.0, 1.0)
        return dq_c_pos * (1.0 - pe), dq_c_pos * pe
    return dq_c_pos, None


def convective_autoconversion_split(dq_c_conv_dt, q_c_u, q_c_crit, pe_max):
    """Physical (Sundqvist 1978) autoconversion split of the detrained
    convective condensate source into surface rain + suspended anvil.

    Replaces the CONSTANT ``precip_efficiency`` of :func:`split_convective_rain`:
    the precipitating fraction at each level EMERGES from the updraft
    cloud-water content ``q_c_u`` (the plume's ``Plume.q_c_u``) instead of being
    imposed uniformly.  Sundqvist (1978) smooth autoconversion threshold:

    .. math::

        pe(z) = pe_{max}\\,\\bigl(1 - e^{-(q_{c,u}(z)/q_{c,crit})^2}\\bigr)

    so a thin updraft (``q_c_u`` \\ll ``q_c_crit``) detrains its condensate as
    suspended anvil (``pe`` -> 0), while a water-loaded updraft (``q_c_u``
    \\gg ``q_c_crit``) autoconverts nearly all of it to rain (``pe`` ->
    ``pe_max``).  This is the physically-motivated in-plume microphysics the
    constant knob approximates: precip efficiency rises with the updraft
    condensate loading rather than being a single tuned number.

    SIGN / UNITS (positive-source convention): ``dq_c_conv_dt`` [kg/kg/s] and
    ``q_c_u`` [kg/kg] are non-negative (a convective source / a water content);
    the rain source ``dq_r`` [kg/kg/s] >= 0 is column-integrated DIRECTLY to
    positive-down surface precip by the pipeline (as for
    :func:`split_convective_rain`).  MASS is conserved exactly — per level
    ``dq_c_new + dq_r == max(dq_c_conv_dt, 0)`` — the split adds no source/sink.

    Parameters
    ----------
    dq_c_conv_dt : jax.Array
        Convective cloud-water source [kg/kg/s], shape (ncol, nlev). Clamped to
        its non-negative part.
    q_c_u : jax.Array
        Updraft cloud-water mixing ratio [kg/kg], shape (ncol, nlev), from the
        plume.  Non-negative; clamped defensively.
    q_c_crit : float
        Critical updraft cloud water [kg/kg] setting the Sundqvist threshold —
        the loading at which ~63 % of the condensate precipitates.
    pe_max : float
        Ceiling on the precipitating fraction, in [0, 1].

    Returns
    -------
    (dq_c_new, dq_r) : tuple[jax.Array, jax.Array]
        Anvil cloud-water remainder ``(1 - pe(z))`` and rain source ``pe(z)`` of
        the positive condensate.  Unlike :func:`split_convective_rain`, ``dq_r``
        is ALWAYS an array (never ``None``): the physical efficiency is a field,
        not a scheme-level on/off — it is simply ~0 wherever ``q_c_u`` -> 0.
    """
    dq_c_pos = jnp.maximum(dq_c_conv_dt, 0.0)
    # Sundqvist (1978) smooth threshold on the updraft cloud-water loading;
    # jnp.maximum(q_c_crit, tiny) is a divide-by-zero floor, not a tunable.
    ratio = jnp.maximum(q_c_u, 0.0) / jnp.maximum(q_c_crit, 1e-30)
    pe = pe_max * (1.0 - jnp.exp(-(ratio * ratio)))
    pe = jnp.clip(pe, 0.0, 1.0)
    return dq_c_pos * (1.0 - pe), dq_c_pos * pe
