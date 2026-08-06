"""
JAX translation of MLLeafBoundaryLayerMod Fortran module.

Leaf boundary layer conductance for heat, water vapour, and CO2.

Original Fortran module: MLLeafBoundaryLayerMod
Fortran lines 1-140

Differentiability notes
-----------------------
The inner layer loop is replaced by ``jax.vmap`` over the full layer
dimension.  ``max()`` calls are replaced by ``jnp.maximum``; the
``dpai == 0`` guard uses ``jnp.where`` instead of a Python ``if``.
``gb_type`` is a module-level Python integer (static at trace time) so
its ``if/elif`` is evaluated once during tracing — no issue.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import grav, tfrz  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.pftconMod import pftcon  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import dc0, dh0, dv0, gb_factor, gbh_min, visc0  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import gb_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401

# ---------------------------------------------------------------------------
# Per-layer kernel — vmapped over layer axis
# ---------------------------------------------------------------------------


def _gb_layer(dpai_ic, wind_ic, tair_ic, tleaf_ic, visc, dh, dv_dh, dc_dh, dl, rhomol_p):
    """Boundary layer conductance for one canopy layer (differentiable).

    All scalar arguments (visc, dh, …) are patch-level constants broadcast
    across layers by the vmap caller.

    Returns:
        Tuple ``(gbh, gbv, gbc)`` in mol/m2/s; all zero when dpai_ic == 0.
    """
    re = wind_ic * dl / visc
    pr = visc / dh
    # Free-convection: clamp negative buoyancy to zero — Fortran lines 80-83
    gr = grav * dl**3 * jnp.maximum(tleaf_ic - tair_ic, 0.0) / (tair_ic * visc * visc)

    # Guard bases of fractional powers against zero to prevent inf gradients.
    # When re=0 or gr=0, x^n has gradient n*x^(n-1) → inf.  Inside
    # jnp.where(dpai>0, ..., 0) the inf gradient is multiplied by the mask
    # 0 → 0*inf = NaN.  jnp.maximum clips to a tiny positive floor so the
    # gradient is finite (n*eps^(n-1)), and 0*finite = 0.
    _eps_pow = jnp.asarray(1.0e-30)
    re_safe = jnp.maximum(re, _eps_pow)
    gr_safe = jnp.maximum(gr, _eps_pow)

    # (i) Laminar forced convection — Fortran lines 87-90
    nu_lam = gb_factor * 0.66 * pr**0.33 * re_safe**0.5
    gbh_lam = jnp.maximum((dh * nu_lam / dl) * rhomol_p, gbh_min)
    gbv_lam = gbh_lam * dv_dh**0.67
    gbc_lam = gbh_lam * dc_dh**0.67

    # (ii) Turbulent forced convection — Fortran lines 92-96
    nu_turb = gb_factor * 0.036 * pr**0.33 * re_safe**0.8
    gbh_turb = jnp.maximum((dh * nu_turb / dl) * rhomol_p, gbh_min)
    gbv_turb = gbh_turb * dv_dh**0.67
    gbc_turb = gbh_turb * dc_dh**0.67

    # (iii) Free convection — Fortran lines 98-101
    nu_free = 0.54 * pr**0.25 * gr_safe**0.25
    gbh_free = (dh * nu_free / dl) * rhomol_p
    gbv_free = gbh_free * dv_dh**0.75
    gbc_free = gbh_free * dc_dh**0.75

    # Select flow regime — gb_type is a static Python int, evaluated at trace time
    # Fortran lines 103-121
    if gb_type == 1:
        gbh_val = gbh_lam
        gbv_val = gbv_lam
        gbc_val = gbc_lam
    elif gb_type == 2:
        gbh_val = jnp.maximum(gbh_lam, gbh_turb)
        gbv_val = jnp.maximum(gbv_lam, gbv_turb)
        gbc_val = jnp.maximum(gbc_lam, gbc_turb)
    elif gb_type == 3:
        gbh_val = jnp.maximum(gbh_lam, gbh_turb) + gbh_free
        gbv_val = jnp.maximum(gbv_lam, gbv_turb) + gbv_free
        gbc_val = jnp.maximum(gbc_lam, gbc_turb) + gbc_free
    else:
        # Unreachable at runtime; provides valid JAX values to satisfy tracer
        gbh_val = gbv_val = gbc_val = jnp.zeros(())

    # Zero-out empty layers without Python if — Fortran lines 123-125
    mask = dpai_ic > 0.0
    return (
        jnp.where(mask, gbh_val, 0.0),
        jnp.where(mask, gbv_val, 0.0),
        jnp.where(mask, gbc_val, 0.0),
    )


# vmap: first four args are per-layer (axis 0); remaining six are patch scalars
_gb_layers = jax.vmap(
    _gb_layer,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None),
)

# Double-vmap: outer axis is sun/shade (axis 0 of tleaf_both = shape (2, nlevmlcan));
# dpai, wind, tair are shared across sun/shade (in_axes=None for outer vmap).
# One GPU kernel dispatch computes boundary layer for both sun and shade leaves.
_gb_layers_both = jax.vmap(
    _gb_layers,
    in_axes=(None, None, None, 0, None, None, None, None, None, None),
)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1, 2))
def LeafBoundaryLayer(
    num_filter: int,
    filter_patch: Sequence[int],
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf boundary layer conductance for heat, water vapour,
    and CO2.

    Mirrors Fortran subroutine ``LeafBoundaryLayer`` (lines 22-140).

    Reference: Bonan (2019) *Climate Change and Terrestrial Ecosystem
    Modeling*, Chapter 10.

    Molecular diffusivities are corrected for temperature and pressure
    relative to standard conditions (Fortran lines 72-76):

    .. code-block:: none

        fac  = 101325 / pref * (tref / tfrz)^1.81
        visc = visc0 * fac
        dh   = dh0 * fac
        dv   = dv0 * fac
        dc   = dc0 * fac

    For each leaf layer with ``dpai > 0``, dimensionless numbers are
    computed (Fortran lines 80-83):

    .. code-block:: none

        Re = wind * dleaf / visc
        Pr = visc / dh
        Gr = grav * dleaf^3 * max(tleaf - tair, 0) / (tair * visc^2)

    Three Nusselt-number regimes are computed (Fortran lines 85-101):

    - **Laminar forced** (Fortran lines 87-90):
      ``Nu = gb_factor * 0.66 * Pr^0.33 * Re^0.5``
    - **Turbulent forced** (Fortran lines 92-96):
      ``Nu = gb_factor * 0.036 * Pr^0.33 * Re^0.8``
    - **Free convection** (Fortran lines 98-101):
      ``Nu = 0.54 * Pr^0.25 * Gr^0.25``

    Conductances from each regime are:

    .. code-block:: none

        gbh = (dh * Nu / dleaf) * rhomol    [mol/m2/s]
        gbv = gbh * (dv/dh)^exponent
        gbc = gbh * (dc/dh)^exponent

    Exponent is 0.67 for forced convection and 0.75 for free convection.
    ``gbh_lam`` and ``gbh_turb`` are floored at ``gbh_min``.

    The ``gb_type`` switch selects which regimes are combined
    (Fortran lines 103-121):

    - **1** — laminar only.
    - **2** — max of laminar and turbulent.
    - **3** — max of laminar and turbulent, plus free convection.

    Layers with ``dpai == 0`` receive all conductances set to zero.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; ``gbh_leaf``, ``gbv_leaf``,
            and ``gbc_leaf`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dleaf = pftcon.dleaf

    gbh = mlcanopy_inst.gbh_leaf
    gbv = mlcanopy_inst.gbv_leaf
    gbc = mlcanopy_inst.gbc_leaf

    for fp in range(num_filter):  # Fortran: do fp = 1, num_filter
        p = filter_patch[fp]
        pft = patch.itype[p]  # JAX int — dynamic index

        # Diffusivity correction for temperature and pressure — Fortran lines 72-76
        pref_p = mlcanopy_inst.pref_forcing[p]
        tref_p = mlcanopy_inst.tref_forcing[p]
        fac = 101325.0 / pref_p * (tref_p / tfrz) ** 1.81
        visc = visc0 * fac
        dh = dh0 * fac
        dv = dv0 * fac
        dc = dc0 * fac
        rhomol_p = mlcanopy_inst.rhomol_forcing[p]
        dl = dleaf[pft]  # JAX dynamic gather
        dv_dh = dv / dh
        dc_dh = dc / dh

        # vmap over all layers (1..nlevmlcan); dpai==0 layers yield zero
        gbh_v, gbv_v, gbc_v = _gb_layers(
            mlcanopy_inst.dpai_profile[p, 1:],  # (nlevmlcan,)
            mlcanopy_inst.wind_profile[p, 1:],
            mlcanopy_inst.tair_profile[p, 1:],
            mlcanopy_inst.tleaf_leaf[p, 1:, il],
            visc,
            dh,
            dv_dh,
            dc_dh,
            dl,
            rhomol_p,
        )

        gbh = gbh.at[p, 1:, il].set(gbh_v)
        gbv = gbv.at[p, 1:, il].set(gbv_v)
        gbc = gbc.at[p, 1:, il].set(gbc_v)

    return mlcanopy_inst._replace(
        gbh_leaf=gbh,
        gbv_leaf=gbv,
        gbc_leaf=gbc,
    )


# ---------------------------------------------------------------------------
# Public: batched sun+shade driver (one GPU dispatch for both leaf types)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1))
def LeafBoundaryLayerBoth(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """Compute leaf boundary layer conductance for both sun and shade in one
    fused GPU kernel dispatch.

    Replaces two sequential ``LeafBoundaryLayer(isun, ...)`` +
    ``LeafBoundaryLayer(isha, ...)`` calls.  The only per-leaf input that
    differs between sun and shade is ``tleaf``; all diffusivity corrections
    and layer-geometry inputs are shared.  By stacking ``tleaf`` as a
    ``(2, nlevmlcan)`` tensor and using ``_gb_layers_both`` (an outer vmap
    over the sun/shade dimension), both leaf types are computed in a single
    XLA kernel — halving the number of GPU dispatches vs the sequential path.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter.
        mlcanopy_inst: Canopy container; ``gbh_leaf``, ``gbv_leaf``,
            and ``gbc_leaf`` are updated for both ``isun`` and ``isha``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dleaf = pftcon.dleaf

    gbh = mlcanopy_inst.gbh_leaf
    gbv = mlcanopy_inst.gbv_leaf
    gbc = mlcanopy_inst.gbc_leaf

    for fp in range(1 if grid is not None else num_filter):
        p = grid.p if grid is not None else filter_patch[fp]  # grid.p may be TRACED (S3)
        pft = patch.itype[p]  # patch.itype is jnp → dynamic gather is traced-p safe

        # Diffusivity correction (same for sun and shade) — Fortran lines 72-76
        pref_p = mlcanopy_inst.pref_forcing[p]
        tref_p = mlcanopy_inst.tref_forcing[p]
        fac = 101325.0 / pref_p * (tref_p / tfrz) ** 1.81
        visc = visc0 * fac
        dh = dh0 * fac
        dv = dv0 * fac
        dc = dc0 * fac
        rhomol_p = mlcanopy_inst.rhomol_forcing[p]
        dl = dleaf[pft]
        dv_dh = dv / dh
        dc_dh = dc / dh

        # Stack sun and shade tleaf: shape (2, nlevmlcan)
        # [0] = isun (index 1), [1] = isha (index 2)
        tleaf_both = jnp.stack(
            [
                mlcanopy_inst.tleaf_leaf[p, 1:, isun],
                mlcanopy_inst.tleaf_leaf[p, 1:, isha],
            ],
            axis=0,
        )

        # One fused vmap over sun/shade + layers — half the GPU dispatches
        # vs two sequential LeafBoundaryLayer calls.
        # Results: (gbh_v, gbv_v, gbc_v) each shape (2, nlevmlcan)
        gbh_v, gbv_v, gbc_v = _gb_layers_both(
            mlcanopy_inst.dpai_profile[p, 1:],
            mlcanopy_inst.wind_profile[p, 1:],
            mlcanopy_inst.tair_profile[p, 1:],
            tleaf_both,
            visc,
            dh,
            dv_dh,
            dc_dh,
            dl,
            rhomol_p,
        )

        # Write back: [0]=sun, [1]=shade
        gbh = gbh.at[p, 1:, isun].set(gbh_v[0]).at[p, 1:, isha].set(gbh_v[1])
        gbv = gbv.at[p, 1:, isun].set(gbv_v[0]).at[p, 1:, isha].set(gbv_v[1])
        gbc = gbc.at[p, 1:, isun].set(gbc_v[0]).at[p, 1:, isha].set(gbc_v[1])

    return mlcanopy_inst._replace(
        gbh_leaf=gbh,
        gbv_leaf=gbv,
        gbc_leaf=gbc,
    )
