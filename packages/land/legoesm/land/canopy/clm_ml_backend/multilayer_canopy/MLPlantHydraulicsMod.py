"""
JAX translation of MLPlantHydraulicsMod Fortran module.

Calculate plant hydraulics for the multilayer canopy model.
Three public routines:

- :func:`PlantResistance`: whole-plant leaf-specific conductance.
- :func:`SoilResistance`: soil hydraulic resistance and fractional
  water uptake per soil layer.
- :func:`LeafWaterPotential`: leaf water potential via analytical
  ODE integration.

Original Fortran module: MLPlantHydraulicsMod
Fortran lines 1-200

Differentiability notes
-----------------------
* All ``float()`` / ``int()`` wrappers removed.
* ``np.asarray()`` calls removed — JAX arrays used directly.
* ``np.`` operations replaced by ``jnp.``; ``math.log``, ``math.sqrt``
  replaced by ``jnp.log``, ``jnp.sqrt``.
* Inner layer loops replaced by ``jax.vmap`` (canopy layers) or
  vectorised JAX operations (soil layers with static ``nlevsoi`` size
  and ``j <= nbedrock`` masking).
* ``if float(jax_value) > 0:`` → ``jnp.where``; ``max(x, 0)`` →
  ``jnp.maximum(x, 0.0)``.
* The soil-layer loop bound ``nbedrock[c]`` (a JAX integer) is handled
  by iterating over the full ``nlevsoi`` range and masking inactive
  layers — no data-dependent loop bounds.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import denh2o, grav
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import rpi as pi  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevsoi  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmh2o  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import dtime_ml  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLpftconMod import MLpftcon  # noqa: F401

# ---------------------------------------------------------------------------
# PlantResistance — per-layer kernel + vmap
# ---------------------------------------------------------------------------


def _lsc_layer(dpai_ic, rsoil_p, gplant_pft):
    """Leaf-specific conductance for one canopy layer (differentiable)."""
    rplant = 1.0 / gplant_pft
    lsc_val = 1.0 / (rsoil_p + rplant)
    return jnp.where(dpai_ic > 0.0, lsc_val, 0.0)


_lsc_layers = jax.vmap(_lsc_layer, in_axes=(0, None, None))


def PlantResistance(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate whole-plant leaf-specific conductance (soil-to-leaf).

    Mirrors Fortran subroutine ``PlantResistance`` (lines 23-75).

    Reference: Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A21)-(A22).

    For each layer ``ic`` with ``dpai > 0``:

    .. code-block:: none

        rplant = 1 / gplant_SPA(pft)        [MPa.s.m2/mmol H2O]
        lsc(p,ic) = 1 / (rsoil(p) + rplant) [mmol H2O/m2/s/MPa]

    Layers with ``dpai == 0`` receive ``lsc = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        mlcanopy_inst: Canopy container; ``lsc_profile`` is updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    gplant_SPA = MLpftcon.gplant_SPA
    lsc = mlcanopy_inst.lsc_profile

    for fp in range(num_filter):  # Fortran: do fp = 1, num_filter
        p = filter_patch[fp]
        pft = patch.itype[p]  # JAX int — dynamic index

        lsc_v = _lsc_layers(
            mlcanopy_inst.dpai_profile[p, 1:],
            mlcanopy_inst.rsoil_soil[p],
            gplant_SPA[pft],
        )
        lsc = lsc.at[p, 1:].set(lsc_v)

    return mlcanopy_inst._replace(lsc_profile=lsc)


# ---------------------------------------------------------------------------
# SoilResistance — vectorised over nlevsoi soil layers
# ---------------------------------------------------------------------------


def SoilResistance(
    num_filter: int,
    filter_patch: Sequence[int],
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate soil hydraulic resistance and fractional water uptake
    from each soil layer.

    Mirrors Fortran subroutine ``SoilResistance`` (lines 77-165).

    Reference: Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A23)-(A28).

    Per-layer calculations (Fortran lines 118-146):

    .. code-block:: none

        hk [mmol/m/s/MPa] = hk_l [mm/s] * (1e-3/head) * denh2o/mmh2o * 1000
        smp_mpa [MPa]      = smp_l [mm] * 1e-3 * head
        root_biomass_density [g/m3] = root_biomass * rootfr / dz    (≥ 1e-10)
        root_length_density  [m/m3] = root_biomass_density / (root_density * pi*r^2)
        root_dist [m]               = sqrt(1 / (root_length_density * pi))
        soilr1 = log(root_dist/r) / (2*pi * rld * dz * hk)         (A23)
        soilr2 = root_resist / (root_biomass_density * dz)          (A24)
        soilr  = soilr1 + soilr2
        rsoil  += 1/soilr    (sum conductances across layers)
        evap[j] = max((smp_mpa - minlwp_SPA) / soilr, 0)           (A26)
        evap[j] = 0 if frozen

    After the layer loop (Fortran lines 149-168):

    .. code-block:: none

        rsoil(p) = lai(p) / rsoil(p)           (A25, resistance form)
        psis(p)  = sum(smp_mpa*evap) / totevap  (A27-A28)
        soil_et_loss(p,j) = evap(j) / totevap

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        soilstate_inst: Soil state container (read-only).
        waterstatebulk_inst: Bulk water state container (read-only).
        mlcanopy_inst: Canopy container; ``psis_soil``,
            ``rsoil_soil``, and ``soil_et_loss_soil`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    minlwp_SPA = -2.0  # Fortran local parameter (line 95)
    head = denh2o * grav * 1.0e-6  # MPa/m  (Python float)

    root_radius_SPA = MLpftcon.root_radius_SPA
    root_density_SPA = MLpftcon.root_density_SPA
    root_resist_SPA = MLpftcon.root_resist_SPA

    dz_col = col.dz
    nbedrock = col.nbedrock
    smp_l = soilstate_inst.smp_l_col
    hk_l = soilstate_inst.hk_l_col
    rootfr = soilstate_inst.rootfr_patch
    h2osoi_ice = waterstatebulk_inst.h2osoi_ice_col

    lai = mlcanopy_inst.lai_canopy
    root_biomass = mlcanopy_inst.root_biomass_canopy
    psis = mlcanopy_inst.psis_soil
    rsoil = mlcanopy_inst.rsoil_soil
    soil_et_loss = mlcanopy_inst.soil_et_loss_soil

    # Static soil-layer index array (1-based, shape (nlevsoi,))
    j_arr = jnp.arange(1, nlevsoi + 1)

    for fp in range(num_filter):  # Fortran: do fp = 1, num_filter
        p = filter_patch[fp]
        c = patch.column[p]  # JAX int — dynamic index
        pft = patch.itype[p]

        # Active-layer mask: j <= nbedrock[c] — Fortran: nlayers = nbedrock(c)
        active_j = j_arr <= nbedrock[c]  # shape (nlevsoi,), JAX bool

        # Root cross-sectional area — Fortran line 111
        rr = root_radius_SPA[pft]
        root_cross_sec_area = pi * rr * rr

        # Vectorised per-soil-layer computation (j=1..nlevsoi)
        # Hydraulic conductivity and matric potential — Fortran lines 116-118
        hk_v = hk_l[c, 1 : nlevsoi + 1] * (1.0e-3 / head) * denh2o / mmh2o * 1000.0
        smp_mpa_v = smp_l[c, 1 : nlevsoi + 1] * 1.0e-3 * head

        # Root biomass density — Fortran lines 120-122
        dz_v = dz_col[c, 1 : nlevsoi + 1]
        rbd_v = jnp.maximum(root_biomass[p] * rootfr[p, 1 : nlevsoi + 1] / dz_v, 1.0e-10)

        # Root length density and mean inter-root distance — Fortran lines 125-128
        rld_v = rbd_v / (root_density_SPA[pft] * root_cross_sec_area)
        # Guard rld_v > 0 for safe 1/rld_v and sqrt(1/rld_v) gradients
        rld_v = jnp.maximum(rld_v, 1.0e-30)
        root_dist_v = jnp.sqrt(1.0 / (rld_v * pi))

        # Soil-to-root (A23) and root-to-stem (A24) resistance
        # hk_v can be 0 for frozen layers; guard prevents NaN in jax.grad
        hk_v_safe = jnp.maximum(hk_v, 1.0e-30)
        soilr1_v = jnp.log(root_dist_v / rr) / (2.0 * pi * rld_v * dz_v * hk_v_safe)
        soilr2_v = root_resist_SPA[pft] / (rbd_v * dz_v)
        soilr_v = soilr1_v + soilr2_v  # total belowground resistance
        # Guard soilr_v > 0 for safe division gradients inside jnp.where
        soilr_v_safe = jnp.maximum(soilr_v, 1.0e-30)

        # Maximum transpiration per layer (A26) — Fortran lines 145-148
        evap_v = jnp.maximum((smp_mpa_v - minlwp_SPA) / soilr_v_safe, 0.0)
        # Zero out frozen layers and below-bedrock layers
        frozen_v = h2osoi_ice[c, 1 : nlevsoi + 1] > 0.0
        evap_v = jnp.where(active_j & ~frozen_v, evap_v, 0.0)

        # Total belowground resistance (A25) — Fortran line 151
        rsoil_sum = jnp.sum(jnp.where(active_j, 1.0 / soilr_v_safe, 0.0))
        rsoil_sum_safe = jnp.maximum(rsoil_sum, 1.0e-30)
        rsoil = rsoil.at[p].set(lai[p] / rsoil_sum_safe)

        # Weighted soil water potential and fractional uptake — Fortran 153-168
        totevap = jnp.sum(evap_v)
        # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
        totevap_safe = jnp.maximum(totevap, 1.0e-30)
        psis_p = jnp.where(totevap > 0.0, jnp.sum(smp_mpa_v * evap_v) / totevap_safe, minlwp_SPA)
        psis = psis.at[p].set(psis_p)

        # Fractional water uptake per layer
        nlayers_f = nbedrock[c].astype(jnp.float32)
        # Guard nlayers_f > 0 for safe 1/nlayers_f gradient inside jnp.where
        nlayers_f_safe = jnp.maximum(nlayers_f, 1.0)
        et_loss_uniform = jnp.where(active_j, 1.0 / nlayers_f_safe, 0.0)
        soil_et_loss_v = jnp.where(totevap > 0.0, evap_v / totevap_safe, et_loss_uniform)
        soil_et_loss = soil_et_loss.at[p, 1 : nlevsoi + 1].set(soil_et_loss_v)

    return mlcanopy_inst._replace(
        psis_soil=psis,
        rsoil_soil=rsoil,
        soil_et_loss_soil=soil_et_loss,
    )


# ---------------------------------------------------------------------------
# LeafWaterPotential — per-layer kernel + vmap
# ---------------------------------------------------------------------------


def _lwp_layer(dpai_ic, zs_ic, lsc_ic, trleaf_ic, lwp_bef_ic, psis_p, head, capac_p, dtime):
    """Leaf water potential ODE solution for one canopy layer (differentiable)."""
    has_pai = dpai_ic > 0.0
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    lsc_safe = jnp.maximum(lsc_ic, 1.0e-30)  # avoid /0 on empty layers
    a = psis_p - head * zs_ic - 1000.0 * trleaf_ic / lsc_safe
    b = capac_p / lsc_safe
    y0 = lwp_bef_ic
    dy = (a - y0) * (1.0 - jnp.exp(-dtime / b))
    return jnp.where(has_pai, y0 + dy, 0.0)


_lwp_layers = jax.vmap(
    _lwp_layer,
    in_axes=(0, 0, 0, 0, 0, None, None, None, None),
)


@partial(jax.jit, static_argnums=(0, 1, 2))
def LeafWaterPotential(
    num_filter: int,
    filter_patch: Sequence[int],
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf water potential by analytically integrating the
    first-order ODE over one multilayer canopy timestep.

    Mirrors Fortran subroutine ``LeafWaterPotential`` (lines 167-200).

    Reference: Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A19)-(A20).

    The ODE ``dy/dt = (a - y) / b`` has the analytical solution over
    a full timestep ``dtime``:

    .. code-block:: none

        a  = psis(p) - head*zs(p,ic) - 1000*trleaf(p,ic,il)/lsc(p,ic)
        b  = capac_SPA(pft) / lsc(p,ic)
        dy = (a - y0) * (1 - exp(-dtime/b))
        lwp(p,ic,il) = y0 + dy

    where ``y0 = lwp_bef(p,ic,il)`` is the leaf water potential from
    the previous timestep.  Layers with ``dpai == 0`` receive
    ``lwp = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; ``lwp_leaf`` is updated for
            leaf type ``il``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    head = denh2o * grav * 1.0e-6  # MPa/m (Python float)
    dtime = dtime_ml  # Python float constant

    capac_SPA = MLpftcon.capac_SPA
    lwp = mlcanopy_inst.lwp_leaf

    for fp in range(num_filter):  # Fortran: do fp = 1, num_filter
        p = filter_patch[fp]
        pft = patch.itype[p]  # JAX int — dynamic index

        lwp_v = _lwp_layers(
            mlcanopy_inst.dpai_profile[p, 1:],
            mlcanopy_inst.zs_profile[p, 1:],
            mlcanopy_inst.lsc_profile[p, 1:],
            mlcanopy_inst.trleaf_leaf[p, 1:, il],
            mlcanopy_inst.lwp_bef_leaf[p, 1:, il],
            mlcanopy_inst.psis_soil[p],
            head,
            capac_SPA[pft],
            dtime,
        )
        lwp = lwp.at[p, 1:, il].set(lwp_v)

    return mlcanopy_inst._replace(lwp_leaf=lwp)
