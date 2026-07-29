"""
JAX translation of MLCanopyNitrogenProfileMod Fortran module.

Canopy profile of nitrogen and photosynthetic capacity.

Original Fortran module: MLCanopyNitrogenProfileMod
Fortran lines 1-175

Differentiability notes
-----------------------
* ``np.asarray()`` calls removed — JAX arrays are used directly.
* ``np.`` operations replaced by ``jnp.`` so gradient flow is maintained
  through the nitrogen scaling computation.
* ``round(float(c3psn[pft])) == 1`` replaced by a JAX-compatible
  ``jnp.where`` branch selection so ``pft`` can be a traced integer.
* ``acclim_type`` and ``kn_val`` are static Python scalars — their
  ``if/elif`` branches are evaluated once at trace time.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import tfrz  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.pftconMod import pftcon  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import (  # noqa: F401
    jmax25_to_vcmax25_acclim,
    jmax25_to_vcmax25_noacclim,
    kp25_to_vcmax25_c4,
    rd25_to_vcmax25_c3,
    rd25_to_vcmax25_c4,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import acclim_type, kn_val, leaf_optics_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nlevmlcan  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLpftconMod import MLpftcon  # noqa: F401


@partial(jax.jit, static_argnums=(0, 1))
def CanopyNitrogenProfile(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    vcmaxpft_jax=None,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Calculate the canopy profile of nitrogen and photosynthetic capacity.

    Mirrors Fortran subroutine ``CanopyNitrogenProfile`` (lines 22-175).

    References:
    Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A1)-(A2).
    Bonan et al. (2021) *Agric. For. Met.*, 306, 108435,
    supplemental eqs. (6)-(12).

    **Canopy-top parameters** (Fortran lines 80-100):

    - ``vcmax25top`` from PFT lookup table.
    - ``jmax25_to_vcmax25`` selected by ``acclim_type``; for
      ``acclim_type == 1`` the ratio is temperature-dependent:
      ``2.59 - 0.035 * clamp(tacclim - tfrz, 11, 35)``.
    - C3: ``jmax25top = ratio * vcmax25top``,
      ``rd25top = rd25_to_vcmax25_c3 * vcmax25top``, ``kp25top = 0``.
    - C4: ``jmax25top = 0``,
      ``rd25top = rd25_to_vcmax25_c4 * vcmax25top``,
      ``kp25top = kp25_to_vcmax25_c4 * vcmax25top``.

    **Leaf nitrogen decay coefficient** (Fortran lines 102-110):

    .. code-block:: none

        kn_val < 0: kn = exp(0.00963 * vcmax25top - 2.43)
        kn_val > 0: kn = kn_val

    **Per-layer nitrogen scaling** (Fortran lines 113-148):
    Iterates from top to bottom (``ic = ncan, ..., 1``).

    .. code-block:: none

        fn     = exp(-kn*pai_above) * (1 - exp(-kn*dpai)) / kn

    For ``leaf_optics_type == 0`` (Bonan et al. 2021, eqs. 8-12):

    .. code-block:: none

        fn_sun = clump / (kn + kb*clump) * exp(-kn*pai_above)
                 * tbi * (1 - exp(-(kn + kb*clump)*dpai))
        fn_sha = fn - fn_sun
        nscale_sun = fn_sun / (fracsun * dpai)
        nscale_sha = fn_sha / ((1-fracsun) * dpai)

    For ``leaf_optics_type == 1``:

    .. code-block:: none

        nscale_sun = nscale_sha = fn / dpai

    All four leaf parameters (``vcmax25``, ``jmax25``, ``rd25``,
    ``kp25``) are scaled proportionally to ``nscale``.

    **Layer weighted mean** (Fortran lines 150-154):

    .. code-block:: none

        vcmax25_profile(ic) = vcmax25_leaf(ic,isun)*fracsun
                            + vcmax25_leaf(ic,isha)*(1-fracsun)

    **Conservation check** (Fortran lines 156-160):
    Numerical sum of ``vcmax25_profile * dpai`` over all layers must
    equal the analytical integral
    ``vcmax25top * (1 - exp(-kn*(lai+sai))) / kn``
    to within 1e-6 umol/m2/s.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``vcmax25_leaf``,
            ``jmax25_leaf``, ``rd25_leaf``, ``kp25_leaf``,
            ``vcmax25_profile``, ``jmax25_profile``, ``rd25_profile``,
            and ``kp25_profile`` are updated.
        vcmaxpft_jax: Optional JAX array override for ``MLpftcon.vcmaxpft``
            (shape matching the PFT dimension).  When provided, this is used
            instead of the module-global ``MLpftcon.vcmaxpft`` so that JAX
            autodiff can trace gradients through vcmaxpft.  When ``None``
            (default), the module-global value is used (standard non-diff path).

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    c3psn = pftcon.c3psn
    # Use explicit JAX arg when provided (differentiable path); else module global.
    _vcmaxpft = MLpftcon.vcmaxpft if vcmaxpft_jax is None else vcmaxpft_jax

    vcmax25_leaf = mlcanopy_inst.vcmax25_leaf
    jmax25_leaf = mlcanopy_inst.jmax25_leaf
    rd25_leaf = mlcanopy_inst.rd25_leaf
    kp25_leaf = mlcanopy_inst.kp25_leaf
    vcmax25_profile = mlcanopy_inst.vcmax25_profile
    jmax25_profile = mlcanopy_inst.jmax25_profile
    rd25_profile = mlcanopy_inst.rd25_profile
    kp25_profile = mlcanopy_inst.kp25_profile

    for fp in range(1 if grid is not None else num_filter):  # Fortran: do fp = 1, num_filter
        p = grid.p if grid is not None else filter_patch[fp]  # grid.p may be TRACED (S3)
        pft = patch.itype[p]  # JAX int — dynamic index (traced-p safe; ncan masked below)

        # is_c3: JAX boolean scalar; used with jnp.where for differentiable branching
        is_c3 = jnp.round(c3psn[pft]) == 1  # Fortran implicit round of 0/1 flag

        # Canopy-top photosynthetic parameters — Fortran lines 80-96
        vcmax25top = _vcmaxpft[pft]  # JAX scalar via dynamic gather

        if acclim_type == 0:  # static branch — evaluated at trace time
            j2v = jmax25_to_vcmax25_noacclim
        elif acclim_type == 1:
            ta_c = jnp.clip(mlcanopy_inst.tacclim_forcing[p] - tfrz, 11.0, 35.0)
            j2v = 2.59 - 0.035 * ta_c  # Fortran line 87
        else:
            endrun(msg=" ERROR: CanopyNitrogenProfile: acclim_type not valid")
            j2v = 0.0  # Unreachable

        # C3/C4 parameter selection — Fortran lines 90-97
        # jnp.where avoids Python if on traced is_c3
        jmax25top = jnp.where(is_c3, j2v * vcmax25top, 0.0)
        rd25top = jnp.where(is_c3, rd25_to_vcmax25_c3 * vcmax25top, rd25_to_vcmax25_c4 * vcmax25top)
        kp25top = jnp.where(is_c3, 0.0, kp25_to_vcmax25_c4 * vcmax25top)

        # Nitrogen decay coefficient — Fortran lines 102-108
        # kn_val is a static Python float; the if/elif branches are trace-time only
        if kn_val < 0.0:
            kn = jnp.exp(0.00963 * vcmax25top - 2.43)
        elif kn_val > 0.0:
            kn = jnp.asarray(kn_val)
        else:
            endrun(msg="ERROR: CanopyNitrogenProfile: incorrect Kn")
            kn = jnp.zeros(())  # Unreachable

        clump = MLpftcon.clump_fac[pft]  # JAX scalar
        lai_p = mlcanopy_inst.lai_canopy[p]
        sai_p = mlcanopy_inst.sai_canopy[p]

        # Per-layer arrays — use JAX arrays directly (no np.asarray sync)
        _dpai = mlcanopy_inst.dpai_profile[p]  # shape (nlevmlcan+1,)
        _kb = mlcanopy_inst.kb_profile[p]
        _tbi = mlcanopy_inst.tbi_profile[p]
        _fracsun = mlcanopy_inst.fracsun_profile[p]

        # ncan is the number of active canopy layers for patch p.
        # We work over the full layer axis (1..nlevmlcan) and rely on
        # dpai==0 masking for inactive layers.  The iteration order
        # top→bottom maps to reversed indices ic=ncan..1.
        ncan_p = mlcanopy_inst.ncan_canopy[p]  # JAX int scalar

        # Build top-to-bottom index order: [ncan, ncan-1, ..., 1]
        # Use static nlevmlcan as the upper bound; mask by dpai later.
        ics = jnp.arange(nlevmlcan, 0, -1)  # [nlevmlcan, ..., 1]
        active = ics <= ncan_p  # mask for actual canopy layers

        dpai_v = _dpai[ics]
        kb_v = _kb[ics]
        tbi_v = _tbi[ics]
        fs_v = _fracsun[ics]

        has_pai = active & (dpai_v > 0.0)
        dpai_safe = jnp.maximum(
            dpai_v, 1.0e-30
        )  # avoid /0; jnp.maximum avoids select_divide_fusion XLA bug

        # pai_above[j] = cumulative dpai of layers above layer ics[j]
        # = sum(dpai_v[0:j])  →  shifted cumsum — Fortran line 148
        # Mask inactive layers to 0 BEFORE cumsum: inactive slots have
        # dpai = spval (1e36), which would poison exp(-kn*pai_above) → 0
        # for all active layers downstream.
        dpai_cumsum = jnp.where(active, dpai_v, 0.0)
        pai_above_v = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dpai_cumsum[:-1])])

        # Integrated nitrogen factor — Fortran line 130
        exp_kn_above = jnp.exp(-kn * pai_above_v)
        fn = exp_kn_above * (1.0 - jnp.exp(-kn * dpai_v)) / kn
        fn = jnp.where(has_pai, fn, 0.0)

        if leaf_optics_type == 0:  # static branch — Fortran lines 132-139
            denom = kn + kb_v * clump
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            denom_safe = jnp.maximum(denom, 1.0e-30)
            fn_sun = (
                clump / denom_safe * exp_kn_above * tbi_v * (1.0 - jnp.exp(-denom_safe * dpai_v))
            )
            fn_sun = jnp.where(has_pai, fn_sun, 0.0)
            fn_sha = fn - fn_sun
            fs_safe = jnp.maximum(fs_v, 1.0e-30)
            fsha_safe = jnp.maximum(1.0 - fs_v, 1.0e-30)
            nscale_sun = jnp.where(has_pai, fn_sun / (fs_safe * dpai_safe), 0.0)
            nscale_sha = jnp.where(has_pai, fn_sha / (fsha_safe * dpai_safe), 0.0)

        elif leaf_optics_type == 1:  # static branch — Fortran lines 140-141
            nscale = jnp.where(has_pai, fn / dpai_safe, 0.0)
            nscale_sun = nscale
            nscale_sha = nscale

        else:
            nscale_sun = jnp.zeros(nlevmlcan)
            nscale_sha = jnp.zeros(nlevmlcan)

        # Scale leaf parameters — Fortran lines 143-146
        vcmax25_sun_v = jnp.where(has_pai, vcmax25top * nscale_sun, 0.0)
        vcmax25_sha_v = jnp.where(has_pai, vcmax25top * nscale_sha, 0.0)
        jmax25_sun_v = jnp.where(has_pai, jmax25top * nscale_sun, 0.0)
        jmax25_sha_v = jnp.where(has_pai, jmax25top * nscale_sha, 0.0)
        rd25_sun_v = jnp.where(has_pai, rd25top * nscale_sun, 0.0)
        rd25_sha_v = jnp.where(has_pai, rd25top * nscale_sha, 0.0)
        kp25_sun_v = jnp.where(has_pai, kp25top * nscale_sun, 0.0)
        kp25_sha_v = jnp.where(has_pai, kp25top * nscale_sha, 0.0)

        # Layer weighted mean — Fortran lines 150-154
        # Direct form vcmax25top * fn / dpai is algebraically equivalent to
        # vcmax25_sun * fracsun + vcmax25_sha * (1-fracsun) but avoids
        # precision loss when fracsun == 0 or fracsun == 1 exactly, which
        # would cause the safe-denominator substitutions to drop a term and
        # break the conservation check below.
        vcmax25_profile_v = jnp.where(has_pai, vcmax25top * fn / dpai_safe, 0.0)
        jmax25_profile_v = jnp.where(has_pai, jmax25top * fn / dpai_safe, 0.0)
        rd25_profile_v = jnp.where(has_pai, rd25top * fn / dpai_safe, 0.0)
        kp25_profile_v = jnp.where(has_pai, kp25top * fn / dpai_safe, 0.0)

        # Reorder from top→bottom (ics order) back to ic=1..nlevmlcan order
        # ics = [nlevmlcan, ..., 1] reversed → [ic=1, ..., ic=nlevmlcan]
        vcmax25_sun_ord = vcmax25_sun_v[::-1]
        vcmax25_sha_ord = vcmax25_sha_v[::-1]
        jmax25_sun_ord = jmax25_sun_v[::-1]
        jmax25_sha_ord = jmax25_sha_v[::-1]
        rd25_sun_ord = rd25_sun_v[::-1]
        rd25_sha_ord = rd25_sha_v[::-1]
        kp25_sun_ord = kp25_sun_v[::-1]
        kp25_sha_ord = kp25_sha_v[::-1]
        vcmax25_prof_ord = vcmax25_profile_v[::-1]
        jmax25_prof_ord = jmax25_profile_v[::-1]
        rd25_prof_ord = rd25_profile_v[::-1]
        kp25_prof_ord = kp25_profile_v[::-1]

        # Bulk JAX write-back — 12 scatter operations
        vcmax25_leaf = vcmax25_leaf.at[p, 1:, isun].set(vcmax25_sun_ord)
        vcmax25_leaf = vcmax25_leaf.at[p, 1:, isha].set(vcmax25_sha_ord)
        jmax25_leaf = jmax25_leaf.at[p, 1:, isun].set(jmax25_sun_ord)
        jmax25_leaf = jmax25_leaf.at[p, 1:, isha].set(jmax25_sha_ord)
        rd25_leaf = rd25_leaf.at[p, 1:, isun].set(rd25_sun_ord)
        rd25_leaf = rd25_leaf.at[p, 1:, isha].set(rd25_sha_ord)
        kp25_leaf = kp25_leaf.at[p, 1:, isun].set(kp25_sun_ord)
        kp25_leaf = kp25_leaf.at[p, 1:, isha].set(kp25_sha_ord)
        vcmax25_profile = vcmax25_profile.at[p, 1:].set(vcmax25_prof_ord)
        jmax25_profile = jmax25_profile.at[p, 1:].set(jmax25_prof_ord)
        rd25_profile = rd25_profile.at[p, 1:].set(rd25_prof_ord)
        kp25_profile = kp25_profile.at[p, 1:].set(kp25_prof_ord)

        # Conservation check — JIT-compatible via debug.callback
        numerical = jnp.sum(vcmax25_prof_ord * _dpai[1:])
        analytical = vcmax25top * (1.0 - jnp.exp(-kn * (lai_p + sai_p))) / kn
        jax.debug.callback(
            lambda n, a: (
                endrun(msg="ERROR: CanopyNitrogenProfile: canopy integration error")
                if abs(float(n) - float(a)) > 1.0e-6
                else None
            ),
            numerical,
            analytical,
        )

    return mlcanopy_inst._replace(
        vcmax25_leaf=vcmax25_leaf,
        jmax25_leaf=jmax25_leaf,
        rd25_leaf=rd25_leaf,
        kp25_leaf=kp25_leaf,
        vcmax25_profile=vcmax25_profile,
        jmax25_profile=jmax25_profile,
        rd25_profile=rd25_profile,
        kp25_profile=kp25_profile,
    )
