"""JAX-native per-step LAI / albedo updater for ``lax.scan`` time loops.

The host-side builders in :mod:`.builders` are convenient at simulation start
but block a ``lax.scan`` body — each call would host-roundtrip through numpy.
This module gives the same end result through a JAX-pure
``(theta_top, doy) -> (land_params, lai_col)`` closure: all static per-column
inputs (PFT lookups, soil_color, glacier / surfdata-covered masks, bare-
fallback templates) are captured once when the factory is built, and the
returned function only ingests the two things that change per step.

See :func:`make_step_land_params_updater` for the entry point.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.land.param_providers import PFTParamProvider
from legoesm.land.canopy.config import CanopyLandParams
from legoesm.land.soil_albedo import soil_albedo, soil_albedo_broadband
from legoesm.land.global_surface_data import interp_monthly

from legoesm.land.boundary_data._internals import (
    CI_DEFAULT, KN_DEFAULT, ALF_DEFAULT,
    M_C3, M_C4, B0_C3, B0_C4,
    TGC_DEFAULT_C, HC_MIN_M,
    EMISS_VEG, RZ0M_BARE,
    GLACIER_ALB_VIS, GLACIER_ALB_NIR, GLACIER_ALBEDO_DEFAULT,
    pft_lookup_arrays,
)
from legoesm.land.boundary_data.builders import (
    dominant_pft_index, glacier_mask,
)
from legoesm.land.boundary_data.gap_fill import (
    surfdata_covered,
    bare_canopy_params, bare_land_surface_params,
    gap_fill_tree,
)


def make_step_land_params_updater(gsd, surface_scheme):
    """Build a JAX-pure ``(theta_top, doy) -> (land_params, lai_col)`` closure
    for use inside a ``lax.scan`` time loop.

    Splits :func:`~legoesm.land.boundary_data.surface_data_to_land_params` +
    :func:`~legoesm.land.boundary_data.fill_land_param_gaps` into a static
    precompute (PFT lookups, soil_color, glacier / surfdata-covered masks,
    bare-fallback templates — host-side, runs once) and a JAX-pure per-step
    updater that ingests the only inputs that change with time: ``theta_top``
    (top-layer wetness; varies with the soil state) and ``doy`` (day-of-year,
    advances each step).  Everything else is captured once.

    Inside the updater:
      - LAI / canopy-height are picked from the monthly climatology via
        :func:`interp_monthly` at the traced ``doy`` and the precomputed
        dominant-PFT (canopy scheme) or PFT-fraction weights (SEB scheme).
      - Soil-background albedo is recomputed from the precomputed soil colour
        and the traced ``theta_top``.
      - Glacier override (ice albedo + zero LAI) and surfdata-gap fill are
        applied with precomputed masks.

    Returns a function ``(theta_top, doy) -> (land_params, lai_col)``:
    ``land_params`` has the same type as
    :func:`~legoesm.land.boundary_data.surface_data_to_land_params` for the
    given scheme; ``lai_col`` is the per-column LAI driving the albedo blend
    (dominant-PFT LAI for canopy, PFT-weighted column LAI for SEB), a handy
    diagnostic for the scan body.
    """
    from legoesm.land.canopy import CanopyConfig
    is_canopy = isinstance(surface_scheme, CanopyConfig)

    # Inputs that don't change across steps (cast once to JAX).
    lai_monthly = jnp.asarray(gsd.lai_monthly)
    htop_monthly = jnp.asarray(gsd.htop_monthly)
    soil_color = jnp.asarray(np.asarray(gsd.soil_color))
    glacier_col = jnp.asarray(glacier_mask(gsd).astype(np.float64))
    covered_jnp = jnp.asarray(surfdata_covered(gsd))

    if is_canopy:
        dom = dominant_pft_index(gsd)
        ncol = int(dom.shape[0])
        lut = pft_lookup_arrays()
        dom_idx = jnp.asarray(dom.astype(np.int32))
        is_veg_dom = lut["is_veg"][dom]              # (ncol,) np
        is_veg_col = jnp.asarray(is_veg_dom.astype(np.float64))
        hc_default = jnp.asarray(lut["hc"][dom])
        fC4 = jnp.asarray(lut["fc4"][dom])
        Vcmax25_C3 = jnp.asarray(lut["vc3"][dom])
        Vcmax25_C4 = jnp.asarray(lut["vc4"][dom])
        rz0m = jnp.asarray(np.where(is_veg_dom > 0.0, lut["rz0m"][dom], RZ0M_BARE))
        rd = jnp.asarray(np.where(is_veg_dom > 0.0, lut["rd"][dom], 0.0))
        bare_fb = bare_canopy_params(ncol)
        full = lambda v: jnp.full(ncol, v)

        def _update_canopy(theta_top: jnp.ndarray, doy: jnp.ndarray):
            """Return ``(CanopyLandParams, lai_col)``: ``lai_col`` is the
            per-column dominant-PFT LAI used in the params, useful as a
            diagnostic in the scan body."""
            lai_m = interp_monthly(lai_monthly, doy)                # (ncol, npft)
            htop_m = interp_monthly(htop_monthly, doy)
            LAI = jnp.take_along_axis(lai_m, dom_idx[:, None], axis=1)[:, 0]
            hc_surf = jnp.take_along_axis(htop_m, dom_idx[:, None], axis=1)[:, 0]
            LAI = jnp.where(is_veg_col > 0.0,
                            jnp.where(jnp.isfinite(LAI), LAI, 0.0), 0.0)
            hc = jnp.where(jnp.isfinite(hc_surf) & (hc_surf > 0.0),
                           hc_surf, hc_default)
            hc = jnp.maximum(hc, HC_MIN_M)
            av, an = soil_albedo(soil_color, theta_top)
            ice = glacier_col > 0.0
            is_veg = jnp.where(ice, 0.0, is_veg_col)
            LAI = jnp.where(ice, 0.0, LAI)
            av = jnp.where(ice, GLACIER_ALB_VIS, av)
            an = jnp.where(ice, GLACIER_ALB_NIR, an)
            lp = CanopyLandParams(
                LAI=LAI, hc=hc, fC4=fC4, FNonVeg=1.0 - is_veg,
                CI=full(CI_DEFAULT), kn=full(KN_DEFAULT),
                Vcmax25_C3_leaf=Vcmax25_C3, Vcmax25_C4_leaf=Vcmax25_C4,
                m_C3=full(M_C3), m_C4=full(M_C4),
                b0_C3=full(B0_C3), b0_C4=full(B0_C4),
                alf=full(ALF_DEFAULT), TgC=full(TGC_DEFAULT_C),
                ALB_VIS=av, ALB_NIR=an,
                emissivity=full(EMISS_VEG), rz0m=rz0m, rd=rd,
            )
            lp_filled = gap_fill_tree(lp, bare_fb, covered_jnp)
            return lp_filled, lp_filled.LAI

        return _update_canopy

    # ---- SEB / slab path: PFT-weighted LAI -> albedo_veg blend ----
    fracs = np.nan_to_num(np.asarray(jnp.mean(gsd.pft_frac, axis=0)), nan=0.0)
    zero_cover = fracs.sum(axis=-1) < 1e-6
    fracs[zero_cover, :] = 0.0
    fracs[zero_cover, 0] = 1.0
    pft_provider = PFTParamProvider.from_defaults(jnp.asarray(fracs))
    base_lp = pft_provider()                                 # static base LandSurfaceParams
    fracs_jnp = jnp.asarray(fracs)
    ncol = int(fracs.shape[0])
    bare_fb = bare_land_surface_params(ncol)
    # Dominant-PFT C4 flag (0/1) — the in-scan twin of surface_data_param_provider's
    # fc4_col (see the rationale there: a big leaf is a single pathway). Folded onto
    # base_lp so the SEB LandSurfaceParams carries fC4 (structure matches bare_fb for
    # the gap_fill_tree tree-map) and the big-leaf FvCB path runs the right pathway.
    fc4_col = jnp.asarray(np.asarray(pft_lookup_arrays()["fc4"])[dominant_pft_index(gsd)])
    base_lp = base_lp._replace(fC4=fc4_col)

    def _update_seb(theta_top: jnp.ndarray, doy: jnp.ndarray):
        """Return ``(LandSurfaceParams, lai_col)``: ``lai_col`` is the
        PFT-weighted column LAI driving the albedo blend, useful as a
        diagnostic in the scan body."""
        lai_m = interp_monthly(lai_monthly, doy)              # (ncol, npft)
        lai_col = jnp.sum(jnp.where(jnp.isfinite(lai_m), lai_m, 0.0) * fracs_jnp,
                          axis=-1)
        soil_bg = soil_albedo_broadband(soil_color, theta_top)
        f_veg = 1.0 - jnp.exp(-0.5 * lai_col)
        alb = base_lp.albedo_veg * f_veg + soil_bg * (1.0 - f_veg)
        alb = jnp.where(glacier_col > 0.0, GLACIER_ALBEDO_DEFAULT, alb)
        lp = base_lp._replace(albedo_veg=alb)
        return gap_fill_tree(lp, bare_fb, covered_jnp), lai_col

    return _update_seb
