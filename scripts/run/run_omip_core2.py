#!/usr/bin/env python
"""legoESM ocean forced by CORE-II Normal-Year Forcing on the eORCA1 tripole
grid -- the faithful counterpart to the NEMO ORCA1 reference (see
``OMIP_faithful.md``).

Both models use the SAME CORE-II forcing: NEMO reads the raw COREv2 files; here
``load_core2_nyf`` reads ``nyf.zarr`` built from those same files by
``scripts/data/build_core2_nyf_zarr.py`` (native 6-hourly winds, so the nonlinear
bulk fluxes match). The eORCA1 grid + land mask + bathymetry come from NEMO's
own ``eORCA1.2_mesh_mask.nc`` (tmaskutil / e3t_0), so the geometry matches too.

Loop: ``load_core2_nyf`` -> per step pick the 6-hourly record -> apply CORE-II
bulk fluxes via ``apply_omip2_surface_fluxes`` (ocean-reaction sign, tripole
rotation) -> ``model.step``. Annual snapshots + scalar diagnostics are written
for scoring against the NEMO climatology.

NOTE the applicator is host-side NumPy, so each step round-trips the state
device<->host. ``--smoke`` reports steps/s so the real run length can be sized;
if throughput is too low, apply forcing every N steps (forcing is 6-hourly).

Usage (GPU sbatch):
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/run_omip_core2.py --smoke
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/run_omip_core2.py --years 5
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax

# Precision: the OMIP run is float64 (x64) by DEFAULT — the scientific
# reference.  ``--fp32`` (single-precision, GPU-memory mode: eORCA025 ¼° fits
# the lat-band SPMD step on a 48 GB GPU in fp32 where f64 OOMs) must leave JAX
# x64 OFF so device arrays default to float32; the matching all-fp32
# ``PrecisionPolicy`` is set from ``args`` in ``main`` (after argparse).  x64
# has to be decided BEFORE any JAX op runs, so the flag is sniffed from argv
# here (a cheap pre-parse; argparse still owns the real flag + validation).
_FP32 = "--fp32" in sys.argv[1:]
if not _FP32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.eos import VALID_FREEZE_SCHEMES

_SEC_PER_DAY = 86400.0
_SEC_PER_6H = 21600.0
_YEAR_S = 365.0 * _SEC_PER_DAY
_MESH = "data/grids/eORCA1.2_mesh_mask.nc"
# NEMO ldf_eiv (nn_aei_ijk_t=21) kappa_GM defaults, defined ONCE and shared by
# `build_tripole`'s signature and the `--gm-aei0` / `--gm-kappa-min` argparse
# defaults so the two can never drift.
# NEMO ldftra.F90:290-293 -- for the LAPLACIAN operator (ORCA1:
# ln_traldf_lap=.true.) the prefactor is zUfac = 1/2 * rn_Ud, so
#     aei0 = 1/2 * rn_Ue * rn_Le = 0.5 * 0.018 * 100e3 = 900 m^2/s,
# which the code's own printout states ("aht0 = 1/2 rn_Ud*rn_Ld",
# ldftra.F90:331) and NEMO's emitted aeiu_2d confirms: max EXACTLY 900
# (measured 2026-08-12, RUN_TRD2 rec 1).  The previous 1800 dropped the 1/2
# and made the "NEMO-faithful" GM cap twice NEMO's.
_GM_AEI0_DEFAULT = 900.0
# Equatorial-taper floor.  NOT a NEMO number: raw NEMO is capped-only and lets
# kappa_GM -> 0 at the equator (the recipe card keeps that, gm_kappa_min=0.0).
# 200 is a production stability knob for the 1-degree global run; the numeric
# value coincides with VisbeckConfig.kappa_min, which the OMIP tripole does NOT
# run (its recipe ships Visbeck disabled -- verified in the run manifest,
# 2026-08-12).  The divergence from the recipe default is deliberate and
# asserted at both ends (tests/unit/test_run_omip_core2_gm_treguier.py).
_GM_KAPPA_MIN_DEFAULT = 200.0
# Isoneutral-slope operators and GM bolus forms selectable on the tripole.
# NEMO ORCA1 runs the STANDARD rotated laplacian (namelist_cfg:
# ln_traldf_lap=.true., ln_traldf_iso=.true., ln_traldf_triad=.false.) with the
# Method of Stabilizing Correction (ln_traldf_msc=.true.), and adds the eddy-
# induced transport to the ADVECTING velocity (LDF/ldftra.F90 `ldf_eiv_trp`,
# PUBLIC "called by traadv.F90") so the bolus rides the monotone FCT limiter.
# "nemo_iso_lap" is that OPERATOR.  It does NOT by itself turn on the
# stabilizing correction: `akz` is gated on the SEPARATE GMRediConfig field
# `msc_stabilize` (default False), so matching ORCA1 needs it too --
# hence --gm-msc-stabilize.  "through_fct" is the bolus routing.  The OMIP default is "centered" slopes
# with an unlimited centred bolus flux -- a documented departure, now
# selectable rather than hard-wired.
_GM_SLOPE_SCHEMES = ("triads", "centered", "nemo_iso_lap")
_GM_BOLUS_FORMS = ("centred", "through_fct")


def real_freshwater_restoring_conflict(freshwater_closure, sss_restore,
                                       sss_restore_normalization,
                                       sss_restore_channel=None):
    """Return the refusal message for an incompatible pairing, else None.

    ``--freshwater-closure real_freshwater`` drops the virtual-salt term, so a
    restoring flux DERIVED as a virtual-salt equivalent (legoESM's historical
    ``normalization="s_target"``: a fixed ``z1``, divided by ``S_target``)
    would apply the wrong relaxation strength -- its realized effect under
    volume-only dilution depends on the LIVE salinity and the ACTUAL top-cell
    thickness.

    ``normalization="live_s"`` is NEMO ``sbcssr`` nn_sssr=2 (sbcssr.F90:132-134,
    ``zerp = zsrp*coefice*(sss_m - sss_target)/MAX(sss_m,1e-20)``), a genuine
    WATER flux -- what ORCA1 runs alongside its variable-volume freshwater
    budget -- so that pairing is allowed.

    Split out of ``main`` so the rule is unit-testable: ``main`` applies it
    only after the model is built, which no unit test can cheaply reach.
    """
    if freshwater_closure != "real_freshwater" or not sss_restore:
        return None
    # EXEMPTION, and the only one: `--sss-restore-channel water_flux` with the
    # NEMO conversion.  That combination is exactly the precondition this
    # guard's own message has always named -- the restoring is routed through
    # `fw.restoring` as a real water flux (driving eta / z-star) with its heat
    # term, and the post-step tracer edit is SKIPPED, so there is no
    # virtual-salt-like operation left for `real_freshwater` to contradict.
    # `live_s` is required alongside it by a separate guard in `main`.
    if (sss_restore_channel == "water_flux"
            and sss_restore_normalization == "live_s"):
        return None
    # NOTE (2026-08-12, codex 9383572 RED): `--sss-restore-normalization live_s`
    # does NOT lift this.  A previous revision let it through on the grounds
    # that live_s makes the restoring "a genuine water flux".  It does not, in
    # THIS code path: `apply_sss_restoring_step*` consume `dS_dt_top` and edit
    # the tracer directly -- the flux never reaches FreshwaterForcing, eta, or
    # the z-star dilution.  And `dS_dt_top` is re-derived as
    # `-freshwater_flux * S_safe / (rho_0*z1)`, in which `S_safe` CANCELS the
    # division that produced the flux, so the normalization changes the applied
    # tendency ONLY where the +/-4 mm/day cap binds.  Restoring is therefore
    # still a virtual-salt-like operation whatever the denominator, and pairing
    # it with a closure that assumes no virtual-salt term is still wrong.
    # Lifting this guard requires ROUTING restoring as a water flux (and
    # removing the tracer-side edit), not renaming its denominator.
    return (
        "--freshwater-closure real_freshwater is not compatible with "
        "--sss-restore: the restoring is applied as a DIRECT SALINITY "
        "TENDENCY (apply_sss_restoring_step consumes dS_dt_top; the "
        "freshwater flux is a diagnostic here and never enters the eta / "
        "z-star volume channel), which is a virtual-salt-like operation the "
        "real_freshwater closure assumes is absent. "
        "--sss-restore-normalization live_s does NOT lift this: S_safe "
        "cancels in the dS_dt_top re-derivation except where the flux cap "
        "binds. Drop --sss-restore, or keep the virtual_salt_flux closure, "
        "until restoring is routed as a real water flux (#1484).")


def _tripole_treguier_gm_redi(gm_aei0, gm_kappa_min):
    """GM/Redi block for ``--gm-treguier`` on the eORCA1 tripole.

    ONE VARIABLE: this is the tripole NEMO-match recipe's OWN GM/Redi block
    with `gm_treguier` flipped on, so a `--gm-treguier` arm differs from the
    control run in the ``treguier`` field and nothing else.

    The base used to be ``run_omip._DEFAULT_BATHY_GM_REDI``, which is the
    LAT-LON bathymetry default (``kappa_GM=kappa_Redi=800``, Visbeck ON,
    ``slope_scheme="triads"``) and NOT the tripole recipe's block
    (``kappa_GM=kappa_Redi=600``, Visbeck OFF, ``slope_scheme="centered"``).
    Selecting it changed FOUR fields at once, so no ``--gm-treguier`` arm could
    be attributed to the Treguier coefficient.  Verified against two run
    manifests (2026-08-12): the arm ran kappa 800/800 + triad slopes against a
    control at 600/600 + centered slopes.

    The recipe validates the Treguier block while building it
    (``validate_treguier_cfg``), so a bad ``aei0``/``kappa_min`` fails here
    rather than inside the first GM tendency.
    """
    from legoesm.ocean.fidelity.nemo_match_recipe import (
        NEMOMatchTripoleRecipeConfig,
        nemo_match_tripole_model_config,
    )
    recipe_cfg = NEMOMatchTripoleRecipeConfig(
        gm_treguier=True,
        gm_aei0=float(gm_aei0),
        gm_kappa_min=float(gm_kappa_min),
    )
    return nemo_match_tripole_model_config(recipe_cfg).gm_redi


# NEMO eORCA geometry + WOA IC loaders now live in the ocean package so the
# coupled-ESM driver (Phase-2 tripole coupler) can build the SAME validated
# cold-start without importing from scripts/ (a layering violation).
from legoesm.ocean.init_tripole import (  # noqa: E402
    read_mesh_mask_bathy,
    compute_woa_3d,
    squeeze_nemo_field_2d as _squeeze2d,
)


def _south_pad_rows(n_lat: int, n_gpus: int) -> int:
    """Number of LAND rows to append at the SOUTH so ``n_lat`` is a multiple of
    ``n_gpus`` (the lat-band SPMD step needs one uniform band per device).

    eORCA025 ``n_lat=1207`` is odd: for ``n_gpus=2`` this returns 1 (-> 1208).
    Returns 0 when already divisible (or ``n_gpus <= 1``).
    """
    if n_gpus <= 1:
        return 0
    rem = n_lat % n_gpus
    return 0 if rem == 0 else (n_gpus - rem)


def _pad_mask_bathy_south(land_mask: np.ndarray, H_bathy: np.ndarray,
                          n_pad: int):
    """Prepend ``n_pad`` LAND rows (mask=0, bathy=0) to the SOUTH of the cell
    ``(n_lat, n_lon)`` land-mask + bathymetry arrays.

    Pairs with :func:`legoesm.grids.tripole.pad_tripole_grid_south` (which pads
    the GRID geometry the same way + keeps the north fold): the padded mask/bathy
    + grid are fed to the SAME ``_init_rest_state`` / WOA-fill path, so the state
    is built on the padded grid with the added rows masked LAND (inert dynamics).
    The wet rows are preserved bit-exact, shifted ``+n_pad`` in the lat index.
    """
    if n_pad <= 0:
        return land_mask, H_bathy
    lm = np.asarray(land_mask)
    hb = np.asarray(H_bathy)
    n_lon = lm.shape[1]
    zeros_lm = np.zeros((n_pad, n_lon), dtype=lm.dtype)
    zeros_hb = np.zeros((n_pad, n_lon), dtype=hb.dtype)
    return (np.concatenate([zeros_lm, lm], axis=0),
            np.concatenate([zeros_hb, hb], axis=0))


def _ew_overlap_fill(a: np.ndarray) -> np.ndarray:
    """Fill the ORCA 2-point cyclic-overlap halo columns of a static field.

    ``col[0] <- col[nx-2]``, ``col[nx-1] <- col[1]`` along axis 1 (longitude).
    For ``(n_lat, n_lon)`` masks/bathy and ``(n_lat, n_lon, nlev)`` IC fields.
    NumPy counterpart of the model's per-step ``_apply_ew_cyclic_overlap`` so
    the static geometry/IC start consistent with the reconnected seam.
    """
    a = np.array(a, copy=True)
    nx = a.shape[1]
    a[:, 0] = a[:, nx - 2]
    a[:, nx - 1] = a[:, 1]
    return a


def smooth_woa_ts(state, grid, passes):
    """Horizontal Laplacian smoothing of the WOA T,S initial condition over
    OCEAN cells, per level.  Removes the spurious grid-scale / over-sharp
    fronts that interpolating + flood-filling WOA onto the tripole introduces
    -- those imply unphysically large geostrophic velocities (>>2 m/s) and the
    cold-start cannot carry them.  The dynamics ARE stable on a smooth
    stratification (uniform-strat rest test), so smoothing the IC toward that
    regime is the natural conditioning."""
    from legoesm.ocean.bathymetry import laplacian_smooth_2d
    mask = np.asarray(state.land_mask.data) > 0.5
    # Cube horizontal fields are (6, n, n) -> laplacian_smooth_2d needs the
    # cube-topology stencil (cross-face neighbours); 2-D grids are (n_lat, n_lon).
    is_cubed = (mask.ndim == 3)
    T = np.array(state.T.data, dtype=np.float64)
    S = np.array(state.S.data, dtype=np.float64)
    nlev = T.shape[-1]
    for arr in (T, S):
        for k in range(nlev):
            orig_k = arr[..., k].copy()
            cur = arr[..., k]
            for _ in range(int(passes)):
                sm = np.asarray(laplacian_smooth_2d(cur, 1, is_cubed=is_cubed))
                cur = np.where(mask, sm, orig_k)
            arr[..., k] = cur
    print(f"[setup] WOA T,S horizontal smoothing: {passes} Laplacian passes/level "
          f"(removes spurious grid-scale fronts)")
    return state._replace(
        T=state.T.replace(data=jnp.asarray(T)),
        S=state.S.replace(data=jnp.asarray(S)),
    )


def apply_balanced_init(state, grid, z_coord, config,
                        taper_lat_deg=8.0, ref_depth_m=1500.0,
                        max_speed=2.5, with_ssh=True):
    """Initialise the cold-start in geostrophic / thermal-wind balance.

    The OMIP WOA cold-start blows up because it starts from REST (u=0) with a
    flat free surface (eta=0): the full baroclinic pressure-gradient force from
    WOA's density fronts is then UNBALANCED, and the violent geostrophic
    adjustment goes nonlinear (conclusively diagnosed -- the partial-cell PGF
    itself is NEMO-class, ~1e-6 m/s2 on a uniform-stratification rest test).

    This puts the flow in balance at t=0 so there is no adjustment shock:

    1. Baroclinic pressure anomaly ``p'`` from WOA T,S (surface-referenced),
       via the SAME ``iterate_eos_and_pressure_anomaly`` the dycore uses.
    2. LEVEL-OF-NO-MOTION reference: subtract the deepest-active ``p'_bottom``
       so the total pressure is flat at the seafloor (deep flow -> 0,
       surface-intensified ~1 m/s -- physical).  ``p_ref = p' - p'_bottom``.
    3. Geostrophic velocity from ``p_ref`` at cell centres,
       ``u_g = -(1/rho_0 f) dp_ref/dy``, ``v_g = +(1/rho_0 f) dp_ref/dx``,
       with the Coriolis singularity regularised near the equator
       ``1/f -> f/(f^2 + f_eps^2)`` (``f_eps = 2 Omega sin(taper_lat)`` -> the
       geostrophic velocity tapers smoothly to zero within ~|lat|<taper_lat).
       Mapped to the C-grid faces with ``cell_to_cgrid_winds`` (fold-aware).
    4. (with_ssh) Balanced free surface ``eta = -p'_bottom/(rho_0 g)`` (area-
       demeaned), so the dycore's total PGF ``-(1/rho_0) grad(p' + rho_0 g eta)
       = -(1/rho_0) grad(p_ref)`` exactly balances the geostrophic velocity.

    Pure IC change -- no dynamics-core modification.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        gradient_x_cgrid, gradient_y_cgrid, cell_to_cgrid_winds,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    from legoesm import constants

    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data
    rho_0 = float(config.rho_0)
    g_val = float(config.g)
    is_pc = isinstance(z_coord, OceanPartialCellCoordinate)
    eos_fn = make_eos_fn(config.eos, getattr(config, "eos_linear", None))
    h_actual = z_coord.h_partial if is_pc else None

    _, _, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask,
        lambda fld: neumann_fill_cgrid(fld, mask, grid=grid),
        eos_fn, z_coord.dz_ref, rho_0, g_val,
        n_iter=2, hi_precision_pressure=True, h_actual=h_actual,
    )                                                    # (n_lat, n_lon, nlev)

    # Reference level for the level-of-no-motion: a FIXED depth (~ref_depth_m)
    # common to all sufficiently-deep columns -- NOT the per-column seafloor
    # (whose depth varies with bathymetry, so a seafloor reference makes p_ref
    # and eta scale with DEPTH instead of the dynamic steric signal -> O(100 m)
    # spurious SSH).  Shallow columns reference their own bottom level.
    z_full = np.asarray(z_coord.z_full_ref)                 # (nlev,), negative
    k_ref = int(np.argmin(np.abs(z_full + ref_depth_m)))    # level nearest ref_depth
    if is_pc:
        bl = jnp.clip(z_coord.bottom_level, 0, z_coord.n_levels - 1)
        k_use = jnp.minimum(bl, k_ref)                      # (n_lat, n_lon)
    else:
        k_use = jnp.full(p_prime.shape[:-1], k_ref, dtype=jnp.int32)
    p_at_ref = jnp.take_along_axis(p_prime, k_use[..., None], axis=-1)  # (...,1)
    p_ref = p_prime - p_at_ref                              # 0 at the reference

    p_ref_filled = neumann_fill_cgrid(p_ref, mask, grid=grid)
    gx_u = gradient_x_cgrid(p_ref_filled, grid)            # (n_lat, n_lon+1, nlev)
    gy_v = gradient_y_cgrid(p_ref_filled, grid)            # (n_lat+1, n_lon, nlev)
    gx_T = 0.5 * (gx_u[:, :-1] + gx_u[:, 1:])              # (n_lat, n_lon, nlev)
    gy_T = 0.5 * (gy_v[:-1] + gy_v[1:])

    # grid-agnostic Coriolis: tripole LatLonCGridGeometry -> f_T, plain
    # LatLonGrid -> f; both expose the grid_coriolis @property -> (n_lat, n_lon).
    f_T = grid.grid_coriolis                                # (n_lat, n_lon)
    f_eps = 2.0 * constants.Omega * float(np.sin(np.deg2rad(taper_lat_deg)))
    inv_f = (f_T / (f_T ** 2 + f_eps ** 2))[..., None]     # -> 0 at the equator

    u_g = -(1.0 / rho_0) * inv_f * gy_T
    v_g = +(1.0 / rho_0) * inv_f * gx_T
    # Safety clip: geostrophy is invalid in the (tapered) equatorial band and
    # at any residual sharp IC front (e.g. flood-fill seams); bound the speed
    # to a physical maximum so those cells start bounded rather than at
    # tens of m/s.  Mid-latitude balanced flow is well below this.
    u_g = jnp.clip(u_g, -max_speed, max_speed)
    v_g = jnp.clip(v_g, -max_speed, max_speed)
    m3 = mask[..., None]
    if is_pc:
        m3 = m3 * z_coord.is_active.astype(m3.dtype)
    u_g = u_g * m3
    v_g = v_g * m3

    u_face, v_face = cell_to_cgrid_winds(u_g, v_g, grid)
    u_face = u_face * state.u_mask.data[..., None]
    u_face = u_face.at[:, -1].set(u_face[:, 0])            # periodic wrap column
    v_face = v_face * state.v_mask.data[..., None]

    repl = dict(
        u=state.u.replace(data=u_face),
        v=state.v.replace(data=v_face),
    )
    umax = float(jnp.nanmax(jnp.abs(u_face)))
    vmax = float(jnp.nanmax(jnp.abs(v_face)))
    if with_ssh:
        eta = -p_at_ref[..., 0] / (rho_0 * g_val)          # (n_lat, n_lon)
        area = grid.area * mask   # grid-agnostic (both grids expose .area)
        eta_mean = jnp.sum(eta * area) / jnp.maximum(jnp.sum(area), 1.0)
        eta = jnp.clip(eta - eta_mean, -5.0, 5.0) * mask   # physical SSH bound
        repl["eta"] = state.eta.replace(data=eta)
        print(f"[setup] balanced init: ref level k={k_ref} (~{ref_depth_m:.0f} m), "
              f"geostrophic u,v (taper {taper_lat_deg} deg, clip {max_speed} m/s) "
              f"max|u|={umax:.3f} max|v|={vmax:.3f} m/s + SSH "
              f"eta[{float(jnp.min(eta)):.2f},{float(jnp.max(eta)):.2f}] m")
    else:
        print(f"[setup] balanced init: ref level k={k_ref} (~{ref_depth_m:.0f} m), "
              f"geostrophic u,v (taper {taper_lat_deg} deg, clip {max_speed} m/s) "
              f"max|u|={umax:.3f} max|v|={vmax:.3f} m/s (NO balanced SSH)")
    return state._replace(**repl)


def make_partial_cell(z_coord, H_bathy, land_mask, thin_threshold=0.3,
                      smoothing_passes=0, min_levels=1):
    """Convert a z* reference coord + bathymetry to an ``OceanPartialCellCoordinate``,
    snapping ``H_bathy`` DOWN to the interface above whenever the bottom partial cell
    would be thinner than ``thin_threshold * dz_ref`` (MOM6/MITgcm thin-cell fix).

    ``smoothing_passes`` > 0 applies that many Laplacian smoothing passes to the
    OCEAN ``H_bathy`` field first (land held fixed), reducing the bathymetric slope
    (r-factor ``|H_i-H_j|/(H_i+H_j)``).  NEMO/ROMS smooth their bathymetry for exactly
    this reason: the spurious partial-cell pressure-gradient seed that blows up the
    WOA cold-start (per-term diag: KE_PGF at the S-Atlantic / Indonesian continental
    SLOPES, the steepest cells) scales with the slope, so gentler topography shrinks
    it.  The max r-factor is reported before/after so the geometry cost is explicit.

    Mirrors the documented-stable ``run_omip.py`` partial-cell setup
    (``run_omip_single``, ~line 3181). The OMIP-faithful runner previously passed the
    plain ``OceanZStarCoordinate`` from ``_create_setup`` straight to the model. With
    that coord, ``J = (eta + H_bathy)/H_max`` uniformly stretches all levels to the
    local depth (terrain-following / sigma-like), AND the Adcroft/SMC03 partial-cell
    PGF correction is gated OFF (``isinstance(z_coord, OceanPartialCellCoordinate)``
    is False in ``ocean_pe_latlon_cgrid``). Over steep equatorial topography (f≈0)
    that drives the spurious bottom meridional-PGF seed that blows up the WOA cold
    start (per-term diag job 8106208: KE_PGF_v ~8.6e-3 m/s2 @ Indonesian seas, bottom
    level). The z-level partial-cell coord (this function) puts levels at FIXED
    reference depths, activates the PGF correction, and applies the thin-cell snap
    that prior work found necessary at the equator.

    Returns ``(z_coord_partial, H_snapped_np, land_mask_np)``.
    """
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    H_np = np.asarray(H_bathy, dtype=np.float64)
    lm0 = np.asarray(land_mask, dtype=np.float64)

    if smoothing_passes and smoothing_passes > 0:
        from legoesm.ocean.bathymetry import laplacian_smooth_2d, compute_max_r_factor
        ocean = lm0 > 0.5
        r_before = float(compute_max_r_factor(H_np, lm0))
        H_s = H_np.copy()
        # Smooth ocean cells only; hold land fixed and re-impose it each
        # pass so the smoother never bleeds land depths into the ocean.
        for _ in range(int(smoothing_passes)):
            H_sm = np.asarray(laplacian_smooth_2d(H_s, 1, is_cubed=False))
            H_s = np.where(ocean, H_sm, H_np)
        H_np = np.where(ocean, H_s, H_np)
        r_after = float(compute_max_r_factor(H_np, lm0))
        print(f"[setup] bathymetry smoothing: {smoothing_passes} Laplacian "
              f"passes, max r-factor {r_before:.3f} -> {r_after:.3f}")

    abs_z_half = np.abs(np.asarray(z_coord.z_half_ref))   # (nlev+1,) positive depths
    dz_ref_np = np.asarray(z_coord.dz_ref)                # (nlev,) positive
    H_snapped = H_np.copy()
    n_snapped = 0
    for k in range(z_coord.n_levels):
        top, bot = abs_z_half[k], abs_z_half[k + 1]
        in_layer = (H_np > top) & (H_np <= bot)
        too_thin = in_layer & ((H_np - top) < thin_threshold * dz_ref_np[k])
        H_snapped = np.where(too_thin, top, H_snapped)
        n_snapped += int(np.sum(too_thin))
    lm = np.asarray(land_mask, dtype=np.float64)
    new_land = (H_snapped <= 0.0) & (lm > 0.5)
    # rn_hmin-style conditioning: mask ocean columns with fewer than
    # ``min_levels`` active reference levels. Single-active-level coastal cells
    # (H <= dz_ref[0]) are a 1/h instability seed at 1/4 deg, where resolved
    # Arctic/coastal shelves collapse to one thin partial layer and a tiny
    # smc03 PGF residual is amplified into a blowup (eORCA025 cold-start seed at
    # 67N/107.5W). ``abs_z_half[k]`` is the top of level k, so n_active =
    # #levels whose top lies above the local seafloor.
    n_masked_shallow = 0
    if min_levels and int(min_levels) > 1:
        n_active = (abs_z_half[None, None, :z_coord.n_levels]
                    < H_snapped[..., None]).sum(axis=2)
        too_shallow = (n_active < int(min_levels)) & (lm > 0.5) & (H_snapped > 0.0)
        n_masked_shallow = int(np.sum(too_shallow))
        H_snapped = np.where(too_shallow, 0.0, H_snapped)
        new_land = new_land | too_shallow
    n_new_land = int(np.sum(new_land))
    lm_out = np.where(new_land, 0.0, lm)
    msg = (f"[setup] partial-cell snap (cutoff {thin_threshold*100:.0f}%): "
           f"{n_snapped} cells snapped, {n_new_land} -> land")
    if min_levels and int(min_levels) > 1:
        msg += f" ({n_masked_shallow} masked for <{int(min_levels)} active levels)"
    print(msg)
    # Storage dtype follows the active precision policy (f64 by default;
    # float32 under --fp32, where an explicit dtype=float64 would warn-and-
    # truncate). resolve_dtype clamps f64->f32 when x64 is off, so this is the
    # one device array the partial-cell coord builds in the run-wide dtype.
    from legoesm.core.precision import resolve_dtype as _resolve_dtype
    _coord_dtype = _resolve_dtype(None, "storage")
    zc = create_partial_cell_coordinate(
        z_coord, jnp.asarray(H_snapped, dtype=_coord_dtype),
    )
    return zc, H_snapped, lm_out


# Single source of truth for --tke-mxl-choice, consumed by BOTH argparse
# (choices=) and orca1_zdftke_config's guard.  Codex 2026-08-07 #4: a finite
# -2..9 sweep in the test is a nearby-mutation check, NOT set equality -- it
# still passes if the builder accepts 10 while argparse does not.  Sharing one
# constant makes divergence impossible by construction instead of policed.
TKE_MXL_CHOICES = (2, 3, 4)   # 2=Veros BL, 3=NEMO nn_mxl=3, 4=NEMO nn_mxl=2


def orca1_zdftke_config(iwm_enabled: bool = False, surface_bc: str | None = None,
                        mxl_choice: int | None = None,
                        n2_mode: str | None = None,
                        n2_eos_form: str | None = None,
                        prognostic: bool | None = None,
                        kappa_convention: str | None = None,
                        shear_production: str | None = None,
                        lc: bool | None = None,
                        etau_mode: str | None = None):
    """NEMO ORCA1 ``&namzdf_tke`` mapped onto :class:`TKEConfig`, value by value.

    Source of truth: ``cfgs/ORCA1/EXP00/RUN_REF/namelist_cfg`` overrides on top
    of ``namelist_ref`` defaults (NEMO 5.0.1).  Structural conventions
    (diagnostic Mode-B TKE, insitu N², gaspar amplitude, Veros mixing-length
    construction) follow the validated DINO NEMO-faithful recipe
    (``ocean/experiments/dino.py::_dino_vertical_mixing_config``); ONLY the
    namelist values change for ORCA1.

    Mapping (namelist -> field):
      rn_ediff = 0.1   -> c_k          (avm = c·mxl·sqrt(e); see amplitude note)
      rn_ediss = 0.7   -> c_eps        (Kolmogorov dissipation coefficient)
      rn_emin  = 1e-6  -> tke_background   [m²/s²]
      rn_emin0 = 1e-4  -> tke_surface_min  [m²/s²]
      nn_pdl   = 1     -> prandtl_mode="richardson" with prandtl_ri_coeff =
                          1/ri_cri, ri_cri = 2/(2 + rn_ediss/rn_ediff) = 2/9
                          (zdftke.F90:772,399: Pr = clamp(Ri/ri_cri, 1, 10) ==
                          clamp(4.5·Ri, 1, 10) — exactly _prandtl_number's form)
      nn_mxl   = 2     -> tke_mxl_choice=4 (NOT 2: the choice numbering is
                          Veros-derived, so choice 2 is Veros Bougeault-
                          Lacarrere and choice 4 is NEMO nn_mxl=2)
      ln_lc    = T     -> lc=True
      rn_lc    = 0.25  -> lc_coeff       (namelist_cfg override of 0.15)
      nn_etau  = 1     -> etau_mode="below_ml"
      rn_efr   = 0.08  -> etau_frac      (namelist_cfg override of 0.05)
      nn_htau  = 1     -> etau_htau_mode="latitude" (0.5–30 m, 45·|sinφ|)
      rn_ebb   = 67.83 -> tke.py module constant _NEMO_TKE_EBB (etau e_sfc;
                          identical value, no config field)
      &namzdf rn_avm0/rn_avt0 (1.2e-4/1.2e-5 backgrounds) -> kappaM_min /
                          kappaH_min.  NEMO composes avm = max(closure, avmb)
                          and avt = max(pdl·avt, avtb) (zdftke.F90:715,723);
                          compute_K_from_tke applies EXACTLY that max via the
                          kappaM_min/kappaH_min floors, so the namzdf
                          backgrounds live INSIDE the closure and build_tripole
                          forces the model-level additive A_v/K_v to molecular
                          (residual +1.4e-6/+1e-10 add — negligible, flagged).
                          With --iwm, NEMO's zdfiwm_init forces avmb/avtb to
                          molecular (rnu=1.4e-6, 1e-10): ``iwm_enabled=True``
                          switches the floors to those values so the wave
                          field is the interior background, as in NEMO.
      nn_avb   = 0     -> bg_diff_scale=0.0 (no Bryan-Lewis depth profile;
                          abyssal mixing comes from zdfiwm as in NEMO)

    nn_eice  = 3     -> eice=3 (under-ice attenuation of lc/etau: the
                          kernels' (1-ice_frac) factor fed max(0,1-4*fi) via
                          surface_forcing.ice_concentration — CLOSED
                          2026-07-18; was a flagged no-ice_frac gap).

    NO TKEConfig counterpart (fidelity gaps, flagged not stubbed):
      ln_mxl0=T / rn_mxl0=0.04  surface mixing length = F(wind stress);
      nn_mxlice=2               under-ice mixing-length scaling;
      rn_bshear=1e-20           background-shear floor (legoESM uses 1e-12);
      surface TKE BC            NEMO Dirichlet e_sfc=rn_ebb·|τ|/ρ0 vs legoESM
                                flux (|τ|/ρ0)^{3/2} (Veros/Wallace form);
      prognostic en carry       NEMO integrates en prognostically; this runs
                                the DINO-validated quasi-steady diagnostic
                                Mode-B (3 backward-Euler iterations).
    """
    from legoesm import constants as _const
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    rn_ediff = 0.1        # namelist_ref &namzdf_tke default (ORCA1 keeps it)
    rn_ediss = 0.7        # namelist_ref &namzdf_tke default (ORCA1 keeps it)
    # zdftke.F90:772 — ri_cri deduced from rn_ediff/rn_ediss; Pr slope = 1/ri_cri.
    pr_ri_slope = (2.0 + rn_ediss / rn_ediff) / 2.0     # = 4.5
    if iwm_enabled:
        # zdfiwm_init: avmb = rnu = 1.4e-6 m²/s, avtb = 1e-10 m²/s — the
        # wave field IS the interior background (matches the model-level
        # A_v/K_v override in the build_tripole iwm block).
        avmb, avtb = _const.nu_ocean_molecular, 1.0e-10
    else:
        avmb, avtb = 1.2e-4, 1.2e-5     # &namzdf rn_avm0 / rn_avt0
    _cfg = TKEConfig(
        c_k=rn_ediff,
        c_eps=rn_ediss,
        tke_background=1.0e-6,          # rn_emin
        tke_surface_min=1.0e-4,         # rn_emin0
        # nn_mxl: choice=3 IS the NEMO nn_mxl construction (lup/ldown |dl/dz|<=e3t
        # sweeps) WITH the ln_mxl0 wind-stress surface anchor that NEMO ORCA1 runs
        # (ln_mxl0=T).  choice=2 (Veros Bougeault-Lacarrere, no ln_mxl0) was the
        # flagged fidelity gap; =3 closes it.  See the A/B campaign (2026-07-24).
        # ORCA1's namelist_cfg sets `nn_mxl = 2` (EXP00/namelist_cfg:444), which
        # is choice 4, NOT 3.  The card carried 3 while its own comment above
        # said 4 was the ORCA1 value -- flagged by codex 9405117 as a direct
        # card inconsistency that invalidates one-step parity.  Choices 3 and 4
        # share l_k = min(lup,ldn) and differ only in the dissipation length
        # (3: sqrt(lup*ldn), 4: min(lup,ldn)), so 4 dissipates more.
        tke_mxl_choice=4,
        # ---- what survived codex 9405307, which REFUTED three of five ------
        # The cumulative offline test (job 9405195) put all five at 0.854
        # Antarctic / 0.930 Arctic and I wired them as a new baseline. Codex
        # refuted three of them and the headline. Reverted, with the reasons
        # kept here so nobody re-wires them:
        #
        # veros_dz_slots=True -- REVERTED. The interior solver does map
        #   dz_cell<->e3t and dz_half<->e3w, but on --partial-cell the tripole
        #   caller supplies dz_ref*J while the real bottom thickness is
        #   h_partial*J, so the partial metric is silently discarded. It is
        #   not an e3t/e3w mapping at partial bottoms. It also COLLIDES with
        #   nemo_z0: both overload dz_surface, one as a Veros surface
        #   half-volume and one as a top-cell face-gradient distance, and on
        #   the centred grid those differ by 2x.
        #
        # n2_mode="nemo_bn2" -- REVERTED, and this one was simply wrong.
        #   ORCA1 selects TEOS-10 (namelist_cfg:307). legoESM's `nemo_bn2`
        #   calls nemo_seos_alpha_beta and its own docstring says "exact bn2
        #   (S-EOS)". That is a DIFFERENT N2, not ORCA1's rn2. Closing this
        #   properly needs a TEOS-10 rab/bn2, not a config flip.
        #
        # tke_surface_bc_level="nemo_z0" -- REVERTED. The placement is right
        #   but the metric is not: dz_surface = -z_full_ref[0]*J is the top
        #   cell's MIDPOINT, i.e. half dz_ref[0], while NEMO's jk=2 lower
        #   coefficient needs the full top-cell e3t(1). That doubles the
        #   virtual-surface coupling.
        #
        # ALSO REVERTED BY THE SAME REVIEW: --grid mpas shares this card and
        # mpas_integration.py:686 fail-loud rejects BOTH n2_mode != "insitu"
        # and veros_dz_slots=True, so the five-field card could not run on
        # MPAS at all. The user's standing ask is three-grid agreement; a card
        # that only one grid can execute is not a baseline.
        # ---- the two the review confirmed --------------------------------
        # Job 9405195, our K_M vs NEMO's own avm after one matched 3600 s step
        # from NEMO's state and en, Antarctic 491743 interfaces: the card as it
        # stood was ~2.6x NEMO, the amplitude fix took it to 1.846, alpha_tke=1
        # OVERSHOT to 0.634, and these five together land it at 0.854
        # (Arctic 0.930) -- inside the pre-registered 0.85-1.20 band.  They
        # converge rather than trade: alpha alone undershoots and these lift it
        # back.  This block is therefore a deliberate NEW BASELINE, not five
        # independent one-variable edits, and must not be compared term-by-term
        # against arms that predate it.
        #
        # veros_dz_slots: NEMO's zzd = -0.5*rn_Dt*mean(avm)/(e3t*e3w) is
        #   flux-form -- gradient across the intervening T cell (/e3t),
        #   divergence into the W control volume (/e3w).  legoESM maps
        #   dz_cell<->e3t and dz_half<->e3w ONLY in this branch; the legacy
        #   default uses dz_half for BOTH, so the coefficient was NEMO's while
        #   the stencil was not (codex 9405117 #2).  Alone: 0.627.
        # n2_mode: the clipped in-situ N2 carries a compressibility bias
        #   (+4.3e-5 measured in a prior session -- enough to stop the EVD
        #   trigger firing in the Arctic at all).  It sets BOTH the buoyancy
        #   length and the buoyancy sink.  k_profiles threads the gdept /
        #   interior-gdepw ladders when this mode is selected.  Alone: 0.808.
        # dissipation_discretization: NEMO linearises the Kolmogoroff sink as a
        #   NEWTON split -- zfact2 = 1.5*rn_Dt*rn_ediss on the diagonal
        #   (zdftke.F90:241,414) plus zfact3 = 0.5*rn_ediss added back
        #   explicitly (:242,:419) -- the correct Jacobian for eps ~ e^{3/2}.
        #   Plain backward Euler puts 1.0 on the diagonal with no add-back.
        #   Codex 9405117 predicted the direction before it was run ("retains
        #   more TKE than plain BE"); alone it lifts 0.634 -> 0.739.
        dissipation_discretization="nemo_1p5_split",
        # tke_surface_bc_level: NEMO holds en(1) at the z=0 W-point and SOLVES
        #   the tridiagonal from jk=2 (zdftke.F90:264,403-410).
        #   "interior_pinned" pins the Dirichlet value AT the first interior
        #   interface instead -- one w-level too deep.  Worth only ~4% on its
        #   own (measured, and it REFUTED a root-cause hypothesis of mine), but
        #   it is what NEMO does.  Requires the surface Dirichlet value, which
        #   this card sets, and dz_surface, which k_profiles:811-815 threads
        #   for exactly this option.
        # STILL NOT NEMO, and not closable here: tke_shear_production stays
        # "squared_centered".  NEMO's zdf_sh2 is face-native with a now x
        # before velocity product and DOUBLES production adjacent to coasts
        # via (2 - umask*umask) (zdfsh2.F90).  "nemo_face_native" implements
        # exactly that and needs the raw C-grid face state plus per-level
        # wumask/wvmask/coast masks, which the offline column probe cannot
        # supply -- so it is untested and deliberately NOT enabled here.  It is
        # the last known card gap and needs a tripole run to evaluate.
        # NEMO nn_bc_surf=1: en(1)=max(rn_emin0, rn_ebb·|τ|/ρ0) Dirichlet surface
        # TKE.  The Veros flux (|τ|/ρ0)^{3/2} default was a flagged gap; the
        # Dirichlet form matches NEMO and cuts the summer-hemisphere warm SST.
        surface_bc="nemo_dirichlet",
        # NEMO integrates `en` PROGNOSTICALLY (one backward-Euler step/model-step
        # carrying OceanState.tke).  The quasi-steady diagnostic Mode-B was the
        # flagged gap; prognostic accumulates the tropical mixing energy and
        # RECOVERS the tropical SST (+0.55→+0.12; SST rmse 1.44→0.90, MLD 74→38 in
        # the controlled tripole d30 A/B) — the faithful tropical fix.  Revert any
        # lever via --tke-surface-bc/--tke-mxl-choice/--tke-prognostic.
        prognostic=True,
        prandtl_mode="richardson",      # nn_pdl=1
        prandtl_ri_coeff=pr_ri_slope,   # 1/ri_cri = 4.5 (NOT the Veros 6.6)
        # TKE VERTICAL-DIFFUSION COEFFICIENT.  NEMO diffuses `en` with the
        # PLAIN viscosity: zzd_up = -0.5*rn_Dt*(avm(k+1)+avm(k))/(e3t*e3w)
        # (zdftke.F90:407-409) is a face coefficient of mean(avm), i.e. x1.
        # TKEConfig's default is the Veros/Gaspar 30.0 -- as its own
        # __param_spec__ reference says outright, "NEMO avm x1 (zdftke);
        # Veros/Gaspar 30" -- so the card was diffusing TKE THIRTY TIMES too
        # fast against a NEMO oracle.  MEASURED on NEMO's own state and `en`,
        # one 3600 s step, Antarctic, 491743 interfaces, scored against NEMO's
        # avm (EVD-free: nn_evdm=0), job 9405026:
        #     zero step                          K_M/avm 0.861
        #     alpha_tke=30 (was the card)  e 3.20x, K_M/avm 1.846
        #     alpha_tke=1.0 (NEMO)         e 0.95x, K_M/avm 0.634
        # Arctic 1.666 -> 0.673.  Nothing else in the closure came within an
        # order of it: Langmuir, nn_etau, the mixing-length choice, the surface
        # BC placement and the dissipation split are each worth 2-9%.
        # HONEST CAVEAT: 1.0 overshoots the other way -- over that hour NEMO's
        # avm rises 5% and ours then FALLS 22%.  1.0 is still the faithful
        # value (it is what NEMO's discretisation computes); a residual of the
        # same order as the 0.861 zero-step offset remains, and the dissipation
        # split moves the wrong way for it.
        # No CLI flag: alpha_tke carries a __param_spec__ (tier 2, bounds
        # 1-90), so `--params vertical_mixing.tke.alpha_tke=30` reverts it.
        alpha_tke=1.0,                  # NEMO zdftke: TKE diffused by avm x1
        # STRATIFICATION. Precise statement, after GLM-5.2 pushed back and
        # zdftke.F90:650 settled it: BOTH forms floor N2 inside the length --
        # NEMO evaluates zrn2 = MAX(rn2, rsmall) and legoESM's choice-3/4
        # branch takes sqrt(max(N2, 1e-12)). So this is NOT "signed vs
        # clipped"; the floor is on both sides. What changes is the N2 VALUE.
        # The in-situ density gradient carries a POSITIVE compressibility bias
        # (+4.3e-5 measured in a prior session) that keeps N2 comfortably
        # above the floor even in neutral water, so the buoyancy length stays
        # SHORT. The adiabatic form reaches the floor where the water really
        # is neutral, the length runs to the sweep bound, and that is NEMO's
        # long zmxlm. Same mechanism as described below, correctly named.
        #
        # Carried from the same review, not acted on: the stacked MIN sweeps
        # make the diffusivity non-differentiable at the mixed-layer base, and
        # in neutral layers the limiter IS effectively the convection
        # parameterisation (length set by ramp geometry, not closure physics)
        # -- the documented resolution-sensitivity of convective mixing in
        # ORCA-type runs. ORCA1 sets no Galperin cap (no rn_clim_galp in the
        # namelist), so NEMO lives with this too; matching it is the goal here.
        #
        # ORIGINAL NOTE: the card computed N2 from the in-situ
        # density gradient CLIPPED at zero; NEMO's rn2 is SIGNED. In
        # convectively neutral water the signed form gives N -> 0, the
        # buoyancy length sqrt(2e)/N blows up, and the lup/ldown sweeps set a
        # LONG length -- which is what NEMO's zmxlm does there. Clipping
        # suppresses exactly those, and job 9407791 showed that is the whole
        # remaining mixing-length deficit: seeding the length with a signed N2
        # moved our zero-step length ratio against NEMO's own from 0.897 to
        # 0.997 (Southern Ocean), 0.914 to 0.998 (tropics) and 0.792 to 1.004
        # (Arctic). The largest correction is the Arctic, the most convective
        # band -- the mechanism's own prediction.
        #
        # This is also the counterpart of alpha_tke: at day 30 the sqrt(2) and
        # alpha fixes drove the Southern Ocean mixed layer from +1.5 m to
        # -19.7 m, because removing two large over-mixing errors left the
        # short-length under-mixing error uncompensated. This is that error.
        #
        # n2_eos_form: ORCA1 runs ln_teos10=.true. (namelist_cfg:308), so the
        # alpha/beta come from the Roquet polynomial with the TEOS-10
        # coefficient set. A previous revision set n2_mode alone and codex
        # 9405307 refuted it -- the bn2 helper defaulted to S-EOS, so the
        # TEOS-10 path was unreachable. Both fields are needed.
        n2_mode="nemo_bn2",
        n2_eos_form="teos10",
        # K-from-TKE AMPLITUDE.  NEMO zdftke tke_avn computes
        #     zsqen = SQRT(en) ; zav = rn_ediff*zmxlm*zsqen
        #     p_avm = MAX(zav, avmb)*wmask
        # -- the sqrt carries `en`, NOT `2*en`; the factor 2 lives in the
        # LENGTH, zmxlm = SQRT(2*en/rn2) (zdftke.F90:651).  (NEMO's own header
        # comment at :150/:553 writes the momentum floor as `avtb`; the CODE
        # uses `avmb` and `avtb` is the TRACER floor -- codex 9400815 #1.
        # nn_pdl=1 changes only avt, and NOT as pdlr*avm: the tracer floor
        # avtb is applied AFTER the Prandtl reduction, which
        # compute_K_from_tke mirrors as K_H = max(kappaH_min, K_M_raw/Pr).)
        # legoESM's `gaspar_sqrt2e` DEFAULT applies sqrt(2*e) in the amplitude
        # as well, and the nn_mxl=2/3 branch already builds the length from
        # sqrt(2)*sqrt(e)/N (tke.py:698-702), so the default DOUBLE-COUNTS the
        # sqrt(2) on exactly the path this card selects.  MEASURED on NEMO's
        # own state (Stage A, commit 39ce0701c, Arctic calm columns):
        #     gaspar_sqrt2e  K_H 4.048e-2 = 4.81x NEMO avt 8.410e-3
        #     veros_sqrte    K_H 2.730e-2 = 3.25x
        # SCOPE OF "exact", stated narrowly on purpose (codex 9400815 #2):
        # `veros_sqrte` is c_k*l_k*sqrt(max(0,e)) and is EXACT against NEMO on
        # wet rows of this card's normal trajectory, because positivity="floor"
        # ends every solve at e >= tke_background = 1e-6 = rn_emin, so
        # sqrt(max(0,e)) == sqrt(e) there.  It is NOT globally identical: NEMO
        # zeroes dry rows via *wmask while this card leaves tke_dry_wmask=False
        # and keeps the background there, and the e<=0 branch differs (ours
        # returns 0, floored back up by kappaM_min/kappaH_min; NEMO floors en
        # first).  Neither is reachable on the floored path this card runs.
        # SIGN, also narrowly: at fixed e/l_k/N2/shear the pin divides K by
        # sqrt(2).  It does NOT follow that the mixed layer shallows by any
        # particular amount -- the prognostic solve is coupled (lower K_M cuts
        # shear production and reinforces; lower K_H cuts buoyancy destruction
        # and offsets), so the integrated response is what the day-90 A/B
        # measures, not something the algebra gives.
        # Revert with --tke-kappa-convention, which reaches the tripole and
        # (since 2026-08-22) the MPAS grid as well, so the same A/B can be run
        # on either.
        kappa_convention="veros_sqrte",
        lc=True,                        # ln_lc
        lc_coeff=0.25,                  # rn_lc (namelist_cfg override)
        etau_mode="below_ml",           # nn_etau=1
        etau_frac=0.08,                 # rn_efr (namelist_cfg override)
        etau_htau_mode="latitude",      # nn_htau=1 (namelist_ref default)
        eice=3,                         # nn_eice=3 — under-ice lc/etau attenuation
        kappaM_min=avmb,                # NEMO avm = max(closure, avmb)
        kappaH_min=avtb,                # NEMO avt = max(pdl·avt, avtb)
        bg_diff_scale=0.0,              # nn_avb=0 — no depth-profile background
    )
    # Surface TKE boundary condition (``--tke-surface-bc``).  DEFAULT keeps the
    # TKEConfig default (``veros_flux``, the flagged fidelity gap listed above).
    # ``nemo_dirichlet`` selects the NEMO ``en(1)=max(rn_emin0, rn_ebb·|τ|/ρ0)``
    # Dirichlet condition (rn_ebb=67.83; tke.py ``_surface_tke_dirichlet``) —
    # closing the surface-BC gap so wind energy enters the near-surface TKE at
    # the NEMO rate rather than the weaker Veros flux ``(|τ|/ρ0)^{3/2}`` form.
    if surface_bc is not None:
        if surface_bc not in ("veros_flux", "nemo_dirichlet"):
            raise ValueError(
                f"orca1_zdftke_config surface_bc {surface_bc!r} invalid; "
                "expected 'veros_flux' or 'nemo_dirichlet' (NEMO nn_bc_surf).")
        _cfg = _cfg._replace(surface_bc=surface_bc)
    # Langmuir + surface-TKE penetration overrides (``--tke-lc``/``--tke-etau``).
    # DEFAULT keeps the ORCA1 card (ln_lc=T, nn_etau=1).  The OFF settings
    # exist to build a "fesom-mimic" card: fesom-jax's CVMix TKE has no
    # Langmuir and no etau penetration, and quantifying the FESOM2 skill gap
    # requires running OUR physics with THOSE branches off (2026-08-18).
    if lc is not None:
        _cfg = _cfg._replace(lc=bool(lc))
    if etau_mode is not None:
        if etau_mode not in ("below_ml", "none"):
            raise ValueError(f"orca1_zdftke_config etau_mode {etau_mode!r} "
                             "invalid; expected 'below_ml' or 'none'.")
        _cfg = _cfg._replace(etau_mode=etau_mode)
    # K-from-TKE amplitude (``--tke-kappa-convention``).  DEFAULT keeps the
    # card value (``veros_sqrte`` = NEMO's ``rn_ediff*zmxlm*sqrt(en)``);
    # ``gaspar_sqrt2e`` restores the legacy sqrt(2)-double-counting amplitude
    # so the fix can be A/B'd against every arm that predates it.
    if kappa_convention is not None:
        if kappa_convention not in ("veros_sqrte", "gaspar_sqrt2e"):
            raise ValueError(
                f"orca1_zdftke_config kappa_convention {kappa_convention!r} "
                "invalid; expected 'veros_sqrte' (NEMO avm = rn_ediff*zmxlm*"
                "sqrt(en)) or 'gaspar_sqrt2e' (the legacy double-count).")
        _cfg = _cfg._replace(kappa_convention=kappa_convention)
    # Shear-production discretisation (``--tke-shear-production``).  DEFAULT
    # keeps the card value (``squared_centered``).  ``nemo_face_native`` is
    # NEMO's zdf_sh2: face-native differences, a now x before velocity
    # product, and production DOUBLED adjacent to coasts via
    # (2 - umask*umask) (zdfsh2.F90:78-94).  It is the LAST unclosed gap on
    # this card and the only one the offline column probe cannot evaluate --
    # it needs the raw C-grid face state, which k_profiles supplies on the
    # tripole under --partial-cell.
    if shear_production is not None:
        if shear_production not in ("squared_centered", "nemo_face_native",
                                    "nemo_face_native_now2", "nemo_burchard"):
            raise ValueError(
                f"orca1_zdftke_config shear_production {shear_production!r} "
                "invalid; expected 'squared_centered', 'nemo_face_native' "
                "(NEMO zdf_sh2, leap-frog family), 'nemo_face_native_now2' "
                "(same face geometry at NOW^2 -- the key_RK3 oracle variant) "
                "or 'nemo_burchard'.")
        _cfg = _cfg._replace(tke_shear_production=shear_production)
    # Mixing-length formulation (``--tke-mxl-choice``).  DEFAULT keeps the card
    # value (2 = Veros Bougeault-Lacarrere, the current production).  3 selects
    # NEMO nn_mxl=3: the lup/ldown |dl/dz|<=e3t sweeps WITH the ln_mxl0 wind-
    # stress surface anchor (l_sfc=max(rn_mxl0, vkarmn*2e5/(rho0*g)*|tau|)) that
    # choice 2 omits — a larger upper-ocean mixing length -> more mixed-layer
    # mixing -> cooler SST (the tropical-warm fix candidate).  The tripole
    # k_profiles path already threads dz_ref/jacobian/taum so choice 3 is live.
    # 4 selects NEMO nn_mxl=2 — the value ORCA1's namelist_cfg ACTUALLY sets
    # (`nn_mxl = 2`, verified in RUN_GATEWAY/namelist_cfg).  Choices 3 and 4
    # share the lup/ldown sweeps and the same eddy-coefficient LENGTH
    # l_k = min(lup,ldn); they differ only in the dissipation length
    # (tke.py:764-785): choice 3 uses l_eps = sqrt(lup*ldn), choice 4 uses
    # l_eps = l_k = min(lup,ldn).  Since min <= sqrt(product), choice 4
    # dissipates MORE (eps = c_eps*e^{3/2}/l_eps).
    # NOTE (codex 2026-08-07 r1+r2, correcting an earlier claim of mine):
    # do NOT say the diffusivity is identical.  Only the LENGTH l_k is
    # shared.  A shorter l_eps raises the dissipation coefficient at fixed
    # positive TKE (tke.py:1007), and K is recomputed from the updated TKE
    # (tke.py:2153), so choice 4 CAN reduce TKE and alter K.  It need not:
    # the two lengths can be equal, later trajectories differ in l_k too,
    # and K can coincide at the floors/ceilings (tke.py:1530).  'Can alter',
    # never 'must'.  Until 2026-08-06 argparse accepted 4 while this
    # builder raised on it, so `--tke-mxl-choice 4` crashed the run.
    if mxl_choice is not None:
        # int(4.9) would silently truncate to 4; argparse blocks that via
        # type=int but a PROGRAMMATIC caller does not (codex 2026-08-07 #5).
        # Reject non-integral numerics (int(4.9)->4 would run choice-4 physics
        # under a nonsense value) and anything outside the SHARED set.  bool is
        # excluded explicitly: True == 1 would otherwise sneak through int().
        if (isinstance(mxl_choice, bool)
                or not isinstance(mxl_choice, (int, np.integer))
                or int(mxl_choice) not in TKE_MXL_CHOICES):
            raise ValueError(
                f"orca1_zdftke_config mxl_choice {mxl_choice!r} invalid; expected "
                f"an int in {TKE_MXL_CHOICES} (2=Veros Bougeault-Lacarrere, "
                "3=NEMO nn_mxl=3, 4=NEMO nn_mxl=2 -- the value ORCA1's "
                "namelist_cfg actually runs).")
        _cfg = _cfg._replace(tke_mxl_choice=int(mxl_choice))
    if n2_mode is not None:
        # Scheme Literal -> validate at config-build time on the STATIC value,
        # never a silent fallback (dispatch hardening).
        _N2_MODES = ("insitu", "insitu_signed", "adiabatic", "nemo_bn2")
        if n2_mode not in _N2_MODES:
            raise ValueError(
                f"orca1_zdftke_config n2_mode {n2_mode!r} invalid; expected "
                f"one of {_N2_MODES}")
        _cfg = _cfg._replace(n2_mode=n2_mode)
    if n2_eos_form is not None:
        _EOS_FORMS = ("seos", "teos10")
        if n2_eos_form not in _EOS_FORMS:
            raise ValueError(
                f"orca1_zdftke_config n2_eos_form {n2_eos_form!r} invalid; "
                f"expected one of {_EOS_FORMS}")
        _cfg = _cfg._replace(n2_eos_form=n2_eos_form)
    # Prognostic vs diagnostic TKE (``--tke-prognostic``).  DEFAULT keeps the
    # card value (False = the DINO-validated quasi-steady Mode-B diagnostic, 3
    # backward-Euler iters).  True selects NEMO's PROGNOSTIC en integration
    # (Mode-A, one backward-Euler step/model-step carrying OceanState.tke) — the
    # closure ACCUMULATES the diurnal-SW + wind TKE that mixes the tropical ML,
    # the candidate for the tropical warm that the mixing-length levers can't
    # touch.  The tripole implicit path carries state.tke
    # (_tke_prognostic_active gates it) so this is live.
    if prognostic is not None:
        _cfg = _cfg._replace(prognostic=bool(prognostic))
    return _cfg


def ah_profile_from_file(grid, path, A_h_base: float):
    """Latitudinal A_h profile from ORCA1's eddy_viscosity_3D.nc.

    Zonal MEDIAN of the surface-level ahmf per source row, interpolated onto
    the grid's nominal latitude rows and returned as a tuple of RATIOS to
    ``A_h_base`` (the hashable static form LateralViscosityConfig carries).
    The bipolar cap's nominal latitudes are distorted, but the file is a
    uniform 20000 there, so the interpolation error multiplies a constant.
    """
    import netCDF4 as nc4
    ds = nc4.Dataset(path)
    try:
        ahm = np.ma.filled(np.ma.masked_invalid(
            ds.variables["ahmf_3d"][:]), np.nan).astype(np.float64).squeeze()
        src_lat = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
    finally:
        ds.close()
    surf = ahm[0]                                  # (ny, nx), level 0
    surf = np.where(surf > 0.0, surf, np.nan)      # 0 = land in this file
    row_lat = np.nanmedian(src_lat, axis=1)        # (ny,)
    row_ahm = np.nanmedian(surf, axis=1)
    good = np.isfinite(row_lat) & np.isfinite(row_ahm)
    if good.sum() < 10:
        raise SystemExit(f"--A-h-profile-file {path}: <10 usable rows")
    order = np.argsort(row_lat[good])
    xs, ys = row_lat[good][order], row_ahm[good][order]
    lat_deg = np.degrees(np.asarray(grid.lat, dtype=np.float64))
    prof = np.interp(lat_deg, xs, ys, left=ys[0], right=ys[-1]) / float(A_h_base)
    print(f"[A_h profile] {path}: ratio min {prof.min():.4f} (lat "
          f"{lat_deg[int(np.argmin(prof))]:.1f}) max {prof.max():.4f}; "
          f"A_h_base {A_h_base:g}")
    return tuple(float(x) for x in prof)


def build_tripole_vmix_config(tripole_vmix: str, iwm=None, tke_eice=None,
                              tke_surface_bc=None, tke_mxl_choice=None,
                              tke_n2_mode=None, tke_n2_eos_form=None,
                              tke_prognostic=None, tke_kappa_convention=None,
                              tke_shear_production=None, tke_lc=None,
                              tke_etau=None):
    """``VerticalMixingConfig`` for ``--tripole-vmix`` (+ optional zdfiwm).

    ``tripole_vmix``: "none" (byte-identical no-closure default), "tke"
    (ORCA1 ``&namzdf_tke`` mapping, :func:`orca1_zdftke_config`) or "kpp"
    (scheme defaults).  Unknown selections raise (dispatch hardening).

    ``iwm`` (an enabled ``IWMConfig`` or None) rides the SAME config: zdfiwm
    is ADDITIVE on top of the closure inside ``compute_vertical_K_profiles``
    — exactly NEMO's zdfphy ordering (zdf_tke computes avt/avm first, zdf_iwm
    then adds onto them), and the same contract the latlon path uses when it
    attaches iwm onto its KPP config.  So ``--tripole-vmix tke --iwm``
    composes; it is NOT an error.

    ``tke_eice`` (``--tke-eice``): None keeps the ORCA1 card default
    (nn_eice=3); 0/1/3 override the under-ice lc/etau attenuation mode for
    A/B runs (0 reproduces the pre-2026-07-18 no-attenuation behaviour).

    ``tke_surface_bc`` (``--tke-surface-bc``): None keeps the TKEConfig default
    (``veros_flux``); ``nemo_dirichlet`` selects the NEMO ``en(1)=rn_ebb·|τ|/ρ0``
    surface condition (closes the flagged surface-BC fidelity gap).  Applicable
    ONLY to the ``tke`` closure; supplied with ``none``/``kpp`` it raises rather
    than silently no-op (dispatch hardening), mirroring the eice contract.
    """
    from legoesm.ocean.physics.vertical_mixing.config import (
        KPPConfig, VerticalMixingConfig,
    )
    for _fl, _v in (("--tke-surface-bc", tke_surface_bc),
                    ("--tke-mxl-choice", tke_mxl_choice),
                    ("--tke-lc", tke_lc),
                    ("--tke-etau", tke_etau),
                    ("--tke-n2-mode", tke_n2_mode),
                    ("--tke-n2-eos-form", tke_n2_eos_form),
                    ("--tke-prognostic", tke_prognostic),
                    ("--tke-kappa-convention", tke_kappa_convention),
                    ("--tke-shear-production", tke_shear_production)):
        if _v is not None and tripole_vmix != "tke":
            raise ValueError(
                f"{_fl} {_v!r} requires --tripole-vmix tke; got --tripole-vmix "
                f"{tripole_vmix!r} (that knob only exists in the TKE closure).")
    _iwm_on = iwm is not None and iwm.enabled
    if tripole_vmix == "none":
        vm = VerticalMixingConfig(scheme="none")
    elif tripole_vmix == "tke":
        _tke = orca1_zdftke_config(iwm_enabled=_iwm_on, surface_bc=tke_surface_bc,
                                   mxl_choice=tke_mxl_choice,
                                   n2_mode=tke_n2_mode,
                                   n2_eos_form=tke_n2_eos_form,
                                   prognostic=tke_prognostic,
                                   kappa_convention=tke_kappa_convention,
                                   shear_production=tke_shear_production,
                                   lc=tke_lc, etau_mode=tke_etau)
        if tke_eice is not None:
            if int(tke_eice) not in (0, 1, 3):
                raise ValueError(
                    f"--tke-eice {tke_eice!r} invalid; expected 0, 1 or 3 "
                    "(NEMO nn_eice modes).")
            _tke = _tke._replace(eice=int(tke_eice))
        vm = VerticalMixingConfig(scheme="tke", tke=_tke)
    elif tripole_vmix == "kpp":
        vm = VerticalMixingConfig(scheme="kpp", kpp=KPPConfig())
    else:
        raise ValueError(
            f"unknown --tripole-vmix {tripole_vmix!r}; expected 'none', "
            "'tke' or 'kpp'.")
    if _iwm_on:
        vm = vm._replace(iwm=iwm)
    return vm


def build_tripole(nlev: int, H_max: float, mesh_path: str,
                  woa_init: bool = False, woa_t=None, woa_s=None, n_gpus: int = 1,
                  pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None, A_h_eq_sigma_deg=None,
                  ke_gradient_scheme=None, partial_cell=False,
                  adaptive_implicit_vertadv=None, bathy_smoothing_passes=0,
                  momentum_time_integrator=None, barotropic_solver=None,
                  barotropic_pcg_variant=None,
                  barotropic_diffusion_alpha=None, n_barotropic_substeps=None,
                  barotropic_time_filter=None, bottom_drag_r=None,
                  C_smag=None, C_leith=None, C_smag_lap=None,
                  momentum_advection=None, slope_foot_alpha=None,
                  slope_foot_n_levels=None, slope_foot_threshold=None,
                  min_levels=1, div_damp_2=None, div_damp_4=None,
                  smag_cfl_safety=None, convection="none",
                  convection_K_conv=1.0, convection_K_bg=1e-5,
                  freeze_floor=None, freezing=None, ew_cyclic_overlap=None,
                  runoff_depth_spread_m=None, tracer_advection=None,
                  mle=None, dz_ref_override=None,
                  bottom_drag_scheme=None, bottom_drag_cd0=None,
                  bottom_drag_cdmax=None, bottom_drag_z0=None,
                  bottom_drag_ke0=None, iwm=None, iwm_forcing_file=None,
                  ddm=None, prescribed_flow=None, no_gm_redi=False,
                  tripole_vmix="none", tke_eice=None, tke_surface_bc=None,
                  tke_mxl_choice=None, tke_prognostic=None,
                  tke_n2_mode=None, tke_n2_eos_form=None,
                  tke_kappa_convention=None, tke_shear_production=None,
                  tke_lc=None, tke_etau=None, A_h_profile_file=None,
                  gm_treguier=False, gm_aei0=_GM_AEI0_DEFAULT,
                  gm_kappa_min=_GM_KAPPA_MIN_DEFAULT,
                  gm_slope_scheme=None, gm_bolus_advection=None,
                  gm_msc_stabilize=None,
                  store_mass_flux=False, store_salt_flux=False):
    """Build the eORCA1 tripole grid + model + initial state with NEMO's mask/bathy.

    Reuses run_omip's validated tripole setup. ``forcing_mode='jra55_do_tropical'``
    selects the surface-forcing ``scheme='none'`` config so the model applies NO
    internal restoring -- CORE-II forcing is applied externally by the applicator.

    ``woa_init`` initialises T/S from the WOA18 climatology (``init_ocean_from_woa``)
    instead of the idealised rest state -- ESSENTIAL for a faithful comparison, since
    NEMO starts from the Gouretski/WOCE climatology; a rest-state vs climatology IC
    confounds model differences with IC differences over a few-year spinup. (WOA18 is
    a close stand-in for NEMO's exact Gouretski IC, which is the further refinement.)
    """
    # Dispatch hardening at the programmatic surface too (argparse `choices`
    # only guards the CLI): ""/None/typos must not silently run as "none".
    if gm_slope_scheme is not None and gm_slope_scheme not in _GM_SLOPE_SCHEMES:
        raise ValueError(
            f"unknown gm_slope_scheme {gm_slope_scheme!r}; expected one of "
            f"{sorted(_GM_SLOPE_SCHEMES)}.")
    if (gm_bolus_advection is not None
            and gm_bolus_advection not in _GM_BOLUS_FORMS):
        raise ValueError(
            f"unknown gm_bolus_advection {gm_bolus_advection!r}; expected one "
            f"of {sorted(_GM_BOLUS_FORMS)}.")
    # `through_fct` is honored ONLY with slope_scheme="nemo_iso_lap": the model
    # gates on exactly that pair (ocean_model_latlon_cgrid `_want_bolus`), so
    # any other slope scheme would silently keep the centred bolus flux while
    # the run manifest claimed FCT routing.  Fail instead.
    if gm_bolus_advection == "through_fct" and gm_slope_scheme != "nemo_iso_lap":
        raise ValueError(
            "gm_bolus_advection='through_fct' requires "
            "gm_slope_scheme='nemo_iso_lap' (the model honors the pair, not "
            f"the flag alone); got gm_slope_scheme={gm_slope_scheme!r}.")
    if tripole_vmix not in ("none", "tke", "kpp"):
        raise ValueError(
            f"unknown tripole_vmix {tripole_vmix!r}; expected 'none', 'tke' "
            "or 'kpp'.")
    from scripts.run import run_omip
    # Pick the tripole resolution from the mesh file: eORCA025 (1/4 deg) vs the
    # default eORCA1 (1 deg). create_tripole_grid reads the grid (glamt/e1t.../
    # tmask + fold) from this SAME file, so the grid and the land_mask/bathy
    # (read below) provably come from one mesh -- assert it to kill any drift.
    _mname = Path(mesh_path).name
    resolution = ("eorca025" if "025" in _mname
                  else "eorca05" if "05" in _mname else "eorca1")
    _grid_mesh = run_omip._parse_resolution("tripole", resolution)["mesh_path"]
    if Path(_grid_mesh).resolve() != Path(mesh_path).resolve():
        raise ValueError(
            f"tripole mesh mismatch: grid built from {_grid_mesh!r} but "
            f"mask/bathy read from {mesh_path!r}. Pass --mesh {_grid_mesh}."
        )
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "tripole", resolution, nlev, H_max,
        physics_preset="full", water_type="II",
        forcing_mode="jra55_do_tropical",
        dz_ref_override=dz_ref_override,
    )
    # Optional dycore-stability overrides (for WOA cold-start tuning): rebuild
    # the config + model from run_omip's validated tripole base, changing only
    # the requested knobs (e.g. pgf_scheme="smc03", higher A_h/B_h).
    _ovr = {k: v for k, v in (("pgf_scheme", pgf_scheme), ("A_h", A_h),
                              ("B_h", B_h), ("K_bih", K_bih),
                              ("A_h_eq_boost", A_h_eq_boost),
                              ("A_h_eq_sigma_deg", A_h_eq_sigma_deg),
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ("momentum_time_integrator", momentum_time_integrator),
                              ("barotropic_solver", barotropic_solver),
                              ("barotropic_implicit_pcg_variant",
                               barotropic_pcg_variant),
                              ("barotropic_diffusion_alpha", barotropic_diffusion_alpha),
                              ("n_barotropic_substeps", n_barotropic_substeps),
                              ("barotropic_time_filter", barotropic_time_filter),
                              ("bottom_drag_r", bottom_drag_r),
                              ("bottom_drag_scheme", bottom_drag_scheme),
                              ("bottom_drag_cd0", bottom_drag_cd0),
                              ("bottom_drag_cdmax", bottom_drag_cdmax),
                              ("bottom_drag_z0", bottom_drag_z0),
                              ("bottom_drag_ke0", bottom_drag_ke0),
                              ("C_smag", C_smag), ("C_leith", C_leith),
                              ("C_smag_lap", C_smag_lap),
                              ("momentum_advection", momentum_advection),
                              ("slope_foot_alpha", slope_foot_alpha),
                              ("slope_foot_n_levels", slope_foot_n_levels),
                              ("slope_foot_threshold", slope_foot_threshold),
                              ("div_damp_2", div_damp_2),
                              ("div_damp_4", div_damp_4),
                              ("smag_cfl_safety", smag_cfl_safety),
                              ("freeze_floor", freeze_floor),
                              ("freezing", freezing),
                              ("ew_cyclic_overlap", ew_cyclic_overlap),
                              ("runoff_depth_spread_m", runoff_depth_spread_m),
                              ("tracer_advection", tracer_advection),
                              ("prescribed_flow", prescribed_flow),
                              ) if v is not None}
    if no_gm_redi:
        # Disable the recipe's GM/Redi (NEMOMatchTripoleRecipeConfig ships
        # gm_redi=True, kappa=600).  Appended AFTER the not-None filter above
        # because the override VALUE here is None.  REQUIRED under
        # --prescribed-flow (validate_prescribed_flow_args): GM bolus is
        # parameterized advection the lever cannot pin, and the model
        # constructor rejects the combination.
        _ovr["gm_redi"] = None
        print("[setup] tripole GM/Redi DISABLED (--no-gm-redi)")
    elif gm_treguier:
        # NEMO-faithful eddy-induced-velocity coefficient.  NEMO ORCA1 runs
        # &namtra_eiv with ln_ldfeiv=.true. and nn_aei_ijk_t=21 -> aeiu/aeiv =
        # F(growth rate of baroclinic instability), a 2-D time-varying field
        # capped at aei0 = 1/2*rn_Ue*rn_Le = 0.5 * 0.018 * 100e3 = 900 m^2/s
        # (ldftra.F90:290-293 sets zUfac = r1_2*rn_Ud for the laplacian, which
        # ORCA1 runs; NEMO's emitted aeiu_2d maxes at exactly 900).  legoESM's
        # TreguierConfig IS that scaling (already used by the DINO oracle card).
        # Visbeck and Treguier are mutually exclusive (gm_redi_latlon_cgrid
        # raises); the tripole recipe already ships Visbeck OFF, so this only
        # turns the Treguier block on.
        _ovr["gm_redi"] = _tripole_treguier_gm_redi(gm_aei0, gm_kappa_min)
        print(f"[setup] tripole GM kappa_GM scheme: TREGUIER (NEMO ldf_eiv "
              f"nn_aei_ijk_t=21, aei0={float(gm_aei0):g} m^2/s, "
              f"kappa_min={float(gm_kappa_min):g} m^2/s) — Visbeck OFF")
    # NEMO-faithful lateral-mixing lane (opt-in).  Composes with whichever
    # kappa_GM scheme is active above: this touches ONLY the operator fields, so
    # `--gm-slope-scheme nemo_iso_lap` alone is a one-variable operator swap
    # against the control, independent of the coefficient.
    if gm_slope_scheme is not None or gm_bolus_advection is not None:
        _gm_base = _ovr.get("gm_redi", config.flat_get("gm_redi"))
        if _gm_base is None:
            raise ValueError(
                "--gm-slope-scheme / --gm-bolus-advection need GM/Redi "
                "enabled; this run has it disabled (--no-gm-redi).")
        _gm_kw = {}
        if gm_slope_scheme is not None:
            _gm_kw["slope_scheme"] = gm_slope_scheme
        if gm_bolus_advection is not None:
            _gm_kw["gm_bolus_advection"] = gm_bolus_advection
        if gm_msc_stabilize is not None:
            _gm_kw["msc_stabilize"] = bool(gm_msc_stabilize)
        _ovr["gm_redi"] = _gm_base._replace(**_gm_kw)
        print(f"[setup] tripole GM/Redi operator: "
              f"slope_scheme={_ovr['gm_redi'].slope_scheme}, "
              f"gm_bolus_advection={_ovr['gm_redi'].gm_bolus_advection}, "
              f"msc_stabilize={_ovr['gm_redi'].msc_stabilize} "
              f"(NEMO ORCA1 = nemo_iso_lap + through_fct + msc_stabilize; "
              f"NOTE the bolus then rides THIS run's tracer limiter, which is "
              f"not necessarily NEMO's FCT)")
    # IMPLICIT vertical mixing (NEMO ln_zdf*, MOM6 CVMix, MPAS all do this; the
    # config default is True). _create_setup()'s arg default is False (explicit) --
    # at the NEMO 75-level grid the explicit KPP vertical-viscosity CFL blows the
    # cold-start (see the same fix in build_latlon_bathy). Force it on uniformly so
    # the tripole matches latlon/mpas; the convection block below is then redundant.
    _ovr["implicit_vertical_mixing"] = True
    # Grid-agnostic convective adjustment (Oceananigans-style enhanced
    # vertical diffusivity where N^2 < 0) and/or the Fox-Kemper MLE
    # restratification.  The tripole base config ships physics=None; opting in
    # attaches an OceanPhysicsConfig.  Convective K flows through the SAME
    # grid-agnostic compute_vertical_K_profiles -> implicit backward-Euler
    # vertical solve; MLE adds an explicit bolus tracer tendency through the
    # same physics_fn pipeline.  Default (both off) leaves the validated
    # faithful config untouched.
    _use_convection = bool(convection and convection != "none")
    _use_iwm = iwm is not None and iwm.enabled
    _use_ddm = ddm is not None and ddm.enabled
    _use_vmix = bool(tripole_vmix and tripole_vmix != "none")
    if (_use_convection or mle is not None or _use_iwm or _use_ddm
            or _use_vmix):
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig, EnhancedDiffusionConfig,
        )
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        # CONVECTION/MLE physics pipeline.  Every other module is explicitly
        # disabled: the OceanPhysicsConfig defaults are NOT inert
        # (lateral_mixing defaults to harmonic -- assumes a cubed-sphere 4-D
        # layout and would crash on the tripole 3-D state; shortwave_penetration
        # defaults ON -- would double-count the shortwave that the dynamics-core
        # external-tau block already applies).  The C-grid model's own
        # config-level A_h/B_h/K_h, bottom_drag_r and the external CORE-II
        # forcing are untouched; the pipeline contributes ONLY the convective K
        # and/or the MLE bolus tracer tendency.
        _conv_cfg = OceanConvectionConfig(
            scheme=convection,
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=convection_K_conv, K_bg=convection_K_bg,
            ),
        ) if _use_convection else OceanConvectionConfig(scheme="none")
        # --tripole-vmix: NEMO's vertical-mixing CLOSURE on the tripole (the
        # last audited namelist gap — NEMO ORCA1 runs zdftke).  "none"
        # (default) keeps the byte-identical no-closure pipeline; "tke"
        # attaches the ORCA1 &namzdf_tke mapping (orca1_zdftke_config);
        # "kpp" attaches KPP defaults.  TKE/CATKE K-profiles are computed
        # inside the implicit solve's compute_vertical_K_profiles fallback
        # (combined.py keeps the pipeline factory a deliberate no-op), which
        # receives the step-time surface_forcing (CORE-II tau_x/tau_y) — so
        # the surface TKE input sees the real wind stress.
        _vm_cfg = build_tripole_vmix_config(
            tripole_vmix, iwm=iwm if _use_iwm else None,
            tke_eice=tke_eice, tke_surface_bc=tke_surface_bc,
            tke_mxl_choice=tke_mxl_choice, tke_prognostic=tke_prognostic,
            tke_n2_mode=tke_n2_mode, tke_n2_eos_form=tke_n2_eos_form,
            tke_kappa_convention=tke_kappa_convention,
            tke_shear_production=tke_shear_production,
            tke_lc=tke_lc, tke_etau=tke_etau)
        if _use_vmix:
            print(f"[setup] tripole vertical-mixing closure: {tripole_vmix}"
                  + (" (ORCA1 namzdf_tke namelist mapping)"
                     if tripole_vmix == "tke" else ""))
            if (tripole_vmix == "tke"
                    and int(getattr(_vm_cfg.tke, "eice", 0)) != 0):
                print(f"[setup] TKE under-ice attenuation eice="
                      f"{int(_vm_cfg.tke.eice)} (NEMO nn_eice): active only "
                      "when an ice concentration reaches the closure "
                      "(--prognostic-sea-ice, or a prescribed SIC via "
                      "--ice-albedo/--ice-thermo/--sss-restore); without one "
                      "the attenuation is inert (open water, fi=0).")
        if _use_iwm or _use_vmix:
            # zdfiwm rides the vertical-mixing config (attached above by
            # build_tripole_vmix_config); with --tripole-vmix none the
            # scheme stays "none" (backgrounds + convection EVD only), with
            # tke/kpp the additive wave K enters compute_vertical_K_profiles
            # AFTER the closure — NEMO's zdfphy ordering.
            # implicit_vertical_mixing is already forced True above.
            # NEMO zdfiwm_init FORCES the model backgrounds to molecular
            # values (avmb = rnu = 1.4e-6 m²/s, avtb = 1e-10 m²/s): the wave
            # field IS the interior background.  Mirror that (codex r1 #2) —
            # keeping the OMIP A_v/K_v floors would double-count backgrounds.
            # Same when a CLOSURE is attached (codex tke r1 #1/#2): NEMO
            # composes avm = max(closure, avmb) / avt = max(pdl·avt, avtb)
            # (zdftke.F90:715,723) — exactly what the closure-internal
            # kappaM_min/kappaH_min floors do (orca1_zdftke_config carries
            # rn_avm0/rn_avt0, or the molecular pair under --iwm; KPP carries
            # its A_bg/K_bg) — while compute_vertical_K_profiles ADDS the
            # model-level A_v/K_v on top.  Keeping the OMIP floors there
            # would double-count the background, so force them molecular
            # (the residual +1.4e-6/+1e-10 additive term is negligible).
            from legoesm import constants as _const
            _ovr["A_v"] = _const.nu_ocean_molecular
            _ovr["K_v"] = 1.0e-10   # NEMO avtb with ln_zdfiwm
            print("[setup] model additive backgrounds forced to molecular "
                  f"(A_v={_ovr['A_v']:g}, K_v={_ovr['K_v']:g}) — "
                  f"{'zdfiwm_init' if _use_iwm else 'closure floors'} own "
                  "the NEMO avmb/avtb backgrounds")
        if _use_ddm:
            # zdfddm double-diffusive mixing rides the vertical-mixing config
            # (additive avt/avs in compute_vertical_K_profiles); unlike zdfiwm
            # it is purely additive -- NO molecular-background override.
            # implicit_vertical_mixing is already forced True above.
            _vm_cfg = _vm_cfg._replace(ddm=ddm)
        _ovr["physics"] = OceanPhysicsConfig(
            vertical_mixing=_vm_cfg,
            lateral_mixing=LateralMixingConfig(scheme="none"),
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=_conv_cfg,
            shortwave_penetration=None,
            mle=mle,
        )
        # Convective adjustment must apply through the implicit vertical
        # solve (backward-Euler is unconditionally stable; an explicit
        # K_conv would violate CFL at ocean dt).  MLE adds only an explicit
        # tracer tendency (no K_v), so it does NOT itself require the implicit
        # solve; only enable it when convection is on.
        if _use_convection:
            _ovr["implicit_vertical_mixing"] = True
        print(f"[setup] tripole physics ENABLED: "
              f"vmix={tripole_vmix} "
              f"convection={convection if _use_convection else 'none'} "
              f"(K_conv={convection_K_conv} K_bg={convection_K_bg}) "
              f"MLE={'ce=%g' % mle.ce if mle is not None else 'off'} "
              f"IWM={'on' if _use_iwm else 'off'} "
              f"DDM={'on' if _use_ddm else 'off'}")
    if store_mass_flux:
        # #1442: keep the tracer-advecting mass flux on the returned state so
        # transport diagnostics integrate the flux the model actually used
        # instead of reconstructing h*u from the post-barotropic velocity.
        # Pure diagnostic -- the trajectory is unchanged.
        _ovr["store_mass_flux"] = True
    if store_salt_flux:
        # Salt analogue: keep the column-integrated advective SALT flux so the
        # gateway accumulator integrates the model's own limited fluxes
        # (exact channel) alongside its upwind estimate.
        _ovr["store_salt_flux"] = True
    if _ovr:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        # #501/#661: _ovr carries FLAT names (A_h/C_smag_lap/barotropic_solver/
        # bottom_drag_r/...) now nested in sub-configs; replace_flat routes them.
        if A_h_profile_file:
            _ovr["A_h_lat_profile"] = ah_profile_from_file(
                grid, A_h_profile_file,
                _ovr.get("A_h", config.lateral_viscosity.A_h))
        config = config.replace_flat(**_ovr)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        print(f"[setup] tripole config override: {_ovr}")
    land_mask, H_bathy = read_mesh_mask_bathy(mesh_path)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (topography removed -- PGF-over-topo control)")
    n_lat, n_lon = int(grid.lat_T.shape[0]), int(grid.lat_T.shape[1])
    if land_mask.shape != (n_lat, n_lon):
        raise ValueError(
            f"mesh mask shape {land_mask.shape} != grid {(n_lat, n_lon)}"
        )
    n_pad = _south_pad_rows(n_lat, n_gpus)
    if n_pad > 0:
        # Multi-GPU lat-band SPMD divisibility: append n_pad LAND rows at the
        # SOUTH (grid geometry + mask + bathy together) BEFORE the state /
        # partial-cell build so everything downstream is consistent.  The bipolar
        # fold stays at the north (pad_tripole_grid_south shifts fold_j/cap_j +n_pad).
        from legoesm.grids.tripole import pad_tripole_grid_south
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        grid = pad_tripole_grid_south(grid, n_pad)
        land_mask, H_bathy = _pad_mask_bathy_south(land_mask, H_bathy, n_pad)
        model = LatLonCGridOceanModel(grid, z_coord, config)  # rebuild on padded grid
        n_lat = int(grid.n_lat)
        print(f"[setup] SPMD south-pad: +{n_pad} LAND rows -> n_lat={n_lat} "
              f"(n_gpus={n_gpus}, fold still north at j={int(grid.fold.fold_j)})")
    if ew_cyclic_overlap:
        # ORCA 2-pt cyclic-overlap fill of the static geometry, applied BEFORE
        # make_partial_cell (codex HIGH): the partial-cell coordinate
        # (z_coord.is_active / h_partial / bottom_level) is built per-column
        # from H_bathy/land_mask, so it must see the overlap-filled (reconnected)
        # seam -- otherwise is_active stays severed at col0/col_{nx-1} while the
        # 2-D state is wet, corrupting the 3-D tracer-flux masks. A raw tmaskutil
        # marks the halo cols col0/col_{nx-1} LAND (the lon-72.5E seam wall).
        # make_partial_cell with NO cross-column smoothing (the faithful config)
        # is column-local, so identical input columns col0==col{nx-2} yield
        # identical outputs -> the overlap survives it.
        if bathy_smoothing_passes and bathy_smoothing_passes > 0:
            raise ValueError(
                "--ew-cyclic-overlap with bathy smoothing > 0 is unsupported: "
                "cross-column smoothing breaks the seam-column identity that the "
                "partial-cell coordinate relies on. Use 0 smoothing passes.")
        land_mask = _ew_overlap_fill(land_mask)
        H_bathy = _ew_overlap_fill(H_bathy)
        _wet = land_mask > 0.5
        print(f"[setup] EW cyclic-overlap ON (ORCA tripole seam): halo cols "
              f"filled col0<-col{n_lon-2}, col{n_lon-1}<-col1; "
              f"seam wet cells {int(_wet[:, 0].sum())}/{int(_wet[:, 1].sum())}")
    if partial_cell:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        z_coord, H_bathy, land_mask = make_partial_cell(
            z_coord, H_bathy, land_mask, smoothing_passes=bathy_smoothing_passes,
            min_levels=min_levels)
        model = LatLonCGridOceanModel(grid, z_coord, config)
    if iwm is not None and iwm.enabled:
        # FINAL model build with the zdfiwm maps (after every config /
        # z_coord rebuild above).  On the eORCA1 tripole the forcing file
        # is on the SAME mesh — the loader passes it through untouched.
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        _iwm_maps = None
        if iwm_forcing_file:
            from legoesm.ocean.iwm_forcing import load_iwm_forcing
            # grid.lat_T/lon_T are stored in RADIANS on the tripole grid;
            # the loader expects degrees (matches the eORCA nav_lat/nav_lon).
            _iwm_maps = load_iwm_forcing(
                iwm_forcing_file,
                np.degrees(np.asarray(grid.lat_T)),
                np.degrees(np.asarray(grid.lon_T)),
                land_mask=land_mask)
        model = LatLonCGridOceanModel(grid, z_coord, config,
                                      iwm_forcing=_iwm_maps)
        print(f"[setup] tripole zdfiwm ENABLED "
              f"(maps={'file:' + iwm_forcing_file if iwm_forcing_file else 'uniform fallback'}, "
              f"mevar={iwm.mevar} tsdiff={iwm.tsdiff})")
    state = run_omip._init_rest_state(
        "tripole", grid, z_coord, H_max,
        H_bathy=jnp.asarray(H_bathy), land_mask=jnp.asarray(land_mask),
    )
    if woa_init:
        from legoesm.core.field import Field
        T_woa, S_woa = compute_woa_3d(grid, z_coord, woa_t, woa_s,
                                      H_bathy, land_mask)
        if ew_cyclic_overlap:
            # Overlap-fill the IC so the halo columns start consistent with
            # their partners (the per-step projection keeps them so).
            T_woa = _ew_overlap_fill(np.asarray(T_woa))
            S_woa = _ew_overlap_fill(np.asarray(S_woa))
        if T_woa.shape != state.T.data.shape:
            raise ValueError(
                f"WOA T shape {T_woa.shape} != state T {state.T.data.shape}"
            )
        state = state._replace(
            T=Field(jnp.asarray(T_woa), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            S=Field(jnp.asarray(S_woa), name=state.S.name,
                    dims=state.S.dims, units=state.S.units),
        )
        print(f"[setup] T/S initialised from WOA18 ({Path(woa_t).name})")
    return grid, z_coord, model, state, np.asarray(H_bathy)


def build_latlon_bathy(nlev: int, H_max: float, mesh_path: str,
                       n_lat: int = 180, n_lon: int = 360,
                       woa_init: bool = False, woa_t=None, woa_s=None,
                       pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None, A_h_eq_sigma_deg=None,
                       ke_gradient_scheme=None, partial_cell=False,
                       adaptive_implicit_vertadv=None, bathy_smoothing_passes=0,
                  momentum_time_integrator=None, barotropic_solver=None,
                  barotropic_pcg_variant=None,
                  barotropic_diffusion_alpha=None, n_barotropic_substeps=None,
                  barotropic_time_filter=None, bottom_drag_r=None,
                  C_smag=None, C_leith=None, C_smag_lap=None,
                  momentum_advection=None, slope_foot_alpha=None,
                  slope_foot_n_levels=None, slope_foot_threshold=None,
                  min_levels=1, div_damp_2=None, div_damp_4=None,
                  smag_cfl_safety=None, freeze_floor=None, freezing=None,
                  use_polar_filter=None, polar_filter_cutoff_lat_deg=None,
                  polar_filter_max_wave_speed=None,
                  polar_filter_safety_factor=None,
                  runoff_depth_spread_m=None, tracer_advection=None,
                  mle=None, dz_ref_override=None, mask_marginal_seas=False,
                  bottom_drag_scheme=None, bottom_drag_cd0=None,
                  bottom_drag_cdmax=None, bottom_drag_z0=None,
                  bottom_drag_ke0=None, iwm=None, iwm_forcing_file=None,
                  ddm=None, vertical_mixing=None,
                  prescribed_flow=None, no_gm_redi=False,
                  store_mass_flux=False, store_salt_flux=False):
    """Build a regular lat-lon C-grid with REALISTIC bathymetry + the run_omip
    production config (smc03 PGF, biharmonic, implicit-CN barotropic, GM/Redi,
    KPP) -- documented to run STABLE 50+ yr with real geometry, unlike the
    tripole (adcroft) path which blows up on a realistic cold-start.

    Bathymetry + land mask are NEMO's OWN eORCA1 fields (tmaskutil / e3t_0)
    regridded to the lat-lon grid (nearest-neighbour, periodic) -- so the
    geometry still matches the NEMO reference.
    """
    from scripts.run import run_omip
    import xarray as xr
    from scripts.validate.compare_omip_nemo import regrid_curv_to_latlon
    res = f"{n_lat}x{n_lon}"
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "latlon", res, nlev, H_max, physics_preset="full", water_type="II",
        use_bathymetry=True, pgf_scheme=pgf_scheme,
        A_h_override=A_h, B_h_override=B_h,
        # _create_setup's bathy branch builds gm_redi=None when no_gm_redi
        # (else the _DEFAULT_BATHY_GM_REDI Visbeck config).  REQUIRED under
        # --prescribed-flow: GM bolus is parameterized advection the lever
        # cannot pin, and the model constructor rejects the combination.
        no_gm_redi=no_gm_redi,
        dz_ref_override=dz_ref_override,
        # KPP boundary-layer-depth override (Ri_crit / Cv): None reproduces the
        # default KPPConfig byte-for-byte; a custom VerticalMixingConfig shoals
        # the diagnosed-too-deep JANUARY winter mixed layer (ll2: NH-midlat Jan
        # MLD 174 m vs NEMO 93 m -> over-mixes away the warm 100 m mode water ->
        # -2.83 C SST). The use_bathymetry path threads this into bathy_physics.
        vertical_mixing=vertical_mixing,
    )
    _ovr = {k: v for k, v in (("K_bih", K_bih),
                              ("A_h_eq_boost", A_h_eq_boost),
                              ("A_h_eq_sigma_deg", A_h_eq_sigma_deg),
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ("momentum_time_integrator", momentum_time_integrator),
                              ("barotropic_solver", barotropic_solver),
                              ("barotropic_implicit_pcg_variant",
                               barotropic_pcg_variant),
                              ("barotropic_diffusion_alpha", barotropic_diffusion_alpha),
                              ("n_barotropic_substeps", n_barotropic_substeps),
                              ("barotropic_time_filter", barotropic_time_filter),
                              ("bottom_drag_r", bottom_drag_r),
                              ("bottom_drag_scheme", bottom_drag_scheme),
                              ("bottom_drag_cd0", bottom_drag_cd0),
                              ("bottom_drag_cdmax", bottom_drag_cdmax),
                              ("bottom_drag_z0", bottom_drag_z0),
                              ("bottom_drag_ke0", bottom_drag_ke0),
                              ("C_smag", C_smag), ("C_leith", C_leith),
                              ("C_smag_lap", C_smag_lap),
                              ("momentum_advection", momentum_advection),
                              ("slope_foot_alpha", slope_foot_alpha),
                              ("slope_foot_n_levels", slope_foot_n_levels),
                              ("slope_foot_threshold", slope_foot_threshold),
                              ("div_damp_2", div_damp_2),
                              ("div_damp_4", div_damp_4),
                              ("smag_cfl_safety", smag_cfl_safety),
                              ("freeze_floor", freeze_floor),
                              ("freezing", freezing),
                              ("runoff_depth_spread_m", runoff_depth_spread_m),
                              ("tracer_advection", tracer_advection),
                              ("prescribed_flow", prescribed_flow),
                              ("use_polar_filter", use_polar_filter),
                              ("polar_filter_cutoff_lat_deg",
                               polar_filter_cutoff_lat_deg),
                              ("polar_filter_max_wave_speed",
                               polar_filter_max_wave_speed),
                              ("polar_filter_safety_factor",
                               polar_filter_safety_factor),
                              ) if v is not None}
    # IMPLICIT vertical mixing of momentum + tracers (NEMO ln_zdf*, MOM6, and
    # tripole all do this; the LatLonCGridOceanConfig default is True). The lat-lon
    # _create_setup() arg default is False (explicit), which silently reproduces
    # the historical explicit-diffusion path: at the NEMO 75-level grid the ~1 m
    # surface cell makes the explicit KPP vertical-viscosity CFL A_v*dt/dz_0^2
    # ~ O(1)>>limit, so the physics momentum tendency (`phys_u`, momentum-diag job
    # 8487918) blows the cold-start in ~2 steps at the Kuroshio. The backward-Euler
    # implicit solve is unconditionally stable -> force it on for the OMIP latlon
    # path (20-level happened to stay under the explicit CFL; 75-level does not).
    _ovr["implicit_vertical_mixing"] = True
    if store_mass_flux:
        # #1442: keep the tracer-advecting mass flux on the returned state so
        # transport diagnostics integrate the flux the model actually used
        # instead of reconstructing h*u from the post-barotropic velocity.
        # Pure diagnostic -- the trajectory is unchanged.  Threaded here as
        # well as in build_tripole because --gateway-transports supports BOTH
        # app grids (SUPPORTED_APP_GRIDS = ("tripole", "latlon")); wiring only
        # the tripole left every supported latlon run silently reconstructing
        # (codex RED 6).
        _ovr["store_mass_flux"] = True
    if store_salt_flux:
        # Salt analogue, threaded in BOTH builders for the same RED-6 reason.
        _ovr["store_salt_flux"] = True
    if _ovr:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        # #501/#661: _ovr carries FLAT names now nested in sub-configs
        # (lateral_viscosity/barotropic/bottom_drag/polar_filter); replace_flat
        # routes them.
        config = config.replace_flat(**_ovr)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        print(f"[setup] latlon config override: {_ovr}")
    if mle is not None:
        # Merge the Fox-Kemper MLE into the EXISTING latlon-bathy physics
        # (KPP + enhanced-diffusion convection); do NOT replace it (that would
        # drop KPP/convection).  The bolus tracer tendency rides the same
        # physics_fn pipeline.
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        if config.physics is None:
            raise ValueError(
                "--mle on latlon_bathy expected a physics config (KPP/"
                "convection) but config.physics is None.")
        config = config._replace(physics=config.physics._replace(mle=mle))
        model = LatLonCGridOceanModel(grid, z_coord, config)
        print(f"[setup] latlon MLE ENABLED: ce={mle.ce:g}")
    if iwm is not None and iwm.enabled:
        # zdfiwm rides the vertical-mixing config inside the EXISTING
        # latlon-bathy physics (KPP + convection) — merge, don't replace
        # (same doctrine as the MLE block above).  The maps + model
        # rebuild happen below, after the land mask is known.
        if config.physics is None:
            raise ValueError(
                "--iwm on latlon_bathy expected a physics config (KPP/"
                "convection) but config.physics is None.")
        _vm_iwm = config.physics.vertical_mixing._replace(iwm=iwm)
        config = config._replace(
            physics=config.physics._replace(vertical_mixing=_vm_iwm))
        # NEMO zdfiwm_init FORCES the model backgrounds to molecular values
        # (avmb = rnu = 1.4e-6 m²/s, avtb = 1e-10 m²/s): the wave field IS
        # the interior background (codex r1 #2).  KPP's own scheme
        # backgrounds (K_bg/A_bg) remain user-tunable via --kpp-k-bg /
        # --kpp-a-bg for the NEMO-faithful configuration.
        from legoesm import constants as _const
        config = config.replace_flat(
            A_v=_const.nu_ocean_molecular, K_v=1.0e-10)
        print("[setup] zdfiwm: model backgrounds forced to molecular "
              f"(A_v={_const.nu_ocean_molecular:g}, K_v=1e-10) per zdfiwm_init")
    if ddm is not None and ddm.enabled:
        # zdfddm double-diffusive mixing rides the vertical-mixing config
        # inside the EXISTING latlon-bathy physics (KPP + convection) -- merge,
        # don't replace (same doctrine as the --mle / --iwm blocks above).
        # Unlike zdfiwm it is purely additive (no molecular-background
        # override) and has NO forcing file, so -- like --mle -- the model is
        # rebuilt here; the partial-cell / iwm rebuilds below re-use this same
        # config, preserving ddm.  Requires implicit vertical mixing (forced
        # True above via _ovr).
        if config.physics is None:
            raise ValueError(
                "--double-diffusion on latlon_bathy expected a physics config "
                "(KPP/convection) but config.physics is None.")
        _vm_ddm = config.physics.vertical_mixing._replace(ddm=ddm)
        config = config._replace(
            physics=config.physics._replace(vertical_mixing=_vm_ddm))
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        model = LatLonCGridOceanModel(grid, z_coord, config)
        print(f"[setup] latlon zdfddm ENABLED "
              f"(rn_avts={ddm.rn_avts:g} rn_hsbfr={ddm.rn_hsbfr:g})")
    e_mask, e_H = read_mesh_mask_bathy(mesh_path)
    ds = xr.open_dataset(mesh_path)
    src_lat = _squeeze2d(ds["gphit"].values)
    src_lon = _squeeze2d(ds["glamt"].values)
    tgt_lat = np.rad2deg(np.asarray(grid.lat))
    tgt_lon = np.rad2deg(np.asarray(grid.lon))
    H_ll, ocean_ll = regrid_curv_to_latlon(
        e_H, src_lat, src_lon, e_mask, tgt_lat, tgt_lon, max_deg=3.0,
    )
    land_mask = (ocean_ll > 0.5).astype(np.float64)
    # Mask the poorly-resolved semi-enclosed / endorheic seas BEFORE the
    # partial-cell + model build so the derived u/v face masks stay consistent.
    # On the regular lat-lon grid the IDW regrid reconnects basins through 1-cell
    # straits and inflates their depth; their brackish/hypersaline WOA T/S then
    # makes a sharp 1-cell front whose baroclinic PGF blows the cold-start at
    # high vertical resolution (diag 8486173: the Caspian, 47.5N/48E lev6,
    # max|u| 0.3->450 m/s in 2 steps). The same mechanism is why the cube masks
    # them. Caveated: these basins are excluded from the open-ocean comparison.
    if mask_marginal_seas:
        # tgt_lat/tgt_lon are 1-D (n_lat,)/(n_lon,) for the regular grid; the
        # box test needs 2-D fields matching land_mask (the cube passes 2-D).
        _lon2d, _lat2d = np.meshgrid(tgt_lon, tgt_lat)   # both (n_lat, n_lon)
        land_mask = _apply_marginal_sea_mask(land_mask, _lat2d, _lon2d)
    H_bathy = np.where(land_mask > 0.5, np.maximum(H_ll, 50.0), 0.0)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (topography removed -- PGF-over-topo control)")
    print(f"[setup] latlon {n_lat}x{n_lon}: ocean cells {int(land_mask.sum())}, "
          f"H_bathy [{H_bathy[land_mask>0.5].min():.0f},{H_bathy.max():.0f}] m")
    if partial_cell:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        z_coord, H_bathy, land_mask = make_partial_cell(
            z_coord, H_bathy, land_mask, smoothing_passes=bathy_smoothing_passes,
            min_levels=min_levels)
        model = LatLonCGridOceanModel(grid, z_coord, config)
    if iwm is not None and iwm.enabled:
        # FINAL model build with the zdfiwm maps (after every config /
        # z_coord rebuild above, so nothing downstream drops them).
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        _iwm_maps = None
        if iwm_forcing_file:
            from legoesm.ocean.iwm_forcing import load_iwm_forcing
            _iwm_maps = load_iwm_forcing(
                iwm_forcing_file, tgt_lat, tgt_lon, land_mask=land_mask)
        model = LatLonCGridOceanModel(grid, z_coord, config,
                                      iwm_forcing=_iwm_maps)
        print(f"[setup] latlon zdfiwm ENABLED "
              f"(maps={'file:' + iwm_forcing_file if iwm_forcing_file else 'uniform fallback'}, "
              f"mevar={iwm.mevar} tsdiff={iwm.tsdiff})")
    state = run_omip._init_rest_state(
        "latlon", grid, z_coord, H_max,
        H_bathy=jnp.asarray(H_bathy), land_mask=jnp.asarray(land_mask),
    )
    if woa_init:
        from legoesm.core.field import Field
        T_woa, S_woa = compute_woa_3d(grid, z_coord, woa_t, woa_s,
                                      H_bathy, land_mask)
        state = state._replace(
            T=Field(jnp.asarray(T_woa), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            S=Field(jnp.asarray(S_woa), name=state.S.name,
                    dims=state.S.dims, units=state.S.units),
        )
        print(f"[setup] latlon-bathy T/S initialised from WOA18 ({Path(woa_t).name})")
    return grid, z_coord, model, state, np.asarray(H_bathy)


def _apply_marginal_sea_mask(land_mask, lat_deg, lon_deg):
    """Set to LAND the poorly-resolved semi-enclosed marginal seas, whose narrow
    sills (e.g. Gibraltar) are sub-grid at coarse cube resolution -> a sharp 1-cell
    WOA density contrast -> an explosive cold-start PGF spike (the cube ignition
    sites). Mirrors the lat-lon Arctic caveat: these basins are excluded from the
    comparison so the open-ocean dynamics can run. ``lat_deg``/``lon_deg`` match
    ``land_mask`` shape; lon normalised to [0, 360)."""
    lat = np.asarray(lat_deg)
    lon = np.asarray(lon_deg) % 360.0
    out = np.asarray(land_mask, dtype=np.float64).copy()
    # (lat0, lat1, lon0, lon1) deg, lon in [0,360); lon0>lon1 means wrap over 0.
    boxes = [
        (30.0, 47.0, 353.0, 360.0),  # W Mediterranean (lon wrap part)
        (30.0, 47.0, 0.0, 37.0),     # Mediterranean (main)
        (40.0, 48.0, 27.0, 42.0),    # Black Sea
        (12.0, 30.0, 32.0, 44.0),    # Red Sea
        (23.0, 31.0, 47.0, 57.0),    # Persian Gulf
        (53.0, 66.0, 10.0, 30.0),    # Baltic
        (51.0, 64.0, 265.0, 285.0),  # Hudson Bay
        (34.0, 50.0, 45.0, 56.0),    # Caspian Sea (endorheic; IDW regrid
                                     # inflates its ~6 m depth to ~184 m and the
                                     # brackish WOA T/S makes a sharp 1-cell
                                     # front -> a baroclinic-PGF cold-start
                                     # blowup at 75-level, lat-lon 8486173. Box
                                     # padded to 50N/56E: the IDW spreads the
                                     # basin past its 47N/54E geographic edge
                                     # (ignition at 48.5N/51E, job 8486227).)
        (43.0, 48.0, 57.0, 62.0),    # Aral Sea (endorheic)
        (41.0, 49.0, 268.0, 285.0),  # Great Lakes (inland; no-op if WOA-land)
    ]
    n_before = int(out.sum())
    for lat0, lat1, lon0, lon1 in boxes:
        in_lat = (lat >= lat0) & (lat <= lat1)
        in_lon = (lon >= lon0) & (lon <= lon1)
        out = np.where(in_lat & in_lon, 0.0, out)
    print(f"[setup] marginal-sea mask: {n_before - int(out.sum())} cells -> land "
          f"(Med/Black/Red/Gulf/Baltic/Hudson; sub-grid sills, caveated)")
    return out


def _regrid_curv_to_points(field2d, src_lat_deg, src_lon_deg, ocean_mask,
                           tgt_lat_deg, tgt_lon_deg, k=4, max_deg=3.0):
    """IDW-regrid a curvilinear 2-D field (ocean cells only) onto ARBITRARY target
    points (any shape, e.g. cube (6,n,n)) using great-circle (chord) kNN. Mirrors
    ``compare_omip_nemo.regrid_curv_to_latlon`` but for point targets (that one
    meshgrids 1-D axes, so it can't take cube cell centres). Returns (values,
    ocean_flag) with the target's shape; ocean_flag=0 where the nearest source
    ocean cell is farther than ``max_deg``."""
    from scipy.spatial import cKDTree

    def _xyz(lat_r, lon_r):
        cl = np.cos(lat_r)
        return np.stack([cl * np.cos(lon_r), cl * np.sin(lon_r),
                         np.sin(lat_r)], axis=-1)

    m = np.asarray(ocean_mask).ravel() > 0.5
    if not m.any():
        raise ValueError("no ocean source cells")
    src_xyz = _xyz(np.deg2rad(np.asarray(src_lat_deg).ravel()[m]),
                   np.deg2rad(np.asarray(src_lon_deg).ravel()[m]))
    vals = np.asarray(field2d, dtype=np.float64).ravel()[m]
    tshape = np.asarray(tgt_lat_deg).shape
    tgt_xyz = _xyz(np.deg2rad(np.asarray(tgt_lat_deg).ravel()),
                   np.deg2rad(np.asarray(tgt_lon_deg).ravel()))
    tree = cKDTree(src_xyz)
    d, idx = tree.query(tgt_xyz, k=k)
    d = np.maximum(d, 1e-12)
    w = (1.0 / d) / (1.0 / d).sum(axis=1, keepdims=True)
    out = (vals[idx] * w).sum(axis=1).reshape(tshape)
    chord = 2.0 * np.sin(np.deg2rad(max_deg) / 2.0)
    ocean = (d[:, 0].reshape(tshape) < chord).astype(np.float64)
    return out, ocean


def build_cubed_sphere(nlev: int, H_max: float, mesh_path: str, n: int = 48,
                       woa_init: bool = False, woa_t=None, woa_s=None,
                       flat_bottom: bool = False, A_h=None, hyperdiff_coeff=None,
                       div_damp_2=None, div_damp_4=None, baroclinic_rk3=None,
                       mask_marginal_seas=False, dt=30.0,
                       velocity_ceiling=None, partial_cell=False,
                       pgf_scheme=None, bottom_drag_r=None,
                       bottom_drag_bbl_thickness=None,
                       bottom_drag_bg_velocity=None,
                       harmonic_cfl_safety=None,
                       bathy_smoothing_passes=0,
                       smc03_bottom_2nd_order=None,
                       barotropic_sw_div_damp_factor=None,
                       barotropic_sw_damp_v=None):
    """Build a cubed-sphere ocean (FV3 C-D grid baroclinic backend) with NEMO's
    OWN eORCA1 bathymetry/land-mask regridded onto the cube cell centres, for the
    faithful CORE-II comparison. The 3rd grid; reuses run_omip._create_setup (FC +
    fv3sw barotropic + face-edge-stability A_h/K_h) and the OMIP-2 applicator
    (grid_type='cubed_sphere'). NOTE: the cube OceanModel still has the documented
    PGF-over-bathy instability (docs/ocean/experiments/cubed_sphere_pgf_stability.md)
    that FC + elevated diffusion only delay; this builder is the harness to drive
    the dycore fix, not a finished faithful path."""
    from scripts.run import run_omip
    from legoesm.ocean.init import rest_state_ocean
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "cubed_sphere", f"C{n}", nlev, H_max, physics_preset="full",
        water_type="II",
    )
    # _create_setup builds the cube with physics=None (face-edge stability), so
    # model.step(surface_forcing=sf) would DROP the CORE-II forcing (the FC cube
    # path applies surface forcing only via physics_fn). Configure EXTERNAL surface
    # forcing so the coupler-provided tau/q_net are applied (atmosphere convention,
    # ocean reaction = -tau -- exactly what compute_omip2_surface_forcing returns).
    # compute_omip2_surface_forcing folds shortwave INTO q_net, and the external
    # scheme deposits the FULL q_net in the surface layer, so DISABLE shortwave
    # penetration (=None) to avoid double-counting solar.
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import (
        LateralMixingConfig, HarmonicConfig,
    )
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    # CRITICAL (iter-33): with external-forcing physics the dynamics-core viscosity
    # branch (`if physics_fn is None`) is SKIPPED — ALL momentum/tracer
    # mixing must come from physics_fn. The default HarmonicConfig A_h=1e4 +
    # enforce_cfl=False is ~4 orders too weak -> near-zero lateral momentum viscosity
    # -> the sharp marginal-sea front jet blows up. Use a STRONG CFL-CAPPED harmonic
    # (the cube analogue of the lat-lon smag-cfl-cap): A_h high, capped per cell at the
    # diffusive-CFL limit A_h*dt/dx^2 <= cfl_safety/4 -> maximal stable viscosity.
    # cfl_dt_estimate=dt so the cap matches the actual timestep.
    _harm_Ah = A_h if A_h is not None else 1.0e9
    # Diffusive-CFL cap safety (A_h·dt/dx² ≤ cfl_safety/4): default 0.20 is ~5×
    # below the forward-Euler stable max (1.0); tunable to probe whether a
    # stronger isotropic ceiling — the upper bound on what flow-adaptive
    # Smagorinsky could deliver at the unstable cell — holds the cold-start mode.
    _harm_cfl = harmonic_cfl_safety if harmonic_cfl_safety is not None else 0.20
    phys = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=LateralMixingConfig(
            scheme="harmonic",
            harmonic=HarmonicConfig(A_h=_harm_Ah, K_h=1.0e3, enforce_cfl=True,
                                    cfl_dt_estimate=float(dt), cfl_safety=_harm_cfl)),
        surface_forcing=SurfaceForcingConfig(scheme="external"),
        bottom_drag=BottomDragConfig(scheme="none"),  # drag via model config, not physics
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=None,
    )
    config = config._replace(physics=phys)
    # Optional momentum-dissipation overrides (debug the wind-stress-driven
    # grid-scale momentum instability on the cube A/CD-grid: the doc's FC-stable
    # cube was tested under thermal restoring only, NOT wind tau).
    _ovr = {k: v for k, v in (("A_h", A_h),
                              ("hyperdiff_coeff", hyperdiff_coeff),
                              ("div_damp_2", div_damp_2),
                              ("div_damp_4", div_damp_4),
                              ("baroclinic_rk3", baroclinic_rk3),
                              ("velocity_ceiling", velocity_ceiling),
                              ("pgf_scheme", pgf_scheme),
                              ("bottom_drag_r", bottom_drag_r),
                              ("bottom_drag_bbl_thickness",
                               bottom_drag_bbl_thickness),
                              ("bottom_drag_bg_velocity",
                               bottom_drag_bg_velocity),
                              ("smc03_bottom_2nd_order",
                               smc03_bottom_2nd_order),
                              ("barotropic_sw_div_damp_factor",
                               barotropic_sw_div_damp_factor),
                              ("barotropic_sw_damp_v",
                               barotropic_sw_damp_v)) if v is not None}
    if _ovr:
        config = config._replace(**_ovr)
        print(f"[setup] cube config override: {_ovr}")
    # NEMO bathy/mask -> cube cell centres (point-target IDW; the curvilinear
    # mesh is the same faithful geometry tripole/latlon use).  Built BEFORE the
    # model so partial cells can fold H_bathy into the vertical coordinate that
    # the model stores and steps with.
    import xarray as xr
    e_mask, e_H = read_mesh_mask_bathy(mesh_path)
    ds = xr.open_dataset(mesh_path)
    src_lat = _squeeze2d(ds["gphit"].values)
    src_lon = _squeeze2d(ds["glamt"].values)
    tgt_lat = np.rad2deg(np.asarray(grid.lat))   # (6, n, n)
    tgt_lon = np.rad2deg(np.asarray(grid.lon))
    H_cs, ocean_cs = _regrid_curv_to_points(
        e_H, src_lat, src_lon, e_mask, tgt_lat, tgt_lon, max_deg=3.0)
    land_mask = (ocean_cs > 0.5).astype(np.float64)
    if mask_marginal_seas:
        land_mask = _apply_marginal_sea_mask(land_mask, tgt_lat, tgt_lon)
    H_bathy = np.where(land_mask > 0.5, np.maximum(H_cs, 50.0), 0.0)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (cube; topography removed)")
    print(f"[setup] cubed_sphere C{n}: ocean cells {int(land_mask.sum())}/"
          f"{land_mask.size}, H_bathy [{H_bathy[land_mask>0.5].min():.0f},"
          f"{H_bathy.max():.0f}] m")
    # Bathymetry smoothing (cube-aware): the cube cold-start blowup is a spurious
    # partial-cell PGF residual at the steepest sub-grid topography (the under-
    # resolved Mediterranean at C32), forcing a basin-scale mode that no faithful
    # viscosity can damp (the grid-scale diffusive-CFL caps A_h below what a
    # 2-3-cell basin mode needs).  The residual scales with the bathymetric slope
    # (r-factor |H_i-H_j|/(H_i+H_j)), so a few Laplacian passes over the OCEAN
    # cells (land held fixed, seam-correct via is_cubed=True) shrink it directly —
    # the proven NEMO/ROMS technique for exactly this seed.
    if bathy_smoothing_passes and bathy_smoothing_passes > 0:
        from legoesm.ocean.bathymetry import laplacian_smooth_2d, compute_max_r_factor
        ocean = land_mask > 0.5
        r_before = float(compute_max_r_factor(H_bathy, land_mask))
        H_s = H_bathy.copy()
        for _ in range(int(bathy_smoothing_passes)):
            H_sm = np.asarray(laplacian_smooth_2d(H_s, 1, is_cubed=True))
            H_s = np.where(ocean, H_sm, H_bathy)
        H_bathy = np.where(ocean, np.maximum(H_s, 50.0), H_bathy)
        r_after = float(compute_max_r_factor(H_bathy, land_mask))
        print(f"[setup] cube bathymetry smoothing: {bathy_smoothing_passes} "
              f"Laplacian passes, max r-factor {r_before:.3f} -> {r_after:.3f}")
    # Partial bottom cells: fold the regridded bathymetry into the vertical
    # coordinate (Adcroft-Hill-Marshall 1997 / Adcroft-Campin 2004) instead of
    # the default pure-z* uniform stretch.  Reuses the canonical
    # ``make_partial_cell`` (thin-cell snap; defaults are cube-safe — no
    # bathy smoothing, min_levels=1).  Prerequisite for the smc03 PGF.  Both
    # backends now carry the partial-cell substrate: the FC A-grid (deprecated)
    # and the C-D grid (cd-grid, atmosphere-matching, the faithful target).
    if partial_cell:
        z_coord, H_bathy, land_mask = make_partial_cell(
            z_coord, H_bathy, land_mask,
        )
    # FV3 C-D grid baroclinic backend (the only cube ocean backend; the
    # deprecated FC-Gram A-grid was removed).
    model = OceanModel(grid, z_coord, config)
    print(f"[setup] cube backend: cd-grid (FV3 C-D)"
          f"{' + partial cells' if partial_cell else ''}")
    state = rest_state_ocean(grid, z_coord, H_max=H_max)
    state = state._replace(
        land_mask=state.land_mask.replace(data=jnp.asarray(land_mask)),
        H_bathy=state.H_bathy.replace(data=jnp.asarray(H_bathy)),
    )
    if woa_init:
        from legoesm.core.field import Field
        T_woa, S_woa = compute_woa_3d(grid, z_coord, woa_t, woa_s,
                                      H_bathy, land_mask)
        state = state._replace(
            T=Field(jnp.asarray(T_woa), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            S=Field(jnp.asarray(S_woa), name=state.S.name,
                    dims=state.S.dims, units=state.S.units),
        )
        print(f"[setup] cube T/S initialised from WOA18 ({Path(woa_t).name})")
    return grid, z_coord, model, state, np.asarray(H_bathy)


def build_mpas_ocean(nlev: int, H_max: float, mesh_path: str, level: int = 6,
                     lloyd_iterations: int = 20, woa_init: bool = False,
                     woa_t=None, woa_s=None, flat_bottom: bool = False,
                     A_h=None, B_h=None, K_bih=None, C_smag_lap=None,
                     pgf_scheme=None, tracer_advection=None, bottom_drag_r=None,
                     bottom_drag_bbl_thickness=None, bottom_drag_bg_velocity=None,
                     partial_cell=False, dz_ref_override=None,
                     n_barotropic_substeps=None,
                     barotropic_solver=None, barotropic_pcg_variant=None,
                     freeze_floor=None, freezing=None,
                     runoff_depth_spread_m=None, mle=None,
                     bottom_drag_scheme=None, bottom_drag_cd0=None,
                     bottom_drag_cdmax=None, bottom_drag_z0=None,
                     bottom_drag_ke0=None, iwm=None, ddm=None,
                     vertical_mixing=None, ew_cyclic_overlap=False):
    """Build an MPAS (icosahedral Voronoi) ocean for the faithful CORE-II NEMO
    comparison — the 4th grid.  Reuses ``run_omip._create_setup('mpas', ...)``
    (the wired MPASOceanModel: KPP + GM/Redi + smc03 PGF + implicit-CN
    barotropic + bottom drag + Smagorinsky), then switches surface forcing to the
    faithful EXTERNAL contract (CORE-II tau/q_net via ``model.step(surface_forcing
    =compute_omip2_surface_forcing(...))``), regrids NEMO eORCA1 bathymetry onto
    the Voronoi cell centres, and (optionally) folds partial cells + WOA IC.

    Unlike the cube (parked, fixed C-resolution), the Voronoi mesh resolution is
    a FREE parameter (``level``: nCells = 10*4^level + 2 → ico5 ~230 km, ico6
    ~115 km ≈ ORCA1, ico7 ~58 km), so MPAS is not inherently resolution-limited.
    Returns ``(mesh, z_coord, model, state, H_bathy)`` — same tuple as the other
    builders.
    """
    from scripts.run import run_omip
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.core.field import Field

    mesh, z_coord, config, _model0, _ = run_omip._create_setup(
        "mpas", f"ico{level}", nlev, H_max,
        physics_preset="full", water_type="II",
        dz_ref_override=dz_ref_override,
        # KPP boundary-layer-depth sensitivity override (Ri_crit / Cv).  None
        # -> _create_setup builds the default KPPConfig (byte-identical to the
        # pre-flag runs); a custom VerticalMixingConfig deepens/shoals the
        # diagnosed-too-shallow subtropical mixed layer (MLD ~half of NEMO).
        vertical_mixing=vertical_mixing,
    )
    # Faithful EXTERNAL surface-forcing contract: the CORE-II tau/q_net from
    # compute_omip2_surface_forcing is deposited by mpas_physics (it negates +
    # edge-projects tau and does Jerlov SW penetration), so q_net already folds
    # shortwave in → disable shortwave_penetration to avoid double-counting solar
    # (mirrors build_cubed_sphere).
    phys = config.physics._replace(
        surface_forcing=SurfaceForcingConfig(scheme="external"),
        shortwave_penetration=None,
    )
    config = config._replace(physics=phys)
    if vertical_mixing is not None and vertical_mixing.scheme == "tke":
        # NEMO zdftke card on MPAS: mirror the TRIPOLE closure environment
        # exactly (codex HIGH x2), or the "same closure across grids" A/B is
        # confounded:
        #  * Backgrounds -> molecular.  NEMO composes avm=max(closure, avmb)
        #    with the closure-internal rn_avm0/rn_avt0 floors (which the card
        #    carries as kappaM_min/kappaH_min); the MPAS model-level A_v/K_v
        #    (1e-4/1e-5) are ADDED on top of the closure profiles
        #    (ocean_model_mpas K_v_cell = background + K_v_tke), so keeping
        #    them double-counts the background — force molecular exactly as
        #    the tripole build does (zdfiwm_init values: rnu=1.4e-6, 1e-10).
        #  * Convection -> none.  The "full" preset ships enhanced-diffusion
        #    convection (K_conv up to 1) that the tripole card does NOT run;
        #    under TKE the closure itself convects (kappaM_max ceiling), so
        #    the preset EVD would be a second, tripole-absent mixer.
        from legoesm import constants as _const
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig,
        )
        phys_tke = config.physics._replace(
            convection=OceanConvectionConfig(scheme="none"),
        )
        config = config._replace(
            physics=phys_tke,
            A_v=_const.nu_ocean_molecular,   # same pair the tripole forces
            K_v=1.0e-10)                     # NEMO avtb

        print("[setup] mpas TKE card: molecular backgrounds "
              "(A_v=1.4e-6, K_v=1e-10) + preset convection stripped "
              "(tripole-equivalent closure environment)")
    if iwm is not None and iwm.enabled:
        raise SystemExit(
            "--iwm is not wired on the MPAS vertical-mixing bridge yet "
            "(lat-lon / tripole only)")
    if ddm is not None and ddm.enabled:
        raise SystemExit(
            "--double-diffusion is not wired on the MPAS vertical-mixing "
            "bridge yet (lat-lon / tripole only)")
    _ovr = {k: v for k, v in (("A_h", A_h), ("B_h", B_h), ("K_bih", K_bih),
                              ("C_smag_lap", C_smag_lap), ("pgf_scheme", pgf_scheme),
                              ("tracer_advection", tracer_advection),
                              ("bottom_drag_r", bottom_drag_r),
                              ("bottom_drag_bbl_thickness", bottom_drag_bbl_thickness),
                              ("bottom_drag_bg_velocity", bottom_drag_bg_velocity),
                              ("bottom_drag_scheme", bottom_drag_scheme),
                              ("bottom_drag_cd0", bottom_drag_cd0),
                              ("bottom_drag_cdmax", bottom_drag_cdmax),
                              ("bottom_drag_z0", bottom_drag_z0),
                              ("bottom_drag_ke0", bottom_drag_ke0),
                              ("n_barotropic_substeps", n_barotropic_substeps),
                              ("barotropic_solver", barotropic_solver),
                              ("barotropic_implicit_pcg_variant",
                               barotropic_pcg_variant),
                              ("freeze_floor", freeze_floor),
                              ("freezing", freezing),
                              ("runoff_depth_spread_m", runoff_depth_spread_m))
            if v is not None}
    if _ovr:
        config = config._replace(**_ovr)
        print(f"[setup] mpas config override: {_ovr}")
    if mle is not None:
        # Fox-Kemper MLE on the Voronoi mesh (NEMO nn_mle=1 bolus restratification).
        config = config._replace(mle=mle)
        print(f"[setup] mpas Fox-Kemper MLE enabled: ce={mle.ce:g}")

    # NEMO eORCA1 bathy/mask -> Voronoi cell centres (point-target IDW, the same
    # faithful geometry tripole/latlon/cube use).
    import xarray as xr
    e_mask, e_H = read_mesh_mask_bathy(mesh_path)
    # ORCA 2-pt cyclic-overlap fill of the eORCA source mask/bathy BEFORE the
    # Voronoi regrid + NN land/sea lookup.  The eORCA mask halo columns are
    # INCONSISTENT with their interior partners (verified: |col0 - col[nx-2]| = 1.0
    # wet/dry mismatch at lon 72.5/73.5E), so the IDW/NN near 72.5E blends a
    # wet-vs-land mismatch -> a spurious ~73E SST/SSS stripe on the MPAS maps
    # (Voronoi has no intrinsic seam; it is IMPORTED from this inconsistent source).
    # Reuses the same _ew_overlap_fill the tripole path applies; gated by
    # --ew-cyclic-overlap (matches tripole's opt-in; default off = byte-identical).
    if ew_cyclic_overlap:
        e_mask = _ew_overlap_fill(np.asarray(e_mask))
        e_H = _ew_overlap_fill(np.asarray(e_H))
    ds = xr.open_dataset(mesh_path)
    src_lat = _squeeze2d(ds["gphit"].values)
    src_lon = _squeeze2d(ds["glamt"].values)
    tgt_lat = np.rad2deg(np.asarray(mesh.latCell))   # (nCells,)
    tgt_lon = np.rad2deg(np.asarray(mesh.lonCell))
    H_pts, _ = _regrid_curv_to_points(
        e_H, src_lat, src_lon, e_mask, tgt_lat, tgt_lon, max_deg=3.0)
    # Land/sea by the NEAREST NEMO source cell's ACTUAL mask (exact NN over ALL
    # source cells), NOT _regrid_curv_to_points' ocean_flag (which wets any
    # target within max_deg of an OCEAN cell -> over-wets coastlines at fine ico
    # resolution, borrowing bathy onto continental cells and contaminating the
    # score).  IDW (H_pts) is used only for the depth of cells classified wet.
    from scipy.spatial import cKDTree as _cKDTree
    def _xyz_deg(latd, lond):
        lr = np.deg2rad(np.asarray(latd).ravel())
        orr = np.deg2rad(np.asarray(lond).ravel())
        cl = np.cos(lr)
        return np.stack([cl * np.cos(orr), cl * np.sin(orr), np.sin(lr)], axis=-1)
    _, _nn = _cKDTree(_xyz_deg(src_lat, src_lon)).query(_xyz_deg(tgt_lat, tgt_lon), k=1)
    land_mask = (np.asarray(e_mask).ravel()[_nn] > 0.5).astype(np.float64)
    H_bathy = np.where(land_mask > 0.5, np.maximum(H_pts, 50.0), 0.0)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (mpas; topography removed)")
    print(f"[setup] mpas ico{level}: ocean cells {int(land_mask.sum())}/"
          f"{land_mask.size}, H_bathy [{H_bathy[land_mask>0.5].min():.0f},"
          f"{H_bathy.max():.0f}] m")
    if partial_cell:
        # min_levels=1 only: the make_partial_cell min_levels>1 path assumes a
        # 2-D leading axis; MPAS cells are 1-D (nCells,).
        z_coord, H_bathy, land_mask = make_partial_cell(z_coord, H_bathy, land_mask)

    model = MPASOceanModel(mesh, z_coord, config)
    print(f"[setup] mpas backend: Voronoi (TRiSK)"
          f"{' + partial cells' if partial_cell else ''}")
    state = rest_state_mpas_ocean(mesh, z_coord, H_max=H_max)
    state = state._replace(
        land_mask=state.land_mask.replace(data=jnp.asarray(land_mask)),
        H_bathy=state.H_bathy.replace(data=jnp.asarray(H_bathy)),
    )
    if woa_init:
        T_woa, S_woa = compute_woa_3d(mesh, z_coord, woa_t, woa_s,
                                      H_bathy, land_mask)
        assert np.asarray(T_woa).shape == tuple(state.T.data.shape), (
            f"WOA shape {np.asarray(T_woa).shape} != state.T {state.T.data.shape}")
        state = state._replace(
            T=Field(jnp.asarray(T_woa), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            S=Field(jnp.asarray(S_woa), name=state.S.name,
                    dims=state.S.dims, units=state.S.units),
        )
        print(f"[setup] mpas T/S initialised from WOA18 ({Path(woa_t).name})")
    return mesh, z_coord, model, state, np.asarray(H_bathy)


_RUNOFF_NC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
              "INPUTS/orca1_inputs/data_repository/input_fields/"
              "runoff-icb_DaiTrenberth_Depoorter.nc")

# NEMO ORCA1 RUN_REF sea-ice diagnostics (SI3): annual-mean `siconc` used as the
# prescribed sea-ice concentration for the SW-albedo surrogate (--ice-albedo).
_SICONC_NC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
              "EXP00/RUN_REF/ORCA1_1y_20000101_20041231_icemod.nc")
# Monthly ESACCI/BIOMER chlorophyll climatology (mg/m^3) on a 0.5deg regular grid,
# the input NEMO ORCA1 reads for ln_qsr_rgb/nn_chldta=1 (--sw-rgb-chl).  CHLA has
# shape (12, 361, 721) with 2D nav_lat/nav_lon, regridded like siconc/runoff.
_CHL_NC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
           "INPUTS/orca1_inputs/data_repository/input_fields/"
           "merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc")
# NEMO ORCA1 domain_cfg: source of the reference 75-level column (e3t_1d) for
# --nemo-vertical, so legoESM matches NEMO's vertical resolution.
_NEMO_DOMAIN_CFG = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/"
                    "ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/"
                    "domain_cfg.nc")


def _load_nemo_e3t_1d(path: str):
    """Read NEMO's 1-D reference layer thicknesses ``e3t_1d`` [m] (length jpk).

    Falls back to differencing ``gdepw_1d`` (interface depths) if ``e3t_1d`` is
    absent.  Returns a 1-D float64 NumPy array (top -> bottom), all > 0."""
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    if "e3t_1d" in ds:
        dz = np.asarray(ds["e3t_1d"].values, dtype=np.float64).ravel()
    elif "gdepw_1d" in ds:
        w = np.asarray(ds["gdepw_1d"].values, dtype=np.float64).ravel()
        dz = np.diff(np.concatenate([w, w[-1:] + (w[-1] - w[-2])]))
    else:
        raise KeyError(f"{path}: no e3t_1d or gdepw_1d for --nemo-vertical")
    if dz.ndim != 1 or dz.size < 2 or not np.all(dz > 0):
        raise ValueError(f"{path}: bad e3t_1d (shape {dz.shape}, must be 1-D >0)")
    return dz
# NEMO ORCA1 RUN_REF MONTHLY ocean grid_T (`tos` = SST [degC]) -> used to give the
# annual-mean siconc a SEASONAL cycle (--ice-albedo-seasonal): NEMO sea ice sits
# at the freezing point, so where the monthly SST is at/below freezing NEMO has
# ice, and where it warms above freezing the ice (and its albedo) is gone.  The
# run wrote only an ANNUAL icemod (no monthly siconc), so the monthly SST is the
# faithful seasonal proxy available without a NEMO re-run.
_TOS_MONTHLY_NC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/"
                   "ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc")


def _load_nemo_cell_area_m2(path=_NEMO_DOMAIN_CFG):
    """NEMO ORCA1 horizontal T-cell area ``e1t*e2t`` [m^2] from ``domain_cfg``
    (shape ``(jpj, jpi)`` == the Dai-Trenberth runoff grid).  This is the EXACT
    NEMO metric used to area-weight the runoff SOURCE integral for conservation.
    Deliberately NOT a gradient-of-lat/lon approximation: a centred ``np.gradient``
    of longitude across the +/-180 seam sees an O(360 deg) jump that wrapping the
    averaged derivative cannot undo, inflating dateline river-cell areas and the
    source total to ~2.95 Sv vs NEMO's true ~1.27 Sv (codex review).  The TARGET
    grids carry their own exact ``.area``/``.areaCell``."""
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    e1t = _squeeze2d(np.asarray(ds["e1t"].values, dtype=np.float64))
    e2t = _squeeze2d(np.asarray(ds["e2t"].values, dtype=np.float64))
    return e1t * e2t


def _area_conservative_scale(field, cell_area, wet_mask, target_integral):
    """Scale a per-area flux ``field`` [X/m^2] so its area-integral over the wet
    cells equals ``target_integral`` [X]: returns ``field * target/current``.
    No-op (returns ``field`` unchanged) when the current integral is <= 0 (an
    all-zero / all-dry field has nothing to scale)."""
    field = np.asarray(field, dtype=np.float64)
    cur = float((field * np.asarray(cell_area) * np.asarray(wet_mask)).sum())
    if cur > 0.0:
        return field * (float(target_integral) / cur)
    return field


# --- KPP boundary-layer-depth sensitivity (vmix structural-residual lever) ---
# Diagnosis (2026-06-22): the OMIP subtropical mixed layer is ~half of NEMO
# (MLD bias -35 m NH-subtropics, model-wide MPAS == tripole, NOT spin-up), so
# surface heat + freshwater fluxes are trapped in a too-thin top layer -> the
# subtropical surface warm+fresh bias.  KPP sets h_bl where the bulk Richardson
# number Ri_b crosses ``Ri_crit``; deepening the layer is a one-knob lever via
# Ri_crit (the threshold) or Cv (the unresolved-shear V_t^2 coefficient).  These
# helpers expose the two knobs WITHOUT mutating the production KPPConfig default.
#
# Accepted ranges (reject parameter-abuse runs that collapse the experiment):
#   Ri_crit -> the KPPConfig ``__param_spec__`` tunable-tier-2 bounds
#              (Large et al. 1994); outside this the BL diagnosis is unphysical.
#   Cv      -> Cv is NOMINALLY a FIXED LMD94 constant (1.6, no tunable bound in
#              the spec); this band is an EXPLICIT experimental sensitivity
#              range, intentionally wider, not a tuning tier.
_KPP_RI_CRIT_RANGE = (0.099, 0.9)
_KPP_CV_RANGE = (0.5, 5.0)

# NEMO rn_vfac (ln_crt_dwn current feedback): fraction of the surface current
# subtracted from the wind before the bulk. NEMO bound is [0, 1] (0 = absolute
# wind = default; 1 = full feedback).
_WIND_VFAC_RANGE = (0.0, 1.0)


def _resolve_wind_vfac(relative_winds: bool, wind_vfac):
    """Resolve the NEMO ``rn_vfac`` current-feedback fraction from the CLI flags.

    ``--wind-vfac X`` sets it explicitly; ``--relative-winds`` is the shorthand
    for the NEMO ``ln_crt_dwn`` default ``rn_vfac = 1.0``.  Passing BOTH is only
    accepted when they agree (``--wind-vfac 1.0``); a contradictory pair is a
    user error and raises (never silently pick one -- the dispatch-hardening
    rule).  Returns ``0.0`` (absolute wind, the byte-identical default) when
    neither is given.  Validated finite and within ``_WIND_VFAC_RANGE``."""
    if wind_vfac is None:
        vfac = 1.0 if relative_winds else 0.0
    else:
        vfac = float(wind_vfac)
        if relative_winds and vfac != 1.0:
            raise SystemExit(
                f"--relative-winds (rn_vfac=1.0) conflicts with --wind-vfac "
                f"{wind_vfac}; pass only one (or --wind-vfac 1.0).")
    lo, hi = _WIND_VFAC_RANGE
    if not (np.isfinite(vfac) and lo <= vfac <= hi):
        raise SystemExit(
            f"--wind-vfac must be finite and within [{lo}, {hi}] (NEMO rn_vfac "
            f"current-feedback fraction); got {vfac!r}.")
    return vfac


def _kpp_vmix_override(kpp_ri_crit=None, kpp_cv=None, kpp_eice=None):
    """Build a KPP ``VerticalMixingConfig`` overriding ONLY the CLI-set knobs.

    Returns ``None`` when neither knob is given so the caller falls through to
    ``_create_setup``'s default ``VerticalMixingConfig(scheme="kpp")`` — i.e.
    byte-for-byte the pre-flag config (no silent re-defaulting of the other
    KPP fields).  ``Ri_crit`` / ``Cv`` must be finite and within their accepted
    range (see ``_KPP_RI_CRIT_RANGE`` / ``_KPP_CV_RANGE``)."""
    if kpp_ri_crit is None and kpp_cv is None and kpp_eice is None:
        return None
    from legoesm.ocean.physics.vertical_mixing.config import (
        KPPConfig, VerticalMixingConfig,
    )

    def _check(name, val, rng):
        lo, hi = rng
        if not (np.isfinite(val) and lo <= val <= hi):
            raise ValueError(
                f"--{name} must be finite and within [{lo}, {hi}] "
                f"(physical KPP boundary-layer range); got {val!r}.")

    kpp = KPPConfig()
    if kpp_ri_crit is not None:
        _check("kpp-ri-crit", kpp_ri_crit, _KPP_RI_CRIT_RANGE)
        kpp = kpp._replace(Ri_crit=float(kpp_ri_crit))
    if kpp_cv is not None:
        _check("kpp-cv", kpp_cv, _KPP_CV_RANGE)
        kpp = kpp._replace(Cv=float(kpp_cv))
    if kpp_eice is not None:
        if int(kpp_eice) not in (0, 1, 3):
            raise ValueError(
                f"--kpp-eice must be 0 (off), 1 (legoESM linear 1-fi) or 3 "
                f"(max(0,1-4*fi), matches NEMO nn_eice=3); got {kpp_eice!r}.")
        kpp = kpp._replace(eice=int(kpp_eice))
    return VerticalMixingConfig(scheme="kpp", kpp=kpp)


def _validate_kpp_grid(grid, kpp_ri_crit=None, kpp_cv=None, kpp_eice=None,
                       mpas_vmix="kpp"):
    """Reject the KPP override flags on grids whose CORE-II builder does not
    thread ``vertical_mixing`` INTO A LIVE KPP scheme (a flag that silently does
    nothing is the dispatch footgun CLAUDE.md forbids).  ``mpas`` and
    ``latlon_bathy`` run KPP (``bathy_physics.vertical_mixing``) so the override
    reaches ``KPPConfig.Ri_crit``/``Cv``.  ``tripole`` ships ``physics=None``
    (the dynamics-core implicit vertical solve, NO KPP boundary layer) so a KPP
    override would be a silent no-op there -> still rejected; ``cubed_sphere``
    is not wired.  Extend this set only when the builder actually runs KPP."""
    if (kpp_ri_crit is not None or kpp_cv is not None
            or kpp_eice is not None) and grid not in (
            "mpas", "latlon_bathy"):
        raise SystemExit(
            f"--kpp-ri-crit/--kpp-cv/--kpp-eice are wired for --grid mpas/latlon_bathy "
            f"(grids that run the KPP boundary layer), not --grid {grid!r}. "
            f"tripole runs the dynamics-core implicit vertical solve (no KPP) "
            f"so the override would silently do nothing; its opt-in closure "
            f"is selected by --tripole-vmix (tke/kpp at scheme defaults, "
            f"no Ri_crit/Cv knobs).")
    if (kpp_ri_crit is not None or kpp_cv is not None
            or kpp_eice is not None) and grid == "mpas" and mpas_vmix == "tke":
        raise SystemExit(
            "--kpp-ri-crit/--kpp-cv/--kpp-eice configure the KPP closure, but "
            "--mpas-vmix tke runs the NEMO zdftke card on MPAS — the KPP "
            "overrides would silently do nothing there. Drop them (TKE's "
            "under-ice lever is --tke-eice on the tripole; the MPAS TKE "
            "bridge sets eice via the card).")
    if kpp_eice is not None and grid == "mpas":
        raise SystemExit(
            "--kpp-eice is wired for --grid latlon_bathy only: the MPAS KPP "
            "vertical-mixing bridge receives no ice concentration yet "
            "(mpas_physics passes tau/q only), so the attenuation would "
            "silently no-op there. Run the under-ice KPP lever on latlon.")


def _validate_tke_card_grid(grid, tripole_vmix="none", tke_eice=None,
                            tke_surface_bc=None, tke_mxl_choice=None,
                            tke_prognostic=None, tke_kappa_convention=None,
                            tke_shear_production=None,
                            tke_n2_mode=None, tke_n2_eos_form=None,
                            tke_lc=None, tke_etau=None, mpas_vmix="kpp"):
    """Reject the zdftke card knobs unless the tke closure is active.

    ``--tke-eice`` / ``--tke-surface-bc`` / ``--tke-mxl-choice`` are applied
    ONLY inside the ``tke``
    branch of ``build_tripole_vmix_config`` (-> ``orca1_zdftke_config``).  Two
    grids reach that branch: the tripole attach block under ``--tripole-vmix
    tke``, and the MPAS builder under ``--mpas-vmix tke`` (the "tripole\\_"
    prefix on the builder is historical; the construction is grid-agnostic and
    the MPAS TKE bridge consumes the same ``TKEConfig``).  Everywhere else the
    knobs are SILENTLY DISCARDED — the dispatch footgun CLAUDE.md forbids:
      * ``--grid latlon_bathy`` / ``cubed_sphere`` (different builder);
      * ``--grid tripole --tripole-vmix none`` (``_use_vmix`` false ->
        build_tripole_vmix_config is never called at all);
      * ``--grid tripole --tripole-vmix kpp`` and ``--grid mpas --mpas-vmix
        kpp`` (the knob is not applied in the kpp branch).
    So require one of the two FULL tke contexts whenever a knob is set
    (codex 2026-07-22 HIGH).

    The MPAS context was opened on 2026-08-22 so the three grids can run the
    SAME closure card: the surface-TKE Dirichlet boundary condition is the
    largest single lever this campaign has measured, and while it was
    tripole-only every cross-grid comparison was made between a tripole
    running the faithful card and an MPAS running a different one.  The MPAS
    TKE bridge rejects, loudly and at factory-build time, the handful of
    ``TKEConfig`` options it does not plumb (``bottom_tke_bc``,
    ``n2_mode='adiabatic'``, ``veros_dz_slots``, ``buoyancy_timing``,
    ``advection_scheme``, ``source_eke_diss``, and ``iwm``), so a knob that
    reaches it either takes effect or raises.
    """
    for _flag, _val in (("--tke-eice", tke_eice),
                        ("--tke-surface-bc", tke_surface_bc),
                        ("--tke-mxl-choice", tke_mxl_choice),
                        ("--tke-lc", tke_lc),
                        ("--tke-etau", tke_etau),
                        ("--tke-n2-mode", tke_n2_mode),
                        ("--tke-n2-eos-form", tke_n2_eos_form),
                        ("--tke-prognostic", tke_prognostic),
                        ("--tke-kappa-convention", tke_kappa_convention),
                    ("--tke-shear-production", tke_shear_production)):
        _active = ((grid == "tripole" and tripole_vmix == "tke")
                   or (grid == "mpas" and mpas_vmix == "tke"))
        if _val is not None and not _active:
            raise SystemExit(
                f"{_flag} configures the zdftke closure and takes effect ONLY "
                "under --grid tripole --tripole-vmix tke or --grid mpas "
                f"--mpas-vmix tke; got --grid {grid!r} --tripole-vmix "
                f"{tripole_vmix!r} --mpas-vmix {mpas_vmix!r}, where it is "
                f"silently discarded. Add the matching --tripole-vmix tke / "
                f"--mpas-vmix tke, or drop {_flag}.")


def _validate_pcg_variant_grid(grid, barotropic_pcg_variant=None,
                               barotropic_solver=None):
    """Reject ``--barotropic-pcg-variant`` on configurations where no PCG
    runs (a flag that silently does nothing is the dispatch footgun
    CLAUDE.md forbids).  ``tripole``/``latlon_bathy``
    (LatLonCGridOceanConfig.barotropic.barotropic_implicit_pcg_variant) and
    ``mpas`` (MPAS config field) dispatch it at the solver entry;
    ``cubed_sphere`` runs the FV3 split-explicit subcycle — no PCG at all.
    An EXPLICIT ``--barotropic-solver explicit_substep`` override also
    bypasses the PCG (codex r1 #4: a single-reduce scaling run with the
    explicit solver would silently measure nothing)."""
    if barotropic_pcg_variant is None:
        return
    if grid not in ("tripole", "latlon_bathy", "mpas"):
        raise SystemExit(
            f"--barotropic-pcg-variant is wired for --grid tripole/"
            f"latlon_bathy/mpas (implicit-CN PCG grids), not --grid "
            f"{grid!r} (the cube's split-explicit subcycle has no PCG, so "
            f"the flag would silently do nothing).")
    if barotropic_solver == "explicit_substep":
        raise SystemExit(
            "--barotropic-pcg-variant selects the implicit-CN PCG "
            "reduction strategy, but --barotropic-solver explicit_substep "
            "runs no PCG — the flag would silently do nothing.  Drop one "
            "of the two.")


def _runoff_component_vars(exclude_isf: bool):
    """Dai-Trenberth NetCDF freshwater components summed by
    :func:`load_runoff_monthly`: rivers (``sorunoff``) + icebergs (``Icb_flux``)
    + ice-shelf melt (``sornfisf``).

    ``exclude_isf=True`` DROPS ``sornfisf``: when ``--isf`` is on, the SAME
    file's ice-shelf melt is separately deposited at depth over the
    [zmin, zmax] cavity band (``apply_isf_prescribed_melt_step``), so also
    summing it into the surface runoff would DOUBLE-COUNT the ice-shelf
    freshwater (codex).  The runner passes ``exclude_isf=args.isf``.
    """
    if exclude_isf:
        return ("sorunoff", "Icb_flux")
    return ("sorunoff", "sornfisf", "Icb_flux")


def load_runoff_monthly(grid, grid_type, lat2d_deg, lon2d_deg, mesh_path,
                        land_mask=None, spread_passes=2, exclude_isf=False):
    """Load NEMO's Dai-Trenberth runoff (the SAME file NEMO ORCA1 uses) and regrid
    each climatological month onto the model grid. Total freshwater = rivers
    (sorunoff) + ice-shelf melt (sornfisf) + icebergs (Icb_flux) [kg/m²/s, +INTO
    ocean]; ``exclude_isf=True`` drops sornfisf because --isf applies it at depth
    (see :func:`_runoff_component_vars`). Returns (12, *lat2d_deg.shape). Ungates
    the SSS comparison (runoff=0 made SSS only informational). Curvilinear ->
    model grid via the same IDW used for bathy; eORCA1 nav_lat/lon are the runoff
    file's own coords."""
    import xarray as xr
    from legoesm.ocean.bathymetry import (
        laplacian_smooth_2d, laplacian_smooth_voronoi)
    ds = xr.open_dataset(_RUNOFF_NC, decode_times=False)
    src_lat = _squeeze2d(ds["nav_lat"].values)
    src_lon = _squeeze2d(ds["nav_lon"].values)
    components = _runoff_component_vars(exclude_isf)
    total = np.zeros_like(np.asarray(ds["sorunoff"].values), dtype=np.float64)
    for v in components:
        if v in ds:
            total = total + np.nan_to_num(np.asarray(ds[v].values, dtype=np.float64))
    # SOURCE = the DISCHARGE cells only (annual runoff > 0): a coastal river-mouth
    # field is sparse, so IDW from ALL cells (incl. zeros) would dilute the discharge
    # to ~0. Routing only from nonzero cells spreads each river to the nearest model
    # coastal cells (codex HIGH). Exact area-integral conservation is enforced
    # below by the area-weighted renorm (every grid receives the same source total).
    annual = total.sum(axis=0)
    src_valid = annual > 0.0
    out = np.zeros((12,) + tuple(np.asarray(lat2d_deg).shape), dtype=np.float64)
    is_cubed = (np.asarray(lat2d_deg).ndim == 3)
    ocean = None
    if land_mask is not None:
        ocean = np.asarray(land_mask) > 0.5
    # Area-conservation inputs: SOURCE cell areas = NEMO's exact e1t*e2t from
    # domain_cfg (the runoff grid's metric), TARGET cell areas from the model grid
    # (``areaCell`` on the MPAS VoronoiMesh, ``.area`` on the C-grid families).
    # Both [m^2]; A_tgt.shape == lat2d_deg.shape == out[m].shape.
    A_src = _load_nemo_cell_area_m2()
    if A_src.shape != total.shape[1:]:
        raise ValueError(
            f"runoff source area {A_src.shape} != runoff field {total.shape[1:]}: "
            f"domain_cfg e1t/e2t must match the Dai-Trenberth grid")
    A_tgt = np.asarray(grid.areaCell if hasattr(grid, "areaCell") else grid.area)
    for m in range(12):
        # k=4 (NOT k=1: _regrid_curv_to_points assumes 2-D kNN -> k=1 crashes, codex HIGH)
        Rm, _ = _regrid_curv_to_points(
            total[m], src_lat, src_lon, src_valid,
            lat2d_deg, lon2d_deg, k=4, max_deg=2.0)
        Rm = np.maximum(Rm, 0.0)
        # COASTAL SPREAD (codex conservation flag + SSS-quality): the NN/IDW
        # regrid concentrates each river in ~1 model cell -> over-fresh spots that
        # hurt SSS. Spread over a coastal band via ocean-masked averaging, then
        # the area-conservative renorm below restores the exact source total.
        # MPAS uses the Voronoi-topology smoother (cellsOnCell neighbour-average);
        # the structured laplacian_smooth_2d does NOT apply to an unstructured mesh
        # (the reason MPAS was previously left UN-spread -> big rivers like the
        # Amazon/Arctic over-concentrated in the ~4 IDW cells -> local -2 to -3 PSU
        # over-freshening; lat-lon/cube were smoothed but MPAS was not).
        if ocean is not None and spread_passes > 0:
            _is_voronoi = (grid_type == "mpas")
            _coc = np.asarray(grid.cellsOnCell) if _is_voronoi else None
            _nec = np.asarray(grid.nEdgesOnCell) if _is_voronoi else None
            for _ in range(int(spread_passes)):
                if _is_voronoi:
                    sm = np.asarray(laplacian_smooth_voronoi(Rm, _coc, _nec, 1))
                else:
                    sm = np.asarray(laplacian_smooth_2d(Rm, 1, is_cubed=is_cubed))
                Rm = np.where(ocean, sm, 0.0)
        # AREA-CONSERVATIVE renorm (replaces the old cell-SUM renorm and now runs
        # on EVERY grid incl. MPAS spread_passes=0): scale the regridded runoff so
        # its area-integral equals the source month total [kg/s], i.e. NEMO's
        # Dai-Trenberth global freshwater input.  This makes MPAS and tripole
        # receive the SAME total -> removes the grid-dependent surface fresh bias
        # (MPAS was -0.50 PSU vs tripole -0.12 from a non-conservative regrid).
        F_src = float((total[m] * A_src).sum())            # kg/s into ocean
        wet = ocean if ocean is not None else np.ones(Rm.shape, dtype=bool)
        if F_src > 0.0 and float((Rm * A_tgt * wet).sum()) <= 0.0:
            warnings.warn(                                 # codex LOW: don't silently skip
                f"runoff month {m}: source {F_src:.3e} kg/s but the regridded "
                f"target integral is <=0 (no wet target cell received runoff -- "
                f"check land_mask / IDW max_deg) -> month NOT conserved",
                RuntimeWarning)
        Rm = _area_conservative_scale(Rm, A_tgt, wet, F_src)
        out[m] = Rm
    # Annual-mean conserved total on BOTH the source and the (renormed) target,
    # in Sv of freshwater (1 Sv = 1e9 kg/s) -> they should match to ~rounding,
    # and be grid-INDEPENDENT (the whole point of the renorm).
    _wet = ocean if ocean is not None else np.ones(out.shape[1:], dtype=bool)
    _src_Sv = float((total.mean(axis=0) * A_src).sum()) / 1.0e9
    _tgt_Sv = float((out.mean(axis=0) * A_tgt * _wet).sum()) / 1.0e9
    _comp_label = "+".join(components) + (
        " (sornfisf EXCLUDED: --isf applies ice-shelf melt at depth)"
        if exclude_isf else "")
    print(f"[setup] runoff: Dai-Trenberth ({_comp_label}) from {int(src_valid.sum())} "
          f"discharge cells, 12 months, {spread_passes} spread passes, "
          f"max {out.max():.2e} kg/m^2/s | conserved total src={_src_Sv:.4f} Sv "
          f"-> target={_tgt_Sv:.4f} Sv (area-weighted, grid-independent)")
    return out


def load_nemo_siconc(grid, grid_type, lat2d_deg, lon2d_deg, siconc_file=None,
                     land_mask=None):
    """Load NEMO ORCA1 sea-ice concentration and IDW-regrid onto the model grid.

    Returns siconc in [0,1], shape ``lat2d_deg.shape`` ((n_lat,n_lon) for
    latlon/tripole; (nCells,) for MPAS; (6,n,n) for the cube), for the SW-albedo
    surrogate (``--ice-albedo``).  The available file is an ANNUAL MEAN
    (``ORCA1_1y_*icemod.nc``), so this is a time-INVARIANT climatology -- load
    ONCE before the time loop.  LIMITATION: an annual mean over-ices the Antarctic
    summer (when the SST warm bias is worst) and under-ices winter; a 12-month
    icemod climatology (if sourced later) would refine this via the
    ``load_runoff_monthly`` monthly pattern.  Regrids from ALL NEMO ocean cells
    (incl. ice-free siconc=0) via the same curvilinear IDW used for runoff/bathy."""
    import xarray as xr
    ds = xr.open_dataset(siconc_file or _SICONC_NC, decode_times=False)
    src_lat = _squeeze2d(ds["nav_lat"].values)
    src_lon = _squeeze2d(ds["nav_lon"].values)
    sic = np.asarray(ds["siconc"].values, dtype=np.float64)
    if sic.ndim == 3:                          # (time, y, x) -> annual climatology
        with warnings.catch_warnings():        # all-NaN land columns -> NaN (kept
            warnings.simplefilter("ignore", RuntimeWarning)  # out via src_valid below)
            sic = np.nanmean(sic, axis=0)
    sic = _squeeze2d(sic)
    # Source ocean mask = finite cells.  VERIFIED for this NEMO ORCA1 icemod file:
    # land is written as _FillValue -> NaN (108k NaN cells; e.g. the Sahara cell is
    # NaN), so finiteness IS the land/ocean discriminator and ice-free OPEN ocean
    # (siconc=0, finite) is correctly retained as IDW source.  PORTABILITY caveat
    # (codex): a NEMO build that writes finite land ZEROS instead would let land
    # cells damp coastal/ice-edge siconc -- use an explicit ocean mask then.
    src_valid = np.isfinite(sic)
    sic = np.clip(np.nan_to_num(sic, nan=0.0), 0.0, 1.0)
    out, _ = _regrid_curv_to_points(
        sic, src_lat, src_lon, src_valid, lat2d_deg, lon2d_deg, k=4, max_deg=2.0)
    out = np.clip(out, 0.0, 1.0)
    if land_mask is not None:
        out = np.where(np.asarray(land_mask) > 0.5, out, 0.0)
    print(f"[setup] sea-ice albedo: NEMO siconc (annual) regridded onto {grid_type}, "
          f"max {float(out.max()):.2f}, ice-covered (>0.15) cell frac "
          f"{float((out > 0.15).mean()):.3f}")
    return out


def load_nemo_chl_monthly(grid, grid_type, lat2d_deg, lon2d_deg, chl_file=None):
    """Load the monthly ESACCI chlorophyll climatology and IDW-regrid onto the grid.

    Returns a ``(12, *lat2d_deg.shape)`` array of surface chlorophyll [mg/m^3] for
    the RGB shortwave-penetration scheme (``--sw-rgb-chl``).  The source
    ``CHLA(12, 361, 721)`` lives on a 0.5deg regular grid with 2D nav_lat/nav_lon,
    so the SAME curvilinear IDW used for siconc/runoff/bathy applies directly.
    Values are left in physical units (the model clamps to NEMO's [0.03, 10] range
    at class-index time); only non-finite source cells are excluded from the IDW.
    Loaded ONCE before the time loop and indexed per step by calendar month."""
    import xarray as xr
    ds = xr.open_dataset(chl_file or _CHL_NC, decode_times=False)
    src_lat = _squeeze2d(ds["nav_lat"].values)
    src_lon = _squeeze2d(ds["nav_lon"].values)
    chl = np.asarray(ds["CHLA"].values, dtype=np.float64)   # (12, y, x)
    if chl.ndim != 3 or chl.shape[0] != 12:
        raise ValueError(
            f"expected monthly CHLA (12, y, x); got shape {chl.shape} in {chl_file or _CHL_NC}"
        )
    out_months = []
    for m in range(12):
        src = chl[m]
        # Valid IDW sources = finite AND positive Chl: a NEMO build that writes
        # land/missing cells as finite 0 would otherwise dilute coastal ocean
        # down to the 0.03 clamp floor (codex MED).  This file's ocean min is
        # ~0.007 mg/m3 with land as _FillValue, so finiteness alone suffices,
        # but the `> 0` guard makes the loader robust to a zero-filled variant.
        src_valid = np.isfinite(src) & (src > 0.0)
        src_filled = np.nan_to_num(src, nan=0.0)
        om, _ = _regrid_curv_to_points(
            src_filled, src_lat, src_lon, src_valid, lat2d_deg, lon2d_deg,
            k=4, max_deg=2.0)
        # Floor at the NEMO clamp minimum so flood-filled land/coast cells never
        # produce a zero/negative Chl that would underflow the class-index log10.
        out_months.append(np.maximum(om, 0.03))
    out = np.stack(out_months, axis=0)
    print(f"[setup] RGB chlorophyll: ESACCI monthly regridded onto {grid_type}, "
          f"range {float(out.min()):.3f}-{float(out.max()):.3f} mg/m3, "
          f"annual-mean {float(out.mean()):.3f}")
    return out


def _ice_presence_from_tos(tos_C, ice_edge_C: float = -1.0, ramp_C: float = 1.0):
    """Sea-ice presence in [0,1] from SST [degC]: 1 where the surface is at/below
    the freezing point (ice), 0 over warm open water, with a smooth tanh ramp of
    half-width ``ramp_C`` centred at ``ice_edge_C``.  NEMO sea ice sits at the
    freezing point, so cold SST is a faithful indicator of ice presence."""
    return 0.5 * (1.0 - np.tanh((np.asarray(tos_C) - ice_edge_C) / ramp_C))


def _seasonal_siconc_from_presence(annual, presence):
    """Combine an annual-mean siconc map with a (12, *grid) ice-PRESENCE stack into
    a (12, *grid) monthly siconc.  Per-cell MEAN-PRESERVING normalisation:

        siconc(m) = clip(annual * presence(m) / mean_m presence, 0, 1)

    Before the [0,1] clip the 12-month MEAN equals the annual-mean siconc, so the
    ANNUAL albedo is CONSERVED: the ice months carry the true (elevated) winter
    concentration and the warm months go to ~0.  This removes the spurious year-
    round summer albedo (the NH cold-bias root cause) WITHOUT the annual SW over-
    absorption a max-normalisation would introduce (codex MEDIUM).  Perennial-ice
    cells (presence~const) are left ~unchanged; cells with no ice in any month
    (mean presence -> 0) yield 0.  The clip caps cells whose reconstructed winter
    concentration exceeds 1 (a cell that is, say, 0.4 annual but iced only 3 months
    is ~fully iced those months)."""
    annual = np.asarray(annual)
    presence = np.asarray(presence)
    mean_p = np.maximum(presence.mean(axis=0), 1.0e-6)         # per-cell mean presence
    season = presence / mean_p[None, ...]                      # 12-mo mean = 1 (pre-clip)
    return np.clip(annual[None, ...] * season, 0.0, 1.0)


def load_nemo_siconc_monthly(grid, grid_type, lat2d_deg, lon2d_deg,
                             siconc_file=None, tos_file=None, land_mask=None,
                             ice_edge_C: float = -1.0, ramp_C: float = 1.0):
    """Build a 12-MONTH sea-ice-concentration climatology on the model grid for the
    SEASONAL SW-albedo surrogate (``--ice-albedo-seasonal``).

    Motivation (codex HIGH; NH cold bias): the annual-MEAN ``siconc`` applied every
    timestep keeps a high ice albedo through the summer in NH seasonal-ice zones
    (Labrador/Greenland/Bering/Okhotsk), suppressing summer SW absorption all year
    -> a large spurious NH cold bias.  The faithful fix is a monthly siconc.  The
    NEMO run wrote only an annual ``siconc`` (no monthly icemod), but it DID write
    monthly ``tos`` (SST); NEMO sea ice sits at the freezing point, so the monthly
    SST is a faithful proxy for WHEN ice is present.

    Construction (NEMO-derived, prescribed -> feedback-safe):
      presence(m,cell) = 0.5*(1 - tanh((tos_m_C - ice_edge_C) / ramp_C))  in [0,1]
        (->1 where the monthly SST is at/below freezing, ->0 over warm open water)
      siconc(m,cell)   = clip(annual_siconc(cell) * presence(m,cell)
                              / mean_m presence, 0, 1)
        (per-cell MEAN-PRESERVING so the 12-month mean equals the annual-mean
         concentration: ice months carry the true winter value, warm months go to
         ~0 -- captures the seasonal cycle and the correct NH/SH phase and conserves
         the annual albedo; perennial-ice cells keep presence~const -> unchanged.
         See :func:`_seasonal_siconc_from_presence`).

    Returns ``(12, *lat2d_deg.shape)`` siconc in [0,1].  LIMITATION: this recovers
    the SEASONALITY of NEMO's ice from its SST; a true monthly icemod climatology
    (a NEMO re-run with the monthly ice stream actually written) would be the next
    refinement.  ``ice_edge_C``/``ramp_C`` set the SST->ice-presence ramp [degC]."""
    import xarray as xr
    # Annual siconc spatial pattern, already regridded onto the model grid.
    annual = load_nemo_siconc(grid, grid_type, lat2d_deg, lon2d_deg,
                              siconc_file=siconc_file, land_mask=land_mask)
    ds = xr.open_dataset(tos_file or _TOS_MONTHLY_NC, decode_times=False)
    src_lat = _squeeze2d(ds["nav_lat"].values)
    src_lon = _squeeze2d(ds["nav_lon"].values)
    tos = np.asarray(ds["tos"].values, dtype=np.float64)        # (time, y, x) degC
    if tos.ndim != 3:
        raise ValueError(f"monthly tos expected (time,y,x), got {tos.shape}")
    # Calendar-month climatology: average every record sharing a calendar month
    # (NYF -> all years share the same forcing; ``tos[k::12]`` are the same month
    # across years; a partial final year just gives some months 1 extra sample --
    # e.g. 40 records = 3y + 4m -> months 0-3 get 4, months 4-11 get 3).
    n_t = tos.shape[0]
    if n_t < 12:                                                # codex LOW: a <12-record
        raise ValueError(                                       # file can't form a 12-mo
            f"monthly tos has only {n_t} records (<12): cannot build a calendar-"      # climatology
            "month climatology.  Provide >=12 monthly records via --tos-monthly-file.")
    presence_m = []
    for m in range(12):
        recs = tos[m::12]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tos_m = np.nanmean(recs, axis=0)                    # (y, x) degC
        src_valid = np.isfinite(tos_m)
        tos_m = np.nan_to_num(tos_m, nan=10.0)                  # land/NaN -> warm (no ice)
        tos_grid, _ = _regrid_curv_to_points(
            tos_m, src_lat, src_lon, src_valid, lat2d_deg, lon2d_deg,
            k=4, max_deg=2.0)
        presence_m.append(
            _ice_presence_from_tos(tos_grid, ice_edge_C, ramp_C))
    presence = np.stack(presence_m, axis=0)                     # (12, *grid)
    out = _seasonal_siconc_from_presence(annual, presence)      # (12, *grid)
    # Summer-vs-winter contrast + annual-mean CONSERVATION diagnostic (codex MEDIUM:
    # the 12-month mean should track the source annual siconc; the [0,1] clip is the
    # only departure, where reconstructed winter ice saturates).
    nh = np.asarray(lat2d_deg) > 45.0
    if nh.any():
        mar = out[2][nh].mean(); sep = out[8][nh].mean()
        mean12 = out.mean(axis=0)
        print(f"[setup] sea-ice albedo: SEASONAL NEMO siconc (12 mo via monthly SST) "
              f"on {grid_type}; NH(>45N) mean siconc Mar={mar:.3f} Sep={sep:.3f} "
              f"(annual={annual[nh].mean():.3f}) -- summer albedo relaxed; "
              f"12-mo-mean vs annual (conservation) NH={mean12[nh].mean():.3f} "
              f"global={mean12.mean():.3f} vs {annual.mean():.3f}")
    return out


# noleap calendar month lengths (NEMO/OMIP convention) + cumulative day bounds.
_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_CUM = np.cumsum(_MONTH_DAYS)  # [31,59,...,365]


def _runoff_month_idx(step: int, dt: float) -> int:
    """Climatological calendar month 0-11 for the perpetual-year model time, using
    NEMO's NOLEAP month lengths (not equal 365/12 bins; codex MEDIUM)."""
    day = (step * dt / _SEC_PER_DAY) % 365.0
    return int(np.searchsorted(_MONTH_CUM, day, side="right"))


def _siconc_at_step(siconc, step: int, dt: float, monthly: bool):
    """Sea-ice concentration for the current step.  ``monthly`` -> index the leading
    12-month axis of ``siconc`` by the calendar month (same NOLEAP binning as the
    runoff climatology); otherwise return the single annual map as-is.  ``None``
    passes through (no ice field loaded)."""
    if siconc is None or not monthly:
        return siconc
    return siconc[_runoff_month_idx(step, dt)]


def _idx_t(step: int, dt: float, n_rec: int) -> int:
    """Nearest 6-hourly CORE-II record for the current model time (perpetual yr)."""
    t = (step * dt) % _YEAR_S
    # CORE-II 6-hourly records are cell-CENTRED at (k+0.5)*6h (see
    # build_core2_nyf_zarr time_s=(arange+0.5)*6h), so the record whose centre is
    # nearest model time t is floor(t/6h), not round(t/6h) -- the latter applies
    # each record with a +3 h phase lead.
    return int(t // _SEC_PER_6H) % n_rec


_SOURCE_REV_UNAVAILABLE = "unavailable"
# Bounds on the provenance probe (codex r9).  Each foreign namespace portion
# costs up to three `git` subprocesses at setup, so on a slow shared filesystem
# an unbounded sweep is a real startup stall; and the recorded string ends up
# in the archive, so it must not grow without limit either.  legoesm has ~10
# portions today and they are normally all in ONE checkout, which the cheap
# `--show-toplevel` fast path settles in a single command each.
_MAX_TREE_PROBES: int = 16
_MAX_TREE_TAGS: int = 4

# --- LEGOESM_* environment in the restart fingerprint ----------------------
# Several numerics levers are ENV-gated rather than CLI flags
# (LEGOESM_BAROCLINIC_F32, LEGOESM_VMIX_F32_SOLVE, LEGOESM_VMIX_BATCHED,
# LEGOESM_TRACER_PAIR, ...), so two legs with identical command lines can
# integrate different numerics.  They are hashed as a fail-CLOSED SWEEP with an
# EXCLUSION list — the same doctrine as _RESTART_FP_EXCLUDE for CLI args, so a
# lever added later is covered automatically.
#
# BUT a BARE prefix sweep is a FALSE-ABORT regression (codex tail-round RED):
# ~120 LEGOESM_* variables exist and many are per-job INFRASTRUCTURE
# (LEGOESM_JIT_CACHE_DIR, LEGOESM_NCPUS, LEGOESM_COORD_PORT, ...) that
# legitimately differ between chained legs.  Hashing those aborts every real
# restart — breaking exactly the feature this exists to protect.
#
# The exclusions are STRUCTURAL FAMILIES, not a hand-listed set of names:
# filesystem locations, interpreters, resource counts, ports and debug dumps
# cannot change the trajectory.  Everything else stays hashed.  If a legitimate
# chain false-aborts, EXTEND THIS LIST — and never add a variable that changes
# numerics.
_RESTART_ENV_EXCLUDE_SUFFIX: tuple[str, ...] = (
    "_DIR", "_PATH", "_ROOT", "_CACHE", "_SRC", "_URL", "_FILE", "_PORT",
    "_LIB", "_REF",
)
_RESTART_ENV_EXCLUDE_EXACT: frozenset[str] = frozenset({
    # interpreters / environments / repo locations
    "LEGOESM_PYTHON", "LEGOESM_PY", "LEGOESM_CONDA_ENV", "LEGOESM_REPO",
    "LEGOESM_CLIMATEEVAL_PYTHON",
    # scheduler + resource shape: these change per job by construction and the
    # model's answer is invariant across them
    "LEGOESM_NCPUS", "LEGOESM_NGPUS", "LEGOESM_SLURM_ACCOUNT",
    "LEGOESM_JAX_COORDINATOR",
    # compile / mesh cache policy switches
    "LEGOESM_JIT_CACHE_MIN_SECS", "LEGOESM_JAX_CACHE_DISABLE",
    "LEGOESM_MESH_CACHE_DISABLE", "LEGOESM_ALLOW_CPU_COMPILE_CACHE",
    # data staging locations carrying none of the suffixes above
    "LEGOESM_CLM_SURFDATA", "LEGOESM_CMIP7_RAW",
    # diagnostics / profiling / test-selection switches
    "LEGOESM_PROFILE_MPI", "LEGOESM_VARIANT_COLORS", "LEGOESM_REGEN_GOLDEN",
    "LEGOESM_SCALING_KIND", "LEGOESM_DEBUG_HELD", "LEGOESM_DUMP_RAD",
    "LEGOESM_JAX_DISTRIBUTED_TEST", "LEGOESM_RUN_AMIP_INTEGRATION",
    "LEGOESM_RUN_SLOW_RCE",
})


def _restart_env_items(environ=None) -> list[tuple[str, str]]:
    """The ``LEGOESM_*`` settings that belong in the restart fingerprint.

    SORTED, so the digest cannot depend on environment iteration order.
    Factored out of ``main`` so BOTH directions are directly testable: a
    numerics gate must be included, an infrastructure path must not.
    """
    import os as _os
    env = _os.environ if environ is None else environ
    return sorted(
        (k, v) for k, v in env.items()
        if k.startswith("LEGOESM_")
        and k not in _RESTART_ENV_EXCLUDE_EXACT
        and not k.endswith(_RESTART_ENV_EXCLUDE_SUFFIX)
    )


def _source_revision(start_dir=None) -> str:
    """Git revision of the checkout this driver is RUNNING FROM.

    Recorded in every ``--restart-save`` archive and compared on resume, so a
    scorecard is never attributed to the wrong revision.

    Three things the naive ``git rev-parse HEAD`` got wrong (codex r6 MEDIUM):

    1. **Scope.** A bare ``git rev-parse`` resolves against the CWD, so a job
       launched from ``$HOME`` or from another worktree recorded a DIFFERENT
       repository's HEAD.  It is scoped to ``__file__``'s directory here, which
       is the tree whose code is actually executing (this repo runs pinned
       worktrees per job precisely because a shared checkout is not
       reproducible).
    2. **Dirty state.** A SHA describes committed content only.  Modified
       tracked files are appended as ``-dirty`` — ``git describe --dirty``
       semantics, i.e. UNTRACKED files are deliberately not counted: this repo
       always carries hundreds of untracked scratch scripts, so counting them
       would pin the marker permanently on and make it uninformative.
    3. **Silent failure.** ``None`` on error was indistinguishable from "the
       archive predates this field", and both sides silently skipped the
       comparison.  A failure is now recorded EXPLICITLY as
       ``_SOURCE_REV_UNAVAILABLE`` so the resume can say the check could not
       run rather than implying it passed.

    4. **Mixed trees** (codex r7 MEDIUM, widened at r8).  The DRIVER script and
       the imported ``legoesm`` packages need not come from the same checkout —
       every sbatch wrapper here sets ``PYTHONPATH`` explicitly, and a wrong
       value silently runs this script against ANOTHER worktree's model code,
       which is exactly the shared-checkout hazard the pinned-worktree workflow
       exists to avoid.  ``legoesm`` is a PEP-420 namespace package, so it is
       not one directory but a LIST (``legoesm.__path__``: ``src/legoesm`` plus
       every ``packages/*/legoesm``); ALL of them are checked, not just the one
       that happens to define this module.  Any divergence from the driver's
       root is recorded as ``+mixedtree:<sha12>[,<sha12>...]``.  ``start_dir``
       skips the cross-check (unit testing of the git plumbing itself).

    5. **Inherited git environment** (codex r8 MEDIUM).  ``git -C <dir>`` does
       NOT override ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_COMMON_DIR``: under
       a hook or a wrapper that exports them, both probes would report an
       unrelated repository as clean and this function would return a
       confidently WRONG plain sha.  They are stripped from the child
       environment.

    Returns the revision string; never raises.
    """
    import os as _os
    import subprocess as _sp
    here = str(Path(__file__).resolve().parent if start_dir is None
               else Path(start_dir))
    # Repository-DISCOVERY variables only: leaving the rest of the environment
    # intact keeps git's own PATH/credential setup working.
    _env = {k: v for k, v in _os.environ.items()
            if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR",
                         "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                         "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                         "GIT_CEILING_DIRECTORIES")}

    def _git(cwd, *a):
        # check=False: a non-repo / missing git is an expected outcome here,
        # not an exception path.
        return _sp.run(["git", "-C", cwd, *a], capture_output=True,
                       text=True, timeout=10, check=False, env=_env)

    def _describe(cwd):
        """``(toplevel, revision-string)`` for one directory, or ``(None, …)``."""
        rev = _git(cwd, "rev-parse", "HEAD")
        if rev.returncode != 0 or not rev.stdout.strip():
            return None, _SOURCE_REV_UNAVAILABLE
        sha = rev.stdout.strip()
        top = _git(cwd, "rev-parse", "--show-toplevel")
        root = top.stdout.strip() if top.returncode == 0 else None
        st = _git(cwd, "status", "--porcelain", "--untracked-files=no")
        if st.returncode != 0:
            # HEAD resolved but the dirty check did not: say so rather than
            # implying a clean tree.
            return root, f"{sha}-dirty-unknown"
        return root, (f"{sha}-dirty" if st.stdout.strip() else sha)

    try:
        root, rev = _describe(here)
        if rev == _SOURCE_REV_UNAVAILABLE or start_dir is not None:
            return rev
        if root is None:
            # HEAD resolved but the repository ROOT did not, so the cross-check
            # below cannot run.  Keep the sha (it is real) but mark it
            # ambiguous rather than reporting a clean, fully-describing
            # revision (codex r8).
            return f"{rev}+mixedtree:unknown"
        # Cross-check EVERY tree that supplies model code.  BOUNDED WORK
        # (codex r9): the cheap `--show-toplevel` probe runs first and, for the
        # overwhelmingly common case of one checkout, is the ONLY command per
        # portion; the two extra commands run only for a portion that actually
        # differs.  Portions are capped and the suffix is truncated so neither
        # the setup latency nor the recorded string can grow without bound on a
        # slow shared filesystem.
        import legoesm as _lego
        found: set[str] = set()
        seen: set[str] = set()
        probes = 0
        truncated = False
        for portion in sorted(getattr(_lego, "__path__", [])):
            pkg_dir = str(Path(portion).resolve())
            if pkg_dir in seen:
                continue
            seen.add(pkg_dir)
            probes += 1
            if probes > _MAX_TREE_PROBES:
                truncated = True
                break
            top = _git(pkg_dir, "rev-parse", "--show-toplevel")
            pkg_root = top.stdout.strip() if top.returncode == 0 else None
            if pkg_root and Path(pkg_root) == Path(root):
                continue          # same checkout: nothing more to ask
            if pkg_root is None:
                # Root unresolved -> we cannot say WHICH tree it is; the
                # docstring promises 'unknown' here (codex r9 LOW).
                tag = "unknown"
            else:
                _, pkg_rev = _describe(pkg_dir)
                tag = (pkg_rev if pkg_rev == _SOURCE_REV_UNAVAILABLE
                       else pkg_rev.split("-")[0][:12])
            found.add(tag)
        if not found and not truncated:
            return rev
        # CANONICAL + HONEST ABOUT TRUNCATION (codex tail round): the portions
        # are visited in sorted order and the tags are sorted before capping,
        # so the recorded provenance is deterministic rather than dependent on
        # __path__ order; and when tags ARE dropped the string says so instead
        # of looking complete.
        tags = sorted(found)
        if len(tags) > _MAX_TREE_TAGS:
            tags = tags[:_MAX_TREE_TAGS] + [f"+{len(found) - _MAX_TREE_TAGS}-more"]
        if truncated:
            tags.append("probe-cap-reached")
        return f"{rev}+mixedtree:{','.join(tags)}"
    except Exception:                       # noqa: BLE001 — provenance only
        return _SOURCE_REV_UNAVAILABLE


def _source_revision_drift_note(parent_sha, this_sha) -> str | None:
    """Warning text for a resume across a source change, or ``None`` if silent.

    OUTCOMES KEPT DISTINCT (codex r6 MEDIUM).  The pre-fix code stored ``None``
    on failure and skipped the comparison whenever either side was falsy, so
    "the check could not run" was indistinguishable from "the check ran and
    matched" — the resume looked verified when nothing had been verified.
    EQUAL-BUT-AMBIGUOUS is a further case: a ``-dirty`` marker does not
    identify WHICH uncommitted edits were present, and ``+mixedtree`` says the
    SHA describes only part of the running code, so equality of two such
    strings is not equality of the code.

    Pure function of the two strings so it is directly testable; ``main`` only
    prints the result.
    """
    unknown = {None, "", _SOURCE_REV_UNAVAILABLE}
    if parent_sha in unknown or this_sha in unknown:
        return ("[warn] source-revision drift check SKIPPED: parent="
                f"{parent_sha or 'absent'}, this leg={this_sha}. The archive "
                "predates the field or the revision could not be resolved — "
                "this is NOT evidence that the code matches.")
    if parent_sha != this_sha:
        return (f"[warn] --restart-from was written at source revision "
                f"{parent_sha[:20]} but this leg is running {this_sha[:20]}: "
                "the model code changed between legs.  The state and "
                "configuration still validated, so the resume proceeds — but "
                "attribute results to BOTH revisions.")
    why = _revision_ambiguity(this_sha)
    if why:
        return (f"[warn] both legs report {this_sha}, but that string does not "
                f"pin the code: {why}  Matching markers do not prove matching "
                "source.")
    return None


def _revision_ambiguity(rev) -> str | None:
    """Why a revision string fails to identify the running code, or ``None``.

    ``-dirty`` / ``-dirty-unknown`` = uncommitted (or unknown) tracked edits;
    ``+mixedtree`` = the driver and the imported ``legoesm`` packages came from
    different checkouts, so ONE sha cannot describe both.
    """
    s = str(rev)
    reasons = []
    if "-dirty-unknown" in s:
        reasons.append("the dirty-state check itself failed, so uncommitted "
                       "tracked edits can neither be confirmed nor ruled out;")
    elif "-dirty" in s:
        reasons.append("the tree had uncommitted changes to tracked files, "
                       "which the sha does not describe;")
    if "+mixedtree" in s:
        reasons.append("the driver script and the imported legoesm packages "
                       "came from DIFFERENT checkouts (PYTHONPATH), so this "
                       "sha describes only the driver;")
    return " ".join(reasons) if reasons else None


def _diag(state, lat2d=None, lon2d=None) -> dict:
    """Cheap scalar diagnostics over ocean cells (one device->host pull).

    ``umax_lat``/``umax_lon``/``umax_lev`` = location of the 3-D max|u| —
    localises WHERE velocity grows/blows up (equator f->0 vs western
    boundaries vs the bipolar cap/fold vs at depth), the key pin-point for
    the dynamics instability seed.
    """
    # T/S: slice FIRST (device-side), THEN convert — only the 2-D surface
    # layer crosses to host (converting the full leaf would assemble the
    # whole 3-D sharded field; codex batch4 HIGH).  u/v genuinely need every
    # level (3-D max|u| + its location), so those are full-leaf pulls.
    T = np.asarray(state.T.data[..., 0])
    S = np.asarray(state.S.data[..., 0])
    u = np.asarray(state.u.data)            # (nlat, nlon+1, nlev)
    # MPAS has no separate v field (u is edge-normal on (nEdges, nlev)).
    has_v = getattr(state, "v", None) is not None
    v = np.asarray(state.v.data) if has_v else None
    m = np.asarray(state.land_mask.data) > 0.5
    au = np.abs(u)
    has_u = bool(au.size and np.isfinite(au).any())
    max_speed = float(np.nanmax(au)) if has_u else float("nan")
    umax_lat = umax_lon = float("nan")
    umax_lev = -1
    # Cube u is 4-D (6, n, n, nlev); the C-grid u is 3-D (n_lat, n_lon+1, nlev).
    # The umax-location pin-point below only makes sense for the 2-D-mappable
    # C-grid case, so skip it (keep max_speed + finite, which are shape-agnostic)
    # when the field is the cube layout or the coord arrays don't match.
    if lat2d is not None and has_u and u.ndim == 4 and np.asarray(lat2d).ndim == 3:
        # Cube A-grid: u (6, n, n, nlev) collocated with T -> lat2d (6, n, n).
        fu, ju, iu, ku = (int(x) for x in
                          np.unravel_index(np.nanargmax(au), au.shape))
        lat2d = np.asarray(lat2d)
        umax_lat = round(float(lat2d[fu, ju, iu]), 1)
        umax_lev = ku
        if lon2d is not None:
            umax_lon = round(float(np.asarray(lon2d)[fu, ju, iu]), 1)
    elif lat2d is not None and has_u and u.ndim == 3:
        ju, iu, ku = (int(x) for x in
                      np.unravel_index(np.nanargmax(au), au.shape))
        lat2d = np.asarray(lat2d)
        jj = min(ju, lat2d.shape[0] - 1)
        # u is a u-FACE field (n_lat, n_lon+1): column iu spans [0, n_lon]. The
        # T-centre coord arrays have n_lon columns and the wrap column n_lon is a
        # copy of column 0, so fold iu back with % (NOT clamp to n_lon-1, which
        # would report the cyclic-seam max ~360 deg away at the far edge).
        ii = iu % lat2d.shape[1]
        umax_lat = round(float(lat2d[jj, ii]), 1)
        umax_lev = ku
        if lon2d is not None:
            umax_lon = round(float(np.asarray(lon2d)[jj, ii]), 1)
    return {
        "mean_sst_C": float(np.nanmean(T[m])) if m.any() else float("nan"),
        "mean_sss": float(np.nanmean(S[m])) if m.any() else float("nan"),
        "max_abs_u": max_speed,
        "max_abs_v": (float(np.nanmax(np.abs(v))) if (v is not None and v.size)
                      else 0.0),
        "umax_lat": umax_lat,
        "umax_lon": umax_lon,
        "umax_lev": umax_lev,
        # Guard ALL prognostic fields -- a blowup that goes non-finite first in
        # S or v (not just T/u) must still trip the ABORT, else a NaN state is
        # silently snapshotted.  (MPAS has no v; skip it there.)
        "finite": bool(np.isfinite(T).all() and np.isfinite(S).all()
                       and np.isfinite(u).all()
                       and (v is None or np.isfinite(v).all())),
    }


def _grid_lat2d_deg(grid, grid_type):
    """Lat/lon in degrees for snapshots/scoring, per grid type. Cube returns the
    (6,n,n) per-face arrays as-is (the scorer flattens source points), tripole the
    2-D curvilinear T arrays, regular lat-lon the meshgridded 2-D axes."""
    if grid_type == "cubed_sphere":
        return (np.rad2deg(np.asarray(grid.lat)),
                np.rad2deg(np.asarray(grid.lon)))
    if grid_type == "mpas":
        # Voronoi cell centres: 1-D (nCells,); the scorer flattens any source.
        return (np.rad2deg(np.asarray(grid.latCell)),
                np.rad2deg(np.asarray(grid.lonCell)))
    if grid_type == "tripole":
        return (np.rad2deg(np.asarray(grid.lat_T)),
                np.rad2deg(np.asarray(grid.lon_T)))
    # regular lat-lon: 1-D radian axes -> 2-D degree meshgrid
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                               np.rad2deg(np.asarray(grid.lat)))
    return lat2d, lon2d


# ===========================================================================
# Prognostic sea-ice coupling (REUSES legoesm.ice.step_sea_ice — the REAL
# model — instead of the freeze-floor / prescribed-siconc / relaxation
# surrogates).  Pure integration glue: it samples the CORE-II forcing with the
# SAME sampler the momentum/heat path uses (sample_omip2_forcing), feeds the
# canonical step_sea_ice, and partitions/routes the returned TileResponse via
# the ONE shared coupler helper (legoesm.coupler.ocean_forcing.
# blend_ice_ocean_forcing) into the EXISTING OceanSurfaceForcing (tau, q_net,
# salt_flux, KPP freshwater) and FreshwaterForcing (evap x f_open, ice_fw)
# channels.  No new sea-ice physics; no new ocean salt/FW applicator.
# ===========================================================================

def _ice_state_spatial_shape(grid, app_grid_type):
    """Spatial shape of a per-cell ice field on the ocean grid.

    MPAS Voronoi -> ``(nCells,)``; lat-lon / tripole C-grid -> ``(n_lat, n_lon)``
    (the ocean T-point shape).  Matches ``_base_spatial_ndim`` in
    ``legoesm.ice.sea_ice`` so ``init_dynamic_ice_state(shape)`` builds a
    single-category state with the right rank for ``step_sea_ice``.
    """
    if app_grid_type == "mpas":
        return (int(np.asarray(grid.latCell).shape[0]),)
    if app_grid_type == "tripole":
        return tuple(int(s) for s in np.asarray(grid.lat_T).shape)
    if app_grid_type == "latlon":
        return (int(np.asarray(grid.lat).shape[0]),
                int(np.asarray(grid.lon).shape[0]))
    raise ValueError(
        f"--prognostic-sea-ice: unsupported grid_type {app_grid_type!r} for the "
        "ice-state spatial shape (supported: mpas, tripole, latlon).")


def _require_prognostic_ice_for_itd_flags(ice_categories, ice_ridging,
                                          prognostic_sea_ice) -> None:
    """Refuse ``--ice-categories``/``--ice-ridging`` without
    ``--prognostic-sea-ice``: their only consumer is the prognostic-ice
    build, so without it the flags would be accepted and silently ignored
    (the surrogate ice paths have no thickness distribution) — the
    accept-then-ignore shape the reachability audit forbids (codex)."""
    if (int(ice_categories) != 1 or ice_ridging) and not prognostic_sea_ice:
        raise ValueError(
            "--ice-categories/--ice-ridging configure the PROGNOSTIC ice "
            "model and require --prognostic-sea-ice; without it the flags "
            "would be accepted and silently ignored (the surrogate ice "
            "paths have no thickness distribution).")


def _resolve_ice_categories(n_categories, ridging, supports_dynamics,
                            grid_desc):
    """Resolve ``--ice-categories`` / ``--ice-ridging`` into the
    ``(n_categories, itd_remap, ridging_enabled)`` SeaIceConfig fields —
    refusing, never silently ignoring, a request that cannot take effect.

    * ``n_categories == 1`` (default): the exact pre-flag configuration —
      ``itd_remap='simple'`` (the SeaIceConfig default; never consulted at one
      category) and no ridging, byte-identical to runs before the flag
      existed.  ``--ice-ridging`` here is REFUSED rather than accepted: the
      step's ridging gate is multi-category-only (``is_multicat``), so the
      flag would parse and then silently do nothing every step.
    * ``n_categories > 1``: ``itd_remap`` is FORCED to ``'lipscomb2001'``
      rather than exposed as a choice — this runner always enables the brine
      tracer, and ``step_sea_ice`` rejects multi-category tracers under the
      ``'simple'`` linear remap (it moves only h/concentration/temperature
      across bins, breaking salt conservation), so a ``'simple'`` option
      could never legally run here (a phantom choice).
    * ``--ice-ridging`` needs the grid's strain-rate operators (the closing
      rate comes from the velocity deformation field): on a grid without
      them (tripole ORCA) ``step_sea_ice`` would raise at entry, so refuse
      up front with the actionable message instead.
    """
    n_cat = int(n_categories)
    if n_cat < 1:
        raise SystemExit(
            f"--ice-categories {n_cat}: need >= 1 thickness categor"
            f"{'y' if n_cat == 1 else 'ies'} (1 = single-category, the "
            "default; >= 2 enables the Lipscomb 2001 ITD).")
    if n_cat == 1:
        if ridging:
            raise SystemExit(
                "--ice-ridging requires --ice-categories >= 2: mechanical "
                "ridging redistributes ice BETWEEN thickness categories, and "
                "the single-category step skips it silently (the gate is "
                "multi-category-only), so accepting the flag here would be a "
                "no-op.")
        return 1, "simple", False
    if ridging and not supports_dynamics:
        raise SystemExit(
            f"--ice-ridging: grid {grid_desc} lacks the strain-rate "
            "operators the ridging closing rate needs (step_sea_ice would "
            "reject it at entry).  Use --grid mpas (or a lat-lon grid), or "
            "drop --ice-ridging (multi-category ITD without ridging still "
            "runs).")
    return n_cat, "lipscomb2001", bool(ridging)


def _apply_ice_init(ice_state, ic):
    """Overwrite the zero-ice cold-start state with the NEMO SI3 ice IC.

    ``ic`` is a :class:`legoesm.ocean.forcing.nemo_native_fields.NemoIceInit`
    already on the model T-grid (clamped, coherent, land-masked — see the
    loader).  Maps only the fields the file AND the state carry:

      at_i -> concentration, ht_i -> h_ice, ht_s -> h_snow, sm_i -> S_ice,
      tmsu -> T_ice (only where a valid on-ice reading exists; elsewhere the
      state keeps its own default so ice-free cells are untouched).

    Dynamics fields (u_ice/v_ice/sigma_*) and melt ponds have no SI3-IC
    counterpart and stay zero (Jan-1 start: ponds are a melt-season
    feature; ice velocity spins up from the ocean/wind stress in a few
    days).  Single-category states only: the SI3 IC file carries AGGREGATE
    fields, so with --ice-categories > 1 the runner applies this IC to the
    single-category state FIRST and then lifts it via
    distribute_dynamic_state_to_categories (delta ITD seeding); a state that
    already carries a category axis raises rather than guessing a split.
    """
    h_old = ice_state.h_ice.data
    conc_np = np.asarray(ic.concentration)
    if h_old.ndim != conc_np.ndim:
        raise ValueError(
            f"--ice-init: ice state h_ice has rank {h_old.ndim} but the "
            f"regridded IC has rank {conc_np.ndim}; multi-category ice "
            "states are not supported (the SI3 IC file carries aggregate "
            "fields only — initialise single-category, or add an ITD "
            "distribution step).")
    for _nm in ("concentration", "h_ice", "h_snow", "S_ice", "T_su"):
        _arr = getattr(ic, _nm)
        if _arr is not None and np.asarray(_arr).shape != tuple(h_old.shape):
            raise ValueError(
                f"--ice-init: regridded IC field {_nm!r} shape "
                f"{np.asarray(_arr).shape} does not match the ice-state "
                f"spatial shape {tuple(h_old.shape)}.")
    dtype = h_old.dtype
    conc = jnp.asarray(conc_np, dtype=dtype)
    upd = dict(
        concentration=ice_state.concentration.replace(data=conc),
        h_ice=ice_state.h_ice.replace(
            data=jnp.asarray(ic.h_ice, dtype=dtype)),
    )
    if ic.h_snow is not None:
        upd["h_snow"] = ice_state.h_snow.replace(
            data=jnp.asarray(ic.h_snow, dtype=dtype))
    if ic.S_ice is not None:
        upd["S_ice"] = ice_state.S_ice.replace(
            data=jnp.asarray(ic.S_ice, dtype=dtype))
    if ic.T_su is not None:
        T_ic = jnp.asarray(ic.T_su, dtype=dtype)
        upd["T_ice"] = ice_state.T_ice.replace(
            data=jnp.where(jnp.isfinite(T_ic), T_ic, ice_state.T_ice.data))
    return ice_state._replace(**upd)


def _surface_uv_faces(state):
    """Top-level (2-D) staggered u/v faces of a lat-lon C-grid family state,
    read DEVICE-SIDE — slice-before-convert (M2 batch4 contract: the ``[..., 0]``
    child never assembles the full, possibly lat-band-sharded, 3-D leaf).

    Layout-polymorphic (scaling-M2 leftover): the ``--spmd-persistent-state``
    lane carries ``v`` as the ``n_lat``-row ``v_lower`` (the staggered top row
    dropped), detected here by ``v`` sharing ``u``'s leading dim (the GLOBAL
    layout's v always has ``n_lat+1`` rows).  The missing top row is the
    pole/cap WALL — identically zero under the v-carrier contract asserted in
    ``shard_state_latlon`` — appended via the SAME shared reconstruction the
    full-state gather uses (``append_vface_wall_row``), so the sharded-state
    read is BIT-identical to reading the gathered state WITHOUT forcing the
    per-step full-state gather (gated by
    tests/parallel/test_persistent_sharded_ocean_loop.py).
    """
    u_face = jnp.asarray(state.u.data)[..., 0]       # (n_lat, n_lon+1)
    v_face = jnp.asarray(state.v.data)[..., 0]       # (n_lat[+1], n_lon)
    if v_face.shape[0] == u_face.shape[0]:           # v_lower carrier layout
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            append_vface_wall_row,
        )
        v_face = append_vface_wall_row(v_face)       # -> (n_lat+1, n_lon)
    return u_face, v_face


def _surface_currents(state, grid, app_grid_type):
    """Top-level ocean currents as GEOGRAPHIC (u_east, v_north) at T points /
    cells [m/s].

    Reused as ``ocean_u`` / ``ocean_v`` for ``step_sea_ice`` — the ice model's
    velocity contract is geographic E/N (free drift mixes them with the
    geographic winds, and the C-grid ice transport rotates E/N onto the local
    faces).  MPAS stores the edge-normal ``u`` (nEdges, nlev); reconstruct
    cell-centred (u, v) with the canonical Perot ``reconstruct_cell_velocity``
    (init_mpas).  Regular lat-lon: the grid axes ARE geographic, so the face
    value at the matching index is an adequate cell-centre proxy for the
    O(0.1 m/s) ocean-ice drag reference (avoids a bespoke face->centre
    average).  TRIPOLE: ``state.u``/``state.v`` are GRID-RELATIVE i/j face
    components — passing them as E/N mixed frames (codex r2 #3: free drift
    then blended them with geographic winds and the transport rotated the mix
    AGAIN in the bipolar cap), so delegate to ``_surface_currents_geographic``
    (face->T-centre average + the canonical renormalised rotation).  All
    C-grid reads go via ``_surface_uv_faces`` (device-side,
    persistent-sharded-layout aware — no full-state gather)."""
    if app_grid_type == "mpas":
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity
        u_sfc, v_sfc = reconstruct_cell_velocity(state.u.data[:, 0], grid)
        return u_sfc, v_sfc
    # latlon / tripole C-grid: u on EW faces (n_lat, n_lon+1), v on NS faces
    # (n_lat+1, n_lon).  Tripole MUST come back geographic (grid-relative i/j
    # would be frame-mixed with the geographic winds in free drift and rotated
    # a second time by the C-grid ice transport, codex r2 #3): reuse the
    # canonical geographic helper (T-centre average + renormalised rotation).
    if app_grid_type == "tripole":
        return _surface_currents_geographic(state, grid, app_grid_type)
    # Regular lat-lon: grid axes ARE geographic; crop to the T shape
    # (n_lat, n_lon) at the surface level (cheap face proxy, no gather).
    u_face, v_face = _surface_uv_faces(state)
    u_sfc = u_face[:, :-1]                      # drop the periodic wrap column
    v_sfc = 0.5 * (v_face[:-1, :] + v_face[1:, :])
    return u_sfc, v_sfc


def _surface_currents_geographic(state, grid, app_grid_type):
    """Top-layer ocean surface current in GEOGRAPHIC (east, north) [m/s] for the
    NEMO ``ln_crt_dwn`` relative-wind subtraction (``rn_vfac``).

    The CORE-II wind (``u10``/``v10``) arrives GEOGRAPHIC, so the current
    subtracted from it in ``air_sea_fluxes`` MUST be geographic too -- subtracting
    a grid-aligned current from a geographic wind corrupts the tripole-fold /
    high-lat stress (the frame-consistency correctness point).  Frame handling:

    * ``mpas`` -- ``_surface_currents`` already reconstructs geographic
      (u_east, v_north) via the Perot ``reconstruct_cell_velocity``; pass through.
    * lat-lon C-grid family (``latlon`` / ``latlon_regional`` / ``tripole``) --
      average the two bracketing faces of each cell to the T-centre (grid-aligned
      i-, j-components) and rotate to geographic with the canonical
      ``coupler.grid_remap.rotate_tpoint_currents_to_geographic`` (angle from the
      grid's ``cos_alpha_u`` / ``sin_alpha_u``; the IDENTITY outside the tripole
      bipolar cap, so a regular lat-lon grid is exact).

    The rotation lives HERE (the top-layer run driver), not in
    ``compute_omip2_surface_forcing``, because the ``legoesm.ocean`` package may
    not import ``legoesm.coupler`` under the import-linter component-independence
    contract; the driver is above both layers and may.  Unsupported grids raise
    (fail loud, never a silent absolute-wind fall-through)."""
    if app_grid_type == "mpas":
        return _surface_currents(state, grid, app_grid_type)
    if app_grid_type not in ("latlon", "latlon_regional", "tripole"):
        raise NotImplementedError(
            f"--relative-winds (ln_crt_dwn) is wired for the lat-lon C-grid "
            f"family + mpas; got app_grid_type={app_grid_type!r}.")
    # Device-side staggered read (persistent-sharded-layout aware — no
    # full-state gather; see _surface_uv_faces).
    u_face, v_face = _surface_uv_faces(state)        # (n_lat, n_lon+1) grid-i,
    #                                                  (n_lat+1, n_lon) grid-j
    u_c = 0.5 * (u_face[:, :-1] + u_face[:, 1:])     # -> (n_lat, n_lon) T-centre
    v_c = 0.5 * (v_face[:-1, :] + v_face[1:, :])     # -> (n_lat, n_lon) T-centre
    cos_a_u = getattr(grid, "cos_alpha_u", None)
    sin_a_u = getattr(grid, "sin_alpha_u", None)
    if cos_a_u is None or sin_a_u is None:
        # Regular lat-lon: grid-i == geographic east, grid-j == north (identity).
        return u_c, v_c
    from legoesm.coupler.grid_remap import rotate_tpoint_currents_to_geographic
    return rotate_tpoint_currents_to_geographic(u_c, v_c, cos_a_u, sin_a_u)


def _build_atm_to_surface_core2(forc, ramp=1.0):
    """Build an :class:`AtmToSurface` from the CORE-II fields ALREADY sampled
    onto the ocean grid by :func:`sample_omip2_forcing`.

    Field mapping (CORE-II -> AtmToSurface), units preserved:
      u10        -> u_lowest          [m/s]
      v10        -> v_lowest          [m/s]
      T_air [K]  -> T_lowest          [K]   (CORE-II air T is already Kelvin)
      q_air      -> q_lowest          [kg/kg]
      sw_down    -> sw_down           [W/m2]
      lw_down    -> lw_down           [W/m2]
      precip     -> precip_total      [kg/m2/s]
      snow       -> precip_snow       [kg/m2/s] (0 when the cache lacks snow)
      slp        -> p_surface=p_lowest[Pa]      (standard atm when slp absent)
    ``rho_lowest`` is moist-air density p/(R_d*T_v) — the SAME form
    ``coupler.surface_exchange.extract_atm_to_surface`` uses.  ``cos_zenith`` /
    ``co2_ppmv`` are inert for the ice model (it has its own SW/albedo path), so
    they are populated as zeros / a default and never read.  ``has_radiation`` /
    ``has_precipitation`` = 1 (CORE-II provides both)."""
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm import constants

    def _arr(name):
        return jnp.asarray(np.asarray(forc[name], dtype=np.float64))

    u10 = _arr("u10"); v10 = _arr("v10")
    T_air = _arr("T_air"); q_air = _arr("q_air")
    sw = _arr("sw_down") * float(ramp)
    lw = _arr("lw_down") * float(ramp)
    precip = _arr("precip") * float(ramp)
    snow_np = forc.get("snow")
    precip_snow = (jnp.asarray(np.asarray(snow_np, dtype=np.float64)) * float(ramp)
                   if snow_np is not None else jnp.zeros_like(precip))
    slp_np = forc.get("slp")
    p_sfc = (jnp.asarray(np.asarray(slp_np, dtype=np.float64))
             if slp_np is not None
             else jnp.full_like(u10, float(constants.p_atm_std)))
    # Moist-air density at the lowest level (p / (R_d * T_v)).
    T_v = T_air * (1.0 + (1.0 / float(constants.epsilon) - 1.0) * q_air)
    rho = p_sfc / (float(constants.R_d) * T_v)
    zero = jnp.zeros_like(u10)
    # has_radiation / has_precipitation are scalar flags (matching the canonical
    # extract_atm_to_surface); the ice model broadcasts them against the field
    # dtype.  cos_zenith / co2_ppmv are inert for the ice model (its own SW /
    # albedo path) -> zero array / scalar default, never read.
    return AtmToSurface(
        sw_down=sw, lw_down=lw, precip_total=precip, precip_snow=precip_snow,
        T_lowest=T_air, q_lowest=q_air, u_lowest=u10, v_lowest=v10,
        p_lowest=p_sfc, p_surface=p_sfc, rho_lowest=rho,
        cos_zenith=zero, co2_ppmv=jnp.asarray(0.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )


def _ice_apply_ew_overlap(ice_pytree):
    """Slave a sea-ice pytree's two longitude HALO columns to their ORCA
    2-point cyclic-overlap partners (``--ew-cyclic-overlap``): every ice
    state field AND every TileResponse array is CELL-CENTRED, so the cell
    rule applies uniformly — ``col[0] <- col[nx-2]``, ``col[nx-1] <- col[1]``
    (same physical columns).

    The ocean model re-imposes this on ITS prognostic state at the end of
    every step (``_apply_ew_cyclic_overlap``), but the ice state + response
    live in the HOST loop and were never projected (codex): the C-grid ice
    transport assumes regular period-``nx`` longitude wrap, off by one on an
    ORCA overlap grid, so the duplicated seam columns would drift apart step
    by step (and the response's halo stresses would feed inconsistent seam
    forcing).  Applied after ice init and after every ``step_sea_ice`` (state
    + response)."""
    def _ovl(a):
        nx = a.shape[1]
        a = a.at[:, 0].set(a[:, nx - 2])
        return a.at[:, nx - 1].set(a[:, 1])
    return jax.tree.map(_ovl, ice_pytree)


def _validate_kpp_freshwater_contract(app_grid_type: str, sf_scheme: str):
    """Fail fast if routing the physical net freshwater through
    ``sf.freshwater`` (the vmix surface-buoyancy channel) would be applied a
    SECOND time by this grid's surface-forcing physics scheme.

    The guard lists the actual ``sf.freshwater``-as-mass consumers, not an
    allow-list (codex r1 #1 + r3 #2 — blanket rejection aborted valid configs):

    * latlon / tripole (LatLonCGrid pipeline): only ``"external"`` consumes
      ``sf.freshwater`` (``physics/surface_forcing/external.py`` applies it as
      a VIRTUAL SALT — a second application on top of the
      ``model.step(freshwater=fw)`` mass channel).  ``restoring`` /
      ``prescribed`` / ``combined`` / ``bulk_formulas`` / ``flux_feedback``
      read their own config/channels and never touch ``sf.freshwater``
      (verified in ``surface_forcing/integration.py``).
    * mpas: NO scheme consumes it — the MPAS ``"external"`` block
      (``mpas_physics.py``) deposits ONLY tau/q_net (freshwater + salt are
      documented to stay on the ``step(freshwater=)`` / in-core channels).

    The cube path never calls this (its 'external' physics deliberately
    consumes ``sf.freshwater`` as the virtual-salt closure, with NO
    ``freshwater=`` step channel — single application by construction).
    """
    unsafe = () if app_grid_type == "mpas" else ("external",)
    if sf_scheme in unsafe:
        raise RuntimeError(
            "OMIP direct-forcing KPP-freshwater contract violated: on grid "
            f"{app_grid_type!r} the surface-forcing scheme {sf_scheme!r} "
            "applies sf.freshwater as virtual salt, so routing the physical "
            "net freshwater through it for KPP buoyancy would double-apply "
            "the freshwater mass (it already enters via model.step("
            "freshwater=fw)).")


def _ice_global_stats(ice_state, grid, ocean_mask):
    """Global prognostic-ice scalars over OCEAN cells:
    ``(ice_area_m2, mean_conc, max_thick_m, conc_masked)``.

    ``ice_area_m2`` = sum(conc * cell_area) [m2]; ``mean_conc`` = mean
    concentration over the ice-covered (conc > 0) wet cells (0.0 when
    ice-free); ``max_thick_m`` = max ice thickness [m] over wet cells.
    ``conc_masked`` is the wet-masked concentration array (reused by the
    stdout diag so the numbers cannot drift apart).  ``grid.area`` is the
    per-cell area [m2] on all three supported grids (VoronoiMesh exposes
    it as ``areaCell``; the C-grid families as the ``.area`` property).
    ``ocean_mask`` restricts the summary to wet cells so a spurious
    land-ice growth (masked out of the ocean budget) does not inflate
    the reported area.

    NOTE (tripole): the sum runs over the FULL array incl. the ORCA
    cyclic-overlap columns + north halo row, so duplicated seam cells
    overcount ``ice_area_m2`` by <~0.6%.  Kept deliberately: it is the
    convention every prior [ice] stdout line used (A/B-comparable run
    logs); the day-90 scoring reads snapshots, never these scalars."""
    conc = np.asarray(ice_state.concentration.data, dtype=np.float64)
    h = np.asarray(ice_state.h_ice.data, dtype=np.float64)
    m_nd = np.asarray(ocean_mask).ndim
    if conc.ndim > m_nd:
        # Multi-category (--ice-categories >= 2): aggregate BEFORE the spatial
        # masking.  The old ``conc.ndim > h.ndim`` test was never true for a
        # real multi-category state (both fields carry the trailing category
        # axis), so the mask broadcast below crashed on the first [ice] diag
        # line (codex).  Volume-weighted mean thickness matches the snapshot
        # writer's aggregation.
        vol = (conc * h).sum(axis=-1)
        conc = conc.sum(axis=-1)
        h = np.where(conc > 0.0, vol / np.maximum(conc, 1.0e-12), 0.0)
    m = np.asarray(ocean_mask, dtype=np.float64) > 0.5
    conc = np.where(m, conc, 0.0)
    h = np.where(m, h, 0.0)
    # VoronoiMesh exposes cell area as ``areaCell``; the C-grid families as the
    # ``.area`` property (see docstring) -- pick whichever this grid has.
    _area = getattr(grid, "area", None)
    if _area is None:
        _area = grid.areaCell
    area = np.asarray(_area, dtype=np.float64)
    ice_area_m2 = float(np.sum(conc * area))
    mean_conc = (float(conc[conc > 0].mean()) if (conc > 0).any() else 0.0)
    return ice_area_m2, mean_conc, float(h.max()), conc


def _prognostic_ice_diag(ice_state, resp, grid, app_grid_type, ocean_mask):
    """One-line verifiable summary over OCEAN cells: ice AREA [10^6 km2], mean
    concentration, max thickness, and the area-mean salt flux over ice-covered
    cells.  Scalars come from :func:`_ice_global_stats` (the same numbers the
    diag CSV logs)."""
    ice_area_m2, mean_conc, max_h, conc = _ice_global_stats(
        ice_state, grid, ocean_mask)
    icy = conc > 1.0e-3
    salt = np.asarray(resp.salt_flux, dtype=np.float64)
    salt_mean = float(salt[icy].mean()) if icy.any() else 0.0
    return (f"ice_area={ice_area_m2 / 1.0e12:.3f}e6 km2 "
            f"mean_conc={mean_conc:.3f} "
            f"max_h={max_h:.3f} m "
            f"mean_salt_flux(icy)={salt_mean:.3e} kg/m2/s "
            f"icy_cells={int(icy.sum())}")


def _amoc26n_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc: bool = True):
    """AMOC@26N [Sv] from the LIVE state (h reconstructed in-run via
    compute_layer_thickness — the snapshot lacks eta/z_coord).  Reuses the
    tested compute_amoc_from_state{,_mpas} (Atlantic-masked moc_streamfunction
    -> max).  Pure NumPy at run-end (no AD/JIT/shared-kernel touch).  Prints +
    writes a scalar file; NaN/skip is non-fatal.  RAPID obs ~17 Sv.

    io_proc=False (non-process-0 under --distributed): compute on every rank (the
    gathered state is replicated; collective-consume identically) but only process
    0 writes transports.txt.  Default True = single-process unchanged."""
    try:
        from legoesm.ocean.vertical import compute_layer_thickness
        h = np.asarray(compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord))
        if app_grid_type == "mpas":
            from legoesm.ocean.spinup import compute_amoc_from_state_mpas
            amoc = float(compute_amoc_from_state_mpas(
                np.asarray(state.u.data), h, grid))
        elif getattr(state, "v", None) is not None:
            from legoesm.ocean.spinup import compute_amoc_from_state
            amoc = float(compute_amoc_from_state(
                np.asarray(state.v.data), h,
                np.asarray(state.land_mask.data), grid))
        else:
            return
        if not io_proc:
            return
        print(f"[transports] AMOC@26N = {amoc:.2f} Sv  (RAPID obs ~17; "
              f"NEMO via scripts/validate/nemo_transports.py)")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "transports.txt").write_text(
            f"amoc26N_Sv {amoc:.4f}\n")
    except Exception as e:  # diagnostic must never crash the run
        print(f"[transports] AMOC@26N diag skipped: {type(e).__name__}: {e}")


def _save_bsf_amoc_diag(state, grid, z_coord, app_grid_type, out_dir,
                        io_proc: bool = True):
    """Barotropic streamfunction (gyres) + AMOC overturning streamfunction
    (Atlantic, lat-depth) + global MOC, at run-end for ALL simulations. Reuses
    the tested ``diagnostics_streamfunction.{barotropic_streamfunction,
    moc_streamfunction}`` (lat-lon C-grid: latlon/tripole). Saves a .npz of the
    fields and PNG maps. Pure NumPy at run-end; never crashes the run.

    MPAS (VoronoiMesh) has no structured lat-lon streamfunction operator here;
    its AMOC@26N scalar is reported by ``_amoc26n_diag`` -- the field maps are
    skipped (noted) until an unstructured-grid streamfunction is wired."""
    if app_grid_type == "mpas" or getattr(state, "v", None) is None:
        print("[transports] BSF/AMOC field maps skipped (no structured C-grid "
              "v-faces on this grid); AMOC@26N scalar is in transports.txt.")
        return
    try:
        from legoesm.ocean.vertical import compute_layer_thickness
        from legoesm.ocean.diagnostics_streamfunction import (
            barotropic_streamfunction, moc_streamfunction,
        )
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        eta = np.asarray(state.eta.data)
        Hb = np.asarray(state.H_bathy.data)
        mask = np.asarray(state.land_mask.data)
        h = np.asarray(compute_layer_thickness(state.eta.data,
                                               state.H_bathy.data, z_coord))
        # Barotropic streamfunction (gyres), [Sv], (n_lat, n_lon).
        bsf = np.asarray(barotropic_streamfunction(u, h, mask, grid))
        # AMOC = Atlantic-masked overturning; also the GLOBAL MOC. moc returns
        # (n_lat+1, nlev) [Sv] on v-faces. Atlantic band -75..15 E matches
        # compute_amoc_from_state. Use the 2-D T-grid longitude (tripole fold),
        # not the legacy 1-D first-row grid.lon (codex review).
        _lon = getattr(grid, "lon_T", None)
        if _lon is None:
            _lon = getattr(grid, "lon2d", None)
        if _lon is None:
            _lon = grid.lon
        lon2d = np.rad2deg(np.asarray(_lon))
        _lonw = (((lon2d + 180.0) % 360.0) - 180.0)
        lon_band = (_lonw >= -75.0) & (_lonw <= 15.0)
        if lon_band.ndim == 1:
            lon_band = np.broadcast_to(lon_band, mask.shape)
        atl_mask = mask * lon_band.astype(mask.dtype)
        amoc = np.asarray(moc_streamfunction(v, h, eta, Hb, atl_mask, grid))
        gmoc = np.asarray(moc_streamfunction(v, h, eta, Hb, mask, grid))
        lat2d = np.rad2deg(np.asarray(grid.lat))
        # v-FACE latitudes (n_lat+1) for the MOC fields, not the T-row centres
        # (codex review). From grid.lat_v if present, else T-row midpoints +
        # extrapolated end faces.
        _latv = getattr(grid, "lat_v", None)
        if _latv is not None:
            _lv2 = np.rad2deg(np.asarray(_latv))
            lat_v = _lv2.mean(axis=-1) if _lv2.ndim > 1 else _lv2
        else:
            rc = lat2d.mean(axis=-1) if lat2d.ndim > 1 else lat2d   # (n_lat,)
            mid = 0.5 * (rc[:-1] + rc[1:])
            lat_v = np.concatenate([[2 * rc[0] - mid[0]], mid,
                                    [2 * rc[-1] - mid[-1]]])         # (n_lat+1,)
        lat_v = np.asarray(lat_v)[:amoc.shape[0]]
        z_cen = (np.abs(np.asarray(z_coord.z_full_ref))
                 if getattr(z_coord, "z_full_ref", None) is not None
                 else np.arange(amoc.shape[1], dtype=float))
        if not io_proc:
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(out_dir) / "bsf_amoc.npz", bsf=bsf, amoc=amoc, gmoc=gmoc,
            lat_v=lat_v, z_center=z_cen,
            lat_T=lat2d, lon_T=lon2d, land_mask=mask)

        def _sx(a, f):  # finite-safe extremum (avoid all-NaN RuntimeWarning)
            return float(f(a)) if np.isfinite(a).any() else float("nan")
        # NADW cell = positive max; AABW = negative min. Print signed extrema.
        print(f"[transports] BSF [{_sx(bsf, np.nanmin):.1f},"
              f"{_sx(bsf, np.nanmax):.1f}] Sv; AMOC psi "
              f"[{_sx(amoc, np.nanmin):.1f},{_sx(amoc, np.nanmax):.1f}] Sv "
              f"(NADW=max); global-MOC [{_sx(gmoc, np.nanmin):.1f},"
              f"{_sx(gmoc, np.nanmax):.1f}] Sv -> bsf_amoc.npz")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # AMOC lat-depth section (x = v-face latitude, y = depth).
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
            lv = np.linspace(-30, 30, 31)
            c0 = ax[0].contourf(lat_v, -z_cen[:amoc.shape[1]],
                                amoc[:lat_v.size].T, levels=lv,
                                cmap="RdBu_r", extend="both")
            ax[0].set_title("AMOC overturning [Sv] (Atlantic)")
            ax[0].set_xlabel("latitude"); ax[0].set_ylabel("depth [m]")
            plt.colorbar(c0, ax=ax[0])
            # BSF map.
            bsf_p = bsf[:, :lon2d.shape[-1]] if bsf.shape[-1] != lon2d.shape[-1] \
                else bsf
            c1 = ax[1].pcolormesh(np.where(mask > 0.5, bsf_p, np.nan),
                                  cmap="RdBu_r", vmin=-60, vmax=60)
            ax[1].set_title("Barotropic streamfunction [Sv]")
            ax[1].set_xlabel("i"); ax[1].set_ylabel("j")
            plt.colorbar(c1, ax=ax[1])
            fig.tight_layout()
            fig.savefig(Path(out_dir) / "bsf_amoc.png", dpi=110)
            plt.close(fig)
            print(f"[transports] saved {Path(out_dir) / 'bsf_amoc.png'}")
        except Exception as pe:
            print(f"[transports] BSF/AMOC plot skipped: {type(pe).__name__}: {pe}")
    except Exception as e:
        print(f"[transports] BSF/AMOC diag skipped: {type(e).__name__}: {e}")


def _acc_drake_diag(state, grid, z_coord, app_grid_type, out_dir,
                    io_proc: bool = True):
    """ACC@Drake [Sv] from the LIVE state (h reconstructed in-run).  Reuses the
    tested compute_acc_from_state{,_mpas}: lat-lon/tripole via barotropic_stream
    function+acc_transport (ψ_bt max−min in the Drake band), MPAS via the
    edge-based section transport.  Pure NumPy at run-end; APPENDS a scalar to
    transports.txt (the AMOC diag writes it first); NaN/skip is non-fatal.  ACC
    spins up in months (wind-driven) so it is meaningful well before AMOC.
    NEMO ORCA1 ref ~159 Sv (obs ~137)."""
    try:
        from legoesm.ocean.vertical import compute_layer_thickness
        h = np.asarray(compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord))
        if app_grid_type == "mpas":
            from legoesm.ocean.spinup import compute_acc_from_state_mpas
            acc = float(compute_acc_from_state_mpas(
                np.asarray(state.u.data), h, grid))
        elif getattr(state, "v", None) is not None:
            from legoesm.ocean.spinup import compute_acc_from_state
            acc = float(compute_acc_from_state(
                np.asarray(state.u.data), h,
                np.asarray(state.land_mask.data), grid))
        else:
            return
        if not io_proc:
            return
        print(f"[transports] ACC@Drake = {acc:.2f} Sv  (obs ~137; NEMO ORCA1 "
              f"~159 via scripts/validate/nemo_transports.py --grid-u)")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "transports.txt", "a") as fh:
            fh.write(f"acc_drake_Sv {acc:.4f}\n")
    except Exception as e:  # diagnostic must never crash the run
        print(f"[transports] ACC@Drake diag skipped: {type(e).__name__}: {e}")


def _gateway_transport_diag(acc, out_dir, io_proc: bool = True):
    """APPEND time-mean Arctic gateway transports to transports.txt.

    ``acc`` is the GatewayAccumulator carried through the step loop (None when
    --gateway-transports is off).  Sign: POSITIVE = INTO the Arctic.  Volume in
    Sv, salt in psu*m^3/s.  Non-fatal, like its siblings.
    """
    if acc is None:
        return
    try:
        if acc.n == 0:
            print("[gateway] no steps accumulated; nothing written")
            return
        rows = acc.as_dict()
        if not io_proc:
            return
        print(f"[gateway] time-mean over {acc.n} steps (+ = INTO the Arctic):")
        for nm, (vol_sv, salt) in rows.items():
            print(f"[gateway]   {nm:<16} {vol_sv:+8.3f} Sv   "
                  f"{salt:+.4e} psu m^3/s")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "transports.txt", "a") as fh:
            fh.write(f"gateway_n_steps {acc.n}\n")
            for nm, (vol_sv, salt) in rows.items():
                fh.write(f"gateway_{nm}_vol_Sv {vol_sv:.6f}\n")
                fh.write(f"gateway_{nm}_salt_psu_m3s {salt:.6e}\n")
            # EXACT advective salt transport (store_salt_flux), APPENDED
            # after the legacy block so every pre-existing line stays
            # byte-identical; absent (legacy accumulator) => no new lines.
            _se_mean = acc.salt_exact_mean
            if _se_mean is not None:
                for i, nm in enumerate(acc.names):
                    fh.write(f"gateway_{nm}_salt_exact_psu_m3s "
                             f"{float(_se_mean[i]):.6e}\n")
    except Exception as e:  # a diagnostic must never crash the run
        print(f"[gateway] transport diag skipped: {type(e).__name__}: {e}")


GATEWAY_CUMULATIVE_CSV = "gateway_transports.csv"


class _GatewayCumulativeCsv:
    """Writer state for ``gateway_transports.csv``.

    Three fields, each load-bearing:

    ``fh``      the open handle, or ``None`` once the writer has been DISABLED.
                A write failure disables it permanently instead of leaving a
                broken handle to be retried at every later cadence (codex
                round-1 YELLOW 2).
    ``names``   the gateway order the HEADER was built from.  A row whose
                accumulator names differ is refused, so a gateway's transport
                can never land under another gateway's column.
    ``last_n``  the accumulator count of the last row written.  Dump points can
                coincide -- the run-end row, the abort row and a cadence row
                can all land on the same step -- and a duplicated endpoint
                would make a reader's window silently zero-length.
    """

    __slots__ = ("fh", "names", "last_n")

    def __init__(self, fh, names):
        self.fh = fh
        self.names = tuple(names)
        self.last_n = None


def _gateway_cumulative_open(out_dir, names, io_proc: bool = True):
    """Open ``gateway_transports.csv`` and write its header.  Returns a handle.

    WHY A SECOND FILE, AND WHY A CSV.  ``transports.txt`` is the run's
    key-value scalar dump: one ``name value`` line per quantity, written ONCE
    at run end.  Repeating that block per cadence would give duplicate keys
    that no existing reader of that file expects, so the whole-run block stays
    exactly as it is (byte-identical) and the per-cadence series goes to its
    own file -- the same split the driver already makes between the run-end
    scalars and ``diag_timeseries.csv``.  The CSV follows that sibling's
    conventions: opened once with a header, one row appended and FLUSHED per
    dump (observable mid-run under a pipe-buffered stdout), process-0 only,
    closed at the end.

    Returns a :class:`_GatewayCumulativeCsv`, or ``None`` when disabled or on
    any failure.  ``None`` makes every other function here a no-op, so a
    writer problem degrades the diagnostic and never the run.
    """
    if not io_proc:
        return None
    # Bound OUTSIDE the try so the except can close a handle that was opened
    # before the header write failed; returning None with the file still open
    # leaked it for the rest of the run (codex round-2 YELLOW 2).
    fh = None
    try:
        from legoesm.ocean.diagnostics_sections import gateway_cumulative_columns
        names = tuple(names)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = open(Path(out_dir) / GATEWAY_CUMULATIVE_CSV, "w")
        fh.write(",".join(gateway_cumulative_columns(names)) + "\n")
        fh.flush()
        return _GatewayCumulativeCsv(fh, names)
    except Exception as e:  # a diagnostic must never crash the run
        print(f"[gateway] cumulative CSV disabled: {type(e).__name__}: {e}")
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001 -- already failing; nothing to add
                pass
        return None


def _gateway_cumulative_row(gw_csv, acc, step, day) -> None:
    """APPEND one CUMULATIVE-sum row and flush.  No-op when either is ``None``.

    CUMULATIVE, never per-window: nothing is reset, so the run-end whole-run
    mean is produced by the very same accumulator and a window is recovered by
    the reader as ``(cumsum_b - cumsum_a) / (n_b - n_a)``.

    A row is SKIPPED (not written) in three cases, each of which would corrupt
    a reader's arithmetic rather than merely lose a row:
      * the accumulator's names disagree with the header's -> mis-assigned
        columns;
      * ``acc.n`` equals the last row's ``n_steps`` -> a duplicated endpoint,
        whose window has zero steps;
      * the writer has already been disabled by an earlier failure.
    """
    if gw_csv is None or acc is None or gw_csv.fh is None:
        return
    try:
        from legoesm.ocean.diagnostics_sections import (
            format_gateway_cumulative_row,
        )
        if tuple(acc.names) != gw_csv.names:
            # Skip the row rather than write one whose columns mean something
            # other than what the header says.  Loud, because a mis-assigned
            # gateway column is exactly the kind of defect that survives review.
            print(f"[gateway] cumulative row SKIPPED at step {step}: "
                  f"accumulator names {tuple(acc.names)} != header names "
                  f"{gw_csv.names}; the columns would be mis-assigned.")
            return
        n = int(acc.n)
        if gw_csv.last_n is not None and n == gw_csv.last_n:
            # Dump points coincide when the run ends exactly on a cadence
            # boundary, or when the abort fires on a step already dumped.  Two
            # identical rows give a reader a zero-step window and a 0/0 mean.
            return
        gw_csv.fh.write(format_gateway_cumulative_row(acc, step, day) + "\n")
        gw_csv.fh.flush()
        gw_csv.last_n = n
    except Exception as e:  # a diagnostic must never crash the run
        # DISABLE, do not merely skip: the handle may be broken (full disk,
        # closed file, dead NFS mount) and retrying it at every later cadence
        # would spam the log and write nothing (codex round-1 YELLOW 2).
        print(f"[gateway] cumulative CSV DISABLED at step {step} after "
              f"{type(e).__name__}: {e}")
        try:
            gw_csv.fh.close()
        except Exception:  # noqa: BLE001 -- already failing; nothing to add
            pass
        gw_csv.fh = None


def _gateway_cumulative_close(gw_csv) -> None:
    """Close the cumulative CSV.  No-op when never opened or already disabled."""
    if gw_csv is None or gw_csv.fh is None:
        return
    try:
        gw_csv.fh.close()
    except Exception as e:  # a diagnostic must never crash the run
        print(f"[gateway] cumulative CSV close failed: {type(e).__name__}: {e}")
    gw_csv.fh = None


def _mht_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc: bool = True):
    """Global meridional ocean heat transport (NH peak / SH min) [PW] from the
    LIVE state.  Reuses the tested compute_mht_from_state{,_mpas} (ρ0·cp·Σ v·θ·h
    per latitude).  Pure NumPy at run-end; APPENDS to transports.txt; non-fatal.
    NH peak obs ~1.8 PW; NEMO ref via scripts/validate/nemo_transports.py --grid-t."""
    try:
        from legoesm.ocean.vertical import compute_layer_thickness
        h = np.asarray(compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord))
        T = np.asarray(state.T.data)
        if app_grid_type == "mpas":
            from legoesm.ocean.spinup import compute_mht_from_state_mpas
            mh = compute_mht_from_state_mpas(np.asarray(state.u.data), T, h, grid)
        elif getattr(state, "v", None) is not None:
            from legoesm.ocean.spinup import compute_mht_from_state
            mh = compute_mht_from_state(
                np.asarray(state.v.data), T, h,
                np.asarray(state.land_mask.data), grid)
        else:
            return
        if not io_proc:
            return
        print(f"[transports] MHT NH peak = {mh['nh_peak_PW']:.2f} PW @ "
              f"{mh['nh_peak_lat']:.0f}N, SH min = {mh['sh_min_PW']:.2f} PW "
              f"(NH obs ~1.8 PW; NEMO via nemo_transports.py --grid-t)")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "transports.txt", "a") as fh:
            fh.write(f"mht_nh_peak_PW {mh['nh_peak_PW']:.4f}\n"
                     f"mht_sh_min_PW {mh['sh_min_PW']:.4f}\n")
    except Exception as e:  # diagnostic must never crash the run
        print(f"[transports] MHT diag skipped: {type(e).__name__}: {e}")


def _save_snapshot(out_dir: Path, tag: str, state, lat2d, lon2d, z_coord=None,
                   io_proc: bool = True, ice_state=None, grid=None,
                   step: int | None = None, day: float | None = None,
                   extra: dict | None = None):
    # io_proc=False (non-process-0 under --distributed): the state is replicated
    # and the host pull below is pure NumPy (no collective), but only process 0
    # writes the file — N processes would otherwise clobber the same .npz.  Still
    # materialize on every rank so the gathered (collective) state is consumed
    # identically.  Default True = single-process byte-identical.
    save_kw = dict(
        T=np.asarray(state.T.data), S=np.asarray(state.S.data),
        u=np.asarray(state.u.data),
        land_mask=np.asarray(state.land_mask.data),
        lat_T=np.asarray(lat2d), lon_T=np.asarray(lon2d),
    )
    # WHEN, inside the payload.  Until now the simulated time lived only in the
    # FILE NAME, so a reader had to trust a convention -- and `snapshot_final`
    # is written whenever a run stops, including early, so its name says nothing
    # about when it stopped.  A comparison against a dated oracle record cannot
    # be checked without this.
    if step is not None:
        save_kw["step"] = np.asarray(int(step))
    if day is not None:
        save_kw["time_days"] = np.asarray(float(day))
    # CELL AREA, so a band or box mean can be area-weighted on the model's own
    # mesh.  cos(lat) is only the right weight on a regular grid; on the
    # eORCA tripole it is wrong wherever the mesh is stretched, and a
    # cross-mesh comparison that used it is reporting a sampling difference as
    # a physical one (codex 2026-08-22 CRITICAL).
    for _area_attr in ("cell_area", "area_T", "areacello"):
        _a = getattr(grid, _area_attr, None) if grid is not None else None
        if _a is not None:
            save_kw["cell_area"] = np.asarray(getattr(_a, "data", _a))
            break
    # Prognostic sea ice (--prognostic-sea-ice): concentration + thickness on
    # the T-grid, ocean-masked (audited output gap — ice growth was invisible
    # in snapshots).  Cell areas are derivable from lat_T/lon_T (or grid
    # files) downstream, so raw per-cell fields keep the npz area-weighting-
    # safe.  EXTRA keys only: the scorers read explicit keys, so old readers
    # are unaffected and old snapshots (without these keys) stay loadable.
    if ice_state is not None:
        _lm = np.asarray(state.land_mask.data, dtype=np.float64)
        _ic = np.asarray(ice_state.concentration.data, dtype=np.float64)
        _ih = np.asarray(ice_state.h_ice.data, dtype=np.float64)
        if _ic.ndim > _lm.ndim:  # multi-category: aggregate (n/a in this runner)
            _vol = (_ic * _ih).sum(axis=-1)
            _ic = _ic.sum(axis=-1)
            _ih = np.where(_ic > 0.0, _vol / np.maximum(_ic, 1.0e-12), 0.0)
        save_kw["ice_concentration"] = _ic * _lm
        save_kw["ice_thickness"] = _ih * _lm
    # MPAS has no separate v field; the scorer reads T/S/land_mask/lat_T/lon_T only.
    if getattr(state, "v", None) is not None:
        save_kw["v"] = np.asarray(state.v.data)
    # The TRACER-ADVECTING mass flux, when the run captured it
    # (``store_mass_flux``/``--gateway-transports``).  #1442: ``state.u`` is the
    # velocity BEFORE the barotropic transport correction
    # ``delta_U = (Hu_avg - Hu_3d)/H_u_old`` -- that correction reaches the
    # tracer flux and never reaches ``state.u`` -- so an offline diagnostic that
    # rebuilds ``h * u`` from a snapshot is missing a depth-uniform mode.
    # MEASURED on eORCA1 (nemolev_trp_icemelt70_d90): the reconstructed net
    # meridional transport differed from the convergence implied by the measured
    # sea-level change by 0.35-1.28 Sv at every latitude -- ~100% of the
    # apparent signal at 66N.  #1440 made the corrected flux available IN STATE;
    # persisting it here is what lets an offline reader use it.
    # EXTRA keys only: old readers ignore them and old snapshots stay loadable.
    _stored_flux = False
    for _name in ("mass_flux_u", "mass_flux_v", "mass_flux_w"):
        _f = getattr(state, _name, None)
        if _f is not None and getattr(_f, "data", None) is not None:
            save_kw[_name] = np.asarray(_f.data)
            _stored_flux = True
    # The flux is THICKNESS-weighted velocity [m^2/s], so a reader still needs
    # the face widths to reach m^3/s (codex: the arrays alone are not
    # self-contained). Those live on the runtime grid, which an offline reader
    # does not have, so save them alongside — only when a flux was stored, and
    # only if the grid exposes them (a regular lat-lon grid may not).
    if _stored_flux and grid is not None:
        for _m in ("dy_u", "dx_v", "dx_u", "dy_v"):
            _val = getattr(grid, _m, None)
            if _val is not None:
                try:
                    save_kw[_m] = np.asarray(_val)
                except Exception:       # pragma: no cover - exotic grid proxy
                    pass
    # Free surface: carried so an OFFLINE tool can re-seed a state from this
    # snapshot without a barotropic-adjustment shock; the scorer ignores it.
    # NB --restart-from does NOT read snapshots (codex r4 LOW): the restart
    # loader rejects an archive with no '_slot_kinds' manifest.  Use
    # --restart-save / save_run_restart for a resumable checkpoint.
    eta = getattr(state, "eta", None)
    if eta is not None:
        save_kw["eta"] = np.asarray(eta.data)
    # Prognostic TKE carry (tke closure prognostic=True, any grid): recorded so
    # an offline re-seed does not re-spin the turbulence from the background
    # value — an uncheckpointed carry cold-starts and the re-seeded run is not
    # a continuation of this one (the #1310 lesson, there an uncheckpointed
    # mass-fixer anchor in the SPECTRAL model).  Again NOT the --restart-from
    # path — see the note above.
    # EXTRA key only — scorers and old readers are unaffected.
    tke = getattr(state, "tke", None)
    if tke is not None:
        save_kw["tke"] = np.asarray(tke.data)
    # Geometry for the offline mixed-layer-depth diagnostic (de Boyer Montegut /
    # Treguier 2023): sea-floor depth + level-centre reference depths.  The MLD
    # scorer derives the per-level wet mask from ``z_center_ref < H_bathy``.
    H_bathy = getattr(state, "H_bathy", None)
    if H_bathy is not None:
        save_kw["H_bathy"] = np.asarray(H_bathy.data)
    if z_coord is not None and getattr(z_coord, "z_half_ref", None) is not None:
        zh = np.asarray(z_coord.z_half_ref)              # (nlev+1,), <=0
        save_kw["z_center_ref"] = np.abs(0.5 * (zh[:-1] + zh[1:]))   # (nlev,) positive
    # Caller-supplied diagnostic arrays (``--kprofile-snapshots`` writes the
    # closure's viscosity and diffusivity here).  Kept as EXTRA keys so an old
    # reader ignores them and an old snapshot still loads.
    if extra:
        for _k, _v in extra.items():
            if _v is not None:
                save_kw[_k] = np.asarray(_v)
    if not io_proc:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"snapshot_{tag}.npz", **save_kw)


def _kprofiles(model, state, sf, dt, z_coord):
    """The closure's own K_M/K_H at this state, for a snapshot.

    NEMO publishes ``avm`` and ``avt`` in its five-day output, so the oracle's
    turbulent Prandtl number is a file read.  Ours was not comparable at all:
    the diffusivities are built inside the step and never persisted, so every
    statement about our equatorial mixing has been an inference from TKE and
    the stratification rather than a measurement of the quantity NEMO
    publishes.

    The diffusivities are built INSIDE the implicit vertical solve, not on the
    tendency -- for the TKE closure ``physics_fn`` deliberately returns
    ``K_v=None`` so the solve computes the profile itself -- so this asks the
    model for ``diagnose_vertical_K``, which runs the same solve setup and
    returns the coefficients at the point it consumes them, after the closure,
    the background floor and the additive internal-wave mixing.  That is the
    quantity NEMO publishes as avt/avm; reading ``K_v`` off the tendency would
    have returned nothing for the production arm.

    Returns an empty dict (and says why) when the model exposes no such method
    or the evaluation fails, so the snapshot is written either way and a
    diagnostic can never end an integration.
    """
    if not hasattr(model, "diagnose_vertical_K"):
        print("[kprofile] SKIPPED: this model exposes no diagnose_vertical_K "
              "(the diffusivity dump is a lat-lon C-grid feature only)",
              flush=True)
        return {}
    try:
        K_H, K_M = model.diagnose_vertical_K(state, dt, surface_forcing=sf)
    except Exception as exc:            # pragma: no cover - config-dependent
        print(f"[kprofile] SKIPPED: diagnose_vertical_K failed: {exc}",
              flush=True)
        return {}
    out = {"K_H_diag": np.asarray(getattr(K_H, "data", K_H)),
           "K_M_diag": np.asarray(getattr(K_M, "data", K_M))}
    # The TRUE interior interface depths the profile lives on, so a reader
    # compares against NEMO's depthw on the same levels instead of a
    # cell-centre midpoint reconstruction (which shifted a 64.96 m interface
    # to 65.12 m and dropped it from a <=65 m window -- codex 2026-08-23).
    z_half = getattr(z_coord, "z_half_ref", None) if z_coord is not None \
        else None
    if z_half is not None:
        out["z_interface_ref"] = np.abs(np.asarray(z_half)[1:-1])
    return out


def _dump_momentum_terms(model, state, sf, dt, lat2d, lon2d, tag=""):
    """DEBUG: per-term momentum-tendency breakdown (which term drives a
    cold-start blowup). Uses the lat-lon C-grid model's public
    ``tendencies_with_diagnostics`` (closure-tested); a no-op on grids without
    it (mpas/cube). Prints each term's global-max |tendency| + its location,
    sorted descending so the dominant (blowup) term is first."""
    if not hasattr(model, "tendencies_with_diagnostics"):
        print(f"[mom-diag{tag}] no per-term diagnostics on this model", flush=True)
        return
    try:
        _tend, diag = model.tendencies_with_diagnostics(
            state, surface_forcing=sf, dt=dt)
    except Exception as exc:                       # debug probe: never abort the run
        print(f"[mom-diag{tag}] diagnostics failed: {exc!r}", flush=True)
        return
    lat = np.asarray(lat2d)
    lon = np.asarray(lon2d)
    rows = []
    for name in diag._fields:
        if not (name.endswith("_u") or name.endswith("_v")):
            continue
        a = np.asarray(getattr(diag, name).data)
        aa = np.where(np.isfinite(a), np.abs(a), 0.0)
        if aa.size == 0:
            continue
        idx = np.unravel_index(int(np.argmax(aa)), a.shape)
        ii = min(int(idx[0]), lat.shape[0] - 1)
        jj = min(int(idx[1]), lat.shape[1] - 1)
        lev = int(idx[2]) if a.ndim >= 3 else -1
        rows.append((float(aa.max()), name, float(lat[ii, jj]),
                     float(lon[ii, jj]), lev))
    print(f"[mom-diag{tag}] per-term |tendency| global-max (m/s^2), "
          "largest first:", flush=True)
    for mx, name, la, lo, lev in sorted(rows, reverse=True):
        print(f"    {name:18s} {mx:.4e} @ ({la:+.1f},{lo:.0f}) lev{lev}",
              flush=True)


def _cli_flags_given(argv=None) -> set:
    """The set of argparse ``dest`` names the user passed explicitly on the CLI.

    Used so a ``--config`` template never overrides a flag the user typed:
    ``--latlon-res`` -> ``latlon_res``.  (Conservative: ``--flag=value`` and
    ``--flag value`` both register the flag.)
    """
    import sys as _sys
    argv = _sys.argv[1:] if argv is None else argv
    given = set()
    for tok in argv:
        if tok.startswith("--"):
            name = tok[2:].split("=", 1)[0]
            given.add(name.replace("-", "_"))
    return given


def _validate_spmd_persistent_state(persistent: bool, n_gpus: int,
                                    distributed: bool) -> None:
    """Fail-fast validation of the ``--spmd-persistent-state`` flag combos
    (pure helper: runs BEFORE any device/data work so a bad combination costs
    nothing; directly unit-tested).

    * requires ``--n-gpus > 1`` — with one device there is no scatter/gather
      to eliminate and the flag would silently mean nothing (dispatch
      hardening: refuse, never no-op);
    * refuses ``--distributed`` — the multi-process host loop runs on the
      all-gathered REPLICATED state on every rank (process-0-gated I/O), and
      the persistent lane's per-leaf host reads (``np.asarray`` of a
      non-fully-addressable array) would crash mid-loop.  Multi-controller
      persistence is the next increment.
    """
    if not persistent:
        return
    if n_gpus <= 1:
        raise SystemExit(
            "--spmd-persistent-state requires --n-gpus > 1: with a single "
            "device there is no per-step scatter/gather to eliminate (the "
            "plain model.step path is already gather-free). Drop the flag "
            "or add --n-gpus N.")
    if distributed:
        raise SystemExit(
            "--spmd-persistent-state is single-controller only (refused with "
            "--distributed): the multi-process host loop operates on the "
            "all-gathered replicated state on every rank, and the persistent "
            "lane's per-leaf host reads (np.asarray on a non-fully-"
            "addressable array) would crash. Run --distributed without "
            "--spmd-persistent-state, or single-process with it.")


class _PersistentStateResidency:
    """Residency tracker for the ``--spmd-persistent-state`` lane (scaling-M2).

    Owns the ONLY mutable residency flag: the persistent lane's ``state`` is
    either lat-band SHARDED (``v``/``v_mask`` carried as the n_lat-row
    ``v_lower``) or GLOBAL (full staggered ``v``), and every layout flip goes
    through :meth:`ensure_sharded` / :meth:`ensure_global` — which also COUNT
    each full-state transfer (the honest-cost contract: a forced per-step
    gather is announced + counted, never silent).  ``enabled=False`` (flag
    off, the default) makes both methods exact identity no-ops, so the
    default driver path is byte-identical.

    The full-STATE counters track LAYOUT FLIPS ONLY — they are NOT the total
    host-transfer cost.  Per-step LEAF host transfers (the forcing builders'
    surface-T slice pulls, the SSS-restore / ice-thermo surface pull+write-
    backs, the WOA-nudge / spin-up-drag FULL-3-D leaf round trips, the
    diag-cadence reads) remain in the persistent lane and are counted
    SEPARATELY via :meth:`count_leaf_slice` / :meth:`count_leaf_full`
    (codex batch4 HIGH: ``full_state_gathers_per_step=0`` must never read as
    "zero transfer cost").  Leaf counting is unconditional — the transfers
    happen on every lane; only the persistent lane REPORTS the totals.

    The flag cannot desync from the state: it flips only here, both methods
    are no-ops unless the flag is in the opposite residency, and every loop
    reassignment of ``state`` either preserves residency (the sharded inner
    step: sharded in -> sharded out; leaf-wise host BCs: layout-preserving
    leaf replaces) or routes through these methods.  Module-level (not a
    ``main()`` closure) so the interleavings are directly unit-tested
    (tests/unit/test_run_omip_core2_spmd_persistent_cli.py — supplementary
    review finding: closure-only bookkeeping was untestable).
    """

    def __init__(self, enabled: bool, shard_fn=None, gather_fn=None):
        if enabled and (shard_fn is None or gather_fn is None):
            raise ValueError(
                "_PersistentStateResidency: enabled=True requires both "
                "shard_fn and gather_fn (the lat-band layout flips).")
        self.enabled = bool(enabled)
        self._shard_fn = shard_fn
        self._gather_fn = gather_fn
        self.sharded = False           # current residency of the loop state
        self.gathers = 0               # full-STATE sharded->global transfers
        self.shards = 0                # full-STATE global->sharded transfers
        # -- per-step LEAF host-transfer counters (honest cost, codex HIGH) --
        self.leaf_slice_pulls = 0      # 2-D (surface/mask) leaf host READS
        self.leaf_slice_writes = 0     # 2-D surface-layer device WRITE-backs
        self.leaf_full_gathers = 0     # FULL-3-D single-leaf host reads
        self.leaf_full_uploads = 0     # FULL-3-D single-leaf device uploads

    def count_leaf_slice(self, *, pulls: int = 0, writes: int = 0) -> None:
        """Record 2-D leaf host transfers: surface-layer / 2-D-mask host
        READS (``pulls``) and surface-layer device WRITE-backs (``writes``).
        Bookkeeping only — never moves data itself."""
        self.leaf_slice_pulls += int(pulls)
        self.leaf_slice_writes += int(writes)

    def count_leaf_full(self, *, gathers: int = 0, uploads: int = 0) -> None:
        """Record FULL-3-D single-leaf host transfers: host reads of an
        entire leaf (``gathers``) and full-leaf device uploads (``uploads``).
        Bookkeeping only — never moves data itself."""
        self.leaf_full_gathers += int(gathers)
        self.leaf_full_uploads += int(uploads)

    def ensure_sharded(self, st):
        """Lay ``st`` out lat-band sharded (identity when disabled or
        already sharded)."""
        if not self.enabled or self.sharded:
            return st
        self.sharded = True
        self.shards += 1
        return self._shard_fn(st)

    def ensure_global(self, st):
        """Gather ``st`` back to the global single-device layout (full
        staggered v) for host-global consumers / I-O (identity when disabled
        or already global)."""
        if not self.enabled or not self.sharded:
            return st
        self.sharded = False
        self.gathers += 1
        return self._gather_fn(st)


def _record_final_state_digest(manifest_path, state) -> None:
    """Record the final ocean state digest into the run manifest (#376 Phase 4).

    Best-effort: a digest/record failure must not fail an otherwise-complete run.
    Gives ``legoesm reproduce --check`` a reference to compare a rerun against.
    """
    if manifest_path is None:
        return
    try:
        from legoesm.driver.restart import pytree_state_digest, record_state_digest
        digest = pytree_state_digest(state)
        record_state_digest(manifest_path, digest)
        print(f"[done] recorded state_digest {digest[:16]}... in {manifest_path}")
    except Exception as exc:  # noqa: BLE001 — provenance is best-effort
        print(f"[warn] state_digest not recorded: {type(exc).__name__}: {exc}")


def validate_prescribed_flow_args(prescribed_flow, grid: str,
                                  spinup_drag_tau_days: float,
                                  no_gm_redi: bool = False) -> None:
    """Arg-validation gates for --prescribed-flow (pure; unit-testable).

    The lever is IN-MODEL config (``LatLonCGridOceanConfig.prescribed_flow``,
    pinned inside ``_step_impl`` where the tracer mass fluxes are built), NOT
    a host-loop post-step reset — so it composes with ``--scan-block`` (the
    scan body calls ``_step_impl`` directly and the pin rides inside it) and
    needs no reference-state capture.  Only the lat-lon C-grid model
    implements it (grid=tripole|latlon_bathy); the cube/MPAS models have no
    ``prescribed_flow`` field and would silently ignore the request.
    """
    if prescribed_flow is None:
        return
    if grid not in ("tripole", "latlon_bathy"):
        raise SystemExit(
            "--prescribed-flow is implemented by the lat-lon C-grid model "
            "only (LatLonCGridOceanConfig.prescribed_flow): use --grid "
            f"tripole or latlon_bathy; got --grid {grid!r}.")
    if float(spinup_drag_tau_days) > 0.0:
        raise SystemExit(
            "--prescribed-flow cannot combine with --spinup-drag-tau-days: "
            "the spin-up Rayleigh drag rescales u/v post-step in the host "
            "loop while the lever pins them in-model — the interaction is "
            "undefined (the drag would be a silent no-op at best). Drop one.")
    if not no_gm_redi:
        # Both supported builders ship GM/Redi ON unconditionally (tripole:
        # NEMOMatchTripoleRecipeConfig.gm_redi=True; latlon_bathy:
        # _DEFAULT_BATHY_GM_REDI), and the model constructor REJECTS
        # prescribed_flow+GM/Redi (bolus transport is parameterized advection
        # the pin cannot isolate).  Require the explicit disable here so the
        # job dies at arg parse, not after the grid/mesh/IC build.
        raise SystemExit(
            "--prescribed-flow requires --no-gm-redi: the tripole and "
            "latlon_bathy configs enable GM/Redi by default, and GM bolus "
            "transport is parameterized ADVECTION that bypasses the pinned "
            "mass-flux block (the model constructor rejects the combination)."
            " Pass --no-gm-redi to run the circulation-isolation experiment.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for run_omip_core2, extracted from ``main`` so the
    argument set is unit-testable (e.g. the #939 ``--polar-filter``
    tri-state disable). ``main`` calls this then ``parse_args``."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--smoke", action="store_true",
                   help="Short 10-day benchmark run (reports steps/s).")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="Timestep [s] (default 3600 = NEMO ORCA1).")
    p.add_argument("--fp32", action="store_true",
                   help="Single-precision (float32) ocean run: leave JAX x64 OFF "
                        "(device arrays default float32) and set the all-fp32 "
                        "PrecisionPolicy. Halves GPU memory (~37 GB/GPU vs ~75 GB "
                        "in f64 for eORCA025 ¼°), so the lat-band SPMD step fits a "
                        "48 GB GPU at N=2 where f64 OOMs. Default (flag absent) = "
                        "f64 reference, byte-unchanged. Validated for the "
                        "tripole / latlon_bathy C-grid path (the SPMD fixed-iter "
                        "barotropic PCG + wright EOS + adcroft/smc03 PGF are "
                        "fp32-safe); mpas / cubed_sphere are NOT fp32-validated "
                        "(carry f64-only / fp32-unvalidated numerics) and are "
                        "rejected. NOTE: do "
                        "NOT also export JAX_ENABLE_X64=1 — that would re-enable "
                        "x64 and silently defeat the fp32 memory saving.")
    p.add_argument("--nlev", type=int, default=40,
                   help="Ocean vertical levels for the tanh z* default grid "
                        "(default 40, climate-usable minimum). For full NEMO "
                        "ORCA1 fidelity use --nemo-vertical (L75, overrides "
                        "--nlev).")
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--nemo-vertical", action="store_true",
                   help="Match NEMO ORCA1's vertical grid: build z_coord from "
                        "NEMO's reference layer thicknesses e3t_1d (75 levels, "
                        "~1 m surface -> ~204 m deep, Madec-Imbard L75) instead "
                        "of the default tanh 20-level stretch. Overrides --nlev / "
                        "--H-max to the NEMO column (better-resolved thermocline "
                        "+ MLD, directly comparable to NEMO).")
    p.add_argument("--nemo-vertical-file", type=str, default=None,
                   help="NetCDF with e3t_1d for --nemo-vertical (default: the "
                        "ORCA1 domain_cfg).")
    p.add_argument("--mesh", type=str, default=_MESH)
    p.add_argument("--grid", type=str, default="tripole",
                   choices=["tripole", "latlon_bathy", "cubed_sphere", "mpas"],
                   help="tripole (eORCA1 same-grid), latlon_bathy (regular lat-lon + "
                        "NEMO bathy + smc03 + polar filter), cubed_sphere (FV3 C-D "
                        "grid + NEMO bathy on cube cells; parked, resolution-limited), "
                        "or mpas (icosahedral Voronoi + NEMO bathy; resolution free via "
                        "--mpas-level).")
    p.add_argument("--latlon-res", type=str, default="180x360",
                   help="lat-lon resolution NxM for --grid latlon_bathy.")
    p.add_argument("--cube-n", type=int, default=48,
                   help="cubed-sphere face resolution n (C-n) for --grid cubed_sphere.")
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Multi-GPU lat-band SPMD ocean step (latlon_bathy / tripole "
                        "only): partition the ocean state by latitude band across N "
                        "local devices via make_sharded_ocean_step_global. n_lat is "
                        "padded with LAND rows at the SOUTH to a multiple of N (the "
                        "tripole north fold stays at the north). The host post-step "
                        "BCs (SSS restore / prognostic ice / geothermal / BBL / nudge) "
                        "run on the gathered GLOBAL state, unchanged. Default 1 = the "
                        "single-device path (byte-identical). Single-process: N must be "
                        "<= jax.local_device_count(). MULTI-NODE: add --distributed and "
                        "launch one process per GPU (mpirun/srun); then N must be <= "
                        "jax.device_count() (the GLOBAL device set across processes).")
    p.add_argument("--distributed", action="store_true",
                   help="Multi-PROCESS jax.distributed bootstrap for the --n-gpus "
                        "lat-band SPMD step: launch ONE process per GPU (mpirun/srun, "
                        "ntasks = total GPUs across nodes) and pass --distributed so "
                        "the GLOBAL jax.devices() spans every node. The lat-band mesh "
                        "is then built over all global devices and the --n-gpus guard "
                        "is relaxed to jax.device_count() (global). The OMIP host loop "
                        "runs identically on every process over the all-gathered "
                        "(replicated) state; I/O (manifest, CSV, snapshots, transports) "
                        "is written ONLY by process 0. Single-process (default, no "
                        "--distributed) is byte-unchanged.")
    p.add_argument("--spmd-persistent-state", action="store_true",
                   help="With --n-gpus > 1: keep the ocean state lat-band "
                        "SHARDED across steps (make_sharded_ocean_step) instead "
                        "of the global-in/global-out wrapper's full-state "
                        "scatter+gather EVERY step (scaling-M2). Host post-step "
                        "BCs run UNCHANGED: the leaf-wise host updates (SSS "
                        "restore / ice-thermo / nudge / drag) read+write "
                        "per-leaf on the addressable sharded arrays, and the "
                        "jnp per-column BCs (geothermal / ISF / BBL) are "
                        "sharding-transparent. The FULL state is gathered only "
                        "at snapshot/abort/final boundaries — prognostic-ice / "
                        "relative-winds surface-current reads are SHARDED "
                        "(scaling-M2 leftovers, PR #980) and no longer force a "
                        "per-step gather; any residual gathers are still "
                        "logged as full_state_gathers_per_step, never silent. "
                        "Default "
                        "OFF = the byte-identical per-step wrapper. Single-"
                        "controller only: refused with --distributed (the "
                        "multi-process host loop needs the replicated gathered "
                        "state on every rank; persistent multi-controller is "
                        "the next increment).")
    p.add_argument("--mpas-level", type=int, default=6,
                   help="MPAS Voronoi subdivision level (nCells=10*4^level+2): "
                        "5~230km, 6~115km (~ORCA1), 7~58km. For --grid mpas.")
    p.add_argument("--mpas-lloyd", type=int, default=20,
                   help="MPAS Lloyd-relaxation iterations at mesh build. For --grid mpas.")
    p.add_argument("--cube-Ah", type=float, default=None,
                   help="cube horizontal viscosity A_h override [m^2/s].")
    p.add_argument("--cube-hyperdiff", type=float, default=None,
                   help="cube biharmonic hyperdiffusion coeff override.")
    p.add_argument("--cube-pgf-scheme", type=str, default=None,
                   choices=[None, "adcroft", "smc03", "zero"],
                   help="cube partial-cell PGF scheme on the cd-grid AL corners "
                        "(adcroft=linear shift [default]; smc03=density-Jacobian, "
                        "the faithful path; zero=DIAGNOSTIC, removes the PGF entirely "
                        "to falsify the PGF-residual-is-the-cold-start-cause hypothesis).")
    p.add_argument("--cube-bottom-drag-r", type=float, default=None,
                   help="cube linear bottom-drag coeff r [m/s] (du/dt|drag=-r*u/h_bot "
                        "on the cd-grid cell-centre bottom level) — the proven "
                        "dissipation-stack piece the cube external-physics path lacked; "
                        "targets the bottom/mid-depth cold-start seed.")
    p.add_argument("--cube-bbl-thickness", type=float, default=None,
                   help="cube bottom-boundary-layer thickness H_BBL [m] (>0 spreads the "
                        "drag over the near-seafloor band instead of one thin partial "
                        "cell — the cold-start thin-bottom-cell blowup fix).")
    p.add_argument("--cube-bottom-drag-bg-vel", type=float, default=None,
                   help="cube MOM6 DRAG_BG_VEL u_bg [m/s] (>0 -> quadratic-with-floor "
                        "bottom drag; recovers linear r at |u|->0).")
    p.add_argument("--cube-harmonic-cfl-safety", type=float, default=None,
                   help="cube harmonic-viscosity diffusive-CFL cap safety (default 0.20; "
                        "A_h*dt/dx^2 <= safety/4). Raise toward ~1.0 for the maximal "
                        "stable isotropic ceiling — the upper bound on flow-adaptive Smag.")
    p.add_argument("--cube-bathy-smoothing", type=int, default=0,
                   help="cube: N Laplacian smoothing passes over the ocean bathymetry "
                        "(cube-seam-aware), reducing the r-factor / per-cell slope that "
                        "seeds the partial-cell PGF cold-start blowup at under-resolved "
                        "marginal seas. Proven NEMO/ROMS technique.")
    p.add_argument("--cube-baro-divdamp", type=float, default=None,
                   help="cube fv3sw barotropic divergence-damping factor (default 120). "
                        "Crank to test/damp the bathy-driven barotropic cold-start mode 2.")
    p.add_argument("--cube-baro-dampv", type=float, default=None,
                   help="cube fv3sw barotropic vorticity-damping coeff (default 0.030).")
    p.add_argument("--cube-smc03-bottom-2nd", action="store_true",
                   help="cube: smc03 PGF uses the 3-point 2nd-order backward bottom-cell "
                        "density slope (curvature-accurate under the pressure-dependent "
                        "EOS) instead of the O(dz)-biased one-sided slope — removes the "
                        "bottom-cell rest PGF residual that seeds the cold-start. "
                        "Linear-EOS bit-exact; requires --cube-pgf-scheme smc03.")
    p.add_argument("--cube-divdamp2", type=float, default=None,
                   help="cube 2nd-order divergence damping [m^2/s].")
    p.add_argument("--cube-divdamp4", type=float, default=None,
                   help="cube 4th-order divergence damping [m^4/s].")
    p.add_argument("--cube-rk3", action="store_true",
                   help="cube: 3-stage SSP-RK3 baroclinic update (vs forward-Euler) "
                        "— the tripole cold-start fix ported to the cube OceanModel.")
    p.add_argument("--cube-velocity-ceiling", type=float, default=None,
                   help="cube: clip |u|,|v| to this [m/s] each step -- bounds the sub-grid "
                        "marginal-sea jet spikes (caveated open-ocean run).")
    p.add_argument("--cube-mask-marginal-seas", action="store_true",
                   help="cube: mask poorly-resolved semi-enclosed marginal seas "
                        "(Med/Black/Red/Gulf/Baltic/Hudson) to land — their sub-grid "
                        "sills seed the cold-start PGF blowup (caveated, like the "
                        "latlon Arctic).")
    p.add_argument("--mask-marginal-seas", action="store_true",
                   help="lat-lon: mask the poorly-resolved semi-enclosed / "
                        "endorheic seas (Med/Black/Red/Gulf/Baltic/Hudson/Caspian/"
                        "Aral/Great Lakes) to land. On the regular grid the IDW "
                        "regrid reconnects them through 1-cell straits and inflates "
                        "their depth; the brackish/hypersaline WOA T/S then seeds a "
                        "sharp-front baroclinic-PGF cold-start blowup at NEMO "
                        "75-level (the Caspian, 47.5N/48E). Caveated open-ocean run.")
    p.add_argument("--woa-init", action="store_true",
                   help="Initialise T/S from WOA18 (faithful IC) vs rest state.")
    p.add_argument("--woa-t", type=str, default="data/woa18/woa18_decav_t00_01.nc")
    p.add_argument("--woa-s", type=str, default="data/woa18/woa18_decav_s00_01.nc")
    p.add_argument("--ke-gradient-scheme", type=str, default=None,
                   choices=["centered", "hollingsworth"],
                   help="KE-gradient discretization for the vector-invariant "
                        "momentum advection. 'hollingsworth' (NEMO nkeg_HW) is "
                        "consistent with the AL81 PV-flux Coriolis term; "
                        "'centered' is the legacy scheme. Default (None) PRESERVES "
                        "the config default, currently 'centered' for BOTH tripole "
                        "and latlon_bathy -- no production default is changed. This "
                        "is a diagnostic A/B knob: job 8106193 showed hollingsworth "
                        "does NOT fix the WOA cold-start blowup (both schemes go "
                        "non-finite by day 0.5 on both grids). For tripole the "
                        "hollingsworth KE stencil also still lacks a fold-aware "
                        "north halo (see run_omip.py tripole config note).")
    p.add_argument("--partial-cell", action="store_true",
                   help="Use OceanPartialCellCoordinate (z-level + partial bottom "
                        "steps, NEMO-faithful) with thin-cell snapping, instead of "
                        "the plain sigma-like z* coord from _create_setup. For "
                        "tripole/latlon this activates the Adcroft/SMC03 partial-cell "
                        "PGF correction; for the cube (FC backend) it builds the "
                        "partial-cell substrate (true bottom-cell thickness + "
                        "below-seafloor masking) — prerequisite for the cube smc03 "
                        "PGF. Matches NEMO's vertical coordinate. Fixes the spurious "
                        "equatorial-bottom PGF cold-start blowup (job 8106208).")
    p.add_argument("--pgf-scheme", type=str, default=None, choices=[None, "adcroft", "smc03"],
                   help="Override tripole PGF scheme (default: run_omip's adcroft).")
    p.add_argument("--A-h", type=float, default=None, help="Override Laplacian viscosity [m2/s].")
    p.add_argument("--tracer-advection", type=str, default=None,
                   help="Override the tracer advection scheme (e.g. ppm_fct "
                        "-- closest to NEMO's FCT2 and less diffusive at "
                        "fronts than the default tvd/Van-Leer; also weno5, "
                        "dst3, superbee). Validated per-scheme by the ocean "
                        "matrix; smoke before production.")
    p.add_argument("--B-h", type=float, default=None, help="Override biharmonic viscosity [m4/s].")
    p.add_argument("--K-bih", type=float, default=None,
                   help="Biharmonic tracer hyperdiffusion [m4/s] -- scale-selectively "
                        "damp a grid-scale baroclinic T/S mode (preserves large-scale gradients).")
    p.add_argument("--flat-bottom", action="store_true",
                   help="Replace bathymetry with a flat bottom (H_max) over ocean cells "
                        "-- controlled test isolating the PGF-over-topography error.")
    p.add_argument("--A-h-eq-boost", type=float, default=None,
                   help="Equatorial Laplacian-viscosity boost factor -- damps the f->0 "
                        "velocity growth (A_h *= 1+(boost-1)*exp(-(lat/sigma)^2)).")
    p.add_argument("--A-h-eq-sigma-deg", type=float, default=None,
                   help="Gaussian half-width [deg] of the equatorial A_h shaping "
                        "(--A-h-eq-boost). NEMO ORCA1's eddy_viscosity_3D ramp is ~7.")
    p.add_argument("--A-h-profile-file", type=str, default=None,
                   help="ORCA1 eddy_viscosity_3D.nc: prescribe the LATITUDINAL "
                        "A_h shape from the oracle's own momentum-viscosity "
                        "file (zonal median, ratio to --A-h). Replaces "
                        "--A-h-eq-boost. Tripole only.")
    p.add_argument("--adaptive-implicit-vertadv", action="store_true",
                   help="Enable adaptive-implicit vertical momentum advection "
                        "(Shchepetkin 2015 / NEMO ln_zad_Aimp) -- removes the vertical-CFL "
                        "limit so the spurious-w 'vertadv' runaway cannot amplify. The "
                        "NEMO-faithful fix for the OMIP cold-start blowup (eORCA OMIP "
                        "production runs set ln_zad_Aimp=.true.).")
    p.add_argument("--slope-foot-alpha", type=float, default=None,
                   help="MOM6 slope-foot viscosity enhancement at topographic slopes "
                        "(WBCs hug slopes -- the built-in WBC-enhanced-viscosity, NEMO-like). "
                        "Multiplies A_h by ~alpha where slope>threshold. 0=off, prod 3.")
    p.add_argument("--slope-foot-n-levels", type=int, default=None,
                   help="Levels from bottom to apply slope-foot (default 5; set =nlev for full column).")
    p.add_argument("--slope-foot-threshold", type=float, default=None,
                   help="Slope r-factor threshold above which slope-foot fires (MOM6 default 0.1).")
    p.add_argument("--momentum-advection", default=None,
                   choices=[None,"vector_invariant","weno5","weno7"],
                   help="Momentum advection scheme. weno5/weno7 = upstream-biased "
                        "(dissipative at sharp jets, NEMO-UP3-like); vector_invariant "
                        "(default) = energy-conserving AL81 (NON-dissipative).")
    p.add_argument("--C-smag", type=float, default=None,
                   help="Biharmonic Smagorinsky coeff (self-activating ~strain, scale-selective).")
    p.add_argument("--C-leith", type=float, default=None,
                   help="Leith biharmonic coeff (self-activating ~|grad vorticity| -- targets the sharp-jet edge).")
    p.add_argument("--C-smag-lap", type=float, default=None,
                   help="Laplacian Smagorinsky coeff (tripole OMIP default 0.33).")
    p.add_argument("--bottom-drag-r", type=float, default=None,
                   help="Linear bottom drag coefficient [m/s] (du/dt|drag=-r*u/h_bot). "
                        "tripole OMIP default is 0 (OFF); NEMO uses implicit quadratic drag.")
    p.add_argument("--bottom-drag-scheme", type=str, default=None,
                   choices=[None, "legacy", "nemo_quadratic", "nemo_loglayer"],
                   help="Bottom-drag law: 'nemo_quadratic' = zdfdrg np_non_lin "
                        "(the ORCA1 namelist: Cd0*sqrt(u^2+v^2+ke0) from the "
                        "bottom-cell full speed), 'nemo_loglayer' = np_loglayer "
                        "(Cd from clip((kappa/ln(0.5*e3t_bot/z0))^2, cd0, cdmax)). "
                        "Default/legacy keeps the historical r/DRAG_BG_VEL path.")
    p.add_argument("--bottom-drag-cd0", type=float, default=None,
                   help="NEMO rn_Cd0 [-] (ORCA1: 1e-3; loglayer Cd minimum)")
    p.add_argument("--bottom-drag-cdmax", type=float, default=None,
                   help="NEMO rn_Cdmax [-] (ORCA1: 0.1; loglayer Cd cap)")
    p.add_argument("--bottom-drag-z0", type=float, default=None,
                   help="NEMO rn_z0 bottom roughness [m] (ORCA1: 3e-3)")
    p.add_argument("--bottom-drag-ke0", type=float, default=None,
                   help="NEMO rn_ke0 background bottom KE [m^2/s^2] (ORCA1: 2.5e-3)")
    p.add_argument("--dm2dc", action="store_true",
                   help="Diurnal cycle on the daily-mean shortwave (NEMO "
                        "ln_dm2dc, sbcdcy/Bernie 2007; ORCA1: .true.).  "
                        "Mean-preserving analytic modulation of the CORE-II "
                        "daily SW.  Host-loop only.")
    p.add_argument("--isf", action="store_true",
                   help="NEMO ISF 'spe' prescribed ice-shelf melt "
                        "(ln_isfpar_mlt, cn_isfpar_mlt='spe'; ORCA1: on): "
                        "monthly melt deposited over the [zmin,zmax] band "
                        "with latent cooling + freezing-point heat content "
                        "+ virtual-salt dilution + eta volume source.  "
                        "Requires --isf-forcing-file.  Host-loop only.")
    p.add_argument("--isf-forcing-file", type=str, default=None,
                   help="NetCDF with sornfisf/sodepmin_isf/sodepmax_isf on "
                        "the eORCA1 grid (the ORCA1 INPUTS "
                        "runoff-icb_DaiTrenberth_Depoorter.nc).")
    p.add_argument("--iwm", action="store_true",
                   help="Internal wave-driven mixing (NEMO zdfiwm, de Lavergne "
                        "2020; ORCA1: ln_zdfiwm=.true.).  lat-lon/tripole only.")
    p.add_argument("--iwm-mevar", action="store_true",
                   help="zdfiwm ln_mevar variable mixing efficiency (ORCA1: off)")
    p.add_argument("--iwm-tsdiff", action="store_true",
                   help="zdfiwm ln_tsdiff differential T/S mixing (ORCA1: off; "
                        "raises — unsupported on the shared-K solve)")
    p.add_argument("--iwm-forcing-file", type=str, default=None,
                   help="de Lavergne power/decay maps (zdfiwm_forcing_TRA.nc "
                        "layout; the ORCA1 INPUTS copy works).  Omit for the "
                        "uniform constant-power fallback.")
    p.add_argument("--iwm-power-bot", type=float, default=1.0e-10,
                   help="Uniform-fallback abyssal-hill power [W/m^2]")
    p.add_argument("--iwm-power-cri", type=float, default=1.0e-10,
                   help="Uniform-fallback critical-slope power [W/m^2]")
    p.add_argument("--iwm-power-nsq", type=float, default=1.0e-5,
                   help="Uniform-fallback N^2-scaled power [W/m^2]")
    p.add_argument("--iwm-power-sho", type=float, default=1.0e-10,
                   help="Uniform-fallback shoaling power [W/m^2]")
    p.add_argument("--iwm-scale-bot", type=float, default=100.0,
                   help="Uniform-fallback abyssal-hill decay scale [m]")
    p.add_argument("--iwm-scale-cri", type=float, default=100.0,
                   help="Uniform-fallback critical-slope decay scale [m]")
    p.add_argument("--double-diffusion", action="store_true",
                   help="Double-diffusive mixing (NEMO zdfddm, Merryfield "
                        "1999; salt-fingering avt/avs).  ADDITIVE on top of the "
                        "vertical-mixing K like --iwm.  lat-lon/tripole only.")
    p.add_argument("--ddm-avts", type=float, default=None,
                   help="zdfddm rn_avts: max salt-fingering salt diffusivity "
                        "[m^2/s] (NEMO namzdf_ddm default 1e-4).  None keeps the "
                        "DoubleDiffusionConfig default.")
    p.add_argument("--ddm-rc", type=float, default=None,
                   help="zdfddm rn_hsbfr: salt-fingering cutoff density ratio "
                        "R_c [1] (NEMO default 1.6).  None keeps the default.")
    p.add_argument("--barotropic-solver", default=None, choices=[None,"explicit_substep","implicit_cn"],
                   help="Override barotropic solver. NEMO uses split-explicit forward-backward "
                        "(=explicit_substep here, with a dissipative cosine time filter); OMIP "
                        "default is implicit_cn (Crank-Nicolson, NEUTRAL -- no fast-gravity-wave damping).")
    p.add_argument("--barotropic-diffusion-alpha", type=float, default=None,
                   help="Barotropic 2D Laplacian damping coefficient (explicit_substep).")
    p.add_argument("--n-barotropic-substeps", type=int, default=None,
                   help="Number of barotropic substeps (explicit_substep).")
    p.add_argument("--barotropic-time-filter", default=None, choices=[None,"box","cosine"],
                   help="Barotropic time-average filter (cosine = more dissipative for fast modes).")
    p.add_argument("--barotropic-pcg-variant", default=None,
                   choices=[None, "standard", "single_reduce"],
                   help="Implicit-CN barotropic PCG reduction strategy "
                        "(latlon_bathy / tripole / mpas; implicit_cn solver "
                        "path only — explicit_substep ignores it upstream by "
                        "config contract). 'standard' = 2 sequential global "
                        "reductions/iter; 'single_reduce' = Chronopoulos-Gear, "
                        "ONE batched reduction/iter — the reduction-latency "
                        "lever for small per-rank tiles / high rank counts / "
                        "multi-node (equivalent in exact arithmetic, differs "
                        "at round-off; solver-tolerance lane, not bit-exact). "
                        "None keeps the config default ('standard').")
    p.add_argument("--momentum-rk3", action="store_true",
                   help="Use 3-stage SSP-RK3 for the outer baroclinic momentum step "
                        "(mirrors NEMO's RK3 / key_RK3) instead of forward-Euler -- the "
                        "NEMO-faithful fix for the cold-start adjustment blowup. 3x tendency cost.")
    p.add_argument("--freeze-floor", action="store_true",
                   help="Floor ocean T at the seawater freezing point (~-1.8 C) each "
                        "step -- a sea-ice thermodynamic surrogate. legoESM has no "
                        "prognostic ice, so high-lat (esp. Arctic) cells over-cool "
                        "3-5 C below NEMO (LIM ice caps SST). NEMO-faithful; removes "
                        "~half the Arctic SST RMSE. Off = bit-exact legacy.")
    p.add_argument("--freeze-scheme", type=str, default="constant",
                   choices=sorted(VALID_FREEZE_SCHEMES),
                   help="Seawater freezing-point (liquidus) scheme (MED-1) for the "
                        "freeze surrogates: 'constant' (default, byte-exact -1.8 C), "
                        "'linear_S' (MOM6 linear liquidus), 'unesco' (UNESCO/Millero, "
                        "NEMO eos_fzp EOS-80: ~-1.92 C at S=35). Non-constant makes "
                        "BOTH the --freeze-floor model clamp AND the --ice-thermo "
                        "under-ice relaxation track the LOCAL surface salinity "
                        "(Arctic-relevant: fresher shelf water freezes warmer). "
                        "Requires --freeze-floor and/or --ice-thermo (else no "
                        "consumer -> hard error).")
    p.add_argument("--ice-ocean-heat-coeff", type=float, default=None,
                   help="Ocean->ice basal turbulent heat-transfer coefficient "
                        "[W/m^2/K] for --prognostic-sea-ice "
                        "(SeaIceConfig.ocean_heat_transfer_coeff; default 20). "
                        "The Antarctic-melt driver probe (2026-07-28) measured "
                        "the constant 20 at 3-5x below NEMO's u*-dependent MIZ "
                        "exchange — ~60-80 approximates NEMO's summer "
                        "marginal-ice-zone melt rate pending the faithful "
                        "u*-dependent scheme.")
    p.add_argument("--prognostic-sea-ice", action="store_true",
                   help="Wire legoESM's REAL prognostic sea-ice model "
                        "(legoesm.ice.step_sea_ice: thermo + dynamics + brine) into "
                        "the run, REPLACING the freeze-floor / prescribed-siconc / "
                        "ice-thermo-relaxation surrogates.  The ice tile's brine-"
                        "rejection salt flux + melt/freeze freshwater + ocean heat "
                        "extraction are routed into the EXISTING surface_forcing "
                        "(salt_flux, q_net) and freshwater (ice_fw) channels — so "
                        "sea-ice salt/FW export balances Arctic river runoff (the "
                        "missing reservoir behind the Arctic SSS crash).  Cold start "
                        "(zero ice; spins up).  Requires --woa-init and a grid that "
                        "supports ice dynamics (mpas primary; tripole / latlon).  "
                        "Mutually exclusive with the ice surrogates.")
    p.add_argument("--prognostic-ice-dynamics", default="mevp",
                   choices=["mevp", "evp", "free_drift"],
                   help="Sea-ice rheology for --prognostic-sea-ice (default mevp). "
                        "'mevp'/'evp' need the grid strain-rate operators; falls "
                        "back to 'free_drift' automatically if the grid lacks them "
                        "(NOT 'none' — that gives no ice drift/export).")
    p.add_argument("--prognostic-ice-salinity", type=float, default=None,
                   help="Bulk salinity of newly-frozen lead/basal ice [PSU] for the "
                        "--prognostic-sea-ice brine closure (BrineConfig.S_ice_new; "
                        "default constants.S_ice_bulk_default ~4 PSU).")
    p.add_argument("--ice-init", type=str, default=None,
                   help="NEMO SI3 ice initial-state file (Ice_initialization.nc: "
                        "at_i/ht_i[/ht_s/sm_i/tmsu]) — start --prognostic-sea-ice "
                        "from NEMO's Jan-1 ice cover instead of the zero-ice cold "
                        "start.  Native eORCA1 embeds exactly on the tripole grid; "
                        "other grids regrid nearest-wet (same convention as "
                        "--nemo-monthly-init).  Fixes the Arctic brine bias: "
                        "freezing ~1.5 m of NEW ice in 90 d over 10-20 m Siberian "
                        "shelf columns injects +1.2..+3.5 PSU brine that NEMO "
                        "(starting WITH that ice) never sees.  Default None = "
                        "byte-identical zero-ice cold start.")
    p.add_argument("--ice-categories", type=int, default=1,
                   help="Number of sea-ice thickness categories for "
                        "--prognostic-sea-ice (default 1 = single-category, "
                        "byte-identical to prior runs).  >= 2 runs the "
                        "multi-category ITD with the tracer-aware Lipscomb "
                        "(2001) incremental remap (itd_remap='lipscomb2001', "
                        "set automatically: the 'simple' linear remap cannot "
                        "carry the brine/snow tracers this runner always "
                        "enables).  CICE-standard is 5.  --ice-init seeds the "
                        "ITD by placing each cell's aggregate ice in the bin "
                        "containing its thickness.")
    p.add_argument("--ice-ridging", action="store_true",
                   help="Enable mechanical ridging (Lipscomb 2007 "
                        "participation/redistribution) for --prognostic-sea-ice. "
                        "Requires --ice-categories >= 2 (ridging moves ice "
                        "between thickness bins) and a grid with strain-rate "
                        "operators (mpas, latlon; NOT tripole) — both checked "
                        "up front, refused with an actionable error rather "
                        "than silently ignored.")
    p.add_argument("--visc-schedule", type=str, default=None,
                   help="Piecewise viscosity schedule 'day:A_h:C_smag_lap,...'"
                        " e.g. '0:1e5:3.0,90:5e4:1.0,180:2e4:0.33' — start at "
                        "the cold-start-stable values, step down toward "
                        "NEMO's eddy-viscosity magnitude (1e3-2e4) once the "
                        "WOA adjustment has passed. One JIT recompile per "
                        "segment. tripole/latlon (LatLonCGridOceanModel).")
    p.add_argument("--bbl-adv", action="store_true",
                   help="NEMO advective bottom-boundary layer (trabbl "
                        "nn_bbl_adv=2, Campin & Goosse 1999): dense shelf "
                        "bottom water advects DOWN the continental slope when "
                        "denser than the deep neighbour (Gibraltar/Med, "
                        "Denmark Strait, Antarctic overflows — unresolved at "
                        "1 deg without it). Host post-step exchange, exactly "
                        "tracer-conserving. latlon/tripole only.")
    p.add_argument("--bbl-gamma-s", type=float, default=20.0,
                   help="Advective-BBL coefficient gamma [s] (NEMO "
                        "rn_gambbl=20).")
    p.add_argument("--runoff-depth-spread-m", type=float, default=None,
                   help="Spread river runoff dilution over the top this-many "
                        "metres (NEMO sbcrnf rn_dep_max=150) instead of a "
                        "single surface cell — fixes the too-fresh/too-shallow "
                        "Amazon-type plume. Column-integral salt unchanged. "
                        "Default None = legacy top-cell (bit-exact).")
    p.add_argument("--runoff-spread-passes", type=int, default=None,
                   help="HORIZONTAL coastal-spread passes for the regridded "
                        "runoff (ocean-masked neighbour-average; Voronoi-topology "
                        "on MPAS, structured laplacian elsewhere). Default None = "
                        "8 on MPAS (big rivers over-concentrate in ~4 IDW cells; "
                        "spread-passes sensitivity-tuned), "
                        "2 on lat-lon/cube. The area-conservative renorm keeps the "
                        "global total exact. Distinct from --runoff-depth-spread-m "
                        "(VERTICAL spread).")
    p.add_argument("--river-mouth-restoring-gate", action="store_true",
                   help="Disable SSS restoring at river-mouth cells (runoff > "
                        "threshold) so the restoring does not fight the river "
                        "plume toward the coarse WOA climatology. NOTE this is "
                        "a legoESM DEVIATION, not NEMO parity: sbcssr's "
                        "(1-2*rnfmsk) mask is INACTIVE in the ORCA1 deck "
                        "(ln_rnf_mouth defaults .false., so rnfmsk==0 and the "
                        "factor is 1) — NEMO restores FULLY at river mouths "
                        "there. Requires --runoff + --sss-restore.")
    p.add_argument("--ew-cyclic-overlap", action="store_true",
                   help="TRIPOLE ONLY: reconnect the ORCA east-west cyclic seam "
                        "(lon ~72.5E on eORCA1). The eORCA1 mesh marks the 2 cyclic "
                        "halo columns LAND, and the C-grid roll is period-nx (off by "
                        "one for an ORCA period-(nx-2) grid) -> a spurious wall + a "
                        "drifting ~1 C SST/SSS seam stripe. Slaves the halo columns to "
                        "their overlap partners each step (col0<-col[nx-2], "
                        "col[nx-1]<-col1) + overlap-fills the mask/bathy/IC. "
                        "ORCA-overlap-specific; do NOT use on a regular lat-lon grid. "
                        "Off = bit-exact legacy.")
    p.add_argument("--polar-filter", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Enable/disable the mask-aware Fourier polar filter (lat-lon "
                        "grid only). Omitted -> the config default applies (the global "
                        "bathy path forces it ON, #939); --no-polar-filter forces it "
                        "OFF (for the #939 A/B or an emergency disable). "
                        "truncate the zonal modes exceeding the per-latitude CFL near the "
                        "converging-meridian poles, where a global lat-lon ocean otherwise "
                        "blows up ~day 0.25. Mask-aware (land filled with ocean zonal mean "
                        "before the FFT, restored after) so continents are not smeared into "
                        "ocean; tracer-conservative (k=0 mode kept). NOTE: lat-lon stays "
                        "imperfect in the land-locked Arctic -- ORCA tripole is the faithful "
                        "path; this is for lat-lon stability + a tropics/mid-lat/SH compare.")
    p.add_argument("--polar-filter-cutoff-lat", type=float, default=None,
                   help="Latitude (deg) poleward of which the polar filter acts (default 60).")
    p.add_argument("--polar-filter-max-wave-speed", type=float, default=None,
                   help="Max wave speed [m/s] setting the CFL wavenumber cap (default 300).")
    p.add_argument("--polar-filter-safety", type=float, default=None,
                   help="Fraction of the CFL wavenumber kept, <1 for margin (default 0.85).")
    p.add_argument("--runoff", action="store_true",
                   help="Apply NEMO's Dai-Trenberth river+ice-shelf+iceberg runoff "
                        "(the SAME file ORCA1 uses) as a per-step freshwater/virtual-salt "
                        "flux -> ungates the SSS comparison (tripole/latlon only).")
    p.add_argument("--no-emp", dest="emp_freshwater", action="store_false",
                   default=True,
                   help="DISABLE the atmospheric P - E surface freshwater flux "
                        "(default ON).  P - E = precip + lh/L_v is the dominant "
                        "OMIP-2 salt-budget term; ON by default so the multi-year "
                        "salinity is faithful.  Use --no-emp only for ablation "
                        "(reproducing the pre-fix runoff-only fresh drift).")
    p.add_argument("--ice-albedo", action="store_true",
                   help="Apply a NEMO-siconc-weighted sea-ice + open-ocean SW "
                        "albedo to the downwelling shortwave (closes the SH/Antarctic "
                        "warm bias: the ocean previously absorbed ~100%% of SW with "
                        "no albedo). Prescribed (annual NEMO siconc) -> feedback-safe.")
    p.add_argument("--siconc-file", type=str, default=None,
                   help="Override the NEMO sea-ice-concentration file for "
                        "--ice-albedo (default = the ORCA1 RUN_REF annual icemod.nc).")
    p.add_argument("--ice-albedo-seasonal", action="store_true",
                   help="Give the --ice-albedo siconc a 12-MONTH seasonal cycle "
                        "(via NEMO's monthly SST, since the run wrote no monthly "
                        "icemod): removes the spurious year-round summer albedo in "
                        "NH seasonal-ice zones that drives the NH cold bias (codex "
                        "HIGH). Indexed by calendar month every step. The same "
                        "monthly siconc also gates SSS restoring (NEMO nn_sssr_ice=0).")
    p.add_argument("--tos-monthly-file", type=str, default=None,
                   help="Override the NEMO monthly grid_T (tos) file used to build "
                        "the seasonal siconc (default = ORCA1 RUN_REF 1m grid_T.nc).")
    p.add_argument("--ice-thermo", action="store_true",
                   help="Prescribed-ice THERMODYNAMIC boundary (codex HIGH): under "
                        "sea ice, cut SW reaching the ocean (--ice-thermo-sw-trans) "
                        "+ suppress turbulent/LW by (1-sic) + relax the surface "
                        "ocean toward freezing (--ice-thermo-tau-days). Two-sided -> "
                        "cools the over-warm Southern-Ocean under-ice cells (the "
                        ">45S warm bias) AND holds the Arctic near freezing, while "
                        "open water keeps the seasonal-albedo NH warming. Uses the "
                        "same prescribed siconc as --ice-albedo (implies a siconc).")
    p.add_argument("--ice-thermo-tau-days", type=float, default=20.0,
                   help="Under-ice freezing-relaxation timescale [days] (default 20; "
                        "physical range 5-30).")
    p.add_argument("--ice-thermo-sw-trans", type=float, default=0.03,
                   help="Fraction of downwelling SW transmitted through ice into the "
                        "ocean (default 0.03; the albedo-only surrogate implies 0.35).")
    p.add_argument("--sss-restore", action="store_true",
                   help="Apply OMIP-2 weak SSS restoring toward the WOA surface "
                        "salinity (the protocol NEMO ORCA1 uses) -> bounds the "
                        "multi-year surface-freshwater drift (5-yr MPAS drifted "
                        "35->31 PSU without it). Requires --woa-init. Interior tau "
                        "via --sss-restore-tau-days; marginal seas use shorter "
                        "built-in OMIP-2 regional taus.")
    p.add_argument("--sss-restore-tau-days", type=float, default=365.0,
                   help="Interior SSS-restoring timescale [days] (default 365 = "
                        "OMIP-2 interior; regional Arctic/Med/SO use shorter "
                        "built-in taus). NEMO ORCA1 RUN_REF equivalent: piston "
                        "-220 mm/day over the 10 m top layer = tau ~45.5 d.")
    p.add_argument("--sss-restore-channel", default=None,
                   choices=["tracer", "water_flux"],
                   help="How SSS restoring reaches the ocean (tripole/latlon). "
                        "Unset = 'tracer' (default, bit-identical to earlier "
                        "runs): a post-step salinity edit, which is a "
                        "virtual-salt-like operation that moves no water and "
                        "carries no heat. 'water_flux' is NEMO nn_sssr=2: the "
                        "flux enters the freshwater budget (so it drives eta / "
                        "z-star dilution, NEMO sshwzv.F90:123) and carries "
                        "qns -= erp*rcp*sst_m (sbcssr.F90:138). The post-step "
                        "edit is then SKIPPED -- the two channels are "
                        "exclusive, since running both applies restoring "
                        "twice. Requires --sss-restore-normalization live_s, "
                        "NEMO's own conversion for a real water flux.")
    p.add_argument("--sss-restore-normalization", default=None,
                   choices=["s_target", "live_s"],
                   help="Denominator of the SSS-restoring salinity->freshwater "
                        "conversion. Unset keeps the card value ('s_target', "
                        "bit-identical to earlier runs). 'live_s' is NEMO "
                        "sbcssr nn_sssr=2 (sbcssr.F90:132-134), which ORCA1 "
                        "runs: zerp = zsrp*coefice*(sss_m - sss_target)/"
                        "MAX(sss_m,1e-20) -- divided by the LIVE surface "
                        "salinity. Under 'live_s' the restoring is a genuine "
                        "water flux, which is what makes it compatible with "
                        "--freshwater-closure real_freshwater.")
    p.add_argument("--sss-restore-bound-mmday", type=float, default=None,
                   help="Bound |restoring FW flux| at this mm/day-equivalent "
                        "(NEMO ln_sssr_bnd: rn_sssr_bnd=4.0 in the ORCA1 "
                        "reference). Default None keeps the loose 200 mm/day "
                        "safety cap. The bound is what lets a SHORT tau hold "
                        "SSS without injecting deep-convection-killing salt "
                        "spikes (the tau=60 AMOC-collapse mechanism).")
    p.add_argument("--sw-rgb-chl", action="store_true",
                   help="Use NEMO's RGB chlorophyll shortwave-penetration scheme "
                        "(ln_qsr_rgb) instead of the uniform 2-band Jerlov default: "
                        "IR + R/G/B bands with chlorophyll-dependent extinction from "
                        "the monthly ESACCI Chl climatology (Morel-Berthon vertical "
                        "profile). Clear subtropical water penetrates deeper (cools "
                        "the surface warm bias); productive subpolar water traps light "
                        "near the surface. latlon/tripole only (wired into the PE "
                        "C-grid step). --chl-file overrides the default ESACCI path.")
    p.add_argument("--chl-file", type=str, default=None,
                   help="Override path to the monthly ESACCI chlorophyll NetCDF "
                        "(default: NEMO ORCA1 INPUTS merged_ESACCI...CHL_REG05.nc).")
    p.add_argument("--sss-ice-gate-nemo", action="store_true",
                   help="Use NEMO's exact under-ice SSS-restoring law (sbcssr "
                        "nn_sssr_ice=0: coefice = 1 - fr_i, zero under full ice) "
                        "instead of the legoESM tanh cutoff. Requires --sss-restore.")
    p.add_argument("--sss-restore-file", type=str, default=None,
                   help="NEMO-native monthly SSS-restoring climatology "
                        "(sn_sss, e.g. sss_climatology_for_restoring.nc, "
                        "presalt 12 x y x x). Replaces the IC-surface "
                        "restoring target with NEMO's own product — the "
                        "IC-surface target holds Arctic shelves several PSU "
                        "too salty. Requires --sss-restore.")
    p.add_argument("--nemo-monthly-init", nargs=2, default=None,
                   metavar=("TEMP_NC", "SALT_NC"),
                   help="Initialise T/S from NEMO's monthly init files "
                        "(sn_tem/sn_sal, woce_*_monthly_init_4p2.nc) at "
                        "--nemo-init-month, replacing the (annual) --woa-init "
                        "T/S AFTER the state build. nemolev ladders only "
                        "(75 levels, no vertical interpolation).")
    p.add_argument("--nemo-init-month", type=int, default=1,
                   help="Month (1-12) of --nemo-monthly-init to use "
                        "(default 1 — a 1 January cold start).")
    p.add_argument("--runoff-depth-nemo-ini", action="store_true",
                   help="NEMO ln_rnf_depth_ini: per-cell runoff spread depth "
                        "proportional to the local climatological runoff max "
                        "(h = 150 m * rnf_max/0.05, floor 1 m, capped at the "
                        "local depth) instead of the flat "
                        "--runoff-depth-spread-m. Requires --runoff; mutually "
                        "exclusive with --runoff-depth-spread-m.")
    p.add_argument("--runoff-dep-max", type=float, default=None,
                   help="Override NEMO rn_dep_max [m] (default 150) used by "
                        "--runoff-depth-nemo-ini: the depth the biggest rivers "
                        "spread over. LOWER keeps river plumes shallower -> more "
                        "surface freshening (Arctic-shelf retention sensitivity: "
                        "the big Siberian rivers otherwise dilute deep -> salty).")
    p.add_argument("--runoff-rnf-max", type=float, default=None,
                   help="Override NEMO rn_rnf_max [kg/m^2/s] (default 0.05) used "
                        "by --runoff-depth-nemo-ini: the climatological runoff at "
                        "which the spread depth reaches rn_dep_max.")
    p.add_argument("--woa-smoothing-passes", type=int, default=0,
                   help="Horizontal Laplacian smoothing passes/level on the WOA T,S IC "
                        "-- removes spurious grid-scale fronts from interpolating/flood-"
                        "filling WOA onto the tripole that imply >>2 m/s geostrophic flow "
                        "and break the cold start. Requires --woa-init. 0=off.")
    p.add_argument("--balanced-init", action="store_true",
                   help="Initialise the cold-start in geostrophic/thermal-wind balance "
                        "(level-of-no-motion velocity from the WOA p' field, equator-tapered) "
                        "+ balanced SSH -- removes the unbalanced-rest adjustment shock that is "
                        "the conclusively-diagnosed cause of the WOA cold-start blowup. "
                        "Requires --woa-init.")
    p.add_argument("--balanced-init-taper-lat", type=float, default=8.0,
                   help="Equatorial taper latitude [deg] for the geostrophic init: the "
                        "1/f singularity is regularised as f/(f^2+f_eps^2) with "
                        "f_eps=2*Omega*sin(taper_lat), so u_g,v_g taper to 0 within ~this "
                        "latitude of the equator.")
    p.add_argument("--no-balanced-ssh", action="store_true",
                   help="With --balanced-init, set only the geostrophic velocity (surface-"
                        "referenced inconsistency control); default also sets the balanced SSH.")
    p.add_argument("--bathy-smoothing-passes", type=int, default=0,
                   help="Laplacian smoothing passes on the OCEAN bathymetry before "
                        "building the partial-cell coord -- reduces the bathymetric "
                        "slope (r-factor) and hence the spurious partial-cell PGF seed "
                        "at steep continental slopes (NEMO/ROMS smooth for this). "
                        "Requires --partial-cell. 0=off.")
    p.add_argument("--output", type=str, default="results/omip_nemo/legoesm_tripole")
    p.add_argument("--config", type=str, default=None,
                   help="Ocean experiment YAML (legoesm.ocean.config."
                        "OceanExperimentConfig). Its ocean.* fields are applied "
                        "onto the built config (lat-lon C-grid grids only) and a "
                        "run_manifest.json (resolved config + config_hash + final "
                        "state_digest) is written under --output for "
                        "`legoesm reproduce`. The mesh/grid still come from "
                        "--grid/--mesh. See issue #376.")
    p.add_argument("--forcing-path", type=str, default=None,
                   help="Directory containing the CORE-II NYF zarr (nyf.zarr) — "
                        "passed as load_core2_nyf(cache_dir=...). Lets a "
                        "fetch-then-run workflow point the loader at staged data "
                        "instead of the default ~/.cache/.../core2_nyf. Set via "
                        "--config forcing.path. See issue #376.")
    p.add_argument("--diag-every-days", type=float, default=30.0)
    p.add_argument("--diag-momentum-step", type=int, default=-1,
                   help="DEBUG (lat-lon/tripole): at every step <= this, dump the "
                        "per-term momentum-tendency breakdown (PGF/Coriolis/advection/"
                        "viscosity/...) global-max + location, to pin the term driving "
                        "a cold-start blowup. -1=off.")
    p.add_argument("--prescribed-flow", choices=("zero", "frozen"), default=None,
                   help="Vertical-physics isolation (IN-MODEL lever: threads "
                        "LatLonCGridOceanConfig.prescribed_flow, pinned inside "
                        "the step where the tracer mass fluxes are built) -- "
                        "'zero' (u=v=eta=0, mass fluxes and w vanish: pure "
                        "column physics on the full grid) or 'frozen' (tracers "
                        "advected by the step-entry flow; u/v/eta never "
                        "evolve). T/S still evolve by vertical mixing, "
                        "convection, surface fluxes, penetrating SW and "
                        "restoring, so vertical-structure changes are isolated "
                        "from circulation feedback. tripole/latlon_bathy only; "
                        "scan-block compatible; rejects --spinup-drag-tau-days; "
                        "REQUIRES --no-gm-redi (GM bolus = parameterized "
                        "advection the pin cannot isolate).")
    p.add_argument("--no-gm-redi", action="store_true",
                   help="Disable GM/Redi isopycnal mixing (tripole recipe and "
                        "latlon_bathy production config ship it ON: kappa=600 "
                        "tripole, Visbeck _DEFAULT_BATHY_GM_REDI latlon). "
                        "REQUIRED with --prescribed-flow: the GM bolus (skew) "
                        "transport is parameterized tracer ADVECTION applied "
                        "outside the pinned mass-flux block, so the model "
                        "constructor rejects prescribed_flow+GM/Redi.")
    p.add_argument("--scan-block", type=int, default=0,
                   help="Issue #354: wrap the time loop in jax.lax.scan, "
                        "fusing this many steps per block (CORE-II forcing "
                        "sampled on-device, no per-step host roundtrip). "
                        "0 (default) = the bit-identical Python loop. "
                        "Tripole only; incompatible with --nudge-woa / "
                        "--spinup-drag. Diagnostics run at block boundaries.")
    p.add_argument("--snapshot-every-days", type=float, default=0.0,
                   help="Write a state snapshot every N sim-days (in addition "
                        "to yearly + final). 0=off. Lets a long run be scored "
                        "mid-flight (e.g. day-30 SST vs NEMO) without waiting "
                        "for the full integration.")
    p.add_argument("--restart-save", type=str, default=None,
                   help="Write a RESUMABLE restart to this .npz path (atomic "
                        "overwrite) at the --snapshot-every-days cadence and "
                        "at the end of the run. Unlike snapshot_*.npz (a "
                        "diagnostic artifact) this carries EVERY prognostic "
                        "and integrator-carry slot, the UNMANGLED sea-ice "
                        "state, and the absolute step counter — so a 72 h job "
                        "chain integrates forward instead of re-paying the "
                        "cold-start spin-up each time. Cadence: "
                        "--restart-every-days (defaults to "
                        "--snapshot-every-days); always also written at the "
                        "end of the run.")
    p.add_argument("--restart-every-days", type=float, default=0.0,
                   help="Cadence for --restart-save [sim-days]. 0 (default) "
                        "follows --snapshot-every-days. With BOTH at 0 the "
                        "restart is written only at the end of the run, so a "
                        "wallclock kill loses the whole leg — the driver warns "
                        "when that is the case.")
    p.add_argument("--restart-from", type=str, default=None,
                   help="Resume from a --restart-save archive: the ocean "
                        "carry, the sea-ice state and the step counter are "
                        "restored, and the loop continues to the ABSOLUTE "
                        "--years target (it does not re-run the completed "
                        "steps). Grid-type, dt, forcing-record count, x64 and "
                        "a digest of the RESOLVED configuration must match "
                        "(hard errors); a source-revision change only WARNS, "
                        "since chaining across a bug fix is a supported "
                        "workflow. SCOPE: this is a guarded RECOVERY resume "
                        "that catches configuration drift, archive truncation/"
                        "corruption and STRUCTURAL tampering (a renamed, "
                        "relabelled or removed slot) — NOT a cryptographically "
                        "strict or bit-identical continuation. There is no "
                        "payload checksum, so an edit to an array's VALUES at "
                        "the same shape resumes silently, and forcing/mesh "
                        "inputs are pinned by PATH, not by content hash.")
    p.add_argument("--kprofile-snapshots", action="store_true",
                   help="Store the vertical viscosity and diffusivity the "
                        "implicit solve consumes (K_M_diag/K_H_diag, with "
                        "z_interface_ref) in each day/final snapshot of the "
                        "standard loop. NEMO publishes avm/avt, so this makes "
                        "the turbulent Prandtl number comparable against the "
                        "oracle instead of inferred from TKE. The TKE closure "
                        "returns no K on the tendency (the solve builds it), "
                        "so this uses the model's diagnose_vertical_K. Costs "
                        "one extra solve-setup per snapshot, changes no "
                        "prognostic field, and is refused with --scan-block "
                        "(that lane does not thread the dump).")
    p.add_argument("--forcing-ramp-days", type=float, default=0.0,
                   help="Ramp the surface forcing 0->full over N days "
                        "(cold-start shock mitigation).")
    p.add_argument("--min-levels", type=int, default=1,
                   help="rn_hmin-style conditioning: mask ocean columns with "
                        "fewer than this many active reference levels (partial-"
                        "cell path only). 1=off. Use 2 at 1/4 deg to remove the "
                        "single-thin-layer coastal cells that seed a 1/h blowup.")
    p.add_argument("--convection", type=str, default="none",
                   choices=["none", "enhanced_diffusion"],
                   help="Grid-agnostic convective adjustment (Oceananigans-"
                        "style enhanced vertical diffusivity where N^2<0), "
                        "applied via the implicit backward-Euler vertical "
                        "solve. Default 'none' preserves the validated "
                        "faithful config (tripole base ships physics=None).")
    p.add_argument("--convection-K-conv", type=float, default=1.0,
                   help="Convective diffusivity K_conv [m^2/s] for "
                        "--convection enhanced_diffusion (default 1.0).")
    p.add_argument("--convection-K-bg", type=float, default=1e-5,
                   help="Background diffusivity K_bg [m^2/s] for convection.")
    p.add_argument("--kpp-ri-crit", type=float, default=None,
                   help="Override the KPP critical bulk Richardson number "
                        "(default 0.3 = LMD94/MOM6). RAISING it deepens the "
                        "boundary layer (mpas: fix subtropical MLD ~half of "
                        "NEMO); LOWERING it shoals it (latlon_bathy: fix the "
                        "too-deep JANUARY winter ML, 174 m vs NEMO 93 m, that "
                        "cools the 100 m mode water). --grid mpas/latlon_bathy "
                        "only (both run KPP); tripole has no KPP boundary layer.")
    p.add_argument("--kpp-cv", type=float, default=None,
                   help="Override the KPP unresolved-shear coefficient Cv "
                        "(default 1.6). RAISING it increases V_t^2 -> deeper "
                        "boundary layer, LOWERING it shoals it (same MLD lever "
                        "as --kpp-ri-crit). --grid mpas/latlon_bathy only.")
    p.add_argument("--kpp-eice", type=int, default=None, choices=[0, 1, 3],
                   help="Under-ice attenuation of the KPP turbulent velocity "
                        "scales (KPP w-scale analogue of ice suppression; "
                        "mirror of --tke-eice). Compact ice scales w_m/w_s by "
                        "(1-eff) so BOTH the boundary-layer depth and mixing "
                        "shrink under ice. None/0 (default) = off; 1 = legoESM "
                        "linear (1-fi) [NOT NEMO nn_eice=1]; 3 = max(0,1-4*fi) "
                        "(matches NEMO nn_eice=3, killed at fi>=0.25). The KPP grids' "
                        "Arctic halocline-erosion lever (over-deep MLD + "
                        "Siberian salty) that --tke-eice fixed only on the TKE "
                        "grid. Needs --prognostic-sea-ice or a prescribed SIC. "
                        "--grid latlon_bathy (MPAS KPP bridge has no ice yet).")
    p.add_argument("--mpas-vmix", type=str, default="kpp",
                   choices=["kpp", "tke"],
                   help="Vertical-mixing CLOSURE on the MPAS Voronoi grid. "
                        "'kpp' (default) = the historical MPAS KPP boundary "
                        "layer (with --kpp-ri-crit/--kpp-cv). "
                        "'tke' = the NEMO ORCA1 &namzdf_tke card "
                        "(orca1_zdftke_config: lc, etau nn_htau=1 latitude "
                        "profile, under-ice eice from the card's nn_eice=3) "
                        "through the SAME grid-agnostic kernel the tripole "
                        "runs — the scheme-matching lever for the cross-grid "
                        "Arctic gap (2x2: TKE-vs-KPP = +0.78 Siberian / "
                        "+36-50 m MLD of the tripole<->MPAS disagreement).")
    p.add_argument("--tripole-vmix", type=str, default="none",
                   choices=["none", "tke", "kpp"],
                   help="Vertical-mixing CLOSURE on the tripole grid (the "
                        "last audited NEMO ORCA1 namelist gap: NEMO runs "
                        "zdftke; the faithful tripole default runs only the "
                        "dynamics-core implicit backgrounds). 'tke' = NEMO "
                        "&namzdf_tke mapped from the ORCA1 namelist "
                        "(rn_ediff/rn_ediss/rn_emin/rn_emin0/nn_pdl/ln_lc/"
                        "rn_lc=0.25/nn_etau=1/rn_efr=0.08/nn_htau=1); 'kpp' = "
                        "KPP defaults. Composes ADDITIVELY with --iwm "
                        "(NEMO zdfphy order: closure first, zdfiwm adds on "
                        "top). Default 'none' is byte-identical. "
                        "STABILITY: TKE x superbee tracer advection blew up "
                        "on DINO in ~15 days — smoke-gate before long runs.")
    p.add_argument("--tke-eice", type=int, default=None, choices=[0, 1, 3],
                   help="Under-ice attenuation of the TKE lc/etau wave "
                        "sources (NEMO nn_eice) for --tripole-vmix tke. "
                        "None (default) keeps the ORCA1 card value (3 = "
                        "max(0,1-4*fi), wave TKE killed at fi>=0.25); 1 = "
                        "(1-fi); 0 = no attenuation (reproduces the "
                        "pre-2026-07-18 behaviour for A/B). The ice "
                        "concentration reaches the closure via "
                        "surface_forcing.ice_concentration under "
                        "--prognostic-sea-ice.")
    p.add_argument("--tke-surface-bc", type=str, default=None,
                   choices=["veros_flux", "nemo_dirichlet"],
                   help="Surface-TKE boundary condition for --tripole-vmix "
                        "tke. None (default) keeps the card default "
                        "(nemo_dirichlet since #1326: NEMO's en(1)="
                        "max(rn_emin0, rn_ebb*|tau|/rho0), rn_ebb=67.83); "
                        "'veros_flux' selects the Veros flux form "
                        "(|tau|/rho0)^{3/2} (the pre-#1326 behaviour, for "
                        "A/B). Requires --tripole-vmix tke (else raises).")
    p.add_argument("--tke-lc", type=str, default=None, choices=("on", "off"),
                   help="Langmuir cell parameterisation in the tripole TKE "
                        "card (NEMO ln_lc). None keeps the ORCA1 card (on). "
                        "'off' exists for the fesom-mimic card: fesom-jax's "
                        "CVMix TKE has no Langmuir term.")
    p.add_argument("--tke-etau", type=str, default=None,
                   choices=("below_ml", "none"),
                   help="Surface-TKE penetration mode (NEMO nn_etau). None "
                        "keeps the ORCA1 card (below_ml). 'none' for the "
                        "fesom-mimic card (fesom-jax has no etau term).")
    p.add_argument("--tke-shear-production", type=str, default=None,
                   choices=["squared_centered", "nemo_face_native",
                            "nemo_face_native_now2",
                            "nemo_burchard"],
                   help="TKE shear-production discretisation for "
                        "--tripole-vmix tke. None (default) keeps the card "
                        "value ('squared_centered': velocities averaged to "
                        "cell centres, then differenced). 'nemo_face_native' "
                        "is NEMO's zdf_sh2 -- face-native differences, a now "
                        "x before velocity product, and production DOUBLED "
                        "adjacent to coasts via (2 - umask*umask) "
                        "(zdfsh2.F90:78-94). Averaging before differencing "
                        "SMOOTHS, so the default is systematically weaker "
                        "than NEMO's; this is the last unclosed gap on the "
                        "ORCA1 card and the only one an offline column probe "
                        "cannot evaluate, because it needs the raw C-grid "
                        "face state. Requires --partial-cell (k_profiles "
                        "builds wumask/wvmask/coast masks from "
                        "z_coord.is_active) and --tripole-vmix tke.")
    p.add_argument("--tke-kappa-convention", type=str, default=None,
                   choices=["veros_sqrte", "gaspar_sqrt2e"],
                   help="Amplitude of K from TKE for --tripole-vmix tke. "
                        "None (default) keeps the card value ('veros_sqrte' "
                        "= NEMO's avm = rn_ediff*zmxlm*sqrt(en), zdftke.F90:"
                        "150/553). 'gaspar_sqrt2e' restores the legacy "
                        "c_k*l_k*sqrt(2*e) amplitude, which DOUBLE-COUNTS the "
                        "sqrt(2) already carried by the nn_mxl=2/3 buoyancy "
                        "length sqrt(2e)/N and measured 4.81x NEMO's avt "
                        "against 3.25x for the NEMO form (Stage A, 39ce0701c) "
                        "-- supply it only to reproduce arms that predate the "
                        "fix. Requires --tripole-vmix tke (else raises).")
    p.add_argument("--tke-n2-mode", default=None,
                   choices=["insitu", "insitu_signed", "adiabatic",
                            "nemo_bn2"],
                   help="Override the stratification the ORCA1 zdftke card "
                        "feeds its closure. The card selects 'nemo_bn2' "
                        "(NEMO's own eosbn2 assembly, what ORCA1 runs). "
                        "'insitu' reverts ONLY that, which is the "
                        "one-variable control: the in-situ density gradient "
                        "carries a +g^2/c^2 = 4.27e-5 s^-2 compressibility "
                        "bias the adiabatic form does not. Requires --grid "
                        "tripole --tripole-vmix tke.")
    p.add_argument("--tke-n2-eos-form", default=None,
                   choices=["seos", "teos10"],
                   help="Which alpha/beta the nemo_bn2 assembly uses. The "
                        "card selects 'teos10' (ORCA1 runs ln_teos10=.true.). "
                        "Inert under every other --tke-n2-mode.")
    p.add_argument("--tke-mxl-choice", type=int, default=None,
                   choices=list(TKE_MXL_CHOICES),
                   help="TKE mixing-length formulation for --tripole-vmix tke. "
                        "None (default) keeps the card value (3 since #1326 = "
                        "NEMO nn_mxl=3: lup/ldown |dl/dz|<=e3t sweeps WITH the "
                        "ln_mxl0 wind-stress surface anchor). 2 = Veros "
                        "Bougeault-Lacarrere (the pre-#1326 behaviour, for "
                        "A/B). 4 = NEMO nn_mxl=2, which is what the ORCA1 "
                        "namelist actually runs: the SAME lup/ldown sweeps as "
                        "3 but a SINGLE length (l_eps = l_k = min(lup,ldn)) "
                        "instead of l_eps = sqrt(lup*ldn). Since min <= sqrt, "
                        "choice 4 dissipates MORE and mixes LESS, and the two "
                        "differ only where lup and ldn diverge (weakly "
                        "stratified deep columns) -- a high-latitude-selective "
                        "lever. NOTE the numbering is Veros-derived and does "
                        "NOT match NEMO's nn_mxl values. Requires "
                        "--tripole-vmix tke (else raises).")
    p.add_argument("--tke-prognostic", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Prognostic TKE for --tripole-vmix tke. Unset (default) "
                        "keeps the card value (PROGNOSTIC Mode-A since #1326: "
                        "NEMO's en integration, one step/model-step carrying "
                        "OceanState.tke). --no-tke-prognostic selects the "
                        "quasi-steady Mode-B diagnostic (3 backward-Euler "
                        "iters/call, ~3x cheaper; the pre-#1326 behaviour, for "
                        "A/B). Requires --tripole-vmix tke (else raises).")
    p.add_argument("--gm-treguier", action="store_true",
                   help="Use the NEMO-faithful flow-dependent GM coefficient "
                        "(TreguierConfig = NEMO &namtra_eiv nn_aei_ijk_t=21: "
                        "aeiu/aeiv = F(growth rate of baroclinic instability), "
                        "capped at aei0) INSTEAD of the tripole default's "
                        "VISBECK adaptive kappa_GM. NEMO ORCA1 runs the former "
                        "(rn_Ue=0.018, rn_Le=100e3, laplacian => "
                        "aei0 = 1/2*rn_Ue*rn_Le = 900 m^2/s); the two "
                        "are mutually exclusive. --grid tripole only.")
    p.add_argument("--gm-slope-scheme", choices=_GM_SLOPE_SCHEMES, default=None,
                   help="Isoneutral-slope operator for GM/Redi (tripole only). "
                        "Unset keeps the recipe's 'centered'. 'nemo_iso_lap' is "
                        "NEMO's STANDARD rotated laplacian, which is what ORCA1 "
                        "runs (namelist_cfg: ln_traldf_lap=.true., "
                        "ln_traldf_iso=.true., ln_traldf_triad=.false.) and "
                        "which carries the akz stabilization of "
                        "ln_traldf_msc=.true.; our default has neither. Required "
                        "for --gm-bolus-advection through_fct.")
    p.add_argument("--gm-bolus-advection", choices=_GM_BOLUS_FORMS, default=None,
                   help="GM eddy-induced (bolus) transport form (tripole only). "
                        "Unset keeps the recipe's 'centred' = a standalone, "
                        "UNLIMITED centred flux. 'through_fct' adds the bolus to "
                        "the tracer advecting mass flux so it rides the monotone "
                        "FCT limiter -- what NEMO does (LDF/ldftra.F90 "
                        "ldf_eiv_trp, called by traadv.F90). Honored ONLY with "
                        "--gm-slope-scheme nemo_iso_lap (the model gates on the "
                        "pair), so passing it alone raises rather than silently "
                        "keeping the centred flux.")
    p.add_argument("--gm-msc-stabilize", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="NEMO ln_traldf_msc (Method of Stabilizing Correction) "
                        "for the isoneutral operator (tripole only). Unset "
                        "keeps the card value (GMRediConfig.msc_stabilize "
                        "defaults False). ORCA1 runs ln_traldf_msc=.true., and "
                        "--gm-slope-scheme nemo_iso_lap does NOT imply it: the "
                        "akz split is gated on this SEPARATE field, so a NEMO "
                        "operator match needs both.")
    p.add_argument("--gm-kappa-min", type=float, default=_GM_KAPPA_MIN_DEFAULT,
                   help="Floor on the Treguier kappa_GM [m^2/s] for "
                        "--gm-treguier. The NEMO tropical taper min(1,|f/f20|) "
                        "sends kappa -> 0 AT THE EQUATOR (2.17%% of eORCA1 wet "
                        "cells measured below taper 0.05), which destabilised a "
                        "1-degree global run; Visbeck carries kappa_min=200 for "
                        "the same reason. NOT NEMO: a nonzero floor is a "
                        "deliberate closure change that keeps a finite bolus "
                        "transport at the equator, so oracle/fidelity runs must "
                        "pass 0 (= raw NEMO capped-only form). Applied before "
                        "the Hallberg resolution scaling, as Visbeck's is.")
    p.add_argument("--gm-aei0", type=float, default=_GM_AEI0_DEFAULT,
                   help="kappa_GM cap [m^2/s] for --gm-treguier = NEMO "
                        "1/2*rn_Ue*rn_Le for the laplacian operator (ORCA1: "
                        "0.5*0.018*100e3 = 900; ldftra.F90:290-293, and NEMO's "
                        "own aeiu_2d maxes at exactly 900). Default 900.")
    p.add_argument("--freshwater-closure", type=str, default=None,
                   choices=["none", "virtual_salt_flux", "real_freshwater"],
                   help="Ocean freshwater closure. 'virtual_salt_flux' "
                        "(current default) applies a virtual salt flux "
                        "-S_ref*F_fw/(rho0*dz0) ON TOP OF the z-star eta "
                        "channel, which already conserves h*S while the "
                        "column stretches -- a spurious salt source NEMO "
                        "does not have under variable volume (measured at "
                        "+10.109 psu.m of excess Arctic salt over 60 d). "
                        "'real_freshwater' keeps the eta/volume channel and "
                        "drops ONLY that virtual-salt term; the genuine "
                        "sea-ice salt flux pathway is unaffected. See "
                        "docs/dev-notes/ocean_real_freshwater_design.md.")
    p.add_argument("--freshwater-salinity", type=str, default="s_ref",
                   choices=["s_ref", "local"],
                   help="Salinity multiplying the freshwater flux in the "
                        "virtual-salt closure. 's_ref' (default) = the fixed "
                        "config S_ref=35 (legacy, bit-identical). 'local' = "
                        "the LOCAL top-cell salinity — NEMO's tra_sbc "
                        "convention (sfx = emp*sss); on fresh shelves "
                        "(Siberian ~27 PSU) the fixed-35 closure "
                        "over-salinifies ice growth by ~1.35x and "
                        "over-dilutes rivers (2026-07-18 Arctic "
                        "halocline-erosion audit). latlon/tripole/mpas.")
    p.add_argument("--gateway-transports", action="store_true",
                   help="Accumulate TIME-MEAN volume and salt transports "
                        "through the Arctic gateways (Bering/Pacific, "
                        "Davis/CAA, Atlantic/Nordic, Siberian) on the "
                        "lat>=66N region boundary and append them to "
                        "transports.txt (upwind AND, via store_salt_flux, "
                        "the model's EXACT advective salt transport; their "
                        "difference is the face-scheme gap plus a one-step "
                        "salinity time-level offset). ALSO dumps the "
                        "accumulator's "
                        "CUMULATIVE sums + step count to "
                        f"{GATEWAY_CUMULATIVE_CSV} at the "
                        "--snapshot-every-days cadence (plus one row at run "
                        "end), so the mean over ANY window is recovered "
                        "exactly by differencing two rows: "
                        "(cumsum_b - cumsum_a) / (n_b - n_a). Nothing is "
                        "reset, so the transports.txt whole-run mean is "
                        "unchanged. Pure diagnostic: reads the state "
                        "after each step and never writes back, so the "
                        "trajectory is bit-identical with the flag off. "
                        "Sign: POSITIVE = INTO the Arctic. tripole / latlon "
                        "host-loop runs only.")
    p.add_argument("--no-normalize-freshwater", action="store_true",
                   help="EXPLICITLY disable the global surface-freshwater "
                        "normalization the latlon/tripole/mpas setups enable "
                        "by default. Re-admits the real CORE-II ~+0.65 Sv "
                        "P-E+R imbalance (~-0.5 PSU/90d global fresh drift) — "
                        "for controlled probes only, e.g. combined with "
                        "--freshwater-salinity local (whose combination WITH "
                        "the normalization is rejected: nonzero global-salt "
                        "covariance).")
    p.add_argument("--relative-winds", action="store_true",
                   help="NEMO ln_crt_dwn current feedback: subtract the ocean "
                        "surface current from the 10-m wind before the bulk "
                        "stress + turbulent fluxes (rn_vfac=1.0). Reduces "
                        "tropical stress by ~20-30 percent, realigns the EUC and "
                        "damps mesoscale (eddy-killing). Shorthand for "
                        "--wind-vfac 1.0. lat-lon / tripole / mpas only "
                        "(cubed_sphere parked). Default off (absolute wind).")
    p.add_argument("--wind-vfac", type=float, default=None,
                   help="NEMO rn_vfac current-feedback fraction in [0, 1] "
                        "(0=absolute wind=default; 1=full feedback="
                        "--relative-winds). Set BOTH only if equal to 1.0.")
    p.add_argument("--mle", action="store_true",
                   help="Enable the Fox-Kemper mixed-layer-eddy (MLE) "
                        "restratification (NEMO tramle nn_mle=1): a bolus "
                        "overturning streamfunction that flattens mixed-layer "
                        "isopycnals (restratifying, shoaling the MLD) in "
                        "mode-water regions. Lat-lon / tripole C-grid only "
                        "(--grid mpas raises). Default off.")
    p.add_argument("--mle-ce", type=float, default=0.06,
                   help="MLE efficiency coefficient rn_ce (NEMO ORCA1 0.06; "
                        "typical 0.06-0.08). For --mle.")
    p.add_argument("--geothermal", action="store_true",
                   help="Geothermal bottom heat-flux boundary condition (NEMO "
                        "ln_trabbc, Emile-Geay & Madec 2009): warm the deepest "
                        "wet cell of each column by the seafloor heat flux. "
                        "Grid-agnostic. Tiny + abyssal -- structural NEMO "
                        "faithfulness, NOT a surface-SST lever on spin-up "
                        "timescales. Default off.")
    p.add_argument("--geothermal-flux-wm2", type=float, default=None,
                   help="Constant seafloor geothermal heat flux [W/m^2] for "
                        "--geothermal. Default = GeothermalConfig default "
                        "(NEMO rn_geoflx_cst).")
    p.add_argument("--div-damp-2", type=float, default=None,
                   help="2nd-order divergence damping [m^2/s] -- suppresses "
                        "grid-scale divergent (checkerboard) modes at small "
                        "high-lat coastal cells. CFL: dt < dx^2/(2*nu).")
    p.add_argument("--div-damp-4", type=float, default=None,
                   help="4th-order (scale-selective) divergence damping [m^4/s] "
                        "-- damps the grid-scale mode far more than the resolved "
                        "flow; gentler CFL than 2nd-order. Try ~1e9-1e10 at 1/4 deg.")
    p.add_argument("--smag-cfl-safety", type=float, default=None,
                   help="Cap the Laplacian-Smagorinsky coeff at smag_cfl_safety*"
                        "area*cos^2(lat)/dt (per-cell tuned viscosity ceiling). "
                        "Lets --C-smag-lap "
                        "be cranked high to damp WBC jets WITHOUT self-CFL at the "
                        "sharp jet. ~0.125 is a safe 2-D Laplacian cap.")
    p.add_argument("--nudge-woa-tau-days", type=float, default=0.0,
                   help="Nudge T,S toward WOA with this timescale [days] from a "
                        "rest start -- gradual cold-start spinup that avoids the "
                        "WOA-IC geostrophic-imbalance blowup. 0=off.")
    p.add_argument("--nudge-release-day", type=float, default=0.0,
                   help="Stop nudging after this model day (0=nudge throughout).")
    p.add_argument("--spinup-drag-tau-days", type=float, default=0.0,
                   help="Rayleigh velocity-damping timescale [days] during the "
                        "spin-up phase -- bleeds off the cold-start geostrophic "
                        "adjustment OVERSHOOT (the ~5-10 m/s transients that trip "
                        "the nonlinear advective blowup) while the stratification "
                        "settles. 0=off.")
    p.add_argument("--spinup-drag-days", type=float, default=0.0,
                   help="Duration [days] of the spin-up velocity-damping phase "
                        "(drag removed afterwards -> free run).")
    return p


def main() -> int:
    # allow_abbrev=False: the module-level x64 toggle is decided by an EXACT
    # "--fp32" argv match (``_FP32``), so the real parser must NOT accept an
    # abbreviation (e.g. "--fp") of --fp32 — that would set args.fp32=True
    # while x64 was already enabled, tripping the consistency guard below.
    # All sbatch wrappers already use full flag names, so this is behaviour-
    # preserving for existing callers.
    p = _build_arg_parser()
    args = p.parse_args()

    # KPP MLD-deepening sensitivity flags are mpas/latlon-only (fail loud).
    _validate_kpp_grid(args.grid, args.kpp_ri_crit, args.kpp_cv, args.kpp_eice,
                       mpas_vmix=args.mpas_vmix)
    # --mpas-vmix is an MPAS selector: on another grid it would be silently
    # ignored (dispatch footgun) — reject whenever the flag was typed
    # EXPLICITLY, even with the 'kpp' default value (codex LOW: an explicit
    # `--mpas-vmix kpp --grid tripole` is still a user error worth surfacing).
    if (args.ice_ocean_heat_coeff is not None
            and not args.prognostic_sea_ice):
        raise SystemExit(
            "--ice-ocean-heat-coeff configures the prognostic sea-ice model "
            "and requires --prognostic-sea-ice (otherwise silently unused).")
    if args.grid != "mpas" and (args.mpas_vmix != "kpp"
                                or "mpas_vmix" in _cli_flags_given()):
        raise SystemExit(
            f"--mpas-vmix {args.mpas_vmix!r} selects the MPAS vertical-mixing "
            f"closure and requires --grid mpas (got --grid {args.grid!r}); "
            "for the tripole use --tripole-vmix.")
    # --kpp-eice (latlon) / the MPAS TKE card need an ice source to bite:
    # surface_forcing.ice_concentration is attached only under
    # --prognostic-sea-ice or a prescribed SIC field (--ice-albedo/
    # --ice-thermo/--sss-restore load it).  Without one, ice_frac stays None:
    # KPP-eice would be silently inert (codex MED), and the MPAS TKE bridge
    # FAIL-FASTS at the first step — reject loudly up front in both cases.
    # NB: the ORCA1 zdftke card DEFAULTS eice=3, so MPAS+tke wants ice unless
    # the run explicitly turns the attenuation off with --tke-eice 0 (which
    # reaches MPAS since 2026-08-22).
    _mpas_tke_eice = (args.tke_eice if args.tke_eice is not None else 3)
    _wants_eice = (args.kpp_eice not in (None, 0)
                   or (args.grid == "mpas" and args.mpas_vmix == "tke"
                       and _mpas_tke_eice != 0))
    if _wants_eice and not (
            args.prognostic_sea_ice or args.ice_albedo or args.ice_thermo
            or args.sss_restore):
        raise SystemExit(
            "--kpp-eice/--tke-eice need a sea-ice source to attenuate "
            "against: add --prognostic-sea-ice (or a prescribed SIC via "
            "--ice-albedo/--ice-thermo/--sss-restore). Without one the "
            "surface ice concentration never reaches the closure.")
    _validate_pcg_variant_grid(args.grid, args.barotropic_pcg_variant,
                               args.barotropic_solver)

    # NEMO ln_crt_dwn relative-wind current feedback (rn_vfac): resolve + range-
    # check the CLI up front (fail before the expensive setup).  0.0 = absolute
    # wind (byte-identical default).  cubed_sphere is parked -> reject early
    # rather than crash mid-loop (the applicator has no cube current rotation).
    _wind_vfac = _resolve_wind_vfac(args.relative_winds, args.wind_vfac)
    if _wind_vfac != 0.0 and args.grid == "cubed_sphere":
        raise SystemExit(
            "--relative-winds/--wind-vfac is not wired for --grid cubed_sphere "
            "(parked); supported grids: tripole / latlon_bathy / mpas.")

    # --spmd-persistent-state combo validation (scaling-M2): fail BEFORE the
    # jax.distributed bootstrap / any expensive setup.
    _validate_spmd_persistent_state(
        args.spmd_persistent_state, args.n_gpus, args.distributed)

    # ------------------------------------------------------------------
    # Multi-PROCESS jax.distributed bootstrap (--distributed): MUST run BEFORE any
    # jax array op / device query so jax.devices() spans every process' GPUs.  A
    # no-op for a single process (default).  Reuses the proven coordinator
    # discovery (rank-0 hostname = coordinator) from parallel.distributed; the
    # lat-band SPMD halo is pure-JAX ppermute/psum (no mpi4jax), so the bootstrap
    # requires only mpi4py.  ``_proc_index`` gates process-0-only I/O below.
    _proc_index = 0
    _proc_count = 1
    if args.distributed:
        from legoesm.parallel.distributed import (
            initialize_jax_distributed_multiprocess,
        )
        _rank, _proc_count = initialize_jax_distributed_multiprocess()
        _proc_index = int(jax.process_index())
        if _proc_count <= 1:
            print("[distributed] --distributed given but launched single-process "
                  "(MPI size 1): running the single-controller path.", flush=True)
        else:
            print(f"[distributed] process {_proc_index}/{_proc_count}: "
                  f"global jax.device_count()={jax.device_count()}, "
                  f"local={jax.local_device_count()}", flush=True)

    def _is_io_proc() -> bool:
        """Only process 0 writes to the shared output dir (manifest / CSV /
        snapshots / transports / digest) — under --distributed every process runs
        the same host loop on the all-gathered replicated state, so N processes
        would otherwise clobber the same files."""
        return _proc_index == 0

    # Process-0-only I/O under --distributed is handled by passing ``io_proc=`` to
    # the file-writing helpers (_save_snapshot / the four transport diags /
    # _record_final_state_digest), NOT by shadowing them with local wrappers.  A
    # local ``def _save_snapshot`` would make the name local to main() and the
    # ``_impl = _save_snapshot`` capture raise UnboundLocalError (codex HIGH).  More
    # importantly, those diags run JAX ops (compute_layer_thickness on the gathered,
    # P()-replicated state) — under multi-controller JAX EVERY rank must dispatch
    # the SAME program in the SAME order, so a process-0-ONLY call would desync the
    # collective program.  Passing io_proc lets every rank run the (identical,
    # replicated) JAX + host pull, and gates ONLY the filesystem write to process 0.

    # --ice-thermo-sw-trans is consumed by BOTH under-ice paths (the --ice-thermo
    # prescribed surrogate AND the --prognostic-sea-ice partition), so validate
    # it whenever either is on (codex r5: prognostic ice previously consumed it
    # unvalidated because the check was gated on --ice-thermo alone).
    if args.ice_thermo or args.prognostic_sea_ice:
        if not (0.0 <= float(args.ice_thermo_sw_trans) <= 1.0):
            raise ValueError("--ice-thermo-sw-trans must be in [0,1] (SW fraction "
                             f"transmitted through ice); got {args.ice_thermo_sw_trans}.")
    if args.ice_thermo:   # codex LOW: reject unphysical prescribed-ice params early
        if not (float(args.ice_thermo_tau_days) > 0.0):
            raise ValueError("--ice-thermo-tau-days must be > 0 (freezing-relaxation "
                             f"timescale [days]); got {args.ice_thermo_tau_days}.")

    if args.freeze_scheme != "constant":
        # The liquidus scheme only feeds the freeze surrogates -- with neither
        # enabled it would be a silent no-op (dispatch-hardening: fail loud).
        if not (args.freeze_floor or args.ice_thermo):
            raise ValueError(
                f"--freeze-scheme {args.freeze_scheme!r} has no consumer without "
                "--freeze-floor (salinity-dependent model freeze floor) and/or "
                "--ice-thermo (salinity-dependent under-ice relaxation target). "
                "Add one of those flags or drop --freeze-scheme.")
        if args.grid == "cubed_sphere":
            raise ValueError(
                "--freeze-scheme is not wired for --grid cubed_sphere (the cube "
                "builder does not thread the freeze-floor config; matching its "
                "existing --freeze-floor gap). Use tripole/latlon_bathy/mpas.")

    if args.prognostic_sea_ice:
        # The REAL prognostic ice model REPLACES the surrogates — never combine
        # (double counting / inconsistent SST clamps).  Fail loud.
        _ice_surrogates = [
            ("--freeze-floor", args.freeze_floor),
            ("--ice-thermo", args.ice_thermo),
            ("--ice-albedo", args.ice_albedo),
            ("--ice-albedo-seasonal", args.ice_albedo_seasonal),
        ]
        _on = [name for name, val in _ice_surrogates if val]
        if _on:
            raise ValueError(
                "--prognostic-sea-ice is mutually exclusive with the sea-ice "
                f"surrogates {_on}: the prognostic model replaces the freeze-floor "
                "T clamp, the prescribed-siconc SW albedo, and the ice-thermo "
                "freezing relaxation (combining them would double-count the ice "
                "boundary).  Drop the surrogate flag(s).")
        if not args.woa_init:
            raise ValueError(
                "--prognostic-sea-ice requires --woa-init: the brine salt budget "
                "and freezing point need a realistic high-latitude T/S, not the "
                "idealised rest state.")
        if args.grid == "cubed_sphere":
            raise ValueError(
                "--prognostic-sea-ice is not wired for --grid cubed_sphere (parked "
                "grid; the OMIP runner does not pass freshwater= on the cube path). "
                "Use --grid mpas (primary), tripole, or latlon_bathy.")
        if int(args.scan_block) > 0:
            raise ValueError(
                "--prognostic-sea-ice cannot run under --scan-block: the ice step "
                "pulls ocean SST to the host each step (like SSS restoring / "
                "ice-thermo), so it is host-loop only. Set --scan-block 0.")

    if args.ice_init is not None and not args.prognostic_sea_ice:
        raise ValueError(
            "--ice-init initialises the PROGNOSTIC ice state and requires "
            "--prognostic-sea-ice (the surrogate paths read the prescribed "
            "NEMO siconc climatology, not this file).")
    _require_prognostic_ice_for_itd_flags(
        args.ice_categories, args.ice_ridging, args.prognostic_sea_ice)

    # --prescribed-flow gates (PRE-BUILD, on the static args): grid support +
    # the --spinup-drag rejection + the --no-gm-redi requirement.  NB: no
    # --scan-block gate — the lever is in-model (inside _step_impl), so the
    # lax.scan block path pins correctly.
    validate_prescribed_flow_args(args.prescribed_flow, args.grid,
                                  args.spinup_drag_tau_days,
                                  no_gm_redi=args.no_gm_redi)

    # Precision policy. The all-fp32 policy is set when --fp32 is given; the
    # module-level argv sniff (``_FP32``) already kept JAX x64 OFF so device
    # arrays default to float32. Defend against a stale/mismatched argv sniff
    # (e.g. --fp32 passed via an args namespace that argv didn't see): the
    # parsed flag is authoritative for the policy, and x64 MUST agree with it.
    if bool(args.fp32) != _FP32:
        raise SystemExit(
            "internal: --fp32 argparse flag disagrees with the module-level "
            f"argv sniff (_FP32={_FP32}, args.fp32={args.fp32}). The x64 toggle "
            "is decided from argv at import; pass --fp32 on the command line.")
    if args.fp32:
        # fp32 is only VALIDATED for the C-grid latlon/tripole SPMD path
        # (the audited dtype-safe OMIP path).  The MPAS / cube backends carry
        # f64-only or fp32-unvalidated numerics internally (e.g. the
        # density-Jacobian analytic-pressure PGF hardcodes float64 to protect
        # an O(5.8e8) cancellation with no x64 fallback) and were not validated
        # in single precision — fail loud rather than silently mis-run them.
        if args.grid not in ("tripole", "latlon_bathy"):
            raise SystemExit(
                f"--fp32 is only validated for --grid tripole / latlon_bathy "
                f"(the SPMD C-grid OMIP path: fixed-iteration barotropic PCG + "
                f"wright EOS + adcroft/smc03 PGF are all fp32-safe). --grid "
                f"{args.grid!r} carries f64-only / fp32-unvalidated numerics "
                f"(density-Jacobian PGF islands, etc.) and is NOT fp32-"
                f"validated. Drop --fp32 or use --grid tripole / latlon_bathy.")
        if jax.config.jax_enable_x64:
            # Belt-and-suspenders: something (e.g. JAX_ENABLE_X64=1 in the env,
            # or an earlier import) re-enabled x64, which would silently keep
            # arrays in float64 and defeat the whole point of --fp32.
            raise SystemExit(
                "--fp32 requires JAX x64 DISABLED, but jax_enable_x64 is True "
                "(likely JAX_ENABLE_X64=1 is exported, or x64 was enabled "
                "before this run). Unset JAX_ENABLE_X64 so float32 is the "
                "default device dtype.")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp32() if args.fp32 else PrecisionPolicy.fp64())
    if args.fp32:
        print("[setup] PRECISION: float32 (JAX x64 OFF, PrecisionPolicy.fp32) "
              "— GPU-memory mode; the f64 reference is the default (no --fp32).")
    from legoesm.ocean.forcing import load_core2_nyf
    from legoesm.ocean.coupler import (
        compute_omip2_surface_forcing,
        compute_omip2_freshwater_forcing,
    )
    from legoesm.core.field import Field

    # --config (#376 Phase 4): a template's run controls (time/output/grid)
    # drive the actual run BEFORE the grid/model are built, so a --config run
    # integrates the dt/duration/grid/output the template describes — not the
    # argparse defaults (codex review HIGH). Explicit CLI flags still win. The
    # ocean.* physics overrides are applied AFTER the model is built (below).
    ocean_adapter = None
    if args.config:
        from legoesm.ocean.config import (
            OceanExperimentConfig, resolve_ocean_run_controls,
        )
        ocean_adapter = OceanExperimentConfig.from_yaml(args.config)
        ocean_adapter.validate_strict()
        _applied = resolve_ocean_run_controls(
            ocean_adapter, args, _cli_flags_given()
        )
        if _applied:
            print(f"[setup] --config {args.config} run controls: {_applied}")
        # Grid backend + mesh + forcing come from --grid/--mesh, not the
        # template (the mesh is a file, not a config field). Say so explicitly so
        # a template's grid/forcing sections are never SILENTLY ignored.
        _gt = ocean_adapter.get("grid.type")
        _fc = ocean_adapter.get("forcing.dataset")
        print(f"[setup] --config note: grid backend/mesh come from --grid="
              f"{args.grid!r} / --mesh (template grid.type={_gt!r}, "
              f"forcing={_fc!r} are advisory).")

    print(f"[setup] building {args.grid} (nlev={args.nlev}, "
          f"woa_init={args.woa_init}) ...")
    if args.ew_cyclic_overlap and args.grid not in ("tripole", "mpas"):
        raise ValueError(
            "--ew-cyclic-overlap is ORCA-cyclic-overlap-specific (the eORCA1 "
            "source mesh): tripole applies it to the C-grid prognostic seam, "
            "mpas applies it to the eORCA source mask/bathy BEFORE the Voronoi "
            "regrid (removes the imported ~73E seam). It is WRONG on a regular "
            f"period-nx lat-lon / cube grid. Got --grid {args.grid!r}.")
    if args.tripole_vmix != "none" and args.grid != "tripole":
        raise ValueError(
            "--tripole-vmix wires the vertical-mixing closure onto the "
            "TRIPOLE physics-attach block only; other grids configure their "
            "closure through their own builders (latlon_bathy ships KPP+EVD). "
            f"Got --tripole-vmix {args.tripole_vmix!r} with --grid "
            f"{args.grid!r}.")
    # The --tke-eice / --tke-surface-bc card knobs take effect ONLY under
    # --grid tripole --tripole-vmix tke; reject every other context (they are
    # silently discarded there) — the --tripole-vmix guard above misses them at
    # its "none" default and under the kpp closure.
    if args.A_h_profile_file and args.grid != "tripole":
        raise SystemExit("--A-h-profile-file is tripole-only (the profile is "
                         "built on the eORCA nominal latitude rows).")
    _validate_tke_card_grid(args.grid, args.tripole_vmix, args.tke_eice,
                            args.tke_surface_bc, args.tke_mxl_choice,
                            args.tke_prognostic, args.tke_kappa_convention,
                            args.tke_shear_production,
                            tke_n2_mode=args.tke_n2_mode,
                            tke_n2_eos_form=args.tke_n2_eos_form,
                            tke_lc=args.tke_lc, tke_etau=args.tke_etau,
                            mpas_vmix=args.mpas_vmix)
    # --gm-treguier is applied in build_tripole's GM/Redi override only; on any
    # other grid (or with GM disabled) it would be silently discarded.
    if args.gm_treguier and args.grid != "tripole":
        raise SystemExit(
            f"--gm-treguier is wired for --grid tripole only (the GM/Redi "
            f"override lives in build_tripole); got --grid {args.grid!r}.")
    if args.gm_treguier and args.no_gm_redi:
        raise SystemExit(
            "--gm-treguier and --no-gm-redi are mutually exclusive: the former "
            "selects the NEMO ldf_eiv kappa_GM scheme, the latter disables "
            "GM/Redi entirely.")
    # Same guards for the OPERATOR flags: they are read only inside
    # build_tripole, so on any other grid they would be silently discarded.
    _gm_op_flags = [n for n, v in (("--gm-slope-scheme", args.gm_slope_scheme),
                                   ("--gm-bolus-advection",
                                    args.gm_bolus_advection),
                                   ("--gm-msc-stabilize",
                                    args.gm_msc_stabilize)) if v is not None]
    if _gm_op_flags and args.grid != "tripole":
        raise SystemExit(
            f"{' and '.join(_gm_op_flags)} is wired for --grid tripole only "
            f"(the GM/Redi override lives in build_tripole); got --grid "
            f"{args.grid!r}.")
    if _gm_op_flags and args.no_gm_redi:
        raise SystemExit(
            f"{' and '.join(_gm_op_flags)} and --no-gm-redi are mutually "
            "exclusive: the former select the GM/Redi operator, the latter "
            "disables GM/Redi entirely.")
    # ORDERED MOST-SPECIFIC-FIRST, and this one is the most specific: a
    # channel selected for a restoring that is switched off.  The previous
    # ordering fixed only the closure check, so `water_flux` ALONE still
    # reported "requires live_s" -- and the test missed it because it supplied
    # live_s (codex 9387497).
    if (args.sss_restore_channel == "water_flux"
            and not getattr(args, "sss_restore", False)):
        raise SystemExit(
            "--sss-restore-channel water_flux without --sss-restore selects a "
            "channel for a restoring that is switched off; drop the flag.")
    # The water-flux channel needs NEMO's own conversion: `s_target` is the
    # virtual-salt form, and routing THAT through the freshwater budget would
    # move water at a rate derived from the wrong denominator.
    if (args.sss_restore_channel == "water_flux"
            and args.sss_restore_normalization != "live_s"):
        raise SystemExit(
            "--sss-restore-channel water_flux requires "
            "--sss-restore-normalization live_s: the water flux is NEMO's "
            "nn_sssr=2 form (divided by the LIVE surface salinity), whereas "
            "the default 's_target' is the virtual-salt conversion and would "
            "move water at the wrong rate.")
    # The default `virtual_salt_flux` closure builds its net INTERNALLY from
    # the FreshwaterForcing (`virtual_salt_flux(freshwater, ...)`,
    # ocean_model_latlon_cgrid.py:5044), and that net INCLUDES `restoring`.
    # So under that closure a populated `fw.restoring` would reach the ocean
    # TWICE: once as volume through eta / z-star, and again as the closure's
    # virtual-salt tendency.  (codex 9387241 RED, verified.)
    if (args.sss_restore_channel == "water_flux"
            and args.freshwater_closure != "real_freshwater"):
        raise SystemExit(
            "--sss-restore-channel water_flux requires --freshwater-closure "
            "real_freshwater: the default virtual_salt_flux closure derives "
            "its salt tendency from the NET freshwater, which already "
            "includes fw.restoring, so routing restoring as water would apply "
            "it twice (volume AND virtual salt).")
    if (args.gm_bolus_advection == "through_fct"
            and args.gm_slope_scheme != "nemo_iso_lap"):
        raise SystemExit(
            "--gm-bolus-advection through_fct requires --gm-slope-scheme "
            "nemo_iso_lap: the model honors the PAIR (ocean_model_latlon_cgrid "
            "`_want_bolus`), so through_fct alone would silently keep the "
            "centred bolus flux while the manifest claimed FCT routing.")
    # Symmetric guard: the two Treguier tunables are read ONLY inside the
    # --gm-treguier branch of build_tripole, so a non-default value passed
    # without the scheme flag would evaporate silently.
    if not args.gm_treguier:
        _stray = [f"--gm-aei0 {args.gm_aei0:g}"
                  if args.gm_aei0 != _GM_AEI0_DEFAULT else None,
                  f"--gm-kappa-min {args.gm_kappa_min:g}"
                  if args.gm_kappa_min != _GM_KAPPA_MIN_DEFAULT else None]
        _stray = [s for s in _stray if s]
        if _stray:
            raise SystemExit(
                f"{' and '.join(_stray)} require --gm-treguier (they only "
                f"configure the NEMO ldf_eiv kappa_GM block); without it the "
                f"value is silently discarded.")
    if args.gm_treguier and args.gm_kappa_min > args.gm_aei0:
        raise SystemExit(
            f"--gm-kappa-min {args.gm_kappa_min:g} exceeds --gm-aei0 "
            f"{args.gm_aei0:g}: the floor would override the NEMO cap on every "
            f"wet cell.")
    if args.river_mouth_restoring_gate and not args.runoff:
        raise ValueError(
            "--river-mouth-restoring-gate requires --runoff (the gate masks "
            "restoring where the Dai-Trenberth runoff field is active; "
            "without --runoff there is no runoff field and the gate would "
            "silently do nothing).")
    if args.sss_ice_gate_nemo and not args.sss_restore:
        raise ValueError(
            "--sss-ice-gate-nemo requires --sss-restore (it only changes the "
            "under-ice weighting of the SSS restoring; with no restoring it "
            "would silently do nothing).")
    # Fox-Kemper MLE: lat-lon / tripole C-grid (mle_latlon_cgrid) AND MPAS
    # Voronoi (mle_mpas, NEMO nn_mle=1 bolus port).  The cube stays unsupported
    # (parked grid; no C-D-grid MLE adapter).
    mle_cfg = None
    if args.mle:
        if args.grid == "cubed_sphere":
            raise ValueError(
                "--mle is not implemented for --grid 'cubed_sphere' (parked "
                "grid; no C-D-grid MLE adapter). Use tripole, latlon_bathy or mpas.")
        from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig
        mle_cfg = MLEConfig(ce=args.mle_ce)
        print(f"[setup] Fox-Kemper MLE requested: ce={args.mle_ce:g}")

    # --nemo-vertical: replace the default 20-level tanh z* with NEMO ORCA1's
    # 75-level reference column (e3t_1d) so vertical gradients (thermocline, MLD)
    # are resolved comparably to NEMO. Overrides --nlev/--H-max to the NEMO column.
    _nemo_dz = None
    if args.nemo_vertical:
        if args.grid == "cubed_sphere":
            raise ValueError(
                "--nemo-vertical is not wired for --grid cubed_sphere (parked "
                "grid; build_cubed_sphere does not thread dz_ref_override). Use "
                "tripole, latlon_bathy, or mpas.")
        _vfile = args.nemo_vertical_file or _NEMO_DOMAIN_CFG
        _nemo_dz = _load_nemo_e3t_1d(_vfile)
        args.nlev = int(_nemo_dz.size)
        args.H_max = float(_nemo_dz.sum())
        print(f"[setup] --nemo-vertical: {args.nlev} levels from {_vfile} "
              f"(dz {_nemo_dz[0]:.2f}->{_nemo_dz[-1]:.1f} m, H_max "
              f"{args.H_max:.0f} m) -- matching NEMO ORCA1 L75.")

    # zdfiwm CLI → IWMConfig (shared with run_omip; None when --iwm absent
    # so the builders' iwm-block stays fully inert on legacy runs).
    from scripts.run.run_omip import build_iwm_config_from_args as _build_iwm
    _iwm_cfg = _build_iwm(args) if args.iwm else None
    # zdfddm CLI -> DoubleDiffusionConfig (None when --double-diffusion absent
    # so the builders' ddm-block stays fully inert on legacy runs).  Additive
    # salt-fingering avt/avs; rides the same implicit-vertical-mixing paths as
    # --iwm (lat-lon/tripole force implicit mixing on).
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
        DoubleDiffusionConfig as _DDMConfig,
    )
    _ddm_cfg = None
    if args.double_diffusion:
        _ddm_kw = {}
        if args.ddm_avts is not None:
            _ddm_kw["rn_avts"] = args.ddm_avts
        if args.ddm_rc is not None:
            _ddm_kw["rn_hsbfr"] = args.ddm_rc
        _ddm_cfg = _DDMConfig(enabled=True, **_ddm_kw)
    # --freeze-scheme -> model-config freezing override (MED-1): None keeps the
    # builders' validated default config untouched (byte-exact legacy).
    _freezing_ovr = None
    if args.freeze_scheme != "constant":
        from legoesm.ocean.eos import FreezingPointConfig
        _freezing_ovr = FreezingPointConfig(scheme=args.freeze_scheme)
    if args.grid == "tripole":
        grid, z_coord, model, state, H_bathy = build_tripole(
            args.nlev, args.H_max, args.mesh,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            n_gpus=args.n_gpus,
            pgf_scheme=args.pgf_scheme, A_h=args.A_h, B_h=args.B_h, K_bih=args.K_bih,
            flat_bottom=args.flat_bottom, A_h_eq_boost=args.A_h_eq_boost,
            A_h_eq_sigma_deg=args.A_h_eq_sigma_deg,
            A_h_profile_file=args.A_h_profile_file,
            ke_gradient_scheme=args.ke_gradient_scheme,
            partial_cell=args.partial_cell,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            bathy_smoothing_passes=args.bathy_smoothing_passes,
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            freeze_floor=(True if args.freeze_floor else None),
            freezing=_freezing_ovr,
            runoff_depth_spread_m=args.runoff_depth_spread_m,
            barotropic_solver=args.barotropic_solver,
            barotropic_pcg_variant=args.barotropic_pcg_variant,
            barotropic_diffusion_alpha=args.barotropic_diffusion_alpha,
            n_barotropic_substeps=args.n_barotropic_substeps,
            barotropic_time_filter=args.barotropic_time_filter,
            bottom_drag_r=args.bottom_drag_r,
            C_smag=args.C_smag, C_leith=args.C_leith, C_smag_lap=args.C_smag_lap,
            momentum_advection=args.momentum_advection,
            slope_foot_alpha=args.slope_foot_alpha,
            slope_foot_n_levels=args.slope_foot_n_levels,
            slope_foot_threshold=args.slope_foot_threshold,
            min_levels=args.min_levels,
            div_damp_2=args.div_damp_2, div_damp_4=args.div_damp_4,
            smag_cfl_safety=args.smag_cfl_safety,
            convection=args.convection,
            convection_K_conv=args.convection_K_conv,
            convection_K_bg=args.convection_K_bg,
            ew_cyclic_overlap=(True if args.ew_cyclic_overlap else None),
            tracer_advection=args.tracer_advection,
            mle=mle_cfg, dz_ref_override=_nemo_dz,
            bottom_drag_scheme=args.bottom_drag_scheme,
            bottom_drag_cd0=args.bottom_drag_cd0,
            bottom_drag_cdmax=args.bottom_drag_cdmax,
            bottom_drag_z0=args.bottom_drag_z0,
            bottom_drag_ke0=args.bottom_drag_ke0,
            iwm=_iwm_cfg, iwm_forcing_file=args.iwm_forcing_file,
            ddm=_ddm_cfg,
            prescribed_flow=args.prescribed_flow,
            no_gm_redi=args.no_gm_redi,
            tripole_vmix=args.tripole_vmix,
            tke_eice=args.tke_eice,
            tke_surface_bc=args.tke_surface_bc,
            tke_mxl_choice=args.tke_mxl_choice,
            tke_n2_mode=args.tke_n2_mode,
            tke_n2_eos_form=args.tke_n2_eos_form,
            tke_prognostic=args.tke_prognostic,
            tke_kappa_convention=args.tke_kappa_convention,
            tke_shear_production=args.tke_shear_production,
            tke_lc=(None if args.tke_lc is None else args.tke_lc == "on"),
            tke_etau=args.tke_etau,
            gm_treguier=args.gm_treguier,
            gm_aei0=args.gm_aei0,
            gm_kappa_min=args.gm_kappa_min,
            gm_slope_scheme=args.gm_slope_scheme,
            gm_bolus_advection=args.gm_bolus_advection,
            gm_msc_stabilize=args.gm_msc_stabilize,
            store_mass_flux=bool(getattr(args, "gateway_transports",
                                         False)),
            store_salt_flux=bool(getattr(args, "gateway_transports",
                                         False)),
        )
        app_grid_type = "tripole"
    elif args.grid == "cubed_sphere":
        grid, z_coord, model, state, H_bathy = build_cubed_sphere(
            args.nlev, args.H_max, args.mesh, n=args.cube_n,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            flat_bottom=args.flat_bottom,
            A_h=args.cube_Ah, hyperdiff_coeff=args.cube_hyperdiff,
            div_damp_2=args.cube_divdamp2, div_damp_4=args.cube_divdamp4,
            baroclinic_rk3=(True if args.cube_rk3 else None),
            mask_marginal_seas=args.cube_mask_marginal_seas,
            dt=args.dt,
            velocity_ceiling=args.cube_velocity_ceiling,
            partial_cell=args.partial_cell,
            pgf_scheme=args.cube_pgf_scheme,
            bottom_drag_r=args.cube_bottom_drag_r,
            bottom_drag_bbl_thickness=args.cube_bbl_thickness,
            bottom_drag_bg_velocity=args.cube_bottom_drag_bg_vel,
            harmonic_cfl_safety=args.cube_harmonic_cfl_safety,
            bathy_smoothing_passes=args.cube_bathy_smoothing,
            smc03_bottom_2nd_order=(True if args.cube_smc03_bottom_2nd else None),
            barotropic_sw_div_damp_factor=args.cube_baro_divdamp,
            barotropic_sw_damp_v=args.cube_baro_dampv,
        )
        app_grid_type = "cubed_sphere"
    elif args.grid == "mpas":
        grid, z_coord, model, state, H_bathy = build_mpas_ocean(
            args.nlev, args.H_max, args.mesh,
            level=args.mpas_level, lloyd_iterations=args.mpas_lloyd,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            flat_bottom=args.flat_bottom, partial_cell=args.partial_cell,
            freeze_floor=(True if args.freeze_floor else None),
            freezing=_freezing_ovr,
            runoff_depth_spread_m=args.runoff_depth_spread_m,
            mle=mle_cfg, dz_ref_override=_nemo_dz,
            barotropic_solver=args.barotropic_solver,
            barotropic_pcg_variant=args.barotropic_pcg_variant,
            # Cross-grid parity (2026-07-18 manifest audit): these two flags
            # were silently IGNORED on MPAS — the call site never passed
            # them, so mpas8_corr ran tvd + adcroft while the tripole ran
            # superbee + smc03 despite byte-identical sbatch flags.  Both
            # ARE supported on the Voronoi core (ocean_model_mpas advection
            # dispatch incl. superbee; ocean_pe_mpas smc03 branch).
            pgf_scheme=args.pgf_scheme,
            tracer_advection=args.tracer_advection,
            bottom_drag_scheme=args.bottom_drag_scheme,
            bottom_drag_cd0=args.bottom_drag_cd0,
            bottom_drag_cdmax=args.bottom_drag_cdmax,
            bottom_drag_z0=args.bottom_drag_z0,
            bottom_drag_ke0=args.bottom_drag_ke0,
            iwm=_iwm_cfg, ddm=_ddm_cfg,
            # --mpas-vmix: 'tke' runs the NEMO ORCA1 zdftke card through the
            # SAME builder the tripole uses (the "tripole_" prefix is
            # historical — pure grid-agnostic config construction; its
            # closure-mismatch rejects also fire here); 'kpp' (default) keeps
            # the historical KPP + --kpp-* overrides.
            # Full #1326 ORCA1 card, including the PROGNOSTIC Mode-A carry —
            # MPASOceanState.tke is seeded by model.seed_tke(state) in the
            # host loop before the first step (pytree-stable carry).
            # The zdftke card knobs reach MPAS too (2026-08-22): the three
            # grids have to be able to run the SAME closure or a cross-grid
            # comparison is measuring the card, not the grid.  Every knob is
            # threaded — a knob accepted by _validate_tke_card_grid and then
            # dropped here is the silent-discard footgun that guard exists to
            # prevent.  iwm stays None: the MPAS TKE bridge rejects it.
            vertical_mixing=(
                build_tripole_vmix_config(
                    "tke", iwm=None,
                    tke_eice=args.tke_eice,
                    tke_surface_bc=args.tke_surface_bc,
                    tke_mxl_choice=args.tke_mxl_choice,
                    tke_prognostic=args.tke_prognostic,
                    tke_n2_mode=args.tke_n2_mode,
                    tke_n2_eos_form=args.tke_n2_eos_form,
                    tke_kappa_convention=args.tke_kappa_convention,
                    tke_shear_production=args.tke_shear_production,
                    tke_lc=args.tke_lc, tke_etau=args.tke_etau)
                if args.mpas_vmix == "tke"
                else _kpp_vmix_override(args.kpp_ri_crit, args.kpp_cv,
                                        args.kpp_eice)),
            ew_cyclic_overlap=bool(args.ew_cyclic_overlap),
        )
        app_grid_type = "mpas"
    else:
        _nlat, _nlon = (int(x) for x in args.latlon_res.split("x"))
        if args.n_gpus > 1 and _nlat % args.n_gpus != 0:
            # The regular lat-lon grid has a SOUTH POLE WALL, not a bipolar fold:
            # padding rows would add unphysical sub-pole latitudes (cos(lat)->
            # negative / tiny metrics), so the fold-preserving south-pad does NOT
            # apply here.  Require a divisible --latlon-res instead (180x360 is
            # divisible by 2/3/4/5/6...; pick e.g. 180/360 for n_gpus|180).
            raise SystemExit(
                f"--n-gpus {args.n_gpus} with --grid latlon_bathy needs "
                f"n_lat ({_nlat}) divisible by n_gpus (the regular grid is "
                f"south-pole-walled, not folded, so it is NOT land-padded). "
                f"Choose --latlon-res with n_lat % {args.n_gpus} == 0.")
        grid, z_coord, model, state, H_bathy = build_latlon_bathy(
            args.nlev, args.H_max, args.mesh, n_lat=_nlat, n_lon=_nlon,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            pgf_scheme=args.pgf_scheme, A_h=args.A_h, B_h=args.B_h, K_bih=args.K_bih,
            flat_bottom=args.flat_bottom, A_h_eq_boost=args.A_h_eq_boost,
            A_h_eq_sigma_deg=args.A_h_eq_sigma_deg,
            ke_gradient_scheme=args.ke_gradient_scheme,
            partial_cell=args.partial_cell,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            bathy_smoothing_passes=args.bathy_smoothing_passes,
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            freeze_floor=(True if args.freeze_floor else None),
            freezing=_freezing_ovr,
            runoff_depth_spread_m=args.runoff_depth_spread_m,
            tracer_advection=args.tracer_advection,
            barotropic_solver=args.barotropic_solver,
            barotropic_pcg_variant=args.barotropic_pcg_variant,
            barotropic_diffusion_alpha=args.barotropic_diffusion_alpha,
            n_barotropic_substeps=args.n_barotropic_substeps,
            barotropic_time_filter=args.barotropic_time_filter,
            bottom_drag_r=args.bottom_drag_r,
            C_smag=args.C_smag, C_leith=args.C_leith, C_smag_lap=args.C_smag_lap,
            momentum_advection=args.momentum_advection,
            slope_foot_alpha=args.slope_foot_alpha,
            slope_foot_n_levels=args.slope_foot_n_levels,
            slope_foot_threshold=args.slope_foot_threshold,
            min_levels=args.min_levels,
            div_damp_2=args.div_damp_2, div_damp_4=args.div_damp_4,
            smag_cfl_safety=args.smag_cfl_safety,
            # tri-state: None -> keep the config default (bathy forces ON, #939);
            # True/False -> explicit override via build_latlon_bathy's _ovr.
            use_polar_filter=args.polar_filter,
            polar_filter_cutoff_lat_deg=args.polar_filter_cutoff_lat,
            polar_filter_max_wave_speed=args.polar_filter_max_wave_speed,
            polar_filter_safety_factor=args.polar_filter_safety,
            mle=mle_cfg, dz_ref_override=_nemo_dz,
            mask_marginal_seas=args.mask_marginal_seas,
            bottom_drag_scheme=args.bottom_drag_scheme,
            bottom_drag_cd0=args.bottom_drag_cd0,
            bottom_drag_cdmax=args.bottom_drag_cdmax,
            bottom_drag_z0=args.bottom_drag_z0,
            bottom_drag_ke0=args.bottom_drag_ke0,
            iwm=_iwm_cfg, iwm_forcing_file=args.iwm_forcing_file,
            ddm=_ddm_cfg,
            # KPP Ri_crit/Cv override (shoal the too-deep winter ML). None
            # unless --kpp-ri-crit/--kpp-cv given -> default KPPConfig unchanged.
            vertical_mixing=_kpp_vmix_override(args.kpp_ri_crit, args.kpp_cv, args.kpp_eice),
            prescribed_flow=args.prescribed_flow,
            no_gm_redi=args.no_gm_redi,
            # #1442: same --gateway-transports override the tripole branch
            # passes.  "latlon" IS in SUPPORTED_APP_GRIDS, so leaving it out
            # made every supported latlon gateway run fall back to the h*u
            # reconstruction the flag exists to replace (codex RED 6).
            store_mass_flux=bool(getattr(args, "gateway_transports",
                                         False)),
            store_salt_flux=bool(getattr(args, "gateway_transports",
                                         False)),
        )
        app_grid_type = "latlon"

    # --config (#376 Phase 4): apply the ocean YAML's explicit ocean.* physics
    # fields onto the built config and rebuild the model (same path the flag
    # overrides above use). Only the lat-lon C-grid model (tripole /
    # latlon_bathy) is wired for YAML config overrides.
    if ocean_adapter is not None:
        _explicit = dict(ocean_adapter.get("ocean") or {})
        if _explicit:
            if args.grid not in ("tripole", "latlon_bathy"):
                raise ValueError(
                    f"--config ocean.* overrides are only supported for the "
                    f"lat-lon C-grid model (grid=tripole|latlon_bathy), not "
                    f"grid={args.grid!r}. Remove the ocean: section or pick a "
                    f"lat-lon grid."
                )
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel,
            )
            _yaml_cfg = ocean_adapter.to_ocean_config()
            # #501/#661: ocean.* YAML keys are FLAT names; grouped members
            # (A_h/C_smag_lap/barotropic_solver/...) live in nested sub-configs,
            # so read via flat_get (getattr would AttributeError on them).
            _ovr = {k: _yaml_cfg.flat_get(k) for k in _explicit}
            # The builders force implicit_vertical_mixing=True (the root-cause
            # fix; explicit KPP vertical viscosity is CFL-unstable at the NEMO
            # 75-level ~1 m top cell -> cold-start blowup). A YAML override must
            # not silently revert that to the explicit path. (codex review)
            if _ovr.get("implicit_vertical_mixing") is False:
                raise ValueError(
                    "--config ocean.implicit_vertical_mixing=false is rejected: "
                    "explicit vertical mixing is CFL-unstable at a NEMO-vertical "
                    "(~1 m surface) grid and blows the cold-start. Remove the "
                    "override (the OMIP builders force implicit) or use a coarse "
                    "vertical grid where the explicit-diffusion CFL is satisfied.")
            # A YAML ocean.physics block REPLACES the whole physics config, which
            # would silently clobber a CLI --kpp-ri-crit/--kpp-cv override that
            # build_latlon_bathy already threaded into live KPP (codex). Fail
            # loud on the conflict rather than let YAML win over the explicit CLI.
            if "physics" in _ovr and (args.kpp_ri_crit is not None
                                      or args.kpp_cv is not None
                                      or args.kpp_eice is not None):
                raise ValueError(
                    "--kpp-ri-crit/--kpp-cv/--kpp-eice conflict with a --config "
                    "ocean.physics block: the YAML physics config would overwrite "
                    "the CLI KPP override. Set Ri_crit/Cv/eice in the YAML "
                    "(ocean.physics.vertical_mixing.kpp) OR drop the ocean.physics "
                    "section and use the CLI flags -- not both.")
            # Same class of conflict for #1442 (codex round-6 RED 3): this
            # rebuild happens AFTER the gateway builders set store_mass_flux
            # from --gateway-transports, so a YAML
            # ``ocean: {store_mass_flux: false}`` would silently switch the
            # capture back off and the gateway accumulator would quietly
            # integrate the h*u reconstruction under a flag that promises the
            # exact flux.  Fail loud; YAML never wins over the explicit CLI.
            if ("store_mass_flux" in _ovr
                    and getattr(args, "gateway_transports", False)
                    and not _ovr["store_mass_flux"]):
                raise ValueError(
                    "--gateway-transports conflicts with --config "
                    "ocean.store_mass_flux=false: the flag turns the capture ON "
                    "so the diagnostic integrates the flux the model actually "
                    "advected with, and the YAML would turn it back off AFTER "
                    "the builder, silently downgrading the diagnostic to the "
                    "h*u reconstruction. Drop one of the two.")
            # Identical conflict class for the EXACT-salt capture: the
            # builders set store_salt_flux from --gateway-transports and the
            # per-step gateway call passes require_salt=True, so a YAML
            # switch-off would turn the promised exact channel into a raise
            # at step 1 -- fail at setup instead, with the reason.
            if ("store_salt_flux" in _ovr
                    and getattr(args, "gateway_transports", False)
                    and not _ovr["store_salt_flux"]):
                raise ValueError(
                    "--gateway-transports conflicts with --config "
                    "ocean.store_salt_flux=false: the flag turns the "
                    "exact-salt capture ON (require_salt=True in the gateway "
                    "accumulator), and the YAML would turn it back off AFTER "
                    "the builder. Drop one of the two.")
            model = LatLonCGridOceanModel(
                grid, z_coord, model.config.replace_flat(**_ovr),
                # Preserve the zdfiwm maps through the YAML rebuild (codex
                # r2 #2: dropping them silently reverts file-map IWM to the
                # uniform fallback).
                iwm_forcing=getattr(model, "_iwm_forcing", None),
            )
            print(f"[setup] --config {args.config} ocean override: {sorted(_ovr)}")

    # OMIP-2 weak SSS restoring toward the WOA surface-salinity climatology (the
    # protocol NEMO ORCA1 uses).  Bounds the multi-year surface-freshwater drift
    # (the 5-yr MPAS run drifted 35->31 PSU without it).  Capture the target =
    # the UNSMOOTHED WOA surface salinity BEFORE --woa-smoothing-passes damps the
    # IC fronts (restoring must target the true climatology, not the smoothed IC).
    visc_schedule = None
    # Index of the NEXT schedule segment to apply. Fast-forwarded below when
    # resuming: left at zero, a restart at day 60 would first re-apply the
    # day-0 viscosity to a day-60 state and integrate with the wrong lateral
    # mixing until the schedule caught up.
    visc_seg_idx = 0
    if args.visc_schedule:
        if app_grid_type not in ("tripole", "latlon"):
            raise ValueError("--visc-schedule is wired for tripole/latlon "
                             f"(LatLonCGridOceanModel); got {args.grid!r}.")
        visc_schedule = []
        for seg in args.visc_schedule.split(","):
            parts = seg.strip().split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"--visc-schedule segment {seg!r} must be "
                    "'day:A_h:C_smag_lap'.")
            _d, _a, _c = (float(parts[0]), float(parts[1]), float(parts[2]))
            # strict validation (codex MED): finite, non-negative; NaN would
            # evade config checks and comparisons.
            if not (np.isfinite(_d) and np.isfinite(_a) and np.isfinite(_c)):
                raise ValueError(f"--visc-schedule segment {seg!r}: all "
                                 "fields must be finite.")
            if _d < 0.0 or _a < 0.0 or _c < 0.0:
                raise ValueError(f"--visc-schedule segment {seg!r}: day, "
                                 "A_h and C_smag_lap must be >= 0.")
            visc_schedule.append((_d, _a, _c))
        _days = [d for d, _, _ in visc_schedule]
        if any(b <= a for a, b in zip(_days, _days[1:])):
            raise ValueError("--visc-schedule days must be STRICTLY "
                             "ascending (duplicates would apply a segment "
                             "one timestep late).")
        print(f"[setup] viscosity schedule: {visc_schedule}")
        # Provenance (codex HIGH): the manifest's runtime_config snapshots
        # only the INITIAL A_h/C_smag_lap; the full schedule is recorded in
        # the manifest command_line AND in this explicit sidecar.
        if _is_io_proc():
            try:
                import json as _json
                Path(args.output).mkdir(parents=True, exist_ok=True)
                with open(Path(args.output) / "visc_schedule.json", "w") as _f:
                    _json.dump({"segments_day_Ah_Csmaglap": visc_schedule}, _f,
                               indent=1)
            except Exception as _e:  # noqa: BLE001 — provenance best-effort
                print(f"[warn] visc_schedule sidecar not written: {_e}")

    # (--prescribed-flow gates ran PRE-BUILD via validate_prescribed_flow_args;
    # the lever itself was threaded into the model config at build.)
    if args.dm2dc and app_grid_type not in ("tripole", "latlon"):
        raise SystemExit(
            "--dm2dc is wired for tripole/latlon (the applicator needs the "
            f"2-D tracer lon/lat); got {args.grid!r}")
    isf_forcing = None
    if args.isf:
        # NEMO ISF 'spe' prescribed melt: load the monthly Depoorter fields
        # on the model tracer grid (eORCA1 passthrough on the tripole;
        # nearest-wet + melt-total-preserving regrid elsewhere).
        if app_grid_type not in ("tripole", "latlon"):
            raise SystemExit(
                "--isf is wired for tripole/latlon (host post-step apply); "
                f"got {args.grid!r}")
        if not args.isf_forcing_file:
            raise SystemExit(
                "--isf requires --isf-forcing-file (the NEMO "
                "runoff-icb_DaiTrenberth_Depoorter.nc layout with sornfisf/"
                "sodepmin_isf/sodepmax_isf)")
        if args.runoff and (Path(args.isf_forcing_file).resolve()
                            != Path(_RUNOFF_NC).resolve()):
            # The --runoff loader dropped sornfisf from _RUNOFF_NC on the
            # assumption --isf deposits the SAME field at depth (double-count
            # fix); with a different ISF source the totals may not correspond
            # (codex r1 #4).  Not fatal — excluding the surface copy is still
            # the right call — but make the mismatch loud.
            warnings.warn(
                f"--isf-forcing-file {args.isf_forcing_file!r} differs from "
                f"the surface-runoff source {_RUNOFF_NC!r}: the surface "
                "sornfisf was excluded from --runoff assuming --isf re-injects "
                "the SAME ice-shelf melt at depth; verify the two files carry "
                "the same sornfisf climatology.", RuntimeWarning)
        from legoesm.ocean.forcing.isf_spe import load_isf_spe_forcing
        if app_grid_type == "tripole":
            _isf_lat = np.degrees(np.asarray(grid.lat_T))
            _isf_lon = np.degrees(np.asarray(grid.lon_T))
        else:
            _isf_lat = np.degrees(np.asarray(grid.lat))
            _isf_lon = np.degrees(np.asarray(grid.lon))
        isf_forcing = load_isf_spe_forcing(
            args.isf_forcing_file, _isf_lat, _isf_lon,
            land_mask=np.asarray(state.land_mask.data))
        _isf_tot = [float((isf_forcing.fwf[m]
                           * np.asarray(grid.area)).sum()) * 1e-9
                    for m in range(12)]
        print(f"[setup] ISF 'spe' melt loaded: monthly totals "
              f"{min(_isf_tot):.3f}-{max(_isf_tot):.3f} mSv-scale "
              f"(x1e6 kg/s), file={args.isf_forcing_file}")
    bbl_geom = None
    bbl_face_widths = None
    if args.bbl_adv:
        # NEMO advective BBL (trabbl nn_bbl_adv=2): static geometry from the
        # partial-cell reference thicknesses + NEMO mask; host post-step
        # application (same pattern as restoring / ice-thermo).
        if app_grid_type not in ("tripole", "latlon"):
            raise ValueError("--bbl-adv is wired for tripole/latlon only "
                             f"(got grid {args.grid!r}).")
        from legoesm.ocean.vertical import OceanPartialCellCoordinate
        if not isinstance(z_coord, OceanPartialCellCoordinate):
            raise ValueError("--bbl-adv requires --partial-cell (the BBL "
                             "geometry comes from per-cell bottom levels).")
        from legoesm.ocean.physics.bbl_adv import bbl_static_geometry
        bbl_geom = bbl_static_geometry(
            jnp.asarray(z_coord.h_partial),
            jnp.asarray(state.land_mask.data))
        if app_grid_type == "tripole":
            # tripole carries face metrics: dy_u (n_lat, n_lon+1 with wrap),
            # dx_v (n_lat+1, n_lon). Interior faces: between cols i,i+1 ->
            # u-face index i+1; between rows j,j+1 -> v-face index j+1.
            _dyu = jnp.asarray(grid.dy_u)[:, 1:-1]
            _dxv = jnp.asarray(grid.dx_v)[1:-1, :]
        else:
            # regular lat-lon: dy const, dx = R cos(lat) dlon at the v-face
            # rows / cell rows (faces share the row latitude for dy_u).
            _lat = np.asarray(grid.lat)            # (n_lat,) rad
            _nlat, _nlon = state.land_mask.data.shape
            _dlat = float(_lat[1] - _lat[0])
            _dlon = 2.0 * np.pi / _nlon
            from legoesm import constants as _const
            _R = float(getattr(grid, "radius", _const.R_earth))
            dy = _R * _dlat
            _dyu = jnp.full((_nlat, _nlon - 1), dy)
            _latv = 0.5 * (_lat[:-1] + _lat[1:])
            _dxv = jnp.asarray(
                (_R * np.cos(_latv) * _dlon)[:, None]
                * np.ones((1, _nlon)))
        bbl_face_widths = (_dyu, _dxv)
        print(f"[setup] BBL-adv ON (Campin-Goosse gamma={args.bbl_gamma_s}s): "
              f"active i-faces "
              f"{int(np.asarray(bbl_geom.u_active).sum())}, j-faces "
              f"{int(np.asarray(bbl_geom.v_active).sum())}")

    lat2d, lon2d = _grid_lat2d_deg(grid, args.grid)

    if args.woa_smoothing_passes and args.woa_smoothing_passes > 0:
        if not args.woa_init:
            raise ValueError("--woa-smoothing-passes requires --woa-init.")
        if args.grid == "mpas":
            raise ValueError(
                "--woa-smoothing-passes is not available for mpas: smooth_woa_ts "
                "uses the structured 2-D laplacian_smooth_2d; a Voronoi "
                "connectivity smoother (cellsOnCell) is future work.")
        state = smooth_woa_ts(state, grid, args.woa_smoothing_passes)

    if args.nemo_monthly_init is not None:
        # Exact-recipe IC: NEMO's own monthly init (sn_tem/sn_sal) at the
        # start month — the annual --woa-init leaves Arctic shelves ~1-3
        # PSU salty vs NEMO's January state.
        if args.woa_smoothing_passes and int(args.woa_smoothing_passes) > 0:
            raise SystemExit(
                "--nemo-monthly-init with --woa-smoothing-passes would "
                "silently smooth away the exact NEMO IC — drop one "
                "(codex r11 MED#2).")
        if app_grid_type == "mpas":
            raise SystemExit(
                "--nemo-monthly-init is wired for the structured grids "
                "(tripole/latlon); MPAS keeps its own IC path.")
        from legoesm.ocean.forcing.nemo_native_fields import (
            load_nemo_monthly_init_ts,
        )
        _T_ic, _S_ic = load_nemo_monthly_init_ts(
            args.nemo_monthly_init[0], args.nemo_monthly_init[1],
            lat2d, lon2d, n_levels=int(z_coord.n_levels),
            month=int(args.nemo_init_month))
        _Td = state.T.data.dtype
        state = state._replace(
            T=state.T.replace(data=jnp.asarray(_T_ic, dtype=_Td)),
            S=state.S.replace(data=jnp.asarray(_S_ic, dtype=_Td)))
        print(f"[setup] NEMO monthly init: month {args.nemo_init_month} "
              f"from {args.nemo_monthly_init[0].rsplit('/', 1)[-1]} / "
              f"{args.nemo_monthly_init[1].rsplit('/', 1)[-1]}")

    sss_restore_cfg = None
    # Defined unconditionally: the step loop reads it next to a
    # `sss_restore_cfg is not None` short-circuit, and relying on that
    # evaluation order for a name to exist is one reorder away from a
    # NameError deep inside a multi-day run.
    _sss_water_flux = False
    sss_restore_target = None
    # Monthly (sn_sss climatology) vs static (IC-surface) SSS target, detected
    # grid-agnostically below: monthly carries a leading 12-month axis ON TOP OF
    # the grid's spatial rank -> structured (12, ny, nx); MPAS (12, nCells).
    _sss_monthly = False
    if args.sss_restore:
        if app_grid_type == "cubed_sphere":
            raise ValueError("--sss-restore: not wired for the cube (parked grid).")
        if not args.woa_init and args.sss_restore_file is None:
            raise ValueError(
                "--sss-restore requires --woa-init (IC-surface target) or "
                "--sss-restore-file (NEMO sn_sss monthly climatology).")
        if not (float(args.sss_restore_tau_days) > 0.0):
            raise ValueError("--sss-restore-tau-days must be > 0 (0 divides by "
                             "zero in build_region_masks; negative = anti-restoring).")
        from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig
        from legoesm import constants
        _cfg_kwargs = {}
        if args.sss_ice_gate_nemo:
            _cfg_kwargs["ice_gate_mode"] = "nemo_linear"
        if args.sss_restore_normalization is not None:
            _cfg_kwargs["normalization"] = args.sss_restore_normalization
        if args.sss_restore_bound_mmday is not None:
            if not (float(args.sss_restore_bound_mmday) > 0.0):
                raise ValueError("--sss-restore-bound-mmday must be > 0.")
            # mm/day water-equivalent -> kg/m^2/s (rho_water * m/day / 86400).
            _cfg_kwargs["max_flux_kg_m2_s"] = (
                float(args.sss_restore_bound_mmday) * 1.0e-3 / 86400.0
                * float(constants.rho_water))
        # Water-flux channel switch, resolved once so the step loop reads a
        # plain bool (and so an unset flag can never accidentally enable it).
        _sss_water_flux = (args.sss_restore_channel == "water_flux")
        sss_restore_cfg = SSSRestoringConfig(
            enabled=True,
            tau_restore_days_default=float(args.sss_restore_tau_days),
            **_cfg_kwargs,
        )
        if args.sss_restore_file is not None:
            from legoesm.ocean.forcing.nemo_native_fields import (
                load_nemo_sss_restoring_climatology,
            )
            sss_restore_target = load_nemo_sss_restoring_climatology(
                args.sss_restore_file, lat2d, lon2d,
                np.asarray(state.land_mask.data) > 0.5)   # (12, n_lat, n_lon)
        else:
            sss_restore_target = np.asarray(
                state.S.data, dtype=np.float64)[..., 0].copy()  # surface SSS
        _wet = np.asarray(state.land_mask.data) > 0.5
        # Monthly iff a leading 12-axis sits on top of the grid's spatial rank
        # (structured 2-D -> 3-D; MPAS 1-D -> 2-D).  The bare ``ndim == 3`` test
        # this replaced mis-classed the MPAS monthly (12, nCells) array (ndim 2)
        # as a static target and applied the (nCells,) wet mask to the 12-axis.
        _sss_monthly = (sss_restore_target.shape[0] == 12
                        and sss_restore_target.ndim == _wet.ndim + 1)
        _bnd = (f"{args.sss_restore_bound_mmday:.1f} mm/day (NEMO ln_sssr_bnd)"
                if args.sss_restore_bound_mmday is not None
                else "200 mm/day safety cap")
        _tgt_kind = ("NEMO sn_sss monthly clim"
                     if _sss_monthly else "WOA surface SSS")
        _tgt_wet = (sss_restore_target[:, _wet]
                    if _sss_monthly
                    else sss_restore_target[_wet])
        print(f"[setup] SSS restoring ON: tau_default="
              f"{args.sss_restore_tau_days:.0f} d + OMIP-2 regional masks; "
              f"flux bound {_bnd}; target = {_tgt_kind} "
              f"[{_tgt_wet.min():.1f},{_tgt_wet.max():.1f}] PSU")


    # apply_balanced_init is the lat-lon C-grid geostrophic cold-start (tripole/
    # latlon).  The cube's FC-gradient balanced-init was removed with the FC
    # A-grid backend; a C-D grid cube balanced-init is future work.
    if args.balanced_init and args.grid == "cubed_sphere":
        raise ValueError(
            "--balanced-init is not available for cubed_sphere: the cube "
            "FC-gradient balanced-init was removed with the deprecated FC A-grid "
            "backend. A C-D grid cube balanced-init is future work."
        )
    if args.balanced_init and args.grid == "mpas":
        raise ValueError(
            "--balanced-init is not available for mpas: apply_balanced_init uses "
            "lat-lon C-grid gradient operators; an MPAS TRiSK geostrophic init is "
            "future work."
        )
    if args.balanced_init and args.grid not in ("cubed_sphere", "mpas"):
        if not args.woa_init:
            raise ValueError("--balanced-init requires --woa-init (it balances the WOA IC).")
        state = apply_balanced_init(
            state, grid, z_coord, model.config,
            taper_lat_deg=args.balanced_init_taper_lat,
            with_ssh=(not args.no_balanced_ssh),
        )


    runoff_monthly = None
    if args.runoff:
        if app_grid_type == "cubed_sphere":
            raise ValueError("--runoff: not wired for the cube (parked grid).")
        # Coastal-spread passes: MPAS uses the Voronoi-topology smoother in
        # load_runoff_monthly (cellsOnCell neighbour-average), so it gets the SAME
        # spreading the structured grids always had.  MPAS needs MORE passes than
        # lat-lon: the IDW k=4 regrid concentrates each river into ~4 Voronoi cells,
        # so un-spread the Amazon/Arctic over-freshen their mouths by -2 to -3 PSU
        # (regional SSS-band diagnostic).  A spread-passes SENSITIVITY (ico6 day-90,
        # SSS bias vs NEMO) is MONOTONE with no downside to the open Pacific/Southern
        # Ocean: Amazon basin -1.04 (sp2) -> -0.70 (sp4) -> -0.335 (sp8); global SSS
        # rmse 1.046 -> 1.01 -> 0.990.  8 still leaves the basin NEGATIVE (the real
        # plume is preserved, not washed out); beyond ~8 the gain plateaus and risks
        # over-diffusing the plume, so 8 is the default.  The area-conservative
        # renorm keeps the global total exact regardless.
        if args.runoff_spread_passes is not None and int(args.runoff_spread_passes) < 0:
            raise SystemExit(
                "--runoff-spread-passes must be >= 0 "
                f"(got {args.runoff_spread_passes})")
        _spread = int(args.runoff_spread_passes) if args.runoff_spread_passes is not None \
            else (8 if app_grid_type == "mpas" else 2)
        runoff_monthly = load_runoff_monthly(
            grid, app_grid_type, lat2d, lon2d, args.mesh,
            land_mask=np.asarray(state.land_mask.data), spread_passes=_spread,
            # --isf deposits the SAME file's sornfisf at depth -> drop it from
            # the surface runoff or the ice-shelf melt is counted twice.
            exclude_isf=args.isf)
    if ((args.runoff_dep_max is not None or args.runoff_rnf_max is not None)
            and not args.runoff_depth_nemo_ini):
        raise SystemExit(
            "--runoff-dep-max/--runoff-rnf-max require --runoff-depth-nemo-ini "
            "(they override the NEMO ln_rnf_depth_ini map; without it they would "
            "silently do nothing).")
    if args.runoff_depth_nemo_ini:
        # NEMO ln_rnf_depth_ini: per-cell spread depth from the runoff
        # climatology maximum — small Arctic/Siberian rivers stay near-surface
        # (a flat 150 m dilutes their shelf plumes several PSU salty).  Wired
        # for the structured C-grids (tripole/latlon) AND the MPAS Voronoi
        # core: the per-cell map (nemo_runoff_depth_map) and the freshwater
        # spread closure (runoff_spread_virtual_salt_tendency_3d) are BOTH
        # grid-agnostic — the MPAS state is the flattened (nCells,) / (nCells,
        # nlev) analogue of the C-grid's per-column arrays — so the same lever
        # keeps river plumes shallow (more shelf-surface freshening) on all
        # three grids.  Column-integral freshwater/salt is unchanged; only the
        # vertical distribution shifts (sign: freshwater +into ocean lowers S).
        if runoff_monthly is None:
            raise SystemExit("--runoff-depth-nemo-ini requires --runoff.")
        if app_grid_type == "cubed_sphere":
            raise SystemExit(
                "--runoff-depth-nemo-ini is not wired for the cube (parked "
                "grid; the OMIP runner folds freshwater onto surface_forcing "
                "there rather than passing freshwater= to the model).")
        if float(getattr(model.config, "runoff_depth_spread_m", 0.0)) > 0.0:
            raise SystemExit(
                "--runoff-depth-nemo-ini and --runoff-depth-spread-m are "
                "mutually exclusive.")
        from legoesm.ocean.forcing.runoff_depth import nemo_runoff_depth_map
        _rd_kw = {}
        if args.runoff_dep_max is not None:
            if not (np.isfinite(args.runoff_dep_max) and args.runoff_dep_max > 0):
                raise SystemExit("--runoff-dep-max must be finite and > 0.")
            _rd_kw["dep_max"] = float(args.runoff_dep_max)
        if args.runoff_rnf_max is not None:
            if not (np.isfinite(args.runoff_rnf_max) and args.runoff_rnf_max > 0):
                raise SystemExit("--runoff-rnf-max must be finite and > 0.")
            _rd_kw["rnf_max"] = float(args.runoff_rnf_max)
        _h_rnf = nemo_runoff_depth_map(
            np.asarray(runoff_monthly), np.asarray(H_bathy), **_rd_kw)
        if app_grid_type == "mpas":
            # Voronoi MPAS: rebuild the model with the per-cell map threaded
            # into MPASOceanConfig.  The freshwater application in
            # ocean_pe_mpas.py reads runoff_depth_spread_map via the shared
            # resolve_runoff_spread_arg selector (same code path as the
            # C-grid).  ``grid`` holds the Voronoi mesh here.  MPAS has no
            # iwm_forcing (rejected in build_mpas_ocean), so the C-grid's
            # iwm_forcing= kwarg is intentionally omitted.
            from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
            model = MPASOceanModel(
                grid, z_coord,
                model.config._replace(
                    runoff_depth_spread_map=jnp.asarray(_h_rnf)))
        else:
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel,
            )
            model = LatLonCGridOceanModel(
                grid, z_coord,
                model.config._replace(
                    runoff_depth_spread_map=jnp.asarray(_h_rnf)),
                iwm_forcing=getattr(model, "_iwm_forcing", None))
        _wetm = np.asarray(state.land_mask.data) > 0.5
        print(f"[setup] NEMO runoff depth map (ln_rnf_depth_ini): "
              f"h_rnf wet range [{_h_rnf[_wetm].min():.1f},"
              f"{_h_rnf[_wetm].max():.1f}] m; "
              f"{(np.asarray(runoff_monthly).max(0)[_wetm] > 0).sum()} "
              f"runoff cells")
    _fw_cfg_kw = {}
    if args.freshwater_salinity != "s_ref":
        _fw_cfg_kw["freshwater_salinity"] = args.freshwater_salinity
    if args.freshwater_closure is not None:
        # Default None means "leave the config default alone", so an
        # unset flag stays bit-identical to previous runs.
        _fw_cfg_kw["freshwater_closure"] = args.freshwater_closure
        # #1484 codex HIGH: SSS restoring is DERIVED as a virtual-salt
        # equivalent -- sss_restoring inverts a target salt tendency into a
        # water flux using S_target and a CONFIGURED z1. Under
        # real_freshwater there is no virtual-salt term at all: the same
        # number acts through volume-only dilution, whose realized strength
        # depends on the LIVE salinity and the ACTUAL top-cell thickness, so
        # the relaxation is the wrong magnitude. Refuse the pairing here --
        # this is where restoring is switched on -- rather than in the ocean
        # config, which has no restoring field to key off (the flux arrives
        # as FreshwaterForcing.restoring, a traced array).
        _fw_conflict = real_freshwater_restoring_conflict(
            args.freshwater_closure, getattr(args, "sss_restore", False),
            args.sss_restore_normalization, args.sss_restore_channel)
        if _fw_conflict is not None:
            raise SystemExit(_fw_conflict)
    if args.no_normalize_freshwater:
        # EXPLICIT opt-out of the global-freshwater normalization.  The
        # CORE-II P-E+R integral is a real ~+0.65 Sv imbalance, so turning
        # this off re-admits a ~-0.5 PSU/90d global-mean fresh drift —
        # accepted ONLY for controlled probes (e.g. --freshwater-salinity
        # local, whose combination with the normalization is rejected by the
        # model config until a joint volume+salt correction exists).
        _fw_cfg_kw["normalize_freshwater"] = False
    if _fw_cfg_kw:
        # NEMO tra_sbc virtual-salt convention (sfx = emp * sss_local) and/or
        # normalization opt-out: rebuild the model with the selections
        # threaded into the dynamics config (the SAME NamedTuple-replace
        # rebuild the runoff-depth-map block uses).  latlon + tripole share
        # LatLonCGridOceanModel; the cube's 'external' physics already
        # applies its virtual salt at the LOCAL S_top, and the spectral path
        # has no freshwater channel — both are rejected upstream of this
        # OMIP host loop for salinity-faithful runs.
        if app_grid_type == "mpas":
            from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
            model = MPASOceanModel(
                grid, z_coord,
                model.config._replace(**_fw_cfg_kw))
        elif app_grid_type in ("latlon", "tripole"):
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel,
            )
            model = LatLonCGridOceanModel(
                grid, z_coord,
                model.config._replace(**_fw_cfg_kw),
                iwm_forcing=getattr(model, "_iwm_forcing", None))
        else:
            raise SystemExit(
                f"--freshwater-salinity/--no-normalize-freshwater are wired "
                f"for latlon/tripole/mpas only (got grid {app_grid_type}).")
        print(f"[setup] freshwater config overrides: {_fw_cfg_kw}")
    # Prescribed sea-ice-concentration field for the SW-albedo surrogate
    # (--ice-albedo) AND the NEMO-faithful SSS-restoring ice gate (nn_sssr_ice=0:
    # no restoring under ice).  Loaded ONCE, regridded onto the model grid; passed
    # to compute_omip2_surface_forcing + the restoring every step (no per-step
    # recompute).  ``--ice-albedo-seasonal`` -> a (12, *grid) monthly climatology
    # (leading month axis indexed each step); otherwise a single annual map.  Also
    # loaded when only --sss-restore is on, so the restoring ice gate is faithful
    # even without the albedo.
    siconc_clim = None
    siconc_monthly = False
    if args.ice_albedo or args.sss_restore or args.ice_thermo:
        if args.ice_albedo_seasonal:
            siconc_clim = load_nemo_siconc_monthly(
                grid, app_grid_type, lat2d, lon2d, siconc_file=args.siconc_file,
                tos_file=args.tos_monthly_file,
                land_mask=np.asarray(state.land_mask.data))
            siconc_monthly = True
        else:
            siconc_clim = load_nemo_siconc(
                grid, app_grid_type, lat2d, lon2d, siconc_file=args.siconc_file,
                land_mask=np.asarray(state.land_mask.data))
    # Monthly chlorophyll for the RGB SW-penetration scheme (--sw-rgb-chl).  Loaded
    # once; indexed per step by calendar month, then attached to the surface
    # forcing as ``chl`` (the PE C-grid step switches to rgb_chl when chl is set).
    chl_clim = None
    if args.sw_rgb_chl:
        if app_grid_type not in ("latlon", "tripole"):
            raise ValueError(
                f"--sw-rgb-chl is wired for latlon/tripole only, not {app_grid_type!r}")
        chl_clim = load_nemo_chl_monthly(
            grid, app_grid_type, lat2d, lon2d, chl_file=args.chl_file)
    # allow_synthetic=False: this NEMO-faithful pipeline MUST use the real
    # 6-hourly CORE-II nyf.zarr; a silent fallback to 365 daily synthetic forcing
    # would corrupt the comparison invisibly. --forcing-path (set via --config
    # forcing.path) threads the staged data dir into the loader's cache_dir so a
    # fetch-then-run workflow finds it (#376).
    forcing = load_core2_nyf(
        allow_synthetic=False,
        cache_dir=(Path(args.forcing_path) if args.forcing_path else None),
    )
    n_rec = int(forcing.u10.shape[0])
    print(f"[setup] grid {lat2d.shape}, forcing records {n_rec}, dt={args.dt}s")
    # RESOLVED forcing archive, always logged.  With --forcing-path unset the
    # loader picks an environment/home-dependent cache, so two chained legs can
    # read DIFFERENT forcing from identical command lines.  It is hashed into
    # the restart fingerprint (a mismatch is then a hard error on resume), but
    # the digest is opaque — printing the path is what makes that error
    # DIAGNOSABLE from the two legs' logs.
    from legoesm.ocean.forcing import core2_nyf_path as _core2_path
    print("[setup] forcing archive: "
          f"{_core2_path(Path(args.forcing_path) if args.forcing_path else None)}",
          flush=True)

    # Prognostic sea-ice (--prognostic-sea-ice): build the canonical SeaIceConfig
    # + a zero-ice cold-start state on the OCEAN grid.  The REAL model
    # (legoesm.ice.step_sea_ice) is stepped each loop iteration and its
    # brine/melt/heat response is routed into the existing ocean channels.
    ice_config = None
    ice_state = None
    if args.prognostic_sea_ice:
        from legoesm.ice import (
            SeaIceConfig, init_dynamic_ice_state, step_sea_ice,
            distribute_dynamic_state_to_categories,
            grid_supports_ice_dynamics, grid_supports_ice_transport,
        )
        from legoesm.ice.config import BrineConfig, RidgingConfig
        # Free-drift fallback if the grid lacks strain-rate operators
        # (NOT 'none', which yields no drift/export).
        _ice_dyn = args.prognostic_ice_dynamics
        # The tripole grid object is a LatLonCGridGeometry: it now supports
        # TRANSPORT (fold-aware donor-cell C-grid advection,
        # grid_supports_ice_transport) but still lacks the curvilinear
        # strain-rate/stress-divergence ops for EVP/mEVP
        # (grid_supports_ice_dynamics), so the rheology degrades to
        # free_drift while the free-drift velocities ADVECT the ice tracers
        # (Fram/Bering export, marginal-zone divergence).  MPAS (VoronoiMesh)
        # remains the fully-supported mEVP target.
        _supports_dyn = grid_supports_ice_dynamics(grid)
        _supports_transport = grid_supports_ice_transport(grid)
        if not _supports_dyn and _ice_dyn in ("mevp", "evp"):
            print(f"[setup] prognostic ice: grid {type(grid).__name__} lacks "
                  f"strain-rate ops -> dynamics {_ice_dyn!r} -> 'free_drift' "
                  f"(transport {'advect' if _supports_transport else 'none'}; "
                  "brine salt + melt freshwater + ocean-heat export PRESERVED). "
                  "Use --grid mpas for full mEVP.")
            _ice_dyn = "free_drift"
        _transport = "advect" if _supports_transport else "none"
        _brine = BrineConfig(enabled=True)
        if args.prognostic_ice_salinity is not None:
            _brine = _brine._replace(S_ice_new=float(args.prognostic_ice_salinity))
        # Multi-category ITD (--ice-categories / --ice-ridging): resolve the
        # request against THIS grid's capabilities (refuse-not-ignore).
        _n_cat, _itd_remap, _ridging_on = _resolve_ice_categories(
            args.ice_categories, args.ice_ridging, _supports_dyn,
            type(grid).__name__)
        ice_config = SeaIceConfig(
            dynamics=_ice_dyn,
            transport=_transport,
            n_categories=_n_cat,
            itd_remap=_itd_remap,     # 'lipscomb2001' whenever _n_cat > 1
            ridging=RidgingConfig(enabled=_ridging_on),
            brine=_brine,            # brine-rejection salt flux -> ocean salt_flux
            # Under-ice transmitted SW is owned by the ICE model (constant-
            # scheme transmittance): the ice EB is debited and the ocean
            # receives it via resp.ocean_heat_extraction (-= sw_penetrated),
            # closing the SW budget the old ocean-side A*tau*swd surrogate
            # left open (codex L1).  The blend below therefore passes
            # sw_transmittance_ice=0.0.
            sw_transmittance_const=float(args.ice_thermo_sw_trans),
        )
        if args.ice_ocean_heat_coeff is not None:
            ice_config = ice_config._replace(
                ocean_heat_transfer_coeff=float(args.ice_ocean_heat_coeff))
        ice_shape = _ice_state_spatial_shape(grid, app_grid_type)
        # Zero-ice cold start (h=0, concentration=0); spins up from the forcing.
        ice_state = init_dynamic_ice_state(ice_shape, S_ice_init=0.0)
        ice_state = ice_state._replace(
            concentration=ice_state.concentration.replace(
                data=jnp.zeros_like(ice_state.concentration.data)))
        if args.ice_init is not None:
            # NEMO SI3 ice IC (at_i/ht_i/ht_s/sm_i/tmsu) -> the prognostic
            # state, replacing the zero-ice cold start.  Same embed /
            # nearest-wet convention as --nemo-monthly-init; zero-ice cells
            # stay zero (incl. S_ice=0, matching the fresh cold-start seed).
            from legoesm.ocean.forcing.nemo_native_fields import (
                load_nemo_ice_init,
            )
            _ice_ic = load_nemo_ice_init(
                args.ice_init, lat2d, lon2d,
                np.asarray(state.land_mask.data))
            ice_state = _apply_ice_init(ice_state, _ice_ic)
            _a0, _c0, _h0, _ = _ice_global_stats(
                ice_state, grid, state.land_mask.data)
            print(f"[setup] ICE INIT from "
                  f"{args.ice_init.rsplit('/', 1)[-1]}: "
                  f"area={_a0 / 1.0e12:.3f}e6 km2 mean_conc={_c0:.3f} "
                  f"max_h={_h0:.3f} m (fields: at_i,ht_i"
                  f"{',ht_s' if _ice_ic.h_snow is not None else ''}"
                  f"{',sm_i' if _ice_ic.S_ice is not None else ''}"
                  f"{',tmsu' if _ice_ic.T_su is not None else ''})")
        if _n_cat > 1:
            # Lift the (possibly IC-seeded) single-category state onto the
            # n_cat-bin ITD: delta seeding into the bin containing each
            # cell's thickness, snow/salinity/ponds riding along in the occupied bin.
            # AFTER --ice-init (the SI3 file carries aggregate fields only)
            # and BEFORE the ew-overlap slaving (tree_map, axis-1 safe on
            # the lifted fields).
            ice_state = distribute_dynamic_state_to_categories(
                ice_state, _n_cat)
        if args.ew_cyclic_overlap and app_grid_type == "tripole":
            # Slave the duplicated ORCA halo columns from the start (the
            # transport step re-imposes this every step below).
            ice_state = _ice_apply_ew_overlap(ice_state)
        from legoesm import constants as _ice_const
        _ice_T_freeze = float(_ice_const.T_freeze)   # degC ocean T -> K for ice
        _cat_str = (f" n_categories={_n_cat} itd={_itd_remap!r}"
                    f" ridging={'ON' if _ridging_on else 'off'}"
                    if _n_cat > 1 else "")
        print(f"[setup] PROGNOSTIC SEA ICE: step_sea_ice dynamics={_ice_dyn!r} "
              f"transport={_transport!r} brine=ON (S_ice_new="
              f"{_brine.S_ice_new:.1f} PSU) on {app_grid_type} shape {ice_shape}"
              f"{_cat_str}; "
              "salt_flux+ice_fw+heat -> existing surface/freshwater channels.")

    dt = float(args.dt)
    ramp_s = float(args.forcing_ramp_days) * _SEC_PER_DAY
    total_days = 10.0 if args.smoke else args.years * 365.0
    n_steps = int(total_days * _SEC_PER_DAY / dt)
    diag_every = max(1, int(args.diag_every_days * _SEC_PER_DAY / dt))
    steps_per_year = int(365.0 * _SEC_PER_DAY / dt)
    out_dir = Path(args.output)

    nudge_tau_s = float(args.nudge_woa_tau_days) * _SEC_PER_DAY
    nudge_release_s = float(args.nudge_release_day) * _SEC_PER_DAY
    drag_tau_s = float(args.spinup_drag_tau_days) * _SEC_PER_DAY
    drag_days_s = float(args.spinup_drag_days) * _SEC_PER_DAY
    if drag_tau_s > 0:
        print(f"[setup] spin-up velocity drag: tau={args.spinup_drag_tau_days}d "
              f"for first {args.spinup_drag_days}d")
    nudge_T = nudge_S = nudge_m3 = None
    if nudge_tau_s > 0:
        nudge_T, nudge_S = compute_woa_3d(
            grid, z_coord, args.woa_t, args.woa_s,
            H_bathy, np.asarray(state.land_mask.data),
        )
        nudge_m3 = np.asarray(state.land_mask.data)[..., None]
        print(f"[setup] nudging T,S -> WOA: tau={args.nudge_woa_tau_days}d, "
              f"release day={args.nudge_release_day or 'never'}")

    snap_every = (int(args.snapshot_every_days * _SEC_PER_DAY / dt)
                  if args.snapshot_every_days > 0 else 0)
    # Restart cadence is INDEPENDENT of the diagnostic snapshot cadence (the two
    # are different contracts), but defaults to it so one flag is usually
    # enough.  Both zero => end-of-run only, which a wallclock kill destroys —
    # warn, because silently having no mid-run checkpoint is the exact failure
    # this feature exists to prevent.
    restart_every = (int(args.restart_every_days * _SEC_PER_DAY / dt)
                     if args.restart_every_days > 0 else snap_every)
    if args.restart_save and restart_every <= 0:
        print("[warn] --restart-save with no cadence (--restart-every-days / "
              "--snapshot-every-days both 0): the restart is written ONLY at "
              "the end of the run, so a wallclock kill loses this leg "
              "entirely.", flush=True)

    # ------------------------------------------------------------------
    # --restart-from: resume the integration (full carry + sea ice + step).
    # Placed AFTER the ocean state, the sea-ice state and the step-derived run
    # controls exist, and BEFORE the step-0 diagnostic, so the [diag] line and
    # the CSV report the RESTORED state rather than the cold-start template.
    # The target is ABSOLUTE (--years / --smoke): a resumed leg integrates from
    # the restart step up to n_steps, it does not re-run n_steps more.
    # ------------------------------------------------------------------
    # RESOLVED-RUN fingerprint for the restart (codex r2/r3 HIGH).
    #
    # EXCLUSION list, not an inclusion list: every CLI setting is hashed unless
    # it is explicitly a run-control / IO knob that legitimately differs
    # between chained legs.  That is fail-CLOSED — a new flag added later is
    # covered automatically, whereas an inclusion list silently omits it.  The
    # earlier version hashed only repr(model.config) and so missed
    # --visc-schedule, --forcing-ramp-days, --sss-restore and every other
    # host-loop forcing knob, all of which change step N+1.
    #
    # Computed HERE, at setup, before the --visc-schedule mid-run model
    # rebuild, so a scheduled leg still matches its siblings.
    #
    # LIMITS, stated rather than papered over: values are hashed via repr(),
    # which ELIDES the interior of a large array (e.g. a runoff-depth map), and
    # the forcing archive is pinned by RESOLVED PATH, not by a content hash.
    # So this detects configuration DRIFT, not a deliberately forged archive or
    # a mutated forcing file at the same path.
    _RESTART_FP_EXCLUDE = frozenset({
        # resume plumbing + the absolute target, which grows leg by leg
        "restart_from", "restart_save", "restart_every_days", "years", "smoke",
        # pure output / cadence knobs
        "output", "snapshot_every_days", "diag_every_days",
        # read-only diagnostics that never touch the state (codex r4 LOW):
        # including them would false-abort a leg that merely turned a
        # diagnostic on or off.
        "gateway_transports", "diag_momentum_step",
    })
    # Path-valued args are normalised before hashing so an equivalent relative
    # path or symlink cannot false-abort a legitimate chained leg.
    _RESTART_FP_PATH_KEYS = frozenset({
        "forcing_path", "mesh", "config", "woa_t", "woa_s", "ice_init",
        "siconc_file", "tos_monthly_file", "chl_file",
        # codex r5 LOW: these were still hashed as RAW text, so an equivalent
        # relative or symlinked spelling false-aborted a valid chained leg.
        "nemo_vertical_file", "isf_forcing_file", "iwm_forcing_file",
        "sss_restore_file",
        # nargs=2: ONE dest holding two paths, normalised element-wise below.
        "nemo_monthly_init",
    })
    _restart_cfg_fp = None
    if args.restart_save or args.restart_from:
        import hashlib as _hashlib
        _fp_items = []
        for _k in sorted(vars(args)):
            if _k in _RESTART_FP_EXCLUDE:
                continue
            _v = getattr(args, _k)
            if _k in _RESTART_FP_PATH_KEYS and _v:
                if isinstance(_v, str):
                    _v = str(Path(_v).resolve())
                elif isinstance(_v, (list, tuple)):
                    _v = [str(Path(_e).resolve()) if isinstance(_e, str) and _e
                          else _e for _e in _v]
            _fp_items.append(f"{_k}={_v!r}")
        # The resolved model + sea-ice configs too: they capture defaults and
        # preset expansions that never appear as an explicit CLI value.
        _fp_items.append(f"model_config={model.config!r}")
        _fp_items.append(f"ice_config={ice_config!r}")
        _fp_items.append(f"grid_type={app_grid_type}")
        # RESOLVED forcing archive, not the raw flag (codex r8 MEDIUM): with
        # --forcing-path unset the loader falls back to an environment/home
        # dependent cache directory, so two legs whose command lines are
        # IDENTICAL (both recording forcing_path=None) can read different
        # CORE-II archives and still produce matching fingerprints.  Resolved
        # through the loader's own helper so the recorded path cannot drift
        # from the loaded one.
        from legoesm.ocean.forcing import core2_nyf_path
        _fp_forcing = core2_nyf_path(
            Path(args.forcing_path) if args.forcing_path else None)
        _fp_items.append(f"forcing_archive={_fp_forcing.resolve()}")
        # BEHAVIOUR-CHANGING ENVIRONMENT (codex r9 HIGH; narrowed in the tail
        # round to stop it false-aborting on per-job cache dirs and ports).
        # See _restart_env_items / _RESTART_ENV_EXCLUDE_* above.
        # JAX_ENABLE_X64 is pinned separately by the archive's _x64 record.
        _fp_items.append(f"env={_restart_env_items()!r}")
        _restart_cfg_fp = _hashlib.sha256(
            "|".join(_fp_items).encode("utf-8")).hexdigest()[:32]
    # SOURCE REVISION (codex r5 HIGH; scoping/dirty/explicit-failure fixed in
    # r6): a changed model implementation with identical options otherwise
    # resumes silently.  Recorded always, via _source_revision() — which scopes
    # the query to THIS script's checkout, marks a dirty tree, and returns
    # _SOURCE_REV_UNAVAILABLE instead of silently omitting the field.
    # DELIBERATELY A WARNING, NOT AN ABORT: chaining a multi-day production run
    # across a bug fix is a legitimate and expected workflow, and a hard error
    # would make the feature unusable exactly when it matters.  The state
    # itself is still validated by the config fingerprint; this line makes the
    # code drift visible in the log and in the archive so a scorecard is never
    # attributed to the wrong revision.  It is provenance, NOT a proof of
    # identical code — see ocean.restart's SCOPE OF THE GUARANTEE.
    _restart_src_sha = None
    if args.restart_save or args.restart_from:
        _restart_src_sha = _source_revision()
        if _restart_src_sha == _SOURCE_REV_UNAVAILABLE:
            print("[warn] could not determine the source revision of "
                  f"{Path(__file__).resolve().parent} (not a git checkout, or "
                  "git unavailable): the restart archive will record "
                  f"{_SOURCE_REV_UNAVAILABLE!r} and the leg-to-leg code-drift "
                  "check cannot run.", flush=True)
        else:
            _amb = _revision_ambiguity(_restart_src_sha)
            if _amb:
                print(f"[warn] source revision {_restart_src_sha} does not "
                      f"describe the code being executed: {_amb} Results from "
                      "this leg are not reproducible from the recorded "
                      "revision alone — commit, and run the driver and the "
                      "packages from ONE checkout, before a production leg.",
                      flush=True)

    start_step = 0
    if args.restart_save or args.restart_from:
        # FAIL FAST: if this build's state exposes a slot the restart
        # persistence policy does not classify, abort now (seconds in) rather
        # than at the first mid-run checkpoint, hours into an integration.
        from legoesm.ocean.restart import validate_restart_policy
        try:
            validate_restart_policy(state, ice_state)
        except KeyError as _pol_err:
            # KeyError's str() is repr-quoted; args[0] is the plain message.
            raise SystemExit(_pol_err.args[0]) from _pol_err
    if args.restart_save and args.restart_from:
        # Refuse to overwrite the archive we are resuming FROM: a leg that
        # blows up after its first cadence write would have destroyed the only
        # good parent restart, i.e. the spin-up this feature exists to keep.
        # Checked BEFORE the load so it costs nothing.
        if Path(args.restart_save).resolve() == Path(
                args.restart_from).resolve():
            raise SystemExit(
                "--restart-save and --restart-from point at the same file "
                f"({args.restart_save}); write the new leg to a distinct path "
                "so the parent restart survives a failed leg.")
    if args.restart_from:
        from legoesm.ocean.restart import load_run_restart
        _rs_path = Path(args.restart_from)
        if not _rs_path.exists():
            raise SystemExit(f"--restart-from: no such file {_rs_path}")
        # load_run_restart RAISES on an ice present/absent mismatch, so the
        # returned ice_state is non-None exactly when this run has prognostic
        # ice — no silent cold-start fallback is possible here.
        state, ice_state, _rs_meta = load_run_restart(
            _rs_path, state, ice_template=ice_state,
            grid_type=app_grid_type, dt_seconds=dt,
            n_forcing_records=n_rec, config_fingerprint=_restart_cfg_fp)
        # Source-revision drift: three DISTINCT outcomes (unknown / mismatch /
        # equal-but-dirty), decided by the pure helper so the logic is unit
        # tested rather than only exercised by a full driver run.
        _drift_note = _source_revision_drift_note(_rs_meta.get("sha"),
                                                  _restart_src_sha)
        if _drift_note:
            print(_drift_note, flush=True)
        start_step = int(_rs_meta["step"])
        if start_step >= n_steps:
            # Never exit silently "already at target" (CLAUDE.md run-target
            # rule): a chain launcher must see WHY nothing ran.
            raise SystemExit(
                f"--restart-from {_rs_path} is already at step {start_step} "
                f"(day {_rs_meta['time_days']:.2f}) but this run targets only "
                f"{n_steps} steps ({total_days:.0f} days).  Raise --years, or "
                "point at an earlier restart.")
        print(f"[restart] resumed from {_rs_path}: step {start_step} "
              f"(day {_rs_meta['time_days']:.2f}), carry slots "
              f"{sorted(_rs_meta['slots'])}"
              + (f", ice slots {sorted(_rs_meta['ice_slots'])}"
                 if _rs_meta["ice_slots"] else ", no sea ice"), flush=True)
        if getattr(args, "gateway_transports", False):
            print("[restart] NOTE --gateway-transports accumulates a TIME MEAN "
                  "from the resume point only; a chained run's per-leg means "
                  "must be recombined offline (weighted by leg length).",
                  flush=True)
    start_day = start_step * dt / _SEC_PER_DAY
    if visc_schedule and start_step:
        # Skip every segment whose start day the checkpoint is already past,
        # so the resumed leg begins on the viscosity the schedule says applies
        # at this time rather than replaying the ramp from the cold start.
        while (visc_seg_idx + 1 < len(visc_schedule)
               and visc_schedule[visc_seg_idx + 1][0] <= start_day):
            visc_seg_idx += 1
        print(f"[restart] viscosity schedule fast-forwarded to segment "
              f"{visc_seg_idx + 1}/{len(visc_schedule)} "
              f"(day {visc_schedule[visc_seg_idx][0]:g})", flush=True)

    print(f"[run] {total_days:.0f} days = {n_steps} steps "
          f"(diag every {diag_every} steps"
          f"{f', snapshot every {snap_every} steps' if snap_every else ''}"
          f"{f', resuming at step {start_step}' if start_step else ''})")
    d0 = _diag(state, lat2d, lon2d)
    print(f"[diag] step {start_step}: {d0}", flush=True)

    # Progress time-series CSV, flushed each diag -> observable mid-run even when
    # stdout is pipe-buffered, and a record for post-hoc analysis.  mkdir on EVERY
    # rank (idempotent, exist_ok=True -> no clobber): if only process 0 made the
    # dir and that raised, the other ranks would run on into the distributed step
    # and hang (codex MED).  Making the dir everywhere removes that single point of
    # failure; the actual writes (CSV / manifest / snapshots) stay process-0 only.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run manifest (#376 Phase 4): capture the FULL ocean experiment identity at
    # run start so the run is reconstructible and `legoesm reproduce` has a
    # reference. The hashed payload is an OceanRunRecord = runtime config + the
    # run controls (dt, total days, grid, mesh, output, forcing, IC) that live
    # OUTSIDE model.config — so two runs that differ only in dt/grid/output get
    # distinct config_hashes (codex review HIGH). Best-effort: a provenance-write
    # failure never aborts a long integration.
    # Process-0 only under --distributed (every process shares one output dir).
    #
    # Resolve the CORE-II cache the run will actually read, so the manifest
    # records the forcing rather than the flag.  --forcing-path wins; otherwise
    # this is the same default the loader takes, and the two cannot drift
    # because both call core2_nyf_cache_dir().
    if args.forcing_path:
        _resolved_forcing_path = str(args.forcing_path)
    else:
        from legoesm.ocean.forcing import core2_nyf_cache_dir
        _resolved_forcing_path = str(core2_nyf_cache_dir())
    print(f"[setup] CORE-II forcing cache: {_resolved_forcing_path}")
    manifest_path = None
    if _is_io_proc():
        try:
            from legoesm.driver.restart import (
                dataset_provenance_entry,
                write_run_manifest,
            )
            from legoesm.ocean.config import OceanRunRecord
            run_record = OceanRunRecord(
                runtime_config=model.config,
                grid=str(args.grid),
                mesh=str(args.mesh),
                nlev=int(args.nlev),
                dt_seconds=float(dt),
                total_days=float(total_days),
                output_path=str(args.output),
                forcing="core2_nyf",
                # RESOLVED, not the flag.  An empty string here used to mean
                # "the default", and the default silently changed from the raw
                # CORE-II winds to the bias-corrected ones -- so every manifest
                # written before this recorded nothing about which forcing the
                # run actually used, and the loader's own provenance line is
                # swallowed by the `| tail` most arm scripts pipe through.
                forcing_path=str(_resolved_forcing_path),
                woa_init=bool(args.woa_init),
                woa_t=str(args.woa_t or ""),
                woa_s=str(args.woa_s or ""),
                latlon_res=str(args.latlon_res),
                smoke=bool(args.smoke),
            )
            manifest_path = write_run_manifest(
                out_dir, run_record, config_kind="ocean",
                runner_tag="run_omip_core2",
                dataset_provenance=[
                    dataset_provenance_entry(pth, dataset_id=did)
                    for did, pth in (
                        ("core2_forcing", _resolved_forcing_path),
                        ("woa_t", args.woa_t),
                        ("woa_s", args.woa_s),
                    )
                    if pth
                ],
            )
            print(f"[setup] wrote run manifest {manifest_path}")
        except Exception as _exc:  # noqa: BLE001 — provenance is best-effort
            print(f"[warn] run manifest not written: {type(_exc).__name__}: {_exc}")

    _csv_cols = ["step", "day", "mean_sst_C", "mean_sss", "max_abs_u",
                 "max_abs_v", "umax_lat", "umax_lon", "umax_lev", "steps_per_s"]
    # Prognostic-ice columns (audited output gap: ice growth was invisible in
    # the run record): global ice area [m2], mean concentration over the
    # ice-covered wet cells, max thickness [m] — the _ice_global_stats trio
    # the stdout [ice] line also reports.  Added ONLY under
    # --prognostic-sea-ice so ice-free runs keep the legacy header.
    if ice_config is not None:
        _csv_cols += ["ice_area_m2", "ice_mean_conc", "ice_max_thick_m"]
        _lm_csv = np.asarray(state.land_mask.data)
    # Process-0-only CSV under --distributed: every process runs the same host
    # loop on the all-gathered replicated state, so a single writer suffices and
    # avoids N processes clobbering the same file.  On non-IO ranks _csv is None
    # and the writer/closer below are no-ops.
    _csv = None
    _csv_appended = False
    if _is_io_proc():
        # Resuming APPENDS to an existing series (a chained leg must not erase
        # the parent leg's record); a fresh run truncates and writes the header.
        _csv_path = out_dir / "diag_timeseries.csv"
        # A header-only or empty file is NOT a parent series: appending to it
        # and then suppressing the restart row would leave the leg with no
        # starting point at all (codex r3 LOW).
        _csv_has_rows = False
        if _csv_path.exists():
            with open(_csv_path) as _fh:
                # Count NON-BLANK lines past the header: a header plus a stray
                # blank line is not a parent series (codex r4 LOW).
                _csv_has_rows = sum(
                    1 for _i, _ln in enumerate(_fh)
                    if _i > 0 and _ln.strip()) > 0
        _csv_append = bool(args.restart_from) and _csv_has_rows
        _csv_appended = _csv_append
        _csv = open(_csv_path, "a" if _csv_append else "w")
        if not _csv_append:
            _csv.write(",".join(_csv_cols) + "\n")

    def _log_diag_csv(step, day, d, rate, ice=None):
        if _csv is None:
            return
        row = (
            f"{step},{day:.3f},{d['mean_sst_C']:.4f},{d['mean_sss']:.4f},"
            f"{d['max_abs_u']:.6e},{d['max_abs_v']:.6e},{d['umax_lat']},"
            f"{d['umax_lon']},{d['umax_lev']},{rate:.3f}")
        if ice_config is not None:
            if ice is not None:
                _ia, _ic, _ih, _ = _ice_global_stats(ice, grid, _lm_csv)
                row += f",{_ia:.6e},{_ic:.4f},{_ih:.4f}"
            else:  # defensive: header promised the columns — never misalign
                row += ",nan,nan,nan"
        _csv.write(row + "\n")
        _csv.flush()

    def _close_csv():
        if _csv is not None:
            _csv.close()

    # Seed the series with the initial state — UNLESS we are appending to a
    # parent leg's CSV, which already logged this exact step as its final row.
    # Keyed off whether we actually appended, not off --restart-from: a resume
    # into a FRESH output dir writes a new CSV that would otherwise have no
    # starting row at all (codex r2 LOW).
    if not (_is_io_proc() and _csv_appended):
        _log_diag_csv(start_step, start_day, d0, 0.0, ice=ice_state)

    def _write_run_restart(step_i: int, day_f: float, st, ice_st) -> None:
        """Write the resumable restart (``--restart-save``), process-0 only.

        Overwrites the SAME path atomically each cadence, so a chain launcher
        always finds one valid, latest checkpoint.  Separate from
        ``_save_snapshot`` on purpose: snapshots are a diagnostic contract with
        downstream scorers (ice fields land-masked + category-aggregated),
        restarts carry every prognostic + integrator-carry slot unmangled.
        """
        if not args.restart_save or not _is_io_proc():
            return
        from legoesm.ocean.restart import save_run_restart
        save_run_restart(args.restart_save, st, step=step_i, time_days=day_f,
                         grid_type=app_grid_type, dt_seconds=dt,
                         n_forcing_records=n_rec,
                         config_fingerprint=_restart_cfg_fp,
                         parent=args.restart_from, sha=_restart_src_sha,
                         ice_state=ice_st)
        print(f"[restart] saved step {step_i} (day {day_f:.2f}) -> "
              f"{args.restart_save}", flush=True)

    # ------------------------------------------------------------------
    # Multi-GPU lat-band SPMD step (--n-gpus N): partition the GLOBAL ocean state
    # by latitude band across N local devices.  ``_ocean_step(state, sf, fw)`` is
    # the single per-step entry the host loop calls; default (N=1) is the plain
    # single-device model.step (byte-identical).  The global-in/global-out wrapper
    # scatters/gathers each step, so the host post-step BCs (SSS restore /
    # prognostic ice / geothermal / BBL / nudge / drag) operate on the gathered
    # GLOBAL state UNCHANGED.  n_lat is already SPMD-divisible (build_tripole
    # south-padded it; the latlon branch errored on a non-divisible --latlon-res).
    # ------------------------------------------------------------------
    if app_grid_type == "mpas":
        # Prognostic-TKE carry seed (MPAS Mode-A): the step carry must be
        # pytree-stable, so the None->Field promotion happens HERE, once,
        # before the first step (no-op unless the prognostic TKE closure is
        # active and the carry is unseeded; a restart-loaded tke passes
        # through untouched).
        state = model.seed_tke(state)
        # MPASOceanModel.step has no t_seconds (dm2dc, its only consumer, is
        # arg-gated to tripole/latlon) -- passing it TypeErrors at step 1.
        _ocean_step = (lambda st, sf, fw, t_sec=None:
                       model.step(st, dt, surface_forcing=sf, freshwater=fw))
    else:
        _ocean_step = (lambda st, sf, fw, t_sec=None:
                       model.step(st, dt, surface_forcing=sf, freshwater=fw,
                                  t_seconds=t_sec))
    # --spmd-persistent-state lane state (scaling-M2): OFF by default so the
    # residency helpers below are no-ops and the loop is byte-identical.
    _spmd_persistent = False
    _pers_shard_fn = _pers_gather_fn = None
    if args.n_gpus > 1:
        if app_grid_type not in ("tripole", "latlon"):
            raise SystemExit(
                f"--n-gpus {args.n_gpus} is only wired for the lat-lon C-grid "
                f"(grid=tripole|latlon_bathy); got grid={args.grid!r}. The cube / "
                f"MPAS SPMD paths are separate.")
        if args.visc_schedule:
            raise SystemExit(
                "--n-gpus > 1 with --visc-schedule is unsupported: the schedule "
                "rebuilds the model mid-loop, which would leave the sharded step "
                "holding a stale model. Run the viscosity schedule single-device, "
                "or drop it for the multi-GPU run.")
        import jax as _jax
        if args.distributed:
            # Multi-process: jax.devices() spans EVERY process' GPUs (the global
            # set), and create_latlon_mesh builds the "lat" mesh over them, so the
            # band axis is sharded ACROSS nodes.  Guard against the global count.
            _global = _jax.device_count()
            if args.n_gpus > _global:
                raise SystemExit(
                    f"--n-gpus {args.n_gpus} > global device count {_global} "
                    f"(jax.device_count() across all --distributed processes). "
                    f"Launch ntasks = N processes, one GPU each (e.g. --nodes=2 "
                    f"--ntasks-per-node=2 --gres=gpu:2 for N=4).")
            if args.n_gpus != _global:
                # The lat-band mesh takes the FIRST n_gpus global devices; a
                # mismatch with the launched device count silently idles ranks and
                # (worse) can place two bands on one node while another idles.
                raise SystemExit(
                    f"--distributed expects --n-gpus ({args.n_gpus}) == global "
                    f"device count ({_global}): one process per GPU, every device "
                    f"in the band mesh. Launch exactly {args.n_gpus} single-GPU "
                    f"processes (ntasks={args.n_gpus}).")
        else:
            _local = _jax.local_device_count()
            if args.n_gpus > _local:
                raise SystemExit(
                    f"--n-gpus {args.n_gpus} > local device count {_local}. This "
                    f"single-controller path uses ONE process' local devices (e.g. "
                    f"a 2-GPU node sees 2). For N spanning multiple nodes, add "
                    f"--distributed and launch one process per GPU (mpirun/srun, "
                    f"ntasks=N).")
        n_lat_final = int(grid.n_lat)
        if n_lat_final % args.n_gpus != 0:
            raise SystemExit(
                f"internal: padded n_lat ({n_lat_final}) not divisible by "
                f"n_gpus ({args.n_gpus}) — the south-pad failed.")
        from legoesm.parallel.mesh import create_latlon_mesh
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            make_sharded_ocean_step_global,
        )
        # Prime the build-once vertex-mask cache from the concrete state BEFORE
        # building the sharded step (the wrapper slices the primed global vmask
        # per band; an unprimed cache raises in _build_band_vertex_masks).
        model.prime_step_caches(state)
        _spmd_mesh = create_latlon_mesh(n_devices=args.n_gpus).mesh
        _tf_spmd = getattr(model.config, "tidal_forcing", None)
        if _tf_spmd is not None and _tf_spmd.enabled:
            raise SystemExit(
                "--n-gpus > 1 with tidal_forcing.enabled=True is unsupported: "
                "the lat-band sharded step does not thread t_seconds, so the "
                "equilibrium tide would be SILENTLY inert. Run the tide "
                "single-device, or disable tidal forcing for the SPMD run.")
        if args.spmd_persistent_state:
            # PERSISTENT lane (scaling-M2 increment 1): the state stays
            # lat-band sharded ACROSS steps via the pure-dynamics inner step
            # (the wrapper docstring's own guidance); shard_state_latlon /
            # gather_state_latlon run only at the residency boundaries the
            # helpers below manage (initial shard, snapshot/abort/final
            # gathers, and the counted per-step gathers forced by host-global
            # consumers).  t_sec is always None here (tide fail-fasts above).
            from legoesm.ocean.dynamics.sharded_ocean_step import (
                gather_state_latlon,
                make_sharded_ocean_step,
                shard_state_latlon,
            )
            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
            _ocean_step = (lambda st, sf, fw, t_sec=None:
                           _spmd_inner(st, dt, surface_forcing=sf,
                                       freshwater=fw))

            def _pers_shard_fn(st, _mesh=_spmd_mesh):
                return shard_state_latlon(st, _mesh)

            def _pers_gather_fn(st, _mesh=_spmd_mesh):
                return gather_state_latlon(st, _mesh)

            _spmd_persistent = True
            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
                  f"rows/band); PERSISTENT sharded state "
                  f"(--spmd-persistent-state): full-state gathers only at "
                  f"snapshot/abort/final + counted per-step forcings.")
        else:
            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
            # t_sec is always None here (tide-enabled fail-fasts above).
            _ocean_step = (lambda st, sf, fw, t_sec=None:
                           _spmd_step(st, dt, surface_forcing=sf,
                                      freshwater=fw))
            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
                  f"rows/band); global-in/global-out wrapper (host BCs on "
                  f"gathered state).")

    # ------------------------------------------------------------------
    # --spmd-persistent-state residency helpers (scaling-M2).  The persistent
    # lane keeps ``state`` in the lat-band SHARDED layout (v/v_mask carried as
    # the n_lat-row ``v_lower``) across steps; these two helpers flip the
    # residency at the classified boundaries and COUNT every full-state
    # transfer so the cost is visible in the run log (never silent).  With the
    # flag OFF both are exact no-ops (byte-identical default path).
    #
    # Host-op classification (scaling-M2 audit):
    # (a) sharded-safe, UNCHANGED on the persistent state: the leaf-wise host
    #     BCs (SSS restore / ice-thermo freeze relax / WOA nudge / spin-up
    #     drag) read+write single cell-centred leaves via np.asarray — an
    #     addressable sharded array assembles to the identical host values,
    #     and the drag's v touch operates on ``v_lower`` exactly (the dropped
    #     pole row is identically 0 and 0*decay == 0); the jnp per-column BCs
    #     (geothermal / ISF) are sharding-transparent under GSPMD; BBL is
    #     value-exact too, but its static lat-neighbour slice updates may make
    #     XLA insert device-side collectives / replicate T,S under eager GSPMD
    #     (correct, device-resident — NOT a host-layout flip, so it is
    #     intentionally outside the gather counters, which track full-state
    #     LAYOUT flips only); the
    #     forcing builders (compute_omip2_surface_forcing / _freshwater_)
    #     np.asarray-read state.T identically (pre-existing per-step host
    #     read, both lanes); _diag is EXACT on the sharded layout (the v top
    #     row it cannot see is identically 0 in the gathered layout too).
    # (b) global reductions: none on the host loop itself (the in-step
    #     reductions run through the SPMD-safe psum paths inside shard_map).
    # (c) host-global consumers needing the FULL (n_lat+1)-v global layout:
    #     the momentum-term debug dump (runs model internals on the host
    #     state) and the real I/O boundaries (snapshot / blowup abort /
    #     final diags+digest) — these gather via _ensure_global_state below
    #     (counted; snapshot-cadence ones re-shard lazily at the next step).
    #     Prognostic sea ice + relative winds are NOT in this class any more
    #     (scaling-M2 leftover): _surface_currents* read 2-D surface u/v
    #     slices DEVICE-SIDE and reconstruct the one dropped staggered top
    #     row as the wall zero (_surface_uv_faces -> append_vface_wall_row,
    #     exact under the v-carrier contract), so they operate on the
    #     sharded state directly — class (a), no forced full-state gather.
    # The residency STATE MACHINE itself is the module-level, unit-tested
    # ``_PersistentStateResidency`` (flag/counter interleavings gated in
    # tests/unit/test_run_omip_core2_spmd_persistent_cli.py); main() only
    # binds it to this run's shard/gather layout flips.
    # ------------------------------------------------------------------
    _pers_res = _PersistentStateResidency(
        _spmd_persistent, _pers_shard_fn, _pers_gather_fn)
    _ensure_sharded_state = _pers_res.ensure_sharded
    _ensure_global_state = _pers_res.ensure_global
    # Snapshot the applicator's monotonic host-pull ledger so the done line
    # can report THIS run's forcing-builder surface-slice pulls (the builders
    # record each 2-D surface-T pull; codex batch4 HIGH — those transfers
    # must appear next to the full-state gather count, never implied zero).
    from legoesm.ocean.coupler.omip2_applicator import host_pull_ledger
    _ledger0 = host_pull_ledger()

    t_wall = time.time()

    # ------------------------------------------------------------------
    # Issue #354: optional lax.scan block-stepping (tripole; no nudge/drag).
    # The CORE-II forcing is sampled on-device (no per-step host roundtrip),
    # so XLA fuses each block of ``--scan-block`` steps.  Default
    # (--scan-block 0) keeps the bit-identical Python loop below.
    # Diagnostics / snapshots / non-finite abort run at BLOCK BOUNDARIES.
    # ------------------------------------------------------------------
    _tti = getattr(getattr(model, "config", None),
                   "tracer_time_integrator", "euler")
    if args.n_gpus > 1 and int(args.scan_block) > 0:
        raise SystemExit(
            "--n-gpus > 1 and --scan-block are mutually exclusive: the lax.scan "
            "block path fuses single-device on-device steps (it does not use the "
            "lat-band sharded step). Pick one — multi-GPU SPMD (the host Python "
            "loop, --scan-block 0) OR single-device scan fusion.")
    # Equilibrium tide disqualifies the scan-block path: its body steps via
    # model._step_impl(...) with no t_seconds (bypassing step()'s eager
    # enabled-but-no-time guard), so an enabled tide would be SILENTLY inert
    # inside the scan.  Fall back to the host loop, which threads t.
    _tf_scan = getattr(getattr(model, "config", None), "tidal_forcing", None)
    _tide_enabled = _tf_scan is not None and _tf_scan.enabled
    use_scan = (int(args.scan_block) > 0 and app_grid_type == "tripole"
                and nudge_tau_s == 0.0 and drag_tau_s == 0.0
                and not args.sss_restore
                and not _tide_enabled
                and _tti != "ab2")
    if args.kprofile_snapshots and use_scan:
        # The scan-block lane does not thread the per-snapshot K dump, so
        # honouring --kprofile-snapshots there would silently write no
        # diffusivities.  Refuse rather than mislead (the fidelity arms run
        # with --sss-restore, which already disables the scan lane).
        raise SystemExit(
            "--kprofile-snapshots is not wired into the --scan-block lane; "
            "drop --scan-block (the standard per-step loop dumps the "
            "diffusivities) or drop --kprofile-snapshots.")
    if int(args.scan_block) > 0 and not use_scan:
        why = ("AB2 tracer time integrator (None->Field carry breaks "
               "lax.scan)" if _tti == "ab2"
               else "tidal_forcing enabled (the scan body does not thread the "
                    "model time the tide needs)" if _tide_enabled
               else "grid!=tripole or WOA-nudging / spin-up-drag / SSS-restoring "
                    "enabled (those need per-step host updates)")
        print(f"[scan] --scan-block ignored: {why}.", flush=True)
    # --gateway-transports lane guards.  MUST precede the `use_scan` branch,
    # which RETURNS from main(): a guard after it never executes and the flag
    # is silently ignored (codex H3).  The predicate lives in the ocean package
    # so it is directly unit-testable; this call site is what makes it bite.
    if getattr(args, "gateway_transports", False):
        from legoesm.ocean.diagnostics_sections import validate_gateway_lanes
        try:
            validate_gateway_lanes(
                use_scan=bool(use_scan),
                spmd_persistent=bool(
                    getattr(args, "spmd_persistent_state", False)),
                app_grid_type=app_grid_type)
        except ValueError as _gw_err:
            raise SystemExit(str(_gw_err))

    if use_scan:
        # NEMO ln_crt_dwn relative winds are host-loop only: the on-device scan
        # body (compute_omip2_surface_forcing_jax) has no current-feedback wiring,
        # so --relative-winds under --scan-block would SILENTLY drop the feedback.
        # Refuse rather than mislead (dispatch hardening).
        if _wind_vfac != 0.0:
            raise SystemExit(
                "[scan] --relative-winds/--wind-vfac is not applied on the "
                "--scan-block fast path (the on-device forcing kernel has no "
                "current-feedback wiring); use the host Python loop "
                "(omit --scan-block).")
        # The scan path applies NONE of the host-loop surface forcing extensions:
        # P - E (precip is not in the on-device stack), Dai-Trenberth runoff, SSS
        # restoring, OR the --ice-albedo SW reduction (siconc not on device).
        # Refuse rather than silently emit a quietly-fresh / no-albedo result if
        # ANY of those is requested.  P - E is ON by default, so the scan path is
        # reachable only with --no-emp AND no --runoff/--sss-restore/--ice-albedo
        # (a pure momentum/heat tripole perf run, issue #354).
        if (args.emp_freshwater or args.runoff or args.sss_restore
                or args.ice_albedo or args.ice_thermo or args.geothermal
                or args.dm2dc or args.isf):
            raise SystemExit(
                "[scan] --scan-block applies no surface salinity/albedo/ice forcing "
                "or geothermal BC or diurnal SW (P - E / runoff / SSS restoring / "
                "ice-albedo / ice-thermo / geothermal / dm2dc are host-loop only), "
                "so it cannot run a faithful integration.  Use the host Python "
                "loop (omit --scan-block), or drop "
                "--runoff/--sss-restore/--ice-albedo/--ice-thermo/--geothermal/"
                "--dm2dc and pass --no-emp for the momentum/heat-only scan path.")
        if args.restart_save or args.restart_from:
            # UNCONDITIONAL refusal (codex r2 HIGH).  The scan body calls
            # model._step_impl DIRECTLY and never calls seed_scan_carry, so it
            # PROMOTES optional slots None -> Field on the first block step
            # (prognostic TKE documents exactly that).  My earlier guard tested
            # the CURRENT state's populated slots, which are all None at setup
            # for a fresh TKE/EKE/leapfrog config — so it passed and the lane
            # then created a carry the restart neither saved nor advanced.
            # There is no cheap value-based test that closes that hole, and the
            # lane's own eligibility gap is pre-existing and out of scope here,
            # so restarts on this lane are refused outright.
            raise SystemExit(
                "--restart-save/--restart-from is not supported with "
                "--scan-block: the scan body steps model._step_impl directly "
                "and never seeds or advances the scan carry, so it can promote "
                "an integrator slot mid-block that the restart neither records "
                "nor continues.  Run the restartable leg on the host Python "
                "loop (omit --scan-block).")
        from legoesm.ocean.coupler.omip2_applicator import (
            build_core2_forcing_device_stack, build_omip2_scan_block_fn,
        )
        f_stack, nn_i, nn_j, gshape = build_core2_forcing_device_stack(
            forcing, grid, "tripole")
        # Scan blocks trace _step_impl directly — prime build-once
        # caches from the concrete state first (vertex-mask constant).
        model.prime_step_caches(state)
        block_fn = build_omip2_scan_block_fn(model, dt, gshape, ramp_s=ramp_s)
        bsz = int(args.scan_block)
        print(f"[run] lax.scan block-stepping: block<={bsz} steps, split at "
              f"diag/snapshot/year boundaries so output cadence matches the "
              f"Python loop (CORE-II forcing fused on-device)", flush=True)

        def _block_steps(step):
            # Cap the block so it ENDS on the next diagnostic / snapshot /
            # year boundary -> the modulo-gated I/O below fires at exactly
            # the same cadence as the Python loop (codex #354 finding 2).
            nb = min(bsz, n_steps - step)
            # No restart_every here: this lane REFUSES restarts (above).
            for period in (diag_every, snap_every, steps_per_year):
                if period and period > 0:
                    nb = min(nb, period - (step % period))
            return max(1, nb)

        # Resume at the restart's absolute step (0 for a fresh run): the block
        # forcing indices below are pure functions of `step`, so continuing the
        # counter reproduces the forcing exactly.
        step = start_step
        while step < n_steps:
            nb = _block_steps(step)
            idx_block = jnp.asarray(
                [_idx_t(step + 1 + k, dt, n_rec) for k in range(nb)],
                dtype=jnp.int32)
            state = block_fn(state, f_stack, nn_i, nn_j, idx_block,
                             jnp.int32(step + 1))
            step += nb
            day = step * dt / _SEC_PER_DAY
            if step % diag_every == 0 or step == n_steps:
                state = jax.block_until_ready(state)
                d = _diag(state, lat2d, lon2d)
                # Throughput of THIS leg: a resumed run has done (step-start_step)
                # steps in (now - t_wall), not `step` of them.
                rate = (step - start_step) / (time.time() - t_wall)
                print(f"[diag] step {step} (day {day:.0f}): {d} | "
                      f"{rate:.2f} steps/s", flush=True)
                _log_diag_csv(step, day, d, rate)
                if not d["finite"]:
                    print("[ABORT] non-finite state", flush=True)
                    _save_snapshot(out_dir, f"blowup_step{step}",
                                   state, lat2d, lon2d, grid=grid,
                                   io_proc=_is_io_proc())
                    _close_csv()
                    return 1
            if snap_every > 0 and step % snap_every == 0 and step != n_steps:
                _save_snapshot(out_dir, f"day{int(round(day)):04d}",
                               state, lat2d, lon2d, z_coord=z_coord,
                               io_proc=_is_io_proc(), grid=grid,
                               step=step, day=day)
                print(f"[snapshot] day {day:.0f} saved", flush=True)
            if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
                yr = step // steps_per_year
                _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d,
                               z_coord=z_coord, grid=grid, io_proc=_is_io_proc())
                print(f"[snapshot] year {yr} saved", flush=True)
        state = jax.block_until_ready(state)
        _io = _is_io_proc()
        _save_snapshot(out_dir, "final", state, lat2d, lon2d, z_coord=z_coord,
                       io_proc=_io, grid=grid, step=step, day=day)
        _amoc26n_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _acc_drake_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _save_bsf_amoc_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _mht_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _record_final_state_digest(manifest_path, state)
        _close_csv()
        rate = (n_steps - start_step) / (time.time() - t_wall)
        print(f"[done] {n_steps - start_step} steps this leg "
              f"(absolute step {n_steps}) @ {rate:.2f} steps/s (scan); "
              f"final: {_diag(state, lat2d, lon2d)}")
        if args.smoke:
            yr_est = steps_per_year / rate / 3600.0
            print(f"[smoke] projected wall-time: {yr_est:.2f} h/yr  "
                  f"({args.years:.0f}yr -> {yr_est*args.years:.1f} h)")
        return 0

    # KPP freshwater-buoyancy contract guard: on the direct-forced non-cube
    # path the host loop routes the PHYSICAL net freshwater (P-E+R+ice_fw)
    # through sf.freshwater as a BUOYANCY-ONLY signal for the vertical-mixing
    # surface-buoyancy diagnosis, while the freshwater MASS enters exactly once
    # via model.step(freshwater=fw).  Fail fast if the config ever drifts onto
    # a scheme that ALSO applies sf.freshwater as a virtual salt.
    if app_grid_type != "cubed_sphere":
        _sfc_cfg = getattr(getattr(model.config, "physics", None),
                           "surface_forcing", None)
        _validate_kpp_freshwater_contract(
            app_grid_type, getattr(_sfc_cfg, "scheme", "none"))

    # Equilibrium-tide wiring: when ocean.tidal_forcing.enabled the model's
    # step() REQUIRES the elapsed model time (it fail-fasts otherwise — the
    # tide would be silently inert).  Thread t = (step-1)*dt as a device
    # scalar (compiled once, no per-step retrace); tide-off passes None and
    # the call is byte-identical to before.
    _tf_cfg = getattr(model.config, "tidal_forcing", None)
    _tide_on = _tf_cfg is not None and _tf_cfg.enabled

    # --prescribed-flow is IN-MODEL (LatLonCGridOceanConfig.prescribed_flow,
    # threaded at build): the pin happens INSIDE _step_impl where the tracer
    # mass fluxes are built, so no host-loop reset / reference capture here.
    if args.prescribed_flow is not None:
        print(f"[prescribed-flow] mode={args.prescribed_flow} (in-model): "
              f"circulation pinned inside every step -- vertical physics "
              f"isolated from circulation feedback.", flush=True)

    # --spmd-persistent-state: static classification of the per-step
    # host-global consumers (class (c) above).  Forced per-step gathers are
    # KEPT + COUNTED + logged up front — never a silent degradation.
    # Prognostic ice / relative winds no longer force one (scaling-M2
    # leftover): their surface-current reads are device-side on the sharded
    # layout (_surface_uv_faces), bit-identical to the gathered read — the
    # list stays as the wiring point for any future host-global consumer.
    _pers_forced = []
    if _spmd_persistent:
        print(f"[spmd-persistent] full_state_gathers_per_step="
              f"{1 if _pers_forced else 0} (full-STATE layout flips ONLY — "
              f"NOT the total transfer cost)"
              + (f" — forced by: {'; '.join(_pers_forced)}" if _pers_forced
                 else "; otherwise full-state gathers only at snapshot/abort/"
                      "final boundaries"), flush=True)
        if ice_config is not None or _wind_vfac != 0.0:
            print("[spmd-persistent] prognostic-ice / relative-winds surface "
                  "currents read the SHARDED state device-side (2-D surface "
                  "u/v slices; the dropped staggered top row is reconstructed "
                  "as the wall zero — no forced per-step full-state gather).",
                  flush=True)
        # Honest-cost companion (codex batch4 HIGH): enumerate the per-step
        # LEAF host transfers that REMAIN in the persistent lane, so a
        # "0 full-state gathers" line is never read as "0 transfer cost".
        # Measured totals are printed in the [spmd-persistent] done lines.
        _leaf_srcs = ["surface-T 2-D slice per surface-forcing build"]
        if args.emp_freshwater or runoff_monthly is not None:
            _leaf_srcs.append("surface-T 2-D slice per freshwater build")
        if sss_restore_cfg is not None:
            _leaf_srcs.append("SSS-restore S-surface pull + write-back")
        if args.ice_thermo:
            _leaf_srcs.append(
                "ice-thermo T-surface pull + write-back"
                + (" (+ per-cell liquidus S-surface pull, --freeze-scheme)"
                   if args.freeze_scheme != "constant" else ""))
        if nudge_tau_s > 0:
            _leaf_srcs.append(
                "WOA-nudge FULL-3D T,S gather + re-upload (while active)")
        if drag_tau_s > 0:
            _leaf_srcs.append(
                "spin-up-drag FULL-3D u,v gather + re-upload (while active)")
        print(f"[spmd-persistent] per-step LEAF host transfers remain "
              f"(counted separately, totals at [done]): "
              f"{'; '.join(_leaf_srcs)}; plus T,S surface + FULL-3D u,v "
              f"reads at diag cadence.", flush=True)
        if args.diag_momentum_step >= 0:
            print(f"[spmd-persistent] --diag-momentum-step "
                  f"{args.diag_momentum_step}: additionally gathers each of "
                  f"the first {args.diag_momentum_step} steps (debug window).",
                  flush=True)
    _pers_needs_prestep_global = bool(_pers_forced)

    # --- Arctic gateway transport accumulator (pure diagnostic) ----------
    _gw_acc = None
    _gw_gates = None
    _gw_geom = None
    _gw_csv = None
    if getattr(args, "gateway_transports", False):
        if getattr(state, "v", None) is None:
            print("[gateway] needs C-grid v faces; DISABLED for this run.")
        else:
            from legoesm.ocean.diagnostics_sections import (
                promote_gateway_geometry, setup_gateway_accumulator,
            )
            _gw_lat = (jnp.degrees(jnp.asarray(grid.lat_T))
                       if hasattr(grid, "lat_T") else jnp.asarray(lat2d))
            _gw_lon = (jnp.degrees(jnp.asarray(grid.lon_T))
                       if hasattr(grid, "lon_T") else jnp.asarray(lon2d))
            # Setup is inside the same non-fatal boundary as the per-step call
            # (codex round-4 RED): a diagnostic must never abort the run.
            try:
                _gw_acc, _gw_gates, _gw_faces = setup_gateway_accumulator(
                    _gw_lat, _gw_lon, state.land_mask.data)
                # Promote ONCE, and prefer the model's own geometry: the ocean
                # model already ran ensure_geometry(grid, metric_convention=
                # config.metric_convention) in its constructor, so model.grid
                # has the RIGHT face metrics.  Promoting inside the per-step
                # call rebuilt every metric array each step and silently
                # defaulted the convention to "exact" (codex r3 finding 1).
                _gw_geom = promote_gateway_geometry(
                    getattr(model, "grid", grid),
                    metric_convention=getattr(model.config,
                                              "metric_convention", "exact"))
            except Exception as _gw_e:
                print(f"[gateway] setup FAILED, diagnostic disabled: "
                      f"{type(_gw_e).__name__}: {_gw_e}")
                _gw_acc = _gw_gates = _gw_geom = None
            if _gw_acc is not None:
                print(f"[gateway] accumulating through {len(_gw_gates.names)} "
                      f"gateways ({int(jnp.sum(_gw_faces.u_sel))} u-faces, "
                      f"{int(jnp.sum(_gw_faces.v_sel))} v-faces); "
                      "+ = INTO Arctic")
                # Per-cadence CUMULATIVE dump.  The run-end transports.txt
                # block is a WHOLE-RUN mean, which cannot separate the
                # cold-start adjustment from the settled window; differencing
                # two rows of this file gives the mean over ANY window:
                #   mean(n_a, n_b] = (cumsum_b - cumsum_a) / (n_b - n_a).
                _gw_csv = _gateway_cumulative_open(
                    out_dir, _gw_acc.names, io_proc=_is_io_proc())
                if _gw_csv is not None:
                    if snap_every > 0:
                        print(f"[gateway] cumulative dumps -> "
                              f"{GATEWAY_CUMULATIVE_CSV} every {snap_every} "
                              f"steps ({args.snapshot_every_days:g} d) plus "
                              f"one at run end; difference two rows for a "
                              f"windowed mean", flush=True)
                    else:
                        # Honest, not silent: with no snapshot cadence the file
                        # gets ONE row (run end) and carries no more
                        # information than transports.txt already does.
                        print(f"[gateway] --snapshot-every-days is 0, so "
                              f"{GATEWAY_CUMULATIVE_CSV} will hold only the "
                              f"run-end row; pass --snapshot-every-days N to "
                              f"make windowed means recoverable", flush=True)

    # RESUME AT THE RESTART'S ABSOLUTE STEP. Starting at 1 replays the whole
    # run against an already-advanced state: a step-4 checkpoint with an
    # 8-step target applied the forcing for steps 1..8 to that state, advanced
    # twelve physical steps, and then labelled the result step 8. The scan
    # lane already continued the counter; this one did not.
    for step in range(start_step + 1, n_steps + 1):
        it = _idx_t(step, dt, n_rec)
        _t_sec = jnp.asarray((step - 1) * dt) if _tide_on else None
        ramp = min(1.0, (step * dt) / ramp_s) if ramp_s > 0 else 1.0
        # Piecewise viscosity schedule (--visc-schedule): at each segment
        # boundary rebuild config+model ONCE (one JIT recompile per segment)
        # with the next (A_h, C_smag_lap).  The high cold-start viscosity is
        # only needed during the WOA adjustment; stepping down from an
        # adjusted state closes the 5-10x gap to NEMO's eddy_viscosity file
        # (the user-flagged 'fuzzier than NEMO') without the day-30 NaN a
        # one-jump drop causes.  Exact + auditable (segments logged).
        if visc_schedule and visc_seg_idx < len(visc_schedule):
            _day0, _ah, _cs = visc_schedule[visc_seg_idx]
            if (step - 1) * dt >= _day0 * 86400.0:
                if (_ah, _cs) != (float(model.config.lateral_viscosity.A_h),
                                  float(model.config.lateral_viscosity.C_smag_lap)):
                    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid \
                        import LatLonCGridOceanModel
                    model = LatLonCGridOceanModel(
                        grid, z_coord,
                        model.config._replace(lateral_viscosity=model.config.lateral_viscosity._replace(A_h=_ah, C_smag_lap=_cs)),
                        # keep the zdfiwm maps through the mid-run rebuild
                        iwm_forcing=getattr(model, "_iwm_forcing", None))
                print(f"[visc-schedule] day {(step-1)*dt/86400.0:.1f}: "
                      f"A_h={_ah:g} C_smag_lap={_cs:g} "
                      f"(segment {visc_seg_idx + 1}/{len(visc_schedule)})",
                      flush=True)
                visc_seg_idx += 1
        # --spmd-persistent-state: consumers below that need the FULL global
        # layout every step force a per-step gather — kept + counted (see the
        # [spmd-persistent] setup log line), never silent.  Since the
        # scaling-M2 leftover this is ONLY the momentum-term debug dump
        # (model internals on the host state) + any future _pers_forced
        # entry; prognostic ice / relative winds read the sharded state
        # device-side (_surface_uv_faces reconstructs the staggered top row).
        if _spmd_persistent and (_pers_needs_prestep_global
                                 or (args.diag_momentum_step >= 0
                                     and step <= args.diag_momentum_step)):
            state = _ensure_global_state(state)
        # Build CORE-II surface forcing and integrate it INSIDE model.step (the
        # dynamics-core external-tau block) -- energetically consistent, unlike
        # the operator-split applicator (which pumped the runaway). Optional
        # cold-start ramp scales the forcing fields.
        # Sea-ice concentration for THIS step: the calendar-month slice when a
        # seasonal (12-month) climatology was loaded, else the single annual map.
        # Drives both the SW albedo (--ice-albedo only) and the SSS-restoring ice
        # gate (below).  ``_sic`` may be loaded for restoring alone (when only
        # --sss-restore is set), so the albedo is gated on --ice-albedo explicitly
        # to keep an SSS-only run's heat budget unchanged (codex HIGH).
        _sic = _siconc_at_step(siconc_clim, step, dt, siconc_monthly)
        # Open-water surface-flux attenuation under sea ice:
        #  * --ice-thermo -> the PRESCRIBED NEMO siconc surrogate (legacy): the
        #    EXISTING under-ice mechanism inside compute_omip2_surface_forcing
        #    (_ice_surface_heat: cut under-ice SW to tau_ice_sw, suppress
        #    open-ocean turbulent+LW by (1-conc)).
        #  * --prognostic-sea-ice -> NO attenuation here: sf below is the
        #    UNMASKED full-cell open-ocean bulk forcing (ice_albedo=None), and
        #    the ENTIRE open-water/ice partition (heat, SW, STRESS, and — via
        #    fw — EVAPORATION) is applied ONCE, after step_sea_ice, by
        #    blend_ice_ocean_forcing with the PRE-step concentration (codex:
        #    the old wiring attenuated only heat/SW and left full open-water
        #    stress + evap acting under ice).
        _ice_alb = _sic if (args.ice_albedo or args.ice_thermo) else None
        _under_ice = args.ice_thermo
        # NEMO ln_dm2dc window for THIS step: NEMO zlo = (nsec_day - dt/2)/rday,
        # zup = zlo + dt/rday (nn_fsbc-equivalent = 1: forcing rebuilt every
        # step here).  Perpetual 365-day calendar, day-of-year 1-based.
        _dm2dc_win = None
        if args.dm2dc:
            # NEMO time axis is defined at step MIDPOINTS (day.F90 seeds
            # nsec_day at dt/2), so build the window from the mid-step
            # time: step k integrates exactly [(k-1)dt, k dt] (codex r3
            # #1 — the end-of-step time was half a step late).
            _t_mid = (step - 0.5) * dt
            _sec_of_day = _t_mid % _SEC_PER_DAY
            _t_lo = (_sec_of_day - 0.5 * dt) / _SEC_PER_DAY
            _dm2dc_win = (
                int((_t_mid / _SEC_PER_DAY) % 365.0) + 1,   # day_of_year
                365.0,
                _t_lo,
                _t_lo + dt / _SEC_PER_DAY,
            )
        # NEMO ln_crt_dwn relative-wind current feedback (rn_vfac): rotate the
        # beginning-of-step ocean surface current to GEOGRAPHIC (the frame the
        # CORE-II u10/v10 arrive in) and pass it to the bulk.  Computed ONCE and
        # shared with the freshwater (evap) forcing below so the latent HEAT
        # (q_net) and the evaporative MASS (P - E) stay the SAME physical flux
        # (E = -lhflx / L_vap).  _wind_vfac == 0.0 (default) => None, no state
        # pull, byte-identical to the absolute-wind path.
        _u_oce = _v_oce = None
        if _wind_vfac != 0.0:
            _u_oce, _v_oce = _surface_currents_geographic(
                state, grid, app_grid_type)
        sf = compute_omip2_surface_forcing(
            state, forcing=forcing, idx_t=it,
            grid=grid, grid_type=app_grid_type,
            ice_albedo=_ice_alb,
            under_ice=_under_ice, tau_ice_sw=args.ice_thermo_sw_trans,
            dm2dc_window=_dm2dc_win,
            u_oce=_u_oce, v_oce=_v_oce,
            wind_current_feedback_vfac=_wind_vfac,
        )
        if ramp < 1.0:
            sf = sf._replace(tau_x=sf.tau_x * ramp, tau_y=sf.tau_y * ramp,
                             q_net=sf.q_net * ramp, sw_down=sf.sw_down * ramp)
        # Surface chlorophyll for THIS step (calendar-month slice); attaching it
        # switches the PE step to NEMO's RGB penetration.  NOT ramped — Chl is a
        # fixed optical climatology, independent of the dynamical spin-up ramp.
        if chl_clim is not None:
            sf = sf._replace(chl=chl_clim[_runoff_month_idx(step, dt)])
        # PRESCRIBED-ice runs (--ice-albedo/--ice-thermo/--sss-restore ice
        # gate, NO --prognostic-sea-ice): thread the SAME climatological
        # concentration to the vertical-mixing closure so the TKE under-ice
        # attenuation (TKEConfig.eice, NEMO nn_eice) is not silently skipped.
        # The prognostic branch overwrites this below with its own
        # partition-time-level concentration after blend_ice_ocean_forcing.
        # Inert unless eice != 0 (consumption is config-gated in k_profiles).
        if _sic is not None and ice_config is None:
            sf = sf._replace(ice_concentration=_sic)
        # DEBUG: per-term momentum-tendency breakdown at the onset steps (pin the
        # term driving the lat-lon 75-level cold-start blowup). sf is finalised
        # for momentum here EXCEPT under --prognostic-sea-ice, where the
        # open-water/ice stress partition (blend_ice_ocean_forcing) still
        # applies below — this diag then shows the full-cell open-ocean tau.
        if args.diag_momentum_step >= 0 and step <= args.diag_momentum_step:
            _dump_momentum_terms(model, state, sf, dt, lat2d, lon2d,
                                 tag=f" step{step}")
        # Surface freshwater (atmospheric P - E + optional Dai-Trenberth runoff)
        # is delivered through the IN-CORE channel
        # ``model.step(..., freshwater=FreshwaterForcing)``: the dynamics core
        # applies the virtual-salt tendency (``config.S_ref``) AND the eta
        # free-surface source inside the barotropic solve, on-device + AD-safe.
        # P - E is ON by default (--no-emp = ablation).  The CUBE's 'external'
        # physics instead consumes ``surface_forcing.freshwater`` directly, so for
        # the cube the NET flux is folded onto ``sf`` and no freshwater= arg is
        # passed (single application; avoids the double-count codex flagged).
        _R = (runoff_monthly[_runoff_month_idx(step, dt)]
              if runoff_monthly is not None else None)
        _want_fw = args.emp_freshwater or (_R is not None)
        # Prognostic sea ice: step the REAL model on the SAME CORE-II forcing
        # (sampled with the SAME sampler the heat/momentum path uses), then route
        # its brine-salt / melt-freshwater / ocean-heat response into the
        # EXISTING surface_forcing (salt_flux, q_net) + freshwater (ice_fw)
        # channels.  Carry the new ice state.  (Validated host-loop only; the cube
        # path is rejected upstream, so this only runs in the else branch below.)
        ice_resp = None
        if ice_config is not None:
            from legoesm.ocean.coupler import sample_omip2_forcing
            forc_ice = sample_omip2_forcing(forcing, it, grid, app_grid_type)
            if _dm2dc_win is not None:
                # SAME diurnal SW modulation the ocean forcing gets (codex r1
                # #2): the ice tile must not integrate the raw daily-mean SW
                # while the ocean sees the sbcdcy-modulated one.
                from legoesm.ocean.coupler.omip2_applicator import (
                    dm2dc_sw_factor,
                )
                forc_ice = dict(forc_ice)
                forc_ice["sw_down"] = (
                    np.asarray(forc_ice["sw_down"], dtype=np.float64)
                    * dm2dc_sw_factor(grid, _dm2dc_win))
            atm_ice = _build_atm_to_surface_core2(forc_ice, ramp=ramp)
            sst_K = jnp.asarray(state.T.data)[..., 0] + _ice_T_freeze
            ocn_u, ocn_v = _surface_currents(state, grid, app_grid_type)
            # Partition time level (codex r4 #1 + r5 #1): the ice model
            # exposes the AGGREGATE concentration its THERMODYNAMICS
            # integrated the atmospheric fluxes over
            # (resp.ice_concentration_thermo = post-transport, pre-thermo),
            # so ice + open water together receive exactly the incident flux
            # (A + (1-A) = 1) even under transport='advect'.  The pre-call
            # concentration is the fallback for response paths that do not
            # populate the field (slab ice).
            _ice_conc_pre = ice_state.concentration.data
            if _ice_conc_pre.ndim > np.asarray(state.land_mask.data).ndim:
                _ice_conc_pre = jnp.sum(_ice_conc_pre, axis=-1)  # multi-cat
            ice_state, ice_resp = step_sea_ice(
                ice_state, atm_ice, sst_K, ocn_u, ocn_v,
                ice_config, U_min=0.0, dt=dt, grid=grid)
            if args.ew_cyclic_overlap and app_grid_type == "tripole":
                # Re-slave the duplicated ORCA halo columns after transport
                # (codex: the C-grid ice advection wraps with period nx, off
                # by one on the 2-point-overlap grid — without this the
                # duplicated seam columns drift apart; the ocean does the same
                # on its own state in _apply_ew_cyclic_overlap).  The RESPONSE
                # is slaved too (codex r2 #4): its halo-column stresses/fluxes
                # feed the blended forcing at the physical seam next to the
                # halo, so unslaved duplicates would diverge there as well.
                ice_state = _ice_apply_ew_overlap(ice_state)
                ice_resp = _ice_apply_ew_overlap(ice_resp)
            if getattr(ice_resp, "ice_concentration_thermo", None) is not None:
                _ice_conc_pre = ice_resp.ice_concentration_thermo
            # The TileResponse is passed to the ocean UNSCALED even under the
            # cold-start ramp (codex r2 #1): step_sea_ice has already committed
            # the FULL exchange to ice_state (ice grew/melted against the full
            # basal heat/brine), so scaling only the ocean-side response would
            # break ice-ocean conservation — and the SW-driven parts already
            # carry the ramp through atm_ice (scaling again would be ramp^2).
            # The ramp is an atmospheric-forcing spin-up crutch; the ice-ocean
            # exchange is an internal coupled flux and must balance exactly.
        if app_grid_type == "cubed_sphere":
            # CUBE is PARKED (cold-start blowup). Its 'external' physics applies
            # surface_forcing.freshwater ONCE as a virtual salt with the LOCAL
            # S_top (not config.S_ref) and NO eta free-surface source -- a
            # lighter treatment than the latlon/MPAS freshwater= path. Acceptable
            # for the parked grid; revisit if the cube is unparked for a
            # salinity-faithful OMIP run.
            if _want_fw:
                from legoesm.ocean.freshwater import net_freshwater_flux
                fw = compute_omip2_freshwater_forcing(
                    state, forcing=forcing, idx_t=it, grid=grid,
                    grid_type=app_grid_type, runoff_R=_R,
                    emp=args.emp_freshwater, ramp=ramp,
                    u_oce=_u_oce, v_oce=_v_oce,
                    wind_current_feedback_vfac=_wind_vfac)
                sf = sf._replace(freshwater=net_freshwater_flux(fw))
            state = model.step(state, dt, surface_forcing=sf,
                               t_seconds=_t_sec)
        else:
            fw = None
            # Build the freshwater struct if EITHER the atmospheric P-E/runoff is
            # wanted OR the prognostic ice needs an ``ice_fw`` carrier (zero P-E-R
            # in that case, so only the ice melt/freeze freshwater is delivered).
            if _want_fw or ice_resp is not None:
                fw = compute_omip2_freshwater_forcing(
                    state, forcing=forcing, idx_t=it, grid=grid,
                    grid_type=app_grid_type, runoff_R=_R,
                    emp=args.emp_freshwater, ramp=ramp,
                    u_oce=_u_oce, v_oce=_v_oce,
                    wind_current_feedback_vfac=_wind_vfac)
            if ice_resp is not None:
                # ONE shared, mask-aware partition (coupler.ocean_forcing):
                # open-water stress/evap/heat/SW x f_open=(1-A) at the SINGLE
                # documented PRE-step concentration (see _ice_conc_pre above);
                # ice basal heat, brine salt, melt/freeze freshwater, and ice
                # stress added exactly once.  sf was built UNMASKED
                # (ice_albedo=None -> raw SW), so the SW split happens here and
                # only here (raw_core2 mode).  sw_transmittance_ice=0.0: the
                # under-ice transmitted SW is delivered by the ICE MODEL
                # (SeaIceConfig.sw_transmittance_const debits the ice EB and
                # routes tau*SW to the ocean via resp.ocean_heat_extraction),
                # so adding the old ocean-side A*tau*swd surrogate here would
                # now DOUBLE-COUNT it (codex L1 — budget closed).
                from legoesm.coupler.ocean_forcing import blend_ice_ocean_forcing
                from legoesm.ice import uses_new_physics
                fw, sf = blend_ice_ocean_forcing(
                    open_sf=sf, open_fw=fw, ice_resp=ice_resp,
                    ice_concentration=_ice_conc_pre,
                    ocean_mask=state.land_mask.data,
                    sw_partition="raw_core2",
                    alpha_ocean=float(_ice_const.alpha_ocean_broadband),
                    sw_transmittance_ice=0.0,
                    ice_owns_snow_reservoir=uses_new_physics(ice_config),
                )
                # Thread the SAME partition-time-level ice concentration to
                # the vertical-mixing closure: the TKE lc/etau under-ice
                # attenuation (TKEConfig.eice, NEMO nn_eice) reads
                # surface_forcing.ice_concentration.  Attach ALWAYS (inert
                # unless eice != 0 — consumption is config-gated in
                # k_profiles, so eice=0 stays bit-identical).
                sf = sf._replace(ice_concentration=_ice_conc_pre)
            elif fw is not None:
                # KPP freshwater-buoyancy contract (codex): sf.freshwater is
                # consumed ONLY by the vertical-mixing surface-buoyancy
                # diagnosis on this direct-forced path (surface-forcing scheme
                # "none" — guarded at setup), so route the PHYSICAL net
                # freshwater P-E+R through it.  The mass/salinity is applied
                # exactly once via model.step(freshwater=fw); the numerical
                # SSS-restoring flux is EXCLUDED (applied as a post-step state
                # update, never through fw.restoring here).  The ice branch
                # above sets the same channel (incl. ice_fw) inside the blend.
                from legoesm.ocean.freshwater import net_freshwater_flux
                sf = sf._replace(
                    freshwater=net_freshwater_flux(fw._replace(restoring=None)))
            # Print at step 1 AND on a daily stride: a step-1-only budget
            # cannot tell INITIALISATION SHOCK from a persistent bias, and
            # the step-1 numbers (Antarctic ice 44.8e-6 vs NEMO's entire
            # runoff+ice 16.7; Arctic ice MELTING +23.6 in January where NEMO
            # freezes at -17.2) are exactly the kind that need a trajectory
            # before they are believed.
            _fwb_stride = max(1, int(round(86400.0 / float(dt))))
            if fw is not None and (step == 1 or step % _fwb_stride == 0):
                _Ab = np.asarray(grid.areaCell if hasattr(grid, "areaCell")
                                 else grid.area)
                _wb = np.asarray(state.land_mask.data) > 0.5
                _ig = lambda _x: (float((np.asarray(_x) * _Ab * _wb).sum()) / 1.0e9
                                  if _x is not None else 0.0)   # noqa: E731
                _P, _E, _Rn, _Ic = (_ig(fw.precip), _ig(fw.evap),
                                     _ig(fw.runoff), _ig(fw.ice_fw))
                print(f"[fwbudget] step={step} day={step * dt / 86400.0:.2f} "
                      f"{app_grid_type}: P={_P:+.4f} E={_E:+.4f} "
                      f"R={_Rn:+.4f} ice={_Ic:+.4f} net(P-E+R+ice)="
                      f"{_P - _E + _Rn + _Ic:+.4f} Sv (raw pre-normalize, "
                      f"area-wtd over wet)", flush=True)
                # PER-BAND split (2026-08-19). The SO freshwater budget probe
                # localised the missing Antarctic summer fresh layer to the
                # runoff+ice channel (NEMO: 16.7e-6 kg/m2/s of a 30.4 total,
                # while our P-E already matches NEMO's open-ocean E-P), and a
                # GLOBAL Sv total cannot show whether OUR runoff+ice reaches
                # that band. Same fields, area-weighted mean per band, in the
                # probe's units so the two are directly comparable.
                # The structured C-grids call it lat_T (or lat); the MPAS
                # Voronoi mesh calls it latCell.  Reading grid.lat
                # unconditionally raised AttributeError on MPAS and killed the
                # run inside a DIAGNOSTIC (jobs 9466912/9466913) -- a print
                # must never be able to end an integration, so an unrecognised
                # geometry skips the band split instead.
                _latsrc = next(
                    (v for v in (getattr(grid, "lat_T", None),
                                 getattr(grid, "lat", None),
                                 getattr(grid, "latCell", None))
                     if v is not None), None)
                _latb = (np.degrees(np.asarray(_latsrc))
                         if _latsrc is not None else None)
                if _latb is not None and _latb.shape == _wb.shape:
                    # TRUE cell area, not cos(lat): on the eORCA1 tripole the
                    # two differ by 0.00-1.72x per cell south of 45S, which
                    # inflated the first Antarctic ice number by ~45%. _Ab is
                    # the same area the global Sv total above already uses.
                    _wgt = _Ab * _wb
                    print(f"[fwbudget-bands] day={step * dt / 86400.0:.2f} "
                          "1e-6 kg/m2/s, + = into ocean (evap +up):",
                          flush=True)
                    for _bn, (_lo, _hi) in (("antarctic_S_of_45S", (-90, -45)),
                                            ("SH_midlat_45S_23S", (-45, -23)),
                                            ("arctic_N_of_45N", (45, 90))):
                        _m = (_latb >= _lo) & (_latb < _hi) & (_wgt > 0)
                        if not _m.any():
                            continue
                        def _bm(_x):
                            if _x is None:
                                return 0.0
                            _a = np.asarray(_x)
                            return float((_a[_m] * _wgt[_m]).sum()
                                         / _wgt[_m].sum())
                        _p, _e = _bm(fw.precip), _bm(fw.evap)
                        _r, _i = _bm(fw.runoff), _bm(fw.ice_fw)
                        print(f"[fwbudget-bands]   {_bn:22s} P={1e6*_p:8.2f} "
                              f"E={1e6*_e:8.2f} R={1e6*_r:8.2f} "
                              f"ice={1e6*_i:8.2f}  P-E+R+ice="
                              f"{1e6*(_p-_e+_r+_i):8.2f}", flush=True)
                else:
                    print(f"[fwbudget-bands] SKIPPED: lat {_latb.shape} does "
                          f"not align with the mask {_wb.shape}", flush=True)
            # _ocean_step = single-device model.step (default), the lat-band
            # SPMD global-in/global-out step (--n-gpus > 1), or the PERSISTENT
            # sharded inner step (--spmd-persistent-state); all apply the
            # in-core wind-stress / heat / freshwater forcing.  Default lanes
            # return a GLOBAL state, so the host post-step BCs below are
            # unchanged; the persistent lane keeps the state SHARDED — the
            # leaf-wise/jnp post-step BCs below operate on it identically (see
            # the residency-helper classification).  t_seconds threads the
            # equilibrium-tide model time (None when tide off; the SPMD path
            # fail-fasts at setup if the tide is enabled).
            # SSS-restoring INPUT ASSEMBLY, hoisted above the ocean step.
            #
            # VALUE-IDENTICAL to assembling it after the step, which is why the
            # move is safe: the sea ice is updated earlier in THIS iteration
            # (step_sea_ice, above), the runoff already fed `fw`, the monthly
            # target selection is pure, and `_sss_ice` reads only `land_mask`
            # (static) plus `ice_state`, which `_ocean_step` does not modify.
            #
            # WHY IT MOVED: routing restoring as a real water flux (NEMO
            # nn_sssr=2) needs these inputs BEFORE the step, because the flux
            # must enter `fw.restoring` / `q_net` instead of being applied as a
            # post-step tracer edit.  It is also the more faithful ordering in
            # its own right: NEMO computes `sbcssr` in the surface-forcing
            # phase from the NOW-level SSS, whereas the post-step call below
            # sees the already-updated salinity.
            _sss_ice = None
            _R_gate = None
            _sss_tgt_step = None
            if sss_restore_cfg is not None:
                # NEMO-faithful ice gate (namsbc_ssr nn_sssr_ice=0: no SSS
                # restoring under sea ice).  Feed the SAME prescribed siconc
                # the albedo uses (``_sic``); with --prognostic-sea-ice the
                # prescribed NEMO siconc is no longer the truth, so gate on the
                # LIVE (ocean-masked) prognostic concentration instead.
                _sss_ice = _sic
                if ice_resp is not None:
                    _lc = ice_state.concentration.data
                    if _lc.ndim > np.asarray(state.land_mask.data).ndim:
                        _lc = jnp.sum(_lc, axis=-1)
                    _sss_ice = _lc * jnp.asarray(state.land_mask.data, _lc.dtype)
                # River-mouth gate (legoESM deviation, see sss_restoring.py):
                # restoring OFF at river mouths so it does not fight the plume
                # toward coarse WOA.  Gated by flag.
                _R_gate = _R if args.river_mouth_restoring_gate else None
                # Monthly (12, ...) NEMO sn_sss target -> this step's month;
                # static IC-surface target unchanged.
                _sss_tgt_step = (sss_restore_target[_runoff_month_idx(step, dt)]
                                 if _sss_monthly
                                 else sss_restore_target)
                # WATER-FLUX CHANNEL (NEMO nn_sssr=2).  Default OFF: the
                # post-step tracer edit below stays the only application, so an
                # unset flag is bit-identical to before.
                #
                # ON: the restoring flux enters `fw.restoring`, so it reaches
                # the ocean the way NEMO's does -- through the freshwater
                # budget, which drives eta / the z-star dilution
                # (`freshwater_eta_tendency = net_freshwater_flux / rho_0`,
                # the analogue of NEMO `pssh(Kaa) = pssh(Kbb) - rDt*(emp/rho0 +
                # hdiv)`) -- and it carries NEMO's heat term
                # (`qns -= erp*rcp*sst_m`, sbcssr.F90:138).  The post-step
                # applier is then SKIPPED; running both would apply restoring
                # TWICE, which is the sharpest failure mode of this change and
                # is asserted against in tests.
                if _sss_water_flux:
                    from legoesm.ocean.forcing.sss_restoring import (
                        compute_sss_restoring_flux as _sss_flux_fn,
                    )
                    _S_now = state.S.data[..., 0]
                    _T_now = state.T.data[..., 0]          # potential temp [degC]
                    _lm = jnp.asarray(state.land_mask.data, _S_now.dtype)
                    _sss_out = _sss_flux_fn(
                        S_model_top=_S_now,
                        S_target=jnp.asarray(_sss_tgt_step, _S_now.dtype),
                        lat_deg=jnp.asarray(lat2d, _S_now.dtype),
                        lon_deg=jnp.asarray(lon2d, _S_now.dtype),
                        ice_concentration=(jnp.zeros_like(_S_now)
                                           if _sss_ice is None
                                           else jnp.asarray(_sss_ice, _S_now.dtype)),
                        config=sss_restore_cfg,
                        river_runoff=(None if _R_gate is None
                                      else jnp.asarray(_R_gate, _S_now.dtype)),
                        sst_C=_T_now,
                    )
                    # Land cells contribute nothing to either budget.
                    _fw_restore = _sss_out["freshwater_flux"] * _lm
                    if fw is None:
                        # `fw` is None when the run has no P-E, runoff or ice
                        # (e.g. --no-emp on a forcing-free probe), and
                        # `_replace` on None would crash at the first step
                        # (codex 9387241).  Under this channel the restoring IS
                        # physical freshwater, so it needs a carrier: build a
                        # zero forcing and put it in the restoring slot.
                        from legoesm.ocean.freshwater import FreshwaterForcing
                        _z = jnp.zeros_like(_fw_restore)
                        fw = FreshwaterForcing(precip=_z, evap=_z, runoff=_z,
                                               ice_fw=_z,
                                               restoring=_fw_restore)
                    else:
                        fw = fw._replace(restoring=_fw_restore)
                    # NEMO's qns is positive INTO the ocean, matching q_net, so
                    # this adds with no sign flip (derivation at the term in
                    # sss_restoring.py).
                    _q_restore = _sss_out["heat_flux"] * _lm
                    sf = sf._replace(
                        q_net=(_q_restore if sf.q_net is None
                               else sf.q_net + _q_restore))
                    # KPP surface buoyancy: under this channel the restoring IS
                    # physical freshwater, so it belongs in the sum that the
                    # tracer-channel branch above deliberately excludes.
                    # Set it even when absent: skipping would drop the
                    # restoring-driven HALINE buoyancy from KPP while volume
                    # and heat still applied (codex 9387497).
                    sf = sf._replace(
                        freshwater=(_fw_restore if sf.freshwater is None
                                    else sf.freshwater + _fw_restore))
            state = _ensure_sharded_state(state)
            state = _ocean_step(state, sf, fw, _t_sec)
        if _gw_acc is not None:
            # READ-ONLY: `state` is never reassigned here, so the trajectory
            # is bit-identical to a run without the flag.
            #
            # NON-FATAL BOUNDARY (codex round-4 RED): this is a DIAGNOSTIC and
            # must never abort a production run, exactly like its sibling
            # _gateway_transport_diag.  gateway_step deliberately raises on a
            # bad geometry, and that raise sits INSIDE the step loop, so
            # without this guard a diagnostic could kill a multi-day run.  On
            # any failure: log once, disable the accumulator, keep integrating.
            # Note it does NOT fall back to per-step promotion -- that was the
            # defect the raise exists to prevent.
            from legoesm.ocean.diagnostics_sections import gateway_step
            try:
                _gw_acc = gateway_step(
                    _gw_acc, _gw_gates, state, z_coord, _gw_geom,
                    min_water_column_m=getattr(model.config,
                                               "min_water_column_m", None),
                    # "stored", not "auto": both supported grid branches turn
                    # store_mass_flux ON for this flag, so a state without the
                    # capture means some config path disabled it -- raise
                    # rather than silently integrate h*u (#1442, codex r6 RED3).
                    source="stored",
                    # Same promise for the EXACT salt channel: the builders
                    # set store_salt_flux with this flag, so a state without
                    # the pair means a config path disabled the capture.
                    require_salt=True)
            except Exception as _gw_e:
                print(f"[gateway] DISABLED at step {step} after "
                      f"{type(_gw_e).__name__}: {_gw_e}")
                # SALVAGE THE TAIL before discarding the accumulator (codex
                # round-1 RED).  gateway_step is pure and the assignment above
                # never happened, so _gw_acc still holds the LAST GOOD state,
                # with n = step - 1.  Without this dump everything accumulated
                # since the previous cadence row is lost -- and so is the
                # whole-run block, because _gateway_transport_diag(None) is a
                # no-op.  The writer suppresses a duplicate n, so a failure on
                # the step right after a cadence dump adds no second row.
                _gateway_cumulative_row(_gw_csv, _gw_acc, step - 1,
                                        (step - 1) * dt / _SEC_PER_DAY)
                _gw_acc = None
        if sss_restore_cfg is not None and not _sss_water_flux:
            # SKIPPED under the water-flux channel: the flux already entered
            # `fw.restoring` / `q_net` BEFORE the step.  Running this as well
            # would apply restoring TWICE, and the run would still look
            # plausible -- which is why this is a hard either/or, never a
            # blend, and why a test asserts the two channels are exclusive.
            #
            # NEMO-faithful ice gate (namsbc_ssr nn_sssr_ice=0: no SSS restoring
            # under sea ice).  Feed the SAME prescribed siconc the albedo uses
            # (``_sic``; None only if neither --ice-albedo nor a siconc field is
            # available, in which case the restoring is ungated as before).
            # With --prognostic-sea-ice the prescribed NEMO siconc is no longer
            # the truth — gate the restoring on the LIVE (ocean-masked) prognostic
            # ice concentration instead, via the SAME ``ice_concentration=``
            # parameter (codex MED): prescribed and live ice must not disagree in
            # the salt-restoring path.
            # `_sss_ice` / `_R_gate` / `_sss_tgt_step` were assembled ABOVE the
            # ocean step (see the hoist comment there).  They are unchanged by
            # the step, so this call is value-identical to the previous inline
            # assembly; the hoist exists so the same inputs can feed the
            # water-flux routing, which must run pre-step.
            # sss_apply pulls ONE 2-D S-surface slice to host and scatters
            # the updated layer back device-side (slice-before-convert
            # contract, locked by tests/unit/test_sss_apply.py) — counted.
            _pers_res.count_leaf_slice(pulls=1, writes=1)
            if app_grid_type == "mpas":
                from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step_mpas
                state = apply_sss_restoring_step_mpas(
                    state, S_target=_sss_tgt_step, ice_concentration=_sss_ice,
                    config=sss_restore_cfg, mesh=grid, dt=dt,
                    river_runoff=_R_gate)
            else:
                from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step
                state = apply_sss_restoring_step(
                    state, S_target=_sss_tgt_step, ice_concentration=_sss_ice,
                    config=sss_restore_cfg, grid=grid, z_coord=z_coord, dt=dt,
                    lat2d_deg=lat2d, lon2d_deg=lon2d,
                    river_runoff=_R_gate)
        if args.ice_thermo and _sic is not None:
            # Prescribed-ice freezing relaxation (the post-step half of the
            # thermodynamic boundary; the SW cut + (1-sic) flux suppression are in
            # compute_omip2_surface_forcing).  Nudge the under-ice surface ocean
            # toward freezing -> cools the over-warm Southern-Ocean under-ice cells
            # (>45S warm bias) + holds the Arctic near freezing.  Grid-agnostic
            # top-cell update (same host-state pattern as the SSS restoring).
            from legoesm.ocean.coupler.omip2_applicator import under_ice_freeze_relax
            # Surface-only op: pull ONLY the 2-D top layer to host (converting
            # the full leaf would assemble the whole 3-D sharded T — codex
            # batch4 HIGH), relax it, scatter it back DEVICE-SIDE.  The
            # in-place [...] assign reproduces the old full-array numpy cast
            # semantics exactly (f64 relax result -> leaf dtype), and the
            # untouched deep layers keep the original device buffer —
            # bit-identical to the old full round trip.  Counted — including
            # the extra S-surface liquidus pull when --freeze-scheme is
            # per-cell (MED-1).
            _pers_res.count_leaf_slice(
                pulls=(2 if args.freeze_scheme != "constant" else 1),
                writes=1)
            T0 = np.asarray(state.T.data[..., 0]).copy()   # copy: device arrays alias
            # --freeze-scheme != constant: per-cell liquidus target from the
            # LOCAL surface salinity (MED-1); constant keeps the fixed -1.8 C
            # scalar byte-identical (S_top=None short-circuits inside).  The
            # S pull slices FIRST too (device-side) — only the 2-D surface
            # layer crosses to host.
            T0[...] = under_ice_freeze_relax(
                T0, _sic, dt, tau_ice_days=args.ice_thermo_tau_days,
                S_top=(np.asarray(state.S.data[..., 0])
                       if args.freeze_scheme != "constant" else None),
                freeze_scheme=args.freeze_scheme)
            state = state._replace(
                T=Field(jnp.asarray(state.T.data).at[..., 0].set(jnp.asarray(T0)),
                        name=state.T.name,
                        dims=state.T.dims, units=state.T.units))
        if args.geothermal:
            # Geothermal bottom heat-flux BC (NEMO ln_trabbc): warm the deepest
            # wet cell of each column by the seafloor heat flux Q_geo. Live
            # partial-cell thickness from compute_layer_thickness (true h_k);
            # wet = h_k > 1e-3 m (below-seafloor rock cells carry h_k = 0). Grid-
            # agnostic (..., nlev) host post-step update, like the SSS/ice-thermo
            # paths. Explicit + unconditionally stable (no inter-level coupling).
            from legoesm.ocean.coupler import apply_geothermal_step
            from legoesm.ocean.physics.geothermal import GeothermalConfig
            from legoesm.ocean.vertical import compute_layer_thickness
            _eta = getattr(state, "eta", None)
            _eta_arr = (state.eta.data if _eta is not None
                        else jnp.zeros_like(jnp.asarray(H_bathy)))
            _dz = compute_layer_thickness(_eta_arr, jnp.asarray(H_bathy), z_coord)
            # Wet = positive live thickness AND ocean (the 2-D land mask): on a
            # pure-z* grid a land column with nonzero eta has dz>0 but must NOT
            # be geothermally heated (codex). land_mask is (...,) -> broadcast.
            _lm = jnp.asarray(state.land_mask.data)[..., None] > 0.5
            _wet = ((_dz > 1e-3) & _lm).astype(_dz.dtype)
            _geo_cfg = GeothermalConfig(enabled=True)
            if args.geothermal_flux_wm2 is not None:
                _geo_cfg = _geo_cfg._replace(flux_wm2=args.geothermal_flux_wm2)
            state = apply_geothermal_step(
                state, dz_live=_dz, wet_cell=_wet, dt=dt, config=_geo_cfg)
        if isf_forcing is not None:
            # NEMO ISF 'spe' prescribed melt (ln_isfpar_mlt, cn_isfpar_mlt=
            # 'spe'): monthly Depoorter melt deposited over the per-column
            # [zmin, zmax] band — latent cooling + melt heat content at the
            # in-situ freezing point + virtual-salt dilution + eta volume
            # source.  Host post-step apply, same geometry inputs as the
            # geothermal BC above.
            from legoesm.ocean.coupler.ice_shelf_apply import (
                apply_isf_prescribed_melt_step,
            )
            from legoesm.ocean.vertical import compute_layer_thickness
            _eta_arr = (state.eta.data if getattr(state, "eta", None)
                        is not None
                        else jnp.zeros_like(jnp.asarray(H_bathy)))
            _dz_isf = compute_layer_thickness(
                _eta_arr, jnp.asarray(H_bathy), z_coord)
            _lm_isf = jnp.asarray(state.land_mask.data)[..., None] > 0.5
            _wet_isf = ((_dz_isf > 1e-3) & _lm_isf).astype(_dz_isf.dtype)
            _mi = _runoff_month_idx(step, dt)
            state = apply_isf_prescribed_melt_step(
                state,
                fwf_kg_m2_s=jnp.asarray(isf_forcing.fwf[_mi]),
                zmin_m=jnp.asarray(isf_forcing.zmin[_mi]),
                zmax_m=jnp.asarray(isf_forcing.zmax[_mi]),
                dz_live=_dz_isf, wet_cell=_wet_isf, dt=dt,
                rho_0=float(model.config.rho_0))
        if bbl_geom is not None:
            # NEMO advective BBL (Campin-Goosse): dense shelf bottom water
            # descends the slope. Host post-step exchange, exactly tracer-
            # conserving; transports recomputed from current bottom T/S.
            from legoesm.ocean.physics.bbl_adv import apply_bbl_adv_step
            state = apply_bbl_adv_step(
                state, bbl_geom, dt,
                gamma_s=args.bbl_gamma_s,
                rho_0=float(model.config.rho_0),
                area_2d=jnp.asarray(grid.area),
                dy_u_faces=bbl_face_widths[0],
                dx_v_faces=bbl_face_widths[1],
                nlev=int(args.nlev))
        if nudge_tau_s > 0 and (nudge_release_s <= 0 or step * dt < nudge_release_s):
            a = dt / nudge_tau_s
            # WOA nudging is genuinely FULL-3-D (every level relaxes toward
            # the climatology), so this is an honest full-leaf host round
            # trip: T and S gathered + re-uploaded every nudging step —
            # counted, never hidden behind full_state_gathers=0 (codex
            # batch4 HIGH).  Device-side nudging is scaling-M2 increment 2.
            _pers_res.count_leaf_full(gathers=2, uploads=2)
            Tn = np.asarray(state.T.data)
            Sn = np.asarray(state.S.data)
            Tn = Tn + a * (nudge_T - Tn) * nudge_m3
            Sn = Sn + a * (nudge_S - Sn) * nudge_m3
            state = state._replace(
                T=Field(jnp.asarray(Tn), name=state.T.name,
                        dims=state.T.dims, units=state.T.units),
                S=Field(jnp.asarray(Sn), name=state.S.name,
                        dims=state.S.dims, units=state.S.units),
            )
        if drag_tau_s > 0 and step * dt < drag_days_s:
            df = float(np.exp(-dt / drag_tau_s))   # Rayleigh decay factor
            _upd = {"u": Field(jnp.asarray(np.asarray(state.u.data) * df),
                               name=state.u.name, dims=state.u.dims,
                               units=state.u.units)}
            # MPAS has no separate v field (u is edge-normal).
            if getattr(state, "v", None) is not None:
                _upd["v"] = Field(jnp.asarray(np.asarray(state.v.data) * df),
                                  name=state.v.name, dims=state.v.dims,
                                  units=state.v.units)
            # Full-3-D leaf round trips (host scalar multiply of u and v) —
            # counted while the spin-up drag is active (codex batch4 HIGH:
            # honest accounting; device-side drag is scaling-M2 increment 2).
            _pers_res.count_leaf_full(gathers=len(_upd), uploads=len(_upd))
            state = state._replace(**_upd)
        if step % diag_every == 0 or step == n_steps:
            state = jax.block_until_ready(state)
            # _diag pulls the 2-D T,S surface slices + the 2-D land mask and
            # the FULL 3-D u(,v) leaves (max|u| + its location need every
            # level) — diag-cadence host reads, counted for the honest-cost
            # done line.
            _pers_res.count_leaf_slice(pulls=3)
            _pers_res.count_leaf_full(
                gathers=1 + int(getattr(state, "v", None) is not None))
            d = _diag(state, lat2d, lon2d)
            # Throughput of THIS leg: a resumed run has done (step-start_step)
            # steps in (now - t_wall), not `step` of them.
            rate = (step - start_step) / (time.time() - t_wall)
            day = step * dt / _SEC_PER_DAY
            print(f"[diag] step {step} (day {day:.0f}): {d} | {rate:.2f} steps/s",
                  flush=True)
            _log_diag_csv(step, day, d, rate, ice=ice_state)
            if ice_resp is not None:
                _ice_diag = _prognostic_ice_diag(ice_state, ice_resp, grid,
                                                 app_grid_type,
                                                 state.land_mask.data)
                print(f"[ice]  step {step}: {_ice_diag}", flush=True)
            if not d["finite"]:
                print("[ABORT] non-finite state", flush=True)
                # persistent lane: the snapshot needs the full staggered-v
                # global layout (abort boundary — one gather, then exit).
                state = _ensure_global_state(state)
                _save_snapshot(out_dir, f"blowup_step{step}", state, lat2d, lon2d,
                               grid=grid, io_proc=_is_io_proc(), ice_state=ice_state)
                _close_csv()
                # Dump what the accumulator reached before the abort: the last
                # cadence row alone would understate the run, and the run-end
                # writer below is never reached on this path.
                _gateway_cumulative_row(_gw_csv, _gw_acc, step, day)
                _gateway_cumulative_close(_gw_csv)
                return 1
        if snap_every > 0 and step % snap_every == 0 and step != n_steps:
            day = step * dt / _SEC_PER_DAY
            # persistent lane: snapshot cadence = a real output boundary; the
            # state is gathered here (counted) and re-sharded lazily at the
            # next step's _ensure_sharded_state.
            state = _ensure_global_state(state)
            _save_snapshot(out_dir, f"day{int(round(day)):04d}", state, lat2d,
                           lon2d, z_coord=z_coord, io_proc=_is_io_proc(),
                           ice_state=ice_state, grid=grid,
                           step=step, day=day,
                           extra=(_kprofiles(model, state, sf, dt, z_coord)
                                  if args.kprofile_snapshots else None))
            print(f"[snapshot] day {day:.0f} saved", flush=True)
            # Same cadence as the snapshot, and AFTER this step's
            # gateway_step, so the row's n_steps matches the snapshot's day.
            # `step != n_steps` above excludes the final step; the run-end row
            # below covers it, so each dump point appears exactly once.
            _gateway_cumulative_row(_gw_csv, _gw_acc, step, day)
        if (restart_every > 0 and step % restart_every == 0
                and step != n_steps):
            # Own cadence, own contract: the state may still be sharded on the
            # persistent SPMD lane, so gather before serialising (idempotent
            # when the snapshot block above already did).
            state = _ensure_global_state(state)
            _write_run_restart(step, step * dt / _SEC_PER_DAY, state, ice_state)
        if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
            yr = step // steps_per_year
            state = _ensure_global_state(state)
            _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d,
                           z_coord=z_coord, grid=grid, io_proc=_is_io_proc(),
                           ice_state=ice_state)
            print(f"[snapshot] year {yr} saved", flush=True)

    state = jax.block_until_ready(state)
    # persistent lane: final I/O boundary — the snapshot / transport diags /
    # state digest all need the full staggered-v global layout (one gather).
    state = _ensure_global_state(state)
    _io = _is_io_proc()
    _save_snapshot(out_dir, "final", state, lat2d, lon2d, z_coord=z_coord,
                   io_proc=_io, ice_state=ice_state, grid=grid,
                   step=step, day=day,
                   extra=(_kprofiles(model, state, sf, dt, z_coord)
                          if args.kprofile_snapshots else None))
    _write_run_restart(n_steps, n_steps * dt / _SEC_PER_DAY, state, ice_state)
    _amoc26n_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _acc_drake_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _save_bsf_amoc_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _mht_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    # Run-end CUMULATIVE row FIRST, then the whole-run mean.  The last window
    # (e.g. days 60-90 of a 90-day run) is only recoverable if the final totals
    # are dumped: the cadence block above skips step == n_steps.
    _gateway_cumulative_row(_gw_csv, _gw_acc, n_steps,
                            n_steps * dt / _SEC_PER_DAY)
    _gateway_cumulative_close(_gw_csv)
    _gateway_transport_diag(_gw_acc, out_dir, io_proc=_io)
    _record_final_state_digest(manifest_path, state)
    _close_csv()
    rate = (n_steps - start_step) / (time.time() - t_wall)
    if _spmd_persistent:
        # The honest cost lines (codex batch4 HIGH): (1) FULL-STATE layout
        # flips the persistent lane actually performed (the old wrapper does
        # 2 per step: scatter + gather) — includes the snapshot/abort/final
        # cadence gathers and any forced per-step ones announced at setup;
        # (2) the per-step LEAF host transfers that REMAIN (forcing-builder
        # surface-T pulls, SSS/ice-thermo surface pull+write-backs, nudge/
        # drag full-3-D round trips, diag-cadence reads) — the persistent
        # lane eliminates full-STATE round trips, NOT these, and they are
        # never reported as zero cost.
        _builder_pulls = (host_pull_ledger()["surface_slice_pulls"]
                          - _ledger0["surface_slice_pulls"])
        _slice_pulls = _pers_res.leaf_slice_pulls + _builder_pulls
        # Per-step rates use THIS LEG's step count: the counters above were
        # zeroed at leg start, so dividing by the absolute n_steps would
        # under-report a resumed leg's cost (codex r1 LOW).
        _leg_steps = max(1, n_steps - start_step)
        print(f"[spmd-persistent] full-STATE gathers={_pers_res.gathers} "
              f"shards={_pers_res.shards} over {_leg_steps} steps this leg "
              f"({_pers_res.gathers / _leg_steps:.4f} gathers/step; "
              f"wrapper lane would be {_leg_steps} + {_leg_steps})", flush=True)
        print(f"[spmd-persistent] LEAF host transfers (NOT in the full-STATE "
              f"count above): 2-D surface-slice pulls={_slice_pulls} "
              f"({_slice_pulls / _leg_steps:.2f}/step; {_builder_pulls} "
              f"from the forcing builders), surface-slice "
              f"write-backs={_pers_res.leaf_slice_writes}, FULL-3D leaf "
              f"gathers={_pers_res.leaf_full_gathers} "
              f"uploads={_pers_res.leaf_full_uploads} (WOA nudge / spin-up "
              f"drag while active + diag-cadence u,v).", flush=True)
    print(f"[done] {n_steps - start_step} steps this leg (absolute step "
          f"{n_steps}) @ {rate:.2f} steps/s; final: {_diag(state, lat2d, lon2d)}")
    if args.smoke:
        yr_est = steps_per_year / rate / 3600.0
        print(f"[smoke] projected wall-time: {yr_est:.2f} h/yr  "
              f"({args.years:.0f}yr -> {yr_est*args.years:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
