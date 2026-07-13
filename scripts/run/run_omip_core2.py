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

_SEC_PER_DAY = 86400.0
_SEC_PER_6H = 21600.0
_YEAR_S = 365.0 * _SEC_PER_DAY
_MESH = "data/grids/eORCA1.2_mesh_mask.nc"


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


def build_tripole(nlev: int, H_max: float, mesh_path: str,
                  woa_init: bool = False, woa_t=None, woa_s=None, n_gpus: int = 1,
                  pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None,
                  ke_gradient_scheme=None, partial_cell=False,
                  adaptive_implicit_vertadv=None, bathy_smoothing_passes=0,
                  momentum_time_integrator=None, barotropic_solver=None,
                  barotropic_diffusion_alpha=None, n_barotropic_substeps=None,
                  barotropic_time_filter=None, bottom_drag_r=None,
                  C_smag=None, C_leith=None, C_smag_lap=None,
                  momentum_advection=None, slope_foot_alpha=None,
                  slope_foot_n_levels=None, slope_foot_threshold=None,
                  min_levels=1, div_damp_2=None, div_damp_4=None,
                  smag_cfl_safety=None, convection="none",
                  convection_K_conv=1.0, convection_K_bg=1e-5,
                  freeze_floor=None, ew_cyclic_overlap=None,
                  runoff_depth_spread_m=None, tracer_advection=None,
                  mle=None, dz_ref_override=None,
                  bottom_drag_scheme=None, bottom_drag_cd0=None,
                  bottom_drag_cdmax=None, bottom_drag_z0=None,
                  bottom_drag_ke0=None, iwm=None, iwm_forcing_file=None,
                  ddm=None):
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
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ("momentum_time_integrator", momentum_time_integrator),
                              ("barotropic_solver", barotropic_solver),
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
                              ("ew_cyclic_overlap", ew_cyclic_overlap),
                              ("runoff_depth_spread_m", runoff_depth_spread_m),
                              ("tracer_advection", tracer_advection),
                              ) if v is not None}
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
    if _use_convection or mle is not None or _use_iwm or _use_ddm:
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig, EnhancedDiffusionConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
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
        _vm_cfg = VerticalMixingConfig(scheme="none")
        if _use_iwm:
            # zdfiwm rides the vertical-mixing config; scheme stays "none"
            # (the tripole oracle has no closure scheme in the pipeline —
            # backgrounds + convection EVD), the additive wave K enters in
            # compute_vertical_K_profiles.  implicit_vertical_mixing is
            # already forced True above.
            _vm_cfg = _vm_cfg._replace(iwm=iwm)
            # NEMO zdfiwm_init FORCES the model backgrounds to molecular
            # values (avmb = rnu = 1.4e-6 m²/s, avtb = 1e-10 m²/s): the wave
            # field IS the interior background.  Mirror that (codex r1 #2) —
            # keeping the OMIP A_v/K_v floors would double-count backgrounds.
            from legoesm import constants as _const
            _ovr["A_v"] = _const.nu_ocean_molecular
            _ovr["K_v"] = 1.0e-10   # NEMO avtb with ln_zdfiwm
            print("[setup] zdfiwm: model backgrounds forced to molecular "
                  f"(A_v={_ovr['A_v']:g}, K_v={_ovr['K_v']:g}) per zdfiwm_init")
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
              f"convection={convection if _use_convection else 'none'} "
              f"(K_conv={convection_K_conv} K_bg={convection_K_bg}) "
              f"MLE={'ce=%g' % mle.ce if mle is not None else 'off'} "
              f"IWM={'on' if _use_iwm else 'off'} "
              f"DDM={'on' if _use_ddm else 'off'}")
    if _ovr:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        # #501/#661: _ovr carries FLAT names (A_h/C_smag_lap/barotropic_solver/
        # bottom_drag_r/...) now nested in sub-configs; replace_flat routes them.
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
                       pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None,
                       ke_gradient_scheme=None, partial_cell=False,
                       adaptive_implicit_vertadv=None, bathy_smoothing_passes=0,
                  momentum_time_integrator=None, barotropic_solver=None,
                  barotropic_diffusion_alpha=None, n_barotropic_substeps=None,
                  barotropic_time_filter=None, bottom_drag_r=None,
                  C_smag=None, C_leith=None, C_smag_lap=None,
                  momentum_advection=None, slope_foot_alpha=None,
                  slope_foot_n_levels=None, slope_foot_threshold=None,
                  min_levels=1, div_damp_2=None, div_damp_4=None,
                  smag_cfl_safety=None, freeze_floor=None,
                  use_polar_filter=None, polar_filter_cutoff_lat_deg=None,
                  polar_filter_max_wave_speed=None,
                  polar_filter_safety_factor=None,
                  runoff_depth_spread_m=None, tracer_advection=None,
                  mle=None, dz_ref_override=None, mask_marginal_seas=False,
                  bottom_drag_scheme=None, bottom_drag_cd0=None,
                  bottom_drag_cdmax=None, bottom_drag_z0=None,
                  bottom_drag_ke0=None, iwm=None, iwm_forcing_file=None,
                  ddm=None, vertical_mixing=None):
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
        dz_ref_override=dz_ref_override,
        # KPP boundary-layer-depth override (Ri_crit / Cv): None reproduces the
        # default KPPConfig byte-for-byte; a custom VerticalMixingConfig shoals
        # the diagnosed-too-deep JANUARY winter mixed layer (ll2: NH-midlat Jan
        # MLD 174 m vs NEMO 93 m -> over-mixes away the warm 100 m mode water ->
        # -2.83 C SST). The use_bathymetry path threads this into bathy_physics.
        vertical_mixing=vertical_mixing,
    )
    _ovr = {k: v for k, v in (("K_bih", K_bih),
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ("momentum_time_integrator", momentum_time_integrator),
                              ("barotropic_solver", barotropic_solver),
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
                              ("runoff_depth_spread_m", runoff_depth_spread_m),
                              ("tracer_advection", tracer_advection),
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
                     pgf_scheme=None, bottom_drag_r=None,
                     bottom_drag_bbl_thickness=None, bottom_drag_bg_velocity=None,
                     partial_cell=False, dz_ref_override=None,
                     n_barotropic_substeps=None,
                     barotropic_solver=None, freeze_floor=None,
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
                              ("freeze_floor", freeze_floor),
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


def _kpp_vmix_override(kpp_ri_crit=None, kpp_cv=None):
    """Build a KPP ``VerticalMixingConfig`` overriding ONLY the CLI-set knobs.

    Returns ``None`` when neither knob is given so the caller falls through to
    ``_create_setup``'s default ``VerticalMixingConfig(scheme="kpp")`` — i.e.
    byte-for-byte the pre-flag config (no silent re-defaulting of the other
    KPP fields).  ``Ri_crit`` / ``Cv`` must be finite and within their accepted
    range (see ``_KPP_RI_CRIT_RANGE`` / ``_KPP_CV_RANGE``)."""
    if kpp_ri_crit is None and kpp_cv is None:
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
    return VerticalMixingConfig(scheme="kpp", kpp=kpp)


def _validate_kpp_grid(grid, kpp_ri_crit=None, kpp_cv=None):
    """Reject the KPP override flags on grids whose CORE-II builder does not
    thread ``vertical_mixing`` INTO A LIVE KPP scheme (a flag that silently does
    nothing is the dispatch footgun CLAUDE.md forbids).  ``mpas`` and
    ``latlon_bathy`` run KPP (``bathy_physics.vertical_mixing``) so the override
    reaches ``KPPConfig.Ri_crit``/``Cv``.  ``tripole`` ships ``physics=None``
    (the dynamics-core implicit vertical solve, NO KPP boundary layer) so a KPP
    override would be a silent no-op there -> still rejected; ``cubed_sphere``
    is not wired.  Extend this set only when the builder actually runs KPP."""
    if (kpp_ri_crit is not None or kpp_cv is not None) and grid not in (
            "mpas", "latlon_bathy"):
        raise SystemExit(
            f"--kpp-ri-crit/--kpp-cv are wired for --grid mpas/latlon_bathy "
            f"(grids that run the KPP boundary layer), not --grid {grid!r}. "
            f"tripole runs the dynamics-core implicit vertical solve (no KPP) "
            f"so the override would silently do nothing.")


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


def _diag(state, lat2d=None, lon2d=None) -> dict:
    """Cheap scalar diagnostics over ocean cells (one device->host pull).

    ``umax_lat``/``umax_lon``/``umax_lev`` = location of the 3-D max|u| —
    localises WHERE velocity grows/blows up (equator f->0 vs western
    boundaries vs the bipolar cap/fold vs at depth), the key pin-point for
    the dynamics instability seed.
    """
    T = np.asarray(state.T.data)[..., 0]
    S = np.asarray(state.S.data)[..., 0]
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


def _surface_currents(state, grid, app_grid_type):
    """Top-level ocean currents (u_east, v_north) at T points / cells [m/s].

    Reused as ``ocean_u`` / ``ocean_v`` for ``step_sea_ice``.  MPAS stores the
    edge-normal ``u`` (nEdges, nlev); reconstruct cell-centred (u, v) with the
    canonical Perot ``reconstruct_cell_velocity`` (init_mpas).  The C-grid
    families (latlon / tripole) store cell-centred ``u`` / ``v`` faces; take the
    surface level directly on the T-shape they already carry — the ice model only
    needs an O(0.1 m/s) drift reference for the ocean-ice drag, so the face value
    at the matching index is an adequate cell-centre proxy (and avoids a bespoke
    face->centre average)."""
    if app_grid_type == "mpas":
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity
        u_sfc, v_sfc = reconstruct_cell_velocity(state.u.data[:, 0], grid)
        return u_sfc, v_sfc
    # latlon / tripole C-grid: u on EW faces (n_lat, n_lon+1), v on NS faces
    # (n_lat+1, n_lon); crop to the T shape (n_lat, n_lon) at the surface level.
    n_lat = int(np.asarray(grid.lat_T if app_grid_type == "tripole"
                           else grid.lat).shape[0]) if app_grid_type != "latlon" \
        else int(np.asarray(grid.lat).shape[0])
    u_face = state.u.data[..., 0]
    v_face = state.v.data[..., 0]
    u_sfc = u_face[:, :-1]                      # drop the periodic wrap column
    v_sfc = 0.5 * (v_face[:-1, :] + v_face[1:, :])
    return u_sfc, v_sfc


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


def _prognostic_ice_diag(ice_state, resp, grid, app_grid_type, ocean_mask):
    """One-line verifiable summary over OCEAN cells: ice AREA [10^6 km2], mean
    concentration, max thickness, and the area-mean salt flux over ice-covered
    cells.

    ``grid.area`` is the per-cell area [m2] on all three supported grids
    (VoronoiMesh exposes it as ``areaCell``; the C-grid families as the
    ``.area`` property).  ``ocean_mask`` restricts the summary to wet cells so a
    spurious land-ice growth (masked out of the ocean budget) does not inflate
    the reported area."""
    conc = np.asarray(ice_state.concentration.data, dtype=np.float64)
    h = np.asarray(ice_state.h_ice.data, dtype=np.float64)
    if conc.ndim > h.ndim:  # safety (single-category here)
        conc = conc.sum(axis=-1)
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
    icy = conc > 1.0e-3
    salt = np.asarray(resp.salt_flux, dtype=np.float64)
    salt_mean = float(salt[icy].mean()) if icy.any() else 0.0
    return (f"ice_area={ice_area_m2 / 1.0e12:.3f}e6 km2 "
            f"mean_conc={float(conc[conc > 0].mean()) if (conc > 0).any() else 0.0:.3f} "
            f"max_h={float(h.max()):.3f} m "
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
                   io_proc: bool = True):
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
    # MPAS has no separate v field; the scorer reads T/S/land_mask/lat_T/lon_T only.
    if getattr(state, "v", None) is not None:
        save_kw["v"] = np.asarray(state.v.data)
    # Free surface: needed to RESTART a run from this snapshot (--restart-from)
    # without a barotropic-adjustment shock; the scorer ignores it.
    eta = getattr(state, "eta", None)
    if eta is not None:
        save_kw["eta"] = np.asarray(eta.data)
    # Geometry for the offline mixed-layer-depth diagnostic (de Boyer Montegut /
    # Treguier 2023): sea-floor depth + level-centre reference depths.  The MLD
    # scorer derives the per-level wet mask from ``z_center_ref < H_bathy``.
    H_bathy = getattr(state, "H_bathy", None)
    if H_bathy is not None:
        save_kw["H_bathy"] = np.asarray(H_bathy.data)
    if z_coord is not None and getattr(z_coord, "z_half_ref", None) is not None:
        zh = np.asarray(z_coord.z_half_ref)              # (nlev+1,), <=0
        save_kw["z_center_ref"] = np.abs(0.5 * (zh[:-1] + zh[1:]))   # (nlev,) positive
    if not io_proc:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"snapshot_{tag}.npz", **save_kw)


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


def main() -> int:
    # allow_abbrev=False: the module-level x64 toggle is decided by an EXACT
    # "--fp32" argv match (``_FP32``), so the real parser must NOT accept an
    # abbreviation (e.g. "--fp") of --fp32 — that would set args.fp32=True
    # while x64 was already enabled, tripping the consistency guard below.
    # All sbatch wrappers already use full flag names, so this is behaviour-
    # preserving for existing callers.
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
                        "threshold), like NEMO sbcssr's (1-2*rnfmsk) damping "
                        "mask — otherwise the restoring fights the river plume "
                        "toward the coarse WOA climatology. Requires --runoff "
                        "+ --sss-restore.")
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
    p.add_argument("--polar-filter", action="store_true",
                   help="Enable the mask-aware Fourier polar filter (lat-lon grid only): "
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
    args = p.parse_args()

    # KPP MLD-deepening sensitivity flags are mpas-only (fail loud, never silent).
    _validate_kpp_grid(args.grid, args.kpp_ri_crit, args.kpp_cv)

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
    if args.grid == "tripole":
        grid, z_coord, model, state, H_bathy = build_tripole(
            args.nlev, args.H_max, args.mesh,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            n_gpus=args.n_gpus,
            pgf_scheme=args.pgf_scheme, A_h=args.A_h, B_h=args.B_h, K_bih=args.K_bih,
            flat_bottom=args.flat_bottom, A_h_eq_boost=args.A_h_eq_boost,
            ke_gradient_scheme=args.ke_gradient_scheme,
            partial_cell=args.partial_cell,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            bathy_smoothing_passes=args.bathy_smoothing_passes,
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            freeze_floor=(True if args.freeze_floor else None),
            runoff_depth_spread_m=args.runoff_depth_spread_m,
            barotropic_solver=args.barotropic_solver,
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
            runoff_depth_spread_m=args.runoff_depth_spread_m,
            mle=mle_cfg, dz_ref_override=_nemo_dz,
            bottom_drag_scheme=args.bottom_drag_scheme,
            bottom_drag_cd0=args.bottom_drag_cd0,
            bottom_drag_cdmax=args.bottom_drag_cdmax,
            bottom_drag_z0=args.bottom_drag_z0,
            bottom_drag_ke0=args.bottom_drag_ke0,
            iwm=_iwm_cfg, ddm=_ddm_cfg,
            vertical_mixing=_kpp_vmix_override(args.kpp_ri_crit, args.kpp_cv),
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
            ke_gradient_scheme=args.ke_gradient_scheme,
            partial_cell=args.partial_cell,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            bathy_smoothing_passes=args.bathy_smoothing_passes,
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            freeze_floor=(True if args.freeze_floor else None),
            runoff_depth_spread_m=args.runoff_depth_spread_m,
            tracer_advection=args.tracer_advection,
            barotropic_solver=args.barotropic_solver,
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
            use_polar_filter=(True if args.polar_filter else None),
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
            vertical_mixing=_kpp_vmix_override(args.kpp_ri_crit, args.kpp_cv),
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
                                      or args.kpp_cv is not None):
                raise ValueError(
                    "--kpp-ri-crit/--kpp-cv conflict with a --config ocean.physics "
                    "block: the YAML physics config would overwrite the CLI KPP "
                    "override. Set Ri_crit/Cv in the YAML "
                    "(ocean.physics.vertical_mixing.kpp) OR drop the ocean.physics "
                    "section and use the CLI flags -- not both.")
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

    sss_restore_cfg = None
    sss_restore_target = None
    if args.sss_restore:
        if app_grid_type == "cubed_sphere":
            raise ValueError("--sss-restore: not wired for the cube (parked grid).")
        if not args.woa_init:
            raise ValueError("--sss-restore requires --woa-init (the restoring "
                             "target is the WOA surface-salinity climatology).")
        if not (float(args.sss_restore_tau_days) > 0.0):
            raise ValueError("--sss-restore-tau-days must be > 0 (0 divides by "
                             "zero in build_region_masks; negative = anti-restoring).")
        from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig
        from legoesm import constants
        _cfg_kwargs = {}
        if args.sss_ice_gate_nemo:
            _cfg_kwargs["ice_gate_mode"] = "nemo_linear"
        if args.sss_restore_bound_mmday is not None:
            if not (float(args.sss_restore_bound_mmday) > 0.0):
                raise ValueError("--sss-restore-bound-mmday must be > 0.")
            # mm/day water-equivalent -> kg/m^2/s (rho_water * m/day / 86400).
            _cfg_kwargs["max_flux_kg_m2_s"] = (
                float(args.sss_restore_bound_mmday) * 1.0e-3 / 86400.0
                * float(constants.rho_water))
        sss_restore_cfg = SSSRestoringConfig(
            enabled=True,
            tau_restore_days_default=float(args.sss_restore_tau_days),
            **_cfg_kwargs,
        )
        sss_restore_target = np.asarray(
            state.S.data, dtype=np.float64)[..., 0].copy()      # surface SSS
        _wet = np.asarray(state.land_mask.data) > 0.5
        _bnd = (f"{args.sss_restore_bound_mmday:.1f} mm/day (NEMO ln_sssr_bnd)"
                if args.sss_restore_bound_mmday is not None
                else "200 mm/day safety cap")
        print(f"[setup] SSS restoring ON: tau_default="
              f"{args.sss_restore_tau_days:.0f} d + OMIP-2 regional masks; "
              f"flux bound {_bnd}; target = WOA surface SSS "
              f"[{sss_restore_target[_wet].min():.1f},"
              f"{sss_restore_target[_wet].max():.1f}] PSU")

    if args.woa_smoothing_passes and args.woa_smoothing_passes > 0:
        if not args.woa_init:
            raise ValueError("--woa-smoothing-passes requires --woa-init.")
        if args.grid == "mpas":
            raise ValueError(
                "--woa-smoothing-passes is not available for mpas: smooth_woa_ts "
                "uses the structured 2-D laplacian_smooth_2d; a Voronoi "
                "connectivity smoother (cellsOnCell) is future work.")
        state = smooth_woa_ts(state, grid, args.woa_smoothing_passes)

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

    lat2d, lon2d = _grid_lat2d_deg(grid, args.grid)
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

    # Prognostic sea-ice (--prognostic-sea-ice): build the canonical SeaIceConfig
    # + a zero-ice cold-start state on the OCEAN grid.  The REAL model
    # (legoesm.ice.step_sea_ice) is stepped each loop iteration and its
    # brine/melt/heat response is routed into the existing ocean channels.
    ice_config = None
    ice_state = None
    if args.prognostic_sea_ice:
        from legoesm.ice import (
            SeaIceConfig, init_dynamic_ice_state, step_sea_ice,
            grid_supports_ice_dynamics,
        )
        from legoesm.ice.config import BrineConfig
        # Free-drift fallback if the grid lacks strain-rate/transport operators
        # (NOT 'none', which yields no drift/export).
        _ice_dyn = args.prognostic_ice_dynamics
        # NOTE: the tripole grid object is a LatLonCGridGeometry, which
        # grid_supports_ice_dynamics() does NOT recognise (it matches LatLonGrid
        # / VoronoiMesh / CubedSphereGrid).  So tripole degrades to free_drift +
        # transport='none'.  The brine SALT flux + melt/freeze FRESHWATER + ocean
        # HEAT extraction (the channels that balance Arctic runoff) are produced
        # by the thermodynamics regardless of the rheology, so export is PRESERVED
        # under free_drift — only the velocity-driven tracer advection / ridging
        # are dropped.  MPAS (VoronoiMesh) is the primary, fully-supported target.
        _supports = grid_supports_ice_dynamics(grid)
        if not _supports and _ice_dyn in ("mevp", "evp"):
            print(f"[setup] prognostic ice: grid {type(grid).__name__} lacks "
                  f"strain-rate/transport ops -> dynamics {_ice_dyn!r} -> "
                  "'free_drift', transport 'none' (brine salt + melt freshwater + "
                  "ocean-heat export PRESERVED; tracer advection/ridging dropped). "
                  "Use --grid mpas for full mEVP + transport.")
            _ice_dyn = "free_drift"
        _transport = "advect" if _supports else "none"
        _brine = BrineConfig(enabled=True)
        if args.prognostic_ice_salinity is not None:
            _brine = _brine._replace(S_ice_new=float(args.prognostic_ice_salinity))
        ice_config = SeaIceConfig(
            dynamics=_ice_dyn,
            transport=_transport,
            brine=_brine,            # brine-rejection salt flux -> ocean salt_flux
        )
        ice_shape = _ice_state_spatial_shape(grid, app_grid_type)
        # Zero-ice cold start (h=0, concentration=0); spins up from the forcing.
        ice_state = init_dynamic_ice_state(ice_shape, S_ice_init=0.0)
        ice_state = ice_state._replace(
            concentration=ice_state.concentration.replace(
                data=jnp.zeros_like(ice_state.concentration.data)))
        from legoesm import constants as _ice_const
        _ice_T_freeze = float(_ice_const.T_freeze)   # degC ocean T -> K for ice
        print(f"[setup] PROGNOSTIC SEA ICE: step_sea_ice dynamics={_ice_dyn!r} "
              f"transport={_transport!r} brine=ON (S_ice_new="
              f"{_brine.S_ice_new:.1f} PSU) on {app_grid_type} shape {ice_shape}; "
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

    print(f"[run] {total_days:.0f} days = {n_steps} steps "
          f"(diag every {diag_every} steps"
          f"{f', snapshot every {snap_every} steps' if snap_every else ''})")
    d0 = _diag(state, lat2d, lon2d)
    print(f"[diag] step 0: {d0}", flush=True)

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
                forcing_path=str(args.forcing_path or ""),
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
                        ("core2_forcing", args.forcing_path),
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
    # Process-0-only CSV under --distributed: every process runs the same host
    # loop on the all-gathered replicated state, so a single writer suffices and
    # avoids N processes clobbering the same file.  On non-IO ranks _csv is None
    # and the writer/closer below are no-ops.
    _csv = None
    if _is_io_proc():
        _csv = open(out_dir / "diag_timeseries.csv", "w")
        _csv.write(",".join(_csv_cols) + "\n")

    def _log_diag_csv(step, day, d, rate):
        if _csv is None:
            return
        _csv.write(
            f"{step},{day:.3f},{d['mean_sst_C']:.4f},{d['mean_sss']:.4f},"
            f"{d['max_abs_u']:.6e},{d['max_abs_v']:.6e},{d['umax_lat']},"
            f"{d['umax_lon']},{d['umax_lev']},{rate:.3f}\n")
        _csv.flush()

    def _close_csv():
        if _csv is not None:
            _csv.close()

    _log_diag_csv(0, 0.0, d0, 0.0)

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
        # MPASOceanModel.step has no t_seconds (dm2dc, its only consumer, is
        # arg-gated to tripole/latlon) -- passing it TypeErrors at step 1.
        _ocean_step = (lambda st, sf, fw, t_sec=None:
                       model.step(st, dt, surface_forcing=sf, freshwater=fw))
    else:
        _ocean_step = (lambda st, sf, fw, t_sec=None:
                       model.step(st, dt, surface_forcing=sf, freshwater=fw,
                                  t_seconds=t_sec))
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
        _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
        # t_sec is always None here (tide-enabled fail-fasts above).
        _ocean_step = (lambda st, sf, fw, t_sec=None:
                       _spmd_step(st, dt, surface_forcing=sf, freshwater=fw))
        print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
              f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} rows/band); "
              f"global-in/global-out wrapper (host BCs on gathered state).")

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
    if int(args.scan_block) > 0 and not use_scan:
        why = ("AB2 tracer time integrator (None->Field carry breaks "
               "lax.scan)" if _tti == "ab2"
               else "tidal_forcing enabled (the scan body does not thread the "
                    "model time the tide needs)" if _tide_enabled
               else "grid!=tripole or WOA-nudging / spin-up-drag / SSS-restoring "
                    "enabled (those need per-step host updates)")
        print(f"[scan] --scan-block ignored: {why}.", flush=True)
    if use_scan:
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
            for period in (diag_every, snap_every, steps_per_year):
                if period and period > 0:
                    nb = min(nb, period - (step % period))
            return max(1, nb)

        step = 0
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
                rate = step / (time.time() - t_wall)
                print(f"[diag] step {step} (day {day:.0f}): {d} | "
                      f"{rate:.2f} steps/s", flush=True)
                _log_diag_csv(step, day, d, rate)
                if not d["finite"]:
                    print("[ABORT] non-finite state", flush=True)
                    _save_snapshot(out_dir, f"blowup_step{step}",
                                   state, lat2d, lon2d, io_proc=_is_io_proc())
                    _close_csv()
                    return 1
            if snap_every > 0 and step % snap_every == 0 and step != n_steps:
                _save_snapshot(out_dir, f"day{int(round(day)):04d}",
                               state, lat2d, lon2d, z_coord=z_coord,
                               io_proc=_is_io_proc())
                print(f"[snapshot] day {day:.0f} saved", flush=True)
            if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
                yr = step // steps_per_year
                _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d,
                               z_coord=z_coord, io_proc=_is_io_proc())
                print(f"[snapshot] year {yr} saved", flush=True)
        state = jax.block_until_ready(state)
        _io = _is_io_proc()
        _save_snapshot(out_dir, "final", state, lat2d, lon2d, z_coord=z_coord,
                       io_proc=_io)
        _amoc26n_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _acc_drake_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _save_bsf_amoc_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _mht_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
        _record_final_state_digest(manifest_path, state)
        _close_csv()
        rate = n_steps / (time.time() - t_wall)
        print(f"[done] {n_steps} steps @ {rate:.2f} steps/s (scan); "
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

    for step in range(1, n_steps + 1):
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
        sf = compute_omip2_surface_forcing(
            state, forcing=forcing, idx_t=it,
            grid=grid, grid_type=app_grid_type,
            ice_albedo=_ice_alb,
            under_ice=_under_ice, tau_ice_sw=args.ice_thermo_sw_trans,
            dm2dc_window=_dm2dc_win,
        )
        if ramp < 1.0:
            sf = sf._replace(tau_x=sf.tau_x * ramp, tau_y=sf.tau_y * ramp,
                             q_net=sf.q_net * ramp, sw_down=sf.sw_down * ramp)
        # Surface chlorophyll for THIS step (calendar-month slice); attaching it
        # switches the PE step to NEMO's RGB penetration.  NOT ramped — Chl is a
        # fixed optical climatology, independent of the dynamical spin-up ramp.
        if chl_clim is not None:
            sf = sf._replace(chl=chl_clim[_runoff_month_idx(step, dt)])
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
            # PRE-step concentration = the partition time level (codex r4 #1):
            # step_sea_ice integrates the atmospheric fluxes over its INPUT
            # state, so the ice tile intercepted A_pre of the incident flux
            # this step; giving open water (1 - A_pre) conserves the delivered
            # atmospheric flux exactly (A_pre + (1-A_pre) = 1).  A post-step A
            # would let a melt-to-open cell receive full open-water forcing
            # over the SAME interval whose energy already melted the ice.
            # KNOWN APPROXIMATION (codex r5 #1): on advective ice runs (MPAS
            # transport='advect') step_sea_ice transports concentration BEFORE
            # thermodynamics, so the exact thermo-time area is post-transport;
            # the per-step difference is O(u*dt/dx) ~ 1e-4 in fraction (CFL-
            # limited) and A_pre is EXACT on the latlon/tripole campaign paths
            # (transport='none').  Exposing the post-transport pre-thermo conc
            # would require an ice-model API change — revisit if MPAS ice
            # budgets ever matter at that order.
            _ice_conc_pre = ice_state.concentration.data
            if _ice_conc_pre.ndim > np.asarray(state.land_mask.data).ndim:
                _ice_conc_pre = jnp.sum(_ice_conc_pre, axis=-1)  # multi-cat
            ice_state, ice_resp = step_sea_ice(
                ice_state, atm_ice, sst_K, ocn_u, ocn_v,
                ice_config, U_min=0.0, dt=dt, grid=grid)
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
                    emp=args.emp_freshwater, ramp=ramp)
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
                    emp=args.emp_freshwater, ramp=ramp)
            if ice_resp is not None:
                # ONE shared, mask-aware partition (coupler.ocean_forcing):
                # open-water stress/evap/heat/SW x f_open=(1-A) at the SINGLE
                # documented PRE-step concentration (see _ice_conc_pre above);
                # ice basal heat, brine salt, melt/freeze freshwater, and ice
                # stress added exactly once.  sf was built UNMASKED
                # (ice_albedo=None -> raw SW), so the SW split happens here and
                # only here (raw_core2 mode).  KNOWN SURROGATE (codex r4 #2,
                # pre-existing): the A*tau_ice_sw*sw_down under-ice SW dribble
                # is NOT subtracted from the ice tile's own energy balance (the
                # simple ice model absorbs all non-reflected SW, no penetration
                # channel), a ~tau_ice_sw non-closure of SW over ice —
                # calibratable to 0 via --ice-thermo-sw-trans.
                from legoesm.coupler.ocean_forcing import blend_ice_ocean_forcing
                fw, sf = blend_ice_ocean_forcing(
                    open_sf=sf, open_fw=fw, ice_resp=ice_resp,
                    ice_concentration=_ice_conc_pre,
                    ocean_mask=state.land_mask.data,
                    sw_partition="raw_core2",
                    alpha_ocean=float(_ice_const.alpha_ocean_broadband),
                    sw_transmittance_ice=float(args.ice_thermo_sw_trans),
                )
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
            if step == 1 and fw is not None:                 # [fwbudget] DIAG (temp)
                _Ab = np.asarray(grid.areaCell if hasattr(grid, "areaCell")
                                 else grid.area)
                _wb = np.asarray(state.land_mask.data) > 0.5
                _ig = lambda _x: (float((np.asarray(_x) * _Ab * _wb).sum()) / 1.0e9
                                  if _x is not None else 0.0)   # noqa: E731
                _P, _E, _Rn, _Ic = (_ig(fw.precip), _ig(fw.evap),
                                     _ig(fw.runoff), _ig(fw.ice_fw))
                print(f"[fwbudget] {app_grid_type}: P={_P:+.4f} E={_E:+.4f} "
                      f"R={_Rn:+.4f} ice={_Ic:+.4f} net(P-E+R+ice)="
                      f"{_P - _E + _Rn + _Ic:+.4f} Sv (raw pre-normalize, "
                      f"area-wtd over wet)", flush=True)
            # _ocean_step = single-device model.step (default) OR the lat-band
            # SPMD global-in/global-out step (--n-gpus > 1); both apply the
            # in-core wind-stress / heat / freshwater forcing.  Returns a GLOBAL
            # state, so the host post-step BCs below are unchanged.  t_seconds
            # threads the equilibrium-tide model time (None when tide off; the
            # SPMD path fail-fasts at setup if the tide is enabled).
            state = _ocean_step(state, sf, fw, _t_sec)
        if sss_restore_cfg is not None:
            # NEMO-faithful ice gate (namsbc_ssr nn_sssr_ice=0: no SSS restoring
            # under sea ice).  Feed the SAME prescribed siconc the albedo uses
            # (``_sic``; None only if neither --ice-albedo nor a siconc field is
            # available, in which case the restoring is ungated as before).
            # With --prognostic-sea-ice the prescribed NEMO siconc is no longer
            # the truth — gate the restoring on the LIVE (ocean-masked) prognostic
            # ice concentration instead, via the SAME ``ice_concentration=``
            # parameter (codex MED): prescribed and live ice must not disagree in
            # the salt-restoring path.
            _sss_ice = _sic
            if ice_resp is not None:
                _lc = ice_state.concentration.data
                if _lc.ndim > np.asarray(state.land_mask.data).ndim:
                    _lc = jnp.sum(_lc, axis=-1)
                _sss_ice = _lc * jnp.asarray(state.land_mask.data, _lc.dtype)
            # River-mouth gate (NEMO sbcssr (1-2*rnfmsk)): pass the per-cell
            # runoff so restoring is OFF at river mouths and does not fight
            # the plume toward coarse WOA (Amazon artifact). Gated by flag.
            _R_gate = _R if args.river_mouth_restoring_gate else None
            if app_grid_type == "mpas":
                from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step_mpas
                state = apply_sss_restoring_step_mpas(
                    state, S_target=sss_restore_target, ice_concentration=_sss_ice,
                    config=sss_restore_cfg, mesh=grid, dt=dt,
                    river_runoff=_R_gate)
            else:
                from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step
                state = apply_sss_restoring_step(
                    state, S_target=sss_restore_target, ice_concentration=_sss_ice,
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
            Tn = np.asarray(state.T.data).copy()   # copy: device arrays alias / are read-only
            Tn[..., 0] = under_ice_freeze_relax(
                Tn[..., 0], _sic, dt, tau_ice_days=args.ice_thermo_tau_days)
            state = state._replace(
                T=Field(jnp.asarray(Tn), name=state.T.name,
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
            state = state._replace(**_upd)
        if step % diag_every == 0 or step == n_steps:
            state = jax.block_until_ready(state)
            d = _diag(state, lat2d, lon2d)
            rate = step / (time.time() - t_wall)
            day = step * dt / _SEC_PER_DAY
            print(f"[diag] step {step} (day {day:.0f}): {d} | {rate:.2f} steps/s",
                  flush=True)
            _log_diag_csv(step, day, d, rate)
            if ice_resp is not None:
                _ice_diag = _prognostic_ice_diag(ice_state, ice_resp, grid,
                                                 app_grid_type,
                                                 state.land_mask.data)
                print(f"[ice]  step {step}: {_ice_diag}", flush=True)
            if not d["finite"]:
                print("[ABORT] non-finite state", flush=True)
                _save_snapshot(out_dir, f"blowup_step{step}", state, lat2d, lon2d,
                               io_proc=_is_io_proc())
                _close_csv()
                return 1
        if snap_every > 0 and step % snap_every == 0 and step != n_steps:
            day = step * dt / _SEC_PER_DAY
            _save_snapshot(out_dir, f"day{int(round(day)):04d}", state, lat2d,
                           lon2d, z_coord=z_coord, io_proc=_is_io_proc())
            print(f"[snapshot] day {day:.0f} saved", flush=True)
        if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
            yr = step // steps_per_year
            _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d,
                           z_coord=z_coord, io_proc=_is_io_proc())
            print(f"[snapshot] year {yr} saved", flush=True)

    state = jax.block_until_ready(state)
    _io = _is_io_proc()
    _save_snapshot(out_dir, "final", state, lat2d, lon2d, z_coord=z_coord,
                   io_proc=_io)
    _amoc26n_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _acc_drake_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _save_bsf_amoc_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _mht_diag(state, grid, z_coord, app_grid_type, out_dir, io_proc=_io)
    _record_final_state_digest(manifest_path, state)
    _close_csv()
    rate = n_steps / (time.time() - t_wall)
    print(f"[done] {n_steps} steps @ {rate:.2f} steps/s; final: {_diag(state, lat2d, lon2d)}")
    if args.smoke:
        yr_est = steps_per_year / rate / 3600.0
        print(f"[smoke] projected wall-time: {yr_est:.2f} h/yr  "
              f"({args.years:.0f}yr -> {yr_est*args.years:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
