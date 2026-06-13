"""legoESM-Veros recipe: global_1deg (the STOCK-RESOLUTION global transfer).

``veros.setups.global_1deg.global_1deg.GlobalOneDegreeSetup`` is the stock
Veros global 1-degree, 115-level world ocean (the pyOM2 1x1 cluster-class
config): a UNIFORM 1.0-degree x 1.0-degree grid (no stretching), vertical
thicknesses READ FROM the forcing file (115 levels, 10 m surface cells,
H = 5500.13 m), kbot from the file salinity zero-count + bathymetry land
mask + THREE hardcoded channel closures (Indonesian throughflow strip,
one Aleutian cell, the English Channel), and the SAME monthly forcing
kernel as global_flexible (MIT-grid wind stress, qnet + qnec*(sst-T) heat
feedback, 30-day SSS restoring, simple ice mask, penetrative-shortwave
qsol column) read STRAIGHT onto the grid (no regridding at all).  This
recipe maps every Veros setting onto EXISTING canonical legoESM config
options (SCOPING_three_setups.md SS C table; recipe = pure config selecting
canonical blocks, NO bespoke numerics; zero new model surface beyond the
already-merged EXT-F2 q_solar channel).

Resolution + dt — STOCK, mirrored exactly (this row's point):

  ``NX, NY, NZ = 360, 160, 115``; ``dt_mom = dt_tracer = 1800 s`` (sync) ⇒
  ``dt = 1800``, ``dt_mom_ratio = 1``.  Explicit-AB2 Coriolis margin at the
  most poleward interior centre (79.5 deg): ``|f|·dt_mom = 2Ω·sin(79.5°)
  ·1800 ≈ 0.258`` — comfortably inside the ≈0.5 margin (the global_4deg
  class value), so NO dt deviation is needed; both sides of the comparison
  run the stock pair.  The COST mitigation is entirely on the oracle side
  (Veros JAX-GPU backend; see ``.physics-validator/transfer_global_1deg/``).

Config deltas vs the matched global_4deg recipe (source of truth:
``veros/setups/global_1deg/global_1deg.py`` + veros/settings.py defaults;
scoping SS C — "= global_flexible row-for-row EXCEPT tke_mxl_choice=1"):

  1. **Uniform 1° grid**: ``x_origin = 91``, ``y_origin = -79``, cyclic x
     → ``create_regional_latlon_grid(periodic_x=True)`` with interior
     centres EXACTLY on Veros xt (90.5 .. 449.5) / yt (-79.5 .. 79.5)
     (first interior centre = origin − Δ/2, the 4deg pattern).  NO new
     grid builder (scoping SS 0.5).
  2. **dz from file**: ``vs.dzt = dz_file[::-1]`` (Veros k=0 deepest) ⇒
     legoESM ``dz_ref = dz_file`` DIRECTLY (file is surface-first: 10 m at
     index 0, 83.3 m at index 114 — verified on the real asset).  Centres
     via the shared ``veros_u_centered_z_centres`` (Veros dzw metric) with
     the interleaving degeneracy guard.
  3. **kbot**: the 4deg-class salt-count rule — ``kbot = 1 + Σ_z(salt==0)``
     (file salinity in Veros z-order), zeroed where ``bathymetry == 0``,
     zeroed where ``kbot ≥ nz`` (only-surface-cell column), THEN the three
     channel closures transcribed verbatim from set_topography
     (global_1deg.py:143-154, 0-based interior indices):
       - Indonesian strip:   ``(207 ≤ i < 214) & (j < 5)``
       - Aleutian cell:      ``(i == 104) & (j == 134)``
       - English Channel:    ``(269 ≤ i < 271) & (j == 130)``
     (No marginal-sea morphology and no ETOPO processing here — the file
     bathymetry is already on-grid.)
  4. GM/Redi: ``K_iso_steep = 50`` (4deg: 1000), ``iso_dslope = iso_slopec
     = 0.005`` → ``S_max = 5e-3``, ``taper_width_frac = 1.0`` (4deg:
     1e-3/4.0) — the global_flexible values.
  5. **EKE FLIP BACK: ``isopycnal_diffusion = True``** (the setup sets
     ``enable_eke_isopycnal_diffusion = True``, like ACC/global_flexible
     and unlike global_4deg's settings-default False) ⇒ K_iso follows the
     prognostic GM coefficient.
  6. **TKE: ``tke_mxl_choice = 1``** — THE one TKE delta vs the
     4deg/flexible block (``TKEConfig.tke_mxl_choice`` dispatches it in
     ``vertical_mixing/tke.py``; everything else — c_k=0.1, c_eps=0.7,
     alpha_tke=30, mxl_min=1e-8, kappaM_min=2e-4, kappaH_min=2e-5, kappaH
     profile, superbee advection — is the verbatim 4deg block).

     ``tke_mxl_choice=1`` is the DEBT-SAFE Veros distance-to-boundary
     limiter (``veros/core/tke.py:43-47``): the buoyancy mixing length is
     capped by ``min(-zw + dzw/2, ht + zw)`` — the distance to the surface
     and to the seafloor — before the ``mxl_min`` floor. legoESM imports
     this faithfully via :func:`veros_mxl_choice1_boundary_cap`, wired in by
     the K-profile orchestrator from the static interface geometry + the
     per-column ``H_bathy`` (= Veros ``ht``). The cap is what makes the path
     debt-safe under the post-mixing surface correction: without it the raw
     length ``sqrt(2e)/sqrt(max(1e-12,N²))`` overflows to ``+inf`` where
     ``N²→0`` at a convecting surface and drove the realized
     implicit-friction increment NaN (the MEASURED south-Pacific
     step-2 blowup, since fixed). The recipe runs at the faithful
     ``mxl_choice=1`` with NO stability fallback — a true like-for-like
     against the Veros global_1deg oracle (the only setup using choice 1).
  7. Lateral viscosity ``A_h = 5e4`` (setup literal, not the degtom³
     formula), cos¹(lat) scaling.
  8. **Penetrative shortwave** (the merged EXT-F2 channel):
     ``FluxFeedbackConfig.penetrative_shortwave=True`` with
     ``shortwave_water_type="I"`` (≡ the setup literals R=0.58, ζ1=0.35,
     ζ2=23.0).  Heat-ownership contract: the harness passes
     ``q_prescribed = qnet − qsol`` (non-solar remainder) + ``q_solar =
     qsol``; legoESM's I(0)=1 full-column deposit is cell-by-cell identical
     to Veros's pen(0)=0 zero-column-sum redistribution on the
     solar-inclusive qnet.  Ice gating of the 3-D solar source is inside
     the scheme.
  9. Wind stress on the MIT grid: the kernel's one-cell x/y shift
     (:func:`veros_mit_tau_shift_1deg`), applied once at data-prep time.
 10. Everything else — eos="veros_gsw", rigid lid + faithful AB2 stack,
     zero bottom drag, cp_0 kernel literal 3991.86795711963, 30-day SSS
     restore, ice mask, 360-day forcing year — is IDENTICAL to the matched
     global_4deg recipe and is imported/reused from it (THE RULE: factored,
     not copied).

``enable_eke_diss_surfbot=True/0.2`` and the idemix flags in the setup are
INERT with ``enable_idemix=False`` (4deg-verified) — unmapped.  The file's
``tidal_energy``/``wind_energy`` fields are idemix-only — unread.

DEDUP NOTE: :func:`veros_mit_tau_shift_1deg`, the shape-generic layout
bridges and :func:`veros_area_t_1deg` are the same constructions as the
NOT-YET-MERGED global_flexible recipe carries (``/tmp`` worktree at the
time of writing); whichever lands second must factor them into ONE shared
fidelity helper module (this build stands alone on origin/main).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.grids.latlon import LatLonGrid, create_regional_latlon_grid
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG

# Shared canonical blocks from the matched recipes (factored, not copied):
# the ACCRecipe container, the global_4deg TKE/EKE/GM blocks this setup
# repeats, the Veros kernel cp_0 literal, the 30-day restore, the
# 360-day-year month weights and the u_centered vertical-centre
# construction.
from legoesm.ocean.fidelity.veros_acc_recipe import ACCRecipe
from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
    GLOBAL4_EKE_CONFIG,
    GLOBAL4_GM_REDI_CONFIG,
    GLOBAL4_TKE_CONFIG,
    T_REST_S,
    VEROS_GLOBAL4_CP0,
    get_periodic_interval_weights,
)
from legoesm.ocean.fidelity.veros_state_bridge import veros_u_centered_z_centres
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import (
    FluxFeedbackConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.state import LatLonCGridOceanConfig, LatLonCGridOceanState
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    create_partial_cell_coordinate,
)

__all__ = (
    "DT_MOM_RATIO",
    "DT_S",
    "DXT_DEG",
    "DYT_DEG",
    "GLOBAL_1DEG_A_H",
    "GLOBAL_1DEG_EKE_CONFIG",
    "GLOBAL_1DEG_GM_REDI_CONFIG",
    "GLOBAL_1DEG_TKE_CONFIG",
    "NX", "NY", "NZ",
    "X_ORIGIN_DEG",
    "Y_ORIGIN_DEG",
    "build_global_1deg_grid",
    "build_global_1deg_model_config",
    "build_global_1deg_physics_config",
    "build_global_1deg_recipe",
    "build_global_1deg_state",
    "build_global_1deg_z_coord",
    "get_periodic_interval_weights",
    "kbot_to_mask_and_h_bathy_1deg",
    "replicate_veros_kbot_1deg",
    "veros_area_t_1deg",
    "veros_full_axes_1deg",
    "veros_mit_tau_shift_1deg",
    "veros_xy_to_legoesm_1deg",
    "veros_xyz_to_legoesm_1deg",
)


# ---------------------------------------------------------------------------
# Constants pulled verbatim from veros/setups/global_1deg/global_1deg.py
# ---------------------------------------------------------------------------

NX = 360
NY = 160
NZ = 115

DXT_DEG = 1.0                 # set_grid: dxt = 1.0
DYT_DEG = 1.0                 # set_grid: dyt = 1.0
X_ORIGIN_DEG = 91.0
Y_ORIGIN_DEG = -79.0

# STOCK synchronous stepping — mirrored exactly on both sides (module doc).
DT_S = 1800.0                 # dt_mom = dt_tracer
DT_MOM_RATIO = 1.0

GLOBAL_1DEG_A_H = 5.0e4       # setup literal A_h [m**2/s]

# The three set_topography channel closures (global_1deg.py:143-154),
# 0-based INTERIOR (i=x, j=y) half-open index ranges — transcribed verbatim
# and mutation-pinned by the unit tests.
CHANNEL_CLOSURES = (
    # (i_lo, i_hi, j_lo, j_hi) with i in [i_lo, i_hi), j in [j_lo, j_hi)
    (207, 214, 0, 5),         # Indonesian throughflow strip
    (104, 105, 134, 135),     # Aleutian islands cell
    (269, 271, 130, 131),     # English Channel
)


# ---------------------------------------------------------------------------
# Grid axes (uniform calc_grid transcription) / builders
# ---------------------------------------------------------------------------


def veros_full_axes_1deg(nx: int = NX, ny: int = NY) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Veros ``calc_grid`` horizontal axes INCLUDING the 2 ghost rows each
    side: ``(xt, yt, yu)`` with shapes ``(nx+4,)``/``(ny+4,)``.

    Both axes are UNIFORM here, so the u_centered recursion collapses to
    the closed form: centres at ``origin − Δ/2 + Δ·(idx − 2)`` (``xu[2] =
    x_origin`` / ``yu[2] = y_origin`` shifts), with the CYCLIC x ghost
    overwrite (values copied from the far side).  Anchored against the
    live oracle dump by the harness gate."""
    xt = X_ORIGIN_DEG - DXT_DEG / 2.0 + DXT_DEG * (np.arange(nx + 4) - 2.0)
    xt[-2:] = xt[2:4]
    xt[:2] = xt[-4:-2]
    yt = Y_ORIGIN_DEG - DYT_DEG / 2.0 + DYT_DEG * (np.arange(ny + 4) - 2.0)
    yu = Y_ORIGIN_DEG + DYT_DEG * (np.arange(ny + 4) - 2.0)
    return xt, yt, yu


def build_global_1deg_grid(ny: int = NY, nx: int = NX) -> LatLonGrid:
    """360x160 uniform 1-degree global grid whose interior centres land
    EXACTLY on Veros xt (90.5..449.5, cyclic) / yt (-79.5..79.5).
    ``create_regional_latlon_grid`` adds N/S wall rows (n_lat=162; rows
    0/161 are marked land by the mask builders — the 4deg convention)."""
    lon_west = X_ORIGIN_DEG - DXT_DEG / 2.0        # 90.5 = first centre
    lon_east = lon_west + nx * DXT_DEG             # 450.5
    lat_south = Y_ORIGIN_DEG - DYT_DEG             # -80 → interior -79.5..79.5
    lat_north = lat_south + ny * DYT_DEG           # +80
    grid, _wall_mask = create_regional_latlon_grid(
        n_lat=ny, n_lon=nx,
        lat_south=lat_south, lat_north=lat_north,
        lon_west=lon_west, lon_east=lon_east,
        radius=VEROS_CONSTANTS_CONFIG.R_earth,
        omega=VEROS_CONSTANTS_CONFIG.Omega,        # full 2Ω·sin(lat) Coriolis
        periodic_x=True,                            # enable_cyclic_x
    )
    return grid


def build_global_1deg_z_coord(dz_file: np.ndarray) -> OceanZStarCoordinate:
    """115-level z-coordinate from the forcing-file ``dz`` (SURFACE-FIRST in
    the file: 10 m at index 0 — Veros stores ``vs.dzt = dz_file[::-1]``, so
    legoESM ``dz_ref = dz_file`` directly).  Centres/dz_half via Veros's
    ``u_centered_grid`` recursion (the shared bridge construction — every
    d/dz, slope and implicit solve sees the oracle's dzw), with the
    interleaving degeneracy guard (the flexible-recipe precedent)."""
    dz_ref_np = np.asarray(dz_file, dtype=np.float64)
    if dz_ref_np.ndim != 1 or dz_ref_np.size != NZ:
        raise ValueError(
            f"global_1deg dz must be a ({NZ},) profile, got {dz_ref_np.shape}")
    if np.any(dz_ref_np <= 0):
        raise ValueError("global_1deg dz must be strictly positive")
    if not dz_ref_np[0] < dz_ref_np[-1]:
        raise ValueError(
            "global_1deg dz must be SURFACE-FIRST (thin cells at index 0; "
            "the file convention — Veros reverses it into vs.dzt)")
    z_full_np = veros_u_centered_z_centres(dz_ref_np)
    iface = -np.concatenate([[0.0], np.cumsum(dz_ref_np)])    # 0 .. -H
    if not (np.all(z_full_np < iface[:-1]) and np.all(z_full_np > iface[1:])):
        raise ValueError(
            "global_1deg z grid is degenerate: the Veros u_centered_grid "
            "centre recursion leaves a cell centre outside its cell")
    dz_ref = jnp.asarray(dz_ref_np, dtype=jnp.float64)
    z_half_ref = jnp.concatenate([
        jnp.zeros(1, dtype=dz_ref.dtype), -jnp.cumsum(dz_ref)])
    z_full_ref = jnp.asarray(z_full_np, dtype=dz_ref.dtype)
    dz_half_ref = jnp.abs(z_full_ref[:-1] - z_full_ref[1:])
    return OceanZStarCoordinate(
        n_levels=NZ, H_max=float(np.sum(dz_ref_np)),
        z_full_ref=z_full_ref, z_half_ref=z_half_ref,
        dz_ref=dz_ref, dz_half_ref=dz_half_ref,
    )


# ---------------------------------------------------------------------------
# Topography / kbot (set_topography verbatim; pure given arrays)
# ---------------------------------------------------------------------------


def replicate_veros_kbot_1deg(bathymetry_xy: np.ndarray,
                              salt_xyz: np.ndarray) -> np.ndarray:
    """Veros global_1deg ``set_topography`` verbatim (global_1deg.py:125-154).

    Parameters: ``bathymetry_xy`` (x, y) — the file bathymetry ON the model
    grid (0 = land); ``salt_xyz`` (x, y, z) in VEROS z-order (k=0 deepest —
    the file array after ``.T`` + ``[:, :, ::-1]``; 0 = land sentinel).
    Returns 1-based interior ``kbot`` (x, y); 0 = land.  Order-faithful:

    1. ``kbot = 1 + Σ_z (salt == 0)``  (the salt zero-count rule);
    2. ``kbot = 0`` where ``bathymetry == 0``;
    3. ``kbot = kbot · (kbot < nz)``  (only-surface-cell columns are land);
    4. the THREE channel closures (:data:`CHANNEL_CLOSURES`, transcribed
       index-for-index: Indonesian i∈[207,214) j<5; Aleutian (104,134);
       English Channel i∈[269,271) j=130).

    Bit-target: the live oracle's ``vs.kbot`` (asserted by the harness —
    the established 4deg gate)."""
    if salt_xyz.shape != (bathymetry_xy.shape[0], bathymetry_xy.shape[1], NZ):
        raise ValueError(
            f"salt_xyz shape {salt_xyz.shape} does not match bathymetry "
            f"{bathymetry_xy.shape} + nz={NZ}")
    mask_salt = salt_xyz == 0.0
    kbot = 1 + mask_salt.astype(np.int64).sum(axis=2)
    kbot = np.where(bathymetry_xy == 0.0, 0, kbot)
    kbot = kbot * (kbot < NZ)
    i, j = np.indices(bathymetry_xy.shape)
    closed = np.zeros(bathymetry_xy.shape, dtype=bool)
    for (i_lo, i_hi, j_lo, j_hi) in CHANNEL_CLOSURES:
        closed |= (i >= i_lo) & (i < i_hi) & (j >= j_lo) & (j < j_hi)
    return np.where(closed, 0, kbot)


def kbot_to_mask_and_h_bathy_1deg(
    kbot_xy: np.ndarray,
    z_ref: OceanZStarCoordinate,
) -> tuple[np.ndarray, np.ndarray]:
    """Veros kbot → legoESM ``(land_mask, H_bathy)`` on the (n_lat, n_lon)
    grid including the two wall rows, with H_bathy SNAPPED to interface
    depths ``|z_half_ref[n_wet]|`` (``n_wet = nz − kbot + 1``) so the
    partial-cell coordinate degenerates to FULL cells — the 4deg pattern on
    the file vertical.

    The interfaces are taken from ``z_ref.z_half_ref`` ITSELF (not an
    independent ``np.cumsum``): the file dz values are not round numbers,
    and ``create_partial_cell_coordinate`` counts interfaces strictly
    shallower than H against that exact array — a re-computed cumsum can
    differ in the last ulp and spawn ~1e-13 m sliver cells (measured:
    57k columns on the synthetic world)."""
    iface_depth = np.abs(np.asarray(z_ref.z_half_ref, dtype=np.float64))
    n_wet = np.where(kbot_xy > 0, NZ - kbot_xy + 1, 0)          # (x, y)
    H_xy = iface_depth[n_wet]
    mask_xy = (n_wet > 0).astype(np.float64)
    nx, ny = kbot_xy.shape
    land_mask = np.zeros((ny + 2, nx))
    land_mask[1:-1, :] = mask_xy.T
    H_bathy = np.zeros((ny + 2, nx))
    H_bathy[1:-1, :] = H_xy.T
    return land_mask, H_bathy


# ---------------------------------------------------------------------------
# Layout bridges + forcing prep helpers (pure; see module DEDUP NOTE)
# ---------------------------------------------------------------------------


def veros_xyz_to_legoesm_1deg(arr_xyz: np.ndarray,
                              fill: float = 0.0) -> np.ndarray:
    """(x, y, z) VEROS z-order (k=0 deepest) → legoESM (lat, lon, z) with
    k=0 SURFACE, plus the two wall rows (shape-generic)."""
    nx, ny, nz = arr_xyz.shape
    out = np.full((ny + 2, nx, nz), fill, dtype=np.float64)
    out[1:-1, :, :] = np.transpose(arr_xyz, (1, 0, 2))[:, :, ::-1]
    return out


def veros_xy_to_legoesm_1deg(arr_xy: np.ndarray,
                             fill: float = 0.0) -> np.ndarray:
    """(x, y) → legoESM (lat, lon) with wall rows (2-D forcing fields)."""
    nx, ny = arr_xy.shape
    out = np.full((ny + 2, nx), fill, dtype=np.float64)
    out[1:-1, :] = arr_xy.T
    return out


def veros_mit_tau_shift_1deg(taux_xym: np.ndarray,
                             tauy_xym: np.ndarray,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """The Veros set_forcing_kernel MIT-grid index shift
    (global_1deg.py:309-310): ``surface_taux[i] = taux[i+1]`` (one cell in
    x) and ``surface_tauy[j] = tauy[j+1]`` (one cell in y), applied ONCE at
    data-prep time on the INTERIOR monthly stacks (x, y, 12).

    Ghost semantics (MEASURED on the live oracle dump, t=0 realized
    ``surface_taux``): Veros NEVER fills the ghost cells of the custom
    ``taux``/``tauy`` variables (they are assigned only at ``[2:-2, 2:-2]``
    and ``enforce_boundaries`` is never called on them), so BOTH shifts
    pull a ZERO ghost into the last interior column/row — the easternmost
    ``surface_taux`` column and the northernmost ``surface_tauy`` row are
    exactly 0 in the oracle.  Replicated, not smoothed (and NOT a cyclic
    roll — the roll assumption fails the harness gate by 0.2 N/m²).  The
    harness gate asserts this against ``surface_taux0``/``surface_tauy0``
    bit-targets."""
    taux_shift = np.concatenate(
        [taux_xym[1:, :, :], np.zeros_like(taux_xym[:1, :, :])], axis=0)
    tauy_shift = np.concatenate(
        [tauy_xym[:, 1:, :], np.zeros_like(tauy_xym[:, :1, :])], axis=1)
    return taux_shift, tauy_shift


def veros_area_t_1deg(
    yt_deg: np.ndarray,
    r_earth: float = VEROS_CONSTANTS_CONFIG.R_earth,
) -> np.ndarray:
    """Veros T-cell area column weights ``dxt·dyt·cost`` [m**2] for the
    uniform 1-degree grid (per-latitude row; broadcast over x)."""
    degtom = r_earth * np.pi / 180.0
    return (DXT_DEG * degtom) * (DYT_DEG * degtom) * np.cos(
        np.deg2rad(np.asarray(yt_deg)))


# ---------------------------------------------------------------------------
# Physics configs (scoping SS C; deltas vs global_4deg documented per line)
# ---------------------------------------------------------------------------

# EKE: identical parameter block to global_4deg/ACC with the FLIP BACK:
# enable_eke_isopycnal_diffusion=True in THIS setup (global_1deg.py:79,
# like ACC/global_flexible; the 4deg's False was the settings default) ⇒
# K_iso = K_gm (the Redi tracer diffusivity follows the prognostic GM
# coefficient).
GLOBAL_1DEG_EKE_CONFIG = GLOBAL4_EKE_CONFIG._replace(
    isopycnal_diffusion=True,     # *** the flip back vs global_4deg ***
)

# GM/Redi: K_iso_0=1000, K_iso_steep=50, iso_dslope=iso_slopec=0.005 ⇒
# S_max=5e-3, taper_width_frac=1.0 (established mapping S_max=iso_slopec,
# frac=iso_dslope/iso_slopec) — the global_flexible values.
GLOBAL_1DEG_GM_REDI_CONFIG = GLOBAL4_GM_REDI_CONFIG._replace(
    S_max=5.0e-3,                 # Veros iso_slopec   (4deg: 1e-3)
    taper_width_frac=1.0,         # iso_dslope/iso_slopec (4deg: 4.0)
    K_iso_steep=50.0,             # Veros K_iso_steep  (4deg: 1000)
    eke=GLOBAL_1DEG_EKE_CONFIG,
)

# TKE: the setup repeats the global_4deg block (c_k=0.1, c_eps=0.7,
# alpha_tke=30, mxl_min=1e-8, kappaM_min=2e-4, kappaH_min=2e-5, kappaH
# profile, superbee advection, r_bot=0) with EXACTLY ONE delta:
#
#   *** tke_mxl_choice = 1 ***  (global_1deg.py:64; 4deg/flexible: 2)
#
# — the simple Blanke-Delecluse-class mixing length (implemented and
# dispatched in vertical_mixing/tke.py; raises on unknown values).
GLOBAL_1DEG_TKE_CONFIG = GLOBAL4_TKE_CONFIG._replace(
    tke_mxl_choice=1,
)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def build_global_1deg_state(
    grid: LatLonGrid,
    z_coord,
    land_mask: np.ndarray,
    H_bathy: np.ndarray,
    T_init: np.ndarray | None = None,
    S_init: np.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Initial state: file T/S (legoESM order/shape) masked by the active
    cells, rest velocity, rigid-lid eta ≡ 0, TKE/EKE carry fields seeded
    (the 4deg seeding convention — see its docstring)."""
    nz = z_coord.n_levels
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        S_uniform=35.0, H_max=float(z_coord.H_max),
        land_mask_override=jnp.asarray(land_mask),
        H_bathy_override=jnp.asarray(H_bathy),
    )
    is_active = jnp.asarray(z_coord.is_active, dtype=state.T.data.dtype)
    if T_init is not None:
        state = state._replace(
            T=state.T.replace(data=jnp.asarray(T_init) * is_active))
    if S_init is not None:
        state = state._replace(
            S=state.S.replace(data=jnp.asarray(S_init) * is_active))

    lm = state.land_mask.data
    dtype = state.T.data.dtype
    wet3 = (lm[:, :, jnp.newaxis] > 0.5) * jnp.ones((1, 1, nz - 1), dtype=dtype)

    tke0 = GLOBAL_1DEG_TKE_CONFIG.tke_background * wet3
    state = state._replace(
        tke=Field(data=tke0, name="tke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"),
        dtke=Field(data=jnp.zeros_like(tke0), name="dtke",
                   dims=("lat", "lon", "level"), units="m^2/s^3"),
    )
    eke0 = GLOBAL_1DEG_EKE_CONFIG.e_min * wet3
    state = state._replace(
        eke=Field(data=eke0, name="eke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"),
        eke_diss=Field(data=jnp.zeros_like(eke0), name="eke_diss",
                       dims=("lat", "lon", "level"), units="m^2/s^3"),
    )
    return state


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def build_global_1deg_physics_config() -> OceanPhysicsConfig:
    """global_1deg physics: prognostic TKE (superbee advection, mxl_choice=1,
    Richardson Prandtl) + flux_feedback surface forcing WITH the
    penetrative-shortwave q_solar channel.  GM/Redi rides the top-level
    dynamics config; IDEMIX off."""
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke", tke=GLOBAL_1DEG_TKE_CONFIG),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(
            scheme="flux_feedback",
            flux_feedback=FluxFeedbackConfig(
                # Veros kernel cp_0 hardcode (global_1deg.py:303) — the same
                # literal as the 4deg/flexible kernels, NOT constants.c_sw.
                c_sw=VEROS_GLOBAL4_CP0,
                rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
                tau_restore_s=T_REST_S,              # t_rest = 30 d
                ice_mask=True,
                # qsol penetrative column: Jerlov "I" ≡ the setup literals
                # rpart=0.58, efold1=0.35, efold2=23.0 (global_1deg.py:182-184).
                penetrative_shortwave=True,
                shortwave_water_type="I",
            ),
        ),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,    # solar is OWNED by flux_feedback here
    )


def build_global_1deg_model_config() -> LatLonCGridOceanConfig:
    """global_1deg dynamics config — the matched-4deg faithful dycore stack
    (same Veros core ⇒ same options) with the setup's parameter deltas
    (A_h literal, GM/Redi block, STOCK synchronous dt)."""
    return LatLonCGridOceanConfig(
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=GLOBAL_1DEG_A_H,                        # setup literal 5e4
        A_h_lat_scaling=True,                       # cos¹(lat) scaling
        A_h_cos_power=1,
        lateral_viscosity_operator="flux_divergence",
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        vertical_momentum_scheme="centered_full",
        tracer_advection="centered",
        bottom_drag_r=0.0,                          # Veros r_bot default 0
        eos="veros_gsw",                            # eq_of_state_type = 5
        implicit_vertical_mixing=True,
        K_v=0.0,
        gm_redi=GLOBAL_1DEG_GM_REDI_CONFIG,
        surface_forcing_implicit=True,              # Veros source placement
        outer_integrator="ab2",
        ab2_scope="advective",
        barotropic_solver="rigid_lid",
        dt_mom_ratio=DT_MOM_RATIO,                  # 1 — STOCK sync stepping
        momentum_friction_additive=True,
        coriolis_scheme="explicit_ab2",             # |f|·dt_mom ≈ 0.26 @79.5°
        physics=build_global_1deg_physics_config(),
    )


def build_global_1deg_recipe(
    dz_file: np.ndarray,
    bathymetry_xy: np.ndarray,
    salt_xyz: np.ndarray,
    temp_xyz: np.ndarray | None = None,
) -> ACCRecipe:
    """One-stop constructor from raw on-grid file arrays (the harness owns
    the netCDF reads; unit tests fabricate FULL-SIZE synthetic arrays —
    the closure indices are absolute, so the shapes are not negotiable).

    Parameters: ``dz_file`` (nz,) SURFACE-FIRST file thicknesses;
    ``bathymetry_xy`` (x, y); ``salt_xyz``/``temp_xyz`` (x, y, z) in VEROS
    z-order (the file array after ``.T`` + ``[:, :, ::-1]``).

    Returns the shared ``ACCRecipe`` container; ``wind_forcing`` is None —
    the monthly forcing is composed per-step by the harness
    (``scripts/validate/ocean_fidelity/run_global_1deg_freerun.py``)."""
    grid = build_global_1deg_grid()
    z_ref = build_global_1deg_z_coord(dz_file)
    kbot = replicate_veros_kbot_1deg(bathymetry_xy, salt_xyz)
    land_mask, H_bathy = kbot_to_mask_and_h_bathy_1deg(kbot, z_ref)
    z_coord = create_partial_cell_coordinate(z_ref, jnp.asarray(H_bathy))
    T_init = (veros_xyz_to_legoesm_1deg(temp_xyz)
              if temp_xyz is not None else None)
    S_init = veros_xyz_to_legoesm_1deg(salt_xyz)
    initial_state = build_global_1deg_state(
        grid, z_coord, land_mask, H_bathy, T_init=T_init, S_init=S_init)
    return ACCRecipe(
        model_config=build_global_1deg_model_config(),
        physics_config=build_global_1deg_physics_config(),
        grid=grid,
        z_coord=z_coord,
        land_mask=jnp.asarray(land_mask),
        initial_state=initial_state,
        wind_forcing=None,
    )
