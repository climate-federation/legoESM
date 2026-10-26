"""legoESM-Veros recipe: global_4deg (the GLOBAL transfer test).

``veros.setups.global_4deg.global_4deg.GlobalFourDegreeSetup`` is the stock
Veros global 4-degree, 15-level world ocean (open Indonesian throughflow),
driven by monthly climatological forcing (Trenberth wind stress, ECMWF heat
flux + SST feedback, SSS restoring, simple ice mask).  This recipe maps every
Veros setting onto EXISTING canonical legoESM config options — the same
oracle-recipe discipline as ``veros_acc_basic_recipe`` (recipe = pure config
selecting canonical blocks; NO bespoke numerics).

Config deltas vs the matched ACC recipe (source of truth:
``veros/setups/global_4deg/global_4deg.py`` + ``veros/settings.py`` defaults;
see ``.physics-validator/transfer_global_4deg/SCOPING.md`` §E):

  1. GM/Redi: ``K_iso_0 = K_iso_steep = 1000`` (ACC: 1000/500);
     ``iso_slopec = 1e-3`` → ``S_max = 1e-3`` and ``iso_dslope = 4e-3`` →
     ``taper_width_frac = iso_dslope/iso_slopec = 4.0`` (ACC: 0.01/0.5).
     The established mapping (dm95 tanh arg ``(S_max−S)/(frac·S_max)`` ≡ the
     Veros form) holds for frac > 1.
  2. **EKE FLIP (loud!): ``isopycnal_diffusion = False``.**  The Veros setup
     does NOT set ``enable_eke_isopycnal_diffusion`` and the settings.py
     default is **False** (veros/settings.py:129) — so unlike the ACC
     (which sets it True), the Redi tracer diffusivity stays the CONSTANT
     ``K_iso_0 = 1000``; only the GM skew coefficient is prognostic-EKE
     driven.  Verified gating: ``ocean_model_latlon_cgrid.py:1703``.
  3. Lateral viscosity ``A_h = (4·degtom)³·2e-11 ≈ 1.76e6 m²/s`` (4° cell vs
     ACC's 2°), same cos¹(lat) scaling (``enable_hor_friction_cos_scaling``,
     ``hor_friction_cosPower = 1``).
  4. **NO bottom friction**: the setup does not set ``r_bot`` and the Veros
     default is ``r_bot = 0.0`` with ``enable_bottom_friction = False``
     (veros/settings.py:50,75) → ``bottom_drag_r = 0.0`` and the TKE
     ``source_bottom_drag_diss = False`` (``K_diss_bot ≡ 0``).  Do NOT reuse
     the ACC's faithful ``R_BOT ≈ 2.76e-3``.
  5. Asynchronous stepping ``dt_mom = 1800`` / ``dt_tracer = 86400`` →
     ``dt = 86400`` with ``dt_mom_ratio = 48`` (ACC ratio: 9).
  6. EOS: ``eq_of_state_type = 5`` → ``eos = "veros_gsw"`` (TEOS-10 gsw port;
     ACC: ``veros_nonlin2``).
  7. TKE superbee advection: ``enable_tke_superbee_advection = True`` →
     ``TKEConfig.advection_scheme = "superbee"`` (absent in the ACC setup).
  8. Surface forcing: the ``flux_feedback`` scheme (Veros set_forcing_kernel
     heat/salt block: prescribed qnet + qnec·(sst−T) feedback, SSS restoring
     at 30 d, simple ice mask) with the Veros kernel's hardcoded
     ``cp_0 = 3991.86795711963`` (NOT ``constants.c_sw = 3994``; 0.05%
     mismatch documented in ``FluxFeedbackConfig``) and ``rho_0 = 1024``
     (= ``settings.rho_0`` default = ``VEROS_CONSTANTS_CONFIG.rho_0``) — ONE
     consistent rho threaded from the same config block the dycore reads.
  9. Grid/bathymetry: 90×40×15 global lat-lon (centres ON Veros xt/yt:
     lon 2..358, lat −78..78, cyclic-x), 15-level ``ddz`` (Veros stores
     ``dzt = ddz[::-1]``, k=0 deepest; legoESM ``dz_ref = ddz`` directly,
     k=0 surface = 50 m), Veros kbot replicated from the forcing-file
     bathymetry+salinity and SNAPPED to interface depths so the
     ``OceanPartialCellCoordinate`` degenerates to FULL cells exactly
     (bit-identical maskT, probe-verified level-by-level — SCOPING §B).
 10. Coriolis: full ``2Ω·sin(lat)`` (Veros set_coriolis) — automatic from the
     lat-lon grid with ``Omega`` pinned to Veros.

Faithful dycore stack — DOCUMENTED CHOICE: global_4deg runs the SAME Veros
core as the matched ACC baseline, so the recipe BAKES IN the ACC's
Veros-faithful driver kwargs (instead of leaving them to ``--flags`` like the
ACC freerun driver): ``outer_integrator="ab2"``, ``barotropic_solver=
"rigid_lid"``, ``coriolis_scheme="explicit_ab2"``, ``ab2_scope="advective"``,
``momentum_friction_additive=True``, ``vertical_momentum_scheme=
"centered_full"``, ``momentum_advection="flux_form"`` (centered),
``tracer_advection="centered"``, ``lateral_viscosity_operator=
"flux_divergence"``, ``implicit_vertical_mixing=True``, ``K_v=0``,
``surface_forcing_implicit=True`` (Veros thermodynamics source placement).
Explicit-AB2 Coriolis stability: ``|f|·dt_mom = 2Ω·sin(78°)·1800 ≈ 0.26``
at the most poleward wet row — well inside the ≈0.5 margin (the global
domain is SAFE here only because dt_mom = 1800 s; see
``check_coriolis_stability``).

Data-prep helpers (pure functions, no I/O — the harness does the netCDF
reads): Veros kbot replication, the legoESM mask/H_bathy mapping, the
(x,y,z)→(lat,lon,z) IC bridge, sentinel masking, the annual-mean qnet
imbalance removal, and the 360-day-year ``get_periodic_interval`` weights
(jnp-traceable for in-scan use).  All probe-PROVEN bit-identical to the live
Veros oracle (SCOPING §B/§C) — lifted from
``.physics-validator/transfer_global_4deg/probe_{kbot_mask,forcing_data}.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import LatLonGrid, create_regional_latlon_grid
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.fidelity.veros_global_common import (
    build_veros_global_state,
    veros_area_t_generic,
)
from legoesm.ocean.fidelity.veros_stepping import veros_faithful_stepping
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig, LateralMixingConfig,
)
from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
from legoesm.ocean.physics.surface_forcing.config import (
    FluxFeedbackConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.state import LatLonCGridOceanConfig, LatLonCGridOceanState
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    OceanZStarCoordinate,
    create_partial_cell_coordinate,
)

# Shared canonical blocks reused from the matched ACC recipe (THE RULE:
# factored, not copied).  ACC_TKE_CONFIG carries the Veros TKE knobs that
# global_4deg sets to the IDENTICAL values (c_k=0.1, c_eps=0.7, alpha_tke=30,
# mxl_min=1e-8, tke_mxl_choice=2, kappaM_min=2e-4, kappaH_min=2e-5,
# enable_kappaH_profile=True) plus the matched-ACC faithful options
# (prognostic carried TKE, adiabatic N², Richardson Prandtl
# [enable_Prandtl_tke default True], source_eke_diss=True).  The ACCRecipe
# NamedTuple is reused as the recipe container.
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACC_TKE_CONFIG,
    ACCRecipe,
)
from legoesm.ocean.fidelity.veros_state_bridge import veros_u_centered_z_centres


# ---------------------------------------------------------------------------
# Constants pulled verbatim from veros/setups/global_4deg/global_4deg.py
# ---------------------------------------------------------------------------

NX = 90
NY = 40
NZ = 15

DXT_DEG = 4.0
DYT_DEG = 4.0
X_ORIGIN_DEG = 4.0
Y_ORIGIN_DEG = -76.0

DT_MOM_S = 1800.0
DT_TRACER_S = 86400.0
DT_MOM_RATIO = DT_TRACER_S / DT_MOM_S    # 48.0 — the asynchronous ratio

# Veros vertical thickness profile ``ddz`` (global_4deg.py:109-112).  Veros
# stores ``vs.dzt = ddz[::-1]`` (k=0 deepest = 690 m); legoESM is k=0 surface,
# so ``dz_ref = ddz`` DIRECTLY (surface cell = 50 m).  SCOPING §B probe
# verified the per-level wet counts level-by-level with this orientation.
GLOBAL4_DDZ = np.array([
    50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0,
    390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0,
])
H_MAX = float(GLOBAL4_DDZ.sum())          # 5200 m

# Veros set_forcing_kernel literals (global_4deg.py:250,259).  cp_0 is a
# SETUP-KERNEL hardcode (≠ settings nor legoesm.constants.c_sw=3994; the
# 0.05% mismatch is documented in FluxFeedbackConfig) — pinned here as the
# recipe's oracle value, exactly like VEROS_CONSTANTS_CONFIG pins rho_0.
VEROS_GLOBAL4_CP0 = 3991.86795711963      # [J/(kg K)]
T_REST_S = 30.0 * 86400.0                 # SSS restoring timescale [s]

# Veros 360-day forcing year (set_forcing_kernel: year_in_seconds = 360*86400).
YEAR_S = 360.0 * 86400.0
N_MONTHS = 12
MONTH_S = YEAR_S / N_MONTHS

# qnec/qnet land sentinel in the forcing files (global_4deg.py:179,183).
FORCING_SENTINEL = -1.0e10


def global_4deg_A_h(r_earth: float = VEROS_CONSTANTS_CONFIG.R_earth) -> float:
    """Veros: ``A_h = (4 * degtom) ** 3 * 2e-11`` ≈ 1.76e6 m²/s (the 4° analog
    of ``acc_A_h``; ``degtom = R_earth·π/180`` is a unit conversion derived
    from the configured radius, not a literal)."""
    degtom = r_earth * float(np.pi) / 180.0
    return (DXT_DEG * degtom) ** 3 * 2.0e-11


# ---------------------------------------------------------------------------
# Physics configs (each value verified against global_4deg.py — see tests)
# ---------------------------------------------------------------------------

# Prognostic EKE — IDENTICAL parameter block to the ACC recipe (the setup sets
# the same eke_* values: eke_k_max=1e4, eke_c_k=0.4, eke_c_eps=0.5,
# eke_cross=2.0, eke_crhin=1.0, eke_lmin=100, superbee advection, 3-D field)
# with ONE LOUD FLIP:
#
#   *** isopycnal_diffusion = False ***  (ACC: True)
#
# ``enable_eke_isopycnal_diffusion`` is ABSENT from the global_4deg setup and
# the Veros settings.py DEFAULT is False (settings.py:129) ⇒ K_iso stays the
# CONSTANT K_iso_0 = 1000 (eke.py "always constant" K_iso branch); only the
# GM skew coefficient is the prognostic c_k·eke_len·√E.
#
# The matched-ACC EKE source options carry over unchanged (same Veros core,
# same forc = K_diss_h − P_diss_skew assembly): source_kdiss_h +
# kdiss_h_flux_form (Veros K_diss_h for the flux-divergence viscosity),
# gm_source_mode="realized_signed" (the literal signed −P_diss_skew),
# n2_mode="adiabatic" (the Veros static stability for eke_len/Eady).
# ``enable_eke_diss_surfbot=True/0.2`` in the setup is INERT with IDEMIX off
# (the redistribution lives only in integrate_idemix; SCOPING §E re-verified)
# ⇒ the EKE dissipation feeds TKE directly = TKE source_eke_diss=True below.
GLOBAL4_EKE_CONFIG = EKEConfig(
    c_k=0.4,                      # Veros eke_c_k
    c_eps=0.5,                    # Veros eke_c_eps
    l_min=100.0,                  # Veros eke_lmin
    kappa_gm_max=1.0e4,           # Veros eke_k_max
    advection_scheme="superbee",  # Veros enable_eke_superbee_advection=True
    mixing_length_scheme="rhines",
    eke_cross=2.0,                # Veros eke_cross
    eke_crhin=1.0,                # Veros eke_crhin
    isopycnal_diffusion=False,    # *** THE FLIP vs ACC — see block comment ***
    eke_3d=True,
    alpha_eke=1.0,                # Veros default (setup leaves it)
    source_kdiss_h=True,
    kdiss_h_flux_form=True,
    gm_source_mode="realized_signed",
    source_p_diss_iso=False,      # gated off (ACC verdict; see eke.py docs)
    n2_mode="adiabatic",
    n2_over_dzw=True,             # Veros dzw slot for the adiabatic-N² divisor
    #                               (deferred EKE-side twin of veros_dz_slots)
)

# GM/Redi: enable_neutral_diffusion + enable_skew_diffusion with
# K_iso_0 = K_iso_steep = 1000, iso_dslope = 4e-3, iso_slopec = 1e-3.
# Established mapping: S_max = iso_slopec, taper_width_frac =
# iso_dslope/iso_slopec (dm95 tanh arg (S_max−S)/(frac·S_max) ≡ Veros form;
# frac = 4.0 > 1 is fine).  Neutral-density slopes + implicit K_33 as ACC.
GLOBAL4_GM_REDI_CONFIG = GMRediConfig(
    kappa_GM=1000.0,              # EKE-off fallback (prognostic eke drives GM)
    kappa_Redi=1000.0,            # Veros K_iso_0 — CONSTANT here (flip #2!)
    S_max=1.0e-3,                 # Veros iso_slopec      (ACC: 0.01)
    taper_width_frac=4.0,         # iso_dslope/iso_slopec (ACC: 0.5)
    implicit_K33=True,
    K_iso_steep=1000.0,           # Veros K_iso_steep     (ACC: 500)
    slope_density="neutral",
    veros_triad_weights=True,     # Veros dzw(pair)/(4 dzt) triad weights, no
    #                               boundary renormalization (see GMRediConfig).
    double_redi_diagonal=True,    # Veros adds K_11/K_22 in BOTH the iso and
    #                               skew passes (diffusion.py:40-47 +
    #                               thermodynamics.py:430-437) — the oracle's
    #                               2× horizontal diagonal (see GMRediConfig).
    eke=GLOBAL4_EKE_CONFIG,
)

# TKE: every shared knob equals the ACC value (the setup repeats the same
# numbers) — reuse the ACC block and change exactly the two deltas:
#   advection_scheme="superbee"  — enable_tke_superbee_advection=True (ACC:
#     absent ⇒ "none").  W-grid superbee + AB2 tendency history (state.dtke).
#   source_bottom_drag_diss=False — r_bot = 0 here (Veros settings default;
#     the setup never enables bottom friction) ⇒ K_diss_bot ≡ 0; the ACC
#     recycled a REAL bottom-drag dissipation, which does not exist here.
GLOBAL4_TKE_CONFIG = ACC_TKE_CONFIG._replace(
    advection_scheme="superbee",
    source_bottom_drag_diss=False,
)


# ---------------------------------------------------------------------------
# Grid / vertical coordinate
# ---------------------------------------------------------------------------


def build_global_4deg_grid() -> LatLonGrid:
    """90×40 4° global grid whose interior centres land EXACTLY on Veros
    xt (2..358, cyclic) / yt (−78..78) — probe-verified (SCOPING §B).
    ``create_regional_latlon_grid`` adds N/S wall rows (n_lat=42; rows 0/41
    are marked land by the mask builders, the ACC-recipe convention)."""
    # Veros u-grid origin convention (same as ACC): first interior T centre at
    # x_origin - dx/2 = 2.0 ⇒ centres 2, 6, ..., 358 == Veros xt[2:-2].
    lon_west = X_ORIGIN_DEG - DXT_DEG / 2.0        # 2.0
    lon_east = lon_west + NX * DXT_DEG             # 362
    lat_south = Y_ORIGIN_DEG - DYT_DEG             # -80 → interior -78..78
    lat_north = lat_south + NY * DYT_DEG           # +80
    grid, _wall_mask = create_regional_latlon_grid(
        n_lat=NY, n_lon=NX,
        lat_south=lat_south, lat_north=lat_north,
        lon_west=lon_west, lon_east=lon_east,
        radius=VEROS_CONSTANTS_CONFIG.R_earth,
        omega=VEROS_CONSTANTS_CONFIG.Omega,        # full 2Ω·sin(lat) Coriolis
        periodic_x=True,                            # Veros enable_cyclic_x
    )
    return grid


def build_global_4deg_z_coord() -> OceanZStarCoordinate:
    """15-level reference z-coordinate from the Veros ``ddz`` (k=0 surface =
    50 m in legoESM ordering; Veros stores the reversed ``ddz[::-1]``).

    Cell CENTRES (and hence ``dz_half_ref``, the vertical-gradient /
    implicit-solve metric) use Veros's ``u_centered_grid`` recursion — NOT
    midpoints — so every ∂/∂z, isoneutral slope and K_33 sees the oracle's
    ``dzw`` (see :func:`..veros_state_bridge.veros_u_centered_z_centres`;
    the same construction ``veros_zt_centres`` already uses for kbot).
    Interfaces (``z_half_ref``) are identical in both conventions."""
    dz_ref = jnp.asarray(GLOBAL4_DDZ, dtype=jnp.float64)
    z_half_ref = jnp.concatenate([
        jnp.zeros(1, dtype=dz_ref.dtype), -jnp.cumsum(dz_ref),
    ])
    z_full_ref = jnp.asarray(
        veros_u_centered_z_centres(np.asarray(GLOBAL4_DDZ)),
        dtype=dz_ref.dtype)
    dz_half_ref = jnp.abs(z_full_ref[:-1] - z_full_ref[1:])
    return OceanZStarCoordinate(
        n_levels=NZ, H_max=H_MAX,
        z_full_ref=z_full_ref, z_half_ref=z_half_ref,
        dz_ref=dz_ref, dz_half_ref=dz_half_ref,
    )


# ---------------------------------------------------------------------------
# Bathymetry ingestion (pure data-prep; probe-proven bit-identical to Veros)
# ---------------------------------------------------------------------------


def veros_zt_centres() -> np.ndarray:
    """Veros computed cell-centre depths ``zt`` (k=0 DEEPEST, negative) from
    ``dzt = ddz[::-1]`` — the array set_topography compares the bathymetry
    against.  NOT the forcing-file ``zt`` variable (surface-first), and NOT
    plain midpoints: Veros builds the vertical with ``u_centered_grid``
    (core/numerics.py:9-21 via :74-76), the REFLECTED recursion
    ``zt[k] = 2·zw[k-1] − zt[k-1]`` with ``zw_raw[k] = Σ_{i≥1} dzt[i]``,
    shifted so the top interface is 0.  Using midpoints instead shifts zt by
    up to 25 m and mis-buckets 56 columns' kbot — the u_centered form
    reproduces the oracle kbot histogram bit-identically
    ([1281, 569, 563, 407, 247, 120, 70, 46, 40, 41, 33, 36, 36, 40, 71, 0])."""
    # Shared canonical construction (also feeds build_global_4deg_z_coord's
    # z_full_ref/dz_half_ref): top-down output, flipped here to Veros order.
    return veros_u_centered_z_centres(np.asarray(GLOBAL4_DDZ))[::-1]


def replicate_veros_kbot(bathymetry_xy: np.ndarray,
                         salt_xyz: np.ndarray) -> np.ndarray:
    """Veros global_4deg ``set_topography`` verbatim (global_4deg.py:124-137).

    Parameters: ``bathymetry_xy`` (x, y) [m, negative depths]; ``salt_xyz``
    (x, y, z) in VEROS z-order (k=0 deepest — i.e. the file array after
    ``.T`` + ``[:, :, ::-1]``, 0 = land sentinel).
    Returns 1-based ``kbot`` (x, y); 0 = all-land column.
    Probe-verified bit-identical to the live Veros oracle (SCOPING §B).
    """
    zt = veros_zt_centres()
    land = (zt[None, None, :] <= bathymetry_xy[..., None]) | (salt_xyz == 0.0)
    kbot = 1 + land.astype(int).sum(axis=2)
    all_land = (bathymetry_xy == 0) | (kbot == NZ)
    return np.where(all_land, 0, kbot)


def kbot_to_mask_and_h_bathy(kbot_xy: np.ndarray,
                             n_lat_grid: int = NY + 2,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Map Veros kbot → legoESM ``(land_mask, H_bathy)`` on the (n_lat, n_lon)
    grid INCLUDING the two wall rows (rows 0 / -1 land, H=0).

    H_bathy is SNAPPED exactly to the interface depth ``Σ dz_ref[:n_wet]``
    (n_wet = NZ − kbot + 1) so ``create_partial_cell_coordinate`` degenerates
    to FULL cells — bit-identical wet geometry to Veros's full-cell maskT
    (probe-verified per level).  Land columns get H_bathy = 0 (the natural
    partial-cell choice; bottom_level = −1)."""
    iface_depth = np.concatenate([[0.0], np.cumsum(GLOBAL4_DDZ)])   # (NZ+1,)
    n_wet = np.where(kbot_xy > 0, NZ - kbot_xy + 1, 0)              # (x, y)
    H_xy = iface_depth[n_wet]
    mask_xy = (n_wet > 0).astype(np.float64)
    # (x, y) → (lat, lon) + wall rows.
    n_extra = n_lat_grid - NY
    if n_extra != 2:
        raise ValueError(
            f"expected a grid with 2 wall rows (n_lat = {NY + 2}), "
            f"got n_lat = {n_lat_grid}"
        )
    land_mask = np.zeros((n_lat_grid, NX))
    land_mask[1:-1, :] = mask_xy.T
    H_bathy = np.zeros((n_lat_grid, NX))
    H_bathy[1:-1, :] = H_xy.T
    return land_mask, H_bathy


def veros_xyz_to_legoesm(arr_xyz: np.ndarray,
                         n_lat_grid: int = NY + 2,
                         fill: float = 0.0) -> np.ndarray:
    """(x, y, z) VEROS z-order (k=0 deepest) → legoESM (lat, lon, z) with k=0
    SURFACE, padded with the two wall rows (``fill``).  The IC/forcing bridge
    (same convention handling as the ACC bridge; SCOPING §C)."""
    out = np.full((n_lat_grid, NX, arr_xyz.shape[2]), fill, dtype=np.float64)
    out[1:-1, :, :] = np.transpose(arr_xyz, (1, 0, 2))[:, :, ::-1]
    return out


def veros_xy_to_legoesm(arr_xy: np.ndarray,
                        n_lat_grid: int = NY + 2,
                        fill: float = 0.0) -> np.ndarray:
    """(x, y) → legoESM (lat, lon) with wall rows (2-D forcing fields)."""
    out = np.full((n_lat_grid, NX), fill, dtype=np.float64)
    out[1:-1, :] = arr_xy.T
    return out


# ---------------------------------------------------------------------------
# Forcing data prep (pure; the harness owns the netCDF I/O)
# ---------------------------------------------------------------------------


def mask_forcing_sentinel(arr: np.ndarray) -> np.ndarray:
    """Veros: ``where(x <= -1e10, 0, x)`` (qnec + qnet sentinel masking)."""
    return np.where(arr <= FORCING_SENTINEL, 0.0, arr)


def remove_qnet_imbalance(qnet_xy12: np.ndarray,
                          area_xy: np.ndarray,
                          wet_surface_xy: np.ndarray,
                          ) -> tuple[np.ndarray, float]:
    """Veros set_initial_conditions annual-mean heat-flux imbalance removal
    (global_4deg.py:185-189), ORDER-FAITHFUL: the mean is taken over ALL
    interior cells (land included), THEN the adjusted flux is masked by the
    wet surface.  Returns ``(qnet_adjusted, mean_flux)``; on the real data
    ``mean_flux == 7.860702e-02 W/m²`` (matches the Veros log to all printed
    digits — probe-verified, asserted by the harness)."""
    mean_flux = float(
        (qnet_xy12 * area_xy[:, :, None]).sum() / N_MONTHS / area_xy.sum()
    )
    qnet_adj = (qnet_xy12 - mean_flux) * wet_surface_xy[:, :, None]
    return qnet_adj, mean_flux


def veros_area_t(yt_deg: np.ndarray,
                 r_earth: float = VEROS_CONSTANTS_CONFIG.R_earth,
                 ) -> np.ndarray:
    """Veros T-cell area column weights ``dxt·dyt·cost`` [m²] for the 4° grid
    (per-latitude; broadcast over x).  ``degtom = R_earth·π/180``."""
    return veros_area_t_generic(yt_deg, DXT_DEG, DYT_DEG, r_earth)


def get_periodic_interval_weights(time_s):
    """Veros ``tools.get_periodic_interval`` for the 360-day monthly cycle
    (veros/tools/setup.py:88-123), jnp-traceable so the harness evaluates it
    INSIDE the jitted scan from the traced model time.

    Returns ``(n1, f1, n2, f2)`` with records anchored at month START
    (t=0 → 100% January; SCOPING §C verified)."""
    t = jnp.asarray(time_s) % YEAR_S
    n1 = jnp.asarray(t // MONTH_S, dtype=jnp.int32)
    n2 = (1 + n1) % N_MONTHS
    f2 = (t - MONTH_S * n1.astype(t.dtype)) / MONTH_S
    f1 = 1.0 - f2
    return n1, f1, n2, f2


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def build_global_4deg_state(
    grid: LatLonGrid,
    z_coord: OceanPartialCellCoordinate,
    land_mask: np.ndarray,
    H_bathy: np.ndarray,
    T_init: np.ndarray | None = None,
    S_init: np.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Initial state: file T/S (already bridged to legoESM order/shape via
    :func:`veros_xyz_to_legoesm`) masked by the active cells, rest velocity,
    rigid-lid eta ≡ 0; TKE/EKE carry fields seeded.

    Seeding note (Veros parity): Veros zero-initialises ``vs.tke``/``vs.eke``
    (no setup seed).  legoESM seeds tke at ``tke_background = 1e-6`` and eke
    at ``e_min = 1e-8`` — the model floors both there anyway on the first
    step, so this is the same effective start (and matches the matched-ACC
    convention).  ``eke_diss`` and the TKE-advection AB2 history ``dtke``
    seed at zero (Veros's zero-initialised eke_diss_iw / dtke[taum1]),
    keeping the ``lax.scan`` carry pytree constant.
    """
    return build_veros_global_state(
        grid, z_coord, land_mask, H_bathy,
        tke_config=GLOBAL4_TKE_CONFIG,
        eke_config=GLOBAL4_EKE_CONFIG,
        T_init=T_init, S_init=S_init,
    )


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def build_global_4deg_physics_config() -> OceanPhysicsConfig:
    """global_4deg physics: TKE (prognostic, Richardson-Prandtl, superbee
    advection) + flux_feedback surface forcing.  GM/Redi rides the top-level
    dynamics config (as in the ACC recipes); IDEMIX off (enable_idemix=False;
    the idemix_* setup flags are inert)."""
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke", tke=GLOBAL4_TKE_CONFIG),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(
            scheme="flux_feedback",
            flux_feedback=FluxFeedbackConfig(
                # Veros kernel cp_0 hardcode — NOT constants.c_sw (0.05% off).
                c_sw=VEROS_GLOBAL4_CP0,
                # ONE consistent rho: the same Veros settings.rho_0 = 1024 the
                # dycore reads via VEROS_CONSTANTS_CONFIG.
                rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
                tau_restore_s=T_REST_S,
                ice_mask=True,
            ),
        ),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def build_global_4deg_model_config() -> LatLonCGridOceanConfig:
    """global_4deg dynamics config — the matched-ACC faithful dycore stack
    (same Veros core ⇒ same options; see the module docstring 'DOCUMENTED
    CHOICE') with the global_4deg parameter deltas (A_h, GM/Redi, dt ratio,
    gsw EOS, zero bottom drag)."""
    return LatLonCGridOceanConfig.from_flat(
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=global_4deg_A_h(VEROS_CONSTANTS_CONFIG.R_earth),
        A_h_lat_scaling=True,                       # cos¹(lat) scaling
        A_h_cos_power=1,
        lateral_viscosity_operator="flux_divergence",
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        vertical_momentum_scheme="centered_full",
        tracer_advection="centered",
        # NO bottom friction: Veros r_bot default 0.0 (settings.py:75),
        # enable_bottom_friction default False — the setup never enables it.
        bottom_drag_r=0.0,
        eos="veros_gsw",                            # eq_of_state_type = 5
        implicit_vertical_mixing=True,
        K_v=0.0,
        gm_redi=GLOBAL4_GM_REDI_CONFIG,
        surface_forcing_implicit=True,              # Veros source placement
        # ---- Veros-faithful time stepping (matched-ACC stack, baked in) ----
        # Shared bundle via veros_stepping.veros_faithful_stepping (#433);
        # dt_mom_ratio=DT_MOM_RATIO=48 (dt arg IS dt_tracer; |f|·dt_mom ≈ 0.26 @78°).
        **veros_faithful_stepping(with_surface_forcing=True,
                                  dt_mom_ratio=DT_MOM_RATIO),
        physics=build_global_4deg_physics_config(),
    )


def build_global_4deg_recipe(
    bathymetry_xy: np.ndarray,
    salt_xyz: np.ndarray,
    temp_xyz: np.ndarray | None = None,
) -> ACCRecipe:
    """One-stop constructor from raw Veros-order data arrays (the harness
    reads them from the netCDF; unit tests fabricate tiny ones).

    Parameters: ``bathymetry_xy`` (x, y); ``salt_xyz``/``temp_xyz`` (x, y, z)
    in VEROS z-order (file array after ``.T`` + ``[:, :, ::-1]``).

    Returns the shared ``ACCRecipe`` container.  ``wind_forcing`` is None —
    the monthly forcing is composed per-step by the harness
    (``scripts/ocean_fidelity/run_global_4deg_freerun.py``)."""
    grid = build_global_4deg_grid()
    z_ref = build_global_4deg_z_coord()
    kbot = replicate_veros_kbot(bathymetry_xy, salt_xyz)
    land_mask, H_bathy = kbot_to_mask_and_h_bathy(kbot, grid.n_lat)
    z_coord = create_partial_cell_coordinate(z_ref, jnp.asarray(H_bathy))
    T_init = (veros_xyz_to_legoesm(temp_xyz, grid.n_lat)
              if temp_xyz is not None else None)
    S_init = veros_xyz_to_legoesm(salt_xyz, grid.n_lat)
    initial_state = build_global_4deg_state(
        grid, z_coord, land_mask, H_bathy, T_init=T_init, S_init=S_init)
    return ACCRecipe(
        model_config=build_global_4deg_model_config(),
        physics_config=build_global_4deg_physics_config(),
        grid=grid,
        z_coord=z_coord,
        land_mask=jnp.asarray(land_mask),
        initial_state=initial_state,
        wind_forcing=None,
    )


__all__ = (
    "DT_MOM_RATIO",
    "DT_MOM_S",
    "DT_TRACER_S",
    "FORCING_SENTINEL",
    "GLOBAL4_DDZ",
    "GLOBAL4_EKE_CONFIG",
    "GLOBAL4_GM_REDI_CONFIG",
    "GLOBAL4_TKE_CONFIG",
    "H_MAX",
    "MONTH_S",
    "N_MONTHS",
    "NX", "NY", "NZ",
    "T_REST_S",
    "VEROS_GLOBAL4_CP0",
    "X_ORIGIN_DEG",
    "Y_ORIGIN_DEG",
    "YEAR_S",
    "build_global_4deg_grid",
    "build_global_4deg_model_config",
    "build_global_4deg_physics_config",
    "build_global_4deg_recipe",
    "build_global_4deg_state",
    "build_global_4deg_z_coord",
    "get_periodic_interval_weights",
    "global_4deg_A_h",
    "kbot_to_mask_and_h_bathy",
    "mask_forcing_sentinel",
    "remove_qnet_imbalance",
    "replicate_veros_kbot",
    "veros_area_t",
    "veros_xy_to_legoesm",
    "veros_xyz_to_legoesm",
    "veros_zt_centres",
)
