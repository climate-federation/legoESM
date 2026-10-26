"""legoESM-Veros recipe: ACC channel (Phase G.0c).

Builds the lat-lon C-grid equivalent of ``veros.setups.acc.acc.ACCSetup``:

- 30 × 42 × 15 grid at 2° horizontal resolution, ~1900 m deep
- Stretched z-coordinate (276 m at depth → 20 m at surface)
- ``y_origin = -40°``, ``x_origin = 0°``; lon wraps at 60°, lat to +44°
- Single-column western wall north of -20° latitude — re-entrant
  channel in the south, closed sub-tropical basin in the north
- Linear T(z) initial 0 (bottom) → 15 °C (surface), S = 35 uniform
- Surface wind stress: sinusoidal in the southern band, 1 - cos in
  the northern band
- T*  surface restoring (30-day timescale), zero in the tropics
- Veros canonical TKE + GM/Redi + linear bottom friction + implicit
  vertical viscosity, cos(lat) lateral viscosity

This module returns the legoESM analog with every scheme + parameter
pinned to the Veros source. All physical constants (g, rho_0, Omega,
R_earth, c_sw, and the derived A_h) are pinned through config
(``ConstantsConfig`` / ``VEROS_CONSTANTS_CONFIG``), so no monkey-patch
of legoESM's constants module is needed (G-C4).

Source of truth: ``veros/setups/acc/acc.py`` in the
team-ocean/veros repository.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.latlon import LatLonGrid, create_regional_latlon_grid
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.fidelity.veros_stepping import veros_faithful_stepping
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig, LateralMixingConfig,
)
from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig, GeometricConfig
from legoesm.core.field import Field
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.state import (
    LatLonCGridOceanConfig, LatLonCGridOceanState, OceanSurfaceForcing,
)
from legoesm.ocean.vertical import OceanZStarCoordinate


# ---------------------------------------------------------------------------
# Constants pulled verbatim from veros/setups/acc/acc.py
# ---------------------------------------------------------------------------

NX = 30
NY = 42
NZ = 15

DXT_DEG = 2.0
DYT_DEG = 2.0
X_ORIGIN_DEG = 0.0
Y_ORIGIN_DEG = -40.0

# Vertical layer-thickness profile (Veros: ddz[::-1] / 2.5).
# ddz published in Veros source:
_ACC_DDZ = np.array([
    50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0,
    390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0,
])
ACC_DZT = _ACC_DDZ[::-1] / 2.5   # shape (15,), dzt[0] = deepest = 276 m, dzt[-1] = surface = 20 m

DT_MOM_S = 4800.0
DT_TRACER_S = 43200.0       # = 86400 / 2

# Coriolis ω: Veros sets settings.omega — we leave that to legoESM's
# canonical constants.Omega, which the recipe-loader will override to
# 7.292115e-5 (Veros value) via the constants-override context manager.

# Lateral viscosity: A_h = (2 · degtom)^3 · 2e-11, with degtom = R_earth · π / 180.
def _degtom(r_earth: float = constants.R_earth) -> float:
    return r_earth * float(np.pi) / 180.0

def acc_A_h(r_earth: float = constants.R_earth) -> float:
    """Veros: ``A_h = (2 * degtom) ** 3 * 2e-11``. Units: m^4/s² × m = m²/s
    after the cos²(lat) scaling and ∇^4 stencil; matches MOM6 / MITgcm
    biharmonic-equivalent harmonic-viscosity convention used in ACC.

    ``r_earth`` defaults to legoESM's; build_acc_model_config passes the Veros
    value so the grid/A_h are pinned via config (G-C4)."""
    return (2.0 * _degtom(r_earth)) ** 3 * 2.0e-11

# Bottom drag. Veros applies a linear drag as a RATE on the bottom cell only:
# du/dt = -r_bot·u with r_bot = 1e-5 1/s and NO division by the cell thickness
# (veros/core/friction.py:284-289). legoESM's linear bottom drag is the stress form
# du/dt = -r·u/h_bot (stress velocity-scale ÷ bottom-cell thickness), so to reproduce
# Veros's drag RATE the coefficient must be r = r_bot · h_bot. The ACC has a flat
# bottom (deepest = thickest z-star layer, 276 m) ⇒ h_bot = max(ACC_DZT).
#
# DISSIPATION AUDIT (2026-05-29): the prior R_BOT=1e-5 mis-mapped Veros's r_bot (copied
# the numeric value without the units/÷h_bot conversion), making legoESM's bottom drag
# ~h_bot≈276× TOO WEAK (~320-day vs Veros's ~28 h timescale). r = r_bot·h_bot ≈ 2.76e-3
# is the FAITHFUL mapping (legoESM's bottom-cell drag RATE then equals Veros's).
#
# IMPORTANT (1-year re-verify): the faithful drag does NOT reconcile the ACC transport —
# it OVER-damps it. @30d it looked like a clean fix (KE +233%→+27%, transport +70%→−22%),
# but @1yr the faithful drag gives transport −83% (18 vs 110 Sv) / KE −29%, whereas the
# mis-mapped r=1e-5 gave +58% / +443%. So legoESM's barotropic transport OVER-RESPONDS to
# bottom drag vs Veros (same bottom-cell rate → 18 Sv legoESM vs 110 Sv Veros): the
# genuine deeper difference is the BAROTROPIC FORMULATION (legoESM split-explicit free
# surface vs Veros rigid-lid/streamfunction `solve_stream.py`). The old r=1e-5 masked it
# via COMPENSATING errors (under-drag ≈ cancelled the over-responsive barotropic). We keep
# the FAITHFUL drag here (match Veros's scheme, not tune to the transport metric); the
# transport residual is the documented barotropic-formulation delta. See strategy §8.
_VEROS_R_BOT = 1.0e-5                       # Veros r_bot [1/s] — bottom-cell drag RATE
R_BOT = _VEROS_R_BOT * float(max(ACC_DZT))  # legoESM bottom_drag_r [m/s] ≈ 2.76e-3 (faithful)

# Veros TKE knobs (verbatim from ACCSetup)
ACC_TKE_CONFIG = TKEConfig(
    c_k=0.1,
    c_eps=0.7,
    alpha_tke=30.0,
    mxl_min=1.0e-8,
    tke_mxl_choice=2,
    kappaM_min=2.0e-4,
    kappaM_max=100.0,            # Veros default kappaM_max (convective ceiling)
    kappaH_min=2.0e-5,
    enable_kappaH_profile=True,
    # ----- Deep-ocean ventilation fix (opt-in; default config bit-identical) -----
    # Veros computes the static-stability N² by adiabatic parcel displacement to
    # the upper cell's pressure (thermodynamics.py:99-103). legoESM's legacy
    # in-situ N² is biased ~6x too stable (compressibility) and NEVER goes
    # negative -> the abyss never convects -> the +5.3°C warm bias. The adiabatic
    # N² recovers Veros's static instability (gate 1: N²<0 count matches Veros
    # MACHINE-close, 2461 vs 2461 on the 60-day bridged state) so the TKE itself
    # convects: the buoyancy length blows up over unstable columns and K_M
    # saturates toward kappaM_max.
    n2_mode="adiabatic",
    # Veros vertical-metric slots (the TKE metric-consistency fix,
    # .physics-validator/accbasic_regression/): adiabatic N² over dzw (the
    # u_centered dz_half, thermodynamics.py:99), buoyancy-length growth
    # allowance + TKE-diffusion face gradients over dzt with dzw control
    # volumes (tke.py:54-65,185-222), surface injection over 0.5·dzw_top
    # (tke.py:225). Without this the chain mixes midpoint/dzw/dzt slots
    # and — on the u_centered z-coordinate below — equilibrates onto a
    # spurious deep-TKE branch (mean TKE ~1e-2 vs Veros's ~1e-4 class,
    # ACC_Basic KE 0.98→0.77).
    veros_dz_slots=True,
    # Veros TKE positivity (tke.py:224-245): the buoyancy sink P_diss_v =
    # kappaH·N² is EXPLICIT in forc, interior TKE may go NEGATIVE (an
    # energy debt — Veros's 10-yr mean TKE is literally negative,
    # -1.5e-3), and only the surface level is clamped at zero. The legacy
    # per-step background floor erases the interior debt every step — a
    # spurious energy injection that (with the metric slots fixed) was
    # still feeding a ~1e-2 deep TKE reservoir on the u_centered
    # coordinate (.physics-validator/tke_metric_fix/ re-ablation).
    positivity="veros_surface_correction",
    # Veros K-from-TKE amplitude (tke.py:73): kappaM = c_k·mxl·sqrt(max(0,e)).
    # The legacy Gaspar form c_k·l_k·√(2e) double-counts the √2 already
    # inside the Veros buoyancy length (mxl = √2·√e/√N̄) — K_M/K_H/P_s
    # ×1.414 vs the oracle wherever the caps/floors don't bind.
    kappa_convention="veros_sqrte",
    # Veros tracer diffusivity K_H = max(kappaH_min, K_M/Prandtl) with the
    # Richardson-dependent Prandtl number (enable_Prandtl_tke=True, the Veros
    # ACC + global default): Pr = max(1, min(10, 6.6*Ri)). In the stratified
    # interior Pr -> 10 (small abyssal K_H ~ kappaH_min, fixing the ~9x
    # over-diffusion from the legacy K_H = max(K_M, kappaH_min) bug chain where
    # the momentum floor kappaM_min leaked into the tracer floor); in a
    # convecting column Ri < 0 -> Pr -> 1 so K_H tracks the large convective K_M.
    prandtl_mode="richardson",
    Prandtl_tke0=10.0,
    # ----- PROGNOSTIC TKE (Veros enable_tke prognostic form) -----
    # Veros ACC runs enable_tke=True PROGNOSTICALLY: one backward-Euler TKE step
    # per model step with dt_tke = dt_mom (tke.py:137), the TKE field carried
    # across steps (state.tke), seeded at tke_background. legoESM's prior recipe
    # ran the Mode-B quasi-steady DIAGNOSTIC chain (n_iterations=3, dt=86400);
    # prognostic=True switches to the faithful carried-TKE form.
    prognostic=True,
    # Energy-recycling sources (Veros ACC: enable_eke=True, enable_idemix=False,
    # so integrate_tke ``forc = K_diss_v - P_diss_v - P_diss_nonlin + eke_diss_iw
    # + K_diss_bot``). legoESM recycles the two terms it can surface today:
    #   - source_eke_diss: the EKE dissipation rate eke_diss_iw (= c_eps·√E·E/L),
    #     carried from the 3-D EKE step (state.eke_diss) — fed within the same
    #     step (legoESM runs EKE before the TKE solve, matching Veros's
    #     eke→tke ordering), so NO lag in the synchronous path.
    #   - source_bottom_drag_diss: K_diss_bot, the bottom-drag KE extraction
    #     (Veros linear_bottom_friction diss = r_bot·u²), surfaced as the
    #     tendency diagnostic tend.K_diss_bot.
    # DEFERRED (legoESM does not yet surface the diagnostics): P_diss_adv
    # (non-conservative advection) and P_diss_nonlin (cabbeling / non-linear EOS).
    source_eke_diss=True,
    source_bottom_drag_diss=True,
    # ----- Veros step order: TKE charges the POST-MIXING stratification -----
    # Veros solves the TKE budget AFTER the implicit T/S vertical mixing
    # (veros.py:263-285): the kappa profiles consumed by the TRACER solve come
    # from the PREVIOUS step's TKE (set_tke_diffusivities at tau), and the TKE
    # forcing charges P_diss_v = kappaH·Nsqr[taup1] — the stratification the
    # implicit solve has ALREADY stabilised (thermodynamics.py:385) — plus the
    # surface buoyancy-flux P_diss_v slot (386-388). legoESM's legacy ordering
    # charged the PRE-mixing N² (the full instability every step) — the
    # identified dominant ACC_Basic residual after the metric-slot fixes
    # (.physics-validator/tke_metric_fix/).
    buoyancy_timing="post_mixing_veros",
    # Veros K_diss_v is the REALIZED implicit-friction dissipation
    # κ·(∂u_new/∂z)·(∂u_old/∂z) (friction.py:131-151), not the pre-solve
    # parameterised K_M·S² (audited at ~6.1× the realized form).
    shear_production="realized_veros",
)

# Veros GM/Redi knobs (verbatim from ACCSetup)
ACC_GM_REDI_CONFIG = GMRediConfig(
    kappa_GM=1000.0,             # EKE-off fallback (overridden by the prognostic eke below)
    kappa_Redi=1000.0,           # EKE-off fallback (overridden by K_iso=K_gm below)
    # ^ Veros ACC sets enable_eke_isopycnal_diffusion=True ⇒ K_iso = K_gm: the Redi
    #   tracer diffusivity follows the prognostic GM coefficient (veros/core/eke.py:
    #   74-75). Reproduced via EKEConfig.isopycnal_diffusion=True below (the step
    #   passes kappa_redi_override = kappa_gm_override), so kappa_Redi=1000 is only
    #   the EKE-off fallback.
    S_max=0.01,                  # ↔ iso_slopec
    taper_width_frac=0.5,        # = iso_dslope / iso_slopec = 0.005 / 0.01
    implicit_K33=True,           # Veros applies the vertical isoneutral diagonal
    #                              K_33 IMPLICITLY (core/isoneutral/diffusion.py);
    #                              fold it into the implicit tracer solve instead of
    #                              the explicit F_z (the tier-2 dtemp_iso residual).
    K_iso_steep=500.0,           # ↔ Veros K_iso_steep (acc.py:41): horizontal-
    #                              diffusion floor on K_11/K_22 at steep slopes.
    veros_triad_weights=True,    # ↔ Veros dzw(pair)/(4·dzt) triad weights with no
    #                              boundary renormalization (see GMRediConfig doc).
    double_redi_diagonal=True,   # ↔ Veros adds the precomputed K_11/K_22 diagonal
    #                              in BOTH the iso and skew passes (diffusion.py:
    #                              40-47 + thermodynamics.py:430-437; acc.py runs
    #                              enable_neutral_diffusion AND enable_skew_
    #                              diffusion) — the oracle's 2× horizontal
    #                              diagonal (see the GMRediConfig field doc).
    slope_density="neutral",     # ↔ Veros isoneutral.py:40-41: build the slopes
    #                              from the LOCALLY-REFERENCED neutral density
    #                              gradient ∂ρ/∂T·∇T+∂ρ/∂S·∇S (get_drhodT/get_drhodS
    #                              at abs(zt)), NOT the in-situ ∇ρ. Removes the
    #                              adiabatic compressibility bias in ∂_zρ that
    #                              collapsed legoESM's K_33 (and S²) 10-25× interior
    #                              / ~10⁴× near surface vs Veros — the genuine
    #                              T_iso oracle-gap lever (corr 0.17 in-situ).
    # Prognostic EKE (Eden-Greatbatch) with the Rhines-limited mixing length —
    # Veros ACC runs enable_eke=True (veros/setups/acc/acc.py:67-75). The closure
    # FORM + params reproduce Veros's GM coefficient K_gm AND eke_len/L_rossby/
    # L_rhines to machine precision (gates E9 + L4). ACC overrides eke_cross=2.0
    # (acc.py:71); the other EKEConfig defaults match ACC (c_k=0.4, c_eps=0.5,
    # l_min=100, k_max=1e4, superbee advection). NB EKEConfig.k_iso=1000 is the
    # eke-FIELD diffusivity (Veros uses ~max(500,2·K_gm) — a minor approximation,
    # distinct from the tracer K_iso above). With eke set, the prognostic
    # c_k·eke_len·√E replaces the constant kappa_GM above. isopycnal_diffusion=True
    # additionally drives the Redi tracer diffusivity K_iso = K_gm (Veros
    # enable_eke_isopycnal_diffusion=True, acc.py:74) — so BOTH the GM skew and the
    # Redi diffusivity are now the prognostic kappa (apples-to-apples with Veros).
    # eke_3d=True: the 3-D (depth-resolved) prognostic EKE on the interior
    # interfaces (W-grid), matching Veros's 3-D ``vs.eke`` — the GM coefficient
    # kappa_GM(z) and the EKE budget (depth-resolved source/sink + implicit
    # vertical EKE diffusion K=alpha_eke·A_v + per-interface horizontal
    # transport) are all depth-resolved. ``alpha_eke=1.0`` (Veros ACC default).
    #
    # EKE SOURCE = apples-to-apples with Veros's ACC EKE forcing
    # (veros/core/eke.py:110 ``forc = K_diss_gm + K_diss_h - P_diss_skew``; in ACC
    # K_diss_gm=0 + P_diss_hmix=0, so the two live terms are K_diss_h and
    # -P_diss_skew). A budget diagnosis showed legoESM's parameterized source
    # carried only ~16% of Veros's forcing (forc/P ≈ 6×); the two opt-ins close it:
    #   - source_kdiss_h=True: route the mean-KE removed by the harmonic lateral
    #     viscosity A_h into EKE (Veros K_diss_h, ~56% of the ACC forcing — the
    #     dominant missing source).
    #   - kdiss_h_flux_form=True: use the FAITHFUL positive-definite flux form
    #     A_h·(div²+<ζ²>) (Veros's clamp-free A_h|∇u|² analogue for legoESM's
    #     vector-Laplacian viscosity) rather than the clamped dynamical -u·A_h∇²u
    #     (which over-credits the domain-integrated KE dissipation by ~11–20%).
    #   - gm_source_mode="realized_signed": the LITERAL SIGNED Veros conversion
    #     -P_diss_skew = -(g/ρ₀)∇(int_drhodX)·F_skew (the dynamic-enthalpy
    #     dissipation of the GM skew flux), built from the SKEW-only isopycnal flux
    #     and Veros's int_drhodT/S integrands. Replaces the positive-definite
    #     parameterized κ_GM·N²·<S²> (which over-counts by ~22%): probe-measured on
    #     the drop snapshot signed 5.94e10 W vs parameterized 7.29e10 vs Veros 5.97e10.
    #   - source_p_diss_iso=True: SUBTRACT the realized signed Redi APE dissipation
    #     -P_diss_iso (Veros's EKE forc sink, veros/core/eke.py:117), built from the
    #     ISO-only flux + the implicit K_33 vertical diagonal dissipation.
    #   - n2_mode="adiabatic": the Eady/deformation-radius N² by adiabatic parcel
    #     displacement to the upper cell's pressure (Veros eke.py:50-54 via
    #     thermodynamics.py:99-105), the true static stability the Veros EKE chain
    #     uses. The legacy in-situ N² is biased ~6x too stable (compressibility) →
    #     ∫N dz ~2.7x too large → eke_len ~23% too long → EKE dissipation (∝1/L)
    #     too weak (~71% of the runaway-EKE bias; the third in-situ-vs-locally-
    #     referenced instance after convection-N² and neutral slopes). Mirrors the
    #     TKE n2_mode="adiabatic" above.
    eke=EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0,
                  isopycnal_diffusion=True, eke_3d=True,
                  source_kdiss_h=True, kdiss_h_flux_form=True,
                  gm_source_mode="realized_signed", source_p_diss_iso=False,
                  # source_p_diss_iso left OFF (experimental): legoESM's
                  # adiabatic-cancelling triads CANNOT reproduce Veros's
                  # -P_diss_iso sink (an artifact of Veros's non-cancelling
                  # discretization); the faithful dynamic-enthalpy form
                  # yields a small spurious +3.2e9 W source instead. Built,
                  # tested, gated -- see the EKE-budget probe verdict.
                  n2_mode="adiabatic",
                  # Veros dzw slot for the adiabatic-N² divisor (the
                  # deferred EKE-side twin of TKE veros_dz_slots; the
                  # u_centered dz_half_ref IS Veros's dzw here).
                  n2_over_dzw=True),
)

# Surface restoring timescale
T_RESTORING_DAYS = 30.0
_SECONDS_PER_DAY = 86400.0

# Veros ACC surface-forcing profile parameters (verbatim from
# veros/setups/acc/acc.py:126-134). Band edges in degrees latitude.
_ACC_TAUX_AMP = 0.1          # wind-stress amplitude [N/m^2]
_ACC_TSTAR_AMP = 15.0        # T* amplitude [degC]
_ACC_WIND_LAT_S = -20.0      # southern edge: sin westerly band below this
_ACC_WIND_LAT_N = 10.0       # northern edge: (1-cos) band above this
_ACC_TSTAR_LAT_S = -20.0     # T* ramps down south of this
_ACC_TSTAR_LAT_N = 20.0      # T* ramps down north of this
# Veros ACC grid latitude extents INCLUDING its 2 ghost cells each side — the
# forcing formulas reference global_min/max(yt) / (yu). Derived from the recipe
# grid constants and VERIFIED against a live ACCSetup grid (yt in [-45, 45]
# step 2 -> 46 cells = NY+4; yu = yt + dy/2 in [-44, 46]).
_VEROS_YT_MIN = Y_ORIGIN_DEG - 2.5 * DYT_DEG            # -45.0
_VEROS_YT_MAX = _VEROS_YT_MIN + (NY + 3) * DYT_DEG      # +45.0  (NY+4 cells)
_VEROS_YU_MIN = _VEROS_YT_MIN + 0.5 * DYT_DEG           # -44.0
_VEROS_YU_MAX = _VEROS_YT_MAX + 0.5 * DYT_DEG           # +46.0
# Salinity is NOT restored in Veros ACC (T-only heat-flux forcing); a huge
# timescale makes the restoring tendency negligibly small (effectively off).
_ACC_NO_SALT_RESTORE_TAU_S = 1.0e30


# ---------------------------------------------------------------------------
# The Veros <-> legoESM WIRING DIAGRAM (oracle-card audit trail)
# ---------------------------------------------------------------------------
# (Veros option / numeric, legoESM config field, chosen block + faithfulness note).
# Mirrors ``NEMO_BLOCK_MAPPING`` (nemo_recipe.py) so every ocean oracle card carries
# the same auditable map a user reads to see which legoESM block each production
# numeric corresponds to.  The dycore-identity rows (those whose middle column is a
# bare top-level ``LatLonCGridOceanConfig`` scheme field) are MACHINE-CHECKED against
# the ``veros_faithful_v1`` catalog entry by tests/ocean/fidelity/
# test_veros_block_mapping.py -> the card and the catalog cannot silently diverge.
VEROS_BLOCK_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("constants", "ConstantsConfig",
     "VEROS_CONSTANTS_CONFIG (Veros base constants; derived recomputed)"),
    ("EOS", "eos",
     "veros_nonlin2 (Veros 2nd-order nonlinear EOS, linearised-about-state)"),
    ("momentum advection", "momentum_advection",
     "flux_form (Veros 2nd-order centered flux-form momentum)"),
    ("momentum flux reconstruction", "momentum_flux_scheme",
     "centered (Veros 2nd-order)"),
    ("tracer advection", "tracer_advection",
     "centered (Veros 2nd-order centered tracer advection)"),
    ("pressure gradient", "pgf_scheme",
     "adcroft (Veros energy-conserving FD hydrostatic PGF)"),
    ("KE gradient", "ke_gradient_scheme",
     "centered (flux-form path; Veros adds the explicit KE gradient)"),
    ("free surface / barotropic", "barotropic_solver",
     "rigid_lid (Veros streamfunction/rigid-lid external mode)"),
    ("Coriolis", "coriolis_scheme",
     "explicit_ab2 (Veros planetary f x u in the AB2-extrapolated tendency)"),
    ("time integration", "outer_integrator",
     "ab2 (Veros Adams-Bashforth-2 for momentum + tracers)"),
    ("AB2 dissipation scope", "ab2_scope",
     "advective (Veros AB2-extrapolates advection only; mixing stays explicit)"),
    ("tracer time integration", "tracer_time_integrator",
     "euler (Veros forward-Euler tracer step under AB2 advection)"),
    ("momentum friction placement", "momentum_friction_additive",
     "True (Veros adds lateral+vertical friction as a separate additive tendency)"),
    ("implicit vertical mixing", "implicit_vertical_mixing",
     "True (Veros implicit backward-Euler vertical mixing)"),
    ("implicit-vmix dzw slot", "implicit_vmix_dzw_slot",
     "True (Veros dzw spacing in the implicit tridiagonal vertical operator)"),
    ("lateral viscosity operator", "lateral_viscosity_operator",
     "flux_divergence (Veros per-component harmonic friction)"),
    ("vertical momentum advection", "vertical_momentum_scheme",
     "centered_full (Veros centered vertical momentum flux)"),
    ("lateral viscosity coefficient", "A_h",
     "acc_A_h (Veros enable_noslip_lateral=False harmonic A_h)"),
    ("vertical mixing closure", "physics.vertical_mixing",
     "ACC_TKE_CONFIG (Veros prognostic TKE closure)"),
    ("GM/Redi eddy parameterisation", "gm_redi",
     "ACC_GM_REDI_CONFIG (Veros GM + Redi isoneutral, dm95 taper)"),
    ("bottom drag", "bottom_drag_r",
     "R_BOT (Veros linear bottom drag r * |u|)"),
)


# ---------------------------------------------------------------------------
# Grid / vertical coordinate / bathymetry
# ---------------------------------------------------------------------------


def build_acc_grid() -> LatLonGrid:
    """Build legoESM lat-lon C-grid whose interior centres coincide with
    Veros ACC's xt/yt EXACTLY.

    Veros's u-centred grid (verified from ``veros/core/numerics.py`` +
    a live ``ACCSetup``): the first interior centre sits half a cell below
    the origin, so
        xt = [-1, 1, 3, ..., 57]   (= x_origin - dx/2 + i·dx)
        yt = [-41, -39, ..., 41]   (= y_origin - dy/2 + j·dy)
    ``create_regional_latlon_grid`` places interior cell ``i`` (i=1..n) at
    ``lower + (i-0.5)·d`` and adds N/S wall rows at index 0 and -1. To make
    the interior centres land ON Veros's xt/yt — so the analytic kbot lands
    Veros's land cells identically AND the per-process tendency comparison
    is at the same physical points — offset the bounds accordingly. Getting
    this wrong (a half-cell lon offset + a one-row lat offset) was the Phase
    G tier-2 residual: density still matched (point-wise in T,S) but the
    western wall mis-aligned and momentum metrics (Coriolis f(lat)) were off.
    """
    lon_west = X_ORIGIN_DEG - DXT_DEG / 2.0        # -1  -> centres -1,1,...,57
    lon_east = lon_west + NX * DXT_DEG             # 59
    lat_south = Y_ORIGIN_DEG - DYT_DEG             # -42 -> interior -41,...,41
    lat_north = lat_south + NY * DYT_DEG           # +42
    grid, _wall_mask = create_regional_latlon_grid(
        n_lat=NY, n_lon=NX,
        lat_south=lat_south, lat_north=lat_north,
        lon_west=lon_west, lon_east=lon_east,
        # Pin Earth radius / rotation to Veros's values via config (so the grid
        # metrics + Coriolis are correct, pinned purely through config).
        radius=VEROS_CONSTANTS_CONFIG.R_earth,
        omega=VEROS_CONSTANTS_CONFIG.Omega,
        periodic_x=True,
    )
    return grid


def build_acc_z_coord() -> OceanZStarCoordinate:
    """Veros ACC vertical coordinate, built from the exact dzt array
    in Veros's ACC source — ``ddz[::-1] / 2.5`` with ddz published as
    ``[50, 70, ..., 690]``.

    legoESM convention is k=0 surface, k=nlev-1 deepest (opposite of
    Veros which has k=0 deepest). We reverse the Veros dzt array
    when populating the legoESM z_coord so each interior k-index
    refers to the same physical layer in both models.
    """
    # legoESM dz_ref: surface (k=0) → bottom (k=nlev-1) = ddz / 2.5 directly.
    dz_ref = _ACC_DDZ / 2.5   # shape (15,), [20, 28, 40, ..., 276]
    dz_ref = jnp.asarray(dz_ref, dtype=jnp.float64)
    H_max = float(jnp.sum(dz_ref))   # 1900 m exactly

    # z_half_ref: interfaces; surface (k=0) at z=0, bottom (k=nlev) at z=-H_max
    z_half_ref = jnp.concatenate([
        jnp.zeros(1, dtype=dz_ref.dtype),
        -jnp.cumsum(dz_ref),
    ])
    # z_full_ref: cell centres — Veros's u_centered_grid recursion, NOT
    # midpoints (the ACC ddz/2.5 is stretched, so the two differ by up to
    # 10 m and dz_half_ref — the vertical-gradient / implicit-solve metric —
    # alternates around the midpoint value exactly as Veros's dzw does; see
    # veros_u_centered_z_centres).  Interfaces above are identical either way.
    from legoesm.ocean.fidelity.veros_state_bridge import (
        veros_u_centered_z_centres,
    )
    z_full_ref = jnp.asarray(
        veros_u_centered_z_centres(np.asarray(dz_ref)), dtype=dz_ref.dtype)
    # dz_half_ref: distance between adjacent cell centres (== Veros dzw
    # interior, z-flipped)
    dz_half_ref = jnp.abs(z_full_ref[:-1] - z_full_ref[1:])

    return OceanZStarCoordinate(
        n_levels=NZ,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


def build_acc_land_mask(grid: LatLonGrid) -> jnp.ndarray:
    """Veros: ``kbot = (x > 1.0 | y < -20)``.

    In legoESM-speak: a single-column western wall north of -20°. The
    lon=0 column is land at y >= -20 and wet south of y=-20 (so the
    Drake-Passage band is re-entrant).
    """
    lat_deg = np.degrees(np.asarray(grid.lat))  # cell-centre latitudes (n_lat,)
    lon_deg = np.degrees(np.asarray(grid.lon))  # cell-centre longitudes (n_lon,)
    # Wet = (lon > 1.0) OR (lat < -20). With lon[0] = 1.0 + 0 = 1.0 (Veros's
    # cell-centre x for column 0 is x_origin + dx/2 = 1.0), this includes
    # x[0]=1.0 — but the Veros condition is *strict* (x > 1.0), so x=1.0
    # is land. We match that.
    wet_lon = lon_deg > 1.0                   # (n_lon,)
    wet_lat = lat_deg < -20.0                 # (n_lat,)
    wet = wet_lon[None, :] | wet_lat[:, None]  # (n_lat, n_lon)
    # build_acc_grid places the interior centres ON Veros's xt/yt, so the
    # kbot above lands Veros's 2-wide western wall (xt=-1,1 <= 1 -> land for
    # lat>=-20) exactly. create_regional_latlon_grid still adds non-physical
    # N/S wall rows at index 0 and -1 (the bridge zero-fills them -> T=S=0 ->
    # rho ~ 997); they are outside Veros's 42-row domain, so mark them LAND.
    # Leaving any of these mis-handled was the Phase G tier-2 density residual.
    wet[0, :] = False
    wet[-1, :] = False
    return jnp.asarray(wet.astype(np.float64))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def build_acc_state(grid: LatLonGrid,
                     z_coord: OceanZStarCoordinate,
                     *,
                     gm_redi: GMRediConfig = ACC_GM_REDI_CONFIG,
                     tke: TKEConfig = ACC_TKE_CONFIG,
                     ) -> LatLonCGridOceanState:
    """Veros ACC initial conditions:

    - T(z) linear from 0 at bottom to 15 °C at surface
    - S uniform 35 PSU
    - u, v, eta zero
    - Bathymetry depth = full H_max where wet, zero where land

    ``gm_redi`` / ``tke`` default to the ACC configs (so the historical call
    ``build_acc_state(grid, z_coord)`` is bit-identical). The acc_basic transfer
    recipe passes its own configs (EKE off ``gm_redi.eke=None`` -> no EKE-field
    seeding; ``tke.prognostic`` may still be True -> TKE field seeded). Gating the
    seed branches on the PASSED configs (not the module-level ACC constants) is
    the only behavioural change, and it is a no-op when the defaults are used.
    """
    H_max = float(np.sum(ACC_DZT))
    land_mask = build_acc_land_mask(grid)

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=15.0, T_deep=0.0,
        S_uniform=35.0, H_max=H_max,
        land_mask_override=land_mask,
        # Veros ACC IC: temp = (1 - zt/zw[0])*15 (acc.py:117) with BOTTOM-FIRST
        # zw, so zw[0] = the top face of the BOTTOM cell (~-1724 m), NOT the
        # bottom interface -H_max. Gate 4's earlier "linear" transcription
        # normalised by -H_max, leaving the t=0 abyss +2.15 K vs Veros
        # (lego +1.00 C vs Veros -1.15 C) — with the ~3 Sv deep ventilation
        # that IC offset persisted as essentially THE ENTIRE 30-yr abyssal
        # warm bias (2.33 vs 0.98 C). "veros_acc_linear" is the literal
        # transcription; the matched-IC 5-yr abyss trajectories agree to
        # ~0.01 K (.physics-validator/eke_acc_residual/spinup_abyss_*).
        stratification="veros_acc_linear",
    )
    # Prognostic EKE: when the recipe runs EKE on (``ACC_GM_REDI_CONFIG.eke`` set),
    # the eddy-energy field must be a Field from step 0. The model step turns ``eke``
    # None -> Field, which would break the constant-pytree carry of a ``jax.lax.scan``
    # forward integration (the free-run driver). Initialise it to the ``e_min`` floor
    # on wet cells (Veros likewise starts ``eke`` at a small positive value). The
    # frozen-state tier-2 probe does NOT read ``eke`` (it takes the constant-kappa GM
    # path), so the committed tier-2 tendency comparison is unchanged.
    #
    # eke_3d=True (the ACC recipe): the eddy-energy field is 3-D on the interior
    # interfaces (n_lat, n_lon, nlev-1) (the W-grid), seeded to e_min on wet
    # columns. Otherwise it is the 2-D depth-integrated (n_lat, n_lon) field.
    eke_cfg = gm_redi.eke
    if eke_cfg is not None:
        lm = state.land_mask.data
        if eke_cfg.eke_3d:
            nlev = z_coord.n_levels
            eke0 = (eke_cfg.e_min * lm)[:, :, jnp.newaxis] * jnp.ones(
                (1, 1, nlev - 1), dtype=lm.dtype)
            eke_dims = ("lat", "lon", "level")
            eke_units = "m^2/s^2"            # specific eddy energy (W-grid)
        else:
            eke_dims = ("lat", "lon")
            if eke_cfg.closure == "geometric":
                # GEOMETRIC prognoses the depth-INTEGRATED ∫E dz [m^3/s^2]. Honor
                # the closure's OWN documented cold-start ∫E dz = e0_per_depth·H
                # (GeometricConfig, Torres et al. 2025 Appendix E), NOT the
                # Eden-Greatbatch specific-energy floor e_min [m^2/s^2]. Flat-bottom
                # ACC ⇒ H = H_max on wet columns. The step writes units "m^3/s^2"
                # (ocean_model_latlon_cgrid.py:1976); Field.units is static pytree
                # metadata, so a seed/step mismatch breaks the jax.lax.scan
                # constant-carry on the FIRST step.
                eke0 = eke_cfg.geometric.e0_per_depth * H_max * lm
                eke_units = "m^3/s^2"
            else:
                eke0 = eke_cfg.e_min * lm
                eke_units = "m^2/s^2"
        state = state._replace(
            eke=Field(data=eke0, name="eke", dims=eke_dims, units=eke_units))
        # eke_diss (Veros eke_diss_iw): the 3-D EKE step writes this every step;
        # seed it to zero so the None -> Field transition never happens mid-scan
        # (constant-pytree carry). Only the 3-D EKE path produces it.
        if eke_cfg.eke_3d:
            nlev = z_coord.n_levels
            ediss0 = jnp.zeros((lm.shape[0], lm.shape[1], nlev - 1),
                               dtype=eke0.dtype)
            state = state._replace(
                eke_diss=Field(data=ediss0, name="eke_diss",
                               dims=("lat", "lon", "level"), units="m^2/s^3"))
    # PROGNOSTIC TKE: when the recipe runs prognostic TKE on, seed state.tke at
    # the tke_background floor on wet columns (interior interfaces, W-grid) so the
    # carried field is a Field from step 0 (the model step would otherwise turn
    # tke None -> Field on the first iteration, breaking the lax.scan carry).
    tke_cfg = tke
    if getattr(tke_cfg, "prognostic", False):
        lm = state.land_mask.data
        nlev = z_coord.n_levels
        wet3 = (lm[:, :, jnp.newaxis] > 0.5)
        tke0 = jnp.where(wet3, tke_cfg.tke_background, 0.0).astype(
            lm.dtype) * jnp.ones((1, 1, nlev - 1), dtype=lm.dtype)
        state = state._replace(
            tke=Field(data=tke0, name="tke", dims=("lat", "lon", "level"),
                      units="m^2/s^2"))
    return state


# ---------------------------------------------------------------------------
# Surface forcing (free-run): Veros ACC wind stress + T* restoring
# ---------------------------------------------------------------------------


def _acc_taux_profile(lat_deg: np.ndarray) -> np.ndarray:
    """Veros ACC zonal wind stress ``taux`` [N/m^2] as a function of cell-centre
    latitude, reproducing ``veros/setups/acc/acc.py:126-129`` exactly. The band
    is selected by the T-point latitude ``yt`` (= ``lat_deg``) while the value
    uses the v-point latitude ``yu = yt + dy/2`` (Veros's discretisation). This
    is the OCEAN-SIDE stress (eastward-positive), applied with a ``+`` sign in
    Veros (``du += surface_taux/(rho_0·dz)``)."""
    yt = np.asarray(lat_deg, dtype=np.float64)
    yu = yt + 0.5 * DYT_DEG                       # v-point lat, as Veros uses
    taux = np.zeros_like(yt)
    south = yt < _ACC_WIND_LAT_S
    north = yt > _ACC_WIND_LAT_N
    taux = np.where(
        south,
        _ACC_TAUX_AMP * np.sin(
            np.pi * (yu - _VEROS_YU_MIN) / (_ACC_WIND_LAT_S - _VEROS_YT_MIN)),
        taux)
    taux = np.where(
        north,
        _ACC_TAUX_AMP * (1.0 - np.cos(
            2.0 * np.pi * (yu - _ACC_WIND_LAT_N) / (_VEROS_YU_MAX - _ACC_WIND_LAT_N))),
        taux)
    return taux


def build_acc_wind_stress(grid: LatLonGrid) -> OceanSurfaceForcing:
    """Veros ACC surface wind stress as an :class:`OceanSurfaceForcing`.

    Veros applies ``surface_taux`` (the ocean-side, eastward-positive stress)
    with a ``+`` sign. legoESM's external surface-forcing path treats ``tau_x``
    as the ATMOSPHERIC stress and flips it (``-tau_x``) to get the ocean
    reaction (``ocean_pe_latlon_cgrid.py:1922``). So we pass ``tau_x = -taux``
    here, which after legoESM's internal flip reproduces Veros's ``+taux`` —
    i.e. westerlies (taux>0) accelerate the ACC eastward. Verified by the sign
    of the spun-up channel jet (must be eastward)."""
    lat_deg = np.degrees(np.asarray(grid.lat))    # cell-centre lat (n_lat,)
    taux = _acc_taux_profile(lat_deg)             # ocean-side stress (n_lat,)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    tau_x = jnp.asarray(
        np.broadcast_to(-taux[:, None], (n_lat, n_lon)))   # negated: see docstring
    tau_y = jnp.zeros((n_lat, n_lon), dtype=tau_x.dtype)
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


def build_acc_t_star(grid: LatLonGrid) -> jnp.ndarray:
    """Veros ACC surface restoring target ``t_star`` [degC] (2-D, broadcast over
    longitude), reproducing ``veros/setups/acc/acc.py:132-134``: 15 degC in the
    band [-20, 20], ramping linearly to 0 at the meridional walls."""
    lat = np.degrees(np.asarray(grid.lat))
    ts = np.full_like(lat, _ACC_TSTAR_AMP)
    south = lat < _ACC_TSTAR_LAT_S
    north = lat > _ACC_TSTAR_LAT_N
    ts = np.where(
        south,
        _ACC_TSTAR_AMP * (lat - _VEROS_YT_MIN) / (_ACC_TSTAR_LAT_S - _VEROS_YT_MIN),
        ts)
    ts = np.where(
        north,
        _ACC_TSTAR_AMP * (1.0 - (lat - _ACC_TSTAR_LAT_N) / (_VEROS_YT_MAX - _ACC_TSTAR_LAT_N)),
        ts)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    return jnp.asarray(np.broadcast_to(ts[:, None], (n_lat, n_lon)))


def build_acc_restoring_config(grid: LatLonGrid) -> RestoringConfig:
    """Veros ACC surface heat-flux forcing as a :class:`RestoringConfig`.

    Veros: ``forc_temp_surface = (dz_surf/(30 d)) · (t_star - T_surf)`` so the
    surface tendency is ``dT/dt = (t_star - T)/(30 d)`` — the layer thickness
    cancels, matching legoESM's ``restoring_surface_forcing`` form
    ``-(T - T*)/tau_T`` exactly with ``tau_T = 30 d``. Salinity is NOT restored
    in Veros ACC, so ``tau_S`` is huge (effectively off); ``S_star`` defaults to
    35 PSU (a no-op given the huge timescale)."""
    return RestoringConfig(
        tau_T=T_RESTORING_DAYS * _SECONDS_PER_DAY,
        tau_S=_ACC_NO_SALT_RESTORE_TAU_S,
        T_star_array=build_acc_t_star(grid),
        implicit=False,    # Veros uses explicit forward-Euler restoring; dt << 2·tau_T
    )


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


class ACCRecipe(NamedTuple):
    """All the pieces needed to run legoESM-Veros ACC:

    - ``model_config``: LatLonCGridOceanConfig matching Veros ACC settings.
    - ``physics_config``: OceanPhysicsConfig matching Veros ACC physics.
    - ``grid``: lat-lon C-grid at Veros ACC's native resolution.
    - ``z_coord``: stretched-z vertical coordinate.
    - ``land_mask``: cell-centre wet mask matching Veros's kbot pattern.
    - ``initial_state``: ocean state ready for the first step.

    All Veros physical constants are pinned through ``model_config.constants``
    (G-C4) — no monkey-patch of legoESM's constants module is needed.
    """
    model_config: LatLonCGridOceanConfig
    physics_config: OceanPhysicsConfig
    grid: LatLonGrid
    z_coord: OceanZStarCoordinate
    land_mask: jnp.ndarray
    initial_state: LatLonCGridOceanState
    # Free-run wind stress (OceanSurfaceForcing) — passed to ``model.step`` as
    # the ``surface_forcing`` arg. ``None`` unless built with surface forcing
    # (the frozen-state tendency probe does not use it).
    wind_forcing: object = None


def build_acc_physics_config(grid: LatLonGrid | None = None, *,
                             with_surface_forcing: bool = False,
                             ) -> OceanPhysicsConfig:
    """Veros ACC physics: TKE + GM/Redi + linear bottom drag (via model
    config), implicit vertical viscosity. IDEMIX is disabled in the ACC
    adapter (``enable_idemix=False`` in Veros) so it is not mapped.

    EKE: the Eden-Greatbatch prognostic-EKE closure (``lateral_mixing/eke.py``
    + ``GMRediConfig.eke``) is ADOPTED in ``ACC_GM_REDI_CONFIG`` (Veros ACC runs
    ``enable_eke=True``). The closure FORM + parameters reproduce Veros's
    ``K_gm`` AND its mixing length ``eke_len``/``L_rossby``/``L_rhines`` to
    machine precision given Veros's own state (gates E9 + L4; strategy doc §8
    EKE ledger). The recipe uses ``mixing_length_scheme="rhines"`` — the
    Rhines-limited ``eke_len = max(lmin, min(eke_cross·L_rossby,
    eke_crhin·L_rhines))`` (~8 km on the developed ACC) with ``eke_cross=2.0`` —
    so the prognostic GM coefficient ``kappa_GM = c_k·eke_len·√E`` is
    apples-to-apples with Veros, NOT the ~25×-too-large Visbeck ``L``.
    ``GMRediConfig.kappa_GM=1000`` is retained as the EKE-off fallback.

    Both the GM *skew* coefficient AND the Redi *tracer* diffusivity are now
    EKE-driven: ``isopycnal_diffusion=True`` reproduces Veros's
    ``enable_eke_isopycnal_diffusion`` (``K_iso = K_gm``), so the step drives
    ``kappa_Redi`` from the same prognostic kappa as GM (the constant
    ``kappa_Redi=1000`` is the EKE-off fallback). GM and Redi are both
    apples-to-apples with Veros.

    Set ``with_surface_forcing=True`` (free-run harness) to activate Veros ACC's
    T* surface restoring via the physics pipeline; ``grid`` is then required (for
    the latitude-dependent T* target). Default ``False`` keeps the prior
    prescribed-zero forcing so the frozen-state tendency probe + committed tier-2
    report are unchanged. The wind stress is applied SEPARATELY via the
    ``OceanSurfaceForcing`` argument to ``model.step`` (see
    :func:`build_acc_wind_stress`), not through this config — the two paths are
    disjoint (restoring -> dT, wind -> du/dv)."""
    if with_surface_forcing:
        if grid is None:
            grid = build_acc_grid()
        surface_forcing = SurfaceForcingConfig(
            scheme="restoring", restoring=build_acc_restoring_config(grid))
    else:
        surface_forcing = SurfaceForcingConfig(scheme="prescribed")
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=ACC_TKE_CONFIG),
        # GM/Redi on the lat-lon C-grid is a DYNAMICS-level process applied via
        # the top-level config.gm_redi field (see build_acc_model_config); the
        # physics-pathway lateral-mixing factory is cubed-sphere-only and raises
        # on lat-lon. So lateral_mixing is "none" here. (Setting GM/Redi ONLY in
        # physics.lateral_mixing — as before — left it INACTIVE for the ACC
        # recipe: config.gm_redi defaults None so the model skipped it, and the
        # tendency probe never invokes the physics pipeline.)
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=surface_forcing,
        bottom_drag=BottomDragConfig(scheme="none"),   # see model config bottom_drag_r
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def build_acc_model_config(grid: LatLonGrid | None = None, *,
                           with_surface_forcing: bool = False,
                           gm_redi: GMRediConfig = ACC_GM_REDI_CONFIG,
                           ) -> LatLonCGridOceanConfig:
    """Veros ACC dynamics: harmonic lateral viscosity with cos(lat)
    scaling, linear bottom drag, implicit vertical viscosity,
    Veros's nonlin3 EOS.

    ``with_surface_forcing`` (free-run) threads through to the physics config to
    activate T* restoring; default ``False`` is the frozen-state-probe config."""
    return LatLonCGridOceanConfig.from_flat(
        # All physical constants pinned to Veros via config (G-C4): g/rho_0 are
        # read by the PE core, the ConstantsConfig by the de-mirrored physics,
        # and R_earth feeds acc_A_h — pinned purely through config.
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=acc_A_h(VEROS_CONSTANTS_CONFIG.R_earth),
        A_h_lat_scaling=True,
        A_h_cos_power=1,
        # Veros ACC lateral viscosity is the component-wise FLUX-DIVERGENCE harmonic
        # friction ∇·(A_h∇u) (core/friction.py harmonic_friction), with the cos(lat)
        # A_h scaling applied inside the flux (enable_hor_friction_cos_scaling,
        # hor_friction_cosPower) — NOT legoESM's default vector Laplacian
        # grad(div)−k×grad(curl) (which carries curvature coupling between u,v that
        # the component-wise form omits). Select flux_divergence for apples-to-apples
        # (doctrine rule H). The paired K_diss_h EKE source automatically follows
        # (the energy-consistent component-wise A_h|∇u|² = Veros calc_diss_u/v),
        # since kdiss_h_flux_form=True is also set on the EKE config below.
        lateral_viscosity_operator="flux_divergence",
        # Veros ACC momentum advection is flux-form (core/momentum.py), centered
        # 2nd-order — so the recipe selects flux_form/centered for true
        # apples-to-apples (doctrine rule H). Validated: vs the developed-flow
        # Veros snapshot this lifts du_adv corr 0.584->0.654 and dv_adv
        # -0.336->0.715 (vector-invariant was anti-correlated). See the §8 ledger.
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        # VERTICAL momentum advection (stage 8): Veros (core/momentum.py
        # momentum_advection) advects the FULL velocity with a 2nd-order
        # CENTERED energy-conserving flux (flux_top = 0.25·(u[k+1]+u[k])·(w+w_east)).
        # legoESM's default ("upwind_perturbation") instead advects only the
        # baroclinic PERTURBATION u' = u - U_bar with 1st-order interface
        # upwind — which (a) OMITS the depth-integral-zero -d/dz(w·U_bar)
        # redistribution of barotropic momentum into shear, and (b) adds an
        # implicit vertical viscosity ~|w|·dz/2 that damps baroclinic shear.
        # Both push the column toward barotropic (the observed ubot/usurf 0.30
        # vs Veros 0.12). The recipe selects the Veros-faithful "centered_full"
        # for apples-to-apples (doctrine rule H). Stability rests on dt_mom +
        # A_v/TKE friction, exactly as in Veros (the centered flux is unlimited
        # and dispersive, with no implicit viscosity).
        vertical_momentum_scheme="centered_full",
        # TRACER advection: the recipe selects legoESM's "centered" scheme to
        # match Veros ACC's UNLIMITED centered 2nd-order tracer flux
        # (``veros/core/advection.py`` ``adv_flux_2nd``;
        # ``enable_superbee_advection=False``) -- horizontal
        # F = 0.5*(T[i]+T[i+1])*(h*u), vertical F = 0.5*(T[k]+T[k+1])*w -- for a
        # true apples-to-apples dycore comparison (doctrine rule H: finish the
        # Veros-numerics options). This SUPERSEDES the prior "documented gap,
        # won't add" framing (commit 7915f6eb): legoESM previously had only
        # limited / high-order lat-lon C-grid tracer schemes
        # (tvd/superbee/dst3/weno5/7); the plain centered 2nd-order option is now
        # added (reusing the same flux-form divergence machinery) and selected
        # here.
        #
        # HONEST CAVEAT (unchanged from the prior note): centered 2nd-order is
        # unlimited and therefore DISPERSIVE -- it can over/undershoot near sharp
        # gradients (non-monotone, locally negative tracers) and carries no
        # implicit diapycnal mixing. legoESM's production default stays TVD (Van
        # Leer), which is robust/monotone. And the climate lever is structural:
        # the residual abyssal warm bias is the too-strong resolved overturning
        # (the too-barotropic flow), and a prior tvd/dst3/weno5 scheme sweep
        # showed LESS-diffusive advection runs the abyss slightly WARMER, not
        # closer to Veros -- so the tracer scheme is matched for FIDELITY, not as
        # a fix for the climate delta (the over-overturning is the vertical
        # momentum partition / eddy form stress, not the tracer scheme).
        tracer_advection="centered",
        bottom_drag_r=R_BOT,
        # ``eq_of_state_type=3`` in Veros dispatches to nonlinear_eq2.py
        # (Vallis 2008) — NOT nonlinear_eq3.py despite the file name.
        # Verified against ``veros/core/density/get_rho.py``.
        eos="veros_nonlin2",
        implicit_vertical_mixing=True,
        # Drop legoESM's constant background TRACER diffusivity K_v (default
        # 1e-4). Veros ACC has NO constant background tracer mixing -- the
        # abyssal floor is kappaH_min=2e-5 via the TKE Prandtl chain (see
        # ACC_TKE_CONFIG.prandtl_mode). Leaving the 1e-4 background on top of
        # the TKE kappaM_min(2e-4) leak gave the ~9x-too-diffusive abyss
        # (3.0e-4 vs Veros ~3.3e-5) that ventilates away the deep stratification.
        # (A_v -- the constant background MOMENTUM viscosity, default 1e-3 -- is
        # left at the legoESM default; Veros's momentum floor is kappaM_min=2e-4
        # via TKE, so A_v is a separate momentum-only delta outside the deep-T
        # warm-bias scope and is not touched here.)
        K_v=0.0,
        # GM/Redi is a TOP-LEVEL (dynamics) field on the lat-lon C-grid — this
        # is what the model actually reads (ocean_model_latlon_cgrid.py:998).
        # Setting it only in physics.lateral_mixing left GM/Redi inactive.
        # ``gm_redi`` defaults to the Eden-Greatbatch ACC config; the GEOMETRIC
        # calibration twin (Stage 0) passes a GMRediConfig whose ``.eke`` carries
        # ``closure="geometric"`` (see ``build_acc_recipe(eke_override=...)``).
        gm_redi=gm_redi,
        # Veros applies the surface TRACER forcing (forc_temp_surface restoring)
        # IMPLICITLY — it enters the backward-Euler vertical-mixing tridiagonal
        # RHS at weight 1.0 (core/thermodynamics.py), NOT as an AB2-extrapolated
        # explicit tendency. Match that placement so the restoring is not
        # over-applied 1.6× by the faithful AB2 outer integrator. Only meaningful
        # with surface forcing present; the frozen-state tendency probe
        # (with_surface_forcing=False, no restoring) stays bit-identical either
        # way, so gate it on with_surface_forcing to keep the probe path
        # unambiguous.
        surface_forcing_implicit=with_surface_forcing,
        # --- Veros-faithful STEPPING COMPOSITION (free-run default) ---
        # These were historically applied as DRIVER OVERRIDES (run_acc_freerun /
        # attractor_drop), so "the faithful recipe" silently required remembering
        # six flags — and the un-overridden recipe ran the matsuno_split/"total"
        # composition, which a realized 90-day energy audit (2026-06-12,
        # .physics-validator/eke_acc_residual/traj_energy_*) showed LEAKS
        # ~+1.8 GW of spurious kinetic energy (0.6% of the 282 GW wind
        # throughput, continuously) where Veros closes to ±0.03 GW. Under this
        # bundle the realized budget matches Veros channel-by-channel to ~1-3%
        # (wind 284 vs 282, bottom drag 22.0 vs 22.1, lateral 58 vs 56,
        # vertical 102.4 vs 102.2, baroclinic conversion 101.6 vs 101.1 GW) and
        # the 30-yr free run lands at KE 0.950x / mean_eke 1.37x the Veros
        # equilibrium (was 1.53x / 4.6x). Gated on with_surface_forcing exactly
        # like surface_forcing_implicit above: the frozen-state tendency-probe
        # path (with_surface_forcing=False) keeps the legoESM defaults and
        # stays bit-identical.
        #
        # The bundle (each ↔ its Veros counterpart; all phase-G options):
        #   outer_integrator="ab2"        ↔ Veros AB2 (AB_eps=0.1 = config
        #                                   default ab2_epsilon, not repeated)
        #   dt_mom_ratio=9.0              ↔ dt_mom=4800 / dt_tracer=43200
        #                                   (pass dt=43200 as the step dt)
        #   barotropic_solver="rigid_lid" ↔ enable_streamfunction
        #   coriolis_scheme="explicit_ab2"↔ tend_coriolisf in du (the energy
        #                                   lever — see the audit note above)
        #   ab2_scope="advective"         ↔ dissipative tendencies at weight
        #                                   1.0 (solve_stream.py placement)
        #   momentum_friction_additive=True ↔ explicit additive du_mix
        # implicit_vmix_dzw_slot ↔ Veros dzw divisor of the implicit T/S +
        # friction solves (#428): the ACC z-coordinate is u_centered, so the
        # dz_half_ref·J slot differs from the midpoint reconstruction. The whole
        # bundle is gated on with_surface_forcing — the frozen-state probe keeps
        # legoESM defaults and stays bit-identical. Shared across all 5 Veros
        # recipes via veros_stepping.veros_faithful_stepping (#433).
        **veros_faithful_stepping(with_surface_forcing=with_surface_forcing,
                                  dt_mom_ratio=9.0),
        physics=build_acc_physics_config(
            grid, with_surface_forcing=with_surface_forcing),
    )


def geometric_eke_config(geom: GeometricConfig = GeometricConfig()) -> EKEConfig:
    """Clean GEOMETRIC EKE closure config for the calibration twin (Stage 0).

    The Eden-Greatbatch ACC config (:data:`ACC_GM_REDI_CONFIG`.eke) sets a stack
    of EG-only flags (``eke_3d=True``, ``isopycnal_diffusion=True``,
    ``gm_source_mode="realized_signed"``, ``source_kdiss_h``/``kdiss_h_flux_form``)
    that the GEOMETRIC closure's cross-validation REJECTS (eke.py:730-763) — its
    source terms (B_C, B_T) and Redi coupling (kappa_n) are self-contained. So the
    twin uses a fresh ``EKEConfig`` carrying ONLY ``closure="geometric"`` + the
    ``GeometricConfig`` (Torres et al. 2025); every other field stays at its
    EKEConfig default. The depth-INTEGRATED 2-D budget requires ``eke_3d=False``
    (the default), which routes the initial-state seeding (build_acc_state:504-516)
    to the 2-D ``(lat, lon)`` eke field the geometric step consumes."""
    return EKEConfig(closure="geometric", geometric=geom)


def build_acc_recipe(*, with_surface_forcing: bool = False,
                     eke_override: EKEConfig | None = None) -> ACCRecipe:
    """One-stop constructor. Use as::

        from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe

        recipe = build_acc_recipe()
        # ... run / probe with recipe.initial_state, .model_config, etc.

    All Veros constants (g, rho_0, Omega, R_earth, c_sw, derived A_h) are
    pinned through config — ``build_acc_model_config`` sets ``constants=
    VEROS_CONSTANTS_CONFIG`` and the grid radius/omega — so no monkey-patch
    or ``override_constants`` context is required (G-C4).

    ``with_surface_forcing=True`` (free-run harness) activates Veros ACC's
    surface forcing: T* restoring through ``physics_config`` and the wind stress
    as ``recipe.wind_forcing`` (pass to ``model.step(..., surface_forcing=
    recipe.wind_forcing)``). Default ``False`` is the frozen-state-probe recipe
    (no forcing), so the committed tier-2 tendency comparison is unchanged.

    ``eke_override`` swaps the EKE closure config in ``model_config.gm_redi.eke``
    (e.g. :func:`geometric_eke_config` for the GEOMETRIC calibration twin). It is
    threaded into BOTH the model config (so the step runs the chosen closure) AND
    ``build_acc_state`` (so the initial eke field is seeded with the matching
    shape — 2-D for geometric/``eke_3d=False``, 3-D for Eden-Greatbatch). ``None``
    keeps the Eden-Greatbatch ACC default, leaving the historical call
    ``build_acc_recipe()`` bit-identical.
    """
    grid = build_acc_grid()
    z_coord = build_acc_z_coord()
    land_mask = build_acc_land_mask(grid)
    gm_redi = (ACC_GM_REDI_CONFIG if eke_override is None
               else ACC_GM_REDI_CONFIG._replace(eke=eke_override))
    initial_state = build_acc_state(grid, z_coord, gm_redi=gm_redi)
    wind_forcing = build_acc_wind_stress(grid) if with_surface_forcing else None
    return ACCRecipe(
        model_config=build_acc_model_config(
            grid, with_surface_forcing=with_surface_forcing, gm_redi=gm_redi),
        physics_config=build_acc_physics_config(
            grid, with_surface_forcing=with_surface_forcing),
        grid=grid,
        z_coord=z_coord,
        land_mask=land_mask,
        initial_state=initial_state,
        wind_forcing=wind_forcing,
    )


__all__ = (
    "ACC_DZT",
    "ACC_GM_REDI_CONFIG",
    "ACC_TKE_CONFIG",
    "ACCRecipe",
    "DT_MOM_S",
    "DT_TRACER_S",
    "NX",
    "NY",
    "NZ",
    "R_BOT",
    "T_RESTORING_DAYS",
    "VEROS_BLOCK_MAPPING",
    "X_ORIGIN_DEG",
    "Y_ORIGIN_DEG",
    "acc_A_h",
    "build_acc_grid",
    "build_acc_land_mask",
    "build_acc_model_config",
    "build_acc_physics_config",
    "build_acc_recipe",
    "build_acc_restoring_config",
    "build_acc_state",
    "build_acc_t_star",
    "build_acc_wind_stress",
    "build_acc_z_coord",
    "geometric_eke_config",
)
