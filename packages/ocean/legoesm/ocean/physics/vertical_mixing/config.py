"""Configuration for ocean vertical mixing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.tidal import TidalMixingConfig


class ConstantVerticalMixingConfig(NamedTuple):
    """Constant-coefficient vertical mixing."""
    A_v: float = 1e-3   # Vertical viscosity [m^2/s]
    K_v: float = 1e-4   # Vertical diffusivity [m^2/s]


class RichardsonVerticalMixingConfig(NamedTuple):
    """Pacanowski & Philander (1981) Richardson-number dependent mixing."""
    K_0: float = 5e-3    # Maximum diffusivity [m^2/s]
    alpha: float = 5.0   # Stability parameter
    n: int = 2           # Exponent
    K_bg: float = 1e-5   # Background diffusivity [m^2/s]
    A_bg: float = 1e-4   # Background viscosity [m^2/s]
    Pr_t: float = 10.0   # Turbulent Prandtl number


class TKEConfig(NamedTuple):
    """Gaspar (1990) / Burchard (2002) prognostic TKE closure.

    Veros's canonical vertical-mixing scheme (``enable_tke=True``).
    Solves a prognostic budget for turbulent kinetic energy per unit
    mass, then derives the eddy diffusivities ``K_M`` / ``K_H`` from
    TKE and a mixing-length closure.

    Closure equations (per column, at interfaces unless noted):

        dTKE/dt = P_s + P_b - eps + d/dz(alpha_tke * K_M * dTKE/dz)

    with

        P_s   = K_M * (du/dz)^2 + K_M * (dv/dz)^2     (shear production)
        P_b   = -K_H * N^2                            (buoyancy work)
        eps   = c_eps * TKE^(3/2) / l_eps             (dissipation)
        K_M   = c_k * l_k * sqrt(2 * TKE)
        K_H   = K_M

    The mixing lengths ``l_k`` (for K_M) and ``l_eps`` (for dissipation)
    follow the Bougeault-Lacarrere asymmetric construction
    (``tke_mxl_choice=2``): an upward and downward integration of TKE
    against the local Brunt-Vaisala frequency gives ``l_up`` and
    ``l_dn``; ``l_k = sqrt(l_up * l_dn)`` and
    ``l_eps = max(l_up, l_dn)``. ``tke_mxl_choice=1`` selects a simple
    parabolic-bounded length scale; choice ``2`` is the production
    default.

    Surface flux into TKE is ``(|tau|/rho_0)^(3/2)`` (Wallace surface
    parameterisation).

    Parameter naming mirrors Veros's settings 1:1 so the legoESM-Veros
    recipe maps each knob directly. Defaults are Veros's DINO / ACC
    canonical values.
    """
    c_k: float = 0.1
    c_eps: float = 0.7
    alpha_tke: float = 30.0
    mxl_min: float = 1.0e-8
    tke_mxl_choice: int = 2
    kappaM_min: float = 2.0e-4
    kappaM_max: float = 100.0            # convective ceiling on K_M [m^2/s] (Veros default)
    kappaH_min: float = 2.0e-5
    enable_kappaH_profile: bool = True
    tke_surface_min: float = 1.0e-4      # surface TKE floor [m^2/s^2]
    tke_background: float = 1.0e-6       # interior TKE floor [m^2/s^2]
    # ----- Static-stability N^2 mode (deep-ocean ventilation / convection) -----
    # ``"insitu"`` (default, BIT-IDENTICAL legacy): N^2 from the in-situ
    #   density difference at native pressures, clipped >= 0 (convection
    #   never fires through the TKE -- a static-stability bias that leaves
    #   the abyss unventilated).
    # ``"adiabatic"``: N^2 by adiabatic parcel displacement to the upper
    #   cell's pressure (Veros thermodynamics.py:99-103), SIGNED (not
    #   clipped). With this, a statically-unstable column gives N^2 < 0, the
    #   buoyancy length scale blows up, and K_M saturates toward
    #   ``kappaM_max`` -- i.e. the TKE itself convects, exactly as Veros's
    #   ``enable_tke`` path does. Requires the caller to pass T/S/pressure
    #   + an EOS to :func:`tke_vertical_mixing`.
    n2_mode: str = "insitu"
    # ----- Tracer/momentum Prandtl chain (abyssal over-diffusion fix) -----
    # ``"unit"`` (default, BIT-IDENTICAL legacy): K_H = max(K_M, kappaH_min)
    #   -- the MOMENTUM floor ``kappaM_min`` leaks into the TRACER floor
    #   (``kappaH_min`` is dead) -> abyssal K_H ~ kappaM_min, ~6-10x too
    #   diffusive vs Veros.
    # ``"constant"``: K_H = max(kappaH_min, K_M / Prandtl_tke0).
    # ``"richardson"``: Veros ``enable_Prandtl_tke=True`` --
    #   Prandtl = max(1, min(10, 6.6 * Ri)) with the gradient Richardson
    #   number Ri = N^2 / max(shear^2, eps); K_H = max(kappaH_min,
    #   K_M / Prandtl). In the stratified interior Pr -> 10 (small abyssal
    #   K_H); in a convecting column Ri < 0 -> Pr -> 1 (K_H tracks the large
    #   convective K_M). This is the Veros ACC default.
    prandtl_mode: str = "unit"
    Prandtl_tke0: float = 10.0           # constant Prandtl number (Veros Prandtl_tke0)
    # ----- Prognostic TKE carry (Veros enable_tke PROGNOSTIC form) -----
    # ``prognostic=False`` (default, BIT-IDENTICAL): the Mode-B quasi-steady
    #   diagnostic chain runs in ``compute_vertical_K_profiles`` — ``tke_old=None``
    #   seeded at background, ``n_iterations=3``, ``dt=86400`` (drives the implicit
    #   solve to the local quasi-steady equilibrium). No TKE field is carried.
    # ``prognostic=True``: the TKE field is CARRIED on the ocean state
    #   (``state.tke``, interior interfaces ``(n_lat, n_lon, nlev-1)``). Each model
    #   step runs ONE backward-Euler TKE solve with ``dt = dt_mom`` (the MOMENTUM
    #   timestep — Veros tke.py:137 ``dt_tke = dt_mom`` even though TKE advances
    #   once per tracer step), ``n_iterations=1``, seeded from the carried field.
    #   The updated TKE is returned and stored back on the state at the END of the
    #   model step. Requires the recipe (or driver) to seed ``state.tke`` so the
    #   ``lax.scan`` carry pytree stays constant.
    prognostic: bool = False
    # ----- Energy-recycling sources (Veros forc += eke_diss_iw + K_diss_bot) -----
    # Each default OFF; only consulted when ``prognostic=True``. They add the
    # dissipated mechanical energy from OTHER schemes back into the TKE source
    # ``forc`` (Veros integrate_tke_kernel forc assembly), recycling energy that
    # would otherwise be lost — the Veros ACC energetically-consistent closure.
    #
    # ``source_eke_diss``: add the EKE dissipation rate (Veros ``eke_diss_iw =
    #   c_int·E`` with ``c_int = eke_c_eps·√E/eke_len`` — the SAME sink the EKE
    #   step already computes). The EKE step runs AFTER the TKE K-profile solve in
    #   the legoESM model step (the K-profiles are computed first in
    #   ``_apply_implicit_vertical_mixing``), so the EKE dissipation is carried
    #   across via a state field (``state.eke_diss``) — a documented ONE-STEP LAG.
    #
    # ``source_bottom_drag_diss``: add ``K_diss_bot`` — the bottom-drag KE
    #   extraction (Veros friction.linear_bottom_friction: ``diss = r_bot·u²`` at
    #   the bottom level, mapped to the W-grid). legoESM surfaces this as a
    #   tendency diagnostic (``LatLonCGridOceanTendencies.K_diss_bot``, mirroring
    #   the ``Ah_visc_u/v`` K_diss_h pattern) and routes it to the TKE source at
    #   the interior interfaces.
    #
    # ``P_diss_adv`` (advective) and ``P_diss_nonlin`` (cabbeling/non-linear EOS)
    # are DEFERRED — they need advection/cabbeling dissipation diagnostics that
    # legoESM does not yet surface (Veros tke.py:142,149). Documented gap.
    source_eke_diss: bool = False
    source_bottom_drag_diss: bool = False


class KPPConfig(NamedTuple):
    """LMD94-style K-Profile Parameterization.

    Extends the Large, McWilliams & Doney (1994) KPP boundary-layer
    parameterization with linear interpolation for h_bl, proper
    turbulent velocity scales, and interior convective instability
    handling.

    The caller should supply surface wind stress (tau_x, tau_y) and
    surface buoyancy flux (B_f) so that friction velocity and turbulent
    velocity scales can be computed correctly.  If these are not
    provided, simplified proxies from the ocean state are used.
    """
    Ri_crit: float = 0.3    # Critical bulk Richardson number (Large, McWilliams & Doney 1994; matches NCAR POP2 / MOM6 default)
    Cv: float = 1.6          # Unresolved shear coefficient
    kappa_vk: float = constants.kappa_vk
    K_max: float = 1.0       # Maximum diffusivity [m^2/s]
    K_bg: float = 1e-5       # Background diffusivity [m^2/s]
    A_bg: float = 1e-4       # Background viscosity [m^2/s]
    gamma_T: float = 6.33    # Non-local transport coefficient for T
    gamma_S: float = 6.33    # Non-local transport coefficient for S
    K_conv: float = 1.0      # Convective mixing diffusivity [m^2/s]
    Ri_conv: float = 0.0     # Ri threshold for convective instability
    K_0_shear: float = 5e-3  # LMD94 interior shear instability peak K [m^2/s]
    Ri_0: float = 0.7        # LMD94 critical Ri for interior shear mixing
    c_s: float = 98.96       # LMD94 scalar stability constant (App. B; V_t^2 + scalar convective scale)
    c_b: float = 0.599       # LMD94 convective velocity scale parameter (legacy single-scale form)
    epsilon_lmd: float = 0.1  # LMD94 surface-layer fraction (App. A/B)
    # LMD94 Appendix B separate momentum/scalar velocity scales w_m, w_s.
    # In the code's sign convention zeta = d/L_MO ≥ 0 for unstable, so the
    # weakly-unstable→convective transition is at |zeta| = zeta_{m,s}_abs
    # (= −zeta_{m,s} of LMD94).  a_*/c_* are chosen by LMD94 so the
    # convective scale w = kappa·(a·u*³ + c·kappa·B_f·d)^{1/3} joins the
    # weakly-unstable kappa·u*·(1+16|zeta|)^{1/4 (m), 1/2 (s)} continuously.
    zeta_m_abs: float = 0.2   # |zeta| momentum weakly→convective join (LMD94 zeta_m=−0.2)
    zeta_s_abs: float = 1.0   # |zeta| scalar   weakly→convective join (LMD94 zeta_s=−1.0)
    a_m: float = 1.26         # LMD94 momentum convective constant
    c_m: float = 8.38         # LMD94 momentum convective constant
    a_s: float = -28.86       # LMD94 scalar convective constant (slope = c_s)
    crossing_sharpness: float = 20.0  # Sigmoid sharpness for h_bl crossing-depth selector
    crossing_threshold: float = 0.1   # Crossing-strength threshold for h_bl blend
    # Timestep [s] used to derive the explicit-diffusion CFL ceiling
    # A_v_max = 0.25 * min(dz_k, dz_k+1)^2 / cfl_cap_dt_s on the MPAS path.
    # MUST be set to the ocean dynamics dt for the cap to be correct: a
    # value smaller than the real dt over-damps; larger risks instability.
    # Default 300.0 preserves the historical hard-coded estimate.
    cfl_cap_dt_s: float = 300.0


class VerticalMixingConfig(NamedTuple):
    """Top-level vertical mixing configuration."""
    scheme: str = "constant"  # "constant", "richardson", "kpp", "tke", "none"
    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    kpp: KPPConfig = KPPConfig()
    tke: TKEConfig = TKEConfig()
    # Tidal mixing is ADDITIVE: when ``tidal.enabled=True`` the
    # caller computes a ``K_tidal(x, y, z)`` field via
    # :func:`legoesm.ocean.physics.vertical_mixing.tidal.compute_tidal_diffusivity`
    # and adds it on top of the diffusivity field from ``scheme``
    # before applying the tracer mixing step.  Default off
    # (``enabled=False``) preserves bit-exact regression on legacy
    # configs.  NOTE: ``make_vertical_mixing_physics`` / ``make_ocean_physics``
    # do NOT apply tidal mixing and RAISE if ``enabled=True`` — apply it via
    # ``ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step`` so it cannot
    # silently no-op inside the physics composition.
    tidal: TidalMixingConfig = TidalMixingConfig()
