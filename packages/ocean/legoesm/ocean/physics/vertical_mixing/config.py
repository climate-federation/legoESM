"""Configuration for ocean vertical mixing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.tidal import TidalMixingConfig
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMConfig,
)
from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
    DoubleDiffusionConfig,
)


__param_spec__ = {
    "ConstantVerticalMixingConfig": {
        "scheme_key": "ocean.vm.constant",
        "excluded": {
            "N_ref": "Gregg et al. (2003) fixed published reference stratification N_0 (5.24e-3 1/s); the latitude-scaling normalisation, not a trained closure knob",
        },
        "params": {
            "A_v": {"units": "m^2/s", "bounds": (0.00033, 0.003), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "constant vertical mixing", "shape": None},
            "K_v": {"units": "m^2/s", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "constant vertical mixing", "shape": None},
            "K_bg_eq": {"units": "m^2/s", "bounds": (3.3e-06, 3e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Gregg et al. (2003) latitude-dependent internal-wave background (CVMix bkgnd) — equatorial diffusivity", "shape": None},
            "K_bg_pole": {"units": "m^2/s", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Gregg et al. (2003) latitude-dependent internal-wave background (CVMix bkgnd) — polar diffusivity", "shape": None},
        },
    },
    "RichardsonVerticalMixingConfig": {
        "scheme_key": "ocean.vm.richardson",
        "excluded": {
            "Pr_t": "dead field: the Pr-Ri scaling is built into the K formula; richardson.py only warns when Pr_t != default and never reads it — phantom trainable",
        },
        "params": {
            "A_bg": {"units": "m^2/s", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Pacanowski-Philander Richardson mixing", "shape": None},
            "K_0": {"units": "m^2/s", "bounds": (0.00165, 0.015), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Pacanowski-Philander Richardson mixing", "shape": None},
            "K_bg": {"units": "m^2/s", "bounds": (3.3e-06, 3e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Pacanowski-Philander Richardson mixing", "shape": None},
            "alpha": {"units": "1", "bounds": (1.65, 15.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Pacanowski-Philander Richardson mixing", "shape": None},
        },
    },
    "TKEConfig": {
        "scheme_key": "ocean.vm.tke",
        "excluded": {
            "bg_diff_amp": "Bryan-Lewis 1979 fixed published arctan offset",
            "bg_diff_arctan_coeff": "Bryan-Lewis 1979 fixed published arctan amplitude",
            "bg_diff_depth_m": "Bryan-Lewis 1979 fixed published transition depth",
            "bg_diff_width_m": "Bryan-Lewis 1979 fixed published transition width",
            "cfl_cap_dt_s": "numerics: solver/CFL/smoothing parameter",
            "kappaH_min": "numerics: floor/cap",
            "kappaM_max": "numerics: floor/cap",
            "kappaM_min": "numerics: floor/cap",
            "mxl_min": "numerics: floor/cap",
            "prandtl_ri_coeff": "Galperin/Veros fixed Pr-Ri slope (6.6)",
            "tke_background": "numerics: floor/cap",
            "tke_surface_min": "numerics: floor/cap",
        },
        "params": {
            "Prandtl_tke0": {"units": "1", "bounds": (3.3, 30.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Gaspar TKE vertical mixing", "shape": None},
            "lc_coeff": {"units": "1", "bounds": (0.05, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "NEMO zdftke rn_lc / Axell 2002 Langmuir cells", "shape": None},
            "etau_frac": {"units": "1", "bounds": (0.01, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "NEMO zdftke rn_efr sub-ML TKE penetration", "shape": None},
            "alpha_tke": {"units": "1", "bounds": (9.9, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Gaspar TKE vertical mixing", "shape": None},
            "bg_diff_scale": {"units": "m^2/s", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Bryan-Lewis (1979) background-diffusivity amplitude", "shape": None},
            "c_eps": {"units": "1", "bounds": (0.231, 2.1), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Gaspar TKE vertical mixing", "shape": None},
            "c_k": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "Gaspar TKE vertical mixing", "shape": None},
        },
    },
    "CATKEConfig": {
        "scheme_key": "ocean.vm.catke",
        "excluded": {
            "c_entr_u": "calibrated to 0 (momentum entrainment-penetration length disabled in Wagner 2025)",
            "c_entr_e": "calibrated to 0 (TKE entrainment-penetration length disabled in Wagner 2025)",
            "c_entr_diss": "calibrated to 0 (dissipation entrainment-penetration length disabled)",
            "minimum_tke": "numerics: floor",
            "minimum_convective_buoyancy_flux": "numerics: regulariser floor",
            "negative_tke_damping_time_s": "numerics: negative-TKE damping timescale (iteration-coupled)",
            "maximum_viscosity": "numerics: cap",
            "maximum_tracer_diffusivity": "numerics: cap",
            "maximum_tke_diffusivity": "numerics: cap",
        },
        "params": {
            "c_surface_shear": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) mixing length", "shape": None},
            "c_bottom_shear": {"units": "1", "bounds": (0.05, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) mixing length", "shape": None},
            "c_ri_lower": {"units": "1", "bounds": (0.05, 0.7), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) stability function", "shape": None},
            "c_ri_width": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) stability function", "shape": None},
            "c_sheared_plume": {"units": "1", "bounds": (0.1, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) convective length", "shape": None},
            "c_hi_u": {"units": "1", "bounds": (0.05, 0.8), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) momentum length", "shape": None},
            "c_lo_u": {"units": "1", "bounds": (0.1, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) momentum length", "shape": None},
            "c_un_u": {"units": "1", "bounds": (0.1, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) momentum length", "shape": None},
            "c_hi_c": {"units": "1", "bounds": (0.02, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) tracer length", "shape": None},
            "c_lo_c": {"units": "1", "bounds": (0.1, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) tracer length", "shape": None},
            "c_un_c": {"units": "1", "bounds": (0.1, 1.8), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) tracer length", "shape": None},
            "c_hi_e": {"units": "1", "bounds": (0.1, 1.8), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) TKE length", "shape": None},
            "c_lo_e": {"units": "1", "bounds": (2.0, 24.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) TKE length", "shape": None},
            "c_un_e": {"units": "1", "bounds": (0.4, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) TKE length", "shape": None},
            "c_conv_u": {"units": "1", "bounds": (1.0, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) convective length", "shape": None},
            "c_conv_c": {"units": "1", "bounds": (1.0, 15.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) convective length", "shape": None},
            "c_entr_c": {"units": "1", "bounds": (0.02, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) entrainment length", "shape": None},
            "c_conv_e": {"units": "1", "bounds": (1.0, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) convective length", "shape": None},
            "c_hi_diss": {"units": "1", "bounds": (0.1, 1.8), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) dissipation length", "shape": None},
            "c_lo_diss": {"units": "1", "bounds": (0.5, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) dissipation length", "shape": None},
            "c_un_diss": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) dissipation length", "shape": None},
            "c_conv_diss": {"units": "1", "bounds": (1.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) dissipation length", "shape": None},
            "c_w_ustar": {"units": "1", "bounds": (1.0, 8.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) surface TKE flux", "shape": None},
            "c_w_conv": {"units": "1", "bounds": (0.1, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "CATKE (Wagner 2025) surface TKE flux", "shape": None},
        },
    },
    "KPPConfig": {
        "scheme_key": "ocean.vm.kpp",
        "excluded": {
            "Cv": "Large 1994 fixed nondim constant",
            "neg_beta_T": "Large 1994 fixed nondim constant (-beta_T, App. B; V_t^2 prefactor)",
            "Ri_conv": "default 0 = disabled/off (enable via config, not training)",
            "a_m": "Large 1994 fixed nondim constant",
            "a_s": "Large 1994 fixed nondim constant",
            "c_b": "Large 1994 fixed nondim constant",
            "c_m": "Large 1994 fixed nondim constant",
            "c_s": "Large 1994 fixed nondim constant",
            "businger_stable_coeff": "Businger-Dyer 1971 fixed MOST stability-function constant",
            "businger_unstable_coeff": "Businger-Dyer 1971 fixed MOST stability-function constant",
            "cfl_cap_dt_s": "numerics: solver/CFL/smoothing parameter",
            "crossing_sharpness": "numerics: solver/CFL/smoothing parameter",
            "crossing_threshold": "numerics: solver/CFL/smoothing parameter",
            "epsilon_lmd": "numerics: floor/cap",
            "gamma_S": "Large 1994 fixed nondim constant",
            "gamma_T": "Large 1994 fixed nondim constant",
            "zeta_m_abs": "Large 1994 fixed nondim constant",
            "zeta_s_abs": "Large 1994 fixed nondim constant",
        },
        "params": {
            "A_bg": {"units": "m^2/s", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "K_0_shear": {"units": "m^2/s", "bounds": (0.00165, 0.015), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "K_bg": {"units": "m^2/s", "bounds": (3.3e-06, 3e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "K_conv": {"units": "m^2/s", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "K_max": {"units": "m^2/s", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "Ri_0": {"units": "1", "bounds": (0.231, 2.1), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "Ri_crit": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP (Large et al. 1994)", "shape": None},
            "ustar_speed_ratio": {"units": "1", "bounds": (0.0033, 0.03), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP u* surface-speed proxy (Large et al. 1994)", "shape": None},
            "langmuir_coeff": {"units": "1", "bounds": (0.02, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "KPP-Langmuir enhancement C_L (McWilliams & Sullivan 2000; Li et al. 2016 CVMix)", "shape": None},
            "langmuir_number_default": {"units": "1", "bounds": (0.2, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "fully-developed-sea turbulent Langmuir number ~0.3 (Van Roekel et al. 2012)", "shape": None},
        },
    },
}


class ConstantVerticalMixingConfig(NamedTuple):
    """Constant-coefficient vertical mixing.

    With ``lat_dependent=False`` (default) the vertical viscosity ``A_v`` and
    diffusivity ``K_v`` are spatial constants (BIT-IDENTICAL legacy).  With
    ``lat_dependent=True`` the background is REPLACED, on the IMPLICIT
    vertical-mixing path (``k_profiles.compute_vertical_K_profiles``), by the
    Gregg et al. (2003) latitude/stratification-scaled internal-wave background
    (CVMix ``bkgnd`` / MOM6 ``Henyey_IGW_background``): the diapycnal
    diffusivity is reduced toward the equator — where the Coriolis parameter
    vanishes and internal-wave breaking is suppressed — ranging from
    ``K_bg_eq`` (equator) to ``K_bg_pole`` (poleward); the momentum viscosity is
    scaled by the SAME factor, preserving the configured Prandtl ratio
    ``A_v/K_v``.  REPLACED means end-to-end: the model-level fallback floors
    the dynamics caller passes (``K_v_background``/``A_v_background``, i.e.
    ``LatLonCGridOceanConfig.K_v``/``A_v``) are suppressed as well, so the
    constant-BACKGROUND contribution lies exactly in ``[K_bg_eq, K_bg_pole]``
    with the configured Prandtl ratio (no double-added constant floor).
    Additive closures configured on top (``enhanced_diffusion`` convection,
    internal-wave mixing) still stack onto that background, and non-wet
    (sub-seafloor) interfaces are zeroed — the exact-range guarantee is for
    the background term at wet interfaces, not the total after other
    closures.  The EXPLICIT ``constant_vertical_mixing`` tendency path cannot
    apply a latitude field and RAISES if ``lat_dependent=True`` (no silent
    no-op).  See ``_shared.latitude_background_diffusivity``.
    """
    A_v: float = 1e-3   # Vertical viscosity [m^2/s]
    K_v: float = 1e-4   # Vertical diffusivity [m^2/s]
    # --- Latitude-dependent internal-wave background (Gregg 2003 / CVMix bkgnd) ---
    lat_dependent: bool = False   # opt-in; False => spatial-constant A_v/K_v (legacy)
    K_bg_eq: float = 1e-5         # equatorial background diffusivity [m^2/s]
    K_bg_pole: float = 1e-4       # polar background diffusivity [m^2/s]
    N_ref: float = 5.24e-3        # Gregg (2003) reference stratification N_0 [1/s]


class RichardsonVerticalMixingConfig(NamedTuple):
    """Pacanowski & Philander (1981) Richardson-number dependent mixing."""
    K_0: float = 5e-3    # Maximum diffusivity [m^2/s]
    alpha: float = 5.0   # Stability parameter
    n: int = 2           # Exponent
    K_bg: float = 1e-5   # Background diffusivity [m^2/s]
    A_bg: float = 1e-4   # Background viscosity [m^2/s]
    Pr_t: float = 10.0   # Turbulent Prandtl number
    # ----- Static-stability N^2 mode for the gradient Richardson number -----
    # ``"insitu"`` (default, BIT-IDENTICAL legacy) / ``"insitu_signed"``: N^2
    #   from the in-situ density difference (``eos.compute_buoyancy_frequency``,
    #   already signed/unclipped here — the downstream ``richardson_number``
    #   clips Ri>=0), which carries the compressibility bias (~too stable).
    # ``"adiabatic"``: PP81's TRUE static stability via adiabatic parcel
    #   displacement to the upper cell's pressure
    #   (``eos.compute_buoyancy_frequency_adiabatic``), SIGNED. Requires the
    #   caller to thread cell-centre pressure ``p_cell`` (+ the model EOS) to
    #   ``richardson_vertical_mixing``. Both integration factory and the
    #   implicit k_profiles path supply it when this is selected.
    n2_mode: str = "insitu"


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
    # --- Galperin Pr-Ri + Bryan-Lewis (1979) bg-diffusivity profile (#518 §10) ---
    prandtl_ri_coeff: float = 6.6        # Galperin/Veros fixed Pr = max(1, min(10, 6.6*Ri)) slope
    bg_diff_amp: float = 0.8             # Bryan-Lewis arctan offset (published fit)
    bg_diff_arctan_coeff: float = 1.05   # Bryan-Lewis arctan amplitude (published fit)
    bg_diff_depth_m: float = 2500.0      # Bryan-Lewis transition depth [m] (published fit)
    bg_diff_width_m: float = 222.2       # Bryan-Lewis transition width [m] (published fit)
    bg_diff_scale: float = 1.0e-4        # abyssal tracer-diffusivity floor amplitude [m^2/s]
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
    # ----- Veros vertical-metric slots (the TKE metric-consistency fix) -----
    # legoESM's historical TKE chain mixes vertical-metric conventions: it
    # uses the centre spacing ``dz_half`` (Veros dzw) in slots where Veros
    # uses the CELL thickness ``dzt``, reconstructs a midpoint interface
    # spacing for the adiabatic N² even on a u_centered (Veros) coordinate,
    # approximates the per-interface control volume as an average of
    # adjacent face spacings, and injects the surface TKE flux over
    # ``dz_half[0]`` instead of Veros's surface half-volume ``0.5·dzw_top``.
    # On a midpoint coordinate the slots nearly coincide; on a Veros
    # u_centered coordinate (the faithful recipes) they alternate by up to
    # ±50% per level and the chain equilibrates onto a spurious deep-TKE
    # branch (ACC_Basic diagnosis, .physics-validator/accbasic_regression/).
    #
    # ``veros_dz_slots=True`` evaluates every slot as Veros does
    # (veros/core/tke.py:54-65,185-225 + thermodynamics.py:99):
    #   - adiabatic N² over the caller's dz_half (= Veros dzw);
    #   - buoyancy-length growth allowance = dzt, Veros pass order;
    #   - TKE-diffusion face gradients over dzt, control volumes = dzw;
    #   - surface injection over 0.5·dzw_top (= -z_full_ref[0]·J).
    # Requires the caller to pass dz_ref/jacobian/dz_surface (the
    # k_profiles bridge does). These are outright metric bugs vs the
    # scheme's own reference — canonical in spirit — but gated behind this
    # flag so every existing default/legacy config stays BIT-IDENTICAL
    # (repo bit-identity doctrine); the Veros-faithful recipes opt in.
    veros_dz_slots: bool = False
    # ----- TKE positivity treatment (Veros tke.py:224-245) -----
    # ``"floor"`` (default, BIT-IDENTICAL legacy): the buoyancy sink is
    #   linearised IMPLICITLY (sign-aware split) and the solved TKE is
    #   floored at ``tke_background`` everywhere + ``tke_surface_min`` at
    #   the top interface. The floor erases the interior energy DEBT that
    #   the stratification sink runs up each step — a systematic spurious
    #   energy injection. On the Veros-faithful recipes this feeds a deep
    #   TKE reservoir (~1e-2 m²/s² vs Veros's 1e-4-class/negative;
    #   .physics-validator/tke_metric_fix/ re-ablation).
    # ``"veros_surface_correction"``: Veros's treatment — the buoyancy work
    #   ``P_b = -K_H·N²`` enters the RHS EXPLICITLY (Veros forc =
    #   K_diss_v − P_diss_v, both explicit), the dissipation linearisation
    #   uses ``sqrt(max(0, e))`` (tke.py:30), interior TKE MAY GO NEGATIVE
    #   (an energy debt; sqrt(max(0,e)) shuts the closure off there), and
    #   only the SURFACE level is clamped at zero (tke.py:238-245 — Veros
    #   records the clamp as ``tke_surf_corr``; legoESM's collapsed surface
    #   point is the topmost interior interface). No ``tke_background`` /
    #   ``tke_surface_min`` floors.
    positivity: str = "floor"
    # ----- K-from-TKE amplitude convention -----
    # ``"gaspar_sqrt2e"`` (default, BIT-IDENTICAL legacy):
    #   K_M = c_k·l_k·sqrt(2·max(e, tke_background)) — the Gaspar form.
    #   On the SIGNED-N² (Veros buoyancy-length) path this DOUBLE-COUNTS
    #   the sqrt(2): the buoyancy length already is mxl = √2·√e/√N̄
    #   (tke.py:34), so K_M comes out ×1.414 vs Veros everywhere the
    #   caps/floors don't bind — and P_s = K_M·S², K_H = K_M/Pr inherit it.
    # ``"veros_sqrte"``: Veros tke.py:73 — K_M = c_k·l_k·sqrt(max(0, e))
    #   (kappaM = c_k·mxl·sqrttke; sqrttke = sqrt(max(0, tke)), consistent
    #   with the negative-TKE energy debt of
    #   positivity="veros_surface_correction").
    kappa_convention: str = "gaspar_sqrt2e"
    # ----- TKE buoyancy-term timing (the Veros step-order option) -----
    # Veros assembles the TKE forcing from REALIZED dissipation diagnostics
    # of the SAME step, evaluated AFTER the implicit T/S vertical mixing
    # (veros.py:239-298 step order: set_tke_diffusivities[tau] → momentum →
    # thermodynamics{advect → vertmix → calc_eq_of_state(taup1) →
    # surf_densityf → diag_P_diss_v} → integrate_tke):
    #   forc = K_diss_v − P_diss_v          (tke.py:142)
    #   P_diss_v = kappaH·Nsqr[taup1]       (thermodynamics.py:385: the
    #     POST-MIXING N² — the implicit solve has already removed most of
    #     the instability, so convective TKE production is only the
    #     residual),
    #   P_diss_v[surface W] = −(g/ρ0)·forc_rho_surface (thermodynamics.py:
    #     386-388 + surf_densityf 304-317 — the surface buoyancy-flux TKE
    #     source/sink at taup1 surface T/S).
    #
    # ``"pre_mixing"`` (default, BIT-IDENTICAL legacy): the TKE budget is
    #   solved BEFORE the tracer implicit mixing, charging the PRE-mixing
    #   N² inside the same K-profile computation that feeds the tracer
    #   solve. With persistent surface-forced instability this charges the
    #   FULL instability every step — measured ~100× too much TKE in
    #   convecting columns vs the Veros equilibrium
    #   (.physics-validator/tke_metric_fix/).
    # ``"post_mixing_veros"``: the Veros ordering inside
    #   ``_apply_implicit_vertical_mixing`` — (1) K_M/K_H for the TRACER and
    #   MOMENTUM solves are derived from the CARRIED tke (Veros
    #   set_tke_diffusivities from tke[tau]; kappa consumed by tracers is
    #   the previous step's TKE); (2) the implicit T/S solve runs; (3) N² is
    #   recomputed from the MIXED T/S (the dzw-slotted adiabatic N²);
    #   (4) ONE backward-Euler TKE solve charges that POST-mixing N² plus
    #   the surface buoyancy-flux P_diss_v slot, with an internal surface-W
    #   row (Veros's surface half-volume point, seeded from the topmost
    #   interior interface — legoESM does not carry it across steps; the
    #   documented residual approximation). Requires ``prognostic=True``,
    #   ``veros_dz_slots=True``, ``n2_mode="adiabatic"`` and
    #   ``positivity="veros_surface_correction"`` (fail loudly otherwise).
    # DEFERRED Veros forc terms (oracle runs enable_conserve_energy=True):
    #   −P_diss_nonlin (cabbeling; needs dynamic-enthalpy diagnostics),
    #   −P_diss_adv (globally-redistributed advection dissipation), and the
    #   no-EKE branch ``+K_diss_gm + K_diss_h − P_diss_skew`` (tke.py:174-176;
    #   acc_basic class). K_diss_gm ≡ 0 in the oracle (no TEM friction).
    #   K_diss_h − P_diss_skew partially cancel; wiring them needs a
    #   TKE-side gate for the flux-form K_diss_h density (today gated by
    #   ``EKEConfig.source_kdiss_h``) + a constant-kappa call of
    #   ``compute_realized_gm_skew_conversion`` in the tendency — real
    #   plumbing, documented gap, not silently approximated.
    buoyancy_timing: str = "pre_mixing"
    # ----- TKE shear-production form (Veros realized K_diss_v) -----
    # ``"pre_solve"`` (default, BIT-IDENTICAL legacy): P_s = K_M·S² from the
    #   PRE-solve velocities — the parameterised explicit form.
    # ``"realized_veros"``: Veros friction.py:131-151 — the realized
    #   implicit-friction dissipation K_diss_v = κ·(∂u_new/∂z)·(∂u_old/∂z)
    #   summed over u and v, faces→centres at the interior interfaces. The
    #   old ocean-expert audit measured the pre-solve explicit K_M·S² as
    #   ~6.1× Veros's realized form on the same state. Only consulted under
    #   ``buoyancy_timing="post_mixing_veros"`` (the realized increments
    #   exist only after the friction solve); fail loudly otherwise.
    shear_production: str = "pre_solve"
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
    # ----- NEMO zdftke surface terms (Langmuir + sub-ML TKE penetration) -----
    # Faithful ports of NEMO 5.0.1 ``zdftke.F90``. BOTH are ON in NEMO's
    # ``namelist_ref`` defaults, hence active in the DINO and ORCA1 oracles;
    # both default OFF here (bit-identical legacy) and are enabled by the
    # NEMO-faithful recipes (DINO).
    # ``lc``: Langmuir-circulation TKE source (Axell 2002; NEMO ln_lc).
    #   Stokes drift from the surface stress (Axell Eq. 44 via |τ| =
    #   ρ_air·C_d·U₁₀²): ½W_lc² = ½·0.016²·|τ|/(ρ_air·C_d); LC depth h_lc
    #   from the cumulative-PE criterion (Axell Eq. 47, zdftke.F90:338-356);
    #   source u_s³·(lc_coeff·sin(πz/h_lc))³/h_lc added to the TKE RHS at
    #   interfaces shallower than h_lc (zdftke.F90:358-370).
    # ``etau_mode``: penetration of surface TKE below the mixed layer due to
    #   near-inertial waves (NEMO nn_etau): "none" (=0, default) |
    #   "below_ml" (=1): e(k) += etau_frac·e_sfc·exp(-z/h_tau) with
    #   e_sfc = max(emin0, ebb·|τ|/ρ0) (zdftke.F90:265,492-496); applied
    #   AFTER the TKE solve, BEFORE the K_M/K_H computation (NEMO step
    #   order: tke_tke ends with etau, then tke_avn computes the K's).
    # ``etau_htau_mode``: h_tau profile (NEMO nn_htau): "constant10m" (=0,
    #   10 m everywhere — the namelist_ref/DINO default) | "latitude" (=1,
    #   max(0.5, min(30, 45·|sin φ|)) m — requires ``lat_deg`` threading).
    lc: bool = False
    lc_coeff: float = 0.15               # NEMO rn_lc — LC vertical-velocity coefficient
    etau_mode: str = "none"              # "none" | "below_ml"  (NEMO nn_etau 0/1)
    etau_frac: float = 0.05              # NEMO rn_efr — fraction of surface TKE penetrating
    etau_htau_mode: str = "constant10m"  # "constant10m" | "latitude" (NEMO nn_htau 0/1)
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
    # ----- Lateral/vertical ADVECTION of the prognostic TKE field -----
    # Veros ``enable_tke_superbee_advection`` (global_4deg sets it True; the
    # ACC setup does not). Valid: "none" | "superbee"; anything else raises at
    # model-config construction AND at the dispatch site (dispatch hardening).
    #
    # ``"none"`` (default, BIT-IDENTICAL): no advection — not a single traced
    #   op is added (static Python gate in the model step).
    # ``"superbee"``: each step computes the W-grid superbee advective
    #   tendency of the CARRIED tke (Veros tke[tau]) from the PRE-STEP
    #   velocities (Veros u[tau] via calculate_velocity_on_wgrid) and applies
    #   it AFTER the implicit TKE solve with Adams-Bashforth-2 weights on the
    #   tendency history (Veros tke.py:286-323):
    #       tke ← tke + dt_tracer·((1.5+AB_eps)·dtke^n − (0.5+AB_eps)·dtke^{n-1})
    #   NOTE the dt's: the implicit TKE solve uses dt_mom (Veros tke.py:137)
    #   but the advection AB2 increment AND its uCFL weight use dt_tracer
    #   (Veros tke.py:318, advection.py:47) — they differ under asynchronous
    #   dt_mom≠dt_tracer stepping (global_4deg: 1800 s vs 86400 s). AB_eps is
    #   the model's ``ab2_epsilon`` (the one AB2 epsilon — not redefined).
    #   The tendency history is carried on ``state.dtke`` (zero on the first
    #   step, like Veros's zero-initialised dtke[taum1]). Requires
    #   ``prognostic=True`` (advecting a diagnostic TKE is a config error).
    advection_scheme: str = "none"
    # Timestep [s] used to derive the explicit-diffusion CFL ceiling
    # A_v_max = 0.25 * min(dz_k, dz_k+1)^2 / cfl_cap_dt_s on the MPAS path
    # (make_tke_profiles_mpas caps the diagnostic K_M / K_H; mirrors the KPP
    # MPAS bridge's KPPConfig.cfl_cap_dt_s exactly — same numerics parameter).
    # MUST be set to the ocean dynamics dt for the cap to be correct: a
    # value smaller than the real dt over-damps; larger risks instability.
    # Default 300.0 preserves the historical hard-coded estimate.
    cfl_cap_dt_s: float = 300.0


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
    # LMD94 Eq. 23 unresolved-shear variance V_t^2 carries a (-beta_T)^1/2
    # prefactor (beta_T = -0.2 fixed, App. B); applied EXPLICITLY in kpp.py so
    # Cv keeps its standard standalone value 1.6 (it does NOT absorb sqrt(0.2)).
    neg_beta_T: float = 0.2   # = -beta_T (LMD94 App. B; V_t^2 prefactor)
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
    # --- Monin-Obukhov similarity (Businger-Dyer) + u* proxy (#518 item 10) ---
    # Businger-Dyer MOST stability-function constants (Businger et al. 1971,
    # used by LMD94): the unstable (1 + 16|zeta|) and stable (1 + 5*zeta)
    # coefficients in the weakly-(un)stable velocity scales.  Fixed published
    # constants (like a_m/c_m above) — excluded from training.
    businger_unstable_coeff: float = 16.0   # (1 + 16|zeta|)^{1/4,1/2} weakly-unstable scale
    businger_stable_coeff: float = 5.0      # 1/(1 + 5*zeta) stable suppression
    # u_star proxy ratio when wind stress is absent: u* ~ ratio*|U_surface|
    # (~sqrt(C_d) drag-like closure knob).
    ustar_speed_ratio: float = 0.01
    # --- Langmuir turbulence (KPP-Langmuir wave-enhanced surface mixing) ---
    # Langmuir circulations (wind + Stokes-drift shear) enhance surface
    # boundary-layer mixing. Enhancement factor eps_L = sqrt(1 + C_L/La_t^2)
    # (>= 1) on the KPP velocity scales, where La_t is the turbulent Langmuir
    # number. Opt-in; default off is byte-identical to classical KPP.
    enable_langmuir: bool = False    # opt-in Langmuir enhancement
    langmuir_coeff: float = 0.08     # C_L in eps_L = sqrt(1 + C_L/La_t^2)
    langmuir_number_default: float = 0.3  # fallback La_t when no Stokes-drift input


class CATKEConfig(NamedTuple):
    """CATKE: Convective-Adjustment Turbulent-Kinetic-Energy vertical mixing.

    One-equation prognostic-TKE closure (Wagner et al. 2025, JAMES,
    doi:10.1029/2024MS004522), calibrated to a suite of large-eddy
    simulations. Distinct from :class:`TKEConfig` (Gaspar/Burchard): CATKE
    derives a DYNAMIC convective mixing length (Deardorff ``w*^3/Jb`` scaling
    — predicting both convective-layer depth AND timescale) and blends three
    Richardson-number regimes (high/low/negative Ri) with SEPARATE coefficients
    for momentum (u), tracers (c) and TKE (e).

    Diffusivities at interfaces:  ``K_X = l_X * w*`` with ``w* = sqrt(max(e, e_min))``.
    Mixing length:  ``l_X = min(H, max(sigma_X * l_stable, l_convective_X))`` with

        l_stable     = min(c_surface_shear*depth, c_bottom_shear*hab, w*/sqrt(N^2+))
        sigma_X(Ri)  = scale(Ri; c_un_X, c_lo_X, c_hi_X, c_ri_lower, c_ri_width)
        l_convective = c_conv_X * w*^3/(Jb+Jb_eps) * (1 - c_sheared_plume*Ri_f)  [convecting]
                     = c_entr_X * Jb/(w* N^2 + Jb_eps)                           [entraining]

    Dissipation:  ``eps = e * sqrt(|e|) / l_D`` (l_D uses the ``c_*_diss`` regime
    coefficients: ``l_stable / sigma_D`` max'd with the convective length).
    Surface TKE flux:  ``Q_e = -c_w_ustar*u*^3 - c_w_conv*wConv^3``.

    Field names map 1:1 to Oceananigans ``CATKEMixingLength`` / ``CATKEEquation``
    (Cˢ→c_surface_shear, Cᶜc→c_conv_c, CʰⁱD→c_hi_diss, ...); defaults are the
    LES-calibrated values from Wagner et al. (2025).
    """
    # --- stable (shear) mixing length: surface/bottom + stratification ---
    c_surface_shear: float = 1.131       # Cˢ  surface-distance coefficient
    c_bottom_shear: float = 0.28         # Cᵇ  bottom-distance coefficient
    c_ri_lower: float = 0.254            # CRi⁰ stability-function lower Ri
    c_ri_width: float = 1.02             # CRiᵟ stability-function width
    c_sheared_plume: float = 0.505       # Cˢᵖ sheared-convective-plume coefficient
    # --- per-variable shear-length coefficients (high / low / negative Ri) ---
    c_hi_u: float = 0.242                # Cʰⁱu momentum, high Ri
    c_lo_u: float = 0.361                # Cˡᵒu momentum, low Ri
    c_un_u: float = 0.370                # Cᵘⁿu momentum, negative Ri
    c_hi_c: float = 0.098                # Cʰⁱc tracers, high Ri
    c_lo_c: float = 0.369               # Cˡᵒc tracers, low Ri
    c_un_c: float = 0.572                # Cᵘⁿc tracers, negative Ri
    c_hi_e: float = 0.548                # Cʰⁱe TKE, high Ri
    c_lo_e: float = 7.863                # Cˡᵒe TKE, low Ri
    c_un_e: float = 1.447                # Cᵘⁿe TKE, negative Ri
    # --- per-variable convective / entrainment length coefficients ---
    c_conv_u: float = 3.705              # Cᶜu  momentum convective length
    c_entr_u: float = 0.0                # Cᵉu  momentum entrainment length
    c_conv_c: float = 4.793              # Cᶜc  tracer convective length
    c_entr_c: float = 0.112              # Cᵉc  tracer entrainment length
    c_conv_e: float = 3.642              # Cᶜe  TKE convective length
    c_entr_e: float = 0.0                # Cᵉe  TKE entrainment length
    # --- dissipation length (CATKEEquation) ---
    c_hi_diss: float = 0.579             # CʰⁱD dissipation, high Ri
    c_lo_diss: float = 1.604             # CˡᵒD dissipation, low Ri
    c_un_diss: float = 0.923             # CᵘⁿD dissipation, negative Ri
    c_conv_diss: float = 3.254           # CᶜD  dissipation convective length
    c_entr_diss: float = 0.0             # CᵉD  dissipation entrainment length
    # --- surface TKE flux ---
    c_w_ustar: float = 3.179             # Cᵂu★ shear-driven surface TKE flux
    c_w_conv: float = 0.383              # CᵂwΔ convective surface TKE flux
    # --- numerics floors / caps (NOT trainable) ---
    minimum_tke: float = 1.0e-9                     # e floor for w* background mixing
    minimum_convective_buoyancy_flux: float = 1.0e-11  # Jb_eps regulariser
    negative_tke_damping_time_s: float = 60.0       # damp spurious negative TKE
    # Diffusivity caps default to inf (= no clip), matching Oceananigans
    # CATKEVerticalDiffusivity; the unconditionally-stable implicit vertical
    # solve tolerates large K.  Set a finite value to clip if desired.
    maximum_viscosity: float = float("inf")           # K_u cap [m^2/s]
    maximum_tracer_diffusivity: float = float("inf")  # K_c cap [m^2/s]
    maximum_tke_diffusivity: float = float("inf")     # K_e cap [m^2/s]


class VerticalMixingConfig(NamedTuple):
    """Top-level vertical mixing configuration."""
    scheme: str = "constant"  # "constant", "richardson", "kpp", "tke", "catke", "none"
    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    kpp: KPPConfig = KPPConfig()
    tke: TKEConfig = TKEConfig()
    catke: CATKEConfig = CATKEConfig()
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
    # Internal wave-driven mixing (NEMO zdfiwm, de Lavergne 2020) is
    # ADDITIVE like tidal, but — unlike tidal — it contributes to BOTH the
    # tracer diffusivity and the momentum viscosity, so it is applied
    # inside ``k_profiles.compute_vertical_K_profiles`` (the implicit
    # vertical-mixing path), AFTER the primary closure — exactly NEMO's
    # zdfphy ordering (zdf_tke, then zdf_iwm adds onto avt/avs/avm).
    # ``make_vertical_mixing_physics`` / ``make_ocean_physics`` RAISE if
    # ``iwm.enabled=True`` (the explicit-tendency path cannot honour it),
    # mirroring the tidal guard above.  Default off ⇒ bit-exact legacy.
    iwm: IWMConfig = IWMConfig()
    # Double-diffusive mixing (NEMO zdfddm; Merryfield 1999) is ADDITIVE like
    # iwm, but contributes a SEPARATE salt (avs) vs heat (avt) diffusivity, so
    # it is applied in the implicit vertical-mixing path AFTER the primary
    # closure and routes the salinity solve through its own diffusivity
    # (``K_v + (avs - avt)``); momentum (avm) is untouched, matching zdfddm.
    # Requires ``implicit_vertical_mixing=True`` (the shared-K explicit/pair
    # path cannot carry avs != avt).  Default off ⇒ bit-exact legacy.
    ddm: DoubleDiffusionConfig = DoubleDiffusionConfig()
