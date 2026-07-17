"""Configuration for gravity wave drag schemes.

Provides configuration NamedTuples for:
1. Rayleigh: simple Rayleigh friction drag
2. Lindzen: smoothed Lindzen (1981) orographic GWD
3. McFarlane: smoothed McFarlane (1987) orographic GWD
4. Hines: Hines (1997) Doppler-spread parameterization
5. PrognosticSpectral: prognostic spectral GWD
6. MLEmulator: ML-based GWD emulator (Equinox MLP)
7. GravityWaveDragConfig: top-level selector

References
----------
- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
- McFarlane, N. A. (1987). The effect of orographically excited gravity
  wave drag on the general circulation of the lower stratosphere and
  troposphere. J. Atmos. Sci., 44, 1775-1800.
- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
  momentum deposition in the middle atmosphere. 1. Basic formulation.
  J. Atmos. Solar-Terr. Phys., 59, 371-386.
"""

from __future__ import annotations

import math
from typing import NamedTuple


__param_spec__ = {
    "E3SMBeresConfig": {
        "scheme_key": "atm.gwd.E3SMBeresConfig",
        "excluded": {
            "mfcc_uh_slope": "stand-in-table-only surrogate slope (default 0 = uh-independent); the real source spectrum is the offline mfcc table, not a trained scalar",
            "storm_speed_min": "ground-relative storm-speed floor used only to truncate the integer Doppler cell-speed CS; a discretisation threshold, not a continuum closure",
        },
        "params": {
            # --- convective source amplitude (Beres et al. 2004) ---
            "cf": {"units": "1", "bounds": (5.0, 60.0), "tunable_tier": 1, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM gw_convect_hcf", "shape": None},
            "al": {"units": "m", "bounds": (1.0e4, 1.0e6), "tunable_tier": 1, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM AL", "shape": None},
            "hdepth_scaling_factor": {"units": "1", "bounds": (0.25, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM hdepth_scaling_factor", "shape": None},
            # --- convective source triggering / search window ---
            "hdepth_min_km": {"units": "km", "bounds": (0.5, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM hdepth_min", "shape": None},
            "z_heat_max": {"units": "m", "bounds": (5.0e3, 4.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004) scheme default", "shape": None},
            "source_wind_p": {"units": "Pa", "bounds": (4.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_beres_src k700 source-wind level", "shape": None},
            # --- analytic stand-in spectrum surrogate (not Beres-faithful) ---
            "mfcc_peak": {"units": "Pa", "bounds": (1.0e-4, 1.0e-1), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "stand-in mfcc surrogate (E3SM mfcc table replacement)", "shape": None},
            "mfcc_c0": {"units": "m/s", "bounds": (5.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "stand-in mfcc surrogate phase-speed width", "shape": None},
            "mfcc_hdepth_growth": {"units": "1/km", "bounds": (0.0, 0.5), "tunable_tier": 3, "transform": "sigmoid", "category": "source_spectrum", "reference": "stand-in mfcc surrogate depth-growth slope", "shape": None},
        },
    },
    "E3SMCAMConfig": {
        "scheme_key": "atm.gwd.E3SMCAMConfig",
        "excluded": {
            "alpha_newtonian": "uniform Newtonian-cooling coefficient (default 0 = off); when used it is replaced by the E3SM height profile, not a single trained scalar",
            # dc is the phase-speed bin width but the frontal source quadrature
            # turns it into a Python integer sub-interval COUNT
            # (e3sm_cam._front_fav: n_sub = int(math.floor(dc/dca + 0.5)) - 1,
            # then jnp.arange(1, n_sub+1)). A traced trainable would hit int() on
            # a tracer / fix a static, non-differentiable quadrature shape. Not
            # trainable until the quadrature uses a fixed max grid with masking.
            "dc": "phase-speed bin width -> Python int quadrature count (jnp.arange shape) in _front_fav; not traceable",
        },
        "params": {
            # --- efficiency (primary amplitude knob) ---
            "effgw": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "efficiency", "reference": "CAM gw_drag effgw_oro/effgw_cm", "shape": None},
            # --- saturation / wave breaking ---
            "fcrit2": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Lindzen (1981)/McFarlane (1987); CAM fcrit2", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "wave_breaking", "reference": "Scinocca (2003); E3SM gw_common umcfac", "shape": None},
            # --- saturation floors ---
            "n2min": {"units": "1/s^2", "bounds": (1.0e-9, 1.0e-7), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "E3SM gw_prof N^2 floor", "shape": None},
            "taumin": {"units": "Pa", "bounds": (1.0e-12, 1.0e-8), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "E3SM gw_common minimum-stress floor", "shape": None},
            "ubmc2mn": {"units": "m^2/s^2", "bounds": (1.0e-3, 1.0e-1), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "E3SM gw_common (u-c)^2 floor", "shape": None},
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling", "shape": None},
            # --- GW-induced eddy diffusion ---
            "dback": {"units": "m^2/s", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "E3SM gw_common background diffusivity dback", "shape": None},
            "prndl": {"units": "1", "bounds": (0.05, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "E3SM gw_diffusion inverse Prandtl number", "shape": None},
            "egwd_max": {"units": "m^2/s", "bounds": (10.0, 500.0), "tunable_tier": 3, "transform": "sigmoid", "category": "saturation", "reference": "E3SM gw_diffusion eddy-diffusivity cap", "shape": None},
            "ediff_kbot_p": {"units": "Pa", "bounds": (3.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_diffusion kbotbg eddy-diffusion bottom level", "shape": None},
        },
    },
    "E3SMFrontalConfig": {
        "scheme_key": "atm.gwd.E3SMFrontalConfig",
        "excluded": {
            "front_spectrum_dc_resolution": "sub-bin c-grid quadrature spacing dca for the Gaussian integration; a numerical resolution of the spectrum, not a closure",
        },
        "params": {
            # --- frontal source amplitude / triggering ---
            "taubgnd": {"units": "Pa", "bounds": (1.0e-4, 1.0e-2), "tunable_tier": 1, "transform": "sigmoid", "category": "momentum_flux", "reference": "Charron & Manzini (2002); CAM taubgnd", "shape": None},
            "frontgfc": {"units": "K^2/(m^2 s)", "bounds": (1.0e-11, 1.0e-9), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "Charron & Manzini (2002); CAM frontgfc trigger threshold", "shape": None},
            "c0": {"units": "m/s", "bounds": (10.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "CAM gw_front Gaussian phase-speed width c0", "shape": None},
            # --- launch / trigger levels ---
            "launch_p": {"units": "Pa", "bounds": (3.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_front kbotbg launch interface", "shape": None},
            "front_p": {"units": "Pa", "bounds": (4.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_front kfront trigger level", "shape": None},
        },
    },
    "E3SMOrographicConfig": {
        "scheme_key": "atm.gwd.E3SMOrographicConfig",
        "excluded": {
            "sgh_default": "default subgrid-orography stddev (default 0 = no oro waves); a per-column boundary input supplied by the dataset, not a trained scalar",
        },
        "params": {
            # --- orographic source activation thresholds ---
            "oro_min_h": {"units": "m", "bounds": (1.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987); E3SM orohmin", "shape": None},
            "oro_min_wind": {"units": "m/s", "bounds": (0.5, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987); E3SM orovmin", "shape": None},
        },
    },
    "GWDMLEmulatorConfig": {
        "scheme_key": "atm.gwd.GWDMLEmulatorConfig",
        "excluded": {
            "norm_u": "input feature-normalisation wind scale; a preprocessing constant for the MLP, not a physical closure",
            "norm_T": "input feature-normalisation temperature scale; a preprocessing constant for the MLP, not a physical closure",
            "norm_z": "input feature-normalisation height scale; a preprocessing constant for the MLP, not a physical closure",
        },
        "params": {
        },
    },
    "HinesConfig": {
        "scheme_key": "atm.gwd.HinesConfig",
        "excluded": {
            "U_mag_floor": "wind-magnitude floor for the direction projection; a divide-by-zero safety floor, not a closure",
            "doppler_sharpness": "sigmoid sharpness of the smooth saturation gate; a differentiability/smoothing width, not a closure",
        },
        "params": {
            # --- launch source spectrum (Hines 1997) ---
            "total_rms_wind": {"units": "m/s", "bounds": (0.5, 10.0), "tunable_tier": 1, "transform": "sigmoid", "category": "source_spectrum", "reference": "Hines (1997) launch rms wind", "shape": None},
            # --- saturation / momentum-flux cap ---
            "Fmax": {"units": "Pa", "bounds": (0.01, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Hines (1997) saturation momentum-flux cap", "shape": None},
            # --- tendency limiters ---
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common umcfac no-reversal limiter", "shape": None},
        },
    },
    "LindzenConfig": {
        "scheme_key": "atm.gwd.LindzenConfig",
        "excluded": {
            "Fr_sharpness": "sigmoid sharpness of the saturation stress-ratio breaking transition (NOT a Froude-number transition); a differentiability/smoothing width, not a closure",
            "crit_level_sharpness": "sigmoid sharpness of the smooth critical-level filter; a differentiability/smoothing width, not a closure",
            "crit_level_floor": "signed source-projected wind U_proj (NOT a wind magnitude) at which the smooth critical-level filter is half-on; a smoothing/regulariser offset, not a closure",
            "N_ref": "declared but never read by lindzen_gwd (N is computed from the local theta gradient); phantom trainable — exposing it would offer a no-op gradient",
        },
        "params": {
            # --- orographic launch amplitude ---
            "h_topo": {"units": "m", "bounds": (50.0, 2000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "orographic", "reference": "Lindzen (1981) subgrid topographic height", "shape": None},
            # --- saturation / wave breaking ---
            "fcrit2": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Lindzen (1981) / E3SM fcrit2: critical Froude number squared scaling the saturation CAP VALUE (effkwv semantics, gw_common.F90:153)", "shape": None, "legacy_name": "critical_Fr"},
            # --- tendency limiters ---
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling (orographic)", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common umcfac no-reversal limiter", "shape": None},
        },
    },
    "McFarlaneConfig": {
        "scheme_key": "atm.gwd.McFarlaneConfig",
        "excluded": {
            "crit_level_floor": "signed source-projected wind U_proj (NOT a wind magnitude) at which the smooth critical-level filter is half-on; a smoothing/regulariser offset, not a closure",
            "crit_level_sharpness": "sigmoid sharpness of the smooth critical-level filter; a differentiability/smoothing width, not a closure",
            "min_wind_sharpness": "sigmoid sharpness of the smooth min-wind activation; a differentiability/smoothing width, not a closure",
            "softmin_sharpness": "sigmoid sharpness of the saturation-cap blend; a differentiability/smoothing width, not a closure",
        },
        "params": {
            # --- orographic launch amplitude / efficiency ---
            "G_0": {"units": "1", "bounds": (0.1, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987)/Palmer et al. (1986) launch-flux factor E", "shape": None},
            "efficiency": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "efficiency", "reference": "McFarlane (1987) breaking efficiency", "shape": None},
            "h_topo": {"units": "m", "bounds": (50.0, 2000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987) subgrid topographic height", "shape": None},
            # --- saturation / wave breaking ---
            "fcrit2": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "McFarlane (1987); E3SM fcrit2 Froude cap", "shape": None},
            "envelope_scale": {"units": "1", "bounds": (0.25, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "McFarlane (1987) vertical envelope scale", "shape": None},
            "directional_spread": {"units": "1", "bounds": (0.25, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "McFarlane (1987) multi-directional spreading factor", "shape": None},
            # --- source activation / clips / tendency limiters ---
            "min_wind": {"units": "m/s", "bounds": (0.5, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987) minimum source-level wind", "shape": None},
            "tau_max": {"units": "Pa", "bounds": (1.0, 50.0), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "McFarlane scheme launch-stress clip", "shape": None},
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling (orographic)", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common umcfac no-reversal limiter", "shape": None},
        },
    },
    "PrognosticSpectralConfig": {
        "scheme_key": "atm.gwd.PrognosticSpectralConfig",
        "excluded": {
            "breaking_sharpness": "sigmoid sharpness of the breaking transition; a differentiability/smoothing width, not a closure",
            "direction_sign_width": "tanh width of the smooth sign(c - U_launch) launch-fixed directional deposition factor; a differentiability/smoothing width, not a closure",
        },
        "params": {
            # --- launch source spectrum ---
            "launch_flux": {"units": "Pa", "bounds": (1.0e-4, 1.0e-2), "tunable_tier": 1, "transform": "sigmoid", "category": "momentum_flux", "reference": "prognostic-spectral scheme launch momentum flux", "shape": None},
            # --- saturation / wave breaking ---
            "breaking_threshold": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Lindzen (1981) Froude breaking threshold", "shape": None},
            # --- prognostic relaxation timescale ---
            "tau_decay": {"units": "s", "bounds": (3.6e3, 2.592e5), "tunable_tier": 2, "transform": "sigmoid", "category": "damping", "reference": "prognostic-spectral relaxation timescale", "shape": None},
        },
    },
    "RayleighConfig": {
        "scheme_key": "atm.gwd.RayleighConfig",
        "excluded": {
        },
        "params": {
            # --- boundary-layer Rayleigh drag ---
            "k_max": {"units": "1/s", "bounds": (1.0e-6, 1.0e-4), "tunable_tier": 1, "transform": "sigmoid", "category": "damping", "reference": "Held & Suarez (1994) boundary-layer drag rate", "shape": None},
            "sigma_b": {"units": "1", "bounds": (0.5, 0.95), "tunable_tier": 2, "transform": "sigmoid", "category": "damping", "reference": "Held & Suarez (1994) boundary-layer top sigma", "shape": None},
            # --- upper sponge ---
            "sponge_k": {"units": "1/s", "bounds": (1.0e-6, 1.0e-4), "tunable_tier": 1, "transform": "sigmoid", "category": "damping", "reference": "Rayleigh sponge drag rate", "shape": None},
            "sponge_top": {"units": "1", "bounds": (0.001, 0.1), "tunable_tier": 2, "transform": "sigmoid", "category": "damping", "reference": "Rayleigh sponge top sigma level", "shape": None},
        },
    },
}


class RayleighConfig(NamedTuple):
    """Configuration for Rayleigh friction drag.

    Fields
    ------
    k_max : float
        Maximum drag coefficient [1/s] (default 1/(1*86400)).
    sigma_b : float
        Boundary layer top sigma level (default 0.7).
    sponge_top : float
        Upper sponge sigma level (default 0.02).
    sponge_k : float
        Upper sponge drag coefficient [1/s] (default 1/(0.5*86400)).
    """
    k_max: float = 1.0 / 86400.0
    sigma_b: float = 0.7
    sponge_top: float = 0.02
    sponge_k: float = 1.0 / (0.5 * 86400.0)


class LindzenConfig(NamedTuple):
    """Configuration for smoothed Lindzen (1981) orographic GWD.

    Fields
    ------
    h_topo : float
        Sub-grid topographic height [m] (default 500).
    k_wave : float
        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    N_ref : float
        RESERVED / currently unused (default 0.01) — N is diagnosed from the
        local stratification (theta gradient via ``brunt_vaisala_n_full``), not
        from this field, so setting it does NOT change the launch/saturation
        stress.
    fcrit2 : float
        Critical Froude number squared scaling the saturation CAP VALUE
        (``tau_sat_eff = fcrit2*tau_sat`` — the oracle ``effkwv = kwv*fcrit2``
        semantics, E3SM gw_common.F90:153; Lindzen's limit scales with
        ``Fr_c²``). The breaking sigmoid activates at the fixed threshold
        ``tau_carry > tau_sat_eff`` and relaxes toward ``tau_sat_eff``; the
        saturation is SOFT (finite-sharpness sigmoid), NOT an exact hard cap.
        Default 1.0 (``Fr_c = 1``) caps at the exact Lindzen ``tau_sat``.
        Formerly named ``critical_Fr`` and wired only as the sigmoid
        ACTIVATION center with an unscaled relaxation target, which left the
        knob inert wherever ``tau_carry <= tau_sat``.
    Fr_sharpness : float
        Sigmoid sharpness for the saturation stress-ratio breaking transition
        (default 20.0). NOT a Froude-number transition (see ``fcrit2``).
    crit_level_sharpness : float
        Sigmoid sharpness [s/m] for the smooth critical-level filter
        (default 10.0).  The orographic wave (c = 0) is absorbed where the
        source-projected wind ``U_proj`` reverses sign (E3SM
        gw_common.F90:492 ``where ubmc*(ubi_above - c) > 0``); the earlier
        ``|U_proj|^3`` saturation alone gave no explicit critical-level
        absorption.  This smooth gate handles the differentiable absorption of
        the carried stress; the deposited drag additionally carries a HARD
        ``U_proj > 0`` positivity mask in the scheme body so the VECTOR sink
        ``u*du_dt + v*dv_dt <= 0`` is enforced STRICTLY (only the vector
        projection is guaranteed — componentwise ``du_dt*u`` can be >0 for an
        oblique wind; the smooth sigmoid alone is never identically zero).
    crit_level_floor : float
        SIGNED source-projected wind ``U_proj`` [m/s] at which the smooth
        critical-level gate ``sigmoid(sharpness*(U_proj - floor))`` is half-on
        (default 0.5) — NOT a wind magnitude; the gate ramps in as ``U_proj``
        drops toward +0.5 and only ASYMPTOTICALLY → 0 for reversed ``U_proj < 0``
        (the separate hard ``U_proj > 0`` mask zeroes the deposited drag exactly).
    tndmax_per_day : float
        Absolute ceiling on ``|du/dt|`` [m/s/day] (default 500.0, CAM
        ``tndmax`` for orographic-only; gw_common.F90:161).  Caps the
        ``stress/(rho*dz)`` accelerations that blow up where the launched
        stress saturates abruptly in a thin or weak-wind layer.
    umcfac : float
        Maximum fraction of ``|c - U_proj|`` (c = 0) the wind may change per
        step (default 0.5, CAM ``umcfac``; gw_common.F90:642), so a single
        step cannot reverse the wind past the phase speed.
    """
    h_topo: float = 500.0
    k_wave: float = 2.0 * math.pi / 100e3
    N_ref: float = 0.01
    fcrit2: float = 1.0
    Fr_sharpness: float = 20.0
    crit_level_sharpness: float = 10.0
    crit_level_floor: float = 0.5
    tndmax_per_day: float = 500.0
    umcfac: float = 0.5


class McFarlaneConfig(NamedTuple):
    """Configuration for smoothed McFarlane (1987) orographic GWD.

    Extends the Lindzen approach with explicit launch flux control
    and directional spreading.

    Fields
    ------
    h_topo : float
        Sub-grid topographic height [m] (default 500).
    k_wave : float
        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    G_0 : float
        Dimensionless launch-flux efficiency factor (default 0.5).
        The orographic launch stress is
        ``tau_0 = G_0 * rho * N * k * h^2 * U`` [Pa] — ``G_0`` is the
        dimensionless prefactor; the dimensional content comes from the
        thermodynamics / wind / wavenumber.  (Earlier docstring labeled
        ``G_0`` as Pa, which combined with the missing ``k_wave`` factor
        in the formula produced stress with the wrong units.)
    efficiency : float
        Breaking efficiency (default 0.5).
    min_wind : float
        Minimum wind for wave activity [m/s] (default 2.0).
    envelope_scale : float
        Vertical envelope scale (default 1.0).
    directional_spread : float
        Multi-directional spreading factor (default 1.0).
    min_wind_sharpness : float
        Sigmoid sharpness for the smooth ``U > min_wind`` activation
        (default 20.0).  Higher values approach a hard step.
    softmin_sharpness : float
        Sigmoid sharpness of the saturation-cap blend (default 50.0).  Feeds
        ``jax.nn.sigmoid(sat_sharpness*excess)`` to blend toward ``tau_sat``
        when ``tau_carry`` exceeds it; higher values give a sharper cap at the
        cost of larger gradients near the kink.
    tau_max : float
        Upper clip on launch stress [Pa] (default 10.0).  Operationally
        protects against runaway stress in pathological columns.
    fcrit2 : float
        Critical Froude number squared (default 1.0, CAM ``fcrit2``).  Used in
        the McFarlane (1987) / E3SM ``gw_oro_src`` displacement-height cap
        ``min(h_disp^2, fcrit2*(U/N)^2)`` (gw_oro.F90:166; ``h_disp = h`` at
        the default ``use_e3sm_hdsp=False``, E3SM's ``2*sgh`` when set) so the
        launched streamline-displacement amplitude saturates at the Fr = 1
        marginal-instability value rather than the raw orographic height.
    use_e3sm_hdsp : bool
        When ``True`` form the streamline displacement as E3SM does —
        ``hdsp = 2*sgh`` (gw_oro.F90:117), i.e. the launch cap becomes
        ``min((2h)^2, fcrit2*(U/N)^2)`` — closing the declared ~4x
        launch-amplitude departure (exactly 4x below the Froude cap, equal
        above it, 1-4x in the band between).  Default ``False`` keeps the
        legacy direct-``h`` displacement (``h_topo`` effectively a tuned
        amplitude).  Requires a real per-column ``h_topo_col`` (the wired
        ``subgrid_topo_stddev``): enabling it on the scalar ``config.h_topo``
        fallback raises, because quadrupling a uniform 500 m pseudo-mountain
        would silently quadruple drag over OCEAN (no landfrac factor in this
        scheme).  Retune ``G_0``/``directional_spread``/``tau_max`` before
        flipping in production (RCE/AMIP-gated).
    use_depth_averaged_source : bool
        When ``True`` the source ``rho``/``N``/``U`` and the wave direction
        come from E3SM's dp-weighted low-level averages over the levels the
        mountain penetrates (``hdsp > sqrt(zm[k]*zm[k+1])``, the shared
        ``oro_source.depth_averaged_oro_source``; gw_oro.F90:119-145), the
        launch wind is the depth-averaged magnitude, and — as in E3SM, where
        tau is held CONSTANT from the surface up to ``src_level``
        (gw_oro.F90:178-186) — NO drag deposits inside the source region.
        Default ``False`` keeps the legacy bottom-midpoint source (surface
        ``rho``/``N``/``U``, deposition allowed from the bottom level).  The
        displacement entering the penetration test follows ``use_e3sm_hdsp``
        (``2*h`` when set, ``h`` otherwise); the oracle-faithful combination
        is both flags ON.  Closes the declared surface-only-source departure
        (PBL-contaminated N/U; nocturnal weak surface wind killing a launch
        a real 700-1400 m average would sustain; spurious low-level
        deposition).  Behavioral -> RCE/AMIP-gated flip.
    crit_level_sharpness : float
        Sigmoid sharpness [s/m] for the smooth critical-level filter
        (default 10.0).  The orographic wave (phase speed ``c = 0``) has its
        critical level where the source-projected wind ``U_proj`` reverses sign
        (at ``U_proj = 0``); the smooth gate ramps in as ``U_proj`` drops toward
        ``crit_level_floor`` (default +0.5 m/s), i.e. marginally BEFORE the true
        reversal.  This reproduces the E3SM ``where ubmc*(ubi_above - c) > 0``
        test (gw_common.F90:492) that the earlier ``|U_proj|`` saturation
        stress silently dropped, letting waves transmit through and
        accelerate a reversed jet.  Higher values approach a hard cutoff.  This
        smooth gate handles the differentiable absorption of the carried stress;
        the deposited drag additionally carries a HARD ``U_proj > 0`` positivity
        mask in the scheme body so the VECTOR sink ``u*du_dt + v*dv_dt <= 0`` is
        enforced STRICTLY (componentwise ``du_dt*u`` can be >0 for oblique winds).
    crit_level_floor : float
        Signed source-projected wind ``U_proj`` [m/s] (NOT a magnitude) at which
        the smooth critical-level gate is half-on (default 0.5).  As ``U_proj``
        drops toward and below this, the gate attenuates the CARRIED stress (not
        ``tau_sat``), driving the propagated wave stress toward ~0.
    tndmax_per_day : float
        Absolute ceiling on ``|du/dt|`` [m/s/day] (default 500.0, CAM
        ``tndmax`` for orographic-only; gw_common.F90:161).  Caps
        physically-implausible accelerations in thin, low-density upper
        layers where ``stress / (rho*dz)`` blows up.
    umcfac : float
        Maximum fraction of ``|c - U_proj|`` the wind may change per step
        (default 0.5, CAM ``umcfac``; gw_common.F90:642).  Prevents the
        single-step tendency from reversing the wind past the phase speed.
    """
    h_topo: float = 500.0
    k_wave: float = 2.0 * math.pi / 100e3
    G_0: float = 0.5
    efficiency: float = 0.5
    min_wind: float = 2.0
    envelope_scale: float = 1.0
    directional_spread: float = 1.0
    min_wind_sharpness: float = 20.0
    softmin_sharpness: float = 50.0
    tau_max: float = 10.0
    fcrit2: float = 1.0
    use_e3sm_hdsp: bool = False
    use_depth_averaged_source: bool = False
    crit_level_sharpness: float = 10.0
    crit_level_floor: float = 0.5
    tndmax_per_day: float = 500.0
    umcfac: float = 0.5


class HinesConfig(NamedTuple):
    """Configuration for Hines (1997) Doppler-spread parameterization.

    Fields
    ------
    m_star : float
        Characteristic vertical wavenumber [1/m] (default 2*pi/2e3).
    total_rms_wind : float
        Total RMS gravity wave wind [m/s] (default 2.0).
    Fmax : float
        Saturation momentum flux cap [Pa] (default 0.1).
    doppler_sharpness : float
        Sigmoid sharpness for Doppler saturation (default 50.0).
    tndmax_per_day : float
        Absolute ceiling on ``|du/dt|`` [m/s/day] (default 400.0, CAM
        ``tndmax`` for spectral/non-orographic sources; gw_common.F90:158).
        Caps the ``Fmax/(rho*dz)`` accelerations that blow up in thin,
        low-density upper layers where a fixed momentum-flux cap is divided
        by a tiny ``rho*dz``.
    umcfac : float
        Maximum fraction of the local wind magnitude the deposition may
        remove per step (default 0.5, CAM ``umcfac``; gw_common.F90:642), so
        the single-step drag cannot reverse the wind.
    """
    m_star: float = 2.0 * math.pi / 2e3
    total_rms_wind: float = 2.0
    Fmax: float = 0.1
    doppler_sharpness: float = 50.0
    U_mag_floor: float = 0.1  # Wind-magnitude floor for projection [m/s]
    tndmax_per_day: float = 400.0
    umcfac: float = 0.5


class PrognosticSpectralConfig(NamedTuple):
    """Configuration for prognostic spectral GWD.

    Fields
    ------
    n_azimuths : int
        Number of azimuthal directions (default 4).
    n_wavenumbers : int
        Number of spectral bins (default 20).
    k_min : float
        Minimum horizontal wavenumber [1/m] (default 2*pi/100e3).
    k_max : float
        Maximum horizontal wavenumber [1/m] (default 2*pi/1e3).
    launch_flux : float
        Source momentum flux [Pa] (default 1e-3).
    breaking_threshold : float
        Froude threshold for wave breaking (default 1.0).
    breaking_sharpness : float
        Sigmoid sharpness for breaking transition (default 10.0).
    direction_sign_width : float
        Width [m/s] of the smooth ``tanh((c - U_launch)/width)`` directional
        sign factor in the stress deposition (default 1.0).  A differentiable
        stand-in for ``sign(c - U_launch)`` evaluated at the LAUNCH (surface)
        level and held FIXED with height (F-GWD-1); small against typical
        intrinsic phase speeds (O(1-100 m/s)) so the sign saturates to +-1
        except within ~1 m/s of a launch-level critical line.
    tau_decay : float
        Relaxation timescale for prognostic spectrum [s] (default 86400).
    thermal_tendency : bool
        If True, return the diagnosed kinetic-energy-to-thermal tendency.
        Set False for SCM realism sweeps where prognostic-spectral momentum
        deposition is exercised but its currently unvalidated energetics must
        not cool/heat the thermodynamic column.
    """
    n_azimuths: int = 4
    n_wavenumbers: int = 20
    k_min: float = 2.0 * math.pi / 100e3
    k_max: float = 2.0 * math.pi / 1e3
    launch_flux: float = 1e-3
    breaking_threshold: float = 1.0
    breaking_sharpness: float = 10.0
    direction_sign_width: float = 1.0
    tau_decay: float = 86400.0
    thermal_tendency: bool = True


class E3SMOrographicConfig(NamedTuple):
    """Configuration for the E3SM/CAM orographic source (``gw_oro_src``).

    Faithful to ``components/eam/src/physics/cam/gw_oro.F90``.

    Fields
    ------
    sgh_default : float
        Default subgrid orographic standard deviation [m] used when no
        per-column ``sgh`` is supplied (default 0.0 -> no oro waves).
        ``hdsp = 2 * sgh`` is the streamline displacement height.
    oro_min_h : float
        Minimum displacement height ``hdsp`` for orographic waves [m]
        (``orohmin`` in oracle, default 10).
    oro_min_wind : float
        Minimum source-level wind for orographic waves [m/s]
        (``orovmin`` in oracle, default 2).
    """
    sgh_default: float = 0.0
    oro_min_h: float = 10.0
    oro_min_wind: float = 2.0


class E3SMFrontalConfig(NamedTuple):
    """Configuration for the E3SM/CAM frontal source (``gw_cm_src``).

    Faithful to ``components/eam/src/physics/cam/gw_front.F90``.

    Fields
    ------
    taubgnd : float
        Background source strength [Pa] (default 1.5e-3, CAM ``taubgnd``).
    frontgfc : float
        Frontogenesis-function critical threshold [K^2/(m^2 s)]
        (default 1.0e-10, CAM ``frontgfc``).
    c0 : float
        Gaussian width in phase speed [m/s] (default 30.0, CAM ``c0``).
    launch_p : float
        Pressure [Pa] of the wave LAUNCH interface ``kbot`` (E3SM ``kbotbg``,
        the interface nearest 500 hPa; default 5.0e4).
    front_p : float
        Pressure [Pa] of the frontogenesis TRIGGER level ``kfront`` (E3SM
        ``kfront``, the level near 600 hPa where ``frontgf`` is tested;
        default 6.0e4).  E3SM tests the trigger at ``kfront`` but launches at
        ``kbot`` — these are NOT the same level.
    front_spectrum_dc_resolution : float
        Sub-bin c-grid spacing [m/s] for the Gaussian phase-speed
        quadrature in ``gw_front_init`` (``dca``); each phase-speed bin
        of width ``dc`` is integrated over ``nint(dc/dca)`` sub-intervals
        (default 0.1, E3SM ``gw_front.F90`` ``dca``).
    latitude_taper : bool
        Apply the ``cos(lat)`` polar taper to the frontal tendencies.  E3SM
        sets this BY DYCORE (gw_drag.F90:829-833: ``do_latitude_taper =
        .not. dycore_is('UNSTRUCTURED')``): ``True`` on structured lat-lon
        grids, ``False`` on the unstructured (SE-family) dycore — which
        dycore a production campaign ran is not provable from the vendored
        tree.  legoESM's
        cubed-sphere / icosahedral / MPAS grids correspond to the
        UNSTRUCTURED branch, so the E3SM-equivalent value there is
        ``False`` — the default ``True`` (legacy, matches E3SM structured)
        suppresses frontal drag toward the poles (→ 0), a first-order
        high-latitude difference.  Flip per grid family; behavioral →
        AMIP-gated.
    """
    taubgnd: float = 1.5e-3
    frontgfc: float = 1.0e-10
    c0: float = 30.0
    launch_p: float = 5.0e4
    front_p: float = 6.0e4
    front_spectrum_dc_resolution: float = 0.1
    latitude_taper: bool = True


class E3SMBeresConfig(NamedTuple):
    """Configuration for the E3SM/CAM Beres (2004) convective source.

    Faithful to ``components/eam/src/physics/cam/gw_convect.F90``
    (``gw_beres_src``) and the namelist defaults in ``gw_drag.F90``.

    The Beres source builds the launched phase-speed spectrum from the
    deep-convective heating profile: the heating depth ``hdepth`` and the
    maximum heating rate ``q0`` index an OFFLINE-generated mean-flux lookup
    table ``mfcc(hdepth, uh, c)`` (the ``newmfspectra*.nc`` /
    ``gw_drag_file`` netcdf, variable ``mfcc``).  That table is NOT bundled
    with this repository, so a documented analytic stand-in spectrum is used
    by default (``use_stand_in_table=True``); set ``mfcc_table`` to the real
    table array to recover bit-faithfulness with E3SM (see ``gw_beres_src``).

    Fields
    ------
    cf : float
        Heating-rate conversion factor ``CF`` (E3SM ``gw_convect_hcf``,
        default 20.0) used to scale the convective heating into the source
        amplitude ``q0 = CF * max(netdt)``.
    hdepth_scaling_factor : float
        Tunable multiplier on the diagnosed heating depth (E3SM
        ``hdepth_scaling_factor``, default 1.0).
    al : float
        Averaging length ``AL`` [m] in the source amplitude
        ``tau0 = mfcc * q0^2 / AL`` (E3SM ``AL = 1.0e5``, default 1.0e5).
    hdepth_min_km : float
        Minimum heating depth [km] for a non-zero source (E3SM ``2.5``,
        default 2.5).  Below this the column launches no convective waves.
    z_heat_max : float
        Maximum altitude [m] of the heating-depth search window (E3SM
        ``20000``, default 20000.0).
    storm_speed_min : float
        Storm-speed floor [m/s] below which the cell speed ``CS`` is zero
        (E3SM ``10``, default 10.0).
    source_wind_p : float
        Pressure [Pa] of the source-wind level ``k700`` (E3SM selects the
        level nearest 700 hPa; default 7.0e4).  The 700 hPa winds set the
        source direction and the storm speed.
    maxh : int
        Heating-depth table dimension (E3SM ``maxh = 20``); the diagnosed
        ``hdepth`` [km] is clamped to ``[1, maxh]`` and rounded to index it.
    maxuh : int
        Mean-wind table half-dimension (E3SM ``maxuh = 40``); ``uh`` is
        clamped to ``[-maxuh, maxuh]`` and rounded to index the table.
    use_stand_in_table : bool
        When ``True`` (default) use the documented analytic stand-in
        spectrum (a normalized Gaussian in phase speed, peak ``mfcc_peak``,
        width ``mfcc_c0``) so the scheme RUNS and is testable WITHOUT the
        offline E3SM table.  This is NOT bit-faithful to Beres (2004); it
        reproduces the ALGORITHM with a clearly-labelled placeholder table.
        Set ``False`` and supply ``mfcc_table`` for the real lookup.
    mfcc_peak : float
        Peak value of the analytic stand-in ``mfcc`` spectrum
        [kg m^-1 s^-2 per (K/s)^2 ... normalized], default 1.0e-2.  Only
        used when ``use_stand_in_table=True``.
    mfcc_c0 : float
        Phase-speed width [m/s] of the analytic stand-in Gaussian
        (default 30.0).  Only used when ``use_stand_in_table=True``.
    mfcc_hdepth_growth : float
        Fractional growth of the stand-in spectrum amplitude per km of
        heating depth (default 0.05), a documented monotone surrogate for
        the real table's deepening-convection dependence.  Only used when
        ``use_stand_in_table=True``.
    mfcc_uh_slope : float
        Fractional change of the stand-in spectrum amplitude per (m/s) of the
        heating-region mean wind ``uh`` (default 0.0 -> uh-independent).  The
        real E3SM ``mfcc`` table depends on ``uh`` (the source spectrum tilts
        with the background wind); the default stand-in is uh-independent for
        simplicity, but a non-zero slope makes the stand-in exercise the
        ``uh`` table-column lookup (used by the oracle harness to validate
        that the ``uh_idx``/``uh_col`` mapping is correct).  Only used when
        ``use_stand_in_table=True``.
    """
    cf: float = 20.0
    hdepth_scaling_factor: float = 1.0
    al: float = 1.0e5
    hdepth_min_km: float = 2.5
    z_heat_max: float = 20000.0
    storm_speed_min: float = 10.0
    source_wind_p: float = 7.0e4
    maxh: int = 20
    maxuh: int = 40
    use_stand_in_table: bool = True
    mfcc_peak: float = 1.0e-2
    mfcc_c0: float = 30.0
    mfcc_hdepth_growth: float = 0.05
    mfcc_uh_slope: float = 0.0


class E3SMCAMConfig(NamedTuple):
    """Configuration for the faithful E3SM/CAM gravity-wave scheme.

    Drives the ``gw_prof`` + ``gw_drag_prof`` spectral solver
    (``components/eam/src/physics/cam/gw_common.F90``) and selects the
    wave source(s).  All tunables match the CAM namelist / module
    parameters; magic numbers from ``gw_common.F90`` (``dback``,
    ``taumin``, ``umcfac``, ``ubmc2mn``, ``n2min``) are kept as named
    fields here so no literal lives in the JAX body (repo rule).

    Fields
    ------
    source : str
        Wave source: ``"orographic"`` (McFarlane c=0), ``"frontal"``
        (uniform Gaussian spectrum tied to frontogenesis), or
        ``"convective"`` (Beres 2004 source from the deep-convective
        heating profile).  The Beres convective source's full
        bit-faithfulness needs an offline ``mfcc`` lookup table that is not
        bundled with the repo; by default ``"convective"`` runs with a
        documented analytic stand-in spectrum (see ``E3SMBeresConfig``).
    pgwv : int
        Half-width of the phase-speed spectrum (waves -pgwv..pgwv).
        ``pgwv=0`` is the single-wave (orographic) path; ``pgwv>0`` the
        full spectral path.  (default 0)
    dc : float
        Phase-speed bin width [m/s] (default 2.5, CAM ``dc``).
    kwv : float
        Effective horizontal wavenumber [1/m] (default 2*pi/1e5).
    fcrit2 : float
        Critical Froude number squared (default 1.0, CAM ``fcrit2``).
    effgw : float
        Tendency efficiency factor applied to the drag (default 0.125,
        CAM orographic ``effgw_oro``; frontal uses ``effgw_cm``).
    alpha_newtonian : float
        Newtonian-cooling coefficient [1/s] used uniformly at all
        interfaces (default 0.0 -> off; CAM uses a height profile).
    dback : float
        Background diffusivity [m^2/s] (``gw_common`` parameter, 0.05).
    taumin : float
        Minimum non-zero stress [Pa] (``gw_common`` parameter, 1e-10).
    umcfac : float
        Max fraction of ``|c-u|`` the wind may change per step
        (``gw_common`` parameter, 0.5).
    ubmc2mn : float
        Minimum ``(u-c)^2`` [m^2/s^2] (``gw_common`` parameter, 0.01).
    tndmax_per_day : float
        Maximum wind tendency before efficiency [m/s/day]
        (``gw_common`` 400 for spectral, 500 for oro-only).
    n2min : float
        Minimum N^2 [1/s^2] (``gw_prof`` parameter, 1e-8).
    dttke_use_intrinsic : bool
        KE->thermal heating form for the spectral path.  ``False`` (default)
        matches the pinned E3SM-3.0.1 oracle ``dttke = sum_l c_l*gwut_l``
        (gw_common.F90:727).  ``True`` uses the newer CAM/EAM-trunk
        intrinsic-frequency form ``sum_l (c_l - ubm)*gwut_l``.
    use_discrete_ke_heating : bool
        Orographic heating closure.  ``False`` (default) keeps the
        continuous-rate identity ``dT/dt = -(u*du + v*dv)/c_pd``.  ``True``
        uses the E3SM driver-level DISCRETE-step closure
        ``dT/dt = -(du*(u + 0.5*dt*du) + dv*(v + 0.5*dt*dv))/c_pd``
        (gw_drag.F90:908-913, default no-energy-fix branch), which returns
        exactly the discrete resolved-KE change as heat so the discrete
        column energy budget closes; the continuous form over-heats by
        ``0.5*dt*(du^2+dv^2)/c_pd`` per step.  Orographic source only.
    use_newtonian_profile : bool
        When ``True`` use the E3SM height-dependent Newtonian-cooling
        profile (``alpha0``/``palph`` from gw_drag.F90, interpolated to the
        column interface pressures) in the spectral saturation/diffusivity
        instead of the single uniform ``alpha_newtonian``.  ``False``
        (default) keeps the uniform value so the clean oracle comparison is
        unchanged.  E3SM uses the profile for spectral sources and a tiny
        floor (1e-6 1/s) for orographic-only.
    use_e3sm_spectral_heating : bool
        E3SM-faithful spectral thermal term (default ``False`` = legacy).
        E3SM's spectral (``ngwv > 0``) ``gw_drag_prof`` UNCONDITIONALLY
        (a) band-limits ``dttke`` to midpoints ``ktop+1..kbotbg``
        (gw_common.F90:726-728; ``ktop = 0``, ``kbotbg`` = the interface
        above 500 hPa) and (b) adds the dse-diffusion heating ``dttdf``
        (``ttgw = dttke + dttdf``, :721,731) — with NO u/v diffusion
        (``egwdffi`` is exported only as the EKGWSPEC diagnostic; E3SM
        never diffuses u/v with it).  ``True`` applies both.  The legacy
        default sums ``dttke`` over ALL levels (a deep Beres source
        deposits ground-relative heating below 500 hPa that E3SM does not)
        and omits ``dttdf``.  Composes with ``do_eddy_diffusion`` (dttdf
        added exactly once).  Behavioral -> AMIP-gated flip.
    do_eddy_diffusion : bool
        STANDALONE ADDITION (default ``False``; spectral path only; NO
        E3SM analog for the momentum part): diffuse u/v through the GW
        eddy diffusivity ``egwdffi`` — E3SM never applies this anywhere —
        and add the ``dttdf`` dse-diffusion heating.  Kept for standalone
        use so the GW momentum eddy flux is not silently dropped when no
        host boundary-layer scheme consumes an exported diffusivity.  For
        the E3SM-faithful thermal term WITHOUT the momentum addition use
        ``use_e3sm_spectral_heating``.
    do_energy_conservation : bool
        When ``True`` (spectral path only) apply the C.-C. Chen column
        momentum & energy fixer (``momentum_energy_conservation``,
        gw_common.F90) so the column total-energy budget self-closes to
        machine precision.  ``False`` (default) leaves the raw tendencies —
        NOTE this is a DEPARTURE from the oracle's shipped behavior: E3SM
        v3.0.1 calls the fixer UNCONDITIONALLY after each spectral
        ``gw_drag_prof`` (Beres gw_drag.F90:800, CM :863); flip owed after
        AMIP validation.
    prndl : float
        Inverse Prandtl number for the GW eddy diffusivity (E3SM
        ``prndl = 0.25``, gw_diffusion.F90).
    egwd_max : float
        Cap on the GW eddy diffusivity at interfaces [m^2/s] (E3SM
        ``150``, gw_diffusion.F90).
    ediff_kbot_p : float
        Pressure [Pa] of the eddy-diffusion bottom level ``kbotbg`` (E3SM
        selects the interface nearest 500 hPa; default 5.0e4).
    orographic : E3SMOrographicConfig
        Orographic source sub-config.
    frontal : E3SMFrontalConfig
        Frontal source sub-config.
    beres : E3SMBeresConfig
        Convective (Beres 2004) source sub-config.
    """
    source: str = "orographic"
    pgwv: int = 0
    dc: float = 2.5
    kwv: float = 2.0 * math.pi / 1.0e5
    fcrit2: float = 1.0
    effgw: float = 0.125
    alpha_newtonian: float = 0.0
    dback: float = 0.05
    taumin: float = 1.0e-10
    umcfac: float = 0.5
    ubmc2mn: float = 0.01
    tndmax_per_day: float = 400.0
    n2min: float = 1.0e-8
    dttke_use_intrinsic: bool = False
    use_discrete_ke_heating: bool = False
    use_e3sm_spectral_heating: bool = False
    use_newtonian_profile: bool = False
    do_eddy_diffusion: bool = False
    do_energy_conservation: bool = False
    prndl: float = 0.25
    egwd_max: float = 150.0
    ediff_kbot_p: float = 5.0e4
    orographic: E3SMOrographicConfig = E3SMOrographicConfig()
    frontal: E3SMFrontalConfig = E3SMFrontalConfig()
    beres: E3SMBeresConfig = E3SMBeresConfig()


class GWDMLEmulatorConfig(NamedTuple):
    """Configuration for ML-based GWD emulator.

    Fields
    ------
    n_input : int
        Number of input features per level (default 7).
    n_hidden : int
        Hidden layer width (default 128).
    n_layers : int
        Number of MLP layers (default 3).
    n_output : int
        Number of output tendencies per level (default 3).
    seed : int
        Random seed for initialization (default 0).
    use_residual : bool
        Scale outputs for residual learning (default True).
    """
    n_input: int = 7
    n_hidden: int = 128
    n_layers: int = 3
    n_output: int = 3
    seed: int = 0
    use_residual: bool = True
    norm_u: float = 30.0     # Wind scale [m/s] for u, v normalization
    norm_T: float = 300.0    # Temperature scale [K]
    norm_z: float = 30000.0  # Height scale [m]


class GravityWaveDragConfig(NamedTuple):
    """Top-level gravity wave drag configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active GWD scheme: "rayleigh", "lindzen", "mcfarlane",
        "hines", "prognostic_spectral", "e3sm_cam", "ml_emulator",
        or "none".
    rayleigh : RayleighConfig
        Configuration for Rayleigh friction scheme.
    lindzen : LindzenConfig
        Configuration for Lindzen orographic scheme.
    mcfarlane : McFarlaneConfig
        Configuration for McFarlane orographic scheme.
    hines : HinesConfig
        Configuration for Hines Doppler-spread scheme.
    prognostic_spectral : PrognosticSpectralConfig
        Configuration for prognostic spectral scheme.
    e3sm_cam : E3SMCAMConfig
        Configuration for the faithful E3SM/CAM gw_drag_prof scheme.
    ml_emulator : GWDMLEmulatorConfig
        Configuration for ML emulator scheme.
    """
    scheme: str = "none"
    rayleigh: RayleighConfig = RayleighConfig()
    lindzen: LindzenConfig = LindzenConfig()
    mcfarlane: McFarlaneConfig = McFarlaneConfig()
    hines: HinesConfig = HinesConfig()
    prognostic_spectral: PrognosticSpectralConfig = PrognosticSpectralConfig()
    e3sm_cam: E3SMCAMConfig = E3SMCAMConfig()
    ml_emulator: GWDMLEmulatorConfig = GWDMLEmulatorConfig()
