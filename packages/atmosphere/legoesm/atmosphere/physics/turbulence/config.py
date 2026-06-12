"""Configuration for atmospheric turbulence / boundary layer schemes.

Provides configuration NamedTuples for:
1. Surface layer: bulk aerodynamic surface fluxes
2. Smagorinsky: constant eddy diffusivity (simplest baseline)
3. Louis (1979): stability-dependent diffusion
4. TKE / Mellor-Yamada 2.5: prognostic TKE closure
5. CLUBB-lite: higher-order closure skeleton
6. Holtslag-Boville: nonlocal K-profile with counter-gradient
7. YSU: nonlocal K-profile with entrainment flux
8. EDMF: eddy-diffusivity mass-flux unified framework
9. ML Turbulence Emulator: Equinox MLP surrogate
10. Top-level TurbulenceConfig that selects the active scheme.

References
----------
- Louis, J.-F. (1979). A parametric model of vertical eddy fluxes in the
  atmosphere. Boundary-Layer Meteorol., 17, 187-202.
- Mellor, G. L., & Yamada, T. (1982). Development of a turbulence closure
  model for geophysical fluid problems. Rev. Geophys., 20, 851-875.
- Holtslag, A. A. M., & Boville, B. A. (1993). Local versus nonlocal
  boundary-layer diffusion in a global climate model. J. Climate, 6,
  1825-1842.
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon. Wea.
  Rev., 134, 2318-2341.
- Siebesma, A. P., et al. (2007). A combined eddy-diffusivity mass-flux
  approach for the convective boundary layer. J. Atmos. Sci., 64, 1230-1248.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # avoid a config <-> clubb import cycle at runtime
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig


class SurfaceLayerConfig(NamedTuple):
    """Configuration for bulk aerodynamic surface fluxes.

    Fields
    ------
    z0 : float
        Roughness length [m] (default 1e-4).
    Cd_neutral : float
        Neutral drag coefficient (default 1.5e-3).
    Ch_neutral : float
        Neutral heat transfer coefficient (default 1.5e-3).
    bulk_scheme : str
        Bulk flux algorithm: "constant", "coare3", "large_yeager"
        (default "constant").
    z_ref : float
        Reference height for MOST bulk formulas [m] (default 10.0).
    bulk_n_iter : int
        Number of MOST iterations (default 5).
    """
    z0: float = 1e-4
    Cd_neutral: float = 1.5e-3
    Ch_neutral: float = 1.5e-3
    bulk_scheme: str = "constant"
    z_ref: float = 10.0
    bulk_n_iter: int = 5


class SmagorinskyConfig(NamedTuple):
    """Configuration for the Smagorinsky–Lilly turbulence closure.

    Deformation-based eddy viscosity
    ``K_m = (C_s · l)^2 · |S| · √(max(0, 1 − Ri/Pr_t))`` with ``K_h =
    K_m / Pr_t`` (Smagorinsky 1963; Lilly 1962 buoyancy correction).

    Fields
    ------
    C_s : float
        Smagorinsky constant (dimensionless); atmospheric value ~0.1–0.25
        (default 0.2).
    l_mix_max : float
        Blackadar (1962) asymptotic mixing length [m] used to build the
        length scale ``l = κz / (1 + κz / l_mix_max)`` (default 100.0;
        shared with the other turbulence closures).
    Pr_t : float
        Turbulent Prandtl number; Kh = Km / Pr_t (default 1.0).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    C_s: float = 0.2
    l_mix_max: float = 100.0
    Pr_t: float = 1.0
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class LouisConfig(NamedTuple):
    """Configuration for Louis (1979) stability-dependent turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Ri_crit : float
        Critical *bulk* Richardson number for the PBL-height diagnosis
        (threaded into ``PBLHeightConfig.Ri_crit``; default 0.25).  The
        Louis stability functions themselves are smooth in the gradient
        Ri and use no critical cutoff.
    b_heat_ratio : float
        Ratio of the heat stability-function coefficient to the momentum
        one, ``b_h / b_m`` (Louis, Tiedtke & Geleyn 1982 use 3b for heat
        vs 2b for momentum ⇒ 1.5).  Gives a stratification-dependent
        turbulent Prandtl number Pr_t = K_m/K_h (>1 stable, <1 unstable);
        ``1.0`` recovers the Louis (1979) ``f_h = f_m`` simplification.
        Momentum K_m is independent of this ratio (default 1.5).  Must be
        positive: like ``b_louis``/``d_louis`` it appears in the stable
        denominator ``1 + 2·b_h·Ri/√(1+d·Ri)``, which a negative value
        could drive through zero (singular K_h).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Ri_crit: float = 0.25
    # Louis stability function coefficients (Holtslag & De Bruin 1988)
    b_louis: float = 5.0
    c_louis: float = 16.6   # Updated from 5.0 (Louis 1979) to 16.6
    d_louis: float = 5.0
    b_heat_ratio: float = 1.5  # b_h/b_m (LTG82: 3b heat vs 2b momentum)
    blend_ri_sharpness: float = 100.0  # sigmoid sharpness [1/Ri] for stable/unstable blend
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class TKEConfig(NamedTuple):
    """Configuration for prognostic TKE / Mellor-Yamada 2.5 turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Ck : float
        TKE -> Km coefficient (default 0.1).
    Ce : float
        TKE dissipation coefficient (default 0.19).
    tke_min : float
        Minimum TKE [m^2/s^2] (default 1e-6).
    Pr_t : float
        Turbulent Prandtl number (default 0.33).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Ck: float = 0.1
    Ce: float = 0.19
    tke_min: float = 1e-6
    Pr_t: float = 0.33
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class MYNN25Config(NamedTuple):
    """Configuration for the MYNN-2.5 turbulence scheme (Nakanishi & Niino 2009).

    Default constants are taken from NN09 Table 1 / eq. 66 and match
    jax_scm's MYNNParams so the oracle-driven SCM benchmarks (GABLS1,
    Wangara, Ekman) can run with bit-equivalent closure coefficients.

    Fields
    ------
    A1, A2 : float
        Stability-function coefficients (momentum, heat).
    B1 : float
        Master length-scale coefficient.  Surface boundary value
        ``qke_sfc = B1^(2/3) · u*²`` (MY82 eq. 54) flows from this.
    B2 : float
        Dissipation length-scale coefficient.
    C1, C2, C3, C4, C5 : float
        Pressure-covariance / return-to-isotropy coefficients.  ``C4`` is
        the cross-correlation coefficient (unused at level 2.5; kept for
        symmetry with the full NN09 closure).
    gamma1 : float
        Critical-flux-Richardson-number numerator coefficient
        (NN09 below eq. A4).
    tke_min : float
        Minimum qke (= 2·TKE) [m²/s²] for numerical safety.
    surface : SurfaceLayerConfig
        Surface-layer (bulk-flux) configuration.
    """
    A1: float = 1.18
    A2: float = 0.665
    B1: float = 24.0
    B2: float = 15.0
    C1: float = 0.137
    C2: float = 0.75
    C3: float = 0.352
    C4: float = 0.0
    C5: float = 0.2
    gamma1: float = 0.235
    tke_min: float = 1e-10
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class CLUBBLiteConfig(NamedTuple):
    """Configuration for CLUBB-lite higher-order closure.

    Simplified implementation of the Cloud Layers Unified By Binormals
    (CLUBB) model. Carries five prognostic second moments (w'², θ_l'²,
    r_t'², w'θ_l', w'r_t') and diagnoses cloud fraction from a Gaussian
    PDF of the saturation deficit.

    References
    ----------
    - Golaz, J.-C., Larson, V. E., & Cotton, W. R. (2002). A PDF-based
      model for boundary layer clouds. Part I. J. Atmos. Sci., 59, 3540.
    - Larson, V. E., & Golaz, J.-C. (2005). Using Assumed PDF shapes.
      J. Atmos. Sci., 62, 3620-3649.

    Fields
    ------
    C1 : float
        Pressure scrambling / return-to-isotropy coeff for w'² (default 4.0).
    C4 : float
        Pressure scrambling for flux moments w'θ_l', w'r_t' (default 3.0).
    C5 : float
        Scalar variance dissipation rate (default 3.0).
    C_eps : float
        TKE dissipation coefficient (default 0.19).
    C_K : float
        Diffusivity coefficient: Km = C_K * l * sqrt(w'²) (default 0.4).
    Pr_t : float
        Turbulent Prandtl number (default 0.33).
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    tke_min : float
        Minimum TKE (w'²) [m²/s²] (default 1e-6).
    var_min : float
        Minimum scalar variance [K² or (kg/kg)²] (default 1e-12).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    C1: float = 4.0
    C4: float = 3.0
    C5: float = 3.0
    C_eps: float = 0.19
    C_K: float = 0.4
    Pr_t: float = 0.33
    l_mix_max: float = 100.0
    tke_min: float = 1e-6
    var_min: float = 1e-12
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class HoltslagBovilleConfig(NamedTuple):
    """Configuration for Holtslag-Boville nonlocal K-profile turbulence.

    Faithful to the E3SM/CAM ``hb_diff.F90`` (``eddy_scheme='HB'``) and
    ``pbl_utils.F90`` oracle.  Default values reproduce the oracle's
    hardcoded ``parameter`` block; see
    ``.physics-validator/holtslag_boville/static.md`` for the block-by-block
    map.  All transition sharpnesses replace the oracle's hard ``if``
    switches with smooth sigmoid blends (differentiability) and default to
    values that recover the oracle within a stated tolerance.

    Oracle constants (CAM names -> here)
    ------------------------------------
    ricr=0.3 -> Ri_crit ; betam=15 -> betam ; betah=15 -> betah ;
    betas=5 -> betas ; fakn=7.2 -> fakn ; fak=8.5 -> fak ;
    sffrac=0.1 -> sffrac ; binm=betam*sffrac, binh=betah*sffrac (derived) ;
    ml2=30^2 -> ml_free (=30 m length scale) ; zkmin=0.01 -> kvf_min ;
    fac=100 -> pblh_ustar_fac ; ustar_min=0.01 -> ustar_min ;
    700 -> pblh_mech_coeff ; free-atm f(Ri) coeffs (18,10,8).

    Fields
    ------
    Ri_crit : float
        Critical bulk Richardson number ``ricr`` (oracle 0.3).
    betam, betah, betas : float
        Monin-Obukhov gradient-function constants (oracle 15, 15, 5).
    fakn : float
        Constant in the turbulent Prandtl number / countergradient
        (oracle 7.2).
    fak : float
        Constant in the surface temperature/humidity excess (oracle 8.5).
    sffrac : float
        Surface-layer fraction of the boundary layer (oracle 0.1).
    ml_free : float
        Free-atmosphere mixing length [m] (oracle 30 m -> ml2=900).
    kvf_min : float
        Floor on the free-atmosphere diffusivity ``zkmin`` [m^2/s]
        (oracle 0.01).
    pblh_ustar_fac : float
        Mechanical term ``fac`` in the bulk-Ri ``vvk`` (oracle 100).
    pblh_mech_coeff : float
        Minimum-mechanical-mixing-depth coefficient ``h>=c*u*``
        (oracle 700).
    pblmaxp : float
        Maximum PBL depth in pressure units [Pa].  The bulk-Ri crossing
        search is limited to levels with ABSOLUTE pressure ``p >= pblmaxp``
        and the no-crossing fallback height is the top of that search region
        (oracle ``npbl`` counts levels with ``pref_mid(k) >= pblmaxp`` and
        the fallback is ``z(pverp-npbl)``; ``pblmaxp = 4e4 Pa`` = 400 hPa,
        an absolute threshold, NOT relative to the surface).
    cloud_pbl_floor_m : float
        Lowest-layer "marine-stratus ventilation" PBL floor [m]: the oracle
        unconditionally sets ``pblh = max(pblh, zi(pver) + 50)`` (the test
        ``cldn(:,pver) >= 0`` is always true), i.e. the PBL top is at least
        the top interface of the lowest model layer plus 50 m (oracle 50).
    ustar_min : float
        Floor on friction velocity [m/s] (oracle 0.01).
    free_ri_unstable_coeff : float
        Coefficient in the unstable free-atm f(Ri)=sqrt(1-c*Ri) (oracle 18).
    free_ri_stable_c1, free_ri_stable_c2 : float
        Stable free-atm f(Ri)=1/(1+c1*Ri*(1+c2*Ri)) (oracle 10, 8).
    pbl_crossing_sharpness : float
        Sigmoid sharpness [1/Ri] selecting the lowest Ri_crit crossing in
        the smooth PBL-height diagnostic (replaces the oracle's hard
        first-crossing scan).  RESIDUAL-GAP NOTE: a crossing whose upper
        level sits exactly ON ``ricr`` (rino_hi == ricr) gets ~0.5 weight
        rather than 1 -- the unavoidable price of a differentiable
        approximation to the oracle's hard step.  For real columns rino
        jumps by O(1) across the crossing so the weight is ~1 and the match
        is exact; raise this sharpness to shrink the on-threshold residual
        at the cost of a steeper gradient (default 100).
    sfc_blend_sharpness : float
        Sigmoid sharpness [1/(z/h)] for the surface-layer vs outer-layer
        blend at ``zh=sffrac`` (replaces oracle hard switch).
    cgs_gate_sharpness : float
        Sigmoid sharpness [1/(z/h)] for the countergradient in-PBL gate at
        the lower full level (oracle ``z(k) < pblh`` is a hard step);
        sharper than the K-profile gate so cgs matches the oracle at the
        PBL-top interface (default 400).
    stable_blend_sharpness : float
        Sigmoid sharpness [1/(z/L)] for the stable ``zl<=1`` vs ``zl>1``
        blend (replaces oracle hard switch).
    unstable_blend_sharpness : float
        Sigmoid sharpness [s^3/m^2] for the unstable (kbfs>0) vs stable
        (kbfs<=0) regime blend on the surface buoyancy flux.
    unstable_kbfs_threshold : float
        Positive offset [m^2/s^3] biasing the unstable indicator so exactly
        neutral kbfs=0 maps to the STABLE branch (oracle ``unstbl = kbfs >
        0`` is a strict inequality).  Default 1e-6 (~ 1e-3 W/m^2 of buoyancy
        flux).  Paired with ``unstable_blend_sharpness=1e7`` this gives
        ``s*threshold = 10``, so kbfs=0 -> sigmoid(-10) ~ 5e-5 (firmly
        stable) while any kbfs >= 2e-6 (~ 2e-3 W/m^2, negligible) ->
        sigmoid(+10) ~ 1 (unstable) -- as close to the oracle strict ``> 0``
        as a smooth-everywhere indicator allows.
    arg_floor : float
        Smooth floor on the ``(1-beta*zl)`` MO arguments so the cube-root
        / sqrt stay real and their gradients finite (default 0.01).
    surface : SurfaceLayerConfig
        Surface-layer (bulk-flux) parameters.
    """
    # --- oracle physical constants ---
    Ri_crit: float = 0.3
    betam: float = 15.0
    betah: float = 15.0
    betas: float = 5.0
    fakn: float = 7.2
    fak: float = 8.5
    sffrac: float = 0.1
    ml_free: float = 30.0
    kvf_min: float = 0.01
    pblh_ustar_fac: float = 100.0
    pblh_mech_coeff: float = 700.0
    pblmaxp: float = 4.0e4
    cloud_pbl_floor_m: float = 50.0
    ustar_min: float = 0.01
    free_ri_unstable_coeff: float = 18.0
    free_ri_stable_c1: float = 10.0
    free_ri_stable_c2: float = 8.0
    # --- smooth-blend sharpnesses (replace oracle hard switches) ---
    pbl_crossing_sharpness: float = 100.0
    sfc_blend_sharpness: float = 80.0
    cgs_gate_sharpness: float = 400.0
    stable_blend_sharpness: float = 20.0
    unstable_blend_sharpness: float = 1.0e7
    unstable_kbfs_threshold: float = 1.0e-6
    arg_floor: float = 0.01
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class YSUConfig(NamedTuple):
    """Configuration for YSU (Yonsei University) PBL turbulence scheme.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Pr_t : float
        Turbulent Prandtl number (default 1.0).
    entrainment_coeff : float
        Entrainment coefficient at PBL top (default 0.2).
    Ri_crit : float
        Critical Richardson number (default 0.25).
    pbl_smooth_sharpness : float
        Sigmoid sharpness for smooth PBL-top detection (default 20.0).
    louis_b : float
        Louis (1982) stability-function ``b`` constant
        (default 5.0).  Used in YSU local-Ri f_stable / f_unstable.
    louis_c : float
        Louis (1982) UNSTABLE-branch denominator coefficient
        (default 5.0).  Multiplies the ``b · l_mix² · sqrt(|Ri|)``
        denominator term in f_unstable.  The repository's louis.py
        distinguishes this from louis_d; YSU now follows the same
        convention.
    louis_d : float
        Louis (1982) STABLE-branch sqrt coefficient (a.k.a. ``b'``)
        (default 5.0).  Appears only in ``sqrt(1 + d · Ri)`` of
        f_stable.
    blend_ri_sharpness : float
        Sigmoid sharpness for stable / unstable blend in
        Richardson-number space (default 100.0 1/Ri).
    countergrad_coeff : float
        Nonlocal countergradient coefficient ``b`` in
        ``γ_c = b·(w'θ')_0 / (w_*·h)`` (Troen & Mahrt 1986; Hong et
        al. 2006).  Drives YSU's defining nonlocal upward heat
        transport in the convective BL (default 6.5).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Pr_t: float = 1.0
    entrainment_coeff: float = 0.2
    entrainment_width_frac: float = 0.3  # Gaussian entrainment width as fraction of h_pbl
    Ri_crit: float = 0.25
    pbl_smooth_sharpness: float = 20.0
    louis_b: float = 5.0
    louis_c: float = 5.0
    louis_d: float = 5.0
    blend_ri_sharpness: float = 100.0
    blend_pbl_sharpness: float = 10.0  # sigmoid sharpness for K-profile->local PBL blend
    countergrad_coeff: float = 6.5
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class TurbulentEDMFConfig(NamedTuple):
    """Configuration for EDMF eddy-diffusivity mass-flux turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Ck : float
        TKE -> Km coefficient (default 0.1).
    Ce : float
        TKE dissipation coefficient (default 0.19).
    tke_min : float
        Minimum TKE [m^2/s^2] (default 1e-6).
    Pr_t : float
        Turbulent Prandtl number (default 0.33).
    n_updrafts : int
        Number of updraft plumes (default 1).
    a_updraft : float
        Updraft area fraction (default 0.1).
    w_updraft_min : float
        Minimum updraft velocity [m/s] (default 0.1).
    entrainment_rate : float
        Lateral entrainment rate [1/m] (default 1e-3).
    detrainment_rate : float
        Lateral detrainment rate [1/m] (default 2e-3).
    parcel_dT : float
        Initial updraft potential-temperature perturbation [K]
        (default 0.5).  Was hardcoded as ``+0.5`` in the scan body
        prior to the audit-driven config migration; lifting it to a
        config field lets users tune the initial buoyancy of the
        plume against scheme calibration data.
    updraft_deactivation_sharpness : float
        Sigmoid sharpness [s/m] for smoothly deactivating the updraft as
        its vertical velocity falls below ``w_updraft_min`` (default 20.0).
        Lifted from a hardcoded literal inside the ``lax.scan`` updraft
        body so the transition width is tunable against calibration.
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Ck: float = 0.1
    Ce: float = 0.19
    tke_min: float = 1e-6
    Pr_t: float = 0.33
    n_updrafts: int = 1
    a_updraft: float = 0.1
    w_updraft_min: float = 0.1
    entrainment_rate: float = 1e-3
    detrainment_rate: float = 2e-3
    parcel_dT: float = 0.5
    updraft_deactivation_sharpness: float = 20.0
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class TurbulenceConfig(NamedTuple):
    """Top-level turbulence configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active turbulence scheme: "smagorinsky", "louis", "tke",
        "mynn25", "clubb_lite", "holtslag_boville", "ysu", "edmf",
        or "none".
    smagorinsky : SmagorinskyConfig
        Configuration for Smagorinsky scheme.
    louis : LouisConfig
        Configuration for Louis scheme.
    tke : TKEConfig
        Configuration for TKE scheme (Mellor-Yamada 1982).
    mynn25 : MYNN25Config
        Configuration for MYNN-2.5 scheme (Nakanishi-Niino 2009).
    clubb_lite : CLUBBLiteConfig
        Configuration for CLUBB-lite scheme.
    clubb : CLUBBConfig or None
        Configuration for the fuller CLUBB scheme (``scheme="clubb"``). ``None``
        selects the CAM-default ``CLUBBConfig()`` (resolved in the dispatcher);
        kept as ``None`` here to avoid a config <-> clubb import cycle.
    holtslag_boville : HoltslagBovilleConfig
        Configuration for Holtslag-Boville scheme.
    ysu : YSUConfig
        Configuration for YSU scheme.
    edmf : TurbulentEDMFConfig
        Configuration for EDMF scheme.
    update_interval_steps : int
        Recompute turbulence every N time steps (1 = every step).
    """
    scheme: str = "smagorinsky"
    smagorinsky: SmagorinskyConfig = SmagorinskyConfig()
    louis: LouisConfig = LouisConfig()
    tke: TKEConfig = TKEConfig()
    mynn25: MYNN25Config = MYNN25Config()
    clubb_lite: CLUBBLiteConfig = CLUBBLiteConfig()
    holtslag_boville: HoltslagBovilleConfig = HoltslagBovilleConfig()
    ysu: YSUConfig = YSUConfig()
    edmf: TurbulentEDMFConfig = TurbulentEDMFConfig()
    clubb: CLUBBConfig | None = None
    update_interval_steps: int = 1
