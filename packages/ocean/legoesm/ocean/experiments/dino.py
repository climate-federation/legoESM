"""DINO (DIabatic Neverworld Ocean) — replication of Kamm et al. (2025).

Reference
---------
Kamm, D., Deshayes, J., & Madec, G. (2025). DINO: a diabatic model of
pole-to-pole ocean dynamics to assess subgrid parameterizations across
horizontal scales. *Geosci. Model Dev.*, 18, 8091-8107.
https://doi.org/10.5194/gmd-18-8091-2025

NEMO reference implementation: https://github.com/vopikamm/DINO/tree/v0.2.0
Zenodo deposit: https://doi.org/10.5281/zenodo.15016824

Scope
-----
Replicates the DINO 1° (R1) configuration on lat-lon (Mercator) and
MPAS regional grids. The domain is a 50°-wide Atlantic-sector basin
spanning ~70°S to 70°N with a re-entrant channel between 45° and 65°S.

Phase 2A — DINOConfig dataclass.

Other phases live in this module: bathymetry (Phase 2B), surface
forcing (Phase 2C), initial conditions (Phase 2D), vertical grid
(Phase 2E), and dispatch wiring (Phase 2F).

See ``docs/ocean/experiments/dino_replication_plan.md`` for the full
plan, decisions log, and audit findings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    create_levy_stretched_z_star,
)


@dataclass
class DINOConfig:
    """Configuration for the DINO experiment (1° R1).

    All numeric defaults reflect the paper (Tables 1-2, Appendices A-D)
    and were cross-checked against the upstream NEMO namelist
    (vopikamm/DINO@v0.2.0, EXPREF/namelist_cfg, fetched 2026-05-14).
    See ``docs/ocean/experiments/dino_replication_plan.md`` for the
    decisions log.
    """

    # ------------------------------------------------------------------
    # Domain (Sect 2.2, Table 2)
    # ------------------------------------------------------------------
    lon_west_deg: float = -50.0    # western basin boundary
    lon_east_deg: float = 0.0      # eastern basin boundary
    lat_max_deg: float = 70.0      # symmetric N/S truncation latitude
    #   NB on the nemo_faithful_grid path the row count is fixed (n_lat=195),
    #   so the Mercator projection truncates the outermost T-centre at ±69.151°
    #   (NEMO's merc_proj) — this 70° gates the valid range, not the exact edge.
    # Opt-in: build the standalone lat-lon grid on NEMO's EXACT DINO R1 mesh
    # (mesh-verified: T-centres lon [1.5,48.5] / faces [1,49], 48 zonal cells,
    # 195 meridional rows with the equator ON a T-point at ±69.151°) instead of
    # the legoESM [-50,0]/198×50 default. Default OFF so existing DINO runs are
    # byte-identical (the 48° vs 50° basin is not comparable). When ON, the lon
    # frame + sill anchor MUST be NEMO's — use ``nemo_faithful_dino_config`` (it
    # co-sets lon_west/lon_east/sill_lon_m_deg); ``dino_lat_lon_grid`` raises if
    # the frame is inconsistent. See docs/ocean/fidelity/dino_tendency_certificate.md.
    nemo_faithful_grid: bool = False
    channel_lat_south_deg: float = -65.0  # southern edge of re-entrant channel
    channel_lat_north_deg: float = -45.0  # northern edge of re-entrant channel
    channel_width_deg: float = 20.0       # Δφ_c in eq A4 (= |lat_n - lat_s|)

    # ------------------------------------------------------------------
    # Reference physical constants used by the paper (Table 1)
    # Different from project defaults; kept as scheme parameters for
    # paper-fidelity replication (rho_0 differs from constants.rho_ocean
    # by ~0.1%, c_p differs from constants.c_sw by ~0.05%).
    # ------------------------------------------------------------------
    rho_0: float = 1026.0          # Boussinesq reference density [kg/m³]
    c_p: float = 3991.86           # Specific heat capacity [J/(kg·K)]

    # ------------------------------------------------------------------
    # Bathymetry (Appendix A, Zenodo namelist)
    # b = g_φ · g_λ · (H_deep - H_shallow) + H_shallow  per the
    # implementation note in the plan (deep at interior, shallow at
    # boundary). The naming here is intentionally clearer than the
    # paper's "H_max"/"H_min".
    # ------------------------------------------------------------------
    H_deep: float = 4000.0         # max depth (deep basin) [m]; namelist rn_H
    H_shallow: float = 2000.0      # min depth (at coast) [m]; namelist rn_hborder
    s_lambda_inv_deg: float = 1.0 / 3.0   # zonal slope param 1/rn_distLam = 1/3 deg⁻¹
    # rn_slp_cha (channel-boundary slope tapering) is intentionally NOT
    # ported: paper Fig 1 reproduction is visually correct without it
    # at our resolution. If a future visual regression shows it matters,
    # add a `channel_wall_slope: float = 1.5` field and thread into
    # `_exp_bathy` for the channel-band tapering.

    # Sill (Scotia Ridge analog), eq. A5 + namelist
    sill_lon_m_deg: float = -50.0  # λ_m, anchored at western wall (Drake Passage)
    sill_lat_m_deg: float = -55.0  # φ_m, mid-channel latitude
    sill_gaussian_width_s: float = 4.0    # rn_ds_width [degrees]
    H_sill: float = 2500.0                # rn_ds_depth [m]

    # No Mid-Atlantic Ridge (paper Sect 2.2 explicitly excludes; namelist
    # rn_mr_* are test scaffolding — see decisions log 2026-05-14).

    # ------------------------------------------------------------------
    # Vertical grid (Appendix C, eq. C3)
    # z(k) = a₂ + a₁·k + a₀·a_cr·ln(cosh((k - k_th)/a_cr))
    # ------------------------------------------------------------------
    n_levels: int = 36             # K, number of full levels
    dz_min: float = 10.0           # Δz_min, top-layer thickness [m]
    k_th: int = 35                 # k_th, inflection level index
    a_cr: float = 10.5             # stretching parameter

    # ------------------------------------------------------------------
    # Surface forcing — wind (Sect 2.3, eq 7; cubic-Hermite smooth-step
    # tau_u(φ) interpolated through the (lat, tau) knots below.
    # ------------------------------------------------------------------
    wind_tau_lats_deg: tuple[float, ...] = (
        -70.0, -45.0, -15.0, 0.0, 15.0, 45.0, 70.0,
    )
    wind_tau_values: tuple[float, ...] = (
        0.0, 0.2, -0.1, -0.02, -0.1, 0.1, 0.0,
    )  # [N/m²]

    # ------------------------------------------------------------------
    # Surface forcing — restoring (Sect 2.3, eqs 8-9; Appendix B eqs B1-B2)
    # Paper uses heat-flux coefficients (W/m²/K and kg/m²/s); the DINO
    # module converts these to timescales internally per layer thickness.
    # ------------------------------------------------------------------
    A_theta: float = 40.0          # T-restoring heat flux coefficient [W/m²/K]
    A_S: float = 3.858e-3          # S-restoring salt flux coefficient [kg/m²/s]
    L_phi_deg: float = 140.0       # meridional extent for cosine profile [deg]

    # Restoring profile values (annual-mean equivalents of eqs B1-B4)
    T_star_eq: float = 27.0        # equatorial target T [°C]
    T_star_n_mean: float = 5.0     # northern boundary target T (annual mean of B3)
    T_star_s_mean: float = -0.5    # southern boundary target T (annual mean of B4)
    # Seasonal T* amplitudes (usrdef_sbc.F90 case 4: T*_s = mean_s - amp_s*c2,
    # T*_n = mean_n + amp_n*c2 with c2 the 21-July-phased cosine).  The
    # oracle hard-codes 0.5 / 3.0 (asymmetric: mild southern, strong
    # northern seasonal swing).
    T_star_seasonal_amp_s: float = 0.5
    T_star_seasonal_amp_n: float = 3.0
    # usrdef_sbc taum westerly boost ("Boost in westerlies for TKE"):
    # taum = |utau| * 1.3 where utau > 0 — TKE surface input only.
    taum_westerly_boost: float = 1.3
    # Run the analytic forcing with the oracle's annual cycle (ln_ann_cyc):
    # T* and Q_sr recomputed per step from the 360-day-year phases.  False
    # (historical default) keeps the precomputed annual-mean arrays
    # bit-exact.  Requires the caller to thread t_seconds into
    # apply_dino_*_surface_forcing (run_dino does).
    forcing_annual_cycle: bool = False
    # Route the wind stress THROUGH model.step(surface_forcing=...) (the
    # dynamics-core external-tau block) instead of the post-step Euler
    # kick, so the TKE closure receives the surface stress (NEMO's taum
    # channel, including the usrdef x1.3 westerly boost that feeds TKE
    # but NOT the momentum).  False (default) keeps the historical
    # post-step wind application bit-exact.
    wind_through_step: bool = False
    # NEMO dynzdf wind placement (the last composition item): deposit the wind
    # stress in the TOP CELL of the implicit vertical solve's RHS
    # (dynzdf.F90:326-335) instead of an explicit surface kick, with the wind's
    # depth-mean carried by F_slow for the barotropic substeps (NEMO zu_frc,
    # dynspg_ts.F90 ~L360).  Requires wind_through_step=True (the stress must
    # reach model.step's surface_forcing).  Default False = prior behaviour.
    surface_stress_implicit: bool = False
    S_star_eq: float = 37.25       # equatorial target S [g/kg]
    S_star_n: float = 35.1         # northern boundary target S [g/kg]
    S_star_s: float = 35.0         # southern boundary target S [g/kg]
    S_star_eq_dip_amp: float = 1.25       # equatorial Gaussian dip amplitude (eq B2)
    S_star_eq_dip_sigma_deg: float = 7.5  # equatorial Gaussian dip width [deg]

    # ------------------------------------------------------------------
    # Solar radiation (eq B5; annual-mean used in Phase 1, full seasonal
    # form deferred to Phase 5)
    # ------------------------------------------------------------------
    Q_sr_amp: float = 230.0                 # peak insolation [W/m²]
    solar_declination_amp_deg: float = 23.5  # declination amplitude (used by
                                            # full B5; annual mean integrates
                                            # over this internally)

    # Jerlov type I shortwave penetration (Sect 2.3 of paper). The
    # ShortwavePenetrationConfig in legoESM has matching defaults —
    # we just request water_type="I" downstream.
    jerlov_water_type: str = "I"

    # ------------------------------------------------------------------
    # Vertical mixing — KPP + enhanced-diffusion convection.
    # Paper uses TKE; we use KPP (decisions log 2026-05-14).
    # Background values match paper namelist (rn_avm0, rn_avt0).
    # ------------------------------------------------------------------
    A_v_bg: float = 1.2e-4         # background vertical viscosity [m²/s]
    K_v_bg: float = 1.2e-5         # background vertical diffusivity [m²/s]
    K_conv: float = 100.0          # enhanced-diffusion convective K [m²/s] (rn_evd)
    # NEMO nn_evdm=1 (DINO namelist): the enhanced vertical diffusion applies
    # to tracers AND momentum. Effective only when the vertical-mixing closure
    # is NOT kpp (KPP carries its own interior convective viscosity — the
    # k_profiles guard rejects nu_conv under kpp to avoid double-counting;
    # NEMO likewise pairs EVD with TKE, never KPP). MPAS is tracer-only
    # (edge-momentum convective mixing needs a TRiSK reconstruction) — the
    # MPAS path keeps nu_conv=0 regardless.
    # NOTE: this default CHANGES the historical --vmix constant/tke lat-lon
    # baselines (they previously ran tracer-only EVD; the pre-2026-07
    # stability numbers — constant NaN d231, tke NaN d226/d39 — were measured
    # WITHOUT momentum EVD). Reproduce those with --evd-momentum off.
    evd_on_momentum: bool = True   # NEMO nn_evdm=1 (momentum EVD, non-kpp closures)
    # Vertical-mixing turbulence closure. The paper (Kamm et al. 2025) uses the
    # NEMO TKE scheme (Blanke & Delecluse 1993); "tke" selects our TKE closure
    # configured to the paper (background visc/diff = A_v_bg / K_v_bg, convective
    # ceiling K_conv, constant background, prandtl_mode="constant"). DINO lat-lon
    # has TWO SW channel∩wall-corner (lat −69.7/lon −49.5, wind τ→0) surface-u
    # instabilities (DIAGNOSED 2026-06-30, _diag_dino_tke_blowup.py):
    #   (A) day ~39, TKE-specific, VISCOSITY-SENSITIVE — the ``tke_momentum_visc_bg``
    #       floor (below) damps it (sweep 1.2e-4→d39, 5e-4→>90d).
    #   (B) day ~230, scheme-GENERAL (NaNs tke AND "constant"), VISCOSITY-INSENSITIVE
    #       (5e-4/1e-3/2e-3 all NaN ~d226-236) — a deep corner numerical mode,
    #       analogous to the MPAS southern-channel one. NOT fixable by viscosity.
    # So with the floor, "tke" runs to ~day 226 (good for the laminar sub-annual
    # spin-up comparison) but is NOT multi-year-stable; only "kpp" (strong surface
    # mixing suppresses BOTH) is multi-year-stable, and stays the DEFAULT. The
    # tracer background K_v_bg is untouched, so the thermocline comparison is
    # unaffected. Set per run via --vmix.
    vmix_scheme: str = "kpp"       # "kpp" (multi-year-stable default) | "tke" (paper, ~226d)
    # TKE-only momentum-viscosity background [m²/s], a FLOOR applied (max with
    # A_v_bg) ONLY when vmix_scheme="tke" — see ``A_v_bg_effective``. 5e-4 (4×
    # the paper's 1.2e-4) damps the day-39 instability (A); raising it further
    # does NOT help instability (B) (2e-3 still NaNs ~d236), so 5e-4 is the
    # chosen floor. kpp/constant are unaffected (they keep the paper A_v_bg).
    tke_momentum_visc_bg: float = 5.0e-4
    # TKE tracer/momentum Prandtl selection. False (default): constant Pr=10
    # (Prandtl_tke0), byte-identical to the legacy DINO tke path and all other
    # recipes. True (nemo_dino_kamm): NEMO zdftke nn_pdl=1 — the Richardson-
    # dependent inverse Prandtl pdlr=max(0.1, ri_cri/max(ri_cri,Ri)) with
    # ri_cri=2/(2+rn_ediss/rn_ediff)=2/(2+c_eps/c_k) (zdftke.F90:399,772), so
    # Pr=1/pdlr=clamp(Ri/ri_cri,1,10). Reproduced EXACTLY by prandtl_mode=
    # "richardson" with prandtl_ri_coeff=1/ri_cri (see _dino_vertical_mixing_config;
    # test_nemo_recipe.test_nemo_tke_prandtl_bit_reproduces_nemo_pdl). Interior
    # (Ri>>ri_cri) drops avt toward 0.1·avm; convecting columns (Ri<0) give
    # Pr=1 (avt tracks the large convective avm).
    tke_prandtl_ri: bool = False

    # ------------------------------------------------------------------
    # Equation of state. The paper (Kamm et al. 2025) uses NEMO's
    # "simplified" S-EOS (Roquet et al. 2015, np_seos) with the DINO
    # coefficients (a0=0.165, b0=0.76554, λ1=0.06, μ1=1.4970e-4
    # thermobaric; λ2=μ2=ν=0, T0=10°C, S0=35). "nemo_seos" selects exactly
    # that oracle EOS — NemoSEOSConfig's defaults ARE the DINO coefficients
    # (mirroring how "veros_gsw" carries the global_4deg oracle coefficients
    # in its default config), so no per-experiment EOS-config threading is
    # needed. The legoESM default is "wright" (Wright 1997 full nonlinear
    # EOS) — a documented fidelity gap vs NEMO alongside the vmix closure.
    # Set per run via --eos.
    # ------------------------------------------------------------------
    eos: str = "wright"            # "wright" (legoESM default) | "nemo_seos" (paper/oracle)
    # EOS depth argument: "insitu" (pressure-based, legoESM default) or "geometric"
    # (NEMO gdept). The DINO tendency certificate (docs/ocean/fidelity/
    # dino_tendency_certificate.md) showed "geometric" is REQUIRED for the S-EOS
    # to match NEMO's rhd (the thermobaric μ1·zh term uses the geometric depth;
    # "insitu" gave a 10× worse depth-proportional error). So the nemo_paper
    # (paper/oracle) recipe sets "geometric"; other recipes keep "insitu".
    eos_depth: str = "insitu"

    # Convective adjustment (enhanced vertical diffusion) trigger fidelity.
    # NEMO's zdfevd is a HARD switch: it applies rn_evd (=K_conv) wherever the
    # local Brunt-Vaisala frequency rn2 < 0, and exactly ZERO otherwise. The
    # legoESM default is a differentiable sigmoid of N^2 (smooth_transition),
    # which LEAKS ~O(K_conv) mixing into weakly-STABLE water (N^2~+1e-6 s^-2) that
    # NEMO never mixes — over-eroding the thermocline (62-day matched-grid check:
    # basin-mean T@262m 8.78 vs NEMO 9.50; the hard step restores 9.55-9.57).
    # NEMO's rn2 (eosbn2) is the ADIABATIC static stability (local alpha,beta),
    # so the faithful pair is hard-step + n2_mode="adiabatic". Set on the
    # nemo_paper (oracle) recipe; other recipes keep the smooth legoESM default.
    # See docs/ocean/fidelity/dino_tendency_certificate.md.
    convection_smooth_transition: bool = True   # False = NEMO hard rn2<0 switch
    convection_n2_mode: str = "insitu"          # "adiabatic" = NEMO eosbn2
    # NEMO zdfevd threshold (zdfevd.F90): EVD fires where the N² <= -1e-12 —
    # a small NEGATIVE threshold that ignores marginally-neutral interfaces
    # (vs the legoESM default 0.0). Default 0.0 keeps other recipes unchanged;
    # nemo_paper sets NEMO's -1e-12. (NEMO's full switch is the two-time-level
    # MIN(rn2, rn2b) <= -1e-12; the rn2b hysteresis is entangled with the TKE
    # n2_before_advection + adiabatic-N² sequencing — a TKE-closure axis, and
    # measurably nil at the 62-day state — so it is characterised, not shipped
    # here. See docs/ocean/fidelity/dino_tendency_certificate.md.)
    convection_n2_threshold: float = 0.0        # -1e-12 = NEMO rn_evd threshold
    # TKE static-stability trigger. Default "insitu" (BIT-IDENTICAL legacy).
    # NEMO's zdftke consumes the SAME rn2 (eosbn2 bn2) as zdfevd — select
    # "nemo_bn2" to feed the TKE buoyancy the exact bn2 N². For the DINO S-EOS
    # bn2 ≡ the "adiabatic" parcel N² to ~9e-7 s^-2 (see the nemo_dino_kamm
    # convection comment), so no recipe switches by default.
    tke_n2_mode: str = "insitu"                 # "nemo_bn2" = NEMO eosbn2 rn2
    # ------------------------------------------------------------------
    # NEMO zdftke closure-IDENTITY axes (Kamm 2025 DINO, namzdf_tke ref
    # defaults). Every default below reproduces the PRIOR DINO TKE behaviour
    # byte-for-byte (Mode-B diagnostic, Veros mixing length + surface flux BC +
    # Gaspar amplitude + backward-Euler dissipation); the nemo_dino_kamm(+_mlf)
    # card flips them to the NEMO-faithful values. Reachable via --recipe and
    # dict-validated --config (the tke_n2_mode precedent — no per-field flag).
    # See zdftke.F90 + cfgs/DINO namzdf_tke and docs/ocean/fidelity.
    # ------------------------------------------------------------------
    # PROGNOSTIC en stepped at the model dt (NEMO tke_tke), carried on
    # state.tke, vs the legoESM Mode-B diagnostic equilibrium (dt=86400, 3 it).
    tke_prognostic: bool = False                # True = NEMO prognostic en
    # Mixing-length construction: 2 = Veros Bougeault-Lacarrere (legacy);
    # 3 = NEMO nn_mxl=3 lup/ldown |dl/dz|<=e3t sweeps + ln_mxl0 wind anchor.
    tke_mxl_choice: int = 2                      # 3 = NEMO nn_mxl=3
    # Surface TKE BC: "veros_flux" (Neumann wind-work flux) vs "nemo_dirichlet"
    # (NEMO en(1)=MAX(rn_emin0, rn_ebb·taum/rho0), zdftke.F90:265).
    tke_surface_bc: str = "veros_flux"          # "nemo_dirichlet" = NEMO
    # Kolmogoroff dissipation discretization: "backward_euler" (fully implicit)
    # vs "nemo_1p5_split" (NEMO zdftke zfact2/zfact3 1.5/0.5 semi-implicit).
    tke_dissipation: str = "backward_euler"     # "nemo_1p5_split" = NEMO
    # K amplitude convention: "gaspar_sqrt2e" (K=c_k·l·sqrt(2e), legacy) vs
    # "veros_sqrte" (K=c_k·l·sqrt(e)) — NEMO avm=rn_ediff·zmxlm·SQRT(en)
    # (zdftke.F90:713) is EXACTLY veros_sqrte (gaspar over-mixes by sqrt(2)).
    tke_kappa_convention: str = "gaspar_sqrt2e"  # "veros_sqrte" = NEMO
    # TKE self-diffusion Schmidt coefficient (alpha_tke). Gaspar/Veros use 30;
    # NEMO zdftke diffuses en with 0.5·(avm[k+1]+avm[k]) (zdftke.F90:406-410),
    # i.e. alpha_tke=1.0. None = keep the TKEConfig default (30).
    tke_alpha: float | None = None              # 1.0 = NEMO en self-diffusion

    # ------------------------------------------------------------------
    # GM/Redi mesoscale eddy parameterization. Adaptive κ via Visbeck 1997
    # (default) or Treguier 1997 (gm_kappa_scheme="treguier", the NEMO
    # nn_aei_ijk_t=21 oracle scaling — supersedes the 2026-05-14 decision
    # to skip Tréguier, added 2026-07-02 after the GM-effectiveness diag).
    # ------------------------------------------------------------------
    use_gm_redi: bool = True
    # Adaptive-κ_GM scaling: "visbeck" (Visbeck 1997, the historical legoESM
    # DINO choice) | "treguier" (Treguier 1997 / NEMO nn_aei_ijk_t=21 — the
    # ACTUAL DINO+ORCA1 oracle scaling; cap aei0 = rn_Ue·rn_Le = 0.03·100 km
    # = 3000 m²/s from the DINO &namtra_eiv). The GM-effectiveness diagnostic
    # (2026-07-01) showed Visbeck κ (200-2000) under-predicts the 1° channel
    # need — Treguier is the faithful option. Lat-lon only (MPAS raises).
    gm_kappa_scheme: str = "visbeck"
    treguier_aei0: float = 3000.0  # Treguier κ cap [m²/s] = rn_Ue·rn_Le
    visbeck_alpha: float = 0.015   # Visbeck dimensionless prefactor
    visbeck_kappa_min: float = 200.0      # κ_GM floor [m²/s]
    visbeck_kappa_max: float = 2000.0     # κ_GM ceiling [m²/s]
    redi_S_max: float = 0.005             # Redi slope tapering threshold
    gm_redi_slope_scheme: str = "triads"  # Griffies 1998 triads (matches NEMO iso-neutral)
    # Lateral TRACER mixing direction: "geopotential" (legacy — iso-level
    # Laplacian K_h = ½·U_T·Δ) or "isoneutral" (NEMO ln_traldf_iso: Redi
    # iso-neutral Laplacian with kappa = ½·U_d·Δ(φ) row-scaled, slope cap
    # rn_slpmax via redi_S_max, EIV-independent — use_gm_redi stays the
    # EIV switch; K_h is zeroed to avoid double-counting; the vertical
    # diagonal K33 is solved IMPLICITLY = NEMO ln_traldf_msc).
    lateral_tracer_mixing: str = "geopotential"
    # Steep-slope handling for the isoneutral operator: "dm95_taper"
    # (legacy — kappa tapers to 0 at steep slopes) or "nemo_cap" (NEMO
    # ldfslp: slope capped at redi_S_max, taper 1 — keeps flattening
    # steep fronts; the oracle semantic).  Only read when
    # lateral_tracer_mixing="isoneutral".
    redi_slope_limit: str = "dm95_taper"
    # Mixed-layer-depth criterion for the NEMO ldfslp slope ramp / native
    # slopes (GMRediConfig.mld_criterion): "rho_c" (default, byte-identical
    # potential-density difference) or "n2_integral" (NEMO zdfmxl.F90:91-105
    # exact integral(MAX(N^2,0) dz) >= g*rho_c/rho0). The nemo_dino_kamm card
    # selects "n2_integral"; only affects runs with the ML ramp / native
    # slopes active. Dispatch raises on an unknown value.
    gm_redi_mld_criterion: str = "rho_c"
    # GM eddy-induced (bolus) advection FORM for gm_redi_slope_scheme=
    # "nemo_iso_lap" (GMRediConfig.gm_bolus_advection): "centred" (default, byte-
    # identical — 2nd-order centred bolus flux inside the iso operator) or
    # "through_fct" (NEMO traadv: the bolus transport is added to the advecting
    # mass flux so it passes through the monotone FCT limiter). The nemo_dino_kamm
    # card selects "through_fct"; only read with slope_scheme="nemo_iso_lap".
    gm_bolus_advection: str = "centred"

    # ------------------------------------------------------------------
    # Lateral mixing of momentum (geopotential / iso-level Laplacian;
    # confirmed match with NEMO via Zenodo namelist 2026-05-14)
    # ------------------------------------------------------------------
    U_M: float = 0.27              # viscous velocity scale [m/s] (rn_Uv)
    # High-latitude / equatorial protection (MOM6 OM4 standard knobs).
    # Diagnosed empirically (2026-05-14) as the late-time stability fix:
    # without these, A_h(j)·cos(φ) drops to 34% at 70°N/S and the
    # equatorial waveguide goes unconstrained (f→0). Enabling them
    # turns a day-28 blowup into a 35-day-stable run with the
    # Hollingsworth correction.
    A_h_floor: float = 1000.0      # min effective A_h [m²/s] after cos(lat) scaling
    A_h_eq_boost: float = 3.0      # equatorial Laplacian-viscosity boost factor
    A_h_eq_sigma_deg: float = 5.0  # boost Gaussian half-width [deg]
    # Tracer iso-neutral diffusion velocity scale (R1 only; eq below
    # Table 2). At R1 our GM/Redi handles this — we keep U_T for
    # reference but do not apply a separate harmonic tracer diffusion.
    U_T: float = 0.027             # tracer diffusivity velocity scale [m/s] (rn_Ut)

    # ------------------------------------------------------------------
    # MPAS-only equatorial viscosity boost (grid-specific stabilizer).
    # The lat-lon path protects the equatorial waveguide with A_h_eq_boost
    # (above): where f→0 the Coriolis restoring vanishes and the forced
    # equatorial jet is numerically unconstrained. The MPAS implicit-CN
    # barotropic solver's Coriolis predictor-corrector has the SAME blind
    # spot — without protection the wind-driven equatorial jet runs away
    # (DIAGNOSED: |u| 1.8→7.6 m/s by day 20 → NaN by day 30; fastest edges
    # all at |lat|<3°; dt-independent, forcing-required, viscosity-of-the-
    # uniform-kind-insensitive → a dynamical f→0 mode, not grid noise).
    # MPASOceanConfig.equatorial_visc_boost applies the SAME tight-Gaussian
    # A_h·(1 + boost·exp(−½(lat/σ)²)) profile the lat-lon A_h_eq_boost uses
    # (σ=5° matches A_h_eq_sigma_deg). IGNORED on the lat-lon path. Boost
    # SWEEP: boost=3 NaNs day50; boost=5 stable but jet peaks 4.5 m/s; boost=8
    # peaks ~2.1 m/s — chosen default (≈9× A_h at the equator, MPAS 3–10 range).
    # SCOPE: this cures the EARLY equatorial mode (stable days 0–~90, was NaN
    # day 30). A SEPARATE instability (#3) limits the full year: the regional
    # Voronoi mesh has distorted (anisotropic dcEdge≪dvEdge) cells in the
    # southern re-entrant-channel / buffer transition (lat≈−67°) — see
    # _diag_dino_mpas_blowup.py — and as the ACC jet matures there (~day 100)
    # those cells go unstable (|u| regrows → NaN ~day 130), viscosity-
    # insensitive (boost 8/12, baro-u-visc 0/1e4/3e4 all NaN day 130). That is
    # a mesh-quality issue (the generator skips Lloyd, voronoi.py:1701), not a
    # config knob; full-year MPAS DINO needs a better southern-channel mesh.
    mpas_equatorial_visc_boost: float = 8.0   # low-lat A_h boost factor [-]

    # ------------------------------------------------------------------
    # Bottom drag (paper Sect 2.1 "nonlinear friction term"; namelist
    # rn_drag = 1e-3 quadratic).
    # ------------------------------------------------------------------
    C_d_bottom: float = 1.0e-3     # quadratic drag coefficient
    bottom_drag_bg_velocity: float = 0.1   # u_bg [m/s] for MOM6 quadratic-with-floor form
    bottom_drag_bbl_thickness: float = 50.0  # BBL thickness [m] for distributed drag
    # NEMO zdfdrg drag law: "legacy" keeps the historical MOM6 form above
    # (bit-exact); "nemo_quadratic" is the DINO reference's ACTUAL namdrg
    # selection (ln_non_lin: r = Cd0*sqrt(u^2+v^2+ke0) with Cd0 = C_d_bottom
    # and the namelist_ref ke0 = 2.5e-3 m^2/s^2); "nemo_loglayer" = the
    # zdfdrg np_loglayer option.  Threaded to the model config's
    # bottom_drag_scheme on BOTH the lat-lon and MPAS DINO paths.
    bottom_drag_scheme: str = "legacy"
    n_barotropic_substeps: int = 30   # baroclinic-to-barotropic step ratio

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    dt: float = 2700.0             # baroclinic timestep [s] (45 min, Table 2)

    # ------------------------------------------------------------------
    # Numerical scheme selections (decisions log 2026-05-14):
    # - PGF: adcroft (closest to NEMO ln_hpg_sco standard Jacobian)
    # - Barotropic solver: implicit_cn (avoid checkerboard noise)
    # - Tracer advection: tvd (start simple; sensitivity study deferred)
    # - hi_precision_pressure: True
    # - Free-slip lateral BC: legoESM default (matches NEMO)
    # - Vector-invariant + AL81 EEN PV-flux: legoESM default
    # ------------------------------------------------------------------
    pgf_scheme: str = "adcroft"
    # Vertical p' quadrature (LatLonCGridOceanConfig.pgf_quadrature):
    # "cell_integral" (legacy) or "nemo_trapezoid" (dynhpg recurrence —
    # the DINO oracle; on DINO's stretched levels the two differ).
    pgf_quadrature: str = "cell_integral"
    # Vertical coordinate for the lat-lon path: "zstar" (legacy — all 36
    # levels compressed to the local bowl depth, terrain-following) or
    # "masked_zco" (NEMO DINO_R1 ln_zco: FLAT geopotential levels with
    # full-cell bottom masking per zgr_msk_top_bot — wet iff
    # gdept(k) < H; the column's depth snaps to the interface below the
    # deepest wet centre).  The oracle runs masked z-levels, so the
    # r1_exact preset selects "masked_zco".  MPAS keeps its own column
    # handling and rejects "masked_zco".
    vertical_coordinate: str = "zstar"
    barotropic_solver: str = "implicit_cn"
    barotropic_implicit_theta_eta: float = 0.55
    # When barotropic_solver="rigid_lid", DINO applies the FULL Veros-faithful
    # ACC stack (ab2 outer + explicit_ab2 Coriolis + ab2_scope="advective" +
    # this dt_mom ratio); a BARE rigid_lid flip runs away because the bottom-drag
    # depth-mean only reaches the streamfunction barotropic balance via the
    # du_diss fold, which is gated on ab2_scope="advective" (veros_acc_recipe.py).
    # dt_mom = dt / ratio under-relaxes momentum to accelerate the ACC spin-up
    # (Veros dt_mom=4800/dt_tracer=43200 ⇒ 9). Ignored unless rigid_lid.
    rigid_lid_dt_mom_ratio: float = 9.0
    # Barotropic averaging filter (explicit_substep only): "cosine"
    # (legacy) | "power_law" | "box" | "nemo_boxcar_centred" (dynspg_ts
    # ln_bt_fw=F + nn_bt_flt=1 — the DINO namelist selection).
    barotropic_time_filter: str = "cosine"
    # In-substep barotropic Coriolis (node 16): "avg" (legacy 4-pt average,
    # which annihilates the 2Δx checkerboard -> spurious deep-equatorial jet)
    # | "een" (NEMO dyn_spg_ts::dyn_cor_2D enstrophy-conserving EEN
    # ln_dynvor_een, which restores the null mode).  DINO uses ln_dynvor_een.
    barotropic_coriolis: str = "avg"
    # Barotropic-Coriolis split (node 16 LIVE application): "frozen" (default)
    # keeps the planetary Coriolis frozen in F_slow across the substep window;
    # "live" removes the pre-step 2D barotropic Coriolis from F_slow and applies
    # it LIVE each substep on the evolving transport (NEMO dyn_spg_ts:296-300 +
    # dyn_cor_2D). Under vorticity_scheme="een_total" the "live" split REQUIRES
    # barotropic_coriolis="een" so the subtraction/live stencils cancel — this
    # is what unblocks the leapfrog (nemo_dino_kamm_mlf) at dt=2700 by
    # continuously restoring the 2Δx C-grid rotational null mode the frozen form
    # leaves undamped under the leapfrog's neutral stability.
    barotropic_coriolis_split: str = "frozen"
    # ln_bt_auto: compute n_barotropic_substeps from the external-wave
    # CFL with this Courant ceiling (rn_bt_cmax); <= 0 disables (use
    # n_barotropic_substeps as-is).
    barotropic_auto_cmax: float = 0.0
    tracer_advection: str = "tvd"
    # Hollingsworth correction for KE gradient (fixes Hollingsworth-
    # Kallberg instability over stratified bathymetry; legoESM #263).
    # Matches NEMO's nn_dynkeg=1 default.
    ke_gradient_scheme: str = "hollingsworth"
    # Lateral momentum viscosity OPERATOR (node 14). "vector_laplacian" (default)
    # applies a single A_h·cos(φ) scalar OUTSIDE the vector Laplacian;
    # "nemo_div_curl" embeds NEMO's ahmt(T)/ahmf(F) = ½·rn_Uv·MAX(e1,e2)
    # coefficient INSIDE the div/curl (NEMO dyn_ldf_lev_lap). On the DINO
    # Mercator grid (e1≈e2) the magnitude matches A_h·cos(φ) to O(Δλ²) (~2e-5
    # worst-case, machine-level only at the equator); the difference is the
    # coefficient-gradient placement cross-terms.
    lateral_viscosity_operator: str = "vector_laplacian"
    # hi_precision_pressure is intentionally NOT a DINOConfig field —
    # the lat-lon dycore already pins it True at ocean_pe_latlon_cgrid.py
    # so a field on this config would never be read.

    # ------------------------------------------------------------------
    # Momentum / Coriolis / time-integrator scheme identity (the L2
    # intercomparison axes — see DINO_RECIPES). Defaults reproduce the legoESM
    # DINO stack (the NEMO double-gyre approximation: vector-invariant EEN
    # momentum, Matsuno-split Coriolis, forward-Euler outer integrator, total AB2
    # scope). The MITgcm / Oceananigans / Veros recipes override these to select
    # each model's canonical blocks (flux-form / WENO momentum, explicit-AB2
    # Coriolis, AB2 outer). Consumed by ``dino_lat_lon_model_config`` (lat-lon
    # C-grid path); the MPAS path keeps its own vector-invariant identity, so the
    # L2 recipes are lat-lon (matching the paper R1 comparison grid).
    # ``barotropic_solver="rigid_lid"`` ALWAYS forces the coordinated
    # Veros-faithful ab2 stack (it is the only barotropic path whose Coriolis is
    # AB2-consistent — see ``rigid_lid_dt_mom_ratio``), overriding these last.
    momentum_advection: str = "vector_invariant"  # "flux_form" (MITgcm) | "weno7" (Oceananigans)
    momentum_flux_scheme: str = "upwind"          # "centered" (MITgcm flux-form advScheme=2)
    coriolis_scheme: str = "matsuno_split"        # "explicit_ab2" (MITgcm/Oceananigans/Veros)
    outer_integrator: str = "forward_euler"       # "ab2" | "leapfrog" (NEMO stp_MLF)
    # Vector-invariant vorticity flux scheme (relative + optionally planetary).
    #   "al81" (default) — relative-vorticity EEN triad; planetary Coriolis in the
    #     separate matsuno_split / explicit_ab2 face-f path.
    #   "een_total" — NEMO ln_dynvor_een: the ABSOLUTE vorticity (f+zeta)/e3f rides
    #     the SAME EEN triad (Coriolis IN the RHS), the faithful DINO form. Pairs
    #     with coriolis_scheme="explicit_ab2" (Matsuno off) — used by the leapfrog
    #     (nemo_dino_kamm_mlf) card.
    vorticity_scheme: str = "al81"
    # Robert-Asselin filter coefficient (rn_atfp) for outer_integrator="leapfrog"
    # (NEMO plain RA, not Williams). NEMO default 0.1. Ignored otherwise.
    asselin_gamma: float = 0.1
    ab2_scope: str = "total"                       # "advective" (Veros; forced by rigid_lid)
    # AB2 time-centering of the barotropic slow forcing F_slow (Oceananigans
    # Gᵁ convention).  REQUIRED whenever coriolis_scheme="explicit_ab2" pairs
    # with barotropic_solver="implicit_cn": the CN predictor gates its own FB
    # Coriolis off (_cori_fac=0) and the outer AB2 deliberately excludes the
    # barotropic increment from extrapolation, so without this flag the
    # barotropic-mode Coriolis is integrated FORWARD EULER at weight 1.0 —
    # unconditionally unstable, |G|=sqrt(1+(f·dt)²) per step (e-fold ≈ 2/(f²·dt):
    # ~6 d at 15° for dt=2700 s).  Diagnosed as the DINO 'oceananigans'-card
    # barotropic blowup (dino_l2_bisect o_ctl: basin-scale off-equatorial eta
    # quadrupole, |eta| 6 m by day 15, growth rate ∝ dt).  The validated-stable
    # Silvestri §5 jet runs the same Coriolis routing WITH this flag on.
    barotropic_slow_forcing_ab2: bool = False     # True (Oceananigans card)

    # ------------------------------------------------------------------
    # Derived / effective properties
    # ------------------------------------------------------------------

    @property
    def A_v_bg_effective(self) -> float:
        """Effective background vertical VISCOSITY [m²/s] for the momentum
        solve (the model ``A_v`` and the TKE ``kappaM_min``).

        For ``vmix_scheme="tke"`` this is ``max(A_v_bg, tke_momentum_visc_bg)``
        — a FLOOR that damps the diagnosed SW channel-corner surface-momentum
        instability (a higher A_v_bg override still wins).  ``kpp``/``constant``
        keep ``A_v_bg`` (they are stable at the paper 1.2e-4).  The tracer
        background ``K_v_bg`` is NOT raised here, so the thermocline mixing
        stays at the paper value.
        """
        if self.vmix_scheme == "tke":
            return max(self.A_v_bg, self.tke_momentum_visc_bg)
        return self.A_v_bg


def dino_r1_exact_config(**overrides) -> DINOConfig:
    """DINO_R1 EXACTNESS preset (Level-1 campaign, step 1).

    Flips every DINOConfig selection whose EXACT NEMO block already exists
    onto the DINO_R1 oracle choice (tests/DINO_R1/EXP00/namelist_cfg — see
    docs/ocean/fidelity/dino_l1_exactness_audit.md):

    * ``eos="nemo_seos"``            — S-EOS, DINO coefficients (nameos)
    * ``vmix_scheme="tke"``          — the oracle closure (namzdf); NOTE the
      known multi-year SW-corner TKE instability (~d226 with the paper
      backgrounds) is accepted here: exactness first, the harness measures
      what the oracle-faithful configuration actually does
    * ``tke_momentum_visc_bg=A_v_bg``— stabilizer floor OFF (oracle
      avm0 = 1.2e-4 exactly; the 5e-4 floor is a legoESM stabilizer)
    * ``bottom_drag_scheme="nemo_quadratic"`` — zdfdrg ln_non_lin (#738)
    * ``use_gm_redi=False``          — ln_ldfeiv = .false. (NO eddy-induced
      velocity at R1)
    * ``lateral_tracer_mixing="isoneutral"``, ``redi_S_max=0.01`` —
      ln_traldf_iso Redi-only Laplacian (kappa = ½·U_d·Δ(φ) row-scaled,
      K_h zeroed) with the vertical diagonal K33 solved implicitly
      (= ln_traldf_msc).  Remaining deviation: NEMO CAPS the slope at
      rn_slpmax + ML ramp; we DM95-taper kappa around S_max
    * ``A_h_floor=0``, ``A_h_eq_boost=1`` — legoESM stabilizers OFF (the
      oracle viscosity is exactly ahm = Uv·Δ/2, no floor, no boost)
    * ``tracer_advection="fct2"`` — NEMO traadv_fct with nn_fct_h =
      nn_fct_v = 2 (2nd-order centred high flux + Zalesak), the oracle
      selection

    The preset also enables the oracle's seasonal forcing
    (``forcing_annual_cycle=True`` — ln_ann_cyc) and routes the wind
    through model.step (``wind_through_step=True`` — the TKE closure
    receives NEMO's taum incl. the x1.3 westerly boost).

    Fields that REMAIN approximate after this preset (later ladder steps,
    tracked in the audit doc):
    centred split-explicit barotropic (implicit_cn here), iso-neutral+MSC
    lateral diffusion, and the MLF leapfrog integrator.

    ``**overrides`` are applied on top (dataclasses.replace semantics).
    """
    import dataclasses as _dc

    base = DINOConfig(
        eos="nemo_seos",
        vmix_scheme="tke",
        bottom_drag_scheme="nemo_quadratic",
        use_gm_redi=False,
        A_h_floor=0.0,
        A_h_eq_boost=1.0,
        tracer_advection="fct2",
        vertical_coordinate="masked_zco",
        pgf_quadrature="nemo_trapezoid",
        lateral_tracer_mixing="isoneutral",
        # #1226: the NEMO ln_traldf_iso operator IS nemo_iso_lap — the
        # DINOConfig default ("triads", the Griffies approximation) left this
        # NEMO-exactness preset running a non-NEMO explicit operator while
        # the isoneutral builder's slope_positions="nemo_native" routed the
        # implicit K33 through the a33 stencil: a mismatched split pair the
        # fn-entry MSC guard now rejects loudly.
        gm_redi_slope_scheme="nemo_iso_lap",
        redi_S_max=0.01,               # rn_slpmax
        redi_slope_limit="nemo_cap",   # ldfslp cap semantics (not DM95)
        # Centred split-explicit barotropic (ln_bt_fw=F + flt=1) is
        # INSEPARABLE from the MLF integrator: the window needs the
        # before-state (t-dt) start. The forward-frame reduction grows
        # energy from day ~30 and NaNs by d180 (job 8826132) — and
        # NEMO's own DINO namelist says the same ("model crashes if
        # ln_bt_fw=T"). The filter + ln_bt_auto blocks are shipped and
        # F90-locked; the PRESET keeps implicit_cn until ladder step 4
        # (MLF) wires the before-state start.
        forcing_annual_cycle=True,
        wind_through_step=True,
    )
    # Stabilizer floor off: the oracle background viscosity is avm0 exactly.
    base = _dc.replace(base, tke_momentum_visc_bg=base.A_v_bg)
    if overrides:
        base = _dc.replace(base, **overrides)
    return base


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# L2 model recipes (docs/next_implementations "DINO two-level recipe" campaign).
#
# A DINO recipe is a PURE CONFIG overlay on ``DINOConfig`` that selects the
# canonical legoESM blocks reproducing another ocean model's DINO numerics —
# NEVER a bespoke solver (oracle-recipe doctrine, CLAUDE.md / docs/ocean/
# fidelity/oracle_recipe_strategy.md). Each dict below is splatted onto a base
# ``DINOConfig`` via :func:`dino_config_for_recipe`; the model-identity block
# choices (momentum / Coriolis / integrator / barotropic solver) mirror the
# verified dycore bundles in ``legoesm.ocean.recipes`` (``mitgcm_v1`` /
# ``oceananigans_v1`` / ``veros_faithful_v1``) so they cannot drift, while EOS +
# vertical mixing + tracer advection are set to each model's DINO choice (the
# thermocline-relevant axes). L1 = the paper/NEMO cards; L2 = the three
# cross-model cards. All are lat-lon (the paper R1 comparison grid); the MPAS
# path keeps its own identity.
#
# EOS/vmix fidelity notes (mapped, some approximate — see the report):
#   * MITgcm EOS: ``unesco80`` ≈ MITgcm JMD95Z (Jackett & McDougall 1995) within
#     ~1e-3 kg/m^3; MITgcm's default MDJWF (McDougall 2003) is NOT bit-exact in
#     legoESM (a documented gap, not stubbed here).
#   * Oceananigans EOS: ``veros_gsw`` IS the TEOS-10 48-term polynomial
#     (SeawaterPolynomials.TEOS10) — exact family match.
#   * Oceananigans vmix ``catke``: Oceananigans' CATKEVerticalDiffusivity; it is
#     prognostic (the lat-lon model carries its TKE state like ``tke``). Its 1°
#     DINO multi-year stability is UNVERIFIED — smoke-gate before a long run;
#     ``richardson`` (≈ RiBasedVerticalDiffusivity) is the diagnostic fallback.
#   * Veros vmix ``tke`` is only stable to ~day 226 on our 1° DINO (the SW-corner
#     mode documented on ``DINOConfig.vmix_scheme``), so the Veros card is a
#     sub-annual comparison; MITgcm (``kpp``) and the L1 cards run to a year.
# ---------------------------------------------------------------------

DINO_RECIPES: dict[str, dict] = {
    # --- L1 — the identity card: legoESM production default (Wright + KPP). ---
    # Equivalent to a bare DINOConfig(); named for uniform selection.
    "legoesm_default": {
        "eos": "wright",
        "vmix_scheme": "kpp",
    },
    # --- L1 — the paper/NEMO-faithful card (Kamm et al. 2025). ---
    "nemo_paper": {
        "eos": "nemo_seos",            # Roquet 2015 S-EOS, DINO coefficients (paper)
        "eos_depth": "geometric",      # NEMO gdept (tendency-certificate: Δrhd 1.5e-5 vs 1.4e-4)
        "vmix_scheme": "tke",          # NEMO TKE (Blanke & Delecluse 1993)
        "tracer_advection": "tvd",     # NEMO FCT/TVD family (R1)
        "gm_redi_slope_scheme": "triads",   # Griffies iso-neutral triads (NEMO)
        "ke_gradient_scheme": "hollingsworth",  # NEMO nn_dynkeg=1
        "barotropic_solver": "implicit_cn",
        # NEMO zdfevd: hard rn2<0 switch on the adiabatic (eosbn2) N^2. The
        # smooth-sigmoid default leaks mixing into stable water and over-cools
        # the thermocline by ~0.7 C (see DINOConfig.convection_smooth_transition).
        "convection_smooth_transition": False,
        "convection_n2_mode": "adiabatic",
        "convection_n2_threshold": -1e-12,   # NEMO zdfevd rn_evd threshold
    },
    # === COMPLETE NEMO-DINO card (Kamm et al. 2025) — EVERY setting explicit and
    # === cited to our NEMO 5.0.2 cfgs/DINO namelist_cfg/_ref, so nothing silently
    # === inherits a legoESM default. Unlike "nemo_paper" (a partial overlay that
    # === falls back to non-NEMO defaults on ~8 fields), this is the audited full
    # === match. See docs/ocean/fidelity/dino_setup_audit.md + the setup+namelist
    # === audit. ONE piece is not yet config-matchable and is flagged TODO below
    # === (Madec ln_traldf_iso + ln_ldfeiv). Grid: pair with nemo_faithful_dino_config
    # === for NEMO's exact 48x195 mesh + bathymetry frame.
    "nemo_dino_kamm": {
        # -- EOS (nameos: ln_seos=T, rn_lambda1=0.06, rn_mu1=1.497e-4) --
        "eos": "nemo_seos",
        "eos_depth": "geometric",                # key_qco z*/gdept depth for S-EOS
        # -- Vertical coordinate (namusr_def: ln_zco_nam=T, ln_zps_nam=F -> full-step z) --
        "vertical_coordinate": "masked_zco",
        # -- Vertical mixing (namzdf: ln_zdftke=T; namzdf_tke rn_ediff=0.1 rn_ediss=0.7) --
        "vmix_scheme": "tke",
        "tke_momentum_visc_bg": 1.2e-4,          # rn_avm0 (NO legoESM 5e-4 stabilizer floor)
        "tke_prandtl_ri": True,                  # nn_pdl=1 Ri-dependent Prandtl (default namzdf_tke)
        # -- NEMO zdftke closure identity (namzdf_tke ref defaults; DINO sets no
        #    &namzdf_tke overrides). Flips the legoESM Mode-B/Veros TKE to the
        #    NEMO-faithful prognostic closure; cures the eq surface avm 32×
        #    over-mixing (mxl 16.8 m vs NEMO ~0.2 m). --
        "tke_prognostic": True,                  # NEMO prognostic en at model dt
        "tke_mxl_choice": 3,                     # nn_mxl=3 lup/ldown + ln_mxl0 (rn_mxl0=0.04)
        "tke_n2_mode": "nemo_bn2",               # zdftke consumes eosbn2's rn2
        "tke_surface_bc": "nemo_dirichlet",      # en(1)=MAX(rn_emin0, rn_ebb·taum/rho0)
        "tke_dissipation": "nemo_1p5_split",     # zdftke zfact2/zfact3 1.5/0.5 split
        "tke_kappa_convention": "veros_sqrte",   # avm=rn_ediff·zmxlm·SQRT(en) (zdftke:713)
        "tke_alpha": 1.0,                        # en self-diffusion 0.5·(avm+avm) (zdftke:406)
        # -- Convection (namzdf: ln_zdfevd=T, rn_evd=100, nn_evdm=1; hard rn2<0 on eosbn2) --
        # NEMO's rn2 is eosbn2 bn2 (S-EOS local alpha,beta at each cell's gdept,
        # geometric zrw interp) — available as convection_n2_mode="nemo_bn2" /
        # tke_n2_mode="nemo_bn2". VERIFIED same-state (bridge restart state) to
        # reproduce NEMO rn2_stg at 99.9% per-depth agreement. Kept at
        # "adiabatic" here because, for the DINO S-EOS, the parcel-displacement
        # "adiabatic" N² is numerically EQUIVALENT to the exact bn2 (corr 1.0000,
        # maxdiff ~9e-7 s^-2 → ~0.1% of marginal interfaces flip) — "adiabatic"
        # already matches NEMO rn2_stg to 99.9% same-state, so the exact bn2 is a
        # no-op-equivalent provenance option, not a fidelity fix. See
        # tests/ocean/unit/test_nemo_bn2.py + docs/ocean/fidelity.
        "convection_smooth_transition": False,
        "convection_n2_mode": "adiabatic",
        "convection_n2_threshold": -1e-12,
        # -- Bottom drag (namdrg: ln_non_lin=T; namdrg_bot rn_Cd0=1e-3, rn_ke0=2.5e-3) --
        "bottom_drag_scheme": "nemo_quadratic",  # r = Cd0*sqrt(u^2+v^2+ke0)
        # -- Tracer advection (namtra_adv: ln_traadv_fct=T, nn_fct_h=nn_fct_v=2) --
        "tracer_advection": "fct2",
        # -- Tracer lateral diffusion (namtra_ldf: ln_traldf_iso + ln_traldf_msc,
        #    nn_aht_ijk_t=20, rn_Ud=0.027, rn_Ld=100e3) + GM (namtra_eiv: ln_ldfeiv=T,
        #    nn_aei_ijk_t=21 => Treguier aei0 = rn_Ue*rn_Le = 0.03*100e3 = 3000). --
        #    NEMO's Madec STANDARD ln_traldf_iso operator + ln_ldfeiv GM bolus
        #    (ldf_eiv_trp_MLF), now implemented in the nemo_iso_lap path (conserving
        #    to machine precision + energetically correct — flattens isopycnals).
        "gm_redi_slope_scheme": "nemo_iso_lap",
        # namtra_adv ln_traadv_fct: NEMO adds the ln_ldfeiv GM bolus to the
        # advecting velocity, so the bolus flux goes THROUGH the monotone FCT
        # limiter (not a separate centred flux) — node 22 of the wiring diagram.
        "gm_bolus_advection": "through_fct",
        "gm_kappa_scheme": "treguier",
        # NEMO zdfmxl N^2-integral MLD criterion for the ldfslp slope ramp
        # (node 6/7); the pot-density default anchors the ML slope ramp at a
        # different depth (slopes corr 0.99 below ML, 0.33 inside).
        "gm_redi_mld_criterion": "n2_integral",
        "redi_S_max": 0.01,                      # rn_slpmax (namtra_ldf ref default)
        # -- Momentum (namdyn_adv: ln_dynadv_vec + nn_dynkeg=1; namdyn_vor: ln_dynvor_een) --
        "ke_gradient_scheme": "hollingsworth",
        # -- Lateral momentum viscosity (namdyn_ldf: ln_dynldf_lap, nn_ahm_ijk_t=20,
        #    rn_Uv=0.27; NO boost/floor). Node 14: NEMO dyn_ldf_lev_lap embeds
        #    ahmt(T)/ahmf(F)=½·rn_Uv·MAX(e1,e2) inside div/curl. --
        "lateral_viscosity_operator": "nemo_div_curl",
        "A_h_eq_boost": 1.0,
        "A_h_floor": 0.0,
        # -- Barotropic / free surface (namdyn_spg: ln_dynspg_ts=T; nn_bt_flt=2; nn_e=30) --
        # FE-FRAME FIDELITY CEILING (verified 2026-07-18, step-dump twin vs NEMO
        # RUN_STEPDUMP kt=5760->5761): this card runs the barotropic mode on the
        # forward_euler outer frame, but NEMO-DINO's dyn_spg_ts is intrinsically
        # the MLF/centred barotropic (ln_bt_fw=F, a 2*nn_e window over 2dt seeded
        # from the before-level). Two MLF-only composition pieces are therefore
        # FAITHFULLY guarded OUT of this frame (do NOT re-wire them here):
        #   * barotropic_time_filter="nemo_boxcar_ab3" (AB3 vel + AM4 ssh temporal
        #     dissipation) requires outer_integrator="leapfrog" — the 2dt window +
        #     before-seed are what make the AM4 boxcar faithful; on the dt/nn_e
        #     forward window the dissipation window is halved (unfaithful).
        #   * barotropic_coriolis_split="live" (in-substep dyn_cor_2D) requires
        #     coriolis_scheme="explicit_ab2" so F_slow CONTAINS the planetary
        #     Coriolis for the pre-step subtraction to cancel; this card uses
        #     matsuno_split (planetary Coriolis applied as a separate rotation,
        #     NOT in F_slow) -> live subtraction would leave an O(1) residual.
        # The step-1 eta residual (interior rms 1.88e-2 m) is NOT a 2dx barotropic
        # checkerboard (2dx-proj/total = 0.000) but a smooth west-wall + equatorial
        # pattern = the forward_euler-vs-MLF integrator-frame difference, which no
        # barotropic sub-step composition can remove. The bit-faithful NEMO-DINO
        # dynspg_ts match is the MLF card (nemo_dino_kamm_mlf), not this FE card.
        # barotropic_diffusion_alpha (0.01, default) is a forward-frame 2dx
        # stability crutch, not a NEMO term; setting it 0 slightly WORSENS step-1
        # eta (1.88->1.94e-2) and does not touch the (non-2dx) residual.
        "barotropic_solver": "explicit_substep",
        "barotropic_time_filter": "nemo_boxcar_centred",
        # namdyn_vor: ln_dynvor_een — enstrophy-conserving EEN barotropic
        # Coriolis (node 16; cures the deep-equatorial jet velocity null mode).
        # "een_metric" = METRIC-COMPLETE: folds NEMO's e1v/r1_e1u + e2u/r1_e2v
        # scale factors into ffu/ffv EXACTLY (dynspg_ts.F90:1349-1379) — the
        # coefficients NEMO's dyn_cor_2D actually uses, so this is the strictly
        # more NEMO-faithful barotropic Coriolis for the card (discrete enstrophy
        # conservation exact on the Mercator grid).
        # NB (twin-verified 2026-07-19): the e1/e2 factors are a ~1% high-lat
        # correction and do NOT cure the |lat|~68deg 2dx ETA runaway — the FE
        # step-twin from the bridged NEMO restart still NaNs at s26 with
        # een_metric, nearly byte-identical to "een" (eta 868 vs 862 @s24). That
        # runaway is a free-surface 2dx eta mode fed through the barotropic
        # PGF/continuity coupling, NOT the Coriolis V->u averaging the EEN
        # restores; the faithful cure remains the standing barotropic free-
        # surface redesign. een_metric is kept as the faithful Coriolis choice.
        "barotropic_coriolis": "een_metric",
        # ln_bt_auto=T + rn_bt_cmax=0.8 => NEMO computes nn_e=23 for this exact
        # DINO 1deg grid at rn_Dt=2700 s (RUN_STEPDUMP/ocean.output: "in
        # iterations nn_e = 23", max courant 0.779). NOT the namelist fallback
        # nn_e=30 (only used when ln_bt_auto=F). Matches the boxcar window that
        # sets the step-averaged eta. Re-derive via barotropic_common.
        # nemo_auto_substeps if the grid or dt changes.
        "n_barotropic_substeps": 23,
        # -- Surface forcing (namusr_def: ln_ann_cyc=T seasonal cycle) --
        "forcing_annual_cycle": True,
        "wind_through_step": True,
    },
    # --- L2 — Veros (Vallis nonlinear EOS, TKE, superbee, streamfunction/AB2). ---
    # Dycore identity: recipes.py::veros_faithful_v1 (rigid_lid → the builder
    # auto-applies ab2 + explicit_ab2 + ab2_scope="advective").
    "veros": {
        "eos": "veros_nonlin2",        # Veros eq_of_state_type=3 (Vallis 2008)
        "vmix_scheme": "tke",          # Veros enable_tke (canonical)
        "tracer_advection": "superbee",  # Veros Sweby superbee tracer flux
        "barotropic_solver": "rigid_lid",  # Veros external-mode streamfunction
        "gm_redi_slope_scheme": "triads",
        "ke_gradient_scheme": "centered",
    },
    # --- L2 — MITgcm (JMD95-family EOS, KPP, DST3 tracer, flux-form centered ---
    # momentum, explicit-AB2 face-f Coriolis, unsplit implicit free surface, AB2).
    # Dycore identity: recipes.py::mitgcm_v1.
    "mitgcm": {
        "eos": "unesco80",             # ≈ MITgcm JMD95Z (MDJWF not bit-exact)
        "vmix_scheme": "kpp",          # MITgcm pkg/kpp
        "tracer_advection": "dst3_multidim",  # MITgcm advScheme=80 (3-DST, multi-dim)
        "momentum_advection": "flux_form",    # MITgcm mom_fluxform
        "momentum_flux_scheme": "centered",   # MITgcm advScheme=2 (centered)
        "coriolis_scheme": "explicit_ab2",    # MITgcm 4-pt face-f Coriolis
        "outer_integrator": "ab2",            # MITgcm Adams-Bashforth
        "ab2_scope": "total",                 # MITgcm momDissip_In_AB=.TRUE.
        "barotropic_solver": "implicit_unsplit",  # implicitFreeSurface (unsplit)
        "ke_gradient_scheme": "centered",     # dropped under flux_form
    },
    # --- L2 — Oceananigans (TEOS-10 EOS, CATKE, WENO tracer + WENOVectorInvariant ---
    # momentum, implicit free surface, AB2). Dycore identity: recipes.py::
    # oceananigans_v1. Coriolis explicit_ab2 is [APPROX] for the vertex-f
    # enstrophy-conserving form (see oceananigans_recipe.py).
    "oceananigans": {
        "eos": "veros_gsw",            # TEOS-10 48-term (SeawaterPolynomials.TEOS10)
        "vmix_scheme": "catke",        # Oceananigans CATKEVerticalDiffusivity
        "tracer_advection": "weno7",   # Oceananigans WENO(order=7)
        "momentum_advection": "weno7",  # Oceananigans WENOVectorInvariant (PR #559)
        "coriolis_scheme": "explicit_ab2",  # [APPROX] face-f ~ vertex-f enstrophy
        "outer_integrator": "ab2",     # Oceananigans QuasiAdamsBashforth2 (default)
        "barotropic_solver": "implicit_cn",  # Oceananigans ImplicitFreeSurface
        # AB2 time-centering of F_slow (Oceananigans Gᵁ): without it the
        # explicit_ab2 × implicit_cn pairing integrates the barotropic-mode
        # Coriolis forward-Euler (unconditionally unstable; the diagnosed
        # DINO-oceananigans barotropic blowup).  Matches the Silvestri §5
        # jet stack, which validates this Coriolis routing WITH the flag.
        "barotropic_slow_forcing_ab2": True,
        "ke_gradient_scheme": "centered",
    },
}

# nemo_dino_kamm_mlf: the Kamm-2025 DINO card WITH NEMO's faithful Modified
# Leap-Frog time integrator (stp_MLF) and the combined-EEN Coriolis-in-RHS
# (ln_dynvor_een, (f+zeta)/e3f through the EEN triad). Inherits every nemo_dino_kamm
# block and overrides ONLY the integrator/Coriolis placement, so a controlled A/B
# vs nemo_dino_kamm isolates the time-integrator change. Node 19 (+ node 13) of the
# DINO wiring diagram.
DINO_RECIPES["nemo_dino_kamm_mlf"] = {
    **DINO_RECIPES["nemo_dino_kamm"],
    "outer_integrator": "leapfrog",       # NEMO stp_MLF (key_qco, no key_RK3)
    "vorticity_scheme": "een_total",      # ln_dynvor_een: (f+zeta) in the EEN triad
    "coriolis_scheme": "explicit_ab2",    # Matsuno rotation OFF; Coriolis in the RHS
    "asselin_gamma": 0.1,                 # rn_atfp (plain Robert-Asselin, not Williams)
    # LIVE in-substep EEN barotropic Coriolis (node 16, NEMO dyn_cor_2D applied
    # each substep on ua_e/va_e): removes the pre-step 2D barotropic Coriolis
    # from F_slow (dynspg_ts:296-300) and re-applies the SAME EEN stencil live,
    # continuously restoring the C-grid 2Δx null mode. "een" is already inherited
    # from nemo_dino_kamm; "live" is what actually fires it under the leapfrog.
    "barotropic_coriolis_split": "live",
    # NEMO nn_bt_flt=2 FULL substep (dynspg_ts.F90 ts_bck_interp): the AB3
    # velocity predictor + the α=0 ssh half-step-back interpolation
    # (0.614/0.285/0.088/0.013) supply NEMO's built-in AM4 TEMPORAL DISSIPATION
    # of the 2Δx barotropic gravity-wave mode.  Under forward_euler (nemo_dino_kamm)
    # the FE numerical damping suppressed that mode with the plain boxcar
    # (nemo_boxcar_centred); the NEUTRAL leapfrog does NOT, so the equatorial
    # C-grid 2Δx eta checkerboard grows and blows the run at dt=2700 (~day 2)
    # WITHOUT this dissipation — the second half of the leapfrog barotropic
    # residual (residual #1b, alongside the Nbb before-level seed).  Boxcar
    # averaging + window are identical to nemo_boxcar_centred.
    "barotropic_time_filter": "nemo_boxcar_ab3",
}

# L2 cards select lat-lon-C-grid-only blocks (flux-form / WENO momentum, AB2
# outer, rigid-lid / unsplit free surface); the L1 cards are grid-portable but
# the DINO recipe surface is scoped to lat-lon for the intercomparison.
DINO_L2_RECIPES: frozenset[str] = frozenset({"veros", "mitgcm", "oceananigans"})


def dino_config_for_recipe(recipe: str,
                           base: DINOConfig | None = None) -> DINOConfig:
    """Return a ``DINOConfig`` overlaid with the named model recipe's blocks.

    ``recipe`` is one of :data:`DINO_RECIPES` (``legoesm_default``,
    ``nemo_paper``, ``nemo_dino_kamm``, ``veros``, ``mitgcm``,
    ``oceananigans``). ``nemo_dino_kamm`` is the COMPLETE NEMO-faithful card
    (Kamm et al. 2025) — every setting explicit + cited to the NEMO namelist,
    vs ``nemo_paper``'s partial overlay that inherits legoESM defaults. The
    recipe is a
    PURE CONFIG overlay (splatted via ``dataclasses.replace``) selecting shared
    canonical legoESM blocks — no bespoke solver.

    Parameters
    ----------
    recipe : str
        A key of :data:`DINO_RECIPES`.
    base : DINOConfig, optional
        Base config to overlay (default a fresh ``DINOConfig()``); pass one to
        combine a recipe with non-scheme setup tweaks (``dt``, grid extent).

    Returns
    -------
    DINOConfig

    Raises
    ------
    ValueError
        On an unknown recipe name (dispatch hardening — a typo must fail loudly,
        never silently select a default).
    """
    import dataclasses
    if recipe not in DINO_RECIPES:
        raise ValueError(
            f"unknown DINO recipe {recipe!r}; choose from "
            f"{sorted(DINO_RECIPES)}")
    if base is None:
        base = DINOConfig()
    return dataclasses.replace(base, **DINO_RECIPES[recipe])


# ---------------------------------------------------------------------
# Phase 2B — Analytical bathymetry (Appendix A; ported from
# vopikamm/DINO@v0.2.0 MY_SRC/usrdef_zgr.F90)
# ---------------------------------------------------------------------

def _smooth_step(x, a: float, b: float):
    """Quintic smooth step: 6t⁵ - 15t⁴ + 10t³ with t = (x-a)/(b-a) clipped.

    Paper eq A2; matches Zenodo `smooth_step` (zgr_lib lines 823-825).
    Returns 0 for x ≤ a, 1 for x ≥ b, smooth in between.
    """
    t = jnp.clip((x - a) / (b - a), 0.0, 1.0)
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _exp_bathy(x, x_left: float, x_right: float, width: float,
               dist_lam: float, dist_taper: float):
    """Tapered exponential — paper eq A1, matches Zenodo `exp_bathy`.

    Three-piece piecewise function: 0 outside [x_left, x_right], rises
    via tapered exponential to 1 in the interior.
    """
    znorm = 1.0 + jnp.exp(-width / dist_lam)

    # Left taper region: x ∈ [x_left, x_left + dist_taper]
    s_left = _smooth_step(x, x_left, x_left + dist_taper)
    val_left = (1.0 - jnp.exp(-(x - x_left) / dist_lam) / znorm) * (1.0 - s_left) + s_left

    # Right taper region: x ∈ [x_right - dist_taper, x_right]
    s_right = 1.0 - _smooth_step(x, x_right - dist_taper, x_right)
    val_right = (1.0 - jnp.exp((x - x_right) / dist_lam) / znorm) * (1.0 - s_right) + s_right

    in_left = (x >= x_left) & (x < x_left + dist_taper)
    in_right = (x > x_right - dist_taper) & (x <= x_right)
    in_interior = (x >= x_left + dist_taper) & (x <= x_right - dist_taper)

    return jnp.where(
        in_left, val_left,
        jnp.where(in_right, val_right,
                  jnp.where(in_interior, jnp.ones_like(x),
                            jnp.zeros_like(x))),
    )


def _gauss_ring(lon, lat, lon0: float, lat0: float, ring_radius: float,
                dist_lam: float, depth_top: float, depth_bot):
    """Gaussian ring centered at (lon0, lat0) with given radius.

    Matches Zenodo `gauss_ring`:
      arg = (-x² - y² + 2·rad·sqrt(x²+y²) - rad²) / dist_lam²
          = -(sqrt(x²+y²) - rad)² / dist_lam²

    Returns depth_bot away from the ring, depth_top on the ring.
    `depth_bot` may be a scalar or an array (e.g., the underlying
    bathymetry, so the ring can only shoal — never deepen — a column).
    """
    x = lon - lon0
    y = lat - lat0
    r = jnp.sqrt(x**2 + y**2)
    arg = -(r - ring_radius) ** 2 / dist_lam ** 2
    return (depth_top - depth_bot) * jnp.exp(arg) + depth_bot


def dino_bathymetry(lon_deg, lat_deg, cfg: DINOConfig | None = None):
    """Build DINO basin bathymetry at the given (lon, lat) points.

    Ports the analytical bathymetry construction from Kamm et al. 2025
    Appendix A (eqs A1-A5) and the upstream NEMO source
    (vopikamm/DINO@v0.2.0 MY_SRC/usrdef_zgr.F90, nn_botcase=1, the
    "bowl_cosh" basin).

    The Mid-Atlantic Ridge is intentionally omitted (paper Sect 2.2;
    `ln_mid_ridge=.false.` in the upstream namelist).

    Sign convention: ``H_bathy`` is positive (depth below sea level),
    bounded between ``H_shallow`` (at coasts) and ``H_deep`` (in the
    deep interior), and ``H_sill`` along the Drake-passage sill ring.

    Parameters
    ----------
    lon_deg, lat_deg : array
        Same-shape arrays of longitudes/latitudes in degrees. Can be
        any rank (will broadcast).
    cfg : DINOConfig, optional
        DINO configuration; defaults to ``DINOConfig()``.

    Returns
    -------
    H_bathy : array
        Same shape as ``lon_deg``. Depth in meters (positive).
    """
    if cfg is None:
        cfg = DINOConfig()

    lon_deg = jnp.asarray(lon_deg)
    lat_deg = jnp.asarray(lat_deg)

    # Domain extents
    lon_min = cfg.lon_west_deg
    lon_max = cfg.lon_east_deg
    lat_min = -cfg.lat_max_deg
    lat_max = cfg.lat_max_deg

    width_lon = lon_max - lon_min     # = 50°
    cha_min = cfg.channel_lat_south_deg
    cha_max = cfg.channel_lat_north_deg
    cha_width = cfg.channel_width_deg  # = 20°

    # Length scales (degrees) — note the Mercator correction on the
    # meridional scale matches the Zenodo code, NOT the paper text.
    # The code computes dist_phi = cos(rad·phi_max) · rn_distLam, which
    # SHRINKS the meridional taper length at high latitudes. Paper text
    # writes s_phi = cos(rad·phi_max) · s_lambda, which is the inverse
    # convention. Trusting the code (produces Fig 1).
    dist_lam_deg = 1.0 / cfg.s_lambda_inv_deg                              # 3°
    dist_phi_deg = math.cos(math.radians(cfg.lat_max_deg)) * dist_lam_deg  # ~1.03°

    # ------------------------------------------------------------------
    # Channel modification (paper eq A4): inside the channel band, the
    # zonal walls vanish so the flow is re-entrant.
    # ------------------------------------------------------------------
    g_phi_cha = _exp_bathy(
        lat_deg, cha_min, cha_max,
        width=width_lon, dist_lam=dist_lam_deg, dist_taper=cha_width / 2.0,
    )
    g_lambda = _exp_bathy(
        lon_deg, lon_min, lon_max,
        width=width_lon, dist_lam=dist_lam_deg, dist_taper=cha_width / 2.0,
    )
    # In the channel band, force g_lambda = 1 (no zonal walls)
    g_lambda = g_lambda * (1.0 - g_phi_cha) + g_phi_cha

    # Meridional taper for the full N/S extent (paper eq A3)
    g_phi = _exp_bathy(
        lat_deg, lat_min, lat_max,
        width=width_lon, dist_lam=dist_phi_deg, dist_taper=cha_width / 2.0,
    )

    # Bathymetry: deep at interior (g=1), shallow at boundaries (g=0)
    bathy = g_lambda * g_phi * (cfg.H_deep - cfg.H_shallow) + cfg.H_shallow

    # ------------------------------------------------------------------
    # Drake-passage sill (paper eq A5): Gaussian ring anchored at the
    # western wall, extending east into the channel. Only allowed to
    # SHOAL the column (depth_bot=bathy means it never deepens).
    # ------------------------------------------------------------------
    sill_taper = _smooth_step(
        lon_deg,
        cfg.sill_lon_m_deg,
        cfg.sill_lon_m_deg + cfg.sill_gaussian_width_s,
    )
    ring_radius = cha_width / 2.0  # zrad in Zenodo code
    sill = _gauss_ring(
        lon_deg, lat_deg,
        lon0=cfg.sill_lon_m_deg,
        lat0=cfg.sill_lat_m_deg,
        ring_radius=ring_radius,
        dist_lam=cfg.sill_gaussian_width_s,
        depth_top=cfg.H_sill,
        depth_bot=bathy,
    )
    bathy = sill_taper * sill + (1.0 - sill_taper) * bathy

    return bathy


# ---------------------------------------------------------------------
# Phase 2C — Surface forcing (Sect 2.3, Appendix B; ported from
# vopikamm/DINO@v0.2.0 MY_SRC/usrdef_sbc.F90)
#
# Surprising findings vs paper text (locked in 2026-05-14):
# - Wind interpolation is NOT PCHIP; the Zenodo source uses cubic
#   Hermite smooth-step (3-2s)·s² between adjacent (lat, tau) knots.
#   See `znl_cbc` in usrdef_sbc.F90.
# - Temperature restoring T*(lat) uses sin((φ+φ_max)·π/(φ_max-φ_min))
#   which is algebraically equivalent to cos(π·φ/L_φ) with L_φ=140 —
#   paper's cosine form is correct.
# - Q_sr (annual mean) is the time-average of the seasonal eq B5,
#   computed by numerical quadrature over a 360-day year.
# - Q_sr / non-solar split (paper eq 8) is applied where the
#   restoring tendency is evaluated, NOT here.
# ---------------------------------------------------------------------

def dino_wind_stress(lat_deg, cfg: DINOConfig | None = None):
    """Zonal wind stress τ_u(lat) using cubic Hermite smooth-step
    interpolation between (lat, tau) knots — matches Zenodo `znl_cbc`.

    Each segment is interpolated with `(3 - 2s)·s²` where
    s = clamp((φ - φ_left) / (φ_right - φ_left), 0, 1). This is a
    cubic with zero derivative at both endpoints — smoother than
    linear interpolation, NOT a true PCHIP.

    Parameters
    ----------
    lat_deg : array
        Latitude in degrees.
    cfg : DINOConfig, optional

    Returns
    -------
    tau_u : array, same shape as lat_deg
        Zonal wind stress [N/m²]. Purely zonal (no meridional component).
    """
    if cfg is None:
        cfg = DINOConfig()

    lat_deg = jnp.asarray(lat_deg, dtype=jnp.float64)
    lats = jnp.asarray(cfg.wind_tau_lats_deg, dtype=jnp.float64)
    taus = jnp.asarray(cfg.wind_tau_values, dtype=jnp.float64)

    # Find the segment containing each lat (vectorized).
    # For each lat point: locate the index ks s.t. lats[ks] <= lat <= lats[ks+1].
    # Use right-side searchsorted so that lat == lats[i] maps to segment [i-1, i].
    idx = jnp.clip(
        jnp.searchsorted(lats, lat_deg, side="right") - 1,
        0, lats.shape[0] - 2,
    )
    lat_lo = lats[idx]
    lat_hi = lats[idx + 1]
    tau_lo = taus[idx]
    tau_hi = taus[idx + 1]

    s = jnp.clip((lat_deg - lat_lo) / (lat_hi - lat_lo), 0.0, 1.0)
    weight = (3.0 - 2.0 * s) * s ** 2  # cubic Hermite smooth-step
    return tau_lo + (tau_hi - tau_lo) * weight


def dino_T_star_annual_mean(lat_deg, cfg: DINOConfig | None = None):
    """Annual-mean temperature restoring target T*(lat), per paper eq B1.

    T*(φ) = T*_n/s + (T*_eq - T*_n/s) · cos(π·φ/L_φ)

    where the n/s subscript switches at the equator. L_φ = 140°.
    Equivalent to the sin-form used in the Zenodo source.
    """
    if cfg is None:
        cfg = DINOConfig()
    lat = jnp.asarray(lat_deg)
    T_star_ns = jnp.where(lat <= 0.0, cfg.T_star_s_mean, cfg.T_star_n_mean)
    # Use cos form (paper); equivalent to NEMO's sin((φ+φ_max)π/L_φ).
    profile = jnp.cos(jnp.pi * lat / cfg.L_phi_deg)
    return T_star_ns + (cfg.T_star_eq - T_star_ns) * profile


def dino_S_star(lat_deg, cfg: DINOConfig | None = None):
    """Salinity restoring target S*(lat) with equatorial Gaussian dip
    per paper eq B2 / Zenodo source.

    S*(φ) = S*_n/s + (S*_eq - S*_n/s) · (1 + cos(2π·φ/L_φ))/2
            − 1.25·exp(−φ²/7.5²)
    """
    if cfg is None:
        cfg = DINOConfig()
    lat = jnp.asarray(lat_deg)
    S_star_ns = jnp.where(lat <= 0.0, cfg.S_star_s, cfg.S_star_n)
    cos_factor = (1.0 + jnp.cos(2.0 * jnp.pi * lat / cfg.L_phi_deg)) / 2.0
    dip = cfg.S_star_eq_dip_amp * jnp.exp(
        -(lat ** 2) / cfg.S_star_eq_dip_sigma_deg ** 2,
    )
    return S_star_ns + (cfg.S_star_eq - S_star_ns) * cos_factor - dip


def dino_Q_sr_annual_mean(lat_deg, cfg: DINOConfig | None = None,
                          n_days: int = 360):
    """Annual-mean Q_sr(lat) by trapezoid quadrature of paper eq B5
    over a 360-day year.

    Computed once at config time (NumPy-side); result is a JAX array.
    Not just `230·cos(φ)` — the polar-night max(., 0) clipping makes
    the high-latitude annual mean fall off faster than cos(φ).

    Parameters
    ----------
    lat_deg : array
    cfg : DINOConfig, optional
    n_days : int
        Number of days per year used for quadrature (DINO uses 360).

    Returns
    -------
    Q_sr_bar : array, same shape as lat_deg, in W/m².
    """
    if cfg is None:
        cfg = DINOConfig()
    lat = np.asarray(lat_deg, dtype=np.float64)
    days = np.arange(1, n_days + 1, dtype=np.float64)
    # Broadcast: (n_days, *lat.shape)
    decl = cfg.solar_declination_amp_deg * np.cos(
        np.pi * (days - 171.0) / 180.0,
    )
    decl_b = decl.reshape((n_days,) + (1,) * lat.ndim)
    arg = np.pi / 180.0 * (lat[None, ...] - decl_b)
    q = np.maximum(cfg.Q_sr_amp * np.cos(arg), 0.0)
    return jnp.asarray(q.mean(axis=0))


def dino_seasonal_cosines(t_seconds, cfg: DINOConfig | None = None):
    """Seasonal phase cosines (usrdef_sbc.F90 compute_day_of_year).

    360-day year; ``c1`` peaks at 21 June (solar declination phase),
    ``c2`` at 21 July (T* phase, one month lag)::

        c1 = cos( (t - 21 Jun) / (half year) * pi )
        c2 = cos( (t - 21 Jul) / (half year) * pi )

    ``t_seconds`` is the model time with NEMO's convention (kt*dt, first
    step ends at t = dt).  Pure jnp; traced-time safe.
    """
    year_h = 360.0 * 24.0
    zt = jnp.mod(jnp.asarray(t_seconds) / 3600.0, year_h)
    half = year_h / 2.0
    c1 = jnp.cos((zt - 171.0 * 24.0) / half * jnp.pi)   # 21 June  (day 171)
    c2 = jnp.cos((zt - 201.0 * 24.0) / half * jnp.pi)   # 21 July  (day 201)
    return c1, c2


def dino_T_star_seasonal(lat_deg, t_seconds, cfg: DINOConfig | None = None):
    """Seasonal T*(lat, t) — usrdef_sbc case 4 with ln_ann_cyc.

    Boundary values swing with the 21-July cosine, ASYMMETRICALLY
    (oracle: south -0.5*c2, north +3.0*c2); the meridional profile is
    the same Munday cosine as the annual-mean form.
    """
    if cfg is None:
        cfg = DINOConfig()
    lat = jnp.asarray(lat_deg)
    _, c2 = dino_seasonal_cosines(t_seconds, cfg)
    T_s = cfg.T_star_s_mean - cfg.T_star_seasonal_amp_s * c2
    T_n = cfg.T_star_n_mean + cfg.T_star_seasonal_amp_n * c2
    T_star_ns = jnp.where(lat <= 0.0, T_s, T_n)
    profile = jnp.cos(jnp.pi * lat / cfg.L_phi_deg)
    return T_star_ns + (cfg.T_star_eq - T_star_ns) * profile


def dino_Q_sr_seasonal(lat_deg, t_seconds, cfg: DINOConfig | None = None):
    """Seasonal Q_sr(lat, t) — usrdef_sbc eq B5 with the annual cycle:

        Q_sr = max( Q0 * cos( pi*(lat - 23.5*c1)/180 ), 0 )

    The declination follows the 21-June cosine; the polar-night clip is
    the max(., 0).  Annual mean of this field == dino_Q_sr_annual_mean
    (same phase convention; locked by test).
    """
    if cfg is None:
        cfg = DINOConfig()
    lat = jnp.asarray(lat_deg)
    c1, _ = dino_seasonal_cosines(t_seconds, cfg)
    decl = cfg.solar_declination_amp_deg * c1
    arg = jnp.pi / 180.0 * (lat - decl)
    return jnp.maximum(cfg.Q_sr_amp * jnp.cos(arg), 0.0)



# ---------------------------------------------------------------------
# Phase 2C-extra — Surface tendency functions with Q_sr / non-solar
# split (paper eqs 7-9). Grid-agnostic: operate on whatever shape the
# surface T, S, T*, S*, Q_sr arrays have. The DINO module owns these
# because the general restoring/wind APIs in legoESM use simpler forms
# (timescale instead of heat-flux coefficient; no Q_sr split).
# ---------------------------------------------------------------------

def dino_top_layer_heat_flux_split(T_sfc_C, T_star, Q_sr,
                                    cfg: DINOConfig | None = None):
    """Surface heat-flux split per paper eq 8.

    Q_ns = A_Θ · (T* − T) − Q_sr     (non-solar; applied to top layer)
    Q_sr = Q_sr                        (solar; distributed via Jerlov)

    Both in W/m². The solar component is returned unchanged so the
    caller can apply it to the column via legoESM's
    ``shortwave_penetration`` module.

    Parameters
    ----------
    T_sfc_C : array
        Top-layer temperature [°C].
    T_star : array
        Restoring target [°C], same shape as T_sfc_C.
    Q_sr : array
        Surface solar flux [W/m²], same shape as T_sfc_C.
    cfg : DINOConfig, optional

    Returns
    -------
    Q_ns : array
    Q_sr : array (passed through unchanged)
    """
    if cfg is None:
        cfg = DINOConfig()
    Q_ns = cfg.A_theta * (T_star - T_sfc_C) - Q_sr
    return Q_ns, Q_sr


def dino_top_layer_T_tendency(T_sfc_C, T_star, Q_sr, dz_0,
                               cfg: DINOConfig | None = None):
    """Top-layer temperature tendency from non-solar heat flux (eq 8).

      dT/dt|_top = (A_Θ · (T* − T) − Q_sr) / (ρ₀ · c_p · dz_0)

    Solar penetration through the column is NOT included here — apply
    via Jerlov-I shortwave penetration separately.
    """
    if cfg is None:
        cfg = DINOConfig()
    Q_ns, _ = dino_top_layer_heat_flux_split(T_sfc_C, T_star, Q_sr, cfg)
    return Q_ns / (cfg.rho_0 * cfg.c_p * dz_0)


def dino_top_layer_S_tendency(S_surface, S_star, dz_0,
                               cfg: DINOConfig | None = None):
    """Top-layer salinity tendency from Haney-style restoring (eq 9).

      dS/dt|_top = A_S · (S* − S) / (ρ₀ · dz_0)
    """
    if cfg is None:
        cfg = DINOConfig()
    return cfg.A_S * (S_star - S_surface) / (cfg.rho_0 * dz_0)


def dino_top_layer_u_tendency(tau_u, dz_0,
                               cfg: DINOConfig | None = None):
    """Top-layer zonal-velocity tendency from wind stress (eq 7).

      dU/dt|_top = τ_u / (ρ₀ · dz_0)
    """
    if cfg is None:
        cfg = DINOConfig()
    return tau_u / (cfg.rho_0 * dz_0)


# ---------------------------------------------------------------------
# Phase 2D — Initial conditions (Appendix D; ported from
# vopikamm/DINO@v0.2.0 MY_SRC/usrdef_istate.F90)
#
# Note: paper eq D4-D5 has a typo. The formula reads
#   T̃(φ,z) = (Θ(z) - Θ|_{z=0}) · (φ_1 - |φ|)/φ_1 + Θ|_{z=0}
# but the surrounding text says "towards bottom values at the poles"
# and the Zenodo source uses `zTbot` (depth-bottom value), not z=0.
# We follow the source / text, not the equation: at the poles, columns
# are isothermal/isohaline at the deep-abyss value (~4°C, ~35.12 g/kg).
# ---------------------------------------------------------------------

def dino_T_profile_1d(z_pos):
    """Equatorial 1D temperature profile Θ(z) per paper eq D2 / Zenodo
    source (usrdef_istate.F90, case 1 vertical profile).

    Parameters
    ----------
    z_pos : array
        Depth (positive down) in meters.

    Returns
    -------
    T : array
        Temperature in °C, same shape as ``z_pos``.
    """
    z = jnp.asarray(z_pos)
    deep = 16.0 - 12.0 * jnp.tanh((z - 400.0) / 700.0)
    shallow = (
        15.0 * (1.0 - jnp.tanh((z - 50.0) / 1500.0))
        - 1.4 * jnp.tanh((z - 100.0) / 100.0)
        + 7.0 * (1500.0 - z) / 1500.0
    )
    weight_deep = (1.0 - jnp.tanh((500.0 - z) / 150.0)) / 2.0
    weight_shallow = (1.0 - jnp.tanh((z - 500.0) / 150.0)) / 2.0
    return deep * weight_deep + shallow * weight_shallow


def dino_S_profile_1d(z_pos):
    """Equatorial 1D salinity profile S(z) per paper eq D3 / Zenodo
    source (usrdef_istate.F90).

    Parameters
    ----------
    z_pos : array
        Depth (positive down) in meters.

    Returns
    -------
    S : array
        Absolute salinity in g/kg, same shape as ``z_pos``.
    """
    z = jnp.asarray(z_pos)
    deep = 36.25 - 1.13 * jnp.tanh((z - 305.0) / 460.0)
    shallow = (
        35.55 + 1.25 * (5000.0 - z) / 5000.0
        - 1.62 * jnp.tanh((z - 60.0) / 650.0)
        + 0.2 * jnp.tanh((z - 35.0) / 100.0)
        + 0.2 * jnp.tanh((z - 1000.0) / 5000.0)
    )
    weight_deep = (1.0 - jnp.tanh((500.0 - z) / 150.0)) / 2.0
    weight_shallow = (1.0 - jnp.tanh((z - 500.0) / 150.0)) / 2.0
    return deep * weight_deep + shallow * weight_shallow


def dino_initial_T_S(
    lat_deg,
    z_full_ref,
    cfg: DINOConfig | None = None,
    *,
    phi_max_deg: float | None = None,
    t_bot: float | None = None,
    s_bot: float | None = None,
):
    """Compute T(lat, z) and S(lat, z) initial conditions for DINO.

    Applies the meridional gradient (paper eq D4-D5 / Zenodo "case 4"):

      T̃(φ, z) = (T_1D(z) - T_bot) · (φ_max - |φ|) / φ_max + T_bot

    At the equator (|φ|=0): T̃ = T_1D(z) (full equatorial profile).
    At the poles (|φ|=φ_max): T̃ = T_bot for all z (isothermal abyssal
    columns — promotes high-latitude deep convection).

    Parameters
    ----------
    lat_deg : array, shape (..., n_lat) or any broadcastable shape
        Latitude in degrees.
    z_full_ref : array, shape (n_levels,)
        Cell-center z values (legoESM convention: NEGATIVE below
        surface). Internally converted to positive depths.
    cfg : DINOConfig, optional
    phi_max_deg, t_bot, s_bot : float, optional
        NEMO ``usr_def_istate`` (case 4) BIT-EXACT overrides. NEMO computes
        the meridional-blend anchors from the *actual model grid*, not the
        1-D reference profile:

        * ``phi_max_deg`` = ``MAXVAL(gphit)`` — the poleward-most T-point
          latitude (69.85° on DINO R1), NOT the nominal truncation
          ``cfg.lat_max_deg`` (70°).
        * ``t_bot`` / ``s_bot`` = ``MINVAL`` of the horizontally-uniform
          profile over the **wet** 3-D field. On a full-step (``ln_zco``)
          grid whose deepest reference level is globally DRY, this is the
          value at the deepest *wet* level, NOT ``T_1d[-1]`` (the deepest
          *reference* level).

        The pure ``(lat, z)`` formula cannot see the ``tmask``/``gphit``, so
        the NEMO-fidelity caller passes these three scalars (computed once
        from the mesh). Defaults (``None``) reproduce the standalone-recipe
        behaviour BYTE-IDENTICALLY (``cfg.lat_max_deg`` / ``T_1d[-1]`` /
        ``S_1d[-1]``).

    Returns
    -------
    T : array, shape (*lat_deg.shape, n_levels)
        Initial conservative temperature [°C].
    S : array, shape (*lat_deg.shape, n_levels)
        Initial absolute salinity [g/kg].
    """
    if cfg is None:
        cfg = DINOConfig()

    lat_deg = jnp.asarray(lat_deg)
    z_full_ref = jnp.asarray(z_full_ref)
    z_pos = -z_full_ref  # legoESM uses z negative below surface; the
                         # Zenodo formulas use depth positive down.

    T_1d = dino_T_profile_1d(z_pos)
    S_1d = dino_S_profile_1d(z_pos)

    # Bottom values (deepest cell-center). For the default DINO config
    # (H_deep=4000 m, 36 levels), these are T_bot ≈ 3.9°C, S_bot ≈ 35.12.
    # NEMO takes MINVAL over the wet 3-D field instead (see docstring); the
    # caller supplies t_bot/s_bot when matching NEMO bit-for-bit.
    T_bot = T_1d[-1] if t_bot is None else jnp.asarray(t_bot, dtype=T_1d.dtype)
    S_bot = S_1d[-1] if s_bot is None else jnp.asarray(s_bot, dtype=S_1d.dtype)

    # Meridional gradient factor (eq D4-D5; matches Zenodo case 4).
    # NEMO uses MAXVAL(gphit) (northernmost T-point); the pure formula
    # defaults to the nominal truncation cfg.lat_max_deg.
    phi_max = cfg.lat_max_deg if phi_max_deg is None else phi_max_deg
    factor = (phi_max - jnp.abs(lat_deg)) / phi_max  # 1 at equator, 0 at poles

    # Broadcast: factor has shape lat_deg.shape; profiles have shape (n_levels,)
    factor_b = factor[..., None]
    T = (T_1d - T_bot) * factor_b + T_bot  # shape (*lat_deg.shape, n_levels)
    S = (S_1d - S_bot) * factor_b + S_bot

    return T, S


# ---------------------------------------------------------------------
# Phase 2E — Vertical grid (Appendix C, eq C3)
# ---------------------------------------------------------------------

def create_dino_z_star(cfg: DINOConfig | None = None) -> OceanZStarCoordinate:
    """36-level Lévy 2010 stretched z* grid for DINO (Appendix C).

    Thin wrapper around ``legoesm.ocean.vertical.create_levy_stretched_z_star``
    that supplies DINO's parameter values. Top layer thickness ≈ 10 m;
    bottom layer ~454 m; surface interface = 0; bottom interface = -4000 m.
    """
    if cfg is None:
        cfg = DINOConfig()
    return create_levy_stretched_z_star(
        n_levels=cfg.n_levels,
        H_max=cfg.H_deep,
        dz_min=cfg.dz_min,
        k_th=float(cfg.k_th),
        a_cr=cfg.a_cr,
    )


def dino_masked_zco_coordinate(z_ref, H_bowl):
    """NEMO ``zgr_msk_top_bot`` masked z-levels for the DINO bowl.

    Reproduces DINO_R1's ``ln_zco`` vertical grid: a cell (i,j,k) is wet
    iff its reference centre depth is above the bathymetry
    (``gdept(k) < H``, usrdef_zgr.F90:471-479), and every wet cell is a
    FULL cell — the column's effective depth snaps to the interface
    below the deepest wet centre.  Implemented as an
    ``OceanPartialCellCoordinate`` whose ``H_bathy`` input is that
    snapped interface depth, so ``h_partial ∈ {0, dz_ref}`` exactly and
    the Jacobian is 1 at η=0.

    Parameters
    ----------
    z_ref : OceanZStarCoordinate
        The 36-level Lévy reference grid (``create_dino_z_star``).
    H_bowl : array (n_lat, n_lon)
        Continuous bowl bathymetry [m, positive down]; <= 0 on land.

    Returns
    -------
    (coord, H_snap) : (OceanPartialCellCoordinate, jnp.ndarray)
        ``H_snap[i,j] = Σ_k h_partial[i,j,k]`` — pass it as the state's
        ``H_bathy`` so geometry and state agree.
    """
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    abs_half = jnp.abs(jnp.asarray(z_ref.z_half_ref))       # (nlev+1,)
    # gdept(k) = the coordinate's own t-depths — ANALYTIC (mi96 zt=k+0.5)
    # when the ladder was built with analytic_t_depths=True; NEMO's wet
    # test uses pdept_1d, and midpoint surrogates put k_bot one level
    # too shallow wherever H falls between the midpoint and the analytic
    # centre (up to ~4.6 m apart on the DINO grid — codex r3 HIGH).
    centers = jnp.abs(jnp.asarray(z_ref.z_full_ref))        # (nlev,)
    H = jnp.asarray(H_bowl)
    # NEMO rule: wet iff gdept(k) < H  (strict; usrdef_zgr WHERE clause)
    n_wet = jnp.sum(centers[None, None, :] < H[..., None], axis=-1)
    H_snap = jnp.where(n_wet > 0, abs_half[n_wet], 0.0)
    coord = create_partial_cell_coordinate(z_ref, H_snap)
    return coord, H_snap


def dino_lat_lon_bowl(grid, cfg: DINOConfig | None = None):
    """Continuous DINO bowl bathymetry H(i,j) [m] on a Mercator grid.

    Single canonical construction (lon wrap to (-180,180] + meshgrid +
    :func:`dino_bathymetry`) shared by the state builder and the
    masked-zco coordinate builder.
    """
    if cfg is None:
        cfg = DINOConfig()
    lat_deg_1d = jnp.degrees(grid.lat)
    lon_deg_1d = jnp.degrees(grid.lon)
    lon_deg_1d = (lon_deg_1d + 180.0) % 360.0 - 180.0
    lon2d, lat2d = jnp.meshgrid(lon_deg_1d, lat_deg_1d, indexing="xy")
    return dino_bathymetry(lon2d, lat2d, cfg)


def dino_lat_lon_vertical(grid, cfg: DINOConfig | None = None):
    """Vertical coordinate for the lat-lon DINO per
    ``cfg.vertical_coordinate`` ("zstar" | "masked_zco")."""
    if cfg is None:
        cfg = DINOConfig()
    z_ref = create_dino_z_star(cfg)
    if cfg.vertical_coordinate == "zstar":
        return z_ref
    if cfg.vertical_coordinate == "masked_zco":
        # NEMO DINO_R1 ladder: jpk = cfg.n_levels counts INTERFACE
        # indices (level jpk is a permanently-masked dummy), so the wet
        # cell count is n_levels-1 = 35, and the t-depths are ANALYTIC
        # (mi96_1d zt = k+0.5) — verified against the reference run's
        # deptht to 1.5e-4 m (float32 file storage).  The legacy
        # "zstar" path keeps the historical 36-cell midpoint ladder.
        z_nemo = create_levy_stretched_z_star(
            n_levels=cfg.n_levels - 1,
            H_max=cfg.H_deep,
            dz_min=cfg.dz_min,
            k_th=float(cfg.k_th),
            a_cr=cfg.a_cr,
            analytic_t_depths=True,
        )
        coord, _H_snap = dino_masked_zco_coordinate(
            z_nemo, dino_lat_lon_bowl(grid, cfg))
        return coord
    raise ValueError(
        f"unknown DINOConfig.vertical_coordinate "
        f"{cfg.vertical_coordinate!r}; expected 'zstar' or 'masked_zco'")


# ---------------------------------------------------------------------
# Phase 3 — MPAS regional mesh wiring (partial-periodic via land mask)
#
# DINO has closed walls at lon = lon_west and lon = lon_east everywhere
# EXCEPT in the channel band (−65 to −45°S). The regional Voronoi mesh
# supports `periodic_x=True` (full re-entrant) and `periodic_x=False`
# (closed everywhere), but not partial periodicity.
#
# Solution: use `periodic_x=True` so the mesh wraps east-west, then
# mark cells in a thin strip near the periodic seam as LAND wherever
# the latitude is outside the channel band. This creates a wall along
# the seam everywhere except in the channel — exactly the DINO
# topology.
# ---------------------------------------------------------------------

def dino_mpas_land_mask(
    mesh,
    cfg: DINOConfig | None = None,
    seam_strip_width_deg: float | None = None,
):
    """Per-cell land mask for a DINO MPAS regional mesh.

    Thin wrapper around ``partial_periodic_seam_wall_mpas`` (which
    provides the general partial-periodic seam-wall pattern) combined
    with a lat-band buffer mask for cells outside ±lat_max_deg.

    Parameters
    ----------
    mesh : VoronoiMesh
    cfg : DINOConfig, optional
    seam_strip_width_deg : float, optional
        See ``partial_periodic_seam_wall_mpas``.

    Returns
    -------
    land_mask : jax array, shape (nCells,)
        1.0 = ocean, 0.0 = land.
    """
    from legoesm.ocean.init_mpas import partial_periodic_seam_wall_mpas

    if cfg is None:
        cfg = DINOConfig()

    # Source 1: buffer cells outside lat band [-lat_max_deg, lat_max_deg]
    lat_deg = jnp.degrees(mesh.latCell)
    in_lat_band = (
        (lat_deg >= -cfg.lat_max_deg) & (lat_deg <= cfg.lat_max_deg)
    ).astype(jnp.float32)

    # Source 2: seam wall outside channel band (general helper)
    return partial_periodic_seam_wall_mpas(
        mesh,
        open_lat_south_deg=cfg.channel_lat_south_deg,
        open_lat_north_deg=cfg.channel_lat_north_deg,
        seam_lon_deg=cfg.lon_west_deg,
        seam_strip_width_deg=seam_strip_width_deg,
        base_mask=in_lat_band,
    )


def dino_mpas_initial_state_arrays(
    mesh,
    z_coord: OceanZStarCoordinate,
    cfg: DINOConfig | None = None,
    seam_strip_width_deg: float | None = None,
):
    """Build (T, S, H_bathy, land_mask) for a DINO MPAS regional mesh.

    Convenience that combines land mask (Phase 3), bathymetry (Phase
    2B), and initial T/S (Phase 2D) for the MPAS path. Does not yet
    construct the full ``MPASOceanState`` — that wiring lives in
    Phase 2F.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    cfg : DINOConfig, optional
    seam_strip_width_deg : float, optional

    Returns
    -------
    T : jax array, shape (nCells, n_levels)
    S : jax array, shape (nCells, n_levels)
    H_bathy : jax array, shape (nCells,)
    land_mask : jax array, shape (nCells,)
    """
    if cfg is None:
        cfg = DINOConfig()

    land_mask = dino_mpas_land_mask(
        mesh, cfg=cfg, seam_strip_width_deg=seam_strip_width_deg,
    )

    # Mesh lonCell is in [0, 2π); wrap to (-180, 180]
    lon_deg = (jnp.degrees(mesh.lonCell) + 180.0) % 360.0 - 180.0
    lat_deg = jnp.degrees(mesh.latCell)

    H_bathy = dino_bathymetry(lon_deg, lat_deg, cfg)
    # Land cells: set H_bathy to 0 so dynamics doesn't try to use them
    H_bathy = jnp.where(land_mask > 0.5, H_bathy, 0.0)

    T, S = dino_initial_T_S(lat_deg, z_coord.z_full_ref, cfg)
    # Mask T, S on land
    mask3d = land_mask[:, None]
    T = jnp.where(mask3d > 0.5, T, 0.0)
    S = jnp.where(mask3d > 0.5, S, 0.0)

    return T, S, H_bathy, land_mask


# ---------------------------------------------------------------------
# Phase 1B + 2F — Lat-lon (Mercator) path. Phase 1B is essentially
# free in legoESM: setting LatLonCGridOceanConfig.A_h_lat_scaling=True
# applies the cos(lat) per-row factor the paper requires, because on
# the Mercator grid dx(j) = R·cos(φ(j))·Δλ and DINO's
# A_h(j) = 0.5·U_M·dx(j) = (0.5·U_M·R·Δλ)·cos(φ(j)).
# ---------------------------------------------------------------------

# NEMO DINO R1 exact grid (mesh_mask-verified against our NEMO 5.0.2 build):
# 48 zonal T-cells over faces [1°, 49°] (T-centres [1.5, 48.5]); 195 meridional
# rows with the equator ON a T-point (j=97, ±69.151° at the truncation). The lat
# count is NEMO's merc_proj truncation (K=97), NOT the floor-based auto-count, so
# it is passed explicitly. create_mercator_grid reproduces glamt/gphit to 3e-6°.
_NEMO_DINO_NLON = 48
_NEMO_DINO_NLAT = 195
_NEMO_DINO_LON_WEST = 1.0    # western cell FACE (rn_lam_min frame)
_NEMO_DINO_LON_EAST = 49.0   # eastern cell FACE


def nemo_faithful_dino_config(base: DINOConfig | None = None) -> DINOConfig:
    """A DINOConfig on NEMO's exact DINO R1 grid frame (opt-in, mesh-verified).

    Sets ``nemo_faithful_grid`` and co-sets the coupled longitude frame the
    bathymetry needs — the basin walls (``lon_west_deg``/``lon_east_deg``) and
    the Drake sill anchor (``sill_lon_m_deg``, which is anchored at the western
    wall) — to NEMO's [1°, 49°] frame. Use this instead of flipping
    ``nemo_faithful_grid`` alone, which would leave the bathymetry in the
    legoESM [-50°, 0°] frame and mark the whole grid as land.
    """
    import dataclasses
    base = base if base is not None else DINOConfig()
    return dataclasses.replace(
        base,
        nemo_faithful_grid=True,
        lon_west_deg=_NEMO_DINO_LON_WEST,
        lon_east_deg=_NEMO_DINO_LON_EAST,
        sill_lon_m_deg=_NEMO_DINO_LON_WEST,
    )


def dino_lat_lon_grid(cfg: DINOConfig | None = None, n_lon: int = 50):
    """Build a Mercator lat-lon grid covering the DINO basin.

    Wraps ``legoesm.grids.latlon.create_mercator_grid`` with the
    DINO-specific domain (lon ∈ [-50°, 0°], lat ∈ [-70°, 70°]).
    Default ``n_lon = 50`` gives 1° equatorial cells — paper R1.

    ``cfg.nemo_faithful_grid`` (opt-in) instead builds NEMO's EXACT DINO R1
    mesh — 48×195 with the equator on a T-point and faces [1°, 49°] — so a
    standalone run matches NEMO cell-for-cell (the ``n_lon`` argument is then
    ignored). The bathymetry lon frame must be NEMO's; build the config via
    :func:`nemo_faithful_dino_config` (this raises otherwise).
    """
    from legoesm.grids.latlon import create_mercator_grid

    if cfg is None:
        cfg = DINOConfig()

    if cfg.nemo_faithful_grid:
        # The bathymetry (dino_bathymetry) reads lon_west/lon_east; if they are
        # not NEMO's [1,49] frame the grid lons fall outside the basin and every
        # cell is masked land. Fail loudly rather than silently produce a dead
        # domain — the coupled fields come from nemo_faithful_dino_config.
        if (cfg.lon_west_deg, cfg.lon_east_deg) != (
                _NEMO_DINO_LON_WEST, _NEMO_DINO_LON_EAST):
            raise ValueError(
                "nemo_faithful_grid=True requires the NEMO [1,49] longitude "
                f"frame, got lon_west={cfg.lon_west_deg}, "
                f"lon_east={cfg.lon_east_deg}. Build the config with "
                "nemo_faithful_dino_config() so the bathymetry frame + sill "
                "anchor are co-set."
            )
        return create_mercator_grid(
            n_lon=_NEMO_DINO_NLON,
            lat_max_deg=cfg.lat_max_deg,
            lon_west_deg=cfg.lon_west_deg,
            lon_east_deg=cfg.lon_east_deg,
            equator_on_tpoint=True,
            n_lat=_NEMO_DINO_NLAT,
        )

    return create_mercator_grid(
        n_lon=n_lon,
        lat_max_deg=cfg.lat_max_deg,
        lon_west_deg=cfg.lon_west_deg,
        lon_east_deg=cfg.lon_east_deg,
    )


def dino_lat_lon_initial_state_arrays(
    grid,
    z_coord: OceanZStarCoordinate,
    cfg: DINOConfig | None = None,
    land_mask_override: jnp.ndarray | None = None,
):
    """Build (T, S, H_bathy, land_mask) for a DINO Mercator grid.

    Combines Phase 2B bathymetry, Phase 2D ICs, and a Phase-3-style
    seam-wall land mask. The lat-lon C-grid uses periodic-X (jnp.roll)
    for longitude operators, so without a land-mask wall the basin
    would be re-entrant at every latitude. We mark the westernmost
    longitude column as land outside the channel band, replicating the
    MPAS approach.

    ``land_mask_override`` (opt-in) SUPPLIES the wet mask directly and
    skips the analytic seam wall — used by the NEMO-bridged comparison
    to make the domain i-periodic/re-entrant (``ln_Iperio=.true.``) with
    NEMO's own ``tmask`` surface (column j=0 stays WET where NEMO is
    wet, no interior land wall).  Default ``None`` = the analytic
    seam-wall mask (byte-identical for the standalone bowl recipes).

    Returns
    -------
    T : array (n_lat, n_lon, n_levels) [°C]
    S : array (n_lat, n_lon, n_levels) [g/kg]
    H_bathy : array (n_lat, n_lon) [m]
    land_mask : array (n_lat, n_lon) (1=ocean, 0=land)
    """
    if cfg is None:
        cfg = DINOConfig()

    lat_deg_1d = jnp.degrees(grid.lat)                        # (n_lat,)

    # Bathymetry (shared canonical construction)
    H_bathy = dino_lat_lon_bowl(grid, cfg)

    # ICs: T(lat, z), S(lat, z) — broadcast over longitude
    T_lat_z, S_lat_z = dino_initial_T_S(lat_deg_1d, z_coord.z_full_ref, cfg)
    T = jnp.broadcast_to(T_lat_z[:, None, :],
                         (grid.n_lat, grid.n_lon, z_coord.n_levels))
    S = jnp.broadcast_to(S_lat_z[:, None, :],
                         (grid.n_lat, grid.n_lon, z_coord.n_levels))

    if land_mask_override is not None:
        # i-periodic / re-entrant (NEMO ln_Iperio): the caller (the NEMO
        # bridge) supplies NEMO's own surface tmask, which is wet at the
        # zonal margins — no interior seam wall.  compute_face_masks
        # (inside rest_state_latlon_cgrid_ocean) derives periodic-
        # consistent u_mask via jnp.roll in lon; N/S stay walled.
        if land_mask_override.shape != (grid.n_lat, grid.n_lon):
            raise ValueError(
                f"land_mask_override shape {land_mask_override.shape} != "
                f"(n_lat, n_lon)=({grid.n_lat}, {grid.n_lon}); a mis-shaped "
                "mask would silently mis-broadcast against T/S/H_bathy."
            )
        land_mask = jnp.asarray(land_mask_override).astype(T.dtype)
    else:
        # Land mask: thin wrapper over the general lat-lon partial-
        # periodic seam-wall helper. Channel is open between
        # channel_lat_south_deg and channel_lat_north_deg.
        from legoesm.ocean.init_latlon_cgrid import (
            partial_periodic_seam_wall_latlon)
        land_mask = partial_periodic_seam_wall_latlon(
            grid,
            open_lat_south_deg=cfg.channel_lat_south_deg,
            open_lat_north_deg=cfg.channel_lat_north_deg,
            seam_column_index=0,
        ).astype(T.dtype)

    # Mask T, S on land (keep ocean values)
    mask3d = land_mask[..., None]
    T = jnp.where(mask3d > 0.5, T, 0.0)
    S = jnp.where(mask3d > 0.5, S, 0.0)
    H_bathy = jnp.where(land_mask > 0.5, H_bathy, 0.0)

    return T, S, H_bathy, land_mask


def dino_lat_lon_state(
    grid,
    z_coord: OceanZStarCoordinate,
    cfg: DINOConfig | None = None,
    land_mask_override=None,
):
    """Build a full ``LatLonCGridOceanState`` for DINO from rest with
    paper IC stratification + bathymetry + seam-wall land mask.

    ``land_mask_override`` (opt-in) makes the domain i-periodic/
    re-entrant (NEMO ``ln_Iperio``) by supplying NEMO's own surface
    ``tmask`` instead of the analytic seam wall — the fix for the
    spurious equatorial land wall that trapped the deep-equatorial jet.
    ``rest_state_latlon_cgrid_ocean`` recomputes u_mask/v_mask atomically
    from it (periodic in lon, walled N/S), avoiding the stale-face-mask
    leak footgun.  Default ``None`` = the analytic seam-wall mask
    (byte-identical for the standalone bowl recipes).

    Returns
    -------
    LatLonCGridOceanState
    """
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    if cfg is None:
        cfg = DINOConfig()

    T, S, H_bathy, land_mask = dino_lat_lon_initial_state_arrays(
        grid, z_coord, cfg, land_mask_override=land_mask_override,
    )

    # Masked z-levels (vertical_coordinate="masked_zco"): the state's
    # H_bathy must be the coordinate's SNAPPED full-cell depth
    # (Σ h_partial), not the continuous bowl — otherwise the Jacobian
    # (H_bathy vs Σ h_partial) is inconsistent at η=0.
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    if isinstance(z_coord, OceanPartialCellCoordinate):
        H_bathy = jnp.sum(z_coord.h_partial, axis=-1)

    # Build a base rest-state with our overrides, then replace T, S
    state = rest_state_latlon_cgrid_ocean(
        grid=grid,
        z_coord=z_coord,
        H_max=cfg.H_deep,
        land_mask_override=land_mask,
        H_bathy_override=H_bathy,
    )
    # Replace T, S with our lat-z structured ICs
    state = state._replace(
        T=Field(data=T, name="T", dims=state.T.dims, units=state.T.units),
        S=Field(data=S, name="S", dims=state.S.dims, units=state.S.units),
    )
    return state


def _dino_vertical_mixing_config(cfg: DINOConfig):
    """Shared DINO vertical-mixing config (used by both grid model-configs so
    the closure is identical across lat-lon and MPAS).

    ``cfg.vmix_scheme == "tke"`` selects the NEMO-style TKE closure the paper
    uses, configured with the paper's constant background visc/diff
    (``A_v_bg`` / ``K_v_bg``) and the convective ceiling (``K_conv``), with the
    depth-dependent Bryan-Lewis background switched OFF (``bg_diff_scale=0``) to
    match the paper's constant background. ``"kpp"`` keeps the prior KPP closure.
    Convective adjustment itself is handled by the enhanced-diffusion convection
    scheme (``K_conv``) on top, as in the paper.
    """
    from legoesm.ocean.physics.vertical_mixing.config import (
        CATKEConfig, KPPConfig, RichardsonVerticalMixingConfig, TKEConfig,
        VerticalMixingConfig,
    )
    if cfg.vmix_scheme == "tke":
        # prandtl_mode="constant" is REQUIRED for the paper background to take
        # effect: with the default "unit" mode kappaH_min is dead (the tracer
        # floor inherits kappaM_min) and kappaM_max is ignored. "constant" gives
        # K_H = max(kappaH_min, K_M / Prandtl_tke0) and applies the kappaM_max
        # ceiling, so K_v_bg (1.2e-5) and the convective ceiling (K_conv) are
        # honored. kappaM_min is the EFFECTIVE momentum-viscosity floor
        # (A_v_bg_effective = max(A_v_bg, tke_momentum_visc_bg) = 5e-4 by default
        # — the SW-corner stabilizer); the TRACER floor kappaH_min stays at the
        # paper K_v_bg (1.2e-5), and Prandtl_tke0 is a FIXED field (default 10,
        # the paper's 1.2e-4/1.2e-5 ratio) — NOT recomputed from the raised
        # momentum floor — so the thermocline mixing is unchanged. The TKE step
        # is fed N2 + shear_sq by the vertical-mixing integration.
        tke = TKEConfig(
            prandtl_mode="constant",
            kappaM_min=cfg.A_v_bg_effective, kappaH_min=cfg.K_v_bg,
            kappaM_max=cfg.K_conv, bg_diff_scale=0.0,
            # NEMO zdftke consumes eosbn2's rn2 — "nemo_bn2" (else "insitu").
            n2_mode=cfg.tke_n2_mode,
            # NEMO zdftke closure-identity axes (see DINOConfig; defaults keep
            # the prior Mode-B/Veros behaviour, nemo_dino_kamm flips to NEMO).
            prognostic=cfg.tke_prognostic,
            tke_mxl_choice=cfg.tke_mxl_choice,
            surface_bc=cfg.tke_surface_bc,
            dissipation_discretization=cfg.tke_dissipation,
            kappa_convention=cfg.tke_kappa_convention,
            # NEMO zdftke surface terms — BOTH are namelist_ref defaults
            # (DINO's namelist_cfg sets no &namzdf_tke overrides, so the
            # oracle runs with ln_lc=T (rn_lc=0.15) and nn_etau=1
            # (rn_efr=0.05, nn_htau=0 → constant 10 m)). Faithful ON.
            lc=True,
            etau_mode="below_ml",
        )
        if cfg.tke_alpha is not None:
            # NEMO en self-diffusion uses 0.5·(avm[k+1]+avm[k]) (alpha_tke=1),
            # vs the Gaspar/Veros default 30. None ⇒ keep the TKEConfig default.
            tke = tke._replace(alpha_tke=cfg.tke_alpha)
        if cfg.tke_prandtl_ri:
            # NEMO nn_pdl=1: Ri-dependent inverse Prandtl. richardson mode with
            # coeff=1/ri_cri reproduces NEMO's Pr=1/pdlr=clamp(Ri/ri_cri,1,10)
            # to machine precision (clamp-to-[1,10] is order-independent). ri_cri
            # is derived from the SAME c_eps/c_k the TKE closure uses, so it
            # auto-tracks a retuned dissipation ratio (NEMO 0.7/0.1 → 2/9 → 4.5).
            ri_cri = 2.0 / (2.0 + tke.c_eps / tke.c_k)
            tke = tke._replace(
                prandtl_mode="richardson", prandtl_ri_coeff=1.0 / ri_cri)
        return VerticalMixingConfig(scheme="tke", tke=tke)
    if cfg.vmix_scheme == "kpp":
        return VerticalMixingConfig(
            scheme="kpp", kpp=KPPConfig(K_bg=cfg.K_v_bg, A_bg=cfg.A_v_bg),
        )
    if cfg.vmix_scheme == "constant":
        # Constant background vertical mixing (no boundary-layer scheme): the
        # model uses the config-level A_v_bg / K_v_bg directly. Used to UNIFY
        # the vertical mixing across grids identically (isolating the pure
        # discretization difference from any KPP-implementation difference).
        return VerticalMixingConfig(scheme="constant")
    if cfg.vmix_scheme == "richardson":
        # Pacanowski & Philander (1981) Richardson-number mixing — the closure
        # Oceananigans exposes as RiBasedVerticalDiffusivity. Diagnostic (no
        # prognostic state). Background floors anchored to the paper A_v_bg/K_v_bg
        # so the thermocline background matches the other cards.
        return VerticalMixingConfig(
            scheme="richardson",
            richardson=RichardsonVerticalMixingConfig(
                K_bg=cfg.K_v_bg, A_bg=cfg.A_v_bg_effective,
            ),
        )
    if cfg.vmix_scheme == "catke":
        # CATKE (Wagner et al. 2025) — Oceananigans' CATKEVerticalDiffusivity,
        # the canonical Oceananigans-DINO closure. PROGNOSTIC (like tke; the
        # lat-lon model carries its TKE state). Coefficients are the LES-
        # calibrated defaults; the model-level A_v/K_v supply the background.
        # NOTE: CATKE at 1° DINO is UNVERIFIED for multi-year stability — smoke-
        # gate before a long run; tke/kpp/richardson are fallbacks.
        return VerticalMixingConfig(scheme="catke", catke=CATKEConfig())
    raise ValueError(
        f"unknown DINOConfig.vmix_scheme {cfg.vmix_scheme!r}; expected "
        "'kpp', 'tke', 'constant', 'richardson' or 'catke'")


def _dino_barotropic_substeps(grid, cfg: DINOConfig) -> int:
    """n_barotropic_substeps, optionally from NEMO ln_bt_auto.

    ``barotropic_auto_cmax > 0``: nn_e = ceil(dt/cmax · max zcu) with
    ``zcu = sqrt(g·H·(1/e1² + 1/e2²))`` (dynspg_ts.F90:1223-1240),
    evaluated conservatively with the basin's deepest wet column and
    the smallest Mercator metrics (poleward rows).  Otherwise the
    configured ``n_barotropic_substeps`` is returned unchanged.
    """
    if cfg.barotropic_auto_cmax <= 0.0:
        return cfg.n_barotropic_substeps
    from legoesm import constants
    from legoesm.ocean.dynamics.barotropic_common import nemo_auto_substeps

    R = float(constants.R_earth)
    cos_min = float(jnp.min(jnp.cos(jnp.asarray(grid.lat))))
    e1_min = R * float(grid.dlon) * cos_min          # zonal, smallest row
    e2_min = R * float(jnp.min(jnp.asarray(grid.dlat))) \
        if hasattr(grid, "dlat") and jnp.ndim(grid.dlat) > 0 \
        else R * float(grid.dlat)
    inv_metric = 1.0 / e1_min ** 2 + 1.0 / e2_min ** 2
    return nemo_auto_substeps(
        cfg.dt, cfg.H_deep, inv_metric, float(constants.g),
        cmax=cfg.barotropic_auto_cmax)


def dino_lat_lon_model_config(
    grid,
    cfg: DINOConfig | None = None,
    physics: bool = True,
):
    """Build (LatLonCGridOceanConfig, OceanPhysicsConfig) for DINO.

    A_h is a SCALAR; the model multiplies it by ``cos(lat)`` per row
    when ``A_h_lat_scaling=True``. With the Mercator grid this exactly
    reproduces DINO's ``A_h(j) = 0.5·U_M·dx(j)``:

        A_h_base = 0.5 · U_M · R · Δλ_rad
        A_h(j)   = A_h_base · cos(φ(j))     [m²/s]

    Same logic for K_h with ``U_T``.
    """
    from legoesm.ocean.state import LatLonCGridOceanConfig

    if cfg is None:
        cfg = DINOConfig()

    R = grid.radius
    dlon_rad = grid.dlon
    A_h_base = 0.5 * cfg.U_M * R * dlon_rad
    K_h_base = 0.5 * cfg.U_T * R * dlon_rad

    # Quadratic-with-floor drag (MOM6 form): r = C_d * u_bg recovers
    # C_d*|u| at |u| >> u_bg.
    bottom_drag_r = cfg.C_d_bottom * cfg.bottom_drag_bg_velocity

    # Lat-lon C-grid GM/Redi is wired through LatLonCGridOceanConfig.gm_redi
    # (NOT OceanPhysicsConfig.lateral_mixing — that factory only supports
    # cubed-sphere). When Visbeck is enabled (always for DINO per the
    # Decisions Log), the kappa_GM/kappa_Redi fields are ignored at
    # runtime — anchored to ``cfg.visbeck_kappa_min`` so the static
    # value stays non-degenerate if the Visbeck path is ever mis-wired.
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, VisbeckConfig,
    )
    if cfg.gm_kappa_scheme not in ("visbeck", "treguier"):
        raise ValueError(
            f"unknown DINOConfig.gm_kappa_scheme {cfg.gm_kappa_scheme!r}; "
            "expected 'visbeck' or 'treguier'")
    if cfg.gm_redi_mld_criterion not in ("rho_c", "n2_integral"):
        raise ValueError(
            "unknown DINOConfig.gm_redi_mld_criterion "
            f"{cfg.gm_redi_mld_criterion!r}; expected 'rho_c' or 'n2_integral'")
    from legoesm.ocean.physics.lateral_mixing.config import TreguierConfig
    if cfg.lateral_tracer_mixing not in ("geopotential", "isoneutral"):
        raise ValueError(
            f"unknown DINOConfig.lateral_tracer_mixing "
            f"{cfg.lateral_tracer_mixing!r}; expected 'geopotential' or "
            "'isoneutral'")
    if cfg.lateral_tracer_mixing == "isoneutral":
        # NEMO ln_traldf_iso (+ ln_traldf_msc): Redi-ONLY iso-neutral
        # Laplacian.  kappa_Redi = aht = ½·U_T·Δx(φ) — the equator value
        # here, row-scaled by cos φ via kappa_redi_lat_scaling (the same
        # Mercator scaling A_h uses).  EIV stays a SEPARATE switch
        # (use_gm_redi): at R1 it is off, so kappa_GM=0 and both adaptive
        # κ diagnostics are disabled.  implicit_K33=True = the MSC
        # (vertical diagonal of the rotated operator solved backward-
        # Euler).  slope_density="neutral" builds slopes from locally-
        # referenced ∂ρ/∂T,∂ρ/∂S like NEMO's neutral slopes.
        # REMAINING DEVIATION (documented, audit row 8): NEMO caps the
        # SLOPE at rn_slpmax and ramps it inside the ML; our DM95 tanh
        # TAPERS kappa to zero around S_max instead.
        if cfg.use_gm_redi:
            raise ValueError(
                "DINOConfig: lateral_tracer_mixing='isoneutral' with "
                "use_gm_redi=True (EIV) is not wired yet — the R1 oracle "
                "runs EIV off; add the combined branch when the "
                "eddy-permitting recipes need it.")
        # Equator aht = ½·U_T·R·dλ = K_h_base (the SAME coefficient the
        # legacy iso-level K_h used); per-row cos φ applied by the model
        # via kappa_redi_lat_scaling.
        gm_redi_cfg = GMRediConfig(
            kappa_GM=0.0,
            kappa_Redi=float(K_h_base),
            kappa_redi_lat_scaling=True,
            S_max=cfg.redi_S_max,
            slope_scheme=cfg.gm_redi_slope_scheme,
            gm_bolus_advection=cfg.gm_bolus_advection,
            slope_density="neutral",
            slope_limit=cfg.redi_slope_limit,
            implicit_K33=True,
            # #1226: the explicit operator and the implicit K33 MUST share one
            # slope discretization.  With the default mode_b placement the
            # operator differences the single interface slope field while the
            # K33 getter built its diagonal from the 8-triad machinery — two
            # slope sets whose amplitudes differ O(35%), so the rotated
            # tensor's PSD condition (implicit S^2 >= explicit S^2, pointwise)
            # fails and the net vertical diffusivity goes NEGATIVE — the
            # kappa-scaled local tracer runaway (stable at kappa=200, T>38C
            # by day 90 at NEMO strength).  nemo_native uses ldfslp's four-
            # position slopes on BOTH sides (bit-identical arrays) with the
            # traldf_iso_a33 mask-normalized w-point kappa — NEMO's own
            # explicit/implicit split, transcribed.
            slope_positions="nemo_native",
            # ln_traldf_msc=T (the DINO namelist): NEMO's Method of
            # Stabilizing Correction — the explicit A33 remainder is bounded
            # by the 1/2 vertical-CFL limit and the implicit solve receives
            # the capped akz.  Without it the explicit off-diagonal terms at
            # NEMO-strength kappa exceed their stability limit: the 1-year
            # r1_exact screen ran away after day ~100 (T 27 -> 52 -> 90 C)
            # with msc off, matching the missing-MSC signature (#1226).
            msc_stabilize=True,
            mld_criterion=cfg.gm_redi_mld_criterion,
            visbeck=VisbeckConfig(enabled=False),
            treguier=TreguierConfig(enabled=False),
        )
    else:
        gm_redi_cfg = GMRediConfig(
            # Placeholders; ignored at runtime because an adaptive κ is
            # enabled.  Anchored to visbeck_kappa_min so the static value
            # is non-degenerate if the adaptive path is ever mis-wired.
            kappa_GM=cfg.visbeck_kappa_min,
            kappa_Redi=cfg.visbeck_kappa_min,
            S_max=cfg.redi_S_max,
            slope_scheme=cfg.gm_redi_slope_scheme,
            gm_bolus_advection=cfg.gm_bolus_advection,
            mld_criterion=cfg.gm_redi_mld_criterion,
            # Exactly ONE adaptive-κ diagnostic on (the GM/Redi dispatch
            # raises if both are enabled): "visbeck" (historical) or
            # "treguier" (the NEMO nn_aei_ijk_t=21 oracle scaling, cap
            # aei0 = rn_Ue·rn_Le).
            visbeck=VisbeckConfig(
                enabled=(cfg.gm_kappa_scheme == "visbeck"),
                alpha=cfg.visbeck_alpha,
                kappa_min=cfg.visbeck_kappa_min,
                kappa_max=cfg.visbeck_kappa_max,
            ),
            treguier=TreguierConfig(
                enabled=(cfg.gm_kappa_scheme == "treguier"),
                aei0=cfg.treguier_aei0,
            ),
        ) if cfg.use_gm_redi else None

    physics_cfg = None
    if physics:
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.convection.config import (
            EnhancedDiffusionConfig, OceanConvectionConfig,
        )
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.shortwave_penetration import (
            ShortwavePenetrationConfig,
        )
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        physics_cfg = OceanPhysicsConfig(
            vertical_mixing=_dino_vertical_mixing_config(cfg),
            # GM/Redi goes on the model config directly, not here.
            lateral_mixing=LateralMixingConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                # NEMO nn_evdm=1: EVD on tracers AND momentum (nu_conv =
                # rn_evd = K_conv). Gated off under kpp (k_profiles rejects
                # the combination — KPP's interior convective viscosity would
                # double-count; NEMO pairs EVD with TKE). The MPAS builder
                # below keeps nu_conv=0 (tracer-only; edge-momentum
                # convective mixing needs a TRiSK reconstruction).
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=cfg.K_conv,
                    nu_conv=(cfg.K_conv
                             if (cfg.evd_on_momentum
                                 and cfg.vmix_scheme != "kpp") else 0.0),
                    # NEMO zdfevd is a hard rn2<0 switch on adiabatic N^2; the
                    # nemo_paper recipe selects it (smooth default over-cools).
                    smooth_transition=cfg.convection_smooth_transition,
                    n2_mode=cfg.convection_n2_mode,
                    n2_threshold=cfg.convection_n2_threshold,
                ),
            ),
            shortwave_penetration=ShortwavePenetrationConfig(
                water_type=cfg.jerlov_water_type,
            ),
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),  # use model-level
        )

    # rigid_lid is the Veros-faithful ACC streamfunction solver; a BARE flip of
    # barotropic_solver runs away (568 Sv → NaN) because the bottom-drag
    # depth-mean never reaches the streamfunction barotropic balance — that fold
    # (ocean_model_latlon_cgrid.py, F_slow += du_diss depth-mean) is gated on
    # tend.du_diss, which is produced ONLY under ab2_scope="advective". So when
    # rigid_lid is selected, apply the FULL coordinated stack (veros_acc_recipe.py:
    # ab2 outer + explicit_ab2 Coriolis + ab2_scope="advective" + dt_mom_ratio).
    # Validated: DINO rigid_lid then spins up STABLY (7→31 Sv/180 d, still rising).
    # Momentum / Coriolis / integrator scheme identity — the L2 recipe axes.
    # Defaults reproduce the legoESM DINO stack (vector-invariant momentum,
    # Matsuno-split Coriolis, forward-Euler outer, total AB2 scope); the
    # mitgcm/oceananigans/veros cards override them via DINOConfig (DINO_RECIPES).
    # rigid_lid ALWAYS forces the coordinated Veros-faithful ab2 stack (the only
    # barotropic path whose Coriolis is AB2-consistent — see the field note), so
    # it wins last.
    _scheme = dict(
        momentum_advection=cfg.momentum_advection,
        momentum_flux_scheme=cfg.momentum_flux_scheme,
        coriolis_scheme=cfg.coriolis_scheme,
        outer_integrator=cfg.outer_integrator,
        vorticity_scheme=cfg.vorticity_scheme,
        asselin_gamma=cfg.asselin_gamma,
        ab2_scope=cfg.ab2_scope,
        # Routed into config.barotropic by from_flat.  Required by the
        # oceananigans card (explicit_ab2 × implicit_cn): keeps the
        # barotropic-mode Coriolis AB2-extrapolated instead of forward-Euler
        # (see DINOConfig.barotropic_slow_forcing_ab2).
        barotropic_slow_forcing_ab2=cfg.barotropic_slow_forcing_ab2,
    )
    if cfg.barotropic_solver == "rigid_lid":
        _scheme.update(
            outer_integrator="ab2", coriolis_scheme="explicit_ab2",
            ab2_scope="advective", dt_mom_ratio=cfg.rigid_lid_dt_mom_ratio,
            # The rigid-lid streamfunction projection removes barotropic
            # inertial modes entirely (the FE-Coriolis hazard the flag cures
            # does not exist there), and the flag's validation rejects
            # ab2_scope="advective" — force it OFF under the coordinated
            # rigid-lid stack regardless of the recipe card.
            barotropic_slow_forcing_ab2=False,
        )

    model_cfg = LatLonCGridOceanConfig.from_flat(
        rho_0=cfg.rho_0,
        # NEMO dynzdf wind placement (see DINOConfig.surface_stress_implicit).
        surface_stress_implicit=cfg.surface_stress_implicit,
        A_h=A_h_base,
        A_h_lat_scaling=True,         # cos(lat) per-row scaling — Phase 1B
        # Node 14: "nemo_div_curl" embeds ahmt/ahmf=½·rn_Uv·MAX(e1,e2) inside the
        # div/curl (ignores A_h_lat_scaling / eq-boost / floor, which stay OFF on
        # the faithful NEMO cards). Default "vector_laplacian" keeps the A_h·cos(φ)
        # scalar path byte-identical for every other recipe.
        lateral_viscosity_operator=cfg.lateral_viscosity_operator,
        K_h=(0.0 if cfg.lateral_tracer_mixing == "isoneutral"
             else K_h_base),   # iso-neutral replaces iso-level diffusion
        A_v=cfg.A_v_bg_effective,   # TKE: 4× floor at the SW-corner stabilizer
        K_v=cfg.K_v_bg,
        bottom_drag_r=bottom_drag_r,
        bottom_drag_bg_velocity=cfg.bottom_drag_bg_velocity,
        bottom_drag_bbl_thickness=cfg.bottom_drag_bbl_thickness,
        bottom_drag_scheme=cfg.bottom_drag_scheme,
        bottom_drag_cd0=cfg.C_d_bottom,
        n_barotropic_substeps=_dino_barotropic_substeps(grid, cfg),
        barotropic_solver=cfg.barotropic_solver,
        barotropic_implicit_theta_eta=cfg.barotropic_implicit_theta_eta,
        barotropic_time_filter=cfg.barotropic_time_filter,
        barotropic_coriolis=cfg.barotropic_coriolis,
        barotropic_coriolis_split=cfg.barotropic_coriolis_split,
        **_scheme,
        tracer_advection=cfg.tracer_advection,
        pgf_scheme=cfg.pgf_scheme,
        pgf_quadrature=cfg.pgf_quadrature,
        ke_gradient_scheme=cfg.ke_gradient_scheme,  # #263 Hollingsworth fix
        # MOM6-style stability protection (diagnosed 2026-05-14)
        A_h_floor=cfg.A_h_floor,
        A_h_eq_boost=cfg.A_h_eq_boost,
        A_h_eq_sigma_deg=cfg.A_h_eq_sigma_deg,
        gm_redi=gm_redi_cfg,           # lat-lon C-grid GM/Redi direct path
        physics=physics_cfg,
        eos=cfg.eos,                   # "wright" (default) | "nemo_seos" (paper)
        eos_depth=cfg.eos_depth,       # "insitu" (default) | "geometric" (nemo_paper)
    )
    return model_cfg, physics_cfg


def dino_mpas_state(
    mesh,
    z_coord: OceanZStarCoordinate,
    cfg: DINOConfig | None = None,
    seam_strip_width_deg: float | None = None,
):
    """Build a full MPASOceanState for DINO from rest with IC stratification.

    Wraps the (T, S, H_bathy, land_mask) arrays from
    ``dino_mpas_initial_state_arrays`` into Field objects matching the
    legoESM MPAS convention, with u = 0, η = 0, w = 0.

    Returns
    -------
    MPASOceanState
    """
    from legoesm.core.state import MPASOceanState

    if cfg is None:
        cfg = DINOConfig()

    T, S, H_bathy, land_mask = dino_mpas_initial_state_arrays(
        mesh, z_coord, cfg, seam_strip_width_deg=seam_strip_width_deg,
    )

    nlev = z_coord.n_levels
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    dtype = T.dtype

    return MPASOceanState(
        u=Field(jnp.zeros((nEdges, nlev), dtype=dtype),
                "u", ("nEdges", "nlev"), "m/s"),
        T=Field(T, "T", ("nCells", "nlev"), "degC"),
        S=Field(S, "S", ("nCells", "nlev"), "PSU"),
        eta=Field(jnp.zeros(nCells, dtype=dtype),
                  "eta", ("nCells",), "m"),
        w=Field(jnp.zeros((nCells, nlev + 1), dtype=dtype),
                "w", ("nCells", "nlev+1"), "m/s"),
        H_bathy=Field(H_bathy, "H_bathy", ("nCells",), "m"),
        land_mask=Field(land_mask, "land_mask", ("nCells",), "1"),
    )


_MPAS_TRACER_ADVECTION = ("upwind", "tvd", "superbee")


def _mpas_tracer_advection(cfg: "DINOConfig") -> str:
    """Validate the configured tracer advection for the MPAS DINO path.

    MPASOceanModel only implements upwind / tvd / superbee; the FCT
    family (ppm_fct, fct2 — e.g. from the latlon-oriented ``r1_exact``
    preset) would be rejected deep inside model construction.  Raise
    here with actionable guidance instead (dispatch discipline).
    """
    if cfg.tracer_advection not in _MPAS_TRACER_ADVECTION:
        raise ValueError(
            f"DINOConfig.tracer_advection={cfg.tracer_advection!r} is not "
            f"available on the MPAS DINO path (supported: "
            f"{_MPAS_TRACER_ADVECTION}). The r1_exact preset is lat-lon "
            "only — override tracer_advection (e.g. "
            "dino_r1_exact_config(tracer_advection='tvd')) to run its "
            "other levers on MPAS."
        )
    return cfg.tracer_advection


def dino_mpas_model_config(
    mesh,
    cfg: DINOConfig | None = None,
    physics: bool = True,
):
    """Build (MPASOceanConfig, OceanPhysicsConfig) for DINO.

    Translates DINOConfig fields to the MPAS model + physics configs
    needed by ``MPASOceanModel``.

    Parameters
    ----------
    mesh : VoronoiMesh
        Used to derive a representative cell-resolution scalar for
        lateral mixing coefficients (``A_h ≈ 0.5·U_M·sqrt(<area>)``).
        For a quasi-uniform regional mesh this is the right magnitude;
        the per-cell variant matters more, but is unimplemented here
        and not pursued (Decisions Log 2026-05-14): a uniform-mesh
        approximation is within ~10% of the proper per-cell scaling.
    cfg : DINOConfig, optional
    physics : bool
        If True, include KPP + GM/Redi + enhanced-diffusion convection
        + Jerlov SW. If False, return ``physics=None`` (dycore only —
        for rest-state smoke tests).

    Returns
    -------
    model_config : MPASOceanConfig
    physics_config : OceanPhysicsConfig or None
    """
    from legoesm.ocean.mpas_config import MPASOceanConfig

    if cfg is None:
        cfg = DINOConfig()
    _mpas_tracer_advection(cfg)   # fail fast BEFORE any mesh access
    if cfg.vertical_coordinate != "zstar":
        raise ValueError(
            f"DINOConfig.vertical_coordinate={cfg.vertical_coordinate!r} "
            "is not available on the MPAS DINO path (MPAS keeps its own "
            "column handling). The r1_exact preset is lat-lon only — "
            "override vertical_coordinate='zstar' to run on MPAS.")
    if cfg.lateral_tracer_mixing != "geopotential":
        raise ValueError(
            f"DINOConfig.lateral_tracer_mixing="
            f"{cfg.lateral_tracer_mixing!r} is not available on the MPAS "
            "DINO path (the Redi-only iso-neutral recipe is wired for the "
            "lat-lon C-grid). Override lateral_tracer_mixing="
            "'geopotential' to run on MPAS.")
    if cfg.barotropic_time_filter != "cosine" or cfg.barotropic_auto_cmax > 0:
        raise ValueError(
            "DINOConfig.barotropic_time_filter="
            f"{cfg.barotropic_time_filter!r} / barotropic_auto_cmax="
            f"{cfg.barotropic_auto_cmax!r}: the MPAS DINO builder does not "
            "thread these (it would silently run different barotropic "
            "numerics — codex r9 P2). The centred split-explicit recipe is "
            "lat-lon only; override barotropic_time_filter='cosine' and "
            "barotropic_auto_cmax=0.0 to run on MPAS.")

    # Representative cell size from mean cell area (m).
    cell_dx_m = float(jnp.sqrt(jnp.mean(mesh.areaCell)))
    A_h = 0.5 * cfg.U_M * cell_dx_m
    K_h = 0.5 * cfg.U_T * cell_dx_m

    # MOM6 quadratic-with-floor drag: r = C_d * u_bg.
    bottom_drag_r = cfg.C_d_bottom * cfg.bottom_drag_bg_velocity

    model_config = MPASOceanConfig(
        rho_0=cfg.rho_0,
        A_h=A_h,
        # Equatorial-waveguide stabilizer (f→0): boosts A_h near the equator
        # with the same tight Gaussian the lat-lon A_h_eq_boost uses. Without
        # it the forced equatorial jet runs away on the implicit-CN MPAS path
        # — see the mpas_equatorial_visc_boost note in DINOConfig.
        equatorial_visc_boost=cfg.mpas_equatorial_visc_boost,
        K_h=K_h,
        A_v=cfg.A_v_bg_effective,   # TKE: 4× floor at the SW-corner stabilizer
        K_v=cfg.K_v_bg,
        bottom_drag_r=bottom_drag_r,
        bottom_drag_bg_velocity=cfg.bottom_drag_bg_velocity,
        bottom_drag_bbl_thickness=cfg.bottom_drag_bbl_thickness,
        bottom_drag_scheme=cfg.bottom_drag_scheme,
        bottom_drag_cd0=cfg.C_d_bottom,
        n_barotropic_substeps=cfg.n_barotropic_substeps,
        barotropic_solver=cfg.barotropic_solver,
        tracer_advection=_mpas_tracer_advection(cfg),
        implicit_vertical_mixing=True,
        eos=cfg.eos,                   # "wright" (default) | "nemo_seos" (paper)
    )

    if not physics:
        return model_config, None

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, LateralMixingConfig, VisbeckConfig,
    )
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig,
    )
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig

    if cfg.use_gm_redi and cfg.gm_kappa_scheme != "visbeck":
        raise ValueError(
            f"DINOConfig.gm_kappa_scheme={cfg.gm_kappa_scheme!r} is not "
            "supported on the MPAS grid (Treguier adaptive kappa is lat-lon "
            "C-grid only); use gm_kappa_scheme='visbeck' or --grid latlon.")
    physics_config = OceanPhysicsConfig(
        vertical_mixing=_dino_vertical_mixing_config(cfg),
        lateral_mixing=LateralMixingConfig(
            scheme="gm_redi" if cfg.use_gm_redi else "none",
            gm_redi=GMRediConfig(
                # Placeholders; ignored at runtime because Visbeck is enabled.
                # Anchored to visbeck_kappa_min so the static value is non-
                # degenerate if the Visbeck path is ever mis-wired.
                kappa_GM=cfg.visbeck_kappa_min,
                kappa_Redi=cfg.visbeck_kappa_min,
                S_max=cfg.redi_S_max,
                slope_scheme=cfg.gm_redi_slope_scheme,
                visbeck=VisbeckConfig(
                    enabled=True,
                    alpha=cfg.visbeck_alpha,
                    kappa_min=cfg.visbeck_kappa_min,
                    kappa_max=cfg.visbeck_kappa_max,
                ),
            ),
        ),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=cfg.K_conv,
                # Same NEMO zdfevd hard-switch fidelity as the lat-lon path.
                smooth_transition=cfg.convection_smooth_transition,
                n2_mode=cfg.convection_n2_mode,
                n2_threshold=cfg.convection_n2_threshold,
            ),
        ),
        shortwave_penetration=ShortwavePenetrationConfig(
            water_type=cfg.jerlov_water_type,
        ),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),  # using model-level drag
    )
    # Wire the physics INTO the model config (MPASOceanConfig.physics) — the
    # MPAS model gates KPP/GM-Redi/convection on ``config.physics is not None``
    # (ocean_model_mpas.py). The lat-lon path does this via
    # LatLonCGridOceanConfig.from_flat(physics=...); without it here the run_dino
    # caller (``model_cfg, _ = dino_mpas_model_config(...)``) drops the returned
    # physics_config and the MPAS ocean runs DYCORE-ONLY (no boundary-layer
    # mixing, no eddy parameterization, no convection).
    model_config = model_config._replace(physics=physics_config)
    return model_config, physics_config


# Convenience: surface-layer restoring timescales derived from heat-flux
# coefficients (eq 8 of paper). Useful for sanity printouts.
def restoring_timescale_T_days(cfg: DINOConfig) -> float:
    """τ_T = ρ₀ · c_p · Δz₀ / A_Θ, in days (≈ 11.85 d for default)."""
    seconds = cfg.rho_0 * cfg.c_p * cfg.dz_min / cfg.A_theta
    return seconds / 86400.0


def restoring_timescale_S_days(cfg: DINOConfig) -> float:
    """τ_S = ρ₀ · Δz₀ / A_S, in days (≈ 30.8 d for default)."""
    seconds = cfg.rho_0 * cfg.dz_min / cfg.A_S
    return seconds / 86400.0


# ---------------------------------------------------------------------
# Experiment-registry interface (expected by AVAILABLE_EXPERIMENTS in
# legoesm.ocean.experiments.__init__). Thin grid-type dispatchers that
# call the lat-lon and MPAS builders defined above.
# ---------------------------------------------------------------------

def create_initial_conditions(grid_type: str, grid, z_coord,
                               config: DINOConfig | None = None):
    """Build the DINO initial state for ``grid_type ∈ {"latlon","mpas"}``."""
    if config is None:
        config = DINOConfig()
    if grid_type == "latlon":
        return dino_lat_lon_state(grid, z_coord, config)
    if grid_type == "mpas":
        return dino_mpas_state(grid, z_coord, config)
    raise ValueError(f"Unknown grid_type: {grid_type!r}; "
                     f"DINO supports 'latlon' (Mercator) and 'mpas' "
                     f"(regional Voronoi).")


def create_forcings(grid_type: str, grid, z_coord,
                    config: DINOConfig | None = None):
    """Return (model_config, physics_config, surface_forcing_arrays).

    ``surface_forcing_arrays`` is a precomputed dict (wind τ, T*, S*,
    Q_sr); apply per timestep via
    ``apply_dino_{lat_lon,mpas}_surface_forcing``.
    """
    if config is None:
        config = DINOConfig()
    if grid_type == "latlon":
        model_cfg, phys_cfg = dino_lat_lon_model_config(grid, config, physics=True)
        forcing = dino_lat_lon_surface_forcing_arrays(grid, config)
    elif grid_type == "mpas":
        model_cfg, phys_cfg = dino_mpas_model_config(grid, config, physics=True)
        forcing = dino_mpas_surface_forcing_arrays(grid, config)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type!r}")
    return {"model_config": model_cfg,
            "physics_config": phys_cfg,
            "surface_forcing": forcing}


def validate_results(final_state, diagnostics: dict | None = None,
                     config: DINOConfig | None = None) -> tuple[bool, str]:
    """Basic shake-down validation: no NaN, T/S in physical range,
    velocities not absurd. Multi-decade-spinup checks (ACC transport,
    MOC topology, σ_2 stratification) are out of scope for the local
    1-year cap and live in the long-run analysis pipeline."""
    if config is None:
        config = DINOConfig()
    notes = []
    for fld in ("u", "T", "S", "eta"):
        data = getattr(final_state, fld).data
        import jax.numpy as jnp
        if not bool(jnp.all(jnp.isfinite(data))):
            return False, f"NaN/Inf in final {fld}"
    import numpy as np
    mask = np.asarray(final_state.land_mask.data) > 0.5
    T = np.asarray(final_state.T.data)
    S = np.asarray(final_state.S.data)
    if mask.any():
        T_oc = T[mask, :] if T.ndim == 3 else T[mask]
        S_oc = S[mask, :] if S.ndim == 3 else S[mask]
        if not (-3.0 < T_oc.min() and T_oc.max() < 32.0):
            return False, f"T out of range [{T_oc.min():.2f},{T_oc.max():.2f}]"
        if not (30.0 < S_oc.min() and S_oc.max() < 40.0):
            return False, f"S out of range [{S_oc.min():.2f},{S_oc.max():.2f}]"
    u_max = float(np.max(np.abs(final_state.u.data)))
    eta_max = float(np.max(np.abs(final_state.eta.data)))
    if u_max > 5.0:
        return False, f"|u| max = {u_max:.2f} m/s — instability"
    if eta_max > 5.0:
        return False, f"|η| max = {eta_max:.2f} m — barotropic blowup"
    notes.append(f"|u|max={u_max:.3f}, |η|max={eta_max:.3f}")
    return True, ", ".join(notes)


EXPERIMENT_CONFIG = {
    "name": "dino",
    "description": (
        "DINO (Diabatic Neverworld Ocean) — Kamm et al. 2025 GMD. "
        "Pole-to-pole sector basin with a re-entrant channel; tests "
        "diabatic processes (convection, deep water formation, MOC) "
        "in an idealized configuration."
    ),
    "scientific_purpose": (
        "Replicate the DINO 1° R1 reference experiment for testing "
        "and training subgrid eddy parameterizations. Provides ACC, "
        "MOC, MHT, and σ_2 stratification benchmarks."
    ),
    "reference": "Kamm, D., Deshayes, J., & Madec, G. (2025). DINO. GMD 18, 8091-8107.",
    "config_class": DINOConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "validate": validate_results,
    "default_duration": 30.0,   # days (1-month shake-down)
    "quick_duration": 1.0,      # days (smoke test)
    "grid_support": {
        "cubed_sphere": False,  # not implemented; basin-channel topology incompatible
        "latlon": True,         # Mercator, n_lon=50 = 1° R1
        "mpas": True,           # regional Voronoi, 97 km
        "spectral": False,
    },
}


# ---------------------------------------------------------------------
# Production-time surface forcing applicators (Phase 4)
#
# Applied as explicit external tendencies after each model.step(), NOT
# through legoESM's `OceanSurfaceForcing` / surface_forcing physics path
# (the general API uses timescales not heat-flux coefficients, and
# doesn't subtract Q_sr per paper eq 8). See Decisions Log 2026-05-14.
# ---------------------------------------------------------------------

def dino_lat_lon_surface_forcing_arrays(grid, cfg: DINOConfig | None = None):
    """Pre-compute lat-lon Mercator forcing fields that don't depend on state.

    Returns dict with:
      ``tau_u_face`` (n_lat, n_lon+1) — zonal wind stress at u-faces
      ``T_star_2d``  (n_lat, n_lon)   — annual-mean T restoring target
      ``S_star_2d``  (n_lat, n_lon)   — S restoring target with eq dip
      ``Q_sr_2d``    (n_lat, n_lon)   — annual-mean surface solar
    """
    if cfg is None:
        cfg = DINOConfig()

    lat_1d = jnp.degrees(grid.lat)               # (n_lat,)
    T_star_1d = dino_T_star_annual_mean(lat_1d, cfg)
    S_star_1d = dino_S_star(lat_1d, cfg)
    Q_sr_1d = dino_Q_sr_annual_mean(lat_1d, cfg)
    tau_u_1d = dino_wind_stress(lat_1d, cfg)     # (n_lat,)

    shape_2d = (grid.n_lat, grid.n_lon)
    tau_u_cell_1d = dino_wind_stress(lat_1d, cfg)      # τ at CELL lats
    # NEMO taum (usrdef_sbc:222-223): |τ|, boosted x1.3 in the westerlies
    # (utau > 0) — the TKE surface input only, never the momentum stress.
    taum_1d = jnp.abs(tau_u_cell_1d) * jnp.where(
        tau_u_cell_1d > 0.0, cfg.taum_westerly_boost, 1.0)
    return {
        "lat_deg_1d": lat_1d,   # for the seasonal (annual-cycle) recompute
        "tau_u_cell_2d": jnp.broadcast_to(tau_u_cell_1d[:, None], shape_2d),
        "taum_2d": jnp.broadcast_to(taum_1d[:, None], shape_2d),
        "T_star_2d": jnp.broadcast_to(T_star_1d[:, None], shape_2d),
        "S_star_2d": jnp.broadcast_to(S_star_1d[:, None], shape_2d),
        "Q_sr_2d":   jnp.broadcast_to(Q_sr_1d[:, None], shape_2d),
        "tau_u_face": jnp.broadcast_to(
            tau_u_1d[:, None], (grid.n_lat, grid.n_lon + 1)
        ),
    }


def dino_step_surface_forcing(forcing):
    """OceanSurfaceForcing for model.step() when ``wind_through_step``.

    Carries ONLY the wind: ``tau_x`` (cell-centred zonal stress; the PE
    external-tau block applies the ocean reaction and the top-layer
    deposit) and ``taum`` (the NEMO stress-modulus channel with the
    usrdef x1.3 westerly boost, consumed by the TKE surface input).
    Heat/salt/SW stay on the analytic post-step applicator.
    """
    from legoesm.ocean.state import OceanSurfaceForcing

    # SIGN CONVENTION: the PE external-tau block treats (tau_x, tau_y) in
    # the ATMOSPHERIC convention and applies the OCEAN REACTION -tau (see
    # compute_omip2_surface_forcing).  DINO's analytic tau is the stress ON
    # the ocean (+0.2 Pa accelerates the ocean eastward), so negate here;
    # taum is a modulus (sign-free).
    return OceanSurfaceForcing(
        tau_x=-forcing["tau_u_cell_2d"],
        tau_y=jnp.zeros_like(forcing["tau_u_cell_2d"]),
        taum=forcing["taum_2d"],
    )


def apply_dino_lat_lon_surface_forcing(state, forcing, z_coord, cfg, dt,
                                        t_seconds=None):
    """Apply DINO surface forcing on the lat-lon Mercator grid.

    Components (paper eqs 7-10):
      eq 7  — wind τ_u → top-layer u tendency
      eq 8  — non-solar T restoring (A_θ(T*-T) − Q_sr) at top layer,
               via legoesm.ocean.physics.surface_forcing.restoring with
               implicit=True (analytical implicit-Euler — stable for
               any dt; required for DINO at paper-spec K_conv=100,
               τ_T=11.85 days; see issue #266 / PR #267)
      eq 9  — A_S(S*-S) salinity restoring at top layer (also implicit)
      eq 10 — Jerlov type I column-distributed Q_sr through all levels

    Returns a new state (immutable update of T, S, u).
    """
    from legoesm.ocean.physics.surface_forcing.config import (
        RestoringConfig, tau_from_flux_coefficient,
    )
    from legoesm.ocean.physics.surface_forcing.restoring import (
        restoring_surface_forcing,
    )

    dz_0 = float(z_coord.dz_ref[0])
    cell_mask = state.land_mask.data

    # Oracle annual cycle (ln_ann_cyc): T* and Q_sr are TIME-DEPENDENT —
    # recompute from the 360-day-year phases at this step's model time.
    # Fail loud if the caller did not thread t_seconds (a silent fallback
    # to the annual-mean arrays would fake the seasonal run).
    T_star_2d = forcing["T_star_2d"]
    Q_sr_2d = forcing["Q_sr_2d"]
    if getattr(cfg, "forcing_annual_cycle", False):
        if t_seconds is None:
            raise ValueError(
                "DINOConfig.forcing_annual_cycle=True requires t_seconds "
                "to be passed to apply_dino_lat_lon_surface_forcing")
        _lat1 = forcing["lat_deg_1d"]
        _shape = T_star_2d.shape
        T_star_2d = jnp.broadcast_to(
            dino_T_star_seasonal(_lat1, t_seconds, cfg)[:, None], _shape)
        Q_sr_2d = jnp.broadcast_to(
            dino_Q_sr_seasonal(_lat1, t_seconds, cfg)[:, None], _shape)

    # T/S restoring via the legoESM module with implicit=True. Paper
    # eq 8 split = subtract_qsr=True (Q_sr provided as sw_down).
    tau_T = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0)
    tau_S = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0)
    restoring_cfg = RestoringConfig(
        tau_T=tau_T, tau_S=tau_S,
        T_star_array=T_star_2d,
        S_star_array=forcing["S_star_2d"],
        subtract_qsr=True,
        implicit=True,
    )
    rest_out = restoring_surface_forcing(
        state.T.data, state.S.data, _LatLonGridShim(state, cell_mask),
        restoring_cfg,
        sw_down=Q_sr_2d, dt=dt,
        rho_0=cfg.rho_0, c_p=cfg.c_p, dz_0=dz_0,
    )

    # Jerlov SW penetration through the column (eq 10): 3D tendency.
    # Q_sr is added to the column distribution AND subtracted from the
    # surface non-solar flux (handled by restoring.subtract_qsr above)
    # so total heat is conserved (eq 8 + eq 10 = A_θ(T*-T)).
    jacobian = jnp.ones_like(state.eta.data)
    sw_cfg = ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type)
    dT_dt_sw = shortwave_penetration_tendency(
        sw_down=Q_sr_2d,
        z_coord_dz_ref=z_coord.dz_ref,
        z_coord_z_half_ref=z_coord.z_half_ref,
        jacobian=jacobian, config=sw_cfg,
        rho_0=cfg.rho_0, c_sw=cfg.c_p,
    )

    # Combine: forward-Euler tracer update with all tendencies summed.
    # Restoring tendency is "effective" (already accounts for implicit
    # Euler at given dt — stable for any dt). Mask land everywhere.
    mask3 = cell_mask[..., None]
    new_T = state.T.data + dt * (rest_out.dT_dt + dT_dt_sw) * mask3
    new_S = state.S.data + dt * rest_out.dS_dt * mask3

    # u tendency at u-faces (eq 7) — SKIPPED when the wind goes through
    # model.step(surface_forcing=...) (wind_through_step: the dynamics-core
    # external-tau block owns it; applying here too would double the wind).
    if getattr(cfg, "wind_through_step", False):
        new_u = state.u.data
    else:
        u_top = state.u.data[..., 0]
        du_dt_top = dino_top_layer_u_tendency(forcing["tau_u_face"], dz_0, cfg)
        u_face_mask = state.u_mask.data
        new_u_top = u_top + dt * du_dt_top * u_face_mask
        new_u = state.u.data.at[..., 0].set(new_u_top)

    return state._replace(
        T=Field(data=new_T, name=state.T.name, dims=state.T.dims, units=state.T.units),
        S=Field(data=new_S, name=state.S.name, dims=state.S.dims, units=state.S.units),
        u=Field(data=new_u, name=state.u.name, dims=state.u.dims, units=state.u.units),
    )


class _LatLonGridShim:
    """Tiny adapter so restoring_surface_forcing can pull a (n_lat, n_lon)
    grid_lat from the state's land_mask shape — avoids passing the full
    grid object through the applicator signature."""
    def __init__(self, state, cell_mask):
        # The arrays we feed already have shape (n_lat, n_lon, nlev) for T/S
        # and (n_lat, n_lon) for sw_down/T_star_array, so grid_lat just
        # needs to broadcast to (n_lat, n_lon).
        self.grid_lat = jnp.zeros_like(cell_mask)  # values irrelevant
                                                    # (T*/S* are user-arrays)


def dino_mpas_surface_forcing_arrays(mesh, cfg: DINOConfig | None = None):
    """Pre-compute MPAS regional forcing fields.

    Returns dict with:
      ``T_star_1d`` (nCells,) — annual-mean T restoring target
      ``S_star_1d`` (nCells,) — S restoring target with eq dip
      ``Q_sr_1d``   (nCells,) — annual-mean surface solar
      ``tau_normal`` (nEdges,) — wind stress projected onto edge normal
        (τ_v = 0 for DINO, so τ_n = τ_u · cos(angleEdge))
    """
    if cfg is None:
        cfg = DINOConfig()

    lat_c_deg = jnp.degrees(mesh.latCell)
    T_star_1d = dino_T_star_annual_mean(lat_c_deg, cfg)
    S_star_1d = dino_S_star(lat_c_deg, cfg)
    Q_sr_1d = dino_Q_sr_annual_mean(lat_c_deg, cfg)

    lat_e_deg = jnp.degrees(mesh.latEdge)
    tau_u_e = dino_wind_stress(lat_e_deg, cfg)         # (nEdges,)
    tau_normal = tau_u_e * jnp.cos(mesh.angleEdge)     # (nEdges,)

    return {
        "T_star_1d": T_star_1d,
        "S_star_1d": S_star_1d,
        "Q_sr_1d": Q_sr_1d,
        "tau_normal": tau_normal,
    }


def apply_dino_mpas_surface_forcing(state, forcing, z_coord, cfg, dt,
                                    t_seconds=None):
    """Apply DINO surface forcing on the MPAS regional mesh.

    Same physics as lat-lon (paper eqs 7-10) with edge-projected wind
    and 1D cell-indexed T*, S*, Q_sr.
    """
    if getattr(cfg, "forcing_annual_cycle", False):
        raise NotImplementedError(
            "DINOConfig.forcing_annual_cycle=True is wired on the lat-lon "
            "DINO path only (seasonal T*/Q_sr recompute); the MPAS apply "
            "still uses the annual-mean arrays — reject rather than "
            "silently run the wrong forcing.")
    dz_0 = float(z_coord.dz_ref[0])
    cell_mask = state.land_mask.data                  # (nCells,)

    # T tendency at top layer (eq 8)
    T_top = state.T.data[:, 0]                         # (nCells,)
    dT_dt_top = dino_top_layer_T_tendency(
        T_top, forcing["T_star_1d"], forcing["Q_sr_1d"], dz_0, cfg,
    )

    # Jerlov SW penetration through the column (eq 10)
    jacobian = jnp.ones_like(state.eta.data)           # (nCells,)
    sw_cfg = ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type)
    dT_dt_sw = shortwave_penetration_tendency(
        sw_down=forcing["Q_sr_1d"],
        z_coord_dz_ref=z_coord.dz_ref,
        z_coord_z_half_ref=z_coord.z_half_ref,
        jacobian=jacobian,
        config=sw_cfg,
        rho_0=cfg.rho_0,
        c_sw=cfg.c_p,
    )  # (nCells, nlev)

    new_T = state.T.data + dt * dT_dt_sw * cell_mask[:, None]
    new_T_top = new_T[:, 0] + dt * dT_dt_top * cell_mask
    new_T = new_T.at[:, 0].set(new_T_top)

    # S tendency at top layer (eq 9)
    S_top = state.S.data[:, 0]
    dS_dt_top = dino_top_layer_S_tendency(
        S_top, forcing["S_star_1d"], dz_0, cfg,
    )
    new_S_top = S_top + dt * dS_dt_top * cell_mask
    new_S = state.S.data.at[:, 0].set(new_S_top)

    # u tendency at edges (eq 7). The MPAS dynamics gates fluxes to
    # land cells internally, so no explicit per-edge mask is needed.
    u_top = state.u.data[:, 0]
    du_dt_top = dino_top_layer_u_tendency(forcing["tau_normal"], dz_0, cfg)
    new_u_top = u_top + dt * du_dt_top
    new_u = state.u.data.at[:, 0].set(new_u_top)

    return state._replace(
        T=Field(data=new_T, name=state.T.name, dims=state.T.dims, units=state.T.units),
        S=Field(data=new_S, name=state.S.name, dims=state.S.dims, units=state.S.units),
        u=Field(data=new_u, name=state.u.name, dims=state.u.dims, units=state.u.units),
    )
