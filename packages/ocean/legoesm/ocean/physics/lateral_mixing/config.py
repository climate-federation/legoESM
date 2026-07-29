"""Configuration for ocean lateral mixing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig


__param_spec__ = {
    "TreguierConfig": {
        "scheme_key": "ocean.lat.treguier",
        "excluded": {
            "kappa_min": "numerics: stability floor on the equatorial taper, NOT a NEMO namelist parameter; default 0 = inactive (enable via config, not training)",
        },
        "params": {
            "aei0": {"units": "m2 s-1", "bounds": (500.0, 10000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "NEMO ldftra nn_aei_ijk_t=21 (Treguier 1997); aei0=rn_Ue*rn_Le", "shape": None},
        },
    },
    "HarmonicConfig": {
        "scheme_key": "ocean.lat.harmonic",
        "excluded": {
            "cfl_dt_estimate": "numerics: solver/CFL/smoothing parameter",
            "cfl_safety": "numerics: solver/CFL/smoothing parameter",
        },
        "params": {
            "A_h": {"units": "m^2/s", "bounds": (3300.0, 30000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "harmonic lateral viscosity/diffusivity", "shape": None},
            "K_h": {"units": "m^2/s", "bounds": (330.0, 3000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "harmonic lateral viscosity/diffusivity", "shape": None},
        },
    },
    "BiharmonicConfig": {
        "scheme_key": "ocean.lat.biharmonic",
        "excluded": {
            "B_h_momentum": "default 0 = disabled/off (enable via config, not training)",
            "B_h_tracer": "default 0 = disabled/off (enable via config, not training)",
            "cfl_dt_estimate": "numerics: solver/CFL/smoothing parameter",
            "cfl_safety": "numerics: solver/CFL/smoothing parameter",
        },
        "params": {
        },
    },
    "VisbeckConfig": {
        "scheme_key": "ocean.lat.visbeck",
        "excluded": {
            "f_min": "numerics: floor/cap",
        },
        "params": {
            "L_fixed": {"units": "m", "bounds": (33000.0, 300000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Visbeck et al. (1997)", "shape": None},
            "L_max": {"units": "m", "bounds": (66000.0, 600000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Visbeck et al. (1997)", "shape": None},
            "L_min": {"units": "m", "bounds": (1650.0, 15000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Visbeck et al. (1997)", "shape": None},
            "alpha": {"units": "1", "bounds": (0.00495, 0.045), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Visbeck et al. (1997)", "shape": None},
            "kappa_max": {"units": "m^2/s", "bounds": (1320.0, 12000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Visbeck et al. (1997)", "shape": None},
            "kappa_min": {"units": "m^2/s", "bounds": (33.0, 300.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Visbeck et al. (1997)", "shape": None},
        },
    },
    "GMRediConfig": {
        "scheme_key": "ocean.lat.gm_redi",
        "excluded": {
            "K_iso_steep": "default 0 = disabled/off (enable via config, not training)",
            "taper_width_frac": "numerics: solver/CFL/smoothing parameter",
            "mld_rho_c": "convention: mixed-layer-depth density criterion (NEMO ldfslp ramp)",
        },
        "params": {
            "S_max": {"units": "1", "bounds": (0.0033, 0.03), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Gent-McWilliams / Redi", "shape": None},
            "kappa_GM": {"units": "m^2/s", "bounds": (330.0, 3000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Gent-McWilliams / Redi", "shape": None},
            "kappa_Redi": {"units": "m^2/s", "bounds": (330.0, 3000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Gent-McWilliams / Redi", "shape": None},
            "surface_complement_depth": {"units": "m", "bounds": (33.0, 300.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Gent-McWilliams / Redi", "shape": None},
            "resfn_gamma": {"units": "1", "bounds": (1.0, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Hallberg 2013 (Ocean Modelling 72, 92) resolution function: grid points per deformation radius at half-suppression", "shape": None},
            "resfn_cbcl_ms": {"units": "m s-1", "bounds": (0.5, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Hallberg 2013 / Chelton et al. 1998 (JPO 28, 433): fixed first-baroclinic gravity-wave speed for L_d = c/|f|", "shape": None},
        },
    },
}


class HarmonicConfig(NamedTuple):
    """Laplacian (harmonic) lateral mixing.

    The explicit 2-D Laplacian operator is stable only when
    ``A_h · dt / dx² ≤ 1/4`` (2-D diffusive CFL, safety factor
    incorporated).  ``cfl_dt_estimate`` lets the scheme cap the
    diffusivity at the explicit limit without knowing the runtime dt;
    set higher (=longer dt) for stricter caps.  The cap is opt-in via
    ``enforce_cfl=True``.
    """
    A_h: float = 1e4   # Horizontal viscosity [m^2/s]
    K_h: float = 1e3   # Horizontal tracer diffusivity [m^2/s]
    enforce_cfl: bool = False         # Apply 2-D explicit CFL cap
    cfl_dt_estimate: float = 3600.0   # Reference dt for the cap [s]
    cfl_safety: float = 0.20          # Margin below 1/4 stability bound


class BiharmonicConfig(NamedTuple):
    """Biharmonic lateral mixing.

    Explicit biharmonic CFL is ``B_h · dt / dx⁴ ≤ 1/16`` (2-D, with a
    safety factor) for the legacy wide outer stencil; the compact outer
    stencil (``compact_outer=True``) has a tighter ``≤ 1/512`` bound
    (its 2Δx eigenvalue is ~1024/dx⁴, vs ~0 for the wide form).  See
    ``HarmonicConfig`` for the analogous CFL knobs.

    ``compact_outer`` selects the outer Laplacian of ``∇⁴ = ∇²(∇²)``:
    ``False`` (default) keeps the legacy wide ``div(grad)`` outer stage,
    which has an EXACT 2Δx null (does NOT damp the grid-scale checkerboard
    the biharmonic exists to remove) — retained as the default so
    coefficients tuned against it stay bit-identical.  ``True`` uses the
    compact outer Laplacian (``(1,-4,6,-4,1)`` stencil, maximal 2Δx
    damping, MOM/MPAS-faithful) and correspondingly narrows the coastal
    Neumann fill reach and the CFL cap.
    """
    B_h_momentum: float = 0.0   # Biharmonic viscosity [m^4/s]
    B_h_tracer: float = 0.0     # Biharmonic tracer diffusivity [m^4/s]
    enforce_cfl: bool = False
    cfl_dt_estimate: float = 3600.0
    cfl_safety: float = 0.05    # Margin below 1/16 stability bound
    compact_outer: bool = False  # Compact 2Δx-damping outer ∇² (MOM/MPAS del4)


class VisbeckConfig(NamedTuple):
    """Adaptive GM coefficient following Visbeck, Marshall, Haine & Spall 1997.

    Replaces the constant ``kappa_GM`` with a flow-dependent, horizontally
    varying coefficient

        κ_Visbeck(x, y) = α · L² · ⟨N · |S|⟩_z

    where ``α`` is a dimensionless tunable coefficient (Visbeck 97 suggest
    ~0.015; MOM6 uses values between 0.005 and 0.1), ``L`` is a mixing
    length — either a fixed width of the baroclinic zone or the local
    first-baroclinic Rossby radius ``N̄·H/|f|`` — and
    ``⟨N·|S|⟩_z`` is a depth-average of the local Eady-like growth rate
    at isopycnal interfaces.  The final coefficient is clamped to
    [``kappa_min``, ``kappa_max``] for numerical safety.

    Using ``N·|S|`` as the growth rate follows from the Eady relation
    ``σ = 0.31·|f|/√Ri`` combined with the thermal-wind identity
    ``Ri = f² / (N²·|S|²)``, which gives ``σ ≈ 0.31·N·|S|``; the 0.31
    factor is absorbed into the tunable ``α``.
    """
    enabled: bool = False
    alpha: float = 0.015
    L_fixed: float = 1.0e5               # Fixed mixing length [m]
    use_rossby_radius: bool = True
    L_min: float = 5.0e3                 # Length-scale floor [m]
    L_max: float = 2.0e5                 # Length-scale ceiling [m]
    kappa_min: float = 1.0e2             # κ floor [m²/s]
    kappa_max: float = 4.0e3             # κ ceiling [m²/s]
    f_min: float = 1.0e-6                # |f| floor for Rossby-radius denom
    # Static-stability N² mode for the Eady-growth / Rossby-radius chain.
    #   "insitu" (default, BIT-IDENTICAL legacy): N² from the in-situ density
    #     gradient (compute_buoyancy_frequency). The in-situ ∂_zρ carries the
    #     adiabatic compressibility term and is biased ~6x too stable, which
    #     makes ∫N dz ~2.7x too large → L_def ~2.1x → eke_len ~23% too long.
    #   "adiabatic": N² by adiabatic parcel displacement to the upper cell's
    #     pressure (Veros thermodynamics.py:99-103, eke.py:50-54), the true
    #     static stability used by the Veros EKE chain. Requires the caller to
    #     supply T, S, an EOS and the cell-centre pressure (or its ingredients)
    #     to displace parcels through the EOS; raises if any is missing.
    n2_mode: str = "insitu"
    # Veros dzw slot for the ADIABATIC N² divisor (mirrors
    # ``EKEConfig.n2_over_dzw``; only consulted with ``n2_mode="adiabatic"``):
    # divide the adiabatic density contrast by the actual centre spacing
    # ``dz_half_ref·J`` (Veros ``dzw``) instead of the midpoint
    # reconstruction. Default False ⇒ BIT-IDENTICAL legacy.
    n2_over_dzw: bool = False


class TreguierConfig(NamedTuple):
    """Treguier et al. (1997) / Held-Larichev (1996) adaptive GM coefficient —
    the NEMO ``nn_aei_ijk_t = 21`` scaling (``ldftra.F90::ldf_eiv``), used by
    BOTH the DINO and ORCA1 oracle configurations:

        κ(x, y) = min( min(1, |f/f₂₀|) · Ro² · T⁻¹ ,  aei0 )

    with the internal Rossby radius ``Ro = clip(0.4·∫N dz/|f|, 2 km, 40 km)``
    and the inverse baroclinic-instability timescale
    ``T⁻¹ = √(Σ N²(S_x²+S_y²)dz / (5 m + Σ dz))`` built from the isopycnal
    slopes.  The fixed factors (0.4, 2/40 km, 20°, +5 m) are hard-coded in the
    NEMO source (module constants in ``_gm_redi_common``); the ONE namelist
    tunable is the cap ``aei0 = rn_Ue·rn_Le`` (DINO: 0.03·100 km = 3000 m²/s;
    ORCA1: 0.018·100 km = 1800 m²/s).

    Mutually exclusive with ``VisbeckConfig.enabled`` (both are adaptive-κ
    diagnostics; the GM/Redi dispatch raises if both are on).
    """
    enabled: bool = False
    aei0: float = 3000.0     # κ cap [m²/s] = rn_Ue·rn_Le (DINO namelist value)
    # Optional FLOOR on the returned κ_GM [m²/s], applied to WET columns only
    # (dry columns stay exactly 0).  The tropical taper ``min(1, |f/f_20|)``
    # drives κ → 0 AT THE EQUATOR — measured on the eORCA1 tripole state, the
    # taper reaches 0.0000 and 2.17% of wet cells fall below 0.05 — and an
    # unfloored zero-GM equatorial band destabilised a 1° global run (non-finite
    # before day 5).  ``VisbeckConfig`` (the coefficient the OMIP tripole
    # otherwise uses) carries its own ``kappa_min`` (200 m²/s) for the same
    # reason.
    #
    # NOT NEMO.  NEMO's ldf_eiv is capped-only and genuinely yields κ → 0 at
    # f = 0; a NONZERO kappa_min is a DELIBERATE closure change that keeps a
    # finite eddy-induced velocity (and therefore a bolus transport) in the
    # equatorial band.  Any oracle/fidelity comparison must run kappa_min=0.0.
    # Default 0.0 = NO floor = byte-identical to the pre-existing behaviour and
    # to NEMO, so the DINO oracle card is unaffected.
    #
    # ORDERING: the floor is applied to the Treguier coefficient BEFORE the
    # optional Hallberg ``resolution_function`` scaling (which multiplies
    # whatever the closure produced — override / Treguier / Visbeck /
    # constant).  With ``resolution_function=True`` the EFFECTIVE κ can
    # therefore fall below ``kappa_min``; this matches how
    # ``VisbeckConfig.kappa_min`` already behaves.
    kappa_min: float = 0.0


class GMRediConfig(NamedTuple):
    """Gent-McWilliams / Redi isopycnal mixing (small-slope formulation).

    Implements the Griffies (1998) skew-flux form for GM and the full
    Redi isopycnal diffusion tensor with DM95 slope tapering.

    When kappa_GM == kappa_Redi (default), the horizontal off-diagonal
    terms cancel and the scheme reduces to horizontal diffusion plus
    an enhanced vertical mixing term proportional to S^2.

    When ``visbeck.enabled = True`` the scalar ``kappa_GM`` is replaced
    by a flow-dependent field computed from the local slope and
    stratification (Visbeck et al. 1997).  ``kappa_Redi`` still
    controls the isopycnal diffusivity.

    ``slope_scheme`` selects the discretisation used to build the
    isopycnal-tensor fluxes (lat-lon C-grid only — the cubed-sphere
    implementation always uses centered):

    - ``"triads"`` (default) — Griffies, Gnanadesikan, Pacanowski et
      al. (1998) triad decomposition.  Each flux is built from four
      quarter-cell triads that use the SAME three density / tracer
      values for both slope and gradient, guaranteeing that the Redi
      flux vanishes exactly for tracers constant along isopycnals
      (e.g. T with a linear EOS).  Required for century-scale climate
      runs and the recommended default for all production work.
    - ``"centered"`` — face-then-interface averaging of centered
      slopes (cheap, but the Redi tendency for ``q = f(ρ)`` retains a
      small residual that accumulates through dynamical feedback).
      Kept as a regression-coverage option and as a fallback for
      cheap short integrations.
    - ``"nemo_iso_lap"`` — NEMO 5.0.2's standard rotated-Laplacian
      iso-neutral operator (``traldf_iso``, ``#define iso_lap``),
      ported term-by-term and verified against NEMO's dumped
      ``ttrd_ldf`` (T corr 0.9997 with NEMO's own slopes; 0.96 with
      legoESM's centered slopes).  This is a **pure Redi** operator
      (NEMO ``traldf_iso`` has no GM bolus term): the dispatcher raises
      if ``kappa_GM != 0`` is requested with this scheme.  The explicit
      operator is the skew / off-diagonal iso-neutral part
      (``ln_traldf_msc=F`` ⇒ the K33 diagonal goes to the implicit
      vertical solve).  v1 flat-bottom / single-slope-field (mode-(b))
      approximation — see the operator docstring.
    """
    kappa_GM: float = 1e3       # GM bolus transport coefficient [m^2/s]
    kappa_Redi: float = 1e3     # Redi isopycnal diffusivity [m^2/s]
    S_max: float = 0.01         # Slope at which DM95 taper crosses 0.5.
                                # Equivalent to Veros's ``iso_slopec``.
    taper_width_frac: float = 0.1
    # ^ Tanh transition half-width as a fraction of ``S_max``. Default
    # 0.1 matches legoESM's pre-2026 hardcoded behavior. Veros's
    # ``iso_dslope`` parameter maps via
    # ``taper_width_frac = iso_dslope / iso_slopec``. For DINO's
    # ``iso_slopec=0.01, iso_dslope=0.005`` this is ``0.5``.
    visbeck: VisbeckConfig = VisbeckConfig()
    # Treguier-1997 adaptive κ (NEMO nn_aei_ijk_t=21, the oracle scaling) —
    # mutually exclusive with visbeck.enabled (dispatch raises on both).
    treguier: TreguierConfig = TreguierConfig()
    slope_scheme: str = "triads"     # "triads" (default), "centered", or "nemo_iso_lap"
    # GM eddy-induced (bolus) advection FORM for slope_scheme="nemo_iso_lap"
    # (NEMO ldf_eiv_trp): "centred" (default, BYTE-IDENTICAL) applies the bolus
    # as a 2nd-order CENTRED advective flux inside the iso operator — dispersive
    # at sharp fronts (over/undershoots), leans on the co-located Redi K to damp
    # 2Δx noise.  "through_fct" exports the bolus TRANSPORT (curl of ψ) to the
    # model step, which adds it to the advecting mass flux BEFORE the tracer
    # scheme, so the bolus flux passes through the monotone FCT/Zalesak limiter —
    # the faithful NEMO traadv form (the eiv velocity is added to the advecting
    # velocity). Tracer advection only (never momentum/continuity/eta). Only read
    # by slope_scheme="nemo_iso_lap"; the lat-lon C-grid model honors it.
    gm_bolus_advection: str = "centred"
    # NEMO ldf_eiv averages kappa onto EACH face before building the bolus
    # streamfunction (ldftra.F90:716-718): zaeiu = 0.5*(zaeiw(i)+zaeiw(i+1)).
    # False (default, bit-identical legacy) reuses the cell-centred kappa for
    # both faces -- exact only for a CONSTANT kappa; the Treguier kappa is
    # spatially 2-D, leaving a half-cell offset (#1226: eiv-transport rel err
    # median 3.4% -> 0.16% with the NEMO averaging).  Oracle cards set True.
    gm_bolus_kappa_face_average: bool = False
    slope_density: str = "in_situ"   # "in_situ" (default) or "neutral"
    # NEMO ln_traldf_msc (Method of Stabilizing Correction): when True the
    # nemo_iso_lap operator adds the akz-stabilized EXPLICIT K33 vertical
    # diagonal (traldf_iso_a33) that the ttrd_ldf dump contains for msc=T configs
    # (e.g. DINO). Default False ⇒ full K33 implicit (GYRE; bit-identical to the
    # prior operator). Only used by slope_scheme="nemo_iso_lap".
    msc_stabilize: bool = False
    # ^ Density gradient used to build the isoneutral SLOPES (NOT the tracer
    # gradients, which are always the raw T/S gradients).
    # - "in_situ" (default): slope = -∇_h ρ / ∂_z ρ from the IN-SITU density ρ.
    #   ∂_z ρ then carries the adiabatic compressibility term ∂ρ/∂p·∂p/∂z
    #   (≈ g·ρ₀/c_s² ≈ 4.5e-3 kg/m³/m), making |∂_z ρ| ~4× too steep, S ~4× too
    #   small, S² ~16×, and the vertical isoneutral diagonal K_33 ∝ S² 10–25×
    #   too small (≫ near the surface). BIT-IDENTICAL to the pre-2026 scheme.
    # - "neutral": build the slope-input density gradients from the LOCALLY-
    #   REFERENCED NEUTRAL form ∂ρ/∂T·∇T + ∂ρ/∂S·∇S with ∂ρ/∂T, ∂ρ/∂S the EOS
    #   partial derivatives at the LOCAL cell pressure (Veros get_drhodT /
    #   get_drhodS at abs(zt); veros/core/isoneutral/isoneutral.py:40-41). This
    #   removes the compressibility bias so the slope, S², and K_33 track Veros.
    #   The stable-strat floor min(0,∂_zρ)-eps is applied to the NEUTRAL ∂_zρ.
    #   ACC recipe opts in. Supported by both slope_scheme="triads" and
    #   "centered" on the lat-lon C-grid.
    #   FOLLOW-UP (documented, NOT built here): Veros sums BOTH kr triad levels
    #   for drodzb and carries the exact metric factors dxu/dxt/dyu/dyt/cost in
    #   the K_11/K_22/K_33 assembly; legoESM uses the upper-cell drdT for ∂_zρ
    #   and the uniform-metric 0.25·Σ. Inert on the uniform ACC channel; a true
    #   tripolar/variable-metric run would want the kr-sum + metric factors.
    surface_complement: bool = True  # Add horizontal diffusion (kappa_Redi)
                                      # in the surface layer where DM95 tapers
                                      # Redi to zero.  Ferrari et al. (2008).
                                      # Uses a fixed 100m depth proxy for the
                                      # mixed layer (should be replaced with
                                      # KPP boundary-layer depth when available).
                                      # Only active for slope_scheme="centered".
    surface_complement_depth: float = 100.0  # Depth [m] of the surface layer
    # Steep-slope limiting of the isoneutral operator (triads only):
    # "dm95_taper" (default, bit-identical) — Danabasoglu-McWilliams 95
    # tanh taper sends kappa·taper -> 0 for |S| >> S_max (Veros
    # convention; the flux DIES at steep slopes).  "nemo_cap" — NEMO
    # ldfslp convention: the SLOPE is capped at ±S_max and the taper is
    # 1, so the flux keeps diffusing ALONG the capped direction at
    # steep fronts (rn_slpmax).  Physically different at fronts: the
    # taper lets them steepen unchecked, the cap keeps flattening them.
    # REMAINING NEMO deviation (documented): the ldfslp mixed-layer
    # linear slope ramp toward the surface is not implemented yet.
    slope_limit: str = "dm95_taper"
    # Slope POSITIONS for slope_scheme="nemo_iso_lap": "mode_b" (default,
    # bit-identical v1) places the single interface slope field at all four
    # NEMO positions (corr 0.96, amplitude ~1.35); "nemo_native" computes the
    # ldfslp four-position slopes (uslp/vslp at tracer levels, wslpi/wslpj at
    # w-points; NEMO sign convention, caps + ML ramp + Shapiro built in —
    # certified corr +0.99, amplitude 1.00-1.03 vs the winter *_stg dump) and
    # feeds the exact traldf_iso stencil. With nemo_native the producer-side
    # slope_limit / nemo_mld_slope_ramp / nemo_slope_shapiro flags are
    # irrelevant to the iso operator (native has NEMO's own limiters), and
    # the dispatch applies NO sign negation (native is already NEMO-signed).
    slope_positions: str = "mode_b"
    # NEMO ldfslp mixed-layer slope ramp (default False = BYTE-IDENTICAL).
    # When True, isoneutral slopes are linearly ramped to 0 through the surface
    # mixed layer (ldfslp.F90:284-297 w-point branch: wslp(k) = gdepw(k)/max(hml,10)
    # * wslp_base, wslp_base = slope just below the ML base), matching NEMO's
    # ldfslp which flattens slopes in the ML where stratification -> 0 makes the
    # raw slope blow up.  MLD from the zdfmxl density criterion (below).  Applied
    # to the final tapered slopes; the interior / below-ML numerics (which already
    # match NEMO) are untouched.  Oracle-matching option; opt-in.
    nemo_mld_slope_ramp: bool = False
    # Density criterion [kg/m^3] for the ramp's mixed-layer depth (NEMO zdfmxl
    # rn_rho_c; potential-density difference from the ~10 m reference level).
    mld_rho_c: float = 0.01
    # Mixed-layer-depth criterion for the ldfslp slope ramp / native-slope
    # anchor.  "rho_c" (default, BYTE-IDENTICAL) = potential-density difference
    # of mld_rho_c from the ~10 m reference; "n2_integral" = NEMO's EXACT
    # zdfmxl.F90:91-105 criterion integral(MAX(N^2,0) dz) >= g*mld_rho_c/rho0
    # (in-situ adiabatic N^2 = rn2b, plus the MAX(N^2,0) clamp).  Set on the
    # nemo_dino_kamm card; all other recipes keep "rho_c".  Dispatch raises on
    # an unknown value (gm_redi_latlon_cgrid._nemo_mld).
    mld_criterion: str = "rho_c"
    # N^2 fed to the NEMO-native isopycnal slopes (ldf_slp).  NEMO's ldfslp
    # consumes ``rn2b`` -- the LINEARISED alpha/beta bn2 of eosbn2.F90 -- not a
    # parcel-displacement N^2.  The two diverge with pressure, so the adiabatic
    # form biases the slopes progressively at depth (#1226: on the DINO twin
    # |wslpi| runs 1.2% high in aggregate, essentially all of it below level 18,
    # with the bottom 8 levels carrying ~60% of the excess).  "nemo_bn2" is
    # S-EOS-specific; "adiabatic" (default) leaves every non-oracle recipe
    # bit-identical.  Dispatch raises on an unknown value
    # (gm_redi_latlon_cgrid._nemo_wpoint_e3w_wmask_n2).
    slope_n2: str = "adiabatic"
    # NEMO ldfslp horizontal (1-2-1)⊗(1-2-1)/16 Shapiro smoother on the final
    # interface slopes (ldfslp.F90:304-315).  legoESM omitted it, leaving the
    # interior slope amplitude ~1.27x too large; wet-renormalized so land drops
    # out.  Applied after the ML ramp (NEMO order).  Oracle-matching; opt-in.
    nemo_slope_shapiro: bool = False
    # NEMO nn_aht_ijk_t=20 grid-size scaling: the effective kappa_Redi is
    # cfg.kappa_Redi * cos(lat) per row (Mercator dx ∝ cos φ, so
    # aht(φ) = ½·U_d·Δx(φ) with cfg.kappa_Redi = the EQUATOR value
    # ½·U_d·R·dλ).  Applied by the lat-lon model as a per-column
    # kappa_redi_override; scalar-kappa paths (MPAS/cube) reject it.
    kappa_redi_lat_scaling: bool = False
    # --- Veros-faithful isoneutral options (oracle-matching; default off) ---
    implicit_K33: bool = False
    # ^ When True, the vertical isoneutral diagonal K_33 = kappa_Redi·S² (the
    # "enhanced vertical mixing ∝ S²" noted above) is REMOVED from the explicit
    # F_z and folded into the IMPLICIT vertical-diffusion tridiagonal solve
    # (backward-Euler), matching Veros (core/isoneutral/diffusion.py:
    # delta = dt/dzw·K_33). The explicit F_z then carries ONLY the off-diagonal
    # skew. Stiff-stable; required to reproduce Veros's dtemp_iso (which folds the
    # implicit K_33 increment into the diagnosed isoneutral tendency). Supported
    # only by slope_scheme="triads" on the lat-lon C-grid model.
    K_iso_steep: float = 0.0
    # ^ Steep-slope floor on the HORIZONTAL isoneutral diffusivity: the effective
    # along-isopycnal diffusivity becomes max(K_iso_steep, kappa·taper) (Veros
    # K_11/K_22, isoneutral.py:128/165; NOT applied to K_33). Default 0 = no floor.
    # NB legoESM clips the slope to S_max BEFORE the DM95 taper, so the taper
    # bottoms at 0.5 (its value at S_max) instead of →0; the floor therefore only
    # bites for kappa < 2·K_iso_steep. At Veros ACC (kappa≈1000, K_iso_steep=500)
    # the clipped-taper diagonal already equals K_iso_steep at steep slopes, so the
    # floor is correct but INERT for ACC — it matches Veros either way. (A fully
    # Veros-faithful steep-slope taper would need the UNCLIPPED slope; that is the
    # deeper slope-stencil difference, deferred.)
    veros_triad_weights: bool = False
    # ^ Veros-faithful u/v-face TRIAD WEIGHTS (default False = bit-identical
    # legacy). Veros weights each u/v-face triad by its vertical pair's W-cell
    # thickness, ``dzw(pair)/(4·dzt(level))`` (isoneutral.py:123-129 sumz,
    # diffusion.py:33-47 — the plain Δtr·dzw/(4·dzt) form), with NO
    # renormalization where triads are missing: at the surface the two
    # "above" triads use the half-cell ``dzw_sfc = dzt[0]/2`` and are DEAD
    # for the off-diagonal (taper→0 via the zeroed dTdz) while their
    # K_iso_steep diagonal floor survives; at the bottom the two "below"
    # triads vanish entirely. legoESM's legacy convention instead
    # renormalizes by the number of valid triads (1/N_valid, equal weights),
    # keeping the boundary diagonal at full strength — a defensible
    # discretization → option, not canonical. On a stretched vertical grid
    # the two weightings differ at EVERY level (global_4deg z1: lego/Veros
    # skew F_x = 1.29 with 1/N, ≈1.0 with dzw weights) and by ~2× in the
    # off-diagonal at the surface level (the tier-2 z0 = 3.2× signature).
    # Only meaningful with slope_density="neutral" (the in_situ slope-clip
    # pins dead-triad tapers at 0.5 instead of 0); the faithful recipes are
    # all neutral.
    double_redi_diagonal: bool = False
    # ^ Veros-faithful DOUBLE-COUNTED Redi horizontal diagonal (oracle quirk;
    # default False = bit-identical single diagonal). Veros adds the
    # PRE-computed diagonal flux ``K_11·∂T/∂x`` / ``K_22·∂T/∂y`` inside
    # ``_calc_tracer_fluxes`` UNCONDITIONALLY (core/isoneutral/diffusion.py:
    # 40-47, 67-77), and with ``enable_neutral_diffusion`` +
    # ``enable_skew_diffusion`` both on (the ACC and global_4deg setups) that
    # kernel runs TWICE per tracer per step (thermodynamics.py:430-437) — the
    # iso pass AND the skew pass each add the full K_11/K_22 diagonal, so
    # Veros's net horizontal isoneutral diffusion carries 2× the diagonal
    # (≈ 2·K_iso_0 ≈ 2000 m²/s of along-isopycnal smoothing). legoESM's
    # single diagonal is the textbook Redi tensor; matching the oracle
    # requires reproducing the double-add, so this is a config-selectable
    # OPTION (judgment: Veros implementation quirk → option, not canonical).
    # When True, the triad assembly adds the diagonal (incl. its K_iso_steep
    # floor) ONE extra time to F_x/F_y, and the realized SIGNED skew
    # conversion carries the same extra diagonal in its skew fluxes (Veros's
    # P_diss_skew includes its skew-pass K_11·∂T/∂x flux). Measured
    # (global_4deg bridged yr-1 state, tier-2 component isolation): without
    # it, lego/Veros TOTAL horizontal flux rms = 0.84 with 0.5-0.6 below
    # 800 m; per-pass components match at 1.0. See
    # .physics-validator/eke_global_runaway/RESULTS.md.
    # Prognostic EKE (Eden-Greatbatch 2008): when not None, kappa_GM becomes
    # prognostic (c_k·L·√E) from the evolving eddy-energy field E, instead of the
    # constant ``kappa_GM`` / Visbeck diagnostic. Selection is presence-based
    # (None = off). The Rossby-radius length uses the ``visbeck`` length params.
    # Veros ACC runs with EKE on (enable_eke=True).
    eke: EKEConfig | None = None
    adjoint_stabilization: str = "none"
    # ^ Long-horizon REVERSE-MODE gradient stabilization for the isoneutral
    # operator (default "none" = exact AD, bit-identical legacy). MECHANISM
    # (probe-verified, .physics-validator/gm_adjoint_stab/RESULTS.md): the
    # slope saturation (DM95 taper; and the ±S_max clip on the in-situ path)
    # bounds the PRIMAL fluxes but the taper does NOT bound the LINEARIZED
    # operator — d(taper·S)/d(state) exceeds the primal coefficient bound via
    # (a) the taper-derivative term S·taper' in the transition band
    # (~1/(2·taper_width_frac) excess; dense-Jacobian rho 1.0→2.4 in band at
    # the ACC kappa, scaling with kappa·dt/dz²) and (b) the UNCLIPPED neutral
    # slope tangent ∂S/∂(∇ρ) ∝ 1/∂_zρ in weakly-stratified cells (the
    # dominant path on the real ACC state; EOS-independent). The tangent/
    # adjoint propagator then has per-step amplification |G| ≫ 1 where the
    # primal is stable (full ACC step: |G| ≈ 78 at constant kappa=1000, vs
    # 1.02 with GM/Redi removed) — parameter adjoints grow ~×2-5/step beyond
    # ~1 model day. NB the legacy slope_density="in_situ" CLIP saturates the
    # tangent as well (clip gradient = 0 outside ±S_max): the in-situ path
    # measures |G| ≈ 1.02 with NO stabilization — the instability is specific
    # to the Veros-faithful UNCLIPPED "neutral" slope path.
    # Options (both PRIMAL-INVISIBLE by construction — stop_gradient only):
    # - "stop_gradient_slopes" (RECOMMENDED, probe-validated): stop_gradient
    #   on the slopes themselves (and hence the tapers computed from them) —
    #   the frozen-coefficient (Picard) linearization of the isoneutral
    #   tensor. Gradients keep the full tracer-flux linearization and the
    #   kappa sensitivity, dropping only the density→tensor feedback. Full
    #   ACC step |G|: 78 → 1.020 (constant kappa), full recipe → 1.002.
    # - "stop_gradient_taper": stop_gradient on the DM95 taper FACTORS only
    #   (the textbook differentiable-solver flux-limiter trick). Kills
    #   mechanism (a) — sufficient on healthily-stratified configs — but NOT
    #   mechanism (b): on the faithful ACC stack |G| stays ≈ 75. Kept as the
    #   finer-grained option; prefer "stop_gradient_slopes".
    # Applies to the tracer-tendency triads (u/v/w), the centered scheme, the
    # implicit-K33 coefficient, and the slope chain feeding Visbeck/EKE.
    # NOT for forward-only runs (no effect); select it for long-horizon
    # gradient-based calibration/DA through GM/Redi. Validated fail-fast by
    # ``validate_adjoint_stabilization`` at every GM/Redi entry point.
    # --- Hallberg (2013) resolution function for kappa_GM (default off) ---
    resolution_function: bool = False
    # ^ When True, multiply the EFFECTIVE GM coefficient ``kappa_GM`` (whatever
    # the active closure produced -- constant / Visbeck / Treguier / prognostic
    # EKE / GEOMETRIC) by the Hallberg (2013) resolution function
    #     f_res = 1 / (1 + (L_d / (resfn_gamma * Delta))**2),
    # with ``Delta = sqrt(cell area)`` the local grid spacing and the
    # first-baroclinic deformation radius ``L_d = resfn_cbcl_ms / |f|`` (|f|
    # floored near the equator).  ``f_res -> 1`` where ``Delta >> L_d`` (coarse,
    # eddies unresolved: full GM) and ``-> 0`` where ``Delta << L_d`` (eddy-
    # resolving: GM off, let the resolved eddies act), matching NEMO5 ldf_eiv /
    # MOM6 resolution-scaled KhTh.  Applied to GM ONLY -- the Redi isopycnal
    # diffusivity ``kappa_Redi`` is NOT scaled (NEMO/MOM6 scale the eddy-
    # transport bolus coefficient, not the along-isopycnal tracer diffusion), so
    # as ``f_res -> 0`` the scheme reduces to pure Redi.  Shared by the lat-lon
    # C-grid, MPAS and cubed-sphere GM/Redi paths.  Default False => the
    # kappa_GM object is returned untouched => BYTE-IDENTICAL.
    #
    # EKE-BUDGET COUPLING (lat-lon prognostic closures; codex MED-3 r2): when
    # an EKE / GEOMETRIC closure is active, the model step scales the
    # GM-DERIVED eddy-energy production by the SAME f_res — parameterized
    # ``kappa*sigma^2`` via ``eke_apply_local_source(production_scale=...)``,
    # the GEOMETRIC baroclinic conversion B_C, and the realized skew
    # conversions (via the scaled kappa handed to the conversion builders) —
    # so the E budget receives exactly the APE->EKE conversion the APPLIED
    # (tapered) coefficient performs; an unscaled production would
    # over-energise E (and hence kappa = c_k*L*sqrt(E)) relative to the
    # realized GM work.  Redi-side terms (kappa_redi_override, -P_diss_iso,
    # GEOMETRIC kappa_n) and the barotropic B_T (kappa_u) stay UNSCALED.
    # See ``gm_resolution_factor`` (the single f_res definition) and
    # ``ocean_model_latlon_cgrid`` step / ``_eke_3d_step``.
    resfn_gamma: float = 2.0
    # ^ Resolution-function width gamma (Hallberg 2013): grid points per
    # deformation radius at which GM is half-suppressed (~1-2 typical).
    resfn_cbcl_ms: float = 2.0
    # ^ Fixed first-baroclinic gravity-wave speed c [m/s] for ``L_d = c/|f|``
    # (Chelton et al. 1998: c1 ~ 2 m/s open-ocean).  A FIXED c bound is used
    # (rather than the flow-dependent ``int(N dz)/pi``) so the resolution
    # function applies uniformly to ALL closures incl. constant-kappa, which
    # never computes ``int(N dz)``.


class LateralMixingConfig(NamedTuple):
    """Top-level lateral mixing configuration."""
    scheme: str = "harmonic"  # "harmonic", "biharmonic", "gm_redi", "none"
    harmonic: HarmonicConfig = HarmonicConfig()
    biharmonic: BiharmonicConfig = BiharmonicConfig()
    gm_redi: GMRediConfig = GMRediConfig()
