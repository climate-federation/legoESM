"""Configuration for ocean lateral mixing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig


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
    safety factor).  See ``HarmonicConfig`` for the analogous CFL knobs.
    """
    B_h_momentum: float = 0.0   # Biharmonic viscosity [m^4/s]
    B_h_tracer: float = 0.0     # Biharmonic tracer diffusivity [m^4/s]
    enforce_cfl: bool = False
    cfl_dt_estimate: float = 3600.0
    cfl_safety: float = 0.05    # Margin below 1/16 stability bound


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
    slope_scheme: str = "triads"     # "triads" (default) or "centered"
    slope_density: str = "in_situ"   # "in_situ" (default) or "neutral"
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
    # Prognostic EKE (Eden-Greatbatch 2008): when not None, kappa_GM becomes
    # prognostic (c_k·L·√E) from the evolving eddy-energy field E, instead of the
    # constant ``kappa_GM`` / Visbeck diagnostic. Selection is presence-based
    # (None = off). The Rossby-radius length uses the ``visbeck`` length params.
    # Veros ACC runs with EKE on (enable_eke=True).
    eke: EKEConfig | None = None


class LateralMixingConfig(NamedTuple):
    """Top-level lateral mixing configuration."""
    scheme: str = "harmonic"  # "harmonic", "biharmonic", "gm_redi", "none"
    harmonic: HarmonicConfig = HarmonicConfig()
    biharmonic: BiharmonicConfig = BiharmonicConfig()
    gm_redi: GMRediConfig = GMRediConfig()
