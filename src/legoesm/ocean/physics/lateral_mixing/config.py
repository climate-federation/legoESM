"""Configuration for ocean lateral mixing schemes."""

from __future__ import annotations

from typing import NamedTuple


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
    S_max: float = 0.01         # Maximum isopycnal slope for tapering
    visbeck: VisbeckConfig = VisbeckConfig()
    slope_scheme: str = "triads"     # "triads" (default) or "centered"


class LateralMixingConfig(NamedTuple):
    """Top-level lateral mixing configuration."""
    scheme: str = "harmonic"  # "harmonic", "biharmonic", "gm_redi", "none"
    harmonic: HarmonicConfig = HarmonicConfig()
    biharmonic: BiharmonicConfig = BiharmonicConfig()
    gm_redi: GMRediConfig = GMRediConfig()
