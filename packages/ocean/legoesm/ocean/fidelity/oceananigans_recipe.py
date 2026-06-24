"""Reusable Oceananigans ocean-numerics recipe card + the Oceananigans<->legoESM
wiring diagram.

Like ``mitgcm_recipe.py`` / ``nemo_recipe.py``, this is a PURE CONFIG layer: it
selects the canonical legoESM blocks that reproduce Oceananigans'
``HydrostaticFreeSurfaceModel`` numerics. It contains NO Oceananigans-specific
solver / grid / bridge / reader — those mimicry-only pieces live in the harness
(``oceananigans_io`` / ``oceananigans_runner`` / ``oceananigans_state_bridge``),
never in the model (oracle-recipe doctrine).

UNVALIDATED-NUMERICS CAVEAT (2026-06-21): unlike the MITgcm card (whose every row
was confirmed bit-faithful against MITgcm's own momentum diagnostics), the rows
below are mapped by READING the Oceananigans source, NOT yet by a term-by-term
match against an imported Oceananigans state — the oracle reference-generation
env is being re-provisioned. Rows flagged ``[APPROX]`` are KNOWN to differ from
the oracle at the discrete level and MUST be confirmed (or refined) by the
fidelity scorecard once a reference exists:

  * ``momentum_advection="vector_invariant"`` is legoESM's Arakawa-Lamb (1981)
    12-point triad PV flux; Oceananigans ``VectorInvariant()`` defaults to an
    EnstrophyConserving vorticity flux + EnergyConserving KE gradient. Same
    family (vector-invariant, enstrophy-conserving spirit), NOT the same stencil.
    (For the WENO cases use ``weno5/7/9`` — those DO match Oceananigans'
    ``WENOVectorInvariant`` per PR #559's FV-WENO fix.)
  * ``coriolis_scheme="explicit_ab2"`` is a 4-point FACE-f average; Oceananigans
    ``HydrostaticSphericalCoriolis(EnstrophyConserving())`` uses VERTEX f with a
    metric-weighted averaging. Tested on §5: the legoESM vertex-f energy-
    conserving variant did NOT help and is not the same as the oracle's
    enstrophy-conserving form (see project memory). Treat as [APPROX].
  * ``barotropic_solver``: Oceananigans ``ImplicitFreeSurface`` is a single
    elliptic eta solve; the closest legoESM block is ``implicit_cn`` (steady
    cases) / ``explicit_substep`` (the eddy §5 split-explicit faithful path).
    Per-deck.

This card HARD-PINS what is common + faithful (linear EOS, Boussinesq reference,
differentiable / no-conservation-fixer runtime flags) and exposes the per-deck
axes (the dimensional knobs, the momentum scheme, the barotropic solver, lateral
viscosity, bottom drag, side BC) as arguments.
"""

from __future__ import annotations

from legoesm.ocean.state import LatLonCGridOceanConfig

# Oceananigans defaults for the validation gyre / jet cases (Boussinesq).
OCEANANIGANS_DEFAULT_G = 9.81
OCEANANIGANS_DEFAULT_RHO_0 = 1000.0


# --- The Oceananigans <-> legoESM WIRING DIAGRAM ---------------------------------
# (Oceananigans construct, legoESM config field, chosen block + fidelity note).
# "[pinned]" = common to every Oceananigans hydrostatic case; "[per-deck]" = a card
# argument; "[APPROX]" = KNOWN discrete-level difference, pending scorecard refinement.
OCEANANIGANS_BLOCK_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("gravitational_acceleration, reference density (Boussinesq)", "g + rho_0",
     "[per-deck] g from free_surface.gravitational_acceleration; rho_0 reference"),
    ("BuoyancyTracer / no tracer (homogeneous gyre)", "eos + eos_linear",
     "[per-deck] linear EOS; b<->T via linear EOS handled in the setup, not core"),
    ("momentum_advection = VectorInvariant()", "momentum_advection",
     "[per-deck; APPROX] vector_invariant (AL81 triad) ~ Oceananigans "
     "EnstrophyConserving vorticity + EnergyConserving KE; weno5/7/9 for "
     "WENOVectorInvariant (matched by PR #559)"),
    ("coriolis = HydrostaticSphericalCoriolis(EnstrophyConserving())",
     "coriolis_scheme",
     "[APPROX] explicit_ab2 face-f 4-point avg ~ Oceananigans VERTEX-f "
     "enstrophy-conserving; refuted as the §5 residual but a discrete difference"),
    ("free_surface = ImplicitFreeSurface(g)", "barotropic_solver",
     "[per-deck] implicit_cn (steady) / explicit_substep (split-explicit, the §5 "
     "eddy faithful path); Oceananigans ImplicitFreeSurface = single elliptic solve"),
    ("closure = HorizontalScalarDiffusivity(nu)", "A_h + lateral_viscosity_operator",
     "[per-deck] A_h = nu; flux_divergence operator (component harmonic friction)"),
    ("u/v bottom drag (FluxBoundaryCondition, -mu*u)", "bottom_drag_r",
     "[per-deck] linear bottom drag r = mu [1/s]"),
    ("surface wind stress (FluxBoundaryCondition)",
     "surface_forcing (OceanSurfaceForcing.tau_x)",
     "[per-deck] external wind stress (sign per legoESM atmospheric-stress "
     "convention; verified by an equivariance test in the setup card)"),
    ("tracer_advection = WENO(order=7) (stratified cases)", "tracer_advection",
     "[per-deck] weno7 (matches Oceananigans WENO tracer; FV-WENO via PR #559)"),
    ("time integration (RK3 / split-explicit AB2 barotropic)",
     "outer_integrator + ab2_scope",
     "[per-deck] ab2 outer; barotropic_slow_forcing_ab2 for the faithful §5 split"),
)


def oceananigans_canonical_ocean_config(
    *,
    eos_linear,
    g: float = OCEANANIGANS_DEFAULT_G,
    rho_0: float = OCEANANIGANS_DEFAULT_RHO_0,
    A_h: float = 0.0,                              # noqa: N803
    K_h: float = 0.0,                              # noqa: N803
    A_v: float | None = None,                      # noqa: N803
    K_v: float | None = None,                      # noqa: N803
    physics=None,
    momentum_advection: str = "vector_invariant",  # VectorInvariant() [APPROX]
    tracer_advection: str = "weno7",               # WENO(order=7)
    barotropic_solver: str = "implicit_cn",        # ImplicitFreeSurface
    coriolis_scheme: str = "explicit_ab2",         # spherical Coriolis [APPROX]
    bottom_drag_r: float = 0.0,                    # linear bottom drag mu
    lateral_side_bc: str = "free_slip",            # Oceananigans default: free-slip
    ab2_epsilon: float = 0.1,
    **overrides,
) -> LatLonCGridOceanConfig:
    """Return the SHARED Oceananigans-faithful ``LatLonCGridOceanConfig`` choices.

    HARD-PINS the Boussinesq linear-EOS + differentiable / no-conservation-fixer
    runtime flags common to every Oceananigans hydrostatic case, and exposes the
    per-deck axes (dimensional knobs, the momentum / tracer / barotropic /
    Coriolis schemes, lateral viscosity, bottom drag, side BC). See
    :data:`OCEANANIGANS_BLOCK_MAPPING` for the [APPROX] rows whose discrete
    fidelity is pending the scorecard.

    ``overrides`` patches any remaining field (e.g. the §5 faithful split stack:
    ``barotropic_slow_forcing_ab2=True``, ``weno_smoothness``, ``A_v``/``K_v``).
    """
    vmix: dict[str, float] = {}
    if A_v is not None:
        vmix["A_v"] = A_v
    if K_v is not None:
        vmix["K_v"] = K_v
    return LatLonCGridOceanConfig(
        g=g,
        rho_0=rho_0,
        eos="linear",
        eos_linear=eos_linear,
        A_h=A_h,
        K_h=K_h,
        physics=physics,
        momentum_advection=momentum_advection,
        tracer_advection=tracer_advection,
        coriolis_scheme=coriolis_scheme,
        outer_integrator="ab2",
        ab2_epsilon=ab2_epsilon,
        barotropic_solver=barotropic_solver,
        lateral_viscosity_operator="flux_divergence",
        A_h_lat_scaling=False,
        C_smag=0.0,
        lateral_side_bc=lateral_side_bc,
        bottom_drag_r=bottom_drag_r,
        differentiable_barotropic=True,
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        # Oceananigans advects the FULL horizontal momentum vertically (w·∂u/∂z over the
        # full u); legoESM's production default advects only the baroclinic perturbation
        # u' (omitting −∂(w·U_bar)/∂z). The perturbation form makes a 3D baroclinic-eddy
        # front go UNSTABLE at the interior (the §5/baroclinic_adjustment runaway: NaN
        # day 18) while the full-velocity form saturates like the oracle (within 2× to
        # 30 d). No effect on single-layer cases (gyre/bickley). This is the faithful
        # Oceananigans choice — the §5/CASE-3 interior-eddy closure.
        weno_vertadv_full_velocity=True,
        **vmix,
        **overrides,
    )
