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
    # Dycore scheme choices default to the `oceananigans_v1` catalog recipe (the
    # SINGLE SOURCE — see legoesm.ocean.recipes); pass a value to override per deck.
    momentum_advection: str | None = None,         # default vector_invariant [APPROX]
    tracer_advection: str | None = None,           # default WENO(order=7)
    barotropic_solver: str | None = None,          # default ImplicitFreeSurface (implicit_cn)
    # Barotropic time filter — consumed ONLY by explicit_substep (implicit_cn has no
    # substep; _validate_config warns if set non-default there). None ⇒ PER SOLVER:
    # power_law (SM2005) for explicit_substep (avoids the cosine bell's eddy
    # over-dissipation, see oceananigans_recipe_wiring_plan.md §8), else cosine.
    barotropic_time_filter: str | None = None,
    coriolis_scheme: str | None = None,            # default explicit_ab2 [APPROX]
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
    from legoesm.ocean.recipes import get_recipe
    vmix: dict[str, float] = {}
    if A_v is not None:
        vmix["A_v"] = A_v
    if K_v is not None:
        vmix["K_v"] = K_v
    # SINGLE SOURCE: the canonical dycore identity is the `oceananigans_v1` catalog
    # recipe (schemes + pinned runtime flags incl. weno_vertadv_full_velocity, the
    # §5/CASE-3 interior-eddy closure). Caller scheme args override per deck.
    bundle = get_recipe("oceananigans_v1")
    for key, val in (("momentum_advection", momentum_advection),
                     ("tracer_advection", tracer_advection),
                     ("barotropic_solver", barotropic_solver),
                     ("coriolis_scheme", coriolis_scheme)):
        if val is not None:
            bundle[key] = val
    # Per-solver default: power_law where it is consumed (explicit_substep), else the
    # inert cosine default (so the implicit_cn default does not trip the no-op warning).
    if barotropic_time_filter is None:
        barotropic_time_filter = (
            "power_law" if bundle["barotropic_solver"] == "explicit_substep" else "cosine")
    # FE-Coriolis hazard (diagnosed via the DINO 'oceananigans'-card barotropic
    # blowup): the EFFECTIVE explicit_ab2 x implicit_cn x ab2-outer combo
    # integrates the barotropic-mode Coriolis forward-Euler (the CN predictor
    # gates its FB Coriolis off; the outer AB2 excludes the barotropic
    # increment) -- unconditionally unstable, |G| = sqrt(1+(f*dt)^2) per step.
    # Default the Oceananigans G^U AB2-chi centering ON for that combo (what
    # Oceananigans itself does); matsuno/other per-deck Coriolis overrides never
    # hit the condition.  A caller override always wins (popped, so **bundle +
    # **overrides cannot collide on a duplicate kwarg).  Callers stepping with
    # the flag on must seed F_slow_{u,v}_prev -- model.seed_scan_carry does it.
    # Merge **overrides keys that also live in the bundle INTO the bundle first
    # (pop, so from_flat never sees a duplicate kwarg), making the condition
    # below read the EFFECTIVE combo -- e.g. a caller overriding
    # outer_integrator="forward_euler" via **overrides must not trip the AB2
    # default decision (codex).
    for _k in [k for k in list(overrides) if k in bundle]:
        bundle[_k] = overrides.pop(_k)
    if "barotropic_slow_forcing_ab2" in overrides:
        bundle["barotropic_slow_forcing_ab2"] = overrides.pop(
            "barotropic_slow_forcing_ab2")
    elif (bundle["coriolis_scheme"] == "explicit_ab2"
          and bundle["barotropic_solver"] == "implicit_cn"
          and bundle.get("outer_integrator") == "ab2"):
        bundle["barotropic_slow_forcing_ab2"] = True
    return LatLonCGridOceanConfig.from_flat(
        **bundle,
        barotropic_time_filter=barotropic_time_filter,
        eos_linear=eos_linear,
        g=g,
        rho_0=rho_0,
        A_h=A_h,
        K_h=K_h,
        physics=physics,
        ab2_epsilon=ab2_epsilon,
        lateral_side_bc=lateral_side_bc,
        bottom_drag_r=bottom_drag_r,
        **vmix,
        **overrides,
    )
