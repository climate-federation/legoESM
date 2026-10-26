"""Silvestri et al. 2024 momentum-advection scheme presets (the comparison matrix).

Maps each paper scheme name to the LatLonCGridOceanConfig overrides that select
it, using the canonical building blocks added in Phase 1 (W9V/W9D smoothness,
UP3 flux-form, OM4p25 / QG-Leith closures). A recipe or driver applies a preset
with ``apply_silvestri_scheme(config, name)``.

Two scheme families:
  * BAROCLINIC-JET (paper §5, Tables 2-3): UP3, W9V, W9D, SM2, QG2 — the 5 main
    cases (+ the repo extras bileith/ebs).
  * 2D-TURBULENCE (paper §4, Table 1): DNS, Leith1, Leith2, W5D, W9D, W5V, W9V —
    vorticity-flux variants on a single layer (added with the 2D experiment).

Faithfulness note: each diffusive scheme (UP3/W9*) carries its OWN implicit
dissipation and runs with NO explicit closure (A_h=B_h=C_smag=C_leith=0); the
dispersive schemes (SM2/QG2) use a 2nd-order energy-conserving vorticity flux
(``vector_invariant``) STABILISED by the explicit OM4p25 / QG-Leith closure.
QG2 uses the FULL QG-Leith with the baroclinic stretching term (B5b) — faithful.
"""

from __future__ import annotations

# Zero ALL the explicit lateral-viscosity knobs — the WENO/UP3 schemes are
# self-dissipating and the explicit closures (SM2/QG2) supply their own.
_NO_EXPLICIT_VISC = dict(
    A_h=0.0, B_h=0.0, C_smag=0.0, C_smag_lap=0.0, C_leith=0.0,
)

# --- Baroclinic-jet (§5) scheme matrix: name -> config overrides ------------
SILVESTRI_JET_SCHEMES: dict[str, dict] = {
    # W9V: WENO9 vector-invariant, velocity/full-divergence smoothness ({ζ;u}+{δU;D}).
    "W9V": dict(momentum_advection="weno9", weno_smoothness="split",
                weno_d_term=True, lateral_friction_scheme="none",
                **_NO_EXPLICIT_VISC),
    # W9D: WENO9 vector-invariant, self-smoothness ({ζ;ζ}+{δU;δU}).
    "W9D": dict(momentum_advection="weno9", weno_smoothness="standard",
                weno_d_term=True, lateral_friction_scheme="none",
                **_NO_EXPLICIT_VISC),
    # UP3: 3rd-order upwind-biased flux-form (implicit dissipation, no closure).
    # The paper's UP3 is Oceananigans UpwindBiased(order=3), which selects the
    # upwind branch by the sign of the TRANSPORT for every flux family
    # (upwind_biased_advective_fluxes.jl:18-24) -- NOT NEMO's advected-velocity
    # pair.  See UP3_REFERENCE_SELECTOR in ocean_pe_latlon_cgrid.
    "UP3": dict(momentum_advection="flux_form",
                momentum_flux_scheme="oceananigans_up3",
                lateral_friction_scheme="none", **_NO_EXPLICIT_VISC),
    # SM2: energy-conserving vorticity flux + OM4p25 Smagorinsky lateral friction.
    "SM2": dict(momentum_advection="vector_invariant",
                lateral_friction_scheme="om4p25", **_NO_EXPLICIT_VISC),
    # QG2: energy-conserving vorticity flux + FULL QG-Leith (C=2) with the
    # baroclinic stretching term (B5b) — the faithful paper QG2.
    "QG2": dict(momentum_advection="vector_invariant",
                lateral_friction_scheme="qg_leith", qg_leith_coeff=2.0,
                qg_leith_stretching=True, **_NO_EXPLICIT_VISC),
}

# The paper's 5 main baroclinic-jet comparison schemes (Fig 7-10).
SILVESTRI_JET_MAIN = ("UP3", "W9V", "W9D", "SM2", "QG2")


def apply_silvestri_scheme(config, scheme: str):
    """Return ``config`` with the named Silvestri jet scheme's overrides applied.

    Raises ValueError on an unknown scheme (dispatch discipline — never silently
    fall through to a default).
    """
    if scheme not in SILVESTRI_JET_SCHEMES:
        raise ValueError(
            f"unknown Silvestri jet scheme {scheme!r}; "
            f"choose from {sorted(SILVESTRI_JET_SCHEMES)}")
    return config.replace_flat(**SILVESTRI_JET_SCHEMES[scheme])  # #501: distributes grouped keys


def scheme_label(scheme: str) -> str:
    """Human/plot label for a scheme."""
    if scheme not in SILVESTRI_JET_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    return scheme
