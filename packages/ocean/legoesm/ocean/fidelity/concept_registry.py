"""Cross-oracle concept registry — the ocean naming "Rosetta stone".

A single machine-readable map from each physical concept the ocean code uses to:
  - its **canonical legoESM identifier** (the one name to use),
  - the **aliases** that legoESM has historically used for it (naming debt),
  - the **oracle aliases** (what Veros / MOM6 / MITgcm call the same thing), and
  - units + status.

Why this exists (per the gap->block / dedup doctrine in
``docs/ocean_fidelity/oracle_recipe_strategy.md`` §3 rule I + §9):

1. **Recognition across oracles.** Different production models name the same
   quantity differently (Veros ``kappaM`` = legoESM ``A_v``; Veros ``r_bot`` =
   legoESM ``bottom_drag_r``). A recipe/bridge that maps an oracle to legoESM
   needs this map; without it you silently grow parallel vocabularies.
2. **Stop synonym/duplication creep.** This registry is the source the
   deterministic naming guard (``tests/ocean/unit/test_concept_registry.py``)
   and the agent-based semantic auditor (lego-modularity-tester dimension 10)
   read, so "the same thing under a new name" gets caught instead of
   accumulating.

This is a DOCUMENTATION/REFERENCE artifact (like ``fidelity/references.py``),
not a runtime physics path. It must stay in sync with reality: the guard test
ratchets the `aliases` of `ratchet=True` concepts (debt may only shrink).

Scope: ocean. Atmosphere/land have their own vocabularies; only cross-ocean and
ocean<->oracle concepts live here.
"""

from __future__ import annotations

from typing import NamedTuple

# Oracle keys used in ``oracle_aliases``. Keep this closed so the consistency
# test can validate every mapping references a known oracle.
KNOWN_ORACLES = frozenset({"veros", "mom6", "mitgcm", "nemo"})

# Status vocabulary:
#   "canonical"        — settled canonical name; aliases are clear debt.
#   "unit-hazard"      — the SAME base name means different UNITS in different
#                        scopes (C_water vs c_water, tau_relax days vs s); new
#                        code MUST carry a unit suffix. Human-review, not auto-
#                        ratcheted (an identifier scan can't tell the units).
#   "entrenched"       — a non-canonical alias dominates by usage (e.g. nlev);
#                        documented + tracked, NOT ratcheted (cleanup is a
#                        dedicated PR, not a per-PR gate).
#   "fidelity-scoped"  — the alias lives only on the oracle side / in the
#                        fidelity harness; no legoESM-core collision.
_STATUSES = frozenset(
    {"canonical", "unit-hazard", "entrenched", "fidelity-scoped"}
)


class ConceptDef(NamedTuple):
    """One physical concept and all the names it travels under."""

    concept: str                       # human description
    canonical: str                     # the legoESM identifier to use everywhere
    units: str                         # SI-ish units, or "count"/"dimensionless"
    status: str                        # one of _STATUSES
    aliases: tuple[str, ...] = ()       # other legoESM identifiers seen (debt)
    oracle_aliases: tuple[tuple[str, str], ...] = ()  # ((oracle, name), ...)
    ratchet: bool = False              # if True, the guard forbids `aliases`
                                       # appearing in files outside the baseline
    note: str = ""


# ---------------------------------------------------------------------------
# The registry. Seeded from the 2026-05-29 cross-oracle survey + CLAUDE.md's
# "Open naming debt" section. Extend (with review) as the dimension-10 auditor
# finds more.
# ---------------------------------------------------------------------------

CONCEPTS: tuple[ConceptDef, ...] = (
    ConceptDef(
        concept="Vertical level count",
        canonical="n_levels",
        units="count",
        status="entrenched",
        aliases=("nlev", "nz"),
        oracle_aliases=(("veros", "nz"),),
        note="nlev dominates (~3566 uses) and is the de-facto local-variable "
             "name; n_levels is the OceanZStarCoordinate field. Cleanup is a "
             "dedicated PR — NOT ratcheted (a per-PR gate over 3566 sites is "
             "impractical).",
    ),
    ConceptDef(
        concept="Surface (skin) temperature",
        canonical="T_sfc",
        units="K",
        status="entrenched",
        aliases=("T_surface", "Ts"),
        note="T_sfc is now the only surface-temperature identifier in source: "
             "T_surface has been fully migrated out (0 identifier sites; one "
             "stray comment remains), and `Ts` persists only as SCM-accumulator "
             "locals. Aliases retained for provenance; not ratcheted here.",
    ),
    ConceptDef(
        concept="Seawater specific heat capacity",
        canonical="c_sw",
        units="J/(kg K)",
        status="canonical",
        aliases=("c_ocean", "c_p"),
        oracle_aliases=(("veros", "cp_0"), ("mitgcm", "HeatCapacity_Cp")),
        ratchet=False,  # c_p/c_ocean are context-dependent (budgets.py uses c_p
                        # as a param correctly defaulting to c_sw; slab-ocean
                        # c_ocean is a domain field) -> human/agent review, not
                        # an identifier scan that would false-positive.
        note="constants.c_sw = 3994.0 (Gill 1982). Slab-ocean configs use "
             "c_ocean; budgets.py takes a c_p arg defaulting to c_sw (OK). DINO "
             "/ NeverWorld2-lite carry c_p=3991.86 as a config-field default — "
             "this is the INTENTIONAL Kamm et al. (2025) paper value (an oracle "
             "value pinned via config, per rule H), NOT debt to 'fix' toward "
             "c_sw. The alias is informational; not auto-ratcheted (c_p/c_ocean "
             "are context-dependent — see dimension-10 for semantic review).",
    ),
    ConceptDef(
        concept="Boussinesq reference density",
        canonical="rho_0",
        units="kg/m^3",
        status="canonical",
        aliases=("rho_ocean", "rho_ref"),
        oracle_aliases=(("veros", "rho_0"), ("mitgcm", "rhoConst")),
        note="constants.rho_ocean is the module constant; config field is "
             "rho_0; LinearEOSConfig.rho_ref is the EOS reference. All ~1024-"
             "1025. Distinct roles -> not ratcheted; documented for the bridge.",
    ),
    ConceptDef(
        concept="Linear bottom-drag coefficient",
        canonical="bottom_drag_r",
        units="m/s",
        status="canonical",
        aliases=("bottom_drag_coeff",),
        oracle_aliases=(("veros", "r_bot"), ("mitgcm", "bottomDragLinear")),
        ratchet=True,
        note="state.py field is bottom_drag_r; experiment configs declare "
             "bottom_drag_coeff and map it at build time. New MODEL/config code "
             "must use bottom_drag_r; experiment-layer uses are baselined.",
    ),
    ConceptDef(
        concept="Vertical momentum viscosity (background/fallback)",
        canonical="A_v",
        units="m^2/s",
        status="fidelity-scoped",
        oracle_aliases=(
            ("veros", "kappaM"), ("veros", "kappaM_min"), ("mitgcm", "viscAr"),
        ),
        note="legoESM uses A_v (momentum) / K_v (tracer); Veros TKE uses "
             "kappaM/kappaH (and *_min floors); MITgcm uses viscAr (the 'r' = "
             "vertical/radial axis). The recipe maps Veros's kappaM_min -> "
             "TKEConfig.kappaM_min; no legoESM-core collision.",
    ),
    ConceptDef(
        concept="Vertical tracer diffusivity (background/fallback)",
        canonical="K_v",
        units="m^2/s",
        status="fidelity-scoped",
        oracle_aliases=(
            ("veros", "kappaH"), ("veros", "kappaH_min"), ("mitgcm", "diffKrT"),
        ),
        note="See A_v. Veros's tracer-side floor is kappaH_min; MITgcm's "
             "vertical tracer diffusivity is diffKrT.",
    ),
    ConceptDef(
        concept="Gent-McWilliams (thickness) diffusivity",
        canonical="kappa_GM",
        units="m^2/s",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "K_gm_0"), ("mitgcm", "GM_background_K")),
        note="legoESM GMRediConfig.kappa_GM; Veros K_gm_0; MITgcm pkg/gmredi "
             "GM_background_K. Distinct from the Redi (isoneutral) diffusivity "
             "below — do not conflate.",
    ),
    ConceptDef(
        concept="Redi (isoneutral) diffusivity",
        canonical="kappa_Redi",
        units="m^2/s",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "K_iso_0"), ("mitgcm", "GM_isopycK")),
        note="legoESM GMRediConfig.kappa_Redi; Veros K_iso_0; MITgcm pkg/gmredi "
             "GM_isopycK. Distinct from the GM (thickness) diffusivity above.",
    ),
    ConceptDef(
        concept="Tracer/momentum relaxation timescale",
        canonical="tau_relax_s",
        units="s (suffix _s) or days (suffix _days)",
        status="unit-hazard",
        aliases=("tau_relax",),
        note="convection/config.py tau_relax = SECONDS; "
             "experiments/phillips_two_layer.py tau_relax = DAYS. New code MUST "
             "carry a _s / _days suffix. Human-review (an identifier scan "
             "cannot tell the units), so not auto-ratcheted.",
    ),
    # --- Prognostic state variables (the oracle->legoESM bridge vocabulary). --
    # These let a state bridge call ``oracle_to_canonical("mitgcm")`` to map a
    # snapshot's field names onto LatLonCGridOceanState fields, instead of each
    # bridge hardcoding its own name table. fidelity-scoped: the legoESM names
    # ARE the canonical state-field identifiers, so no core collision.
    ConceptDef(
        concept="Zonal velocity (C-grid u-face)",
        canonical="u",
        units="m/s",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "u"), ("mitgcm", "U"), ("mitgcm", "uVel")),
        note="LatLonCGridOceanState.u. Veros 'u' sits at the EAST T-cell face; "
             "MITgcm 'U' (UVEL) sits at the WEST face — the bridge owns that "
             "half-cell convention, verified by equivariance tests.",
    ),
    ConceptDef(
        concept="Meridional velocity (C-grid v-face)",
        canonical="v",
        units="m/s",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "v"), ("mitgcm", "V"), ("mitgcm", "vVel")),
        note="LatLonCGridOceanState.v. MITgcm 'V' (VVEL) sits at the SOUTH face.",
    ),
    ConceptDef(
        concept="Vertical velocity",
        canonical="w",
        units="m/s",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "w"), ("mitgcm", "W"), ("mitgcm", "wVel")),
        note="LatLonCGridOceanState.w (diagnostic on most configs).",
    ),
    ConceptDef(
        concept="Potential temperature",
        canonical="T",
        units="degC",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "temp"), ("mitgcm", "Theta"), ("mitgcm", "T")),
        note="LatLonCGridOceanState.T. MITgcm dumps potential temperature as "
             "'T'/'Theta' (THETA); Veros names it 'temp'. Both degC.",
    ),
    ConceptDef(
        concept="Salinity",
        canonical="S",
        units="g/kg",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "salt"), ("mitgcm", "S"), ("mitgcm", "Salt")),
        note="LatLonCGridOceanState.S. MITgcm 'S'/'Salt' (SALT); Veros 'salt'.",
    ),
    ConceptDef(
        concept="Free-surface height anomaly",
        canonical="eta",
        units="m",
        status="fidelity-scoped",
        oracle_aliases=(("mitgcm", "Eta"), ("mitgcm", "ETAN")),
        note="LatLonCGridOceanState.eta. MITgcm implicit free surface 'Eta'/"
             "'ETAN'. Veros has no direct eta (barotropic psi); left unmapped.",
    ),
    ConceptDef(
        concept="Horizontal (Laplacian) viscosity",
        canonical="A_h",
        units="m^2/s",
        status="fidelity-scoped",
        oracle_aliases=(("veros", "A_h"), ("mitgcm", "viscAh")),
        note="legoESM A_h; Veros A_h; MITgcm viscAh. Lateral Laplacian momentum "
             "viscosity (sets the Munk layer width delta_M=(A_h/beta)^(1/3)).",
    ),
    ConceptDef(
        concept="Water heat capacity (land/lake)",
        canonical="C_water (volumetric, J/m^3/K) | c_water (specific, J/kg/K)",
        units="J/m^3/K (C_water) vs J/(kg K) (c_water)",
        status="unit-hazard",
        note="land/soil_thermal.py C_water is VOLUMETRIC; "
             "coupler/lake/config.py c_water is SPECIFIC. Same base name, "
             "different units -> any new use must disambiguate. (Land/lake "
             "scope; listed here because it is the canonical example of the "
             "unit-hazard class.)",
    ),
)


def ratcheted_aliases() -> dict[str, str]:
    """{alias: canonical} for every concept with ``ratchet=True`` — the set the
    deterministic naming guard forbids in new code (outside its baseline)."""
    out: dict[str, str] = {}
    for c in CONCEPTS:
        if c.ratchet:
            for a in c.aliases:
                out[a] = c.canonical
    return out


def oracle_to_canonical(oracle: str) -> dict[str, str]:
    """{oracle_name: legoESM_canonical} for one oracle — used by bridges/recipes
    to translate an oracle's vocabulary into legoESM's."""
    out: dict[str, str] = {}
    for c in CONCEPTS:
        for orc, name in c.oracle_aliases:
            if orc == oracle:
                out[name] = c.canonical
    return out


__all__ = [
    "ConceptDef",
    "CONCEPTS",
    "KNOWN_ORACLES",
    "ratcheted_aliases",
    "oracle_to_canonical",
]
