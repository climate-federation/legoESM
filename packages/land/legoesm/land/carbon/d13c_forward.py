"""Per-archetype SIMULATED leaf carbon-isotope discrimination (leaf delta13C) forward.

The MODELLED counterpart of :mod:`legoesm.land.carbon.d13c_observations` (the OBSERVED
side): a per-archetype simulated LEAF delta13C [permil], differentiable in the coupled
stomatal / water-use-efficiency parameters (the Ball-Berry slope ``g1_bb`` etc.) of
:class:`legoesm.land.carbon.stomata.StomataConfig`.

Leaf delta13C is a WATER-USE-EFFICIENCY constraint: photosynthetic 13C discrimination is
set by the intercellular-to-ambient CO2 ratio ``Ci/Ca`` (Farquhar, Ehleringer & Hubick
1989).  A leaf that keeps its stomata more open (higher ``g1``) runs a higher ``Ci/Ca``,
discriminates MORE against the heavier 13C, and so has a MORE NEGATIVE leaf delta13C.  This
is a NEW, INDEPENDENT lever versus SIF (fluorescence, :mod:`sif_forward`) and the live-pool
biomass/LAI streams (allocation, :mod:`live_pool_forward`): SIF/biomass do not see the
stomatal ``Ci/Ca`` at all, whereas delta13C is a direct integrated record of it.

Like the SIF forward and UNLIKE the SOM-SOC forward
(:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced`), this is a
**single-step** photosynthesis diagnostic -- there is NO multi-year spin-up scan.  It
reuses the model's OWN coupled-Farquhar solve
(:func:`legoesm.land.carbon.stomata.solve_coupled_farquhar_ci`) and the SHARED per-archetype
growing-season forcing (:func:`legoesm.land.carbon.archetype_forcing.build_archetype_forcing`,
also used by the SIF forward -- one forcing definition, no copy-paste).

UNLIKE the SIF forward, ``Ci`` is the TRAINED quantity here (it depends on the stomatal
conductance parameters), so the forward CANNOT precompute a static ``Ci``: it RE-SOLVES the
coupled Farquhar system with the TRACED :class:`StomataConfig` on every optimizer step
(still single-step -- no scan -- and differentiable in the stomatal params, so the gradient
reaches ``g1``).  Only the parameter-independent CLIMATE forcing + PFT ``Vc_max25`` are
precomputed once (see :func:`build_d13c_forward`).

C3 vs C4 (CRITICAL for correctness).  This forward implements the **C3** discrimination form
``Delta = a + (b - a)*Ci/Ca`` ONLY.  The model's Farquhar biochemistry is C3-only
(:mod:`legoesm.land.carbon.stomata` -- the C4 PFTs ``c4_grass`` / ``crop_c4`` are run through
C3 kinetics as a documented approximation), so a FAITHFUL C4 intercellular CO2 -- and hence a
faithful C4 discrimination (C4 plants discriminate FAR less, leaf delta13C ~ -11..-14 permil
vs C3 -25..-32 permil, because of their CO2-concentrating mechanism) -- is NOT available.
Applying the C3 form to a C4 leaf would be a magnitude/sign error.  This module therefore
computes the C3 form for EVERY archetype (finite + differentiable), and the C4 archetypes are
MASKED OUT of the calibration delta13C term by the trainer / the observed loader
(:func:`legoesm.land.carbon.d13c_observations.c4_archetype_mask`).  Implementing the
Farquhar-Cerling C4 discrimination form ``Delta_C4 = a + (b4 + (b3 - s)*phi - a)*Ci/Ca`` with
a Collatz C4 biochemistry (so a faithful C4 ``Ci`` exists) is a documented FOLLOW-UP.

SIGN CONVENTION (delta / Delta notation; permil).  ``Delta`` (DISCRIMINATION) is defined
POSITIVE and grows with ``Ci/Ca`` (more open stomata -> more discrimination).  The LEAF
composition is ``delta13C_leaf = delta13C_air - Delta`` -- the leaf is 13C-DEPLETED relative
to the atmosphere, so its delta13C is MORE NEGATIVE than air (delta13C_air ~ -8 permil).
Hence ``d(delta13C_leaf)/d(Ci/Ca) = -(b - a) < 0``: higher ``Ci/Ca`` gives a MORE NEGATIVE
leaf delta13C (the monotonicity the correctness gate asserts).  Physical C3 leaf delta13C
lies in ~ -22..-34 permil.

All functions are pure JAX and differentiable in the ``StomataConfig`` water-use-efficiency
leaves.
"""

from __future__ import annotations

# --- isotope discrimination (Farquhar, Ehleringer & Hubick 1989) -------------
# Photosynthetic 13C discrimination against 13CO2 during C3 assimilation, simplified linear
# form Delta = a + (b - a) * Ci/Ca [permil]; leaf delta13C = delta13C_air - Delta.
# Farquhar, G. D., Ehleringer, J. R. & Hubick, K. T. (1989): Carbon isotope discrimination
# and photosynthesis. Annu. Rev. Plant Physiol. Plant Mol. Biol. 40, 503-537.
# Published constants (NOT tunable): the fixed diffusion / Rubisco fractionations and the
# free-tropospheric boundary-condition delta13C of CO2.
_A_DIFFUSION_PERMIL = 4.4     # 13C fractionation by CO2 diffusion through stomata [permil]
_B_RUBISCO_PERMIL = 27.0      # net 13C fractionation by Rubisco carboxylation [permil]
_DELTA13C_AIR_PERMIL = -8.0   # delta13C of well-mixed atmospheric CO2 (boundary cond.) [permil]

# --- numerics -----------------------------------------------------------------
# Divide-safety floor on ambient CO2 Ca [umol/mol] for the Ci/Ca ratio; Ca is ~400 in the
# forcing (and Ci is clipped to [1, Ca] in the solver), so this only guards a degenerate
# hand-built forcing, never the calibration path.
_CA_FLOOR_PPMV = 1.0


__physics_contract__ = {
    "summary": (
        "Single-step per-archetype simulated LEAF carbon-isotope discrimination (leaf "
        "delta13C) for the Stage-B carbon calibration's delta13C observation stream.  C3 "
        "photosynthetic discrimination Delta = a + (b - a)*Ci/Ca (Farquhar 1989) with the "
        "intercellular:ambient CO2 ratio Ci/Ca from the model's own coupled Farquhar-stomata "
        "solve; leaf delta13C = delta13C_air - Delta.  Trains the stomatal / water-use-"
        "efficiency parameters (Ball-Berry g1_bb) through Ci, which depends on the traced "
        "StomataConfig.  C3-only: C4 PFTs (whose C3-kinetics Ci is not a faithful C4 leaf "
        "state) are MASKED by the trainer, not run through the C3 form.  No spin-up scan; "
        "does not re-derive photosynthesis or the discrimination beyond the Farquhar 1989 form."
    ),
    "inputs": {
        "table": "ArchetypeTable (static; climate features + PFT id)",
        "stomata_config": "StomataConfig (water-use-efficiency leaves; may be TRACED)",
    },
    "outputs": {"leaf_delta13C": "permil"},
    "sign_convention": (
        "Delta (discrimination) POSITIVE, increasing with Ci/Ca (a=4.4 diffusion < b=27 "
        "Rubisco, so b-a>0 and Delta in [a,b]).  leaf delta13C = delta13C_air - Delta, so the "
        "leaf is 13C-DEPLETED relative to air (more negative than delta13C_air ~ -8 permil).  "
        "d(delta13C_leaf)/d(Ci/Ca) = -(b-a) < 0: higher Ci/Ca -> more discrimination -> MORE "
        "NEGATIVE leaf delta13C.  Physical C3 leaf delta13C ~ -22..-34 permil."
    ),
    "conserves": [],  # isotope-ratio diagnostic; not a mass budget
    "differentiable": True,
    "reference": (
        "Farquhar, Ehleringer & Hubick (1989), Annu. Rev. Plant Physiol. 40, 503-537 "
        "(C3 discrimination Delta = a + (b-a) Ci/Ca; a=4.4, b=27 permil).  Ci from the "
        "model's own coupled Farquhar-stomata solve (stomata.solve_coupled_farquhar_ci)."
    ),
    "idealized_test": (
        "tests/land/unit/test_d13c_forward.py: leaf delta13C in the physical C3 range "
        "(~ -22..-34 permil) for C3 archetypes; MONOTONIC the right way "
        "(d(delta13C)/d(Ci/Ca) < 0, and higher Ball-Berry g1_bb -> more negative leaf "
        "delta13C); finite non-zero grad wrt g1_bb; build == simulate."
    ),
}


def leaf_d13c_from_ci_ca(ci_ca):
    """Leaf delta13C [permil] from the intercellular:ambient CO2 ratio ``Ci/Ca`` (C3 form).

    Farquhar (1989) simplified linear discrimination ``Delta = a + (b - a) * Ci/Ca`` and the
    leaf composition ``delta13C_leaf = delta13C_air - Delta`` (the leaf is 13C-depleted
    relative to air).  Pure, differentiable, monotone DECREASING in ``Ci/Ca``
    (``d(delta13C_leaf)/d(Ci/Ca) = -(b - a) < 0``).  Valid for C3 photosynthesis ONLY (the
    caller masks C4 archetypes -- see the module docstring).
    """
    discrimination = _A_DIFFUSION_PERMIL + (
        _B_RUBISCO_PERMIL - _A_DIFFUSION_PERMIL) * ci_ca
    return _DELTA13C_AIR_PERMIL - discrimination


def _leaf_ci_ca(table, stomata_config):
    """Per-archetype ``Ci/Ca`` from the model's own coupled Farquhar-stomata solve ``(n_arch,)``.

    Solves :func:`legoesm.land.carbon.stomata.solve_coupled_farquhar_ci` at the shared
    representative growing-season forcing (built ONCE, parameter-independent) with the
    per-archetype PFT ``Vc_max25`` spliced in and the (possibly TRACED) stomatal
    water-use-efficiency leaves from ``stomata_config``.  Returns ``Ci / Ca`` -- the
    water-use-efficiency ratio the C3 discrimination reads; ``Ci`` carries the gradient to
    the stomatal params (higher ``g1`` -> more open stomata -> higher ``Ci``).
    """
    import jax.numpy as jnp

    from legoesm.land.carbon.archetype_forcing import (
        REF_BETA,
        REF_LAI,
        build_archetype_forcing,
    )
    from legoesm.land.carbon.stomata import solve_coupled_farquhar_ci

    forcing = build_archetype_forcing(table)
    # Per-archetype PFT capacity (Vc_max25) + the traced stomatal leaves; enabled=True
    # selects the coupled Farquhar path.  Vc_max25 stays per-archetype (a scalar override
    # would collapse every archetype's capacity), so the trainer never trains it here.
    cfg = stomata_config._replace(enabled=True, Vc_max25=forcing.Vc_max25)
    leaf = solve_coupled_farquhar_ci(
        forcing.T_leaf, forcing.sw_down, forcing.co2_ppmv,
        forcing.q_air, forcing.p_surface, REF_LAI, REF_BETA, cfg)
    return leaf.Ci / jnp.maximum(forcing.co2_ppmv, _CA_FLOOR_PPMV)


def simulate_archetype_d13c(table, stomata_config):
    """Per-archetype simulated LEAF delta13C ``(n_arch,)`` [permil] (C3 discrimination).

    All-in-one forward: solve the model's coupled Farquhar at each archetype's representative
    growing-season climate (:func:`_leaf_ci_ca`) then apply the Farquhar (1989) C3
    discrimination (:func:`leaf_d13c_from_ci_ca`).  Differentiable in the ``StomataConfig``
    water-use-efficiency leaves (``Ci`` depends on them -- a single-step AD graph, no spin-up
    scan).  Used by the unit test / AD de-risk; the trainer uses :func:`build_d13c_forward`
    to avoid rebuilding the (static) climate forcing each step.

    C3-only: the C4 archetypes are computed here (finite) but MUST be masked out of the
    calibration term by the caller (their C3-kinetics ``Ci`` is not a faithful C4 leaf state;
    see the module docstring + :func:`legoesm.land.carbon.d13c_observations.c4_archetype_mask`).

    Parameters
    ----------
    table : ArchetypeTable
        Stage-A archetypes (static; the isotope forward differentiates only the stomatal
        params).
    stomata_config : StomataConfig
        Coupled-stomata configuration; its water-use-efficiency leaves (``g1_bb`` etc.) may
        be TRACED overrides spliced via ``apply_param_overrides`` inside a loss (production
        defaults untouched).  ``Vc_max25`` is overridden per-archetype from the PFT table.
    """
    return leaf_d13c_from_ci_ca(_leaf_ci_ca(table, stomata_config))


def build_d13c_forward(table):
    """Precompute the static per-archetype climate forcing ONCE and return
    ``d13c_fn(stomata_config) -> (n_arch,)`` leaf delta13C [permil].

    Mirrors :func:`legoesm.land.carbon.sif_forward.build_sif_forward` (a static precomputed
    state + a differentiable kernel), EXCEPT the kernel here RE-SOLVES the coupled Farquhar
    system: ``Ci`` -- and hence the discrimination -- depends on the TRAINED stomatal
    parameters, so only the parameter-independent climate forcing + PFT ``Vc_max25`` are
    frozen (:func:`legoesm.land.carbon.archetype_forcing.build_archetype_forcing`), and the
    returned closure applies the differentiable Farquhar solve + Farquhar-1989 discrimination.
    The stomatal-parameter gradient is a pure single-step graph (no spin-up scan).
    Numerically identical to :func:`simulate_archetype_d13c` for any ``stomata_config``.
    """
    import jax.numpy as jnp

    from legoesm.land.carbon.archetype_forcing import (
        REF_BETA,
        REF_LAI,
        build_archetype_forcing,
    )
    from legoesm.land.carbon.stomata import solve_coupled_farquhar_ci

    forcing = build_archetype_forcing(table)   # static, parameter-independent

    def d13c_fn(stomata_config):
        cfg = stomata_config._replace(enabled=True, Vc_max25=forcing.Vc_max25)
        leaf = solve_coupled_farquhar_ci(
            forcing.T_leaf, forcing.sw_down, forcing.co2_ppmv,
            forcing.q_air, forcing.p_surface, REF_LAI, REF_BETA, cfg)
        ci_ca = leaf.Ci / jnp.maximum(forcing.co2_ppmv, _CA_FLOOR_PPMV)
        return leaf_d13c_from_ci_ca(ci_ca)

    return d13c_fn
