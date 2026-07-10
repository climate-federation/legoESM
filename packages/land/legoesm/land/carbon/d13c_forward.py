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

C3 vs C4 (CRITICAL for correctness -- TWO faithful cases, NOT a C4 mask).  This forward
implements BOTH discrimination pathways and selects per archetype with the shared PFT
classifier (:func:`legoesm.land.surface_params.is_c4_pft_id` on ``table.pft_id``):

* **C3** (:func:`leaf_d13c_from_ci_ca`): ``Delta_C3 = a + (b - a)*Ci/Ca`` (Farquhar 1989),
  with ``Ci/Ca`` from the model's own coupled C3 Farquhar-stomata solve.  leaf delta13C ~
  -22..-34 permil; strongly, monotonically Ci/Ca-dependent.
* **C4** (:func:`leaf_d13c_c4_from_ci_ca`): ``Delta_C4 = a + (b4 + (b3 - s)*phi - a)*Ci/Ca``
  (Farquhar 1983; Henderson 1992; Cerling 1997), with a C4-characteristic FIXED Ci/Ca
  (``D13CConfig.ci_ca_c4`` ~ 0.4) and the bundle-sheath leakiness ``phi``
  (``D13CConfig.phi_c4_leakiness``).  leaf delta13C ~ -11..-14 permil -- DISTINCTLY less
  negative than C3, and only WEAKLY (even non-monotonically) Ci/Ca-dependent, because the
  CO2-concentrating mechanism buffers it (the small bracket ``(b4 + (b3-s)*phi - a)``).

selection: ``leaf_delta13C = jnp.where(is_c4, delta_C4, delta_C3)`` -- BOTH branches finite +
differentiable, so the C4 archetypes now return a FAITHFUL C4 value and are INCLUDED in the
calibration delta13C loss (no more C4 masking).

Why a FIXED C4 Ci/Ca (option (a)).  The model's Farquhar biochemistry is C3-only
(:mod:`legoesm.land.carbon.stomata` -- the C4 PFTs ``c4_grass`` / ``crop_c4`` are run through
C3 kinetics), so the solved ``Ci`` is NOT a faithful C4 leaf state (it lacks the CO2-
concentrating bundle-sheath step).  Rather than feed a wrong C3-kinetics Ci into the C4 form,
the C4 branch regulates to a prescribed C4-characteristic Ci/Ca (C4 leaves hold Ci/Ca ~ 0.4
relatively tightly), so the C4 delta13C depends MAINLY on ``phi`` -- the natural, physically-
primary C4 lever.  ``phi`` is a tunable ``D13CConfig`` closure (tier 2); ``ci_ca_c4`` is FIXED
(exposing both would be a non-identifiable ``(phi, Ci/Ca)`` degeneracy -- see
:class:`legoesm.land.carbon.config.D13CConfig`).  A Collatz C4 biochemistry that would supply
a faithful solved C4 ``Ci`` (so the C4 branch could read a model Ci like the C3 branch) is a
documented FOLLOW-UP; it would REPLACE the fixed setpoint, not the discrimination form.

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

# --- C4 isotope discrimination (Farquhar 1983; Henderson 1992; Cerling 1997) --
# C4 photosynthetic 13C discrimination Delta_C4 = a + (b4 + (b3 - s)*phi - a) * Ci/Ca [permil];
# leaf delta13C = delta13C_air - Delta_C4.  The diffusion a (_A_DIFFUSION_PERMIL) and Rubisco
# b3 (_B_RUBISCO_PERMIL) fractionations are SHARED with the C3 form; b4 and s below are the
# C4-specific published constants.  The bundle-sheath leakiness phi and the C4-characteristic
# Ci/Ca are NOT here -- they are the (tunable / fixed) D13CConfig knobs (config is not a
# provenance constant).  Because the CO2-concentrating mechanism buffers it, C4 discrimination
# is SMALL (Delta ~ 2-4 permil) and only weakly Ci/Ca-dependent -> leaf delta13C ~ -11..-14
# permil, distinctly less negative than C3.
# Farquhar, G. D. (1983): On the nature of carbon isotope discrimination in C4 species,
#   Aust. J. Plant Physiol. 10, 205-226.
# Henderson, S. A., von Caemmerer, S. & Farquhar, G. D. (1992): Short-term measurements of
#   carbon isotope discrimination in several C4 species, Aust. J. Plant Physiol. 19, 263-285.
# Cerling, T. E. et al. (1997): Global vegetation change through the Miocene/Pliocene
#   boundary, Nature 389, 153-158.  Published constants (NOT tunable):
_B4_PEP_PERMIL = -5.7   # net 13C fractionation of CO2->HCO3- + PEP carboxylation (~25C) [permil]
_S_LEAK_PERMIL = 1.8    # 13C fractionation of CO2 leaking out of the bundle sheath [permil]

# --- numerics -----------------------------------------------------------------
# Divide-safety floor on ambient CO2 Ca [umol/mol] for the Ci/Ca ratio; Ca is ~400 in the
# forcing (and Ci is clipped to [1, Ca] in the solver), so this only guards a degenerate
# hand-built forcing, never the calibration path.
_CA_FLOOR_PPMV = 1.0


__physics_contract__ = {
    "summary": (
        "Single-step per-archetype simulated LEAF carbon-isotope discrimination (leaf "
        "delta13C) for the Stage-B carbon calibration's delta13C observation stream.  TWO "
        "faithful pathways selected per archetype by is_c4(pft_id): C3 discrimination "
        "Delta_C3 = a + (b - a)*Ci/Ca (Farquhar 1989) with Ci/Ca from the model's own coupled "
        "Farquhar-stomata solve (trains the Ball-Berry g1_bb through Ci); and C4 discrimination "
        "Delta_C4 = a + (b4 + (b3 - s)*phi - a)*Ci/Ca (Farquhar 1983 / Henderson 1992 / Cerling "
        "1997) at a FIXED C4-characteristic Ci/Ca (D13CConfig.ci_ca_c4) with the bundle-sheath "
        "leakiness phi (D13CConfig.phi_c4_leakiness, the C4 lever).  leaf delta13C = "
        "delta13C_air - Delta.  C4 archetypes are INCLUDED in the loss (no masking).  No "
        "spin-up scan; does not re-derive photosynthesis or the discrimination beyond these "
        "published forms."
    ),
    "inputs": {
        "table": "ArchetypeTable (static; climate features + PFT id -> C3/C4 selector)",
        "stomata_config": "StomataConfig (C3 water-use-efficiency leaves; may be TRACED)",
        "d13c_config": "D13CConfig (C4 phi leakiness [TRACED lever] + fixed ci_ca_c4)",
    },
    "outputs": {"leaf_delta13C": "permil"},
    "sign_convention": (
        "Delta (discrimination) POSITIVE; leaf delta13C = delta13C_air - Delta, so the leaf is "
        "13C-DEPLETED relative to air (more negative than delta13C_air ~ -8 permil).  C3: "
        "a=4.4 diffusion < b=27 Rubisco, so b-a>0 and Delta in [a,b]; "
        "d(delta13C_leaf)/d(Ci/Ca) = -(b-a) < 0 (higher Ci/Ca -> MORE NEGATIVE), physical C3 "
        "leaf delta13C ~ -22..-34 permil.  C4: the bracket (b4 + (b3-s)*phi - a) is SMALL "
        "(b4=-5.7, s=1.8; ~ -4.8 at phi=0.21), so Delta_C4 ~ 2-4 permil and leaf delta13C ~ "
        "-11..-14 permil -- distinctly LESS negative than C3, only weakly Ci/Ca-dependent."
    ),
    "conserves": [],  # isotope-ratio diagnostic; not a mass budget
    "differentiable": True,
    "reference": (
        "C3: Farquhar, Ehleringer & Hubick (1989), Annu. Rev. Plant Physiol. 40, 503-537 "
        "(Delta = a + (b-a) Ci/Ca; a=4.4, b=27 permil).  C4: Farquhar (1983) Aust. J. Plant "
        "Physiol. 10, 205-226; Henderson, von Caemmerer & Farquhar (1992) ibid. 19, 263-285 "
        "(phi ~ 0.2-0.3); Cerling et al. (1997) Nature 389, 153-158.  C3 Ci from "
        "stomata.solve_coupled_farquhar_ci; C4 Ci/Ca a fixed C4-characteristic setpoint."
    ),
    "idealized_test": (
        "tests/land/unit/test_d13c_forward.py: C3 archetypes leaf delta13C in ~ -22..-34 "
        "permil, MONOTONIC (d/d(Ci/Ca) < 0, higher g1_bb -> more negative); C4 archetypes "
        "distinctly less negative ~ -16..-10 permil (a C4 in the C3 band -- or vice versa -- "
        "FAILS); finite non-zero grad wrt g1_bb (C3) AND phi (C4); build == simulate."
    ),
}


def leaf_d13c_from_ci_ca(ci_ca):
    """Leaf delta13C [permil] from the intercellular:ambient CO2 ratio ``Ci/Ca`` (C3 form).

    Farquhar (1989) simplified linear discrimination ``Delta = a + (b - a) * Ci/Ca`` and the
    leaf composition ``delta13C_leaf = delta13C_air - Delta`` (the leaf is 13C-depleted
    relative to air).  Pure, differentiable, monotone DECREASING in ``Ci/Ca``
    (``d(delta13C_leaf)/d(Ci/Ca) = -(b - a) < 0``).  This is the C3 branch; C4 archetypes use
    the SEPARATE :func:`leaf_d13c_c4_from_ci_ca` (selected per archetype by ``is_c4``, NOT
    masked -- see the module docstring).
    """
    discrimination = _A_DIFFUSION_PERMIL + (
        _B_RUBISCO_PERMIL - _A_DIFFUSION_PERMIL) * ci_ca
    return _DELTA13C_AIR_PERMIL - discrimination


def leaf_d13c_c4_from_ci_ca(ci_ca, phi_leakiness):
    """Leaf delta13C [permil] for a **C4** leaf (Farquhar 1983 / Henderson 1992 / Cerling 1997).

    C4 photosynthetic discrimination
    ``Delta_C4 = a + (b4 + (b3 - s) * phi - a) * Ci/Ca`` and the leaf composition
    ``delta13C_leaf = delta13C_air - Delta_C4``, where ``a`` (diffusion) and ``b3`` (Rubisco)
    are SHARED with the C3 form (:func:`leaf_d13c_from_ci_ca`), ``b4`` is the net PEP-
    carboxylation fractionation, ``s`` the fractionation of CO2 leaking out of the bundle
    sheath, and ``phi`` the bundle-sheath LEAKINESS.  Pure, differentiable in BOTH ``ci_ca``
    and ``phi_leakiness``.

    Distinct from C3 by design: because ``phi`` is small, the bracket
    ``(b4 + (b3 - s) * phi - a)`` is small (and can be negative), so ``Delta_C4`` is SMALL
    (~ 2-4 permil) and the C4 leaf delta13C (~ -11..-14 permil) is DISTINCTLY less negative
    than C3 and only WEAKLY Ci/Ca-dependent -- the CO2-concentrating mechanism buffers it.
    ``phi`` is the natural C4 water-use-efficiency / delta13C lever
    (``d(Delta_C4)/d(phi) = (b3 - s) * Ci/Ca > 0`` -> more leaky -> MORE discrimination -> MORE
    negative leaf delta13C).  The C4 forward feeds a FIXED C4-characteristic ``ci_ca`` here
    (the C3-kinetics model Ci is not a faithful C4 leaf state -- see the module docstring).
    """
    discrimination = _A_DIFFUSION_PERMIL + (
        _B4_PEP_PERMIL + (_B_RUBISCO_PERMIL - _S_LEAK_PERMIL) * phi_leakiness
        - _A_DIFFUSION_PERMIL) * ci_ca
    return _DELTA13C_AIR_PERMIL - discrimination


def _archetype_is_c4(table):
    """Static per-archetype C4 selector ``(n_arch,)`` bool from ``table.pft_id``.

    Delegates to the SHARED PFT classifier
    (:func:`legoesm.land.surface_params.is_c4_pft_id`) -- one source of truth with the
    observed-side mask (:func:`legoesm.land.carbon.d13c_observations.c4_archetype_mask`).
    Pure NumPy on the STATIC PFT id, so the result is a compile-time selector for the
    ``jnp.where(is_c4, delta_C4, delta_C3)`` split (no data-dependent branching).
    """
    from legoesm.land.surface_params import is_c4_pft_id

    return is_c4_pft_id(table.pft_id)


def _select_c3_c4(is_c4, ci_ca_c3, d13c_config):
    """Per-archetype leaf delta13C ``(n_arch,)`` [permil]: C4 branch where ``is_c4`` else C3.

    ``is_c4`` is the static per-archetype C4 selector (:func:`_archetype_is_c4`); ``ci_ca_c3``
    is the model's coupled C3 Farquhar-stomata ``Ci/Ca`` (per archetype).  The C4 branch
    IGNORES ``ci_ca_c3`` and uses the FIXED C4-characteristic ``d13c_config.ci_ca_c4`` with the
    (possibly TRACED) leakiness ``d13c_config.phi_c4_leakiness``.  Both branches are finite and
    differentiable, so ``jnp.where`` carries the g1_bb gradient through the C3 archetypes and
    the phi gradient through the C4 archetypes (a C4 archetype's delta13C does NOT depend on
    the C3 stomatal params, and a C3 archetype's does NOT depend on phi -- the correct
    decoupling).  The C4 value is identical across C4 archetypes (a fixed setpoint), so the C4
    term calibrates ``phi`` against the cover-weighted-mean observed C4 delta13C.
    """
    import jax.numpy as jnp

    d13c_c3 = leaf_d13c_from_ci_ca(ci_ca_c3)
    d13c_c4 = leaf_d13c_c4_from_ci_ca(
        d13c_config.ci_ca_c4, d13c_config.phi_c4_leakiness)
    return jnp.where(jnp.asarray(is_c4), d13c_c4, d13c_c3)


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


def simulate_archetype_d13c(table, stomata_config, d13c_config=None):
    """Per-archetype simulated LEAF delta13C ``(n_arch,)`` [permil] (C3 + C4 discrimination).

    All-in-one forward: solve the model's coupled C3 Farquhar at each archetype's
    representative growing-season climate (:func:`_leaf_ci_ca`), then select PER ARCHETYPE
    (:func:`_select_c3_c4`, keyed on ``is_c4(table.pft_id)``) between the Farquhar (1989) C3
    discrimination (:func:`leaf_d13c_from_ci_ca`, from the solved C3 ``Ci/Ca``) and the
    Farquhar-Cerling C4 discrimination (:func:`leaf_d13c_c4_from_ci_ca`, at the FIXED C4
    ``ci_ca_c4`` with the leakiness ``phi``).  Differentiable in BOTH the ``StomataConfig``
    water-use-efficiency leaves (``g1_bb``; C3 archetypes, through the solved ``Ci``) and the
    ``D13CConfig`` leakiness ``phi`` (C4 archetypes) -- a single-step AD graph, no spin-up
    scan.  Used by the unit test / AD de-risk; the trainer uses :func:`build_d13c_forward` to
    avoid rebuilding the (static) climate forcing each step.

    C4 archetypes now return a FAITHFUL C4 value and are INCLUDED in the calibration loss (no
    masking -- see the module docstring).

    Parameters
    ----------
    table : ArchetypeTable
        Stage-A archetypes (static; ``pft_id`` selects the C3/C4 branch, ``Vc_max25`` from the
        PFT table).
    stomata_config : StomataConfig
        Coupled C3-stomata configuration; its water-use-efficiency leaves (``g1_bb`` etc.) may
        be TRACED overrides spliced via ``apply_param_overrides`` inside a loss (production
        defaults untouched).  ``Vc_max25`` is overridden per-archetype from the PFT table.
    d13c_config : D13CConfig, optional
        C4 discrimination knobs (``phi_c4_leakiness`` [TRACED C4 lever] + fixed ``ci_ca_c4``).
        Defaults to :class:`legoesm.land.carbon.config.D13CConfig` (phi=0.21, Ci/Ca=0.4).
    """
    from legoesm.land.carbon.config import D13CConfig

    if d13c_config is None:
        d13c_config = D13CConfig()
    return _select_c3_c4(
        _archetype_is_c4(table), _leaf_ci_ca(table, stomata_config), d13c_config)


def build_d13c_forward(table):
    """Precompute the static per-archetype climate forcing ONCE and return
    ``d13c_fn(stomata_config, d13c_config=None) -> (n_arch,)`` leaf delta13C [permil].

    Mirrors :func:`legoesm.land.carbon.sif_forward.build_sif_forward` (a static precomputed
    state + a differentiable kernel), EXCEPT the kernel here RE-SOLVES the coupled C3 Farquhar
    system: the C3 ``Ci`` -- and hence the C3 discrimination -- depends on the TRAINED stomatal
    parameters, so only the parameter-independent climate forcing + PFT ``Vc_max25`` (and the
    static ``is_c4`` selector) are frozen
    (:func:`legoesm.land.carbon.archetype_forcing.build_archetype_forcing`), and the returned
    closure applies the differentiable Farquhar solve + the per-archetype C3/C4 discrimination
    (:func:`_select_c3_c4`).  The stomatal-parameter (C3) and leakiness ``phi`` (C4) gradients
    are a pure single-step graph (no spin-up scan).  Numerically identical to
    :func:`simulate_archetype_d13c` for any ``stomata_config`` / ``d13c_config``.
    """
    import jax.numpy as jnp

    from legoesm.land.carbon.archetype_forcing import (
        REF_BETA,
        REF_LAI,
        build_archetype_forcing,
    )
    from legoesm.land.carbon.config import D13CConfig
    from legoesm.land.carbon.stomata import solve_coupled_farquhar_ci

    forcing = build_archetype_forcing(table)   # static, parameter-independent
    is_c4 = jnp.asarray(_archetype_is_c4(table))   # static per-archetype C3/C4 selector

    def d13c_fn(stomata_config, d13c_config=None):
        if d13c_config is None:
            d13c_config = D13CConfig()
        cfg = stomata_config._replace(enabled=True, Vc_max25=forcing.Vc_max25)
        leaf = solve_coupled_farquhar_ci(
            forcing.T_leaf, forcing.sw_down, forcing.co2_ppmv,
            forcing.q_air, forcing.p_surface, REF_LAI, REF_BETA, cfg)
        ci_ca = leaf.Ci / jnp.maximum(forcing.co2_ppmv, _CA_FLOOR_PPMV)
        return _select_c3_c4(is_c4, ci_ca, d13c_config)

    return d13c_fn
