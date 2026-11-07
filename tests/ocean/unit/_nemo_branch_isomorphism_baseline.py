"""Registry for the NEMO branch-isomorphism ratchet (fidelity audit, 2026-09).

USER PRINCIPLE this enforces (see ``docs/ocean/fidelity/nemo_branch_isomorphism_map.md``):
legoESM's branch structure must be ISOMORPHIC to NEMO's. Where NEMO shares one
routine across configurations, legoESM must share ONE implementation too — the
only legitimate branch points are NEMO's own namelist/cpp switches. A SECOND
legoESM implementation of the same NEMO routine (a scheme-local twin, a
'_ws'/'_nemo_kmm' copy of a shared helper, a per-case selector NEMO does not
have) is an ARTIFICIAL BRANCH even when both copies are individually correct,
because a fix landed in one does not reach the other — this happened for real
during the audit (``_stage_vertical_up3`` / live-HPG fixes landed in the WS
"ladder #2" only; see the doc's ranked item 3).

``ROUTINE_REGISTRY`` transcribes the audit's "Routine -> implementation map"
(one entry per NEMO routine the audit examined). Each entry names:
  * ``disposition`` — the audit's classification: ``SHARED`` (one legoESM
    impl reached by every card that reaches the NEMO routine — correct),
    ``NEMO_SWITCH`` (several impls, each a REAL distinct NEMO namelist arm —
    legitimate), ``ARTIFICIAL_BRANCH`` (several impls for what NEMO treats as
    ONE arm — a defect), ``OTHER_RECIPE`` (several impls, but the "extra" one(s)
    are another oracle's/recipe's own named transcription, not a duplicate of
    the same reference arm — legitimate, see the 2026-09-02 reclassification
    below), ``ABSENT`` (NEMO routine has no legoESM counterpart), or
    ``UNVERIFIED`` (audit did not trace far enough to classify). A row whose
    doc disposition cell literally reads "see S-XX" (the routine shares its
    underlying legoESM symbol(s) with another already-registered row) keeps
    that literal text here too, e.g. ``"see S-09"`` — such a row's duplication
    is enforced entirely on the row it points at, never under its own id (see
    ``_is_delegated`` in the checker).
  * ``impls`` — the legoESM symbols (module path under the ``legoesm``
    namespace + a name AST-resolvable inside it) the audit identified as
    implementing this routine. Populated only where the audit named a
    concrete, independently-defined symbol (a top-level or nested ``def``/
    ``class``) that a grep against the tree at audit time (2026-09-01/02)
    confirmed exists — see the doc's per-row ``file:line`` citations. Left
    empty where the audit's note names a config selector, an inline branch
    with no enclosing symbol of its own, or a directory glob rather than one
    checkable symbol; those rows are still listed (full map coverage) but
    carry no AST/duplication enforcement. NOT every one of the audit's ~48
    ``*_evaluation`` sub-selectors is transcribed (rank-10 in the doc's ranked
    list) — two representative, individually-verified pairs are (TKE solver,
    GM/Redi face-thickness); the rest is documented debt, not silently
    dropped coverage (see the doc for the full ~48-name list).
  * each ``Impl`` additionally carries a structured ``reference`` (see
    ``Reference`` below) — which model/recipe this implementation transcribes,
    the arm/scheme value that selects it, and a citation. This is the
    mechanism that closes the 2026-09-02 independent-review BLOCKING finding
    (see below): duplication is now detected from the *references themselves*,
    not from the ``disposition`` label a row happens to carry.

``ARTIFICIAL_BRANCH_BASELINE`` is the SHRINK-ONLY allow-list: a routine_id may
appear here only while its registry row genuinely needs one — see
``_row_needs_baseline_entry`` in the checker for the exact, disposition-
independent condition (an ``ARTIFICIAL_BRANCH`` row with >=2 distinct impls, OR
>=2 impls that share the identical (model, arm) reference, OR any impl whose
reference is not yet triaged (``model="unclassified"``)). The reason string is
lifted from the audit's own note / ranked-branches section. Collapsing a
branch (the audit's collapse plans), or fully triaging every impl's reference,
means this dict's entry must be deleted in the SAME PR (stale-entry check).

Per USER CORRECTION (2026-09-02): legoESM hosts several recipes (NEMO, Veros,
MITgcm, Oceananigans, legacy legoESM), so a second implementation is only a
defect when two implementations claim the SAME reference arm — a NEMO
namelist/cpp switch OR a named non-NEMO reference (recipe + citation). Each
baseline entry therefore also carries a ``kind`` classifying which case it
is: ``"nemo_duplicate"`` (two impls of one NEMO arm — the real defect this
ratchet exists for), ``"other_recipe"`` (the "duplicate" is actually another
oracle's own transcription — not a defect), ``"orphan"`` (one of the impls
belongs to no live card/reference and should just be deleted), or
``"unclassified"`` (not yet triaged).

**2026-09-02 independent-review fixes + reclassification pass** (see
``nemo_branch_isomorphism_map.md`` §6 for the full table and citations):

1. (Finding 1) 13 of the doc's 51 rows were silently absent from
   ``ROUTINE_REGISTRY`` — S-11, S-12, S-15, S-22, S-30, S-31, S-36, S-39,
   S-43, S-44, M-02, M-03, M-06. All 13 are now transcribed: fully enforced
   where the audit named an AST-resolvable symbol, or as an explicit
   placeholder (``impls`` limited to a single enclosing-function citation, or
   empty) with a one-line reason in the ``nemo_switch``/notes field naming the
   function and branch sites by line, per the same convention already used
   for S-33/S-34's un-enforced divisor branch. ``test_no_scheme_duplication.py``
   now has a coverage test that fails if any doc row id goes missing again.
2. (Finding 2) S-25's disposition had been transcribed as
   ``"ABSENT (ORCA1 only, no NEMO counterpart)"``, silently exempting its 2
   implementations from the ratchet. The doc's own row is (and was)
   ``ARTIFICIAL_BRANCH`` — restored here, then reclassified ``OTHER_RECIPE``
   in the same pass below (its third arm, ``nemo_advective``, is now also
   transcribed).
3. Of the doc's 19 literally-``ARTIFICIAL_BRANCH``-tagged rows, an
   independent reclassification (table reproduced in the doc, §6) found 7
   genuine ``nemo_duplicate`` defects (S-12, S-16 (see the 2026-09-02 USER
   DECISION note below — given a real reference and reclassified
   ``OTHER_RECIPE`` after this pass first ran), S-19, S-30, S-35,
   S-42, M-01 — kept ``ARTIFICIAL_BRANCH``, baselined below where >=2 distinct
   AST symbols exist) and 12 ``other_recipe`` rows (S-03, S-04, S-07, S-09,
   S-18 (see the 2026-09-02 SECOND reclassification note below — collapsed by
   1d6a7448d after this pass first ran, then reclassified), S-25, S-27,
   S-28_29's S-29 component, S-32, S-33_34, S-40 — moved to
   disposition ``OTHER_RECIPE`` and OUT of the baseline dict, since the
   "duplicate" arm is a legitimately different, separately-cited reference;
   the registry's ``nemo_switch`` field now names each arm's reference).
   0 rows classified ``orphan``. No row's classification is contested between
   the reclassification table and the audit doc (see the doc's new
   "Disagreements" subsection) — the nemo_duplicate/other_recipe/orphan axis
   is new with this pass, not a revision of the doc's own SHARED/NEMO_SWITCH/
   ARTIFICIAL_BRANCH/ABSENT/UNVERIFIED dispositions.
   S-12 and S-30 were genuine ``nemo_duplicate`` defects by the audit's own
   framing (S-30 is the doc's rank-3 collapse item) whose two branch sites were
   inline code inside one function (``_step_impl``), not AST-distinct symbols —
   same carve-out as S-33/S-34, so they never carried a baseline entry.
   **COLLAPSED 2026-09-02**: the pre-barotropic WS momentum ladder is deleted
   and the barotropic solve is seeded with the BEFORE velocity, as
   ``stp_2D`` does (``stp2d.F90:177-186,280-281``); one ladder survives, after
   the solve. Both rows are now ``SHARED`` with that single site. The move was
   gated at ≤2 float64 ulp per certified row with T/S bit-identical — see
   ``legoesm.ocean.fidelity.ulp_move_gate`` and the phase-3 receipts.

**2026-09-02 second independent review (HOLD, one BLOCKING finding fixed in
this pass)**: the checker only ever validated rows tagged
``disposition="ARTIFICIAL_BRANCH"``. The reviewer demonstrated that flipping a
genuine duplicate's disposition to ``"OTHER_RECIPE"`` and deleting its
baseline entry left the suite green — ``OTHER_RECIPE`` rows were never
inspected at all. Fixed by making every ``Impl`` carry a structured
``Reference(model, arm, citation)`` (see below) and adding disposition-
INDEPENDENT checks in ``find_isomorphism_violations``:
  (a) implementations are grouped by ``(reference.model, reference.arm)`` for
      EVERY row regardless of ``disposition``; a group of >=2 needs a baseline
      entry whose ``kind`` is ``nemo_duplicate`` or ``unclassified``;
  (b) any impl whose ``reference.model == "unclassified"`` requires the row
      to carry a baseline entry (any kind) — an uncited reference can no
      longer just be silently written down; it must be either cited for real
      or flagged for someone to triage;
  (c) an ``OTHER_RECIPE`` row (when it has >=2 impls to compare) must resolve
      to >=2 distinct ``(model, arm)`` groups, and no more than one of its
      impls may cite ``model="nemo"`` — two NEMO arms hiding behind
      "OTHER_RECIPE" is NEMO_SWITCH/ARTIFICIAL_BRANCH territory, not a genuine
      cross-recipe fork;
  (d) an impl with ``reference.model == "nemo"`` and ``reference.arm`` in
      ``("", "none")`` is invalid — a NEMO reference must name which arm it
      is, or it cannot be distinguished from any other NEMO impl in the same
      row.
See ``tests/ocean/unit/test_no_scheme_duplication.py`` for the 7 synthetic
self-checks proving each of these branches (plus the pre-existing 3) actually
fires.

**2026-09-02 THIRD independent review (HOLD, "common duplicate shape"
laundering fixed in this pass)**: the reviewer showed the second-pass fix
still missed the ordinary case — a row whose two impls simply cite DIFFERENT
``arm`` strings (S-16's "dyn_spg_ts_continuity" vs
"dyn_spg_ts_continuity_generic", S-18's real-but-uninvolved MITgcm citation,
S-19's differently-named generic arm) never matches the (model, arm)-group
rule, and once flipped to ``OTHER_RECIPE`` with its baseline entry deleted, 2
distinct groups with only 1 ``nemo`` impl satisfies the diversity rule too —
so the row escapes both existing checks for free. Fixed by adding
``Reference.selected_by`` (the recipe/card names that actually route to an
impl) plus a disposition-independent rule: any impl whose model is not
``nemo``/``paper`` and whose ``selected_by`` names no genuine non-NEMO recipe
is an UNREFERENCED ARM; a row pairing a ``nemo`` impl with one requires a
baseline entry regardless of disposition or the arm's own label. Populated
``selected_by`` only where ``nemo_branch_isomorphism_map.md`` §6.1 or the
recipe catalog itself gives a real, grep-verified name (``omip_nemo_match_
tripole_v1``/``_mpas_v1`` for S-03's insitu bn2; the Veros fidelity recipe
files for S-03's adiabatic and S-07/S-07b's TKE/Langmuir defaults;
``veros_faithful_v1`` for S-25's centered_full; ``default_wright_v1``/
``legoesm_linear_v1`` for S-28_29's mitgcm adcroft citation; ``veros_faithful_
v1``/``oceananigans_v1``/``mitgcm_v1`` for S-32's flux_divergence) — never
invented. This surfaced 6 rows (S-02, S-04, S-15, S-25, S-26, S-32) that were
previously unflagged despite pairing a ``nemo`` impl with a non-NEMO/non-paper
arm nothing actually selects; each gets a new ``kind='unclassified'`` entry
below with an honest "genuinely untriaged" reason — none of them invents a
selector that is not real. See ``test_no_scheme_duplication.py`` for the 2
new self-checks (the exact S-16-shaped laundering, and a fabricated
``selected_by`` name) plus a live re-run of the reviewer's three flips
(S-16/S-18/S-19), each confirmed to go red under this fix.

**2026-09-02 SECOND reclassification (S-18 only, post-1d6a7448d)**: re-tracing
the actual call graph shows the "LOCK/OVERFLOW resolve MIN" finding above was
itself a misreading, not a fact that changed. Both the WS-RK3 stage transport
(``_nemo_ws_qco_stage_faces``) and the MLF tracer transport already called the
shared kernel ``nemo_qco_live_face_geometry_from_operands`` even at audit time
— commit 1d6a7448d ("One NEMO qco face thickness, reached by the RK3 and MLF
lanes alike") describes itself as a "PURE REFACTOR" with "the operand order,
the arithmetic ... unchanged", and the diff confirms it: it merely routes both
lanes through one adapter symbol, ``vertical.nemo_qco_live_face_geometry_cgrid``
(grep-verified: both call sites resolve to it on this checkout;
``nemo_qco_live_face_thicknesses`` is now a thin wrapper serving only the
separate S-19/wzv and S-43/GM-Redi consumers) and adds a non-vacuous pinning
test. The row's actual second impl, ``min_cell_to_uface``, was never computing
this quantity for LOCK/OVERFLOW/ORCA1 — the earlier pass conflated its own
citation of ``zad_qco_evaluation``/``wzv_call2_evaluation`` (S-19's gates, read
inside the PE lane's ``nemo_qco_wzv_operands``) with this row's own gate
(``_NEMOWSRK3TestHooks.legacy_stage_min_face_thickness``, which no card ever
sets True). S-18 is reclassified ``OTHER_RECIPE`` and dropped from the
baseline: ``min_cell_to_uface``'s real, separately-cited role (the MOM6/MITgcm
hFacW=min convention for the PE-lane depth-average/slow-forcing) is genuinely
reached by the non-NEMO recipes too (``veros_faithful_v1``/``mitgcm_v1``/
``oceananigans_v1`` structurally cannot reach the NEMO arm — it needs raw NEMO
mesh operands their z-coordinates never carry), unlike the still-open S-26
case this docstring's ``Reference`` section now cites instead. See
``docs/ocean/fidelity/nemo_branch_isomorphism_map.md``'s "Triage against HEAD
648e5cd69" section for the full trace.

Re-derive by re-reading ``docs/ocean/fidelity/nemo_branch_isomorphism_map.md``
and re-grepping each ``impls`` symbol; do not hand-edit the disposition
without updating the doc it transcribes.
"""

from __future__ import annotations

from typing import NamedTuple

# The only kinds a baseline entry's ``kind`` field may take. See the
# docstring above. Kept here (not just in the test) so both the checker and
# any future reclassification tooling import one source of truth.
VALID_BASELINE_KINDS = frozenset({
    "nemo_duplicate", "other_recipe", "orphan", "unclassified",
})

# The only models a ``Reference.model`` may take. ``nemo``/``veros``/
# ``mitgcm``/``oceananigans``/``paper`` require a real, non-empty citation
# (a NEMO file:line, a source path, a DOI/paper, or a named scheme paper).
# ``legoesm_legacy`` is for CONFIRMED pre-existing legoESM code with no
# external reference (citation = "legoESM legacy pre-existing: <module>").
# ``unclassified`` is reserved for genuinely untriaged references (an audit
# note that says UNVERIFIED, not "this is just an unreferenced default") —
# any impl carrying it forces the row to have a baseline entry (see the
# checker's ``_row_needs_baseline_entry``), so an untriaged reference cannot
# just be written down and forgotten.
VALID_REFERENCE_MODELS = frozenset({
    "nemo", "veros", "mitgcm", "oceananigans", "legoesm_legacy", "paper",
    "unclassified",
})


class Reference(NamedTuple):
    """A structured citation for one ``Impl``.

    ``model`` — which recipe/oracle this implementation transcribes; one of
    ``VALID_REFERENCE_MODELS``.
    ``arm`` — the namelist/cpp switch value or scheme name that selects this
    implementation (e.g. ``"ln_teos10"``, ``"nemo_bn2"``, ``"adiabatic"``).
    Never ``""``/``"none"`` when ``model="nemo"`` (see checker rule (d) above)
    — a NEMO reference must name its actual arm.
    ``citation`` — never empty. A NEMO file:line, a source path, a DOI/paper,
    or ``"legoESM legacy pre-existing: <commit or module>"``.
    ``selected_by`` — (2026-09-02 second-review fix, closes the "common
    duplicate shape" laundering: a row escapes the (model, arm)-matching
    duplicate rule for free the moment its two impls' ``arm`` strings differ,
    even when neither is genuinely selected by anything.) The recipe/card
    names (verifiable against ``list_recipes()`` or the closed
    ``KNOWN_CARDS``/``EXTRA_NON_NEMO_RECIPE_FILES`` lists below) that
    ACTUALLY route to this implementation. Empty by default — do not invent
    one; leave it empty wherever the audit doc's "selected by" column names
    nothing concrete for this specific arm. A non-``nemo``, non-``paper``
    impl with an empty ``selected_by`` is an UNREFERENCED ARM (see the
    checker's ``_is_legitimately_referenced``): ``legoesm_legacy``/
    ``unclassified`` carry no external validation at all, and even a
    ``veros``/``mitgcm``/``oceananigans`` model's OWN citation is not by
    itself proof that a real non-NEMO recipe runs it FOR that reason (S-26's
    ``al81`` cites no recipe at all and stays genuinely untriaged for exactly
    this reason). S-18 was believed to be such a case ("Adcroft, Hill &
    Marshall 1997" for its MIN convention, but every named consumer was a
    NEMO-fidelity CARD, not a distinct model recipe) until the 2026-09-02
    reclassification below found real non-NEMO consumers
    (``veros_faithful_v1``/``mitgcm_v1``/``oceananigans_v1``) for it — see the
    row's own note for the measurement.
    ``paper`` is the one model that stands on its own without a
    ``selected_by``: it names an external, peer-reviewed FORMULA (Wright
    1997, Visbeck et al. 1997), not a claim of "this literally transcribes
    model X's recipe" that needs a live consumer to back it up.
    """
    model: str
    arm: str
    citation: str
    selected_by: tuple[str, ...] = ()


# Closed list of NEMO-fidelity CARD/driver identifiers a ``selected_by`` tuple
# may verifiably name (2026-09-02 second-review fix, item 3). These are real,
# grep-confirmed entry points — but note they do NOT legitimize a non-nemo,
# non-paper impl on their own (see ``_is_legitimately_referenced`` in the
# test): a NEMO card landing on a non-NEMO-cited default is precisely the
# S-16/S-18/S-19 defect, not a distinct recipe's deliberate choice. This dict
# exists so a real card name in ``selected_by`` is not mistaken for a
# made-up/misspelled one by the verifiability check.
KNOWN_CARDS: dict[str, str] = {
    "DINO": "ocean/experiments/dino.py:53 (DINOConfig)",
    "GYRE": "ocean/fidelity/nemo_recipe.py:347 (nemo_lat_lon_model_config)",
    "LOCK": "ocean/fidelity/nemo_testcase_recipe.py:191 (build_lock_exchange_zco_card)",
    "OVERFLOW": "ocean/fidelity/nemo_testcase_recipe.py:233 (build_overflow_zps_card)",
    "ORCA1": "scripts/run/run_omip_core2.py:1 (CORE-II eORCA1 driver module)",
}

# Genuine non-NEMO fidelity recipe FILES the audit doc names (S-03/S-07/S-07b)
# that are not entries in ``legoesm.ocean.recipes.list_recipes()`` (that
# catalog holds only the scheme-bundle dicts; these are standalone recipe
# modules under ``ocean/fidelity/``) — grep-confirmed to exist. Naming one of
# these in ``selected_by`` DOES legitimize a non-nemo impl (see
# ``_is_legitimately_referenced``): each is a real, separate model's own
# recipe, not a NEMO card falling through to a default.
EXTRA_NON_NEMO_RECIPE_FILES: dict[str, str] = {
    "veros_acc_recipe": "ocean/fidelity/veros_acc_recipe.py",
    "veros_acc_basic_recipe": "ocean/fidelity/veros_acc_basic_recipe.py",
    "veros_global_1deg_recipe": "ocean/fidelity/veros_global_1deg_recipe.py",
    "veros_global_4deg_recipe": "ocean/fidelity/veros_global_4deg_recipe.py",
    "veros_global_flexible_recipe": "ocean/fidelity/veros_global_flexible_recipe.py",
}


class Impl(NamedTuple):
    # Path under the ``legoesm`` namespace (passed to
    # ``tests.legoesm_paths.legoesm_source_path``), e.g. ``"ocean/eos.py"``.
    rel_path: str
    # Name of a ``def``/``class`` AST-resolvable anywhere in that file
    # (top-level or nested — a nested helper like ``_replace_stage_mean``
    # is a legitimate implementation symbol for its own row).
    symbol: str
    # Structured citation — see ``Reference`` above.
    reference: Reference


class BaselineEntry(NamedTuple):
    reason: str   # the audit's note explaining why this branch is tolerated
    kind: str     # one of VALID_BASELINE_KINDS — see module docstring


class RoutineRow(NamedTuple):
    routine_id: str          # matches the doc's S-xx / M-xx row id
    nemo_routine: str        # short label, e.g. "eos_rab", "stp_MLF"
    disposition: str         # SHARED | NEMO_SWITCH | ARTIFICIAL_BRANCH | OTHER_RECIPE | ABSENT | UNVERIFIED | "see S-XX"
    nemo_switch: str         # the namelist/cpp switch(es), or "none" / "n/a" (also carries free-text notes)
    impls: tuple[Impl, ...] = ()


_EOS = "ocean/eos.py"
_DINO = "ocean/experiments/dino.py"
_NEMO_RECIPE = "ocean/fidelity/nemo_recipe.py"
_TESTCASE_RECIPE = "ocean/fidelity/nemo_testcase_recipe.py"
_RESTORING = "ocean/physics/surface_forcing/restoring.py"
_BULK_OMIP = "ocean/bulk_flux_omip.py"
_CONST_VMIX = "ocean/physics/vertical_mixing/constant.py"
_TKE = "ocean/physics/vertical_mixing/tke.py"
_ENH_DIFF = "ocean/physics/convection/enhanced_diffusion.py"
_GM_REDI = "ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py"
_LM_CONFIG = "ocean/physics/lateral_mixing/config.py"
_LCOPS = "ocean/dynamics/latlon_cgrid_operators.py"
_VERTICAL = "ocean/vertical.py"
_OMLC = "ocean/dynamics/ocean_model_latlon_cgrid.py"
_OPL = "ocean/dynamics/ocean_pe_latlon_cgrid.py"
_BLC = "ocean/dynamics/barotropic_latlon_cgrid.py"
_BLC_IMPLICIT = "ocean/dynamics/barotropic_implicit_latlon_cgrid.py"
_RIGID_LID = "ocean/dynamics/rigid_lid_latlon_cgrid.py"
_BAROTROPIC_COMMON = "ocean/dynamics/barotropic_common.py"
_PGF_SMC03 = "ocean/dynamics/pgf_smc03.py"
_PGF_AHH08 = "ocean/dynamics/pgf_ahh08.py"
_BBL = "ocean/physics/bbl_adv.py"
_ADVECTION = "ocean/advection.py"
_SHORTWAVE = "ocean/physics/shortwave_penetration.py"


ROUTINE_REGISTRY: tuple[RoutineRow, ...] = (
    RoutineRow("S-01", "sbc", "NEMO_SWITCH", "ln_usr vs bulk formulae", (
        Impl(_DINO, "dino_step_surface_forcing",
             Reference("nemo", "ln_usr_dino", "stpmlf.F90:170 (ln_usr, DINO usrdef_sbc)")),
        Impl(_RESTORING, "restoring_surface_forcing",
             Reference("nemo", "ln_usr_gyre", "stprk3.F90:138 (ln_usr, GYRE usrdef_sbc)")),
        Impl(_BULK_OMIP, "air_sea_fluxes",
             Reference("nemo", "bulk", "sbc bulk-formulae dispatch (ORCA1 OMIP forcing)")),
    )),
    RoutineRow("S-02", "eos_rab", "NEMO_SWITCH", "ln_teos10 / ln_seos (ORCA1 'wright' = no NEMO arm)", (
        Impl(_EOS, "nemo_roquet_alpha_beta",
             Reference("nemo", "ln_teos10", "eosbn2.F90:490+ (ln_teos10, Roquet et al. polynomial)")),
        Impl(_EOS, "nemo_seos_alpha_beta",
             Reference("nemo", "ln_seos", "eosbn2.F90:490+ (ln_seos, simplified EOS)")),
        Impl(_EOS, "eos_density_derivatives",
             Reference("legoesm_legacy", "generic_derivative_dispatch",
                        "legoESM legacy pre-existing: ocean/eos.py generic finite-difference EOS derivative helper")),
        Impl(_EOS, "wright_eos",
             Reference("paper", "wright",
                        "Wright (1997), An equation of state for use in ocean models, "
                        "J. Atmos. Oceanic Technol. 14:735-741 (ORCA1 default, no NEMO arm)")),
    )),
    RoutineRow("S-03", "bn2", "OTHER_RECIPE", (
        "none (one NEMO routine); reclassified 2026-09-02 (CONFIRMED): "
        "nemo_bn2=NEMO eosbn2.F90, reached by DINO/GYRE/ORCA1; adiabatic=Veros's "
        "own scheme (config.py:310-316); insitu=unreferenced legacy/KPP default"
    ), (
        Impl(_EOS, "compute_buoyancy_frequency_nemo_bn2",
             Reference("nemo", "nemo_bn2", "eosbn2.F90:253-288")),
        Impl(_EOS, "compute_buoyancy_frequency",
             Reference("legoesm_legacy", "insitu",
                        "legoESM legacy pre-existing: ocean/eos.py (unreferenced legacy/KPP default, "
                        "audit CONFIRMED no NEMO/Veros/MITgcm citation)",
                        selected_by=("omip_nemo_match_tripole_v1", "omip_nemo_match_mpas_v1"))),
        Impl(_EOS, "compute_buoyancy_frequency_adiabatic",
             Reference("veros", "adiabatic", "Veros config.py:310-316 (Veros's own N2 scheme)",
                        selected_by=("veros_global_4deg_recipe", "veros_acc_recipe",
                                     "veros_acc_basic_recipe"))),
    )),
    RoutineRow("S-04", "bn2 live-e3w/gdepw geometry", "OTHER_RECIPE", (
        "none; reclassified 2026-09-02 (CONFIRMED gating / PLAUSIBLE no-alt-user): "
        "nemo arm cites eosbn2.F90 (config.py:352-356, DINO+GYRE only); default "
        "arm names no reference and no non-NEMO recipe is known to need it"
    ), (
        Impl(_EOS, "nemo_bn2_live_ladders",
             Reference("nemo", "nemo_bn2_live", "eosbn2.F90:253-258 (config.py:352-356 gate)")),
        Impl(_EOS, "nemo_bn2_depth_ladders",
             Reference("legoesm_legacy", "default_ladder",
                        "legoESM legacy pre-existing: ocean/eos.py (default depth-ladder, unreferenced; "
                        "DINO+GYRE gate to the nemo arm instead)")),
    )),
    RoutineRow("S-05", "zdf_phy -> zdf_cst", "SHARED", "ln_zdfcst", (
        Impl(_CONST_VMIX, "constant_vertical_mixing",
             Reference("nemo", "ln_zdfcst", "zdfphy.F90 (ln_zdfcst dispatch)")),
    )),
    RoutineRow("S-06", "zdf_phy -> zdf_tke", "SHARED", "ln_zdftke", ()),
    RoutineRow("S-07", "zdf_tke solver sub-branch", "OTHER_RECIPE", (
        "none (tke_solver_evaluation); reclassified 2026-09-02 (CONFIRMED): "
        "default=5 Veros fidelity recipes construct TKEConfig untouched, nemo "
        "arm (dino.py:1247-1256)=DINO+GYRE"
    ), (
        Impl(_TKE, "_nemo_literal_tke_solve",
             Reference("nemo", "nemo_tke_solver", "zdftke.F90 (dino.py:1247-1256 gate)")),
        Impl(_TKE, "_solve_tke_backward_euler",
             Reference("legoesm_legacy", "default_tke_solver",
                        "legoESM legacy pre-existing: ocean/physics/vertical_mixing/tke.py "
                        "(Veros-default TKEConfig construction, unreferenced)",
                        selected_by=("veros_acc_recipe", "veros_acc_basic_recipe",
                                     "veros_global_1deg_recipe", "veros_global_4deg_recipe",
                                     "veros_global_flexible_recipe"))),
    )),
    RoutineRow("S-07b", "zdf_tke langmuir sub-branch", "OTHER_RECIPE", (
        "none (tke_langmuir_evaluation); reclassified 2026-09-02 (CONFIRMED): "
        "same Veros-default-vs-DINO/GYRE-nemo-arm split as S-07"
    ), (
        Impl(_TKE, "_nemo_literal_langmuir_operands",
             Reference("nemo", "nemo_langmuir", "zdftke.F90 Langmuir term (dino.py gate)")),
        Impl(_TKE, "nemo_langmuir_tke_source",
             Reference("legoesm_legacy", "default_langmuir",
                        "legoESM legacy pre-existing: ocean/physics/vertical_mixing/tke.py "
                        "(Veros-default construction, unreferenced despite the 'nemo' name)",
                        selected_by=("veros_acc_recipe", "veros_acc_basic_recipe",
                                     "veros_global_1deg_recipe", "veros_global_4deg_recipe",
                                     "veros_global_flexible_recipe"))),
    )),
    RoutineRow("S-08", "zdf_evd", "NEMO_SWITCH", "ln_zdfevd, nn_evdm", (
        Impl(_ENH_DIFF, "enhanced_diffusion_convection",
             Reference("nemo", "ln_zdfevd", "zdfevd.F90:93-120")),
    )),
    RoutineRow("S-09", "ldf_slp", "OTHER_RECIPE", (
        "none (~12 *_evaluation sub-selectors); reclassified 2026-09-02 "
        "(CONFIRMED top-level slope_scheme; PLAUSIBLE for the ~8 untraced "
        "sub-fields): default=5 Veros recipes construct GMRediConfig untouched, "
        "nemo arm (slope_scheme='nemo_iso_lap')=DINO+GYRE. Only one AST-checkable "
        "symbol exists (the nemo override) — the Veros-default arm is the absence "
        "of that override, not a second symbol; not diversity-enforced here."
    ), (
        Impl(_GM_REDI, "compute_nemo_native_slopes",
             Reference("nemo", "nemo_iso_lap",
                        "ldfslp.F90 (nemo_recipe.py:392 gate); Veros-default GMRediConfig has no "
                        "distinct override symbol to compare against")),
    )),
    RoutineRow("S-10", "ldf_tra / ldf_eiv", "NEMO_SWITCH", "nn_aht_ijk_t / nn_aei_ijk_t (Visbeck = no NEMO arm)", (
        Impl(_LM_CONFIG, "TreguierConfig",
             Reference("nemo", "nn_aei_ijk_t=21_treguier", "ldftra.F90:332 (nn_aei_ijk_t=21, Treguier form)")),
        Impl(_LM_CONFIG, "VisbeckConfig",
             Reference("paper", "visbeck",
                        "Visbeck, Marshall, Haine & Spall (1997) eddy diffusivity parameterization "
                        "(no NEMO nn_aei_ijk_t arm)")),
        Impl(_OMLC, "static_kappa_redi_override",
             Reference("nemo", "nn_aht_ijk_t=0_constant", "ldftra.F90:290 (constant scalar kappa, omlc:1388)")),
    )),
    RoutineRow("S-11", "ldf_dyn (lateral viscosity coefficient)", "SHARED", (
        "nn_ahm_ijk_t=0; scalar A_h/A_h_lat_scaling config constant, no "
        "function symbol to check"
    ), ()),
    RoutineRow("S-12", "stp_2D pre-step (Kbb RHS seeding dyn_spg_ts)", "SHARED", (
        "none; COLLAPSED 2026-09-02 (same commit as S-30, which is this defect "
        "at the stprk3_stg altitude). _step_impl now seeds the barotropic solve "
        "the way stp_2D does: ONE Kbb RHS, its depth mean carried in "
        "F_slow_u/F_slow_v, and the BEFORE velocity handed to the solver "
        "(u_star = u0, omlc:4448, transcribing stp2d.F90:177-186 and the "
        "CALL dyn_spg_ts(kt, Kbb, Kbb, ...) at stp2d.F90:280-281). The WS "
        "momentum ladder that used to be evaluated here purely to build that "
        "seed is deleted; the one surviving ladder is S-30's."
    ), (
        Impl(_OMLC, "_step_impl",
             Reference("nemo", "stp_2D_preseed",
                        "stp2d.F90:177-186,280-281 (single Kbb RHS depth mean + BEFORE "
                        "velocity seed; inline in _step_impl at omlc:4448, not an "
                        "AST-distinct symbol)")),
    )),
    RoutineRow("S-13", "dyn_spg_ts external substep loop", "SHARED", "ln_dynspg_ts (cross-oracle: implicit/rigid-lid are non-NEMO)", (
        Impl(_BLC, "barotropic_substeps_latlon_cgrid",
             Reference("nemo", "ln_dynspg_ts", "dynspg_ts.F90:~560-800 (external substep loop)")),
    )),
    RoutineRow("S-14", "dyn_spg_ts external velocity update gate", "ARTIFICIAL_BRANCH", "ln_dynadv_vec .OR. lk_linssh (legoESM gate ANDs in an extra rk3_ws conjunct NEMO's does not have; no current card mis-routed)", (
        Impl(_BLC, "_nemo_flux_form_external_velocity_update",
             Reference("nemo", "ln_dynadv_vec_flux_arm", "dynspg_ts.F90:719-763 (flux-form velocity update)")),
        Impl(_BLC, "substep_body",
             Reference("nemo", "ln_dynadv_vec_vector_arm",
                        "dynspg_ts.F90:719-763 (vector-form update, inline in barotropic_latlon_cgrid.py)")),
    )),
    RoutineRow("S-15", "dyn_spg_ts backward face depth (zhu_bck)", "NEMO_SWITCH", (
        "key_qcoTest_FluxForm; the two NEMO arms coincide algebraically on a "
        "lat-lon C-grid (e1e2t(i,j)==e1e2t(i+1,j)); min_rule = no NEMO arm"
    ), (
        Impl(_BLC, "nemo_ssh_avg_face_depth",
             Reference("nemo", "key_qcoTest_FluxForm", "dynspg_ts.F90:738-747 (zhu_bck, e1e2t-weighted avg)")),
        Impl(_BLC, "_min_rule_face_depths",
             Reference("legoesm_legacy", "min_rule",
                        "legoESM legacy pre-existing: barotropic_latlon_cgrid.py (min_rule, no NEMO arm; "
                        "algebraically coincides with the nemo arm on a lat-lon C-grid)")),
    )),
    RoutineRow("S-16", "dyn_spg_ts continuity/transport/spg", "OTHER_RECIPE", (
        "USER DECISION 2026-09-02: the generic arm is kept as a legitimate "
        "non-NEMO fork (it is the dycore of 19 idealized test-matrix "
        "experiments) and given a real reference instead of collapsing onto "
        "the NEMO-literal arm — see nemo_branch_isomorphism_map.md's dated "
        "addendum under the S-16 section for the evidence."
    ), (
        Impl(_BLC, "nemo_literal_continuity_divergence",
             Reference("nemo", "dyn_spg_ts_continuity", "dynspg_ts.F90:~640-700,~840")),
        Impl(_BLC, "_run_substep_loop",
             Reference("legoesm_legacy", "dyn_spg_ts_continuity_generic",
                        "legoESM legacy: in-house forward-backward split-explicit C-grid "
                        "solver, introduced with no external citation by commit adbb49f83 "
                        "(2026-04-08, 'Add C-grid lat-lon ocean model to fix checkerboard "
                        "instability', #87); its time-averaging and BEBT closure were later "
                        "explicitly modeled on the MOM6/ROMS split-explicit family per their "
                        "own commit messages -- Hallberg (1997) J. Comput. Phys. 135 and "
                        "Shchepetkin & McWilliams (2005) Ocean Modelling 9 (commits "
                        "06f4de939 2026-04-11 and 3d0170d04 2026-04-20, #205) -- but this "
                        "arm is not a literal transcription of either paper.",
                        selected_by=("default_wright_v1", "legoesm_linear_v1",
                                     "legoesm_nemo_like_v1"))),
    )),
    RoutineRow("S-17", "dyn_cor_2D (barotropic Coriolis)", "NEMO_SWITCH", "ln_dynvor_ene/een ('avg'/'frozen' = no NEMO arm)", (
        Impl(_BLC, "een_barotropic_coriolis",
             Reference("nemo", "ln_dynvor_een", "dynspg_ts.F90:359,689 (een barotropic Coriolis)")),
        Impl(_BLC, "barotropic_coriolis_een_pre_step",
             Reference("nemo", "ln_dynvor_een_prestep", "dynspg_ts.F90:359,689 (een pre-step build)")),
    )),
    RoutineRow("S-18", "dom_qco_r3c (r3u/r3v face thickness)", "OTHER_RECIPE", (
        "key_qco; reclassified 2026-09-02 (post-1d6a7448d, CONFIRMED): the "
        "prior 'LOCK/OVERFLOW resolve MIN' finding was itself a misreading, "
        "not a fact 1d6a7448d changed -- both the WS-RK3 stage transport "
        "(_nemo_ws_qco_stage_faces) and the MLF tracer transport already "
        "delegated to the shared kernel "
        "nemo_qco_live_face_geometry_from_operands at audit time (commit "
        "1d6a7448d, 'One NEMO qco face thickness, reached by the RK3 and MLF "
        "lanes alike', calls itself a PURE REFACTOR, arithmetic unchanged); it "
        "unified both call sites onto one adapter symbol, "
        "vertical.nemo_qco_live_face_geometry_cgrid (verified: both call "
        "sites, omlc:5260 included, grep to it on this checkout), and pinned "
        "it with a non-vacuous test. min_cell_to_uface's only "
        "remaining role for this specific quantity is a private, "
        "never-selected test hook (_NEMOWSRK3TestHooks."
        "legacy_stage_min_face_thickness, omlc:1049, set True only inside "
        "scripts/validate/.../nemo_testcase_phase3_stage_sweep_gate.py, never "
        "by a production card); every other min_cell_to_uface call this row "
        "used to cite (omlc:1379-1384 Matsuno-split Coriolis depth-average, "
        "S-27; omlc:1093 the REFERENCE e3u_0 at eta=0, itself NEMO's own "
        "domzgr min-rule per usrdef_zgr.F90:179-186) is a DIFFERENT NEMO "
        "routine, not dom_qco_r3c. What remains a real, cited, separately-"
        "reached arm is min_cell_to_uface's MOM6/MITgcm hFacW=min role in the "
        "PE-lane depth-average/slow-forcing (opl:1364, omlc:4125), used "
        "identically by every lat-lon C-grid card AND by the non-NEMO "
        "recipes (they can never reach the NEMO arm -- it requires raw NEMO "
        "mesh operands their z-coordinates do not carry)."
    ), (
        Impl(_VERTICAL, "nemo_qco_live_face_geometry_cgrid",
             Reference("nemo", "dom_qco_r3c_shared_kernel",
                        "domqco.F90:165-170,219-222 (dom_qco_r3c / dom_qco_r3c_RK3, one shared "
                        "kernel since 1d6a7448d; nemo_qco_live_face_thicknesses, vertical.py:286, "
                        "is now a thin wrapper delegating to "
                        "nemo_qco_live_face_geometry_from_operands, vertical.py:140 -- the same "
                        "kernel this symbol calls -- for the separate S-19/wzv and S-43/GM-Redi "
                        "consumers)")),
        Impl(_LCOPS, "min_cell_to_uface",
             Reference("mitgcm", "min_rule_transport_face",
                        "Adcroft, Hill & Marshall (1997) MOM6/MITgcm hFacW=min convention "
                        "(latlon_cgrid_operators.py:277-284); cited in place at "
                        "ocean_model_latlon_cgrid.py:1379-1384 as commit 1d6a7448d's own "
                        "'real legoESM scheme selection with its own reference ... not a "
                        "defective NEMO transcription'",
                        selected_by=("DINO", "LOCK", "OVERFLOW", "ORCA1",
                                      "veros_faithful_v1", "mitgcm_v1", "oceananigans_v1"))),
    )),
    RoutineRow("S-19", "wzv", "ARTIFICIAL_BRANCH", "none (np_velocity/np_transport arg)", (
        Impl(_VERTICAL, "diagnose_w_from_flux_div",
             Reference("legoesm_legacy", "wzv_generic",
                        "legoESM legacy pre-existing: ocean/vertical.py (generic w diagnostic, "
                        "unreferenced default). MEASURED 2026-09-02 against the NEMO arm on the "
                        "certified L1 cards at the WS-RK3 stage call site (stprk3_stg.F90:297): "
                        "stage w differs by 3.0e-18 m/s (OVERFLOW) / 2.4e-21 m/s (LOCK), i.e. "
                        "1e-15 of |w|, and every kt=1..10 trajectory row is BIT-IDENTICAL. The "
                        "branch is arithmetic association only -- it does NOT own OVERFLOW's "
                        "2.599e-7 kt=2 velocity debt. Numbers + the non-vacuity control (a "
                        "planted +1e-6 on the literal w moves kt=2 T by 5.1e-9) in "
                        "scripts/validate/ocean_fidelity/testcases/nemo_testcase_wzv_arm_probe.py "
                        "and docs/ocean/fidelity/testcases/"
                        "nemo_testcases_l1_wzv_arm_preregister.md",
                        selected_by=("LOCK", "OVERFLOW", "ORCA1",
                                      "veros_faithful_v1", "mitgcm_v1",
                                      "oceananigans_v1"))),
        Impl(_OPL, "nemo_qco_wzv_operands",
             Reference("nemo", "wzv_nemo_literal",
                        "sshwzv.F90:273-387 (qco arm :331-336) + divhor.F90:108-141; the "
                        "np_velocity/np_transport argument is the only NEMO-side branch. "
                        "Constructible on EVERY card since the mesh operands are now resolved by "
                        "vertical.py nemo_qco_resolved_mesh_operands -- NEMO's own hu_0/e1e2*/"
                        "e2u/e1v when the card carries mesh_mask.nc (DINO/GYRE, byte-identical), "
                        "otherwise rebuilt from the card's grid + reference ladder "
                        "(domain.F90:145; usrdef_zgr.F90:179-186). The two remaining reachability "
                        "walls are NEMO's own: zad_qco_evaluation='nemo_literal' gates the MLF/"
                        "dynzad call sites (stpmlf.F90:227,270) and requires "
                        "vertical_momentum_scheme='nemo_advective', which NEMO's RK3 flux-form "
                        "lane does not run (dynadv_up3 + ln_zad_Aimp instead)",
                        selected_by=("DINO", "GYRE"))),
    )),
    RoutineRow("S-20", "wAimp", "SHARED", "ln_zad_Aimp", (
        Impl(_VERTICAL, "nemo_wicker_aimp_partition_transport",
             Reference("nemo", "ln_zad_Aimp",
                       "sshwzv.F90:773-843 (adaptive-implicit w split; OVERFLOW raw "
                       "e3w_0 from usrdef_zgr.F90:157-168)")),
    )),
    RoutineRow("S-21", "stage transport triplet zFu/zFv/zFw", "SHARED", "none (RK3 identity)", (
        Impl(_OMLC, "_nemo_ws_stage_transport",
             Reference("nemo", "rk3_stage_transport_identity", "stprk3_stg.F90:257-277")),
    )),
    RoutineRow("S-22", "dyn_adv dispatch", "NEMO_SWITCH", (
        "ln_dynadv_vec / _cen2 / _up3; dispatch resolved inside the main "
        "baroclinic tendency function, no separate dispatcher symbol"
    ), (
        Impl(_OPL, "latlon_cgrid_ocean_baroclinic_tendencies",
             Reference("nemo", "dyn_adv_dispatch", "dynadv.F90:87-89 (ln_dynadv_vec/_cen2/_up3 dispatch)")),
    )),
    RoutineRow("S-23", "dyn_adv_up3 vertical flux", "SHARED", "ln_dynadv_up3", (
        Impl(_VERTICAL, "nemo_up3_vertical_momentum_advection",
             Reference("nemo", "ln_dynadv_up3", "dynadv_up3.F90:239-365")),
    )),
    RoutineRow("S-24", "dyn_zad", "SHARED", "ln_dynadv_vec", (
        Impl(_VERTICAL, "nemo_advective_vertical_momentum_advection",
             Reference("nemo", "ln_dynadv_vec_zad", "dynzad.F90:86-119")),
    )),
    RoutineRow("S-25", "vertical momentum advection, non-NEMO arms", "OTHER_RECIPE", (
        "none; reclassified 2026-09-02 (CONFIRMED) -- restores the doc's own "
        "ARTIFICIAL_BRANCH disposition (previously mistranscribed here as "
        "ABSENT): upwind_perturbation=unclaimed default (ORCA1 lands here -- "
        "a driver reachability gap, not a duplicate), centered_full=Veros "
        "core/momentum.py, nemo_advective=NEMO dynzad.F90 (see S-24)"
    ), (
        Impl(_VERTICAL, "flux_form_vertical_momentum_advection",
             Reference("legoesm_legacy", "upwind_perturbation",
                        "legoESM legacy pre-existing: ocean/vertical.py (unclaimed default, "
                        "ORCA1 lands here — driver reachability gap)")),
        Impl(_VERTICAL, "flux_form_vertical_momentum_advection_centered",
             Reference("veros", "centered_full", "Veros core/momentum.py (centered vertical momentum advection)",
                        selected_by=("veros_faithful_v1",))),
        Impl(_VERTICAL, "nemo_advective_vertical_momentum_advection",
             Reference("nemo", "ln_dynadv_vec_zad", "dynzad.F90:86-119 (see S-24)")),
    )),
    RoutineRow("S-26", "dyn_vor (vor_ene/vor_ens/vor_een)", "ABSENT (vor_ens missing; al81 = no NEMO arm)", "ln_dynvor_ene/_ens/_een", (
        Impl(_LCOPS, "pv_flux_al81_partial_cell",
             Reference("legoesm_legacy", "al81",
                        "legoESM legacy pre-existing: ocean/dynamics/latlon_cgrid_operators.py "
                        "(al81, no NEMO arm; ORCA1 lands here)")),
        Impl(_LCOPS, "pv_flux_ene",
             Reference("nemo", "ln_dynvor_ene", "dynvor.F90 (vor_ene arm)")),
    )),
    RoutineRow("S-27", "Coriolis time placement", "OTHER_RECIPE", (
        "none (NEMO always inside dyn_vor RHS); reclassified 2026-09-02 "
        "(CONFIRMED): matsuno_split=unclaimed default incl. ORCA1 (live risk "
        "there, inert on LOCK/OVERFLOW at f=0), explicit_ab2=Veros's own scheme "
        "AND NEMO's own Coriolis placement (DINO+GYRE+veros_faithful_v1+"
        "oceananigans_v1+mitgcm_v1)"
    ), (
        Impl(_OMLC, "_forward_backward_coriolis_3d",
             Reference("legoesm_legacy", "matsuno_split",
                        "legoESM legacy pre-existing: ocean_model_latlon_cgrid.py (Matsuno rotation "
                        "substep, unclaimed default incl. ORCA1)")),
        Impl(_OPL, "latlon_cgrid_ocean_baroclinic_tendencies",
             Reference("veros", "explicit_ab2",
                        "Veros-faithful in-RHS Coriolis (docstring 'VEROS-FAITHFUL'; also NEMO's real "
                        "placement per dynspg_ts.F90, cited for GYRE's use)")),
    )),
    RoutineRow("S-28_29", "dyn_hpg -> hpg_sco / ORCA1 PGF", "OTHER_RECIPE", (
        "ln_hpg_sco vs ln_hpg_zps (hpg_zps/zco/djc ABSENT); reclassified "
        "2026-09-02 (CONFIRMED, S-29 component): adcroft cites Adcroft, "
        "Hallberg & Hill 2008 (pgf_ahh08.py; a citation-year mismatch vs "
        "state.py's field comment is a separate, unresolved finding), smc03 "
        "cites Shchepetkin & McWilliams 2003 -- both real, separately-cited, "
        "recipe-consumed papers; the defect is that the ORCA1 driver cannot "
        "even select nemo_sco (--pgf-scheme choices=[adcroft,smc03]), a "
        "reachability gap, not duplicated NEMO work"
    ), (
        Impl(_OPL, "_bc_ke_and_pressure_gradients",
             Reference("nemo", "nemo_sco", "dynhpg.F90:117-123,340-390 (ln_hpg_sco)")),
        Impl(_PGF_SMC03, "compute_pressure_at_target_smc03",
             Reference("paper", "smc03", "Shchepetkin & McWilliams (2003) PGF scheme")),
        Impl(_PGF_AHH08, "column_pressure_integrals_ahh08",
             Reference("mitgcm", "adcroft", "Adcroft, Hallberg & Hill (2008) MITgcm PGF scheme (pgf_ahh08.py)",
                        selected_by=("default_wright_v1", "legoesm_linear_v1"))),
    )),
    RoutineRow("S-30", "stage momentum time-stepping (one WS ladder)", "SHARED", (
        "key_qco; COLLAPSED 2026-09-02 (the audit's rank-3 item). The WS stage "
        "recurrence used to be written TWICE inside _step_impl -- once before "
        "the barotropic solve to build its seed, once after it, corrected -- "
        "which is how a fix to _stage_vertical_up3/live-HPG landed in the "
        "second copy only. The pre-solve copy is deleted (see S-12: NEMO's "
        "stp_2D has no ladder there); the ONE surviving ladder runs after the "
        "barotropic solve at omlc:4817-5025 for every rk3_ws step, and the "
        "private stage_barotropic_correction hook now ablates the per-stage "
        "external-mode replacement inside it instead of selecting a second "
        "recurrence. Still inline code, not an AST-distinct symbol."
    ), (
        Impl(_OMLC, "_step_impl",
             Reference("nemo", "ws_stage_ladder",
                        "stprk3_stg.F90:344-374 (single post-barotropic ladder, inline in "
                        "_step_impl at omlc:4817-5025, not AST-distinct)")),
    )),
    RoutineRow("S-31", "stage 2/3 eos(Kmm)+dyn_hpg(Kmm) operands", "SHARED", "none", (
        Impl(_OMLC, "_stage_hpg_operands",
             Reference("nemo", "stage23_hpg_operands", "stprk3_stg.F90:317-320")),
    )),
    RoutineRow("S-32", "dyn_ldf -> ldf_lap (viscosity operator form)", "OTHER_RECIPE", (
        "ln_dynldf_lap (one NEMO arm, 3 legoESM operator forms); reclassified "
        "2026-09-02 (CONFIRMED): nemo_div_curl=DINO+GYRE (NEMO "
        "dyn_ldf_lev_lap, dino.py:914), flux_divergence=Veros core/friction.py "
        "harmonic_friction, vector_laplacian (ORCA1's default)=unreferenced"
    ), (
        Impl(_LCOPS, "nemo_ldf_lap_viscosity_cgrid",
             Reference("nemo", "nemo_div_curl", "dino.py:914 (NEMO dyn_ldf_lev_lap)")),
        Impl(_LCOPS, "vector_laplacian_dissipation_cgrid",
             Reference("legoesm_legacy", "vector_laplacian",
                        "legoESM legacy pre-existing: ocean/dynamics/latlon_cgrid_operators.py "
                        "(unclaimed default incl. ORCA1)")),
        Impl(_LCOPS, "flux_divergence_viscosity_cgrid",
             Reference("veros", "flux_divergence", "Veros core/friction.py harmonic_friction",
                        selected_by=("veros_faithful_v1", "oceananigans_v1", "mitgcm_v1"))),
    )),
    RoutineRow("S-33_34", "dyn_zdf / e3w(Kmm) divisor", "OTHER_RECIPE", (
        "S-34 IS NOW SHARED (collapsed 2026-09-02): NEMO's e3w(Kmm) divisor "
        "is no longer a selectable arm. The `implicit_vmix_e3t_now_divisor` "
        "flag is DELETED and the divisor is UNBRANCHED inside the NEMO "
        "identity `zdf_implicit_solver_evaluation=\"nemo_literal\"`, produced "
        "by ONE canonical symbol -- `physics/vertical_mixing/implicit_solver."
        "py::nemo_e3w_kmm` -- which BOTH the tracer solve and the momentum "
        "solve call (e3uw_0 == e3w_0, zgr_lib.F90:111-112). Every card that "
        "runs the routine (DINO nemo_dino_kamm_mlf, LOCK_EXCHANGE, OVERFLOW, "
        "GYRE) reaches that one implementation; the hidden-default defect "
        "(certified DINO card silently on the unreferenced legacy midpoint) is "
        "GONE. Receipt: docs/ocean/fidelity/dino_zdf_divisor_arm_receipt.md. "
        "The row stays OTHER_RECIPE because it also covers S-33, whose "
        "`shared_thomas`/`nemo_literal` solver-evaluation fork is unchanged "
        "and is a genuine cross-recipe fork (ORCA1 + every catalog recipe on "
        "shared_thomas). Two arms survive at the divisor site and are NOT "
        "NEMO's: the Veros dzw slot (`implicit_vmix_dzw_slot`, cited to Veros "
        "thermodynamics.py:267, selected by veros_faithful_v1) and the legacy "
        "midpoint default (unclaimed default incl. ORCA1) -- both reached only "
        "OFF the NEMO identity. Still only one AST-checkable symbol "
        "(omlc:7537's `_apply_implicit_vertical_mixing` encloses both sites), "
        "so the residual risk noted before persists for S-33; not "
        "diversity-enforced here."
    ), (
        Impl(_OMLC, "_apply_implicit_vertical_mixing",
             Reference("nemo", "zdf_implicit_program",
                        "trazdf.F90:219-221 / dynzdf.F90:182-195 (S-34's divisor is now the "
                        "single shared nemo_e3w_kmm; OVERFLOW supplies usrdef_zgr.F90:157-168's "
                        "raw 20 m W ladder and domzgr_substitute.h90:131-133 T/U/V stretch; "
                        "the remaining fork on this symbol is "
                        "S-33's shared_thomas/nemo_literal solver evaluation, omlc:7537, "
                        "not AST-distinct — see docstring)")),
    )),
    RoutineRow("S-35", "stage-3 zub barotropic correction", "ARTIFICIAL_BRANCH", "none (NEMO: dyn_zdf then zub, one site)", (
        Impl(_OMLC, "_replace_stage_mean",
             Reference("nemo", "zub_before_dynzdf", "stprk3_stg.F90:433-446 (site a, before implicit solve)")),
        Impl(_OMLC, "_fixed_depth_means",
             Reference("nemo", "zub_after_dynzdf",
                        "stprk3_stg.F90:433-446 (site b, after implicit solve, "
                        "nemo_stage_mean_imposition gate)")),
    )),
    RoutineRow("S-36", "tra_adv_trp (reuse momentum zF triplet)", "SHARED", (
        "none; private var _nemo_ws_live_stage_geometry + "
        "kmm_tracer_transports config flag (omlc:4213-5240), not a separate "
        "function symbol"
    ), ()),
    RoutineRow("S-37", "tra_adv dispatch", "ARTIFICIAL_BRANCH", "ln_traadv_fct (SHARED for D/L/O/G; ORCA1 runs superbee, a Veros scheme, no NEMO arm)", (
        Impl(_OMLC, "_nemo_ws_rk3_tracer_pair_step",
             Reference("nemo", "ln_traadv_fct_stage01", "traadv.F90:355-364 (stage 0-1, centered)")),
        Impl(_OMLC, "compute_advection_flux_div_pair",
             Reference("nemo", "ln_traadv_fct_flux_div", "traadv.F90 (flux-divergence pair helper)")),
        Impl(_ADVECTION, "fct_tracer_advection",
             Reference("nemo", "ln_traadv_fct_stage2", "traadv.F90:279-283 (stage 2 FCT, ll_dofct)")),
        Impl(_ADVECTION, "_veros_superbee_face_flux",
             Reference("veros", "superbee", "Veros superbee face-flux scheme (ORCA1 default, ABSENT NEMO arm)")),
    )),
    RoutineRow("S-38", "tra_adv_fct implicit-w treatment", "UNVERIFIED", "nn_fct_imp", (
        Impl(_ADVECTION, "fct_tracer_advection",
             Reference("unclassified", "nn_fct_imp",
                        "traadv_fct.F90:143,439-453 (ll_zAimp1 vs legoESM's nemo_rk3_two_step + "
                        "fct_implicit_w; not traced line-by-line, audit disposition UNVERIFIED)")),
    )),
    RoutineRow("S-39", "tracer stage time-stepping (qco (1+r3t) weighting)", "SHARED", "key_qco", (
        Impl(_OMLC, "_nemo_ws_rk3_tracer_pair_step",
             Reference("nemo", "key_qco_tracer_stage_weighting", "stprk3_stg.F90:540-560")),
    )),
    RoutineRow("S-40", "tra_sbc placement (DINO-scoped selector)", "OTHER_RECIPE", (
        "none (nn_fsbc; RK3 cards have no equivalent knob); reclassified "
        "2026-09-02 (CONFIRMED): leapfrog_rhs=NEMO tra_sbc Nnn-RHS placement "
        "(certified MLF card; run_dino.py:625-634 hard-forbids the bad "
        "leapfrog+applied_now pairing), applied_now=DINO's non-MLF oracle "
        "sub-recipes (veros/mitgcm/oceananigans variants). Only one "
        "AST-checkable symbol exists (the selector function); not "
        "diversity-enforced here."
    ), (
        Impl(_DINO, "_check_surface_tendency_placement",
             Reference("nemo", "tra_sbc_placement_multi_arm",
                        "trasbc.F90 Nnn-RHS placement (+applied_now legacy default; both branches "
                        "inside one selector, not AST-distinct — see docstring)")),
    )),
    RoutineRow("S-41", "tra_qsr", "NEMO_SWITCH", "ln_qsr_2bd / ln_qsr_rgb", (
        Impl(_SHORTWAVE, "apply_shortwave_penetration",
             Reference("nemo", "ln_qsr_2bd_or_rgb", "traqsr.F90 (ln_qsr_2bd/ln_qsr_rgb)")),
    )),
    RoutineRow("S-42", "bbl + tra_bbl", "ARTIFICIAL_BRANCH", "ln_trabbl, nn_bbl_adv=2 (one NEMO program, ONE transcription, two composition SITES)", (
        Impl(_BBL, "apply_bbl_adv_tendency",
             Reference("nemo", "trabbl_bbl_exchange",
                        "trabbl.F90:129-136,243-284 -- the single transcription of the 3-leg "
                        "exchange; in-stage site (WS-RK3 stage-3 tracer RHS, OVERFLOW)")),
        Impl(_BBL, "apply_bbl_adv_step",
             Reference("nemo", "trabbl_bbl_exchange",
                        "trabbl.F90:129-136,243-284 -- CALLS the same bbl_transports + "
                        "apply_bbl_adv_tendency (no arithmetic of its own since 2026-09-02); what "
                        "remains is the host post-step forward-Euler PLACEMENT, ORCA1")),
    )),
    RoutineRow("S-43", "tra_ldf", "see S-09", "ln_traldf_lap+ln_traldf_iso", (
        Impl(_GM_REDI, "compute_nemo_native_slopes",
             Reference("nemo", "nemo_iso_lap", "ldfslp.F90 / traldf_iso.F90 (ln_traldf_lap+ln_traldf_iso; see S-09)")),
    )),
    RoutineRow("S-44", "tra_zdf", "see S-33_34", "always", (
        Impl(_OMLC, "_apply_implicit_vertical_mixing",
             Reference("nemo", "e3w_kmm_divisor_multi_arm", "trazdf.F90 (always; see S-33_34)")),
    )),
    RoutineRow("S-45", "tra_npc", "ABSENT", "ln_zdfnpc", ()),
    # 2026-09-02 SCOPE FIX (see the map doc's "S-46" section): ONE legoESM
    # implementation of the UP3 T-point upwind selector, reached by TWO
    # reference arms that are named by the public
    # ``momentum_flux_scheme`` value -- so the arm follows the reference the
    # caller CLAIMS, not the time integrator it happens to use.  The bare
    # "upwind3" (which served both references and resolved by
    # ``momentum_time_integrator``) is removed from
    # VALID_MOMENTUM_FLUX_SCHEME, so a caller that names no reference fails
    # validation.  Both Impl rows point at the SAME symbol on purpose: that
    # sharing IS the isomorphism result for this row.
    RoutineRow("S-46", "dyn_adv_up3 T-point upwind selector", "SHARED", (
        "none (one NEMO routine); the two Impl rows are two REFERENCE arms of "
        "ONE legoESM implementation, selected by momentum_flux_scheme "
        "('nemo_up3' / 'oceananigans_up3'). Private ablation hook only: "
        "_NEMOWSRK3TestHooks.legacy_up3_transport_sign_selector. "
        "2026-09-03: the SELECTOR itself is integrator-agnostic, but public "
        "construction of momentum_flux_scheme='nemo_up3' now REQUIRES "
        "momentum_time_integrator='rk3_ws' -- its only face-thickness-faithful "
        "home is the WS-RK3 stage program (see S-47); _validate_config refuses "
        "the pairing elsewhere (tests/ocean/unit/test_config_footguns.py)"
    ), (
        Impl(_OPL, "_up3_reconstruct",
             Reference("nemo", "ln_dynadv_up3",
                        "dynadv_up3.F90:166,169-170 (zui = puu(ji)+puu(ji+1): the "
                        "T-point along-flow fluxes select the upwind curvature by "
                        "the ADVECTED-VELOCITY pair; magnitude by the transport "
                        "pair :176; F-point :179-187 and vertical :294-295 by the "
                        "transport pair)",
                        selected_by=("LOCK", "OVERFLOW"))),
        Impl(_OPL, "_up3_reconstruct",
             Reference("oceananigans", "oceananigans_up3",
                        "Oceananigans upwind_biased_advective_fluxes.jl:18-24 "
                        "(u~ = symmetric_interpolate(Ax_q, U) -- the TRANSPORT -- "
                        "then upwind_biased_product, so every flux family selects "
                        "by the transport pair); Silvestri et al. 2024 'UP3'",
                        selected_by=("oceananigans_v1",))),
    )),
    RoutineRow("S-47", "dyn_adv_up3 face thickness e3u(Kmm)", "SHARED", (
        "none (one macro; RK3 identity). 2026-09-03: because this fix is wired "
        "only inside the momentum_time_integrator='rk3_ws' stage program, "
        "_validate_config now refuses momentum_flux_scheme='nemo_up3' on any "
        "other momentum_time_integrator -- without this stage pair, nemo_up3 "
        "falls back to the legacy min-of-stretched-T-thickness rule this row "
        "measured as first-order wrong, a pairing NEMO never runs (see S-46; "
        "tests/ocean/unit/test_config_footguns.py)"
    ), (
        Impl(_OPL, "_bc_horizontal_momentum_advection_flux_form",
             Reference("nemo", "dyn_adv_up3_e3u_kmm",
                       "dynadv_up3.F90:160,205-207; domzgr_substitute.h90:127; the WS-RK3 "
                       "stage pair from _nemo_ws_qco_stage_faces via momentum_flux_face_thickness "
                       "(tests/ocean/unit/test_nemo_ws_hadv_face_thickness.py)")),
    )),
    RoutineRow("S-48", "stprk3_stg stage velocity umask rank", "SHARED",
               "none (one array; RK3 identity)", (
        Impl(_OMLC, "_replace_stage_mean",
             Reference("nemo", "stprk3_stg_stage_umask",
                       "stprk3_stg.F90:367,375,382 (stage-1/2 velocity update) and "
                       ":444 (barotropic correction) carry umask(ji,jj,jk); "
                       "dynadv_up3.F90:142-143,160,166-176 read those zeros "
                       "(tests/ocean/unit/test_nemo_ws_stage_face_mask_rank.py)")),
        Impl(_OMLC, "_mom_pert_ws",
             Reference("nemo", "stprk3_stg_transport_umask",
                       "stprk3_stg.F90:273-274: the SAME barotropic correction is "
                       "masked inside the advective transport dyn_adv_up3 consumes, "
                       "zFu = e2u*e3u(Kmm)*( uu(Kmm) + zub*umask(ji,jj,jk) )")),
    )),
    RoutineRow("S-49", "zdf_drg/dyn_drg/dyn_zdf bottom-drag composition",
               "SHARED", "ln_non_lin + ln_drgimp + ln_dynspg_ts", (
        Impl(_OPL, "nemo_bottom_drag_rate_faces",
             Reference("nemo", "nonlinear_implicit_split_explicit_drag",
                       "zdfdrg.F90:138-190; dynspg_ts.F90:699-705,1584-1643; "
                       "dynzdf.F90:148-160,293-305 (selected as the inseparable "
                       "zdf_drag_in_matrix+zdf_baroclinic_only+"
                       "barotropic_drag_substep identity)",
                       selected_by=("DINO", "GYRE"))),
    )),
    RoutineRow("S-50", "hpg_sco literal operands and recurrence", "SHARED",
               "ln_hpg_sco", (
        Impl(_EOS, "nemo_roquet_density_anomaly_ratio",
             Reference("nemo", "hpg_sco_density_operand",
                       "eosbn2.F90:265-288: source-associated prd = zn*r1_rho0-1",
                       selected_by=("DINO", "GYRE"))),
        Impl(_LCOPS, "nemo_hpg_sco_literal_cgrid",
             Reference("nemo", "hpg_sco_recurrence",
                       "dynhpg.F90:340-390: bottom-up e3w(Kmm) trapezoid and "
                       "native U/V face-gradient recurrence",
                       selected_by=("DINO", "GYRE"))),
    )),
    RoutineRow("S-51", "sbc U/V coastal surface-stress factors", "SHARED",
               "none", (
        Impl(_OPL, "surface_stress_faces",
             Reference("nemo", "sbcmod_coastal_stress_factors",
                       "sbcmod.F90:539-546: face average followed by "
                       "(2-umask)*MAX(adjacent tmask); private legacy hook "
                       "is a gate-only ablation",
                       selected_by=("DINO", "GYRE", "ORCA1"))),
    )),
    RoutineRow("M-01", "stp_MLF whole-step composition", "ARTIFICIAL_BRANCH", "absence of key_RK3 (nemo_mlf selected by no card)", (
        Impl(_OMLC, "_leapfrog_step",
             Reference("nemo", "stp_MLF", "stpmlf.F90:108-473 (two _step_impl passes)")),
        Impl(_OMLC, "_nemo_mlf_step",
             Reference("nemo", "stp_MLF",
                        "stpmlf.F90:108-473 (single-pass transcription; also cites a spec doc; "
                        "tests/ocean/unit/test_nemo_mlf_step_transcription.py)")),
    )),
    RoutineRow("M-02", "ssh_nxt (+div_hor)", "SHARED", "none", (
        Impl(_BLC, "barotropic_substeps_latlon_cgrid",
             Reference("nemo", "ssh_nxt_div_hor", "sshwzv.F90; stpmlf.F90:214")),
    )),
    RoutineRow("M-03", "ssh_atf (Asselin filter)", "SHARED", "rn_atfp", (
        Impl(_OMLC, "_thickness_weighted_asselin",
             Reference("nemo", "rn_atfp", "sshwzv.F90; stpmlf.F90:316 (Asselin filter)")),
    )),
    RoutineRow("M-04", "tra_atf_qco / dyn_atf_qco (tracer_combine)", "ARTIFICIAL_BRANCH", "key_qco ('concentration' = no NEMO arm)", (
        Impl(_OMLC, "thickness_weighted_tracer_combine",
             Reference("legoesm_legacy", "concentration",
                        "legoESM legacy pre-existing: ocean_model_latlon_cgrid.py (concentration form, "
                        "no NEMO arm under key_qco; drifts heat +8.6e-6)")),
        Impl(_OMLC, "thickness_weighted_tracer_content",
             Reference("nemo", "thickness_weighted",
                        "traatfqco.F90/dynatfqco.F90; stpmlf.F90:394-395 (key_qco thickness-weighted form)")),
    )),
    RoutineRow("M-05", "mlf_baro_corr", "ARTIFICIAL_BRANCH", "ln_dynspg_ts ('off' = no NEMO arm)", (
        Impl(_BAROTROPIC_COMMON, "after_level_column_mean_reconcile",
             Reference("legoesm_legacy", "off",
                        "legoESM legacy pre-existing: ocean/dynamics/barotropic_common.py (off default, "
                        "no NEMO arm for an MLF+dynspg_ts card)")),
        Impl(_BAROTROPIC_COMMON, "nemo_literal_after_level_reconcile",
             Reference("nemo", "nemo_mlf_baro_corr", "stpmlf.F90:392 (ln_dynspg_ts mlf_baro_corr)")),
    )),
    RoutineRow("M-06", "dyn_ldf/tra_ldf at Kbb inside one pass", "see M-01", "none", (
        Impl(_OMLC, "_leapfrog_step",
             Reference("nemo", "stp_MLF", "stpmlf.F90:250,368 (two-pass tendency pipeline; see M-01)")),
        Impl(_OMLC, "_nemo_mlf_step",
             Reference("nemo", "stp_MLF", "stpmlf.F90:250,368 (_ldf_state= single pass; see M-01)")),
    )),
    RoutineRow("CARD-ASSEMBLY", "card assembly (three assemblers, one RK3 identity)", "ARTIFICIAL_BRANCH", "n/a — legoESM-side; NEMO has one namelist per configuration", (
        Impl(_TESTCASE_RECIPE, "_model_config",
             Reference("legoesm_legacy", "testcase_bare_from_flat",
                        "legoESM legacy pre-existing: ocean/fidelity/nemo_testcase_recipe.py "
                        "(bare from_flat() assembler, LOCK/OVERFLOW)")),
        Impl(_NEMO_RECIPE, "nemo_lat_lon_model_config",
             Reference("nemo", "nemo_recipe_assembler",
                        "n/a — legoESM-side card assembler matching NEMO's namelist structure (GYRE)")),
        Impl(_DINO, "dino_lat_lon_model_config",
             Reference("nemo", "dino_recipe_assembler",
                        "n/a — legoESM-side card assembler matching NEMO's namelist structure (DINO)")),
    )),
)


# Shrink-only allow-list: routine_id -> the audit's reason this artificial
# branch is currently tolerated. A row may be listed ONLY while
# ``_row_needs_baseline_entry`` (test_no_scheme_duplication.py) says it
# genuinely needs one — an ARTIFICIAL_BRANCH row with >=2 distinct impls, a
# reference-duplicate group, or an untriaged (``unclassified``) reference.
# Once a collapse PR lands (see the doc's "collapse plan" per row), or every
# impl's reference is fully triaged, DELETE the entry in that same PR. A row
# reclassified "other_recipe" moves its rationale into the registry row's own
# ``nemo_switch`` text and its disposition to "OTHER_RECIPE" instead — it is
# no longer a baseline exception (see the 2026-09-02 reclassification note in
# the module docstring), UNLESS one of its impls is still ``unclassified``
# (S-38) or its impls still form a same-(model,arm) duplicate group.
ARTIFICIAL_BRANCH_BASELINE: dict[str, BaselineEntry] = {
    "S-02": BaselineEntry(
        reason=(
            'eos_rab: eos_density_derivatives (generic finite-difference derivative dispatch, '
            'arm=generic_derivative_dispatch) has no recipe naming it in selected_by -- not one '
            'of the 19 rows the 2026-09-02 reclassification pass audited, so this is genuinely '
            'untriaged, not a confirmed non-issue: unknown whether any recipe actually needs this '
            'arm distinctly from nemo_roquet/nemo_seos/wright.'
        ),
        kind='unclassified',
    ),
    "S-04": BaselineEntry(
        reason=(
            "bn2 live-e3w/gdepw geometry: the doc's own OTHER_RECIPE note says 'no Veros/MITgcm/"
            "Oceananigans recipe touches tke_n2_evaluation_stage at all' for the default_ladder "
            'arm -- genuinely no non-NEMO recipe selects it away from nemo_bn2_live_ladders '
            '(DINO+GYRE). Kept OTHER_RECIPE per the audit (a NEMO-only opt-in switch, not a '
            'driver reachability gap), but the disposition-independent unreferenced-arm rule '
            'still requires this entry since nothing legitimizes the default arm.'
        ),
        kind='unclassified',
    ),
    "S-15": BaselineEntry(
        reason=(
            'dom_qco_r3c backward face depth (zhu_bck): _min_rule_face_depths (min_rule) has no '
            'recipe in selected_by -- not one of the 19 rows the 2026-09-02 pass audited. The two '
            'NEMO arms coincide algebraically on a lat-lon C-grid, so this is lower-risk than '
            'S-18/S-19, but genuinely untriaged: unknown whether any recipe needs min_rule '
            'distinctly from nemo_ssh_avg_face_depth.'
        ),
        kind='unclassified',
    ),
    "S-25": BaselineEntry(
        reason=(
            "vertical momentum advection, non-NEMO arms: upwind_perturbation is the doc's own "
            "'unclaimed default, run by most catalog recipes + ORCA1 (no override)' -- no recipe "
            'deliberately selects it away from nemo_advective_vertical_momentum_advection. '
            'centered_full IS properly referenced (veros_faithful_v1); this entry covers only the '
            'unreferenced default arm the disposition-independent rule still flags.'
        ),
        kind='unclassified',
    ),
    "S-26": BaselineEntry(
        reason=(
            'dyn_vor: pv_flux_al81_partial_cell (al81) has no recipe in selected_by -- not one of '
            "the 19 rows the 2026-09-02 pass audited; the row's own note says only 'ORCA1 lands "
            "here', and ORCA1 is a NEMO-fidelity card, not a distinct recipe. Genuinely untriaged."
        ),
        kind='unclassified',
    ),
    "S-32": BaselineEntry(
        reason=(
            "dyn_ldf -> ldf_lap: vector_laplacian_dissipation_cgrid (vector_laplacian) is the "
            "doc's own 'unclaimed default incl. ORCA1' / 'unreferenced' arm -- no recipe "
            'deliberately selects it away from nemo_div_curl. flux_divergence IS properly '
            'referenced (veros_faithful_v1, oceananigans_v1, mitgcm_v1); this entry covers only '
            'the unreferenced legacy default the disposition-independent rule still flags.'
        ),
        kind='unclassified',
    ),
    "S-14": BaselineEntry(
        reason=(
            'dyn_spg_ts external velocity update gate carries an extra '
            "momentum_time_integrator=='rk3_ws' conjunct NEMO's own gate (dynspg_ts.F90:719) "
            'does not have — latent mis-route for any future MLF+flux-form card; no current '
            'card affected.'
        ),
        kind='unclassified',
    ),
    "S-19": BaselineEntry(
        reason=(
            'wzv: one NEMO routine (sshwzv.F90), two legoESM impls; default is the non-NEMO '
            '(generic) one. Reclassified 2026-09-02: NEMO_DUPLICATE, CONFIRMED — same '
            'LOCK/OVERFLOW-reach correction as S-18. COLLAPSE REFUSED 2026-09-02 (round 2): '
            'routing the WS-RK3 stage wzv through the literal arm leaves LOCK byte-identical '
            'but moves OVERFLOW kt=8/9/10 before.T by 1.2/1.6/1.6 ulp, and ulp_move_gate '
            'holds TRACER rows to BIT-IDENTITY, not to MAX_ULP_MOVE — so the gate goes red '
            'and nothing was landed. The earlier "every kt=1..10 row BIT-IDENTICAL" claim in '
            'the map is RETRACTED for OVERFLOW at this HEAD. diagnose_w_from_flux_div also '
            'cannot be deleted: MPAS (ocean_pe_mpas.py:312, ocean_model_mpas.py:980), the '
            'C-D grid (ocean_pe_cdgrid.py:361) and the lat-lon PE lane '
            '(ocean_pe_latlon_cgrid.py:1369) all call it, and NEMO cards themselves reach it '
            'at the MLF call-2 site because they resolve wzv_call2_evaluation="generic" — so '
            'the row cannot become OTHER_RECIPE either.'
        ),
        kind='nemo_duplicate',
    ),
    "S-35": BaselineEntry(
        reason=(
            'stage-3 zub barotropic correction: NEMO does dyn_zdf then zub, one site '
            '(stprk3_stg.F90:430,433-446); L/O apply the correction only BEFORE the implicit '
            'solve (wrong side), GYRE has both sites. Reclassified 2026-09-02: NEMO_DUPLICATE, '
            'CONFIRMED.'
        ),
        kind='nemo_duplicate',
    ),
    "S-37": BaselineEntry(
        reason=(
            'tra_adv: ORCA1 runs superbee (Veros), which maps to no NEMO arm '
            '(traadv_mus/_ubs/_qck are ABSENT).'
        ),
        kind='unclassified',
    ),
    "S-38": BaselineEntry(
        reason=(
            'tra_adv_fct implicit-w treatment: nn_fct_imp -> ll_zAimp1 (traadv_fct.F90:143,'
            '439-453) not traced line-by-line against legoESM\'s nemo_rk3_two_step + '
            'fct_implicit_w. Audit disposition UNVERIFIED — genuinely untriaged, not yet a '
            'confirmed defect or a confirmed non-issue.'
        ),
        kind='unclassified',
    ),
    "S-42": BaselineEntry(
        reason=(
            'bbl: ONE transcription of trabbl.F90:243-284 (bbl_transports + '
            'apply_bbl_adv_tendency), reached from two composition SITES. OVERFLOW folds it '
            "into the stage-3 tracer RHS (NEMO's own site, stprk3_stg.F90:468,498,588); the "
            'OMIP driver calls apply_bbl_adv_step, which since 2026-09-02 adds no arithmetic '
            'of its own beyond one forward-Euler update -- it calls the same two operators. '
            'It does feed them the REFERENCE ladder (geom.h_ref/dep_bot) where the in-model '
            'site feeds live stage thickness and live bottom depth '
            '(ocean_model_latlon_cgrid.py:1290-1298), an O(eta/H) ~ 3e-4 operand difference. '
            'The 0.25*V/dt transport cap NEMO has none of is DELETED on Rule 9 alone (NOT on '
            'a recipe argument -- recipes.py has zero BBL mentions, and ~30 committed OMIP '
            'decks DID select the capped path via --bbl-adv; what carries the deletion is '
            'that trabbl.F90:243-284 clamps neither transport). Measured inert: the exchange '
            'fraction reduces to g*gamma*(drho/rho0)*dt/e1t, max 0.048 on ORCA1 metrics at a '
            'Denmark-Strait-exceeding contrast against a 0.25 cap; non-vacuously live on a '
            'tiny-area face (tests/ocean/unit/test_bbl_adv.py). What is left is a PLACEMENT '
            'branch, and it is not config-flippable: the in-model BBL hook exists only in the '
            'WS-RK3 tracer lane (ocean_model_latlon_cgrid.py:6007), while ORCA1 runs the '
            'forward-Euler tracer lane (the config default, state.py:2071) -- which itself '
            'has no NEMO arm. Setting bbl_adv_option=2 on ORCA1 would have SILENTLY run NO '
            'BBL. 2026-09-02 USER DECISION: the pair is now cross-checked at construction -- '
            'LatLonCGridOceanModel._validate_config (ocean_model_latlon_cgrid.py:3342-3361) '
            'raises ValueError naming the field and the mismatched lane whenever '
            'bbl_adv_option=2 is paired with a tracer_time_integrator other than rk3_ws, and '
            'points at the driver-side apply_bbl_adv_step (run_omip_core2.py, --bbl-adv) as '
            'the alternative. Verified inert on production: run_omip_core2.py and every '
            'committed sbatch never set bbl_adv_option (grep, zero hits), so ORCA1 stays at '
            'the config default (0) and never trips the new guard. Non-vacuity: '
            'tests/ocean/unit/test_config_footguns.py::'
            'test_validate_config_rejects_bbl_in_stage_off_the_rk3_ws_lane. Collapsing the '
            'placement itself remains a separate, second ASK, not a flip.'
        ),
        kind='unclassified',
    ),
    "M-01": BaselineEntry(
        reason=(
            'RANK 5. stp_MLF: two implementations. _nemo_mlf_step (single '
            'dyn_ldf(Kbb,Kmm) pass, structurally faithful to stpmlf.F90:275,437) is selected '
            'by no card; _leapfrog_step (two _step_impl passes) is what the certified DINO '
            'card runs. Reclassified 2026-09-02: NEMO_DUPLICATE, CONFIRMED — (b) has real '
            'committed test coverage (test_nemo_mlf_step_transcription.py), so it is '
            'validated-but-unpromoted, not dead code / ORPHAN. MEASURED 2026-09-02 on the '
            'certified card (mlf_step_mechanism_ab.py, fp64, both arms on the SAME state '
            'under the SAME jit, identity control 0.0): the two DIFFER from step 1 in every '
            'prognostic field -- T 4.963e-4 degC, S 4.192e-5 PSU, u 5.196e-6 m/s, '
            'v 9.568e-6 m/s, eta 5.725e-6 m, i.e. 8.6e10 ulp of field scale against a 2-ulp '
            'collapse bar. So this row is a measured DIFFERENCE, not a re-association, and '
            'the card is NOT repointed. Ablating GM/Redi cuts the TRACER gap ~2900x '
            '(confirming the transcription test s predicted mechanism) and leaves the '
            'MOMENTUM gap bit-unchanged, which that test attributes to XLA fusion noise -- '
            'unsettled, recorded rather than folded in. Independently, nemo_mlf hard-requires '
            'implicit_vmix_e3t_now_divisor=True (ocean_model_latlon_cgrid.py:3019) which the '
            'DINO card sets False, so a config-level repoint would move S-34 at the same '
            'time.'
        ),
        kind='nemo_duplicate',
    ),
    "M-04": BaselineEntry(
        reason=(
            "tra_atf_qco/dyn_atf_qco tracer_combine: 'concentration' (the default, resolved "
            'by the certified DINO card) has no NEMO arm under key_qco and drifts heat by '
            "+8.6e-6 vs NEMO's +3.4e-16."
        ),
        kind='unclassified',
    ),
    "M-05": BaselineEntry(
        reason=(
            "mlf_baro_corr: 'off' (barotropic_after_reconcile default) has no NEMO arm for an "
            'MLF+dynspg_ts card; DINO itself is on the faithful value, but the default is the '
            'non-NEMO one.'
        ),
        kind='unclassified',
    ),
    "CARD-ASSEMBLY": BaselineEntry(
        reason=(
            'Three separate card-assembly paths for one RK3/MLF identity: LOCK/OVERFLOW build '
            'via a bare from_flat() that silently keeps 6 non-NEMO defaults (al81, '
            'matsuno_split, barotropic_coriolis=avg, een_seed=window_start, '
            'zad_qco_evaluation=generic, wzv_call2_evaluation=generic) the GYRE/DINO '
            'assemblers would set to the NEMO arm.'
        ),
        kind='unclassified',
    ),
}
