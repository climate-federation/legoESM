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
    that literal text here too, e.g. ``"see S-09"``.
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

``ARTIFICIAL_BRANCH_BASELINE`` is the SHRINK-ONLY allow-list: a routine_id may
appear here only if its registry row is ``disposition="ARTIFICIAL_BRANCH"``
AND has >=2 *distinct* ``impls`` right now. The reason string is lifted from
the audit's own note / ranked-branches section. Collapsing a branch (the
audit's collapse plans) means the fix reduces that row's distinct impls to
<=1 and this dict's entry must be deleted in the SAME PR (stale-entry check).

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
   independent reclassification (table reproduced in the doc, §6) found 8
   genuine ``nemo_duplicate`` defects (S-12, S-16, S-18, S-19, S-30, S-35,
   S-42, M-01 — kept ``ARTIFICIAL_BRANCH``, baselined below where >=2 distinct
   AST symbols exist) and 11 ``other_recipe`` rows (S-03, S-04, S-07, S-09,
   S-25, S-27, S-28_29's S-29 component, S-32, S-33_34, S-40 — moved to
   disposition ``OTHER_RECIPE`` and OUT of the baseline dict, since the
   "duplicate" arm is a legitimately different, separately-cited reference;
   the registry's ``nemo_switch`` field now names each arm's reference).
   0 rows classified ``orphan``. No row's classification is contested between
   the reclassification table and the audit doc (see the doc's new
   "Disagreements" subsection) — the nemo_duplicate/other_recipe/orphan axis
   is new with this pass, not a revision of the doc's own SHARED/NEMO_SWITCH/
   ARTIFICIAL_BRANCH/ABSENT/UNVERIFIED dispositions.
   S-12 and S-30 are genuine ``nemo_duplicate`` defects by the audit's own
   framing (S-30 is the doc's rank-3 collapse item) but their two branch sites
   are inline code inside one function (``_step_impl``), not AST-distinct
   symbols — same carve-out as S-33/S-34, so they carry no baseline entry
   (mechanically unenforceable, not silently dropped: the reason lives in the
   row's own ``nemo_switch`` text).

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


class Impl(NamedTuple):
    # Path under the ``legoesm`` namespace (passed to
    # ``tests.legoesm_paths.legoesm_source_path``), e.g. ``"ocean/eos.py"``.
    rel_path: str
    # Name of a ``def``/``class`` AST-resolvable anywhere in that file
    # (top-level or nested — a nested helper like ``_replace_stage_mean``
    # is a legitimate implementation symbol for its own row).
    symbol: str


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
        Impl(_DINO, "dino_step_surface_forcing"),
        Impl(_RESTORING, "restoring_surface_forcing"),
        Impl(_BULK_OMIP, "air_sea_fluxes"),
    )),
    RoutineRow("S-02", "eos_rab", "NEMO_SWITCH", "ln_teos10 / ln_seos (ORCA1 'wright' = no NEMO arm)", (
        Impl(_EOS, "nemo_roquet_alpha_beta"),
        Impl(_EOS, "nemo_seos_alpha_beta"),
        Impl(_EOS, "eos_density_derivatives"),
        Impl(_EOS, "wright_eos"),
    )),
    RoutineRow("S-03", "bn2", "OTHER_RECIPE", (
        "none (one NEMO routine); reclassified 2026-09-02 (CONFIRMED): "
        "nemo_bn2=NEMO eosbn2.F90, reached by DINO/GYRE/ORCA1; adiabatic=Veros's "
        "own scheme (config.py:310-316); insitu=unreferenced legacy/KPP default"
    ), (
        Impl(_EOS, "compute_buoyancy_frequency_nemo_bn2"),
        Impl(_EOS, "compute_buoyancy_frequency"),
        Impl(_EOS, "compute_buoyancy_frequency_adiabatic"),
    )),
    RoutineRow("S-04", "bn2 live-e3w/gdepw geometry", "OTHER_RECIPE", (
        "none; reclassified 2026-09-02 (CONFIRMED gating / PLAUSIBLE no-alt-user): "
        "nemo arm cites eosbn2.F90 (config.py:352-356, DINO+GYRE only); default "
        "arm names no reference and no non-NEMO recipe is known to need it"
    ), (
        Impl(_EOS, "nemo_bn2_live_ladders"),
        Impl(_EOS, "nemo_bn2_depth_ladders"),
    )),
    RoutineRow("S-05", "zdf_phy -> zdf_cst", "SHARED", "ln_zdfcst", (
        Impl(_CONST_VMIX, "constant_vertical_mixing"),
    )),
    RoutineRow("S-06", "zdf_phy -> zdf_tke", "SHARED", "ln_zdftke", ()),
    RoutineRow("S-07", "zdf_tke solver sub-branch", "OTHER_RECIPE", (
        "none (tke_solver_evaluation); reclassified 2026-09-02 (CONFIRMED): "
        "default=5 Veros fidelity recipes construct TKEConfig untouched, nemo "
        "arm (dino.py:1247-1256)=DINO+GYRE"
    ), (
        Impl(_TKE, "_nemo_literal_tke_solve"),
        Impl(_TKE, "_solve_tke_backward_euler"),
    )),
    RoutineRow("S-07b", "zdf_tke langmuir sub-branch", "OTHER_RECIPE", (
        "none (tke_langmuir_evaluation); reclassified 2026-09-02 (CONFIRMED): "
        "same Veros-default-vs-DINO/GYRE-nemo-arm split as S-07"
    ), (
        Impl(_TKE, "_nemo_literal_langmuir_operands"),
        Impl(_TKE, "nemo_langmuir_tke_source"),
    )),
    RoutineRow("S-08", "zdf_evd", "NEMO_SWITCH", "ln_zdfevd, nn_evdm", (
        Impl(_ENH_DIFF, "enhanced_diffusion_convection"),
    )),
    RoutineRow("S-09", "ldf_slp", "OTHER_RECIPE", (
        "none (~12 *_evaluation sub-selectors); reclassified 2026-09-02 "
        "(CONFIRMED top-level slope_scheme; PLAUSIBLE for the ~8 untraced "
        "sub-fields): default=5 Veros recipes construct GMRediConfig untouched, "
        "nemo arm (slope_scheme='nemo_iso_lap')=DINO+GYRE"
    ), (
        Impl(_GM_REDI, "compute_nemo_native_slopes"),
    )),
    RoutineRow("S-10", "ldf_tra / ldf_eiv", "NEMO_SWITCH", "nn_aht_ijk_t / nn_aei_ijk_t (Visbeck = no NEMO arm)", (
        Impl(_LM_CONFIG, "TreguierConfig"),
        Impl(_LM_CONFIG, "VisbeckConfig"),
        Impl(_OMLC, "static_kappa_redi_override"),
    )),
    RoutineRow("S-11", "ldf_dyn (lateral viscosity coefficient)", "SHARED", (
        "nn_ahm_ijk_t=0; scalar A_h/A_h_lat_scaling config constant, no "
        "function symbol to check"
    ), ()),
    RoutineRow("S-12", "stp_2D pre-step (Kbb RHS seeding dyn_spg_ts)", "ARTIFICIAL_BRANCH", (
        "none; one function _step_impl runs a full WS 3-stage momentum ladder "
        "(omlc:4253-4335) only to seed the barotropic solve, where NEMO "
        "evaluates a single Kbb RHS (stp2d.F90:127-196) -- same defect as S-30 "
        "at a different altitude; both sites live inside _step_impl, not "
        "AST-distinct symbols, so not baseline-enforced here"
    ), (
        Impl(_OMLC, "_step_impl"),
    )),
    RoutineRow("S-13", "dyn_spg_ts external substep loop", "SHARED", "ln_dynspg_ts (cross-oracle: implicit/rigid-lid are non-NEMO)", (
        Impl(_BLC, "barotropic_substeps_latlon_cgrid"),
    )),
    RoutineRow("S-14", "dyn_spg_ts external velocity update gate", "ARTIFICIAL_BRANCH", "ln_dynadv_vec .OR. lk_linssh (legoESM gate ANDs in an extra rk3_ws conjunct NEMO's does not have; no current card mis-routed)", (
        Impl(_BLC, "_nemo_flux_form_external_velocity_update"),
        Impl(_BLC, "substep_body"),
    )),
    RoutineRow("S-15", "dyn_spg_ts backward face depth (zhu_bck)", "NEMO_SWITCH", (
        "key_qcoTest_FluxForm; the two NEMO arms coincide algebraically on a "
        "lat-lon C-grid (e1e2t(i,j)==e1e2t(i+1,j)); min_rule = no NEMO arm"
    ), (
        Impl(_BLC, "nemo_ssh_avg_face_depth"),
        Impl(_BLC, "_min_rule_face_depths"),
    )),
    RoutineRow("S-16", "dyn_spg_ts continuity/transport/spg", "ARTIFICIAL_BRANCH", "none", (
        Impl(_BLC, "nemo_literal_continuity_divergence"),
        Impl(_BLC, "_run_substep_loop"),
    )),
    RoutineRow("S-17", "dyn_cor_2D (barotropic Coriolis)", "NEMO_SWITCH", "ln_dynvor_ene/een ('avg'/'frozen' = no NEMO arm)", (
        Impl(_BLC, "een_barotropic_coriolis"),
        Impl(_BLC, "barotropic_coriolis_een_pre_step"),
    )),
    RoutineRow("S-18", "dom_qco_r3c (r3u/r3v face thickness)", "ARTIFICIAL_BRANCH", "key_qco", (
        Impl(_LCOPS, "min_cell_to_uface"),
        Impl(_VERTICAL, "nemo_qco_live_face_thicknesses"),
    )),
    RoutineRow("S-19", "wzv", "ARTIFICIAL_BRANCH", "none (np_velocity/np_transport arg)", (
        Impl(_VERTICAL, "diagnose_w_from_flux_div"),
        Impl(_OPL, "nemo_qco_wzv_operands"),
    )),
    RoutineRow("S-20", "wAimp", "SHARED", "ln_zad_Aimp", (
        Impl(_VERTICAL, "nemo_wicker_aimp_partition_transport"),
    )),
    RoutineRow("S-21", "stage transport triplet zFu/zFv/zFw", "SHARED", "none (RK3 identity)", (
        Impl(_OMLC, "_nemo_ws_stage_transport"),
    )),
    RoutineRow("S-22", "dyn_adv dispatch", "NEMO_SWITCH", (
        "ln_dynadv_vec / _cen2 / _up3; dispatch resolved inside the main "
        "baroclinic tendency function, no separate dispatcher symbol"
    ), (
        Impl(_OPL, "latlon_cgrid_ocean_baroclinic_tendencies"),
    )),
    RoutineRow("S-23", "dyn_adv_up3 vertical flux", "SHARED", "ln_dynadv_up3", (
        Impl(_VERTICAL, "nemo_up3_vertical_momentum_advection"),
    )),
    RoutineRow("S-24", "dyn_zad", "SHARED", "ln_dynadv_vec", (
        Impl(_VERTICAL, "nemo_advective_vertical_momentum_advection"),
    )),
    RoutineRow("S-25", "vertical momentum advection, non-NEMO arms", "OTHER_RECIPE", (
        "none; reclassified 2026-09-02 (CONFIRMED) -- restores the doc's own "
        "ARTIFICIAL_BRANCH disposition (previously mistranscribed here as "
        "ABSENT): upwind_perturbation=unclaimed default (ORCA1 lands here -- "
        "a driver reachability gap, not a duplicate), centered_full=Veros "
        "core/momentum.py, nemo_advective=NEMO dynzad.F90 (see S-24)"
    ), (
        Impl(_VERTICAL, "flux_form_vertical_momentum_advection"),
        Impl(_VERTICAL, "flux_form_vertical_momentum_advection_centered"),
        Impl(_VERTICAL, "nemo_advective_vertical_momentum_advection"),
    )),
    RoutineRow("S-26", "dyn_vor (vor_ene/vor_ens/vor_een)", "ABSENT (vor_ens missing; al81 = no NEMO arm)", "ln_dynvor_ene/_ens/_een", (
        Impl(_LCOPS, "pv_flux_al81_partial_cell"),
        Impl(_LCOPS, "pv_flux_ene"),
    )),
    RoutineRow("S-27", "Coriolis time placement", "OTHER_RECIPE", (
        "none (NEMO always inside dyn_vor RHS); reclassified 2026-09-02 "
        "(CONFIRMED): matsuno_split=unclaimed default incl. ORCA1 (live risk "
        "there, inert on LOCK/OVERFLOW at f=0), explicit_ab2=Veros's own scheme "
        "AND NEMO's own Coriolis placement (DINO+GYRE+veros_faithful_v1+"
        "oceananigans_v1+mitgcm_v1)"
    ), (
        Impl(_OMLC, "_forward_backward_coriolis_3d"),
        Impl(_OPL, "latlon_cgrid_ocean_baroclinic_tendencies"),
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
        Impl(_OPL, "_bc_ke_and_pressure_gradients"),
        Impl(_PGF_SMC03, "compute_pressure_at_target_smc03"),
        Impl(_PGF_AHH08, "column_pressure_integrals_ahh08"),
    )),
    RoutineRow("S-30", "stage momentum time-stepping (WS ladder written twice)", "ARTIFICIAL_BRANCH", (
        "key_qco; one function _step_impl, ladder #1 at omlc:4253-4335 (seeds "
        "the barotropic solve only, no barotropic correction) and ladder #2 "
        "at omlc:4830-4886 (barotropic-corrected, kept, overwrites "
        "state_new.u/v) -- both always run when momentum_time_integrator= "
        "'rk3_ws'; genuine nemo_duplicate per the audit (rank-3 collapse item, "
        "an already-realized bug: a fix to _stage_vertical_up3/live-HPG landed "
        "in ladder #2 only) but the two sites are inline code inside one "
        "function, not AST-distinct symbols, so not baseline-enforced here"
    ), (
        Impl(_OMLC, "_step_impl"),
    )),
    RoutineRow("S-31", "stage 2/3 eos(Kmm)+dyn_hpg(Kmm) operands", "SHARED", "none", (
        Impl(_OMLC, "_stage_hpg_operands"),
    )),
    RoutineRow("S-32", "dyn_ldf -> ldf_lap (viscosity operator form)", "OTHER_RECIPE", (
        "ln_dynldf_lap (one NEMO arm, 3 legoESM operator forms); reclassified "
        "2026-09-02 (CONFIRMED): nemo_div_curl=DINO+GYRE (NEMO "
        "dyn_ldf_lev_lap, dino.py:914), flux_divergence=Veros core/friction.py "
        "harmonic_friction, vector_laplacian (ORCA1's default)=unreferenced"
    ), (
        Impl(_LCOPS, "nemo_ldf_lap_viscosity_cgrid"),
        Impl(_LCOPS, "vector_laplacian_dissipation_cgrid"),
        Impl(_LCOPS, "flux_divergence_viscosity_cgrid"),
    )),
    RoutineRow("S-33_34", "dyn_zdf / e3w(Kmm) divisor", "OTHER_RECIPE", (
        "none (one function, internal divisor branch -- not AST-distinct "
        "symbols; not baseline-enforced here); reclassified 2026-09-02 "
        "(CONFIRMED selection facts, PLAUSIBLE exact F90 citation): NEMO arm "
        "cites trazdf.F90:219-220 (LOCK/OVERFLOW/GYRE), Veros arm cites Veros "
        "thermodynamics.py:267 (veros_faithful_v1), legacy midpoint "
        "(DINO+unclaimed default+ORCA1) names no reference"
    ), (
        Impl(_OMLC, "_apply_implicit_vertical_mixing"),
    )),
    RoutineRow("S-35", "stage-3 zub barotropic correction", "ARTIFICIAL_BRANCH", "none (NEMO: dyn_zdf then zub, one site)", (
        Impl(_OMLC, "_replace_stage_mean"),
        Impl(_OMLC, "_fixed_depth_means"),
    )),
    RoutineRow("S-36", "tra_adv_trp (reuse momentum zF triplet)", "SHARED", (
        "none; private var _nemo_ws_live_stage_geometry + "
        "kmm_tracer_transports config flag (omlc:4213-5240), not a separate "
        "function symbol"
    ), ()),
    RoutineRow("S-37", "tra_adv dispatch", "ARTIFICIAL_BRANCH", "ln_traadv_fct (SHARED for D/L/O/G; ORCA1 runs superbee, a Veros scheme, no NEMO arm)", (
        Impl(_OMLC, "_nemo_ws_rk3_tracer_pair_step"),
        Impl(_OMLC, "compute_advection_flux_div_pair"),
        Impl(_ADVECTION, "fct_tracer_advection"),
        Impl(_ADVECTION, "_veros_superbee_face_flux"),
    )),
    RoutineRow("S-38", "tra_adv_fct implicit-w treatment", "UNVERIFIED", "nn_fct_imp", (
        Impl(_ADVECTION, "fct_tracer_advection"),
    )),
    RoutineRow("S-39", "tracer stage time-stepping (qco (1+r3t) weighting)", "SHARED", "key_qco", (
        Impl(_OMLC, "_nemo_ws_rk3_tracer_pair_step"),
    )),
    RoutineRow("S-40", "tra_sbc placement (DINO-scoped selector)", "OTHER_RECIPE", (
        "none (nn_fsbc; RK3 cards have no equivalent knob); reclassified "
        "2026-09-02 (CONFIRMED): leapfrog_rhs=NEMO tra_sbc Nnn-RHS placement "
        "(certified MLF card; run_dino.py:625-634 hard-forbids the bad "
        "leapfrog+applied_now pairing), applied_now=DINO's non-MLF oracle "
        "sub-recipes (veros/mitgcm/oceananigans variants)"
    ), (
        Impl(_DINO, "_check_surface_tendency_placement"),
    )),
    RoutineRow("S-41", "tra_qsr", "NEMO_SWITCH", "ln_qsr_2bd / ln_qsr_rgb", (
        Impl(_SHORTWAVE, "apply_shortwave_penetration"),
    )),
    RoutineRow("S-42", "bbl + tra_bbl", "ARTIFICIAL_BRANCH", "ln_trabbl, nn_bbl_adv=2 (one NEMO program, two compositions)", (
        Impl(_BBL, "apply_bbl_adv_tendency"),
        Impl(_BBL, "apply_bbl_adv_step"),
    )),
    RoutineRow("S-43", "tra_ldf", "see S-09", "ln_traldf_lap+ln_traldf_iso", (
        Impl(_GM_REDI, "compute_nemo_native_slopes"),
    )),
    RoutineRow("S-44", "tra_zdf", "see S-33_34", "always", (
        Impl(_OMLC, "_apply_implicit_vertical_mixing"),
    )),
    RoutineRow("S-45", "tra_npc", "ABSENT", "ln_zdfnpc", ()),
    RoutineRow("M-01", "stp_MLF whole-step composition", "ARTIFICIAL_BRANCH", "absence of key_RK3 (nemo_mlf selected by no card)", (
        Impl(_OMLC, "_leapfrog_step"),
        Impl(_OMLC, "_nemo_mlf_step"),
    )),
    RoutineRow("M-02", "ssh_nxt (+div_hor)", "SHARED", "none", (
        Impl(_BLC, "barotropic_substeps_latlon_cgrid"),
    )),
    RoutineRow("M-03", "ssh_atf (Asselin filter)", "SHARED", "rn_atfp", (
        Impl(_OMLC, "_thickness_weighted_asselin"),
    )),
    RoutineRow("M-04", "tra_atf_qco / dyn_atf_qco (tracer_combine)", "ARTIFICIAL_BRANCH", "key_qco ('concentration' = no NEMO arm)", (
        Impl(_OMLC, "thickness_weighted_tracer_combine"),
        Impl(_OMLC, "thickness_weighted_tracer_content"),
    )),
    RoutineRow("M-05", "mlf_baro_corr", "ARTIFICIAL_BRANCH", "ln_dynspg_ts ('off' = no NEMO arm)", (
        Impl(_BAROTROPIC_COMMON, "after_level_column_mean_reconcile"),
        Impl(_BAROTROPIC_COMMON, "nemo_literal_after_level_reconcile"),
    )),
    RoutineRow("M-06", "dyn_ldf/tra_ldf at Kbb inside one pass", "see M-01", "none", (
        Impl(_OMLC, "_leapfrog_step"),
        Impl(_OMLC, "_nemo_mlf_step"),
    )),
    RoutineRow("CARD-ASSEMBLY", "card assembly (three assemblers, one RK3 identity)", "ARTIFICIAL_BRANCH", "n/a — legoESM-side; NEMO has one namelist per configuration", (
        Impl(_TESTCASE_RECIPE, "_model_config"),
        Impl(_NEMO_RECIPE, "nemo_lat_lon_model_config"),
        Impl(_DINO, "dino_lat_lon_model_config"),
    )),
)


# Shrink-only allow-list: routine_id -> the audit's reason this artificial
# branch is currently tolerated. A row may be listed ONLY while its
# ROUTINE_REGISTRY entry is disposition="ARTIFICIAL_BRANCH" with >=2 distinct
# impls; once a collapse PR lands (see the doc's "collapse plan" per row) and
# the row drops to <=1 distinct impl, DELETE the entry in that same PR. A row
# reclassified "other_recipe" moves its rationale into the registry row's own
# ``nemo_switch`` text and its disposition to "OTHER_RECIPE" instead — it is
# no longer a baseline exception (see the 2026-09-02 reclassification note in
# the module docstring).
ARTIFICIAL_BRANCH_BASELINE: dict[str, BaselineEntry] = {
    "S-14": BaselineEntry(
        reason=(
            'dyn_spg_ts external velocity update gate carries an extra '
            "momentum_time_integrator=='rk3_ws' conjunct NEMO's own gate (dynspg_ts.F90:719) "
            'does not have — latent mis-route for any future MLF+flux-form card; no current '
            'card affected.'
        ),
        kind='unclassified',
    ),
    "S-16": BaselineEntry(
        reason=(
            'dyn_spg_ts continuity/transport/spg: 5 generic-vs-nemo_literal branch pairs for '
            'one NEMO program (dynspg_ts.F90:~640-840); default is the generic arm, which '
            'ORCA1 runs. Reclassified 2026-09-02: NEMO_DUPLICATE, CONFIRMED.'
        ),
        kind='nemo_duplicate',
    ),
    "S-18": BaselineEntry(
        reason=(
            'RANK 1 (highest-value collapse). dom_qco_r3c r3u/r3v face thickness: '
            "MIN-of-live-thickness vs NEMO's surface-weighted MEAN (domqco.F90:165-170) for "
            'one quantity; DINO/GYRE reach MEAN at ldf_slp/dyn_zad, LOCK/OVERFLOW/ORCA1 reach '
            'MIN everywhere including the shared dyn_adv/tra_adv operand. Mean-vs-min '
            'measurement was IN FLIGHT at audit time — do not collapse ahead of it. '
            'Reclassified 2026-09-02: NEMO_DUPLICATE, CONFIRMED — LOCK/OVERFLOW resolve MIN '
            'on this checkout too, a live gap on the certified L1 cards, not just ORCA1.'
        ),
        kind='nemo_duplicate',
    ),
    "S-19": BaselineEntry(
        reason=(
            'wzv: one NEMO routine (sshwzv.F90), two legoESM impls; default is the non-NEMO '
            '(generic) one. Reclassified 2026-09-02: NEMO_DUPLICATE, CONFIRMED — same '
            'LOCK/OVERFLOW-reach correction as S-18.'
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
    "S-42": BaselineEntry(
        reason=(
            'bbl: shared transport/tendency arithmetic, two compositions — OVERFLOW folds it '
            "into the stage-3 tracer RHS (NEMO's site), ORCA1 calls a host post-step Euler "
            'with an extra transport cap NEMO has none of. Reclassified 2026-09-02: '
            'NEMO_DUPLICATE, CONFIRMED.'
        ),
        kind='nemo_duplicate',
    ),
    "M-01": BaselineEntry(
        reason=(
            'RANK 5. stp_MLF: two implementations, one dead. _nemo_mlf_step (single '
            'dyn_ldf(Kbb,Kmm) pass, structurally faithful) is selected by no card; only a '
            'probe script calls it directly. Reclassified 2026-09-02: NEMO_DUPLICATE, '
            'CONFIRMED — (b) has real committed test coverage '
            '(test_nemo_mlf_step_transcription.py), so it is validated-but-unpromoted, not '
            'dead code / ORPHAN.'
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
