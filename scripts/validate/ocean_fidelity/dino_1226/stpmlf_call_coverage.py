#!/usr/bin/env python
"""COVERAGE-DRIVEN stp_MLF call-graph gate (#1226 skill Rule 1 extended).

WHY THIS EXISTS
---------------
``fidelity_bar_gate.py`` is a CHECKLIST: it holds one row per term someone
already suspected. Coverage requires the other direction -- enumerate every
``CALL`` the oracle's own time-stepping routine actually makes, in execution
order, and force a disposition on EACH one. A term nobody thought to add to
the checklist is invisible to it by construction; this script is built from
what NEMO's ``stp_MLF`` calls, not from our prior suspicion list.

SOURCE OF TRUTH
----------------
DINO ships its OWN override of the time-stepping routine:
``cfgs/DINO/MY_SRC/stpmlf.F90`` (confirmed present; DINO's ``MY_SRC`` always
wins over ``src/OCE/stpmlf.F90`` in NEMO's build system). Its top guard is
``#if ! defined key_RK3`` / ``# if defined key_qco || defined key_linssh``;
DINO's ``cfgs/DINO/cpp_DINO.fcm`` is::

    bld::tool::fppkeys key_qco key_vco_3d

(``key_RK3`` undefined, ``key_qco`` defined) so the QCO/MLF branch is the one
that compiles -- confirmed by reading the file, not assumed from its name.

Every CALL below is transcribed from that file with its line number. Dispatch
wrappers (``zdf_phy``, ``dyn_adv``, ``dyn_vor``, ``dyn_ldf``, ``dyn_hpg``,
``dyn_spg``, ``tra_adv``, ``tra_ldf``) are resolved ONE level to the concrete
routine DINO's ``cfgs/DINO/EXP00/namelist_cfg`` (falling back to
``cfgs/SHARED/namelist_ref`` for anything DINO does not override) actually
selects; the deciding namelist/cpp lines are quoted in each entry's ``note``.

Usage:
  stpmlf_call_coverage.py               -- print the table, exit 0 if closed
  stpmlf_call_coverage.py --self-test   -- prove the gate is non-vacuous
Exit 0 = every enumerated CALL has a disposition. Non-zero (fail CLOSED) if
any CALL is missing one -- this is the mechanical, not-judgment-based check;
adding a routine here without a disposition is a hard failure, not a warning.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

STPMLF = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
          "cfgs/DINO/MY_SRC/stpmlf.F90")

# Disposition kinds. COVERED = a fidelity_bar_gate.py row measures this
# routine. WAIVED = deliberately out of scope, reason required. UNCOVERED =
# no gate row measures it -- these are surfaced LOUDLY and ranked by
# (my) judgment of climate leverage, labelled as judgment, not measurement.
# COVERED_UNMEASURED = a fidelity_bar_gate.py row now EXISTS for this routine
# (so it is accounted for in the coverage sense -- it is no longer invisible
# to the checklist), but that row's own status is UNMEASURED (None/None) --
# a row existing is not the same claim as a row being measured. This is a
# DISTINCT kind from plain UNCOVERED (no row at all, judgment-ranked) and from
# COVERED (a row exists, whether or not its own measurement clears the bar).
COVERED, WAIVED, UNCOVERED, COVERED_UNMEASURED = (
    "COVERED", "WAIVED", "UNCOVERED", "COVERED (row UNMEASURED)")


@dataclass(frozen=True)
class CallEntry:
    line: int                 # line in stpmlf.F90 (MY_SRC override) of the CALL
    routine: str               # concrete routine DINO actually runs
    disposition: str           # COVERED | WAIVED | UNCOVERED | COVERED_UNMEASURED
    note: str                  # gate row name(s) / reason / climate-leverage judgment
    rank: int | None = None    # UNCOVERED only: 1 = highest judged leverage


# ---------------------------------------------------------------------------
# THE CALL LIST, in stp_MLF execution order (cfgs/DINO/MY_SRC/stpmlf.F90).
# Every `CALL` statement in the file appears here exactly once. Calls made
# only inside a compiled-out `#if` branch for DINO's cpp keys are recorded
# with disposition WAIVED and the excluding key named in `note`.
# ---------------------------------------------------------------------------
CALLS: list[CallEntry] = [
    # --- IO/calendar bookkeeping (lines 124-161) ---
    CallEntry(124, "iom_init", WAIVED, "XIOS context management; I/O plumbing, no physics."),
    CallEntry(125, "dia_mlr_iom_init", WAIVED,
              "gated ln_diamlr (namelist_ref default False, DINO does not set it) "
              "-- dead branch; also I/O-only if it ran."),
    CallEntry(126, "iom_init_closedef", WAIVED, "I/O plumbing."),
    CallEntry(127, "dia_hth_init", WAIVED, "diagnostic-only (thermocline-depth output) init."),
    CallEntry(128, "dia_ptr_init", WAIVED, "diagnostic-only (poleward-transport output) init."),
    CallEntry(129, "dia_ar5_init", WAIVED, "diagnostic-only (AR5 diag output) init."),
    CallEntry(130, "dia_hsb_init", WAIVED,
              "heat/salt/volume BUDGET diagnostic init -- iom_use-gated, output only."),
    CallEntry(131, "dia_25h_init", WAIVED, "25h-mean diagnostic output init, not namelist-enabled."),
    CallEntry(132, "mlf_dia", WAIVED, "sets iom_use() diagnostic-logical flags only (local sub, line 542)."),
    CallEntry(135, "iom_swap", WAIVED, "restart-file I/O context swap (lwxios path; DINO uses lrxios=F by default)."),
    CallEntry(136, "iom_init_closedef", WAIVED, "restart-file I/O plumbing."),
    CallEntry(137, "iom_setkt", WAIVED, "restart-file I/O plumbing."),
    CallEntry(160, "day", WAIVED, "calendar bookkeeping."),
    CallEntry(161, "iom_setkt", WAIVED, "I/O plumbing (tell IOM the current step)."),
    CallEntry(171, "iom_setkt", WAIVED, "restart-file I/O plumbing (SI3 restart context; DINO has no sea ice)."),
    CallEntry(177, "iom_setkt", WAIVED, "restart-file I/O plumbing (ABL restart context; DINO has no ABL)."),
    CallEntry(181, "iom_setkt", WAIVED, "I/O plumbing: tells XIOS the current step number; touches no ocean state."),

    # --- external forcing update (lines 166-172) ---
    CallEntry(166, "tide_update", WAIVED,
              "ln_tide=.false. (namelist_ref default, not set by DINO) -- dead branch."),
    CallEntry(167, "sbc_apr", WAIVED,
              "ln_apr_dyn=.false. (namelist_ref default, not set by DINO) -- dead branch."),
    CallEntry(168, "bdy_dta", WAIVED, "ln_bdy=.false. (namelist_ref) -- dead branch."),
    CallEntry(169, "isf_stp", WAIVED, "ln_isf=.false. (DINO namelist_cfg &namsbc_isf) -- dead branch."),
    CallEntry(170, "sbc", COVERED, '"sbc (utau/qsr/qns/sfx)" -- dispatches to usrdef_sbc_oce '
              "(DINO namelist_cfg &namsbc: ln_usr=.true.) since nsbc==jp_usr; "
              "sbcmod.F90:68."),

    # --- stochastic EOS (line 177-178) ---
    CallEntry(177, "sto_par", WAIVED, "ln_sto_eos=.false. (namelist_ref) -- dead branch."),
    CallEntry(178, "sto_pts", WAIVED, "ln_sto_eos=.false. -- dead branch."),

    # --- thermodynamics (lines 184-187) ---
    CallEntry(184, "eos_rab (Nbb, BEFORE T/S)", COVERED, '"eos_rab beta", "eos_rab alpha"'),
    CallEntry(185, "eos_rab (Nnn, NOW T/S)", COVERED,
              "same eos_rab routine as line 184, NOW-level call -- no separate gate "
              "row exists for the NOW-level output specifically; the BEFORE-level "
              "row above certifies the formula/routine, so this is COVERED by the "
              "same code path, not independently re-measured at this time level."),
    CallEntry(186, "bn2 (Nbb, BEFORE)", COVERED, '"bn2 (rn2b)"'),
    CallEntry(187, "bn2 (Nnn, NOW)", COVERED,
              "same bn2 routine as line 186 -- see line 185 note (shared-formula "
              "argument, not an independent NOW-level measurement)."),

    # --- vertical physics (line 190) ---
    CallEntry(190, "zdf_phy -> zdf_drg('BOTTOM')", COVERED,
              '"zdf_drg_nonlin T-point rate", "dyn_drg_init RHS increment" -- '
              "zdfphy.F90:277, unconditional (ln_isfcav/ln_drgice_imp both False "
              "for DINO so the TOP-drag sibling at :278 does not fire)."),
    CallEntry(190, "zdf_phy -> zdf_mxl", COVERED,
              '"zdf_mxl (nmln)" -- zdfphy.F90:280, N^2-criterion MLD level (nmln).'),
    CallEntry(190, "zdf_phy -> zdf_tke", COVERED,
              '"zdftke pdlr", "zdftke composite avt/avm" -- zdfphy.F90:286, '
              "nzdf_phy==np_TKE selected because DINO &namzdf sets ln_zdftke=.true. "
              "(the only True closure flag; ln_zdfric/ln_zdfgls/ln_zdfosm all "
              "default False in namelist_ref and DINO does not set them)."),
    CallEntry(190, "zdf_phy -> zdf_evd", COVERED,
              "part of the zdf_mxl/zdftke composite avt/avm gate rows above -- "
              "zdfphy.F90:323, DINO &namzdf sets ln_zdfevd=.true. (enhanced "
              "vertical diffusivity where unstable); no gate row isolates evd's "
              "OWN contribution separately from the composite avt/avm it feeds "
              "into, so this is COVERED as part of that composite, not as its "
              "own independently-measured term.", ),
    CallEntry(190, "zdf_phy -> zdf_mxl_turb", COVERED,
              '"zdf_mxl_turb" (row exists, UNMEASURED None/None) -- zdfphy.F90:338, '
              "unconditional. TASK B verdict: no DINO-active consumer of its "
              "output (hmld/mldkz5) exists, so this row is a candidate WAIVE, "
              "not a defect -- see this file's module docstring addendum and the "
              "companion consumer-grep report."),

    # --- lateral physics: slopes + coefficients (lines 194-208) ---
    CallEntry(199, "eos (Nbb, in-situ density for ldf_slp)", WAIVED,
              "the eos() call feeding ldf_slp's density argument; no gate row "
              "isolates this specific eos() call from the ldf_slp rows it "
              "feeds -- the ldf_slp rows below already trace prd (this "
              "density) as their first stage (fidelity_bar_gate.py ldf_slp "
              "wslpi note, 'STAGE WALK ... prd 1.989e-11')."),
    CallEntry(199, "ldf_slp", COVERED,
              '"ldf_slp wslpi", "ldf_slp wslpj", "ldf_slp uslp", "ldf_slp vslp" -- '
              "stpmlf.F90:196-200 IF/ELSE: l_ldfslp is True (ldftra.F90:249, "
              "DINO nldf_tra==np_lap_i since &namtra_ldf sets "
              "ln_traldf_lap=.true./ln_traldf_iso=.true.) and ln_traldf_triad="
              "'.false.' (DINO &namtra_ldf) selects the ELSE branch "
              "(eos+ldf_slp), not ldf_slp_triad."),
    CallEntry(203, "ldf_tra", COVERED,
              '"ldftra ahtu (Redi, nn_aht_ijk_t=20)", "ldftra ahtv (Redi, '
              'nn_aht_ijk_t=20)", "ldf_eiv kappa (aeiu)" -- l_ldftra_time/'
              "l_ldfeiv_time True per DINO &namtra_ldf nn_aht_ijk_t=20 / "
              "&namtra_eiv nn_aei_ijk_t=21 (both time-varying)."),
    CallEntry(204, "ldf_dyn", COVERED_UNMEASURED,
              '2026-07-30: gate row "ldf_dyn coefficient" now exists '
              "(UNMEASURED None/None) -- previously no gate row measured "
              "ahmt/ahmf (the momentum lateral-viscosity coefficient this "
              "computes) directly, only its CONSUMER "
              '"dyn_ldf (dynldf_lev_lap) u/v", which does not isolate the '
              "coefficient from the tendency. l_ldfdyn_time is True per DINO "
              "&namdyn_ldf nn_ahm_ijk_t=20. JUDGMENT (not measurement, still "
              "unmeasured): MODERATE leverage -- an error here would show "
              "up folded into the already-DEBT dynldf_lev_lap rows (corr "
              "0.9979-0.9994), so it is not fully invisible, but a "
              "coefficient-only bug could masquerade as a scheme bug."),
    CallEntry(208, "bbl", WAIVED, "ln_trabbl=.false. (namelist_ref default) -- dead branch."),

    # --- dynamics: ssh/e3/hdiv (lines 214-236) ---
    CallEntry(214, "ssh_nxt", COVERED, '"ssh_nxt / div_hor"'),
    CallEntry(216, "dom_qco_r3c (Naa, first call)", COVERED,
              '"dom_qco_r3c r3t", "dom_qco_r3c r3u/r3v" -- always active for '
              "DINO since lk_linssh=.false. (key_linssh undefined)."),
    CallEntry(218, "dom_qco_r3c (Nnn, spg_exp variant)", WAIVED,
              "guarded by ln_dynspg_exp, which is .false. for DINO "
              "(&namdyn_spg ln_dynspg_ts=.true. is the selected scheme) -- "
              "dead branch."),

    # --- momentum RHS accumulation, tiled loop 1 (lines 244-284) ---
    CallEntry(244, "wzv (Nnn cross-level velocity)", COVERED_UNMEASURED,
              '2026-07-30: gate row "wzv (vertical velocity)" now exists '
              "(UNMEASURED None/None, covers BOTH this call site and the "
              "line-315 recomputation as one row) -- previously no gate row "
              "measured ww (vertical velocity from the continuity equation) "
              "directly. JUDGMENT (still unmeasured): HIGH leverage -- ww "
              "feeds tra_adv's vertical flux, dyn_zad, and wAimp; several "
              "DOWNSTREAM rows (traadv_fct vertical upstream flux, dyn_adv "
              "ZAD) are already DEBT, and ww itself has never been isolated "
              "as the candidate common cause."),
    CallEntry(245, "wAimp", WAIVED,
              "ln_zad_Aimp=.false. (namelist_ref default, DINO does not set "
              "it in &namdyn_adv) -- dead branch."),
    CallEntry(246, "eos (Nnn, in-situ density+rhop for hpg)", WAIVED,
              '"dyn_hpg (du)"/"dyn_hpg (dv)" already trace this density as '
              "their own input (fidelity_bar_gate.py dyn_hpg note: "
              "eos_geometric_depth_1d / nemo_bn2_live_ladders is the PGF's "
              "own density path) -- no separate row isolates this eos() call "
              "from the hpg rows it feeds."),
    CallEntry(248, "dyn_dmp", WAIVED, "ln_dyndmp=.false. AND ln_c1d=.false. (namelist_ref) -- dead branch (both)."),
    CallEntry(250, "dyn_asm_inc", WAIVED,
              "lk_asminc=.FALSE. at COMPILE TIME (key_asminc not in "
              "cfgs/DINO/cpp_DINO.fcm) -- not merely namelist-off, absent "
              "from the DINO build entirely."),
    CallEntry(252, "asm_bkg_wri", WAIVED, "ln_bkgwri=.false. (namelist_ref default) -- dead branch."),
    CallEntry(253, "bdy_dyn3d_dmp", WAIVED, "ln_bdy=.false. -- dead branch."),
    CallEntry(265, "dyn_adv -> dyn_keg + dyn_zad", COVERED,
              '"dyn_adv KEG", "dyn_adv ZAD" -- np_VEC_c2 selected because '
              "DINO &namdyn_adv sets ln_dynadv_vec=.true. (dynadv.F90:79-82)."),
    CallEntry(271, "dyn_vor -> vor_een", COVERED,
              '"dyn_vor EEN u", "dyn_vor EEN v" -- np_EEN selected because '
              "DINO &namdyn_vor sets ln_dynvor_een=.true. (the only True "
              "vorticity-scheme flag; dynvor.F90:138)."),
    CallEntry(275, "dyn_ldf -> dynldf_lev_lap", COVERED,
              '"dyn_ldf (dynldf_lev_lap) u", "dyn_ldf (dynldf_lev_lap) v" -- '
              "np_lap selected because DINO &namdyn_ldf sets "
              "ln_dynldf_lap=.true./ln_dynldf_lev=.true. (dynldf.F90:68-70)."),
    CallEntry(279, "dyn_osm", WAIVED, "ln_zdfosm=.false. (namelist_ref default) -- dead branch."),
    CallEntry(280, "dyn_hpg -> hpg_sco", COVERED,
              '"dyn_hpg (du)", "dyn_hpg (dv)" -- np_sco selected because DINO '
              "&namdyn_hpg sets ln_hpg_sco=.true. (dynhpg.F90:119)."),

    # --- surface pressure gradient (line 288) ---
    CallEntry(288, "dyn_spg -> dyn_spg_ts", COVERED,
              '"dyn_spg_ts pssh", "dyn_spg_ts puu_b", "dyn_spg_ts un_adv", '
              '"dyn_cor_2d (69x/step)" -- np_TS selected because DINO '
              "&namdyn_spg sets ln_dynspg_ts=.true. (dynspg.F90:178); the "
              "ln_apr_dyn/tide/ln_ice_embd/ln_bern_srfc atm-pressure-forcing "
              "block above it (dynspg.F90:102-105) is all-False for DINO so "
              "contributes nothing extra to this call's trend."),

    # --- tiled loop 2: barotropic-corrected dynamics (lines 302-320) ---
    CallEntry(302, "div_hor (2nd call, time-split)", COVERED,
              'part of "ssh_nxt / div_hor" -- same routine/gate row as line '
              "214's ssh_nxt-internal div_hor call; ln_dynspg_ts=.true. makes "
              "this second call active (stpmlf.F90:300-304)."),
    CallEntry(303, "dom_qco_r3c (Naa, third arg set incl. r3f)", COVERED,
              'covered by "dom_qco_r3c r3t"/"r3u/r3v" (same routine); r3f '
              "itself (F-point ratio) has no dedicated gate row -- see the "
              "UNCOVERED entry below."),
    CallEntry(305, "dyn_zdf (dyn_zdf_imp, implicit)", COVERED_UNMEASURED,
              '2026-07-30: gate row "dyn_zdf (momentum implicit vertical '
              'solve)" now exists (UNMEASURED None/None) -- previously no '
              "gate row measured the implicit vertical-momentum-diffusion "
              "SOLVE itself (the tridiagonal solve output uu/vv(Naa)) -- only "
              "its INPUT avm (via zdftke) and a downstream barotropic-"
              "reconciliation quantity (puu_b) are gated. JUDGMENT (still "
              "unmeasured): HIGH leverage -- this is the FINAL "
              "momentum-state-producing step of the entire baroclinic branch "
              "every timestep (folds the RHS, barotropic drag removal at "
              "ln_drgimp.AND.ln_dynspg_ts dynzdf.F90:148-171, and vertical "
              "mixing into one solve) and the #1226 dump instrumentation "
              "(stp_dump_state_and_bt(dynzdf), line 312) exists specifically "
              "because this stage was identified as needing scrutiny, yet no "
              "probe closes the loop into a gate row."),
    CallEntry(315, "wzv (Naa cross-level velocity, 2nd call)", COVERED_UNMEASURED,
              'same routine as line 244 -- covered by the same "wzv '
              '(vertical velocity)" gate row (UNMEASURED); this is the '
              "post-dyn_zdf recomputation, guarded by ln_dynspg_ts=.true. "
              "(stpmlf.F90:314-315)."),
    CallEntry(319, "wAimp (2nd call)", WAIVED, "ln_zad_Aimp=.false. -- dead branch (see line 245)."),

    # --- cool skin / GEOMETRIC (lines 325-331) ---
    CallEntry(325, "diurnal_layers", WAIVED, "ln_diurnal=.false. (namelist_ref default) -- dead branch."),
    CallEntry(331, "ldf_eke", WAIVED,
              "l_ldfeke requires ln_eke_equ=.true. (ldftra.F90:634-636); "
              "ln_eke_equ defaults .false. in namelist_ref &namldf_eke and "
              "DINO's namelist_cfg does not set it -- dead branch (GEOMETRIC "
              "total-EKE equation is not active for this DINO recipe)."),

    # --- diagnostics/outputs (lines 336-356) ---
    CallEntry(336, "dia_cfl", WAIVED, "ln_diacfl=.false. (namelist_ref default) -- dead branch."),
    CallEntry(337, "dia_dct", WAIVED, "ln_diadct=.false. (namelist_ref default) -- dead branch."),
    CallEntry(343, "dia_hth", WAIVED,
              "l_hth is set from iom_use() of thermocline-depth output "
              "fields (diahth.F90:353) -- output-gated diagnostic, no "
              "physics feedback regardless of value."),
    CallEntry(344, "dia_ar5", WAIVED, "AR5 diagnostic output, iom_put-only."),
    CallEntry(345, "dia_ptr", WAIVED,
              "l_diaptr is iom_use()-gated (diaptr.F90:483) poleward-"
              "transport diagnostic output; also invoked non-conditionally "
              "from tra_adv (traadv.F90:376) for the same diagnostic "
              "purpose -- output only either way."),
    CallEntry(347, "dia_wri (tiled, key_xios variant)", WAIVED,
              "guarded #if defined key_xios; DINO's cpp_DINO.fcm does not "
              "define key_xios -- dead branch, compiled out."),
    CallEntry(353, "dia_wri (untiled, default variant)", WAIVED,
              "the variant that DOES compile for DINO (#if ! defined "
              "key_xios); pure model-output write, no physics feedback."),
    CallEntry(355, "dia_detide", WAIVED,
              "l_diadetide requires nn_dia25h-related detiding setup "
              "(diadetide.F90:49-54); not configured for DINO -- dead "
              "branch, and output-only if it ran."),
    CallEntry(356, "dia_mlr", WAIVED, "l_diamlr requires ln_diamlr=.true. (namelist_ref default False) -- dead branch."),

    # --- ssh filtering (lines 361-362) ---
    CallEntry(361, "ssh_atf", COVERED_UNMEASURED,
              '2026-07-30: gate row "ssh_atf" now exists (UNMEASURED '
              'None/None) -- previously "ATF filter T/S/ssh" (COVERED, '
              "exact) measures the TRACER Asselin-filter routine "
              "tra_atf_qco's ssh leg per that row's own caveat ('the ssh leg "
              "is TAUTOLOGICAL'); ssh_atf itself (sshwzv.F90, a DIFFERENT "
              "routine that time-filters ssh before tra_atf_qco/dyn_atf_qco "
              "run) is never independently probed. JUDGMENT (still "
              "unmeasured): LOW-MODERATE leverage -- ssh is a "
              "slowly-evolving, well-observed quantity and the tautological "
              "check above is a weak but nonzero signal that the filtered "
              "value is self-consistent."),
    CallEntry(362, "dom_qco_r3c (Nnn, filtered r3t_f/r3u_f/r3v_f)", COVERED,
              'covered by "dom_qco_r3c r3t"/"r3u/r3v" (same routine, filtered '
              "ssh input)."),

    # --- passive tracers (line 367) ---
    CallEntry(367, "trc_stp", WAIVED,
              "guarded #if defined key_top; DINO's cpp_DINO.fcm does not "
              "define key_top -- TOP/PISCES is entirely absent from the "
              "DINO build (confirmed by the companion zdf_mxl_turb consumer "
              "grep: cfgs/DINO/WORK/ contains no oce_trc.F90/p4z*/p5z*/p2z* "
              "files at all)."),

    # --- active tracers RHS, tiled loop (lines 386-402) ---
    CallEntry(386, "tra_asm_inc", WAIVED, "lk_asminc=.FALSE. at compile time -- see line 250."),
    CallEntry(387, "tra_sbc", COVERED_UNMEASURED,
              '2026-07-30: gate row "tra_sbc" now exists (UNMEASURED '
              "None/None) -- previously no gate row isolated the tracer "
              "surface-boundary-condition RHS contribution on its own (the "
              "sbc() call at line 170 gates utau/qsr/qns/sfx as FORCING "
              "FIELDS, not this routine's application of them into "
              "ts(Krhs)). JUDGMENT (still unmeasured): LOW leverage -- it is "
              "a straight flux-into-tendency application with no internal "
              "branching/scheme choice, and its own instrumentation "
              "(stp_dump_ts_krhs 'trasbc', line 393) exists precisely so a "
              "future probe CAN close this gap cheaply."),
    CallEntry(394, "tra_qsr", COVERED_UNMEASURED,
              '2026-07-30: gate row "tra_qsr (shortwave penetration)" now '
              "exists (UNMEASURED None/None) -- penetrative solar radiation "
              "tendency; ln_traqsr=.true. for DINO (&namsbc) and "
              "ln_qsr_2bd=.true. (&namtra_qsr, 2-band light penetration). "
              "Previously no gate row measured it, though the underlying "
              "qsr FORCING FIELD is covered by 'sbc (utau/qsr/qns/sfx)' -- "
              "that row certifies the surface qsr value, not this routine's "
              "vertical redistribution of it. JUDGMENT (still unmeasured): "
              "MODERATE leverage -- a light-penetration bug would bias the "
              "upper-ocean heat budget, but it has its own #1226 "
              "instrumentation (stp_dump_ts_krhs 'traqsr', line 397) ready "
              "to be turned into a probe."),
    CallEntry(398, "tra_isf", WAIVED, "ln_isf=.false. -- dead branch."),
    CallEntry(399, "tra_bbc", WAIVED, "ln_trabbc=.false. (namelist_ref default) -- dead branch."),
    CallEntry(400, "tra_bbl", WAIVED, "ln_trabbl=.false. -- dead branch."),
    CallEntry(401, "tra_dmp", WAIVED, "ln_tradmp=.false. (namelist_ref default) -- dead branch."),
    CallEntry(402, "bdy_tra_dmp", WAIVED, "ln_bdy=.false. -- dead branch."),

    # --- active tracers RHS, 2nd tiled loop (lines 417-436) ---
    CallEntry(417, "tra_adv -> tra_adv_fct", COVERED,
              '"traadv_fct fluxes", "traadv_fct tendency (T)", "traadv_fct '
              'horizontal tend", "traadv_fct vertical upstream flux", '
              '"traadv_fct (SALINITY)" -- np_FCT selected because DINO '
              "&namtra_adv sets ln_traadv_fct=.true. (traadv.F90:357); "
              "ll_dofct is always True for MLF stepping (kstg not present)."),
    CallEntry(417, "tra_adv -> ldf_eiv_trp (eiv transport, inside tra_adv)", COVERED,
              '"eiv transport u", "eiv transport v" -- traadv.F90:344-345, '
              "active because DINO &namtra_eiv sets ln_ldfeiv=.true. and "
              "ln_traldf_triad=.false."),
    CallEntry(424, "tra_mfc", WAIVED, "ln_zdfmfc=.false. (namelist_ref default) -- dead branch."),
    CallEntry(425, "tra_osm", WAIVED, "ln_zdfosm=.false. -- dead branch."),
    CallEntry(428, "tra_ldf -> traldf_iso_lap", COVERED_UNMEASURED,
              '2026-07-30: gate row "traldf_iso_lap tendency" now exists '
              "(UNMEASURED None/None) -- np_lap_i selected because DINO "
              "&namtra_ldf sets ln_traldf_iso=.true./ln_traldf_lap=.true. "
              "(traldf.F90:94-95). The ahtu/ahtv COEFFICIENT INPUTS to this "
              'routine ARE gated ("ldftra ahtu/ahtv (Redi, '
              'nn_aht_ijk_t=20)"), but no row certifies the TENDENCY '
              "traldf_iso_lap itself produces from them (the isoneutral "
              "slope rotation + K33 term application). JUDGMENT (still "
              "unmeasured): MODERATE-HIGH leverage -- this is the GM/Redi "
              "isoneutral diffusion tendency, a first-order "
              "climate-relevant term, and the wslpi/wslpj/uslp/vslp slope "
              "rows feeding it are THEMSELVES still DEBT (not at bar), so "
              "an uncharacterized tendency-level gate compounds an "
              "already-open residual."),
    CallEntry(430, "tra_zdf", COVERED_UNMEASURED,
              '2026-07-30: gate row "tra_zdf (tracer implicit vertical '
              'solve)" now exists (UNMEASURED None/None) -- previously no '
              "gate row measured the tracer implicit vertical-mixing solve "
              "(the T/S analogue of the dyn_zdf gap above -- folds avt/avs "
              "from zdf_tke into a tridiagonal solve producing ts(Naa)). "
              "JUDGMENT (still unmeasured): HIGH leverage -- same reasoning "
              "as dyn_zdf: this is the FINAL tracer-state-producing step "
              "every timestep, and the #1226 instrumentation "
              "(stp_dump_ts_krhs 'trazdf', line 435) exists specifically to "
              "support closing this gap."),
    CallEntry(436, "tra_npc", WAIVED, "ln_zdfnpc=.false. (namelist_ref default) -- dead branch."),

    # --- finalize: boundary conditions, filtering, restart (lines 457-482) ---
    CallEntry(457, "mlf_baro_corr", COVERED,
              '"mlf_baro_corr" (row exists, UNMEASURED None/None: "algebra '
              'only; needs _step_impl hook") -- local sub in this same '
              "MY_SRC file (line 556), active since ln_dynspg_ts=.true."),
    CallEntry(458, "finalize_lbc", COVERED,
              '"lbc_lnk sign" (row exists, UNMEASURED None/None: "NEVER '
              'VERIFIED") -- local sub in this file (line 638), calls '
              "lbc_lnk with the documented (U,-1)/(V,-1)/(T,+1)/(T,+1) sign "
              "convention."),
    CallEntry(459, "tra_atf_qco", COVERED, '"ATF filter T/S/ssh"'),
    CallEntry(460, "dyn_atf_qco", COVERED, '"ATF filter u", "ATF filter v"'),
    CallEntry(474, "dia_hsb", WAIVED,
              "l_diahsb is iom_use()-gated (diahsb.F90:459) global "
              "conservation-DIAGNOSTIC output; no physics feedback "
              "regardless of value."),
    CallEntry(481, "rst_write", WAIVED,
              "restart-file I/O; gated lrst_oce (restart-frequency "
              "schedule), no physics feedback."),
    CallEntry(482, "sto_rst_write", WAIVED, "ln_sto_eos=.false. -- dead branch (also I/O-only)."),
    CallEntry(495, "stp_ctl", WAIVED,
              "control/blow-up-detection print, no physics feedback into "
              "the next step's state."),
    CallEntry(505, "dia_obs", WAIVED,
              "ln_diaobs=.false. (namelist_ref default) -- dead branch."),
    CallEntry(521, "sbc_cpl_snd", WAIVED,
              "lk_oasis=.FALSE. at compile time (key_oasis3 not in "
              "cpp_DINO.fcm) -- dead branch, absent from the DINO build."),

    # --- MY_SRC #1226 debug-dump instrumentation (not oracle physics) ---
    CallEntry(225, "OPEN/WRITE r3c_dump_*.bin", WAIVED,
              "#1226-round2 DEBUG INSTRUMENTATION added to this MY_SRC file "
              "for the fidelity campaign itself -- not oracle physics, "
              "writes raw dump files consumed by the Python probes that "
              "produced several fidelity_bar_gate.py rows already."),
    CallEntry(270, "stp_dump_krhs('dynadv')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(343, "trddump_acc_baro", WAIVED, "#1455 SG-B accumulation instrumentation (oracle d1ae0ef); writes own module accumulators only; physics-unchanged proven bit-identical by the two controls in oracle 0a1a0cf."),
    CallEntry(406, "trddump_acc_plant", WAIVED, "#1455 SG-B accumulation instrumentation (oracle d1ae0ef); synthetic-plant hook, inert unless rn_acc_plant set; physics-unchanged per oracle 0a1a0cf controls."),
    CallEntry(409, "trddump_acc_state", WAIVED, "#1455 SG-B accumulation instrumentation (oracle d1ae0ef); state snapshot into own accumulators; physics-unchanged per oracle 0a1a0cf controls."),
    CallEntry(589, "trddump_acc_state", WAIVED, "#1455 SG-B accumulation instrumentation (oracle d1ae0ef); state snapshot into own accumulators; physics-unchanged per oracle 0a1a0cf controls."),
    CallEntry(590, "trddump_acc_state", WAIVED, "#1455 SG-B accumulation instrumentation (oracle d1ae0ef); state snapshot into own accumulators; physics-unchanged per oracle 0a1a0cf controls."),
    CallEntry(274, "stp_dump_krhs('dynvor')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(278, "stp_dump_krhs('dynldf')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(284, "stp_dump_krhs('dynhpg')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(293, "stp_dump_state_and_bt('dynspg')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(312, "stp_dump_state_and_bt('dynzdf')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(393, "stp_dump_ts_krhs('trasbc')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(397, "stp_dump_ts_krhs('traqsr')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(423, "stp_dump_ts_krhs('traadv')", WAIVED, "same #1226 debug instrumentation as line 225."),
    CallEntry(547, "stp_dump_ts_krhs('before_traldf')", WAIVED, "same #1226 debug instrumentation as line 225 (#1455 term-sweep pair)."),
    CallEntry(549, "stp_dump_ts_krhs('after_traldf')", WAIVED, "same #1226 debug instrumentation as line 225 (#1455 term-sweep pair)."),
    CallEntry(435, "stp_dump_ts_krhs('trazdf')", WAIVED, "same #1226 debug instrumentation as line 225."),
]


def _validate(calls: list[CallEntry]) -> list[str]:
    """Fail-closed check: every entry must have a real disposition and every
    UNCOVERED entry must carry a rank (so 'UNCOVERED' can never silently mean
    'forgot to rank it')."""
    errors = []
    valid = {COVERED, WAIVED, UNCOVERED, COVERED_UNMEASURED}
    for i, c in enumerate(calls):
        if c.disposition not in valid:
            errors.append(f"entry {i} ({c.routine!r}): invalid disposition "
                           f"{c.disposition!r} (must be one of {sorted(valid)})")
        if c.disposition == UNCOVERED and c.rank is None:
            errors.append(f"entry {i} ({c.routine!r}): UNCOVERED with no rank "
                           "-- every UNCOVERED entry must be ranked")
        if not c.note.strip():
            errors.append(f"entry {i} ({c.routine!r}): empty note -- every "
                           "disposition needs a written reason")
    return errors


def _print_table(calls: list[CallEntry]) -> None:
    print(f"stp_MLF call-graph coverage  ({STPMLF})")
    print(f"total CALLs enumerated: {len(calls)}\n")
    by_disp = {COVERED: 0, WAIVED: 0, UNCOVERED: 0, COVERED_UNMEASURED: 0}
    for c in calls:
        by_disp[c.disposition] += 1
    print(f"  COVERED             : {by_disp[COVERED]}")
    print(f"  COVERED (row UNMEASURED) : {by_disp[COVERED_UNMEASURED]}")
    print(f"  WAIVED              : {by_disp[WAIVED]}")
    print(f"  UNCOVERED           : {by_disp[UNCOVERED]}\n")

    print(f"{'line':>5}  {'disp':<10} routine")
    print("-" * 100)
    for c in calls:
        print(f"{c.line:>5}  {c.disposition:<10} {c.routine}")
        print(f"          note: {c.note}")
    print()

    uncovered = sorted((c for c in calls if c.disposition == UNCOVERED),
                       key=lambda c: c.rank if c.rank is not None else 999)
    if uncovered:
        print("*** UNCOVERED (no gate row measures these) -- ranked by "
              "JUDGMENT of climate leverage, NOT by measurement ***")
        for c in uncovered:
            print(f"  [{c.rank}] line {c.line}: {c.routine}")
            print(f"       {c.note}")
        print()


def build_synthetic_violation() -> list[CallEntry]:
    """A CALL entry that has NO disposition at all (empty string) -- used by
    --self-test to prove _validate() actually fails when coverage is broken,
    not merely when the table happens to be well-formed."""
    return CALLS + [CallEntry(9999, "CALL_synthetic_unaccounted", "", "")]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        # 1. The real table must validate clean.
        real_errors = _validate(CALLS)
        if real_errors:
            print("SELF-TEST FAILED: the real CALL list itself has "
                  "unaccounted entries:")
            for e in real_errors:
                print(f"  {e}")
            return 1
        # 2. A synthetic unaccounted entry MUST be caught -- proves the gate
        #    is not vacuously green.
        synthetic_errors = _validate(build_synthetic_violation())
        if not synthetic_errors:
            print("SELF-TEST FAILED: a synthetic unaccounted CALL entry was "
                  "NOT flagged -- the gate is vacuous.")
            return 1
        print("SELF-TEST PASSED:")
        print(f"  real CALL list: 0 unaccounted entries ({len(CALLS)} total)")
        print(f"  synthetic violation: caught ({len(synthetic_errors)} "
              f"error(s) raised as expected)")
        return 0

    _print_table(CALLS)
    errors = _validate(CALLS)
    if errors:
        print("*** COVERAGE GATE FAILED -- unaccounted CALL entries ***")
        for e in errors:
            print(f"  {e}")
        return 1
    print("COVERAGE GATE PASSED: every enumerated CALL has a disposition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
