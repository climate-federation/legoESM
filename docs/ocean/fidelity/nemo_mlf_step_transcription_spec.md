# nemo_mlf step transcription — implementation spec (P0)

Produced 2026-08-07 by the design lane from the term+join audits. DECISIONS RESOLVED
(maintainer-delegated, recorded in the git log): (1) mlf_baro_corr WAIVED-with-citation
(W1a immaterial, YAGNI); (2) A/B via env-gated driver override, named recipe only at P5;
(3) finalize_lbc commit-point resolved by probe in P1; (4) nemo_kmm divisor HARD-REQUIRED
under nemo_mlf (transcription semantics; the standalone-A/B verdict governs only the
existing leapfrog card); (5) budget tolerance pulled from last recorded run at P3.

# SPEC: `nemo_mlf` step — line-by-line transcription of `stpmlf.F90`

**Status:** design only, no code changed. Read-only lane per task instructions.
**Sources cited:** `stpmlf.F90` (NEMO, `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/stpmlf.F90`), `stpmlf_call_coverage.py` (the join/term audit tool — 106-CALL table, `scripts/validate/ocean_fidelity/dino_1226/stpmlf_call_coverage.py`), current reimplementation `_leapfrog_step` (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6906-7390`), plan doc (`docs/ocean/fidelity/nemo_faithful_ocean_implementation_plan.md`).

## 1. The step skeleton

Rows are `stpmlf_call_coverage.py`'s `CALLS` list (COVERED / COVERED_UNMEASURED entries only; N/A = WAIVED rows dropped per instructions, cited by line for completeness). Format: **NEMO `stpmlf.F90:line`** | **routine** | **lego kernel** | **time levels in→out** | **injection point** | **governing flag**.

| # | NEMO line | routine | lego kernel (file:line / config) | levels in→out | injection point | governs |
|---|---|---|---|---|---|---|
| 1 | `170` | `sbc` | surface-forcing input (`freshwater`/`surface_forcing` args, unchanged) | Nbb,Nnn in → forcing out | pre-RHS | always on |
| 2 | `184-185` | `eos_rab` (Nbb, Nnn) | `eos.compute_ocean_rho`/`eos_rab` kernel, gate row `"eos_rab beta/alpha"` COVERED (`stpmlf_call_coverage.py:113`) | Nbb in, Nnn in → rab_b/rab_n | pre-vertical-physics | `eos.*` |
| 3 | `186-187` | `bn2` (Nbb, Nnn) | `bn2` kernel, gate row `"bn2 (rn2b)"` COVERED (`:119`) | Nbb,Nnn in → rn2b/rn2 | pre-vertical-physics | — |
| 4 | `190` | `zdf_phy → zdf_drg` | bottom-drag kernel, COVERED (`:125`) | Nnn | RHS-at-Nnn | `zdf_drag_in_matrix`/config |
| 5 | `190` | `zdf_phy → zdf_mxl` | MLD (nmln) kernel, COVERED (`:129`) | Nnn | diagnostic feeding TKE | `nn_htau=1` (kamm card) |
| 6 | `190` | `zdf_phy → zdf_tke` | prognostic TKE kernel, COVERED (`:131`) | Nnn (+ TKE carry) | produces K_v/A_v for implicit-solve | `vertical_mixing.tke.prognostic` |
| 7 | `190` | `zdf_phy → zdf_evd` | enhanced-diffusivity fold, COVERED as part of TKE composite (`:136`) | Nnn | folds into avt/avm | `ln_zdfevd` equiv |
| 8 | `199` | `eos`(Nbb, in-situ) + `ldf_slp` | isoneutral-slope kernel, COVERED (`:151-163`, wslpi/wslpj/uslp/vslp rows) | **Nbb** density+N² in → slopes | RHS-at-Nnn (feeds tra_ldf/GM) | `gm_redi.*`, `l_ldfslp` equiv |
| 9 | `203` | `ldf_tra` | Redi/GM coefficient kernel (`ahtu`/`ahtv`/`aeiu`), COVERED (`:164-168`) | Nbb | coefficient, feeds tra_ldf | `nn_aht_ijk_t=20` equiv |
| 10 | `204` | `ldf_dyn` | momentum lateral-viscosity coefficient, COVERED_UNMEASURED (`:169-180`) | Nbb | coefficient, feeds dyn_ldf | `nn_ahm_ijk_t=20` equiv |
| 11 | `214` | `ssh_nxt` (incl. `div_hor`) | ssh/hdiv kernel, COVERED (`:184`) | Nbb,Nnn → **Naa** ssh | state-commit (ssh only) | barotropic solver |
| 12 | `216` | `dom_qco_r3c` (1st) | thickness-ratio kernel, COVERED (`:185-187`) | Naa ssh → r3t/r3u/r3v(Naa) | state-commit | z* always-on |
| 13 | `244` | `wzv` (1st) | vertical-velocity-from-continuity kernel, COVERED_UNMEASURED (`:194-203`) | Nbb,Nnn,Naa → ww | RHS-at-Nnn (feeds tra_adv/dyn_zad) | — |
| 14 | `246` | `eos` (Nnn, in-situ+rhop) | density kernel for HPG, WAIVED as own row but traced into dyn_hpg rows (`:207-212`) | **Nnn** | RHS-at-Nnn | — |
| 15 | `265` | `dyn_adv → dyn_keg+dyn_zad` | KEG+ZAD kernels, COVERED (`:220-222`) | Nnn | RHS accumulate (Nrhs) | `ln_dynadv_vec` equiv |
| 16 | `271` | `dyn_vor → vor_een` | EEN vorticity kernel, COVERED (`:223-226`) | Nnn | RHS accumulate | `vorticity_scheme="een_total"` |
| 17 | `275` | `dyn_ldf → dynldf_lev_lap` | lateral-friction kernel, COVERED (`:227-230`) | **Nbb** (forward-in-time, NOT Nnn) | RHS accumulate | dissipative-scope |
| 18 | `280` | `dyn_hpg → hpg_sco` | HPG kernel, COVERED (`:232-234`) | Nnn | RHS accumulate | `ln_hpg_sco` equiv |
| 19 | `288` | `dyn_spg → dyn_spg_ts` | split-explicit barotropic solver, COVERED (`:237-243`) | Nbb,Nnn,Nrhs → **Naa** u,v,ssh,u_b,v_b | state-commit (barotropic mode + eta) | `explicit_substep` |
| 20 | `302` | `div_hor` (2nd, time-split) | same kernel as row 11, COVERED (`:246-249`) | Naa | state-commit | — |
| 21 | `303` | `dom_qco_r3c` (Naa, +r3f) | same kernel as row 12, COVERED (`:250-253`) | Naa | state-commit | — |
| 22 | `305` | `dyn_zdf` (implicit) | implicit momentum-vertical-mixing solve, COVERED_UNMEASURED (`:254-268`) | Nrhs (RHS), avm → **Naa** u,v | **implicit-solve, folds barotropic-drag removal** | `implicit_vertical_mixing` |
| 23 | `315` | `wzv` (2nd) | same kernel as row 13, COVERED_UNMEASURED (`:269-273`) | Naa | state-commit (diagnostic ww) | `explicit_substep` |
| 24 | `387` | `tra_sbc` | surface tracer BC, COVERED_UNMEASURED (`:336-346`) | **Nnn** | RHS-at-Nnn (Krhs) | `surface_tendency_placement="leapfrog_rhs"` (ALREADY BUILT — see §2) |
| 25 | `394` | `tra_qsr` | penetrative-SW kernel, COVERED_UNMEASURED (`:347-359`) | Nnn | RHS-at-Nnn | `ln_traqsr`/`ln_qsr_2bd` equiv |
| 26 | `417` | `tra_adv → tra_adv_fct` | FCT/Zalesak tracer advection, COVERED (`:367-372`) | Nnn (FCT bounds base = **Nbb** per `_fct_tracer_before`, already built) | RHS accumulate | `tracer_advection="fct"` |
| 27 | `417` | `tra_adv → ldf_eiv_trp` | GM bolus-transport-inside-advection, COVERED (`:373-376`) | Nnn | RHS accumulate | `ln_ldfeiv` equiv |
| 28 | `428` | `tra_ldf → traldf_iso_lap` | isoneutral-Redi tendency, COVERED_UNMEASURED (`:379-393`) | **Nbb** (forward-in-time) | RHS accumulate | dissipative-scope |
| 29 | `430` | `tra_zdf` | implicit tracer-vertical-mixing solve, COVERED_UNMEASURED (`:394-404`) | Nrhs, avt → **Naa** T,S | **implicit-solve, folds RHS into e3t(Kaa)·T(Kaa) matrix** | `implicit_vertical_mixing` |
| 30 | `457` | `mlf_baro_corr` | barotropic-mode-vs-3D-depth-mean reconciliation, COVERED (row exists, "algebra only", `:408-411`) | **Naa** u,v (post dyn_zdf) → **Naa** u,v corrected | **post-implicit-solve state-commit** — currently MISSING as an explicit call (see §6, W1a) | new: dispositive per W1a finding |
| 31 | `458` | `finalize_lbc` | halo/sign-convention application, COVERED (row exists, UNMEASURED, `:412-416`) | Naa | state-commit | halo exchange (always) |
| 32 | `459` | `tra_atf_qco` | tracer Robert-Asselin filter (thickness-weighted content form), COVERED (`:417`) | Nbb,Nnn,Naa → filtered Nnn (next Nbb) | filter, produces next-step Nbb | residual #2 (CLOSED, already built) |
| 33 | `460` | `dyn_atf_qco` | momentum Robert-Asselin filter (plain velocity form), COVERED (`:418`) | Nbb,Nnn,Naa → filtered Nnn | filter, produces next-step Nbb | residual #2's momentum sibling |
| 34 | `361` | `ssh_atf` | ssh Robert-Asselin filter, COVERED_UNMEASURED (`:310-321`) | Nbb,Nnn,Naa → filtered Nnn | filter | plain-RA, already built |
| 35 | `362` | `dom_qco_r3c` (filtered) | same kernel as row 12/21, COVERED (`:322-324`) | filtered ssh → r3t_f/r3u_f/r3v_f | state-commit | — |
| 36 | `472-474` | (inline) r3t/r3u/r3v(Nnn) ← filtered | thickness-ratio carry-forward, no separate kernel | filtered → Nnn | state-commit | — |
| 37 | `478-481` | (inline) time-level swap | `Nrhs=Nbb; Nbb=Nnn; Nnn=Naa; Naa=Nrhs` | — | **the return contract**: caller receives Naa-as-new-Nnn, filtered-Nnn-as-new-Nbb | structural, not a kernel |

Rows dropped as N/A (WAIVED, dead branches / I/O / diagnostics per the audit): lines 124-169 (I/O, calendar, dead-branch forcing), 177-178 (stochastic EOS off), 208 (BBL off), 218/245/319 (spg_exp/wAimp dead branches for DINO), 248-253 (assim/damping off), 279 (OSMOSIS off), 325/331 (diurnal-layers/GEOMETRIC-EKE off), 336-356 (diagnostics), 398-402 (isf/bbc/bbl/dmp off), 424-425 (mfc/osm off), 436 (npc off), 474-505/521 (I/O, restart, coupler-send).

**What the plan doc's join audit already verified as MATCH — cited, not re-derived:**
- Row 30 (`mlf_baro_corr`): **W1a RESOLVED** — the plan doc (`nemo_faithful_ocean_implementation_plan.md:39-48`) certifies this is IMMATERIAL for the kamm card specifically (no depth-mean source exists in lego's solve under `surface_stress_implicit=False`); `mlf_baro_corr_probe.py` committed, round-trip test passes. **Not** a MATCH in the sense of "already transcribed" — a MATCH in the sense of "provably a no-op here", so the skeleton can OMIT it for the kamm card without loss, but must still be a real call (guarded no-op) for any future card where `surface_stress_implicit=True`.
- Row 22 (`dyn_zdf`) / Row 29 (`tra_zdf`): **W2 RESULT** (`:62-88`) — target algebra CONFIRMED MATCH (both fold RHS into `e3t(Kaa)T(Kaa)=e3t(Kbb)T(Kbb)+2dt·e3t(Kmm)·RHS`), `ah_wslp2`/GM-Redi-vertical-term threading CONFIRMED MATCH (config-gated, RETRACTED as a gap). The one confirmed DIFF is the **implicit divisor time-level** (`e3w(Kaa)` vs NEMO's `e3w(Kmm)`) — this is §6 risk item.

## 2. Tendency-interface delta

Per-term status against the `return_rate`/explicit-arg pattern (`surface_tendency_placement="leapfrog_rhs"`, `_external_tracer_rate`, `_fct_tracer_before`, `_barotropic_before_state` — all in `_step_impl` signature, `ocean_model_latlon_cgrid.py:2841-2849`):

**Already conformant** (state the term, cite the kwarg):
- Row 8 (`ldf_slp`): consumes Nbb density/N² explicitly — `_step_impl` already reads `entry_state.T_before`/`N_before` per the `_n2_nemo_before_tracers` hook (`:4631`).
- Row 17/28 (`dyn_ldf`/`tra_ldf` at Nbb): conformant via the **existing two-pass hack** (`_ab2_scope_override="advective"` + a second `_step_impl` call on `nbb`, `_leapfrog_step:7114-7123`) — this is the thing being REPLACED by the transcription, not a term needing a new kwarg. Under `nemo_mlf`, this becomes ONE pass with the dissipative RHS terms evaluated on `state_before` inline, not a second full `_step_impl` invocation.
- Row 19 (`dyn_spg_ts`): conformant via `_barotropic_before_state` (seeds from Nbb, already built, residual #1 CLOSED).
- Row 24 (`tra_sbc`): conformant via `_external_tracer_rate` (residual placement fix, already built, PR history `fb059a9cb`/`3207405e1`).
- Row 26 FCT base: conformant via `_fct_tracer_before` (already built).
- Rows 32-34 (Asselin filters): conformant — these are POST-solve filter steps already isolated as their own code block (`_leapfrog_step:7279-7324`), not tendency injections; no interface change needed, just re-sequencing relative to row 30.

**NOT yet conformant — needs a new return_rate-style signature change:**
- Row 22 (`dyn_zdf`): `_apply_implicit_vertical_mixing` (`:5252`) currently **mutates/returns full state** (`naa`), not a `(rate, diagnostics)` pair — it IS the state-commit step by design (matches NEMO's own contract: `dyn_zdf` folds RHS into the solve, it does not return a separable rate). **No change needed** — this term's contract in NEMO is itself "solve produces state", so `return_rate=True` does not apply; the delta here is purely the **divisor time-level** (§6), not the interface shape.
- Row 29 (`tra_zdf`): same reasoning — `tra_zdf` is a solve-to-state routine in NEMO too. Interface already matches the CONTRACT (state-commit), no `return_rate` change needed. Flag the **W2 divisor gap** (`implicit_vmix_dzw_slot`, needs new `nemo_kmm` slot value per plan doc `:74-76`) as the one real signature-adjacent addition: a new enum value on an EXISTING kwarg, not a new kwarg.
- Row 30 (`mlf_baro_corr`): **currently has no lego call at all** for the kamm card (proven immaterial, §1). For `nemo_mlf` to be a literal transcription rather than "transcription except this one call", it needs an explicit **guarded pass-through kernel**: `mlf_baro_corr(naa_after_dyn_zdf, barotropic_mode_from_dyn_spg_ts) -> naa_corrected`, active only when `surface_stress_implicit=True` (currently never set True outside `explicit_substep`, which already `raise`s against it per `_validate_config`). DECISION-NEEDED: build this as a real (currently-always-no-op) kernel now for transcription completeness, or WAIVE it explicitly in the `nemo_mlf` card with a comment citing W1a, deferring the kernel until a card sets `surface_stress_implicit=True`. Recommend the latter (YAGNI — building a kernel with zero live callers fails the "no unrequested abstraction" bar) but it IS a DECISION, not a default.
- Rows 4-7 (`zdf_phy` composite): currently one fused call inside `_step_impl`'s tendency stage — conformant in the sense that it produces `K_v_phys`/`A_v_phys`/`tke_source` as explicit returned values already threaded through `_leapfrog_step:7080-7101`. No change needed.

**Summary**: the ONLY term requiring a genuine interface change is row 30 (`mlf_baro_corr`, currently absent) and the divisor enum value on `implicit_vmix_dzw_slot` (row 29, W2). Everything else already has a signature-conformant hook — the gap is entirely in **composition order**, not term interfaces. This matches the maintainer's framing exactly: the kernels are done, the orchestration is not.

## 3. State carrier

**Already threaded** (no new fields needed): `Nbb` = `{u,v,T,S,eta}_before` on `LatLonCGridOceanState` (`state.py:468-472`) — populated every step by `_leapfrog_step`'s Asselin-filter tail (`:7318-7324`) and by the cold-start branch (`:7014-7025`). `Naa` = the function's own return value (no separate field — it IS the next `state`). `Nrhs` = never materialized as state; it's the tendency accumulator inside `_step_impl`/`tendencies_with_diagnostics`, matching NEMO's own `ts(Krhs)`/`uu(Nrhs)` semantics (zeroed and rebuilt each step, not carried).

**TKE carry**: `state.tke` (3-D Field, W-grid interior interfaces, `state.py:423-429`) — already exists, already threaded through `_apply_implicit_vertical_mixing` as `tke_old`/`return_tke` (`_leapfrog_step:7236,7246-7247`). No new field.

**e3/ssh levels**: derived on-the-fly via `compute_layer_thickness(eta, H_bathy, z_coord)` at whichever eta is in scope (`e3t_now`, `e3t_bef`, `e3t_aft`, `e3t_flt` — all computed inline in the Asselin-filter block, `:7302-7310`). This is NEMO's own "recompute e3 from ssh on read" convention (z*), not a stored array — CONFIRMED matching NEMO's QCO/`dom_qco_r3c` design (recomputes `r3t`/`r3u`/`r3v` from ssh each call, rows 12/21/35 above), so no new carrier needed; this is a FAITHFUL-but-different mechanism already correctly chosen (§6 risk register keep-decision).

**New fields needed for `nemo_mlf`**: **none identified.** The state already carries every NEMO time-level array the transcription touches. If row 30's `mlf_baro_corr` kernel is built (§2 DECISION-NEEDED), it reads/writes existing `u`,`v` fields only — no new field.

**SegmentCarry**: N/A — ocean's leap-frog does NOT use `packages/coupler/legoesm/driver/compiled_segments.py::SegmentCarry` (that's the atmosphere/coupler hot-loop carry). Ocean's before-state IS the state pytree itself. No SegmentCarry-change checklist applies. (Verified: `grep SegmentCarry` in ocean dynamics returns nothing; the only `SegmentCarry` class in the repo is the coupler one.)

## 4. Config surface

**Dispatch site**: `_step_jitted` (`ocean_model_latlon_cgrid.py:6342-6421`), which already validates `outer_integrator in ("forward_euler", "ab2", "leapfrog")` and `raise ValueError` on unknown (`:6166-6169`) — dispatch-hardening-compliant pattern to extend, not replace.

**Proposed**: add `"nemo_mlf"` as a fourth literal:
```python
_oi = getattr(self.config, "outer_integrator", "forward_euler")
if _oi not in ("forward_euler", "ab2", "leapfrog", "nemo_mlf"):
    raise ValueError(...)
...
elif _oi == "nemo_mlf":
    new_state = self._nemo_mlf_step(state, dt, freshwater=..., ...)
elif _oi == "leapfrog":
    new_state = self._leapfrog_step(...)   # RETAINED, unchanged
```
- **Default untouched**: `outer_integrator` default stays `"forward_euler"`; `"leapfrog"` (the current `_leapfrog_step` reimplementation) stays the default for the kamm card unless/until `nemo_mlf` clears the verification ladder (§5) and a maintainer decision flips the card default. Both paths coexist — REQUIRED for the A/B in §5a.
- **Validate-strict membership**: `outer_integrator` is already validated at construction (`_validate_config`, `:1955-1958` region) — add `"nemo_mlf"` to that same membership set (same file, same function), and it needs the same construction-time compatibility raises that `"leapfrog"` currently has (e.g. `ab2_scope="total"` requirement at `:1987`, `surface_stress_implicit=False` implications) — audited per-guard, not blanket-copied, since `nemo_mlf`'s single-pass composition may relax some of `_leapfrog_step`'s two-pass-specific guards (e.g. the `ab2_scope` override becomes moot if `nemo_mlf` doesn't call `_step_impl` in scoped-override mode at all — DECISION-NEEDED, see §7).
- **Dispatch-hardening test**: `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` grows by one entry for this new `_oi` guard (grow-only per CLAUDE.md).
- **Kamm-card migration plan**: the DINO kamm card (`dino.py`, `nemo_dino_kamm_mlf` recipe) currently sets `outer_integrator="leapfrog"`. Add a **second** named recipe variant (e.g. `nemo_dino_kamm_mlf_transcribed` or a `step_composition="nemo_mlf"` sub-flag on the same recipe — DECISION-NEEDED which naming) that is identical in every OTHER config field (same `ab2_scope`, same `barotropic_time_filter`, same `implicit_vmix_e3t_now_divisor`) and differs ONLY in `outer_integrator`. This is what makes §5a's bit-comparison a controlled comparison (CLAUDE.md controlled-comparison rule: change ONE variable). The OLD `"leapfrog"` path is RETAINED, not deprecated, until the A/B (§5) either shows parity (promote `nemo_mlf` to default, keep `leapfrog` for reference) or shows `nemo_mlf` fixes the form-stress deficit (promote immediately, `leapfrog` becomes the historical/reference implementation, not removed — CLAUDE.md "no deprecated backward-compat wrappers" applies to dead code, not to a scientifically-meaningful alternate implementation kept for A/B).

## 5. Verification ladder

Rung order, cheapest/most-diagnostic first. Instruments already exist (cited); "expected outcome" states what PASS looks like, "failure meaning" states what a rung failure localizes.

**(a) Bit-comparison vs current `leapfrog` step, single-step, matched config.**
- Instrument: new — a direct pytest comparing `_leapfrog_step(state, dt, ...)` vs `_nemo_mlf_step(state, dt, ...)` output on an identical input state, same config except `outer_integrator`.
- Where they SHOULD agree: any term/join the plan doc's audit already certifies as MATCH with no ordering difference — i.e., rows where `_leapfrog_step`'s two-pass composition and NEMO's single-pass composition happen to produce the same arithmetic (e.g. row 24 `tra_sbc` placement, row 19 `dyn_spg_ts` seeding — both already NEMO-faithful in `_leapfrog_step`). They should NOT bit-agree wherever `_leapfrog_step`'s reimplementation genuinely diverges from NEMO's order — that is precisely rows 17/28 (dissipative-at-Nbb via a **second full `_step_impl` pass** vs NEMO's single-pass-with-Nbb-read) and row 22/29's divisor (§6 W2 gap) and row 30 (`mlf_baro_corr` absent vs present-but-provably-no-op).
- Expected outcome: **NOT bit-identical** — a clean disagreement in exactly the rows named above, DAT-agreement everywhere else. A surprise disagreement outside that predicted set is a new bug in the transcription; a surprise AGREEMENT in a predicted-divergent row means the current reimplementation is more faithful than believed (retract the plan doc's characterization for that row, cite the new evidence — Rule 11 discipline).
- Failure meaning: if MORE rows diverge than predicted, the transcription has an undocumented composition error — audit against the skeleton table in §1 before proceeding.

**(b) Replay suite per-step vs NEMO restarts.**
- Instrument: `scripts/validate/ocean_fidelity/dino_1226/multistep_replay.py` (already exists; feeds `tests/ocean/unit/test_dino_1492_multistep_replay.py`, wired per commit `0c92bd468`).
- Expected outcome: `nemo_mlf` should show EQUAL OR BETTER per-step correlation/ratio against NEMO's own `stp_dump_*` dumps than `leapfrog` does at the SAME rows (rows 17/22/28/29/30 specifically — the ones §5a predicts diverge). This is the rung that actually tests "is the transcription MORE faithful", not just "is it different."
- Failure meaning: if `nemo_mlf` matches NEMO's dumps WORSE than `leapfrog` at any row, the transcription has a bug (wrong time-level read, wrong sign, wrong divisor) — go back to that row in §1, re-derive from `stpmlf.F90` line-by-line, don't hand-wave.

**(c) Budget closure.**
- Instrument: `scripts/validate/ocean_fidelity/dino_1226/global_closure_audit.py`, `tracer_content_conservation.py`, `budget_fullframe.py`/`budget_pointwise.py` (all exist).
- Expected outcome: heat/salt/volume budgets close to the SAME tolerance `leapfrog` already achieves (plan doc doesn't report a specific number here — DECISION-NEEDED: pull the exact current closure tolerance from `budget_fullframe.py`'s last recorded run before setting the bar, per Rule 1b — do not eyeball it).
- Failure meaning: a budget leak that `leapfrog` didn't have localizes to whichever row's composition changed (most likely: the divisor fix, row 22/29, or the tracer-combine reweighting at the transcription's flux-form-inside-implicit-solve boundary — since `nemo_mlf` folds RHS into the solve rather than combine-then-correct, per plan doc W2 `:81-88`).

**(d) 90-day acceptance gate.**
- Instrument: `scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py`, `dino_90d_screen.py`, `kamm_twin_90d.py` (built + demonstrated per plan doc `:119`, commit `1bfb710dd`).
- Expected outcome: `nemo_mlf` passes the SAME acceptance gate `leapfrog` currently passes (ACC 0.09 Sv per plan doc `:119`), at minimum. This rung does NOT test for IMPROVEMENT on the form-stress deficit — that's a longer-horizon question (rung e).
- Failure meaning: a 90-day-horizon-only divergence (not visible in single-step/budget rungs) — points at compounding error in a term whose single-step signature is small but non-zero (candidate: the divisor gap, which the plan doc explicitly frames as O(dη)/step, i.e. small per-step, potentially large accumulated).

**(e) 10-yr from-rest vs NEMO matched-year (the climate-leverage rung).**
- Instrument: `dino_year5_compare.py`, `kamm_run5y_v3.py`, `dino_year_screen_fullframe.py` (exist).
- Expected outcome: THIS is the rung that could show the transcription closing the open form-stress deficit (plan doc §W1) — since W1a/W1b/W1c are ALL join-layer suspects that a literal transcription would, by construction, resolve (a transcription cannot have an ordering-difference bug because there is no reimplemented ordering left to differ). DECISION-NEEDED / stated explicitly: this is a HOPE, not a proven prediction — the plan doc's W1c (flank density-tendency budget) has NOT been run to completion, so we do not yet know whether the form-stress deficit is fully explained by composition-ordering (which `nemo_mlf` fixes structurally) or partly by something else (e.g. a genuinely-still-open term-level DEBT row). State this rung's outcome as informative either way: closes the gap → composition was the cause (major finding); doesn't close it → composition was NOT the (sole) cause, W1c's density-tendency-budget instrument is still needed on TOP of the transcription.
- Failure meaning ("doesn't close the gap"): do NOT read this as "transcription failed" — re-run rung (a)-(d) first to confirm the transcription itself is faithful before concluding the form-stress deficit has a non-compositional cause; only after (a)-(d) all pass does a rung-(e) miss cleanly implicate a different mechanism (Rule 4 ablation logic: the transcription is the "remove X" experiment for composition-ordering as a candidate cause).

## 6. Risk register

Each item: **mechanism** | **keep/transcribe decision** | **why**.

1. **The five residual-fixes in `_leapfrog_step`'s docstring (`:6904-6990`).** All five must be carried forward EXPLICITLY, not silently dropped by virtue of rewriting the function:
   - **#1 barotropic-before-seed** (`_barotropic_before_state`, dynspg_ts.F90:494-503) — **TRANSCRIBE, keep the mechanism.** It's a NEMO transcription itself (`ln_bt_fw=.FALSE.` centred barotropic), so `nemo_mlf` should call `dyn_spg_ts` with the SAME Nbb-seeded call, not rediscover it. Already conformant per §2.
   - **#1b `nemo_boxcar_ab3` barotropic time filter** — **KEEP**, it's the `dyn_spg_ts` internals (AB3 predictor + ssh half-step-back interpolation), unrelated to step-composition; the transcription doesn't touch this file, only its CALL SITE.
   - **#2 thickness-weighted tracer Asselin filter** (`_thickness_weighted_asselin`, `thickness_weighted_tracer_combine`) — **TRANSCRIBE, keep the mechanism**, but re-verify it's invoked in NEMO's actual ORDER: `tra_atf_qco` (row 32) happens AFTER `mlf_baro_corr`/`finalize_lbc` (rows 30-31), which the current `_leapfrog_step` doesn't explicitly sequence (no `finalize_lbc`-equivalent halo step is broken out). DECISION-NEEDED: does lego's halo-exchange/masking (already applied continuously via masks, not a discrete `finalize_lbc` call) need an explicit "commit point" in `nemo_mlf` to match NEMO's ordering, or is it provably order-independent because lego halo-fills are idempotent? Flag for the implementation lane to resolve with a probe, don't assume.
   - **#3 live per-substep barotropic Coriolis** (`barotropic_coriolis_split="live"`) — **KEEP**, internal to `dyn_spg_ts`'s substep loop, same reasoning as #1b.
   - **#4 dissipative-at-Nbb via a second `_step_impl` pass** — **TRANSCRIBE THE PHYSICS (Nbb-evaluated diffusion), REPLACE THE MECHANISM.** This is the highest-risk item: `_leapfrog_step` achieves "diffusion at Nbb" by running the ENTIRE tendency pipeline twice (once at Nnn for advection, once at Nbb for dissipation) and combining. NEMO does it in ONE pass, reading `puu(:,:,:,Kbb)` only for the `dyn_ldf`/`tra_ldf` CALL's argument (line 275/437) while everything else in the same pass reads Nnn. `nemo_mlf` must do the SAME — one tendency pass, with `dyn_ldf`/`tra_ldf` explicitly reading `state.T_before`/`u_before` as their OWN local argument, not a whole-state substitution. This changes the RHS-accumulation internals of `tendencies_with_diagnostics`/`_step_impl` (§2's identified interface gap) — likely the single largest implementation-lane task. Getting this wrong (e.g. reading Nbb for MORE terms than just ldf) would silently change other terms' time level — the exact Rule 1d failure mode from the oracle-fidelity skill.
   - **#5 barotropic substep CFL rescale** (`_barotropic_substep_scale=2`) — **KEEP**, mechanical consequence of `rDt=2dt` vs `dt`; applies identically once `nemo_mlf` also uses `rDt=2dt`.

2. **W2's FCT Nbb-base choice** (`_fct_tracer_before`) — **TRANSCRIBE, already correct.** NEMO's own FCT bounds are computed at whatever level the limiter call receives; `_leapfrog_step` already passes `(T_before, S_before)` correctly per its own comment (`:7090-7094`). No change, just re-verify the SAME call survives the single-pass restructure (risk: if `dyn_ldf`/`tra_ldf`'s Nbb-read merges into the Nnn pass per item 1 above, confirm `tra_adv`'s FCT call still gets `_fct_tracer_before` — don't let the merge silently drop this kwarg).

3. **The FAITHFUL-but-different mechanism: recompute-thickness-on-read (z*/QCO).** Already analyzed in §3 as a MATCH (both NEMO's `dom_qco_r3c` and lego's `compute_layer_thickness` recompute e3 from ssh rather than storing it) — **KEEP as-is**, this is not a risk, it's a confirmed-equivalent design choice; listed here only because the task explicitly asked for it to be enumerated.

4. **W2's implicit divisor time-level** (`e3w(Kaa)` vs NEMO's `e3w(Kmm)`, plan doc `:68-77`) — **TRANSCRIBE per the plan doc's own W2 recommendation**: add a `nemo_kmm` slot value to `implicit_vmix_e3t_now_divisor`/`implicit_vmix_dzw_slot` (config already has this machinery per #428, cited in the plan doc). This is explicitly flagged by the plan doc as a form-stress SUSPECT (§0 of this spec, row 22/29) — highest scientific stakes of any single risk-register item. **DECISION-NEEDED**: does `nemo_mlf` inherit whatever the config's `implicit_vmix_e3t_now_divisor` is set to (i.e., stay a SEPARATE selectable option orthogonal to `outer_integrator`, per the recipe-strategy doctrine of "config selecting shared canonical blocks"), or does `nemo_mlf` HARD-REQUIRE `nemo_kmm` as part of being a literal transcription? Recommend: **hard-require** — a transcription that still permits a non-NEMO divisor stops being a transcription at that row; enforce with a construction-time `raise` (dispatch-hardening pattern, §4) if `outer_integrator="nemo_mlf"` and `implicit_vmix_e3t_now_divisor != "nemo_kmm"`.

5. **`mlf_baro_corr` (row 30) build-or-waive** — see §2's DECISION-NEEDED. Risk if WAIVED: `nemo_mlf` is then "transcription except this one call", contradicting the task's framing of a LINE-BY-LINE transcription; must be documented as an explicit, written, reason-cited WAIVER (oracle-fidelity Rule 1: "Waiving becomes a deliberate act that leaves a reason in the diff") citing W1a, not silently dropped.

6. **Two-pass → one-pass restructure changes JIT/AD shape.** `_leapfrog_step` currently calls `_step_impl` (a large JIT-traced function) TWICE per step. `nemo_mlf` collapsing to one call changes compiled-graph size/shape — likely a PERFORMANCE IMPROVEMENT (fewer traced ops) but must be verified for (a) gradient correctness under `eqx.filter_value_and_grad` (CLAUDE.md `.raw`/non-donating variant rule) since the tendency-pipeline internals are being restructured, not just re-called, and (b) no accidental retrace from a new closure captured inside the restructured function (CLAUDE.md JIT/compilation common mistake: "Python defs in `_single_step` recreated each trace" — audit the implementation for this specifically since it's exactly the kind of refactor that introduces it).

## 7. Effort estimate per phase, with review gates

| Phase | Scope | Effort | Review gate |
|---|---|---|---|
| **P0 — this spec** | Design doc (this deliverable) | done | maintainer read; DECISION-NEEDED items above resolved before P1 starts |
| **P1 — single-pass tendency restructure** | Collapse the two-`_step_impl`-pass composition (risk #1 item 4) into one pass with Nbb-scoped `dyn_ldf`/`tra_ldf` reads; thread `_fct_tracer_before` through the merge (risk #2); NO new config surface yet, build behind a feature branch / private method not yet wired to `outer_integrator` | 3-5 days (the single largest item — touches `tendencies_with_diagnostics`/`_step_impl`'s RHS-accumulation core) | codex/agent adversarial review MANDATORY (CLAUDE.md: dycore/physics edit, multi-file, touches AD/conservation) before merge of ANY of this phase |
| **P2 — divisor + config surface** | `nemo_kmm` slot value (risk #4), `outer_integrator="nemo_mlf"` dispatch + construction-time raises (§4), `mlf_baro_corr` build-or-waive decision executed (risk #5) | 2-3 days | dispatch-hardening test + construction-time raise tests, same-PR |
| **P3 — verification ladder (a)-(c)** | Bit-comparison test, replay-suite run, budget-closure run | 2-3 days (instruments exist, this is running + reading them, not building) | Rule 1b gate script extension: `fidelity_bar_gate.py` gets `nemo_mlf` rows |
| **P4 — verification ladder (d)-(e)** | 90-day acceptance gate run, 10-yr from-rest matched-year comparison | 1-2 weeks compute + analysis (multi-day model runs) | full oracle-fidelity Rule 1-11 discipline on the climate-comparison claim; this is where the form-stress-deficit finding (or non-finding) gets written up |
| **P5 — kamm-card default flip (conditional)** | Only if P3+P4 show `nemo_mlf` ≥ `leapfrog` on every rung: flip the DINO card default; `leapfrog` retained as reference implementation, not deleted | 1 day + review | maintainer sign-off (this is a scientific-default change, not mechanical) |

**Total: ~3-4 weeks**, dominated by P1 (the actual transcription work) and P4 (compute-bound climate verification), consistent with the plan doc's own W1/W2 effort framing ("1 substantial lane... ~1 week" for W2 alone, plus W1's "unknown, days to weeks" for the form-stress resolution this transcription is gambling on resolving structurally).

---

**Open DECISION-NEEDED items requiring maintainer input before P1 starts** (collected from above, not re-argued):
1. §2/§6-5: build a real (currently-inert) `mlf_baro_corr` kernel now, or WAIVE explicitly for the kamm card.
2. §4: naming/mechanism for the kamm-card A/B variant (`step_composition=` sub-flag vs. a second named recipe).
3. §6-1: whether lego needs an explicit `finalize_lbc`-equivalent commit point or can rely on provably-idempotent continuous masking.
4. §6-4: hard-require `nemo_kmm` divisor when `outer_integrator="nemo_mlf"`, or leave it a free-standing orthogonal option (recommended: hard-require, stated as a recommendation not a unilateral choice).
5. §5c: pull the exact numeric budget-closure tolerance from the last `budget_fullframe.py` run rather than eyeballing a bar.
---

## P3 RESULTS (2026-08-07)

### Rung (b) — replay vs NEMO per-step restarts: DIFFERENT BUT EQUALLY ACCURATE
Both arms `8 passed` on the real RUN_TWIN_STEP1 tiles, fp64. The transcription is CONFIRMED
live and materially different from step 1 — lockstepped in ONE process from ONE bridged IC,
arrays diffed directly: max|du| 2.8e-3 at k=1 growing to 9.4e-3 at k=32; max|dT| 1.3e-3 -> 7.1e-2.
**But its error magnitude vs NEMO is statistically indistinguishable from the two-pass path
(slope ratio 1.0000).** Correct phrasing: "different but equally accurate", NOT "more faithful".
- Rows 17/28 (ldf-at-Nbb): an apparent T improvement was **DOWNGRADED CONFIRMED -> PLAUSIBLE**
  — the margin (4.8e-3) is ~7% of the arm-to-arm divergence (7.1e-2) and the sign flips at k=17-19.
- Rows 22/29 (divisor): different state, equal accuracy.
- Row 30 (`mlf_baro_corr`): EQUAL at fp-noise (max|de3t| 1e-9), consistent with W1a's no-op.

**RETRACTION recorded (the lane's own, caught by peer challenge):** an initial "bit-identical"
claim was a rounded ratio of SUMMARY STATISTICS, not an array diff — refuted by its own printed
endpoints. It also compared max|A−N| vs max|B−N| when the two maxima sit at DIFFERENT CELLS,
hiding a 2.8e-3 m/s momentum difference. Both are canonical CLAUDE.md failure modes; fixed by
lockstep array diffing.

### Rung (c) — budget closure: COMPOSITION-NEUTRAL, no new leak
Bar PULLED from the recorded artifact (resolved decision 5), not eyeballed. Harness self-check
passed (injected leak 5.0 -> detected 5.000000); the control reproduced the pre-existing recorded
artifact BIT-IDENTICALLY on state integrals, proving the env-knob edit is a no-op.

| quantity | recorded bar | leapfrog | nemo_mlf | ratio |
|---|---|---|---|---|
| volume | 2.5253e-16 | 2.5253e-16 | 2.5253e-16 | 1.0000 |
| heat | 5.3983e-06 | 5.3983e-06 | 5.3984e-06 | 1.0000 |
| salt | 1.4320e-09 | 1.4320e-09 | 1.4323e-09 | 1.0002 |

Notably the divisor change (`e3w(Kaa)`->`e3w(Kmm)`, the highest-stakes risk-register item) is
**budget-neutral to 4 s.f.** — it changes the implicit-solve matrix semantics without breaking
the fold's flux-form conservation.
**GAP (lane-flagged, rerun queued):** this used `applied_now`, whose known placement defect
leaves a ~5.4e-06 heat residual in BOTH arms — a background 3+ orders above where a composition
leak would live, so the test had little power. The NEMO-faithful config is `nemo_mlf` +
`leapfrog_rhs` (~1e-9 background). Rerun in flight.

### What P3 means for P4
The composition is faithful (no leak) and live (materially different state) — but P3 CANNOT show
it is more accurate, because at a 1-day horizon the fidelity difference is inside noise on every
path. **P4's 10-year climate run is therefore the sole decider** for whether the form-stress
deficit is compositional.

## P4 RESULTS (2026-08-07) — rung (d) real gain; rung (e) HAD NO POWER (design error, corrected)

### Rung (d) — 90-day acceptance gate: NO REGRESSION, material gain (CONFIRMED)
Tally `PASS 1 | FAIL 4 | level 5x` in all three arms; self-checks + non-vacuity passed.

| metric (floor units) | control | +divisor only | nemo_mlf |
|---|---|---|---|
| ACC | 16.9 | 17.8 | **10.2** |
| S-band sigma MAX | 32.9 | 33.2 | **16.4** |
| deep contrast | 4.6 PASS | 4.9 PASS | 4.6 PASS |

**Three-arm design solved the composition/divisor conflation**: the divisor-only arm is
flat-to-adverse, so the gain is the COMPOSITION's.

### Rung (e) — 10-year climate: NO DISCRIMINATING POWER BY CONSTRUCTION
**This was a design error in the brief (mine).** The deficit ONSETS y11-13 and reaches 0.9590 at
y15-20; at y6-10 the CONTROL ITSELF sits at 1.0020 — healthy. Both arms are healthy through y10,
so the experiment could not distinguish them on the phenomenon it was built to test. "The window
cannot show full closure" was written as a caveat when it was fatal to the test's power.

Readouts at y6-10 (for the record, none decisive): ACC window err 0.198 -> 0.066 Sv but paired
t=-1.00 (PLAUSIBLE, not significant, driven by y10 alone); F_topo 1.0020 -> 1.0010, t=-1.39
(NULL, 0.3 sd); contrast NULL; heave MIXED — east improves +1.3 m but **west and mid move AWAY**
(-0.5, +0.6 m), and west is the 53%-concentration flank.

**Divisor CONFIRMED NULL at the climate horizon** (paired t=-0.01) — which calibrates the test
(not insensitive) and makes composition attribution clean.

The lane independently reproduced the single-year trap the brief warned of: y10 F_topo readings
(control 1.0049, divisor-only 1.0002) look favourable and collapse on the window mean.

### ⇒ P4b: extend nemo_mlf y11->y20 (running)
The only readout that can answer the question. Control at y15-20: F_topo 0.9590 (sd 0.0034),
heave -37 m (y15) / -67 m (y20) west. If the west flank stays wrong-signed there, the composition
is NOT the cause and the answer lies in the kernels — which is the evidence that would justify a
full NEMO-JAX kernel port.

## P4b VERDICT (2026-08-07) — THE COMPOSITION IS **NOT** THE CAUSE. CONFIRMED.

20-year nemo_mlf arm (all STABLE, fp64, census exact; y1-y10 **bit-identical** to the 10-year
arm — determinism confirmed). Instrument controls all reproduced exactly before any number
(armB y10 F_topo 1.0049; NEMO y10 ACC 121.07; control window 0.9590 sd 0.0034; control west
heave -36.9/-67.3 m).

| readout, y15-20 window | control | nemo_mlf | direction |
|---|---|---|---|
| **F_topo ratio** | **0.9590** (sd .0034) | **0.9577** (sd .0043) | AWAY (-0.0013, sub-floor; paired-t -2.76 only because the arms are near-identical) |
| ACC window err | +6.003 Sv (66.0 floors) | +6.154 Sv (67.6) | AWAY, +1.7 floors, n.s. (t=+1.13) |
| West heave y15/y20 | -36.9 / -67.3 m | -37.8 / -68.2 m | AWAY; **anomaly PERSISTS and grows, rate unchanged** |
| Upper contrast err y20 | +0.029936 | +0.030181 | marginally adverse |

**The pre-registered criterion fired**: the west-flank sign anomaly (the 53%-concentration flank)
was to decide this, and it persists and slightly amplifies. The two arms are climate-inert
relative to each other — ACC within 0.5 Sv, F_topo within 0.003, heave within 1.1 m over 20 years.

**What P4b supports**: a DIRECTION test — nemo_mlf does not bend the trend toward NEMO.
**What it does NOT support**: any claim about the deficit's equilibrium magnitude (neither arm is
equilibrated at y20 — control F_topo 1.0020 at y6-10 -> 0.9590 at y15-20, no plateau; ACC
overshoot still growing 1.02 -> 1.05), nor ruling out a late-onset effect beyond y20.

### What the transcription IS worth (kept — it is not a failure)
Faithful (budget-neutral at full sensitivity, heat ratio 0.9914), live (max|du| 2.8e-3 from step
1), a real 90-day gate gain attributable to the composition (ACC 16.9 -> 10.2 floors, sigma-max
32.9 -> 16.4, with the divisor-only arm flat), ~2.7x faster, and it exposed TWO latent bugs in
existing code (K33 time-level mix; scan-carry seeding) plus a harness wiring gap. It stays as a
selectable, verified integrator.

### ⇒ Where the deficit must now live: THE KERNELS
Exonerated so far: every operator at same-state (6 exonerations), all four momentum sinks, three
join suspects, and now the step composition. Remaining candidates: **the 26 DEBT rows at
1e-4..1e-6**, whose "climate-inert individually" status rests on acceptance runs that PREDATE the
0.091 Sv noise floor. A systematic 1e-4 over ~230k steps (20 yr x 11520) is not small.
**Most suspicious family: `traadv_fct` (5 rows, ~1e-4..1e-5, CONFIRMED LOCAL to the FCT/Zalesak
chain)** — tracer advection over steep topography at the sill flank is exactly where a limiter
would diverge, and it is the largest DEBT family sitting in the density-transport path.

## traadv_fct EXONERATED 2026-08-07 (zero compute — the evidence already existed)

Launched as the post-P4b prime suspect; the lane found the investigation had **already been run
on this branch a week earlier** (commit `0fe45672d`, #1455) and reported it rather than
re-running. Twice-exonerated:
1. **Not inherited**: ww-substitution (NEMO's own `wzv_dump_ww_call2` into lego's FCT w_half)
   DEGRADES all 5 rows by 3-5 orders (tendency-T corr 0.999991 -> 0.251229) ⇒ the residual is
   LOCAL to lego's Zalesak chain.
2. **Not co-located with the heave**: the clip-status-shift census (107 wet-to-wet faces) puts
   only **4.7%** in the k=8-14 band, with the note stating explicitly the differing cells are
   "NOT dominated by that band or by topo steps specifically" — a direct negative on the
   spatial-co-location test.
3. **The scheme is not the lever at all**: "the FCT limiter is the climate lever" was RETRACTED
   (2026-07-27) — swapping in CENTERED (unlimited) advection reproduces the FCT run to 4 decimal
   places over 5 years, **ACC identical**.

CAVEAT kept: the full-scheme ablation was a 5-year ACC comparison, not the y15-20 F_topo window —
suggestive, not byte-matched. Closing that gap costs a 20-yr centered-advection arm and is judged
not worth it against a twice-exonerated family. The 603-vs-662 clip-count remains UNRECONCILED
debt but is explicitly not topography-tracking.

### PROCESS LESSON (mine)
I launched this lane without first checking whether the campaign had already measured it. With
~2 months of artifacts, **"has this already been measured?" must precede "let's measure it."**
The lane caught it; the cost was one agent instead of a 20-year GPU run.

### The elimination ledger is now nearly total
Exonerated: 6 operators at same-state · 4 momentum sinks · 3 join suspects · the step composition
(P4b) · the entire tracer-advection scheme. What remains for the flank heave:
- **`zdftke composite avt/avm` — corr 0.966 / ratio 1.071, by far the LARGEST gate residual**
  (~7%, vs everything else at 1e-4..1e-6). Note vertmix was exonerated for deep-box HEAT DELIVERY
  by the twin budget, which does NOT exonerate a 7% avt/avm error from shaping flank density.
- the barotropic/`dyn_spg_ts` rows, `dyn_vor EEN`, `dom_qco_r3c`, `dyn_cor_2d`, `dyn_drg_init`
- **the honest alternative**: no single row owns it, and the deficit is emergent from the
  accumulated 1e-4..1e-6 across many rows — in which case the only path is W4, grinding them all
  to the class bars, and re-testing.

## zdftke: THE 7% SURVIVES EXACT TRANSCRIPTION (2026-08-07) — escalation, not a gap

Step-0 evidence inventory (made mandatory after the traadv lane found a week-old answer) fired
its escape clause. **CONFIRMED by git ancestry against the measurement commits**: every stage the
campaign ever identified as a candidate for the composite's 7% is ALREADY exact-transcribed and
ALREADY LIVE on the card that produced the number —
- `tke_buoyancy_sink="nemo_explicit"` (zdftke.F90:495) — card default, `dino.py:1069`
- `tke_dissipation="nemo_1p5_split"` (zdftke.F90:241-242,414,419, incl. the zzd_up/zzd_lw
  tridiagonal assembly) — card default, `dino.py:1063`
- `tke_mxl_choice=3` (lup/ldown envelopes) — independently oracle-verified against a
  from-Fortran NumPy transcription at rtol=1e-12, 7/7 tests
all wired in `775cb72fb`, confirmed an ancestor of both `e0fac585e` (the composite measurement)
and `1b1738352` (the sh2-elimination measurement).

**sh2 is exonerated as the driver** (`1b1738352`): substituting NEMO's OWN dumped sh2 moves corr
only 0.9633→0.9656 and moves the ratio AWAY from 1.0. (sh2's own 3-way probe disagreement remains
open but is known not to explain the composite.) The MXL thread closed separately 2026-08-04
(true `tke_dump_rn2` re-walk, 0.00992→0.00069).

⇒ **This is THE RULE's escalation condition one level up: the exact NEMO match was already made,
for every identified stage, and the row still does not clear.** The lane correctly REFUSED to
invent a new "transcription" of already-faithful code to force the number down.

**One stage was never audited line-by-line: the surface/bottom TKE boundary conditions**
(surface injection/`rn_ebb`, the `nn_htau=1` latitude penetration profile the card sets, bottom
BC, `en_min` floors). Walk launched, with the CHEAP DISCRIMINATOR FIRST: decompose the existing
7% by level — a BC defect must be surface-concentrated (k=0-2). If it is not, the block is
exonerated without any transcription work.

**If the BC block is clean, the campaign's own alternative hypothesis stands**: no single row owns
the deficit; it is emergent from accumulated 1e-4..1e-6 across many rows, and the only path is
W4 — grind every DEBT row to its class bar and re-test.

## zdftke bottom BC: one REAL transcription bug found — and it is INERT in DINO (2026-08-07)

Follow-up to the escalation above. The one un-audited zdftke stage (surface/bottom
BCs) was walked line-by-line. Result: **a genuine transcription defect, plus a
retraction of the attribution that led to it.**

### The bug (CONFIRMED — code, both sides)

`zdftke.F90:282-287` does NOT use the plain T-point average that the drag rate
uses. It uses the **wet-only SUM**

```fortran
zmsku = ( 2. - umask(ji-1,jj,mbkt) * umask(ji,jj,mbkt) )
zebot = - 0.001875 * rCdU_bot * SQRT( (zmsku*(uu(ji)+uu(ji-1)))**2 + ... )
```

with **no `0.5`** — contrast `zdfgls.F90:203`, which writes the same mask
expression *with* the `0.5`, and `zdfdrg.F90:174-181`, which forms
`zut = uu(ji)+uu(ji-1)` and then divides it back out via `SQRT(0.25*(zut²+zvt²))`.
So zdftke's bottom velocity is **twice the wet-only average**.

legoESM passed the plain average → **2× low in the flat interior, 4× low at a
bathymetry step** (20.7% of the 9920 wet columns, measured).

The missing `0.5` is **structural, not a NEMO slip**: it cancels the `0.5`
already inside the `0.001875` prefactor (`= (rn_ebb0/rho0)*0.5`,
zdftke.F90:284), leaving `en_bot = (rn_ebb0/rho0)·Cd|U|²  ∝ u_*²` — the same
form as the surface BC `en(1) = zbbrau*taum` (:266). (Surfaced by the
adversarial review; it is the argument that closes the "maybe NEMO forgot the
0.5" objection, which the code-diff alone does not.)

Fixed at `ocean_model_latlon_cgrid.py::_tke_bottom_dirichlet`, which now takes
the **RAW face-staggered state** and masks the faces with the canonical
`compute_face_masks_3d` before summing (raising, never silently degrading, if
handed an already-collapsed T-point field). NEMO's `* ssmask` (:288) is now
ported too. Tests: `TestNemoBottomTkeVelocityConvention` — three cases against
a numpy re-derivation of the F90 line on a staircase bathymetry, **verified
non-vacuous** by reverting the production line and confirming both numeric
tests go red.

**Review-caught defect in my first attempt (worth recording).** I initially
derived the masked sum as `2*zmsku*<u>_plain`, justified by "a dry face carries
u == 0 exactly". That is true in NEMO (`dynzdf.F90:121-150` umasks `puu` at
every level) and **FALSE in legoESM**: the prognostic `u` carries only the 2-D
column mask, and the barotropic correction adds a uniform-in-k increment, so
sub-seafloor faces hold a small non-zero velocity (measured: 40/648 columns,
max |u| 1.48e-2 m/s, zero vertical spread — the uniform add is the source).
The shortcut would have been exact in 570/648 columns and *worse than the
pre-fix code* in the 78 step columns it was meant to fix. It also hand-rolled
geometry that `compute_face_masks_3d` already owns, silently dropping that
function's seam-wall and meridional-periodicity exclusions — the
port-the-formula-but-not-its-exclusions failure mode.

### RETRACTION — this bug does NOT own the dist=0 residual

The walk reported the plain average as "a large, plausible driver of the
concentrated dist=0 corr degradation (0.887-0.896)". **That is refuted.**

Index-convention-free check (`_nemo_zebot_floor.py`, evaluates NEMO's own
`zebot` from its dumped `rCdU_bot` and restart `ub`/`vb`, so no dump-level
indexing enters):

```
rCdU_bot            min -1.1397e-04   max -5.0000e-05
zmsk-weighted |U|   max  3.0633e-01   mean 1.1879e-02
zebot               max  5.2233e-08     vs rn_emin = 1.0e-06
FLOOR BINDS for 1.0000 of 9920 wet columns   (headroom 19.1x)
```

NEMO's bottom TKE Dirichlet is **floor-bound in every wet column with 19×
headroom**. legoESM's is likewise uniformly `1e-06` (instrumented through the
production path: `_nemo_bottom_wet_only_doubling` fires twice, `zmsku ∈ {1,2}`
with 20.7% doubled, Dirichlet min=max=1e-06). A 4× amplitude error is still 5×
under the floor. Re-running `zdftke_bottom_bc_isolate.py` with the fix
reproduces the pre-fix numbers to 4 decimals (dist=0 corr 0.8872/0.8956) —
as it must.

**⇒ the bottom TKE BC row was already AT BAR by construction, and is
climate-inert in DINO.** The fix ships because it is faithful and matters for
any card with stronger bottom flow, not because it moves this campaign.

### Where the dist=0 residual actually lives — FOUND (2026-08-07)

**The `nn_mxl=3` ldown (bottom-up) mixing-length sweep owns it. CONFIRMED.**
Same root cause class as the review defect above: *legoESM does not zero below
the seafloor where NEMO does.*

Discriminator at dist=0 (n=9920, fp64, `LEGOESM_NEMO_E3T=both`, recipe
`nemo_dino_kamm_mlf`, all dumps registered in `time_levels.py`):

| quantity @dist=0 | corr | ratio | verdict |
|---|---|---|---|
| `en` vs `tke_dump_en.bin` | 0.99950 | 1.0000 | input clean |
| `rn2` vs `tke_dump_rn2.bin` | 1.00000 | 1.0000 | input clean |
| **`zmxlm` vs `tke_dump_zmxlm.bin`** | **0.8743** | **1.0763** | **owner** |
| `zmxld` | 0.9841 | 1.0333 | ≈ √1.0763 |
| avt / avm | 0.8872 / 0.8956 | 1.0743 / 1.0764 | tracks `zmxlm` exactly |

`l_eps = √(lup·ldn)` being off by exactly the square root of `l_k = min(lup,ldn)`
says ONE sweep is long and the other is right.

Mechanism, with the oracle's lines. `zdftke.F90:469`
`en = MAX(en, rn_emin) * wmask` ⇒ **`en == 0` at every dry sub-seafloor
w-point**; `:651` `zmxlm = MAX(rmxl_min, SQRT(2·en/zrn2))` has no `wmask`, so
dry rows sit at exactly `rmxl_min = 0.01 m`; the CASE-3 ldown sweep
(`:696-699`) then runs *through* them, giving at each column's deepest wet
interface `ldn(mbkt) = MIN(rmxl_min + e3t(mbkt+1), l_int(mbkt))`. That
prediction reproduces NEMO's dumped `zmxlm@dist0` with `rel_med = 0.0`
(99.01% of columns within rtol 1e-6).

legoESM computes `l_int` from the **carried** `e`, which below the seafloor is
`tke_background = 1.0e-6`, **not 0** (measured: legoESM `e@dist=-2` = 1.0e-06,
NEMO `en@dist=-2` = 0.0 exactly). With `N2 = 0` there, `l_int ≈ 1414 m`, so
each dry row re-widens the ldown carry (measured `l_k` at sub-seafloor rows:
506 m / 634 m vs NEMO's 0.01 m) and the bottom limitation is destroyed before
it reaches the seafloor. Structural root: `compute_mixing_lengths`
(`tke.py:632-642`) takes **no wet mask** — it cannot know where the seafloor is.

The bottom limit binds in **390/9920 = 3.93%** of columns; on that subset
legoESM's `zmxlm` ratio is **2.94** with corr **0.024** (uncorrelated), on the
other 9530 it is **0.999969**. `0.0393 × 1.94 = 0.076` — those 390 columns are
the ENTIRE 1.0763 dist=0 ratio, which is essentially the whole `zdftke
composite` gate row (1.0708).

Projection (correcting `zmxlm` at dist=0 only, everything else byte-identical;
a LOWER bound, since dist=1/2 stay uncorrected):

| row | as-is | projected |
|---|---|---|
| avt @dist=0 | corr 0.8872, ratio 1.0743 | corr 0.9897, ratio 1.0012 |
| avm @dist=0 | corr 0.8956, ratio 1.0764 | corr 0.9980, ratio 1.0032 |

Fix (not yet written — needs its own adversarial review): force `l_int` (or
`e`) to `cfg.mxl_min` on dry w-interfaces before both sweeps, which means
plumbing a wet mask into the shared `compute_mixing_lengths`.

Separate, lower-leverage, and MASKED by the above once fixed (record as DEBT,
PLAUSIBLE): `zdftke.F90:650` floors `rn2` at `rsmall = 0.5·EPSILON(1_wp) =
1.11e-16` (NEMO built with promoted reals — reconstruction matches 96.05% in
double vs 17.66% in single); legoESM uses `max(N2, 1.0e-12)` (`tke.py:691`), a
1e4 difference affecting the 3.79% of dist=0 cells with `rn2 < 1e-12`.
