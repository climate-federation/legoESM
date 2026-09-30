# NEMO-testcases L2 GYRE round 104 receipt: the `zdfsh2` shear-production walk

Date: 2026-09-17. Branch `fidelity/nemo-testcases-l2-gyre-codex2`, incoming tip
`704dbf8faaee79ee0a45a48d1f0ffe27f80a3dcd`.

## Verdict

**HELD; no production physics and no configuration landed.** Round 103 named
the shear production `p_sh2` as the sole magnitude owner of the TKE
right-hand-side miss and stopped, because `p_sh2` is written in a different
routine and campaign decision 41 requires a term to be walked at its own
stage. This is that walk.

**The first statement whose output is not bit-identical to NEMO's is the
shear divisor**

> `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:102`
> `/ ( (e3w_1d(jk  ) *(1._wp+r3u(ji,jj,Kmm))) * (e3w_1d(jk) *(1._wp+r3u(ji,jj,Kbb))) )`

and its `v` twin at `zdfsh2.f90:107`. **It is not the arithmetic that is
wrong; it is the free-surface field legoESM routes into it.** NEMO reaches
this statement through `zdfphy.f90:319-320`'s
`CALL zdf_sh2( Kbb, Kmm, avm_k, sh2 )`, which the step program calls at
`stprk3.f90:167-168` as `CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )` — both formal
time levels bound to the **step-entry** slot, the previous line being the
commented-out `Nnn` variant the authors replaced. So `r3u(ji,jj,Kmm)` and
`r3u(ji,jj,Kbb)` at `:102` are BOTH the step-entry free-surface ratio.
legoESM feeds that construction the RK3 stage-3 `Kmm = N+1/2` ssh instead:
one field, `0.5*(step-entry ssh + after ssh)`, is built for `tra_zdf`'s
`e3w(Kmm)` divisor, where it is correct, and the same field is then handed to
the shear's `r3u`/`r3v`, where it is a different time level from NEMO's.

**WHAT THE DIFF TOUCHES, stated in the verdict so a mechanical scope audit is
not blindsided by the diffstat.** This round is measurement-only in substance
AND in effect: **not one line of production physics changed.** The diff adds
one measurement module and one direct test, extends the consolidated stage
gate with a sub-walk and a plant, and registers fourteen compiled-source
citations in the citation audit (plus a rigid re-anchor of two of that
audit's own self-citations whose line numbers moved because the gate file
grew). No shared numeric line moved, so no card's numbers can move, and none
did: every headline below is identical before and after by construction.

**Headline numbers: NOT RE-MEASURED this round, and unchanged by
construction.** kt2 U/V `2.7377110452773967e-12` / `3.284922138989399e-12`;
kt3 T/S `1.627497246303733e-4` / `6.327735185607253e-6`; day-30 T RMS
`1.2397011295506804e-2` K. These are the round-96/97 before arm restated from
operator note (J). The trajectory ladder and the days 1-30 arm were NOT run,
because there is no candidate to judge. The claim that they are unchanged is
a property OF THE DIFF, not of a run: `git diff --name-only 704dbf8faaee..HEAD`
lists nothing under `packages/`, so no file the production step imports was
touched. That is checkable in one command and is stated as such rather than as
a measurement that was taken.

No NEMO source was modified and neither `makenemo` nor `mpirun` was run. No
card, default, scheme selection, threshold, coefficient, carried state,
restart schema, year harness, reconciliation gate, freshwater pair or #1484
guard changed.

## Registration

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round104.md`, committed as
`05d34a76696f` before any new measurement. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round104/`.

## Instrument validation before any new science

The consolidated gate was re-run at the incoming tip in `--mode
stage-tke-walk` with nothing changed, purely to reproduce known values
(`instrument_repro.json`, 12 m 35 s). It reproduces **every** round-103 row
exactly:

| row | round 103 | this tip |
|---|---|---|
| `en_entry` | 0 / 0 | 0 / 0 |
| `en_after_boundaries` | 0 / 0 | 0 / 0 |
| `en_after_langmuir` | 0 / 0 | 0 / 0 |
| `rhs_pre_sweep` | 11,993 / `5.488912518947231e-10` | 11,993 / `5.488912518947231e-10` |
| `en_post_sweep` | 979 / `6.809688229969524e-12` | 979 / `6.809688229969524e-12` |
| `zd_up` / `zd_lw` / `zdiag` | 0 / 0 each | 0 / 0 each |
| `p_sh2_operand` | 17,400 / `3.811744924985501e-14` | 17,400 / `3.811744924985501e-14` |

Cells are `unequal / max abs`. That is prediction **P1, CONFIRMED**.

## Compiled-source basis

The record's compiled branch is
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90`, byte-identical to the
Round-101 build's copy of the same file (`diff -q`, zero differences). On
this card `cpl_sdrftx .AND. ln_stshear` is false, so the executed branch is
the ELSE at `zdfsh2.f90:97-109`, and the routine's output-bearing assignments
inside `DO jk = 2, jpkm1` (`zdfsh2.f90:83`) are, in compiled order:

| # | statement | line | recorded output? |
|---|---|---|---|
| S1 | `zsh2u` | `zdfsh2.f90:99-103` | no |
| S2 | `zsh2v` | `zdfsh2.f90:104-108` | no |
| S3 | `p_sh2` | `zdfsh2.f90:112-113` | yes (`sh2`) |

followed by the surface/bottom zeroing at `zdfsh2.f90:116-119`, which writes
`jk = 1` and `jk = jpk` and therefore touches no cell this walk scores.
Because S1 and S2 have no recorded output, they are attributed by
one-variable operand swaps against S3 rather than by rows of their own — the
technique the operator endorsed in note (Q) and round 103 used.

Supporting statements cited and mechanically verified: the free-surface ratio
at `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domqco.f90:266-267` (inside
`dom_qco_r3c_RK3`, the entry `stprk3_stg.f90:178` calls), the precomputed
reciprocal it multiplies at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domhgr.f90:170`, and the wet-face
masks at `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/dommsk.f90:237-242`.

## No acquisition was needed, and that is a measurement

The preregistration claimed every operand `zdf_sh2` reads at this instant is
already in an admitted record, and said the round would STOP FOR RECORD if
that were wrong. It held. Two things had to be checked rather than assumed,
and both are now hard refusals inside the probe rather than prose:

1. **Which viscosity.** `zdf_sh2` reads `avm_k` as passed at
   `zdfphy.f90:319`, which is the Round-59 record's `avm_entry`. The round-46
   stage record's `tke_avm_k` is the POST-`zdf_phy` array and the two differ
   in **6,294 of 20,416** owned cells. Using the convenient one would have
   silently answered a different question.
2. **Which reference thickness.** `zdfsh2.f90:102` names `e3w_1d(jk)`, a
   one-dimensional ladder. The record stores the three-dimensional `e3w_0`.
   The probe asserts that field is horizontally uniform (it is, exactly, on
   this zco card) before reading a column out of it, and refuses otherwise.

A third: both face windows reach one cell outside the owned domain, so the
face sums would need a viscosity halo the Round-59 record does not carry.
Every window-boundary face on this card is LAND — `umask` is identically zero
at NEMO `ji = 1, 2, 33, 34` and `vmask` at `jj = 1, 2, 23, 24` — so the
wet-face mask annihilates those terms and a zero pad is exact, not merely
adequate. The probe refuses loudly if that ever stops being true.

## The reference side, proven before anything is read from it

`nemo_testcase_l2_gyre_round104_shear_replay.py` rebuilds `zdfsh2.f90:99-113`
from NEMO's OWN recorded operands in the compiled association and compares
against NEMO's OWN recorded `sh2`:

| arm | unequal | max abs | class |
|---|---:|---|---|
| every operand recorded | **0** of 20,416 | `0.0` | BIT |

That is prediction **P2, CONFIRMED**, and it is what makes every row below
interpretable: without it the reference side of the comparison would itself
be unproven.

## The walk, given NEMO's recorded stage entry, through the production step

All rows are scored over the SAME 20,416 owned cells the round-103 walk used,
of which 17,400 are wet and 3,016 dry; both denominators are given wherever a
count appears. Cells are `unequal / max abs`.

| row | unequal | max abs | class |
|---|---:|---|---|
| `all_operands_recorded` (vs NEMO's recorded `sh2`) | 0 | `0.0` | BIT |
| `production_step_vs_recorded` | 17,400 of 20,416 (= 17,400 of 17,400 wet) | `3.811744924985501e-14` | DEBT |
| `model_operand_replay_vs_production` | 17,400 | `3.811744924985501e-14` | DEBT |

**Ten of the twelve operand rows are captured quantities and every one is
bit-identical to NEMO's. The remaining two are NOT captured, and saying so is
the point.**

| operand | unequal | class | is this what production consumed? |
|---|---:|---|---|
| `u_now`, `u_before`, `v_now`, `v_before` | 0 each | BIT | YES — the bridged state arrays |
| `avm_face_u`, `avm_face_v` | 0 each | BIT | YES — built from the bridged viscosity |
| `wumask`, `wvmask` | 0 each | BIT | YES — the model's own face masks |
| `coast_u`, `coast_v` | 0 each | BIT | YES — the model's own face masks |
| `divisor_u`, `divisor_v` | 0 each | BIT | **NO** — see below |

**CORRECTION, after an independent review, to an earlier draft of this
receipt.** That draft said "every operand legoESM feeds the routine is
bit-identical to NEMO's", listing the divisor among them. That sentence was
WRONG and is withdrawn. The divisor rows are built by the gate's mirror from
the STEP-ENTRY free surface — NEMO's time level — and they are bit-identical
to NEMO's for exactly that reason. The production step builds its divisor
from a DIFFERENT free-surface field, and this round did not capture that
field: exposing it needs a trace threaded out of the model's shear helper,
which is a larger diff than this round took. So the correct statement is that
ten operands are captured and bit, and the divisor's free-surface factor is
the one quantity the census does NOT verify — which is precisely the quantity
the rest of the walk goes on to name.

The five one-variable swap rows are therefore trivially inert — substituting a
group of operands that already equal the recorded ones changes nothing, and
all five sit at the production row's own 17,400 / `3.811744924985501e-14`.
**Prediction P3 is REFUTED.** It predicted that exactly one of four
model-constructed groups, the live face-metric divisor, would reproduce the
production output when swapped in. None does. P3 is reported as refuted and is
not restated as a narrower success. Its *target* was right and its *mechanism*
was wrong: the divisor is indeed the owner, but not through an operand the
swap machinery can substitute, because the mirror and the production step
build it from different free-surface fields in the first place.

The third row is the one that carries the round. The gate rebuilds the
compiled statement using NEMO's own time level for the face metric and the
model's values for everything else, and scores that against what the
production step actually produced; it is NOT bit. **That row is the finding,
not a defect in the mirror.** The module's own docstring warns that a non-BIT
row here means either the mirror is wrong or the production step consumes
something the mirror does not know about; it is the second, and the next
section shows which quantity, by changing it and nothing else.

## What it is consuming: the free-surface TIME LEVEL

`zdfsh2.f90:102` divides by `(e3w_1d(jk)*(1+r3u(ji,jj,Kmm))) *
(e3w_1d(jk)*(1+r3u(ji,jj,Kbb)))`. On the RK3 GYRE run both formal levels are
the same slot: `stprk3.f90:167-168` calls `CALL zdf_phy( kstp, Nbb, Nbb,
Nrhs )` — the line above it is the commented-out `Nnn` variant the authors
replaced — and `zdfphy.f90:319-320` passes those two through unchanged into
`CALL zdf_sh2( Kbb, Kmm, avm_k, sh2 )`. The record agrees: at the kt=2
stage-1 slot `r3u_Kbb` and `r3u_Kmm` are bit-identical arrays, as are `u_Kbb`
and `u_Kmm`, which is why the model's `nemo_face_native_now2` variant using
one field for both factors is right.

legoESM builds ONE free-surface field for the whole vertical-diffusion call,
`0.5*(step-entry ssh + after ssh)`, which is NEMO's stage-3 `Kmm = N+1/2` ssh
and is CORRECT for `tra_zdf`'s `e3w(Kmm)` divisor, and then hands that same
field to the shear's `r3u`/`r3v`, where NEMO's is the step-entry one.

**The discriminator, run through the real production step.** The same step,
with that one field changed to the step-entry ssh and nothing else:

| row | unequal | max abs | class |
|---|---:|---|---|
| `time_level.step_entry_ssh_production_vs_recorded` | **0** of 20,416 | `0.0` | BIT |
| `time_level.step_entry_ssh_vs_model_operand_replay` | **0** of 20,416 | `0.0` | BIT |

The first says legoESM's shear production becomes bit-identical to NEMO's.
The second is the control that makes the attribution legitimate: it proves
the gate's operand mirror IS the model's shear expression, so the one
non-bit row above is attributable rather than resting on an unproven
reconstruction. Gate verdict `CONFIRMED_SHEAR_FACE_METRIC_TIME_LEVEL`.

**What that discriminator is NOT, stated precisely.** ONE FIELD is changed;
the WHOLE STEP is not. The existing model hook that changes it also returns
`tra_zdf`'s `e3w(Kmm)` divisor to the step-entry ssh, where NEMO wants
`N+1/2`, so downstream of the shear the two arms are not comparable and NO
other row from this arm is read. For the `p_sh2` row itself there is no
confound, and that is not an assertion: the shear is computed BEFORE the
diffusion solve that consumes the same field, so nothing the hook changes
downstream can reach it. The independent reviewer checked that ordering in the
source and agreed. The hook is therefore a discriminator and NOT a candidate
fix; the real fix threads a separate step-entry field to the shear alone and
is written up in OPEN rather than attempted this round.

**What is still NOT captured.** Neither arm captures the production step's
own face metric directly. The attribution rests on the combination of: ten
captured operands all bit; the compiled statement provably exact on NEMO's
operands; and the production output going from 17,400 unequal to 0 when one
named free-surface field, and nothing else, is changed. That is a complete
account of the remaining degree of freedom, but it is an elimination argument
with one direct measurement, not a direct capture of the wrong divisor. A
direct capture is the first thing round 105 should add alongside the fix.

## Magnitude: prediction P5 is REFUTED, and that refutation is the useful part

P5 predicted that the miss is a last-bit effect on a large-`sh2` cell, with
the relative difference at the argmax cell at most `1e-10`. Measured:

| quantity | value |
|---|---|
| `max|sh2|` in the record | `5.5633265996063366e-08` |
| `max|delta p_sh2|` | `3.811744924985501e-14` |
| argmax cell, `|delta| / |sh2|` | `6.851558019348533e-07` |
| relative difference over wet cells, median | `3.330e-07` |
| relative difference over wet cells, max | `7.296e-07` |

Seven-hundred-thousandths, not `1e-10`. **P5 is REFUTED**, and its own
falsifier says what that means: the miss is STRUCTURAL, not a rounding one,
which is exactly the signature of a wrong time level and is what ruled out
every last-bit candidate before the walk found the real one. The prediction
is reported as refuted and is not rewritten.

## Prediction P4: a real source deviation, measured INERT

P4 named a different statement: legoESM's inline free-surface ratio in the
shear path DIVIDES by `e1e2u`, where `domqco.f90:266-267` MULTIPLIES by the
precomputed reciprocal `r1_e1e2u`, which `domhgr.f90:170` sets to
`1._wp / e1e2u`. `x/a` and `x*(1/a)` are not the same double.

The deviation is REAL and measured: against NEMO's recorded `r3u_Kmm` the
divide form differs in **58 of 704** u-faces at max `1.058791e-22`
(`2.217e-16` relative) and `r3v` in **54** at `5.293956e-23`, while the
multiply form reproduces NEMO's recorded ratio with max abs difference
`0.0`. legoESM's own SHARED implementation of the same NEMO statement, in
`vertical.py`'s live-face geometry builder, already uses the multiply form;
the TKE shear path carries a SECOND, divergent copy of it.

**And it is measurably INERT at this stage, so P4 is REFUTED as the owner.**
`1e-22` in `r3u` is eight orders of magnitude below one unit in the last
place of `e3w*(1+r3u)`, so the `divisor_u`/`divisor_v` rows above are BIT and
no `p_sh2` cell can move. P4 is therefore reported as a confirmed source
deviation and a refuted attribution — the same shape as operator note (Q)'s
inert `zpelc` recurrence, and it is carried into OPEN as a duplicate-numerics
finding rather than as a candidate.


## Prediction ledger

Every prediction is reported exactly as frozen, including the two that were
refuted. Neither refutation is restated as a narrower success.

| # | prediction | outcome |
|---|---|---|
| P1 | the unmodified gate reproduces every round-103 row at this tip | **CONFIRMED**, all eight rows identical |
| P2 | a transcription fed entirely from NEMO's recorded operands reproduces NEMO's recorded `sh2` BIT | **CONFIRMED**, 0 of 20,416 |
| P3 | exactly one of four model-constructed operand groups, the live face-metric divisor, reproduces the production `p_sh2` | **REFUTED** — none does, because all twelve operands are already bit-identical; the owner is a field none of the four groups names |
| P4 | the first non-bit statement upstream is `domqco.f90:266-267`, the multiply-by-reciprocal | **REFUTED as the owner**, CONFIRMED as a source deviation: real at `1.06e-22` in `r3u`, measurably inert in the divisor |
| P5 | relative difference at the argmax cell at most `1e-10` | **REFUTED**, measured `6.851558019348533e-07`; the miss is structural, and P5's own falsifier says so |
| P6 | the eager / isolated-JIT / production discriminator for round 103's last-bit residue | **NOT RUN**, see OPEN item 2 |
| P7 | nothing lands; the status is HELD | **CONFIRMED** |

## Rule 12 and the testcase dispositions

No production numeric line changed, so there is nothing for the trajectory
ladder to judge and it was not run. Per-testcase, with the shared-statement
risk stated rather than assumed:

| testcase | disposition |
|---|---|
| GYRE (kt=1..10, days 1-30) | UNMOVED BY CONSTRUCTION. The diff adds a measurement module, a test, a gate sub-walk and citation-map entries; it touches no file the production step imports. |
| DINO | UNMOVED THIS ROUND (no imported file changed), AT RISK NEXT ROUND. **This row is READ OFF THE CONFIGURATION AND THE SOURCE, not measured — no DINO case was executed this round.** DINO's NEMO-identity settings also select the live-face shear metric at step entry, so the fix named in OPEN would reach DINO. DINO runs the Modified-Leap-Frog program, where `stpmlf.F90:210` calls `zdf_phy( kstp, Nbb, Nnn, Nrhs )` — the two levels are DIFFERENT there — so a blanket "use the step entry" fix would be wrong for DINO and the fix must branch on the already-existing shear variant. This is the single largest reason the round HELD instead of landing. |
| LOCK_EXCHANGE, OVERFLOW | NOT EXECUTED. Neither runs the turbulence closure's shear path on its card. |
| ORCA2 | UNMEASURED-WITH-SPEC, unchanged from round 103. |

## Plants

Three controls, each shown firing, and each stated for exactly what it proves.

1. **Record replay, `--plant operand-ulp` — the ARITHMETIC control.** Advances
   one bit of a single viscosity face. One unit in the last place of ONE face
   is only half a unit in the last place of the two-face sum at
   `zdfsh2.f90:112`, so round-to-nearest can absorb it — the first two
   versions of this plant were absorbed exactly that way and reported green.
   The plant therefore walks the live faces in descending contribution and
   takes the first whose corruption reaches the output, refusing loudly if
   none does. Fired at u-face index `(16, 30, 0)` on a baseline viscosity sum
   of `0.0024844304807471983` carrying a shear term of
   `5.939825108094368e-08`: exactly **1** cell moved, max
   `1.3234889800848443e-23`. Exit 1, printed `STATUS PLANT-FIRED`.
2. **Stage gate, `--plant stage-shear-operand-ulp` — the PRODUCTION-ROW
   control.** An earlier version of this plant was VACUOUS as an arithmetic
   control and an independent reviewer was right to say so: it corrupted the
   reference of the baseline production row, which differs in every wet cell,
   so the "add exactly one" contract forced it onto a DRY cell where both
   sides are zero by construction. It now targets the TIME-LEVEL row, which
   is bit-exact over all 20,416 cells, and is REQUIRED to land on a cell the
   shear genuinely produced (`recorded != 0.0`), refusing otherwise. Fired at
   `(1, 1, 0)`, a WET cell whose `p_sh2` is `1.7204494884191844e-08`: clean
   `0` unequal, planted **1**, max `3.308722450212111e-24`, plant target
   `GYRE-zco.kt2.tke_shear.time_level.step_entry_ssh_production_vs_recorded`.
   The gate's exit code is 1 for every plant and therefore discriminates
   nothing; the proof is the internal hard check, and the run's status label
   reads `PLANT-FIRED`. The string `STATUS PASS` appears **zero** times in
   its log.
3. **Citation audit.** The clean run is 274 citations, 0 unmapped, 0
   failures, 0 map-audit failures, 9 of 9 self-test controls fired, `STATUS
   PASS`, exit 0. Its receipt-path plant on `domqco.F90:189-209` exits 1 with
   `SYMBOL-NOT-AT-LINE`. Separately, each of this round's **14** new
   compiled-source citations was checked clean AND shifted by two lines: all
   14 clean rows are `OK` and all 14 shifted rows fail, so none of them can
   pass vacuously.

**On "a plant run must never print a success word", stated exactly rather
than as a blanket claim.** The two plants this round touches print no success
STATUS: the replay prints `STATUS PLANT-FIRED` and the stage gate's label is
forced to `PLANT-FIRED`. Two pre-existing strings do contain success words in
plant output and neither is a status: the citation gate's JSON carries its
self-test control labels, two of which read "unplanted baseline must pass",
and the stage gate's JSON carries an unrelated pre-existing verdict field
`CONFIRMED_INHERITED_MODEL_TAUM` from the surface-operand attribution. A log
scraper looking for a status will not be misled; a scraper grepping for the
substring "pass" anywhere would be. That is named here rather than left for
the next reviewer to find, and tightening those two pre-existing strings is
not this round's diff.

**Commit stamps.** The clean shear walk and the plant are both stamped at the
same commit, re-run together after the review. The citation runs are stamped
one commit earlier; those two commits differ only in the stage gate and the
citation script, and the citation gate's own numbers do not depend on the
stage gate.

## Independent adversarial review

An independent adversarial pass was run in-sandbox against this round's diff,
its preregistration and an earlier draft of this receipt
(`codex_review.log`). **Its verdict was `DO NOT SHIP`**, on findings that
were correct, and it is quoted rather than summarised:

> - Blocking: claim (2) is false. The "model operand" census constructs its
>   divisor using `state.eta.data`, the step-entry field. Production instead
>   receives `_nemo_ws_zdf_eta_kmm`, normally the half-step field. Therefore
>   the reported BIT `divisor_u/v` rows are not operands legoESM actually
>   feeds production. The receipt's "every operand" statement is refuted by
>   its own instrumentation.
> - The mirror's documented contract says a non-BIT production comparison
>   means it is wrong or omits an operand. That comparison is indeed non-BIT.
>   Exact agreement only after changing production to the mirror's chosen eta
>   validates the counterfactual arm, not the baseline operand
>   reconstruction. Direct capture of baseline `face_metrics` is missing.
> - The time-level source reading is otherwise correct: NEMO calls
>   `zdf_phy(kstp,Nbb,Nbb,Nrhs)` at `stprk3.f90:168`, passes those slots
>   unchanged at `zdfphy.f90:319-320`, and the record identifies
>   `(Kbb,Kmm)=(3,3)`. The no-Stokes ELSE branch and levels `2:jpkm1` are
>   also correct; surface/bottom zeroing cannot touch the scored slice.
> - The whole-step hook is not a downstream confound for the captured
>   `p_sh2`: shear is calculated before the diffusion solve. But calling it
>   "exactly one thing changed" is false for the whole step because the same
>   eta also changes `tra_zdf`.
> - The stage plant is vacuous as an arithmetic-path control: it corrupts the
>   reference at an already-equal dry cell, not a consumed shear operand. It
>   would fire for any fixed production array containing one matching cell.
>   The separate record-replay plant does exercise arithmetic.
> - "No plant prints a success word" is false: the citation plant emits two
>   strings containing "must pass". The stage-plant artifact also contains an
>   unrelated `CONFIRMED_INHERITED_MODEL_TAUM`.
> - P3, P4, and P5 are explicitly reported as refutations; I found no quiet
>   rewrite. No production package or configuration line changed. However,
>   "headline numbers unchanged" was inferred, not rerun, and the DINO
>   disposition is also asserted without a round-104 execution.
>
> DO NOT SHIP

**What was done about each, with no finding dismissed.**

| finding | classification | action |
|---|---|---|
| the divisor rows are not captured operands, so "every operand is BIT" is false | **CONFIRMED BUG IN THE CLAIM** | the sentence is withdrawn in the verdict section and in the operand table, which now says per row whether the quantity was captured; the walk's non-matching row is described as the finding, not as a mirror defect |
| direct capture of the production face metric is missing | **CONFIRMED GAP** | stated as a gap in "What is still NOT captured", and made OPEN item 0 for round 105 |
| the source reading is correct | agreement, recorded | quoted, because an independent re-reading of the compiled call chain is the strongest evidence this round has |
| "exactly one thing changed" is false for the whole step | **CONFIRMED IMPRECISION** | rewritten as "one field is changed; the whole step is not", with the reviewer's own check that the shear precedes the solve quoted as the reason the `p_sh2` row is unconfounded |
| the stage plant was vacuous as an arithmetic control | **CONFIRMED DEFECT IN THE CONTROL** | the plant was MOVED to the time-level row and now refuses unless it lands on a wet cell; re-run, it fires at `(1, 1, 0)` on a `p_sh2` of `1.72e-08` |
| "no plant prints a success word" is too broad | **CONFIRMED OVERCLAIM** | narrowed to a precise statement naming both pre-existing strings and what they are |
| headline numbers inferred, DINO asserted | **CONFIRMED, both** | both are now explicitly labelled as not measured this round, with the basis given |

The clean walk, the plant, the receipt corrections and the control fix were
all re-run or re-committed after the review. **A second review pass has not
been run on this corrected text**; that is recorded as a limitation rather
than left implicit, and round 105 should re-review before building on it.


## Tests

Focused suites, verbatim:

```
65 passed in 6.79s
```

covering `tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round104_shear_replay.py`
(16 tests, new this round),
`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py` and
`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round46_kt2_stage_gate.py`.

The new tests are non-vacuous by construction rather than by assertion: one
of them hand-evaluates the compiled statement at a named cell and demands the
same bit pattern, and twelve of them are a parametrised sweep proving that
each of the twelve operands, perturbed alone, changes the output — so no row
can pass because an argument is silently ignored.

**The one known-red test is not ours, and we added no tenth offender.**
`test_every_report_emitter_stamps_the_worktree` fails on the lane with nine
offenders. Re-run at this tip it still lists exactly those nine; this round's
new module is not among them, because it stamps the worktree in its report.


## Evidence

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round104/`.

| artifact | SHA-256 |
|---|---|
| `instrument_repro.json` | `4120dd653631d2ec43bfafd4a076a468831ef1701973c288584a53a9d3539f9b` |
| `record_replay.json` | `fb64e71dbadbff76417c51f4e3e47ffb349ab06bac3f9b12c38be4068c6c4274` |
| `record_replay_plant.json` | `d9d7a707cba9c40a983f90d5d4ff1c07181c3813b98514e5829a9241b9101fcf` |
| `shear_walk.json` | `6bb12a25973c7e6fdb350f3fee412db6dda76cfc51e445ebc6bbcb49e455eb1d` |
| `shear_plant.json` | `c0c1ee03f8aa24c4835d341891391788f7338846bfc61822f34f96652d85cd9e` |
| `citation_gate.json` | `678a84898630cff135c647f8f8b6a99f10d9c91d4581af3ff8ad1cd7f4aa7b83` |
| `citation_gate_plant.json` | `165a80b9af98ef85c9f51a24168085611b61106190971ff0ba38173778e4cb52` |
| `citation_gate_round104_controls.log` | `125f6cf54a60e57d86c21a3b5425dcb9742a0acd405923ba1295562a7dc79e6c` |
| `focused_pytest.xml` | `dbf6355fadddb4b4a13e2031935de6df5f4dca533d492583da316439551e8df2` |
| `codex_review.log` | `bdebd0587f3ca181eac8738b9de52b436e0d090621f8b6f142d9fd9df0cb2b67` |

The shear walk and its plant were re-run after the review; the fingerprints
above are of the RE-RUN artifacts, stamped `b4992e0fda7c` and `5255664a0c3b`
respectively, and the superseded first-run numbers are not quoted anywhere in
this receipt. The `instrument_repro` and citation artifacts are from earlier
commits on this same branch, each stamped in its own file.

A fingerprint pins ONE run of the tool that produced it. Re-running a gate
after this receipt is committed invalidates the hash even when every number
is identical; refresh the hash in the same commit, or do not re-run.

## OPEN - round 105

0. **CAPTURE THE PRODUCTION FACE METRIC DIRECTLY.** This round's attribution
   is an elimination argument closed by one direct measurement, not a direct
   capture: the wrong divisor itself was never read out of the production
   step. Threading the shear helper's face metric into the existing statement
   trace is a small diagnostic diff of the same shape as round 103's, and it
   converts the divisor row from "would be, on NEMO's time level" into a
   captured operand. Do this alongside item 1 so the fix is judged against a
   captured baseline.
1. **LAND THE TIME LEVEL, under the ladder.** The statement is named and its
   local proof is closed: with the step-entry free-surface field, `p_sh2` is
   bit-identical to NEMO's over every one of the 20,416 owned cells. What is
   NOT done is the fix itself. The shape it must take: the shear's
   free-surface field has to be threaded separately from the one
   `tra_zdf` consumes, because NEMO wants the step entry for the first
   (`stprk3.f90:168`) and `N+1/2` for the second. The branch is on the
   ALREADY-EXISTING shear variant, not on a new configuration field, so it is
   a bug fix and not a configuration choice — but it changes shared numerics,
   so it lands only if the Rule-12 ladder (kt=1..10) and the days 1-30 arm
   pass against the round-96/97 before arm. Round 103 showed `p_sh2` owns the
   whole right-hand-side magnitude, so this fix is expected to close
   `zdftke.f90:439-442` as well; that expectation is PLAUSIBLE and must be
   measured, not assumed.
2. **Round 103's OPEN item 2 was NOT run, and here is why.** Its
   discriminating measurement asks whether the `1.1102230246251565e-16`
   residue between a NumPy transcription of the right-hand side and the
   model's evaluation of it is XLA fusion or an association difference. The
   arm it needs — the model's right-hand-side expression evaluated eagerly
   and under a JIT of an isolated closure — has to be written against the
   solver's own assembly, which is a separate diff from this round's, and
   this round's budget went into the shear walk instead. It is not blocking:
   the residue is six orders of magnitude below the `p_sh2` term and cannot
   own the block's debt either way. It stays open with its measurement
   unchanged.
3. **A SECOND implementation of NEMO's free-surface ratio exists and is the
   divergent one.** The campaign's shared builder in `vertical.py` uses
   NEMO's multiply-by-reciprocal and reproduces the recorded ratio exactly;
   the TKE shear path has its own inline copy that divides instead and
   differs in 58 u-faces. It is inert today (measured above) and would stop
   being inert the moment the free-surface field or the mesh changed scale.
   Folding the shear path onto the shared builder is the right fix and is a
   one-variable change that must itself pass the ladder.
4. **`p_pdlr` (`zdftke.f90:421`) is still UNMEASURED-WITH-SPEC**, unchanged
   from round 103's OPEN item 3.
5. **Do not re-walk what is closed.** The three matrix writes remain BIT and
   were re-measured only as regression. The `zdfsh2` operand census is closed:
   all twelve operands BIT. The ordered `zpelc` recurrence remains inert
   (operator note Q), and `domqco`'s divide-versus-multiply is now a second
   measured-inert item on that list.
6. **No acquisition is pending.** The admitted round-46 and round-59 records
   contain every array this round and the named fix need.
