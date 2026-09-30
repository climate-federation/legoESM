# Preregistration: NEMO-testcases L2 GYRE round 104 `zdfsh2` shear-production walk

Date: 2026-09-17. Frozen at the incoming lane tip
`704dbf8faaee79ee0a45a48d1f0ffe27f80a3dcd`, before any new production-step
boundary is exposed and before any physics or configuration change. Evidence
belongs under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round104/`.

## What round 103 established, carried here as premises

1. Inside the compiled TKE matrix block, `zd_up` (`zdftke.f90:434`),
   `zd_lw` (`:435`) and `zdiag` (`:436`) are each BIT with zero unequal cells
   given NEMO's recorded stage entry, through the production step. The first
   non-bit output on the `en` path is the right-hand-side assignment at
   `zdftke.f90:439-442`, 11,993 of 20,416 owned cells, max
   `5.488912518947231e-10`.
2. The magnitude of that miss is owned by ONE consumed operand, the shear
   production `p_sh2`: 17,400 of 20,416 owned cells (= 17,400 of 17,400 wet),
   max `3.811744924985501e-14`. The budget closes,
   `rn_Dt * max|delta p_sh2| = 14400 * 3.811744924985501e-14 =
   5.488912691979121e-10` against `5.488912518947231e-10`, 3.2e-8 relative.
3. `p_sh2` is computed in a DIFFERENT routine, `zdfsh2`, so under decision 41
   it is walked at its own stage. That is this round.
4. Round 103's post-hoc one-variable swap is not exact: 11,029 cells at
   `1.1102230246251565e-16` between a NumPy transcription of the RHS statement
   and the model's evaluation of it inside the full step. Cause undecided
   between XLA fusion/FMA and an association difference. Round 103 named the
   discriminating measurement; this round carries it forward (P6).

## Instrument calibration already run before this file was committed

Per the campaign's instrument rule the existing consolidated gate
(`nemo_testcase_l2_gyre_round46_kt2_stage_gate.py --mode stage-tke-walk`) was
re-run at this tip with nothing changed, purely to reproduce known values
(`instrument_repro.json` / `.log`). No new science was read before this file
was committed; the reproduction is scored under P1 below and is reported as a
prediction like every other, including if it is refuted.

## The compiled statements this round walks

The record's compiled branch is
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90`. On this card
`cpl_sdrftx .AND. ln_stshear` is false, so the executed branch is the ELSE at
`:97-109`, and the routine's three output-bearing assignments, in compiled
order inside `DO jk = 2, jpkm1` (`:83`), are

| # | statement | compiled line |
|---|---|---|
| S1 | `zsh2u(ji,jj) = ( p_avm(ji+1,jj,jk) + p_avm(ji,jj,jk) ) * ( uu(jk-1,Kmm) - uu(jk,Kmm) ) * ( uu(jk-1,Kbb) - uu(jk,Kbb) ) / ( (e3w_1d(jk)*(1+r3u(Kmm))) * (e3w_1d(jk)*(1+r3u(Kbb))) ) * wumask` | `zdfsh2.f90:99-103` |
| S2 | the `zsh2v` analogue at v-faces | `zdfsh2.f90:104-108` |
| S3 | `p_sh2(ji,jj,jk) = 0.25 * ( ( zsh2u(ji-1,jj) + zsh2u(ji,jj) ) * ( 2. - umask(ji-1,jj,jk)*umask(ji,jj,jk) ) + ( zsh2v(ji,jj-1) + zsh2v(ji,jj) ) * ( 2. - vmask(ji,jj-1,jk)*vmask(ji,jj,jk) ) )` | `zdfsh2.f90:112-113` |

Only S3's output is recorded (`sh2`, Round-59 record). S1 and S2 have no
recorded reference, so they are attributed by one-variable operand swaps
against S3, the technique the operator endorsed in note Q and round 103 used.

**No new NEMO acquisition is needed, and this is a claim, not an assumption.**
Every operand `zdfsh2` reads at this instant is already in an admitted record:
`avm_entry` and the reference `sh2` in
`round59/oracle_tke_operands/oracle_tke_operands_kt00000002.bin`; and
`u_Kbb`, `u_Kmm`, `v_Kbb`, `v_Kmm`, `r3u_Kbb`, `r3u_Kmm`, `r3v_Kbb`,
`r3v_Kmm`, `e3w_0`, `umask`, `vmask`, `wmask`, `e1e2t`, `e1e2u`, `e1e2v`,
`r1_e1e2u`, `r1_e1e2v`, `ssh_Kbb` in the round-46 kt=2 stage-1 record
(`round46/oracle_kt2_stage`). If that claim is wrong the round STOPS FOR
RECORD and writes an acquisition script rather than substituting a proxy.

## Frozen predictions

**P1 — instrument.** Re-running the unmodified gate at this tip in
`--mode stage-tke-walk` reproduces round 103's `NEMO_TKE_RECORDED` rows
exactly: `en_entry` 0/0, `en_after_boundaries` 0/0, `en_after_langmuir` 0/0,
`rhs_pre_sweep` 11,993 / `5.488912518947231e-10`, `en_post_sweep` 979 /
`6.809688229969524e-12`, `zd_up`/`zd_lw`/`zdiag` 0/0 each, and the
`p_sh2_operand` row 17,400 / `3.811744924985501e-14`.
REFUTED if any one of those numbers differs.

**P2 — reference side.** A NumPy transcription of `zdfsh2.f90:99-113` fed
ENTIRELY from NEMO's own recorded operands (avm from Round 59; `uu`/`vv` at
the kt=2 stage-1 slot, where `Kbb = Kmm = 3`; `r3u`/`r3v` at the same slot;
`e3w_0`; `umask`/`vmask`/`wmask`) reproduces NEMO's recorded `sh2` BIT, 0 of
20,416 cells unequal.
REFUTED by one or more unequal cells, in which case the alignment or the
citation is wrong, the round names that as its finding, and no attribution
built on the transcription is reported.

**P3 — the miss is not in the S3 combine.** Substituting into that replay,
one at a time, the FOUR operand groups `zdfsh2` receives that the model
constructs rather than reads from NEMO — (a) the live face-metric divisor
`(e3w*(1+r3u))*(e3w*(1+r3u))` and its v analogue, (b) the face masks
`wumask`/`wvmask`, (c) the coast factors `2 - umask*umask` and
`2 - vmask*vmask`, (d) the avm face sum `p_avm(ji+1)+p_avm(ji)` including its
periodic seam — exactly ONE group reproduces the production `p_sh2`, and it
is (a), the live face-metric divisor.
REFUTED if group (a) alone does not reproduce the production `p_sh2`, or if
more than one group moves the row, or if none does. A refutation names the
group that does move it, and that group's statement becomes the walk's named
owner instead.

**P4 — the named statement.** The model's live `r3u`/`r3v` are NOT
bit-identical to NEMO's recorded `r3u_Kbb`/`r3v_Kbb`, and the first statement
in compiled order whose output is not bit-identical, upstream of `zdfsh2` and
feeding its divisor, is `dom_qco_r3c` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domqco.f90:214-215`:
`pr3u = 0.5_wp * ( e1e2t(ji)*pssh(ji) + e1e2t(ji+1)*pssh(ji+1) ) * r1_hu_0 *
r1_e1e2u`, where `r1_e1e2u = 1._wp / e1e2u` is a PRECOMPUTED reciprocal
(`domhgr.f90:170`). legoESM's `nemo_qco_live_face` branch instead DIVIDES by
`e1e2u`. `x * (1/a)` and `x / a` are not the same double, so this differs in
essentially every column, which is the shape of the observed miss (all 17,400
wet cells).
REFUTED if the model's live `r3u`/`r3v` are bit-identical to NEMO's recorded
values, in which case the divisor difference lives in `e3w` or in the
association of the multiply, and this round names that instead.

**P5 — magnitude is reported, not assumed.** `max|sh2|` in the Round-59
record is `5.5633265996063366e-08` and the observed `max|delta p_sh2|` is
`3.811744924985501e-14`, a ratio of `6.85e-7` if both fell on the same cell.
A last-bit `r3` difference cannot produce a `6.85e-7` RELATIVE error. So the
prediction is that they do NOT fall on the same cell: at the cell carrying
`max|delta p_sh2|`, the relative difference `|delta| / |sh2|` is at most
`1e-10`, i.e. the miss is a last-bit effect on a large-`sh2` cell and not a
structural error.
REFUTED if the relative difference at that cell exceeds `1e-10`; a refutation
means a STRUCTURAL difference, not a rounding one, and outranks P4 as the
round's finding.

**P6 — carried forward from round 103's OPEN item 2.** Evaluating the model's
right-hand-side expression on NEMO's recorded operands (a) eagerly and (b)
under `jax.jit` of an ISOLATED closure gives results that are bit-identical to
each other and to the NumPy transcription, while (c) the same expression
inside the production step differs from all three in 11,029 cells at
`1.1102230246251565e-16`. That pattern identifies XLA fusion inside the full
step, not an association difference in the model's source.
REFUTED if (a) already differs from the NumPy transcription, in which case the
model's association differs from NEMO's and that is a second, AT-BAR-level
in-block owner; or if (a) and (b) differ from each other, in which case the
isolated-closure JIT is the discriminator and note L's effect is reproduced at
closure scale. Every arm is labelled "isolated-closure eager",
"isolated-closure JIT" and "production step" per note L-amend; the second is
never called production.

**P7 — landing.** No candidate fix lands this round. The named owner is
upstream of `zdfsh2` (P4) or inside it (P3 refuted); either way a fix moves a
SHARED metric or a shared shear helper that DINO and every other card run, and
decision 41 forbids landing downstream of an owned non-exact upstream output.
The status is therefore HELD. REFUTED if the walk names an owner whose fix is
confined to the GYRE card's own selected arm AND the Rule-12 ladder (kt=1..10)
and the days 1-30 arm pass against the immutable round-96/97 before arm
(kt2 U/V `2.7377110452773967e-12` / `3.284922138989399e-12`; kt3 T/S
`1.627497246303733e-4` / `6.327735185607253e-6`; day-30 T RMS
`1.2397011295506804e-2` K), in which case it LANDS.

## Controls that must fire

The new `zdfsh2` rows are worthless unless they can fail. A plant
`stage-shear-operand-ulp` advances exactly one bitwise cell of the recorded
`sh2` reference on the binding `NEMO_TKE_RECORDED` arm, exercising the same
path as the clean run. The clean `all_operands_recorded` row must have zero
unequal cells and the planted row exactly one; the sub-walk aborts internally
if the targeted row does not move from clean to clean+1, so the proof is a
hard check and not the exit code. The plant run must NOT print a success word
anywhere — round 103 fixed one that printed `STATUS PASS` while firing, and
that pattern is followed.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No configuration field, default, scheme
selection, threshold or carried state changes. The year harness, the
reconciliation gate, the freshwater pair and the #1484 guard are read only.
Held patches under `manifests/` are not re-evaluated: none of them names a
statement in `zdfsh2.f90:83-119` or `domqco.f90:214-215`.
