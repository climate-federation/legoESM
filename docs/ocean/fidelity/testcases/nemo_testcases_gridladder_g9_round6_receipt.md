# Grid ladder rung G9 — round 6 receipt (gridladder_r51, third resume)

Status: **LANDED FIX; G9 NOT CERTIFIED; DOWNSTREAM ACQUISITION NEEDED.**
The independent review refuted the second-resume XLA attribution. G9's
stage-3 literal FCT now divides by NEMO's stored Kmm thickness. That closes
the isolated after-advection comparison exactly, but it does not change any
kt-1 scored row and kt 2 remains over the bar. The admitted record does not
contain the next discriminating NEMO value, so no further production change
is made.

- codex review: operator
- independent review: operator (Claude)
- evidence: `gridladder_rounds/round51/`
- implementation: G9-selected stage-3 literal FCT only

## 1. Record status remains honestly PARTIAL

The round-51 FCT file has five present frames: `entry`, `first_midpoint`,
`average_rhs`, `anti_pre`, and `anti_post`. All 19 present arrays for each
tracer are finite and read completely. Its declared sixth frame,
`rhs_final`, is absent and is not inferred or described as measured.

Admission also compares all 42 pre-existing kt-1 and kt-2 records with the
round-46 source byte-for-byte. That passivity result remains 42 / 42. The
reviewer's scripts, source snapshot, both score files, and logs were copied
out of `/tmp` to
`gridladder_rounds/round51/independent_review_third_resume/`.

The earlier claim (a) is narrowed: its evidence covers **kt 2, stage 1 only**.
It is not evidence that the kt-1 stage-3 program or all later stages are
bitwise exact.

## 2. Corrected FCT owner and fix

The 16 first-midpoint differences are not an XLA reconstruction of one
stored value. They are all at `(y=4, x=50)`, levels
`0,1,2,3,5,6,7,8,9,11,13,14,15,17,18,19`, and arise because the whole-step
card supplied the thickness rebuilt from averaged SSH while NEMO's midpoint
division reads stored `e3t(Kmm)` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:538`).
In the default hybrid stage, NEMO stores that Kmm ratio as the half-sum of
the two ratios (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221`).

The card already constructs that stored stage-3 array for vertical diffusion.
The G9-only literal-FCT branch now routes the same array to the FCT divisor;
the live transport geometry is otherwise unchanged. A NumPy transcription
of the recorded operands and the review plant both move after-advection from
21 T / 17 S unequal wet cells to 0 / 0. Doubling the selected divisor moves
all 430,552 wet cells, proving that the operand is live.

The new synthetic x64 test calls the helper through `jax.jit` with traced
tracer, stored thickness, live thickness, and timestep. It is bit-exact
against the stored-divisor transcription on every element. Removing the
new routing makes all 32 elements fail; the negative-control output is
retained as `exact_divisor_test_old_wiring_plant.log`.

The change is inside the pre-existing static condition
`stage3_krhs_terms is not None and stage_index == 2`, selected only by G9.
Consequently the same 26 non-G9 cards retain their round-50 byte-identical
jaxprs; G9 alone reaches the assignment.

## 3. Downstream walk and acquisition boundary

The fix is deliberately not credited with the scored kt-1 residue. The
whole-step kt-1 rows are bitwise unchanged: T remains 105 unequal values and
S remains 96, both AT-BAR; ssh, `uu_b`, `vv_b`, u, and v remain exact.

Using only admitted records, the first exposed downstream disagreement is
the pre-implicit tracer content. With later source terms removed, the card's
after-advection content differs at 646 T / 235 S wet cells. After SBC the
counts are 643 / 248; after all explicit RHS terms T has 612 unequal wet
cells. Stored Kbb and Kaa thickness operands are bitwise exact, and the
recorded-operand NumPy/JAX transcription of NEMO's content expression is
bitwise exact. Supplying the source-rebuilt NEMO content closes the implicit
solve and end-of-stage result, so the remaining owner is no later than the
second recurrence in `tra_zdf_imp`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:284-290`).

The existing record saves Krhs boundaries and Kbb/Kmm/Kaa operands, but not
the value actually stored by that compiled recurrence. A further fix would
therefore choose between unmeasured alternatives. The required operator-only
acquisition is:

1. Add a WRITE-only observer named
   `oracle_g9_trazdf_zrhs_kt00000001.bin` to `tra_zdf_imp` for kt 1, stage 3,
   tracers T and S.
2. At level 1 capture `pt(...,Kaa)` immediately after the assignments at
   lines 284-285 and before it is consumed by level 2. At levels 2 through
   `jpkm1`, capture scalar `zrhs` immediately after lines 288-289 and before
   line 290. Store the resulting two interior 3-D arrays plus kt, Kbb, Kmm,
   Krhs, Kaa, dimensions, halo, and precision in the header.
3. Keep the observer additive, require finite/read-complete arrays, and prove
   all 42 pre-existing round-46 records byte-identical before admission.
4. Compare this frame bitwise with the card's existing
   `expose_pre_implicit_content` output. Do not build or run NEMO in an agent
   session.

## 4. Both-label kt 1..10 rescore

The rescore used CPU production JIT, x64, and the round-46 admitted roots.
Rows are ssh, `uu_b`, `vv_b`, u, v, T, and S.

| label | kt | exact / AT-BAR / DEBT | rows over the bar |
|---|---:|---:|---|
| independent | 1 | 5 / 2 / 0 | none |
| independent | 2 | 0 / 3 / 4 | ssh, u, v, S |
| independent | 3..10 | 0 / 0 / 7 each | all seven rows |
| given entry | 1,3,5,7,9 | 5 / 2 / 0 each | none |
| given-entry two-step | 2 | 0 / 3 / 4 | ssh, u, v, S |
| given-entry two-step | 4 | 0 / 0 / 7 | all seven rows |
| given-entry two-step | 6,8 | 0 / 2 / 5 each | ssh, u, v, T, S |
| given-entry two-step | 10 | 0 / 3 / 4 | ssh, u, v, S |

The complete JSON and logs are
`independent_review_third_resume/g9_independent_kt1_10.{json,log}` and
`independent_review_third_resume/g9_given_entry_kt1_10.{json,log}`.
G9 remains uncertified, so none of its proposed fold import is certified by
this round.

## 5. Gates and disposition

The receipt-specific citation gate passes with 3 citations, 0 failures,
0 unmapped citations, and 0 failing map entries. The ORCA2 default gate and
private-import ratchet pass together: **144 passed in 1324.13 s**.

ASKED: correct the refuted XLA attribution; route stored Kmm thickness to the
G9 literal FCT; add an exact planted test; preserve non-G9 graphs; walk the
next owner; rescore both labels; run citation/default/private gates; commit.

UNASKED, offered for revert: none.

UNVERIFIED: the missing FCT `rhs_final` frame and NEMO's compiled stage-3
`trazdf` content store. Neither is represented as observed.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r51/repo
ACQUISITION_NEEDED: operator must capture `oracle_g9_trazdf_zrhs_kt00000001.bin` exactly as section 3 defines; agents must not build or run NEMO.
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — stored Kmm divisor fixed and exact-tested; both labels rescored; G9 remains uncertified pending the stage-3 `trazdf` content record.
