# Preregistration: NEMO-testcases L2 GYRE round 101 TKE statement boundary

Date: 2026-09-16. Frozen at incoming production tip
`569599e61db0e88970e8f2d7d15636383f61cb6f` before creating or parsing any
Round-101 NEMO record and before changing production physics. Evidence belongs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round101/`.

## Question and magnitude

Round 100 left the first owned program at kt=1 stage 1. Its post-hoc comparison
of the independently registered Round-97 full-RHS arm and Round-99
full-RHS/W/clock arm found the same Rule-12 veto in both: 85/954 moved rows,
56 violating rows, no classification change, and first-over-bar kt2 U/V. The
W/clock additions therefore do not explain or repair the veto. This round
will preserve that comparison as post-hoc attribution and inspect the one
same-stage state family already non-bit in the model chain: the TKE closure
that feeds the staged vertical solve.

The magnitude targets remain kt3 T `1.627497246303733e-4 K` and day-30 T RMS
`1.2397011295506804e-2 K`. A TKE statement is not presumed to own either
number. No candidate can land without closing the production stage row and
passing the complete 954-row Rule-12 gate.

## Compiled execution order

The record-producing compiled program constructs `rn2b`, copies it to `rn2`,
and calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` before the external mode and all three
RK3 stages at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:154-201`.
The active `np_TKE` arm computes shear, calls `zdf_tke`, and copies its
`avm_k/avt_k` output before EVD at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:317-354`.
Inside that call, the compiled order is `tke_tke` and then `tke_avn` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-201`.

The first routine first assigns the surface and bottom TKE boundaries at
`zdftke.f90:273-315`, executes the active Langmuir update
`en = en + rn_Dt*zus3*(zwlc**3)/zhlc` at `zdftke.f90:318-385`, forms the
matrix and complete budget RHS at `zdftke.f90:394-434`, and then performs the
three recurrences and terminal floor/mask at `zdftke.f90:465-483`.
These are executing lines: the admitted Round-59 header resolves
`ln_lc=.true.`, `nn_pdl=1`, `nn_mxl=3`, and `kt=2`.

## Why the existing record cannot decide the statement

The admitted Round-59 record provides `en_entry`, the matrix, the assembled
`rhs_pre_sweep`, and `en_post_sweep`, but no `en` image after the boundary
assignments or after the Langmuir update. Round 62 consequently localized the
first recorded residual only to 238 RHS cells and explicitly could not
distinguish a subterm. A source reconstruction or an isolated JIT closure is
not a production-step substitute under operator note L-amend.

Therefore the next discriminating measurement genuinely needs two absent
NEMO operands. Round 101 will add a WRITE-only, inactive-off recorder to a new
target cloned from the exact Round-59 source card. It makes no namelist,
configuration, timestep, or numerical change.

## Frozen record and controls

The new fixed-layout record is
`oracle_tke_statement_walk_kt00000002.bin`. It contains a 16-byte magic,
thirteen native 32-bit header integers, and five `32*22*31` binary64 arrays in
this exact order:

1. `en_entry`, immediately before `CALL tke_tke`;
2. `en_after_boundaries`, after the surface/bottom assignments and before the
   `ln_lc` arm;
3. `en_after_langmuir`, after the active Langmuir arm and before matrix/RHS;
4. `rhs_pre_sweep`, after the budget assignment and before recurrence one;
5. `en_post_sweep`, after recurrence three and the terminal floor/mask.

The expected physical EOF is exactly `873028` bytes
(`16 + 13*4 + 5*32*22*31*8`). The reader must reject wrong magic, wrong
header, truncation, trailing bytes, non-binary64 builds, a producer-stamp
mismatch, or a non-kt2 record. The entry, RHS and post-sweep duplicate fields
must be bit-identical to the independently admitted Round-59 record. A
one-ULP change to a consumed new boundary must make the named comparison row
non-bit and the plant process exit nonzero. Every non-zero acquisition exit
must print a named `REFUSE:` line first.

The acquisition must use a new target name, clone `GYRE_PISCES`, copy the
Round-59 source card file by file, apply only additive patches, prove the
patched sources with `gfortran -fsyntax-only`, preserve the ten-step horizon,
run the standard twin admission against the Round-59 run, and admit only the
new statement-walk record. The agent must not run `makenemo` or `mpirun`.

## Frozen predictions and falsifiers

The working prediction is that a time-matched production step agrees through
`en_after_boundaries` and first becomes non-bit at `en_after_langmuir`; that
would name the active compiled assignment at `zdftke.f90:374-380`. This is
**confirmed** only if the production-step entry and boundary rows are BIT and
the post-Langmuir row is non-BIT. It is **refuted** by any earlier boundary
miss or by a BIT post-Langmuir row.

The fall-through outcomes are registered now:

- non-BIT `en_after_boundaries` moves the first statement to the compiled
  surface/bottom block at `:273-315`;
- BIT through Langmuir but non-BIT `rhs_pre_sweep` names the complete budget
  assignment at `:416-433`;
- BIT through the RHS but non-BIT `en_post_sweep` names the first recurrence
  at `:466-468` and requires a recurrence-by-recurrence continuation;
- all five rows BIT refutes TKE as the compensating owner at this boundary.

No numerical candidate is eligible in this record-request round. After the
record arrives, every model-side boundary will be measured through the full
public production step; labels will remain `isolated-closure eager`,
`isolated-closure JIT`, and `production step`, and only the last can support a
stage claim. If the predicted Langmuir boundary fires, the existing literal
implementation is re-evaluated only at this stage and only with a
production-step plant; prior eager-only and trajectory-refused results are
not promoted.

## Frozen lane dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | remain at kt=1 stage 1; add only the missing compiled-order TKE boundaries |
| GYRE kt=1--10 | no candidate this round; retain the immutable Round-96/97 baseline and Round-97/99 post-hoc attribution |
| GYRE days 1--30 | no candidate this round; retain day-30 T RMS `1.2397011295506804e-2 K` |
| LOCK_EXCHANGE-zco | no shared numerical change; construction/coverage only |
| OVERFLOW-zps | no shared numerical change; partial-cell construction retained |
| DINO | **SHARED-STATEMENT RISK:** it executes the shared TKE program; no neutrality claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** resolve its integrator, then record native production-step TKE boundaries with the same plants |

No production configuration, carried state, coefficient, stabilizer, year
harness, reconciliation gate, freshwater pair, #1484 guard, NEMO source, or
NEMO executable may change in the repository.
