# Preregistration: NEMO-testcases L2 GYRE round 102 TKE production boundary

Date: 2026-09-16. Frozen at incoming production tip
`29af6665f77ca8b958af33c91eaa08d1e0aa010f`, after the operator's Round-101
acquisition exited 66 but before parsing any of its five scientific arrays,
before exposing a new legoESM production-step boundary, and before changing
production physics. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round102/`.

## Operational fact and admission question

The operator's run reached NEMO `STOP 0` and produced
`oracle_tke_statement_walk_kt00000002.bin` at the preregistered physical EOF
of 873028 bytes. Its executable and copied executable have the same frozen
SHA-256, and the target run's independently written legacy Round-59 operand
record is byte-identical to the admitted Round-59 source record. Admission
nevertheless stopped at the new-record duplicate check, before any new
statement boundary was interpreted scientifically.

The compiled writers expose a registered semantic mismatch in that check.
The new writer copies every `jpk` value of `en` into `r101_rhs` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:96-103`
and writes it at `:116-123`. The independently admitted writer initializes
its RHS record to exact zero and copies only `1:jpkm1`, explicitly describing
`jpk` as outside the solve, at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:162-190`.
The compiled solve consumes levels `2:jpkm1` and states that `e(jpk)=0` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:399-443`; its
recurrences likewise stop at `jpkm1` at `:475-493`.

Before admission, the existing consolidated stage gate will report, for each
duplicate pair, both the whole stored array and the compiled-consumed domain
`1:jpkm1`, plus the unconsumed `jpk` sentinel. The frozen prediction is:

- `en_entry` and `en_post_sweep` are BIT over their complete stored arrays;
- `rhs_pre_sweep` is BIT over every `1:jpkm1` cell;
- any RHS duplicate inequality is confined to `jpk`, where one recorder
  copied live, unconsumed storage and the other deliberately wrote zero.

This prediction is **confirmed** only by zero unequal cells on every complete
entry/post-sweep array and every consumed RHS cell. Any earlier mismatch, any
consumed RHS mismatch, or a mismatch outside the registered RHS `jpk` sentinel
**refutes** it. Confirmation retracts Round 101's over-broad whole-array RHS
duplicate claim in the executable gate: the exact bar remains zero unequal on
every compiled-consumed cell, while the non-semantic sentinel is reported and
must never be called BIT. A one-ULP plant in a consumed RHS cell must make the
gate exit nonzero. Refutation makes the record inadmissible and requires a new
additive recorder under a new NEMO target; no scientific array will then be
used this round.

The immutable acquisition producer remains commit
`29af6665f77ca8b958af33c91eaa08d1e0aa010f`. Round-102 analysis commits may
not rewrite that provenance. The admission script will pin and verify the
producer commit, binary digest, source-card manifests, exact EOF, unchanged
restart/mesh inputs, and standard twin outputs independently of the clean
analysis commit that runs the gate. It will use the existing run directory
and must not invoke `makenemo` or `mpirun`.

## Scientific question and execution modes

If and only if the record is admitted, extend the existing Round-46/51
consolidated stage-twin gate; do not build a second harness. Drive kt=1 stage
1 through `LatLonCGridOceanModel.step`, which calls the production
`self._step_jitted`, from NEMO's recorded stage entry. Expose the actual TKE
images at the matching boundaries from inside that full compiled step. Report
three labels distinctly wherever present:

1. `isolated-closure eager`;
2. `isolated-closure JIT`;
3. `production step`.

Only `production step` can name a statement. The binding raw-bit domain for
each NEMO/model boundary is all 32 x 22 owned cells on levels `1:jpkm1`; the
unconsumed `jpk` image is separately reported. No tolerance or wet-mask
suppression is allowed. A one-ULP perturbation of a consumed production-step
boundary must flip its named row and make the plant exit nonzero.

The compiled order is fixed as follows. `zdf_tke` brackets `tke_tke` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:185-207`.
The entry is recorded at `:268`; the surface and bottom assignments execute at
`:280-322`; the after-boundary record is taken at `:324`; the active Langmuir
calculation and assignment execute at `:326-393`, with the write to `en` at
`:382-391`; its boundary record is taken at `:395`; the matrix and complete
budget RHS execute at `:399-443`; the RHS record is taken at `:472-473`; and
the three recurrences plus floor/mask execute at `:475-495`.

## Frozen predictions, first-statement rule, and falsifiers

The Round-101 prediction is retained without seeing the record: the production
step will be BIT through `en_after_boundaries` and first become non-BIT at
`en_after_langmuir`. This is **confirmed** only if `en_entry` and
`en_after_boundaries` are BIT on the complete binding domain and
`en_after_langmuir` is non-BIT. It is **refuted** by an earlier non-BIT row or
by a BIT post-Langmuir row.

The predeclared fall-through is:

- non-BIT after the boundary block names the first executing assignment in
  `zdftke.f90:280-322` whose preceding production boundary is BIT;
- BIT through Langmuir but non-BIT at `rhs_pre_sweep` names the first
  non-bit statement within `:399-443` after a statement-ordered continuation;
- BIT through the RHS but non-BIT at `en_post_sweep` names the first
  recurrence/floor statement within `:475-493` after a recurrence-ordered
  continuation;
- all five boundaries BIT refutes TKE as this stage's first owned statement.

The first non-bit statement is the first compiled write whose immediately
preceding production boundary is BIT and whose output boundary is non-BIT.
An inherited mismatch cannot be called owned. The consolidated kt=1/kt=2
given-NEMO-entry and chained stage tables must be regenerated in the same run.
The first owned stage remains kt=1 stage 1 unless those production rows refute
it.

No numerical candidate is preregistered yet because the discriminating
operand is deliberately unknown. If the walk names a uniquely source-cited,
same-stage correction without a configuration or carried-state choice, a
committed addendum must freeze its exact candidate values, affected rows, and
falsifiers before applying or scoring it. Otherwise the round remains HELD.
No eager-only or isolated-JIT equality makes a candidate.

## Rule-12 and magnitude registration

Any eligible addendum candidate must make the named stage-twin row BIT given
NEMO's entry and then pass the complete 954-row ladder. No AT-BAR row may
leave the bar, first-over-bar may not move earlier, and every moved row must be
registered. The immutable before arm is the Round-96/97 arm. The mandatory
headline rows are kt2 U/V
`2.7377110452773967e-12` / `3.284922138989399e-12`, kt3 T/S
`1.627497246303733e-4` / `6.327735185607253e-6`, and day-30 T RMS
`1.2397011295506804e-2 K`. A candidate that fails either stage exactness or
Rule 12 is held, not landed.

## Frozen lane dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | remain at kt=1 stage 1; admit and walk only the missing TKE statement boundaries |
| GYRE kt=1--10 | run only for a separately committed eligible candidate; otherwise retain the immutable before arm |
| GYRE days 1--30 | run only after a ladder-eligible candidate; otherwise retain day-30 T RMS `1.2397011295506804e-2 K` |
| LOCK_EXCHANGE-zco | must execute any shared statement or be shown not to execute it before landing |
| OVERFLOW-zps | must execute any shared statement or be shown not to execute it before landing |
| DINO | **SHARED-STATEMENT RISK:** it executes the shared TKE program; no neutrality claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** resolve its integrator and score the identical production boundaries |

No production configuration, carried state, coefficient, stabilizer, year
harness, reconciliation gate, freshwater pair, #1484 guard, NEMO source, or
NEMO executable may change without a new preregistered and user-authorized
decision.

## Pre-measurement addendum: represented entry domain

Frozen after the duplicate/admission audit above confirmed the record, but
before reading or comparing either new scientific boundary. The model carries
NEMO TKE levels `2:jpkm1` as its 29-level prognostic field. It does not carry
level 1 because the compiled program overwrites that level unconditionally at
the first surface-boundary statement (`zdftke.f90:280-289`) before any later
consumer. Calling a reconstructed current-forcing surface value the model's
*entry* would therefore be false.

The entry row binds, without a wet mask or tolerance, over all 32 x 22 x 29
represented cells corresponding to NEMO `2:jpkm1`. Starting at
`en_after_boundaries`, the trace prepends the model's actual virtual-surface
Dirichlet operand, so every later row binds over all 32 x 22 x 30 NEMO levels
`1:jpkm1` as frozen above. The omitted entry level is reported
`UNMEASURED_WITH_SPEC` with this exact overwrite specification; it cannot hide
or excuse an interior entry mismatch. The production trace must capture the
actual values used by the full jitted step: it may assemble the model's split
surface/interior representation for scoring, but may not substitute a NEMO
value or an isolated replay.
