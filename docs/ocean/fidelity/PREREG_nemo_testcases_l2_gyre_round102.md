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

## Post-boundary preregistered decomposition

Frozen after the admitted pre-closure production run reported all five input
fields and `en_entry` BIT, followed by 260 unequal cells at
`en_after_boundaries` with maximum absolute difference
`1.734723475976807e-18`, and before decomposing that row. The original
Langmuir-first prediction is therefore **REFUTED** and remains so.

The boundary block has only two active compiled writes in order: the surface
assignment at `zdftke.f90:284-289`, followed by the bottom assignment at
`:299-308` (the `ln_isfcav` arm at `:309-320` is inactive on this card). The
existing consolidated gate will split the already measured boundary image
into:

1. surface level 1;
2. NEMO-changed interior cells, defined bitwise by
   `en_after_boundaries != en_entry` on levels `2:jpkm1`;
3. unchanged interior cells, the exact complement.

The frozen prediction is that the surface and unchanged-interior rows are BIT
and all 260 inequalities lie on the changed-interior bottom assignment. That
would name the first non-bit statement as
`en(ji,jj,mbkt(ji,jj)+1) = MAX(zebot,rn_emin)*ssmask(ji,jj)` at compiled
`zdftke.f90:307`. A non-bit surface row instead names the earlier compiled
surface assignment at `:285`; a non-bit unchanged-interior row refutes the
two-statement mapping and forbids an ownership claim. The three masks must
partition every binding cell exactly, and the production ULP plant must still
add exactly one inequality to its named full boundary row.

## Post-decomposition operand attribution

Frozen after the decomposition reported that the surface assignment alone
contains all 260 boundary inequalities (maximum absolute difference
`1.734723475976807e-18`), while the NEMO-changed bottom set is empty and the
unchanged interior is BIT, but before exposing or comparing the production
stress modulus. The bottom-first prediction above is therefore **REFUTED**
and remains in this preregistration.

The compiled statement first evaluates
`zbbrau = rn_ebb / rho0` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:257` and then writes
`en(ji,jj,1) = MAX(rn_emin0,zbbrau*taum(ji,jj))` at `:284-289`.
The shared implementation evaluates the same source association. The frozen
hypothesis is therefore that the observed surface image is inherited from
the production forcing's reconstructed stress modulus, not owned by this TKE
assignment: the production `taum` will be non-BIT against Round 59's recorded
`taum_entry`, a scalar replay from NEMO's recorded `rn_ebb`, `rho0`,
`rn_emin0`, and `taum_entry` will reproduce the Round-101 surface image with
zero unequal cells, and replacing only `OceanSurfaceForcing.taum` by the
recorded modulus before calling the full production `step` will make the
surface-assignment row BIT.

This is confirmed only if all three clauses hold in the production-step
instrument. It is refuted if the recorded-operand scalar replay is non-BIT,
if the production modulus is already BIT, or if the one-operand production
substitution leaves any surface cell unequal. The comparison domain is all
32 x 22 owned surface cells without tolerance or mask. The existing model-
forcing arm must remain reported separately and may not be relabelled as
NEMO-entry. If confirmed, the first non-bit TKE statement must be sought
after the boundary block using the NEMO-`taum` production arm; the surface
statement is inherited and is not an eligible patch. If refuted, no TKE
candidate is eligible until the exact constant/evaluation owner inside this
same compiled statement is decomposed in a new committed addendum.

## Post-surface Langmuir evaluation discriminator

Frozen after the production-step operand test confirmed all three clauses
above. With recorded NEMO `taum`, `en_entry` and `en_after_boundaries` are BIT,
then `en_after_langmuir` first becomes DEBT with 3,223 unequal cells and
maximum absolute difference `2.833486841675906e-05`. The model-forcing
diagnostic remains separately non-BIT at the inherited surface write (303
unequal `taum` cells produce 260 unequal surface cells); it is not the arm
used for statement ownership.

The NEMO-identity GYRE card currently selects
`tke_langmuir_evaluation="vectorized"`. The compiled program instead forms
`zWlc2 = zcsd*taum` at `zdftke.f90:350-352`, then forms `zpelc(1)` and the
ordered vertical recurrence
`zpelc(jk)=zpelc(jk-1)+MAX(rn2b,0)*gdepw*e3w` at `:355-361`, before the
reverse crossing search, `zhlc`, and source write at `:363-393`.

Without changing the shared card, the existing consolidated gate will add
one explicitly labelled diagnostic production arm whose only static change
is `tke_langmuir_evaluation="nemo_literal"`; it will retain the same recorded
NEMO closure memory and recorded `taum` and will execute through the full
`LatLonCGridOceanModel.step` / `_step_jitted` path. The frozen prediction is
that this arm makes `en_after_langmuir` BIT. Because `zWlc2` consumes the
already BIT recorded `taum` and the same source constants in both modes, a
BIT literal arm names the first non-bit model statement as the vectorized
replacement of NEMO's ordered `zpelc` recurrence at compiled `:355-361`.

The prediction is confirmed only if the diagnostic arm remains BIT through
the boundary block and makes `en_after_langmuir` BIT on all 32 x 22 x 30
cells. Any boundary mismatch or any post-Langmuir mismatch refutes it and
forbids the recurrence attribution; the next round must then expose the live
`rn2b`, `gdepw`, `e3w`, mask, crossing-index, and source operands individually.
Even if confirmed, this round will not change the card: selecting a different
configured evaluation is a user decision, so the literal arm is evidence for
`DECISION_NEEDED`, not a landing candidate. No 954-row ladder or day-30 arm is
authorized for this diagnostic-only configuration.

## Instrument-routing correction after fail-closed refusal

Frozen after the literal diagnostic exited nonzero before executing a numeric
statement with the named message
`tke_langmuir_evaluation='nemo_literal' requires bottom_level and w_active`.
That refusal is pre-science and does not refute or confirm the numerical
prediction. Inspection of the shared production call shows `w_active` is
already routed by `tke_dry_wmask=True`, but `_tke_bottom_level` routes the
partial-cell `bottom_level` only when the independent bottom-Dirichlet option
is enabled. The compiled Langmuir search independently initializes `imlc` from
`mbkt+1` at `zdftke.f90:363`; the literal implementation therefore correctly
requires this operand even when `bottom_tke_bc=False`.

Before rerunning the frozen diagnostic, route the existing coordinate's
`bottom_level` when either the bottom TKE boundary or literal Langmuir program
consumes it, and relax the silent-no-op guard only for active literal
Langmuir. This is operand plumbing, not a card-selection change: the current
configured vectorized arm must remain byte-for-byte unchanged, and supplying
`bottom_level` to any configuration that consumes neither feature must still
raise. A focused test must cover all three cases. The numerical prediction
and falsifier above remain unchanged.

## Production-plant selector correction

Frozen after the first `stage-tke-production-ulp` invocation exited 1 but its
JSON revealed `plant_target=null`, `clean_n_unequal=0`, and `n_unequal=0`.
That result is **INVALID**, not a passing plant: the main program's generic
plant exit policy made the command red even though the row perturbation never
ran. The row selector retained the obsolete label `NEMO_RECORDED`, while the
binding production arm is labelled `NEMO_TKE_RECORDED`.

Before rerunning, update that selector and its hermetic test to the binding
label and require the reported target to be non-null. The rerun confirms the
control only if the clean boundary has zero unequal cells, the planted row has
exactly one unequal cell, `plant_target` names that row, and the process exits
nonzero. Any other result invalidates the production statement instrument.
