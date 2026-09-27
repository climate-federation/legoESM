# Round 182 receipt — DINO geometry blocker discrimination

**Status: STOPPED_FOR_DECISION.**  Round 181's carried step-entry `rn2b`
candidate remains held.  The DINO raw mesh is already exact and positive.
Routing NEMO's live `Kmm` thickness through every first-step consumer removes
the reported geometry refusal, but the completed unchanged/control DINO step
then returns every prognostic family non-finite.  The same geometry guard only
detects that whole-state failure at step 2.  Repairing that broader developed-
state bridge is outside this round's geometry scope, so both production and
tests were restored byte-for-byte and no trajectory baseline moved.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round182.md`, commit
`1f32d54c1`.  Diagnostic instrument:
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round182_dino_e3w_trace.py`.
Production restoration: `aa20cb13a`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round182/`.

## Compiled-source route

The DINO z-coordinate branch first constructs the full-grid 3-D scale factors
and depths, and only afterwards derives the wet top/bottom indices, at
`DINO/BLD/ppsrc/nemo/usrdef_zgr.f90:123-135`.  The called MI96 routine forms
`pe3t`/`pe3w` from depth and reconstructs depth from those scale factors at
`DINO/BLD/ppsrc/nemo/zgr_lib.f90:199-209`.  Thus the raw geometry is a full-
grid positive operand; it is not a wet mask.

The compiled domain separately sums reference depth through `tmask` and makes
the reciprocal zero on dry points at `DINO/BLD/ppsrc/nemo/domain.f90:194-216`.
The QCO routine consequently computes `r3t = pssh * r1_ht_0` at
`DINO/BLD/ppsrc/nemo/domqco.f90:186-207`.  The active `bn2` statement divides
by `e3w_3d * (1+r3t(Kmm))` before multiplying by `wmask` at
`DINO/BLD/ppsrc/nemo/eosbn2.f90:1507-1517`.

For the DINO slope call, the stage passes `Nbb,Nnn` to `ldf_slp` at
`DINO/BLD/ppsrc/nemo/stpmlf.f90:197-216`.  The callee names those arguments
`Kbb,Kmm` and uses `r3t(Kmm)` for the live mixed-layer depth at
`DINO/BLD/ppsrc/nemo/ldfslp.f90:139-177`, for the W-level MLD distance at
`DINO/BLD/ppsrc/nemo/ldfslp.f90:215-216`, and for the W-point limiter and
depth at `DINO/BLD/ppsrc/nemo/ldfslp.f90:320-345`.  The temporary diagnostic
repair therefore routed the returned `Kmm` stretch through all of those
consumers; it did not replace bad cells or relax the positivity guard.

## Discriminating measurements

The committed mesh gate passed before any repair.  Both `e3w_1d` and the full
`e3w_0` report zero unequal cells and zero maximum difference from the DINO
oracle, and the z-coordinate raw `e3w_0`, `e3uw_0`, and `e3vw_0` identities
are exact.  This refutes the hypothesis that the DINO raw mesh is zero or
non-positive on land/halo cells.

The trace wraps the established production DINO twin and labels every call to
the single `bn2` implementation.  It changes no state unless its explicit
plant is requested.  On the restored starting implementation, its first four
production-JIT rows are:

| call / consumer | minimum finite value (m) | non-finite | zero | negative |
|---|---:|---:|---:|---:|
| 1, step-entry `rn2b` | 1.0284750425959922e1 | 0 | 0 | 0 |
| 2, step-entry `rn2` | 1.0284750425959922e1 | 0 | 0 | 0 |
| 3, native-slope W N2 | 1.0289052027610524e1 | 347,200 | 0 | 0 |
| 4, native-slope MLD N2 | 1.0289052027610524e1 | 347,200 | 0 | 0 |

The 347,200 invalid cells are non-finite, not dry zeros, and include the
entire live interior operand.  Preregistered prediction 2, which required
invalid values only outside the wet interface, is **REFUTED**.

With the source-exact `Kmm` route applied temporarily, calls 1 through 20 of
the first step all have zero non-finite, zero, and negative cells.  Calls 19
and 20 have minimum `1.0284750498704209e1 m`.  Nevertheless the completed
first production step reports:

| returned field | non-finite cells |
|---|---:|
| `eta` | 10,348 |
| `T` | 342,134 |
| `S` | 342,134 |
| `u` | 379,692 |
| `v` | 372,528 |

Those counts cover every reported prognostic family.  Calls 21 and 22, which
belong to the following step, then see 347,200 non-finite geometry values and
the existing guard refuses them.  Therefore the guard is a downstream
detector, not the cause of the first-step state destruction.  The evidence
does not yet localize which upstream developed-state bridge operand causes
that destruction; attributing it to geometry or to Round 181's `rn2b` route
would be false.

The production call plant sets exactly one cell of call 1 to zero.  It prints
`TRACE_PLANT call=1`, reports `zero=1 invalid=1`, triggers the named
`raw-mesh e3w_int` refusal, and exits 1.  Separately, removing the temporary
`Kmm` route from the focused literal-slope test makes that test fail at the
same named guard; restoring it makes the test pass.  The diagnostic is thus
non-vacuous at the production closure and at the focused consumer.

## Frozen-prediction disposition and landing verdict

| prediction | disposition |
|---|---|
| raw stored mesh has no invalid cell | **CONFIRMED** by the mesh gate |
| invalid live cells occur only outside wet interfaces | **REFUTED**: 347,200 non-finite interior cells |
| NEMO-scoped geometry route preserves wet values and removes first-step invalid geometry | **CONFIRMED for calls 1-20**, but the whole step returns non-finite state |
| invalid-cell plant fires and exits nonzero | **CONFIRMED** for the one-cell production plant; the preregistered four-class matrix was not reached |
| both DINO arms finish and execute the recurrence | **REFUTED** by the control arm's whole-state first-step failure |
| Round-181 local proof and complete trajectory reproduce | **NOT RUN** because the DINO prerequisite failed |
| Round-181 GYRE trajectory values reproduce | **NOT RUN**; no new arm was promoted |
| census is complete | **UNCHANGED** from Round 181; both DINO Kamm cards remain mandatory and unmeasured |

Round 181 remains the authoritative measurement of the held candidate:
day-30 T RMS `2.3276772050683987e-06 K`, day-240
`6.5861718814795174e-05 K`, and day-360
`2.6709923853294689e-03 K`, with kt=1/2 unchanged and first-over-bar at kt=3.
Those numbers are quoted, not remeasured in Round 182.  The corresponding
production default remains `recompute`; no Round-181 candidate hunk is present
in the final tree.

Because neither DINO Kamm arm can complete even with the geometry statement
transcribed, Round 182 cannot establish whether the held candidate changes a
DINO certified number.  GYRE, generic-card, tank, month, and year gates were
not rerun: the blast-radius prerequisite stopped the landing before those
measurements, and the restored production implementation is byte-identical to
the starting tip.

## Review, citations, and tests

REVIEW_PENDING

CITATION_PENDING

TESTS_PENDING

Choices made: none.  No NEMO acquisition is required.

## DECISION NEEDED

May Round 181's compiled-source-exact, one-variable carried-`rn2b` candidate
land with DINO registered **UNMEASURED** because the unchanged/control DINO
developed-state arm independently destroys its full first-step prognostic
state?  **Pick: yes** — preserve the DINO refusal as explicit debt and land
the 28.24x / 249.74x / 4.20x GYRE day-30/day-240/day-360 improvement.  The
alternative is to require a separately scoped DINO developed-state bridge
repair and a new certified arm before reconsidering the candidate.

## OPEN

Pending the decision, either (a) reapply and land Round 181 under a user-
approved DINO-unmeasured waiver, rerunning its complete GYRE and census gate,
or (b) preregister a DINO developed-state bridge walk that names the first
non-finite statement inside the first production step.  Do not treat the
step-2 geometry refusal as that first statement.
