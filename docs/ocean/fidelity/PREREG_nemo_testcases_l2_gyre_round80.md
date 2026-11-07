# Preregistration: NEMO-testcases L2 GYRE round 80 held-candidate disposition

Date: 2026-09-13. Frozen after reading the Round 78 and Round 79 records, the
operator's failed acquisition log, the shared oracle-relative comparison gate,
and the compiled GYRE branch, but before acquiring the missing parent ladder or
rerunning the independent review.

## Scope and inherited blocker

Round 79 changed the one shared split-explicit program to carry NEMO's six
absolute AB3/AM4 histories. Its direct source-order gate made the kt=2
substep-1 history association bit-exact and moved the first non-bit statement
to substep-2 `un_e`. The candidate remains **HOLD**, not landed: it lacks both
the exact preregistration-commit GYRE ladder and a completed independent Codex
review.

The operator ran the Round 79 acquisition script without its required parent
clone argument. It exited at shell parameter expansion before running the
ladder or review. This inherited failure is not a measurement and is not
reinterpreted as one. Round 80 changes no physics, model state, configuration,
restart semantics, test threshold, harness, reconciliation gate, freshwater
pair, #1484 guard, or held manifest.

## Compiled statement and frozen interpretation

The compiled GYRE branch initializes the six history arrays only under
`ll_init`, selects the current external velocity from `puu_b/pvv_b`, and zeros
the after accumulators separately at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378`.
It consumes the absolute current/before/twice-before velocities at the midpoint
at the same compiled source's `:481-509`, then rotates the absolute histories
and makes the next substep current value at `:783-795`. Therefore the missing
whole-step comparison remains required; the exact midpoint replay alone does
not certify the independently advanced trajectory.

The comparison is frozen as the shared cellwise oracle-relative gate already
used by the campaign. Its before arm is a newly captured full kt=1..10 ladder
from exact commit `0078f9cc851c921176afd8e30660373129321712`. Its after arm is
the existing clean Round 79 artifact made at `add5cbd555b99ebe32ce4e46ae69538f67951883`,
whose JSON and residual sidecar SHA-256 values are already sealed in the Round
79 receipt. No scratch toggle, reconstructed legacy state, or different gate
may replace either arm.

## Frozen predictions and falsifiers

1. The corrected acquisition takes no positional arguments, creates a new
   disposable shared clone, materializes exact commit `0078f9cc`, proves that
   clone clean, and stamps that exact commit into the parent ladder. Any other
   commit, dirty stamp, missing sidecar, or absent 10-step coverage refuses the
   record.
2. The parent and after reports have identical NEMO oracle fields, selectors,
   precision policy, scored-row names, shapes, and masks. The shared comparison
   must preserve every parent AT-BAR row and must not move `first_over_bar`
   earlier. Its three-ULP and AT-BAR-to-DEBT plants must exit nonzero.
3. Round 79 preregistered kt=2 U/V as the first whole-step rows that must move.
   That statement is tested literally. If both are unchanged, the prediction
   is retained as **REFUTED**, even if later registered rows move and the
   candidate passes the oracle-relative admission bar. No post-hoc relabeling
   of the kt index is allowed.
4. The independent review reads the full `0078f9cc..ac501360` diff, Round 79
   preregistration and receipt, the acquired parent comparison, and the restart
   tests. A final `DO NOT SHIP` forbids promotion. `HOLD` keeps every named
   correction or discriminating measurement open. Only `SHIP` plus a passing
   parent comparison can promote the candidate.
5. If either missing prerequisite is unavailable in this sandbox, Round 80
   stops for the external record with a no-argument `run.sh`; it does not start
   the next pressure/tendency/source walk and does not claim the held physics
   candidate landed.

## Rule 12 disposition

| lane | frozen Round 80 action |
|---|---|
| GYRE source order | Preserve Round 79 result: substep-1 midpoint exact; first remaining row substep-2 `un_e`; no new source walk until HOLD clears |
| GYRE kt=1..10 | Acquire exact `0078f9cc` parent and compare cellwise with sealed Round 79 after arm; every moved row disclosed; no AT-BAR loss; no earlier first-over-bar |
| GYRE days 1..30 | Preserve the already controlled recorded-before comparison; do not rerun or replace either arm in this record-recovery round |
| LOCK_EXCHANGE-zco | Preserve Round 79's measured 50-row comparison; no new change or inference |
| OVERFLOW-zps | Preserve compiled non-execution of the changed continuation branch; no new change or inference |
| DINO | SHARED-STATEMENT RISK remains explicit: DINO uses its separate leapfrog carry, while restart-format compatibility remains a shared surface; no DINO neutrality inferred |
| ORCA2 | UNMEASURED-WITH-SPEC remains unchanged: independently aligned T/S/U/V/SSH plus six absolute histories, native masks, elementwise fp64 equality and normalized L-infinity through kt=1..10; reject AT-BAR loss, earlier first-over-bar, or wet-face history mismatch |

## Exit rule and next walk

This round is **LANDED** only if both inherited prerequisites pass and the held
Round 79 candidate can be promoted without contradicting its frozen register.
Otherwise it is **STOPPED_FOR_RECORD** with one corrected acquisition script.
After promotion, and not before, the magnitude-ranked walk starts immediately
after the exact substep-1 midpoint and records the pressure-gradient,
Coriolis/drag/forcing, velocity update, and swap operands that create the
substep-2 `un_e` mismatch.
