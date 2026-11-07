# ORCA2 round 68 preregistration — resume the independent month acquisition

Date frozen: 2026-09-28

Base: `950b1e4c281ff3677b4613bd5987cf932e1cd6a9`

Claim label: **independent**.  This round repairs the acquisition instrument;
it does not score a month.  legoESM's later month starts from its own ORCA2
card state and will be compared only with NEMO's own from-rest trajectory.

Sea ice remains out of scope.  The card's six-entry `unmeasured_features`
tuple and all selectors remain frozen.

## Observed failure boundary

The operator's round-67 run completed the ten-step calibration with NEMO
`STOP 0`, then the Python admission process failed before the 240-step target
was staged.  The traceback ends at the gate's repository-local
`scripts.validate` import.  The launcher exports package directories but not
the repository root, so direct execution of the gate cannot resolve the
namespace package.  This is an instrument invocation defect, not a NEMO or
model result.

## Frozen predictions and falsifiers

1. Adding the repository root to the launcher's existing `PYTHONPATH` makes
   the committed gate importable when invoked exactly as the launcher invokes
   it.  Continued `ModuleNotFoundError`, or a different import failure, refutes
   the diagnosis.
2. The completed calibration is retained read-only and must admit bit-exact
   against every variable in the four pinned restart shards.  Its recorded
   producer commit must equal the round-67 tip
   `950b1e4c281ff3677b4613bd5987cf932e1cd6a9`; any unequal payload or provenance
   mismatch refuses the resume.
3. The month target is absent.  Resume stages and runs only the 240-step arm;
   it never rebuilds or reruns the completed calibration.  Existing month
   output, missing calibration output, or any deck delta besides `nn_itend`
   and `nn_stock` refuses before `mpirun`.
4. A resumed month records the round-68 launcher commit separately from the
   round-67 calibration commit.  Admission validates both explicit commits;
   it never trusts a commit value read from a record as its own expectation.
5. The round remains `STOPPED_FOR_RECORD`: the operator must run the repaired
   resume mode because PMIx is unavailable in the sandbox.  Month ranking is
   deferred until the resulting record passes the admission gate.

Failed predictions remain **REFUTED** in the receipt.

## Required controls and validation

- Test the launcher-equivalent direct gate import with the exported
  `PYTHONPATH`.
- Show the existing calibration passes its complete bit-exact admission.
- Plant wrong calibration and month producer commits; both must refuse.
- Shell syntax, focused tests, shared-card battery, citation audit with a
  firing plant, default citation audit, and separate read-only Codex review.
- No `packages/` file changes; model trajectory gates and the wide ocean
  battery are not applicable to this acquisition-instrument repair.

ASKED: repair the operator-reported round-67 admission failure and resume the
missing independent month comparator.

UNASKED: configuration, selector, threshold, forcing, stabiliser, carried
state, model arithmetic, sea-ice implementation, and the held QCO/RK change.
