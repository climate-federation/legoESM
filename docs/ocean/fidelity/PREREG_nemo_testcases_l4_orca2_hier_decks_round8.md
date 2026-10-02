# ORCA2 hierarchy decks round 8 preregistration — rung-5 admission repair

Date: 2026-10-01

Base: `a4df4b624` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side acquisition and admission only.  The operator completed the
round-7 rung-6 calibration and rung-5 runs, but admission crashed in its first
plant because the round-7 gate passed a removed `tke_active` keyword to the
rung-6 resolved-state validator.  This round repairs that caller, admits the
existing runs without rerunning NEMO, proves the repaired recorder passive on
rung 6, and only after admission reads the compiled shortwave fallback needed
to define rung 4.  No legoESM package, card, recipe, physics, configuration,
hierarchy switch, threshold, carried state, or sea-ice declaration changes.

All run claims are labelled **independent**: each NEMO rung starts from its own
from-rest T/S initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD8-P1 admission crash | The crash is solely the stale keyword at the round-7 calibration-to-rung-6 validator boundary; rung 6 already owns the TKE-off resolved checks. | Removing only that keyword lets the field-name plant reach and fire in the record parser. | The plant still crashes before its intended mutation or another interface mismatch appears. |
| HD8-P2 active-runoff passivity | The repaired recorder is passive when runoff is active. | All 480 rung-6 calibration frames and both terminal restart shards are byte-identical to the admitted rung-6 record. | Any frame or restart shard differs. |
| HD8-P3 inactive-runoff record | The completed rung-5 run contains the declared self-describing schema. | All 480 frames contain eight finite PRESENT fields and exactly `rnf` and `rnf_tsc` ABSENT; month products and step-240 restart shards are finite and complete. | Any field census, header, payload, month product, or terminal restart fails admission. |
| HD8-P4 admission non-vacuity | Every inherited and round-7 plant binds after the call repair. | All sixteen launcher plants report `PLANT-FIRED`, including field-name first, then the clean admission passes. | Any plant stays green or crashes outside the planted failure. |
| HD8-P5 rung-4 shortwave boundary | Compiled source and the resolved reference namelist select exactly one namelist-only lower-rung behavior when RGB shortwave is removed. | One source-cited setting produces a one-module rung-5-to-rung-4 delta with no other assignment change. | Two defensible readings remain, or the behavior requires a CPP-key/build change; report `DECISION_NEEDED` and do not construct rung 4. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The call repair must be one argument deletion plus a direct regression test
that fails with the stale keyword.  Admission uses the existing operator runs;
it must not invoke `makenemo` or `mpirun`.  Rung-6 calibration identity is a
hard prerequisite to rung-5 admission.  The checker continues deriving payload
lengths from self-describing headers and reaching physical EOF.

Rung 4 may be staged only after rung 5 admits and the compiled shortwave
dispatcher plus `namelist_ref` establish a unique namelist-only fallback.
Otherwise this round stops for the user decision.  No NEMO run is attempted in
the sandbox.
