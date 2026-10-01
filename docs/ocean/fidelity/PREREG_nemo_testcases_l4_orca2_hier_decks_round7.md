# ORCA2 hierarchy decks round 7 preregistration — recorder ABSENT repair

Date: 2026-10-01

Base: `769526b4c` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side instrumentation only.  The operator's rung-5 run stopped
before its first time step with SIGSEGV after runoff was disabled.  This round
source-resolves that failure, ports the main lane's round-87 explicit ABSENT
record convention to the hierarchy's round-69 recorder, proves the repaired
writer is passive on admitted rung 6, and hands off a fresh rung-5 acquisition.
It changes no legoESM package, card, recipe, physics, configuration, hierarchy
switch, threshold, or carried state.  Rung 4 and below remain out of scope.

All run claims are labelled **independent**: NEMO starts from the rung's own
from-rest T/S initialization.

## Frozen oracle reading

The compiled recorder writes ten named fields.  Eight are allocated
unconditionally by the surface core: `utau`, `vtau`, `emp`, and `fr_i` in
`sbc_oce.f90:195-205`; `qns`, `qsr`, `sfx`, and `taum` in
`sbc_oce.f90:210-216`.  Only `rnf` and `rnf_tsc` are module-gated: the first is
allocated only under `ln_rnf` in `sbc_oce.f90:201`, and the second only inside
the runoff allocator in `sbcrnf.f90:146-152`.  The inherited recorder
dereferences both unconditionally in `l4_r69_surface.f90:85-87`.

This census also freezes the boundary of the repair.  The current recorder has
no chlorophyll, eddy-induced-velocity, mixed-layer-eddy, bottom-boundary-layer,
or geothermal field.  Turning those later modules off cannot dereference such
fields in this ten-field schema.  `qsr` is a core surface-flux array allocated
independently of the shortwave-penetration switch; it remains PRESENT.  Any
future schema extension must declare an owner switch per new field before it is
recorded.

The main lane's round-87 repair defines the reusable convention: each record
remains self-describing; a disabled owner produces the normal field name plus
the header `(rank,n1,n2,n3)=(0,0,0,0)` and no payload; a checker accepts ABSENT
only when the resolved namelist and `ocean.output` both report that owner off.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD7-P1 crash owner | The rung-5 SIGSEGV is the recorder's first unconditional dereference of an unallocated runoff field. | An additions-only patch guarding only `rnf` and `rnf_tsc` permits the rung-5 record to start; the source census shows no other conditional field in the schema. | The repaired run still fails before step 1, or another current-schema field is unallocated. |
| HD7-P2 active-runoff passivity | When `ln_rnf=true`, the repaired writer emits exactly the legacy version-1 bytes. | A fresh repaired-build rung-6 calibration yields 480/480 frame files byte-identical to the admitted rung-6 record and bit-identical terminal restart payloads. | Any frame byte or terminal payload differs. |
| HD7-P3 inactive-runoff schema | When `ln_rnf=false`, each frame contains eight PRESENT finite fields and two ABSENT runoff fields, with no plausible zero replacement. | All 480 rung-5 frames parse through physical EOF; each has exactly `rnf` and `rnf_tsc` ABSENT; the resolved owner is false in both deck and output. | Any different census, payload for an off-owner field, malformed header, trailing byte, or owner disagreement. |
| HD7-P4 checker non-vacuity | The checker binds to names, payload lengths, owner state, finiteness, and EOF. | Plants for a bad name, truncation, non-finite present payload, ABSENT-as-zero, and owner-on all fire. | Any plant stays green. |
| HD7-P5 rung-5 completeness | The repaired run completes the unchanged rung-5 protocol. | 480 frames, finite month products, finite fp64 step-240 ocean restart shards, resolved runoff-off consequences, and complete SHA-256 inventory all admit. | Any product is absent, malformed, non-finite, or unpinned. |
| HD7-P6 acquisition disposition | Neither the repaired rung-6 calibration nor repaired rung-5 record exists before handoff. | Preflight and synthetic plants pass/fire; round ends `STOPPED_FOR_RECORD` with one committed launcher that runs calibration first. | An admissible repaired record already exists. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The patch must be additions-only and apply at fuzz zero to the source file used
by the record build.  The checker derives every payload length from the record's
own header and reaches physical EOF; it does not predict byte counts.

The operator launcher must use fresh build and run targets, first run the exact
admitted rung-6 deck, refuse unless all 480 frames reproduce byte-for-byte and
terminal restart payloads reproduce bit-for-bit, and only then run rung 5.  It
must retain the rung-5 one-module deck delta and every round-6 admission plant.
No NEMO run is attempted in the sandbox.
