# ORCA2 round 204 preregistration — admit OMT-0 frames and score the card

Date: 2026-10-09. Frozen base: `ba3932304`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round204/`.
All trajectory results are labelled **independent OMT-0**. The shipped rung-10
ORCA2 card, sea ice, its six selectors and `unmeasured_features` stay unchanged.

## Prior evidence and frozen diagnosis

The operator completed the round-203 uninstrumented calibration and both
instrumented twins through `STOP 0`. Clean admission then refused
`orca2_omt0_frames_10step_a_np2: binary changed`. Every record plant except the
cadence and changed-binary plants refused at that same line, so those controls
did not exercise their intended faults.

The gate first validates the instrumented binary against its pinned
`b54b3778...` digest, then calls round 199's `_run_provenance`, whose compiled
default hard-pins the uninstrumented `c4907e47...` digest. Thus the second check
rejects the already-validated instrumented executable. This is a checker defect,
not a NEMO result. The repair may parameterise only that provenance digest; its
default must remain the round-199 uninstrumented digest.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R204-P1 | The refusal is solely the inherited hard-coded provenance digest. | With the helper passed the already pinned instrument digest, admission advances beyond binary validation; every older round-199 caller behaves identically under the unchanged default. | Any other prerequisite fails before frame parsing: **REFUTED**; name it and stop. |
| R204-P2 | Both completed twins contain the rank/step/stage-complete record. | Exactly 80 frames per twin cover ranks 0/1, kt 1..10 and stages 0..3; every self-describing header and all five finite fp64 payloads parse. | Missing/extra/malformed/non-finite frame: **REFUTED**; no trajectory score. |
| R204-P3 | The writer is passive and deterministic. | All 400 twin field comparisons are array-equal and both instrumented terminal restart ranks are byte-identical to the uninstrumented calibration (four comparisons). | Any bit differs: **REFUTED**; instrument remains inadmissible. |
| R204-P4 | Each admission plant reaches its named predicate. | Cadence, header, field-name, truncation, nonfinite, missing-frame, twin-ULP, terminal-byte and changed-binary logs contain distinct matching refusal text; no plant is accepted merely by an earlier binary check. | Repeated unrelated refusal or green plant: instrument invalid. |
| R204-P5 | The explicit OMT-0 card differs from independent rung 0 only by Decision 103's five OFF modules. | Resolved config diff names momentum advection, momentum lateral diffusion, tracer advection, tracer lateral diffusion and bottom drag only; geometry/entry are identical. | Any sixth physics/state/config delta: `DECISION_NEEDED`; do not score. |
| R204-P6 | OMT-0 executes kt=1..10 but is not bit-exact. | Both independent and given-entry ladders reach kt=10; their first non-bit retained-core boundary is external-stage SSH at kt=1 stage 1 or an earlier source-ordered retained statement named from the record. | Earlier refusal means the card is incomplete; exact ladder refutes the expected retained-core debt but is accepted as the stronger result. |
| R204-P7 | OMT-0 shares NEMO's registered month boundary. | Candidate remains finite through kt=10 and first refuses/stops no earlier than NEMO's registered kt=11 velocity stop; kt=10 per-field rms/max are reported. | Earlier candidate stop is debt and the first boundary owns the next walk; no stabiliser. |
| R204-P8 | No shipped-card or shared behavior changes. | No `packages/` diff unless the already user-authorised TSUNAMI OFF arms must be imported; if imported, the full shared-statement gates run and GYRE stays within its standing gate. | Unauthorised default/card/sea-ice change: remove it. |

## Landing predicate

The record may be admitted only after R204-P1..P4 pass. The OMT-0 card may land
only after R204-P5's exact five-module diff, both ten-step ladders, the kt=11
boundary check, the ordinary ORCA2/GYRE/DINO/tank gates for any `packages/`
change, citation gate and non-vacuous tests pass. OMT-1 does not begin first.

ASKED choices: Decision 103's OMT-0 deck/card and exact record protocol.
UNASKED choices: empty.
