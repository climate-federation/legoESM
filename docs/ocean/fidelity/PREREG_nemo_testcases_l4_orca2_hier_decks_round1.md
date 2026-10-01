# ORCA2 hierarchy decks round 1 preregistration — rung 10 identity

Date: 2026-10-01

Base: `5c47d7340` (`fidelity/nemo-testcases-orca2-decks`).

Scope: NEMO-side only.  This round stages and gates hierarchy rung 10, the
unchanged one-category ORCA2 ocean-ice deck recorded by round 69.  It changes
no legoESM package, card, recipe, selector, physics, configuration, threshold,
or carried state.  Rungs 1 through 9 and main-lane rung 0 are out of scope.

## Frozen source

The source record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/acquisition/orca1ice_surface_only_240step_np2`.
The rung uses its `namelist_cfg` and `namelist_ice_cfg` byte for byte, its
input manifest, and its already-built instrumented binary.  The build's CPP
keys remain `key_si3 key_qco key_vco_1d3d key_RK3`; no rebuild is authorized.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD1-P1 deck identity | Rung 10 is the source deck, with zero differing namelist lines. | Both committed namelist SHA-256 values equal the round-69 values and the parsed assignment maps are identical. | Any byte, assignment, or file-inventory difference. |
| HD1-P2 producer identity | The reused binary and compiled writer are exactly the admitted round-69 producer. | Binary, CPP-key file, compiled `stprk3`, and compiled writer SHA-256 values equal their frozen pins. | Any digest differs, any rebuild occurs, or the writer call is absent. |
| HD1-P3 operand identity | Given identical deck, inputs, binary, two-rank layout, and from-rest protocol, the rung-10 surface operand records are raw-bit identical to round 69. | All ten named payloads on both ranks for every step 1 through 10 compare equal as `uint64`, including signed zero; the self-describing parser reaches physical EOF. | Any field, rank, step, name, shape, precision, or payload differs. |
| HD1-P4 month identity | The 240-step rung-10 run reproduces the admitted round-69 month. | Both ocean restart shards and both ice restart shards are NetCDF-payload identical except permitted timestamp metadata; required terminal ocean fields are finite fp64 at step 240. | Missing shard, non-finite payload, wrong step/dtype, or any non-timestamp payload difference. |
| HD1-P5 resolved deck | The run resolves the shipped one-category SI3 deck from rest and retains `ln_spc_dyn=.true.`. | `ocean.output` prints step 240, restart cadence 240, from-rest provenance, `jpl=1`, SI3 selection, and the special dynamics switch true. | Any resolved value differs or the ice namelist is not read. |
| HD1-P6 output inventory | The record contains the month products consumed by the existing scorer plus a complete per-step finite surface stream. | Required T/U/V/W monthly files, terminal restarts, and 480 finite self-describing rank-step frames exist; SHA256SUMS covers every regular file selected by the manifest. | A required product is absent, an unexpected oracle stream appears, a frame is non-finite, or a ledger target is missing. |
| HD1-P7 disposition | No real hierarchy record exists before the operator run. | Preflight and every synthetic violation pass/fire; round ends `STOPPED_FOR_RECORD` with one acquisition path. | An already-existing admissible rung-10 record is found. |

Predictions are frozen before the hierarchy gate reads or compares a newly
produced rung record.  Failed predictions remain in the receipt as REFUTED.

## Controls

The gate must refuse at least: a changed deck byte, a changed manifest field,
a malformed or truncated self-describing frame, a missing rank-step frame, a
one-ULP operand change, a one-ULP terminal restart change, a non-finite terminal
field, a wrong resolved switch, and an incomplete SHA-256 inventory.  A control
that does not fire blocks admission.

## Landing predicate

This is an acquisition-only round.  It lands the committed deck, manifest,
gate, launcher, tests, and receipt only if preflight and all plants pass.  The
record remains **UNMEASURED** until the operator runs the launcher; no model
trajectory gate is applicable because no model file changes.
