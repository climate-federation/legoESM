# Preregistration — ORCA2 round 130 rung-0 independent month

Date: 2026-10-03. Base: `59ca6ad42`. Every number in this round is
**independent**: legoESM starts from the hierarchy rung-0 card's own
climatological T/S, zero velocity, and zero sea surface, and is scored after
240 three-hour steps against NEMO's admitted rung-0 from-rest terminal
restart. No Decision-52 entry bridge is used. No configuration, forcing,
carried state, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Admitted record and protocol

Round 83 admitted the two-rank terminal restart at kt=240 under the dynamics-
core-only rung-0 deck. The run has exact-zero surface fluxes, constant vertical
mixing, no ice, and only the declared linear implicit bottom-drag gap in the
legoESM card. The scorer must re-admit the record's binary/deck provenance,
both rank shards, kt, fp64 payloads, finiteness, SHA-256 digests, and geographic
orientation before scoring any model value.

The production model runs JIT on CPU with fp64/x64/libm. It consumes one fixed
zero freshwater/surface-forcing object for all 240 steps. Each step is checked
for finite T, S, u, v, and sea surface. The terminal comparison reports, for
each field, bit equality, unequal count, RMS, maximum absolute difference, and
argmax coordinates, then ranks fields independently by RMS and maximum.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R130-P1 | The baseline completes all 240 steps with finite fp64 state. | `steps_completed=240`, no first-nonfinite row. | Stop at and report the first non-finite step, field, and index; no terminal ranking. |
| R130-P2 | The admitted round-83 terminal record is unchanged and complete. | Both rank shards match their pinned SHA-256, carry kt=240 fp64 finite payloads, cover 148x180 once, and match card latitude/longitude bits. | Refuse the score as `STOPPED_FOR_RECORD`. |
| R130-P3 | All five terminal fields are non-bit; temperature ranks first by both RMS and maximum. Predicted RMS order: T, ssh, S, u, v. Predicted maximum order: T, S, ssh, u, v. | The measured orders equal the two frozen lists. | Mark each disagreeing prediction **REFUTED**, retain the measured complete ranking, and do not change the metric. |
| R130-P4 | The scorer binds on terminal payload integrity and finiteness. | One-ULP digest and terminal-nonfinite plants each refuse the otherwise admitted report. | The scorer is invalid; fix it before citing any month number. |
| R130-P5 | Measurement-only code moves no certified trajectory or model file. | No `packages/` diff; the rung-0/rung-7/GYRE/DINO/tank baselines remain the round-129 final-tip values by construction. | Any model diff or configuration delta ends the round without a scientific claim. |

## Round bar

This round lands the scorer and baseline evidence, not the held round-129
physics chain. A valid result requires the committed scorer, direct tests, two
firing plants, record re-admission, a complete five-field ranking, citation
gate with a firing plant, focused tests, and the prescribed ocean-fidelity
battery. The next source walk may use the month scorer only after all of these
bind.
