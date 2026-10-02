# Preregistration — ORCA2 round 97 SPG record admission and walk

Date: 2026-10-02. Base: `ae79a83f7`. Scope is ocean only. Every SPG
measurement is labelled **independent** because rung 0 starts from NEMO's own
from-rest state.

## Prior evidence and frozen diagnosis

The round-96 target exists and its run reached `STOP 0`. It contains two
rank-tagged SPG records, twenty kt=1..10 restart shards, a producer commit
stamp, and a toolchain manifest. Admission stopped only because the launcher
required the stamped commit object to exist in the operator checkout. That is
the prohibited moving-commit pin: the producer content, not repository object
availability, determines whether the record is admissible.

## Frozen repair and measurements

1. Replace the commit-object lookup with fail-closed verification of the
   recorded SHA-256 manifest for the Decision-83 deck patch, SPG source patch,
   writer, checker, round-96 preregistration, and pinned pre-patch compiled
   source. Keep the producer token as provenance and in each record stamp, but
   do not require that token to resolve as a local Git object.
2. Keep the existing harmonized namelist, deck/input manifests, built binary,
   compiled writer layout, record payload hashes, and restart comparison
   checks unchanged.
3. Run `--admit-existing`; all seven plants must fire and all twenty terminal
   restarts must be byte-identical to the round-93 baseline.
4. Parse both self-describing SPG records and walk the independent rung-0
   split-explicit statements in compiled source order. Name the first active
   non-bit boundary. If the walk reaches `dyn_cor_2D`, test the recorded
   Coriolis coefficient against the four-cell masked `e3f_0vor` denominator as
   the preregistered one-variable arm.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R97-P1 | The existing round-96 run is complete; only its commit-object admission pin failed. | `STOP 0`, two records, twenty restarts, and all content manifests exist; content-based admission passes. | Missing or hash-moved content, incomplete payload, failed plant, or moved restart: **REFUTED**; withhold the record and write a fresh-target acquisition. |
| R97-P2 | The instrument is observational. | All twenty target restart shards are byte-identical to round 93. | Any byte moves: **REFUTED**; reject every SPG number. |
| R97-P3 | The first active SPG non-bit boundary is no later than the first recorded after-SSH boundary. | Entry rows before it are bit-exact and the named boundary has at least one active unequal cell on either rank. | All rows through after-SSH are bit-exact: **REFUTED**; continue in compiled order without changing the record. |
| R97-P4 | If reached, the ORCA2 two-dimensional Coriolis coefficient uses NEMO's four-cell masked `e3f_0vor` and not plain T-cell thickness at land-adjacent faces. | Recorded coefficient is bit-exact under the four-cell denominator and non-bit under the plain-thickness control. | Four-cell form is non-bit or the control cannot fire: **REFUTED**; name the observed boundary without landing the arm. |
| R97-P5 | Decision-83's cited-inert harmonization leaves rung-0 numerics unchanged. | All twenty terminal restarts are byte-identical to round 93. | Any restart differs: **REFUTED**; stop with `DECISION_NEEDED` and the first differing shard. |

## Landing and refusal bar

The provenance repair changes no model file or scientific configuration. A
physics landing is allowed only for one compiled, measured statement under the
ORCA2 ladder and all shared-card gates. The held round-94 slow-depth statement
is not mixed into this round. No stabilizer, sea-ice change, carried-state
change, selector change, threshold change, or `unmeasured_features` change is
permitted.
