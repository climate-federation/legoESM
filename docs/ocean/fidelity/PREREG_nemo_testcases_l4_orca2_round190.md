# ORCA2 round 190 preregistration — independent month growth table

Date: 2026-10-08. Base: `06bacd0f456a822efaeced3a13463d4d33e16d8a`.
Every scientific number in this round is **independent**: the rung-0 card
starts from its corrected initial state, not NEMO's recorded entry. Given-entry
rung-7 results are not part of this measurement. Sea ice and the shipped card
remain untouched.

## Frozen scope and statistic

First admit the already-produced round-189 twin record without rebuilding or
rerunning NEMO. The scientific table uses the admitted NEMO restarts at steps
10, 20, ..., 90 and 95 and the matching independent legoESM rung-0 states.
For each step and each field T, S, u, v, and SSH it reports whole active-domain
RMS error and maximum absolute error with its global index. The interval growth
factor is the current step's largest fieldwise maximum divided by the previous
step's largest maximum; the exact entry supplies the fixed `2e-10` floor for
the first interval. The first strict factor greater than ten owns the coarse
walk. No weighting, threshold, cadence, field set, or comparison source may be
changed after measurement.

No model file, physical configuration, forcing, carried state, stabiliser,
selector, sea-ice selector, or `unmeasured_features` entry changes. No
in-executable observer is authorised. Any stage/operator refinement must use
completed passive states and offline replay only.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R190-P1 | The round-189 bounded-list record is admissible and completes the immutable round-186 prefix. | Both twins and both ranks contain finite, correctly stamped steps 95 and 96; step 95 is twin-bit-exact; step 10 recalibrates bit-exactly to round 83; all record plants fire. | Any missing, truncated, misstamped, non-finite, moved, or uncalibrated payload: **REFUTED** and stop for a new record; do not score it. |
| R190-P2 | Step 10 remains the first coarse interval with strict greater-than-ten growth. | The step-10 largest maximum divided by `2e-10` exceeds ten; no earlier sampled restart exists after the exact entry; later rows cannot change the first interval. | Step 10 is at or below ten times the floor: **REFUTED**; report the measured first interval without changing the rule. |
| R190-P3 | Step 95 is finite but explosive, and its addition changes magnitudes without changing the step-10 first-boundary verdict. | Every scored field is finite at step 95 and at least one step-95 maximum exceeds the step-90 maximum by more than ten. | Any non-finite step-95 value or growth factor at most ten: **REFUTED**; retain the observed row and boundary. |
| R190-P4 | The completed-RHS vertical average remains the first source-ordered non-bit statement at kt=1 stage 1. | The existing certified passive rung-0 ladder and source-associated offline replay reproduce the round-187 boundary; the restart table alone creates no new statement claim. | A passive replay disagrees: **REFUTED**; reconcile the instruments before citing either statement. |
| R190-P5 | The step-96 refusal is downstream of measurable finite growth rather than the first debt. | Step 95 is finite and the previously certified kt=1 stage-1 boundary remains first. | Step 95 is already non-finite or kt=1 changes: **REFUTED**; report the earlier boundary. |
| R190-P6 | All measurement controls are non-vacuous. | Missing-rank, twin-ULP, calibration, hidden-deck, missing/truncated/misstamped sentinel, and a one-ULP scored-state plant each make their owning gate refuse. | Any plant stays green: instrument invalid; quote no scientific result. |

The round lands no physics unless a cited source statement is implemented and
passes Decision 96 plus the full ORCA2, GYRE, DINO, and tank gates. Otherwise
it ends **HELD** with the completed growth table and the first unresolved
source statement in OPEN.
