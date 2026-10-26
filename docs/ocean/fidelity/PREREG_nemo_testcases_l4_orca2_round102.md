# Preregistration — ORCA2 round 102 merged-tree GYRE admission

Date: 2026-10-02. Base: `87279760b351e7a95284b7f47631ae68a955d3a1`.
This round applies the standing Decisions 43/45/55/59 gate to the shared
stage-one tracer-thickness-ratio statement already named in round 101. It
does not change model physics, a card, a deck, carried state, a threshold, a
stabilizer, sea ice, or the ORCA2 `unmeasured_features` tuple.

The controlled before member is the certified seed-0 GYRE member tagged
`r6carried`. The candidate is a fresh CPU/fp64 seed-0 360-day member made by
the current merged tip, with daily snapshots. NEMO is the existing seed-0
from-rest record. The existing year harness and year-owner scorer define the
state, masks, staggering, and RMS; this round adds no alternative metric.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R102-P1 | The fresh current-tip year reproduces round 100's merged member bit for bit. | All 360 daily snapshots have identical SHA-256 digests and the three registered T RMS values equal `2.3432465132112266e-06`, `6.58170609494473e-05`, `5.4077419367442036e-05` K. | Any digest or registered value differs: reconcile the run before applying the landing gate. |
| R102-P2 | The already-landed NEMO-cited statement passes Decisions 43/59/AW. | Relative to certified T RMS, day 30 and day 240 move toward NEMO; day 360 moves away by less than `2.0e-9 K` (ten `2.0e-10 K` floor units); all registered year rows are present. | A registered row moves away by at least `2.0e-9 K`, or any required row is absent: stop for Decision 85 with measured values. |
| R102-P3 | The GYRE ten-step gate remains unchanged. | The current 70-row artifact is array-identical to the certified artifact, no kt=1 AT-BAR row leaves, and first-over-bar is not earlier. | Any row moves outside its allowance: merge remains HELD. |
| R102-P4 | Round 100's rung-7 refusal is a signed-zero input-contract defect, not a numerical entry mismatch. | A gate-only bit-canonicalization justified by the NEMO record admits exactly the same numerical entry and the trajectory gate runs without relaxing later bit predicates. | Any nonzero value changes, or the trajectory exposes an unregistered moved row: keep HELD. |
| R102-P5 | The admitted rung-0 frames support the missing kt=1..10 gate without a new oracle record. | A committed gate scores all available entry/stage rows through kt=10 and carries the existing first non-bit statement forward. | Any required frame is absent: report its exact stream and request acquisition. |

## Landing bar

The merge lands only if R102-P1 through P3 pass mechanically. The receipt
registers all eight certified year rows (days 30, 60, 90, 120, 180, 240,
300, 360), states direction versus NEMO, records the admitted snapshot
digests, and quotes the GYRE numbers the source lane should adopt. The gate
must include a planted year regression that fires. Rung-7 and rung-0 work is
then attempted in order; failure there is reported explicitly and no physics
change is made to hide it.
