# Preregistration — ORCA2 round 92 rung-0 card and stage-1 boundary

Date: 2026-10-01. Base: `de1bdae43`. Every measured number in this round is
labelled **independent**: the state comes from the admitted rung-0 from-rest
NEMO record, never from the shipped ORCA2+SI3 identity.

## Frozen measurement

Before this file was committed, the admitted frame headers and the existing
ORCA2 card, ladder, hierarchy-deck, and WS-RK3 source paths were inspected. No
rung-0 legoESM card was instantiated, no frame payload was assembled into a
candidate state, and no legoESM or NEMO trajectory was run.

The round will produce:

1. One explicit rung-0 card derived from the existing ORCA2 geometry card.
   Every Decision-80 excluded module is stated off, constant mixing is stated
   at `rn_avm0=1.2e-4` and `rn_avt0=1.2e-5`, and the shipped card (including
   its sea-ice `unmeasured_features`) remains byte-for-byte unchanged.
2. A self-describing two-rank frame bridge that assembles only owned cells and
   replaces all five prognostic entry fields from the admitted stage-0 `Nbb`
   record. A header/layout/field or signed-bit mismatch refuses execution.
3. One production-JIT CPU/fp64 stage-1 replay with literal zero surface
   forcing, scored bitwise against both admitted rank-complete stage-1 frames.
4. The earliest cross-model non-bit boundary supported by the record. If the
   stage-1 aggregate differs but the record has no internal frame that can
   distinguish its first producer, the result is explicitly UNMEASURED and a
   fail-closed additions-only acquisition is written; no statement is guessed.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R92-P1 | The rung-0 specialization can be expressed without a new solver or card-scoped physics formula. | Existing ORCA2 geometry/dynamics are reused; only resolved module selectors, constant coefficients, surface metadata, and entry state differ. | Any required new numerical formula or ambiguous selector: **REFUTED** and `DECISION_NEEDED`; do not land. |
| R92-P2 | Replacing all five fields from the admitted two-rank stage-0 frames reproduces NEMO's independent entry bit-for-bit. | T, S, u, v, and ssh each have zero unequal bits over the full 148x180 owned domain. | Any unequal bit, overlap/gap, or header disagreement: **REFUTED**; repair the bridge or stop for a new record. |
| R92-P3 | The explicit rung-0 card reaches the admitted stage-1 boundary without executing a module that Decision 80 excludes. | The card census is exact, excluded-module plants fire, and the production step returns a finite stage-1 state. | An excluded module is live, a selector is inherited, or execution refuses before stage 1: **REFUTED**; name the exact unresolved selector/refusal. |
| R92-P4 | At least one stage-1 field is non-bit against NEMO. | The bitwise two-rank comparison reports a nonzero unequal count and magnitude at stage 1. | All five fields exact: **REFUTED**; continue to stage 2 in compiled order. |
| R92-P5 | The shared GYRE trajectory is observationally unchanged by the card-only addition. | Base/tip ten-step residual ladders are array-equal and 0 comparison rows differ; base/tip day-30 snapshots are byte-identical. | Any GYRE movement: **REFUTED**; do not land the package change. |

## Landing and refusal bar

The rung-0 card may land only if its selector census and planted violations
pass, stage-0 is full-domain bit-exact, the existing shipped ORCA2 card is
unchanged, and the prescribed GYRE/DINO/tank/generic-card and focused gates are
green. A first non-bit statement is named only when an admitted pair of frames
brackets it. No threshold, stabilizer, NEMO source, carried-state convention,
or sea-ice selector may change. Failed predictions remain recorded as
**REFUTED**.
