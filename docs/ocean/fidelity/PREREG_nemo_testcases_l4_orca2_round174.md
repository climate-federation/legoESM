# Preregistration — ORCA2 round 174 passive stage-growth boundary

Date: 2026-10-08. Frozen base: `47a0978d0`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round174/`.

Every number in this round is **independent**: hierarchy rung 0 starts from
its own climatological T/S, zero velocity and zero sea surface. No rung-7
given-entry number is mixed into the result. Sea ice, all six sea-ice
selectors and the shipped card's `unmeasured_features` tuple remain unchanged.

## Frozen record and order

The oracle is the admitted round-90 rung-0 frame record: 80 self-describing
rank shards covering entry plus stages 1--3 for kt=1..10. The compiled program
calls the external mode before RK3, then stages 1, 2 and 3 in that order and
writes each completed stage immediately after its call
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:200-233`). The gate must
re-admit all shards with exactly-once rank coverage before scoring.

The candidate is the unchanged complete private arm from rounds 163--173:
raw NEMO reference face depth, no extra compact V mask, the seven-array
external-mode association, and materialised V transport. Separate existing
stage-exposure models publish completed stages 1 and 2; they never feed the
ordinary model whose returned stage 3 advances the trajectory. No callback,
observer, new returned component or executable hook is added.

For each kt=1..8 and completed stage 1..3, the gate reports max absolute
candidate-minus-NEMO error for T, S, u and v. Boundaries are ordered by
`(kt, stage)`; within a boundary fields are ordered T, S, u, v. The growth
ratio is `current_max / max(previous_max, 2e-10)`, where `current_max` is the
largest of the four field maxima. The first ratio strictly greater than 10
owns the next source walk. The kt=1 entry is the preceding exact boundary but
is not itself a completed-stage row.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R174-P1 | The existing frame record remains admissible without reinterpretation. | Exactly 80 self-describing shards parse, cover both ranks exactly once at every boundary, and retain the registered five fields. | Any header, registry, placement, coverage or census check fails. |
| R174-P2 | The completed-stage instrument is passive. | The ordinary complete arm reproduces the content-pinned round-166 kt=1..7 state digests exactly, stage calls never advance the carried state, and the kt=8 refusal remains the registered raw-mesh `e3w_int` refusal. | Any digest moves, a stage output feeds the carried state, or the ordinary arm fails earlier/differently. |
| R174-P3 | The first >10x growth boundary is kt=1 stage 1. | Its four-field maximum divided by the exact-entry floor exceeds 10 and no earlier completed-stage boundary exists. | kt=1 stage 1 stays at or below 10x, or an instrument/admission prerequisite fails. |
| R174-P4 | The table localises growth upstream of kt=8 HPG. | All available kt=1..8 completed-stage field maxima are emitted in order; a non-finite/refused kt=8 stage 3 is recorded as a terminal boundary, not silently dropped. | The table skips an available boundary, changes field order, or reports an HPG operand claim before selecting the growth boundary. |
| R174-P5 | This is measurement-only. | No `packages/`, card, configuration, carried state, stabiliser, threshold, sea-ice selector or halo-unit production path changes. | Any such change lands. |

Failed predictions remain in the receipt. A zero-to-nonzero transition is
measured against the frozen 2e-10 floor; it is never divided by zero. A later
larger explosion cannot displace an earlier qualifying boundary.

## Controls and terminal rule

The classifier must reject planted admission, digest-passivity, boundary
order and growth-selection changes. A one-ULP field plant must move its row.
All candidate, oracle and geometry arrays print float64; the backend is CPU,
JAX uses production JIT, and the policy is fp64/libm.

After the table selects a boundary, stop before any in-executable observer.
The next round replays source-ordered pure operators offline from that passive
completed boundary against an existing NEMO operand record, or requests a
self-describing rank-complete record if one is missing. No statement below a
non-bit input is attributed by inference.

ASKED choices: continue the independent rung-0 walk using passive completed
stage states, per B57 addendum 3.  
UNASKED choices: empty.

## Instrument correction after the first refused run

The first committed run refused at kt=1 before emitting a scientific row. Its
local digest helper serialized the dtype as `dtype.str` and the shape with
`repr`, while the pinned round-166 producer serialized them with `str`. The
arrays were never compared because those different prefixes necessarily give
different hashes. This is an instrument-format defect, not trajectory drift.
The corrected gate reuses round 166's exact digest helper rather than
restating it. Predictions, thresholds, boundary order and terminal rule are
unchanged.

The second committed run passed kt=1..7 passivity and reached kt=8, then the
classifier refused because completed stages 1 and 2 contain non-finite T. That
is the boundary the requested table must expose, not an instrument failure:
the oracle remains finite and the detached stage calls did not feed the
ordinary trajectory. The corrected classifier reports each candidate
non-finite count, retains the maximum over finite pairs, and assigns the
boundary maximum/incoming growth ratio `Infinity`. It still refuses any
non-finite oracle value. The first-growth selector is upstream at kt=1 stage 1,
so this correction cannot change the owning boundary.

The third committed run reached the same kt=8 stage-1/2 non-finite rows but
the classifier refused before writing them because IEEE `Infinity / Infinity`
produced `NaN`, which is not equal to itself during the derived-table check.
The corrected selector defines consecutive non-finite maxima as ratio 1.0;
finite-to-non-finite remains `Infinity` and non-finite-to-finite is 0.0. This
only makes the frozen selector total over its registered domain and cannot
move the already-selected kt=1 stage-1 boundary.
