# ORCA2 round 103 — rung-0 ten-step ladder gate

**Date:** 2026-10-02  
**Base:** `04396546bc37aea5346fb621509a7b8a0f00d793`  
**Preregistration:** `d2412d9aa`  
**Measurement commit:** `2e98c69a46ba1cd77ec0826dae9e66206b853331`  
**Disposition:** **LANDED — rung-0 kt=1..10 gate; rung-7 replay remains OPEN**

## Claim boundary

Every rung-0 number below is **independent**: the replay starts from NEMO
rung 0's own from-rest entry. The separate rung-7 attempt is **given NEMO's
recorded entry** under Decision 52. No table mixes these claim classes.

This round adds a measurement gate and tests only. It changes no `packages/`
file, production card, deck, carried state, threshold, stabilizer, sea-ice
selector, or ORCA2 `unmeasured_features` tuple. The gate-local rung-0 card
continues to declare `linear_implicit_bottom_drag` unbuilt; no substitute
drag or stabilizer was added.

## Rung-0 record and gate — independent

The existing round-90 record admits exactly 80 self-describing shards: two
ranks, ten steps, and entry plus three stage boundaries. The compiled writer
records the `Nbb` entry before step work at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:92-108` and writes stage 1
immediately after the compiled call at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:214-217`. The gate assembles
each global field with exact once-only rank coverage, requires fp64/libm,
production JIT and CPU, checks every candidate and oracle value is finite,
and emits 40 checkpoints / 200 field rows.

The kt=1 entry is bit-exact in all five fields: 0 / 3,223,440 unequal. Exactly
5/200 rows are `AT_BAR_BIT_EXACT`; they are the five kt=1 entry rows. The
remaining 195 rows are `DEBT`. The first non-bit checkpoint remains kt=1
stage 1, with T first in the field registry:

| field | unequal / compared | max abs | rms |
|---|---:|---:|---:|
| T | 582,469 / 799,200 | `0.0013726778718971544 K` | `1.1801028691523244e-05 K` |
| S | 430,551 / 799,200 | `0.0011734531121732061 PSU` | `1.0414642920662687e-05 PSU` |
| u | 443,423 / 799,200 | `0.06171520235435013 m/s` | `0.0008585437906091959 m/s` |
| v | 440,386 / 799,200 | `0.03402112470518942 m/s` | `0.0008932210107169522 m/s` |
| ssh | 16,433 / 26,640 | `0.13145859582012723 m` | `0.006243384623854269 m` |

Field order is not source order. The gate therefore carries forward the
already measured first source statement rather than attributing the row to T:
NEMO's vector-invariant vertical average of the completed 3-D RHS at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219`. Round 94's
source-associated arm made that boundary bit-exact but remains held because
49/70 certified GYRE rows violated the immutable 2-ULP gate. This round does
not land or rescore that physics arm.

At kt=10, the independent end-of-step rows are:

| field | unequal / compared | max abs | rms |
|---|---:|---:|---:|
| T | 430,552 / 799,200 | `0.8929704006615697 K` | `0.006317420525208628 K` |
| S | 430,552 / 799,200 | `0.4156326670091559 PSU` | `0.0016155151271706628 PSU` |
| u | 444,108 / 799,200 | `0.6887186133134653 m/s` | `0.004518520574918117 m/s` |
| v | 450,657 / 799,200 | `1.555861413441975 m/s` | `0.005041743314861328 m/s` |
| ssh | 16,433 / 26,640 | `0.42915183233564513 m` | `0.028059540421323897 m` |

The complete machine-readable result is `round103/rung0_ladder.json`, SHA-256
`683f70e6cc4cc9ca6abdc286904a2781fde68e5429ef3d3c1d31c8e7f44540c9`.
The one-bit kt=1 entry plant exits 2 at `STATUS PLANT-FIRED`; R103-P1, P2, P3,
and P4 are **CONFIRMED**.

## Retained instrument refusal

The first implementation requested the broad live-operand trace. It refused
before step 1 with `live WS-RK3 operand trace is incomplete` because that
instrument requires TKE internals and rung 0 intentionally selects constant
mixing. No numerical row came from that run. The final gate reuses round 92's
passive combined momentum/tracer stage exposure for stages 1 and 2 and the
ordinary returned state for stage 3. The failed development log was
overwritten by the required final committed-tree rerun, so it is not offered
as a durable evidence artifact. The broad trace is retracted as a rung-0-
capable instrument and is not interpreted as model behavior.

## Rung-7 replay — given NEMO's entry

The correct admitted record is the round-5
`orca1ice_surface_entry_every_step_a_np2` root, not the incomplete pinned
phase-2x root used by round 102's aborted command. Its admission is PASS:
20/20 surface frames, 20/20 full-rank entry frames, 30/30 manifest rows
hash-verified, and no missing stream. Therefore no acquisition is needed.

Two clean-tree full replay attempts, followed by a two-step discriminator,
were terminated externally before Python emitted JSON and before the shell
wrote its exit marker. The zero-byte logs are retained. This reproduces round
102's full-replay cutoff but is not a scientific refusal and provides no
ten-step row. R103-P5 is **UNMEASURED**, not refuted.

## Validation and review

- Final committed-tree rung-0 gate: PASS, 40 checkpoints / 200 rows, exit 0.
- Final one-bit real-record plant: `STATUS PLANT-FIRED`, exit 2.
- Focused frame/card/ladder tests: 30 passed.
- No `packages/` diff exists, so the shared GYRE trajectory and year cannot
  move by construction; no GYRE result is re-pinned in this round.
- Separate read-only Codex review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

## OPEN

1. Split the shipped rung-7 replay into bounded clean processes without
   changing its state, forcing, precision, checkpoint definitions, or row
   predicates; then emit the missing 200-row given-entry table. The record is
   complete, so this is not an acquisition item.
2. Resume the rung-0 barotropic source-order walk from round 99: coefficient
   zero signs, accumulation/final scale, the independent fold-row
   `ffv_nw`/`ffv_ne` debt, and the 68-cell substep-2 U residual.
3. The package-exposed rung-0 card and independent 240-step month remain
   unmeasured; neither is implied by this gate-only landing.

## UNVERIFIED

- Rung-7 kt=1..10 rows on the merged tree.
- Rung-0 independent month and a production-selectable rung-0 card.
- No NEMO acquisition was run or requested.

## Choices

ASKED: Decision 52's rung-7 entry bridge and Decision 80's rung-0 hierarchy
remain unchanged. UNASKED: none.
