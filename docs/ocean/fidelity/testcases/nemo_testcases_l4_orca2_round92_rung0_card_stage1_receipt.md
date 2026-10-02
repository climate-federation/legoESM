# ORCA2 round 92 — rung-0 card and stage-1 boundary

**Verdict: STOPPED_FOR_RECORD.** A gate-local rung-0 measurement card exactly
reproduces all 3,223,440 admitted independent entry values, and the compiled
stage-1 boundary is non-bit in all five fields. The admitted record has no
rank-complete internal boundary between them, so no first non-bit statement is
claimed. A new additions-only acquisition is preflighted to record the first
five compiled momentum accumulators on both ranks. No `packages/` file, NEMO
source tree, configuration selector, threshold, stabilizer, carried-state
convention, or sea-ice selector changed.

Base: `de1bdae43`. Preregistration:
`PREREG_nemo_testcases_l4_orca2_round92.md` at `92053a781`. Every number below
is labelled **independent**: it starts from NEMO rung 0's own from-rest entry,
not the shipped ORCA2+SI3 entry.

## Explicit measurement card

The card is deliberately local to the gate and is not exposed by the package
dispatcher. It derives the shipped ORCA2 geometry, masks, EOS, horizontal
dynamics, partial cells, fold, and WS-RK3 program, then states the rung-0
hierarchy selectors explicitly:

| Rung-0 family | Resolved measurement value |
|---|---|
| surface | exact-zero `nemo_flx_zero`; no restoring or freshwater budget |
| vertical closure | constant; `A_v=1.2e-4`, `K_v=1.2e-5`; no latitude scaling |
| retained convection | enhanced diffusion, matching shipped `ln_zdfevd` |
| excluded closures | TKE off, IWM off, DDM off |
| excluded interior modules | GM/EIV off, MLE off, BBL off, runoff spreading off |
| excluded forcing | shortwave off, runoff off, ice off |
| exact-input debt | `linear_implicit_bottom_drag` only |

NEMO initializes the constant background arrays and W masks at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-228`, selects constant
closure and leaves wave closures behind their switches at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90:262-284`. NEMO's resolved
linear drag constructs its coefficient and calls the linear branch at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfdrg.f90:540-547`. legoESM has an
explicit linear representation but no exact `ln_drgimp` composition, so the
card declares that single gap and its execution validator refuses. The replay
is a labelled measurement arm, not an execution-ready card.

The shipped ORCA2 card remains unchanged, including its complete seven-item
`unmeasured_features` tuple ending in `si3_jpl5_layered_prather_state`. Thus
the out-of-scope sea-ice selectors were neither interpreted nor modified.

## Independent entry result

The two self-described rank slabs cover the 148x180 global domain exactly once.
The bridge replaces T, S, u, v, and ssh from the admitted stage-0 `Nbb` frames.
NEMO writes that entry before surface, vertical-physics, barotropic, or stage
work (`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:92-108`). Its
from-rest initialization also constructs both barotropic velocity carries from
the exactly zero three-dimensional velocity and copies them to `Kmm`
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/istate.f90:143-162`).

| Field | Compared values | Unequal bits | max abs | rms |
|---|---:|---:|---:|---:|
| T | 799,200 | 0 | 0 | 0 |
| S | 799,200 | 0 | 0 | 0 |
| u | 799,200 | 0 | 0 | 0 |
| v | 799,200 | 0 | 0 | 0 |
| ssh | 26,640 | 0 | 0 | 0 |

R92-P2 is **CONFIRMED**. The entry-bit and rank-layout plants both fire.

## First measured non-bit boundary

The production-JIT CPU/fp64/libm replay uses exact-zero surface inputs and the
gate-local selector census above. NEMO performs surface, EOS/buoyancy,
vertical-physics, lateral-physics, external-mode, and stage-1 work in the order
shown by
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:147-217`. Its admitted
stage-1 frame is written immediately after the complete call
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:214-217`).

| Field | Unequal / compared | max abs | rms |
|---|---:|---:|---:|
| ssh | 16,433 / 26,640 | 0.122843845486 m | 0.006536001786 m |
| u | 444,098 / 799,200 | 0.064217063155 m/s | 0.001241043651 m/s |
| v | 440,917 / 799,200 | 0.032637398336 m/s | 0.001193084885 m/s |
| T | 582,469 / 799,200 | 0.001454189104 K | 1.39635309e-5 K |
| S | 430,552 / 799,200 | 0.001148259455 PSU | 1.01061884e-5 PSU |

This confirms R92-P4, but it does **not** identify a statement. The record
brackets a complete stage; its inherited internal streams are rank-0 legacy
records and were not admitted for this hierarchy claim. The first non-bit
statement therefore remains `UNMEASURED_STAGE1_INTERNAL_BOUNDARY`. Calling T
the first statement merely because it is first in the comparison table would
confuse field order with compiled execution order.

## Required acquisition

The missing streams are exactly two self-describing kt=1 records, one per MPI
rank, containing the 3-D U/V accumulator after HPG, LDF, VOR, KEG, and ZAD.
Those calls are the compiled vector-invariant order at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:139-166`. The committed
writer only appends calls after those existing statements. Its header carries
magic, version, kt, levels, rank, local dimensions, global origin, owned bounds,
precision, field count, and each array's name/rank/dimensions/payload.

`nemo_testcase_l4_orca2_round92_rhs_acquisition/run.sh --preflight-only`
passes its pinned-source, additions-only, source-order, fp64, and Fortran syntax
checks. Its missing-HPG layout plant fires. The operator-run admission will run
five independent corruption plants, assemble the owned rank slabs exactly once,
and require all 20 ocean restart shards to be byte-identical to the admitted
round-90 parent. It uses the new target `ORCA2_OMIP_L4_R92RHS` and never treats
an absent or unnamed payload as zero.

## Prediction ledger

| Prediction | Verdict |
|---|---|
| R92-P1: selectors only; no new numerical formula | **CONFIRMED for the measurement card** — one pre-existing exact-input gap is declared; no card lands |
| R92-P2: five-field entry is bit-exact | **CONFIRMED** — 0 / 3,223,440 unequal |
| R92-P3: excluded modules remain off through stage 1 | **CONFIRMED** — exact census, finite replay; both selector plants fire |
| R92-P4: at least one stage-1 field is non-bit | **CONFIRMED** — all five are non-bit |
| R92-P5: GYRE unchanged by a card-only package addition | **NOT RUN / INAPPLICABLE** — no package addition lands |

## Gates, tests, and review

The four focused rung-0 controls pass. The real card replay passes and all four
real-record plants fire. The acquisition source patch applies at fuzz 0,
removes zero source lines, compiles against the pinned build modules, passes
shell/Python syntax checks, and its layout plant fires.

The required separate read-only review failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

Citation and full-suite results are recorded in the committed gate output and
the final report for this round.

## OPEN

1. Operator runs the committed acquisition and admits the two rank records.
2. Compare the five NEMO accumulators with legoESM's existing production
   source-order accumulator trace; name the first unequal statement. If all
   five are exact, continue at the next compiled boundary rather than blaming
   the unresolved drag by inference.
3. Only after a statement is identified may the package rung-0 card be proposed
   under the normal GYRE/DINO/tank/card gates. The given-entry and independent
   ten-step ladders and independent month remain subsequent rung-0 work.

## UNVERIFIED

- The first non-bit internal statement is not yet measured.
- The exact implicit linear-drag composition remains unbuilt.
- No rung-0 card is exposed through the package dispatcher.
- Both ten-step ladders and the independent month remain unmeasured.
