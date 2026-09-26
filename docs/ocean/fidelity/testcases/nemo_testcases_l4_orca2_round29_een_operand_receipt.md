# NEMO testcase Lane 4 — ORCA2 card round 29 EEN-operand receipt

Date: 2026-09-26

Parent: `23eb11d18b9a081b1acb764b8ee52f1c3f265ebf`

Status: **HELD — ROUND 28'S EEN ARM WAS ALREADY DENOMINATOR-ONLY.**
At the first production EEN call changed by NEMO's carried raw F thickness,
the total-vorticity numerator and all six face-transport inputs are
bit-identical to the parent.  Only the live F thickness used as EEN's
reciprocal denominator differs.  No model statement lands; the experimental
`packages/` replay is reverted and the final `packages/` tree is identical to
the parent.

Every result below is **independent with Decision-52 SSH**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple are unchanged.  Evidence
is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round29/`.

## Compiled statements and operand boundary

The admitted build computes the reciprocal of live F thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738`.
It then selects the absolute/relative vorticity numerator at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:741-779`,
forms the two face transports, assembles the triads, and adds the U/V tendency
at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:784-802`.
The frozen reference part of the live denominator is constructed and exchanged
at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937`.
The admitted `ocean.output` resolves `ln_dynvor_een = T` and
`ln_dynvor_msk = F`, so these are executed statements.

Round 28's EEN-only diff changed only the live vertex-thickness override passed
to the production EEN function.  Round 29 wrapped that existing function with
a WRITE-only callback and recorded its exact arguments and return arrays during
the stage-2 exposure.  It did not reconstruct the operator in the probe.

## Independent production-call result

The parent and raw-F captures each contain three production EEN calls.  Every
call's output changes.  At the first changed call, the live F denominator
differs in **455,904 / 809,070 cells**, maximum **651.2256783597969 m**.

| operand class at first changed call | exact result |
|---|---:|
| relative-vorticity numerator `zeta` | bit-identical |
| vertex Coriolis `f_vtx` | bit-identical |
| U face thickness and velocity `h_u`, `u` | bit-identical |
| V face thickness and velocity `h_v`, `v` | bit-identical |
| U/V three-dimensional masks | bit-identical |
| live F denominator | **455,904 unequal; max 651.2256783597969 m** |

The exposed stage-2 EEN tendency reproduces the saved round-27/28 arrays
bit-for-bit in both arms.  Parent versus raw-F changes **413,554 / 803,640 U**
cells, maximum **2.1873555668998308e-05 m s-2**, and
**412,558 / 804,600 V** cells, maximum
**3.213274876559519e-05 m s-2**.  Thus the round-28 trajectory arm, including
its kt=4 raw-`e3w` refusal, is assigned specifically to the denominator
statement; no numerator, transport, or tendency-association split remains.

The first NEMO trajectory mismatch remains the admitted round-28 result:
kt=1 stage-1 temperature.  Round 29 does not claim a new ten-step trajectory;
it replays the exact round-28 source arm only through the stage-2 operand
boundary and compares its result to the saved round-27/28 arrays.

**INSTRUMENT CORRECTION:** the first observer version required the raw EEN
function return to equal the later exposed stage accumulator.  The production
run refused that assumption: the two are distinct arithmetic boundaries.  The
gate now compares raw EEN calls to raw EEN calls and separately proves observer
passivity against the saved exposed accumulator.  No scientific number from
the refused version is retained.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R29-P1 | **CONFIRMED** | At the earliest changed EEN call, all eight numerator/transport inputs are bit-identical and only live F thickness differs. |
| R29-P2 | **CONFIRMED** | Both exposed arrays equal round 27/28 bit-for-bit; raw versus parent reproduces 413,554 U / 412,558 V unequal cells. |
| R29-P3 | **CONFIRMED** | The observed parent exposed arrays equal the saved parent arrays bit-for-bit. |
| R29-P4 | **CONFIRMED** | The planted transport-input digest is refused. |

No failed prediction was rewritten.

## Gate, review, and tests

The round-29 outcome gate exits 2 with `HELD`.  Its transport-input plant exits
1 with `first changed EEN call also changes numerator or transport input`.

The required separate `codex exec --sandbox read-only` review was attempted on
the committed round diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

Validation results are recorded in the final follow-up commit for this receipt.

No GYRE/DINO/lock-exchange/overflow landing gate is claimed: this round lands
no model statement, and `git diff 23eb11d18 -- packages` is empty at the final
tip.

## Choices

ASKED: Decision 54 and round 28's OPEN item authorize this EEN operand census.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
score, sea-ice selector, or NEMO source changed.

## OPEN

1. Decision 54 remains held.  Split the denominator's compiled sub-operands in
   order: frozen `e3f_0vor`, live `r3f`, then frozen `fe3mask`; keep numerator,
   face transports, masks, metrics, forcing, and score fixed.
2. Preserve the round-28 LDF-only artifact for the eventual whole-bundle retry;
   it is finite and improves late U/V, but is not a landing verdict.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The inherited duplicate citation-map literal keys remain open.
