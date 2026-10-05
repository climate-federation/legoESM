# NEMO testcase Lane 4 — ORCA2 card round 32 EEN landing and read-outs

Date: 2026-09-26

Parent: `ee7eb6048934502db5f8e3d30047e233f01dc372`

Landing commit: `2b03a6db5feb03dc035d97a177d59a9061c20497`

Status: **LANDED — DECISION 54 IS COMPLETE.**  The one remaining compiled
vorticity-denominator statement now uses NEMO's repaired `e3f_0vor` reference.
The GYRE ten-step ladder and all 360 daily snapshots are byte-identical.  The
ORCA2 ladder completes all 40 checkpoints without moving a bit-identical row
out of its bar.  The ordered operator replay is not bit-exact and its frozen
prediction is retained as **REFUTED**; no repair for that separate debt lands.

Trajectory results below are **independent with Decision-52 SSH**.  The direct
operator replay is **given NEMO's entry**.  The six sea-ice selectors and the
card's `unmeasured_features` tuple are unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round32/`.

## Compiled statements and the one landed change

The admitted compiled build forms the masked four-cell reference at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:914-919`,
performs the F-fold exchange at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:935`, and then
replaces remaining zero values with the mesh F thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:937`.  The
executing reciprocal consumes that repaired array at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738`.

The landing changes only the production call that builds this vorticity
reference: it supplies the card's carried mesh `e3f_0` as the zero-substitution
operand.  The construction gate reports 0 / 799,200 unequal cells for the
masked average, the repaired fold order, and the final repaired array.  Its
operand plant reports 416,202 differences and exits nonzero.  Before the
repair, the final construction had 335,196 unequal cells.

This is the vorticity half left open by round 31.  Lateral diffusion continues
to consume its independently routed mesh reference.  Decision 54 is therefore
complete without conflating NEMO's two different F-thickness consumers.

## Independent GYRE safety gates

The base `ee7eb6048` and landing `2b03a6db5` ten-step runs compare exactly:
all 70 certified rows have zero field move and zero oracle-residual worsening,
and all 210 arrays in the residual archives are `np.array_equal`.  Both
archives have SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
The three-ULP worsening plant exits nonzero.

The separately run base and tip 360-day members have **0 / 360** differing
daily snapshots.  Their pinned NEMO temperature RMS values are unchanged:

| day | RMS T, K | snapshot SHA-256 |
|---:|---:|---|
| 30 | `6.572574374770603e-05` | `a66143733bcc9e4efaa22fe2b3a831629e4d57ffd701ddd827d5553c3e0b007a` |
| 240 | `1.644836070117868e-02` | `0d4f16d0c51da705d5cab1e73bc8c59262322e0997f09f521f5f430b6f87af54` |
| 360 | `1.1225660018551306e-02` | `c6b7e1523b9a6dbb6299bc4d26e5b25c7e53de911249a3df282fa7924db7db96` |

The day-gap rows are identical between base and tip.  This measures rather
than infers round 31's zero-coefficient argument.

## Independent ORCA2 ladder

Both runs complete kt=1..10, 40 checkpoints each.  The first non-bit statement
remains kt=1 stage-1 temperature, and no bit-identical row leaves the bar.
The first movement caused by the landing is kt=5 stage-2 salinity.  Of 200
scored rows, 80 move in content: one maximum moves toward NEMO, two move away,
and 77 retain the same maximum.  The kt=10 stage-3 velocity maxima are:

| field | parent, m/s | landed, m/s |
|---|---:|---:|
| U | `0.4230544199344073` | `0.4230544199344075` |
| V | `0.6838675521949863` | `0.6838675521949865` |

These are the preregistered round-31-arm values to one ULP.  The truncation and
velocity outcome plants both exit nonzero.

## Ordered read-outs

### Round-24 compiled-statement replay — REFUTED

Given NEMO's recorded kt=2 entry, the canonical production replay is not
bit-exact against the literal statements at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.
All 411,736 scored U cells differ, with maximum `2.2089693992382724e-05`
m/s2, and all 412,537 scored V cells differ, with maximum
`1.798072838581657e-05` m/s2.  A second round-32 replay that explicitly
supplies the six stored reciprocal operands reproduces the same leading
scale; it does not close the row.

The first draft of the round-32 replay omitted the landed metric reciprocal
boundary and reproduced the historical discrepancy.  That instrument result
is retracted in code: the final probe reconstructs all six stored operands.
Because both the corrected probe and the pre-existing canonical gate remain
red, R32-P3 is **REFUTED**, not repaired post hoc.  This read-out does not veto
the already-measured one-statement EEN landing; it opens a separate
lateral-diffusion operator debt.

### Carried `hf_0` in `r3f` — measured, not landed

NEMO builds the carried column depth from `e3f_0` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-206`; the live
F ratio is formed at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286` and is
consumed by the reciprocal at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738`.

The carried and reconstructed column depths differ in 14,324 / 26,640 cells,
maximum `651.239260895305` m.  Substituting only carried `hf_0` changes the
denominator in 376,492 / 809,070 cells, but the first raw EEN U and V outputs
remain bit-identical.  At exposed stage 2, U moves in 412,170 cells with
maximum `5.153126997217792e-09` m/s and V moves in 411,143 cells with maximum
`2.5500881043307236e-09` m/s.  Those equal the registered round-30 ceilings,
so R32-P4 is **CONFIRMED**.  The vacuity plant replaces carried depth with the
reconstruction and is refused explicitly.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R32-P0 | **CONFIRMED** | Construction is 0 / 799,200 unequal; ORCA2 completes 40/40; velocity maxima equal the frozen values to one ULP; no exact row leaves the bar. |
| R32-P1 | **CONFIRMED** | GYRE has 70/70 unchanged rows, 210/210 exact residual arrays, 360/360 exact snapshots, and all three pinned year scores. |
| R32-P2 | **CONFIRMED** | The exact inherited card inventory passes 170/170; the post-receipt push battery is reported below. |
| R32-P3 | **REFUTED** | The landed production replay differs in every scored U and V cell. |
| R32-P4 | **CONFIRMED** | First raw EEN output is exact and the stage-2 maxima equal, but do not exceed, the registered bounds. |
| R32-P5 | **CONFIRMED** | Construction, ladder, GYRE, `hf_0`, and citation plants all exit nonzero. |

Failed prediction R32-P3 is retained exactly as preregistered.

## Gates, review, and tests

The exact shared-card inventory is green: 160 DINO/rule-12/lock/overflow
tests pass with nine warnings, and the remaining tank round-34 file passes
10/10, for **170/170**.  The focused round-31/32 binding suite passes 13/13.

The one required `tests/ocean/fidelity -n 12` battery collected 1,889 tests
and reached the inherited final-tail stall.  It was interrupted after a
bounded 17-minute idle wait at 99%, after emitting **1,860 passed, 7 skipped,
8 failed, and 14 unfinished**.  Three failures were this round's citation
ratchet correctly observing the three-line `vertical.py` shift; the citations
were rigidly re-anchored and the complete focused citation suite then passed
16/16.  The other five are the inherited SI3 scalar-math provenance gate,
round-129 worktree stamp, stale round-51 suffix assertion, legacy report-
emitter stamp ratchet, and missing `hires_lane_surface` case-board row.  No
round-32 test fails.

The required separate `codex exec --sandbox read-only` review was attempted
on the committed round diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The receipt citation gate passes all seven compiled citations with zero
failures, zero unmapped citations, and zero map-audit failures; all nine
self-tests fire.  Its real +2-line plant on the landed vorticity citation exits 1 with
`SYMBOL-NOT-AT-LINE`.

The final ORCA2 push battery runs the citation gate, TKE terms, recipe,
freshwater closure, and parallel-receipt gate against the committed receipt:
**127/127 pass in 474.23 s**.

## Choices

ASKED: the B7 addendum authorizes the one repaired vorticity-denominator
statement after the GYRE year, card, and push gates.  That statement lands.

UNASKED: none.  No configuration value, stabilizer, NEMO source, carried
state, sea-ice selector, score, or acquisition changed.  The `hf_0` arm is a
read-out only.

## OPEN

1. The given-NEMO-entry lateral-diffusion replay is non-bit on every scored
   U/V cell.  Start from its live production operands and walk one variable at
   a time; do not reuse the refuted bit-exact claim.
2. NEMO's carried `hf_0` is a measured secondary owner with an exact first raw
   EEN cancellation and a `5.15e-09` / `2.55e-09` m/s exposed effect.  It is
   registered, not landed.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. Decision 52's independent ORCA2 initial-state transcription and year
   comparison remain owed.
6. The inherited duplicate citation-map literal keys remain open.
