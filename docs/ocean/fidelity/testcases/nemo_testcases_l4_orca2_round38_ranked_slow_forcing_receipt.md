# NEMO testcase Lane 4 — ORCA2 card round 38 ranked slow-forcing receipt

Date: 2026-09-26

Parent: `4cda04817736a4bc48f40d4bea54e1c17a83a0c4`

Status: **HELD — THE FIRST RANKED SLOW-FORCING PRODUCER OPERAND IS THE
U-FACE THICKNESS, BUT ITS ISOLATED SUBSTITUTION IS A REAL CANCELLING-PAIR
FAILURE.**  No production model file changed.

Every number below is **given NEMO's entry**.  No independent trajectory
number is claimed.  The six sea-ice selectors and the card's
`unmeasured_features` tuple are unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round38/`.

## Compiled order and admitted record

The executing build records and then consumes `e3u_3d`, `uu(Krhs)`, and
`umask` in the vertical average at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:189-204`, calls
the baroclinic drag at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:218-221`, and
adds wind at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:225-235`.

The existing operator-run acquisition admits without repair: both ranked
streams have the frozen 22,814,424-byte size, distinct rank headers and
digests, `MPIRUN_RC=0`, `RUN DONE`, `STOP 0`, and one marker per rank.  Both
ranks' recorded operands replay their recorded depth mean and wind boundary
bit-for-bit.  R38-P1 and R38-P3 are **CONFIRMED**.

## Round-20 reproduction

Before the ranked payload is used, the inherited source boundary reproduces:
substep-1 rank-1 `zu_frc` differs on 64/64 disputed east-source cells, maximum
`7.356481146903598e-18` m/s2 and 383,252 ULP.  Substituting only recorded
`zu_frc` closes exit U at 0/64 unequal.  R38-P2 is **CONFIRMED**.

## First non-bit producer and isolated arms

On the 64 disputed rank-1 rows, the first non-bit operand in compiled order is
`e3u_3d`: 1,754/1,920 wet layer values differ, maximum
`0.027675060932892848` m.  `umask` is exact, but `uu(Krhs)` is not: 1,753/1,920
values differ, maximum `1.4862887125471208e-17` m/s2.  The reciprocal reference
depth also differs on 64/64 cells, maximum `1.2435455028960568e-06` 1/m.
R38-P4 is **REFUTED**: thickness, not the completed 3-D RHS, is first.

The one-variable arms are decisive:

| depth-mean replay on 64 disputed cells | unequal | maximum m/s2 |
|---|---:|---:|
| production operands | 64 | `7.356587026022005e-18` |
| recorded thickness only | 64 | `9.423577925168289e-11` |
| recorded 3-D RHS only | 36 | `4.235164736271502e-22` |

Replacing the first differing operand alone worsens the boundary by more than
seven orders of magnitude.  R38-P5 is **REFUTED** and R38-P6 is **CONFIRMED**:
no statement is eligible to land.  This is the campaign's cancelling-pair
signal, not permission to retain or repair either half independently.

## Controls, review, and tests

The swapped-rank header plant and a one-ULP mutation of the scored recorded
depth-mean boundary both fire.  The first attempted scientific plant changed
one RHS element by one ULP, but the 30-level reduction absorbed it and the
control did not fire.  That control is **RETRACTED** and remains in
`plant.log`; the replacement targets the actual scored boundary and fires.

The required separate `codex exec --sandbox read-only` review was attempted at
the committed diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The focused ranked-reader plus citation-gate battery passes 18/18.  The
receipt citation gate passes all three compiled citations with zero failures,
zero unmapped citations, and zero map-audit failures; its rigid two-line plant
fails with `SYMBOL-NOT-AT-LINE`.

The required `tests/ocean/fidelity -n 12` battery was launched once.  It
reached 99% and the inherited final-tail stall, then was interrupted after a
bounded silent wait and is not represented as green.  It emitted exactly the
five failures already listed by rounds 36 and 37: SI3 MY_SRC provenance, the
stale GYRE member/gate stamp, the round-51 trace suffix, the unstamped legacy
report emitters, and the missing `hires_lane_surface` case-board row.  Both
round-38 tests passed in that battery.

The gate and plant run from committed code at fp64/libm with production JIT.
No ORCA2 or GYRE trajectory gate is eligible because the isolated statement
misses the exact bar and the `packages/` tree is unchanged.

## Choices

ASKED: resume round 20's ranked slow-forcing producer walk.

UNASKED: none.  No configuration value, carried state, stabilizer, sea-ice
selector, score, NEMO source, acquisition, or model statement changed.

## OPEN

1. The actual whole-card first non-bit statement remains kt=1 stage-1
   temperature.  It owns the next round under the lane's rank-first rule.
2. The slow-forcing branch now has a registered cancelling pair: compare the
   compiled products and partial sums before changing either thickness or the
   3-D RHS.  Do not land the thickness arm independently.
3. The northern-fold mask and wind-stress operands (668/35 cells) remain
   reported, not landed.
4. Decision 52's independent initial-state year comparison remains owed.
5. The inherited duplicate citation-map literal keys remain open.
