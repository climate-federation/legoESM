# Receipt — VORTEX_SMT round 19 (lane round 231): same-stage LDF pair

**Status: HELD.**  No physics, card, carried state, trajectory, or certified
number changed.  Combining NEMO's closed-bottom horizontal W-mask statement
with its recorded live Kmm T divisor removes 93.1802314% of the production-JIT
tracer-LDF RHS maximum, from `7.2621621143171395e-11` to
`4.9526264855300654e-12 K s-1`.  This refutes the preregistered less-than-90%
prediction, but the result remains ten orders of magnitude above the recorded
compiled-rounding floor.  The first remaining source boundary is the live Kmm
U/V face thickness in A11/A22, so the pair stays held and no trajectory gate
is entered.

Base: `501810fe4d` (round 230).  Preregistration: `37efdb57b`.
Measurement instrument: `ba43053e9`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round231/`.

## 1. Compiled statements and frozen controls

The R16 build first forms A11/A22 from the live Kmm U/V thickness, then the
two four-W-mask reciprocals and A13/A23, and finally the horizontal fluxes at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`.
At the deepest tracer level this reads NEMO's closed `jpk` W level rather than
wrapping the surface.  The same operator then adds horizontal and vertical
flux divergence with the stored area reciprocal and live Kmm T thickness at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`
and `:327-331`.

The stamped production run reproduces every frozen control:

| production-JIT arm | RHS maximum (`K s-1`) | fraction removed |
|---|---:|---:|
| ordinary production | `7.2621621143171395e-11` | `0` |
| closed-bottom W mask only | `4.8307864472774789e-11` | `0.3348005220437417` |
| live divisor only | `8.4505340078032935e-11` | `-0.1636388550378563` |
| closed-bottom W mask + live divisor | `4.9526264855300654e-12` | `0.9318023144131400` |
| all recorded fluxes + live divisor, post-hoc floor | `2.9778502051908996e-22` | not a production arm |

The aggregate pre-LDF, post-LDF, and additional-LDF maxima are unchanged at
`7.418332614861356e-11`, `2.0915088416728622e-07`, and
`2.090767008411376e-07 K`.  R19-P1 is confirmed.

## 2. Pair result and first remaining statement

The pair's compiled-order table is:

| row | unequal cells | maximum absolute difference |
|---|---:|---:|
| hmsku / hmskv | 0 / 0 | 0 / 0 |
| A13 / A23 | 0 / 0 | 0 / 0 |
| A11 / A22 | 36,032 / 36,012 | 170.383900019 / 170.383895118 |
| zfu / zfv | 6,543 / 6,548 | 0.0905765696 / 1.6392616702 |
| tracer gradients dit / djt / dkt | 0 / 0 / 0 | 0 / 0 / 0 |
| vertical flux zfw below / above | 0 / 0 | 0 / 0 |
| LDF RHS | 8,641 | `4.9526264855300654e-12 K s-1` |

R19-P2 predicted less than 90% removal and is **REFUTED** by the measured
93.18%.  Because the conditional trigger for R19-P3 was not met, the live-face
third arm was not run post hoc.  This preserves the preregistered decision tree.

The pair is nevertheless not source-closed: A11/A22 are the first non-bit
outputs in the compiled order.  NEMO multiplies its metric ratios by
`e3u_3d*(1+r3u(Kmm)*umask)` and
`e3v_3d*(1+r3v(Kmm)*vmask)` before it forms the now-exact mask cross terms in
the cited `:243-259` block.  The pair deliberately leaves those
face thicknesses on legoESM's ordinary path.  Therefore R19-P5 is not reached:
90% magnitude removal alone is insufficient; the local row must also reach the
compiled-rounding floor before any shared-card trajectory gate can run.

The controlled-arm checks pass.  dit, djt, dkt, zfw below, and zfw above are
bitwise identical to the ordinary production step.  The existing closed-mask
non-vacuity test fails on the superseded periodic floor, and a one-ULP change to
the pair's live-divisor input changes the full-step tendency digest; the plant
prints `STATUS PLANT-FIRED` and exits 1.  R19-P4 is confirmed for the measured
pair.  The unexecuted third-arm plants are not claimed.

## 3. Landing, certified rows, and blast radius

No physics candidate lands.  The new code is a private write-only diagnostic
operand whose default preserves the round-230 expression; ordinary cards
cannot select it.  The mandatory DINO month gate still ran and reports:

> DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.053801168e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS

The certified GYRE rows therefore remain unchanged before and after:

| row | before | after | status |
|---|---:|---:|---|
| kt2 U | `8.326672684688674e-17` | same | AT-BAR |
| kt2 V | `9.71445146547012e-17` | same | AT-BAR |
| kt3 T | `5.861944241472192e-10 K` | same | DEBT |
| kt3 S | `1.6244926507906096e-11` | same | DEBT |
| day-30 T rms | `2.3432437414839976e-06 K` | same | certified |
| day-240 T rms | `6.5817049818294640e-05 K` | same | certified |
| day-360 T rms | `5.4077372201617810e-05 K` | same | certified |

No SMT/flat-VORTEX, tank, generic-GYRE, GYRE ladder/year, or ORCA2 trajectory
row is re-baselined.  This is not a waiver for a landing: there is no physics
landing.

**UNASKED list: EMPTY.**  No physical/configuration/default choice was made.

## 4. Tests, citations, and review

The final focused suite reports:

> 137 passed in 187.58s (0:03:07)

It covers the complete receipt-citation suite, every SMT-card test, and the
complete GM/Redi unit file.  The new direct mask-scope test proves that the
private arm changes only the deepest horizontal W-mask pair; its old periodic
value fails the bottom assertions.  The pair's full-production divisor plant
exits 1 with `STATUS PLANT-FIRED`.

The required separate read-only Codex review could not initialize inside the
sandbox.  Its complete verdict transcript is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**; there is no reviewer
`DO NOT SHIP` verdict.  No physics candidate lands.  The round citation gate
finds four unique mapped compiled-source citations, zero failures, zero
unmapped citations, and zero map-audit failures.  Shifting the horizontal-flux
citation by two lines makes the gate print `SYMBOL-NOT-AT-LINE` and exit 1.
Both runs carry a clean committed-worktree stamp.

## 5. OPEN

Round 232 remains inside the same `traldf_iso` stage.  Add the already-recorded
live Kmm U/V face thicknesses to the measured mask-plus-divisor pair in one
production-JIT arm.  Require A11/A22 to become bit-exact and the RHS to reach
the `2.9778502051908996e-22 K s-1` floor.  If A11/A22 close but the RHS does
not, the next statement is the parenthesised horizontal-flux association at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259`.
Only a locally closed set proceeds to the complete shared-card landing gates;
otherwise every member stays held.  SMT-4 remains behind this same-stage walk.

No NEMO acquisition and no configuration decision is required.
