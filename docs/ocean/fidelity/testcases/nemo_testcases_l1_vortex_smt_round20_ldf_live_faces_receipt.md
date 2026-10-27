# Receipt — VORTEX_SMT round 20 (lane round 232): live-face LDF set

**Status: HELD.**  No physics, card, carried state, trajectory, or certified
number changed.  Adding NEMO's complete live Kmm U/V face thicknesses to the
Round-231 mask-plus-divisor pair makes A11/A22 and A13/A23 bit-exact under the
full production JIT step.  The next compiled statements, the horizontal U/V
flux assignments, remain non-bit on 1,757/1,767 cells and leave a
`2.5978639524806749e-14 K s-1` RHS maximum.  This is 99.964% below the
ordinary production residual but eight orders above the preregistered
`1e-20 K s-1` closure bound, so no trajectory gate is eligible.

Base: `a9cf6fe559` (round 231).  Preregistration: `06da07f8b`.
Discriminator extension: `e6df8730c`; corrected-record default: `63b5fdfd3`;
compiled live-face operand repair: `391c5afb0`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round232/`.

## 1. Compiled order and frozen controls

NEMO forms A11/A22 from the reference face thickness and Kmm free-surface
ratio, then the closed-bottom mask factors and A13/A23, in
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`.
The parenthesised U/V horizontal fluxes occupy
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259`.
The operator finally adds the flux divergence with its live Kmm T thickness at
`:306-310` and `:327-331`.

The valid final run resolves the preregistered Round-228 corrected record and
reproduces every frozen scalar:

| production-JIT arm | RHS maximum (`K s-1`) | fraction removed |
|---|---:|---:|
| ordinary production | `7.2621621143171395e-11` | `0` |
| closed-bottom W mask | `4.8307864472774789e-11` | `0.3348005220437417` |
| mask + live T divisor | `4.9526264855300654e-12` | `0.9318023144131400` |
| mask + live T divisor + live U/V faces | `2.5978639524806749e-14` | `0.9996422740347040` |
| recorded fluxes + live T divisor, post-hoc floor | `2.9778502051908996e-22` | not a production arm |

The aggregate pre-LDF, post-LDF and additional-LDF maxima also reproduce
`7.418332614861356e-11`, `2.0915088416728622e-07`, and
`2.090767008411376e-07 K`.  R20-P1 is confirmed.

## 2. Instrument retractions

Two preserved files in the evidence directory are invalid scientific runs and
are not used above:

* `live_face_set.json` followed the walk's stale default to the Round-227
  record.  Its malformed scalar-factor rows exposed the mismatch even though
  the aggregate RHS controls happened to reproduce.  The default now points
  fail-closed to the admitted Round-228 record.
* `live_face_set_round228.json` used the corrected record but supplied raw
  `e3u_3d/e3v_3d` to a hook that consumes the complete live face thickness.
  It omitted NEMO's recorded `(1+r3u/v(Kmm)*mask)` factor and is retracted.

The final arm constructs exactly the operand in compiled lines 243-244 before
passing it to the production hook.  Its artifact records the Round-228 oracle
root and committed SHA `391c5afb0`.  These corrections are instrument changes,
not model changes.

## 3. First non-bit statement

The final compiled-order table is:

| row | unequal cells | maximum absolute | verdict |
|---|---:|---:|---|
| hmsku / hmskv | `0 / 0` | `0 / 0` | BIT |
| A11 / A22 | `0 / 0` | `0 / 0` | BIT |
| A13 / A23 | `0 / 0` | `0 / 0` | BIT |
| zfu / zfv | `1,757 / 1,767` | `4.5474735088646412e-12 / 3.6379788070917130e-12` | first non-bit |
| dit / djt / dkt | `0 / 0 / 0` | `0 / 0 / 0` | unchanged BIT |
| vertical flux below / above | `0 / 0` | `0 / 0` | unchanged BIT |
| LDF RHS | `8,619` | `2.5978639524806749e-14 K s-1` | non-bit |

R20-P2 is confirmed: all four tensor coefficients close.  R20-P3 is
**REFUTED**: the RHS does not reach either the exact post-hoc floor or the
`1e-20` acceptance bound.  Since every recorded input to the horizontal-flux
assignment is now bit-exact, the first remaining non-bit statements are
NEMO's parenthesised `zfu/zfv` evaluations at compiled lines 254-259.  The
current transcription evaluates the four vertical-gradient addends in a
different association; that is the discriminating one-variable arm for the
next round, not a landed claim in this one.

Both input controls are non-vacuous.  A one-ULP U-face change makes one A11
cell non-bit and changes the tendency digest; the independent V-face change
does the same for A22.  Both runs print `STATUS PLANT-FIRED` and exit 1.
R20-P4 is confirmed.

## 4. Landing and blast radius

R20-P5 is not entered because local production closure is mandatory before a
shared-card trajectory gate.  Production remains byte-for-byte unchanged: the
round edits only the committed measurement script and its private test hook
inputs.  Therefore no SMT/flat-VORTEX, tank, generic-GYRE, certified GYRE
ladder/year, DINO, or ORCA2 trajectory row is re-baselined.

The certified GYRE values remain:

| row | certified value/status |
|---|---:|
| kt2 T / S | `6.054357687161262e-16 / 5.786374251651969e-16`, AT-BAR |
| kt2 U / V | `8.326672684688674e-17 / 9.71445146547012e-17`, AT-BAR |
| kt3 T / S | `5.861944241472192e-10 / 1.6244926507906096e-11`, DEBT |
| day 30 T rms | `2.3432437414839976e-06 K` |
| day 240 T rms | `6.5817049818294640e-05 K` |
| day 360 T rms | `5.4077372201617810e-05 K` |

DINO shares the production operator, but the private operand arm is not
selectable by any card and no production expression changed.  Its certified
month row remains `2.053801168e-03 K`; it is not claimed as remeasured here.
LOCK_EXCHANGE and OVERFLOW likewise cannot execute a private diagnostic arm.

For ORCA2, the measured statement remains **UNMEASURED** on its own developed
state.  Its rung-0 ISO/MSC tracer path executes the same compiled lines
254-259, so the next merge pointer is specific: test the pairwise vertical-
gradient association before attributing any remaining rung-0 tracer-LDF row.

**UNASKED list: EMPTY.**  No physics, configuration, default, threshold,
record source, state, or stabiliser choice was made.

## 5. Tests, citations, and review

The final focused suite reports:

> 137 passed in 181.09s (0:03:01)

It covers the complete receipt-citation suite, every SMT-card test, and the
complete GM/Redi unit file.  The round citation gate finds four mapped
compiled-source citations, zero failures, zero unmapped citations, and zero
map-audit failures.  Shifting the horizontal-flux citation by two lines makes
the gate print `SYMBOL-NOT-AT-LINE` and exit 1.  The independent U-face and
V-face production plants also each print `STATUS PLANT-FIRED` and exit 1.

The required read-only Codex review was attempted and could not initialize in
the sandbox.  Its complete verdict transcript is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**; there is no reviewer
`DO NOT SHIP` verdict.  No physics candidate lands.

## 6. OPEN

Round 233 remains in the same compiled `traldf_iso` stage.  Add one private
production-JIT arm that evaluates the four vertical-gradient addends in the
pairwise parenthesisation NEMO uses at compiled lines 254-259 while retaining
the exact mask, live face and divisor set measured here.  Require zfu/zfv to
become bit-exact and the RHS to reach the `2.9778502051908996e-22 K s-1`
recorded floor.  If it closes, run the full shared-card landing gates; if it
does not, split the outer multiply/add association inside the same two
statements.  Do not start SMT-4 until this same-stage walk closes or the
remaining residue is proven to be the production-JIT rounding floor.

No NEMO acquisition and no configuration decision is required.
