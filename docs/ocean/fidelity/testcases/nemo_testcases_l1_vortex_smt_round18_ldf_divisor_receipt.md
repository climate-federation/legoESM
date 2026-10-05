# Receipt — VORTEX_SMT round 18 (lane round 230): live tracer divisor

**Status: HELD.**  No physics, card, carried state, trajectory, or certified
number changed.  The one-variable production-JIT arm leaves every upstream
flux bitwise unchanged but replacing only the final volume divisor with
NEMO's live Kmm T thickness makes the LDF RHS maximum **16.36% worse**.  The
preregistered magnitude-closure prediction is therefore refuted and the
candidate does not enter the trajectory landing gate.

Base: `536f3da0dee7` (round 229).  Preregistration: `6de0984db`.
Measurement instrument: `b0507e737`; citation re-anchor: `65a43a4b2`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round230/`.

## 1. Compiled statement and frozen reproduction

NEMO first forms the horizontal and vertical tracer fluxes.  It then adds
their divergence to `Krhs`, multiplying by stored `r1_e1e2t` and dividing by
the live Kmm T-point thickness
`e3t_3d*(1+r3t(Kmm)*tmask)` in the ordinary-level loop at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310` and
again in the deepest-level arm at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
Those are the compiled statements tested; no physical inference substitutes
for the source.

Before the new arm, the committed production-JIT walk reproduced every frozen
Round-229 scalar:

| boundary | maximum absolute difference |
|---|---:|
| production LDF RHS | `7.2621621143171395e-11 K s-1` |
| recorded horizontal fluxes only | `4.8452277849849912e-11 K s-1` |
| all recorded fluxes | `4.8452277850122817e-11 K s-1` |
| all recorded fluxes plus NEMO divisor | `2.9778502051908996e-22 K s-1` |
| pre-LDF trajectory boundary | `7.418332614861356e-11 K` |
| post-LDF trajectory boundary | `2.0915088416728622e-07 K` |

Thus R18-P1 is confirmed and the instrument still sees the Round-229 result.

## 2. One-variable production result

The existing full production step was JIT-compiled with one private operand
override: only the final divisor thickness was replaced by the recorded NEMO
array.  Slopes, tensor factors, horizontal and vertical fluxes, metrics,
tracer state, and returned model state remained on the ordinary path.

| production-JIT row | unequal cells | maximum absolute |
|---|---:|---:|
| upstream `zfu` | 0 | 0 |
| upstream `zfv` | 0 | 0 |
| upstream `zfw_kp1` | 0 | 0 |
| upstream `zfw_top` | 0 | 0 |
| LDF RHS, old divisor | 8,641 | `7.2621621143171395e-11 K s-1` |
| LDF RHS, live NEMO divisor | 8,641 | `8.4505340078032935e-11 K s-1` |

The changed operand removes `-0.16363885503785625` of the baseline maximum:
it worsens the row by 16.36%.  R18-P2 and R18-P5 required production closure
below `1e-20 K s-1` and at least 90% magnitude removal.  Both are **REFUTED**.
R18-P3, which would promote the model-built literal thickness as a landing
candidate, is not reached; this arm deliberately used NEMO's recorded
thickness, the strongest possible test of the divisor statement itself.

The production input is controlled.  A one-ULP increase at active cell
`(j,i,k)=(31,31,0)` changes the full-step tendency digest from
`07515806fe0c46739a14686f9e779bf5be505e80835c7a74a8208c930b254037` to
`2466ca5e4442865eb50be11c142df135cb6a4e8ab3dfa96c0c3f062cfc917e15`;
the gate prints `STATUS PLANT-FIRED` and exits 1.  Independently, replacing the
new divisor use with the old Jacobian thickness makes the focused test fail at
its required changed-output assertion.  The clean/default call remains
bit-identical when no private override is supplied.

## 3. Retraction and interpretation

Round 229 called the divisor "the remaining magnitude owner" because the
post-hoc reconstruction reached `2.98e-22 K s-1` after substituting both the
recorded fluxes and the divisor.  That **single-owner wording is retracted**.
The production experiment shows the exact divisor alone is harmful; the
post-hoc closure belongs to a cancelling **same-stage set** consisting of the
horizontal-flux family plus the divisor.  The reconstructed floor remains a
valid statement about that joint set, not evidence that the divisor alone
owns the production error.

This is also why the source-exact statement does not land: the round's frozen
magnitude gate fires before any trajectory gate.  The model-code diff is
measurement-only: an optional private divisor operand and its tests.  The
default is `None`, which selects the previous expression byte-for-byte.  No
card, configuration field, default, coefficient, threshold, state layout, or
stabiliser changed.

## 4. Certified rows and blast radius

Because R18-P2 fails, R18-P4 is not entered and no certified reference is
re-pinned.  The current certified GYRE values remain:

| row | certified value/status |
|---|---:|
| kt2 T / S | `6.054357687161262e-16` / `5.786374251651969e-16`, AT-BAR |
| kt2 U / V | `8.326672684688674e-17` / `9.71445146547012e-17`, AT-BAR |
| kt3 T / S | `5.861944241472192e-10` / `1.6244926507906096e-11`, DEBT |
| day 30 T rms | `2.3432437414839976e-06 K` |
| day 240 T rms | `6.5817049818294640e-05 K` |
| day 360 T rms | `5.4077372201617810e-05 K` |

The mandatory private-directory DINO month gate was run despite the unchanged
default path.  It reports:

> DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.053801168e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS

No SMT/flat-VORTEX, tank, generic-GYRE, GYRE ladder/year, or ORCA2 trajectory
row is claimed from the private arm.  They are unchanged by construction
because ordinary calls cannot supply the private override; the unit test pins
that default equivalence.  This is not used as a waiver for a physics landing:
there is no physics landing.

**UNASKED list: EMPTY.**  No scientific or configuration choice was made.

## 5. Tests, citations, and review

The focused divisor test passes; its old-expression mutation fails as intended.
The production-input plant exits 1 and reports `STATUS PLANT-FIRED`.  The
repository citation suite passes after the required SequenceMatcher re-anchor;
the one enclosing program span grows by exactly the inserted diagnostic line.

The final focused suite, direct receipt citation gate and shifted-citation
plant results are recorded in the round evidence directory.  The separate
read-only Codex review verdict is quoted in the final committed revision of
this section.

## 6. OPEN

Round 231 stays inside this same `traldf_iso` stage and measures the cancellation
directly.  Apply the already source-proven closed-bottom W-mask patch together
with the live Kmm T divisor in one production-step arm, keeping all other
operands fixed.  If that pair still removes less than 90%, split the remaining
horizontal family by adding the already-recorded live U/V face thicknesses as
a third arm.  Only a same-stage set that closes production under JIT proceeds
to the full trajectory gates; otherwise all members remain held.  This order
is forced by the measured facts: the W-mask arm alone removed 33.48%, the
divisor alone worsens 16.36%, and recorded horizontal fluxes plus the divisor
jointly reach the compiled-rounding floor.

No NEMO acquisition and no configuration decision is required.
