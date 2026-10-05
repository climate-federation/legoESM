# Receipt — VORTEX_SMT round 22 (lane round 234): LDF divergence association

**Status: HELD.** No physics, card, carried state, trajectory, or certified
number changed. Starting from bit-exact horizontal and vertical fluxes, all
four cumulative source-order arms are bit-identical to the Round-233 control:
8,613 wet cells remain unequal at
`2.5978639524806749e-14 K s-1`. The compiled divergence association is inert,
so the preregistered closure prediction is refuted and no trajectory landing
gate is eligible.

Base: `888a3201e` (round 233). Preregistration: `27f1ae769`, corrected before
measurement at `74ae7f8e6`. Discriminator and unit coverage: `5e8fa86dd`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round234/`.

## 1. Compiled order and frozen controls

NEMO forms the horizontal fluxes at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259`.
The regular-level update then evaluates the U-face difference, V-face
difference, vertical-flux difference, area reciprocal and live-thickness
division at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`;
the deepest-level form uses the lower vertical flux alone at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
The instrument records `rhs_increment` as `rhs_after-rhs_before` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90:149-151`,
so its `2.9778502051908996e-22 K s-1` recorded-input residual is the comparison
floor, not an exact-zero target.

The production-JIT run reproduces every frozen scalar:

| arm | zfu/zfv unequal | RHS unequal cells | RHS max abs (`K s-1`) | verdict |
|---|---:|---:|---:|---|
| ordinary production | not substituted | `8,641` | `7.2621621143171395e-11` | control reproduced |
| complete live operands | `1,757 / 1,767` | `8,619` | `2.5978639524806749e-14` | control reproduced |
| exact literal fluxes | `0 / 0` | `8,613` | `2.5978639524806749e-14` | control reproduced |
| source-rounded differences | `0 / 0` | `8,613` | `2.5978639524806749e-14` | inert |
| plus source-rounded horizontal sum | `0 / 0` | `8,613` | `2.5978639524806749e-14` | inert |
| plus source-rounded vertical sum | `0 / 0` | `8,613` | `2.5978639524806749e-14` | inert |
| plus source-rounded area multiply/divide | `0 / 0` | `8,613` | `2.5978639524806749e-14` | inert |
| recorded fluxes + stored operands, post-hoc floor | `0 / 0` | `8,451` | `2.9778502051908996e-22` | floor only |

The aggregate pre-LDF, post-LDF and additional-LDF maxima remain
`7.418332614861356e-11`, `2.0915088416728622e-07`, and
`2.090767008411376e-07 K`. All hmsku/hmskv, A11/A22/A13/A23, dit/djt/dkt and
both vertical-flux controls remain bit-exact. R22-P1 is confirmed.

## 2. Refutation and first non-bit statement

R22-P2 predicted that retaining the compiled regular/deepest divergence order
would reach the `2.9778502051908996e-22 K s-1` floor. It is **REFUTED**: every
cumulative arm has the same tendency digest
`6300f14abb814b3af465a3439a143f9c8ff2d2f8ae38d1f76de39787a108d542` and the
same `2.5978639524806749e-14 K s-1` maximum.

The first non-bit source statement after exact fluxes remains the regular/deepest
RHS update at compiled lines 306-310/327-331, but none of its tested arithmetic
boundaries owns the residue. Accordingly this round does not claim a narrower
source owner. The next unmeasured arithmetic boundary is NEMO's stored area
reciprocal: the existing diagnostic proves `1/(e1t*e2t)` is bit-exact only
after materialisation, not through the fused production consumer. That is the
next discriminating measurement, not a result of this round. If it is inert,
the final wet-mask/update seam is next.

R22-P3 therefore ends with no closing arm and no candidate. Both production
plants are non-vacuous: independent one-ULP U- and V-face thickness changes
make their literal flux non-bit, change the final tendency digest, print
`STATUS PLANT-FIRED`, and exit 1. R22-P4 is confirmed.

## 3. Landing and blast radius

R22-P5 is not entered because local production closure is mandatory before a
shared-card trajectory gate. The new modes are reachable only through the
private write-only diagnostic hook. The default remains the prior vectorized
expression, and the direct unit test proves both default identity and execution
of all seventeen source-round boundaries in the final arm. Production physics
and every certified registry are unchanged.

The certified GYRE values remain:

| row | certified value/status |
|---|---:|
| kt2 T / S | `6.054357687161262e-16 / 5.786374251651969e-16`, AT-BAR |
| kt2 U / V | `8.326672684688674e-17 / 9.71445146547012e-17`, AT-BAR |
| kt3 T / S | `5.861944241472192e-10 / 1.6244926507906096e-11`, DEBT |
| day 30 T rms | `2.3432437414839976e-06 K` |
| day 240 T rms | `6.5817049818294640e-05 K` |
| day 360 T rms | `5.4077372201617810e-05 K` |

DINO shares the production operator, so its mandatory month gate ran in a
fresh directory despite the private-only selector. It reports
`2.053801168e-03 K` against the `2.244317642e-03 K` bar: **PASS**, unchanged at
the printed precision. LOCK_EXCHANGE, OVERFLOW, the generic GYRE recipe, both
SMT cards, all six flat VORTEX cards, the certified GYRE ladder/year and ORCA2
cannot select the private modes and are not re-baselined.

For ORCA2, the regular/deepest update is **UNMEASURED** on its own developed
state. Its rung-0 ISO/MSC path shares the cited statements. The exact merge
pointer is to inject the stored `r1_e1e2t` into the production closure after
the exact-flux arm; source-ordering the three flux differences is exonerated.

**UNASKED list: EMPTY.** No physics, configuration, default, threshold,
record source, state, or stabiliser choice was made.

## 4. Tests, citations, and review

The final focused suite reports:

> TEST_SUMMARY_PENDING

The round citation gate and its shifted-citation plant are recorded under the
round evidence directory. The cumulative default receipt gate is also required
to pass because the round edited a model file.

The required read-only Codex review was attempted after the diff and evidence
were complete. Its complete verdict transcript is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**; there is no reviewer
`DO NOT SHIP` verdict. No physics candidate lands.

## 5. OPEN

Round 235 stays in the same compiled statement and uses the existing Round-228
record; no acquisition is needed. Starting from the exact-flux plus exact-live-
thickness arm, inject NEMO's recorded `r1_e1e2t` as a production-JIT operand
and compare it with a source-rounded locally reconstructed reciprocal. Require
the RHS to reach the `2.9778502051908996e-22 K s-1` floor. If both are inert,
split the final wet-mask multiply from the higher-level Krhs update seam. Do not
start SMT-4 while this same-stage boundary remains owned.

No NEMO acquisition and no configuration decision is required.
