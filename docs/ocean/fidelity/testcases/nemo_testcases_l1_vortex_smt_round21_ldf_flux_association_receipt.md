# Receipt — VORTEX_SMT round 21 (lane round 233): horizontal LDF flux association

**Status: HELD.** No physics, card, carried state, trajectory, or certified
number changed. Bare transcription of NEMO's pairwise parenthesisation is inert
under the production JIT. Retaining each compiled source-operation boundary
makes both horizontal fluxes bit-exact, but the consumed LDF RHS remains
non-bit on 8,613 cells with the same
`2.5978639524806749e-14 K s-1` maximum. The next owned boundary is therefore
the downstream flux-divergence/update statement, not the horizontal-flux
assignment. No trajectory landing gate is eligible.

Base: `59639d281` (round 232). Preregistration: `f04c4183d`. Pairwise
instrument: `aa133ba85`; literal discriminator: `b817bd352`; production plants:
`26db3c89a`; unit coverage: `dd0101710`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round233/`.

## 1. Compiled order and frozen controls

NEMO forms A11/A22 from the live Kmm U/V face thicknesses, then forms the
closed-bottom mask factors and A13/A23, in
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`.
The U/V horizontal-flux assignments, including their pairwise four-point
vertical-gradient sums and the enclosing multiply/add order, are
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259`.
NEMO consumes those fluxes in the regular and deepest-level divergence
updates at `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`
and `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.

Both production-JIT measurements reproduce the Round-232 controls:

| production-JIT arm | zfu unequal / max abs | zfv unequal / max abs | RHS unequal / max abs (`K s-1`) |
|---|---:|---:|---:|
| ordinary production | not substituted | not substituted | `8,641 / 7.2621621143171395e-11` |
| mask + live T divisor + live U/V faces | `1,757 / 4.5474735088646412e-12` | `1,767 / 3.6379788070917130e-12` | `8,619 / 2.5978639524806749e-14` |
| pairwise four-point sum | `1,757 / 4.5474735088646412e-12` | `1,767 / 3.6379788070917130e-12` | `8,619 / 2.5978639524806749e-14` |
| literal compiled boundaries | `0 / 0` | `0 / 0` | `8,613 / 2.5978639524806749e-14` |
| recorded fluxes + live T divisor, post-hoc floor | `0 / 0` | `0 / 0` | `8,451 / 2.9778502051908996e-22` |

The aggregate pre-LDF, post-LDF and additional-LDF maxima reproduce
`7.418332614861356e-11`, `2.0915088416728622e-07`, and
`2.090767008411376e-07 K`. All A11/A22/A13/A23, dit/djt/dkt, and vertical-flux
control rows remain bit-exact. R21-P1 is confirmed.

## 2. Discriminating result and retraction

R21-P2 predicted that NEMO's pairwise four-point association would close zfu,
zfv and the RHS. It is **REFUTED**: the pairwise arm is bit-for-bit identical
to Round 232's complete live-face arm. Production XLA reassociates the bare
parentheses.

The preregistered P4 split then retained the compiled operation boundaries
around the diagonal product, four-point sum, cross product, inner sum and
outer coefficient multiply. This is a private measurement arm through the
full production JIT step. It closes both fluxes exactly:

| source row | unequal cells | maximum absolute | verdict |
|---|---:|---:|---|
| hmsku / hmskv | `0 / 0` | `0 / 0` | BIT |
| A11 / A22 | `0 / 0` | `0 / 0` | BIT |
| A13 / A23 | `0 / 0` | `0 / 0` | BIT |
| zfu / zfv | `0 / 0` | `0 / 0` | BIT |
| dit / djt / dkt | `0 / 0 / 0` | `0 / 0 / 0` | BIT |
| vertical flux below / above | `0 / 0` | `0 / 0` | BIT |
| LDF RHS | `8,613` | `2.5978639524806749e-14 K s-1` | non-bit |

The exact flux result proves the source assignment, but it does **not** improve
the consumed RHS magnitude. Accordingly the stronger P2 closure claim remains
refuted, and this exact local statement is not a landing candidate. The first
owned non-bit boundary after the now-exact fluxes is NEMO's source-ordered
divergence/update at compiled lines 306-310, with its deepest-level form at
327-331. No mechanism inside that statement is claimed yet.

Both source-input controls are non-vacuous under the production step. A
one-ULP U-face change makes the literal zfu and RHS rows non-bit; an independent
V-face change does the same for zfv and the RHS. Both runs print
`STATUS PLANT-FIRED` and exit 1. R21-P3 is confirmed.

## 3. Landing and blast radius

R21-P5 is not entered because the local RHS did not close. The literal
evaluation remains reachable only through a private diagnostic override; the
default is explicitly pinned to the prior vectorized expression, and the unit
test proves both the default identity and literal-branch execution. Production
configuration and behavior therefore remain unchanged.

The certified GYRE values remain:

| row | certified value/status |
|---|---:|
| kt2 T / S | `6.054357687161262e-16 / 5.786374251651969e-16`, AT-BAR |
| kt2 U / V | `8.326672684688674e-17 / 9.71445146547012e-17`, AT-BAR |
| kt3 T / S | `5.861944241472192e-10 / 1.6244926507906096e-11`, DEBT |
| day 30 T rms | `2.3432437414839976e-06 K` |
| day 240 T rms | `6.5817049818294640e-05 K` |
| day 360 T rms | `5.4077372201617810e-05 K` |

DINO shares the production operator. Despite the private-only change, its
mandatory month gate was run in a fresh work directory and reports
`2.053801168e-03 K` against the `2.244317642e-03 K` bar: **PASS**, unchanged at
the printed precision.
LOCK_EXCHANGE, OVERFLOW, the generic GYRE recipe, the eight VORTEX registries,
the certified GYRE ladder/year, and ORCA2 cannot select this private arm and
are not re-baselined.

For ORCA2, the result is **UNMEASURED** on its own developed state. Its rung-0
ISO/MSC path shares compiled lines 306-310 and 327-331. The precise merge
pointer is to test the source-ordered horizontal-plus-vertical flux divergence
and its enclosing live-volume division after importing the exact-flux arm;
pairwise horizontal-flux parentheses alone are exonerated.

**UNASKED list: EMPTY.** No physics, configuration, default, threshold,
record source, state, or stabiliser choice was made.

## 4. Tests, citations, and review

The final focused suite reports:

> 138 passed in 197.63s (0:03:17)

It covers the complete receipt-citation suite, every SMT-card test, and the
complete GM/Redi unit file. The round citation gate finds four mapped
compiled-source citations, zero failures, zero unmapped citations, and zero
map-audit failures. The cumulative default receipt gate also passes with 274
citations and zero unmapped citations after the required old-to-new
`SequenceMatcher` re-anchor. Shifting the horizontal-flux citation by two lines
makes the gate print `SYMBOL-NOT-AT-LINE` and exit 1.

The required read-only Codex review was attempted and could not initialize in
the sandbox. Its complete verdict transcript is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**; there is no reviewer
`DO NOT SHIP` verdict. No physics candidate lands.

## 5. OPEN

Round 234 stays in the same compiled `traldf_iso` stage and uses the existing
Round-228 record; no acquisition is needed. Starting from the exact literal
zfu/zfv arm, transcribe the source-ordered regular divergence at compiled lines
306-310 and the deepest-level form at 327-331 through the production JIT. Split
the horizontal differences, their sum with the vertical difference, and the
enclosing area/live-thickness factor one variable at a time. Require the RHS to
reach the `2.9778502051908996e-22 K s-1` recorded-input floor before any shared
landing gate. Do not start SMT-4 while this same-stage boundary remains owned.

No NEMO acquisition and no configuration decision is required.
