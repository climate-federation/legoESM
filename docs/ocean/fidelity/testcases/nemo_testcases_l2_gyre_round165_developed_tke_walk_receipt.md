# Round 165 receipt — developed-state TKE statement walk

**Status: HELD.** No production physics, card, carried state, or configuration
changed.  The admitted day-180 production-JIT walk makes the TKE entry,
surface boundary, Langmuir boundary, and all three matrix rows bit-exact from
NEMO's recorded inputs.  The first non-bit compiled statement is the TKE RHS:
14,649/18,000 cells differ, max `3.053494349730679e-11` m2/s2.  Its first
non-bit source-order operand is already `p_sh2`: 17,400/17,400 wet solved
interfaces differ, max `2.1204821986655657e-15` s-2.  The RHS transcription is
therefore not a candidate; the walk continues upstream through the developed
shear producer.

Preregistration is commit `ae1aec23a`.  The frozen record-layout prediction
was partly refuted and is retained below.  The authoritative scientific run
was made from clean commit `d0ecc72086532fd945c9708af9ee8513689a3fef` and is
`phase3/round165/developed_tke_walk.json`; evidence and logs are under
`phase3/round165/`.

## Acquisition admission

The operator's NEMO run completed normally and wrote both requested records,
but its shell post-check compared their padded 16-byte magic to an unpadded
15-byte value and refused them.  Reading the compiled writers settled the
layout rather than guessing: the operand writer emits its magic and thirteen
header integers at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r54_tke.f90:128-130`, and
the statement writer does the same at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:65-67`.

The corrected fail-closed admission reused the existing binary and reports:

- step-1080 restart byte-identical to the uninstrumented Round-132 restart;
- operand record 4,546,636 bytes, SHA-256
  `d72042cb7979ac42859e38ba83bd598c2268c96f8ac7dac89faf8cf4308799ad`;
- statement record 873,028 bytes, SHA-256
  `0261d268ab244e692f9a465c1a423563a7a1ebd566d841b5cfc9b08115926ed9`;
- producer commit `9f53be73a948fea4804f127dbf9d9a68c0f2899a` and both record stamps valid.

The preregistered `(Kbb,Kmm)=(3,3)` prediction is **REFUTED**: both compiled
headers contain `(1,1)`.  The preregistered claim that both headers advertise
32x22 global extents is also **REFUTED**: the operand writer records 32x22,
whereas the statement writer records global 36x26 plus owned bounds
`i=3:34,j=3:24`.  Those are producer facts, not reader choices.  The corrected
admission checks each exact tuple independently.  Its wrong-magic plant exits
1 with a named `REFUSE`; no NEMO rerun was needed.

## Given-entry production walk

The bridge loads the admitted step-1080 restart, the recorded stage-1 `ssha`,
and the record's own `taum_entry`.  Supplying the recorded stress modulus is
essential to a *given-NEMO-entry* proof: a preliminary diagnostic run using a
stress reconstructed from components produced 188 boundary mismatches and is
not the scientific arm.  With the recorded operand, those 188 cells return to
BIT.

The execution path is `LatLonCGridOceanModel.step` through its production JIT.
The full RK3 step makes four mixing-length calls: two consume the traced entry
energy and two consume the traced post-sweep energy.  The instrument selects
the final closure by a bitwise match of its consumed energy to
`en_post_sweep[...,1:]`; the two matching outputs are bit-identical.  It never
selects by callback order.  The observer and ordinary production states have
zero unequal bytes.

| compiled-order boundary | cells unequal / scored | max abs | verdict |
|---|---:|---:|---|
| entry `en` | 0 / 17,400 | 0 | BIT |
| surface/bottom boundary | 0 / 18,000 | 0 | BIT |
| after Langmuir | 0 / 18,000 | 0 | BIT |
| matrix upper | 0 / 17,400 | 0 | BIT |
| matrix lower | 0 / 17,400 | 0 | BIT |
| matrix diagonal | 0 / 17,400 | 0 | BIT |
| RHS before sweep | 14,649 / 18,000 | `3.053494349730679e-11` | FIRST NON-BIT |
| energy after sweep/floor | 2,342 / 18,000 | `4.643221571121181e-13` | non-bit |

The active surface assignment is compiled at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:284`; the active
Langmuir energy write is at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:387`; and the three
matrix writes are
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:433-435`.  Their BIT
rows establish the compiled order before the RHS, rather than inferring it
from the final closure.

The first non-bit statement is exactly NEMO's RHS sum at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:438-441`.  The
following recurrences and energy floor are compiled at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:475-492`, so their
non-bit output is downstream and not a new owner.

## Operand and closure localization

| source-order RHS/carry operand | cells unequal / scored | max abs | verdict |
|---|---:|---:|---|
| entry energy | 0 / 17,400 | 0 | BIT |
| entry `avm` | 0 / 17,400 | 0 | BIT |
| entry `dissl` | 0 / 17,400 | 0 | BIT |
| `p_sh2` | 17,400 / 17,400 | `2.1204821986655657e-15` | FIRST NON-BIT |
| entry `avt` | 0 / 17,400 | 0 | BIT |
| `rn2` | 16,394 / 17,400 | `1.8431436932253575e-18` | later non-bit |

The source order is literal in the RHS citation above: shear precedes the
`avt*rn2` and dissipation terms.  The magnitude check is also consistent:
`rn_Dt * max|delta p_sh2| = 3.0534943660784147e-11`, versus the observed RHS
maximum `3.053494349730679e-11` (ratio `0.9999999946462204`).  This is a
magnitude bound, not a claim that the smaller `rn2` difference is zero.

The downstream closure remains non-bit as expected: momentum and dissipation
mixing lengths differ in 15,106 and 16,024 cells (both max
`4.828905275644502e-09` m); `avm`, `avt`, and `dissl` differ in 6,563, 6,461,
and 15,886 cells, with maxima `1.578432523574591e-11`,
`1.5940453956808653e-09`, and `4.382728988755469e-13`.  NEMO's final
mixing-length merge is
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:689-692`, its
viscosity/diffusivity/dissipation writes are
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:700-705`, and its
Prandtl update to `avt` is
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:711`.

The frozen prediction that entry, boundary, and Langmuir rows are BIT is
**CONFIRMED**.  The frozen prediction that the RHS is first non-bit and that
`p_sh2` is its first non-bit operand is **CONFIRMED**.  No RHS arithmetic patch
was formed: replacing a correct statement while its first operand is wrong
would be a stabilizer/cancellation, not a NEMO transcription.

## Landing and certified trajectory rows

There is no one-variable candidate this round, so the Decision-43/45/55
trajectory gate is **NOT RUN**.  No certified row moved.  The immutable
Round-163 production headline remains:

| certified row | value | status |
|---|---:|---|
| kt2 T | `1.4210854715202004e-14` K | AT-BAR |
| kt2 S | `2.1316282072803006e-14` | AT-BAR |
| kt2 U | `8.326672684688674e-17` m/s | AT-BAR |
| kt2 V | `9.714451465470120e-17` m/s | AT-BAR |
| kt3 T | `4.940071072212504e-07` K | DEBT |
| kt3 S | `4.008629872487290e-08` | DEBT |
| day-30 T rms | `6.572574374770603e-05` K | immutable after arm |
| day-240 T rms | `1.644836070117868e-02` K | immutable after arm |
| day-360 T rms | `1.122566001855131e-02` K | immutable after arm |

With no shared implementation change there is no DINO, generic-GYRE, tank,
or ORCA2 trajectory movement to score.  ORCA2 remains unmeasured under its
pending explicit stage-solve decision; this round did not answer it.

## Controls, tests, citations, and review

The production entry-ULP plant changes one nonzero TKE cell and exits 1 with
`STATUS PLANT-FIRED`; it moves the registered entry and boundary rows.  The
record wrong-magic plant also exits nonzero with a named refusal.  The focused
parser/stage/year-owner battery reports:

> 60 passed in 33.92s

FINAL_TEST_RESULTS_PLACEHOLDER

FINAL_CITATION_RESULTS_PLACEHOLDER

FINAL_REVIEW_RESULTS_PLACEHOLDER

## OPEN — Round 166

1. Stay at the admitted day-180 entry and walk the `p_sh2` producer in
   compiled `zdf_sh2` order under the production JIT.  Extend the existing
   shear walk; do not build a second bridge.  Score its velocity differences,
   live Kmm thickness/divisors, masks, and face-to-T averaging separately.
2. Name the first non-bit shear statement.  Re-evaluate the held Round-105
   routing split only if this developed-state walk reaches that exact
   statement and its one-variable production proof closes.
3. A candidate lands only through the full Decision-43/45 gate as amended by
   Decision 55.  Otherwise remain HELD with the upstream statement and its
   day-240 magnitude relevance.  Do not act on pending Decisions 53, 54, 57,
   or 58.
