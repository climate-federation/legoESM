# ORCA2 round 267 — source-order retraction and vorticity boundary

Date: 2026-10-13. Base: `a864b99f4`. Preregistration: `4ea260891`.
Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried state,
stabiliser, sea-ice declaration, or `unmeasured_features` entry at its final
tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, using CPU fp64/libm production JIT and the already admitted round-222
record. No NEMO acquisition was needed.

## Loud retraction of round 266

Round 266's claim that the HPG accumulator assignment was the first non-bit
statement is **RETRACTED**. The round-266 executable observer exposed a
source-order option, but its measurement driver did not select that option.
Consequently, the field labelled `after_hpg_u` was the older combined
KE-plus-HPG diagnostic bucket, not NEMO's accumulator immediately after HPG.
The initial round-267 run faithfully reproduced that invalid comparison and
is retained as `*_refuted_missing_source_order.{json,log}`. It cannot support
a physics claim.

The measurement driver now requires and reports the source-order arm. The
gate refuses a report without that stamp, and the corrected output no longer
prints the round-266 HPG-owner verdict. This is the first-class tool retraction
required by the campaign, not a prose-only correction.

## Compiled program and corrected boundary

For stages 2 and 3, the compiled vector-form branch calls HPG first and dumps
the accumulator immediately afterward at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:372-405`.
It then calls vorticity and records the next cumulative boundary at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:407-421`.
Inside HPG, NEMO assigns the literal surface and interior sums directly at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:401-402` and
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:426-427`.

The corrected source-order replay is identical under both labels. HPG is
AT-BAR: maximum `1.0524384369634682e-19 m/s2`, RMS
`1.4037726890957292e-21 m/s2`, with `22,817/399,600` bit-unequal slots.
Its literal self-replay is array-exact. The private proposed arm that removed
the candidate barrier changes no value and is therefore unnecessary and
refuted.

The first valued cumulative boundary is now **after vorticity**, exactly at
the second call/dump above: maximum `1.7082229745349855e-6 m/s2`, RMS
`2.1588341793822282e-8 m/s2`, `249,101/399,600` unequal slots, argmax
`[j=147,i=38,k=14]`. Replaying the candidate accumulation from its own HPG
and vorticity term closes to `3.3881317890172014e-21 m/s2`; therefore the
accumulation itself is faithful and the vorticity operator value is the next
owner class. This round names a recorded operator boundary, not a leaf
statement inside `dyn_vor`; the leaf attribution remains OPEN.

The completed stage-2 RHS remains non-bit at maximum
`1.706908474056055e-6 m/s2`. The matched stage-2 tracer state remains within
the campaign floor under both labels: T maximum
`2.220446049250313e-16 K` and S maximum
`7.105427357601002e-15 PSU`. Independent T has 53 bit-unequal cells;
given-entry T has 10,884; both S rows have seven.

## Frozen predictions and controls

- R267-P1 is **REFUTED**: the corrected instrument retracts round 266's
  source-order claim.
- R267-P2 is **REFUTED**: baseline HPG already closes and the proposed carry
  arm is bit-inert.
- R267-P3 is **REFUTED**: the next boundary is VOR, not KEG.
- R267-P4 is **HELD**: the completed RHS does not close.
- R267-P5 is **CONFIRMED**.

All nine planted violations refuse: missing label, record identity,
passivity, missing source-order stamp, HPG carry, next-boundary selection,
completed-RHS closure, baseline reproduction, and one-bit sensitivity. The
false initial outputs remain in evidence with their failed interpretation;
they were not overwritten.

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round267/`.
Principal SHA-256 values are:

- classification: `3d4c9f89af15318450dbef0540833c1b80beaa485bd464bc351323054c44151d`;
- independent measurement: `d07a90f3bef7dadad7a839d0d1725cad3edd9e1d1bf22226c8bed4275a3f6020`;
- given-entry measurement: `1aba1e8727db2dc748ba3857934cb39e81adde740f31860c83593ba3ec0d46b0`;
- refuted independent run: `b7c1706bbaed158dd5e1219de260a6b430c9bf83e2325ba0c089e00938180d60`;
- refuted given-entry run: `2ea93ab09de7e0b70320e19d0c095012f69c81581bd4d77061495219073126ca`.

The round-266/267 focused gate battery passes `20/20`. The private HPG and
matched-stage-ratio hooks were used only for measurement and then retired;
the final production package files are byte-identical to base. Because no
production statement lands, ORCA2 trajectory and shared GYRE/DINO/tank gates
are not invoked as landing evidence.

The cumulative default citation audit and this receipt's four-citation audit
both pass with zero unmapped citations, map-audit failures, or receipt
failures. The planted two-line shift of the first compiled-stage citation
refuses with exit 1. The separate `codex exec --sandbox read-only` review was
attempted, but its in-process app-server client cannot initialise because the
read-only sandbox denies its required write; **independent review unavailable
in-sandbox**. The diagnostics are retained in `citation_gate*.json` and
`codex_readonly_review.log`.

The required once-only `tests/ocean/fidelity -n 12` battery reached 91%
before the pytest controller vanished without a summary, with 3,166 passed,
seven skipped, and four named reds. All four reproduce serially: the known
GYRE-year harness-hash ratchet; a source-layout plant preflight that sees the
round receipt as an uncommitted path; thirteen legacy unscoped allow-dirty
drivers; and thirteen legacy unstamped report emitters. The latter three
classes are tree/process ratchets rather than round-267 scientific failures;
the source-layout plant is rechecked from the committed receipt tip below.
No round-267 test is red.

## OPEN

At the same matched stage-2 program point, walk the executed EEN vorticity
operator in compiled source order from the existing passive stage operands.
Split its planetary and relative-vorticity contributions and their live EEN
coefficients one variable at a time, requiring the candidate-side replay to
reproduce the corrected cumulative VOR row before naming a leaf. Keep both
labels separate. Do not revisit HPG or attribute the downstream ADV row.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
