# Preregistration: NEMO-testcases L2 GYRE round 75 advective-mean layout repair

Date: 2026-09-12. Frozen after the operator-reported round-74 record-size
refusal and inspection of the failed record's byte count and compiled writer,
before editing or exercising the replacement acquisition card.

## Boundary and immutable failure

Round 73's first non-bit statement remains the kt=2 stage-1 `un_adv` input:
580 of 580 wet U cells differ, with maximum absolute difference
`0.00012029895814569258`. Round 74 repaired the acquisition preflight but the
operator's full run refused at the record-size gate. The failed record is
3,041,176 bytes, 748,800 bytes below the declared 3,789,976-byte layout.

The compiled round-74 writer emits ten named two-dimensional arrays per each
of 50 external substeps at
`GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:581-584`.
The two velocity slots are `l2_u_mid` and `l2_v_mid`, but the kt=2 opening block
allocates only `l2_adv_before_u` and `l2_adv_before_v` at the same compiled
source's `:442-450`. The velocity snapshots are allocated and populated only
inside the kt=1 instrumentation arm at `:390-393` and `:503-506`. Therefore
the kt=2 stream writes zero bytes for those two unallocated allocatables.
The deficit is exactly `50 * 2 * 36 * 26 * 8 = 748800` bytes. This is an
acquisition-instrument defect, not a scientific result.

## Frozen repair and falsifiers

Prepare a new acquisition target `GYRE_OMIP_L2_P3_SM_R75ADV3` and evidence
directory `round75/oracle_advmean_kt2`; neither may already exist. Keep the
R72 source configuration, namelists, compiled model statements, ten-step run,
and all production code unchanged.

In the WRITE-only patch, retain the kt=1 velocity snapshot fields. For kt=2,
write the live allocated `ua_e` and `va_e` arrays into the same two velocity
slots, between `zhU`/`zhV` and `zhup2_e`/`zhvp2_e`. Stream writes have no record
markers, so splitting the kt=1 and kt=2 velocity clauses must preserve the one
header and the reader's existing ten-field order. Do not allocate or carry a
new array and do not change any numerical statement.

Prediction: exact preprocessing followed by `gfortran -fsyntax-only` produces
empty output and exits zero. A static layout proof must identify exactly ten
two-dimensional fields in both the kt=1 and kt=2 substep arms and prove the
declared byte count is 3,789,976. A plant that removes one kt=2 velocity field
must exit nonzero. Shell parsing, Python compilation, and focused acquisition
tests must pass. Any failure records this prediction as REFUTED and no
acquisition card is handed off.

The full operator run must emit exactly 3,789,976 bytes; replay all 100 U/V
entry-plus-increment associations and both normalizations bit-for-bit; retain
the exact header, EOF, stamp, final restart, and mesh mask checks; and make the
stamp, header, truncation, replay-ULP, consumed-field, and layout plants exit
nonzero. The commit stamp must name the clean committed checkout that runs it.

## Source boundary and Rule 12

After acquisition, the next source walk remains the kt=2 U accumulator in
compiled order: entry `un_adv`, `wgtbtp2`, `zhU`, `r1_e2u`, left-associated
increment, and exit `un_adv`, followed only while exact by normalization and
pre/post-boundary values. The active numerical accumulator remains at
`GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:573-579`; this round
does not measure or edit it.

No production statement is eligible. GYRE kt1--10 and days 1--30,
LOCK_EXCHANGE, OVERFLOW, and DINO are UNREACHED because production is
unchanged. DINO retains shared-accumulator and 96--98% cancellation risk.
ORCA2 remains UNMEASURED WITH SPEC: resolve its compiled external-mode card;
record every substep entry, weight, transport, reciprocal metric, exit,
normalization, and boundary handoff for kt1--10; replay in compiled order;
register every moved row; preserve every AT-BAR row; and forbid an earlier
first-over-bar boundary.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. The round-70 patch remains held.
