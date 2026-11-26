# ORCA2 round 265 — matched stage-2 U RHS boundary

Date: 2026-10-13. Base: `3ca3e4d14`. Preregistration: `aba133100`.
Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry at
its tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, on CPU fp64/libm production JIT. No NEMO acquisition was needed.

## Compiled order and admitted record

The scalar stage ratios are built, associated with scalar sign `+1`, and
stored in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178`.
For stage 2, the compiled vector-invariant branch accumulates the completed
RHS in EOS, HPG, VOR, and ADV order, then performs the velocity update in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-424`.

The admitted record has two distinct streams. The composition stream records
the post-update alias, while the pre-update stream writes `Krhs` before
`Krhs == Kaa` is overwritten; the compiled recorder states that reason and
writes U/V at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:443-455`.
The generic reader now accepts the record's explicit ORCA2 dimensions rather
than hard-coding GYRE dimensions. Its header, dimensions, finiteness, payload
length, and SHA-256 identity are gated.

## Frozen predictions and result

R265-P1 is **CONFIRMED** under both labels. The source-complete private unit
reproduces round 264: matched stage-2 fold-band T is
`2.220446049250313e-16 K`, S is `7.105427357601002e-15 PSU`, and the no-observer
prefix completes kt=1 and kt=2 with all 18 carried arrays finite.

R265-P2 is **CONFIRMED at the boundary level** under both labels. In compiled
order, Kbb U is bit-exact. Kmm U differs only below the campaign floor
(`1.734723475976807e-18 m/s`). The completed pre-update `Krhs_u` is the first
valued operand: fold-band maximum `1.706908474056055e-6 m/s2`, RMS
`1.5162263749041033e-7 m/s2`, 2,699/8,100 unequal, argmax `[j=2,i=38,k=14]`.
The raw updated U then has maximum `0.009217305759902698 m/s`; the matched
post-association completed-U boundary remains
`0.003382014663707235 m/s`.

Substituting the admitted pre-update RHS makes raw Kaa U array-exact
(0/399,600 unequal). The corrected and post-associated completed U remain
within the floor: maximum `2.7755575615628914e-17 m/s`, RMS
`6.422926018364848e-19 m/s`, with only last-bit differences. Independent and
given-entry results are identical for every number above.

R265-P3 is **PENDING**, not confirmed: the completed RHS is a multi-operator
boundary, not a leaf NEMO statement. The gate deliberately reports
`PENDING_RHS_OPERATOR_SPLIT_BOUNDARY_CLOSES`; it cannot call the RHS an owner.
R265-P4 is therefore **NOT REACHED** and no trajectory or Decision-96 landing
vote is taken. R265-P5 is **CONFIRMED**: all eleven planted violations refuse,
covering label/prefix coverage, record identity, passivity, round-264
reproduction, source order, first-boundary sensitivity, RHS closure, matched
program point, prefix finiteness, and one-bit sensitivity.

## Instrument corrections retained

Two in-round instrument failures are retained and excluded from claims.
First, the initial private integration associated `r3va` but omitted the
resolved live V-mask half of round 264's coupled context; it moved stage-2 S
to `1.793867 PSU` and T to `0.096881 K`, and the classifier refused. Second,
the first RHS replay read the aliased composition `Krhs`, producing an
impossible `2452.8349769 m/s` raw-U residual. Reading the dedicated pre-update
stream fixes that record-semantics error and yields the exact closure above.
The package arm and temporary helpers are retired; both touched model files
are byte-identical to base `3ca3e4d14`.

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round265/`.
Principal SHA-256 values are:

- final classification: `0124c468d263aa97103cc2c207d3e9bbbe37dbe74a7bc50aa26a85bfe1aa6368`;
- independent measurement: `875e89a1acedf0bedf797a1eeffa35a18f46a0b15da9d9c0b571f217e7b527ef`;
- given-entry measurement: `ac04d0dc7ccf6d1de398483c92b7ade13bd47a6e7cfed66775c9b12311affba3`;
- independent / given finite-prefix controls:
  `865ffac33de658d611c8b62bd63d87cc364f725f5ed6ff9250d0e9a6339ab1c3` /
  `69808a031a5c30b4af12f4c7c94446b703ec986740fd5c7c807937d5e480e797`.

The focused round-260 through round-265 plus citation set passes 104/104; its
log SHA-256 is
`86c26f0ed92de6f817ce3122744fa1b07358f63291d587681f1289b732329b48`.
The once-only `tests/ocean/fidelity -n 12` battery collected 3,446 tests and
reached 99%. All fourteen round-265 tests passed. It emitted the four
registered pre-existing reds—GYRE round-129 record-backed harness pin,
allow-dirty scope, worktree stamp, and SI3 scalar-math provenance—then waited
indefinitely on the LOCK planted-stage end-to-end node and was terminated.
That exact hung node passed serially in 607.87 s; the worktree-stamp ratchet
reproduced with the existing 13-file offender set. Battery and serial
disposition SHA-256 values are
`59b4de63ca7c5d00c7027e59db4506b7bd44d95258b3a95b4637b2b9bee1038d`
and `255264955b907cef22b316aaea1a46b76d2b693a916920ff4067184042d998f7`.

The default cumulative citation gate and this receipt's gate pass with zero
failures and no unmapped citations; the planted two-line shift of the
pre-update citation fails. Their JSON SHA-256 values are
`befc59c7c4628b332206f16060ad5968dc374991179466433506e3e727f9829b`,
`18f675c23339e71e25f9b62000926cd33e9570b2db99dca6b1227f670d840d27`,
and `94c3490bbac1d1adfc8592e2bcb7e0d0e4a79b5ea336b59292b17f9dc3905abb`.

The requested separate `codex exec --sandbox read-only` review failed before
review because its in-process app server could not initialize on the
read-only filesystem. **Independent review unavailable in-sandbox.** Its log
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

Split the admitted completed stage-2 RHS offline, at matched program points,
in the compiled sequence recorded in `oracle_rkstage2_terms_kt00000001.bin`:
entry accumulator, EOS/HPG, VOR, then vector-invariant ADV. The existing HPG
operand and literal streams are also admitted, so no acquisition is required.
Name the first accumulator leaving the floor, then walk only that operator's
operands to a leaf compiled statement. Keep the complete fold unit private;
only a cited leaf transcription that closes the RHS reaches OMT-4 kt=1..10
and the full Decision-96 gates.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
