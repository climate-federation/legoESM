# ORCA2 round 252: direct slow-V mask boundary

Date: 2026-10-12.  This is an ocean-only ORCA2 hierarchy round.  SI3 and the
ORCA2 card's `unmeasured_features` are unchanged.

## Claim labels and frozen question

Every number below is reported separately as **independent** (the OMT-4
card's own initial state) or **given NEMO's entry** (the admitted bridge).
The committed preregistration is
`PREREG_nemo_testcases_l4_orca2_round252.md`.  It asks for the first non-bit
statement in the direct executing chain from the exact depth mean through
bottom drag, wind and the final two-dimensional Coriolis association on the
35 northern-fold halo V faces.

NEMO first forms the reference-thickness depth mean, calls the bottom-drag
correction, and adds wind in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:198-230`.  The compiled
drag routine adds the V correction without a surface-mask multiply in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1462-1463`.  The
record is the admitted OMT-4 round-222 pair; both ranks and both twins carry
exact defined bytes.

## Direct source-order split

The committed round-252 gate ran production JIT, CPU, fp64/libm with no
incoming, depth-mean, or drag override.  It reproduced the same table for
both claim labels:

| boundary / operand | unequal on the 35 faces | maximum |
|---|---:|---:|
| depth mean | 0 | 0 |
| live reciprocal | 13 | 2.9717310957946257e-3 |
| drag coefficient, bottom V, barotropic V, residual, increment | 0 | 0 |
| unmasked post-drag addition | 0 | 0 |
| compact surface V mask | 35 | 1 |
| observed post-drag before | 35 | 1.6557659420864476e-6 m s-2 |
| resolved NEMO surface V mask | 0 | 0 |
| observed post-drag after | 15 | 2.117582368135751e-22 m s-2 |
| observed final slow V after | 15 | 2.117582368135751e-22 m s-2 |

There are zero changes outside the registered 35-face support.  The compact
mask reproduces the old observed value bit-for-bit; the unmasked/resolved
offline replay reproduces NEMO bit-for-bit.  Thus the first non-bit executing
statement was the extra compact surface-mask association after bottom drag.
The implementation reuses the existing D114 resolved operand builder and
keeps NEMO's literal unmasked addition on the measured fold unit; it creates
no second mask or mesh builder.

R252-P1 and P2 are **CONFIRMED**.  R252-P3's prediction of production
bit-exact closure is **REFUTED and retained**: production is at the fixed
floor but has the explicit 15-cell last-bit census above.  The classification
is `AT_BAR_R252_RESOLVED_MASK_CANDIDATE`, not bit exact.  Decision 96 permits
this recorded near-zero ratchet only if every trajectory predicate below is
green.

## Trajectory disposition

The production statement moves no scored trajectory row:

| card / label | result against the pre-round artifact |
|---|---|
| OMT-4 kt=1..10, independent | 0/200 rows moved; no exact loss; first debt unchanged |
| OMT-4 kt=1..10, given NEMO's entry | 0/200 rows moved; no exact loss; first debt unchanged |
| hierarchy rung 0 kt=1..10, independent | 0/200 rows moved; no exact loss; first debt unchanged |
| shipped rung 10 kt=1..10, given NEMO's entry | numerical tree byte-identical; provenance only moved |

The OMT-4 artifact SHA-256 is
`d486a79cf8e6c5cc9c01f0f99c6386df85e05eaf39e3dbc19bcad16bea8cf167`;
rung 0 is
`cf3ec7d832c9d461e034ffe02ceca59d7a84fabc001b75ceb461428351e20257`.
The shipped rung-10 scientific tree has zero non-provenance leaf changes.

GYRE's 210-row residual archive remains byte-identical at SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
All 30 fresh daily snapshots are byte-identical to the certified pre-round
member; day 30 remains
`3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
The focused ORCA2/DINO/tank/OVERFLOW/card battery passes 74/74.

The first DINO invocation exposed and stopped an unbound ORCA-only selector
on the DINO path before step 1.  The review fix initializes that selector to
false outside NEMO's literal ORCA depth branch.  This failed run is retained
as an instrumented regression finding rather than quoted as a science score.
The final current-tip DINO result is recorded in the gate section below.

## Held complete unit

With the complete round-217 fold/transport unit, the raw V surface-mask plus
T-pivot association, the resolved three-dimensional V mask, and both held
external-mode associations enabled, independent OMT-4 completes kt=1..7 and
still refuses at kt=8 on `raw-mesh e3w_int must contain only finite values >
0`.  Therefore this mask statement is necessary but not the complete unit's
final compensating partner.  The private unit remains **HELD** and no held
association is promoted in this round.  R252-P4's predicted kt=8 completion
is **REFUTED**.

## Gates, tests and review

The final current-tip DINO from-rest month completed 960 steps and passed:
day-30 wet three-dimensional T RMS is `2.056821682e-03 K` against the
`2.244317642e-03 K` bar.  Its snapshot is byte-identical to round 238
(`d2e254e5e9d3f9dbddc53c4c9676ea2d992af232fda58ac51cfc1710cbd46b55`).
The log SHA-256 is
`b82812a42350d50c60667a31da41daedfbe0339da7052c210b3d36f12869419c`.

The focused scientific battery passed 74/74.  The required once-only
`tests/ocean/fidelity -n 12` battery collected 3,282 tests and reached 98%
with 3,211 explicit PASS lines before its last xdist worker failed to return
a terminal summary.  It printed exactly four red node IDs.  All four were
re-run serially, in isolation, and reproduced their registered pre-existing
conditions:

1. SI3 scalar-math `MY_SRC` provenance;
2. the GYRE round-129 certified-year harness stamp;
3. the round-35 process-global allow-dirty scoping ratchet; and
4. the report-emitter worktree-stamp ratchet.

No round-252 test failed.  The once-only and isolated logs have SHA-256
`7c4fdd11ec6202ab5104813ac843713227274fc94119e3a179deec3031482b4f`
and `57e397c8bd4199ace73dd48b7c56b1a71ec9662e330895a2179bfdfbbc31ce29`,
respectively.  Per the one-battery rule, the full parallel battery was not
re-run.

Both the round receipt and the cumulative default receipt pass the citation
gate with no unmapped citation; the deliberate citation plant refuses.

The required independent command was attempted as `codex exec --sandbox
read-only`.  It exited before review because its in-process app-server client
could not initialize on the read-only filesystem.  Therefore **independent
review unavailable in-sandbox**.  The retained review-attempt log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

Resume from the now-source-ordered direct slow-V chain.  With the resolved
mask statement in production, re-run the kt=7-to-8 passive boundary table for
the complete private unit and name the first downstream statement that makes
live W thickness invalid.  The unit and the two held associations remain
private; do not attribute the kt=8 damage to the now-at-bar post-drag row.
