# NEMO testcase Lane 4 — ORCA2 card round 41 RHS-family score receipt

Date: 2026-09-27

Parent: `512c25ee0e467a0e59d6f7e534212cdf51fd69b6`

Measurement commit: `41883c3c333cd9ec23608cce07be1ed84aed56d9`

Status: **STOPPED_FOR_RECORD — HPG IS THE FIRST NON-BIT STAGE-1 MOMENTUM
OPERATOR, BUT ITS INTERNAL STATEMENT BOUNDARIES ARE NOT RECORDED.**  No
production model file changed.

All scientific results below are **given NEMO's entry**.  No independent
initial-state result is mixed into the tables.  The six sea-ice selectors and
the card's `unmeasured_features` tuple remain unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round41/`.

## Admission and reproduction

The operator-completed Round-40 acquisition admits without repair.  Both
35,434,332-byte per-rank streams retain their admitted digests.  Their
post-ZAD U/V fields close bit-for-bit against the same run's completed RHS on
both ranks; both inherited ranked slow-forcing streams and all four ocean/ice
restarts are byte-identical.  Header, truncation, swapped-rank, final-ULP,
parent-byte, and restart-byte plants all fire.  R41-P1 is **CONFIRMED**.

Round 39 reproduces exactly at the current parent, so R41-P2 is **CONFIRMED**:

| compiled boundary on 64 rank-1 U rows | unequal / scored | maximum |
|---|---:|---:|
| per-level thickness × RHS × mask | 1,754 / 1,920 | `5.321462756514503e-08` |
| first partial sum, level 0 | 64 / 64 | `9.242193862245139e-11` |
| completed 30-level sum | 64 / 64 | `2.721729894586411e-07` |
| completed sum × production reciprocal | 64 / 64 | `7.356587026022005e-18` m/s2 |

Two refused invocations used the phase-2 record and then the Round-20 ranked
record as the inherited boundary root.  Their marker/file censuses correctly
refused.  They produced no scientific output and are excluded.  The admitted
Round-19 pre-exchange root is the exact boundary record used by Rounds 20,
38, and 39; the successful reproduction above uses it.

## First non-bit operator

The executing compiled order is EOS, HPG, LDF, VOR, KEG, and ZAD at
`ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/stp2d.f90:155-187`.
The score uses the production operator's already-computed raw terms and its
existing write-only compiled-order arm.  All ten raw U/V terms are bit-exact
between the ordinary and ordered traces, so the arm changes only association.

On the 64 disputed rank-1 U rows, HPG is first: 1,753 / 1,754 wet layer
values differ, maximum `1.4862887125471208e-17` m/s2.  Every later boundary
has the same count and maximum, and its residual is bit-identical to HPG's
residual cell by cell:

| boundary | unequal / scored | maximum m/s2 | residual vs post-HPG |
|---|---:|---:|---:|
| HPG | 1,753 / 1,754 | `1.4862887125471208e-17` | reference |
| LDF | 1,753 / 1,754 | `1.4862887125471208e-17` | 0 unequal |
| VOR | 1,753 / 1,754 | `1.4862887125471208e-17` | 0 unequal |
| KEG | 1,753 / 1,754 | `1.4862887125471208e-17` | 0 unequal |
| ZAD | 1,753 / 1,754 | `1.4862887125471208e-17` | 0 unequal |

R41-P3 and R41-P4 are **CONFIRMED**.  The full owned-domain walk also names
HPG first on both ranks.  Its first U rows are 226,187 / 226,236 unequal on
rank 0 (maximum `5.930675880746355e-17` m/s2) and 186,746 / 186,794 on rank 1
(maximum `3.963574209646251e-17` m/s2).  The V HPG rows are 226,564 / 226,637
and 188,494 / 188,538 unequal, with maxima `2.362041079622794e-06` and
`3.1384570529614414e-06` m/s2 respectively.  These broader rows are reported,
not substituted; the registered 64-row slow-forcing boundary remains this
walk's scored claim.

NEMO's executing HPG routine builds the along-surface pressure-gradient
accumulator, the s-coordinate correction, and their sum at
`ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/dynhpg.f90:340-451`.
The admitted record has only the post-routine value, so it cannot name which
of those statements first differs.  R41-P5 is **CONFIRMED**: no isolated
statement reaches a zero-cell bar and nothing lands.

## Acquisition and controls

The fail-closed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round41_stage1_hpg_acquisition/run.sh`.
It uses the new target `ORCA2_ORCA1ICE_OMIP_L4_R41HPG1` and emits one
32,119,476-byte file per rank.  The registered payload contains stage-1
`rhd`, live `e3w`, live `gdept_z0`, `zhpi`, `zuap`, the final sum, and both
stored metric reciprocals.  The existing stage-2 HPG writer is retained and
must remain byte-identical.  The acquisition also requires exact inherited
RHS-family streams and four exact restarts, exact headers and sizes, unique
completion markers, and a clean producer commit.

The clean committed preflight passes, the patched Fortran syntax-compiles,
and the writer-layout plant fires with exit 69.  Per policy, `mpirun` was not
attempted in the sandbox.

The score's first-boundary plant replaces the recorded HPG boundary by the
candidate and moves the selector away from HPG.  The residual-equality plant
changes one active LDF oracle value by one representable step and breaks the
cellwise identity.  Both controls fire.  Focused and wide-test results are
recorded below.

The required separate `codex exec --sandbox read-only` review result is
recorded below.

## Choices

ASKED: score the acquired five compiled momentum boundaries in order.

UNASKED: none.  No model, configuration, carried state, stabilizer, NEMO
source, sea-ice selector, score, or scientific threshold changed.

## OPEN

1. The operator runs the Round-41 acquisition.  The next round scores `rhd`,
   `e3w`, `gdept_z0`, `zhpi`, `zuap`, and the final sum in compiled order and
   names the first non-bit HPG statement.
2. The thickness/RHS/reference-depth compensation remains held; HPG now owns
   the completed-RHS half on the registered 64-row boundary, but no operand
   changes before its internal statement walk.
3. The actual whole-card first non-bit row remains kt=1 stage-1 temperature;
   the barotropic/transport handoff remains its upstream owner candidate.
4. The northern-fold mask/wind debt and Decision-52 independent year remain
   open.
