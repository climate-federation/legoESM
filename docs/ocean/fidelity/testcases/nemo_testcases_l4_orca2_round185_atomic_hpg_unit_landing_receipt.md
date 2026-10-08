# ORCA2 round 185: atomic fold-HPG unit landing receipt

Date: 2026-10-08  
Status: **LANDED**  
Claim labels: rung 0 and its month are **independent**; rung 7 is **given
NEMO's entry**.

## Frozen question and correction

Round 185 tested the exact round-184 four-operand unit under Decision 96 with
score-equal rows excluded from the vote.  This corrects, rather than silently
rewrites, round 184: its `HELD` verdict counted 34 bit-moved but RMS-equal rows
as adverse votes.  The corrected executable gate defines score-moved rows as
`toward + away`, and its `score-equal-vote` plant proves that equal rows cannot
change the majority.  The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round185.md`.

## Compiled statements and landed unit

The record build is `ORCA2_OMIP_L4_R182HPGFOLD`; this is the branch that ran.
NEMO forms the surface V pressure-gradient north product from the literal
`jj+1` density at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416`; the same north
association continues through the vertical recurrence at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-453`.

The admitted R92 build constructs reference face depths from the raw partial
face thickness and face masks at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:193-200`, constructs and
exchanges the V mask at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dommsk.f90:206-232`, and stores the
reference-depth reciprocal at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:212-215`.  Its vector-invariant
slow V forcing is the depth average at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`.

Those four statements are one measured cancelling unit: folded north density,
raw `e3v`, exchanged `vmask`, and raw `r1_hv0`.  The operand gate reports the
folded HPG component bit-exact, the four-operand endpoint bit-exact, and a
68-face target on row 147 (1,319 wet levels).  Production now takes the folded
north density automatically for a NEMO literal SCO grid with an active fold,
and uses the raw recorded V depth-average operands only where the raw face is
wet and the compact reconstruction is dry.  There is no public selector or
stabiliser.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round185/atomic_hpg_reclassified.json`.

## Decision-96 census

The corrected rung-0 **independent** census touches 83/200 rows at the bit
level.  Of the 49 rows whose RMS score moves, 26 move toward NEMO and 23 away;
34 additional rows are explicitly registered as `bit-moved, score-equal`.
The JSON records every touched row and its before/after RMS, maximum,
mean-over-unequal, unequal-cell counts, and first unequal index.  The first
over-bar row moves toward NEMO, no bit-identical row is lost, and kt=10 stage-3
SSH maximum stays exactly `0.42832517646246693 m`.  Thus the corrected vote is
a strict majority toward NEMO and all Decision-96 predicates pass.

The rung-7 **given NEMO's entry** ladder is unchanged on all 200 rows; its
kt=10 stage-3 SSH maximum remains `0.30375337870920743 m`.  Fresh production
runs reproduce both frozen private-arm trajectories: the rung-0 report is row
for row identical, and all 40 rung-7 candidate checkpoints are array-identical
(the removed private-arm metadata is the only report difference).

Artifacts:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round185/rung0_production.json`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round185/rung7_production.json`

## Independent month boundary

The production rung-0 run begins array-identically to NEMO in T, S, u, v, and
SSH.  It completes 95 finite steps and step 96 again refuses on the existing
`raw-mesh e3w_int must contain only finite values > 0` live-thickness guard.
There is no non-finite state claim.  The landing therefore leaves the corrected
entry's month boundary unchanged.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round185/independent_month_boundary.json`.

## Shared blast radius

- GYRE: the 70-row ten-step registry and all 210 residual arrays are
  array-identical to round 184.  A fresh 360-day run is byte-identical at all
  twelve snapshots to the certified round-163 member.  The T RMS values remain
  day 30 `2.3432419318363155e-06 K`, day 240
  `6.5816987106668941e-05 K`, and day 360
  `5.4077212586815052e-05 K`.  The corresponding snapshot SHA-256 values are
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
  `2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
  and `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`.
- DINO: the CPU month gate passes at day-30 wet-3D T RMS
  `2.056821682e-03 K` against the fixed `2.244317642e-03 K` bar.  Its planted
  `6.981690958e-03 K` score fails.
- LOCK_EXCHANGE and OVERFLOW: each is array-identical to round 163 across all
  50 certified rows, with zero worsening ULPs and unchanged first-over-bar
  rows (kt=8 U and kt=2 T/U respectively).  Each `worsen-3ulp` plant fails.
- Shared-card census: PASS; `card_reference` moves the GYRE control and fails.

All evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round185/`.

## Validation and review

Focused validation passes 60/60 tests.  The required single
`tests/ocean/fidelity -n 12` invocation collected 2,871 tests and was stopped
after prolonged 97% tail inactivity; it had four failures.  Isolated reruns
identify exactly the four registered pre-existing reds: SI3 scalar-math
`MY_SRC` provenance, the stale certified-year-harness spread record, the
allow-dirty scope ratchet, and the worktree-stamp ratchet.  No round-185
focused test fails.

The separate `codex exec --sandbox read-only` review was attempted and returned
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The default receipt citation gate passes with zero unmapped citations.  The
round-185 receipt gate and its shifted-source plant are recorded alongside the
other evidence.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R185-P1 | **CONFIRMED** | 26 RMS rows toward, 23 away, and 34 score-equal bit moves; strict majority uses 49 score-moved rows. |
| R185-P2 | **CONFIRMED** | First debt toward, zero exact-row losses, unchanged rung-0 SSH maximum, and rung 7 unchanged. |
| R185-P3 | **CONFIRMED** | Production rung-0/rung-7 trajectories reproduce the frozen private arm; all four operands close side by side. |
| R185-P4 | **CONFIRMED** | GYRE year/ladder, DINO, both tanks, and shared-card census pass; every planted violation fires. |
| R185-P5 | **CONFIRMED** | The independent month completes 95 steps and reaches the same step-96 guard, not earlier. |
| R185-P6 | **CONFIRMED** | Operand, exact-row-loss, false-majority, score-equal-vote, DINO, tank, shared-card, and citation controls fire. |

## OPEN

1. The independent rung-0 month still refuses at step 96 on the live-thickness
   guard.  The next round must print the refusing cell and the ten-step-block
   growth table before attributing an owner.
2. Re-test the held halo/V-transport cancelling unit on this corrected-entry,
   landed-HPG tree under the same RMS-vote and SSH predicates.  It remains
   private unless Decision 96 passes.
3. Rung 7's first non-bit row remains kt=1 stage-1 T, given NEMO's entry; sea
   ice remains exactly the card's unmeasured-feature declaration.
