# ORCA2 round 184 — atomic HPG fold/depth-average unit

Date: 2026-10-08. Base `34fab9763`; measurement commit `cd7ffe529`;
classifier correction `61e24d894`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round184/`.
Status: **HELD**.

The rung-0 numbers below are **independent hierarchy rung 0**. The rung-7
numbers are **given NEMO's entry**. They are deliberately separated. No
production configuration, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

The complete four-operand unit is not a Decision-96 net improvement. On the
independent rung-0 ladder it moves 83/200 rows: 26 toward NEMO by RMS, 23
away, and 34 aggregate-equal. Thus only 26/83 moved rows improve, not the
required strict majority. No bit-exact row leaves the bar; the first debt
(kt=1 stage-1 T) moves toward by `2.202285662861181e-20 K` RMS; kt=10
stage-3 ssh maximum remains exactly `0.42832517646246693 m`.

The given-NEMO-entry rung-7 ladder is unchanged on all 200 rows. Its kt=10
stage-3 ssh maximum remains `0.30375337870920743 m`. An unchanged companion
ladder passes the standing shared-statement predicate, but cannot repair the
failed independent rung-0 predicate. No physics statement lands.

The first worsened rung-0 row is kt=1 stage-2 T, whose RMS moves from
`2.0045321383145108e-05 K` to `2.0045321383145294e-05 K`. The next registered
cancelling partner is therefore the already-held seven-array V-transport/halo
unit: it consumes the same northern-fold V path immediately downstream. This
is the discriminating private pair for the next round, not permission to land
either half.

## Source statement and atomic operand proof

NEMO's surface fold-neighbour density product is executed at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416`; the same
association continues through the interior accumulator at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-453`. NEMO constructs
the reference V depth from recorded `e3v` and `vmask` at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:193-200`, constructs the face
masks and exchanges the fold at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dommsk.f90:206-232`, and stores the
reciprocal at `ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:212-215`. The
consumer is the written depth average at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`.

The round-184 classifier re-admits round 183's side-by-side record proof. On
the registered 68 north-fold V faces, the pre-substitution debts are raw HPG
on 1,319 wet levels, `e3v` on 27 cells, `vmask` on 1,319 levels, and
`r1_hv0` on all 68 faces. The north-fold HPG replay, recorded raw-HPG
component, and complete four-operand depth-average endpoint each close
bit-for-bit. The private arm is one switch; none of its four members is
independently selectable. Every unregistered face remains on the production
path.

## Ladder classification

### Independent hierarchy rung 0

| quantity | before | atomic arm | direction |
|---|---:|---:|---|
| moved rows | — | 83/200 | registered |
| RMS row census | — | 26 toward / 23 away / 34 equal | **not a majority toward** |
| maximum row census | — | 7 toward / 2 away / 74 equal | informative |
| exact-row losses | 0 | 0 | passes |
| first debt, kt1 stage1 T RMS | 1.1801028691523249e-05 | 1.1801028691523227e-05 | toward |
| kt10 stage3 ssh maximum | 0.42832517646246693 | 0.42832517646246693 | equal |

### Given NEMO's entry, shipped rung 7

| quantity | before | atomic arm | direction |
|---|---:|---:|---|
| moved rows | 0 | 0/200 | unchanged |
| exact-row losses | 0 | 0 | passes |
| kt10 stage3 ssh maximum | 0.30375337870920743 | 0.30375337870920743 | equal |

Artifact: `atomic_hpg_unit_gate.json`. Its three planted violations all fire:
operand closure, a previously exact row leaving the bar, and an incomplete
direction census. Status is
`HELD_ATOMIC_HPG_UNIT_NOT_NET_IMPROVEMENT`.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R184-P1 exact admitted operands | **CONFIRMED** by the 1,319/27/1,319/68 side-by-side census and exact HPG replay. |
| R184-P2 exact 68-face endpoint | **CONFIRMED**: 0/68 unequal after all four substitutions. |
| R184-P3 Decision-96 net improvement on both ladders | **REFUTED**: independent rung 0 has only 26/83 moved rows toward by RMS; rung 7 is unchanged. The arm is HELD. |
| R184-P4 non-vacuous and indivisible | **CONFIRMED**: 83 rung-0 rows move; all three classifier plants refuse. |
| R184-P5 shared landing gates | **INELIGIBLE** because P3 refuted the landing. The disabled/default path is nevertheless proven GYRE-identical below. |

## Shared-path, citation, tests, and review

Because the reproducible private hook touches shared model files, GYRE was run
at the base and tip even though no physics landed. The ten-step residual
artifacts contain 210 arrays with 0 unequal by `np.array_equal`; the offline
gate passes 70 rows with maximum worsening 0 ULP and retains first-over-bar at
kt=3. The 30-day members have 30/30 byte-identical daily snapshots; day 30 is
`3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b` on both
trees. This proves the private hook's disabled path is inert.

The focused battery passes 24/24. The one required
`tests/ocean/fidelity -n 12` battery reported 2,848 passes, seven skips, and
exactly four registered pre-existing reds: SI3 scalar-math provenance,
round-35 stamp scope, the worktree-stamp ratchet, and the GYRE round-129
record-backed pin. Nine cases were unreported when the xdist controller stayed
nonterminal at 99%. Six trajectory controls and one selector control pass in
isolation. The two remaining stage-sweep subprocess controls reproduce the
registered nonterminal class: `test_prediction_plant_is_fail_closed` times out
mechanically after 120 seconds, and `test_undetected_plant_exits_two` remains
silent beyond 180 seconds and was interrupted. No round-184 test fails.

The default citation gate passes 274 citations with no unmapped citation or
audit failure after the required SequenceMatcher re-anchor. The round-184
receipt gate and its shifted-line plant are recorded with this receipt.

The separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**:
`failed to initialize in-process app-server client: Read-only file system`.

## OPEN

1. Keep the four-operand HPG/depth-average arm private and HELD.
2. On the corrected independent entry, combine it privately with the held
   seven-array V-transport/halo unit and score the kt=1 stage-1 to stage-2 T
   boundary first. Offline replay only; neither half lands separately.
3. If that complete pair becomes a Decision-96 net improvement, run both
   ladders, GYRE, DINO, tanks, and the independent month boundary before a
   production landing. Otherwise continue from the first worsened statement.
4. The independent rung-0 month step-96 live-thickness refusal remains
   **UNMEASURED-with-spec** in this held round.

No acquisition is needed. ASKED choices: the atomic four-operand unit under
Decision 96. UNASKED choices: empty.
