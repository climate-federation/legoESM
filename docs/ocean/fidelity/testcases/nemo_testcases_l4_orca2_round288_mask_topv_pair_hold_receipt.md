# ORCA2 round 288 — resolved-mask / top-V pair hold

Date: 2026-10-16

Status: **HELD**.  NEMO's resolved three-dimensional stage mask by itself
reproduces the registered kt=4 live-W-thickness refusal.  Adding NEMO's
completed-stage V fold exchange prevents that refusal and completes kt=10,
but its first matched effect is kt=1 stage-1 V and moves the RMS **away** from
NEMO.  The completing trajectory then grows explosively.  The atomic pair
therefore fails frozen prediction R288-P3 and never enters the full landing
gates.  Both statements remain off in production.

Every number below is **independent rung 0**.  It uses NEMO's rung-0 initial
state, not the given-entry twin, and is not mixed with a given-entry number.

## Provenance and instrument

- Lane base: `7f6b144f0`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round288.md`, commit `e4db9dc0f`.
- Measurement implementation: `98644f96b`, with fail-closed boundary and
  matched-stage repairs in `4c08e30b9` and `b9d6b49bf`.
- Candidate measurement commit: `4c08e30b9fa998389c8f04724a33446e523523ad`.
- Admitted oracle record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round90/acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2`
  (80 rank-complete entry/stage frames).
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round288/`.
- Execution: CPU, production JIT, fp64, x64 and libm policy.
- Claim input: the round-276 production control, 200 rows.
- Search before build: the round-174/249 stage-growth patterns and the
  existing round-286 resolved-mask helper were reused.  No new executable
  observer was added; the already-gated completed-stage hooks supplied the
  matched states.

The temporary package arm was removed in commit `58ba3e7f4`.  Relative to the
lane base, `packages/` has no diff.  No card selector, deck, NEMO source,
record, initial state, carried state, runoff setting, or sea-ice setting was
changed.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R288-P1 | CONFIRMED | Production completes 200 rows; mask-only completes kt=1..3, then the kt=4 stage-1 completed state is non-finite and the ordinary run fires `raw-mesh e3w_int must contain only finite values > 0`. |
| R288-P2 | CONFIRMED | Mask plus top-V completes kt=1..10 with no terminal refusal. |
| R288-P3 | REFUTED | First pair effect is kt=1 stage-1 V; RMS increases from 8.749916434572169e-4 to 8.849698307689673e-4 m/s.  No exact stage row is lost. |
| R288-P4 | OFF_CLAIM | P3 is red, so OMT/rung-10/GYRE/DINO/tank landing gates were not run. |
| R288-P5 | CONFIRMED | All three known-answer plants fire and exit nonzero. |

The pair's first effect is the source-ordered completed-stage V association,
not a later live-thickness guard.  At kt=1 stage 1 its maximum V error is
unchanged at 3.3671380625114315e-2 m/s, while RMS moves away as shown above;
the unequal-cell count changes from 438,259 to 434,420.  Fewer unequal bits
are not a fidelity vote when the registered RMS worsens.

The top-V statement does suppress the mask-only candidate's immediate tracer
growth: kt=1 stage-2 S RMS/max change from 2.1566070108516914e-2 /
7.273862831443509 PSU to 4.463617972535543e-3 / 1.5272997426403592 PSU.
That is a stabilising interaction, not a qualifying landing.  By kt=3 stage 3,
the completing pair already has S RMS/max 2.7593704784805287e-2 /
11.13555040846023 PSU.  At kt=10 stage 3 its RMS errors are T
3.907204554795108e7 K, S 5.486156760138972e8 PSU, U
5.191551249218675e12 m/s, V 6.331716583659298e12 m/s and SSH
6.870298809190192e-2 m.  The completing arm is therefore not a hidden near-bar
trajectory.

Artifact SHA256 values:

- `mask_only.json`:
  `3b5d7fea77ce060a35d7db0e4d21b132ae21600cc6a84f235f89f845ffc81785`
- `pair.json`:
  `3ab673dd400178e48e497d88d91989cc1ffc24fa946ed32fbb99fc8eef5852a9`
- `classification.json`:
  `882b4a8286397f197ad7783c207f36217ca09c74e37075fe142a164001b67106`

The `one-bit`, `missing-single-failure`, and `fake-pair-pass` plants each
print `STATUS PLANT-FIRED` and exit 2.  The classifier status is
`HELD_R288_PAIR`; it reports no exact-stage-row loss.

## Validation

- Focused round-288 classifier tests pass 6/6; the focused cumulative citation
  tests pass 17/17.  Their logs have SHA256
  `28ea3f40e4e4da4bdea6442aa0a6b7f6d842f553bd713542c3019b623bab086f`
  and
  `99df6ad9258095de79b14b98f849e19c52843c836f7c0d8c390a0d8a561421a7`.
- This receipt's citation gate finds three mapped live citations with no
  failure; the cumulative default-receipt gate is also clean.  A rigid
  two-line shift of the completed-stage association exits 1 with
  `SYMBOL-NOT-AT-LINE`.
- The single permitted `tests/ocean/fidelity -n 12` battery collected 3,611
  tests and reached 99%.  One G1 worker then aborted inside the JAX compiler;
  the replacement controller stopped after its remaining work, without a
  summary.  The eight ordinary reds printed before the abort are all the
  registered branch-tip reds: the round-129 GYRE record hash, two round-33
  source anchors, the OMT-1 module-edge ratchet, the round-228 citation entry,
  dirty-escape scope, worktree stamping, and SI3 scalar-math provenance.
- The crashed worker printed seven G1 failures.  Each exact node ID passes in
  a fresh serial process (`-p no:xdist`), in 34.23--103.47 seconds.  The full
  battery was not rerun.  Its log SHA256 is
  `897112ce433c6f23b749df79ca9fd64494d59770d407e001ac81da52e56df412`;
  the seven-test serial log SHA256 is
  `09ea849473cb9ba9e9f1c30a06f90accfd0664bbb673eb6e95c242f6f1299dac`.
- `git diff --check` is clean.
- Independent review: **operator**, per the lane instruction.  The candidate
  does not land and no nested review was attempted in this sandbox.

## Configuration choices

- **ASKED:** none.  Decision 126 remains pending and is not re-raised.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange remains off.
- Sea ice remains exactly as the shipped ORCA2 card carries it, including its
  `unmeasured_features` tuple.

## Compiled source citation

The record build creates, bottom-clears and associates the resolved U/V/F
masks at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232`.  The ORCA2
vector branch multiplies the stage update by that V mask at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.  After the
stage update, NEMO associates U and V with sign -1 and T and S with sign +1 at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820`.  This last
statement is the first matched statement moved by the pair and its stage-1 V
RMS moves away, so it cannot be promoted as Decision 125's missing partner.

## OPEN

1. Keep the resolved stage mask and completed-stage top-V exchange held.
2. At the matched post-association kt=1 stage-1 boundary, split the top-V
   exchange's copied source row, pivot-row support and sign against the NEMO
   record.  Name why a source-exact-looking exchange moves V RMS away before
   considering another partner.
3. Leave shipped runoff unchanged while Decision 126 remains pending.
