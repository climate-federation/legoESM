# ORCA2 round 286 — resolved WS-RK3 stage mask held

Date: 2026-10-15

Status: **HELD**

This round keeps the two start labels separate.  The source replay below is
`independent OMT-4`; the trajectory gate is `independent rung 0`.  No
`given NEMO's entry` number is mixed into either table.  Sea ice and its six
selectors are unchanged.  Decision 126's runoff choice remains pending and
was neither changed nor re-raised.  Decision 125's top-V exchange remains off.

## Statement and source

The live NEMO branch builds the three-dimensional face masks, applies its
bottom limits, and associates them laterally at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232`.  The vector-form
stage update consumes those arrays at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.
legoESM's current production path instead reconstructs its stage masks from
compact T-cell activity at `ocean_model_latlon_cgrid.py:6048-6052`.

The private candidate routed only this WS-RK3 consumer through the existing
resolved NEMO operand source.  It added no builder, default, configuration
choice, stabiliser, carried state, runoff change, or sea-ice change.  Producer
commit `78a389a0a2fb51557cbd175219e60d1980559ff4` was clean, CPU-only, JIT,
fp64/libm.  The failed production candidate was reverted in commits
`e7f0270583` and `7200f07c66`; the final tree has no model-file difference
from round 285.

## Frozen predictions

- R286-P1 **CONFIRMED at the executing consumer**: the independent OMT-4
  kt=1 stage-1 raw V update moved from 1,319/415,175 unequal cells
  (max 0.011299245240990687 m/s) to 0/415,175, bit-exact.  The recorder's
  standalone `vmask` row stayed at 1,319 unequal because it records the old
  compact diagnostic rather than the routed consumer; the gate requires that
  row to stay unchanged and does not cite it as the candidate mask.
- R286-P2 **CONFIRMED**: the compact consumer reproduces 1,319 update
  differences; the one-bit known-answer control exits 2.
- R286-P3 **REFUTED**: the same-tree independent rung-0 control completes all
  200 ladder rows, while the candidate completes kt=1..3 and refuses during
  kt=4 before that checkpoint is emitted: `raw-mesh e3w_int must contain only
  finite values > 0`.  A candidate that cannot complete the ladder is not a
  Decision-96 net improvement.
- R286-P4 **NOT RUN, off-claim**: the first required ORCA2 card was already
  red.  The candidate was reverted, so the final shared implementation is
  byte-for-byte the round-285 model tree.
- R286-P5 **NOT RUN by preregistered order**: the held fold unit may be scored
  only after P1-P4 pass.  It remains held; no downstream-damage walk was
  performed.

The stage replay gate reports `PASS_R286_STAGE_MASK_REPLAY`; its planted bit
reports `STATUS PLANT-FIRED`.  The trajectory refusal gate reports
`PASS_R286_TRAJECTORY_REFUSAL_GATE`; deleting the refusal signature makes its
plant exit 2.  Artifact SHA256 values are:

| artifact | SHA256 |
|---|---|
| `stage1_mask_measurement_raw.json` | `558cce477ddd028548528a0a3f9495192e43c87b4f38163314400353017607c9` |
| `stage_mask_replay_gate.json` | `3d88c2c6519c6c3d5c94180efc2f3f9160a986beca6ff6a139dd7b78ceab86b0` |
| `rung0_independent_control.json` | `4e221e81832fee61a2b75f1b5e0c4809435e4350d26d495ad00dc2711151e83c` |
| `rung0_independent_candidate.log` | `e07b8a64f70b765b8ced9b0c952b54fc2115e6c93181b81c3d5942bf75b95ded` |
| `trajectory_refusal_gate.json` | `3654fa27632f155e954cc453ba2a6a9528215e1729bc914c87ca60fc634b69d2` |

Evidence lives in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round286/`.
Independent review: operator.

## OPEN

1. The resolved stage mask is a source-exact local closer but exposes the
   rung-0 live-thickness failure by kt=4.  Keep it out of production and treat
   that first matched passive boundary—not the later guard—as the next owner
   if this cancelling unit is resumed.
2. Decision 125's completed-stage top-V exchange remains off.  Its matched
   stage-1 vector-form V-update diagnosis remains the active campaign item.
3. Decision 126 remains pending; the shipped runoff setting is unchanged.
