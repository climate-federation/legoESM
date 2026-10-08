# Preregistration — ORCA2 round 177 rung-0 independent initial state

Date: 2026-10-08. Frozen base: `8214fa83a`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round177/`.

Every trajectory number in this round is **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry result is mixed into an independent table. The shipped rung-10
ORCA2 card, all six sea-ice selectors and its `unmeasured_features` tuple stay
unchanged.

## Frozen statement and controlled change

The compiled rung-0 oracle applies ORCA_R2's regional T/S alterations only
inside `nn_cfg == 2 .AND. ln_tsd_dmp`
(`ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90:218-254`). The resolved
rung-0 namelist sets `ln_tsd_dmp=.false.`
(`round175/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg:51`). The private
rung-0 card currently inherits the shipped card's unconditionally altered T/S
(`nemo_testcase_recipe.py:1434-1474,1650-1651`).

The one-variable change is gate-local: rebuild only the private rung-0 card's
T/S with the existing `build_orca2_initial_ts(...,
apply_hand_alterations=False)` control. Its grid, geometry, model config,
velocity, sea surface and every forcing remain identical. The shipped card
continues to take the default alteration-enabled path.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R177-P1 | The corrected private rung-0 independent entry is bit-exact to NEMO. | T/S/u/v/ssh each have zero unequal active cells and `np.array_equal` is true against the kt=1 entry record. | Any field differs. |
| R177-P2 | The change removes exactly the alteration block and nothing else. | Old-to-new active differences are T=1,283 and S=720; u/v/ssh stay byte-identical; the shipped card is byte-identical before/after construction and still differs from rung 0 by the same T/S block. | Any count moves, any non-T/S field moves, or the shipped card changes. |
| R177-P3 | The independent ten-step ladder still first leaves bit identity at kt=1 stage 1 T, but every row is remeasured and previous independent numbers are superseded. | The gate completes 200 rows and reports that same first boundary; no prior JSON is reused. | The first boundary or completion changes. |
| R177-P4 | Because both registered growth columns are outside the alteration boxes, the production independent month still first becomes non-finite at step 36 in T at `[86,159,0]`. | A fresh production-JIT CPU run reports exactly that step, field and cell. | Any earlier/later step, other field/cell, or 240-step completion. |
| R177-P5 | After the entry is exact, the rank-complete stage-1 offline replay reaches an internal row and no longer stops at the initial-state guard. | The first scored debt is below the five entry rows, or all replayed rows stay at the floor and the gate stops on its registered unavailable candidate operand. | Any entry row differs or the gate still names the alteration guard. |
| R177-P6 | The round changes no production physics. | No `packages/` file, shipped card selector, carried state, stabiliser, threshold or sea-ice field changes. | Any such change lands. |

Failed predictions remain in the receipt. The comparison floor stays `2e-10`;
bit-exact means `np.array_equal` on binary64 arrays. Every new or extended gate
must carry a planted violation that fires.

## Verification and terminal rule

Run on CPU under fp64/libm. Commit the private-card change before measurement,
then require its clean commit stamp. Rerun the independent entry gate, all 200
ten-step ladder rows, the independent month to its first non-finite boundary or
step 240, and the admitted rank-complete stage-1 offline table. Because no
`packages/` file changes, GYRE/DINO/tank trajectories are unchanged by
construction and are not trajectory claims for this round.

The first non-bit source-ordered statement after the corrected entry owns the
next walk. Do not infer an operator below a missing record or unavailable
candidate operand. No model statement lands this round.

ASKED choices: make hierarchy rung 0 obey its resolved
`ln_tsd_dmp=.false.` oracle branch and remeasure its independent trajectory.  
UNASKED choices: empty.
