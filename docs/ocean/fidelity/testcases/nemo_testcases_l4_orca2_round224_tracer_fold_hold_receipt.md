# ORCA2 round 224 — OMT-4 northern-fold tracer statement

Date: 2026-10-10. Frozen base: `f969dce99`. Preregistration commit:
`8e8453bcb`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round224/`.

This round changes no shipped card, carried state, stabiliser, sea-ice
selector, or `unmeasured_features` tuple. A production arm was committed for
measurement and then retracted after its frozen falsifier fired; the final
tree has no `packages/` diff against the round base. Every trajectory number
below is labelled **independent**.

## First source-ordered statement

The compiled RK3 dispatcher runs centred advection at stages 1--2 and FCT at
stage 3 in
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540`.
NEMO's centred V flux reads the exchanged northern tracer at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_cen.f90:149-160`.
The stage-3 FCT program first forms donor-cell U/V fluxes from the before
tracer at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:495-539`,
then averages a second donor-cell pass at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:562-573`.
Its centred high-order horizontal flux is at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:191-199`.

The offline pure-JIT replay consumes only OMT-4's admitted kt=1 completed
states; it adds no in-executable observer. Stages 1 and 2 are bit-exact to the
literal T-pivot association for both T and S. The regular-wall control fires
on 1,319 values per tracer and stage (maximum 0.3447385433884961 K and
0.7728019055653377 PSU at stage 2), so this exoneration is non-vacuous.

At stage 3, the first changed source-order field is `first_v_raw`. The
grid-aware fold arm changes 6,286 T+S values there, maximum
249.5292683007613 in the recorded metric-weighted flux units. The preceding
U and W donor fluxes are bit-exact (0 differing values); zero changed V
values lie off the northern fold. The midpoint donor pass changes 7,600
values, maximum 249.52883910082684. Therefore the first non-bit executable
statement is NEMO's donor-cell V flux using the exchanged northern tracer,
with the midpoint pass sharing the same association.

The clean final replay is stamped to `6c6dad44432d136bebcbea079dabb30d440d5a43`
and reports `PASS_R224_FIRST_STATEMENT_DONOR_V_FOLD`. Artifact SHA-256:

- `tracer_fold_walk_final.json`:
  `2ac7b667f1c1f8f5876f1b06d79bd26a700677bcea48155d18ffc27fd87dc320`
- `tracer_fold_walk_final.log`:
  `300ef680d36531f9f113ded0fb069e9fc8b83d75493fa2a1e09c0335e1a3ae9a`

## Sufficiency arm: REFUTED and retracted

The private production arm supplied the T-pivot fold to both donor passes and
the centred high-order V flux. Thus it tested the entire source-ordered face
association, not only the first named statement. On the complete atomic
fold/association/V-transport unit, the **independent** OMT-4 trajectory
completed kt=1--7 and refused again during kt=8 on exactly:

`raw-mesh e3w_int must contain only finite values > 0`.

The refusal log SHA-256 is
`973bbd48ba821edff92502b57124cf1d6d560d6dd4acccef930d646bc29f90af`,
byte-identical to round 223's pre-arm atomic refusal log. R224-P4 is therefore
**REFUTED**: the face association is real but insufficient, no Decision-96
census can be manufactured from a non-completing candidate, and the
production change was retracted in commit `6c6dad444`.

Because the final tree has no model-file diff, the certified rung-0, rung-10,
GYRE, DINO, lock-exchange, and overflow trajectories are unchanged by
construction. This round lands only the committed measurement and its
falsification record.

## Preregistration dispositions

| ID | disposition |
|---|---|
| R224-P1 | **CONFIRMED**: stages 1--2 are literal-fold exact; the 1,319-value wall control fires. |
| R224-P2 | **CONFIRMED**: `first_v_raw` is first; U/W stay bit-exact and off-fold change is zero. |
| R224-P3 | **CONFIRMED**: both donor V passes move under the fold association. |
| R224-P4 | **REFUTED**: the source-complete face arm reaches the identical kt=8 refusal. |
| R224-P5 | **CONFIRMED**: wall, source-order, support, and sufficiency plants all refuse. |

## Validation and review

The focused gate suite passes 5/5. Each of the four planted violations exits
nonzero with `STATUS PLANT-FIRED`. Citation, cumulative citation, independent
review, and the prescribed ocean-fidelity battery results are recorded below
after their final-tree executions.

## OPEN

Stay on OMT-4. The donor and centred face associations are measured and
insufficient. Walk the live `nonosc` limiter from the admitted stage-3 state,
starting with the exchanged north-neighbour stencil and the guarded
`zup/zpos` and `zdo/zneg` budgets, then its V-face coefficient. Replay only
from passive completed states; do not add an in-executable observer. Re-score
the complete atomic unit only after the first non-bit limiter statement is
named. OMT-5 remains blocked behind this walk.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
