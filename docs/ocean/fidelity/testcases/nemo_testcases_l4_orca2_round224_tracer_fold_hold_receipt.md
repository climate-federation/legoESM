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

The focused gate suite passes 5/5 (log SHA-256
`9f6861b9d0df9f2a48509f9b9eb83f15d2fd2c7de6d6afa71ac05607dddaacb3`).
Each of the four planted violations exits nonzero with
`STATUS PLANT-FIRED`; their log hashes are, in order, wall
`3c04946d39d50849dcee98a2fd7aaf366ca3ca5b5d81601a7f136e8d157519da`,
source-order
`67979f18113364e01c66b809453a2cafe3510be6a1f6262c3528d40bc5130b03`,
support
`4d9c0f79ab3cc0cfea80f1f530eaf93101d53e13c3c95272e110c5708154f241`,
and sufficiency
`82be91bbca38b37c14880cdf839cad58bff2989e562f4e959742798a01666487`.

The round citation gate passes 5/5 compiled citations, and the cumulative
default gate passes 274 with zero unmapped citations, failures, or map-audit
failures. Shifting the donor range by two lines makes the round gate fail
`SYMBOL-NOT-AT-LINE`. The round/default/plant JSON SHA-256 values are
`80b40276bf90fe8132744cfcc227a4227283e3e7cb569d6cf806058c8e913356`,
`0ccf69df7544094101b44fbcdcc07b06a2a37315bd94b84b0f66cfc9e7a8b218`,
and `db2b63e1315e8237ddd904ef39827aed28c43b20e4afa3c6a55cf628f180c6b5`.

Independent review was attempted with `codex exec --sandbox read-only` and
exited 1 before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. **Independent review unavailable
in-sandbox**; this is not a PASS. Review-log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The prescribed `tests/ocean/fidelity -n 12` battery collected 3,088 tests and
reached 99%. It recorded 3,058 passed, seven skipped, and the same four
registered failures as round 223: the GYRE round-129 spread-floor record
stamp, allow-dirty scope, worktree-stamp ratchet, and SI3 scalar-math
provenance gate. Nineteen tests remained unclassified when every real pytest
process disappeared without a terminal summary; the idle wrapper was
interrupted and the battery was not relaunched. It is not called PASS. Log
SHA-256:
`db66615756322eed0157972d171760ae886df171081cacf1cc9bbc4b00eba89e`.

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
