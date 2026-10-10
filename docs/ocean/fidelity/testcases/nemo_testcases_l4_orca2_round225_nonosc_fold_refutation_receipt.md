# ORCA2 round 225 — OMT-4 live-nonosc fold refutation

Date: 2026-10-10. Frozen base: `730ae56be`. Preregistration commit:
`1c263bcbb`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round225/`.

This round changes no shipped card, carried state, stabiliser, sea-ice
selector, or `unmeasured_features` tuple. The literal production candidate was
committed solely to measure the frozen sufficiency prediction, then retracted
after the prediction failed. The final tree has no `packages/` diff against
the frozen base. All trajectory statements below are labelled
**independent**.

## First source-ordered statement

The resolved compiled program calls the live optimized `nonosc`, not
`nonosc_org`, at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:306-316`.
It sets the exact finite dry sentinel and bound slices at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:768-821`,
forms the seven-member maximum and minimum at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:798-861`,
builds and guards the beta budgets at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:862-878`,
and selects the V-face coefficient at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:888-915`.
The relevant T-pivot halo association executes at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:581-638`.

The offline pure-JIT replay consumes only the admitted OMT-4 kt=1 completed
stage states and reports zero in-executable observers. The first **bit**
difference is NEMO's exact `HUGE(1._wp)` dry sentinel: 261 non-north member
values differ for each tracer, all at dry sentinels, and zero wet limiter
bounds change. This refutes R225-P1 as written because a member preceding the
north stencil differs. It is inert on the wet calculation.

The first **effective** statement is the seven-neighbour `zup` maximum. The
T-pivot north source differs at all 1,366 active fold-support cells for both T
and S. The literal maximum changes 52 T values (maximum 0.1091150076639269 K)
and 42 S values (maximum 0.0773959115316103 PSU); the corresponding `zdo`
minimum changes 61 T values (maximum 0.16280626380051064 K) and 50 S values
(maximum 0.12577644442392 PSU). Zero changed bound lies off the fold.

The source guards are live: 16 fold-support values per tracer take a guarded
no-division arm, every guarded result is finite, and the deliberately
unguarded replay is non-finite at all 16. The sign-selected adjacent beta pair
reproduces the literal V coefficient bit-for-bit. On 1,319 active nonzero
antidiffusive fold faces, the coefficient differs at seven T and nine S
values, maximum absolute difference 1.0; zero off-fold coefficient changes.

The final clean gate is stamped to `f13ba80da4bb78e806c62683c779d8cd5a081af0`
and reports `PASS_R225_FIRST_NONBIT_NONOSC_FOLD`. Evidence SHA-256:

- `nonosc_fold_walk_final.json`:
  `c3a0760ecf0d1bf361a689a7656fc31dce7467f0701d0434c816f2a205327c70`
- `nonosc_fold_walk_final.log`:
  `b11db06aa339e40a40b1112b920c2d3fac0bdc0ca9d87e610123b79a470eebe2`

## Sufficiency: REFUTED and retracted

The private candidate atomically combined round 224's donor/centred face
association with the exact dry sentinel, T-fold upper/lower neighbour,
guarded beta selection, and live north V coefficient. On the complete
fold/association/V-transport unit, the **independent** OMT-4 trajectory again
completed kt=1--7 and refused during kt=8 on exactly:

`raw-mesh e3w_int must contain only finite values > 0`.

The candidate log is byte-identical to round 223's pre-arm baseline and round
224's face-only arm, SHA-256
`973bbd48ba821edff92502b57124cf1d6d560d6dd4acccef930d646bc29f90af`.
R225-P4 is therefore **REFUTED**, not unmeasured. The candidate was retracted
in `8e70f8284`; the round's gate now requires the identical logs and records
the failed prediction mechanically. A Decision-96 census cannot be formed
from a candidate that does not complete the ladder.

The next source-ordered boundary is NEMO's final limited-flux divergence and
tracer-RHS update at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:322-329`.
No claim is made yet about its sufficiency.

Because the final tree has no model-file diff, the certified OMT-4, rung-0,
rung-10, GYRE, DINO, lock-exchange, and overflow trajectories are unchanged
by construction. This round lands only the committed probe, its controls, and
the failed-candidate record.

## Preregistration dispositions

| ID | disposition |
|---|---|
| R225-P1 | **REFUTED**: the exact dry sentinel is an earlier bit difference, though inert at every wet bound. |
| R225-P2 | **CONFIRMED**: 16 guarded-off divisions per tracer remain finite; the unguarded plant is non-finite at all 16. |
| R225-P3 | **CONFIRMED**: the selected beta pair reproduces the changed V coefficient bit-for-bit, only on the fold. |
| R225-P4 | **REFUTED**: the atomic candidate reaches the byte-identical kt=8 refusal. |
| R225-P5 | **CONFIRMED**: all five plants refuse. |

## Validation and review

The focused round-225 plus citation-gate suite passes 23/23 (log SHA-256
`a0884ddc8369bc59e2b20961a37c8b21c85fcb8d04020de1c7810ee1c24a1af7`).
Each of the north-source,
fold-support, guard, coefficient, and sufficiency plants exits 2 with
`STATUS PLANT-FIRED`. Their log SHA-256 values are respectively
`34f7dc8b5cc2434d25f20b533877767aed0e67a145f79141a1c41e151fa212c1`,
`f6766ad2626299e9bc51268729bfc0f04e0f3910db0d456c45584ac6772295fd`,
`482b20f6f7e358bb286308cf8333fdf44e6198defc7ce21a646345b83dfd6fa5`,
`227ef7ce66de2d077c974fe6539e903cc4631dcc0cca56ff180ad47e620a6966`,
and `0fc6443069ed55fff6a613fa6a9d6a0558e78621581f38f93998fb62b1cf0ce3`.

The round citation gate passes all 7 cited compiled spans; the cumulative
default gate passes 274 citations with zero failures, unmapped citations, or
map-audit failures. Shifting the guarded-budget range by two lines makes the
round gate fail `SYMBOL-NOT-AT-LINE`. Round/default/plant JSON SHA-256 values
are `50bec612a9fe51c2690e73565522ed0371154cc9160991ef7d102787c2e2a75e`,
`c2a5bfc3017a81867da6418fcbe343343ac797be60cdba57cb67310e2f0a24ed`,
and `df28c243d4e65f1e2e06a8f8b9c3a0a59d24be4ee25e27341da12e104b270f8e`.

Independent review was attempted with `codex exec --sandbox read-only` and
exited 1 before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. **Independent review unavailable
in-sandbox**; this is not a PASS. Review-log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The single prescribed `tests/ocean/fidelity -n 12` battery collected 3,094
tests and reached 98%. It classified 3,051 passed, seven skipped, and four
registered pre-existing failures: the GYRE round-129 spread-floor record,
the allow-dirty scope ratchet, the worktree-stamp ratchet, and the SI3
scalar-math provenance gate. Thirty-two tests remained unclassified when
every real pytest process disappeared without a terminal summary; the idle
wrapper was interrupted and the battery was not relaunched. This result is
not called PASS. Log SHA-256:
`5be62a9bc88b52dd62c1dacb4f6516d04803a12cfb5c4c494461d624833a7ce5`.

## OPEN

Stay on OMT-4. The source-literal donor/centred face association and entire
live northern-fold `nonosc` sequence are real but insufficient. Offline-walk
the final limited-flux divergence and the tracer-RHS update at compiled lines
322--329, then the stage update, from passive completed states. The first
literal statement that removes or delays the kt=8 live-W refusal is the
compensating partner; score it only as one atomic unit under Decision 96.
OMT-5 remains blocked.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
