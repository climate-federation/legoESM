# ORCA2 round 238 — exact face-geometry landing

Date: 2026-10-11. Frozen base: `f9b157da2`. Preregistration commit:
`107ad52eb`. Implementation commit: `728ee561d`. Status: **LANDED**.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round238/`.

This round implements Decision 114 only. Decision 115 was already present in
the base. The round-217 V-transport unit remains out of production for round
239. No deck, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Cited statement and implementation

The executing ORCA2 record build reads the reference U/V face thicknesses at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188`, with the
compiled three-dimensional association at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90:114-121`. It derives
the U/V/F masks from `tmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-213` and exchanges
them with sign `+1` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:228-232`.
The executing T-pivot V rule writes the pivot and halo rows from the mirrored
source rows at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:973-980`.

legoESM now routes every WS-RK3 stage-geometry consumer through the existing
resolved builder. A card carrying NEMO's raw mesh must carry all four of
`e3u_0`, `e3v_0`, `umask`, and `vmask`; partial provenance raises. Cards
without the raw mesh retain the pre-existing reconstructed builder. There is
one implementation of the face geometry and no new selector.

The once-only full battery caught the first implementation routing generic
and OVERFLOW coordinates that carried only an unrelated raw T thickness into
the face-mesh resolver. Commit `06130089a` fixed the scope discriminator: a
card takes the resolved source only when it carries the complete
`nemo_een_barotropic` face bundle. The three affected generic/OVERFLOW tests
then pass in isolation. Both ORCA2 claim labels were re-measured on the final
committed tree (`88baca68c`) and retain the exact table below.

## Exact operand gate — both claim labels

Every completed-state trace is passive. The raw T/U/V/F mask registry is
bit-exact under both labels. The production-path boundary is:

| label | operand | unequal / support | maximum |
|---|---|---:|---:|
| independent | live `e3v` | 0 / 226,637 | 0 m |
| independent | `vmask` | 0 / 399,600 | 0 |
| given NEMO's entry | live `e3v` | 0 / 226,637 | 0 m |
| given NEMO's entry | `vmask` | 0 / 399,600 | 0 |

The independent debt is not conflated with the geometry result. `vn_adv`
remains 919/8,589 unequal (maximum 24.341577728515905), and `zFv` remains
19,516/226,637 unequal (maximum 339,003.55099822127). Fold-band maxima are
0.0009604808130845299 PSU for S and 0.0013611107582029436 K for T. Thus
R238-P1 and P2 are **CONFIRMED**, while any stronger claim that geometry alone
closes the external-mode debt is **REFUTED**.

## ORCA2 and OMT trajectories

All before/after numerical artifacts are unchanged. OMT-1, OMT-2, OMT-3,
OMT-4, and independent rung 0 have byte-identical JSON files to round 237.
The shipped rung-10 JSON differs only in the stamped producer commit and
worktree root; every scientific value, row and verdict is identical. Hence
zero scored rows moved, no exact row was lost, the first debt did not move,
and no signed-zero or two-ULP count changed.

| gate | claim label | result |
|---|---|---|
| OMT-1, kt=1..8 | independent + given NEMO's entry | byte-identical |
| OMT-2, kt=1..10 | independent + given NEMO's entry | byte-identical |
| OMT-3, kt=1..10 | independent + given NEMO's entry | byte-identical |
| OMT-4, kt=1..10 | independent + given NEMO's entry | byte-identical |
| rung 0, kt=1..10 | independent | byte-identical |
| shipped rung 10, kt=1..10 | given NEMO's entry | numerical tree identical |

R238-P3 is **CONFIRMED**. OMT-4 also completes all 96 admitted independent
steps with an exact entry and no non-finite value or runtime refusal. Rung 0
again completes 99 steps and reaches the registered raw-mesh `e3w_int` guard
at step 100, not earlier. R238-P4 is **CONFIRMED**.

## Shared-card protections

GYRE's 210 ladder residual arrays are `np.array_equal` to round 237 and the
NPZ SHA-256 remains
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The fresh 360-day member is byte-identical at every 30-day snapshot. The
day-30/240/360 SHA-256 values remain
`3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
`2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
and `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`.
The pinned day-30/240/360 T RMS values therefore remain
`2.3432419318363155e-06`, `6.5816987106668941e-05`, and
`5.4077212586815052e-05 K`.

The fresh DINO CPU/fp64 month completes all 960 steps. Its day-30 wet 3-D T
RMS is unchanged at `2.056821682e-03 K`, below the fixed
`2.244317642e-03 K` bar; the snapshot digest is the same
`d2e254e5e9d3f9dbddc53c4c9676ea2d992af232fda58ac51cfc1710cbd46b55`.
The final generic/OVERFLOW route rerun passes 3/3, and the complete card
constructibility/recipe subset passes inside the focused battery. R238-P5 is
**CONFIRMED**.

## Validation and controls

The final focused production-route, raw-operand, recipe and card battery
passes 66/66. The production-route, face-thickness and fold-mask plants all
refuse. The default citation gate passes with zero unmapped citations,
failures or map-audit errors after the required SequenceMatcher re-anchoring.

The once-only `tests/ocean/fidelity -n 12` battery reports 3,149 passed,
seven skipped and ten failed. Six non-registered failures were repaired and
pass in isolated reruns: the generic route, two OVERFLOW controls, two stale
Decision-115 helper calls, and round 228's stale citation anchor. The four
remaining failures are registered pre-existing reds: the GYRE round-129
record stamp, allow-dirty escape-scope ratchet, worktree-stamp ratchet, and
SI3 scalar-math provenance. No round-238 test is red.

Independent review was attempted with `codex exec --sandbox read-only` and
failed before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. Independent review unavailable
in-sandbox; the exact output is retained in `independent_review.log`.

## Preregistered predictions

| prediction | disposition |
|---|---|
| R238-P1 source and registry | **CONFIRMED** |
| R238-P2 exact live `e3v` and `vmask` | **CONFIRMED**; external-mode closure claim **REFUTED** |
| R238-P3 trajectory registration | **CONFIRMED**, zero numerical rows moved |
| R238-P4 month boundaries | **CONFIRMED** |
| R238-P5 preserved cards | **CONFIRMED** |
| R238-P6 implementation integrity | **CONFIRMED**, with four registered battery reds and review unavailable in-sandbox |

## OPEN

Round 239 re-scores the complete round-217 unit with Decisions 115 and 114
both landed. It first tests whether the Ve_rhs fold pair closes the remaining
35-face completed V-RHS debt with raw `e3v`, then measures the unit atomically
on OMT-4 and rung 0. The independently measured `vn_adv`/`zFv` debt remains
open and must not be described as part of this geometry landing.

No acquisition is needed. No configuration or sea-ice decision is open.
