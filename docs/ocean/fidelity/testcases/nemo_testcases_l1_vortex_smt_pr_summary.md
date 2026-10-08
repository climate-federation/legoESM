# VORTEX seamount fidelity — PR summary after PR #1869

This branch carries the VORTEX-with-topography work from rounds 218–246 on
top of PR #1869, plus the inert merge of pinned main `471bee222` certified in
round 247. It adds the explicit SMT-1 through SMT-4 mini-ladder, the admitted
NEMO records and permanent gates, four source-derived partial-cell
statements, and daily 100-day comparisons with movies. It does not claim that
SMT-4's remaining day-100 error has a unique process owner.

## What lands

| item | NEMO statement and local proof | trajectory result |
|---|---|---|
| partial-cell U/V face thickness | NEMO uses the shallower neighbour's reference thickness; the compiled-source proof and zero-ULP geometry audit are in `nemo_testcases_l1_vortex_smt_round4_face_thickness_landing_receipt.md` | SMT vector rows move strongly toward NEMO; existing cards are measured |
| EVD trigger and composition | NEMO evaluates its own `rn2` under the card's EOS and replaces `avt`/`avm` where the trigger fires; proof is in `nemo_testcases_l2_gyre_round221_evd_fix_receipt.md` | false below-bottom SMT-1 triggers disappear; the forced unstable-column plant is non-vacuous |
| implicit bottom-drag divisor | NEMO uses bottom `e3u_3d/e3v_3d` times the after-level stretch; proof is in `nemo_testcases_l1_vortex_smt_round11_drag_divisor_landing_receipt.md` | SMT-2 kt2 U falls from `2.193e-04` to `3.179e-09` |
| isoneutral tracer-LDF mask and live divisor | the closed deepest W mask and live stage-3 `Kmm` thickness are compiled at `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`, `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`, and `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331` | 30/34 moved aggregate rows improve; the disclosed near-zero ratchet remains registered under Decisions 95/96 |

Each implementation is shared where NEMO's branch is shared and is selected
explicitly where NEMO's program differs. No stabiliser absent from NEMO and no
silent library-default choice is introduced.

## Mini-ladder and long-run evidence

The vector seamount ladder adds one ORCA2-facing module per rung:

| rung | added module | day-100 wet T RMS vs NEMO [K] |
|---|---|---:|
| SMT-1 | background vertical mixing + EVD | `4.3321114781972461e-05` |
| SMT-2 | implicit linear bottom drag | `8.1037591477894766e-06` |
| SMT-3 | isoneutral lateral tracer diffusion | `1.7729713625071864e-04` |
| SMT-4 | level Laplacian lateral momentum diffusion | `2.5527080520554426e-04` |

Each rung has an explicit card, admitted NEMO kt=1–10 and daily 100-day
records, a 50-row registry, a daily table, and a movie. The round-247 pinned
main merge reproduces these four endpoints exactly. GYRE remains at its
round-237 certified year values, including day 30
`2.3432419318363155e-06`, day 240 `6.5816987106668941e-05`, and day 360
`5.4077212586815052e-05 K`; its full 954-row ladder is unchanged.

## ORCA2 hand-off

The receipts carry explicit ORCA2 pointers for the four shared statements.
The ORCA2 lane has already measured the face-thickness statement after its
merge; it must select and remeasure the EVD composition, bottom-drag divisor,
and tracer-LDF pair under its own rung-0 card rather than infer their effects
from VORTEX. The live FCT limiter itself was bit-faithful given NEMO's
stage-3 transports; the recorded ORCA2 step-36 pointer is its guarded V-face
coefficient, not a VORTEX-derived replacement.

## Registered debt

- SMT-3's accepted Decision-96 landing retains 40 cellwise two-ULP ratchet
  violations in near-zero cells; 34 aggregate rows move, 30 toward and four
  away, with no certified row lost.
- The same landing recorded 46 GYRE cellwise ratchet rows while leaving all
  954 aggregate ladder rows and first-over-bar unchanged.
- SMT-4's day-100 T RMS is `2.5527080520554426e-04 K`. Turning off any single
  ranked process family made it worse, so HPG, momentum LDF, tracer LDF,
  bottom drag, vertical mixing/EVD, and the barotropic path do not provide a
  unique owner. The stage-3 HPG association is locally source-order exact but
  trajectory-inert. This remains debt, not an attribution claim.

## PR gate result

Pinned main is merged at `cd1dcf7225423e3d60720de7c3b50966edcd65a5` with
zero conflicts and no lane/main overlap under `packages/ocean` or
`src/legoesm`. The merge audit also tested main's reachable shared
tridiagonal-scan change through the long trajectories.

- GYRE: 954/954 ladder rows and all 40 year score values unchanged; pinned
  day-30/day-240/day-360 snapshot digests unchanged.
- SMT-1..4: all four 50-row registries reproduced; 3,200/3,200 daily score
  scalars unchanged; the four day-100 endpoints above reproduced exactly.
- Controls: the six flat VORTEX cards, two base seamount cards, LOCK_EXCHANGE,
  and OVERFLOW each reproduce all 50 rows with zero cell movement.
- DINO: day-30 T RMS `2.056821682e-03 K`, exact to the lane reference and
  below the fixed bar.
- Tests: focused merge suite `105 passed, 16 skipped`; push-gate battery
  `126 passed`.
- Citations: PR-summary gate passes with zero unmapped claims; its shifted
  citation plant exits nonzero.

The detailed command provenance and decisive lines are in
`nemo_testcases_l2_gyre_round247_main_pin_merge_pr_prep_receipt.md` and
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round247/`.
