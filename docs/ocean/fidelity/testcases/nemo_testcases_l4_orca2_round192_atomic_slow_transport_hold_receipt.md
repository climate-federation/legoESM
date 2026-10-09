# ORCA2 round 192 — atomic slow-depth and V-transport hold

Date: 2026-10-09. Base `4191b3671`; preregistration `19aa48b48` plus
provenance correction `92d99fee8`; candidate `5dfe5e0f7`; restored tree
`4f1031438`. Status: **HELD**.

Every trajectory number is **independent hierarchy rung 0**. The private card
starts from its corrected, bit-exact initial state. No given-NEMO-entry rung-7
number is claimed. The shipped rung-10 card, sea ice, all six ice selectors,
configuration, forcing, carried state, stabilisers and `unmeasured_features`
tuple are unchanged.

## Result

The complete cited unit still makes rung 0 non-finite. Adding NEMO's exact
completed-RHS depth-average association to the already-held raw-depth,
unmasked/materialised V-transport and seven-array external-mode unit does not
remove its terminal boundary:

| run | last completed gate progress | first refusal |
|---|---|---|
| atomic candidate | kt=8 stage 3 executable returned; kt=8 stages 1–2 were exposed | scoring kt=8 stage-2 T: candidate is non-finite |
| restored production | all 40 checkpoints through kt=10 stage 3 | none; `PASS_RUNG0_TEN_STEP_LADDER` |

The terminal occurs before a complete 200-row candidate exists, so the unit
cannot satisfy Decision 96. No toward/away census is reported, and rung 7,
the month, GYRE, DINO and tanks are deliberately not run: none can reverse the
dispositive independent-rung-0 refusal.

The candidate implements NEMO's vector-invariant completed-RHS vertical
average at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219` and enables the
existing private arms for raw midpoint face depth at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, the
unmasked separately materialised V transport at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`, and the
seven-array association at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.

This is the same terminal class as rounds 155 and 164, now measured with the
round-190 exact slow-depth statement included. Therefore the slow-depth
association is **REFUTED as the missing compensating partner** for the atomic
V-transport unit. The candidate commit is retained in history as measurement
provenance and immediately reverted. `git diff 4191b3671..4f1031438 --
packages` is empty.

Machine evidence under `orca2_rounds/round192/`:

- `rung0_atomic_candidate.log`, SHA-256
  `13b7a3f116f1c836e0285a4ae28fe45739ecc2c936c2707811f3370b566e0c9a`;
- `rung0_restored_control.json`, SHA-256
  `6859faeeb8e4bda64625a78e68060abd1666cd96b0e1abb80557215e9e81b73c`;
- the candidate arithmetic control passed 1/1, transcript SHA-256
  `be05f2cf0a1e040c3b519ff669fa01f190607234ddf766d7448093e606328d5e`;
- the real entry-bit plant refused before measurement, transcript SHA-256
  `037244c9cc58f0832e1c183aaf5c95197cb2d2520508ecdc0541b2b89ba1a6d9`.

## Frozen prediction ledger

| prediction | result |
|---|---|
| R192-P1 combined candidate preserves every local exact prerequisite | **UNMEASURED_THIS_ROUND**: the ten-step gate does not emit local substep rows before its terminal refusal; rounds 146–154, 190 and 191 remain the separate admitted witnesses. |
| R192-P2 exact slow forcing removes the kt=8 terminal | **REFUTED**: kt=8 stage-2 T is non-finite. |
| R192-P3 Decision-96 rung-0 net improvement | **UNMEASURED_TERMINAL_R192-P2**: no complete candidate row set exists. |
| R192-P4 rung 7 remains executable with no exact loss | **UNMEASURED_TERMINAL_R192-P2**. |
| R192-P5 month advances beyond step 96 | **UNMEASURED_TERMINAL_R192-P2**. |
| R192-P6 controls remain non-vacuous | **PARTIAL**: candidate-arm dependency checks and the entry-bit plant bind; downstream census plants are unreachable without 200 candidate rows. |

## Validation, review and choices

The restored independent gate passes all 200 rows, retains bit-exact entry
T/S/u/v/SSH, keeps the first debt at kt=1 stage-1 T, and reproduces kt=10
stage-3 SSH maximum `0.42832517646246693 m`. The candidate arithmetic test
passes before measurement; the entry-bit plant exits 2 with
`STATUS PLANT-FIRED`.

No configuration choice was made. ASKED choices: the complete unit named by
round 191 and Decision 96. UNASKED choices: empty. No stabiliser or tolerance
was added. The final package tree is identical to the round base.

## OPEN

1. Reapply the same complete private unit and rerun the passive completed-stage
   growth table on the current landed-HPG tree. The next owner is the first
   stage whose error grows by more than ten, not the downstream kt=8 NaN.
2. At that first growth boundary, replay operators offline in compiled order
   from the completed passive state. No in-executable observer is permitted.
3. Keep the complete slow-depth/raw-depth/no-extra-V-mask/seven-array/
   materialised-V-transport unit private until a newly named partner makes the
   rung-0 ladder complete and pass Decision 96.
