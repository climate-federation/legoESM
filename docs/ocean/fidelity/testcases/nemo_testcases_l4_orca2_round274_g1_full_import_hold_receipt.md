# ORCA2 round 274 — certified G1 import held by the Decision-96 census

Date: 2026-10-13

Lane base: `c0638af25a84dd4d522374c918dc3a967fd54f73`

Labels: **independent** and **given NEMO's entry**.  They are never mixed in
one score.

## Verdict

**HELD.**  The complete thirteen-statement G1 activation is not a net
improvement on independent rung 0: among the 200 RMS-scored rows, 28 move
toward NEMO, 165 move away and 7 are score-equal.  No exact row is lost and
the first non-bit row moves toward, but the strict RMS-majority predicate in
Decision 96 is false.  The final kt=10 stage-3 SSH maximum moves from
`0.3840334668286476` to `8.389195250067887 m`.

The imported implementations, the certified G1 card, and its record tooling
remain in the tree.  Every new selector is false by default.  The production
ORCA2, OMT-1..4 and shipped-rung cards leave all thirteen round-274 selectors
off; the private arm in the round-274 gate is their only activation outside
the G1 card.  Thus no ORCA2 physics selection lands in this round.

Sea ice is unchanged.  In particular, no ice selector and no ORCA2
`unmeasured_features` entry was edited.

## Frozen predictions

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round274.md`, committed as
`dda822c5b` before the measurements.

| prediction | result | evidence |
|---|---|---|
| R274-P1: all thirteen certified statements import once and remain explicit | **CONFIRMED for the implementation and G1 card; HELD on ORCA2 selection** | imported grid-ladder tip `92435ddbde02dd6f6a3f44ec3ae36ba38f60b2dd`; focused support tests |
| R274-P2: strict RMS majority toward on every claimed ORCA2/OMT ladder | **REFUTED at the first required ladder, independent rung 0** | `round274/rung0_control.json`, `rung0_candidate.json`, `rung0_decision96_final.log` |
| R274-P3: shared-card controls unchanged | **CONFIRMED for the safe final tree's GYRE ladder/day 30; remainder off-claim after P2 failed** | `gyre_compare.json`, `gyre_ladder.residuals.npz`, `gyre30.log` |
| R274-P4: re-rank the first ORCA2 boundary after landing | **UNMEASURED-with-spec** | activation did not land; a boundary on the rejected arm is not a production claim |
| R274-P5: citation/focused/push gates green | **CONFIRMED except the registered full-fidelity resource stall** | citation audit 274/274; focused results below; push battery below |

The exact-loss control fires:
`STATUS PLANT-FIRED: previously exact rows left the bar: [(1, 'entry', 'T')]`.

## Imported cited unit

The implementation is the grid-ladder's landed code, not a new derivation.
The thirteen selectors are:

1. `barotropic_een_ratio_fold_exchange`;
2. `barotropic_een_fe3mask_pivot_row`;
3. `rk3_stage1_rhs_nemo_literal`;
4. `rk3_stage_velocity_written_order`;
5. `een_vorticity_written_order`;
6. `zdf_barotropic_removal_nemo_literal`;
7. `rk3_fold_exchange_top_v_row`;
8. `rk3_stage_r3f_nemo_literal`;
9. `barotropic_een_ratio_written_order`;
10. `nemo_r3t_stored_product`;
11. `rk3_stage_r3t_nemo_literal`;
12. `nemo_eos_gdept_rounded_product`;
13. `rk3_stage_r3uv_nemo_literal`.

The compiled G1 source is
`cfgs/ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo`.  It writes the stage-two and
stage-three F ratios at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:218`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:239`, and
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546`; constructs the stage residual at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414` and adds it at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:417-418`; exchanges completed stage state at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:572-573`; uses the live
partial-cell U/V thickness in the relative circulation at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:766-767`; and applies the
written EEN triads at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:779-784`.  The compiled
vertical update and barotropic removal are at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:145-150` and
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:160-161`.
The imported grid-ladder round-6 through round-9 receipts retain the other
leaf citations and their G1 bit-exact controls.

Card census on the safe final tree:

| card family | thirteen selectors | execution claim |
|---|---:|---|
| `ORCA2G1-flat` | ON | certified G1 program, 100 steps |
| OMT-1..4 private round-274 arm | ON | measurement arm only |
| rung 0 private round-274 arm | ON | measurement arm only; rejected |
| production OMT-1..4, rung 0 and shipped ORCA2 | OFF | retained production path |
| GYRE, DINO, tanks and generic cards | OFF | defaults retained |

## Decision-96 measurement

The same-tree control disables only the thirteen selectors.  The candidate
enables all thirteen.  Both use the corrected independent rung-0 entry and
the admitted rank-complete record.

| quantity | control | complete G1 arm |
|---|---:|---:|
| RMS census toward / away / equal | — | **28 / 165 / 7** |
| max census toward / away / equal | — | **42 / 142 / 16** |
| exact-row losses | — | **0** |
| first non-bit row | kt=1 stage-1 T, RMS `1.1745202172532108e-05 K` | same row, `1.1745202172532015e-05 K` (toward) |
| kt=10 stage-3 SSH RMS / max | `0.026832591469308 / 0.3840334668286476 m` | `0.07883665867179786 / 8.389195250067887 m` |
| kt=10 stage-3 T RMS / max | `0.005931985294097096 / 0.8634177027234093 K` | `6.721841925555034e7 / 3.4644277058446556e10 K` |
| kt=10 stage-3 S RMS / max | `0.0014564401561608952 / 0.4156931329393778 PSU` | `1.1779563120395916e9 / 6.411368489681729e11 PSU` |
| kt=10 stage-3 u RMS / max | `0.0044017065072820966 / 0.35916459073435314 m s-1` | `1.3707218333889734e12 / 5.819603234610368e14 m s-1` |
| kt=10 stage-3 v RMS / max | `0.004314463377274061 / 0.514763697406635 m s-1` | `1.6925823896659463e12 / 5.888163902400892e14 m s-1` |

This falsifies R274-P2 before OMT or shipped-rung landing claims can be made.
The already-produced OMT-1..4 candidate reports all pass their internal
ladder completeness gates, but their D96 comparisons and the rung-10
candidate are deliberately off-claim after the first required predicate
failed.

## Controls and gates

- Safe final-tree GYRE: 70/70 certified rows unchanged, first-over-bar step 3
  unchanged, residual archive SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5`.
  Day-30 snapshot is byte-identical to round 270, SHA256
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- Citation gate: PASS, 274 citations, zero failed map entries and zero
  unmapped citations.  All nine self-tests, including shifted and widened
  anchors, fire or pass as intended.
- G1 card/record support: 82/83 passed on the combined focused run; the one
  paired-patch whitespace control passed in isolation after normalising both
  committed patch artifacts.  The four combined-lane selector-count controls
  pass after requiring exactly the G1 and round-270 ORCA2 sites.
- Required `tests/ocean/fidelity -n 12` was launched once.  It reached 99%
  with the known resource-failure pattern and a worker stall; it was not run
  a second time.  The isolated G1 support failures were dispositioned above.
- Independent review: **independent review unavailable in-sandbox**.  The
  read-only `codex exec` failed before review with `failed to initialize
  in-process app-server client: Read-only file system`.

## Provenance

Evidence lives in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round274/`.
Key SHA256 values:

- rung-0 control: `20392ee8aa5e39e2f7706de1e980a7053e2b8e4c877a8e0ed6fba1dbcf70ae7b`;
- rung-0 candidate: `f4f17e338d5adec87b89c71927e17ad30b4f8d6f88f7302516a2a37663310709`;
- citation log: `31a5e11b934c7e6baadf89ff217ea6c06a6e79cc46706477a19ada3bf88554fc`.

## OPEN

Round 275 bisects the thirteen imported statements by compiled source unit on
the same independent rung-0 protocol, starting at kt=1 and never carrying the
explosive kt=10 state upstream as an owner.  Each arm must first reproduce its
own live completed boundary, then be scored by Decision 96.  The first unit
that changes the safe trajectory names the interaction; no selector is
promoted as part of a cancelling group without its own census.  If no single
unit causes the growth, test the smallest source-ordered pair that shares a
live ratio or stage consumer.  No NEMO acquisition is presently required.
