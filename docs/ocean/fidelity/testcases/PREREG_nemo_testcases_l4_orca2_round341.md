# ORCA2 round 341 preregistration — full current-tip re-score, source-order walk, and first non-bit owner

Frozen 2026-10-27 before running the round-341 current-tip ladder re-score or
committing the round-341 offline source-order walk gate.
Base: `a91d673b6fcf` (`a91d673b6fcff38b097f7c0fc8ddb009c05d7a32`, round 336
landed).  This round changes no configuration value, lands no model code, and
never runs NEMO.

## Measurements

- Re-score all 200 `(kt, checkpoint, field)` rows (`kt = 1..10`, `entry`,
  `stage1`, `stage2`, `stage3`, full depth `k = 0..29` for `T, S, u, v` plus
  2-D `ssh`) for both rung-10 comparison-card labels (`given_nemo_entry` and
  `independent`), together with the other six Decision-122 pairs (`omt0/independent`
  [200 rows], `rung0/independent` [200 rows], `omt1/independent` [160 rows],
  `omt2/independent` [200 rows], `omt3/independent` [200 rows], and
  `omt4/independent` [200 rows]), totaling 1,560 rows at current tip
  `a91d673b6fcf` under production JIT on CPU (`fp64`, `libm`).
- Classify each row as `BIT_EXACT` when every binary64 word is bitwise equal,
  `AT_BAR_NOT_EXACT` when `unequal > 0` and
  `max_abs / max(max_abs(oracle), 1.0) <= 1.0e-15`, and `OVER_BAR` otherwise.
  Publish the full 1,560-row table (`max_abs`, `rms`, `unequal`, `count`,
  `oracle_max_abs`, `normalization_scale`, `normalized_max_abs`,
  `classification`) and the full 200-row rung-10 table augmented with both the
  2-D surface-mask wet audit (`r333_wet._native_masks`) and the true 3-D
  native-mask (`tmask`, `umask`, `vmask`, `ssmask`) wet/dry census.
- Name the first non-bit row per label under both:
  1. the raw binary64 score (`unequal > 0` across the scored array), and
  2. the 3-D wet-cell score (`unequal_3d_wet > 0` where 3-D `tmask`, `umask`,
     `vmask`, or 2-D `ssmask` is active) and wet-value score (`unequal_wet > 0`
     with non-zero numeric residual).
- Walk both the first raw non-bit row (`kt = 1, stage1, v`) and the first
  3-D-wet non-bit / first wet-value row (`kt = 10, stage2, T`) in executing
  NEMO source order using offline replay from the admitted
  production-arithmetic records only (`round 5` full-entry/stage/barotropic/surface
  records, `round 326` `hm_i`/stage records, `round 328` ZDF-terms records, and
  `round 336` `kt = 2` `tra_zdf` and `TKE` records; treat the `round 328` `dyn`
  walk, `round 331/332` `TKE`/source-order/raw-`V` records, and `round 333`
  stage-3 tracer records as provenance-flagged `PLAUSIBLE` per the round-336
  producer audit).
- If a single source-cited statement (`record-build ppsrc` citation + `legoESM`
  line) is `CONFIRMED` on the admitted production-arithmetic records, stop with
  the non-landed fix `PROPOSAL` and its verification on the recorded inputs
  across `kt = 1..10` (do not land model code in `packages/` this round).  If a
  discriminating NEMO record is missing, design the extension of the reviewed
  round-336 acquisition tools (observers outside loops, pristine census) and
  stop with `ACQUISITION_NEEDED`.

## Predictions and falsifiers

- **R341-P1 — Full 1,560-row current-tip census.**  Across the eight pairs
  (1,560 rows total), the current tip `a91d673b6fcf` produces `473 BIT_EXACT`,
  `237 AT_BAR_NOT_EXACT`, and `850 OVER_BAR` rows:
  - `omt0/independent`: `16 BIT_EXACT`, `23 AT_BAR_NOT_EXACT`, `161 OVER_BAR`
    (first non-bit: `kt = 1, entry, T`, `unequal = 151,917`, `max_abs = 0.0`,
    `rms = 0.0`, `AT_BAR_NOT_EXACT`).
  - `rung0/independent`: `11 BIT_EXACT`, `10 AT_BAR_NOT_EXACT`, `179 OVER_BAR`
    (first non-bit: `kt = 1, stage1, T`, `unequal = 151,917`, `max_abs = 0.0`,
    `rms = 0.0`, `AT_BAR_NOT_EXACT`).
  - `omt1/independent`: `48 BIT_EXACT`, `33 AT_BAR_NOT_EXACT`, `79 OVER_BAR`
    (first non-bit: `kt = 1, entry, T`, `unequal = 151,917`, `max_abs = 0.0`,
    `rms = 0.0`, `AT_BAR_NOT_EXACT`).
  - `omt2/independent`: `48 BIT_EXACT`, `33 AT_BAR_NOT_EXACT`, `119 OVER_BAR`
    (first non-bit: `kt = 1, entry, T`, `unequal = 151,917`, `max_abs = 0.0`,
    `rms = 0.0`, `AT_BAR_NOT_EXACT`).
  - `omt3/independent`: `36 BIT_EXACT`, `25 AT_BAR_NOT_EXACT`, `139 OVER_BAR`
    (first non-bit: `kt = 1, entry, T`, `unequal = 151,917`, `max_abs = 0.0`,
    `rms = 0.0`, `AT_BAR_NOT_EXACT`).
  - `omt4/independent`: `12 BIT_EXACT`, `15 AT_BAR_NOT_EXACT`, `173 OVER_BAR`
    (first non-bit: `kt = 1, entry, T`, `unequal = 151,917`, `max_abs = 0.0`,
    `rms = 0.0`, `AT_BAR_NOT_EXACT`).
  - `rung10/given_nemo_entry` and `rung10/independent`: row-for-row identical
    on all 200 rows, each with `151 BIT_EXACT`, `49 AT_BAR_NOT_EXACT`, and
    `0 OVER_BAR`.
  Any missing row, dirty worktree, or differing count refutes `R341-P1`.
- **R341-P2 — Rung-10 first non-bit row (raw vs 3-D wet) and 48 dry-velocity rows.**
  On both rung-10 labels:
  - The first raw non-bit row (and first 2-D surface-mask non-bit row) is
    `kt = 1, stage1, v` (`unequal = 7,534`, `max_abs = 0.0`, `rms = 0.0`,
    `AT_BAR_NOT_EXACT`).  All 48 non-bit `u`/`v` rows (`kt = 1..10 stage1/2/3 v`
    [30 rows], `kt = 2..10 entry u` [9 rows], `kt = 2..10 entry v` [9 rows])
    have `max_abs = 0.0`, `rms = 0.0`, and `unequal_3d_wet = 0`: every unequal
    word lies on a 3-D dry face (`umask == 0` or `vmask == 0`; at
    `kt = 1, stage1, v`: `5,550` sub-bottom partial-step dry faces with
    `ssvmask == 1, vmask == 0` plus `1,984` surface-column dry faces with
    `ssvmask == 0, vmask == 0`).
  - The first (and only) 3-D-wet non-bit row and first wet-value row across all
    200 rows on both labels is `kt = 10, stage2, T` (`unequal = 1`,
    `unequal_3d_wet = 1`, `unequal_3d_dry = 0`,
    `max_abs = 2.220446049250313e-16` K = 1 ULP at `(j = 18, i = 34, k = 3,
    rank = 0)`, `rms = 3.5125902029716364e-19` K, `AT_BAR_NOT_EXACT`).
  Any nonzero 3-D-wet difference on the 48 `u`/`v` rows or any other non-bit
  3-D-wet row refutes `R341-P2`.
- **R341-P3 — Source-order walk of `kt = 1, stage1, v`.**  From the admitted
  production-arithmetic `round 5` and `round 326` records, all `226,637`
  rank-0 3-D-wet `v` faces at `kt = 1, stage1` are bit-exact (`0` unequal), and
  all `7,534` unequal words against `oracle_stage_kt00000001_s1.bin` occur on
  3-D-dry faces (`vmask == 0`) where NEMO's stage-dump canonicalizer
  `l4_canon_3d` (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:479-525`;
  `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:846-892`)
  initializes `zfield = +0.0_wp` and copies `pfield` only where
  `vmask(jif,jjf,jk) /= 0._wp` (`CONFIRMED` on `round 5` / `round 326`; the
  live dry-sign comparison against the round-332 raw post-fold `V` record is
  reported as `PLAUSIBLE` per the round-336 producer audit).
- **R341-P4 — Source-order walk of `kt = 10, stage2, T` and single owning statement (`CONFIRMED`).**
  Offline source-order replay of `stp_RK3_stg(2)` at `kt = 10` from the
  admitted production-arithmetic `round 5` and `round 326` records confirms
  that a single source-cited statement owns the 1-ULP departure at
  `(j = 18, i = 34, k = 3, rank = 0)`:
  - Record-build `ppsrc` citation:
    `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:177`
    (storing `r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)` at Stage 1,
    swapped into `Kmm` before Stage 2 at
    `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:209`, and
    expanding `e3t(ji,jj,jk,Kmm)` in Stage-2 centered tracer advection at
    `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:158-160,219-220,226-227`;
    identical in `ORCA2_ORCA1ICE_OMIP_L4_R326HMI`).
  - `legoESM` line:
    `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2293-2295`
    (invoked for Stage 2 at `:9570-9572` with `_h_live_one_third` instead of
    `_h_s2_kmm`, and passed at `:2665,2706` into `_nemo_cen2_tracer_rhs` at
    `:2129-2130,2143`).
  - Verification on recorded inputs across `kt = 1..10`: using the stored
    Stage-2 `Kmm` thickness `_h_s2_kmm = nemo_source_round(_h_ref_ws * _qt_13[..., None])`
    (`ocean_model_latlon_cgrid.py:8820-8823`) in Stage-2 `_nemo_cen2_tracer_rhs`
    changes `e3t_kmm[18,34,3]` at `kt = 10` from `0x40234d4bd6440686`
    (`9.650969215206043`) to `0x40234d4bd6440687` (`9.650969215206045`),
    `Krhs[18,34,3]` from `0xbee1c450c2a60ef6` to `0xbee1c450c2a60ef5`, and
    `T_stage2[18,34,3]` from `0xbff663da73f853c7` (`-1.399378254897998`) to
    `0xbff663da73f853c6` (`-1.3993782548979978`), matching NEMO's
    `oracle_stage_kt00000010_s2.bin` bit for bit (`0` unequal words across all
    `228,641` rank-0 interior 3-D-wet `T` and `S` cells at every step
    `kt = 1..10`, and `200 / 200` 3-D-wet `BIT_EXACT` rows on rung 10).
  Any remaining interior 3-D-wet `T` or `S` mismatch at Stage 2 across
  `kt = 1..10` under the stored `e3t(Kmm)` replay refutes `R341-P4`.
