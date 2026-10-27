# ORCA2 round 341 — current-tip 8-pair re-score, 3-D wet audit, and Stage-2 `e3t(Kmm)` source-order walk

Date: 2026-10-27

Base: `a91d673b6fcf` (`a91d673b6fcff38b097f7c0fc8ddb009c05d7a32`, Round 336 landed); preregistration: `5fae40ddd` (`5fae40dddf5a6f280448a5b326c8c3df5f5c6335`, `docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round341.md`); rescore and source-order walk gate: `a104ed2a4` (`a104ed2a4e79a6f3a5704d221b9718113e846bc7`).
Disposition: **HELD (STOP WITH CONFIRMED SINGLE-STATEMENT FIX PROPOSAL AND VERIFICATION ON RECORDED INPUTS; NO MODEL CODE LANDED THIS ROUND)**.
independent review: operator.

This round changes no model code in `packages/`, changes no configuration value, and never builds or runs NEMO.

## 1. Current-tip 8-pair re-score and first non-bit row per label

Measured on clean committed worktree `a104ed2a4e79a6f3a5704d221b9718113e846bc7` under CPU `fp64` (`libm`, production JIT) with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:407-502` and aggregated with `:505-655` (unit controls in `tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py:22-114`; self-test artifact `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/selftest.json`, SHA-256 `79a0b32a2f3b5e4d52d6ed8ef857fc038b9ec2ae1c7610bbe46e8ce68ef7cdfb`; all four planted controls `signed-zero`, `wet-3d-mask`, `stage2-thickness`, and `walk-closure` fire `STATUS PLANT-FIRED` with exit code `2`).

- Full 1,560-row table across all eight pairs: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/full_ladder.tsv` (SHA-256 `9e76cffa7c55b5db294822764abb647825771f57cc2258c30d2471a616784170`).
- Full 400-row (`2 x 200`) Rung-10 3-D and 2-D wet/dry audit table: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/rung10_ladder_3d_wet.tsv` (SHA-256 `cbf70db6b5e0a25987c2ac8a0ea85a99c7a04b93566258e2fbfecdedccb6b5b4`).
- Aggregate summary JSON: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/aggregate.json` (SHA-256 `ca6e10fdece3ddb9ba08a651aa7996f5e37fd454ad6d92faf61745126f36fd21`).

### 1.1 Eight-pair classification census and first raw non-bit row (`R341-P1` CONFIRMED)

| Pair (`rung / label`) | `BIT_EXACT` | `AT_BAR_NOT_EXACT` | `OVER_BAR` | Rows | First raw non-bit row `(kt, checkpoint, field)` | `unequal` | `max_abs` | `rms` | First row class | Pair JSON SHA-256 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| `omt0/independent` | 16 | 23 | 161 | 200 | `(1, entry, T)` | 151,917 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `d70d09d3526764e195896dc979358b4d4fd98e306ce5238a5b41d680ecd7d76a` |
| `rung0/independent` | 11 | 10 | 179 | 200 | `(1, stage1, T)` | 151,917 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `f2b88943ac49b3ed76a6e3d3318f73e22a6687c892db377f4c50abcf50771921` |
| `omt1/independent` | 48 | 33 | 79 | 160 | `(1, entry, T)` | 151,917 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `f32e81d760a12c7ff31d2a7f039872de181d4b692160e5e025a731207524b9c8` |
| `omt2/independent` | 48 | 33 | 119 | 200 | `(1, entry, T)` | 151,917 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `43d4e00ccc3f4fbe93f2566257cc465ac621c98bfdfc3a1dc22883125f6e9397` |
| `omt3/independent` | 36 | 25 | 139 | 200 | `(1, entry, T)` | 151,917 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `be6770ee09a115d20104b7f601173981d0f9b74fd12708cf752a0057b7660332` |
| `omt4/independent` | 12 | 15 | 173 | 200 | `(1, entry, T)` | 151,917 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `c53618a448df52576d6dad88b2cd914473a0ea0257a2dbc8e960faa6fcd90f5e` |
| `rung10/given_nemo_entry` | 151 | 49 | 0 | 200 | `(1, stage1, v)` | 7,534 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `741cbe5e1924cc2f6b4dcfac84dcfc64dc4fbac45103445378626486fb32d13f` |
| `rung10/independent` | 151 | 49 | 0 | 200 | `(1, stage1, v)` | 7,534 | `0.0e+00` | `0.0e+00` | `AT_BAR_NOT_EXACT` | `a8f01b301363d4a1b3120bdcf6d836f7c28ac6fb374a1a59d9922a8bd49d4d58` |
| **Total** | **473** | **237** | **850** | **1,560** | — | — | — | — | — | — |

### 1.2 Rung-10 first non-bit row under raw vs 3-D wet scoring (`R341-P2` CONFIRMED)

`rung10/given_nemo_entry` and `rung10/independent` are row-for-row identical across all 200 rows (`rung10_labels_row_for_row_identical = true`):

1. **First raw non-bit row (and first 2-D surface-mask non-bit row)**: `kt = 1, stage1, v` (`unequal = 7,534` out of `399,600` rank-0 words, `max_abs = 0.0`, `rms = 0.0`, `AT_BAR_NOT_EXACT`). Across all 200 rows of Rung 10, all 48 non-bit `u`/`v` rows (`kt = 1..10 stage1/2/3 v` [30 rows], `kt = 2..10 entry u` [9 rows], `kt = 2..10 entry v` [9 rows]) have `max_abs = 0.0`, `rms = 0.0`, `wet_3d_bit_unequal = 0`, and `unequal_3d_wet = 0`: every single unequal word lies on a 3-D dry face (`umask == 0` or `vmask == 0`). At `kt = 1, stage1, v`, all `226,637` rank-0 3-D wet `v` faces (`vmask != 0`) are bit-exact (`0` unequal), while the `7,534` dry `-0.0` vs `+0.0` words partition into `5,550` sub-bottom partial-step dry faces (`ssvmask == 1, vmask == 0`) and `1,984` surface-column dry faces (`ssvmask == 0, vmask == 0`).
2. **First (and only) 3-D-wet non-bit row and first wet-value row**: `kt = 10, stage2, T` (`unequal = 1`, `wet_3d_bit_unequal = 1`, `unequal_3d_wet = 1` out of `233,341` rank-0 3-D wet `T` cells, `unequal_3d_dry = 0`), located at `(j = 18, i = 34, k = 3, rank = 0)` (`coast = true`, `under_ice = true`, `fold = false`, `seam = false`, `ice_fraction = 0.8657079645688692`), with `max_abs = 2.220446049250313e-16` K (1 ULP), `rms = 3.5125902029716364e-19` K, `wet_3d_rms = 4.596684518556674e-19` K, and `wet_3d_normalized_max_abs = 7.349646833843343e-18 <= 1.0e-15` (`AT_BAR_NOT_EXACT`). Scored on true 3-D wet cells (`tmask`, `umask`, `vmask`, `ssmask`), Rung 10 is **`199 / 200` `BIT_EXACT`, `1 / 200` `AT_BAR_NOT_EXACT`, `0 / 200` `OVER_BAR`**.
3. **First `OVER_BAR` row on Rung 10 (`kt = 1..10`)**: **`None`** (`0 / 200` `OVER_BAR` rows under both raw and 3-D wet scoring).

### 1.3 Full 200-row Rung-10 comparison card table (`given_nemo_entry` and `independent`, identical)

| `kt` | `checkpoint` | `field` | `max_abs` | `rms` | `unequal` | `count` | `wet_3d_count` | `wet_3d_bit_unequal` | `unequal_3d_wet` | `unequal_3d_dry` (`subbottom` + `surface`) | Raw class | 3-D wet class |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `entry` | `u` | `0.0` | `0.0` | 0 | 799,200 | 413,030 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `entry` | `v` | `0.0` | `0.0` | 0 | 799,200 | 415,175 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage1` | `v` | `0.0` | `0.0` | 7,534 | 399,600 | 226,637 | 0 | 0 | 7,534 (5,884 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 1 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage2` | `v` | `0.0` | `0.0` | 7,969 | 399,600 | 226,637 | 0 | 0 | 7,969 (6,319 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 1 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 1 | `stage3` | `v` | `0.0` | `0.0` | 8,134 | 399,600 | 226,637 | 0 | 0 | 8,134 (6,484 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 1 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 2 | `entry` | `v` | `0.0` | `0.0` | 16,756 | 799,200 | 415,175 | 0 | 0 | 16,756 (13,396 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 2 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage1` | `v` | `0.0` | `0.0` | 10,789 | 399,600 | 226,637 | 0 | 0 | 10,789 (9,139 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 2 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage2` | `v` | `0.0` | `0.0` | 10,697 | 399,600 | 226,637 | 0 | 0 | 10,697 (9,047 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 2 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 2 | `stage3` | `v` | `0.0` | `0.0` | 9,621 | 399,600 | 226,637 | 0 | 0 | 9,621 (7,971 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 2 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 3 | `entry` | `v` | `0.0` | `0.0` | 19,057 | 799,200 | 415,175 | 0 | 0 | 19,057 (15,697 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 3 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage1` | `v` | `0.0` | `0.0` | 9,880 | 399,600 | 226,637 | 0 | 0 | 9,880 (8,230 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 3 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage2` | `v` | `0.0` | `0.0` | 9,458 | 399,600 | 226,637 | 0 | 0 | 9,458 (7,808 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 3 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 3 | `stage3` | `v` | `0.0` | `0.0` | 9,592 | 399,600 | 226,637 | 0 | 0 | 9,592 (7,942 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 3 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 4 | `entry` | `v` | `0.0` | `0.0` | 19,064 | 799,200 | 415,175 | 0 | 0 | 19,064 (15,704 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 4 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage1` | `v` | `0.0` | `0.0` | 8,905 | 399,600 | 226,637 | 0 | 0 | 8,905 (7,255 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 4 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage2` | `v` | `0.0` | `0.0` | 9,297 | 399,600 | 226,637 | 0 | 0 | 9,297 (7,647 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 4 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 4 | `stage3` | `v` | `0.0` | `0.0` | 9,099 | 399,600 | 226,637 | 0 | 0 | 9,099 (7,449 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 4 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 5 | `entry` | `v` | `0.0` | `0.0` | 18,843 | 799,200 | 415,175 | 0 | 0 | 18,843 (15,483 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 5 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage1` | `v` | `0.0` | `0.0` | 10,135 | 399,600 | 226,637 | 0 | 0 | 10,135 (8,485 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 5 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage2` | `v` | `0.0` | `0.0` | 10,335 | 399,600 | 226,637 | 0 | 0 | 10,335 (8,685 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 5 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 5 | `stage3` | `v` | `0.0` | `0.0` | 10,119 | 399,600 | 226,637 | 0 | 0 | 10,119 (8,469 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 5 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 6 | `entry` | `v` | `0.0` | `0.0` | 19,524 | 799,200 | 415,175 | 0 | 0 | 19,524 (16,164 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 6 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage1` | `v` | `0.0` | `0.0` | 9,948 | 399,600 | 226,637 | 0 | 0 | 9,948 (8,298 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 6 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage2` | `v` | `0.0` | `0.0` | 9,203 | 399,600 | 226,637 | 0 | 0 | 9,203 (7,553 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 6 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 6 | `stage3` | `v` | `0.0` | `0.0` | 8,614 | 399,600 | 226,637 | 0 | 0 | 8,614 (6,964 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 6 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 7 | `entry` | `v` | `0.0` | `0.0` | 18,271 | 799,200 | 415,175 | 0 | 0 | 18,271 (14,911 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 7 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage1` | `v` | `0.0` | `0.0` | 10,012 | 399,600 | 226,637 | 0 | 0 | 10,012 (8,362 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 7 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage2` | `v` | `0.0` | `0.0` | 9,717 | 399,600 | 226,637 | 0 | 0 | 9,717 (8,067 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 7 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 7 | `stage3` | `v` | `0.0` | `0.0` | 9,285 | 399,600 | 226,637 | 0 | 0 | 9,285 (7,635 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 7 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 8 | `entry` | `v` | `0.0` | `0.0` | 18,586 | 799,200 | 415,175 | 0 | 0 | 18,586 (15,226 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 8 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage1` | `v` | `0.0` | `0.0` | 11,393 | 399,600 | 226,637 | 0 | 0 | 11,393 (9,743 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 8 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage2` | `v` | `0.0` | `0.0` | 11,034 | 399,600 | 226,637 | 0 | 0 | 11,034 (9,384 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 8 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 8 | `stage3` | `v` | `0.0` | `0.0` | 11,004 | 399,600 | 226,637 | 0 | 0 | 11,004 (9,354 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 8 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 9 | `entry` | `v` | `0.0` | `0.0` | 19,902 | 799,200 | 415,175 | 0 | 0 | 19,902 (16,542 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 9 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage1` | `v` | `0.0` | `0.0` | 9,610 | 399,600 | 226,637 | 0 | 0 | 9,610 (7,960 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 9 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage2` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage2` | `v` | `0.0` | `0.0` | 9,111 | 399,600 | 226,637 | 0 | 0 | 9,111 (7,461 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 9 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 9 | `stage3` | `v` | `0.0` | `0.0` | 9,684 | 399,600 | 226,637 | 0 | 0 | 9,684 (8,034 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 9 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `entry` | `T` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `entry` | `S` | `0.0` | `0.0` | 0 | 799,200 | 430,552 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `entry` | `u` | `0.0` | `0.0` | 2,053 | 799,200 | 413,030 | 0 | 0 | 2,053 (373 + 1,680) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 10 | `entry` | `v` | `0.0` | `0.0` | 18,803 | 799,200 | 415,175 | 0 | 0 | 18,803 (15,443 + 3,360) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 10 | `entry` | `ssh` | `0.0` | `0.0` | 0 | 26,640 | 16,433 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage1` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage1` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage1` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage1` | `v` | `0.0` | `0.0` | 8,529 | 399,600 | 226,637 | 0 | 0 | 8,529 (6,879 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 10 | `stage1` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage2` | `T` | `2.2204460492503131e-16` | `3.5125902029716364e-19` | 1 | 399,600 | 233,341 | 1 | 1 | 0 | `AT_BAR_NOT_EXACT` | `AT_BAR_NOT_EXACT` |
| 10 | `stage2` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage2` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage2` | `v` | `0.0` | `0.0` | 9,251 | 399,600 | 226,637 | 0 | 0 | 9,251 (7,601 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 10 | `stage2` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage3` | `T` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage3` | `S` | `0.0` | `0.0` | 0 | 399,600 | 233,341 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage3` | `u` | `0.0` | `0.0` | 0 | 399,600 | 226,236 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |
| 10 | `stage3` | `v` | `0.0` | `0.0` | 9,947 | 399,600 | 226,637 | 0 | 0 | 9,947 (8,297 + 1,650) | `AT_BAR_NOT_EXACT` | `BIT_EXACT` |
| 10 | `stage3` | `ssh` | `0.0` | `0.0` | 0 | 13,320 | 8,794 | 0 | 0 | 0 | `BIT_EXACT` | `BIT_EXACT` |

## 2. Offline source-order walk of the first non-bit rows

Executed with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:660-1301` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/walk.json`, SHA-256 `f78653fff1b5eca59a43aad27797b900cac9fea9917623cd94fc92a50f5ef191`) using only the admitted production-arithmetic records (`round 5` full-entry/stage/barotropic/surface records and `round 326` `hm_i`/stage records; `cross_record_r5_vs_r326_stage12_unequal = 0` across all `kt = 1..10`, Stages 1 and 2, and all five fields `T, S, u, v, ssh`).

### 2.1 Walk of the first raw non-bit row (`kt = 1, stage1, v`) — `CONFIRMED` (`R341-P3`)

- **3-D wet vs 3-D dry partition**: At `kt = 1, stage1, v`, all `226,637` rank-0 3-D wet `V` cells (`vmask(ji,jj,jk) != 0`) are bitwise identical between `legoESM` and `oracle_stage_kt00000001_s1.bin` (`wet_3d_unequal = 0`, `max_abs = 0.0`, `rms = 0.0`). All `7,534` unequal binary64 words are `-0.0` (`0x8000000000000000`) in `legoESM` vs `+0.0` (`0x0000000000000000`) in `oracle_stage_kt00000001_s1.bin` on 3-D dry cells (`vmask(ji,jj,jk) == 0`): `5,550` sub-bottom partial-step dry cells below the ocean floor in wet water columns (`ssvmask(ji,jj) == 1, vmask(ji,jj,jk) == 0`) plus `1,984` surface-column dry cells (`ssvmask(ji,jj) == 0, vmask(ji,jj,jk) == 0`).
- **Why the 2-D surface-mask audit reported `signed_zero_only_wet = 5,550`**: `r333_wet._native_masks` broadcasts the 2-D surface mask `ssvmask` (`k = 0`) across all 30 vertical levels rather than applying the 3-D partial-step `vmask(ji,jj,jk)`.
- **Owning statement (`CONFIRMED` on admitted `round 5` and `round 326` records)**: NEMO's stage/entry dump canonicalizer `l4_canon_3d` at `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:479-525` (and identically at `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:846-892`) initializes `zfield(:,:,:) = 0._wp` (`+0.0`) and copies `pfield(ji,jj,jk)` only where `vmask(jif,jjf,jk) /= 0._wp` (`oracle_dry_non_poszero_words = 0`).
- **Corroborating `PLAUSIBLE` check against the round-332 raw post-fold `V` record**: Per the Round-336 producer audit, the round-332 raw post-fold `V` record (`oracle_r332_raw_v_kt00000001_r0000.bin`) is treated as provenance-flagged `PLAUSIBLE` because its producer binary carried the round-331/332 `zdftke`/`stp_rk3_stg` observers. Comparing that raw post-fold `vv(:,:,:,Kaa)` against `oracle_stage_kt00000001_s1.bin` gives `0` 3-D wet differences (`wet_3d_unequal_vs_canon = 0`) and `10,700` dry `-0.0` words (`5,884` sub-bottom dry + `4,816` surface dry) that `l4_canon_3d` zeroes to `+0.0`.

### 2.2 Walk of the first 3-D-wet non-bit / first wet-value row (`kt = 10, stage2, T`) — `CONFIRMED` single statement (`R341-P4`)

At `kt = 10`, Rung-10 `entry` and `stage1` are `100%` 3-D-wet bit-exact across `T, S, u, v, ssh` (`0` unequal 3-D wet words). Walking `stp_RK3_stg(2)` in NEMO source order from the admitted `round 5` and `round 326` records isolates the single 1-ULP departure at `(j = 18, i = 34, k = 3, rank = 0)`:

1. **Stage-1 `r3t` store and Stage-2 `Kmm` swap in NEMO**:
   - At Stage 1, NEMO computes `r3t(:,:,Kbb) = ssh(:,:,Kbb) * r1_ht_0(:,:)` and `r3ta(:,:) = ssha(:,:) * r1_ht_0(:,:)` and stores the Stage-1 thickness ratio in `r3t(:,:,Kaa)` at `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179` (specifically `:177`: `r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)`).
   - Between Stage 1 and Stage 2, `stp_RK3` swaps `Naa` into `Nnn` (`Kmm`) and calls `stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )` at `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:216-222`.
   - In Stage-2 centered tracer advection (`tra_adv_cen`), NEMO divides the horizontal and vertical advective flux divergences by the macro `e3t(ji,jj,jk,Kmm)`, which expands in the preprocessed record-build source to `e3t_3d(ji,jj,jk) * (1._wp + r3t(ji,jj,Kmm) * tmask(ji,jj,jk))` at `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160` and `:202-227`.
2. **Corresponding `legoESM` statements**:
   - `legoESM` already computes the NEMO-stored Stage-1 stretch `_qt_13 = nemo_r3t_rk3_stage1_stretch(state.eta.data, _eta_end_of_step, _r1_ht0_ws)` at `ocean_model_latlon_cgrid.py:8759-8769` and the stored Stage-2 `Kmm` thickness `_h_s2_kmm = nemo_source_round(_h_ref_ws * _qt_13[..., None])` at `:8820-8823`.
   - `legoESM` already passes `_h_s2_kmm` to Stage-2 `tra_sbc_RK3` (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:282-286` and `:318-324`) at `ocean_model_latlon_cgrid.py:9163-9170` and passes `_qt_13` as `q_now` to the Stage-2 tracer update (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`) at `ocean_model_latlon_cgrid.py:2877-2879`.
   - **The single owning statement**: In `_nemo_ws_stage_transport` at `ocean_model_latlon_cgrid.py:2293-2295` (called for Stage 2 at `:9570-9572` with `_h_live_one_third` and `eta_stage=_eta_live_one_third`), `legoESM` recomputes `h_stage = nemo_qco_live_t_thickness(eta_stage, H_bathy, z_coord, jnp.asarray(h_stage).dtype, e3t_0=h_ref)` from `eta_stage = (2/3) * ssh_b + (1/3) * ssha` (i.e. `((r2_3 * ssh_b + r1_3 * ssha) * r1_ht_0)` instead of NEMO's stored `r2_3 * (ssh_b * r1_ht_0) + r1_3 * (ssha * r1_ht_0)`), unpacks `h_stage` from `stage_geom` at `:2665-2666`, and passes `h_stage` into `_nemo_cen2_tracer_rhs` at `:2705-2706`, where it divides the horizontal and vertical advective flux divergences at `:2130` and `:2143`.

### 2.3 Exact binary64 trace at `(j = 18, i = 34, k = 3, rank = 0)`, `kt = 10, stage2, T`

| Operand / intermediate | Stored `r3t(Kmm)` (`NEMO` / proposed) | Rebuilt `eta_stage * r1_ht_0` (`legoESM` current) | ULP diff |
|---|---|---|---:|
| `ssh_b` (`Kbb`) | `0xc004dac971407002` (`-2.6068295333661817`) | `0xc004dac971407002` (`-2.6068295333661817`) | 0 |
| `ssha` (`stp_2D Kaa`) | `0x3fd4c94d4afbb824` (`0.32478649447046082`) | `0x3fd4c94d4afbb824` (`0.32478649447046082`) | 0 |
| `r1_ht_0` | `0x3f9642c8590b2164` (`0.021739130434782608`) | `0x3f9642c8590b2164` (`0.021739130434782608`) | 0 |
| `r3t_b = ssh_b * r1_ht_0` | `0xbfad03e09d914dec` (`-0.056670207247090904`) | `0xbfad03e09d914dec` (`-0.056670207247090904`) | 0 |
| `r3ta = ssha * r1_ht_0` | `0x3f7ceb8cede3c88b` (`0.0070605759667491481`) | `0x3f7ceb8cede3c88b` (`0.0070605759667491481`) | 0 |
| `r3t(Kmm)` (`stprk3_stg logic`) | `0xbfa2236fdf220b97` (`-0.035426612842477549`) | `0xbfa2236fdf220b98` (`-0.035426612842477556`) | 1 ULP |
| `1 + r3t(Kmm)` | `0x3feeddc9020ddf47` (`0.9645733871575225`) | `0x3feeddc9020ddf46` (`0.96457338715752239`) | 1 ULP |
| `e3t_0` | `0x402402c768e8a600` (`10.00542762603709`) | `0x402402c768e8a600` (`10.00542762603709`) | 0 |
| `e3t(Kmm) = e3t_0 * (1 + r3t(Kmm))` | `0x40234d4bd6440687` (`9.6509692152060449`) | `0x40234d4bd6440686` (`9.6509692152060431`) | 1 ULP |
| `h_scaled = ((du + dv) * r1_e1e2t)` | `0x3f15e5f2c773b397` (`8.3535140095582181e-05`) | `0x3f15e5f2c773b397` (`8.3535140095582181e-05`) | 0 |
| `krhs_h = -(h_scaled / e3t(Kmm))` | `0xbee226f39c21e2df` (`-8.6556218585760697e-06`) | `0xbee226f39c21e2e0` (`-8.6556218585760714e-06`) | 1 ULP |
| `v_scaled = ((ztFw_top - ztFw_bot) * r1_e1e2t)` | `0xbebdbf795f7c1fc3` (`-1.7731157395134987e-06`) | `0xbebdbf795f7c1fc3` (`-1.7731157395134987e-06`) | 0 |
| `v_div = v_scaled / e3t(Kmm)` | `0xbe88a8b65ef4fa70` (`-1.8372411101672376e-07`) | `0xbe88a8b65ef4fa71` (`-1.8372411101672378e-07`) | 1 ULP |
| `Krhs = krhs_h - v_div` | `0xbee1c450c2a60ef5` (`-8.4718977475593455e-06`) | `0xbee1c450c2a60ef6` (`-8.4718977475593472e-06`) | 1 ULP |
| `(1 + r3t(Kbb)) * T(Kbb)` | `0xbff520edea197e7a` (`-1.3205393929019054`) | `0xbff520edea197e7a` (`-1.3205393929019054`) | 0 |
| `rDt * (1 + r3t(Kmm)) * Krhs` | `0xbfa697e29eaf72ae` (`-0.044127542372483716`) | `0xbfa697e29eaf72b0` (`-0.04412754237248373`) | 2 ULP |
| `numerator` | `0xbff5d5acff0efa0f` (`-1.364666935274389`) | `0xbff5d5acff0efa10` (`-1.3646669352743892`) | 1 ULP |
| `1 + r3t(Kaa)` (`Stage 2`) | `0x3fef34cc88015959` (`0.9751951843598291`) | `0x3fef34cc88015959` (`0.9751951843598291`) | 0 |
| `T(Kaa)` (`Stage 2`, oracle `0xbff663da73f853c6`) | `0xbff663da73f853c6` (`-1.3993782548979978`) | `0xbff663da73f853c7` (`-1.3993782548979981`) | 1 ULP |

## 3. Non-landed single-statement fix proposal and verification across `kt = 1..10`

Per instruction 3 of the Round-341 prompt, because a single source-cited statement is `CONFIRMED` on the admitted production-arithmetic records, this round stops with the fix `PROPOSAL` and its verification on the recorded inputs without landing model code in `packages/`.

### 3.1 Fix `PROPOSAL` (not landed in `packages/` this round)

- **Proposed model change**: Under the existing `_cfg_b.rk3_stage2_tracer_nemo_stored_thickness` selector (no new configuration knob; already enabled on the ORCA2 NEMO recipe and used to form `_h_s2_kmm` at `ocean_model_latlon_cgrid.py:8820-8823`), pass `_h_s2_kmm` as the Stage-2 `Kmm` cell thickness into `_nemo_ws_stage_transport` at `ocean_model_latlon_cgrid.py:9570-9572` and retain that passed `h_stage` in `_nemo_ws_stage_transport` at `ocean_model_latlon_cgrid.py:2293-2295` when `rk3_stage2_tracer_nemo_stored_thickness` is active (instead of overwriting `h_stage` with `nemo_qco_live_t_thickness(eta_stage, ...)`), so that Stage-2 `_nemo_cen2_tracer_rhs` (`ocean_model_latlon_cgrid.py:2665-2666` and `:2705-2706`) divides by NEMO's stored `e3t_3d * (1 + r3t(Kmm) * tmask)` (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:177` and `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160`, `:202-227`).

### 3.2 Verification on the admitted `round 5` and `round 326` records across `kt = 1..10`

Across all `kt = 1..10`, every column where `e3t_kmm_stored != e3t_kmm_rebuilt` lies strictly inside the rank-0 interior (`1 <= j <= 146, 1 <= i <= 88`), where all horizontal (`i +/- 1, j +/- 1`) and vertical (`k +/- 1`) stencil neighbors are available in the admitted rank-0 records. Replaying Stage 2 (`stp_RK3_stg(2)`) across all `228,641` rank-0 interior 3-D wet cells for `kt = 1..10` (`2,286,410` wet cell-steps per tracer) gives:

| `kt` | Interior 3-D wet cells | Wet columns with `e3t_stored != e3t_rebuilt` | Wet 3-D cells with `e3t_stored != e3t_rebuilt` | `T` unequal (current `e3t_rebuilt`) | `T` unequal (proposed `e3t_stored`) | `S` unequal (current `e3t_rebuilt`) | `S` unequal (proposed `e3t_stored`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 228,641 | 1 | 16 | 0 | **0** | 0 | **0** |
| 2 | 228,641 | 1 | 19 | 0 | **0** | 0 | **0** |
| 3 | 228,641 | 2 | 19 | 0 | **0** | 0 | **0** |
| 4 | 228,641 | 1 | 4 | 0 | **0** | 0 | **0** |
| 5 | 228,641 | 1 | 23 | 0 | **0** | 0 | **0** |
| 6 | 228,641 | 2 | 11 | 0 | **0** | 0 | **0** |
| 7 | 228,641 | 0 | 0 | 0 | **0** | 0 | **0** |
| 8 | 228,641 | 4 | 39 | 0 | **0** | 0 | **0** |
| 9 | 228,641 | 2 | 14 | 0 | **0** | 0 | **0** |
| 10 | 228,641 | 1 | 3 | 1 | **0** | 0 | **0** |
| **Total (`kt = 1..10`)** | **2,286,410** | **15** | **148** | **1** | **0** | **0** | **0** |

With the single-statement fix proposal (`e3t_kmm_stored` = `_h_s2_kmm`), both `T` and `S` at Stage 2 have **`0` unequal interior 3-D wet words across all `kt = 1..10`**, eliminating the sole 3-D-wet departure on Rung 10 (`kt = 10, stage2, T`) and making all **`200 / 200` Rung-10 rows `100%` `BIT_EXACT` on 3-D wet cells** (`152 / 200` raw `BIT_EXACT`, `48 / 200` `AT_BAR_NOT_EXACT` dry signed-zero `u`/`v` rows, `0 / 200` `OVER_BAR`).
