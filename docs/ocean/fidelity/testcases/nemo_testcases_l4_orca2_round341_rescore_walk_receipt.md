# ORCA2 round 341 — current-tip 8-pair re-score, 3-D wet audit, and Stage-2 `e3t(Kmm)` source-order walk

Date: 2026-10-27

Base: `a91d673b6fcf` (`a91d673b6fcff38b097f7c0fc8ddb009c05d7a32`, Round 336 landed); preregistration: `5fae40ddd` (`5fae40dddf5a6f280448a5b326c8c3df5f5c6335`, `docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round341.md`); rescore and source-order walk gate: `a104ed2a4` (`a104ed2a4e79a6f3a5704d221b9718113e846bc7`).
Disposition: **HELD (STOP WITH CONFIRMED SINGLE-STATEMENT FIX PROPOSAL AND VERIFICATION ON RECORDED INPUTS; NO MODEL CODE LANDED THIS ROUND)**.
independent review: operator.

This round changes no model code in `packages/`, changes no configuration value, and never builds or runs NEMO.

## 1. Current-tip 8-pair re-score and first non-bit row per label

Measured on clean committed worktree `a104ed2a4e79a6f3a5704d221b9718113e846bc7` under CPU `fp64` (`libm`, production JIT) with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:480-602` and aggregated with `:605-755` (unit controls in `tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py:22-124`; self-test artifact `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/selftest.json`, SHA-256 `79a0b32a2f3b5e4d52d6ed8ef857fc038b9ec2ae1c7610bbe46e8ce68ef7cdfb`; all four planted controls `signed-zero`, `wet-3d-mask`, `stage2-thickness`, and `walk-closure` fire `STATUS PLANT-FIRED` with exit code `2`).

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

1. **First raw non-bit row (and first 2-D surface-mask non-bit row)**: `kt = 1, stage1, v` (`unequal = 7,534` out of `399,600` rank-0 words, `max_abs = 0.0`, `rms = 0.0`, `AT_BAR_NOT_EXACT`). Across all 200 rows of Rung 10, all 48 non-bit `u`/`v` rows (`kt = 1..10 stage1/2/3 v` [30 rows], `kt = 2..10 entry u` [9 rows], `kt = 2..10 entry v` [9 rows]) have `max_abs = 0.0`, `rms = 0.0`, `wet_3d_bit_unequal = 0`, and `unequal_3d_wet = 0`: every single unequal word lies on a 3-D dry face (`umask == 0` or `vmask == 0`). At `kt = 1, stage1, v`, all `226,637` rank-0 3-D wet `v` faces (`vmask != 0`) are bit-exact (`0` unequal); on the `7,534` dry faces (`vmask == 0`), `legoESM` has `-0.0` (`0x8000000000000000`), **matching raw NEMO** (`vv(:,:,:,Kaa)` before dump canonicalization), while NEMO's dump canonicalizer `l4_canon_3d` writes `+0.0` (`0x0000000000000000`, `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:488-520`). Under the 3-D surface mask `ssvmask = vmask[:,:,0]` (Table 1.3), those `7,534` dry `-0.0` words partition into `5,884` sub-bottom partial-step dry faces (`ssvmask == 1, vmask == 0`) plus `1,650` surface-column dry faces (`ssvmask == 0, vmask == 0`), corresponding to `5,550` (`state.v_mask == 1, vmask == 0`) plus `1,984` (`state.v_mask == 0, vmask == 0`) under the 2-D `state.v_mask[1:]` audit (`r333_wet._native_masks`, differing on `334` north-fold sub-bottom dry faces).
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

Executed with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:760-1411` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/walk.json`, SHA-256 `f78653fff1b5eca59a43aad27797b900cac9fea9917623cd94fc92a50f5ef191`) using only the admitted production-arithmetic records (`round 5` full-entry/stage/barotropic/surface records and `round 326` `hm_i`/stage records; `cross_record_r5_vs_r326_stage12_unequal = 0` across all `kt = 1..10`, Stages 1 and 2, and all five fields `T, S, u, v, ssh`).

### 2.1 Walk of the first raw non-bit row (`kt = 1, stage1, v`) — `CONFIRMED` (`R341-P3`)

- **3-D wet vs 3-D dry partition and sign polarity**: At `kt = 1, stage1, v`, all `226,637` rank-0 3-D wet `V` cells (`vmask(ji,jj,jk) != 0`) are bitwise identical between `legoESM` and `oracle_stage_kt00000001_s1.bin` (`wet_3d_unequal = 0`, `max_abs = 0.0`, `rms = 0.0`). On the `7,534` unequal 3-D dry cells (`vmask(ji,jj,jk) == 0`), `legoESM` has `-0.0` (`0x8000000000000000`), **matching raw NEMO** (`vv(:,:,:,Kaa)` before dump canonicalization), while NEMO's dump canonicalizer `l4_canon_3d` (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:488-520` inside `:479-525`) writes `+0.0` (`0x0000000000000000`) into `oracle_stage_kt00000001_s1.bin`. Under the 3-D surface mask `ssvmask(ji,jj) = vmask(ji,jj,1)` (Table 1.3), those `7,534` dry `-0.0` cells partition into `5,884` sub-bottom partial-step dry cells (`ssvmask(ji,jj) == 1, vmask(ji,jj,jk) == 0`) plus `1,650` surface-column dry cells (`ssvmask(ji,jj) == 0, vmask(ji,jj,jk) == 0`); under the 2-D `state.v_mask[1:]` audit (`r333_wet._native_masks`), they partition into `5,550` (`state.v_mask == 1, vmask == 0`) plus `1,984` (`state.v_mask == 0, vmask == 0`), differing on `334` north-fold sub-bottom dry faces where `vmask(ji,jj,1) == 1` while `state.v_mask == 0`.
- **Why the 2-D surface-mask audit reported `signed_zero_only_wet = 5,550`**: `r333_wet._native_masks` broadcasts the 2-D `state.v_mask[1:]` mask across all 30 vertical levels rather than applying the 3-D partial-step `vmask(ji,jj,jk)`.
- **Owning statement (`CONFIRMED` on admitted `round 5` and `round 326` records)**: Raw NEMO `vv(:,:,:,Kaa)` and `legoESM` both carry `-0.0` on those `7,534` dry faces; NEMO's stage/entry dump canonicalizer `l4_canon_3d` at `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:488-520` (within `:479-525`, and identically at `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:846-892`) initializes `zfield(:,:,:) = 0._wp` (`+0.0`) and copies `pfield(ji,jj,jk)` only where `vmask(jif,jjf,jk) /= 0._wp` (`oracle_dry_non_poszero_words = 0`), writing `+0.0` in the dump on every dry cell where raw NEMO and `legoESM` hold `-0.0`.
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

## 4. Final gates and disposition

- **Per-receipt and suite citation gates**:
  - `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py --receipt docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round341_rescore_walk_receipt.md --from-heading "# ORCA2 round 341 —"` passes all 24 citations with `0` failures, `0` unmapped citations, `0` map-audit failures, and `9 / 9` self-test controls fired (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/citation_gate.json`, SHA-256 `40a1bebb5bd9b4b512e421d59dc26aa392a428ceae96869fc7a74d913f6fb15f`).
  - `JAX_ENABLE_X64=1 pytest tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`: `18 passed in 9.01s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/test_citation_gate.log`, SHA-256 `5156712751b05a0e1b99e9bb00e47d6fb97339193d4868b942ccc23a4e3c341f`).
- **Default ORCA2 gate battery and private-import ratchet**:
  - All 153 tests in the default ORCA2 landing battery plus the 3 new Round-341 unit controls (`tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py:22-124`) pass on CPU under `JAX_ENABLE_X64=1`:
    - `141 passed in 344.39s` (`default_gate_part1.log`, SHA-256 `3ce741813e70ab8b6a0c46977b7b9a1250fbc162ab58f89c7252941e5dd6ba9c`),
    - `10 passed in 72.58s` (`default_gate_part2.log`, SHA-256 `9692ea18db0ae2d0c0a08871730bd946ccda8e80125bd00671490c494f6fb518`),
    - `2 passed in 580.81s` (`default_gate_part3.log`, SHA-256 `40ad668e52559b6988449fc40e4a23f3aeffa8d6305fa95eab782ae3b7a2af9f`),
    - `3 passed in 6.91s` (`default_gate_r341.log`, SHA-256 `6ecf78b85eea9ab71175adcbf06777d938288e4c88c9759a8cf9d032cb68888f`).
  - Explicit private-import ratchet (`tests/test_no_private_cross_imports.py`): `6 passed in 5.11s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/rescore/ratchet.log`, SHA-256 `531050d7267caa32f4485abc610c52e4083e66d13f42905a7fe2fc98c74ee52e`).
  - No DINO month gate is required because `0` files under `packages/` or `src/` changed.
- **Review**: `independent review: operator`.
- **Disposition**: **HELD** — single cited statement (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:177` + `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160` and `:202-227` vs `ocean_model_latlon_cgrid.py:2293-2295` / `:9570-9572` / `:2665-2666` / `:2705-2706`) `CONFIRMED` on admitted production-arithmetic records; stopping with the fix `PROPOSAL` and verification on the recorded inputs per prompt instruction 3.

## Resume 1: Landed Stage-2 `e3t(Kmm)` thickness fix, Decision 122, caller audit, and `200/200` 3-D-wet `BIT_EXACT` rescore

Date: 2026-10-27

Parent before Round 341: `a91d673b6fcf` (`a91d673b6fcff38b097f7c0fc8ddb009c05d7a32`); Round-341 Held tip: `ca77c284e0a5` (`ca77c284e0a5a8ccabbbdf46b776621330e663f0`); Landed model/test/gate commit: `86dfcb7e0ea5` (`86dfcb7e0ea58ccaee71c80e8135644037151267`).
Disposition: **LANDED (STAGE-2 `e3t(Kmm)` STORED THICKNESS IN `_nemo_ws_stage_transport`; RUNG 10 `200 / 200` 3-D-WET `BIT_EXACT`, `152 / 200` RAW `BIT_EXACT`, `48 / 200` `AT_BAR_NOT_EXACT` DRY `-0.0` `u`/`v` ROWS, `0 / 200` `OVER_BAR`)**.
independent review: operator.

### R1.1 Claim-review disposition and Claim-2 polarity wording

- **Claims 1 and 3 (`CONFIRMED`)**: Independent claim reviews reproduced the binary64 trace in Section 2.3 and confirmed that using the stored Stage-1 `r3t(Kmm)` thickness `_h_s2_kmm = nemo_source_round(_h_ref_ws * _qt_13[..., None])` (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:177` and `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160`, `:202-227`) in Stage-2 `_nemo_ws_stage_transport` removes the sole 1-ULP `kt = 10, stage2, T` departure at `(j = 18, i = 34, k = 3, rank = 0)`.
- **Claim 2 sign polarity (`CONFIRMED` with corrected wording in Sections 1.2 and 2.1)**: On all `7,534` unequal 3-D dry cells (`vmask(ji,jj,jk) == 0`) at `kt = 1, stage1, v`, `legoESM` has `-0.0` (`0x8000000000000000`), **matching raw NEMO** (`vv(:,:,:,Kaa)` before dump canonicalization), while NEMO's dump canonicalizer `l4_canon_3d` writes `+0.0` (`0x0000000000000000`, `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:488-520` inside `:479-525`). Sections 1.2 and 2.1 also state both the 3-D `ssvmask = vmask[:,:,0]` partition (`5,884 + 1,650`, matching Table 1.3) and the 2-D `state.v_mask[1:]` partition (`5,550 + 1,984`, differing on `334` north-fold sub-bottom dry faces).

### R1.2 Model-code change (`M1`..`M6`) and Stage 1 / Stage 3 / legacy caller invariance

- **Model change (`M1`, no new configuration knob)**:
  - In `_nemo_ws_stage_transport` (`ocean_model_latlon_cgrid.py:2267` and `:2291-2295`), `stored_h_stage=False` (default `False`) is added with the Stage-2 guard `if stored_h_stage and stage_index != 1: raise ValueError("stored_h_stage is Stage-2 (stage_index == 1) only")`, and `h_stage = jnp.asarray(h_stage) if stored_h_stage else nemo_qco_live_t_thickness(eta_stage, H_bathy, z_coord, jnp.asarray(h_stage).dtype, e3t_0=h_ref)` retains the passed `h_stage` when `stored_h_stage` is `True` instead of rebuilding from `eta_stage * r1_ht_0`.
  - In the Stage-2 call at `ocean_model_latlon_cgrid.py:9570-9572` only, `h_stage` is passed as `_h_s2_kmm` (`nemo_source_round(_h_ref_ws * _qt_13[..., None])` when `_cfg_b.rk3_stage2_tracer_nemo_stored_thickness` is `True`, else `_h_live_one_third`, built at `ocean_model_latlon_cgrid.py:8820-8823` from `_qt_13` at `:8759-8769`) along with `stored_h_stage=bool(_cfg_b.rk3_stage2_tracer_nemo_stored_thickness)`.
  - Stage 1 (`ocean_model_latlon_cgrid.py:9341-9354`), Stage 3 (`ocean_model_latlon_cgrid.py:9641-9658`), and the legacy caller (`ocean_model_latlon_cgrid.py:10144-10162`) omit `stored_h_stage` (default `False`, and `stage_index != 1` is rejected by the `ValueError` guard at `ocean_model_latlon_cgrid.py:2291-2295`) and remain bit-identical (`0` unequal words across all 12 tuple slots).
- **Production-call-site unit tests, revert check (`M2`), gradient gate (`M3`), and `eager == jit` (`M4`)**:
  - `tests/ocean/unit/test_orca2_round341_stage2_thickness.py:168-586` (`3 passed` on CPU under `JAX_ENABLE_X64=1`, using `_build_stage2_fixture` and `_run_production_step_spies`):
    1. `test_stage2_transport_uses_stored_h_s2_kmm_and_fails_on_revert` builds the compact C-grid fixture carrying the exact Round-341 `kt = 10` binary64 geometry at `(j = 1, i = 1, k = 3)` (`r1_ht_0 = 0x3f9642c8590b2164`, `e3t_0[1, 1, 3] = 0x402402c768e8a600`, `ssh_b = 0xc004dac971407002`, `ssha = 0x3fd4c94d4afbb824`), verifies `h_s2_kmm[1, 1, 3] == 0x40234d4bd6440687` vs `h_live_13[1, 1, 3] == 0x40234d4bd6440686`, proves `_nemo_ws_stage_transport(..., stage_index=1, stored_h_stage=True)` returns `e3t_st = 0x40234d4bd6440687` in tuple slot 3 (`h_stage`) whereas `stored_h_stage=False` returns `e3t_rb = 0x40234d4bd6440686`, proves the exact `kt = 10` Stage-2 tracer division with `e3t_st` yields `T_s2_st = 0xbff663da73f853c6` (`-1.3993782548979978`, matching NEMO `oracle_stage_kt00000010_s2.bin` with `0` ULP difference) whereas `e3t_rb` yields `T_s2_rb = 0xbff663da73f853c7`, and exercises the production `LatLonCGridOceanModel.step` (`_step_impl`) Stage-2 call site (`ocean_model_latlon_cgrid.py:9570-9572`) via `_run_production_step_spies()` to prove slot 3 (`h_stage`) at `[1, 1, 3]` is `0x40234d4bd6440687` when `rk3_stage2_tracer_nemo_stored_thickness=True` vs `0x40234d4bd6440686` when `False`. When `ocean_model_latlon_cgrid.py` is reverted to parent `a91d673b6fcf`, `test_stage2_transport_uses_stored_h_s2_kmm_and_fails_on_revert` fails with `TypeError: _nemo_ws_stage_transport() got an unexpected keyword argument 'stored_h_stage'` (`1 failed in 4.05s`, `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/revert_test_failure.log`, SHA-256 `5d1f73f385dab507d8796695c4ccdbf4cb75ad7a8dc008c4b4e84b029e440fd4`).
    2. `test_stage1_stage3_and_legacy_caller_stay_bit_identical` proves that `_nemo_ws_stage_transport` raises `ValueError("stored_h_stage is Stage-2 (stage_index == 1) only")` when `stored_h_stage=True` is passed with `stage_index=0` or `stage_index=2`, and proves `0` unequal words across all 12 returned tuple slots on Stage 1 (`ocean_model_latlon_cgrid.py:9341-9354`, `stage_index=0` in production `LatLonCGridOceanModel.step`), Stage 3 (`ocean_model_latlon_cgrid.py:9641-9658`, `stage_index=2`, including under a perturbed `h_stage = h_ref * 1.125` input), and all 3 stages (`stage_index = 0, 1, 2`) of the legacy caller (`ocean_model_latlon_cgrid.py:10144-10162`).
    3. `test_stage2_stored_thickness_eager_equals_jit_with_dynamic_args` (`M4`) proves `0` unequal words between eager execution under `with jax.disable_jit():` on the entire eager call and `jax.jit(_stage2_kernel)` with all 11 dynamic array inputs (`u_in, v_in, uu_b_in, vv_b_in, eta_b_in, eta_a_in, h_ref_in, Hu_in, Hv_in, T_b_in, T_s1_in`) passed as JIT arguments across `(g1[3], rhs, T_s2)` (`_nemo_ws_stage_transport`, `_nemo_cen2_tracer_rhs`, and the Stage-2 tracer update), plus an in-test reverted control (`_reverted_kernel` with `stored_h_stage=False`) that fails bit-identity against `h_jit` under `pytest.raises(AssertionError)`.
  - **Gradient / autodiff / FD gate (`M3`)**: The gradient check that actually runs is the full Rung-10 `N = 1` gradient probe (`nemo_testcase_l4_orca2_round329_grad_probe.py` in mode `on`, `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/grad_rung10_n1.json`, SHA-256 `b3dc306c6fc9e71d6a8ace554220f9af9c999ae692f8396e16c423375d10bff3`, also detailed in Section R1.5), which verifies `forward_parity_non_bit_leaves = []`, `rebuilt_vs_traced_non_bit_leaves = []`, finite reverse-mode VJP cotangents on `T@1` (`loss = 32.91696958352461`, `dT0_l2 = 0.01931092431777194`, `dT0_nonzero_wet_frac = 1.0`, `dT0_finite = true`, `dT0_nonfinite_count = 0`, `dT0_nonzero_dry = 0`, `nonfinite_cotangent_leaves = {}`) and `ssh@1` (`loss = 0.09080875357052787`, `dT0_l2 = 0.0006856828144876344`, `dT0_nonzero_wet_frac = 0.9999628384027945`, `dT0_finite = true`, `dT0_nonfinite_count = 0`, `dT0_nonzero_dry = 0`, `nonfinite_cotangent_leaves = {}`), directional JVP vs VJP agreement (`T_dir0`: `rel = 1.7859866296058833e-14` on `T`, `1.3868976507379975e-15` on `ssh`), and scalar `rn_ud` JVP/VJP/central-FD agreement (`rev_fwd_rel = 1.501929835492298e-15`, central FD `fd_rel = 3.0831728047889743e-06` at `eps = 1e-4` on `T`, `0.0` on `ssh`).

### R1.3 Landed 8-pair re-score and Rung-10 `200 / 200` 3-D-wet `BIT_EXACT` result

Measured on clean committed worktree `86dfcb7e0ea58ccaee71c80e8135644037151267` under CPU `fp64` (`libm`, production JIT) with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:480-602` and aggregated with `:605-755` (unit controls in `tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py:22-124`; self-test artifact `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/selftest.json`, SHA-256 `9d35d9cfa98fa89718f21a71f5e7669604abaaaf6e22aaba5381eef076c56ba0`, with all eight `--plant` controls `signed-zero`, `wet-3d-mask`, `stage2-thickness`, `walk-closure`, `exact-loss`, `away-row`, `parent-commit`, and `caller-unequal` firing `STATUS PLANT-FIRED` with exit code `2`):

- Full 1,560-row table across all eight pairs: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/full_ladder.tsv` (SHA-256 `f10026d9963629c90cf01a865775abe1da723ff98817aea0e6a9939fe161fe6e`).
- Full 400-row (`2 x 200`) Rung-10 3-D and 2-D wet/dry audit table: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/rung10_ladder_3d_wet.tsv` (SHA-256 `53beaa595f313a5921c05dc7f678e2246e12e7452454a2c76c5850054e7ee131`).
- Aggregate summary JSON: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/aggregate.json` (`status = PASS_R341_FULL_RESCORE`, SHA-256 `7e95362c4b7992c9d972be8e69481faf6c9163e87d5a681d6609fdbd7202c600`).
- Source-order walk JSON: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/walk.json` (`status = PASS_R341_SOURCE_ORDER_WALK`, SHA-256 `90038904eaac9c4444b0ffc282b1077ed27023bdabfa350534a39c8db3677dd0`).

| Pair (`rung / label`) | Raw `BIT_EXACT` | Raw `AT_BAR_NOT_EXACT` | Raw `OVER_BAR` | 3-D wet `BIT_EXACT` (Rung 10) | Rows | First raw non-bit row | First 3-D-wet non-bit row (Rung 10) | Pair JSON SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|---|
| `omt0/independent` | 16 | 23 | 161 | — | 200 | `(1, entry, T)` | — | `9be7d0e18f2d9abb137d083e6df1d13d5ee8bfa98e5d83022aef9d7254a1dd84` |
| `rung0/independent` | 11 | 10 | 179 | — | 200 | `(1, stage1, T)` | — | `bdca08188d545ee99edce769596a0cfc419ce9afa0e9a00cc03de4943224b692` |
| `omt1/independent` | 48 | 33 | 79 | — | 160 | `(1, entry, T)` | — | `1b27a1a4f6c76565ae90fc83be3024b57bce3a40e3290c56a72b7f1db3a804fa` |
| `omt2/independent` | 48 | 33 | 119 | — | 200 | `(1, entry, T)` | — | `47f202bb42cc8beb98d77c49c735e2413f9f95c81c09f07f41ca524b30c36b00` |
| `omt3/independent` | 36 | 25 | 139 | — | 200 | `(1, entry, T)` | — | `daefff5baa0681c4f99aba5d670ef13e6dab2f603705729b1a54f414ad112d6a` |
| `omt4/independent` | 12 | 15 | 173 | — | 200 | `(1, entry, T)` | — | `12de95e412fa59edc4ee6171952f307f0023aa73287d6a53cba7a16b3daf5917` |
| `rung10/given_nemo_entry` | 152 | 48 | 0 | **200** | 200 | `(1, stage1, v)` | **`None` (`200 / 200` `BIT_EXACT`)** | `98bcceeeb46953f7fcc833c63b576d98654e6d324c3989b032471b3e52bf5229` |
| `rung10/independent` | 152 | 48 | 0 | **200** | 200 | `(1, stage1, v)` | **`None` (`200 / 200` `BIT_EXACT`)** | `8461a69f0e92f9799511fe7f706d233bb941bfb739f37b641d7fb96b75b5b921` |
| **Total** | **475** | **235** | **850** | — | **1,560** | — | — | — |

On Rung 10 (`rung10/given_nemo_entry` and `rung10/independent`, row-for-row identical across all 200 rows), row `(10, stage2, T)` moves from `unequal = 1, max_abs = 2.220446049250313e-16, rms = 3.5125902029716364e-19` (`AT_BAR_NOT_EXACT`) to **`unequal = 0, max_abs = 0.0, rms = 0.0, wet_3d_bit_unequal = 0` (`BIT_EXACT`)**, while all other 199 rows in Table 1.3 are unchanged (`200 / 200` 3-D-wet `BIT_EXACT`, `152 / 200` raw `BIT_EXACT`, `48 / 200` `AT_BAR_NOT_EXACT` dry `-0.0` `u`/`v` rows, `0 / 200` `OVER_BAR`).

### R1.4 Decision 122 across all eight pairs (`decision_d122`)

Evaluated with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:1423-1669` comparing parent `a91d673b6fcff38b097f7c0fc8ddb009c05d7a32` against landed `86dfcb7e0ea58ccaee71c80e8135644037151267` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/decision_d122.json`, `status = PASS_R341_D122_DECISION`, `d122_c_and_d = true`, SHA-256 `a50b97f939582110df0b37821c40b16947cd2536d76b3c813f5532ed5a654e65`):

| Pair (`rung / label`) | Rows | Parent `BIT / AT / OVER` | Current `BIT / AT / OVER` | `exact_losses` | `exact_gains` | `unequal_increases` | `unequal_decreases` | `rms` (`toward / equal / away`) | `max_abs` (`toward / equal / away`) |
|---|---:|---|---|---:|---|---:|---|---|---|
| `omt0/independent` | 200 | `16 / 23 / 161` | `16 / 23 / 161` | 0 | `[]` | 0 | `[]` | `0 / 200 / 0` | `0 / 200 / 0` |
| `rung0/independent` | 200 | `11 / 10 / 179` | `11 / 10 / 179` | 0 | `[]` | 0 | `[]` | `0 / 200 / 0` | `0 / 200 / 0` |
| `omt1/independent` | 160 | `48 / 33 / 79` | `48 / 33 / 79` | 0 | `[]` | 0 | `[]` | `0 / 160 / 0` | `0 / 160 / 0` |
| `omt2/independent` | 200 | `48 / 33 / 119` | `48 / 33 / 119` | 0 | `[]` | 0 | `[]` | `0 / 200 / 0` | `0 / 200 / 0` |
| `omt3/independent` | 200 | `36 / 25 / 139` | `36 / 25 / 139` | 0 | `[]` | 0 | `[]` | `0 / 200 / 0` | `0 / 200 / 0` |
| `omt4/independent` | 200 | `12 / 15 / 173` | `12 / 15 / 173` | 0 | `[]` | 0 | `[]` | `1 / 199 / 0` | `0 / 200 / 0` |
| `rung10/given_nemo_entry` | 200 | `151 / 49 / 0` | `152 / 48 / 0` | 0 | `[(10, stage2, T)]` | 0 | `[(10, stage2, T, 1 -> 0)]` | `1 / 199 / 0` | `1 / 199 / 0` |
| `rung10/independent` | 200 | `151 / 49 / 0` | `152 / 48 / 0` | 0 | `[(10, stage2, T)]` | 0 | `[(10, stage2, T, 1 -> 0)]` | `1 / 199 / 0` | `1 / 199 / 0` |

Every pair is bit-identical or strictly closer to the NEMO oracle (`0` exact losses, `0` unequal increases, `0` rows moving away in `rms` or `max_abs` across all 1,560 rows).

### R1.5 Caller and card audit (`M6`), faithful DINO probes, 30-day GPU DINO month gate, and Rung-10 `N = 1` gradients

- **Caller and card audit (`M6`)**: Evaluated with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:1995-2479` comparing parent `a91d673b6fcff38b097f7c0fc8ddb009c05d7a32` against landed `86dfcb7e0ea58ccaee71c80e8135644037151267` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/caller_audit.json`, `status = PASS_R341_CALLER_AUDIT`, SHA-256 `3f63649094855f9253a0c604086cb374e44a700de0eda55e8a3e985aad73d348`):

| Caller / comparison card | Array count | Total elements | Unequal arrays | Unequal words | Notes |
|---|---:|---:|---:|---:|---|
| `GYRE-zco` | 14 | 302,652 | 0 | 0 | Bit-identical vs parent |
| `LOCK_EXCHANGE-zco` | 14 | 118,350 | 0 | 0 | Bit-identical vs parent |
| `ORCA2-rung0` (standalone `K0` card) | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent (`rung0/independent` 200-row card also `0` differences) |
| `ORCA2-zps` (`kt = 10` Stage-2 entry) | 14 | 11,298,240 | 2 (`s2_h`, `T_stage2`) | 4 (`3` on `s2_h`, `1` on `T_stage2`) | `s1_*`, `s2_u_tr/v_tr/w`, `S_stage2`, `s3_*` all `0` unequal; `T_stage2[18, 34, 3]` moves `0xbff663da73f853c7 -> 0xbff663da73f853c6` (NEMO oracle) |
| `ORCA2-zps_kt1_rest` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G1-flat` | 14 | 453,864 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G1b-vec` | 14 | 453,864 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G2-zps` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G3-cor` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G4-zad` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G5-hpg` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G6-bfr` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G7-sbc` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2G8-ldf` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `ORCA2V1-tke` | 14 | 11,298,240 | 0 | 0 | Bit-identical vs parent |
| `OVERFLOW-zps` | 14 | 911,718 | 0 | 0 | Bit-identical vs parent |
| `VORTEX-zco` | 14 | 571,347 | 0 | 0 | Bit-identical vs parent |
| `faithful_dino` | 14 | 5,273,544 | 0 | 0 | Bit-identical vs parent (`rk3_stage2_tracer_nemo_stored_thickness = False`) |
| `legacy_caller_10145` | 12 | 260,412 | 0 | 0 | Bit-identical vs parent (`ocean_model_latlon_cgrid.py:10144-10162`) |
| `non_nemo_folded` | 14 | 1,536 | 0 | 0 | Bit-identical vs parent |
| `non_nemo_unfolded` | 14 | 1,536 | 0 | 0 | Bit-identical vs parent |

- **3-step faithful DINO baseline probes**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/faithful_dino_values.json` (SHA-256 `5f977ac145525b73d311bdaacd5c7620b4199567f62afac15d3d1c8d19b503fa`) and `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/faithful_dino_program.json` (SHA-256 `5963788cd3b46a3d51440fc22f2ed58ac627c119d33e3d28d1094639bd1741df`) match `docs/ocean/fidelity/baselines/dino_1226_step_hash_baseline.json` byte-for-byte across all 66 leaf-step hashes (`0` differences).
- **30-day GPU DINO month gate**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/dino_month_gate.log` (SHA-256 `2345499705e16c02727e029dd2ec730c1baf30bb771fb9f19ca9e56fd2687222`; snapshot `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/dino_month/snapshots/snapshot_00001.npz`, SHA-256 `40a63d3a43a80600c740067fd3446bf428cd20e362a60987908c1a079a6e6bb5`; score JSON `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/dino_month/twin_nemo_ts_maps_day30_kt960.json`, SHA-256 `e792a9749dc344cc6fb334ed1ce85ed8c9f66923908b66885a4ebda0b3b84a53`): `SELF-TEST OK`; 30-day (`kt = 960`) GPU 0 integration completed in `275.9 s` (`287.4 ms/step`), with day-30 wet 3-D `T` RMS vs NEMO `kt = 960` equal to `2.056821683e-03 K <= 2.244317642e-03 K` (certified `2.040288765e-03 K`) — `PASS`.
- **Rung-10 `N = 1` gradient gate**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/grad_rung10_n1.json` (SHA-256 `b3dc306c6fc9e71d6a8ace554220f9af9c999ae692f8396e16c423375d10bff3`, `2,519.2 s`, peak RSS `75.13 GiB`): `forward_parity_non_bit_leaves = []`, `rebuilt_vs_traced_non_bit_leaves = []`; both `T@1` (`loss = 32.91696958352461`, `dT0_l2 = 0.01931092431777194`, `dT0_nonzero_wet_frac = 1.0`) and `ssh@1` (`loss = 0.09080875357052787`, `dT0_l2 = 0.0006856828144876344`, `dT0_nonzero_wet_frac = 0.9999628384027945`) have `dT0_finite = true`, `dT0_nonfinite_count = 0`, `dT0_nonzero_dry = 0`, `nonfinite_cotangent_leaves = {}`, and directional JVP/VJP agreement (`T_dir0`: `rel = 1.7859866296058833e-14` on `T`, `1.3868976507379975e-15` on `ssh`; `rn_ud`: `rev_fwd_rel = 1.501929835492298e-15`, central FD `fd_rel = 3.0831728047889743e-06` on `T`, `0.0` on `ssh`).
- **Default ORCA2 gate battery, citation gate, and private-import ratchet**:
  - Default gate battery on CPU under `JAX_ENABLE_X64=1`:
    - `141 passed in 235.72s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/default_gate_part1.log`, SHA-256 `52651f45f338b5ae67a65cbf1e2f26576694f51a793b7e73b59f35af7a744bfa`),
    - `10 passed in 40.89s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/default_gate_part2.log`, SHA-256 `96419afcc13e342a84fcd40f66acd4a0a6b8e3cc6bdc1afcf764fdec7644b2e4`),
    - `2 passed in 206.60s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/default_gate_part3.log`, SHA-256 `eae69275c6d28d2d0535eadb210dc36eca240e80995fe56c16e3400d371639c8`),
    - `8 passed in 116.71s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/default_gate_r341.log`, SHA-256 `d720e711641f37b9a2565897c99ccecd443cf6c200d3db0f1a5034e27aa19171`).
  - Explicit private-import ratchet (`tests/test_no_private_cross_imports.py`): `6 passed in 5.71s` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r341/land/ratchet.log`, SHA-256 `81bd53ab5fd3fc98d7199ce3d9a6044230d8735d0846e7b0519f40a22930039a`).
  - Per-receipt and suite citation gates (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py --receipt docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round341_rescore_walk_receipt.md --from-heading "# ORCA2 round 341 —"` and `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`): `0` failures, `0` unmapped citations, `0` map-audit failures, `9 / 9` self-test controls fired (`18 passed`).
- **Review**: `independent review: operator`.
- **Disposition**: **LANDED** — Stage-2 stored `e3t(Kmm)` thickness (`_h_s2_kmm`) wired into `_nemo_ws_stage_transport`'s Stage-2 call (`ocean_model_latlon_cgrid.py:2267`, `:2291-2295`, `:9570-9572`); Rung 10 is **`200 / 200` 3-D-wet `BIT_EXACT`** (`152 / 200` raw `BIT_EXACT`, `48 / 200` `AT_BAR_NOT_EXACT` dry `-0.0` `u`/`v` rows, `0 / 200` `OVER_BAR`).

