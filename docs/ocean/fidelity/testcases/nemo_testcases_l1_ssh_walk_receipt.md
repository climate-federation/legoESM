# OVERFLOW-zps kt>=3 SSH walk: receipt for the seed arm

Preregistration: `nemo_testcases_l1_ssh_walk_preregister.md` (commit
`14fa71454`).  Base tree for every "before" row: `f9f027525` (= `59d1fcbb8` +
the S-21 landing); "after" = the same tree plus the loop-entry seed fix
(`_nemo_literal_seed_from_card_mesh`), fp64 (`PrecisionPolicy.fp64()` +
`JAX_ENABLE_X64=1`, arrays printed `float64`), CPU.  The one-variable arm is
the private `_NEMOWSRK3TestHooks.legacy_seed_min_rule_faces` (harness only;
NEMO has no such switch); its legacy side reproduces the pre-fix tree
BIT-FOR-BIT: OVERFLOW kt=1..10 trajectory 50/50 rows, LOCK 50/50 rows, and the
kt=2 re-seeded frame walk (`seed_fix/frame_walk_kt2_legacy_arm.json`).

## Verdict

**CONFIRMED owner of the kt>=3 OVERFLOW SSH walk**: the card-mesh
`nemo_literal` loop-entry seed (`barotropic_latlon_cgrid.py`
`_depth_average_to_faces`, min-of-stretched-cells rescale).  NEMO seeds the
external loop with the carried `puu_b(Kmm)` (`dynspg_ts.F90:487`), which is
the `e3u_0/hu_0` column mean of `uu(Kmm)` imposed at `stprk3_stg.F90:439-446`
(measured on the oracle: `uu_b(Kbb)` vs that mean of the dumped `uu(Kbb)`
`4e-19 / 7e-18 / 2e-17` at kt=2/3/4).  legoESM's seed returned a uniform
velocity scaled by `(1+r3t_e)/(1+r3t_w)` on every face whose two columns have
different reference depths; re-injected each step, it is the SSH walk.

**REFUTED (P5, the `u` half)**: the kt>=3 `u` walk is NOT the seed.  With the
seed fixed the OVERFLOW `u` row is `2.644e-05 -> 2.630e-05` at kt=10 (1.01x)
while SSH drops 62.7x; the `u` walk (`4.55e-10 -> 8.8e-9 -> 1.27e-7 -> 9.9e-7
-> 2.6e-5`, kt=2..5,10, roughly x8-x20 per step) is a separate, baroclinic
owner that lives in the stage ladder's seed error (`4.55e-10` at kt=2, faces
19/21) and stays UNMEASURED-owner here.

## Predictions vs outcomes

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | re-seeded kt=2 `u_entry` `1.055e-9 -> <= 1e-15`; substep-1 `transport_u`/`eta_continuity`/`eta_pgf` to the `1e-15` class; first DEBT becomes `slow_u` `1.30e-10` unchanged | `u_entry` `6.9e-18`; `transport_u` `3.6e-15`, `eta_continuity` `1.7e-18`, `eta_pgf` `1.7e-18`; first DEBT `slow_u` `1.296e-10` (was `1.296e-10`) | CONFIRMED |
| P2 | re-seeded kt=3, kt=4 `u_entry` `<= 1e-15` | `3.5e-18`, `2.1e-17` (first DEBT `slow_u` `9.63e-10`, `1.90e-9`, unchanged) | CONFIRMED |
| P3 | OVERFLOW kt=2 rows bit-identical | T `1.119105e-14`, u `4.551736e-10`, ssh `1.050549e-14`, S, v all SAME | CONFIRMED |
| P4 | OVERFLOW kt=3 SSH `3.775e-9` decreases, remainder `<= 3.5e-9` | `3.774823e-09 -> 2.440693e-09` | CONFIRMED |
| P5 | kt=4 SSH/u, kt=10 SSH/u each drop `>= 10x` (refuted if `< 2x`) | SSH: kt=4 `1.0715e-06 -> 2.2627e-08` (47x), kt=5 178x, kt=10 `9.237444e-05 -> 1.473509e-06` (62.7x).  `u`: kt=4 `1.6839e-07 -> 1.2710e-07` (1.32x), kt=10 `2.644302e-05 -> 2.629597e-05` (1.01x) | SSH CONFIRMED; **`u` REFUTED** (`< 2x`) |
| P6 | OVERFLOW kt=60 drops, no factor | SSH `7.751811e-05 -> 3.315149e-05` (2.34x; kt=20 40x, kt=40 5.6x); T `1.184430e-05 -> 1.184427e-05`; u `2.507516e-04 -> 2.505926e-04` | SSH drops; T/u unchanged (consistent with the refuted P5-u) |
| P7 | LOCK rows move by ulps only, kt=2 AT-BAR | kt=2..7 bit-identical; from kt=8 the ssh/u rows move in their 8th-9th significant digit (`6.590670e-14 -> 6.590670e-14`, rel `1.3e-8` of the residual, i.e. `~1e-21` of the field); 85/300 rows touched at that level over kt=1..60 | CONFIRMED |
| P8 | kt=1 19-frame gate, stage sweep kt=2 unchanged | kt=1 gate: 0/152 production+legacy rows differ from the clean-`59d1fcbb8` run (`ssh_walk/head_kt1`); stage sweep kt=2: OVERFLOW 96/96 and LOCK 54/54 rows bit-identical (`ssh_walk/s21_after` vs `ssh_walk/seed_after`) | CONFIRMED |
| P9 | 6120-step statistics fp64+fp32, direction-free | 6120-step OVERFLOW-zps statistics (`nemo_testcase_full_statistics.py`, fp64 candidate + fp32 precision floor, scored against the registered NEMO N2 run; before = `stage3_selector/after` at HEAD, after = `ssh_walk/stats_after` on the S-21 + seed tree, ONE statistics pair for both landings): hist `0.06208 -> 0.06127`, census `0.017642 -> 0.016899`, u_linf `1.9940 -> 2.0436` (worse, floor `1.72 -> 2.50`), plume_descent `1499.80 -> 1499.79` (two-valued instrument, same branch), plume_front `117.125 -> 117.125` km, T_linf `0.37925 -> 0.37925`; overall status `OUTSIDE -> OUTSIDE` | reported |

T improves too: OVERFLOW kt=10 `4.046700e-08 -> 2.128213e-09` (19x), kt=5
`3.538e-10 -> 1.787e-10`.

## Frame walk after the fix (re-seeded arm, substep 1 unless noted)

| kt | eta_entry | u_entry | transport_u | eta_continuity | pgf_u | slow_u | u_exit | eta_exit (jn=4) | first DEBT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | `0.0` | `6.9e-18` (was `1.05e-9`) | `3.6e-15` (was `5.27e-7`) | `1.7e-18` (was `1.76e-9`) | `1.4e-20` (was `3.45e-11`) | `1.30e-10` (same) | `4.32e-10` (was `9.37e-10`) | `4.29e-9` (was `5.18e-9`) | slow_u |
| 3 | `0.0` | `3.5e-18` (was `3.08e-7`) | `3.6e-15` (was `1.54e-4`) | `1.4e-17` (was `5.14e-7`) | `1.1e-19` (was `1.01e-8`) | `9.63e-10` (same) | `3.21e-9` (was `2.75e-7`) | `3.15e-8` (was `1.16e-6`) | slow_u |
| 4 | `0.0` | `2.1e-17` (was `4.32e-6`) | `1.1e-14` (was `2.16e-3`) | `5.6e-17` (was `7.20e-6`) | `8.7e-19` (was `1.39e-7`) | `1.90e-9` (same) | `6.34e-9` (was `3.85e-6`) | `8.22e-8` (was `1.63e-5`) | slow_u |

Rows of a stagger with no active face (the tank's V faces) are inventoried,
not scored (`pgf_v` holds `1.6e-3..3.3e-3` on masked faces in BOTH trees;
`ssvmask` zeroes it at `dynspg_ts.F90:757-760`); the gate now says so.

## Trajectory before -> after (OVERFLOW-zps)

| kt | T | u | SSH |
|---:|---|---|---|
| 2 | `1.119105e-14` -> same | `4.551736e-10` -> same | `1.050549e-14` -> same |
| 3 | `6.851053e-12 -> 6.832934e-12` | `9.105049e-09 -> 8.843722e-09` | `3.774823e-09 -> 2.440693e-09` |
| 4 | `6.030536e-11 -> 5.452812e-11` | `1.683854e-07 -> 1.270969e-07` | `1.071516e-06 -> 2.262702e-08` |
| 5 | `3.538167e-10 -> 1.787193e-10` | `2.503111e-06 -> 9.918410e-07` | `1.480568e-05 -> 8.313770e-08` |
| 10 | `4.046700e-08 -> 2.128213e-09` | `2.644302e-05 -> 2.629597e-05` | `9.237444e-05 -> 1.473509e-06` |
| 20 | `5.631183e-08 -> 2.476749e-08` | `8.758696e-05 -> 9.101258e-05` | `1.523681e-04 -> 3.798106e-06` |
| 40 | `5.811660e-07 -> 5.525444e-07` | `1.440634e-04 -> 1.437992e-04` | `1.071662e-04 -> 1.930713e-05` |
| 60 | `1.184430e-05 -> 1.184427e-05` | `2.507516e-04 -> 2.505926e-04` | `7.751811e-05 -> 3.315149e-05` |

LOCK_EXCHANGE-zco: kt=2 `0.0 / 2.276825e-17 / 0.0`, kt=3
`1.184238e-16 / 7.267792e-15 / 1.355253e-18`, kt=10
`1.693460e-14 / 1.314215e-11 / 3.574534e-13`, kt=60
`8.755466e-10 / 7.496604e-08 / 1.609001e-08` -- all unchanged to the printed
digits (ulp moves from kt=8, see P7).

## Faithful-but-worse, disclosed (Rule 8)

The 3-D `u` row is WORSE after the fix at kt = 11, 12, 15, 16, 20, 21, 25, 26, 29, 30, 34, 35, 38, 39, 43, 44, 47, 48, 52
(e.g. kt=11 `2.602e-05 -> 3.321e-05`, kt=20 `8.759e-05 -> 9.101e-05`,
peak ratio `1.28x`), converging again by kt=40 (`1.4406e-04 -> 1.4380e-04`);
SSH improves 5-178x and T never worsens over the same window.  kt=2 is
bit-identical, so no operand cancelled the seed; the reading is trajectory
reshuffling of the separately-owned baroclinic `u` walk (open item 1), not a
broken compensation.  NOT reverted.

## What remains open, ranked

1. `u` walk (OVERFLOW kt=10 `2.63e-5`): baroclinic, in the stage ladder;
   grows ~x10/step from the kt=2 stage-3 remainder `4.55e-10` (faces 19/21).
   UNMEASURED owner.
2. `slow_u` (NEMO `zu_frc` = `Ue_rhs`): `1.30e-10 / 9.63e-10 / 1.90e-9` with an
   exact entry -- the next barotropic operand.  LOCALISED (measured, re-seeded
   arm, substep 1): the residual sits at NEMO u-columns 21 (`-1.30e-10`), 19
   (`+5.9e-11`), 20 (`-4.1e-12`) at kt=2, and 21/19/20 again at kt=3
   (`-9.6e-10/+6.0e-10/-9.1e-11`) and kt=4 -- the density-front faces, whose
   two columns have EQUAL level counts (25|25, no truncation, no partial-cell
   mismatch).  That refutes BOTH candidates named in the preregistration:
   (a) the step-entry `ww` guess (`stprk3.F90:217` -> `stp2d.F90:151-155`)
   reaches the 2-D RHS only through the bottom flux of a TRUNCATED u-column
   (`dynadv_up3.F90:306,350`), and (b) the F_slow min-rule live face thickness
   mis-weights only faces with a partial-cell MISMATCH (independent review: its
   normalised weight error is `4.65e-9` at one face, u-column 23, and
   `<= 1.7e-16` elsewhere, so it cannot supply `1.3e-10` in any case).  The
   `slow_u` remainder is co-located with the kt=2 stage-3 `u` remainder
   (faces 19/21, `4.55e-10`): one front-localised operand family in the
   depth-mean of `hpg + adv_up3` at Kbb.  UNMEASURED owner; the next round's
   instrument is the per-term split of `Ue_rhs` at faces 19-21 (NEMO's
   `oracle_rhs_kt2` gives `hpg+ldf+vor`; `Ue_rhs` minus its `e3u_0` mean is
   the ADV-only 2-D RHS) against legoESM's per-term depth means on the same
   state.
3. The reference-mesh literal seed (`_nemo_literal_seed_from_reference_mesh`,
   DINO) assumes `e3u_0 == e3t_0` (zco).  Correct for DINO; would be the same
   defect on a carried-mesh zps card.  Not touched (no such card certified).

## Controls run

- legacy arm bit-identical to the pre-fix tree (OVERFLOW/LOCK kt=1..10, kt=2
  frame walk);
- kt=1 19-frame gate bit-identical to clean `59d1fcbb8` (0/152 rows);
- `tests/ocean/unit/test_barotropic_seed_qco_faces.py`: the fixed seed equals
  the `e3u_0` column mean to `1e-15` of scale on the OVERFLOW kt=2 entry, the
  legacy rule differs by `> 1e-9` of scale at the shelf break and by
  `<= 1e-15` on every flat face (both fail on the reverted code);
- isomorphism tripwire (`test_no_scheme_duplication.py -k isomorphism`) green
  with the S-18 seed corollary added to the map.

## Reviews

Codex CLI and the GLM tool were unavailable in this session (no codex
binary/credentials, no `mcp__zai__ask_glm` tool), so the dual review ran as two
independent reviewer subagents on the full `59d1fcbb8..` diff (committed +
uncommitted), one on the diff contract, one on the mechanism.  Both verdicts
and the disposition of every finding:

**Reviewer A (diff, code-reviewer): REQUEST CHANGES -> addressed.**
1. The seed fix reaches every `nemo_literal` card without a carried mesh
   (LOCK/OVERFLOW, and a DINO kamm recipe built WITHOUT the bridge mesh);
   production DINO returns early via the bridge.  Disclosed under UNASKED.
2. The `legacy_seed_min_rule_faces` control was threaded only into the
   standard-halo solver and `--arm-legacy-seed-faces` only into `--kt>=2`:
   both now REFUSE where the control cannot land (fail closed).
3. `_score_substeps` downgraded masked-stagger rows unconditionally, weaker
   than the kt=1 gate: they now keep `score_frame`'s gross-zero verdict, carry
   `alignment_row=False`, are listed under `masked_stagger_rows_over_bar`, and
   can never be the first divergence.  Also taken: the reader now pins the
   `(Kbb,Kmm,Kaa)` triple to kt parity; the re-seed report names the fields
   overwritten and the fields kept.  Verified clean by A: the S-21 difference
   form (one consumer of the transport, both calls share `skip_ldf=False`,
   the same UP3 selector, no `extra_rhs`), fp32 dtype safety, the stagger
   map, both new tests as genuine one-variable controls.  Cost note: stage 1
   now evaluates two extra momentum-only `tendencies()` per step (disclosed
   under UNASKED).

**Reviewer B (mechanism, physics-validator): SHIP-WITH-NOTES.**
Verified independently: `domzgr_substitute.h90:126` `e3u = E3u_0*(1+r3u*umask)`
under `key_vco_3d`; `domqco.F90:214-219`; OVERFLOW `usrdef_zgr.F90:184`
`pe3u = pe3t`; on `mesh_mask.nc` `max|e3u_0 - min(e3t_i, e3t_{i+1})| = 0.0`
on all 16900 wet U faces; legoESM's rebuilt `e3u_0`, `umask3`, `hu_0`,
`area_T` match NEMO's EXACTLY (0.0); the `(1+r3t_e)/(1+r3t_w)` algebra
reproduces by hand; legacy arm bit-identical.  Findings: (1) the map's
"leading PLAUSIBLE owner of `slow_u`" (F_slow min-rule weights) is REFUTED by
measurement (weight error `4.65e-9` at one face, `<= 1.7e-16` elsewhere) --
the map note is corrected and the receipt's open item 2 now carries the
discriminating measurement (which refutes candidate (a) as well); (2) the
`u` row is worse at kt=11-34 -- disclosed above (Rule 8); (3) placeholders
and the regenerated walk artifacts -- done; (4) latent: `z_coord=None` now
raises (listed under choices), `min_water_column_m` floors the reference
ladder exactly as the stage kernel's `_h_ref_ws` does, `linear_free_surface`
is not honoured inside the new seed (no linssh NEMO card is certified; the
old path did not honour it either), and the carried-mesh seed's
`e3u_0 == e3t_0` assumption stays documented debt (open item 3).


## Artifacts (`/data/abyssal/dbalwada/nemo-testcases-l1/`)

| artifact | sha256 |
|---|---|
| `barotropic_walk/oracle_kt1_4/oracle_bt_frames_kt00000001.bin` | `589e5384796aa8168f834e452c7b5cd414dbd98c22938c30d60e945401148a4b` |
| `barotropic_walk/oracle_kt1_4/oracle_bt_frames_kt00000002.bin` | `b0c595a9443104a6d8a907edfbc515b6eb614635a326008acdf5bef24bcd8345` |
| `barotropic_walk/oracle_kt1_4/oracle_bt_frames_kt00000003.bin` | `bed470d3aec82686f34e4d62d97604fa5167c1a91a8c4c9edba5e9974331d77e` |
| `barotropic_walk/oracle_kt1_4/oracle_bt_frames_kt00000004.bin` | `78fc03962b0583dc0d25c2ef0cc2b2e6b9ed8d2307df9f7fc1b17367f877747c` |
| `barotropic_walk/oracle_kt1_4/oracle_overflow_bt_substeps_kt00000001_call1.bin` | `2228dfcf4564b59ddc855c1c331a5da041a9502cef17297a0696c5913a78bf46` |
| `barotropic_walk/oracle_kt1_4/oracle_overflow_bt_substeps_kt00000002_call1.bin` | `ca53d58e427710c954dbb020af9ab92cffe7bcfa60b8a12e4644ef4d24ef58a4` |
| `barotropic_walk/oracle_kt1_4/oracle_overflow_bt_substeps_kt00000003_call1.bin` | `33d717ce4d9004652e663c4a50d1b4f73ea8b655fdd805808cf0dfb9ec6529f0` |
| `barotropic_walk/oracle_kt1_4/oracle_overflow_bt_substeps_kt00000004_call1.bin` | `a5eae6347a217a749516ac81499e4a735ce9cc956fc3edcf6362eb81c0493a82` |
| `barotropic_walk/oracle_kt1_4/oracle_rhs_kt00000001.bin` | `c6d0f7233b0c48a80a4f5eecd1dd63aa83bd42ba94755665a8da5be60cba3490` |
| `barotropic_walk/oracle_kt1_4/oracle_rhs_kt00000002.bin` | `0221f254f45e7fd646adb866185799341df4cc588aa8e250eaa9f50b2ad66e77` |
| `barotropic_walk/oracle_kt1_4/oracle_rhs_kt00000003.bin` | `aba5533dc17ef30032b01e6ddd0ab35efa78fd633932a589699f5556c03595d8` |
| `barotropic_walk/oracle_kt1_4/oracle_rhs_kt00000004.bin` | `65a19a044bf4094f2f2bc85b5406e0b818dd48c01fa130a2cd93160b5b31a198` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000001_s1.bin` | `14689bb6bd84c080d2df1e1b8016f35cecc121a9fd0902eb76b627b12eea7578` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000001_s2.bin` | `fa5ae4cfbde1be709fa16f7668dc91b33d12f99e3f043729c2033dc6184591be` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000001_s3.bin` | `e2145deda38f6a657262f9acdcb02e128922da361667674bb770874d7adfd900` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000002_s1.bin` | `2e4f5eeac1e630cc852f0bef895421b93b36c3b1b816c4dd5cca08ea52523d23` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000002_s2.bin` | `c9796350594ffa659a3d945ce73a94a33b306da16c5b09997845921aedd041b6` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000002_s3.bin` | `c31e6bc12eec479393fdbb9e8706fbd83243f8be62d6be14fc89db538d530e89` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000003_s1.bin` | `0a9e0e586ebfab58d66ebd627fdc5311a51151cef701603bf87c9a4f6fc51818` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000003_s2.bin` | `d8f9bb030ddb824729f4895154c8dcc84950da33cf328e629f1a633f211275c6` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000003_s3.bin` | `1887f09f232605d1c67b7423efee9977fb7cfdc212655cf3851cbad3ce5f48b0` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000004_s1.bin` | `6385890232ddeee2a565519656f7fe12e7a491f4df1982322b3b637fbc84c948` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000004_s2.bin` | `a22770a0d12a4779442da853730d3a850bf3d5d8dd20433a2f37c93a375c0468` |
| `barotropic_walk/oracle_kt1_4/oracle_stage_kt00000004_s3.bin` | `8e10826df8661e24d394e73f77f05a3aee07039607eeaf9a13a1b012d4a4adf2` |
| `barotropic_walk/oracle_kt1_4/oracle_step_entry_kt00000001.bin` | `cf0183e563aba8b8848bc5dea470c9e50aab2d987ff5da86241e8f82494d5c66` |
| `barotropic_walk/oracle_kt1_4/oracle_step_entry_kt00000002.bin` | `3a181baa1e02586757b846b28982020fcdd55b33e7c0706adcb3efb32709f906` |
| `barotropic_walk/oracle_kt1_4/oracle_step_entry_kt00000003.bin` | `5cb5b11d1adc1f3c6c5f0ffa09e57c385ea9019a0c3766e4952714401dd5b4d4` |
| `barotropic_walk/oracle_kt1_4/oracle_step_entry_kt00000004.bin` | `9c9f355cce15fac9b03be426b7377ea86405dd0d4f8deeac7e2ca3a0b7e2ac52` |
| `barotropic_walk/oracle_kt1_4/oracle_stp2d_kt00000001.bin` | `af26e09437ec712faef4e15d29ff8d230cf46e0ccbf164041d02940ef6de5426` |
| `barotropic_walk/oracle_kt1_4/oracle_stp2d_kt00000002.bin` | `837b06995318862080227fc4bc79416bd78833de6471ae48f799ef536c253b75` |
| `barotropic_walk/oracle_kt1_4/oracle_stp2d_kt00000003.bin` | `94d8b382f9b5d50dd43600376855b21009bc78523b5a4604032e63a7b6c762ac` |
| `barotropic_walk/oracle_kt1_4/oracle_stp2d_kt00000004.bin` | `52315783ed116d61e536ebd6eb26c72d85bf70353694cfd70d2558d513a79315` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000001_s1.bin` | `378c7b0af065caa63372ba545638407d01ab8f0e7621ef6a6a216bd915da1f3c` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000001_s2.bin` | `1379150c9c1bf1dfef800f61ad0902893abf7f8d950508d03188ef9b77481b64` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000001_s3.bin` | `445eaa15c4fedf494581b5279d4d05580510103cb8585b972a5c9a5afe2d9d37` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000002_s1.bin` | `b30877d16ad75161534d4f0b8fb8e64241a479d7bbd755532dbd908498060592` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000002_s2.bin` | `f7a722551f0eb3d82b553ff28fbcecbce595490af99f40a82f5852b2a1dfe89e` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000002_s3.bin` | `0f0814e901de087bf77fc26de4a07012cb1b99a9647d36dab10f6f77b8e959fb` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000003_s1.bin` | `a6a46e545b595c0ef7c949bef5cd7e47ea63a627409b8f57a4f38aca9d036b1d` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000003_s2.bin` | `a8be98718f325b155afa9513a839f7927d6bdd8e5ee55b0aebcd4cb70eba4bd7` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000003_s3.bin` | `864333188e1b326dd964c1f9f3cdfa841b9011d28a5ec287acaa812077985656` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000004_s1.bin` | `ea4a6850ecb6cb97bccef96855e530824aff8e75684abf991b240ce08f4d4a46` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000004_s2.bin` | `b88c48ce108330caa3c129106a21a926fc32f853c5b01d28ee89454924870edf` |
| `barotropic_walk/oracle_kt1_4/oracle_transport_kt00000004_s3.bin` | `a6711a200421e6cdf75a1201c6bfa73ab8dd3dccc6058a543594634d268806b6` |
| `barotropic_walk/ssh_walk/frame_walk_kt2.json` | `203aa763e2c9bd263881ca21903ee3dc232c11975fae8235eb07f8fc619c1856` |
| `barotropic_walk/ssh_walk/frame_walk_kt3.json` | `0ee82fef8c683c4dd48e4ec844fa40b9f4dc4be83bb272eb800700e6a32d3615` |
| `barotropic_walk/ssh_walk/frame_walk_kt4.json` | `461e638eb19728304ef113dd918be6c53f6c52386495bd2b930c8fa6cd6f5669` |
| `ssh_walk/seed_fix/frame_gate_kt1.json` | `1608604c1d6866a938714af20687ad3a5ac5c091beec2876ff53fe32df610e1a` |
| `ssh_walk/seed_fix/frame_walk_kt2.json` | `d85b5b0c8f262a7cd6d9a9659f1ac6310f073cc34eced1127aaa987101e4c1ab` |
| `ssh_walk/seed_fix/frame_walk_kt2_legacy_arm.json` | `b6f45ee740de7d92c89f1b4052e1e60b1775fa5349813483566ff785791a401a` |
| `ssh_walk/seed_fix/frame_walk_kt3.json` | `2db7d28d260d43c946cb3018787e5775baed3258bf384ea1b5ca8993d5ed0236` |
| `ssh_walk/seed_fix/frame_walk_kt4.json` | `e4e102b63e4fc67f1634fbd413ff611fb6902bb431e70bc369b940baec4cde83` |
| `ssh_walk/head_kt1/frame_gate_kt1.json` | `e04c6508c3bb637d738508993480481694071e28ce1ce6b12844fe17e6be39a9` |
| `ssh_walk/s21_before/lock_stage_sweep_kt2.json` | `397e5c8544864f1b0b8d7a6b90f4b068cc1545e69c76e93d1c8a1be426d1db24` |
| `ssh_walk/s21_before/lock_trajectory_kt10.json` | `7a67e604747dd7a569fdc53fa82eaeb3ab6242cf9102eb5a0aa62298a451c7f9` |
| `ssh_walk/s21_before/lock_trajectory_kt60.json` | `7a04859e03414676549ce5a8c84e0314363d5f2ef0c01096a6e70fe025dce8d4` |
| `ssh_walk/s21_before/overflow_stage_sweep_kt2.json` | `42bcf89722775daad5edc2cfc1b3878cb534cbd3dbc45ae6eb1ad8007e148ee8` |
| `ssh_walk/s21_before/overflow_trajectory_kt10.json` | `dbf487fa82e8177afbac6f544dc1040566c7f34c908f26693542925a289e295f` |
| `ssh_walk/s21_before/overflow_trajectory_kt60.json` | `4bc184fd1fdf4254a29fbcb017152a1856ab26445aac932dca9aca5012569d28` |
| `ssh_walk/s21_after/lock_stage_sweep_kt2.json` | `a676f5ed45d25afc588061bd701a24d1baaf11b81f9cf84488aa7fe07727277d` |
| `ssh_walk/s21_after/lock_trajectory_kt10.json` | `0e689f39e804d6bd115ce8324afd96bd3f03adf8e0d18618076848c9d20a1cd3` |
| `ssh_walk/s21_after/lock_trajectory_kt60.json` | `d79dcf32b7c4edb9fb000e886d4577eff65c1884d25a398e6cdcd15b29b91893` |
| `ssh_walk/s21_after/overflow_stage_sweep_kt2.json` | `a75d19e107d25dc5860a1a3a7b4de2cebbb893cf387f3110c70f653d4b7807b5` |
| `ssh_walk/s21_after/overflow_trajectory_kt10.json` | `07685d8a1e12a4a2141be2649a380ca70805d3514f2824ceaf63a1a4d771eb12` |
| `ssh_walk/s21_after/overflow_trajectory_kt60.json` | `1bc3be88fa858f7bb8f1adf249f39f0bcd71c55711ce35bc591854d84db65410` |
| `ssh_walk/seed_after/lock_stage_sweep_kt2.json` | `d9efb4b9bd058b7d59818d549340844755b8684ad7afc9d486f8ff1ab7cffa8e` |
| `ssh_walk/seed_after/lock_trajectory_kt10.json` | `c4781249e8f5a9389635f0a93154959585ce97ba7a1964aa95b3ac0a45574b6b` |
| `ssh_walk/seed_after/lock_trajectory_kt60.json` | `f96295c403e5c611830d2ea41380ced91b33347fa0d06223b001a41345c73002` |
| `ssh_walk/seed_after/overflow_stage_sweep_kt2.json` | `a75d19e107d25dc5860a1a3a7b4de2cebbb893cf387f3110c70f653d4b7807b5` |
| `ssh_walk/seed_after/overflow_trajectory_kt10.json` | `3454ebef43e3c2529a41fcc3b068fdb460ca9dc7af17459429ac8141bb0725f8` |
| `ssh_walk/seed_after/overflow_trajectory_kt60.json` | `a5d7276f52fcfd1a0a5a97861a976e090ec36a2f678850db7ca7fb11d659f577` |
| `ssh_walk/seed_legacy_arm/lock_trajectory_kt10.json` | `1249e9020a9c0a3d8995a70c3c3a7556b13ab96e9b7381844325b1c312a01ff5` |
| `ssh_walk/seed_legacy_arm/overflow_trajectory_kt10.json` | `fce8d3f73429b6c21b846b211695dcd85d5e4888d2be46533ba29082e4dac051` |
| `ssh_walk/stats_after/overflow_statistics.json` | `7885d38e9fa2168963ea7cda76f2b0d8e2a5d6f843198c724b59e289c91de366` |
| `ssh_walk/stats_after/legoesm/overflow_zps/fp32/metadata.json` | `232943b542f2949f5499d86978a295d5ed7855fa9df839c08db025575222631c` |
| `ssh_walk/stats_after/legoesm/overflow_zps/fp64/metadata.json` | `ce74b0d6fe7bb408eb180859ccfe2291974a565e69edd2b0d953cd46fe934763` |
| `ssh_walk/stats_after/legoesm/overflow_zps/fp32/states.npz` | `f0313b2344f8e0fb4e33c57483eb8a2cb85657f313eff9e96902cb1446ee70ad` |
| `ssh_walk/stats_after/legoesm/overflow_zps/fp64/states.npz` | `94c6f7d4fe08502789c6e282fe312adf66dc998b1e8cbe108d7be234fdafdba2` |

legoESM commits on `fidelity/overflow-ssh-walk`: `f9f027525` (S-21), `14fa71454` (gate + preregistration), `04e227fdc` (seed fix); NEMO instrument copy `tests/OVERFLOW_OMIP_L1_BTWALK4` (MY_SRC diff vs BTWALK: dump gates `kt <= nit000+3`, per-kt `oracle_rhs`, new `stp2d.F90` dump).
