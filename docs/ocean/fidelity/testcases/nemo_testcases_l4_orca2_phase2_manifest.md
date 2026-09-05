# NEMO testcase Lane 4 ORCA2 Phase-2 entry manifest

Every digest below is SHA-256. Large artifacts remain under
`/data/abyssal/dbalwada/nemo-testcases-l4`; none is committed.

## Provenance

| item | value |
|---|---|
| branch base | `5bbb9ee56ad885e35abd8b244831697194e0e0c7` |
| Phase-1 accepted oracle commit | `e6f4c3719714` |
| NEMO source | `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796` |
| input archive, 1,365,673,011 bytes | `5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4` |
| preregistration commit | `325e49a74` |
| card construction commit | `2c60066a9` |

## Consumed deck and oracle artifacts

| artifact | bytes | digest |
|---|---:|---|
| `inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc` | see Phase 1 | `7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839` |
| `inputs/ORCA2_ICE_v5.0.0/data_1m_potential_temperature_nomask.nc` | see Phase 1 | `ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a` |
| `inputs/ORCA2_ICE_v5.0.0/data_1m_salinity_nomask.nc` | see Phase 1 | `ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c` |
| `runs/instrumented_reviewfix_10step_np2/oracle_step_entry_kt00000001.bin` | 14,288,048 | `e1d2251c5897ca5f93bdafa0b88d2abc412ee20fe41e681e97f6e870963337a9` |
| `runs/instrumented_reviewfix_10step_np2/mesh_mask_0000.nc` | 18,278,828 | `0f373c6609d287bc818b1f6ef96bafdcc0b620c730219018de9728369d01ec5f` |
| `runs/instrumented_reviewfix_10step_np2/mesh_mask_0001.nc` | 18,278,828 | `0b6903ce4e508bbd3f3c05dfbd8290548549ed52b13f081810e717071e977933` |

## Retained CPU gate evidence

Root: `/data/abyssal/dbalwada/nemo-testcases-l4/phase2`

| artifact | bytes | digest | exit |
|---|---:|---|---:|
| `orca2_phase2_entry_gate.json` | 1,538 | `d58e1f336eaa7910aa3a83a8d0a0bb7c5b032d7a3039f8cf0b60e8917b8d7703` | 0 |
| `baseline.log` | 1,538 | `d58e1f336eaa7910aa3a83a8d0a0bb7c5b032d7a3039f8cf0b60e8917b8d7703` | 0 |
| `plant_grid.log` | 29 | `6d8f91568ba679de0b48299cd4e38b5effdc00c9d9d56f01c935b6d6ef656518` | 1 |
| `plant_fold.log` | 32 | `cf1a9d1eabbffdddfd90328d95cb6fd30c79cb19444eb6d12f287345a24f10d4` | 1 |
| `plant_dummy.log` | 24 | `b6a1d6c568dc959d5297c2584ffc8e917f17144626644ea13c09c617cfe2ca44` | 1 |
| `plant_interp.log` | 47 | `311e8c946093f7bfe1e79d59c4880bce5cefcb94bd3893e64586b8a662cf8563` | 1 |
| `plant_T.log` | 43 | `54b60c50200cb0ec84a4441d03264f13b0ea4cddd8dd05a220b57b55ed611245` | 1 |
| `plant_S.log` | 43 | `47e31e0d42b3a76aaa1bb6652dee684dfcfb0d40435afe4a29ea561358776b35` | 1 |
| `plant_zero.log` | 43 | `b2e9dcb6ef0dbae4a4fc039933f8e605413137fd5a825569dc8f5d5f449c711f` | 1 |
| `plant_halo.log` | 44 | `89070d8a2d07a26a0819b873cdeb4fedf7c67a0be35520ceff27c7b15921e3d2` | 1 |
| `plant_coverage.log` | 41 | `b6f821d0e5e6d709504741b97f25f416ccab532a4611d8264a7289d1e945dbf1` | 1 |

All retained measurements set `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and
the card sets fp64 plus scalar-libm. The focused unit command collected 28
tests and passed all 28.

## Committed implementation and evidence

| artifact | digest before receipt commit |
|---|---|
| `packages/core/legoesm/grids/tripole.py` | `519222dfed89f2bf2d6a84ba768e922fdbb165f0df6a9fb8caa283794366b967` |
| `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` | `377c8c187c3932dd89019d353ef34ff8d336786b0abb555f2c6a19f63d402a75` |
| `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` | `9a7ff06da326765cabdcee58a768315785a06fb14f34a49dc75afa4e65db519b` |
| `tests/ocean/unit/test_nemo_testcase_recipe.py` | `c9a5cc2be7951cc6d13636b28dcb79f33e4c4c39e6331ec3908df9448c6b0135` |
| `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2_gate.py` | `48e1c827e2057074eac2d1dc72caecebc992f32c8ab0db3b8dc9c7a3b3f22e5d` |
| `docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2_preregister.md` | `1150b1b7d0c5f4b6a0bb524586f216260b470d405104309c6deb3b781aa26469` |
| `docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2_receipt.md` | `fec5465d452cd3e953b969ddc2ab41b8fd0191ca81dbcaa0e87acaac2e0691e8` |

The manifest deliberately does not hash itself. Git object IDs and the final
bundle digest are reported in the handoff, providing the immutable review
closure for all committed files.

## Phase-2j reproducible V2 and downstream handoff

Twin A is pinned as `VARIANT_ORACLE_V2` only after twin B reproduced all
92 frozen streams byte-for-byte.  The complete record manifest, terminal
restart/output manifest, identity gates, plants, V2 numerical rescores,
retained failed diagnostic attempts, WZV scalar-math build, zero-`_ZGV`
evidence and both prepared launchers are enumerated in
`nemo_testcases_l4_orca2_phase2j_artifacts.sha256`, whose SHA-256 is
`33ffce86fad37de6b8013c54a61c96f8fc48f213c21caa59ccb19a3b96b66426`.

The precise first shared debt is the frozen EEN coefficient stream
`oracle_bt_ene_coeff_kt00000001.bin`, SHA-256
`e7282ddc105300f8ee101d00ba9859eeed487a9c480947ae55fd5aac9f2c9363`.
Lane 4 makes no shared external-mode change.  With the external result supplied
from the oracle, the stage-1 horizontal `zFu/zFv` products are exact.  The next
user-shell acquisition adds one WRITE-only WZV/runoff operand stream; its
executable SHA-256 is
`78d0a06c2f21cd174579d60d65083e050fa396ec321ad33a6a69a8cbae89357a`.
No new large artifact is committed.

## Phase-2k reproducible WZV extension and shared-clock handoff

The two Phase-2j WZV acquisitions produced 93 / 93 raw-byte-identical
records.  Twin A extends `VARIANT_ORACLE_V2`; twin B is the retained witness.
The 93-entry record manifest has SHA-256
`ea400055d19bb31be0242a397872df5441a9893d8bd94b2419ae029f96c3449d`.
The appended 24,899,672-byte WZV record has SHA-256
`245be2ea348002b93358e5dd723f0b84198af47c3d38f0bc0bb1acb0fc3100fe`.
Four restart shards and eight history payloads remain exact against the
uninstrumented variant control; all legacy, surface, O1, reproducibility and
WZV plants exit nonzero.

At the resolved stage-1 clock, transport divergence, surface runoff, QCO WZV
and `pFw` are each exact at 0 / 233,341.  The production full-step clock arm
is over bar at 233,341 / 233,341, maximum absolute error
`7.329623319094706e-05 m s-1`.  This is
`GYRE_OWNER_SHARED_WZV_STAGE_CLOCK`; Lane 4 stops without changing that shared
wiring.  The complete data-artifact inventory is
`phase2k/nemo_testcases_l4_orca2_phase2k_artifacts.sha256`, SHA-256
`7f8525dd42568c69a81266a6b120fc61dce1676205e58a88cc69634a9617edad`.
No large artifact is added to Git.

## Phase-2b post-`sbc` exchange handoff

This continuation stopped before legoESM arithmetic because the inherited SI3
record precedes iceberg, runoff, FWB, final halos and U/V interpolation.  Large
artifacts remain under the same Lane-4 data root.

| artifact | bytes | digest |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2b_exchange_schema.exe` | 54,904,016 | `27f02c52b9509201319136deecdf36321ce31324c260796fe646f3bad73a91ec` |
| `build/build_ORCA2_OMIP_L4_phase2b_exchange_schema.log` | 4,711 | `2262044d9422a8c5553a09a3fd3f147918ef74b5dfeeef0ee734aa614f3d7dc3` |
| `build/phase2b_exchange_schema_MY_SRC.sha256` | 2,100 | `5dd1f5079839e1bccb91efbd7ff694ed384bffce2bff941c27aab941ec4df337` |
| `build/phase2b_exchange_schema_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `manifests/instrumented_phase2b_exchange_schema_10step_np2_prepared_all_files.sha256` | 11,265 | `b3d61cd6833e22286d522e7b68daa9ea66022a006a2f07b4b8e2c72362b266f9` |
| retracted flat-schema current census | 10,824 | `6bb61199452c58ca4661b6552e1fa3f2cee71128a5f47a91291d2ec45f284a9a` |

| committed/prepared file | digest before handoff-receipt commit |
|---|---|
| `cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3.F90` in NEMO data tree | `1d8d6ef62550d757859c282865e14d7f24fb28a51069799cf27271cad1b59bf7` |
| `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py` | `007be5c29655000705c8d515527b393bf2393b96dbb0ce08323dac9f641c3bed` |
| accepted schema-correct launcher | `56dc3104e2f7c56d35d16c87a5687d4d3f7fc0fc3b55e4a492abda43fba72351` |
| retracted launcher, exits 69 | `7edbff8a2f9930148407d6eecc257c1aeb89487cbf9ac85491d0c4a28898e5c6` |
| Phase-2b preregistration | `e269ba82d2c36285c010f1169eae308d667f7e8589758aab6c069b2bf5d98b02` |

The Phase-2b preregistration commits are `109152161` and `383dcefdc`; the
instrument/gate/launcher preparation commit is `9a816aaaa`.  The handoff
receipt is intentionally hashed by the final Git commit rather than by this
self-excluding manifest.

## Phase-2c icebergs-off VARIANT handoff

User Decision 7 supersedes the unexecuted icebergs-on Phase-2b handoff.  The
old artifacts remain `SHIPPED_DECK_RECORD`; its retained launcher now exits 69
and has SHA-256
`895684724ca23e424f6c2991186f522f201f3b94697980b044b8087cc0761d3d`.
The comparison arm changes only `ln_icebergs=.true.` to `.false.` and is
labelled `VARIANT`.

Preregistration is commit `82974b40d`, the Coriolis repair is `af54d11ef`,
and the gate/launcher preparation is `ade8a9ffd`.

### Geometry repair and retained CPU evidence

| artifact | bytes | digest |
|---|---:|---|
| `phase2c/entry_baseline.log` | 1,538 | `d58e1f336eaa7910aa3a83a8d0a0bb7c5b032d7a3039f8cf0b60e8917b8d7703` |
| `phase2c/orca2_phase2_entry_gate.json` | 1,538 | `d58e1f336eaa7910aa3a83a8d0a0bb7c5b032d7a3039f8cf0b60e8917b8d7703` |
| `phase2c/plant_coriolis_swap.log` | 35 | `84fccf7d19dec25931c39584d667cf8836f8952fb95c0fb2e2a0397f9330269a` |
| `phase2c/focused_tests.log` (`28 / 28`) | 521 | `55a3c3b4c29e698629b29cee539e384afe0f2fb16f71b85ff9079f2bba96b80d` |
| `phase2c/synthetic_variant_surface_input.bin` (test only) | 4,101,508 | `8e4ab66260aa8ef324989bdc85efe6b5c20b4dd4ef0b405d55cc4481426c11a4` |
| `phase2c/synthetic_variant_surface_gate.log` | 274 | `a9b602b34b9a3893cc6d1feb0516cd58e57c91e214632d3bef7c8e782320813c` |

The repaired entry artifact is byte-identical to the Phase-2 entry baseline.
The Coriolis swap plant exits 1.  The synthetic surface record passes the
derived 35-field schema and all six mutations return `PASS_NONZERO`.

| committed implementation/gate | digest before handoff-receipt commit |
|---|---|
| `packages/core/legoesm/grids/latlon.py` | `32e103185cded916807e74b065a60101aacd306f0faee6f1a277aa7edc911739` |
| `packages/core/legoesm/grids/tripole.py` | `da42cfda711a232c44f19dc3cba21371f9766311c26f472bd609e1429f2fd502` |
| `packages/core/legoesm/grids/halo_latlon.py` | `db7b830502ad6ea192eace5f9cd8245431c777539d3918bc3aed399ef6292afd` |
| `packages/core/legoesm/parallel/latlon_mpi.py` | `d46c2eb69dc62544f5a8205d7247494c744c7fa5b76fffb7082da546f7ea037d` |
| `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` | `448c5f8a63c14c550e23e38d2f68117c95ba8d03ebdeb4ee4c4da46f99cb64f7` |
| `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` | `c45f98296990e3196d02cf81f6d4d17a07a24d5725bef99f9ccbf21223083491` |
| `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` | `1a0494ac769c7229f2cbeaf264152693531f3754956790c19e9d804b88d08f4c` |
| `packages/ocean/legoesm/ocean/state.py` | `a30bbd8e2182a56f70543b1a8a4dfe45419225ba111d1c60e02a7f17bcf50fef` |
| `packages/ocean/legoesm/ocean/vertical.py` | `ab8f835f01fb5a23c8a824b839c703eaa645bb236c7f0d8603a1f1f8ba4709ae` |
| `tests/ocean/unit/test_nemo_testcase_recipe.py` | `903ada0acc20c7175037095a7ae9b1c7cf79b3000a79a0f120c1d95f3101a4a1` |
| `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2_gate.py` | `e39ea6db689aa8f959d0df171911a526cd1133c4c81827a71fb7c089cb2b6c8a` |
| `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py` | `f834b1783a6ed4f1263f94c4fea917c264b4e6a1df847a670261951b5a416834` |

### NEMO build and prepared runs

| artifact | bytes | digest |
|---|---:|---|
| copied-config `MY_SRC/stprk3.F90` | source | `045ca27f0679bdbf4c11cd150e3ccd56ab0a84ac8b0c6beac733de09b3eb757f` |
| `build/nemo_ORCA2_OMIP_L4_phase2c_variant.exe` | 54,904,016 | `b31fc33edd3109a41640f9fb59f915d90a52f1f0509fde7c33254cf46b896f28` |
| `build/build_ORCA2_OMIP_L4_phase2c_variant.log` | 4,642 | `9777ab99e03c848eb3390ecf8cc362c17c82a18edc37d5b2d5b7e88c79dcf73d` |
| `build/phase2c_variant_MY_SRC.sha256` | 2,100 | `54b3bff73af525906e1117cf26af23c75609de4307a44c169c9156ff001edfba` |
| `build/phase2c_variant_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| accepted uninstrumented scalar-math executable | 54,746,696 | `c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343` |
| superseded icebergs-on current 63-item census | 11,265 | `72a4891487ade34553f0e3631c1fb8dab46d1e577251e50982ae611af38db649` |
| instrumented ten-step 63-item census | 11,076 | `0229bc1e9b92427e6f6b107962453fa3eac6f33dd5b11cf008b15acf7c21eb5a` |
| uninstrumented ten-step 63-item census | 11,202 | `28bb86dfd516ea826b2e308a5889eda61e0fb27c20af9f9736e1c080d7dd253e` |
| uninstrumented 30-day 63-item census | 11,139 | `60493aa81a7c8188dba9821b9b877238a9f82285e92d49d77f5e4404e7ccc05e` |

| committed launcher | digest |
|---|---|
| `run_variant_icebergs_off_instrumented_10step_np2.sh` | `e9f16a444616e14faefdb633adbe4f68981e4c04599829585e696935f9d1e63e` |
| `run_variant_icebergs_off_uninstrumented_10step_np2.sh` | `b647d4a3c1f302f26d44bd078e1787e879828c8ea794807dbd96920e69b519a5` |
| `run_variant_icebergs_off_uninstrumented_30day_np2.sh` | `af46ce7583c4d077a4db4f45ba6dd83bd229819bfb17d7aac7833bdbb471022a` |

The two ten-step deck manifests are
`e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059`;
the 30-day deck manifest is
`2ce5f93ae9d6355e1d4d94e2f5e4576f98eec3ec0e61f3abb69d8c149028683a`;
all three input manifests are
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
The final handoff receipt and this self-excluding manifest are bound by the
final Git commit and bundle digest.

## Phase-2d accepted VARIANT and O1 acquisition handoff

Phase-2d descends from `6e935823577b3a60056e96bfca429f43e84ecac3`.
The ordered ladder preregistration is commit `9ebd3c91e`; the accepted-oracle
gate is `b07d7107d`; the explicit icebergs-off card and entry gate are
`3a3f12e03`; and the O1 acquisition addendum is `d433d2f48`.  The comparison
oracle is the `ln_icebergs=.false.` `VARIANT`; shipped-deck records remain
preserved but are superseded for comparison.

### Accepted run and gate artifacts

Root: `/data/abyssal/dbalwada/nemo-testcases-l4/phase2d`

| artifact | bytes | digest |
|---|---:|---|
| `variant_oracle_gate_final.json` | 116,573 | `43c6d0d26503548ecec54f9c16d62ac60de1e3477cc82ee621ff6f7d2d9d71e3` |
| `variant_oracle_gate_final.stdout.log` | 116,573 | `43c6d0d26503548ecec54f9c16d62ac60de1e3477cc82ee621ff6f7d2d9d71e3` |
| `variant_legacy_90_records.sha256` | 8,987 | `790f7e0b46d9beaa11e62a4d0e8bef9091e40fd8a635b4c3db990288ed6c908a` |
| `variant_all_91_records.sha256` | 9,095 | `d528c89cc82c9c5cc2ae8b203e40cd296ce1067f8742fc5f06daeda8c1588684` |
| accepted post-`sbc` surface record | 4,101,508 | `42d1f9735a17652e6d00d4e1641cb513b67bcc221d11605d27a7971483ed16eb` |
| `shipped_phase1_gate_regression_v2.json` | 82,458 | `cd34c630d67aa6d19bc9fdb163d581be0d347f674b474ddde8fb2414f7f5d93f` |
| `shipped_phase1_gate_regression_v2.stdout.log` | 82,458 | `cd34c630d67aa6d19bc9fdb163d581be0d347f674b474ddde8fb2414f7f5d93f` |
| `orca2_variant_entry_gate.json` | 1,555 | `aad0f9a637484f5f05debf54e5e58e1eeed966d8a71eda557770bf03d610a202` |
| `orca2_variant_entry_gate.stdout.log` | 1,555 | `aad0f9a637484f5f05debf54e5e58e1eeed966d8a71eda557770bf03d610a202` |
| instrumented ten-step complete manifest (187 files) | 17,481 | `a7c1a30836c4651a99f71ad748ce292c2f66a2f7d4d16b0d2e046150fd4fe081` |
| uninstrumented ten-step complete manifest (96 files) | 8,386 | `f9edd367755fe8d67f1f7b123a38aafbb15b39bd4a99001ebdb34b6315155fb3` |
| uninstrumented 30-day complete manifest (93 files) | 8,131 | `6f82643f2d15bf9084cfbc39b782ea141270465b2d211ae2d4ee19cea64bfdbf` |

The accepted gate schema-walks 91 records, reports dynamic identity counts of
four restart shards and eight history payloads, and passes ten inherited,
six surface, and one cross-run planted controls.  The entry gate runs CPU,
production JIT, fp64 and scalar-libm; all five entry fields are `0 / n`, and
all eleven entry plants exit nonzero.

### O1 operand acquisition

The ordered ladder stops before arithmetic because the accepted records do
not separate full post-`fld_read` mapped inputs from the pre-SI3 NCAR bulk
leaf.  One rank-0, kt=1, WRITE-only record has therefore been prepared.

| artifact | bytes | digest |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2d_o1.exe` | 54,920,400 | `c9c25e1aeb17d88f8b41e6a3f263d0684678067d3aba2931913effe3da55503b` |
| `build/build_ORCA2_OMIP_L4_phase2d_o1.log` | 10,867 | `c074be7470f1435c26f3475e17eeab75691ab8dd97429da63560ce347bb27cb3` |
| `build/phase2d_o1_MY_SRC.sha256` | 1,092 | `7009f39fe854f89173bceba26fcf4bb8a8034cb228650e63a8586905a52d49a1` |
| `build/phase2d_o1_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| copied-config `MY_SRC/sbcblk.F90` | source | `75177950f32faa52bb3e59111b6adec6b4b07cf8f6044720a8873810671a0eed` |
| prepared 63-item run census | 5,343 | `512ba04697f897991005542a5ae9115419f48b902639269cb065152219a3be16` |
| committed acquisition launcher | 2,272 | `88590198b3222fc8c76eb5c68404f24c19cc3c98abf0978ecdc0f1bd3c6d768c` |
| `phase2d/o1_synthetic_schema.log` | 1,261 | `3256b2a69675db6c1708aade22d7896ed6914709a2c9064b1df4286de541a3e9` |
| failed missing-`.venv` synthetic command log, retained/retracted | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The synthetic walk derives two headers with 9 and 20 fields and a total of
3,090,336 bytes; both header-count and digest-bound one-ULP plants return
`PASS_NONZERO`.  The final handoff receipt, acquisition gate, and this
self-excluding manifest are bound by the handoff commit and bundle digest.

## Phase-2e malformed O1 record and schema-fix handoff

The corrective preregistration is commit `6dba63974`.  The executed acquisition
run is retained but retracted as an O1 oracle: frame 2 has 23,232 trailing
bytes because three full-domain arrays were written under a reduced-domain
header.  No O1 numerical comparison was made.

Root: `/data/abyssal/dbalwada/nemo-testcases-l4/phase2e`

| executed-run evidence | bytes | digest |
|---|---:|---|
| malformed O1 record | 3,113,568 | `982347ad6f617e388104c108a782b21e79d2ee95583872849cd865b8e24c4f31` |
| `o1_gate_malformed.json` | 54 | `fd2e1c1cf015fe62be68b43e391b9b5fef37f3a29de518a2a7d2a968d40f8ee1` |
| `o1_gate_malformed.stdout.log` | 54 | `fd2e1c1cf015fe62be68b43e391b9b5fef37f3a29de518a2a7d2a968d40f8ee1` |
| `inherited_and_ordinary_identity.json` | 40,027 | `f3da75ce9657bd201734ec06eff397fc9e86acecdb2419f5b7005605a2044963` |
| `non_o1_schema_summary.json` | 30,364 | `869857b953aed24c1f289eecba969068db01822960803737f9ff4137dc1c5e65` |
| complete executed-run manifest, 188 items | 17,579 | `84f154edccfc9cb8c3e0303003f85d87000e87bd70fd2abae644bca949471196` |

The ordinary-output gate reports 4 / 4 exact restart shards and 8 / 8 exact
history payloads under the timestamp rule.  Strict inherited-record identity
is 84 / 91; the gate retains all per-file digests and fails closed pending an
explicit decision about source-undefined slots.

### Schema-fixed build and prepared run

| prepared artifact | bytes | digest |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2e_o1_schemafix.exe` | 54,920,400 | `a39000462c7da6278faa197757ffa6fb78593e3001862054cc197d617b1aa20a` |
| `build/build_ORCA2_OMIP_L4_phase2e_o1_schemafix.log` | 10,994 | `2b1b3f170c26165bde40035137cf67504db13509f670e41c277e6c21f5e54d64` |
| `build/phase2e_o1_schemafix_MY_SRC.sha256` | 2,100 | `91bac34d01d7eb0517637e66ff7ebcaa74a9570a9785eeb413809d2828457cf3` |
| `build/phase2e_o1_schemafix_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| copied-config `MY_SRC/sbcblk.F90` | source | `fcb1d0add456f709b49c78e1e6a1c12fd79c59b80da81fe2f55ef7fe03d6c3ab` |
| prepared 63-item run census | 5,343 | `121b3b7827ba64653178f93b150d217a10ced598342eb82f915d9611e08d2078` |
| committed schema-fix launcher | 2,282 | `44b826fdea3b063e31242088755337093b5997e5232942d1280fa6a76662752c` |

The acquisition gate and final receipt are hashed by the handoff commit.  This
manifest deliberately excludes its own digest; the bundle digest closes the
review artifact set.

## Phase-2f O1 validation and canonical-writer handoff

Decision 8 treats the seven differing inherited streams as instrument hygiene.
The schema-fixed run passes O1 and ordinary-output validation.  Its inherited
records are 84 / 91 raw exact and 91 / 91 exact over source-defined bytes; seven
one-ULP defined-slot plants and both O1 plants return `PASS_NONZERO`.  The
994,665-byte JSON and stdout under `phase2f/` each have SHA-256
`c778b59ac130fbcaf21e184246e2b99fb03d025bf3a13feaacc928f4bbc85d89`.

The config-local writers now emit zero-first canonical views over the owned wet
rank-0 region.  No model field is assigned and no frozen schema count changes.
One scalar-math executable is shared by two prepared 10-step runs.

| artifact | bytes | digest |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2f_canonical.exe` | 55,129,776 | `5befe21268dc7e2487525aadf4dec4fb1660451eb70bc33c2ffa8e6ceda196c5` |
| `build/build_ORCA2_OMIP_L4_phase2f_canonical.log` | 7,388 | `eb4e03322be645cab982501843ae51bd632617f28ad876908520d07248fc2cc5` |
| `build/phase2f_canonical_MY_SRC.sha256` | 2,270 | `5a9bbfdaed3e7d4da05b6fb389fdd7cb058255b9bbcb33a29809220fe1d563a9` |
| `build/phase2f_canonical_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| arm-A prepared 63-item manifest | 11,517 | `cc6b165c962fc638df48490f4375d8ddbc6a9a287de1fca52e2b2f6fb7ff2589` |
| arm-B prepared 63-item manifest | 11,517 | `9ca72ba32c2a539b93ac1b7ad09c9f2a56e14b7e0bf1c863f6f99b56eae1b160` |
| committed arm-A launcher | 2,276 | `c1bf68b6a71e31813bb00ac46355c8bd39d1576bdfd6362d99ea1a91b2ab84da` |
| committed arm-B launcher | 2,276 | `2abd81f23c9c4fbbeee81cb9df5c3be659788c169a2d2764a1ff5b2fb8ecc62d` |

The handoff receipt and this self-excluding manifest are closed by the handoff
commit and bundle digest.  The numerical ladder is not entered in this turn.

## Phase-2g VARIANT V2, O1 map, and shared-bulk handoff

Phase-2g descends from `d44ca7abaaf4461ffb57014bbff31d89216c9c26`.
Preregistration is `f93e43a01`; the defined-field identity gate is
`e46670bdd`; the shared `fld_read` implementation and numerical gate are
`ee1bbfcc2`; the canonical O1 writer gate is `a49189b63`; the replacement
launchers are `36f036ebe`; and the explicit card selector is `5260fa8ae`.

The Phase-2f twin runs prove 91 / 91 inherited records raw-byte exact.  Run A
is pinned as VARIANT oracle V2; its O1 remains provisional because two complete
source-unowned fields differ between twin allocations.  Defined O1 storage is
359,640 f64 exact and excludes exactly 26,640 f64 values (`pcd_du` and `qlwn`).

| retained evidence under `phase2g/` | bytes | digest |
|---|---:|---|
| `variant_v2_all_91_records.sha256` | 9,095 | `9455cccb19ccbecd06f457fb6db71f8fdebc8bfebd2dc4d4909e5cb25de46668` |
| `variant_v2_legacy_90_records.sha256` | 8,987 | `13b34464996a1967ce8f5c93cf59ff62ea77cbcea4de0880c8b19f8164b1d26c` |
| `canonical_a_phase2b_gate.json` | 116,574 | `fe07a0dc8e6d3382c282fc647cecaa9cac46448f68603e98f7fe592b90d39d62` |
| `canonical_b_phase2b_gate.json` | 116,574 | `c4420e784c780b7924ed79a375bcd56dd0e5d7d1ecd436022ec8fd23896c0d6c` |
| `o1_defined_twin_gate.json` | 197,768 | `a2e841d22ced403946826e4a8b0b1bcbb8beb32d441943567148a521cc640718` |
| `o1_numerical_gate.json` | 4,244 | `1055cccaebc49b8da1893ba74fa8981c33a56f8affac7c3e26a5af87dfff736d` |
| `card_entry_gate.json` | 1,503 | `893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12` |

O1-M is exact for all nine CORE fields, each 0 / 13,320.  O1-B is the first
over-bar boundary: the oracle-input-substituted shared NCAR bulk leaf has debt
in all seven compared outputs and is owned by `LANE3B_OWNER`.  Per the standing
ownership rule, no Lane-4 shared-operator fix was made and later ocean stages
were not entered.  SI3 remains oracle-supplied and unmeasured pending its
branch merge.

The O1 acquisition writer now emits zero for its two wholly unowned fields.
The replacement scalar-math binary and twin-run handoff are:

| artifact | bytes | digest |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2g_o1canon.exe` | 55,129,776 | `a34c795bb273b338f5b9dd271843aa46f123d4af3dea7f8de8c028b515ddf31a` |
| successful build log | 10,866 | `c0d73f008e64653e49d3c26536bab5af880f6fa07845dd853857349aeccf5a74` |
| retained/retracted failed build log | 3,391 | `a07079631b4808d02c036b9c4a519c7228870e6b594e537f923ce875da3a9d10` |
| `phase2g_o1canon_MY_SRC.sha256` | 2,270 | `ac8b80f6cdb2eda8c9cf5a5c2e102ce1d9785dc2f3ac1ffdc8dedb2f1dad04b1` |
| `phase2g_o1canon_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| arm-A prepared manifest | 5,343 | `828b6ad4802197576068e0b01332aca543ffc2984d468b16b97778e3c6404c81` |
| arm-B prepared manifest | 5,343 | `9deb253ddf3a3a2de9d257105c543146c81e6ebfceb29d70ede39985ec63de75` |
| committed arm-A launcher | 2,274 | `29a22e637398a1a4bab5d6927eb3f318cfa41da240c63edc1297360c57d15a95` |
| committed arm-B launcher | 2,274 | `7d900db5693b030837953a681125cce114d1448b47f0e017dacd8ef69af89ea5` |

The earlier Phase-2f receipt's layout label is corrected to the actually
resolved `jpni=2`, `jpnj=1`.  No run bytes or namelist values are changed by
that documentary correction.  This manifest deliberately excludes its own
digest; the final handoff commit and bundle close the committed evidence set.

## Phase-2h systematic writer audit and replacement-twin handoff

Phase-2g's O1-canonical twins contain 92 records each.  The fail-closed raw
gate reports 91 / 92 exact; only the stage-1 transport operand record differs,
with 232 `zub` and 198 `zvb` f64 slots.  Accordingly, the former 91-record V2
label is superseded by `CANONICAL_CANDIDATE_91`.  No complete VARIANT V2 set is
pinned before the systematic twins return 92 / 92.

| retained/prepared artifact | bytes | digest |
|---|---:|---|
| Phase-2g twin reproducibility JSON | 27,544 | `6f3f06182af0ba8330c3b1e66ff102683684dae156c697f3512dfa582a2fd69e` |
| binding reproducibility plant JSON | 27,469 | `edfbc9786cc7b59cb6fa2a55f715e0a1c5a83786267fb55d4759aafca7275513` |
| 92 / 92 A-versus-A polarity control JSON | 27,469 | `02c1e0fcbe03ec73b9370f2f0cbc08563ab44af3175d1ad17714974f47e5e84e` |
| systematic scalar-math binary | 55,519,664 | `c47a1a6bd9b1873f28cf3eaa34a6264cd3ed65e24d16f8d145c1137e2671a308` |
| successful build log | 77,616 | `7146ac801ee88981a4dac1115970588550eec3062200d436bf60e0ad50e198b3` |
| preserved failed build log | 37,452 | `efd317c60b1abb9d66b6bd8cf59fadbeddf853aef1e7750b3cc682dbabe780d0` |
| systematic MY_SRC manifest | 1,190 | `4ea891a7810f5818b6f3c6d15bb5dbcd49a44a4b8d808f6756fda6e9460106ff` |
| zero-line `_ZGV*` result | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| arm-A prepared manifest | 313 | `3c3c437452df02a2dbbbd4ed3f6bd87c185857319acf6215136043f3dd3dc20e` |
| arm-B prepared manifest | 313 | `f339aff29876373dddaac09d6149509964f9e2d5bc4c00a230c56073e3347570` |
| arm-A launcher | 2,277 | `b0d479d2c9db7b66a25489d608e6dbe272cd9c4ca6c68733890414fa58f0a381` |
| arm-B launcher | 2,277 | `137772b2c9c41691cc91c7a91fd73092e195e57aeeea688b5d5d206f821eaf45` |
| repeated O1 numerical gate JSON | 4,244 | `7541f93a973d5bc1a2f6841689c1b49ac01d7cd01a9f5813de4111d281a398f5` |
| post-bulk RGB gate JSON | 1,055 | `236e507cb354cef9aaf45e805c29f087ab803d37805d2b800d812104a11a8e76` |
| RGB binding-plant stdout | 58 | `e5cffa2a0c66073742c9bbc0024ad0b106103a70d1e1b513b410ef3309df59bd` |

The exact O1 handoff record is 3,090,336 bytes, SHA-256
`751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045`.
All nine CORE/fld_read fields remain exact at 0 / 13,320.  O1-B remains a
`LANE3B_OWNER` debt and is handed to the ice-thermo lane against that record.
Using oracle-supplied bulk and SI3 outputs, the continued ladder's first debt
is O2-RGB at 149,647 / 233,341 wet T cells, owner `ORCA2_OWNER`.  Downstream
boundaries were not entered.  The Phase-2h receipt and this self-excluding
manifest are closed by the final handoff commit and bundle digest.
