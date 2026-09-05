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
