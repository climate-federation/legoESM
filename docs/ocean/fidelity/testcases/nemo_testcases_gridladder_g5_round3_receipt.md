# NEMO test cases: grid ladder G5 round 3 receipt — ice/ocean top drag

**Date:** 2026-10-16  
**Rung:** G5 = certified G4 + ORCA2 SI3 only  
**Verdict:** the review fixes land; **G5 remains HELD, not certified**

The preregistration was committed at `784857c14` before candidate
measurement. The original candidate put wind into the barotropic RHS before
the top-cell drag residual. The independent Claude review held it because NEMO
calls `dyn_drg_init` first
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:219`) and adds
wind afterward (`:228-231`). Commit `568ea3fde` now preserves that order.

The fix removes the entire kt-2 U/V/ssh debt and makes both the external-mode
exit and the stage-1..3 U/V/ssh handoffs bit-exact. GIVEN-NEMO-ENTRY is now
AT-BAR through kt 10. The independent chain remains AT-BAR through every
surface field at kt 9, but its all-level U row first becomes DEBT there; at kt
10 surface and all-level U/V are DEBT. The certification condition is
therefore not met. The remaining owner is registered as
`kt9_independent_all_level_u_continuation`.

## NEMO deck and source

ORCA2 resolves `ln_drgimp=.true.` and `ln_drgice_imp=.true.` from
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/EXP00/namelist_ref:817-818`; the general
top-drag defaults are `rn_Cd0=1.e-3` and `rn_Uc0=0.4` at `:823-824`, and SI3's
ice/ocean coefficient is `rn_Cd_io=5.e-3` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/EXP00/namelist_ice_ref:136`. For this
no-cavity deck, `zdf_drg` copies `rCdU_ice` to `rCdU_top` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/zdfdrg.f90:178-190`; SI3
forms the recorded, non-positive value at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/iceupdate.f90:403`.

The four executing consumers, opened at their enclosing routines, are:

| consumer | G5 compiled source |
|---|---|
| U/V frozen external coefficient | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:1428-1433` (`dyn_drg_init`, :1404) |
| top baroclinic residual, forward/NOW branch | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:1466-1489` (`dyn_drg_init`) |
| non-wetting external-mode drag | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:633-640` (`dyn_spg_ts`, :136) |
| `dyn_zdf` top barotropic RHS and U/V implicit diagonal | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzdf.f90:158-180`, `:303-314`, `:470-482` (`dyn_zdf`, :67) |

The apparent naming mismatch in the third row is NEMO's own source: this deck
has `ll_wd=.false.`, so the feature called implicit ice drag executes the
explicit external-substep statements at :635-639; the 3-D solve remains
implicit.

## Landing and order correction

The record already contains `rCdU_ice`; no acquisition was needed. D123's G5
forcing route supplies it through `OceanSurfaceForcing`. The shared face helper
preserves its negative sign, U/V staggering, north fold, association, and
NEMO's positive signed zero under a traced `jax.jit` input
(`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4077-4107`).

The residual now starts from the pre-wind depth mean, while the already-correct
post-wind value remains the operand of the 3-D perturbation split. After the
top residual is accumulated, wind is added once
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6660-6803`).
This is the literal ordering of NEMO's `CALL dyn_drg_init` at `stp2d.f90:219`
before its wind loops at `stp2d.f90:228-231`.

The same coefficient continues to feed the external substep loop
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7613-7619`)
and the `dyn_zdf` top RHS/diagonal
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13539-13574`).
`implicit_ice_ocean_top_drag` remains default false and G5-only. Missing input,
non-implicit vertical mixing, a non-baroclinic-only solve, bottom-drag
composition, or the wide-halo external solver still raises.

## Input, coefficient, and signed-zero controls

The kt 1..10 forcing identity has 16,433 wet T cells and zero unequal input
rows. At kt 1, all 3,630 ice-covered wet cells have nonzero `rCdU_ice`, range
`[-1.5504269195258426e-3, -4.816384579465245e-10]` m/s.

The coefficient test now passes `rCdU_ice` as a traced argument to `jax.jit`,
not as an eager or closed-over constant. It scores all 15,853 wet U faces and
15,875 wet V faces, including 68 north-fold V faces: zero unequal, zero maximum
absolute error, and zero negative-zero card faces. The coefficient plant still
refuses with exit 1. The earlier prose claim based only on eager execution is
superseded by this compiled measurement.

## Consumer non-vacuity

The committed full-model gate first proves the ordinary kt-1/2 chain AT-BAR,
then omits exactly one downstream consumer through a private static test hook
(`tests/ocean/fidelity/test_nemo_testcase_g2_orca2bathy_ladder.py:414-438`).
Each omission is red:

| omitted consumer | first red row | witness |
|---|---|---|
| top baroclinic residual | kt 2 ssh | DEBT, max `8.141e-02`, 10,505 unequal |
| external-substep drag | kt 1 ssh | DEBT, max `1.814e-02`, 10,109 unequal |
| `dyn_zdf` top RHS/diagonal | kt 1 U | DEBT, max `2.914e-01`, 3,519 unequal |

Thus consumers 2–4 cannot be removed while retaining the G5 result. The
controls change no public configuration and are not constructible by cards.

## kt 1 staged score — all levels and barotropic rows

Bar is normalized max absolute error <= `1e-15`; `A` means AT-BAR. U/V/T/S
summaries cover all 30 carried levels split by NEMO's full/partial depth class;
the displayed max/unequal is the surface row, whose counts equal the all-level
counts for the only non-exact stage-2 cells.

| exposure | u | v | T | S | ssh | uu_b | vv_b |
|---|---|---|---|---|---|---|---|
| stage 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| stage 2 | A 0/0 | A 0/0 | A 1.110e-16/1 | A 7.105e-15/2 | A 0/0 | A 0/0 | A 0/0 |
| stage 3 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |

Stage-2 T and S are **AT-BAR but not exact**. Their all-level full-cell
relative errors are `3.69e-18` and `1.75e-16`; partial and dry rows are exact.
Stages 1 and 3 are bit-exact on all levels and all barotropic rows.

## Both labels, kt 1..10

Cell format is `status max_abs/unequal` for the surface or 2-D row.

### INDEPENDENT

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| 2 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/2 |
| 3 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/2 |
| 4 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 1.066e-14/3 |
| 5 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 1.066e-14/4 |
| 6 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 1.421e-14/5 |
| 7 | A 1.887e-15/428 | A 7.633e-17/490 | A 7.633e-17/500 | A 2.269e-15/434 | A 2.345e-15/435 | A 0/0 | A 1.421e-14/5 |
| 8 | A 1.332e-15/1457 | A 1.804e-16/1717 | A 2.220e-16/1755 | A 1.291e-15/1525 | A 1.225e-15/1516 | A 2.220e-16/2 | A 1.421e-14/8 |
| 9 | A 2.665e-15/2927 | A 2.776e-16/3206 | A 4.441e-16/3232 | A 2.234e-15/2716 | A 2.325e-15/2696 | A 4.441e-16/7 | A 1.421e-14/16 |
| 10 | A 3.553e-15/4972 | A 5.551e-16/5465 | A 7.034e-16/5447 | D 2.084e-14/4506 | D 2.336e-14/4539 | — | — |

There are 68 surface rows: 66 AT-BAR and 38 bit-identical. The 36 all-level
rows contain 33 AT-BAR. The first DEBT is kt-9 U: full relative `4.819e-16`,
partial relative `1.252e-15`. At kt 10, all-level U is `4.157e-15` full /
`5.711e-15` partial and V is `2.521e-15` full / `3.494e-15` partial.

### GIVEN-NEMO-ENTRY (odd entries plus their two-step continuations)

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| 2 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/2 |
| 3 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| 4 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/1 |
| 5 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/1 |
| 6 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/2 |
| 7 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/1 |
| 8 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/2 |
| 9 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/3 |
| 10 | A 0/0 | A 0/0 | A 0/0 | A 1.665e-16/5 | A 1.874e-16/5 | — | — |

All 68 surface rows and all 20 all-level rows are AT-BAR; 59 surface rows are
bit-identical. Before the order fix, the receipt's GIVEN-NEMO-ENTRY kt-3 U/V
rows were DEBT at `5.845e-14` / `5.945e-14` (the requested 5.8e-14 debt).
After the fix both are bit-exact. This distinction supersedes rather than
silently deleting the reviewed pre-fix result.

## Decision 122: ORCA2_OMIP_L4

The same enclosing subroutines and identical line numbers execute in the full
ORCA2 build:

| statement | ORCA2_OMIP_L4 compiled source |
|---|---|
| `rCdU_top=rCdU_ice` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:178-190` (`zdf_drg`, :150) |
| SI3 formation | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:403` (`ice_update_tau`, :332) |
| external coefficient | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1428-1433` (`dyn_drg_init`, :1404) |
| top baroclinic residual | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1466-1489` (`dyn_drg_init`) |
| external substep drag | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:633-640` (`dyn_spg_ts`, :136) |
| 3-D top RHS and U/V diagonal | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:158-180`, `:303-314`, `:470-482` (`dyn_zdf`, :67) |

ORCA2 resolves the same drag flags/defaults at
`ORCA2_OMIP_L4/EXP00/namelist_ref:817-824` and the same SI3 coefficient at
`ORCA2_OMIP_L4/EXP00/namelist_ice_ref:136`. This receipt supplies the D122
source pointer; it does not claim a full-ORCA2 trajectory measurement.

## Evidence, tests, and review

Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round33`.
The new trajectory, stage, and omission files stamp legoESM commit
`568ea3fdedd01bb9676e627175058e981a979c29` and packages tree
`d6b0559d91fa03595a66aee6f1412587d6d37bfc`.

| artifact | SHA-256 |
|---|---|
| `g5_forcing_identity.json` | `451e47fb3951243725e1b8dda9eefb7177f3321b893f5c6767d20fa065b24d46` |
| `g5_top_drag_jit.json` | `bd56bfa84fe7e5d534f3ab91d727bac8d0ad694392dcf22d265903134fe124af` |
| `g5_stages_kt1_alllevels_orderfix.json` | `a182d24e31700530e2fc74b13ce1ca056ef216a86b10a87567d8415765a824ba` |
| `g5_independent_orderfix_kt1_10.json` | `c10007b5181798239cc6cf6a58b992fd4ed0ed83a20e559ad8452ed5009ad52a` |
| `g5_given_orderfix_kt1_10.json` | `362cd9cccc0118b43379f7220d14a9a84798a8f29430d27b27ff51953539fb61` |
| `g5_without_residual_kt1_2.json` | `1d574e1589da43a72cfb209999e5e4007378b8de47b081114519997113d37ffc` |
| `g5_without_substeps_kt1_2.json` | `f44d17748502a0318f16ca01e0c6f88d3a6eb8a58f2f4039fbf25e6f267bad35` |
| `g5_without_dyn_zdf_kt1_2.json` | `6cea301498f75b782a6a93fbf12af6c2d151f4e78d751342dabec34014127ef1` |

- Focused G5 + G2 ladder, CPU fp64: 22 passed, 8 slow deselected.
- G2/G3/G4 fast regression, CPU fp64: 56 passed, 12 slow deselected.
- Isolated consumer gate, CPU fp64: 1 passed in 648.41 s; all three omissions red.
- Coefficient plant: exit 1. Citation gate: 17 passed after the 74-anchor
  mechanical reanchor required by this round's insertions.
- Independent Claude review: HOLD on the pre-fix commit; all five findings are
  dispositioned here. Per the resume instruction, no Codex review was run; the
  operator will re-review with Claude.

ASKED: fix the residual/wind order with both NEMO citations; individually
non-vacuous consumers 2–4; compiled signed-zero parity; all-level and
barotropic stage rows with honest stage-2 T/S; both labels kt 1..10; the
pre-fix GIVEN kt-3 U/V debt; corrected `dyn_zdf` subroutine citation; focused
tests and receipt. UNASKED choices: none.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r33/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: HELD — the kt-2 order debt is fixed and given-entry is AT-BAR through kt 10, but independent all-level U first exceeds the bar at kt 9.
