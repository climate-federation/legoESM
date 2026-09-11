# NEMO testcase L2 GYRE round 56 — derived TKE floor receipt

Date: 2026-09-11. Parent `03098e6fe91c`; implementation/acquisition commit
`21e85d25284d`; CPU/fp64. **ACQUISITION COMPLETE; TRAJECTORY UNMEASURED.**
The clean producer stamp exists and the operand record was acquired at that
commit. No round-56 trajectory was run.

**ROUND-58 RETRACTION:** the acquisition is not admissible for operand
substitution. Its assumed-shape entry dummies rebased reduced NEMO arrays;
the new source-exact Prandtl calibration rejects the record with 635 unequal
wet cells. The executable retraction and full source evidence are in
`nemo_testcases_l2_gyre_round58_tke_operand_bounds_receipt.md`.

## Independent-review findings

| finding | verdict and disposition |
|---|---|
| Writer imports `rn2/rn2b` from the wrong module | **CONFIRMED.** They are declared by `src/OCE/oce.F90:31`; the writer now imports them from `oce`, while `en/avmb/avtb/avtb_2d` remain from `zdf_oce`. |
| Full-domain shape guard/record | **CONFIRMED.** R46 `MY_SRC/zdftke.F90:218,233` (compiled `BLD/ppsrc/nemo/zdftke.f90:223,238`) declares `p_pdlr` as `T2D(0),jpk` and the row work arrays as `T1Di(0),jpk`; MY_SRC `:571` (compiled `:576`) does the same for both mixing lengths. `BLD/inc/do_loop_substitute.h90:72-73,87-89` resolves these to the inner tile. The writer records explicit `ntsi:ntei,ntsj:ntej` arrays, stamps normalized `32,22,31,30`, and the new shape plant corrupts an extent. |
| `p_pdlr(:,jj,:)` rebases the local row | **CONFIRMED.** The explicit `T2D(0)` dummy now preserves the compiled lower bounds and copies `p_pdlr(ntsi:ntei,jj,:)`; every reduced T1Di/T2D operand is treated likewise. |
| Literal floor | **CONFIRMED.** Compiled R46 `BLD/ppsrc/nemo/zdftke.f90:815-817` evaluates `1.e-6_wp/(rn_ediff*SQRT(rn_emin))`; `:829-832` overwrites `rn_mxl0`. The shared JAX path now evaluates that association in fp64 from `TKEConfig.c_k` and `TKEConfig.tke_background`. |
| Wrong citation | **CONFIRMED.** The overwrite is shipped `src/OCE/ZDF/zdftke.F90:859-862` (compiled `:829-832`), now cited that way. |
| Zero-step floor mismatch | **CONFIRMED.** The probe calls the same derived-floor helper; a structural test refuses the deleted field. |
| Raw fallback/dead field/missing mask | **CONFIRMED.** The compiled false arm uses raw `rn_mxl0` at `ppsrc:614-615`, so the dishonest fallback and zero-consumer `mxl0_min_m` were removed. The executing true arm multiplies by `tmask` at `:606-611`; `surface_tmask` is now required and tested. |

The derived floor owns the initialization/interior floor/sweeps at compiled
`:593-595,610-621,650-674`; the resulting lengths feed `avm/avt/dissl` at
`:682-693`. Direct fp64 checks give
`9.99999999999999847e-03` (bits `4576918229304087674`), one ULP below literal
`0.01`, for GYRE and both DINO NEMO cards.

**RETRACTION (round-57 review):** this was not one ULP on every executing card.
The legacy `nemo_v1` card's `_nemo_tke_config` carried `mxl_min=1e-8`, so its
resolved floor moved to `9.99999999999999847e-03`, about `1e6` times larger.
Its 24-test recipe gate checks construction and stepping, not compiled-oracle
TKE bits; that card is UNMEASURED-with-spec debt pending an
`en/avm/avt/dissl` score using its own compiled operands.

## Syntax-only compilation

Scratch files were copied from R46 `MY_SRC/zdftke.F90` and shipped
`src/OCE/ZDF/zdfphy.F90`, then dry-patched. Exact commands (each exit 0; stdout
and stderr empty):

```text
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/WORK -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90 -o /tmp/gyre-r56-syntax.tydsyu/l2_r54_tke.f90
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/WORK -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc /tmp/gyre-r56-syntax.tydsyu/zdftke.F90 -o /tmp/gyre-r56-syntax.tydsyu/zdftke.f90
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/WORK -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc /tmp/gyre-r56-syntax.tydsyu/zdfphy.F90 -o /tmp/gyre-r56-syntax.tydsyu/zdfphy.f90
/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran -fsyntax-only -ffree-line-length-none -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc -J /tmp/gyre-r56-syntax.tydsyu /tmp/gyre-r56-syntax.tydsyu/l2_r54_tke.f90
/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran -fsyntax-only -ffree-line-length-none -I /tmp/gyre-r56-syntax.tydsyu -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc -J /tmp/gyre-r56-syntax.tydsyu /tmp/gyre-r56-syntax.tydsyu/zdftke.f90
/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran -fsyntax-only -ffree-line-length-none -I /tmp/gyre-r56-syntax.tydsyu -I /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc -J /tmp/gyre-r56-syntax.tydsyu /tmp/gyre-r56-syntax.tydsyu/zdfphy.f90
```

## Measurements and Rule 12

| GYRE metric | before | after |
|---|---:|---:|
| kt 1,2,4,5,6,7,8,9,10 ladder rows | BLOCKED—no clean commit stamp | BLOCKED—no clean commit stamp |
| kt3 max T | historical `8.7413164119e-3 K`; same-stamp arm BLOCKED | preregistered `1.6275031290e-4 K`; UNMEASURED |
| kt3 max S | historical `1.3568885211e-3 g/kg`; same-stamp arm BLOCKED | preregistered `6.3278533133e-6 g/kg`; UNMEASURED |
| daily gaps, days 1–30 | all BLOCKED—no clean commit stamp | all BLOCKED—no clean commit stamp |

The prediction is REFUTED by any larger after kt3 maximum; equality/worsening
of day-30 T RMS refutes the direction prediction. DINO's resolved operand moves
down one ULP on both NEMO cards; no trajectory claim is made. LOCK_EXCHANGE
`EXP00/namelist_cfg:131` and OVERFLOW `EXP00/namelist_cfg:129` resolve
`ln_zdfcst=.true.` and do not execute TKE. ORCA2 remains
UNMEASURED-with-spec: bit-score closure outputs from its own compiled operands.

The clean producer stamp is
`phase3/round56/oracle_tke_operands/producer_commit.txt` and equals
`21e85d25284de001c575e23f351a45c426ba5e15`. Acquisition validation PASS;
twin admission PASS (`exact=43/63`, `changed=20`, `admitted=132`). All seven
plant processes exited nonzero, but the shape plant was initially caught by
the older fixed-GYRE extent guard rather than the new stamped-array-shape
guard; round 57 corrects that plant.

Test-status correction: the branch has one red focused test,
`tests/ocean/unit/test_rk3_ws_and_mxl3.py::test_rk3_ws_differs_from_rk3_and_is_finite`.
It is pre-existing and also fails at `3a8d94ea1985`; it is not introduced by
round 56. The earlier `133 passed`, `109 passed`, `3 passed` summary omitted
that known red and is withdrawn.

ASKED: all changes above. UNASKED: none; no freshwater pair, #1484 guard,
ladder card resolution, NEMO source/data, or configuration choice was changed.
Open: operator commit, acquisition, then calibration/substitution and the two
trajectory arms.
