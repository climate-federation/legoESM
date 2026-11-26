# SI3 Ladder Rung O6 Preregistration — Round 14 (`si3_r14`): five-category ridging/rafting and `ice_cor(kt,1)`

- Date: 2026-10-25. Lane `si3`, round 14, rung O6 (`docs/ice/fidelity/si3_ladder_design.md:226`).
- Parent commit: `e8ee89ece8cb`. Model functions preregistered: `si3_ice_dyn_rdgrft` (ridging module) and `si3_ice_cor_dyn` (ITD module).
- Status: committed BEFORE any O6 measurement. No NEMO array has been read for O6. The acquisition tool is NOT built this round.
- Amended 2026-10-25, still before any O6 measurement: the deck is the unmodified production deck (DECISION 174, section 1).
- Reference build cited: `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/` (the production-deck record build of DECISION 165). Its `icedyn_rdgrft.f90`, `icecor.f90`, `iceitd.f90`, `icevar.f90` and `icedyn.f90` are byte-identical to `_O1_REF`, `_O1_REC`, `_O2_REF`, `_O2_REC`, `_O5_REF` and `_O7_REF`. `_O5_REC` differs (its observers), so it is not cited.

## 1. Deck — DECISION 174: the unmodified production deck

The unmodified production `ORCA2_SAS_ICE` deck, thermodynamics on (the design
row as amended by DECISION 174). With `ln_icethd = T`, `ice_dyn_rdgrft_init`
keeps the deck's porosity `.3`, snow survival `.5` and pond survival `1`
(its override at `icedyn_rdgrft.f90:1273-1276` runs only when
`ln_icethd = F`), so the seawater-trapping, snow-loss and pond-loss
statements and their ocean fluxes run with live operands. The `ln_icethd = F`
overlay is not acquired.

Every value is read from the resolved `output.namelist.ice`/`ocean.output`,
never the deck text. One MPI rank.

## 2. Steps and ranks

`nn_it000 = 1`, `nn_itend = 2` (two ice steps, as DECISION 165); `mpirun -np 1`.
One rank matters here: `iterate_ridging` is one flag over the rank's packed
cell list, so the number of ridging passes depends on the decomposition.

## 3. Arrays and call points (GIVEN-NEMO-ENTRY only, binary64, NEMO's declared extents)

R1, `ice_dyn_rdgrft` entry (immediately before `icedyn.f90:193`), full `(jpi,jpj[,k,jl])`:
`a_i v_i v_s sv_i oa_i a_ip v_ip v_il` `(jpi,jpj,jpl)`; `e_s (jpi,jpj,nlay_s,jpl)`,
`e_i szv_i (jpi,jpj,nlay_i,jpl)`; `at_i ato_i divu_i delta_i sst_m sss_m`;
`wfx_dyn sfx_dyn sfx_bri hfx_dyn wfx_snw_dyn wfx_pnd`; `hi_max(0:jpl)`; `tmask(:,:,1)`;
scalars `rDt_ice r1_Dt_ice r1_nlay_i nn_icesal jpl nlay_i nlay_s` and the resolved
`namdyn_rdgrft` values as printed by `ice_dyn_rdgrft_init` after its `ln_icethd` override.

R2, packed intermediates (1-D, length `npti`, with `nptidx(1:npti)`):
after `:301` `closing_net opning zdivu`; after `:309` `apartf(:,0:jpl) closing_gross opning`;
after `:323` `ipti iptidx`; per pass at passes {1, 2, last}: after `:348`
`apartf aridge araft hi_hrdg hraft hrmin hrmax closing_gross opning`, after `:352`
`a_i_2d v_i_2d ato_i_1d`, after `:382` `closing_net opning ato_i_1d zdivu`; and the
pass count `iter - 1` at `:388`.

R3, `ice_dyn_rdgrft` exit (between `icedyn.f90:193` and `:194`): every R1 array.

C1, `ice_cor(kt,1)` entry: R3 plus `sz_i t_su t_i t_s h_i h_s h_ip h_il vt_i rn_amax_2d hi_mean sfx_res hfx_res wfx_res`.
C2, after `icecor.f90:99` (`ice_itd_reb`): `a_i v_i v_s oa_i a_ip v_ip v_il e_s e_i szv_i h_i t_su`.
C3, `ice_cor` exit (after `icecor.f90:130`): every C1 array.

## 4. Frozen predictions (model at this round's commit, `transcendentals="libm"`, CPU fp64, jit)

- P1: R3 replays from R1 through `si3_ice_dyn_rdgrft` exactly (0 ULP, every cell, halos included).
- P2: the model's selected cells equal NEMO's `iptidx(1:ipti)`; its pass count equals NEMO's `iter - 1`.
- P3: each R2 packed array replays exactly at its pass (scored through a test-only trace of the same function, added with the acquisition).
- P4: C3 replays from C1 through `si3_ice_cor_dyn` exactly; C2 replays from C1 exactly.
- P5: NEMO's own budget (`icectl.f90:176-178` with `e1e2t`) closes on R1->R3 and C1->C3 to within NEMO's `rn_icechk_glo` threshold, and the model's closes identically (same bits, so same residual).
- P6: halos of every R3/C3 array equal R1/C1 (NEMO writes interior only).
- Falsifier for each: any nonzero ULP. The first non-bit statement is named with its `ppsrc` line before any model change lands.

## 5. Controls the score must carry

One-bit and swapped-field payload plants on the reader; a wrong-pass plant
(score R3 against the model stopped one pass early) must fire; a halo plant;
a ridging-constant plant (`hrdg_hi_min`) and a rebinning plant (`0.5` share)
in the model must turn P1/P4 red.

## 6. Not exercised (stated, not claimed)

- Multi-rank decompositions (`iterate_ridging` is per rank); exponential
  redistribution (`ln_distf_exp`) is unit-tested only; linear participation
  (`ln_partf_lin`) and `nn_icesal /= 4` are refused, not transcribed.
- MIN/MAX operand order on a signed-zero tie was read from the object code at
  `icedyn_rdgrft.f90:295` and `:713` only; elsewhere a tie needs a `-0`
  operand that the fixtures never produce (PLAUSIBLE until a record shows one).

## 7. Choices

| choice | status |
|---|---|
| deck (section 1) | ASKED: DECISION 174, unmodified production deck |
| `nn_itend = 2`, one rank | UNASKED, following DECISION 165; offered for revert |
| R2 passes {1, 2, last} | UNASKED; offered for revert |
| cited build `ORCA2_SAS_ICE_O7_REC` | UNASKED, mechanical (byte-identical alternatives) |
