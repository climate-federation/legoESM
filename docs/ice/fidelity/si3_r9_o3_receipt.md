# SI3 round 9 receipt — O3 initial HFN thickness-distribution fit

## Own errors, first

1. No preregistration was committed before measuring (standing rule 3). The
   first measurement was an uncommitted probe; this receipt therefore carries
   no frozen prediction, only the committed gate's result and its plants.
2. That first probe's "jit" captured every input as a closure constant, so
   XLA constant-folded it; its numbers are RETRACTED. The committed gate passes
   every input as a jit argument, and the result below is from that gate.

## Outcome

O3 is bit-exact against the ADMITTED O2 record; no acquisition was needed.
O2 now computes its five-category arrays through the O3 fit, and its real
admission is still 0 ULP everywhere.

## Which NEMO branch the deck executes, and why

`ice_var_itd` is a generic interface over four routines
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:95`:
`MODULE PROCEDURE ice_var_itd_1c1c, ice_var_itd_Nc1c, ice_var_itd_1cMc, ice_var_itd_NcMc`).
`ice_istate` calls it once per row with rank-1 inputs and rank-2 outputs
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:357-362`, first line
`CALL ice_var_itd( zht_i_ini(:,jj)     , zht_s_ini(:,jj)     , zat_i_ini(:,jj)   ,`).
Only `ice_var_itd_1cMc` declares that rank pair
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1203-1204`:
`REAL(wp), DIMENSION(:),   INTENT(in)    ::   phti, phts, pati` and
`REAL(wp), DIMENSION(:,:), INTENT(inout) ::   ph_i, ph_s, pa_i`), so the
compiler resolves the call to `:1182` (`SUBROUTINE ice_var_itd_1cMc`). The
other three arms are not executed on this deck and were not transcribed.
The build compiles with `-fdefault-real-8 -O3 ... -fno-tree-vectorize` and no
`-march` (`ORCA2_SAS_ICE_O2_REF/BLD/arch_nemo.fcm:27`), so `2.` is binary64,
no statement is fused, and `EXP` is the scalar glibc call.

## What was transcribed (one shared ice package, `si3_orca2_state.py`)

- `si3_var_itd_1cmc` (`packages/ice/legoesm/ice/si3_orca2_state.py:2529-2638`):
  the reciprocal `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1221` (`WHERE( phti(:) /= 0._wp )   ;   z1_hti(:) = 1._wp / phti(:)`),
  the gamma-integral areas and volumes of categories 1 to 4 (`:1232`, first
  line `pa_i(ji,jl) = pati(ji) * z1_hti(ji) * (  ( phti(ji) + 2.*hi_max(jl-1) ) * EXP( -2.*hi_max(jl-1)*z1_hti(ji) ) &`)
  and of category 5 (`:1257`), the snow load and snow-to-ice transfer
  (`:1280-1283`, first line `zdh = MAX( 0._wp, ( rhos * ph_s(ji,jl) + ( rhoi - rho0 ) * ph_i(ji,jl) ) * r1_rho0 )`),
  the per-category temperatures and salinity (`:1294`, `pt_su(:,jl) = ptmsu(:)`)
  and the pond fractions (`:1301` and `:1321`). The small numbers are NEMO's
  (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/par_ice.f90:134-135`). `EXP` is the
  precision-policy transcendental (`packages/ice/legoesm/ice/si3_orca2_state.py:2556`);
  every other operation is source-rounded so XLA cannot fuse it.
- `si3_o3_istate_fill_fields` (`packages/ice/legoesm/ice/si3_orca2_state.py:2645-2759`):
  the fit call, the layer-temperature copies, the ice indicator
  (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:376`), the salinity clamp
  (`:383`, `s_i(ji,jj,jl) = MIN( MAX( rn_simin , s_i(ji,jj,jl) ) , rn_sinew * sss_m(ji,jj) )`)
  and profile (`:386`), volumes (`:390-392`), snow and ice energies
  (`:398-399`, `:409-412`), layer salt (`:406`), pond volumes (`:420-421`),
  `t1_ice` (`:426`) and the concentration cap (`:432`). The HFN bounds,
  salinity profile and category sums are O1's existing helpers, reused.
- O2 wiring (`packages/ice/legoesm/ice/si3_orca2_state.py:2927-2934`): O2
  calls the fit; NEMO's six recorded arrays survive only as the test-only
  `_control_exact_fitted` argument.

Searched before building: `grep -rn -i "ice_var_itd|abraham|1cMc|gamma function|distribute"`
over `packages/`, `src/` and the O1/O2 tools. Found only
`itd.distribute_to_categories` and `state.distribute_dynamic_state_to_categories`,
which put all ice into the one category containing the mean thickness. That is
a different method with a different interface, so it was not extended.

## Results (CPU, binary64, production jit with inputs as arguments)

**CONFIRMED:** the committed gate
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:120-188`)
feeds O2's own scalar state into the fit, then into NEMO's
`ice_var_glo2eqv(1)` and `ice_var_agg(1)` (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90:336-337`),
and compares with NEMO's `entry` snapshot written at the end of `ice_init`.

- 55 arrays scored: the 12 extensive per-category fields (including all 10
  ice and 5 snow layers), all equivalents, all aggregates, the eight O2 sums
  and masses, and `t1_ice` on its defined interior.
- Wet maximum 0 ULP, full-array maximum 0 ULP, every array.
- 16,849 wet cells: 16,433 interior, 281 cyclic seam, 135 north fold; 0 ULP
  in each subset.
- jit equals eager to 0 ULP.
- Score file `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round9/o3_score.json`.

**CONFIRMED, plants:** a one-ULP change in `v_i` fails (exit 1); XLA's native
`exp` fails (2 ULP, first at zero-based `(31, 149, 3)`, a fold cell).
Reassociating one category-5 product, or the layer-salt product, turns the
oracle test and the record test red; both reverts were checked clean.

**CONFIRMED, two branches never run on this deck:** removing the snow-to-ice
transfer or the concentration cap leaves the record score at 0 ULP. With
2 m / 0.2 m (north) and 1 m / 0.2 m (south) the snow load never exceeds
buoyancy, and total concentration is at most 0.9, below the 0.997 cap. Both
branches are pinned instead by synthetic inputs that do run them
(`tests/ice/unit/test_si3_o3_itd_fit.py:104-150`), including an independent
pure-Python transcription that matches bit for bit
(`tests/ice/unit/test_si3_o3_itd_fit.py:93-101`).

**CONFIRMED, NEMO behaviour reproduced:** when the transferred ice is clipped
at the category's upper bound, NEMO still removes the full snow amount
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1280-1283`), so snow plus
ice mass is not conserved in that case. legoESM reproduces this literally.

**Measured alternative arms (not certified):** native `exp` on CPU gives at
most 4 ULP (5.8e-16 normalised). GPU spot-check (idle GPU 1, `nvidia-smi`
before and after recorded, native `exp` because the glibc arm is CPU-only)
gives at most 14 ULP and 1.59e-15 normalised on the layer energies, above the
1e-15 bar. **PLAUSIBLE:** the GPU excess comes from the GPU `exp`; not
localised.

## Controls for every other caller

- O2 real admission rescored with the fit inside: PASS, every comparison
  0 ULP (`o2_rescore_with_o3_fit.json`). O2 suite 12/12, O1 suite 15/15,
  O3 suite 8/8.
- Parent `62a28e1324b2` versus current: D5 C1D year under CPU jit, all 4,613
  non-provenance values identical (shipped-arm hash
  `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`); the
  non-SI3, column-model, lane-3 and O1 callers bit-identical (O2's harness,
  its own plant fires); all 31 lane-3 test files give the same outcome on both
  trees.
- Gradients: finite; central difference matches to better than 1e-5. The
  `WHERE`/`IF`/`MIN`/`MAX` switches are not differentiable at their
  thresholds; forward arithmetic is NEMO-literal and divisions carry
  gradient-only safe denominators.
- Ratchets: the same 11 pre-existing failures on parent and current; the
  private-import ratchet passes 6/6.

## CHOICES

- ASKED: transcribe the fit, measure on the admitted record, wire O2 on 0 ULP.
- UNASKED: certify under the glibc `exp` arm (`transcendentals="libm"`), the
  campaign's existing NEMO-certification arm. Offered for revert.
- UNASKED: the O2 gate and tests now select that arm, because O2 now runs the
  fit. Offered for revert.
- UNASKED: the fit lives next to O1/O2 rather than in `itd.py`. Offered for
  revert.
- UNASKED: one O2-receipt sentence was updated for the moved code, and the
  citation map was re-anchored. Offered for revert.

Review: codex review: operator. Independent review: none run in this round.
