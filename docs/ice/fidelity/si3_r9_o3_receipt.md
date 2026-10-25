# SI3 round 9 receipt — O3 initial HFN thickness-distribution fit

## Own errors, first

1. No preregistration was committed before measuring (standing rule 3). The
   first measurement was an uncommitted probe; this receipt therefore carries
   no frozen prediction, only the committed gate's result and its plants.
2. That first probe's "jit" captured every input as a closure constant, so
   XLA constant-folded it; its numbers are RETRACTED. The committed gate passes
   every input as a jit argument, and the result below is from that gate.
3. The first version of this receipt said "O2 now computes its five-category
   arrays through the O3 fit, and its real admission is still 0 ULP
   everywhere". Measured since (section "Which O2-admission comparisons read
   the fit"): only O2's initial-state comparison reads the fit; its four step
   comparisons, the restart comparisons and the restart-file check start from
   NEMO's recorded state and do not. The sentence over-claimed and is replaced.
4. The first dependence probe of the resumed round nudged one category by one
   ULP in a halo column; the five-category sum and the halo-excluding global
   sum absorbed it, so it showed no dependence anywhere. Retracted; the
   committed probe below perturbs one interior column by a relative 2**-10.

## Outcome

O3 is bit-exact against the ADMITTED O2 record; no acquisition was needed.
O2's initial-state comparison (its category sums, snow-ice mass and SSH
adjustment) now takes its candidate from our fit and stays 0 ULP. O2's step and
restart comparisons start from NEMO's recorded entry state, as the ladder's
exact-input rule intends. Stepping kt 1 and kt 2 from the entry state O3 builds
is also 0 ULP, scored separately under its own label.

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
- O2 wiring (`packages/ice/legoesm/ice/si3_orca2_state.py:2927-2934`): O2's
  initial state calls the fit; NEMO's six recorded arrays survive only as the
  test-only `_control_exact_fitted` argument. O2's step does not consume the
  fit: it advances the state it is given
  (`packages/ice/legoesm/ice/si3_orca2_state.py:3039-3046`, first line
  `new_state = si3_o1_step(`).

Searched before building: `grep -rn -i "ice_var_itd|abraham|1cMc|gamma function|distribute"`
over `packages/`, `src/` and the O1/O2 tools. Found only
`itd.distribute_to_categories` and `state.distribute_dynamic_state_to_categories`,
which put all ice into the one category containing the mean thickness. That is
a different method with a different interface, so it was not extended.

## Results (CPU, binary64, production jit with inputs as arguments)

**CONFIRMED:** the committed gate
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:153-221`)
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
- Score file `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round9/o3_score.json`
  (SHA-256 `6db4dfbd5910fa8688a5cf69c65d6d15498807effff23854a9ccbd7167323b2c`,
  scored at commit `6ccf54900116`).

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

## Which O2-admission comparisons read the fit (measured)

**CONFIRMED.** The committed probe
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:225-277`)
runs the unmodified O2 admission twice with every comparator replaced by a
recorder, the second time with the fit's six O2-feeding outputs multiplied by
1 + 2**-10 in the first interior ice column
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:74-91`).
A comparison reads the fit iff a bit of its candidate moves. Output
`si3_rounds/round9/r9b/o2_fit_dependence.json`; pinned by
`tests/ice/unit/test_si3_o3_itd_fit.py:191-202`.

| O2-admission comparison | candidate starts from | reads the fit | moved |
|---|---|---|---|
| initial state, 26 values | our scalar state and fit | yes | 10: the six category sums but lid volume, both masses, the SSH adjustment, both SSH levels |
| kt 1 state, 111 arrays | NEMO's recorded entry | no | 0 |
| kt 2 state, 111 arrays | our kt 1 | no | 0 |
| restart entry, 111 arrays | NEMO's restart file | no | 0 |
| restart step, 111 arrays | NEMO's restart file | no | 0 |
| restart step vs kt 2, 111 arrays | NEMO's restart file | no | 0 |
| restart-file variables, 40 | our kt 1 | no | 0 |

Lid volume cannot move: NEMO's entry lid volume is identically zero on this
deck. The kt 1 candidate is computed from the recorded entry
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1386`,
`state_kt1, _ = o2_step_jit(state_entry, sss_entry)`). That is the ladder's
exact-input rule, not a gap
(`docs/ice/fidelity/si3_ladder_design.md:195-197`, first line
`The O rungs separate **operator admission** from **composition**. O1-O16 replay`:
"replay the exact NEMO input at each named source boundary ... only O17
advances the complete live deck"; O3's row,
`docs/ice/fidelity/si3_ladder_design.md:223`: "O2 scalar state as exact input"). Those numbers carry the
GIVEN-NEMO-ENTRY label and are not mixed with the next section's.

## Steps from our own fitted entry (label O3-FIT-ENTRY)

**CONFIRMED.** What O3 owns is the initial category state the first step
consumes. The committed score
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:280-344`)
builds the entry state from the fit (every prognostic category field,
equivalents, aggregates, `t1_ice`, both masses), keeps NEMO's entry only for
fields ice_istate's fit does not produce (`u_ice`, `v_ice`, `cnd_ice`, `t_si`,
`snwice_fmass`, Prather moments, category bounds), steps kt 1 and kt 2 under
CPU jit with every array an argument, and scores against NEMO's step records.

- kt 1 and kt 2: wet and full maximum 0 ULP; entry masses 0 ULP. Output
  `si3_rounds/round9/r9b/o3_entry_steps.json`.
- Fit output perturbed as above: refuses at kt 1 (`v_i`). So the step reads
  the fit.
- NEMO re-sets the snow-ice mass after `ice_istate`, from the aggregates, in
  `ice_update_init`
  (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceupdate.f90:482-483`, first line
  `snwice_mass  (:,:) = tmask(:,:,1) * ( rhos * vt_s(:,:) + rhoi * vt_i(:,:) + rhow * (vt_ip(:,:) + vt_il(:,:)) )`).
  It differs from the `ice_istate` value by 1 ULP in 3,704 cells; using the
  `ice_istate` value made kt 1 refuse by 1 ULP. The O3-built entry reuses the
  package's existing transcription of that statement
  (`packages/ice/legoesm/ice/si3_orca2_state.py:1219-1245`). No model code changed.
- Pinned by `tests/ice/unit/test_si3_o3_itd_fit.py:205-212`.

## Controls for every other caller

- O2 real admission rescored with the fit inside: PASS, every comparison
  0 ULP; the output file is byte-identical to round 7's admitted O2 score
  (both SHA-256 `3748bf8619742796f9c7d7c66c892f946b71b22343ada95b5adf03f53d9ff2f5`). O2 suite 12/12, O1 suite 15/15,
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
- Ratchets: the CLAUDE.md CI ratchet suite (`tests/test_no_hardcoded_constants.py`,
  `tests/test_dispatch_hardening.py`, `tests/test_validate_strict_coverage.py`,
  `tests/test_physics_contracts.py`, `tests/unit/test_params_reachability_audit.py`,
  `tests/unit/test_param_bounds_agree.py`, `tests/test_no_private_cross_imports.py`,
  `tests/test_no_inline_physics_coeffs.py`, `tests/test_param_specs.py`) gives
  11 failed on a clean checkout of the lane tip `62a28e1324` and on this
  branch, and the 38 failure-detail lines are identical, so none is caused by
  this branch (`si3_rounds/round9/r9b/ratchets_{lane_tip,current}.txt`). The 11:
  `test_no_hardcoded_physical_constants[tests/unit/test_jra55_do.py]`;
  `test_no_dispatch_guard_removed`;
  `test_no_inline_physics_coefficients[...]` for `ice/bitz_lipscomb.py`,
  `ice/si3_orca2_state.py`, `land/restart.py`, `ocean/physics/geothermal.py`,
  `ocean/physics/lateral_mixing/mle.py`,
  `ocean/physics/lateral_mixing/mle_latlon_cgrid.py`;
  `test_config_module_has_valid_param_spec[...]` for `ice/si3_orca2_state.py`,
  `ocean/physics/lateral_mixing/mle.py`,
  `ocean/physics/shortwave_penetration.py`. The private-import ratchet passes.

## Precision arm

No earlier SI3 O rung selected an arm (O1 and O2 used `PrecisionPolicy.fp64()`
at the lane tip), whereas the ORCA2 ocean ladder gates
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_eos_gate.py:108`,
`policy = PrecisionPolicy.fp64(transcendentals="libm")`) and lane 3's ADV2D/RHG
oracle recipe
(`packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py:152`)
already certify under the glibc arm. The choice stays with the user.

## CHOICES

- ASKED: transcribe the fit, measure on the admitted record, wire O2 on 0 ULP.
- UNASKED: certify under the glibc `exp` arm (`transcendentals="libm"`), the
  campaign's existing NEMO-certification arm. Offered for revert.
- UNASKED: the O2 gate and tests now select that arm, because O2 now runs the
  fit. Offered for revert.
- UNASKED: the fit lives next to O1/O2 rather than in `itd.py`. Offered for
  revert.
- ASKED (resume item 1): the fit-dependence probe and the O3-FIT-ENTRY
  step score, including which entry fields stay NEMO's.
- UNASKED: perturbation size 2**-10 in the first interior ice column. Offered
  for revert.
- UNASKED: one O2-receipt sentence was updated for the moved code, and the
  citation map was re-anchored. Offered for revert.

Review: codex review: operator (first pass HOLD on the O2-wiring claim, measured above); Argon LAND. Independent review of this resume: operator.
