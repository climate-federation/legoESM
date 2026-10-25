# SI3 round 7 receipt — O2 ready to land

## Outcome

Round `si3_r7` is ready to land. The operator acquired and admitted the O2
record at producer commit `6644b2ce1c6449b9ec7134c32d8bfc7e2b0b58f9`.
The retained record was then replayed on CPU in binary64 through production
`jax.jit`; no NEMO build or run was repeated during scoring.

The landed O1 contract is preserved. The O2 implementation is based on lane
tip `042c7bd74c2944dab8c39318687ab0cef54f9d2b`. The unrelated ORCA2
round-329 cherry-pick `11db3ee221` was rebased out; it is not an ancestor of
this branch, and O2 does not import or call `orca2_initial_ice_ssh`.

## Results

**CONFIRMED:** O2 is bit-exact against the admitted NEMO record. The compiled
CPU replay (`execution_backend=cpu`, `execution_x64=true`,
`execution_jit=true`) scored all `24` O2 arrays plus `area` and `zsshadj`, all
`107` state/moment/equivalent/aggregate arrays at continuous `kt=1` and
`kt=2`, restart entry, restart advance to `kt=2`, and restart `kt=2` versus
continuous `kt=2`, and all `40` registered NetCDF restart variables.

- O2 initial state: wet maximum `0` ULP, full-array maximum `0` ULP, maximum
  normalised error `0.0`; `area=357005237105466.1` and
  `zsshadj=0.09976769334054654` are each `0` ULP.
- Continuous `kt=1` and `kt=2`: wet and full-array maxima are `0` ULP for
  both steps; maximum normalised errors are `0.0`.
- Restart entry, restart advance, and restart-versus-continuous `kt=2`: wet
  and full-array maxima are `0` ULP; all `40` restart variables are `0` ULP.
- The wet T-cell census is `16,433` interior, `281` cyclic-seam, and `135`
  north-fold cells. Because every scored wet array is bit-identical, both
  seam and fold subsets are `0` ULP. The first interior wet T cell is
  zero-based `(2, 21)`.

There is no first non-bit array or location and therefore no owning NEMO
statement to propose. **PLAUSIBLE:** none; the verdict uses no unmeasured
explanation. The permanent score is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round7/o2_jit_score.json`
(SHA-256 `3748bf8619742796f9c7d7c66c892f946b71b22343ada95b5adf03f53d9ff2f5`).
Its `gate_changed_since_run=true` records only the post-acquisition change
from eager replay to compiled replay plus added result fields; producer
commit, admitted binaries, inputs, record bytes, masks, scored arrays, and
bars are unchanged.

## Review repairs

All eight Claude findings were reproduced and repaired:

1. The TEOS-10 freezing point now calls the shared `nemo_eos_fzp` and uses
   NEMO's literal `z1_S0 = 1._wp / 35.16504_wp`
   (`src/OCE/TRA/eosbn2.F90:1674-1683`). The wrong
   `0.875 / 30.76941` expression and its hard-coded test oracle are gone.
2. O2 no longer invents post-fit category totals from scalar thickness times
   concentration. It takes NEMO's six fitted per-category arrays as exact
   inputs from the O1 entry record, then transcribes only the category sums,
   snow/ice/pond/lid mass, and SSH reduction
   (`src/ICE/iceistate.F90:388-427`). O3 still owns the category fit.
3. O2's levitating-ice SSH calculation is implemented only in the selectable
   SI3 production path. The ORCA2 lane's round-329 helper has no O2 caller and
   is not carried on this branch.
4. The ponds-off and lids-off branches are implemented literally
   (`src/ICE/iceistate.F90:295-304`) and directly tested with nonzero pond/lid
   inputs.
5. Continuous step 2 again consumes the recorded step-2 `sss_m`; the landed
   O1 `kt2-sss-perturb` plant proves that dependency.
6. The committed log excerpt is now selected from O1's admitted
   `ORCA2_SAS_ICE` SAS run with `ln_icedyn = F` and `ln_icethd = F`. Preflight
   validates it directly and never rewrites its flags.
7. O2 imports O1's halo-free restart alignment helper and NEMO restart-layout
   fixture, and restores the `restart-disabled`, `rst-step-name`, and
   `ocean-restart-missing` plants.
8. The O2 full-array ULP is independently computed rather than copied from the
   wet ULP. The final preflight JSON records the exact producer commit and is
   regenerated only after the final commit.

## Model and bar

`si3_o2_scalar_initial_state` requires explicit masks, both pre-branch SSH
levels, and all six fitted category arrays. It refuses the wrong initialization
mode, EOS, embedded-ice mode, restart mode, and category shape. It uses the
shared DDPDD reducer from `packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py`
for the two `glob_2Dsum` calls (`src/OCE/LBC/lib_mpp.F90:1244-1267`).

The preregistered bar is unchanged: all 24 scored O2 fields and both scalar
reductions must be at 0 ULP. The difference is ownership: fitted categories
are inputs, while only O2-owned sums, mass, and SSH are scored as O2
transcriptions. The trajectory and restart fields retain O1's 0-ULP bar. No
bar or scoring mask changed for the result above.

O2 remains fail-closed in `step_sea_ice`: it exposes initialization through
`si3_o2_step` but refuses to invent the still-unowned `ice_update_flx`
heat/freshwater/salt response.

## Acquisition contract

The operator-only tool is
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh`.
Its default is `--preflight-only`; `--run` is operator-only. It checks the two
input-bundle hashes, available memory and disk, reference-config registration,
rendered namelists, patched Fortran syntax, the real four-rank writer layout,
O1's NEMO restart layout, the admitted SAS log excerpt, same-program binaries,
producer stamps, both restart dependencies, and every negative-control plant.

The restart run requires and copies both the SI3 and SAS ocean restart into a
separate input directory. Outputs use distinct names and directories. Failed
or admission-refused evidence is retained under the round-7 evidence root.

The read-only preflight wrote
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round7/o2_preflight_receipt.json`.
The admitted record's `producer_commit` is the binding provenance stamp; the
post-acquisition score is separate and does not alter producer evidence.

## Verification and choices

Focused O2 tests cover the literal freezing point, pond/lid switches, exact
category inputs, DDPDD reductions, compiled writer, real log provenance,
same-program checks, JIT/JVP/VJP, O1 restart layout, step-2 salinity selection,
and all admission plants. The O1 regression suite and the imported ORCA2
round-329 tests also pass. Default receipt-citation and private-import gates,
plus an explicit citation-gate invocation for every SI3 receipt, gate handoff.
Post-score verification passes: focused O2 `12/12`; private-import ratchet
`6/6`; default citation gate and all six SI3 per-receipt invocations each
report zero failures, zero stale full-map entries, and zero unmapped citations.

`ASKED`: repair all review findings, preserve the O2 bar with exact
fitted-category inputs, admit the operator-run record, and score it under CPU
production JIT.
`UNASKED`: none.

`CHOICES — ASKED (DECISION 161, 2026-10-24, user: "Your pick")`: the one
shared `nemo_eos_fzp` uses NEMO's literal statement for every caller. The
pre-existing ice-shelf and OMIP-driver callers do NOT keep a separate parent
rounding; measured above, their returned outputs are unchanged (0/4,001 and
0/808 + 0/101 values) and only intermediate freezing points move by 1-2 ULP.

Independent review disposition: Claude review was HOLD/HOLD; every listed
finding above is confirmed and repaired. Argon's three distinct findings are
addressed below.

## Argon independent-review follow-up

1. The citation gate again audits every `CITATION_MAP` entry on every
   invocation, exactly as the lane-tip gate does. All stale entries were
   re-anchored against their current files; the default gate and every SI3
   per-receipt invocation therefore exercise the same full-map audit.
2. The `hemisphere-swap` and `stop-gradient` controls now select the private
   plant arms in `si3_o2_scalar_initial_state`/`si3_o2_step`. They no longer
   manufacture a bad candidate by swapping the input config or stopping the
   gate's arguments before the model call. The focused test invokes both plant
   arms and requires `STATUS PLANT-FIRED` from each.
3. The six NEMO per-category arrays are **inputs to O2**, not exclusions from
   its bar. NEMO first fits the scalar state into all `jpl` categories in
   `src/ICE/iceistate.F90:309-348`; the fit itself includes category-dependent
   gamma integrals, caps, and snow-to-ice transfer in
   `src/ICE/icevar.F90:1211-1272`. NEMO then sums the fitted arrays in
   `src/ICE/iceistate.F90:388-400` and
   `src/ICE/icevar.F90:122-130`. The O2 operator correspondingly requires
   `fitted_v_i`, `fitted_v_s`, `fitted_a_i`, `fitted_a_ip`, `fitted_v_ip`, and
   `fitted_v_il`, checks their `(jpi, jpj, 5)` shape, and sums them in
   `packages/ice/legoesm/ice/si3_orca2_state.py:4542-4723` (since O3, round
   `si3_r9`, the arrays come from the O3 fit and NEMO's recorded arrays are
   the test-only `_control_exact_fitted` control). Thus O3 owns the
   nonlinear fit, while O2's 0-ULP bar still covers `vt_i`, `vt_s`, `at_i`,
   `at_ip`, `vt_ip`, `vt_il`, `snwice_mass`, and `snwice_mass_b` computed from
   those exact five-category inputs.

## Claude re-review follow-up

1. `tbo-halo-leak`, `pond-unscaled`, `ssh-sign-or-div`, and `dom-uniq-fold`
   now call `si3_o2_scalar_initial_state` with, respectively,
   `_plant_tbo_halo_leak`, `_plant_pond_unscaled`, `_plant_ssh_sign`, and
   `_plant_omit_nfd_uniq`. The focused test records selection of each private
   model switch and requires the resulting production-code-path candidate to
   fail with `STATUS PLANT-FIRED`; no gate-side output is forged.
2. Commit `11db3ee221` was removed with `git rebase --onto dc0da32e48
   11db3ee221`. Its ORCA2 preregistration, D122 gate, tests, and 141-line
   ladder-gate addition are absent; `git merge-base --is-ancestor 11db3ee221
   HEAD` returns `1`.
3. Parent/current caller measurement used parent `dc0da32e48` and CPU fp64.
   The OMIP-driver input was 4,001 surface salinities from 0 through 40 PSU at
   0.01-PSU spacing. Its Celsius `nemo_eos_fzp` boundary changed at 29/4,001
   values (maximum 2 ULP; maximum absolute change
   `4.440892098500626e-16` °C), while the real
   `_ice_freezing_temperature_K` return changed at 0/4,001 values (0 ULP).
   The ice-shelf input was 101 eight-level columns: base salinity 30 through
   35 PSU plus a 0 through 0.8 PSU vertical gradient, 200-m levels,
   `zmin=150 m`, `zmax=750 m`, and `fwf=5e-4 kg m-2 s-1`. Its internal
   depth-adjusted freezing point changed at 9/808 values (maximum 1 ULP;
   maximum absolute change `4.440892098500626e-16` °C), while the real
   `isf_prescribed_melt_tendencies` outputs changed at 0/808 `dT_dt`, 0/808
   `dS_dt`, and 0/101 `eta_dot` values (all 0 ULP). Across the combined 4,809
   salinity values, the literal multiply-by-reciprocal square-root operand
   differed from the parent division at 757 values. The shared change remains
   selected pending the user decision above; no caller-specific rounding
   policy was implemented.
