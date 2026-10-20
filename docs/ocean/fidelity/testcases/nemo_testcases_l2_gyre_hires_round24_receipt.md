# NEMO testcase L2 GYRE higher-resolution round 24 closing receipt

Date: 2026-10-20. Branch: `fidelity/gyre-high-resolution`.

## Decision

**AUDIT PASSED; merged digest accepted. CLOSING CHECK PREREGISTERED; GPU result
pending.** The user selected "Audit, then accept." The exact resolved
`GYRE-zco` diff from `7e78361fb1` to the merged tree has 33 additions, zero
removals, and zero changes to pre-existing values. Thirty-two additions are
false selectors and execute the pre-merge arm. The remaining addition is an
explicit `rn_bt_alpha = 0.07`; the pre-merge AB3/AM4 path already supplied the
same `0.07` constant. Independently built pre- and post-merge cards produce the
same SHA-256 for every path/dtype/shape/byte of all 24 state leaves after each
of ten steps. The current digest `dca02dffba3eb14f` therefore replaces
`56c5171d43504b54` as the admitted shipped-card digest.

The five-year `r4_smag` climate PASS is carried forward explicitly as
provenance. It was earned at commit `219db484b`, not rerun on this merge, and
transfers only because the user-authorized resolved-config audit and complete
ten-step state-byte comparison prove the imported fields do not move GYRE.
The result remains PASS against both registered vector- and scalar-math NEMO
builds; this does not turn the long climate comparison into a trajectory claim.

## Finding -> disposition -> evidence

### 1. Exact resolved-configuration audit -> all additions inert on GYRE

The permanent regression test asserts this exact 33-row inventory. "Old arm"
below means the false branch is the formula/path present at `7e78361fb1`; it
does not mean the surrounding physical operator is inactive.

| resolved path | after | GYRE execution path and disposition |
|---|---:|---|
| `barotropic.barotropic_fold_top_v_face_ssh` | false | `_nemo_fold_top_v_wet` returns `None`; no fold-image face-SSH operand |
| `barotropic.barotropic_flux_update_nemo_literal` | false | `nemo_flux_form_velocity_update` takes its old algebraic update |
| `barotropic.barotropic_substep_dt_nemo_literal` | false | substep clock divides by integer `nn_e`, the old arm |
| `barotropic.barotropic_entry_depth_nemo_literal` | false | `_nemo_entry_face_depths` returns the old summed depth |
| `barotropic.barotropic_een_ratio_fold_exchange` | false | `nemo_een_ratio_fold_exchange` returns all three inputs unchanged |
| `barotropic.barotropic_een_fe3mask_pivot_row` | false | paired fold selector remains `(false,false)`; old pivot source |
| `barotropic.barotropic_een_ratio_written_order` | false | `_nemo_een_ratio_written_order` returns the already-divided ratio |
| `barotropic.barotropic_literal_fold_unit_vector_form` | false | vector-form literal fold unit is not selected |
| `barotropic.nemo_barotropic_filter_alpha` | 0.07 | live AB3/AM4 value; exactly replaces the pre-merge `_NEMO_BT_ALPHA = 0.07` default |
| `barotropic.nemo_literal_external_mode_flux_form` | false | literal flux-form external-mode bundle is not selected |
| `een_fold_image_v_row` | false | `_een_fold_image_v_row` returns its six inputs unchanged |
| `rk3_stage1_ssh_nemo_literal` | false | stage-one SSH uses the prior `before + (after-before)/3` expression |
| `rk3_stage1_rhs_nemo_literal` | false | GYRE flux form keeps the prior perturbation RHS selection |
| `rk3_stage_velocity_written_order` | false | all three stages use the prior RK velocity composition |
| `een_vorticity_written_order` | false | EEN triad keeps the prior accumulator composition |
| `zdf_barotropic_removal_nemo_literal` | false | `_zdf_residual_nemo_literal` returns false; prior add-back path |
| `zdf_momentum_nemo_face_thickness` | false | implicit momentum ZDF keeps its prior face-thickness construction |
| `rk3_fold_exchange_top_v_row` | false | no T-pivot top-V exchange on closed-basin GYRE |
| `rk3_stage_r3f_nemo_literal` | false | `_nemo_ws_stage_r3f` returns `None`; stage uses its pre-existing ratio |
| `rk3_stage_r3t_nemo_literal` | false | `_nemo_ws_stage_r3t_stretch` returns `None`; prior live stretch |
| `nemo_eos_gdept_rounded_product` | false | EOS depth product is not given the ORCA2 materialization arm |
| `rk3_stage_r3uv_nemo_literal` | false | `_nemo_ws_stage_pair` returns `None`; prior live face ratios |
| `zdf_tracer_nemo_rounded_thickness` | false | tracer ZDF keeps prior unmaterialized thickness |
| `barotropic_rhs_ascending_level_sum` | false | depth reduction remains `jnp.sum`, the pre-merge path |
| `zdf_momentum_stage_r3u_nemo_literal` | false | no stored stage-ratio interface ladder in momentum ZDF |
| `een_nemo_relative_vorticity` | false | EEN uses its prior vorticity path and no NEMO F-area replacement |
| `een_nemo_top_f_u_halo` | false | no T-pivot top-F/U halo arm in the curl |
| `barotropic_wind_nemo_stored_depth` | false | wind keeps its prior live-column depth reciprocal |
| `surface_stress_nemo_fold_vector_sign` | false | surface stress keeps prior face signs |
| `zdf_surface_stress_nemo_face_thickness` | false | implicit stress keeps prior top-face thickness |
| `rk3_stage_tracer_written_order` | false | tracer RK updates keep their prior composition |
| `rk3_stage3_heat_nemo_written` | false | stage-three heat uses the prior combined source-rate path |
| `rk3_stage3_tracer_nemo_stored_thickness` | false | stage-three tracer ZDF uses the prior mean thickness |

This audit is deliberately executable: the test reads all 33 paths from the
resolved current card, asserts their exact values, and the card digest test
recomputes `dca02dffba3eb14f`. A changed selector, missing field, extra field in
the pre/post diff, or changed legacy value is not admitted by this receipt.

### 2. Ten-step byte comparison -> exact on every state word

The trajectory gate now hashes path, dtype, shape, and C-order bytes for every
state leaf after every step. The same committed hashing program was applied to
the pre-merge worktree at `7e78361fb1` and the merged tree. Both sides have 24
leaves and both produce:

| step | whole-state SHA-256 |
|---:|---|
| 1 | `20b095a840e680dc5e15acaf5689c5e1b96a47979ec2f40f2c969c428aedbdb5` |
| 2 | `f851ac2710b3ca63984846f1a77dfc4014651797dd9b4f418f35c9fb848af629` |
| 3 | `61cc76db690e7d689f1a2f23cfa06bc8dc14fb056cbfda4b1df74775264dcd46` |
| 4 | `cce107fe25f9db1e0b5459b48bbe37e301909d4f285162d61fe0dcface10ff50` |
| 5 | `63394d664c5c2707cdda6098e32400dede4931ad052f30a36fe824769cc817b3` |
| 6 | `aaaec1eb4d3db1d08d6b217f28d2a4cdd8b293bcdc4b57138c9c17c22a8947b7` |
| 7 | `abe237fabfef3a7b53104d1e9189fea6788a7c67ced7948a70dbe904d571dccc` |
| 8 | `fb191c26100e593f2ec59fb4220f9ddb6066075b658bd23019f1fa9b9610f961` |
| 9 | `a03091102b6f5b0767b1289b5f778eaf00ac9bc50835f6fd860e9e4c00e7d624` |
| 10 | `b17e7e20bbe55b6a38f79205f2cae514552d674892423c3ee4d7636533f70799` |

The hash control changes when one float64 word moves by one ULP. The ordinary
70-row NEMO-distance ceiling also passes, so accepting the digest neither hides
a moved unscored carry nor worsens a scored oracle row.

### 3. Closing rounding-order check -> preregistered and exact seed proved

The preregistration is commit `3117150a54f2e4abf29b18f7d29d5b8557ca56f8`,
SHA-256 `3a9cc855a1f90fb7ee8a7f13a3844d07f431f1dc7a13eeece67f84a2139ab95b`.
It fixes the intervention, fields, masks, days, regional/growth reductions,
controls, and three terminal verdicts before the GPU data exist.

The committed CPU/libm seed builder passed at commit `8706859962d4dfca8b2f3500009764553c2acdd8`.
It independently reconstructs the round-22 pattern as materialized-coefficient
post-LDF minus live-composed post-LDF: U maximum
`1.852884572118782e-21` over 3,789 unequal active words and V maximum
`3.3881317890172014e-21` over 3,261. Evidence is
`gyrehr_rounds/round24/seed_preflight/seed_manifest.json`, SHA-256
`a09d3801094e8d999a21dfb10c7b77d2683489dc8431abd0514e2ac11046b9f2`;
the seed NPZ is pinned inside it at
`f22f48546ec9fe396753d8bf6b47e6541b220269b594e188e99fc437981a8bc9`.

The GPU wrapper requires an explicit physical device, rebuilds that exact
seed under certified CPU/libm, proves an identity transform leaves the first
step byte-identical, proves the seeded transform moves the first-step state,
then advances both arms through the same ordinary compiled-day production
stepper for 180 days. It scores the result against the pinned round-11 NEMO
envelope. This sandbox has no CUDA device, so the terminal scientific verdict
is **UNMEASURED** until the operator runs the one command in the final report.

## Certification boundary and OPEN

Certified and carried forward:

- shipped `GYRE-zco`, 31 levels, WS-RK3, split-explicit free surface, TKE/EVD,
  Redi tracers, scalar-libm CPU ladder, digest `dca02dffba3eb14f`;
- ten merged-card steps byte-identical to the pre-merge card over every state
  leaf, plus the inherited 70-row no-worsening NEMO gate;
- `r4_smag` five-year climate PASS against both registered NEMO builds, earned
  at `219db484b` and transferred only across the exact audit above;
- stage-1 LDF residual classified as coefficient/operator composition, with
  no transcription owner and no production-physics edit.

OPEN: one closing sensitivity result. Until its two-arm GPU run is scored, the
measured `3.3881317890172014e-21` residue is a boundary localization, not a
measured explanation of the 30--180-day separation. No shared-path change is
warranted now. After the registered score, `CONFIRM_FULL`,
`REFUTE_ROUNDING_EXPLANATION`, or `PARTIAL_OR_INCONCLUSIVE` closes the hunt
without a post-hoc bar.

Round 18, round 19, and this closing claim remain **HELD / UNREVIEWED-BY-GLM**.
GLM is unavailable as recorded by the operator. No physics fix depends on the
unreviewed result.

## Tests, controls, and choice audit

The post-repair current high-resolution split reports `205 passed, 2
deselected`. The time-level/card-reconciliation/phase-3 split reports `62
passed, 2 deselected`. The compiler-budget gradient split reports nine generic
cases passed, the one-step GYRE case passed, the registered six-step long case
skipped, and the SWG card case passed: `11 passed, 1 skipped`. Aggregate:
`FULL_LANE_BATTERY_SPLIT_PASS: 278 passed, 4 deselected, 1 skipped`.

The citation-map drift caused by the new 24-line hash helper was repaired only
after re-reading the moved record-magic check; all 17 citation tests and their
plants pass. The committed-receipt citation gate reports `PASS` with one
resolved citation and no failures; shifting that citation by two lines reports
`SYMBOL-NOT-AT-LINE` and exits 1.

Choice audit -- **ASKED**: exact config diff, ten-step whole-state byte audit,
digest adoption, explicit climate-PASS provenance transfer, preregistered
closing perturbation experiment, exact GPU command, receipt, and gates.
**UNASKED**: no physics, deck, bar, NEMO source/run, GPU run, issue comment, or
external publication was made.

## Compiled-source citation

The residual used as the closing seed sits at the boundary between NEMO's
mode-32 coefficient construction and halo exchange
(`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`) and the already
verified two-pass consumer. Round 24 changes neither program.
