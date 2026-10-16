# SWG online learning, round 6: training-side receipt

Date: 2026-10-14. Base: lane tip `a0b5535b2`. All changes are uncommitted in
the work copy `claude-swgonl-r6-training` (the operator commits). CPU only,
fp64 (`JAX_ENABLE_X64=1`). Scope: every finding of the round-5 Claude review
in the lenses objective-optimiser-gradient, data-contract and state-operator,
plus sea-level-balanced starts. No training driver was written.

Commands below run from the work copy with
`PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`
and `python -m pytest -q -p no:cacheprovider <node id>`. A "reversion" is a
scratch mutant that undoes one fix in a scratch copy of the module (the work
copy is never edited by it): table `mutants.py` and `mutants_extra.py`,
runner `mutrun.py`, driver `reversions.py`, results `reversions_final.json`, all in
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r6/training/impl/`. Every
reversion listed below made its test fail (`1 failed`, exit code 1).

## What changed

- The Eq. 7 point loss weights `dx^2` on every interior F point
  (`swg_online.py:1119`, `eq7_point_weights`), first ring included, zero on
  wall vertices and dry points. An explicit `wall_ring` option, `"weighted"`
  or `"excluded"`, has no default; anything else raises (`swg_online.py:1107`).
- `L` is half the deck's square side: `basin_length_m` is the namelist's
  `rn_domsiz_m = 2L` (Maddison Table 1, "Domain size 2L 4,000 km") and the
  loss halves it (`swg_online.py:1146`), after checking it against the card's
  own wet rectangle, so passing `L` itself or any other literal raises.
- Eq. 7 takes no data normalisers: the rotational-velocity term and the
  `normalizers` argument left the point loss (`swg_online.py:1169`), and the
  contract's `training_normalizers`, `normalizers_for_split` and the legacy
  `balanced_normalizers_from_training` were deleted. The dormant energy score
  still takes caller-supplied `BalancedNormalizers`.
- The symmetric objective's coarse Gaussian is registered on F points: the
  stencil spans the wall-to-wall vertex rectangle with odd parity about the
  wall vertices at all four walls (`swg_online.py:383`); walls stay zero.
- The streamfunction inversion uses the F interior (walls at the wall
  vertices) and returns rotational velocity on the native U/V faces
  (`swg_online.py:421`). The card must be fp64 (`swg_online.py:242`).
- Sea-level-only start rebalancing, `build_start_rebalancer`
  (`swg_online.py:985`), the operator's registered rule, reusing the lane's
  closed-wall Neumann projection, now `closed_wall_divergent_part`
  (`swg_online.py:930`), moved from the admission script into the package.
  Stop at `rms(a_div) <= 1e-3 rms(a - a_div)` (`swg_online.py:1041`, exact
  rest included), cap 20, update `eta += (phi - <phi>)/g` with the card's
  gravity (`swg_online.py:1052`), raise on non-convergence. Offline numpy
  plus the certified no-closure step; nothing is differentiated.
- The dataset contract (`swg_online.py:628`): every window starts from the
  stored state with its sea level rebalanced (`swg_online.py:830`);
  validation runs for every construction path (`swg_online.py:649`);
  provenance must be unperturbed, verified, fp64 truth from one truth run on
  the registered fine deck (`swg_online.py:501`), whose restart the named
  truth run lists with the same digest (`swg_online.py:544`); each day file
  holds exactly 24 records (`swg_online.py:475`); day files are hashed and
  decoded once per cache (`swg_online.py:446`); a batch validates only if this
  contract issued its exact arrays, start hours and dtypes under its split
  label (`swg_online.py:862`, `swg_online.py:867`); training draws an offset
  per (seed, window length, epoch) (`swg_online.py:809`); held-out splits use
  one fixed chronological lattice (`swg_online.py:813`); a split without one
  full batch raises (`swg_online.py:825`).
- The optimiser factory gains explicit `adam_b1`, `adam_b2` and
  `adam_epsilon_convention` (`training.py:206`, `training.py:208`), defaults
  equal to the historical Optax update. `"keras_epsilon_hat"`
  (`training.py:287`) reproduces Keras AdamW, whose epsilon sits on the
  uncorrected second moment, through `optax.inject_hyperparams`
  (`training.py:298`).
- The admission tool judges fast energy against the same-operator truth
  (`nemo_testcase_l1_swg_online_balanced_start.py:114`,
  `nemo_testcase_l1_swg_online_balanced_start.py:116`), refuses negative
  inputs (`nemo_testcase_l1_swg_online_balanced_start.py:104`), admits the
  rebalanced start that training uses
  (`nemo_testcase_l1_swg_online_balanced_start.py:577`) with the stored start
  as a diagnostic, reports the truth references
  (`nemo_testcase_l1_swg_online_balanced_start.py:588`,
  `nemo_testcase_l1_swg_online_balanced_start.py:592`), selects no objective
  (`nemo_testcase_l1_swg_online_balanced_start.py:627`), reads its truth
  through the package loader (`nemo_testcase_l1_swg_online_balanced_start.py:276`),
  and its legacy raw-versus-conservative mode now reports stored-start
  admission only (`nemo_testcase_l1_swg_online_balanced_start.py:210`).

## Finding dispositions

| Finding | Disposition | Evidence (command, decisive line) | Reversion that makes its test fail |
|---|---|---|---|
| OOG-01 / D1-N1 first interior F row and column unweighted | Fixed | `test_eq7_weights_every_interior_f_point_and_nothing_else` (`test_swg_online.py:211`): `1 passed`; weights nonzero exactly on the 6,241 interior F points | `loss_old_T_rectangle_mask` |
| V-02 no test pins the weighted points | Fixed (same test rejects an all-points mask) | as above | `loss_all_points_mask`, `loss_wall_ring_defaults_to_weighted` |
| OOG-02 L is the full width, loss 1/16 of Eq. 7 | Fixed: L = `rn_domsiz_m`/2 from the deck, checked against the grid | `test_eq7_scale_uses_half_the_deck_domain_and_ignores_velocity` (`test_swg_online.py:181`): `1 passed` | `loss_full_width_as_L`, `loss_L_not_checked_against_deck` |
| OOG-03 epsilon regime and Keras convention | Fixed in the factory (Keras form available and pinned); regime measured below; which convention experiment one uses is an owner confirmation | `test_keras_epsilon_hat_adamw_matches_the_keras_update` (`test_create_optimizer.py:143`): factory equals a NumPy transcription of Keras 3.6.0 AdamW to `1e-14` over six steps at abs(g) from 1 to 1e-7 | `optimizer_keras_is_optax_form`, `optimizer_keras_epsilon_ignored` |
| OOG-04 optimiser tests do not pin epsilon or defaults | Fixed | `test_constant_unclipped_adamw_matches_hand_computed_first_step` (`test_create_optimizer.py:107`, abs(g) = eps, fails for 1e-8); `test_historical_defaults_match_previous_transform_bit_for_bit` (`test_create_optimizer.py:82`, three nonzero steps, clipping active) | `optimizer_epsilon_ignored`, `optimizer_default_epsilon_1e-3`, `optimizer_default_no_clipping` |
| OOG-05 absent-potential test cannot fail | Deferred: `tests/ocean/unit/test_swg_swe_rk3_stage.py` is not this task's file | the review measured that eight sibling tests catch the mutant | none here |
| OOG-06 zero-at-start test uses zero fields | Fixed: a non-trivial random state gives exactly zero eagerly (under jit at most `1e-12` of the checkerboard rise: the closed-over target can be constant-folded) | `test_named_objectives_are_zero_at_start_through_one_diagnostic` (`test_swg_online.py:246`): `1 passed` | `loss_target_always_smoothed` |
| OOG-07 template blind to divergent noise; analytic ratio wrong | Refuted as a defect (Eq. 7 is a vorticity loss by definition); the misleading analytic-ratio field was removed and the tool reports the checkerboard rise through the real loss for both ring settings (`nemo_testcase_l1_swg_online_balanced_start.py:600`) | `test_dual_operator_start_evaluation_runs_the_real_body` (`test_swg_online_balanced_start.py:177`): `1 passed` | none (no code fix) |
| OOG-08 rotational inversion sized for T, F velocities | Fixed: F-interior DST, native face velocities | `test_rotational_velocity_recovers_a_nondivergent_cgrid_flow` (`test_swg_online.py:345`): exact flow recovered to `1e-11` | `rotational_centred_F_point_velocity` |
| OOG-09 no precision-policy check | Fixed: a non-fp64 card raises | `test_closure_geometry_refuses_a_card_not_built_in_fp64` (`test_swg_online.py:358`) | `geometry_no_fp64_check` |
| V-01 / D1-N2 symmetric Gaussian registered on T | Fixed: F rectangle, odd parity at all walls (a stated choice) | `test_symmetric_gaussian_is_registered_on_f_points` (`test_swg_online.py:280`): mirror-equivariant to round-off, nothing on walls, interior equals the separable Gaussian, a unit vortex beside any wall keeps `w0 + w1` | `gaussian_old_T_slice_reflect`, `gaussian_even_about_walls` |
| DC-01 normalisers raise on multi-chunk splits | Resolved by removal (DC-08) | no caller remains (`grep`) | none (deleted code) |
| NEW-01 read-only array under fp64 | Resolved by removal (DC-08) | as above | none |
| DC-02 adjacent driver chunks refused | Fixed: the shared restart hour is served by the earlier chunk's in-scan record; a chunk's stored t0 serves only a record's first hour and is checked to agree within `1e-10` of the field maximum (`swg_online.py:767`, `swg_online.py:735`) | `test_shared_restart_hour_uses_the_inscan_record_within_round_off` (`test_swg_online.py:678`): a one-ulp disagreement is accepted and the in-scan word served; `1e-6` relative is refused | `contract_adjacency_check_off`, `contract_shared_hour_from_stored_t0` |
| DC-03 dataset tests are a weak net | Fixed: 27 contract, loader and sampler tests and plants, each with its reversion in this table | `tests/ml/test_swg_online.py`: `50 passed` | every row from DC-02 to NEW-02 |
| DC-04 never ran on real data; one-segment split yields nothing | (d) fixed: fewer than one batch raises. (a)-(b) deferred: no admissible real v2 record exists (the CPU segments regenerated here have no next-day verification, which needs the GPU restarts). (c) by design: one contiguous record per split | `test_a_split_without_one_full_batch_raises` (`test_swg_online.py:672`) | `sampler_short_split_yields_nothing` |
| DC-05 isolation by label only | Fixed: issued-batch content digests (arrays, start hours, dtypes); constructor and `dataclasses.replace` validate; splits immutable | `test_batch_membership_is_bound_to_the_issued_arrays` (`test_swg_online.py:834`): held-out arrays under train labels, a relabelled batch, an edited word, float32 or float copies, a float window length, a hand-built batch and a second contract instance all raise | `membership_label_only`, `contract_validation_only_in_from_json`, `membership_digest_ignores_dtype`, `membership_start_hours_cast_to_int`, `membership_window_days_not_int` |
| DC-06 one offset phase for all windows; held-out set moves | Fixed: offset per (seed, window, epoch); held-out fixed lattice, `epoch`/`seed` refused there | `test_training_sampler_redraws_offset_per_window_and_epoch` (`test_swg_online.py:627`), `test_heldout_splits_use_one_fixed_chronological_lattice` (`test_swg_online.py:654`) | `sampler_one_draw_for_all_window_lengths`, `sampler_offset_always_zero`, `sampler_no_shuffle`, `sampler_epoch_ignored`, `sampler_constant_seed`, `sampler_windows_overlap_by_half`, `sampler_heldout_randomised_like_train` |
| DC-07 24 records and provenance unchecked | Fixed (`swg_online.py:475`, `swg_online.py:516`, `swg_online.py:520`, `swg_online.py:530`, `swg_online.py:544`, `swg_online.py:676`) | `test_contract_refuses_each_planted_violation` (`test_swg_online.py:793`, 17 plants) and `test_day_file_plants_raise_when_the_day_is_read` (`test_swg_online.py:805`, 3 plants) | `contract_provenance_unchecked` (5 plants), `contract_restart_snapshot_unchecked` (2), `contract_truth_listing_unchecked`, `contract_restart_name_unchecked`, `contract_one_truth_run_unchecked`, `contract_gap_check_off`, `contract_split_order_abs_gap`, `contract_coverage_check_off`, `contract_manifest_pin_off`, `contract_admits_v1`, `contract_admits_incomplete`, `contract_duplicate_unchecked`, `loader_24_records_unchecked`, `loader_lineage_unchecked`, `loader_digest_unchecked` |
| DC-08 normalisers not consumed | Removed: Eq. 7 and the template's network scales (Maddison Eq. 6) are physical, not data statistics; the method was broken in every lane configuration (NEW-01, DC-01), so no consumer ever ran it | `test_point_and_energy_losses_share_interface_and_plants_fail` (`test_swg_online.py:134`): normalisers passed to the point loss raise | none (deleted code) |
| NEW-02 per-batch reload | Fixed: per-day-file cache | `test_day_files_are_hashed_once_while_cached` (`test_swg_online.py:873`) | `loader_reloads_every_batch` |
| D1-01 blind early/late rule selects the objective | Fixed: truth-relative rule, no selection | `test_admission_is_relative_to_same_operator_truth_and_fails_closed` (`test_swg_online_balanced_start.py:71`), `test_dual_summary_counts_and_never_selects_an_objective` (`test_swg_online_balanced_start.py:233`), `test_dual_operator_cli_exit_code_is_admission_not_selection` (`test_swg_online_balanced_start.py:260`) | `tool_blind_early_late_rule`, `tool_admission_accepts_negative`, `tool_selects_objective`, `tool_exit_by_eight_of_ten_filtered` |
| D1-02 filtered state unbalanced near walls | Start: fixed by rebalancing. Target: owner decision, exposed as `wall_ring` | measurements below | see D1-01 and the balanced-start row |
| D1-03 dual analyzer omits truth, body untested | Fixed: truth fast energy and vorticity RMS reported; body executed; truth read through the lane loader | `test_dual_operator_start_evaluation_runs_the_real_body` (rebalanced start ratio exactly 1, stored start rejected); `test_hourly_target_loader_checks_lineage_through_the_lane_loader` (`test_swg_online_balanced_start.py:154`) | `tool_admits_the_stored_start`, `tool_loader_unvalidated` |
| D1-04, D1-05 driver writer and toy-model claim | Deferred: the driver and its tests are not this task's files | none | none |
| D1-06 even-reflection plant | Deferred: `test_swg_coarse_graining.py` is not this task's file | none | none |
| D1-N3 box start unbalanced | Fixed: the box start is rebalanced too | box starts: 6 iterations, see below | `contract_start_not_rebalanced` |
| Balanced starts (task item 3) | Implemented | `test_rebalancer_moves_only_wet_sea_level_keeps_volume_and_balances` (`test_swg_online.py:422`), `test_rebalancer_refuses_non_convergence_and_bad_settings` (`test_swg_online.py:452`), `test_rebalancer_accepts_an_exact_rest_state_unchanged` (`test_swg_online.py:473`), `test_contract_batches_start_from_rebalanced_stored_states` (`test_swg_online.py:601`) | `rebalance_noop`, `rebalance_corner_anchored_potential`, `rebalance_nondeterministic`, `rebalance_returns_unconverged`, `rebalance_rest_state_unconverged`, `contract_t0_one_hour_late`, `contract_targets_start_at_t0`, `contract_template_reads_box`, `contract_start_not_rebalanced` |

Reversion totals: 62 mutants, 66 test runs, every one `1 failed`
(`reversions_final.json`). The first pass missed two: an even-parity
Gaussian (now caught by the wall-mass assertion) and an Optax-path epsilon
mutant paired with the Keras test (the Keras path now has its own). In the
final pass one mutant was malformed (`False and` disabled only the status
clause); it was split into two precise mutants, both caught.

Equivalent mutants, stated so silence is not read as coverage: removing the
explicit area-mean subtraction in the rebalancing update changes nothing
measurable, because the Neumann solve already returns a zero-mean potential,
so the volume test is shown non-vacuous with a corner-anchored potential; the
bounds check inside `batches()` is redundant with the lattice it checks.

## Measurements

No-closure Eq. 7 loss, means over real starts. "Tip" is the lane tip's code
(old mask, `L = rn_domsiz_m`, stored start); "after" is this round (all
interior F points, `L = rn_domsiz_m/2`, rebalanced start). Three 8-day
schema-v2 segments regenerated on CPU by the work copy's own driver (starts
731, 931, 1091); the 10-start rows use the review's ten 2-day CPU segments.
Script `m01_loss_grad.py`; results `m01_loss8.json`, `m01_loss2.json`.

| Objective | Lead | Starts | Tip | After, ring weighted | After, ring excluded | First-ring share |
|---|---|---|---|---|---|---|
| symmetric | 1 d | 10 | 0.001476 | 0.02441 | 0.02354 | 3.6% |
| symmetric | 2 d | 10 | 0.005348 | 0.08855 | 0.08534 | 3.6% |
| symmetric | 4 d | 3 | 0.02035 | 0.3344 | 0.3252 | 2.7% |
| symmetric | 8 d | 3 | 0.06853 | 1.115 | 1.094 | 1.8% |
| template | 1 d | 10 | 0.003977 | 0.07378 | 0.06350 | 13.9% |
| template | 2 d | 10 | 0.01465 | 0.2743 | 0.2342 | 14.6% |
| template | 4 d | 3 | 0.05293 | 0.9968 | 0.8464 | 15.1% |
| template | 8 d | 3 | 0.1513 | 3.009 | 2.418 | 19.6% |

The tip rows reproduce the review's 10-start symmetric means (0.00148,
0.00535). After/tip is 16.3-16.6 for symmetric (16 from `L`, the rest the
restored south/west ring) and 18.3-19.9 for template. Rebalancing the start
moves the weighted loss by at most 0.1% (symmetric) and 1.7% (template, 1 d).

First-stage gradients at initialisation (real 7,696-parameter network, seed
20261013, unit network scales, 1-day window, starts 731/931/1091; results
`m01_grad.json`). Step fractions are `abs(g)/(abs(g)+eps)`: Optax form with
`eps = 1e-7`, Keras form with `eps/sqrt(1-0.999) = 3.16e-6` at step one.

| Objective, case | Median abs(g) | Below 1e-7 | Below 3.16e-6 | Median step/lr, Optax | Median step/lr, Keras |
|---|---|---|---|---|---|
| symmetric, tip | 4.1e-8 to 5.3e-8 | 77-85% | 100% | 0.29-0.35 | 0.013-0.017 |
| symmetric, after, weighted | 6.7e-7 to 9.3e-7 | 6.1-8.1% | 92-98% | 0.87-0.90 | 0.18-0.23 |
| template, tip | 1.1e-7 to 1.6e-7 | 34-47% | 100% | 0.52-0.62 | 0.034-0.049 |
| template, after, weighted | 3.8e-6 to 6.0e-6 | 1.0-1.6% | 29-44% | 0.97-0.98 | 0.54-0.66 |

The tip's symmetric row at start 731 reproduces the review exactly (median
4.47e-8, 82.4% below 1e-7, step fractions 0.309 and 0.0139). No parameter has
an exactly zero or non-finite gradient in any case.

Rebalancing reproduces the operator's measurement: on the stored t0 of start
731, box converges in 6 iterations and filtered in 5, the rebalanced sea level
equals the measurement's to `8.3e-16` and `1.0e-15` m, a re-run is bitwise
identical, and each start takes about 0.02 s after compilation.

## Review

A read-only codex adversarial review ran five passes over this diff (logs
`codex_review*.log`, answers `codex_review*_last.md` in the scratch folder).
Fixed from its findings: the admission tool's own loader checked neither
record count nor lineage nor provenance (it now uses the package loader); the
admission rule accepted negative inputs; the rebalancer called exact rest
unconverged; the stream provenance did not tie the restart to a snapshot its
truth run lists; the batch digest dropped dtypes and coerced start hours; a
float window length validated; a restart path naming another day passed.
Open, not fixed in code: (A) the admission
tool pins each segment's manifest to the digest it has when read, because no
registered digests exist for the ten dual segments (decision below); (C) the
two external call sites below. The fifth pass found no code defect; its only
finding was this paragraph's then-unfilled verdict. GLM was not available to
this subagent, so the dual-review rule is met by codex alone.

## Decisions for the owner

1. `wall_ring`: Eq. 7 as written weights the first interior ring
   (`"weighted"`); under the template objective that ring carries 14-20% of
   the no-closure loss. There is no default; the driver must pass one.
2. Objective: the admission tool reports both operators and selects nothing.
3. Optimiser convention: the template's Keras-default AdamW is
   `adam_epsilon_convention="keras_epsilon_hat"` with `adam_epsilon=1e-7`;
   the real-data seam test uses it. With unit network scales its first steps
   stay epsilon-limited for `symmetric` (92-98% of parameters below the
   step-one epsilon). The network's input/output scales set that regime;
   Maddison's are physical (Eq. 6: `1/(abs(beta) L)` and
   `abs(tau0) pi/(rho0 D L)`), and none is chosen yet.
4. Registered digests for the admission run: requiring a pins file that maps
   each of the ten segments to its manifest digest (codex finding A) would add
   one required argument to the operator's admission command.
5. Confirm two choices made here: odd wall parity for the symmetric Gaussian,
   and a fixed held-out lattice (offset 0, chronological order).

## Outside this task's files

- `scripts/bench/profile_swg_online.py:104-108` and
  `nemo_testcase_l1_swg_online_gpu_smoke.py:273-278` call
  `deterministic_trajectory_loss` with `normalizers=` and without
  `wall_ring=`; both now raise `TypeError` at run time. Each needs one line
  removed and one explicit `wall_ring=` added; their `basin_length_m` is
  already correct. Their unit tests do not reach the call and pass.
- The design note still states the 8/10 automatic selection, the blind
  early/late rule, contract-computed normalisers, the rotational-velocity
  option and stored (unbalanced) starts; it needs the operator's edit.
- The lane citation gate maps none of this receipt's citations (it covers
  oracle files), and it refuses to stamp an uncommitted tree; a local checker
  verifies all of them (`check_citations.py`, `CITATIONS 72 checked, 0
  failing`, and a planted two-line shift fails it).

## Choices made

- ASKED (task): ring option with no default; `L` from the deck; rebalancing
  rule, tolerance `1e-3`, cap 20, card gravity; truth-relative admission with
  factor 2; normalisers removed (offered); Keras epsilon pinned to `1e-7`.
- UNASKED: odd wall parity for the symmetric Gaussian; fixed held-out
  lattice; the adjacent-hour tolerance `1e-10` of the field maximum; a passed
  next-day verification and a truth-run listing required for every source; the
  volume bound `1e-12` of the basin volume in the tool; a 128-file day cache;
  explicit `adam_b1`/`adam_b2` fields; the admission tool's vorticity-error RMS
  taken over interior F points (a constant factor, verdict unchanged); the
  legacy mode's summary keys renamed. Each can be reverted on request.

## Tests

Final runs on the final code, one file group at a time:

- `tests/ml/test_swg_online.py`: `50 passed in 129.39s`.
- `tests/ocean/fidelity/test_swg_online_balanced_start.py`: `9 passed in 16.00s`.
- `tests/ml/test_create_optimizer.py tests/ml/test_profile_swg_online.py
  tests/ocean/fidelity/test_swg_online_gpu_smoke.py
  tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`:
  `2 failed, 29 passed in 10.33s`; both failures are the citation gate
  refusing to stamp a dirty tree ("6 tracked file(s) modified"), which is this
  round's uncommitted state by instruction.
- `tests/test_dispatch_hardening.py tests/test_validate_strict_coverage.py
  tests/unit/test_audit_mlad_fixes.py`: `134 passed, 1 warning in 31.30s`.
- `ruff check` on the six changed files: `All checks passed!`.

## UNVERIFIED

- GPU: rebalancing runs on the default device; CPU and GPU rebalanced starts
  have not been compared, and nothing here ran on a GPU.
- The contract has not run on a real schema-v2 record: none with a passed
  next-day verification exists. Before training, `from_json` and one batch
  per split and window length must run on the real record.
- The Keras form was read from the Keras 3.6.0 wheel; the version Maddison
  used is not stated. Keras 2's legacy Adam used the same epsilon-hat form
  (recalled, not read).
- The two external call sites above were read, not run.
