# GYRE NEMO-fidelity merge-main receipt — 2026-09-16

## Scope and provenance

This receipt covers merge commit `524a7487a9fbeceda3d574e4edfddd0b61cae39f`,
whose parents are lane tip `62003edf55b967047222ed16d5d975454bef618c`
and `gh/main` `946351212fb243121396c7105e047eead24ad559`.  Evidence is
under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/merge_main_2026-09-16/`.

## Textual conflict

The conflict was the `nemo_1p5_split` literal-matrix `diag` in
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py`.  Main retained
the older trailing-identity construction from `5ccac70f8772` (slice
`dissl[..., :N-1]`, then append one); the lane correction was introduced by
`a71aa8e2c5a` and hardened by `66db3d887dd4`, with its evidence in
`nemo_testcases_l2_gyre_tke_runaway_receipt.md`.  Later main-side touches
`4eba0eb9e8f3` and `1c8777e45b49` did not change the bottom-row identity.

Decision: retain the lane's full `dissl[..., :N]` term.  The compiled GYRE
source
`cfgs/GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/zdftke.f90:414-425`
builds `zdiag` for every `jk=2..jpkm1`, including the dissipation term at
line 425.  The same compiled source at `:474-475` then solves
`en(:,:,jpkm1)` by dividing by `zdiag(:,:,jpkm1)`.  NEMO therefore does not
install an identity diagonal on the deepest carried row.

## Auto-merged overlap audit

Each main-side hunk was read from
`git diff 62003edf55b9...946351212fb -- <file>` and checked against the resolved
GYRE card.

| File | GYRE-card execution verdict |
|---|---|
| `barotropic_common.py` | No. The hunk generalizes the SPMD global-dot reducer used by implicit PCG; GYRE uses `barotropic_solver=explicit_substep`. |
| `ocean_model_latlon_cgrid.py` | Mixed. The new `A_h is None` guards are evaluated but false because GYRE pins `A_h=100000`. The large Euler-start rewrite is in the leapfrog path; GYRE uses the forward-Euler outer driver with the coupled WS-RK3 stage program, so that hunk is not executed. |
| `ocean_tendency_common.py` | No. The new `owned_mask` argument is used by the MPAS caller, not the lat-lon GYRE model. |
| `experiments/dino.py` | No. These are DINO experiment-domain/configuration changes. |
| `fidelity/nemo_state_bridge.py` | No. These hunks alter the DINO bridge/setup path, not the testcase GYRE bridge. |
| `gm_redi_latlon_cgrid.py` | Mixed. The face-mask statement executes on GYRE. The new MSC resolver does not because the card has `msc_stabilize=False`. The executing mask change is trajectory-inert by the exact comparison below. |
| `k_profiles.py` | Yes, but inert. GYRE executes `nemo_z0`; the merged literal-matrix path derives its face metric from the NEMO metric ladder and does not consume the removed midpoint `dz_surface` use. |
| `state.py` | Yes for config construction only. The new resolution-scaled `A_h` metadata/resolver is bypassed by GYRE's explicit `A_h=100000`; barotropic edits in the overlap are comments. |
| `test_leapfrog_integrator.py` | No model-card execution; test-only. Its merged fixtures were reconciled with the new prognostic barotropic-pair, RK3 validation, and drag-helper keyword contracts. |
| `test_tke_carried_coefficients.py` | No model-card execution; test-only operand updates for the `nemo_z0` face metric. |
| `test_tke_nemo_terms.py` | No model-card execution; test-only hand-solve and face-metric coverage. |
| `test_validate_strict_coverage.py` | No model-card execution; validation-ratchet tests only. |

## Trajectory proof

The ladder was run in the gate's documented low-memory
`--trajectory-only` mode with `--max-step 10` and compared to the recorded lane
artifact `round96/ladder.json`.  Because the merge was necessarily uncommitted
at measurement time, the gate's explicit `LEGOESM_GATE_ALLOW_DIRTY=1` escape
was used; the artifact records `clean=false`, the escape, and the diff hash.
The comparison reported:

`ORACLE_RELATIVE_COMPARE PASS: rows=70 max_worsening_ulps=0 ...`

All five primary fields plus both carried barotropic fields were unchanged at
every step:

| kt | primary rows identical | barotropic rows identical | total |
|---:|---:|---:|---:|
| 1 | 5/5 | 2/2 | 7/7 |
| 2 | 5/5 | 2/2 | 7/7 |
| 3 | 5/5 | 2/2 | 7/7 |
| 4 | 5/5 | 2/2 | 7/7 |
| 5 | 5/5 | 2/2 | 7/7 |
| 6 | 5/5 | 2/2 | 7/7 |
| 7 | 5/5 | 2/2 | 7/7 |
| 8 | 5/5 | 2/2 | 7/7 |
| 9 | 5/5 | 2/2 | 7/7 |
| 10 | 5/5 | 2/2 | 7/7 |

The day-gap member was run for days 1--30 and scored with
`nemo_testcase_l2_gyre_year_owners.py --day-gap`.  Direct structured comparison
against the lane-tip artifact found all 30 complete row dictionaries equal.
The requested headline values are:

| Row | merged tree | lane tip `62003edf55b9` | Result |
|---|---:|---:|---|
| kt2 U max absolute residual | `2.7377110452773967e-12` | `2.7377110452773967e-12` | bit-identical |
| kt2 V max absolute residual | `3.284922138989399e-12` | `3.284922138989399e-12` | bit-identical |
| kt3 T max absolute residual | `0.0001627497246303733` | `0.0001627497246303733` | bit-identical |
| kt3 S max absolute residual | `6.327735185607253e-06` | `6.327735185607253e-06` | bit-identical |
| day-30 T RMS | `0.012397011295506804` | `0.012397011295506804` | bit-identical |

Artifacts: `ladder.json`, `ladder_comparison.json`, and `day_gap.json` in the
evidence directory.  `ladder_comparison.json` records `status=PASS`, 70 rows,
zero oracle-residual worsening ULPs, zero previous-legoesm field movement, and
no violations.

## Focused tests

The requested combined pytest process reached the repository's documented JAX
per-process compiler limit and aborted without a summary.  Running exactly the
same seven files in isolated processes produced 268 passes:

- `16 passed in 1.71s`
- `62 passed in 44.28s`
- `24 passed in 287.50s (0:04:47)`
- `16 passed in 7.09s`
- `41 passed in 150.48s (0:02:30)`
- `46 passed in 26.21s`
- `63 passed in 3.83s`

The citation suite's quoted line above is from a clean tree after the merge
commit.  The separately documented pre-existing
`test_rk3_ws_differs_from_rk3_and_is_finite` failure is outside this requested
selection.

## Independent review

`independent review unavailable in-sandbox`

The required command exited 1 before reviewing the tree:
`failed to initialize in-process app-server client: Read-only file system`.
